"""nb1090 — MIRROR DATASETS across promiscuous NRs (user's request).

Build PXR-like blind challenges on other promiscuous nuclear receptors that have RICHER public data
(FXR 2816, PPARg 2516 EC50/AC50 cpds; plus poorer RXRa 432, LXRa 447, and papyrus-PXR 737 as controls).
Apply the SAME pipeline we used on PXR and measure, where GROUND TRUTH EXISTS, exactly where the gap is.

For each target:
  - analog-expansion-style scaffold split into ~400 train / ~250 blinded test (mimics PXR's data-poor regime)
  - baseline LGBM(combined Morgan+RDKit)            -> RAE_baseline           (mirror of nb3200's base)
  - kNN retrieval RAE + median top-1 test->train sim (difficulty profile of the split)
  - noise floor estimate from near-duplicate pairs   -> RAE_floor              (mirror of PXR 0.174)
  - alignment Spearman(dist,|dy|) Morgan/ErG/physchem (does chem-FP anti-alignment replicate?)
  - cliff structure (% high-sim pairs that are cliffs; ErG cliff-resolution)
  - metric-learning redundancy: learned activity-metric kNN; does it ADD over baseline? (saturation proof replicate?)
  - *** COVERAGE ORACLE ***: retrain on the FULL rich set (minus test) and rescore the SAME test
       -> RAE_rich. Gap(baseline - rich) = the coverage lever the saturation proof predicts is the real headroom.
       (impossible on PXR; this is the whole point of the mirror.)

Writes data/processed/nb1090_mirror.json with the cross-target table.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist, pdist, squareform
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
import torch, torch.nn as nn

P = "data/processed"
PARQ = "data/external/papyrus_pxr_nr.parquet"
TARGETS = ["FXR", "PPARg", "RXRa", "LXRa", "PXR"]   # rich -> poor, PXR as in-panel control
N_TRAIN, N_TEST = 400, 250
SEED = 42
rng = np.random.default_rng(SEED)


def murcko(s):
    m = Chem.MolFromSmiles(str(s))
    if not m: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return None


def erg(smiles):
    out = [np.array(rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles(str(s))), np.float32)
           if Chem.MolFromSmiles(str(s)) else None for s in smiles]
    dim = next(len(v) for v in out if v is not None)
    return np.array([v if v is not None else np.zeros(dim, np.float32) for v in out])


def tani(A):
    A = (A > 0).astype(np.float32); inter = A @ A.T
    s = A.sum(1)[:, None] + A.sum(1)[None, :] - inter
    return inter / np.clip(s, 1, None)


def lgbm_fit_predict(Xtr, ytr, Xte):
    m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbose=-1)
    m.fit(Xtr, ytr); return m.predict(Xte)


class Emb(nn.Module):
    def __init__(self, d, z=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, z))
    def forward(self, x): return self.f(x)


def metric_learn_add(Xtr, ytr, Xte, yte, anchor):
    """Learn ||f(a)-f(b)||~|dy|; return (learned-space alignment on test, kNN retrieval RAE, blend delta vs anchor)."""
    sc = StandardScaler().fit(np.vstack([Xtr, Xte]))
    Xt = torch.tensor(sc.transform(Xtr).astype(np.float32)); Xe = torch.tensor(sc.transform(Xte).astype(np.float32))
    yt = torch.tensor(ytr.astype(np.float32)); n = len(Xt)
    m = Emb(Xt.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(300):
        m.train(); idx = torch.randint(0, n, (2048, 2))
        de = torch.norm(m(Xt[idx[:, 0]]) - m(Xt[idx[:, 1]]) + 1e-8, dim=1)
        dy = torch.abs(yt[idx[:, 0]] - yt[idx[:, 1]])
        loss = nn.functional.smooth_l1_loss(de, dy)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Ztr = m(Xt).numpy(); Zte = m(Xe).numpy()
    iu = np.triu_indices(len(yte), 1); dyt = np.abs(yte[:, None] - yte[None, :])[iu]
    align = spearmanr(squareform(pdist(Zte))[iu], dyt).correlation
    D = cdist(Zte, Ztr); pred = np.zeros(len(yte))
    for i in range(len(yte)):
        k = np.argsort(D[i])[:5]; w = 1 / (D[i][k] + 1e-6) ** 2; pred[i] = np.sum(w * ytr[k]) / np.sum(w)
    base = rae(yte, anchor); best = base
    for w in np.linspace(0, 1, 21):
        best = min(best, rae(yte, (1 - w) * anchor + w * pred))
    return float(align), float(rae(yte, pred)), float(best - base)


def noise_floor(smi, y):
    """Estimate irreducible RAE from near-duplicate pairs (Tanimoto>=0.95): sigma=robust|dy|/sqrt(2)."""
    M = morgan_fp_batch(smi).astype(np.float32); S = tani(M)
    iu = np.triu_indices(len(y), 1); s = S[iu]; dy = np.abs(y[:, None] - y[None, :])[iu]
    near = dy[s >= 0.95]
    if len(near) < 20: return None, len(near)
    sigma = np.median(near) / (np.sqrt(2) * 0.6745)          # robust per-measurement sigma
    mad = np.mean(np.abs(y - np.median(y)))
    return float(sigma * np.sqrt(2 / np.pi) / mad), int(len(near))


def split_target(df_t):
    """analog-expansion-style: scaffold-disjoint test (~N_TEST), data-poor train (~N_TRAIN)."""
    df_t = df_t.copy(); df_t["scaf"] = df_t["std_smiles"].map(murcko)
    df_t = df_t.dropna(subset=["scaf"]).reset_index(drop=True)
    folds = scaffold_kfold_indices(df_t["scaf"].tolist(), n_splits=max(2, round(len(df_t) / N_TEST)), seed=SEED)
    # pick the fold closest to N_TEST as the blinded test
    te_idx = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - N_TEST))
    te_set = set(te_idx.tolist()); tr_pool = np.array([i for i in range(len(df_t)) if i not in te_set])
    te = te_idx[:N_TEST] if len(te_idx) > N_TEST else te_idx
    tr_full = tr_pool                                          # full rich train (for coverage oracle)
    tr_poor = rng.permutation(tr_pool)[:N_TRAIN]               # PXR-sized poor train
    return df_t, tr_poor, tr_full, te


def run_target(name, df):
    df_t = df[df["target_name"] == name].drop_duplicates("inchikey").reset_index(drop=True)
    if len(df_t) < N_TEST + 120:
        print(f"[{name}] only {len(df_t)} cpds — skip"); return None
    df_t, tr_poor, tr_full, te = split_target(df_t)
    smi = df_t["std_smiles"].tolist(); y = df_t["pec50"].to_numpy().astype(float)
    smte = [smi[i] for i in te]; yte = y[te]
    print(f"[{name}] cpds={len(df_t)} poor-train={len(tr_poor)} rich-train={len(tr_full)} test={len(te)}", flush=True)

    Xc = impute(combined(smi))                                 # shared featurization
    # difficulty: median top-1 test->poor-train Morgan sim
    M = morgan_fp_batch(smi).astype(np.float32)
    sim_te_tr = (M[te] @ M[tr_poor].T)
    s = M.sum(1); sim_te_tr = sim_te_tr / np.clip(s[te, None] + s[tr_poor][None, :] - sim_te_tr, 1, None)
    med_sim = float(np.median(sim_te_tr.max(1)))

    pred_poor = lgbm_fit_predict(Xc[tr_poor], y[tr_poor], Xc[te])
    pred_rich = lgbm_fit_predict(Xc[tr_full], y[tr_full], Xc[te])
    rae_poor, rae_rich = rae(yte, pred_poor), rae(yte, pred_rich)

    # kNN retrieval RAE (Morgan) from poor train
    knn = np.zeros(len(te))
    for i in range(len(te)):
        row = sim_te_tr[i]; k = np.argsort(row)[::-1][:5]; w = row[k] ** 2 + 1e-6
        knn[i] = np.sum(w * y[tr_poor][k]) / np.sum(w)
    rae_knn = rae(yte, knn)

    floor, n_near = noise_floor(smi, y)

    # alignment on test
    iu = np.triu_indices(len(te), 1); dyt = np.abs(yte[:, None] - yte[None, :])[iu]
    Mte = M[te]; mdist = (1 - tani(Mte))[iu]
    align = {"Morgan": spearmanr(mdist, dyt).correlation,
             "ErG": spearmanr(squareform(pdist(StandardScaler().fit_transform(erg(smte))))[iu], dyt).correlation}
    # cliff structure
    msim = tani(Mte)[iu]; hi = msim >= 0.6; cliff = hi & (dyt >= 1.0)
    cliff_frac = float(cliff.sum() / max(hi.sum(), 1))
    erg_d = squareform(pdist(StandardScaler().fit_transform(erg(smte))))[iu]
    erg_cliff_res = float(spearmanr(erg_d[hi], dyt[hi]).correlation) if hi.sum() > 10 else None

    # metric-learning redundancy over the poor baseline (does learned activity-metric ADD?)
    Xml = np.hstack([erg(smi), Xc[:, -217:]])                  # ErG + rdkit-desc tail (aligned ingredients)
    ml_align, ml_knn_rae, ml_delta = metric_learn_add(Xml[tr_poor], y[tr_poor], Xml[te], yte, pred_poor)

    res = dict(target=name, n_cpds=int(len(df_t)), n_poor=int(len(tr_poor)), n_rich=int(len(tr_full)),
               n_test=int(len(te)), median_top1_sim=med_sim,
               rae_baseline_poor=float(rae_poor), rae_rich=float(rae_rich),
               coverage_gap=float(rae_poor - rae_rich), rae_knn=float(rae_knn),
               noise_floor=floor, n_near_dup_pairs=n_near,
               align_Morgan=float(align["Morgan"]), align_ErG=float(align["ErG"]),
               cliff_frac_highsim=cliff_frac, erg_cliff_resolution=erg_cliff_res,
               ml_learned_align=ml_align, ml_knn_rae=ml_knn_rae, ml_blend_delta=ml_delta)
    print(f"  RAE poor={rae_poor:.4f} rich={rae_rich:.4f} (coverage gap {rae_poor-rae_rich:+.4f}) | "
          f"floor={floor} | med-sim={med_sim:.2f} | align M={align['Morgan']:+.3f} ErG={align['ErG']:+.3f} | "
          f"cliff%={cliff_frac:.2f} | ML add={ml_delta:+.4f}", flush=True)
    return res


def main():
    df = pd.read_parquet(PARQ)
    df = df[df["standard_type"].isin(["EC50", "AC50"])].reset_index(drop=True)
    rows = [r for t in TARGETS if (r := run_target(t, df))]
    out = pd.DataFrame(rows)
    print("\n=== MIRROR-DATASET SUMMARY (PXR reference: baseline 0.4416, floor 0.174) ===")
    cols = ["target", "n_cpds", "rae_baseline_poor", "rae_rich", "coverage_gap", "noise_floor",
            "median_top1_sim", "align_Morgan", "align_ErG", "cliff_frac_highsim", "ml_blend_delta"]
    print(out[cols].to_string(index=False))
    json.dump(rows, open(f"{P}/nb1090_mirror.json", "w"), indent=2)
    print(f"\nwrote {P}/nb1090_mirror.json")


if __name__ == "__main__":
    main()
