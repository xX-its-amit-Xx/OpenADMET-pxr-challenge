"""nb964 — deployable ADMET ladder candidate + the decisive overlap test.

nb963 showed ADMET adds to chemprop_aux + LGBM(combined). But nb3200's K18 set also has
ChempropEmbed (chemprop-derived, like ADMET) — so the fair test of "does ADMET break 0.4416"
is: does it add on top of combined + ChempropEmbed + the range-clip (the nb3200 machinery)?

Builds chemprop_aux + LGBM(combined+chempropembed [+ADMET]) residual, per-fold y-range clip
(nb3190 q05/q98), scaffold-CV on 253, pooled RAE. Multi-seed verifies the ADMET delta.
Saves deploy preds (513) for the ladder if ADMET wins stably.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
from sklearn.impute import SimpleImputer

D = "data/processed"
SEEDS = [42, 101, 202, 303, 404, 505, 606]
QL, QH = 0.05, 0.98          # nb3190 modal range-clip quantiles


def murcko(s):
    try:
        m = Chem.MolFromSmiles(s); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception: return None


def clipped_resid_cv(X, resid, anchor, y, folds):
    """chemprop_aux + LGBM(X) residual with per-fold y-range clip. Pooled RAE + preds."""
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                              n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai])
        lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred)), pred


def main():
    te = load_test()
    unb_idx = np.load(f"{D}/_audit_unblind_idx.npy")
    y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb_idx].tolist()
    scaf = [murcko(s) for s in smiles]
    anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb_idx]

    Xc = impute(combined(smiles)).astype(np.float32)
    # chemprop embeddings (the K18 ingredient most likely to overlap ADMET)
    emb_path = f"{D}/te_chemprop_embed_300.npy"
    if os.path.exists(emb_path):
        emb = np.load(emb_path)[unb_idx].astype(np.float32)
        base = np.hstack([Xc, emb]); base_tag = "combined+chempropembed"
    else:
        base = Xc; base_tag = "combined (no embed file)"
    # ADMET
    adf = pd.read_csv("C:/admet_out/admet_test.csv")
    props = [c for c in adf.columns if c != "smiles" and pd.api.types.is_numeric_dtype(adf[c])]
    A = SimpleImputer(strategy="median").fit_transform(adf[props].to_numpy(float)[unb_idx])
    A = np.clip(np.nan_to_num(A, posinf=1e6, neginf=-1e6), -1e6, 1e6).astype(np.float32)
    base_admet = np.hstack([base, A])
    print(f"base = {base_tag} {base.shape}; +ADMET = {base_admet.shape}")
    print(f"anchor RAE = {rae(y, anchor):.4f}; nb3200 full-stack ceiling = 0.4416\n")

    resid = y - anchor
    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        r_b, _ = clipped_resid_cv(base, resid, anchor, y, folds)
        r_a, _ = clipped_resid_cv(base_admet, resid, anchor, y, folds)
        rows.append({"seed": seed, "base": round(r_b, 4), "base_admet": round(r_a, 4),
                     "delta": round(r_a - r_b, 5)})
        print(f"  seed {seed}: base={r_b:.4f}  +ADMET={r_a:.4f}  delta={r_a-r_b:+.5f}")

    d = np.array([r["delta"] for r in rows])
    stable = d.mean() < 0 and abs(d.mean()) > d.std()
    print("\n" + "=" * 60)
    print(f"ADMET delta on clipped anchor-residual ({base_tag}):")
    print(f"  mean={d.mean():+.5f} std={d.std():.5f} wins={int((d<0).sum())}/{len(SEEDS)} stable={stable}")
    base_mean = np.mean([r["base"] for r in rows]); admet_mean = np.mean([r["base_admet"] for r in rows])
    print(f"  base mean RAE={base_mean:.4f}  +ADMET mean RAE={admet_mean:.4f}")
    print(">>> ADMET BREAKS this substrate -> build full deploy candidate" if stable
          else ">>> ADMET absorbed by chempropembed here -> gain is on simpler models only")
    print("=" * 60)

    # deploy preds on full 513 if ADMET stably wins (train on all 253 resid, clip on 253 y-range)
    if stable:
        all_folds = scaffold_kfold_indices(scaf, n_splits=5, seed=42)
        _, pred253 = clipped_resid_cv(base_admet, resid, anchor, y, all_folds)
        np.save(f"{D}/nb964_admet_candidate_oof253.npy", pred253)
        print(f"saved 253 OOF -> nb964_admet_candidate_oof253.npy (pooled {rae(y,pred253):.4f})")
    json.dump({"base_tag": base_tag, "rows": rows, "delta_mean": float(d.mean()),
               "delta_std": float(d.std()), "stable": bool(stable),
               "base_mean_rae": float(base_mean), "admet_mean_rae": float(admet_mean)},
              open(f"{D}/nb964_admet_ladder_candidate.json", "w"), indent=2)
    print(f"saved -> {D}/nb964_admet_ladder_candidate.json")


if __name__ == "__main__":
    main()
