"""nb1012 — does REAL Vina docking pose-energy add to combined+chempropembed on the 253?
The compound pEC50 track USING STRUCTURES. Physics-based score (not a learned embedding) = the signal most
likely to ESCAPE the chempropembed sink (cycle-291). Vina best-pose energy is 1 scalar per ligand.
STEP 1 diagnostic: does vina correlate with pEC50 / the anchor residual at all? (signal vs noise)
STEP 2 integration: residual LGBM on combined+chempropembed +/- vina, multi-seed, range-clip (nb1006 protocol).
Also tested on a LIGHTER base (combined only) to separate 'no signal' from 'absorbed by chempropembed'.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
from scipy.stats import pearsonr, spearmanr

D = "data/processed"; U = "C:/pxr_struct/dock"
SEEDS = [42, 101, 202, 303, 404, 505, 606]
QL, QH = 0.05, 0.98


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def clipped(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred))


def main():
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb]
    Xc = impute(combined(smiles)).astype(np.float32)
    emb_cp = np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)

    vina513 = np.load(f"{U}/vina_scores_test.npy").astype(np.float64)
    v = vina513[unb]
    n_fail = int(np.isnan(v).sum())
    med = np.nanmedian(v); v = np.where(np.isnan(v), med, v)            # impute failed docks
    vz = ((v - v.mean()) / (v.std() + 1e-9)).astype(np.float32).reshape(-1, 1)
    print(f"vina on 253: finite={253-n_fail}/253 mean={v.mean():.2f} std={v.std():.2f} range[{v.min():.1f},{v.max():.1f}]\n")

    # STEP 1 diagnostic: does vina carry pEC50 signal? (lower energy = tighter binding -> expect NEGATIVE corr with pEC50)
    resid = y - anchor
    print("=== diagnostic ===")
    print(f"  corr(vina, pEC50)        = {pearsonr(v, y)[0]:+.3f} (p={pearsonr(v,y)[1]:.3f})  spearman={spearmanr(v,y)[0]:+.3f}  [expect NEGATIVE: tighter=more active]")
    print(f"  corr(vina, anchor resid) = {pearsonr(v, resid)[0]:+.3f} (p={pearsonr(v,resid)[1]:.3f})  [does it explain anchor error?]")

    base_full = np.hstack([Xc, emb_cp]); base_lite = Xc
    print("\n=== integration (residual LGBM, range-clip, 7 seeds) ===")
    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        bf = clipped(base_full, resid, anchor, y, folds)
        bfv = clipped(np.hstack([base_full, vz]), resid, anchor, y, folds)
        bl = clipped(base_lite, resid, anchor, y, folds)
        blv = clipped(np.hstack([base_lite, vz]), resid, anchor, y, folds)
        rows.append({"seed": seed, "d_full": round(bfv - bf, 5), "d_lite": round(blv - bl, 5)})
        print(f"  seed {seed}: full {bf:.4f}->{bfv:.4f}({bfv-bf:+.4f}) | lite {bl:.4f}->{blv:.4f}({blv-bl:+.4f})")
    df = np.array([r["d_full"] for r in rows]); dl = np.array([r["d_lite"] for r in rows])
    f_stable = df.mean() < 0 and abs(df.mean()) > df.std(); l_stable = dl.mean() < 0 and abs(dl.mean()) > dl.std()
    print("\n" + "=" * 64)
    print(f"vina on combined+chempropembed: mean={df.mean():+.5f} std={df.std():.5f} wins={int((df<0).sum())}/7 stable={f_stable}")
    print(f"vina on combined-only (lite):   mean={dl.mean():+.5f} std={dl.std():.5f} wins={int((dl<0).sum())}/7 stable={l_stable}")
    if f_stable:
        print(">>> VINA POSE-ENERGY BREAKS THE SINK on the deploy substrate -> FIRST REAL STRUCTURE LADDER BREAK")
    elif l_stable:
        print(">>> vina helps the LITE base but ABSORBED by chempropembed (real signal, redundant with GNN) -> extract richer pose features")
    else:
        print(">>> vina pose-energy: no stable gain (check diagnostic: signal-less docking vs absorbed)")
    print("=" * 64)
    json.dump({"n_fail": n_fail, "corr_vina_pec50": float(pearsonr(v, y)[0]), "corr_vina_resid": float(pearsonr(v, resid)[0]),
               "d_full_mean": float(df.mean()), "d_full_std": float(df.std()), "full_stable": bool(f_stable),
               "d_lite_mean": float(dl.mean()), "d_lite_std": float(dl.std()), "lite_stable": bool(l_stable)},
              open(f"{U}/nb1012_vina_integration.json", "w"), indent=2)
    print(f"saved -> {U}/nb1012_vina_integration.json")


if __name__ == "__main__":
    main()
