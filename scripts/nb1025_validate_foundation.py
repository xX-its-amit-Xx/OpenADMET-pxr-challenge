"""nb1025 — validate the Kaggle FOUNDATION (Hill) model on the honest held-out 253.

The model trained only on train-assay labels (CRC + counter + single-conc dose), never the 253 -> its 513
predictions are a clean held-out test. Questions:
  1. standalone RAE vs nb3200 (0.4416)  -- can a dose-aware multi-assay model match/beat the deploy anchor?
  2. as a FEATURE on the nb3200 substrate (clipped residual, 30 seeds) -- orthogonal contribution? (vs rich-z -0.008)
  3. simple blend with nb3200 -- convex weight.
  4. NOVELTY-STRATIFIED: does it help specifically on the low-train-similarity (F2 novel-scaffold) tail? -- the thesis
     is that the 8131 scaffold-EXCLUSIVE single-conc compounds extend coverage where CRC data couldn't.

Run after: python scripts/kaggle_push.py --nb 1024 --pull  (pulls found_pec50_513.npy to submissions/kaggle_nb1024/)
"""
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; QL, QH = 0.05, 0.98


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def find_pred():
    for p in [f"submissions/kaggle_nb1024/found_pec50_513.npy",
              f"submissions/kaggle_nb1024/found_pred_513.parquet"]:
        if os.path.exists(p):
            return p
    hits = glob.glob("submissions/**/found_pec50_513.npy", recursive=True) + \
           glob.glob("submissions/**/found_pred_513.parquet", recursive=True)
    return hits[0] if hits else None


def clipped(X, resid, anchor, y, folds):
    pred = anchor.copy()
    for tri, vai in folds:
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(X[tri], resid[tri])
        p = anchor[vai] + m.predict(X[vai]); lo, hi = np.quantile(y[tri], QL), np.quantile(y[tri], QH)
        pred[vai] = np.clip(p, lo, hi)
    return float(rae(y, pred))


def main():
    pp = find_pred()
    assert pp, "no Kaggle prediction found — run kaggle_push.py --nb 1024 --pull first"
    if pp.endswith(".npy"):
        pred513 = np.load(pp)
    else:
        pred513 = pd.read_parquet(pp).sort_values("test_pos")["pec50"].to_numpy()
    print(f"loaded {pp} | 513 preds, {np.isfinite(pred513).sum()} finite")

    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    fpred = pred513[unb]
    # impute any nan with train mean
    tr = load_train().dropna(subset=["pec50"]); tmean = tr["pec50"].mean()
    fpred = np.where(np.isfinite(fpred), fpred, tmean)

    anchor = np.load(f"{D}/nb3200_pred_oof.npy"); resid = y - anchor
    r_anchor = rae(y, anchor); r_found = rae(y, fpred)
    print(f"\nnb3200 anchor RAE = {r_anchor:.4f} | foundation standalone RAE = {r_found:.4f}  "
          f"({'BEATS' if r_found < r_anchor else 'worse than'} anchor)")

    # simple blend grid
    best_w, best_r = 0.0, r_anchor
    for w in np.linspace(0, 1, 21):
        r = rae(y, (1 - w) * anchor + w * fpred)
        if r < best_r:
            best_r, best_w = r, w
    print(f"best convex blend: w_found={best_w:.2f} RAE={best_r:.4f} (delta {best_r-r_anchor:+.4f})")

    # as a FEATURE on nb3200 (clipped residual, 30 seeds)
    te = load_test(); smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    base = np.hstack([impute(combined(smiles)).astype(np.float32), np.load(f"{D}/te_chemprop_embed_300.npy")[unb].astype(np.float32)])
    fz = ((fpred - fpred.mean()) / (fpred.std() + 1e-9)).reshape(-1, 1).astype(np.float32)
    ds = []
    for s in range(1400, 1430):
        f = scaffold_kfold_indices(scaf, 5, seed=s)
        ds.append(clipped(np.hstack([base, fz]), resid, anchor, y, f) - clipped(base, resid, anchor, y, f))
    ds = np.array(ds); st = ds.mean() < 0 and abs(ds.mean()) > ds.std()
    print(f"as FEATURE on nb3200 (30 seeds): {ds.mean():+.5f} +/- {ds.std():.5f} wins {int((ds<0).sum())}/30 stable={st}  (vs rich-z -0.008)")

    # NOVELTY stratification: top-1 Tanimoto to CRC train
    trfp = morgan_fp_batch(tr["smiles"].tolist()); tefp = morgan_fp_batch(smiles)
    trfp = trfp.astype(bool); tefp = tefp.astype(bool)
    top1 = np.zeros(len(smiles))
    for i in range(len(smiles)):
        inter = (tefp[i] & trfp).sum(1); uni = (tefp[i] | trfp).sum(1)
        top1[i] = np.max(inter / np.clip(uni, 1, None))
    novel = top1 < np.median(top1)
    for name, mask in [("NEAR (sim>=median)", ~novel), ("NOVEL (sim<median)", novel)]:
        ra = rae(y[mask], anchor[mask]); rf = rae(y[mask], fpred[mask])
        rb = rae(y[mask], (1 - best_w) * anchor[mask] + best_w * fpred[mask])
        print(f"  {name:22s} n={mask.sum():3d}  anchor {ra:.4f}  found {rf:.4f}  blend@{best_w:.2f} {rb:.4f}")

    json.dump({"anchor": r_anchor, "found_standalone": r_found, "best_blend_w": float(best_w),
               "best_blend_rae": best_r, "feature_delta": float(ds.mean()), "feature_stable": bool(st)},
              open(f"{D}/nb1025_foundation_validation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
