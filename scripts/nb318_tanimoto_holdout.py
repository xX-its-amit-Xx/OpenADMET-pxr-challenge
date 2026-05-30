"""nb318 -- Tanimoto-stratified 10% holdout for OOD validation.

Scaffold CV gives 0.28 OOF but LB is 0.74 — the gap is the OOD analog-expansion
distribution shift. A Tanimoto-stratified holdout should better mimic the test
distribution: pick the 414 train compounds whose nearest-neighbor Tanimoto to
the REST of train is LOWEST. These are the most "alone" compounds — closest
proxy we have to the test set, which has median top-1 Tanimoto of 0.52 to train.

Outputs:
  - data/processed/tanimoto_holdout_idx.npy  (414 train-row indices = holdout)
  - data/processed/tanimoto_inscope_idx.npy  (3725 indices = fit pool)
  - data/processed/nb318_holdout_metrics.csv (per-model RAE/Spearman on holdout)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import r2_score

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED


def main():
    print("=== nb318: Tanimoto-stratified 10% OOD holdout ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    n = len(smiles_tr)

    print(f"Computing Morgan FPs for {n} train compounds...")
    fps = []
    for s in smiles_tr:
        m = Chem.MolFromSmiles(s) if s else None
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None)
    valid_mask = np.array([f is not None for f in fps])
    valid_idx = np.where(valid_mask)[0]
    valid_fps = [fps[i] for i in valid_idx]

    print(f"\nComputing top-1 NN Tanimoto for each compound (excluding self)...")
    top1_sim = np.full(n, np.nan)
    for k, gi in enumerate(valid_idx):
        sims = np.array(BulkTanimotoSimilarity(fps[gi], valid_fps))
        sims[k] = -1  # exclude self
        top1_sim[gi] = float(sims.max())
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(valid_idx)}")

    # Pick the 10% MOST DISSIMILAR (lowest top-1 NN sim) as holdout
    n_holdout = int(0.10 * n)
    order_by_sim = np.argsort(top1_sim)  # NaNs go to end
    holdout_idx = order_by_sim[:n_holdout]
    inscope_idx = np.array([i for i in range(n) if i not in set(holdout_idx)])

    print(f"\nHoldout = {len(holdout_idx)} compounds (top-1 NN Tanimoto: "
          f"min={top1_sim[holdout_idx].min():.3f}, "
          f"max={top1_sim[holdout_idx].max():.3f}, "
          f"median={np.median(top1_sim[holdout_idx]):.3f})")
    print(f"In-scope = {len(inscope_idx)} compounds (median top-1 sim: "
          f"{np.median(top1_sim[inscope_idx]):.3f})")
    # Compare to test set's median top-1 to train (≈0.52 from EDA)
    print(f"Reference: test set median top-1 NN to train = ~0.52 (from EDA)")

    np.save(DATA_PROCESSED / "tanimoto_holdout_idx.npy", holdout_idx)
    np.save(DATA_PROCESSED / "tanimoto_inscope_idx.npy", inscope_idx)
    np.save(DATA_PROCESSED / "tanimoto_top1_sim.npy", top1_sim)
    print(f"\nSaved tanimoto_holdout_idx.npy, tanimoto_inscope_idx.npy, tanimoto_top1_sim.npy")

    # ============================
    # Score every cached OOF on the Tanimoto holdout
    # ============================
    print(f"\n=== Scoring all OOFs on the {len(holdout_idx)}-compound Tanimoto-OOD holdout ===")
    rows = []
    for oof_path in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        name = oof_path.stem.replace("oof_", "")
        try:
            o = np.load(oof_path)
            if o.shape != (n,): continue
            if not np.isfinite(o).all(): continue
            ho_y = y[holdout_idx]; ho_p = o[holdout_idx]
            r = rae(ho_y, ho_p)
            sp, _ = spearmanr(ho_y, ho_p)
            kd, _ = kendalltau(ho_y, ho_p)
            r2 = r2_score(ho_y, ho_p)
            mae = np.abs(ho_y - ho_p).mean()
            rows.append(dict(model=name, holdout_rae=r, holdout_spearman=sp,
                             holdout_kendall=kd, holdout_r2=r2, holdout_mae=mae))
        except Exception:
            continue
    sc = pd.DataFrame(rows).sort_values('holdout_rae').reset_index(drop=True)
    sc.to_csv(DATA_PROCESSED / "nb318_holdout_metrics.csv", index=False)
    print(f"\nTop 20 models by Tanimoto-OOD holdout RAE:")
    print(sc.head(20).to_string(index=False))
    print(f"\nWrote nb318_holdout_metrics.csv")

    # Compare scaffold-CV-RAE vs Tanimoto-OOD-RAE for the top-25 by either
    # (rough cross-check)
    print("\n--- Cross-check: scaffold-CV RAE vs Tanimoto-OOD RAE ---")
    # Compute scaffold-CV RAE same way (full-train OOF rae)
    sc_check = []
    for r in rows:
        nm = r['model']
        try:
            o = np.load(DATA_PROCESSED / f"oof_{nm}.npy")
            scaf_rae = rae(y, o)
            sc_check.append((nm, scaf_rae, r['holdout_rae']))
        except: pass
    sc_check.sort(key=lambda x: x[1])
    print(f"{'model':<35}{'full-OOF RAE':<15}{'tanimoto-OOD RAE':<18}{'gap':<10}")
    for nm, sr, hr in sc_check[:15]:
        gap = hr - sr
        flag = " *" if gap > 0.2 else ""
        print(f"{nm[:34]:<35}{sr:<15.4f}{hr:<18.4f}{gap:+.4f}{flag}")


if __name__ == "__main__":
    main()
