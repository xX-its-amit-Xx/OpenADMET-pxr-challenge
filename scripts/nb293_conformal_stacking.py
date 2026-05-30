"""nb293 -- Conformal-calibrated stacking with per-test-compound adaptive weights.

For each base model, compute Mondrian conformal intervals using scaffold-based
buckets. At inference time the blend weight for a base model is
proportional to (1 / interval_width) for that specific test compound.
Effectively shifts ensemble weight toward whichever sub-model is most
confident on this exact analog.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.stats import spearmanr

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb293: Conformal-calibrated stacking ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()

    # 4-way base — te files for last two are prefixed te_oof_
    oof_paths = ['oof_nb224_pool_plus_2', 'oof_nb179_stack', 'oof_multi_template_delta', 'oof_delta_loso']
    te_paths  = ['te_nb224_pool_plus_2', 'te_nb179_stack', 'te_oof_multi_template_delta', 'te_oof_delta_loso']
    base_names = ['nb224', 'nb179s', 'mtd', 'loso']
    oofs = [np.load(DATA_PROCESSED / f"{p}.npy") for p in oof_paths]
    tes  = [np.load(DATA_PROCESSED / f"{p}.npy") for p in te_paths]
    n_te = len(smiles_te)

    # Morgan FPs for Tanimoto neighborhood lookup
    print("Computing Morgan FPs for train/test...")
    fps_tr = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) if Chem.MolFromSmiles(s) else None for s in smiles_tr]
    fps_te = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) if Chem.MolFromSmiles(s) else None for s in smiles_te]

    # Conformal residuals per model: |y - oof|
    resids = [np.abs(y_tr - o) for o in oofs]

    # For each test compound, identify K nearest TRAIN compounds.
    # Per model, compute per-compound "expected interval" = average residual
    # of its K nearest neighbors. Lower = model is more confident on this region.
    print("\nComputing per-test conformal intervals via Tanimoto-NN of residuals...")
    K = 30
    intervals_per_model = np.zeros((len(base_names), n_te))
    for i, fp in enumerate(fps_te):
        if fp is None:
            intervals_per_model[:, i] = 1.0
            continue
        sims = np.array(BulkTanimotoSimilarity(fp, [f for f in fps_tr if f is not None]))
        # Map back to original train indices
        valid_idx = [j for j, f in enumerate(fps_tr) if f is not None]
        order = np.argsort(sims)[::-1][:K]
        nn_idx = [valid_idx[k] for k in order]
        for m_i, r in enumerate(resids):
            intervals_per_model[m_i, i] = r[nn_idx].mean()
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n_te}")

    print(f"Interval stats per model:")
    for n, iv in zip(base_names, intervals_per_model):
        print(f"  {n}: median={np.median(iv):.3f}  mean={iv.mean():.3f}")

    # Per-test weighting: w_m_i = 1/interval_m_i, then normalise across models
    inv = 1.0 / (intervals_per_model + 1e-6)
    w_norm = inv / inv.sum(axis=0, keepdims=True)  # (M, n_te)
    # Build te blend
    M_te = np.column_stack(tes)  # (n_te, M)
    te_blend = (M_te * w_norm.T).sum(axis=1)
    print(f"\nConformal-weighted test blend: mean={te_blend.mean():.3f}  std={te_blend.std():.3f}")

    # For OOF: use uniform 0.25 weights as proxy (since we can't conformally
    # weight train compounds without leakage); save uniform for SLSQP downstream
    M_oof = np.column_stack(oofs)
    oof_blend_uniform = M_oof.mean(axis=1)
    r = rae(y_tr, oof_blend_uniform)
    sp, _ = spearmanr(y_tr, oof_blend_uniform)
    print(f"OOF uniform-mean (4-way): RAE={r:.4f}  Spearman={sp:.4f}")

    np.save(DATA_PROCESSED / "oof_nb293_conformal.npy", oof_blend_uniform)
    np.save(DATA_PROCESSED / "te_nb293_conformal.npy", te_blend)
    print(f"\nSaved oof_nb293_conformal.npy and te_nb293_conformal.npy")

    # Submit
    sub = pd.DataFrame({'Molecule Name': te_df['Molecule Name'], 'SMILES': te_df['SMILES'], 'pEC50': te_blend})
    out = SUBMISSIONS / "nb293_conformal_blend.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
