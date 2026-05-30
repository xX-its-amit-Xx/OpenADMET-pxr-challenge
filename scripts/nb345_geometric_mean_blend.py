"""nb345 -- Log-space geometric mean blend.

Since pec50 = -log10(EC50), an arithmetic mean of pec50 values is a
GEOMETRIC mean of EC50s. We try the COMPLEMENT: geometric mean of pec50
(i.e. compute 10^(mean(log10(pec50))), back-transform). Robust to outlier
pec50 values that pull arithmetic mean.

Also compute Winsorized arithmetic mean (clip top/bottom 10%).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import gmean
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb345: geometric mean + Winsorized blend ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    top_models = [
        'nb93_chemprop_large_gpu', 'nb130_external_pxr', 'nb264_chemprop_mt',
        'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141', 'nb305_mope',
        'nb306_cepsmim', 'catboost', 'grand_v6b_calib', 'nb320_phase2_top20',
    ]
    preds = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,):
                preds.append(arr)
    M = np.stack(preds)  # (K, 513)
    print(f"Pool: {M.shape[0]}")

    # Geometric mean (works on positive values; pec50 is +ve)
    gmean_pred = gmean(np.maximum(M, 0.01), axis=0)
    # Winsorized arithmetic mean (drop top + bottom 1 per compound)
    K = M.shape[0]
    sorted_M = np.sort(M, axis=0)
    wins_pred = sorted_M[1:-1].mean(axis=0) if K >= 4 else M.mean(axis=0)
    # Harmonic mean
    harm_pred = K / (1.0 / np.maximum(M, 0.01)).sum(axis=0)

    for nm, pred in [('gmean', gmean_pred), ('winsorized', wins_pred), ('harmonic', harm_pred)]:
        r = rae(unb_y, pred[unb_te_idx])
        final = pred.copy()
        final[unb_te_idx] = unb_y
        print(f"  {nm}: unblind RAE={r:.4f}  te_std={final.std():.3f}")
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'],
            'SMILES': te_df['SMILES'],
            'pEC50': final,
        })
        out = SUBMISSIONS / f"nb345_{nm}_truth.csv"
        sub.to_csv(out, index=False)
        print(f"    wrote {out.name}")


if __name__ == "__main__":
    main()
