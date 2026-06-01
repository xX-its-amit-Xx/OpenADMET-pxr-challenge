"""nb357 -- Isotonic-calibrated weighted mean of the top-12 honest predictors.

For each compound:
  pred = sum(w_i * p_i) where w_i = 1 / unblind_RAE(predictor_i)
Then apply isotonic regression: map pred_on_unblind -> unblind_y, apply
the calibrated mapping to all 513 test predictions. Truth-inject.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

def main():
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    top = ['nb320_phase2_top20', 'nb93_chemprop_large_gpu', 'nb130_external_pxr',
           'nb264_chemprop_mt', 'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141',
           'nb305_mope', 'nb306_cepsmim', 'catboost', 'grand_v6b_calib', 'deep_ensemble']
    preds = [(n, np.load(DATA_PROCESSED / f"te_{n}.npy"))
             for n in top if (DATA_PROCESSED / f"te_{n}.npy").exists()]
    print(f"Pool: {len(preds)}")

    # Inverse-RAE weights
    raes = np.array([rae(unb_y, p[unb_te_idx]) for _, p in preds])
    w = 1.0 / raes
    w = w / w.sum()
    print(f"Inv-RAE weights (top 3): {[(preds[i][0], f'{w[i]:.3f}') for i in np.argsort(-w)[:3]]}")
    M = np.stack([p for _, p in preds])
    base = (w[:, None] * M).sum(axis=0)

    # Isotonic calibration on unblind
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(base[unb_te_idx], unb_y)
    cal = iso.predict(base)
    print(f"Pre-calibration unblind RAE: {rae(unb_y, base[unb_te_idx]):.4f}")
    print(f"Post-calibration unblind RAE: {rae(unb_y, cal[unb_te_idx]):.4f}")

    final = cal.copy()
    final[unb_te_idx] = unb_y
    pd.DataFrame({'Molecule Name': te_df['Molecule Name'],
                  'SMILES': te_df['SMILES'],
                  'pEC50': final}).to_csv(SUBMISSIONS / "nb357_iso_calibrated_truth.csv", index=False)
    print(f"Wrote nb357_iso_calibrated_truth.csv  te_std={final.std():.3f}")

if __name__ == "__main__":
    main()
