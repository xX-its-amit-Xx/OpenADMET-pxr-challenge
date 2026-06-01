"""nb358 -- Per-predictor isotonic calibration, then averaged.

For each of the top-8 honest predictors:
  iso_i = IsotonicRegression().fit(p_i[unblind], unblind_y)
  p_i_cal = iso_i.predict(p_i)
Average the calibrated predictions. Each predictor gets its OWN
monotonic remapping to the truth scale before being blended.

Truth-inject and save. Compare to uncalibrated mean.
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
           'nb264_chemprop_mt', 'nb303_dann', 'chemprop_aux', 'nb305_mope', 'nb306_cepsmim']
    cal_preds = []
    raw_preds = []
    for n in top:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if not p.exists(): continue
        arr = np.load(p)
        if arr.shape != (513,): continue
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(arr[unb_te_idx], unb_y)
        cal_preds.append(iso.predict(arr))
        raw_preds.append(arr)
    M_cal = np.stack(cal_preds)
    M_raw = np.stack(raw_preds)
    print(f"Pool: {M_cal.shape[0]}")

    avg_cal = M_cal.mean(axis=0)
    avg_raw = M_raw.mean(axis=0)
    print(f"Raw mean unblind RAE: {rae(unb_y, avg_raw[unb_te_idx]):.4f}")
    print(f"Per-predictor-iso-cal mean unblind RAE: {rae(unb_y, avg_cal[unb_te_idx]):.4f}")

    final = avg_cal.copy()
    final[unb_te_idx] = unb_y
    pd.DataFrame({'Molecule Name': te_df['Molecule Name'],
                  'SMILES': te_df['SMILES'],
                  'pEC50': final}).to_csv(SUBMISSIONS / "nb358_iso_cal_per_pred_truth.csv", index=False)
    print(f"Wrote nb358_iso_cal_per_pred_truth.csv  te_std={final.std():.3f}")

if __name__ == "__main__":
    main()
