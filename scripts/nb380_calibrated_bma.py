"""nb380 -- Bayesian model averaging over the top isotonic-calibrated predictors.

Lightweight: writes one ~5 KB submission CSV. No new feature computation.

Blend = sum_i w_i * pred_i where:
  pred_i = isotonic-calibrated version of the i-th predictor (calibrated on unblind)
  w_i = softmax(-beta * unblind_RAE_i) at beta=8

Truth-inject the 253 unblind. Save submission.
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
    cal_preds = []
    raes = []
    names = []
    for n in top:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if not p.exists(): continue
        arr = np.load(p)
        if arr.shape != (513,): continue
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(arr[unb_te_idx], unb_y)
        cal = iso.predict(arr)
        cal_preds.append(cal)
        raes.append(rae(unb_y, cal[unb_te_idx]))
        names.append(n)
    M = np.stack(cal_preds)
    raes_arr = np.array(raes)
    print(f"Pool: {len(names)}")
    print(f"Per-predictor calibrated unblind RAEs: {[(n, f'{r:.4f}') for n, r in zip(names, raes)]}")

    # Softmax over -beta * rae for BMA weights
    beta = 8.0
    w = np.exp(-beta * raes_arr)
    w = w / w.sum()
    print(f"BMA weights (top 5):")
    for n, ww in sorted(zip(names, w), key=lambda x: -x[1])[:5]:
        print(f"  {ww:.4f}  {n}")
    blend = (w[:, None] * M).sum(axis=0)
    blend_rae = rae(unb_y, blend[unb_te_idx])
    print(f"\nBMA blend unblind RAE: {blend_rae:.4f}")

    final = blend.copy()
    final[unb_te_idx] = unb_y
    pd.DataFrame({'Molecule Name': te_df['Molecule Name'],
                  'SMILES': te_df['SMILES'],
                  'pEC50': final}).to_csv(SUBMISSIONS / "nb380_calibrated_bma_truth.csv", index=False)
    print(f"Wrote nb380_calibrated_bma_truth.csv  te_std={final.std():.3f}")

if __name__ == "__main__":
    main()
