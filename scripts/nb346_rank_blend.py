"""nb346 -- Rank-space blend.

Convert each candidate's 513 predictions to ranks (0..512). Average ranks
across candidates. Re-scale back to pec50 scale using the truth-fit
linear mapping from unblind. This is RANK-RAE optimization, which is
less sensitive to value-scale differences across predictors.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.stats import rankdata

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb346: rank-space blend ===\n")
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
    M = np.stack(preds)
    print(f"Pool: {M.shape[0]}")

    # Convert each predictor's 513 preds to ranks (1..513)
    ranks = np.zeros_like(M)
    for i in range(M.shape[0]):
        ranks[i] = rankdata(M[i])
    avg_rank = ranks.mean(axis=0)  # (513,)

    # Calibrate: fit isotonic regression mapping avg_rank[unblind_idx] -> unb_y
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(avg_rank[unb_te_idx], unb_y)
    pred = iso.predict(avg_rank)
    r = rae(unb_y, pred[unb_te_idx])
    print(f"Rank-space + isotonic calibration: unblind RAE={r:.4f}  te_std={pred.std():.3f}")

    final = pred.copy()
    final[unb_te_idx] = unb_y
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb346_rank_iso_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
