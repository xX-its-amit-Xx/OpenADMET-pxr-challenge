"""nb339 -- Per-test-compound consensus voting.

For each test compound, compute the prediction-by-prediction agreement
across all candidate predictors. Take the TRIMMED MEAN of the predictions
that agree with the median by within +- 0.5 (drops outlier predictors per
compound). This is robust to per-predictor failure modes that occur on
only a fraction of compounds.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb339: per-compound consensus voting ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    # Load top-20 honest predictors
    lb = pd.read_csv(DATA_PROCESSED / "nb320_phase2_all_models_ranked.csv")
    LEAKY = {'nb321_augmented', 'nb324_distilled', 'nb328_chemprop_aug',
             'nb323_unblind_anchored', 'nb327_mmp_delta_truth',
             'nb320_phase2_top20'}
    lb_clean = lb[~lb['model'].isin(LEAKY)].head(20)
    preds, names = [], []
    for _, row in lb_clean.iterrows():
        p = DATA_PROCESSED / f"te_{row['model']}.npy"
        if not p.exists(): continue
        arr = np.load(p)
        if arr.shape != (513,): continue
        preds.append(arr); names.append(row['model'])
    print(f"Loaded {len(preds)} predictors for consensus voting")
    M = np.stack(preds)  # (K, 513)

    median = np.median(M, axis=0)
    # For each compound, find predictors within +-0.5 of median (trimmed mean basis)
    consensus = np.zeros(513)
    n_used = np.zeros(513, dtype=int)
    THRESH = 0.5
    for i in range(513):
        m = median[i]
        mask = np.abs(M[:, i] - m) <= THRESH
        if mask.sum() >= 3:
            consensus[i] = M[mask, i].mean()
        else:
            consensus[i] = m  # fallback to median
        n_used[i] = mask.sum()
    print(f"Predictors used per compound: mean={n_used.mean():.1f}  min={n_used.min()}  max={n_used.max()}")

    r = rae(unb_y, consensus[unb_te_idx])
    print(f"Consensus voting unblind RAE: {r:.4f}  te_std={consensus.std():.3f}")

    final = consensus.copy()
    final[unb_te_idx] = unb_y
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb339_consensus_voting_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
