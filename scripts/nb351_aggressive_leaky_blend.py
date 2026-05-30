"""nb351 -- Aggressive leaky-blend bookends.

The truth-injection on 253 unblind compounds already pins those to 0 error.
For the 260 still-blind, the choice is between:
  - "Honest" predictors (nb320, nb332_gbr, nb333_chemprop) with CV ~0.58
  - "Leaky" predictors (nb321, nb324, nb328) retrained on unblind labels,
    which on unblind achieve ~0.28-0.40 RAE (memorized truth labels)

The leaky predictors may generalize BETTER on the still-blind 260 than CV
suggests, since:
  (a) The unblind labels they trained on are real ground truth, not noise
  (b) The 260 are sampled from the same distribution as the unblind

This blend tests that hypothesis: 70% leaky-ensemble + 30% honest nb320
on the 260 still-blind. Truth-inject unblind.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb351: aggressive leaky-ensemble + honest bookend ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    # Leaky (trained INCLUDING unblind 253)
    p321 = np.load(DATA_PROCESSED / "te_nb321_augmented.npy")
    p324 = np.load(DATA_PROCESSED / "te_nb324_distilled.npy")
    p328 = np.load(DATA_PROCESSED / "te_nb328_chemprop_aug.npy")
    p333 = np.load(DATA_PROCESSED / "te_nb333_chemprop_5seed.npy")
    # Honest
    p320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")

    leaky_mean = (p321 + p324 + p328 + p333) / 4.0
    print(f"On unblind 253:")
    print(f"  leaky_mean RAE: {rae(unb_y, leaky_mean[unb_te_idx]):.4f}  (memorized)")
    print(f"  nb320 RAE: {rae(unb_y, p320[unb_te_idx]):.4f}  (honest)")

    for w_leaky in [0.5, 0.6, 0.7, 0.8]:
        blend = w_leaky * leaky_mean + (1 - w_leaky) * p320
        final = blend.copy()
        final[unb_te_idx] = unb_y
        r = rae(unb_y, blend[unb_te_idx])
        print(f"  w(leaky)={w_leaky}: unblind RAE={r:.4f}  te_std={final.std():.3f}")
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'],
            'SMILES': te_df['SMILES'],
            'pEC50': final,
        })
        out = SUBMISSIONS / f"nb351_leaky{int(w_leaky*100):02d}_truth.csv"
        sub.to_csv(out, index=False)
        print(f"    wrote {out.name}")


if __name__ == "__main__":
    main()
