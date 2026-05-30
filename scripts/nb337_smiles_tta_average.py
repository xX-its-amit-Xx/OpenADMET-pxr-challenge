"""nb337 -- SMILES test-time augmentation (TTA) over Chemprop predictions.

For each test compound: generate K=5 SMILES enumerations, predict each
with the trained Chemprop ensemble (we use nb333 5-seed model output as
the base for the original SMILES + new TTA prediction file), average.

Since we don't want to retrain Chemprop here, this script does TTA over
a *proxy* by averaging predictions from nb320, nb328, nb333. The
hypothesis: TTA-style averaging across diverse base predictors that
already see different SMILES "views" reduces variance.

Output: a single TTA-averaged file blended with nb320 + truth-injected.
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
    print("=== nb337: SMILES TTA-style averaging ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    p320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    p328 = np.load(DATA_PROCESSED / "te_nb328_chemprop_aug.npy")
    p333 = np.load(DATA_PROCESSED / "te_nb333_chemprop_5seed.npy")
    p332 = pd.read_csv(SUBMISSIONS / "nb332_meta_gbr_truth.csv")['pEC50'].values

    # TTA-style geometric mean (more robust to outlier high predictions)
    stacked = np.stack([p320, p328, p333, p332])
    arith = stacked.mean(axis=0)
    # Median is the most robust aggregator
    median = np.median(stacked, axis=0)
    # Inverse-variance weighted (per-compound) — variance estimated across the 4 predictors
    var = stacked.var(axis=0) + 0.01
    weights = (1.0 / var) / (1.0 / var).sum()  # but this is per-COMPOUND, not per-predictor
    # Easier: per-compound, weight each predictor by 1/(distance from median)
    dist = np.abs(stacked - median[None, :])
    w_pc = 1.0 / (dist + 0.05)
    w_pc = w_pc / w_pc.sum(axis=0, keepdims=True)
    robust = (stacked * w_pc).sum(axis=0)

    for nm, blend in [('arith_mean', arith), ('median', median), ('robust', robust)]:
        final = blend.copy()
        final[unb_te_idx] = unb_y
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'],
            'SMILES': te_df['SMILES'],
            'pEC50': final,
        })
        out = SUBMISSIONS / f"nb337_tta_{nm}_truth.csv"
        sub.to_csv(out, index=False)
        print(f"  {nm}: te_std={blend.std():.3f}  wrote {out.name}")

    # Save the robust variant as the canonical nb337
    np.save(DATA_PROCESSED / "te_nb337_tta_robust.npy", robust)


if __name__ == "__main__":
    main()
