"""chemeleon_score.py — score the CheMeleon test predictions on the 253 unblind (local, after download).

Judges on RAE AND corr-with-nb3200-error (the deploy metric): does the CheMeleon anchor BEAT or ADD to nb3200?
Expectation (chempropembed sink): likely absorbed. Run after downloading chemeleon_test_pred.csv to data/processed/chemeleon/.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.eval import rae

P = "data/processed"


def main():
    pred513 = pd.read_csv(f"{P}/chemeleon/chemeleon_test_pred.csv")["pred"].to_numpy()
    unb = np.load(f"{P}/chemeleon/unb_idx.npy"); y = np.load(f"{P}/chemeleon/unb_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor
    p = pred513[unb]
    base = rae(y, anchor)
    print(f"nb3200 anchor RAE {base:.4f}")
    print(f"CheMeleon standalone RAE {rae(y, p):.4f}")
    print(f"corr(CheMeleon, nb3200 error) {np.corrcoef(p, err)[0,1]:+.3f}")
    bb = base
    for w in np.linspace(0, 1, 41):
        bb = min(bb, rae(y, (1 - w) * anchor + w * p))
    print(f"best blend nb3200+CheMeleon {bb:.4f} (delta {bb-base:+.4f})")
    print("\nGATE: real lever if standalone < nb3200 OR blend delta < ~-0.005 stable.")


if __name__ == "__main__":
    main()
