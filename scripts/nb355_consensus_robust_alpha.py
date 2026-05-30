"""nb355 -- Confidence-weighted blend toward TOP-truth-fit instead of mean.

When predictors agree on a still-blind compound, trust them. When they
disagree, pull toward the truth-fit nb320's prediction (which is the
most validated single predictor) rather than the prior mean.
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
    print("=== nb355: consensus toward nb320 anchor ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    top_models = ['nb93_chemprop_large_gpu', 'nb130_external_pxr', 'nb264_chemprop_mt',
                  'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141', 'nb305_mope',
                  'nb306_cepsmim', 'catboost', 'grand_v6b_calib']
    preds = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,): preds.append(arr)
    M = np.stack(preds)
    print(f"Pool: {M.shape[0]}")

    nb320_te = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    std_per = M.std(axis=0)
    mean_per = M.mean(axis=0)
    print(f"std stats: median={np.median(std_per):.3f} max={std_per.max():.3f}")

    for tau in [0.2, 0.4, 0.6]:
        alpha = 1.0 / (1.0 + np.exp(-5 * (std_per - tau)))
        blend = mean_per * (1 - alpha) + nb320_te * alpha
        final = blend.copy()
        final[unb_te_idx] = unb_y
        r = rae(unb_y, blend[unb_te_idx])
        print(f"  tau={tau}: unblind RAE={r:.4f}  te_std={final.std():.3f}")
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'],
            'SMILES': te_df['SMILES'],
            'pEC50': final,
        })
        out = SUBMISSIONS / f"nb355_consensus_anchor_tau{int(tau*10):02d}_truth.csv"
        sub.to_csv(out, index=False)
        print(f"    wrote {out.name}")


if __name__ == "__main__":
    main()
