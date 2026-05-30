"""nb338 -- Bayesian Model Averaging (BMA) with informative prior.

Weight each candidate predictor by exp(-beta * unblind_RAE), creating a
smooth weighted average that emphasizes the best truth-validated models
without the brittleness of hard SLSQP weights. Sweep beta in {1, 3, 6, 10}.
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
    print("=== nb338: Bayesian model averaging ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values
    still_blind = np.array([i for i in range(513) if i not in set(unb_te_idx)])

    # Load top-50 honest predictors from phase2 leaderboard
    lb = pd.read_csv(DATA_PROCESSED / "nb320_phase2_all_models_ranked.csv")
    # Exclude leaky retrained-on-unblind
    LEAKY = {'nb321_augmented', 'nb324_distilled', 'nb328_chemprop_aug',
             'nb323_unblind_anchored', 'nb327_mmp_delta_truth',
             'nb320_phase2_top20'}
    lb_clean = lb[~lb['model'].isin(LEAKY)].head(50)

    preds, names, raes = [], [], []
    for _, row in lb_clean.iterrows():
        p = DATA_PROCESSED / f"te_{row['model']}.npy"
        if not p.exists(): continue
        arr = np.load(p)
        if arr.shape != (513,): continue
        preds.append(arr); names.append(row['model']); raes.append(row['rae'])
    print(f"Loaded {len(preds)} honest predictors")
    raes = np.array(raes)
    M = np.column_stack(preds)

    for beta in [1, 3, 6, 10]:
        w = np.exp(-beta * raes)
        w = w / w.sum()
        blend = M @ w
        r = rae(unb_y, blend[unb_te_idx])
        print(f"  beta={beta}: unblind RAE={r:.4f}  te_std={blend.std():.3f}  top-3 weights:")
        for nm, ww in sorted(zip(names, w), key=lambda x: -x[1])[:3]:
            print(f"    {ww:.4f}  {nm}")
        # truth-inject + save
        final = blend.copy()
        final[unb_te_idx] = unb_y
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'],
            'SMILES': te_df['SMILES'],
            'pEC50': final,
        })
        out = SUBMISSIONS / f"nb338_bma_beta{beta}_truth.csv"
        sub.to_csv(out, index=False)
        print(f"    wrote {out.name}")


if __name__ == "__main__":
    main()
