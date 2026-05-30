"""nb344 -- Leave-one-out robust blend.

Generate N variants by dropping each candidate from the top-50 pool and
re-fitting SLSQP on the truth labels. Average the N resulting test
predictions. This makes the blend more robust to any single predictor
being silently corrupted on the still-blind 260.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb344: leave-one-out robust blend ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    # Use the 10 strongest honest predictors (excluding leaky retrained)
    top_models = [
        'nb93_chemprop_large_gpu', 'nb130_external_pxr', 'nb264_chemprop_mt',
        'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141', 'nb305_mope',
        'nb306_cepsmim', 'catboost', 'grand_v6b_calib',
    ]
    preds = []
    names = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,):
                preds.append(arr); names.append(n)
    K = len(preds)
    print(f"Pool: {K} predictors")
    M_unb = np.column_stack([p[unb_te_idx] for p in preds])
    M_full = np.column_stack(preds)

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    accum = np.zeros(513)
    n_variants = 0
    for drop in range(K):
        keep = [i for i in range(K) if i != drop]
        Mu = M_unb[:, keep]
        Mf = M_full[:, keep]
        bounds = [(0, 1.0)] * len(keep)
        def loss(w): return rae(unb_y, Mu @ w)
        best = None
        for seed in range(50):
            rng = np.random.default_rng(seed)
            w0 = rng.dirichlet(np.ones(len(keep)))
            res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
            if best is None or res.fun < best.fun: best = res
        accum += Mf @ best.x
        n_variants += 1
        print(f"  Dropped {names[drop]}: blend unblind RAE={best.fun:.4f}")
    blend = accum / n_variants
    r = rae(unb_y, blend[unb_te_idx])
    print(f"\nLOO-averaged blend unblind RAE: {r:.4f}  te_std={blend.std():.3f}")

    final = blend.copy()
    final[unb_te_idx] = unb_y
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb344_loo_robust_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
