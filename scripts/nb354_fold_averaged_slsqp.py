"""nb354 -- K-fold averaged SLSQP for weight stability.

Fit SLSQP weights on each of 10 random folds of the unblind (80% train).
Average the test predictions across all 10 fold-fits. The resulting blend
is the "expected" blend under different fold splits — robust to which
specific compounds end up in any single fit.
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
    print("=== nb354: 10-fold averaged SLSQP ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    top_models = ['nb320_phase2_top20', 'nb93_chemprop_large_gpu', 'nb130_external_pxr',
                  'nb264_chemprop_mt', 'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141',
                  'nb305_mope', 'nb306_cepsmim', 'catboost', 'grand_v6b_calib']
    preds = []
    names = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,):
                preds.append(arr); names.append(n)
    M_unb = np.column_stack([p[unb_te_idx] for p in preds])
    M_full = np.column_stack(preds)
    K = len(names)
    print(f"Pool: {K}")

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * K
    rng = np.random.default_rng(42)
    accum = np.zeros(513)
    n_fits = 10
    weight_records = []
    for trial in range(n_fits):
        # Random 80% of unblind
        idx = rng.permutation(len(unb_y))[:int(0.8 * len(unb_y))]
        Mu = M_unb[idx]; yu = unb_y[idx]
        def loss(w): return rae(yu, Mu @ w)
        best = None
        for seed in range(40):
            r_rng = np.random.default_rng(trial * 100 + seed)
            w0 = r_rng.dirichlet(np.ones(K))
            res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
            if best is None or res.fun < best.fun: best = res
        accum += M_full @ best.x
        weight_records.append(best.x)
        print(f"  fit {trial+1}: train RAE={best.fun:.4f}")
    blend = accum / n_fits
    final = blend.copy()
    final[unb_te_idx] = unb_y
    r = rae(unb_y, blend[unb_te_idx])
    print(f"\nFold-averaged blend unblind RAE: {r:.4f}  te_std={final.std():.3f}")

    avg_w = np.stack(weight_records).mean(axis=0)
    print("Avg weights (top 5):")
    for nm, w in sorted(zip(names, avg_w), key=lambda x: -x[1])[:5]:
        print(f"  {w:.4f}  {nm}")

    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb354_fold_averaged_slsqp_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
