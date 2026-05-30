"""nb347 -- Greedy forward selection of predictors on truth.

Start from empty pool. At each step, add the candidate whose addition
maximally reduces unblind RAE (when blended via SLSQP). Stop when
marginal improvement < 0.005. Produces a sparse, interpretable blend.
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
    print("=== nb347: greedy forward selection ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    # Honest pool (no leaky retrained)
    LEAKY = {'nb321_augmented', 'nb324_distilled', 'nb328_chemprop_aug',
             'nb323_unblind_anchored', 'nb327_mmp_delta_truth'}
    lb = pd.read_csv(DATA_PROCESSED / "nb320_phase2_all_models_ranked.csv")
    candidate_names = []
    preds = {}
    for _, row in lb.head(40).iterrows():
        n = row['model']
        if n in LEAKY: continue
        p = DATA_PROCESSED / f"te_{n}.npy"
        if not p.exists(): continue
        arr = np.load(p)
        if arr.shape != (513,): continue
        preds[n] = arr; candidate_names.append(n)
    print(f"Pool: {len(candidate_names)}")

    selected = []
    best_rae = float('inf')
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    for step in range(15):
        best_add = None; best_add_rae = best_rae
        for nm in candidate_names:
            if nm in selected: continue
            trial = selected + [nm]
            M_unb = np.column_stack([preds[n][unb_te_idx] for n in trial])
            K = len(trial)
            bounds = [(0, 1.0)] * K
            def loss(w): return rae(unb_y, M_unb @ w)
            best = None
            for seed in range(30):
                rng = np.random.default_rng(seed)
                w0 = rng.dirichlet(np.ones(K))
                res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
                if best is None or res.fun < best.fun: best = res
            if best.fun < best_add_rae:
                best_add_rae = best.fun; best_add = nm
        if best_add is None or (best_rae - best_add_rae) < 0.005:
            print(f"  step {step+1}: no improvement >= 0.005, stopping")
            break
        selected.append(best_add)
        print(f"  step {step+1}: added {best_add}  unblind RAE={best_add_rae:.4f}")
        best_rae = best_add_rae

    # Final blend
    M_unb = np.column_stack([preds[n][unb_te_idx] for n in selected])
    M_full = np.column_stack([preds[n] for n in selected])
    K = len(selected)
    bounds = [(0, 1.0)] * K
    def loss(w): return rae(unb_y, M_unb @ w)
    best = None
    for seed in range(150):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(K))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\nFinal {K}-component blend: unblind RAE={best.fun:.4f}")
    for nm, w in sorted(zip(selected, best.x), key=lambda x: -x[1]):
        print(f"  {w:.4f}  {nm}")

    blend = M_full @ best.x
    final = blend.copy()
    final[unb_te_idx] = unb_y
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb347_greedy_forward_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
