"""nb322 -- 5-fold cross-validation of nb320 top-50 SLSQP weight stability.

Split the 253 unblind into 5 folds. For each: fit SLSQP weights on 4 folds,
evaluate on 5th. Reports mean +- std of fold RAE and weight stability.
If weight stability is poor, the 0.5609 is overfit to those exact 253.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED


def main():
    print("=== nb322: 5-fold weight stability ===\n")
    te_df_blind = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df_blind['Molecule Name'])}
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    unb_idx = []
    unb_y = []
    for _, r in unb.iterrows():
        if r['Molecule Name'] in name_to_idx:
            unb_idx.append(name_to_idx[r['Molecule Name']])
            unb_y.append(r['pEC50'])
    unb_idx = np.array(unb_idx); unb_y = np.array(unb_y)
    print(f"Unblind: {len(unb_y)}")

    # Load Phase 2 leaderboard top-50
    lb = pd.read_csv(DATA_PROCESSED / "nb320_phase2_all_models_ranked.csv")
    top_names = lb.head(50)['model'].tolist()
    preds = {}
    for n in top_names:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,):
                preds[n] = arr
    names = [n for n in top_names if n in preds]
    print(f"Loaded {len(names)} top models\n")

    # 5-fold split on unblind
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rae = []
    weight_history = []
    for fi, (tr_f, va_f) in enumerate(kf.split(unb_idx)):
        tr_idx = unb_idx[tr_f]; va_idx = unb_idx[va_f]
        tr_y = unb_y[tr_f]; va_y = unb_y[va_f]
        M_tr = np.column_stack([preds[n][tr_idx] for n in names])
        M_va = np.column_stack([preds[n][va_idx] for n in names])
        def loss(w): return rae(tr_y, M_tr @ w)
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0, 1.0)] * len(names)
        best = None
        for seed in range(100):
            rng = np.random.default_rng(seed)
            w0 = rng.dirichlet(np.ones(len(names)))
            res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
            if best is None or res.fun < best.fun: best = res
        va_pred = M_va @ best.x
        va_rae = rae(va_y, va_pred)
        fold_rae.append(va_rae)
        weight_history.append(best.x)
        active = [(n, w) for n, w in zip(names, best.x) if w >= 0.005]
        active.sort(key=lambda x: -x[1])
        print(f"Fold {fi+1}: train_rae={best.fun:.4f}  val_rae={va_rae:.4f}  active={len(active)}")
        for n, w in active[:5]:
            print(f"    {w:.4f}  {n}")

    print(f"\n=== Summary ===")
    print(f"Fold RAE: mean={np.mean(fold_rae):.4f}  std={np.std(fold_rae):.4f}  min={min(fold_rae):.4f}  max={max(fold_rae):.4f}")
    print(f"Full-train RAE (no holdout): 0.5609 (from nb320)")
    print(f"Generalization gap (CV - full-train): {np.mean(fold_rae) - 0.5609:+.4f}")

    # Weight stability: how much do top-5 weights move across folds?
    W = np.stack(weight_history)  # (5, n_models)
    mean_w = W.mean(axis=0); std_w = W.std(axis=0)
    print(f"\nTop 10 models by mean weight (over 5 folds):")
    order = np.argsort(-mean_w)[:10]
    for idx in order:
        print(f"  {names[idx]:<32} mean={mean_w[idx]:.4f}  std={std_w[idx]:.4f}  CV={std_w[idx]/(mean_w[idx]+1e-6):.2f}")


if __name__ == "__main__":
    main()
