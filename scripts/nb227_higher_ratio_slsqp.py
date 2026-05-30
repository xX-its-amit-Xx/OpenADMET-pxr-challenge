"""nb227 -- SLSQP with higher collapse threshold (raise from 0.58 to {0.65, 0.70}).

Hypothesis: nb224's te_std=0.598 was binding at ratio=0.58, meaning the
unconstrained SLSQP wanted to predict even more conservatively. The actual
test set likely has train-like variance (std~1.12), so our predictions are
over-compressed. By forcing a higher ratio, we get predictions with higher
variance — at the cost of slightly higher OOF RAE.

Try ratio thresholds:
  - 0.65 (mild tightening; OOF should bump slightly)
  - 0.70 (moderate; te_std → ~0.72 if oof_std stays ~1.03)
  - 0.80 (aggressive)

If the leaderboard RAE responds positively to higher te_std, this is the
direct optimization path. The post-hoc rescaling we queued (224_match) is
the alternative test.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

sys.path.insert(0, str(Path(__file__).parent))
from nb197_dense_grid import build_candidates, load_pool, NB188_POOL, SEED

N_FOLDS = 5


def constrained_slsqp_with_threshold(X_tr, X_te, y_tr, threshold, n_starts=1500, prev_best=999):
    """Constrained SLSQP with custom ratio threshold."""
    n_m = X_tr.shape[1]
    n_tr = len(y_tr)
    best_r, best_w = 1e9, None

    def obj(w):
        return rae(y_tr, X_tr @ w)

    def obj_grad(w):
        pred = X_tr @ w
        mae_mean = np.abs(y_tr - y_tr.mean()).mean()
        return X_tr.T @ np.sign(pred - y_tr) / (n_tr * mae_mean)

    def ratio_con(w):
        pred_tr = X_tr @ w; pred_te = X_te @ w
        std_tr = pred_tr.std()
        if std_tr < 1e-9: return -1.0
        return pred_te.std() / std_tr - threshold

    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1},
        {"type": "ineq", "fun": ratio_con},
    ]
    bounds = [(0, 1)] * n_m
    rng = np.random.default_rng(SEED)

    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(n_m))
        res = minimize(obj, w0, jac=obj_grad, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-11})
        if res.success and ratio_con(res.x) >= -1e-6:
            r = obj(res.x)
            if r < best_r:
                best_r, best_w = r, res.x.copy()

    if best_w is None:
        return 1e9, None, None, 0

    oof_b = X_tr @ best_w; te_b = X_te @ best_w
    ratio = te_b.std() / oof_b.std()
    print(f"  threshold={threshold}: n={n_m}  RAE={best_r:.6f}  ratio={ratio:.4f}  te_std={te_b.std():.4f}")
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb227: SLSQP with higher collapse threshold ===\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    base_oofs, base_tes, base_stems = load_pool(n_tr)
    oof183 = np.load(DATA_PROCESSED / "oof_nb183_qreg_poly10.npy").astype(np.float64).flatten()
    te183  = np.load(DATA_PROCESSED / "te_nb183_qreg_poly10.npy").astype(np.float64).flatten()
    oof183 = np.where(np.isfinite(oof183), oof183, np.nanmean(oof183))
    te183  = np.where(np.isfinite(te183),  te183,  np.nanmean(te183))
    base_oofs.insert(0, oof183); base_tes.insert(0, te183)
    base_stems.insert(0, "nb183_qreg_poly10")

    # nb224 pool: NB188 + nb219 + nb228
    POOL_PLUS = list(NB188_POOL) + ["nb219_aug_30pct", "nb228_medchem"]
    pool_oofs_tr, pool_oofs_te, pool_names = [], [], []
    for stem in POOL_PLUS:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        te_p  = DATA_PROCESSED / f"te_{stem}.npy"
        if not oof_p.exists(): continue
        oof_m = np.load(oof_p).astype(np.float64).flatten()
        te_m  = np.load(te_p).astype(np.float64).flatten()
        if len(oof_m) != n_tr or len(te_m) != len(te_df): continue
        oof_m = np.where(np.isfinite(oof_m), oof_m, np.nanmean(oof_m))
        te_m  = np.where(np.isfinite(te_m),  te_m,  np.nanmean(te_m))
        pool_oofs_tr.append(oof_m); pool_oofs_te.append(te_m); pool_names.append(stem)

    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    alpha_dense_15 = sorted(set([round(a, 5) for a in
        list(np.linspace(0.0005, 0.004, 30)) + [0.005, 0.006, 0.007, 0.008, 0.009]]))
    alpha_high = [0.01, 0.015, 0.02, 0.03]

    cand_a_oofs, cand_a_tes, _ = build_candidates(
        base_oofs, base_tes, base_stems, y_tr, splits, poly2,
        div_k_list=[15, 20], alpha_list=alpha_dense_15 + alpha_high)
    cand_a_25_oofs, cand_a_25_tes, _ = build_candidates(
        base_oofs, base_tes, base_stems, y_tr, splits, poly2,
        div_k_list=[25], alpha_list=[0.008, 0.01, 0.015, 0.02, 0.03])
    all_cands_oofs = cand_a_oofs + cand_a_25_oofs
    all_cands_tes  = cand_a_tes  + cand_a_25_tes
    print(f"\nTotal candidates: {len(all_cands_oofs)}")

    X_tr = np.column_stack(pool_oofs_tr + all_cands_oofs)
    X_te = np.column_stack(pool_oofs_te + all_cands_tes)
    print(f"Matrix shape: train={X_tr.shape}  test={X_te.shape}\n")

    # Run SLSQP at multiple thresholds
    te_raw = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smap = dict(zip(te_raw["Molecule Name"], te_raw["SMILES"]))

    for thr in [0.65, 0.70, 0.75, 0.80]:
        print(f"\n--- threshold = {thr} ---")
        r, oof_b, te_b, ratio = constrained_slsqp_with_threshold(
            X_tr, X_te, y_tr, thr, n_starts=1000, prev_best=999)
        if oof_b is None:
            print(f"  No solution at threshold {thr}")
            continue
        name = f"227_thr{int(thr*100):02d}"
        np.save(DATA_PROCESSED / f"oof_{name}.npy", oof_b)
        np.save(DATA_PROCESSED / f"te_{name}.npy", te_b)
        sub = pd.DataFrame({
            "SMILES": [smap.get(n, "") for n in te_df["name"]],
            "Molecule Name": te_df["name"],
            "pEC50": te_b,
        })
        sub.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
        print(f"  Saved {name}.csv (OOF={r:.4f}, ratio={ratio:.3f}, te_std={te_b.std():.3f})")


if __name__ == "__main__":
    main()
