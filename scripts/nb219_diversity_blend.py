"""nb219 -- SLSQP blend with diversity injection from nb215/216/217.

The new models (nb215/216/217) are individually weaker (RAE ~0.5) but have
correlation 0.83-0.86 with nb212 vs 0.99+ for existing pool members. nb216 has
ratio=0.66 and nb217 has ratio=0.61, well above the 0.58 constraint -- they
could serve as ratio inflators, freeing weight from nb205 onto lower-RAE models.

Pool:
- nb211, nb205, nb157, nb167, nb169 (existing nb212 pool)
- nb215, nb216, nb217 (new diverse models)
- nb218 (residual learner, if available)

SLSQP with 8000 starts, ratio constraint >= 0.58, nb212 weights as warm start.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
SEED = 42
COLLAPSE_THRESH = 0.58
PREV_BEST = 0.296172

t0 = time.time()

# Pool composition
EXISTING_POOL = [
    ("nb211_div15_chemprop_blend", 0.5276),  # nb212 weights
    ("nb205_ratio_inflate",        0.2686),
    ("nb157_optuna_lgbm_mae",      0.1383),
    ("nb167_xgboost_mae",          0.0379),
    ("nb169_rf_et_mae",            0.0277),
]
NEW_DIVERSE = [
    "nb215_chemist_features",
    "nb216_aux_stack",
    "nb217_pxr_modes",
    "nb218_residual_blend",     # optional, may not exist yet
]


def constrained_slsqp(X_tr, X_te, y_tr, n_starts=8000, prev_best=PREV_BEST, tag="",
                      warm_weight_vecs=None):
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
        return pred_te.std() / std_tr - COLLAPSE_THRESH

    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1},
        {"type": "ineq", "fun": ratio_con},
    ]
    bounds = [(0, 1)] * n_m
    rng = np.random.default_rng(SEED)

    def _try_start(w0):
        nonlocal best_r, best_w
        res = minimize(obj, w0, jac=obj_grad, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-8})
        if res.success and ratio_con(res.x) >= -1e-6:
            r = obj(res.x)
            if r < best_r:
                best_r, best_w = r, res.x.copy()

    if warm_weight_vecs is not None:
        for w0 in warm_weight_vecs:
            if len(w0) == n_m:
                _try_start(w0)

    # Single-model starts
    for i in range(n_m):
        for main_w in [0.5, 0.7, 0.9, 0.95]:
            w0 = np.ones(n_m) * (1.0 - main_w) / max(n_m - 1, 1)
            w0[i] = main_w
            w0 = np.clip(w0, 0, 1); w0 /= w0.sum()
            _try_start(w0)

    # Random Dirichlet starts
    for _ in range(n_starts):
        # Mix of uniform and skewed Dirichlet
        if rng.random() < 0.5:
            w0 = rng.dirichlet(np.ones(n_m))
        else:
            w0 = rng.dirichlet(np.ones(n_m) * 0.3)  # more concentrated
        _try_start(w0)

    if best_w is None:
        return 1e9, None, None, 0
    oof_b = X_tr @ best_w; te_b = X_te @ best_w
    ratio = te_b.std() / oof_b.std()
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb219: Diversity blend (existing pool + new diverse models) ===\n", flush=True)
    print(f"nb212 best: {PREV_BEST}  ratio=0.5800\n", flush=True)

    tr_df = load_train(); te_df = load_test()
    y_tr = tr_df["pec50"].values.astype(np.float64)

    pool_oofs, pool_tes, pool_names = [], [], []
    nb212_warm = []

    print("Loading existing pool (nb212 components)...", flush=True)
    for stem, w in EXISTING_POOL:
        op = DATA_PROCESSED / f"oof_{stem}.npy"; tp = DATA_PROCESSED / f"te_{stem}.npy"
        if not op.exists():
            print(f"  MISSING: {stem}"); continue
        o = np.load(op).flatten().astype(np.float64)
        t = np.load(tp).flatten().astype(np.float64)
        pool_oofs.append(o); pool_tes.append(t); pool_names.append(stem)
        nb212_warm.append(w)
        print(f"  {stem}: RAE={rae(y_tr,o):.4f}  ratio={t.std()/o.std():.4f}", flush=True)

    print("\nLoading new diverse models...", flush=True)
    for stem in NEW_DIVERSE:
        op = DATA_PROCESSED / f"oof_{stem}.npy"; tp = DATA_PROCESSED / f"te_{stem}.npy"
        if not op.exists():
            print(f"  SKIP: {stem} (not yet built)"); continue
        o = np.load(op).flatten().astype(np.float64)
        t = np.load(tp).flatten().astype(np.float64)
        # Sanity check
        if len(o) != len(y_tr) or len(t) != len(te_df):
            print(f"  SKIP: {stem} (wrong shape)"); continue
        # Replace any nan
        o = np.where(np.isfinite(o), o, np.nanmean(o))
        t = np.where(np.isfinite(t), t, np.nanmean(t))
        pool_oofs.append(o); pool_tes.append(t); pool_names.append(stem)
        ratio = t.std()/o.std()
        print(f"  {stem}: RAE={rae(y_tr,o):.4f}  ratio={ratio:.4f}", flush=True)

    n_pool = len(pool_oofs)
    print(f"\nFinal pool: {n_pool} models", flush=True)

    X_tr = np.column_stack(pool_oofs)
    X_te = np.column_stack(pool_tes)

    # Warm starts: nb212 weights padded with zeros for new models
    w_nb212 = np.zeros(n_pool)
    n_existing = len(nb212_warm)
    w_nb212[:n_existing] = nb212_warm
    w_nb212 /= w_nb212.sum()
    warm = [w_nb212]

    # Also try nb212 + small weight on each new model
    for i in range(n_existing, n_pool):
        for new_w in [0.02, 0.05, 0.10, 0.15]:
            w0 = np.zeros(n_pool)
            w0[:n_existing] = np.array(nb212_warm) * (1 - new_w)
            w0[i] = new_w
            w0 /= w0.sum()
            warm.append(w0)

    print(f"\n--- SLSQP: {n_pool} models, 8000 starts + {len(warm)} warm starts ---", flush=True)
    r_best, oof_best, te_best, ratio_best = constrained_slsqp(
        X_tr, X_te, y_tr, n_starts=8000, prev_best=PREV_BEST, tag="nb219",
        warm_weight_vecs=warm)

    flag = "PASS" if ratio_best >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio_best >= COLLAPSE_THRESH and r_best < PREV_BEST) else ""
    print(f"\n=== Summary ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"  nb212 (prev best): RAE={PREV_BEST:.6f}  ratio=0.5800  [PASS]", flush=True)
    print(f"  nb219 SLSQP:       RAE={r_best:.6f}  ratio={ratio_best:.4f}  [{flag}]{beat}", flush=True)

    if oof_best is not None:
        # Find weights via re-solve - but we don't have them stored. Print top by inspection
        # Re-solve for the best weights
        # Just print what we know: which models contributed
        # Actually we need to re-get the weights from the optimization
        pass

    # Save
    if oof_best is not None:
        out_stem = "nb219_diversity_blend"
        np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof_best)
        np.save(DATA_PROCESSED / f"te_{out_stem}.npy",  te_best)
        sub = pd.DataFrame({
            "SMILES": te_df["smiles"].values,
            "Molecule Name": te_df["name"].values,
            "pEC50": te_best,
        })
        sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
        if r_best < PREV_BEST and ratio_best >= COLLAPSE_THRESH:
            print(f"\n*** SAVED NEW BEST: {SUBMISSIONS}/{out_stem}.csv ***", flush=True)
        else:
            print(f"\nSaved baseline: {SUBMISSIONS}/{out_stem}.csv", flush=True)


if __name__ == "__main__":
    main()
