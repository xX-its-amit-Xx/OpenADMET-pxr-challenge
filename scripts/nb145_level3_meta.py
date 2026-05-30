"""nb145 — Level-3 Meta-Stack.

Take OOF predictions from the top meta-models and stack them with a very
shallow XGBoost (max_depth=2, 6-8 inputs). With so few inputs, overfitting
is minimal while the model learns compound-specific blend weights.

Inputs (Level-2 OOF predictions):
  - nb143_oofassay_meta    (OOF+assay, ~0.3186)
  - nb141_xgb_ablation     (best ablation variant, ~0.3186 or similar)
  - nb136_xgb_meta         (OOF+str+assay, 0.3334)
  - nb134_grand_v9         (grand ensemble, 0.3303)
  - nb139_adaptive_blend   (adaptive gating, 0.3544)
  - nb113_counter_delta    (orthogonal: selectivity-based, 0.4446)
  - nb144_grand_v10        (if available)

Strategy: XGBoost + LGBM with max_depth=2-3, very few estimators.
Also tests Ridge regression (linear blend).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy import stats, optimize

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5


def load_meta_oofs(n_tr, y_tr):
    """Load only the top meta-model OOF predictions for level-3 stacking."""
    PRIORITY_STEMS = [
        "nb143_oofassay_meta",
        "nb141_xgb_ablation",
        "nb136_xgb_meta",
        "nb134_grand_v9",
        "nb144_grand_v10",
        "nb139_adaptive_blend",
        "nb142_xgb_calibrated",
        "nb109_calibrated",
        "nb113_counter_delta",
    ]
    results = []
    for stem in PRIORITY_STEMS:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        if not oof_p.exists():
            print(f"  SKIP (not found): {stem}")
            continue
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists(): break
        else:
            print(f"  SKIP (no test): {stem}")
            continue
        try:
            oof = np.load(oof_p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
            r = rae(y_tr, oof)
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
            print(f"  LOADED: {stem:40s}  RAE={r:.4f}  ratio={ratio:.2f}")
        except Exception as e:
            print(f"  ERROR: {stem} — {e}")
    return results


def optimize_weights_slsqp(oof_mat, y_tr, w0=None):
    k = oof_mat.shape[1]
    if w0 is None: w0 = np.ones(k) / k
    def objective(w):
        return np.mean(np.abs(y_tr - oof_mat @ w)) / np.mean(np.abs(y_tr - y_tr.mean()))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * k
    res = optimize.minimize(objective, w0, method="SLSQP", bounds=bounds,
                            constraints=constraints, options={"ftol": 1e-9, "maxiter": 1000})
    return res.fun, res.x


def main():
    print("=== nb145: Level-3 Meta-Stack ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Loading level-2 OOF predictions...")
    models = load_meta_oofs(n_tr, y_tr)
    n_mod = len(models)
    print(f"\nLoaded {n_mod} meta-models for L3 stacking")

    if n_mod < 2:
        print("Not enough models. Run nb143+ first.")
        return

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    stems   = [m["stem"] for m in models]

    # Baseline: simple SLSQP on full OOF (no CV, just for reference)
    print(f"\nBaseline SLSQP blend ({n_mod} models):")
    r_slsqp, w_slsqp = optimize_weights_slsqp(oof_mat, y_tr)
    print(f"  SLSQP: RAE={r_slsqp:.4f}")
    for stem, w in zip(stems, w_slsqp):
        print(f"    {stem:40s}  w={w:.3f}")

    results = {"slsqp_blend": (r_slsqp, oof_mat @ w_slsqp, te_mat @ w_slsqp, 1.0)}

    # XGBoost level-3 (max_depth=2, very shallow)
    for depth, n_est in [(2, 100), (3, 150), (2, 200)]:
        label = f"XGB depth={depth} n={n_est}"
        print(f"\n{label}:")
        params = dict(n_estimators=n_est, max_depth=depth, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                      reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
                      device="cpu", verbosity=0, n_jobs=4, random_state=SEED)
        oof = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = xgb.XGBRegressor(**params)
            m.fit(oof_mat[tr_idx], y_tr[tr_idx], verbose=False)
            oof[va_idx] = m.predict(oof_mat[va_idx])
        m_full = xgb.XGBRegressor(**params)
        m_full.fit(oof_mat, y_tr, verbose=False)
        te_pred = m_full.predict(te_mat)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        print(f"  RAE={r:.4f}  ratio={ratio:.2f}")
        results[label] = (r, oof, te_pred, ratio)

    # LGBM level-3 (very shallow)
    for nl in [8, 16]:
        label = f"LGBM num_leaves={nl}"
        print(f"\n{label}:")
        lgb_params = dict(n_estimators=150, num_leaves=nl, learning_rate=0.05,
                          min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=0.1, verbose=-1, n_jobs=4, random_state=SEED)
        oof = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = lgb.train(lgb_params, lgb.Dataset(oof_mat[tr_idx], label=y_tr[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(oof_mat[va_idx])
        m_full = lgb.train(lgb_params, lgb.Dataset(oof_mat, label=y_tr),
                           callbacks=[lgb.log_evaluation(-1)])
        te_pred = m_full.predict(te_mat)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        print(f"  RAE={r:.4f}  ratio={ratio:.2f}")
        results[label] = (r, oof, te_pred, ratio)

    # Ridge (linear combo)
    for alpha in [0.01, 0.1, 1.0]:
        label = f"Ridge alpha={alpha}"
        oof = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            ridge = Ridge(alpha=alpha, positive=True)
            ridge.fit(oof_mat[tr_idx], y_tr[tr_idx])
            oof[va_idx] = ridge.predict(oof_mat[va_idx])
        ridge_full = Ridge(alpha=alpha, positive=True)
        ridge_full.fit(oof_mat, y_tr)
        te_pred = ridge_full.predict(te_mat)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        results[label] = (r, oof, te_pred, ratio)

    # Summary
    print(f"\n=== Level-3 Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= 0.58 else "WARN"
        print(f"  {k:40s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= 0.58}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBest L3: {best_label}  OOF RAE={best_r:.4f}")

    # Final blend of best L3 with best L2
    best_l2_r = min(m["rae"] for m in models)
    best_l2_oof = models[0]["oof"]  # sorted by RAE
    best_l2_te  = models[0]["te"]
    for alpha in [0.3, 0.5, 0.7]:
        b_oof = alpha * best_oof + (1-alpha) * best_l2_oof
        print(f"  L3({alpha:.1f}) + L2-best({1-alpha:.1f}): RAE={rae(y_tr, b_oof):.4f}")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb145_level3_meta.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb145_level3_meta.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "145_level3_meta.csv", index=False)
    print(f"\nSaved: submissions/145_level3_meta.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
