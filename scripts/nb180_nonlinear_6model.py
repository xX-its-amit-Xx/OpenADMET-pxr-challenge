"""nb180 — Nonlinear meta-learner on the 6-model SLSQP ensemble.

The linear blend (0.3001) uses exactly 6 models with optimized weights:
  nb167_xgboost_mae      w=0.347-0.354
  nb156_catboost_mae     w=0.228-0.233
  nb154_lgbm_mae_filtered w=0.140-0.146
  nb162_mixed_pool       w=0.122-0.125
  nb165_multiseed_162c   w=0.051-0.082
  nb149_meta_maeloss     w=0.075-0.085

Can a nonlinear function of these 6 OOF predictions beat the linear blend?

With 6 features and 4139 samples, overfitting is minimal. This tests whether
the 6-model ensemble errors have nonlinear structure that can be exploited.

Tests:
  A: LGBM MAE (low depth, high regularization) on 6 OOF features
  B: CatBoost MAE depth=4 on 6 OOF features
  C: XGBoost MAE depth=3 on 6 OOF features
  D: Polynomial features degree=2 + Ridge on 6 OOF features
  E: LGBM + CatBoost + XGB blend (3-way on 6 features)
  F: Add log/exp transformations of the 6 features
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58
SEED = 42

# The 6 SLSQP-selected models (confirmed by nb175/nb176)
SIX_MODELS = [
    "nb167_xgboost_mae",
    "nb156_catboost_mae",
    "nb154_lgbm_mae_filtered",
    "nb162_mixed_pool",
    "nb165_multiseed_162c",
    "nb149_meta_maeloss",
]


def load_six_models(n_tr):
    oofs = []; tes = []; loaded = []
    for stem in SIX_MODELS:
        p = DATA_PROCESSED / f"oof_{stem}.npy"
        if not p.exists():
            print(f"  MISSING: {stem}")
            continue
        for pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{pref}{stem}.npy"
            if te_p.exists(): break
        else:
            print(f"  MISSING TE: {stem}")
            continue
        oof = np.load(p).astype(np.float64).flatten()
        te  = np.load(te_p).astype(np.float64).flatten()
        if len(oof) != n_tr: continue
        oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
        te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
        oofs.append(oof); tes.append(te); loaded.append(stem)
    return np.column_stack(oofs), np.column_stack(tes), loaded


def cv_model(model_cls, params, X_tr, y_tr, X_te, splits, label, seeds=(42, 123, 456, 789, 1234)):
    n_tr = len(y_tr)
    seed_oofs = []; seed_tes = []
    for seed in seeds:
        if hasattr(model_cls, 'fit'):
            # sklearn-style: set random_state
            params_s = {**params}
            if 'random_state' in params_s: params_s['random_state'] = seed
            if 'random_seed' in params_s:  params_s['random_seed']  = seed
            oof_s = np.full(n_tr, np.nan)
            for fold, (tr_idx, va_idx) in enumerate(splits):
                m = model_cls(**params_s)
                m.fit(X_tr[tr_idx], y_tr[tr_idx])
                oof_s[va_idx] = m.predict(X_tr[va_idx])
            m_f = model_cls(**params_s)
            m_f.fit(X_tr, y_tr)
            seed_oofs.append(oof_s)
            seed_tes.append(m_f.predict(X_te))
        print(f"    seed {seed}: RAE={rae(y_tr, seed_oofs[-1]):.4f}", flush=True)
    oof_ms = np.mean(seed_oofs, axis=0); te_ms = np.mean(seed_tes, axis=0)
    r = rae(y_tr, oof_ms); ratio = te_ms.std() / oof_ms.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label}] RAE={r:.4f}  ratio={ratio:.3f}  [{flag}]", flush=True)
    return r, oof_ms, te_ms, ratio


def cv_lgbm(params, X_tr, y_tr, X_te, splits, label, seeds=(42, 123, 456, 789, 1234)):
    n_tr = len(y_tr)
    seed_oofs = []; seed_tes = []
    for seed in seeds:
        params_s = {**params, "random_state": seed}
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = lgb.train(params_s, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
            oof_s[va_idx] = m.predict(X_tr[va_idx])
        m_f = lgb.train(params_s, lgb.Dataset(X_tr, label=y_tr),
                        callbacks=[lgb.log_evaluation(-1)])
        seed_oofs.append(oof_s); seed_tes.append(m_f.predict(X_te))
        print(f"    seed {seed}: RAE={rae(y_tr, oof_s):.4f}", flush=True)
    oof_ms = np.mean(seed_oofs, axis=0); te_ms = np.mean(seed_tes, axis=0)
    r = rae(y_tr, oof_ms); ratio = te_ms.std() / oof_ms.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label}] RAE={r:.4f}  ratio={ratio:.3f}  [{flag}]", flush=True)
    return r, oof_ms, te_ms, ratio


def main():
    print("=== nb180: Nonlinear meta-learner on 6 SLSQP models ===\n")
    print("Linear blend (0.3001) — can nonlinear beat it?\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    X6_tr, X6_te, loaded = load_six_models(n_tr)
    print(f"Loaded {len(loaded)} of 6 models: {loaded}")
    print(f"Feature matrix: train={X6_tr.shape}, test={X6_te.shape}")

    # Linear blend reference
    from scipy.optimize import minimize
    def obj(w):
        return np.mean(np.abs(y_tr - X6_tr@w))/np.mean(np.abs(y_tr-y_tr.mean()))
    res = minimize(obj, np.ones(6)/6, method='SLSQP',
                   bounds=[(0,1)]*6, constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                   options={'ftol':1e-10})
    oof_linear = X6_tr@res.x; te_linear = X6_te@res.x
    ratio_linear = te_linear.std()/oof_linear.std()
    print(f"Linear SLSQP reference: RAE={res.fun:.6f}  ratio={ratio_linear:.3f}")

    # Extended features: original 6 + mean + std + pairwise interactions
    X6_ext_tr = np.hstack([
        X6_tr,
        X6_tr.mean(axis=1, keepdims=True),
        X6_tr.std(axis=1, keepdims=True),
        X6_tr.max(axis=1, keepdims=True) - X6_tr.min(axis=1, keepdims=True),
    ])
    X6_ext_te = np.hstack([
        X6_te,
        X6_te.mean(axis=1, keepdims=True),
        X6_te.std(axis=1, keepdims=True),
        X6_te.max(axis=1, keepdims=True) - X6_te.min(axis=1, keepdims=True),
    ])
    print(f"Extended features: {X6_ext_tr.shape}")

    results = {}

    # --- A: LGBM MAE on 6 OOF features ---
    print(f"\n--- A: LGBM MAE on 6 features (low depth) ---")
    lgbm_params_a = dict(
        objective="regression_l1", metric="mae",
        n_estimators=500, num_leaves=8, learning_rate=0.02,
        min_child_samples=20, subsample=0.8, colsample_bytree=1.0,
        reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=4,
    )
    r_a, oof_a, te_a, ratio_a = cv_lgbm(lgbm_params_a, X6_tr, y_tr, X6_te, splits, "LGBM_6feat")
    if ratio_a >= COLLAPSE_THRESH:
        results["A_lgbm_6"] = (r_a, oof_a, te_a, ratio_a)

    # --- A2: LGBM MAE on extended features ---
    print(f"\n--- A2: LGBM MAE on extended features ({X6_ext_tr.shape[1]}) ---")
    r_a2, oof_a2, te_a2, ratio_a2 = cv_lgbm(lgbm_params_a, X6_ext_tr, y_tr, X6_ext_te, splits, "LGBM_6ext")
    if ratio_a2 >= COLLAPSE_THRESH:
        results["A2_lgbm_6ext"] = (r_a2, oof_a2, te_a2, ratio_a2)

    # --- B: CatBoost depth=3 on 6 features ---
    print(f"\n--- B: CatBoost depth=3 on 6 features ---")
    cat_params_b = dict(
        loss_function="MAE", eval_metric="MAE", iterations=600,
        learning_rate=0.02, depth=3, l2_leaf_reg=10.0,
        subsample=0.8, thread_count=4, allow_writing_files=False, verbose=0,
    )
    seeds_cat = [42, 123, 456, 789, 1234]
    n_tr_ = n_tr
    oofs_b = []; tes_b = []
    for seed in seeds_cat:
        oof_s = np.full(n_tr_, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = CatBoostRegressor(**{**cat_params_b, "random_seed": seed})
            m.fit(X6_tr[tr_idx], y_tr[tr_idx])
            oof_s[va_idx] = m.predict(X6_tr[va_idx])
        m_f = CatBoostRegressor(**{**cat_params_b, "random_seed": seed})
        m_f.fit(X6_tr, y_tr)
        oofs_b.append(oof_s); tes_b.append(m_f.predict(X6_te))
        print(f"    seed {seed}: RAE={rae(y_tr, oof_s):.4f}", flush=True)
    oof_b = np.mean(oofs_b, axis=0); te_b = np.mean(tes_b, axis=0)
    r_b = rae(y_tr, oof_b); ratio_b = te_b.std()/oof_b.std()
    flag_b = "PASS" if ratio_b >= COLLAPSE_THRESH else "FAIL"
    print(f"  [CatBoost_d3_6feat] RAE={r_b:.4f}  ratio={ratio_b:.3f}  [{flag_b}]")
    if ratio_b >= COLLAPSE_THRESH:
        results["B_cat_6"] = (r_b, oof_b, te_b, ratio_b)

    # --- C: XGBoost depth=2 on 6 features ---
    print(f"\n--- C: XGBoost depth=2 on 6 features ---")
    r_c, oof_c, te_c, ratio_c = cv_model(
        XGBRegressor,
        dict(objective="reg:absoluteerror", n_estimators=500, learning_rate=0.02,
             max_depth=2, subsample=0.8, reg_lambda=5.0, random_state=42,
             n_jobs=4, verbosity=0),
        X6_tr, y_tr, X6_te, splits, "XGB_d2_6feat",
    )
    if ratio_c >= COLLAPSE_THRESH:
        results["C_xgb_6"] = (r_c, oof_c, te_c, ratio_c)

    # --- D: Polynomial degree=2 + Ridge ---
    print(f"\n--- D: Degree-2 polynomial + Ridge on 6 features ---")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X6_poly_tr = poly.fit_transform(X6_tr)
    X6_poly_te = poly.transform(X6_te)
    print(f"  Polynomial features: {X6_poly_tr.shape}")
    # Ridge CV on polynomial features
    best_r_d = 1e9; best_alpha_d = 1.0
    for alpha_ridge in [0.001, 0.01, 0.1, 1.0, 10.0]:
        oof_d = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = Ridge(alpha=alpha_ridge)
            m.fit(X6_poly_tr[tr_idx], y_tr[tr_idx])
            oof_d[va_idx] = m.predict(X6_poly_tr[va_idx])
        r_d = rae(y_tr, oof_d)
        if r_d < best_r_d:
            best_r_d = r_d; best_alpha_d = alpha_ridge
    # Final model with best alpha
    oof_d = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = Ridge(alpha=best_alpha_d)
        m.fit(X6_poly_tr[tr_idx], y_tr[tr_idx])
        oof_d[va_idx] = m.predict(X6_poly_tr[va_idx])
    m_f = Ridge(alpha=best_alpha_d)
    m_f.fit(X6_poly_tr, y_tr)
    te_d = m_f.predict(X6_poly_te)
    r_d = rae(y_tr, oof_d); ratio_d = te_d.std()/oof_d.std()
    flag_d = "PASS" if ratio_d >= COLLAPSE_THRESH else "FAIL"
    print(f"  Poly+Ridge (alpha={best_alpha_d}): RAE={r_d:.4f}  ratio={ratio_d:.3f}  [{flag_d}]")
    if ratio_d >= COLLAPSE_THRESH:
        results["D_poly_ridge"] = (r_d, oof_d, te_d, ratio_d)

    # --- E: 3-way blend of A+B+C ---
    all_nonlinear = [(r_a, oof_a, te_a), (r_b, oof_b, te_b), (r_c, oof_c, te_c)]
    if all(v is not None for v in [oof_a, oof_b, oof_c]):
        print(f"\n--- E: Blend A+B+C nonlinear predictions ---")
        for wa, wb, wc in [(1/3, 1/3, 1/3), (0.5, 0.3, 0.2), (0.4, 0.4, 0.2)]:
            oof_e = wa*oof_a + wb*oof_b + wc*oof_c
            te_e  = wa*te_a  + wb*te_b  + wc*te_c
            r_e = rae(y_tr, oof_e); ratio_e = te_e.std()/oof_e.std()
            flag = "PASS" if ratio_e >= COLLAPSE_THRESH else "FAIL"
            print(f"  w=({wa:.2f},{wb:.2f},{wc:.2f})  RAE={r_e:.4f}  ratio={ratio_e:.3f}  [{flag}]")
            if ratio_e >= COLLAPSE_THRESH and r_e < results.get("E_best_r", 1e9):
                results["E_best_r"] = r_e
                results["E_best"] = (r_e, oof_e, te_e, ratio_e)

    print(f"\n=== Summary ===")
    clean = {k: v for k, v in results.items() if not k.endswith("_r") and v[3] >= COLLAPSE_THRESH}
    for k, (r, _, _, ratio) in sorted(clean.items(), key=lambda x: x[1][0]):
        print(f"  {k:30s}  RAE={r:.6f}  ratio={ratio:.3f}  PASS")

    print(f"\nLinear reference: 0.300101")
    if not clean:
        print("  No nonlinear models passed collapse check!")
        return

    best_label = min(clean, key=lambda k: clean[k][0])
    best_r, best_oof, best_te, best_ratio = clean[best_label]
    print(f"\nBEST: {best_label}  RAE={best_r:.6f}  ratio={best_ratio:.3f}")

    if best_r < 0.3001:
        print("*** BEATS LINEAR BLEND! New best! ***")
    else:
        print("Linear blend (0.3001) wins — nonlinear adds nothing")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb180_nonlinear_6model.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb180_nonlinear_6model.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "180_nonlinear_6model.csv", index=False)
    print(f"Saved: submissions/180_nonlinear_6model.csv  OOF RAE={best_r:.6f}")


if __name__ == "__main__":
    main()
