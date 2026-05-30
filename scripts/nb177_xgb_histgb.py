"""nb177 — XGBoost 10-seed + HistGradientBoosting on mixed pool.

Current best: 0.3001 (linear blend of 6 models). Plateau confirmed by nb176
(100-start SLSQP, Optuna 500 trials all = 0.3001). Need new diverse base model.

nb167 XGBoost (5 seeds) = 0.3038 — has dominant weight (0.35) in ensemble.
This script tries:
  A: XGBoost 10 seeds (more seeds = lower variance, might beat 0.3038)
  B: XGBoost Optuna-tuned hyperparams (25 trials, then best config 5 seeds)
  C: sklearn HistGradientBoosting (MAE) on top-80 OOF pool — genuinely different implementation
  D: sklearn HistGradientBoosting (MAE) on mixed pool (125 features)
  E: Blend A (xgb-10seed) with best existing CatBoost
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58
SEED = 42

EXCLUDE_META = {
    "nb164_grand_v14", "nb170_grand_v15", "nb176_optuna_weights",
    "nb172_bootstrap_ensemble", "nb173_softmax_sweep",
    "nb175_bayes_blend", "nb174_top10_lgbm",
    "nb108_grand_v2", "nb112_grand_v3", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
    "nb134_grand_v9", "nb144_grand_v10", "nb151_grand_v11",
    "nb153_grand_v12", "nb155_grand_v13",
    "nb177_xgb_histgb",
}

EXCLUDE_STACKS = {
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb136_xgb_meta", "nb134_grand_v9", "nb141_xgb_ablation",
}


def load_base_oofs(n_tr, y_tr, top_n=80, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in EXCLUDE_META or stem in EXCLUDE_STACKS:
            continue
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists():
                break
        else:
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
            r = rae(y_tr, oof)
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results[:top_n]


def load_mixed_pool(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    """Base OOFs + 6 original anchors (same as nb162 mixed pool)."""
    base = load_base_oofs(n_tr, y_tr, top_n=9999, thresh=thresh)
    anchors_needed = [
        "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
        "nb136_xgb_meta", "nb134_grand_v9", "nb141_xgb_ablation",
    ]
    base_stems = {m["stem"] for m in base}
    extras = []
    for stem in anchors_needed:
        if stem in base_stems:
            continue
        p = DATA_PROCESSED / f"oof_{stem}.npy"
        if not p.exists():
            continue
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists():
                break
        else:
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
            extras.append(dict(stem=stem, oof=oof, te=te))
        except Exception:
            pass
    return base, extras


def cv_multiseed(model_cls, params_list, X_tr, y_tr, X_te, splits, label, seeds):
    """params_list[i] overrides base params for seed i."""
    n_tr = len(y_tr)
    seed_oofs = []; seed_tes = []
    for seed, params in zip(seeds, params_list):
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = model_cls(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof_s[va_idx] = m.predict(X_tr[va_idx])
        m_f = model_cls(**params)
        m_f.fit(X_tr, y_tr)
        seed_oofs.append(oof_s)
        seed_tes.append(m_f.predict(X_te))
        print(f"    seed {seed}: RAE={rae(y_tr, oof_s):.4f}", flush=True)
    oof_ms = np.mean(seed_oofs, axis=0)
    te_ms  = np.mean(seed_tes,  axis=0)
    r_ms   = rae(y_tr, oof_ms)
    ratio  = te_ms.std() / oof_ms.std()
    flag   = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label}] RAE={r_ms:.4f}  ratio={ratio:.2f}  [{flag}]", flush=True)
    return r_ms, oof_ms, te_ms, ratio


def main():
    print("=== nb177: XGBoost 10-seed + HistGB ===\n")
    print("Target: beat 0.3001. Need new diverse base model.\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load top-80 base OOFs
    base_models = load_base_oofs(n_tr, y_tr, top_n=80)
    print(f"Top-80 base models loaded ({len(base_models)} pass threshold)")
    X80_tr = np.column_stack([m["oof"] for m in base_models])
    X80_te = np.column_stack([m["te"]  for m in base_models])

    # Load mixed pool
    base_all, anchors = load_mixed_pool(n_tr, y_tr)
    print(f"Mixed pool: {len(base_all)} base + {len(anchors)} anchors = {len(base_all)+len(anchors)} total")
    Xmix_tr = np.column_stack([m["oof"] for m in base_all] + [m["oof"] for m in anchors])
    Xmix_te = np.column_stack([m["te"]  for m in base_all] + [m["te"]  for m in anchors])

    results = {}
    SEEDS_10 = [42, 123, 456, 789, 1234, 2024, 314, 999, 7, 31415]
    SEEDS_5  = SEEDS_10[:5]

    XGB_BASE = dict(
        objective="reg:absoluteerror", eval_metric="mae",
        n_estimators=800, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.05, reg_lambda=1.0,
        n_jobs=4, verbosity=0,
    )

    # --- A: XGBoost 10 seeds on top-80 ---
    print(f"\n--- A: XGBoost 10-seed on top-80 ({X80_tr.shape[1]} features) ---")
    params_10 = [{**XGB_BASE, "random_state": s} for s in SEEDS_10]
    r_a, oof_a, te_a, ratio_a = cv_multiseed(
        XGBRegressor, params_10, X80_tr, y_tr, X80_te, splits, "XGB_10seed_top80", SEEDS_10)
    if ratio_a >= COLLAPSE_THRESH:
        results["A_xgb10_top80"] = (r_a, oof_a, te_a, ratio_a)

    # --- B: Optuna-tuned XGBoost on top-80 (25 trials) ---
    print(f"\n--- B: Optuna-tuned XGBoost on top-80 (25 trials) ---")
    def xgb_objective(trial):
        params = dict(
            objective="reg:absoluteerror", eval_metric="mae",
            n_estimators=trial.suggest_int("n_estimators", 500, 1500),
            learning_rate=trial.suggest_float("lr", 0.01, 0.08, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("mcw", 3, 15),
            reg_alpha=trial.suggest_float("alpha", 0.0, 0.5),
            reg_lambda=trial.suggest_float("lambda", 0.5, 5.0),
            random_state=42, n_jobs=4, verbosity=0,
        )
        oof_t = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = XGBRegressor(**params)
            m.fit(X80_tr[tr_idx], y_tr[tr_idx])
            oof_t[va_idx] = m.predict(X80_tr[va_idx])
        return rae(y_tr, oof_t)

    study_b = optuna.create_study(direction="minimize",
                                   sampler=optuna.samplers.TPESampler(seed=42))
    study_b.optimize(xgb_objective, n_trials=25, show_progress_bar=False)
    print(f"  Optuna best RAE={study_b.best_value:.4f} params: {study_b.best_params}")

    # Train 5-seed ensemble with best params
    best_p = study_b.best_params
    best_xgb_params = dict(
        objective="reg:absoluteerror", eval_metric="mae",
        n_estimators=best_p["n_estimators"], learning_rate=best_p["lr"],
        max_depth=best_p["max_depth"], subsample=best_p["subsample"],
        colsample_bytree=best_p["colsample_bytree"], min_child_weight=best_p["mcw"],
        reg_alpha=best_p["alpha"], reg_lambda=best_p["lambda"],
        n_jobs=4, verbosity=0,
    )
    print("  Training 5-seed ensemble with Optuna best params...")
    params_5_opt = [{**best_xgb_params, "random_state": s} for s in SEEDS_5]
    r_b, oof_b, te_b, ratio_b = cv_multiseed(
        XGBRegressor, params_5_opt, X80_tr, y_tr, X80_te, splits, "XGB_opt_5seed", SEEDS_5)
    if ratio_b >= COLLAPSE_THRESH:
        results["B_xgb_opt5seed"] = (r_b, oof_b, te_b, ratio_b)

    # --- C: HistGradientBoosting (MAE) on top-80 ---
    print(f"\n--- C: HistGradientBoosting (MAE) on top-80 ---")
    HGB_BASE = dict(
        loss="absolute_error", learning_rate=0.05, max_iter=500,
        max_leaf_nodes=63, min_samples_leaf=10, l2_regularization=0.1,
        max_features=0.8, random_state=42,
    )
    seeds_hgb = [42, 123, 456, 789, 1234]
    params_hgb = [{**HGB_BASE, "random_state": s} for s in seeds_hgb]
    r_c, oof_c, te_c, ratio_c = cv_multiseed(
        HistGradientBoostingRegressor, params_hgb,
        X80_tr, y_tr, X80_te, splits, "HGB_top80_5seed", seeds_hgb)
    if ratio_c >= COLLAPSE_THRESH:
        results["C_hgb_top80"] = (r_c, oof_c, te_c, ratio_c)

    # --- D: HistGradientBoosting on mixed pool ---
    print(f"\n--- D: HistGradientBoosting (MAE) on mixed pool ({Xmix_tr.shape[1]} features) ---")
    r_d, oof_d, te_d, ratio_d = cv_multiseed(
        HistGradientBoostingRegressor, params_hgb,
        Xmix_tr, y_tr, Xmix_te, splits, "HGB_mixed_5seed", seeds_hgb)
    if ratio_d >= COLLAPSE_THRESH:
        results["D_hgb_mixed"] = (r_d, oof_d, te_d, ratio_d)

    # --- E: Blend A (XGB-10) + nb156 CatBoost ---
    nb156_oof_p = DATA_PROCESSED / "oof_nb156_catboost_mae.npy"
    nb156_te_p  = DATA_PROCESSED / "te_nb156_catboost_mae.npy"
    if nb156_oof_p.exists() and nb156_te_p.exists() and "A_xgb10_top80" in results:
        nb156_oof = np.load(nb156_oof_p).astype(np.float64)
        nb156_te  = np.load(nb156_te_p).astype(np.float64)
        if nb156_oof.ndim == 2: nb156_oof = nb156_oof[:, 0]
        if nb156_te.ndim  == 2: nb156_te  = nb156_te[:, 0]
        print(f"\n--- E: Blend XGB-10seed + CatBoost (nb156) ---")
        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
            oof_e = alpha * oof_a + (1 - alpha) * nb156_oof
            te_e  = alpha * te_a  + (1 - alpha) * nb156_te
            r_e = rae(y_tr, oof_e); ratio_e = te_e.std() / oof_e.std()
            flag = "PASS" if ratio_e >= COLLAPSE_THRESH else "FAIL"
            print(f"  alpha(xgb)={alpha:.1f}  RAE={r_e:.4f}  ratio={ratio_e:.2f}  [{flag}]")
            if ratio_e >= COLLAPSE_THRESH and r_e < results.get("E_best", (1e9,))[0]:
                results[f"E_blend_a{alpha}"] = (r_e, oof_e, te_e, ratio_e)

    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:35s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    if not results:
        print("  No passing results!")
        return

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid:
        valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBEST: {best_label}  RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb167 XGB-5seed: 0.3038, current ensemble: 0.3001)")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb177_xgb_histgb.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb177_xgb_histgb.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "177_xgb_histgb.csv", index=False)
    print(f"Saved: submissions/177_xgb_histgb.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
