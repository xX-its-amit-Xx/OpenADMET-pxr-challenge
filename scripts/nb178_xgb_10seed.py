"""nb178 — XGBoost MAE 10-seed on pure base pool (fix of nb177).

nb177 contaminated the top-80 with meta-learner OOFs (nb162/nb165/nb149 etc.)
because its exclusion list was incomplete. This script replicates nb167 exactly
(same META_STEMS, same assay features, same hyperparams) but uses 10 seeds
instead of 5 and adds Optuna-tuned hyperparams as a second config.

nb167 (5 seeds, correct base pool): 0.3038 PASS ratio=0.58

Goal: push below 0.3038 with 10 seeds + better tuning.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58
SEED = 42

# All meta-learners / derived models — same as nb167 + new ones since then
META_STEMS = {
    "nb136_xgb_meta", "nb138_elnet_blend", "nb139_adaptive_blend",
    "nb140_xgb_lgbm_meta", "nb141_xgb_ablation", "nb142_xgb_calibrated",
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb146_pca_oof_meta", "nb147_oofrdkit_meta", "nb148_meta_disagreement",
    "nb149_meta_maeloss", "nb150_residual_ensemble", "nb151_grand_v11",
    "nb152_lgbm_mae_tuned", "nb153_grand_v12", "nb154_lgbm_mae_filtered",
    "nb155_grand_v13", "nb156_catboost_mae", "nb157_optuna_lgbm_mae",
    "nb158_collapse_fix", "nb159_variance_scaled_ensemble", "nb160_pca_meta",
    "nb161_neural_meta", "nb162_mixed_pool", "nb163_lgbm_colsample_low",
    "nb163_lgbm_colsample",
    "nb164_grand_v14", "nb165_multiseed_162c", "nb166_catboost_v2",
    "nb167_xgboost_mae",
    "nb168_multiseed_catboost", "nb169_rf_et_mae",
    "nb170_grand_v15", "nb171_catboost_extended",
    "nb172_bootstrap_ensemble", "nb173_softmax_sweep",
    "nb174_top10_lgbm", "nb175_bayes_blend", "nb176_optuna_weights",
    "nb177_xgb_histgb", "nb178_xgb_10seed",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
    "nb108_grand_v2", "nb112_grand_v3", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
}

ANCHOR_STEMS = [
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb136_xgb_meta", "nb134_grand_v9", "nb141_xgb_ablation",
]

LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
)


def load_base_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH, exclude_stems=None):
    exclude_stems = exclude_stems or set()
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in exclude_stems:
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
    return results


def build_assay_features(tr, te, splits, y_tr, n_tr):
    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw    = raw_train[emax_col].values.astype(np.float64)
    emax_log    = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_imputed = np.where(np.isnan(pec50_null), np.nanmedian(pec50_null), pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)
    X_str    = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))
    oof_emax = np.full(n_tr, np.nan)
    oof_null = np.full(n_tr, np.nan)
    oof_sel  = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_str[va_idx])
        m_nl = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=null_imputed[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_str[va_idx])
        m_sl = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=selectivity[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx] = m_sl.predict(X_str[va_idx])
    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_str_te)
    te_null = m_nl_f.predict(X_str_te)
    te_sel  = m_sl_f.predict(X_str_te)
    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null, np.log1p(np.clip(oof_emax, 0, None))])
    assay_te  = np.column_stack([te_emax,  te_null,  te_sel,  np.zeros(len(X_str_te)), np.log1p(np.clip(te_emax, 0, None))])
    return assay_oof, assay_te


def cv_xgb_multiseed(params_base, seeds, X_tr, y_tr, X_te, splits, label):
    n_tr = len(y_tr)
    seed_oofs = []; seed_tes = []
    for seed in seeds:
        params = {**params_base, "random_state": seed}
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = XGBRegressor(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx], verbose=False)
            oof_s[va_idx] = m.predict(X_tr[va_idx])
        m_f = XGBRegressor(**params)
        m_f.fit(X_tr, y_tr, verbose=False)
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
    print("=== nb178: XGBoost MAE 10-seed (pure base pool, fixed) ===\n")
    print("nb167 (5 seeds, correct exclusion): 0.3038 PASS")
    print("Goal: 10 seeds + Optuna-tuned to push below 0.3038\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te_df, splits, y_tr, n_tr)

    print("Loading pure base OOFs (excluding all meta-stacks)...")
    base_mods = load_base_oofs(n_tr, y_tr, exclude_stems=META_STEMS)
    print(f"  {len(base_mods)} pure base models")
    for m in base_mods[:10]:
        print(f"    {m['stem']:55s}  RAE={m['rae']:.4f}")

    top80_mods = base_mods[:80]
    oof80 = np.column_stack([m["oof"] for m in top80_mods])
    te80  = np.column_stack([m["te"]  for m in top80_mods])
    X_tr_80 = np.hstack([oof80, assay_oof])
    X_te_80 = np.hstack([te80,  assay_te])
    print(f"  Top-80 + assay features: {X_tr_80.shape}")

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

    # --- A: 10-seed XGBoost (same params as nb167, pure base pool) ---
    print(f"\n--- A: XGBoost 10-seed on pure top-80 + assay ({X_tr_80.shape[1]} features) ---")
    r_a, oof_a, te_a, ratio_a = cv_xgb_multiseed(XGB_BASE, SEEDS_10, X_tr_80, y_tr, X_te_80, splits, "XGB_10seed_purebase")
    if ratio_a >= COLLAPSE_THRESH:
        results["A_xgb10_purebase"] = (r_a, oof_a, te_a, ratio_a)

    # --- B: Optuna-tuned XGBoost (25 trials, 5 seeds with best params) ---
    print(f"\n--- B: Optuna XGBoost (25 trials) on pure top-80 + assay ---")
    def xgb_objective(trial):
        params = dict(
            objective="reg:absoluteerror", eval_metric="mae",
            n_estimators=trial.suggest_int("n_estimators", 500, 1500),
            learning_rate=trial.suggest_float("lr", 0.01, 0.08, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("mcw", 2, 15),
            reg_alpha=trial.suggest_float("alpha", 0.0, 0.5),
            reg_lambda=trial.suggest_float("lambda", 0.1, 5.0),
            random_state=42, n_jobs=4, verbosity=0,
        )
        oof_t = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = XGBRegressor(**params)
            m.fit(X_tr_80[tr_idx], y_tr[tr_idx], verbose=False)
            oof_t[va_idx] = m.predict(X_tr_80[va_idx])
        return rae(y_tr, oof_t)

    study_b = optuna.create_study(direction="minimize",
                                   sampler=optuna.samplers.TPESampler(seed=SEED))
    study_b.optimize(xgb_objective, n_trials=25, show_progress_bar=False)
    bp = study_b.best_params
    print(f"  Optuna best single-seed RAE={study_b.best_value:.4f}")
    print(f"  Best params: n_est={bp['n_estimators']} lr={bp['lr']:.4f} depth={bp['max_depth']} "
          f"ss={bp['subsample']:.2f} cs={bp['colsample_bytree']:.2f} mcw={bp['mcw']}")

    best_xgb = dict(
        objective="reg:absoluteerror", eval_metric="mae",
        n_estimators=bp["n_estimators"], learning_rate=bp["lr"],
        max_depth=bp["max_depth"], subsample=bp["subsample"],
        colsample_bytree=bp["colsample_bytree"], min_child_weight=bp["mcw"],
        reg_alpha=bp["alpha"], reg_lambda=bp["lambda"],
        n_jobs=4, verbosity=0,
    )
    print("  Training 5-seed ensemble with Optuna best params...")
    r_b, oof_b, te_b, ratio_b = cv_xgb_multiseed(best_xgb, SEEDS_5, X_tr_80, y_tr, X_te_80, splits, "XGB_opt_5seed_purebase")
    if ratio_b >= COLLAPSE_THRESH:
        results["B_xgb_opt5_purebase"] = (r_b, oof_b, te_b, ratio_b)

    # --- C: 10-seed with Optuna params ---
    if ratio_b >= COLLAPSE_THRESH:
        print("\n--- C: 10-seed with Optuna best params ---")
        r_c, oof_c, te_c, ratio_c = cv_xgb_multiseed(best_xgb, SEEDS_10, X_tr_80, y_tr, X_te_80, splits, "XGB_opt_10seed_purebase")
        if ratio_c >= COLLAPSE_THRESH:
            results["C_xgb_opt10_purebase"] = (r_c, oof_c, te_c, ratio_c)

    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:40s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    if not results:
        print("  No passing results!")
        return

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid:
        valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBEST: {best_label}  RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb167 5-seed pure base: 0.3038; ensemble best: 0.3001)")

    if best_r < 0.3038:
        print("*** NEW BEST XGBoost! Beat 0.3038! ***")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb178_xgb_10seed.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb178_xgb_10seed.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "178_xgb_10seed.csv", index=False)
    print(f"Saved: submissions/178_xgb_10seed.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
