"""nb179 — Fix XGBoost collapse and push below 0.3001.

nb178 found:
  A: 10-seed default XGB on pure base = 0.3024, ratio=0.57 FAIL
  B: Optuna XGB 5-seed = 0.2994 (!), ratio=0.57 FAIL (Optuna params: n_est=1364, lr=0.0124, depth=7, ss=0.64, cs=0.96, mcw=12)

Both fail by ratio=0.57 (< 0.58 threshold). Need to either:
  1. Find XGB params that give ratio >= 0.58 while staying low RAE
  2. Blend failing-but-accurate XGB with nb167 (ratio=0.58) to rescue
  3. Try shallower depth (depth=4,5) which reduces test extrapolation

Strategy:
  A: XGB depth=5, Optuna lr/reg, 5 seeds (shallower = better ratio)
  B: XGB depth=4, Optuna lr/reg, 5 seeds
  C: XGB depth=6, strong L2 (lambda=10), 5 seeds
  D: Blend nb178-style 10-seed (reconstruct in memory) with nb167 OOF
  E: Optuna over (depth, lambda) constrained to ratio >= 0.58
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
    "nb177_xgb_histgb", "nb178_xgb_10seed", "nb179_xgb_collapse_fix",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
    "nb108_grand_v2", "nb112_grand_v3", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
}

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
    print(f"  [{label}] RAE={r_ms:.4f}  ratio={ratio:.3f}  [{flag}]", flush=True)
    return r_ms, oof_ms, te_ms, ratio


def main():
    print("=== nb179: XGBoost Collapse Fix ===\n")
    print("nb178 A: 0.3024 ratio=0.570 FAIL (10-seed default)")
    print("nb178 B: 0.2994 ratio=0.570 FAIL (Optuna: depth=7, lr=0.0124)")
    print("Goal: push ratio to 0.58 without sacrificing RAE\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te_df, splits, y_tr, n_tr)

    print("Loading pure base OOFs...")
    base_mods = load_base_oofs(n_tr, y_tr, exclude_stems=META_STEMS)
    print(f"  {len(base_mods)} pure base models")

    top80 = base_mods[:80]
    X80_tr = np.hstack([np.column_stack([m["oof"] for m in top80]), assay_oof])
    X80_te = np.hstack([np.column_stack([m["te"]  for m in top80]), assay_te])
    print(f"  Top-80 + assay: {X80_tr.shape}")

    # Also prepare top-60 (might have better ratio)
    top60 = base_mods[:60]
    X60_tr = np.hstack([np.column_stack([m["oof"] for m in top60]), assay_oof])
    X60_te = np.hstack([np.column_stack([m["te"]  for m in top60]), assay_te])
    print(f"  Top-60 + assay: {X60_tr.shape}")

    # No-assay version (assay features may cause collapse)
    X80_tr_na = np.column_stack([m["oof"] for m in top80])
    X80_te_na = np.column_stack([m["te"]  for m in top80])

    # Load nb167 for blending rescue
    nb167_oof_p = DATA_PROCESSED / "oof_nb167_xgboost_mae.npy"
    nb167_te_p  = DATA_PROCESSED / "te_nb167_xgboost_mae.npy"
    nb167_oof = np.load(nb167_oof_p).astype(np.float64) if nb167_oof_p.exists() else None
    nb167_te  = np.load(nb167_te_p).astype(np.float64)  if nb167_te_p.exists() else None
    if nb167_oof is not None:
        r167 = rae(y_tr, nb167_oof)
        ratio167 = nb167_te.std() / nb167_oof.std()
        print(f"  nb167 reference: RAE={r167:.4f}  ratio={ratio167:.3f}")

    results = {}
    SEEDS_5 = [42, 123, 456, 789, 1234]

    # nb178 best Optuna params (from run)
    OPT_PARAMS = dict(
        objective="reg:absoluteerror", eval_metric="mae",
        n_estimators=1364, learning_rate=0.0124, max_depth=7,
        subsample=0.64, colsample_bytree=0.96, min_child_weight=12,
        reg_alpha=0.46, reg_lambda=4.08,
        n_jobs=4, verbosity=0,
    )

    BASE_PARAMS = dict(
        objective="reg:absoluteerror", eval_metric="mae",
        n_estimators=800, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.05, reg_lambda=1.0,
        n_jobs=4, verbosity=0,
    )

    # --- A: Optuna params but depth=5 (shallower -> better ratio) ---
    print(f"\n--- A: Optuna params, depth=5, top-80 ---")
    params_a = {**OPT_PARAMS, "max_depth": 5}
    r_a, oof_a, te_a, ratio_a = cv_xgb_multiseed(params_a, SEEDS_5, X80_tr, y_tr, X80_te, splits, "XGB_opt_d5_top80")
    if ratio_a >= COLLAPSE_THRESH:
        results["A_opt_d5"] = (r_a, oof_a, te_a, ratio_a)

    # --- B: Optuna params but depth=4 ---
    print(f"\n--- B: Optuna params, depth=4, top-80 ---")
    params_b = {**OPT_PARAMS, "max_depth": 4}
    r_b, oof_b, te_b, ratio_b = cv_xgb_multiseed(params_b, SEEDS_5, X80_tr, y_tr, X80_te, splits, "XGB_opt_d4_top80")
    if ratio_b >= COLLAPSE_THRESH:
        results["B_opt_d4"] = (r_b, oof_b, te_b, ratio_b)

    # --- C: Default params, NO assay features (simpler feature space) ---
    print(f"\n--- C: Default params, no assay, top-80 ---")
    r_c, oof_c, te_c, ratio_c = cv_xgb_multiseed(BASE_PARAMS, SEEDS_5, X80_tr_na, y_tr, X80_te_na, splits, "XGB_base_noassay_top80")
    if ratio_c >= COLLAPSE_THRESH:
        results["C_base_noassay"] = (r_c, oof_c, te_c, ratio_c)

    # --- D: Default params, top-60 + assay ---
    print(f"\n--- D: Default params, top-60 + assay ---")
    r_d, oof_d, te_d, ratio_d = cv_xgb_multiseed(BASE_PARAMS, SEEDS_5, X60_tr, y_tr, X60_te, splits, "XGB_base_top60")
    if ratio_d >= COLLAPSE_THRESH:
        results["D_base_top60"] = (r_d, oof_d, te_d, ratio_d)

    # --- E: Optuna params, high L2 lambda=20 (strong regularization -> better ratio) ---
    print(f"\n--- E: Optuna params, lambda=20, top-80 ---")
    params_e = {**OPT_PARAMS, "reg_lambda": 20.0, "max_depth": 7}
    r_e, oof_e, te_e, ratio_e = cv_xgb_multiseed(params_e, SEEDS_5, X80_tr, y_tr, X80_te, splits, "XGB_opt_lam20_top80")
    if ratio_e >= COLLAPSE_THRESH:
        results["E_opt_lam20"] = (r_e, oof_e, te_e, ratio_e)

    # --- F: Blend best-so-far with nb167 to rescue ratio ---
    if nb167_oof is not None:
        print(f"\n--- F: Rescue blend with nb167 ---")
        # Try each config A-E and blend the one with best RAE regardless of ratio
        all_configs = [
            ("A_opt_d5", r_a, oof_a, te_a, ratio_a),
            ("B_opt_d4", r_b, oof_b, te_b, ratio_b),
            ("C_base_noassay", r_c, oof_c, te_c, ratio_c),
            ("D_base_top60", r_d, oof_d, te_d, ratio_d),
            ("E_opt_lam20", r_e, oof_e, te_e, ratio_e),
        ]
        # Find best by RAE regardless of pass/fail
        best_cfg = min(all_configs, key=lambda x: x[1])
        best_name, best_r_raw, best_oof_raw, best_te_raw, best_ratio_raw = best_cfg
        print(f"  Best raw config: {best_name}  RAE={best_r_raw:.4f}  ratio={best_ratio_raw:.3f}")

        for alpha in [0.7, 0.8, 0.85, 0.9, 0.95]:
            oof_f = alpha * best_oof_raw + (1 - alpha) * nb167_oof
            te_f  = alpha * best_te_raw  + (1 - alpha) * nb167_te
            r_f   = rae(y_tr, oof_f)
            ratio_f = te_f.std() / oof_f.std()
            flag = "PASS" if ratio_f >= COLLAPSE_THRESH else "FAIL"
            print(f"  alpha={alpha:.2f}  RAE={r_f:.4f}  ratio={ratio_f:.3f}  [{flag}]")
            if ratio_f >= COLLAPSE_THRESH:
                if r_f < results.get("F_best_r", 1e9):
                    results["F_best_r"] = r_f
                    results[f"F_blend_a{alpha}"] = (r_f, oof_f, te_f, ratio_f)

    print(f"\n=== Summary ===")
    for k, v in sorted(
        [(k, v) for k, v in results.items() if k != "F_best_r"],
        key=lambda x: x[1][0]
    ):
        flag = "PASS" if v[3] >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:35s}  RAE={v[0]:.4f}  ratio={v[3]:.3f}  [{flag}]")

    clean = {k: v for k, v in results.items() if k != "F_best_r" and v[3] >= COLLAPSE_THRESH}
    if not clean:
        print("  No passing results — all collapse below threshold")
        return

    best_label = min(clean, key=lambda k: clean[k][0])
    best_r, best_oof, best_te, best_ratio = clean[best_label]
    print(f"\nBEST: {best_label}  RAE={best_r:.4f}  ratio={best_ratio:.3f}")
    print(f"(nb167 XGB-5seed: 0.3038, ensemble best: 0.3001)")

    if best_r < 0.3001:
        print("*** NEW BEST! Beat ensemble 0.3001! ***")
    elif best_r < 0.3038:
        print("*** Beat nb167 XGB-5seed (0.3038) — new XGBoost SOTA ***")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb179_xgb_collapse_fix.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb179_xgb_collapse_fix.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "179_xgb_collapse_fix.csv", index=False)
    print(f"Saved: submissions/179_xgb_collapse_fix.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
