"""nb167 — XGBoost with MAE objective on mixed pool.

XGBoost `objective="reg:absoluteerror"` directly optimizes L1/MAE — same
rationale as nb149's LGBM `regression_l1` discovery. Mixed pool = base models
(exclude meta-stacks) + 6 anchor meta-stacks (same as nb162-C setup).

Adds third diverse algorithm (XGBoost) alongside LGBM and CatBoost.

Tests:
  A: XGB_MAE, base+6 anchors, depth=6
  B: XGB_MAE, base+6 anchors, depth=8
  C: XGB_MAE, base+6 anchors, depth=4 (shallower)
  D: XGB_MAE, top-80 base models only (no anchors)
  E: Multi-seed (5x) on best passing config
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import lightgbm as lgb

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58

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
    "nb164_grand_v14", "nb165_multiseed_162c", "nb166_catboost_v2",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
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
            if te_p.exists(): break
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
    X_str = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))
    oof_emax = np.full(n_tr, np.nan); oof_null = np.full(n_tr, np.nan); oof_sel = np.full(n_tr, np.nan)
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
    assay_te  = np.column_stack([te_emax, te_null, te_sel, np.zeros(len(X_str_te)), np.log1p(np.clip(te_emax, 0, None))])
    return assay_oof, assay_te


def cv_xgb_mae(params, X_tr, y_tr, X_te, splits, label):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = XGBRegressor(**params)
        m.fit(X_tr[tr_idx], y_tr[tr_idx], verbose=False)
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)
    m_full = XGBRegressor(**params)
    m_full.fit(X_tr, y_tr, verbose=False)
    te_pred = m_full.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label:55s}] RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]", flush=True)
    return r, oof, te_pred, ratio


def main():
    print("=== nb167: XGBoost MAE on Mixed Pool ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    print("Loading base model OOFs (exclude meta-stacks)...")
    base_mods = load_base_oofs(n_tr, y_tr, exclude_stems=META_STEMS)
    print(f"  {len(base_mods)} base models loaded")

    # Mixed pool: base + 6 anchors
    all_mods = load_base_oofs(n_tr, y_tr, exclude_stems=set())
    stem_map = {m["stem"]: m for m in all_mods}
    anchors = [stem_map[s] for s in ANCHOR_STEMS if s in stem_map]
    print(f"  {len(anchors)} anchors found: {[a['stem'] for a in anchors]}")

    mixed_mods = base_mods + anchors
    oof_mat_mixed = np.column_stack([m["oof"] for m in mixed_mods])
    te_mat_mixed  = np.column_stack([m["te"]  for m in mixed_mods])
    X_tr_mixed = np.hstack([oof_mat_mixed, assay_oof])
    X_te_mixed  = np.hstack([te_mat_mixed, assay_te])
    print(f"  Mixed pool features: {X_tr_mixed.shape}")

    # Top-80 base only pool
    top80_mods = base_mods[:80]
    oof_mat_80 = np.column_stack([m["oof"] for m in top80_mods])
    te_mat_80  = np.column_stack([m["te"]  for m in top80_mods])
    X_tr_80 = np.hstack([oof_mat_80, assay_oof])
    X_te_80  = np.hstack([te_mat_80, assay_te])

    XGB_BASE = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=800,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )

    results = {}

    # A: depth=6 on mixed pool
    print("\n--- A: depth=6, mixed pool ---")
    params_a = {**XGB_BASE, "max_depth": 6}
    r_a, oof_a, te_a, ratio_a = cv_xgb_mae(params_a, X_tr_mixed, y_tr, X_te_mixed, splits, "xgb_d6_mixed")
    results["A_d6_mixed"] = (r_a, oof_a, te_a, ratio_a)

    # B: depth=8 on mixed pool
    print("\n--- B: depth=8, mixed pool ---")
    params_b = {**XGB_BASE, "max_depth": 8}
    r_b, oof_b, te_b, ratio_b = cv_xgb_mae(params_b, X_tr_mixed, y_tr, X_te_mixed, splits, "xgb_d8_mixed")
    results["B_d8_mixed"] = (r_b, oof_b, te_b, ratio_b)

    # C: depth=4 on mixed pool
    print("\n--- C: depth=4, mixed pool ---")
    params_c = {**XGB_BASE, "max_depth": 4}
    r_c, oof_c, te_c, ratio_c = cv_xgb_mae(params_c, X_tr_mixed, y_tr, X_te_mixed, splits, "xgb_d4_mixed")
    results["C_d4_mixed"] = (r_c, oof_c, te_c, ratio_c)

    # D: depth=6 on top-80 base only
    print("\n--- D: depth=6, top-80 base only ---")
    params_d = {**XGB_BASE, "max_depth": 6}
    r_d, oof_d, te_d, ratio_d = cv_xgb_mae(params_d, X_tr_80, y_tr, X_te_80, splits, "xgb_d6_top80")
    results["D_d6_top80"] = (r_d, oof_d, te_d, ratio_d)

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:40s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    # Multi-seed on best passing config
    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if valid:
        best_label = min(valid, key=lambda k: valid[k][0])
        best_r, best_oof_single, best_te_single, _ = valid[best_label]
        print(f"\nBest passing: {best_label}  RAE={best_r:.4f}")

        # Determine which pool was best
        use_mixed = "top80" not in best_label
        X_tr_best = X_tr_mixed if use_mixed else X_tr_80
        X_te_best  = X_te_mixed  if use_mixed else X_te_80

        # Extract depth from label
        depth_str = best_label.split("_")[1]
        depth_val = int(depth_str[1:])
        best_params = {**XGB_BASE, "max_depth": depth_val}

        print(f"\n--- E: multi-seed (5x) on best={best_label} ---")
        seed_oofs = [best_oof_single]; seed_tes = [best_te_single]
        for seed in [123, 456, 789, 1234]:
            oof_s = np.full(n_tr, np.nan)
            params_s = {**best_params, "random_state": seed}
            for fold, (tr_idx, va_idx) in enumerate(splits):
                m = XGBRegressor(**params_s)
                m.fit(X_tr_best[tr_idx], y_tr[tr_idx], verbose=False)
                oof_s[va_idx] = m.predict(X_tr_best[va_idx])
            m_f = XGBRegressor(**params_s)
            m_f.fit(X_tr_best, y_tr, verbose=False)
            seed_oofs.append(oof_s); seed_tes.append(m_f.predict(X_te_best))
            print(f"  seed {seed}: RAE={rae(y_tr, oof_s):.4f}", flush=True)
        oof_ms = np.mean(seed_oofs, axis=0); te_ms = np.mean(seed_tes, axis=0)
        r_ms = rae(y_tr, oof_ms); ratio_ms = te_ms.std() / oof_ms.std()
        flag = "PASS" if ratio_ms >= COLLAPSE_THRESH else "FAIL"
        print(f"  [multi-seed] RAE={r_ms:.4f}  ratio={ratio_ms:.2f}  [{flag}]")
        results["E_multiseed"] = (r_ms, oof_ms, te_ms, ratio_ms)

    # Final best (prefer passing)
    all_valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not all_valid:
        all_valid = results
        print("\nWARN: no config passed threshold — saving best-ratio config")
    final_label = min(all_valid, key=lambda k: all_valid[k][0])
    final_r, final_oof, final_te, final_ratio = all_valid[final_label]

    print(f"\n=== FINAL BEST: {final_label}  RAE={final_r:.4f}  ratio={final_ratio:.2f} ===")
    print(f"(nb156 CatBoost best: 0.3083, nb162 LGBM best: 0.3071)")

    final_te_out = np.clip(final_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb167_xgboost_mae.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb167_xgboost_mae.npy",  final_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": final_te_out})
    sub.to_csv(SUBMISSIONS / "167_xgboost_mae.csv", index=False)
    print(f"Saved: submissions/167_xgboost_mae.csv  OOF RAE={final_r:.4f}")


if __name__ == "__main__":
    main()
