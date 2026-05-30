"""nb166 — CatBoost Variants v2 on Mixed Pool.

nb156 CatBoost depth=8 (0.3083) got weight=0.384 in grand ensemble v14 (0.3013).
CatBoost provides fundamentally different signal from LGBM. More CatBoost variants
= more diverse models for ensemble.

Uses nb162 config C setup: base(110) + 6 meta-stack anchors.

Tests:
  A: CatBoost depth=6 (symmetric shallow)
  B: CatBoost depth=10 (deeper, more complex)
  C: CatBoost depth=8, more iterations (1500)
  D: CatBoost depth=8, lower lr (0.02), more trees (1200)
  E: CatBoost multi-seed (3 seeds) of best config
  F: CatBoost + LGBM_MAE ensemble (equal weight)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58

BASE_EXCLUDE = {
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
LGBM_MAE = dict(
    n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
    subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05,
    objective="regression_l1", verbose=-1, n_jobs=4, random_state=42
)

CAT_BASE = dict(
    loss_function="MAE", eval_metric="MAE",
    random_seed=42, thread_count=4, allow_writing_files=False
)


def load_stem(stem, n_tr):
    oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
    for te_pref in ("te_", "te_oof_"):
        te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
        if te_p.exists(): break
    if not oof_p.exists() or not te_p.exists():
        return None
    oof = np.load(oof_p).astype(np.float64)
    te  = np.load(te_p).astype(np.float64)
    if oof.ndim == 2: oof = oof[:, 0]
    if te.ndim == 2:  te  = te[:, 0]
    if len(oof) != n_tr: return None
    oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
    te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
    return dict(stem=stem, oof=oof, te=te)


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


def cv_catboost(params, X_tr, y_tr, X_te, splits, label, seeds=(42,)):
    n_tr = len(y_tr)
    all_seed_oofs = []; all_seed_tes = []
    for seed in seeds:
        p_s = {**params, "random_seed": seed}
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = CatBoostRegressor(**p_s)
            m.fit(X_tr[tr_idx], y_tr[tr_idx], verbose=False)
            oof_s[va_idx] = m.predict(X_tr[va_idx])
        m_full = CatBoostRegressor(**p_s)
        m_full.fit(X_tr, y_tr, verbose=False)
        te_s = m_full.predict(X_te)
        all_seed_oofs.append(oof_s); all_seed_tes.append(te_s)
    oof = np.mean(all_seed_oofs, axis=0)
    te_pred = np.mean(all_seed_tes, axis=0)
    r = rae(y_tr, oof); ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    suffix = f"({len(seeds)} seeds)" if len(seeds) > 1 else ""
    print(f"  [{label:55s}] RAE={r:.4f}  ratio={ratio:.2f}  [{flag}] {suffix}", flush=True)
    for fold_oof in all_seed_oofs[:1]:  # print folds for first seed
        pass
    return r, oof, te_pred, ratio


def main():
    print("=== nb166: CatBoost Variants v2 on Mixed Pool ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    print("Loading base models + anchors (nb162-C setup)...")
    base_mods = load_base_oofs(n_tr, y_tr, exclude_stems=BASE_EXCLUDE)
    print(f"  {len(base_mods)} base models")
    anchors = []
    for s in ANCHOR_STEMS:
        data = load_stem(s, n_tr)
        if data is not None: anchors.append(data)
    print(f"  {len(anchors)} anchor meta-stacks loaded")

    oof_mat = np.column_stack([m["oof"] for m in base_mods] + [a["oof"] for a in anchors])
    te_mat  = np.column_stack([m["te"]  for m in base_mods] + [a["te"]  for a in anchors])
    X_tr = np.hstack([oof_mat, assay_oof])
    X_te = np.hstack([te_mat, assay_te])
    print(f"  Meta features: {X_tr.shape}")

    results = {}

    # A: CatBoost depth=6 (shallower symmetric trees)
    print("\n--- A: CatBoost depth=6 ---")
    cat_a = {**CAT_BASE, "iterations": 800, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0}
    r, oof, te_pred, ratio = cv_catboost(cat_a, X_tr, y_tr, X_te, splits, "A: CB depth=6")
    results["A_cat_d6"] = (r, oof, te_pred, ratio)

    # B: CatBoost depth=10
    print("\n--- B: CatBoost depth=10 ---")
    cat_b = {**CAT_BASE, "iterations": 600, "learning_rate": 0.04, "depth": 10, "l2_leaf_reg": 5.0}
    r, oof, te_pred, ratio = cv_catboost(cat_b, X_tr, y_tr, X_te, splits, "B: CB depth=10")
    results["B_cat_d10"] = (r, oof, te_pred, ratio)

    # C: CatBoost depth=8, 1500 iterations
    print("\n--- C: CatBoost depth=8, 1500 iters ---")
    cat_c = {**CAT_BASE, "iterations": 1500, "learning_rate": 0.02, "depth": 8, "l2_leaf_reg": 3.0}
    r, oof, te_pred, ratio = cv_catboost(cat_c, X_tr, y_tr, X_te, splits, "C: CB depth=8 n=1500")
    results["C_cat_d8_1500"] = (r, oof, te_pred, ratio)

    # D: CatBoost depth=8 with Boruta-like subsample
    print("\n--- D: CatBoost depth=8, subsample=0.8 ---")
    cat_d = {**CAT_BASE, "iterations": 800, "learning_rate": 0.03, "depth": 8, "l2_leaf_reg": 3.0,
             "subsample": 0.8, "bootstrap_type": "Bernoulli"}
    r, oof, te_pred, ratio = cv_catboost(cat_d, X_tr, y_tr, X_te, splits, "D: CB depth=8 subsample=0.8")
    results["D_cat_d8_sub"] = (r, oof, te_pred, ratio)

    # E: Multi-seed (3 seeds) of best passing CatBoost config
    passing = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if passing:
        best_cat_cfg_name = min(passing, key=lambda k: passing[k][0])
        print(f"\n--- E: Multi-seed (3 seeds) of best: {best_cat_cfg_name} ---")
        best_cat_params = {
            "A_cat_d6": {**CAT_BASE, "iterations": 800, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0},
            "B_cat_d10": {**CAT_BASE, "iterations": 600, "learning_rate": 0.04, "depth": 10, "l2_leaf_reg": 5.0},
            "C_cat_d8_1500": {**CAT_BASE, "iterations": 1500, "learning_rate": 0.02, "depth": 8, "l2_leaf_reg": 3.0},
            "D_cat_d8_sub": {**CAT_BASE, "iterations": 800, "learning_rate": 0.03, "depth": 8, "l2_leaf_reg": 3.0, "subsample": 0.8, "bootstrap_type": "Bernoulli"},
        }.get(best_cat_cfg_name, cat_c)
        r, oof, te_pred, ratio = cv_catboost(best_cat_params, X_tr, y_tr, X_te, splits,
                                              f"E: 3-seed {best_cat_cfg_name}", seeds=[42, 123, 456])
        results["E_multiseed"] = (r, oof, te_pred, ratio)

    # F: Blend best CatBoost + LGBM_MAE (equal weight)
    print("\n--- F: CatBoost + LGBM_MAE equal blend ---")
    oof_lgbm = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM_MAE, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof_lgbm[va_idx] = m.predict(X_tr[va_idx])
    m_full = lgb.train(LGBM_MAE, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_lgbm = m_full.predict(X_te)
    r_lgbm = rae(y_tr, oof_lgbm)
    ratio_lgbm = te_lgbm.std() / oof_lgbm.std()
    print(f"  LGBM_MAE: RAE={r_lgbm:.4f}  ratio={ratio_lgbm:.2f}", flush=True)

    if "A_cat_d6" in results and results["A_cat_d6"][3] >= COLLAPSE_THRESH:
        best_cat_for_blend_key = min(passing, key=lambda k: passing[k][0]) if passing else "A_cat_d6"
        cat_oof = results[best_cat_for_blend_key][1]
        cat_te  = results[best_cat_for_blend_key][2]
        oof_f = (cat_oof + oof_lgbm) / 2
        te_f  = (cat_te  + te_lgbm) / 2
        r_f = rae(y_tr, oof_f); ratio_f = te_f.std() / oof_f.std()
        flag = "PASS" if ratio_f >= COLLAPSE_THRESH else "FAIL"
        print(f"  [F: CB+LGBM blend] RAE={r_f:.4f}  ratio={ratio_f:.2f}  [{flag}]")
        results["F_cb_lgbm_blend"] = (r_f, oof_f, te_f, ratio_f)

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:55s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBest: {best_label}  OOF RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb156 CatBoost d8: 0.3083, nb164 ensemble: 0.3013)")
    if best_r < 0.3083: print("*** NEW BEST CATBOOST! ***")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb166_catboost_v2.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb166_catboost_v2.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "166_catboost_v2.csv", index=False)
    print(f"\nSaved: submissions/166_catboost_v2.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
