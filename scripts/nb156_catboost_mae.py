"""nb156 — CatBoost with MAE Loss on Clean Base OOF.

Same filtered setup as nb154 (exclude meta-stack stems) but using CatBoost
instead of LightGBM. CatBoost uses symmetric trees + ordered boosting, which
may complement LGBM's MAE approach.

Also tests: gradient-boosted MAE via sklearn GradientBoostingRegressor.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

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
    "nb155_grand_v13",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
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

    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null,
                                  np.log1p(np.clip(oof_emax, 0, None))])
    assay_te  = np.column_stack([te_emax, te_null, te_sel,
                                  np.zeros(len(X_str_te)),
                                  np.log1p(np.clip(te_emax, 0, None))])
    return assay_oof, assay_te


def cv_model(model_class, params, X_tr, y_tr, X_te, splits, label, is_catboost=False, is_sklearn=False):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    models_per_fold = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        if is_catboost:
            m = CatBoostRegressor(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx], verbose=False)
        elif is_sklearn:
            m = GradientBoostingRegressor(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)
        models_per_fold.append(m)

    if is_catboost:
        m_full = CatBoostRegressor(**params)
        m_full.fit(X_tr, y_tr, verbose=False)
    elif is_sklearn:
        m_full = GradientBoostingRegressor(**params)
        m_full.fit(X_tr, y_tr)
    te_pred = m_full.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    print(f"  [{label:55s}] RAE={r:.4f}  ratio={ratio:.2f}", flush=True)
    return r, oof, te_pred, ratio


def main():
    print("=== nb156: CatBoost/GBM MAE Meta-Stack on Clean Base OOF ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    print("Loading base model OOFs (excluding meta-stacks)...")
    mods = load_base_oofs(n_tr, y_tr, exclude_stems=META_STEMS)
    print(f"  {len(mods)} base models loaded")
    oof_mat = np.column_stack([m["oof"] for m in mods])
    te_mat  = np.column_stack([m["te"]  for m in mods])
    X_tr = np.hstack([oof_mat, assay_oof])
    X_te = np.hstack([te_mat, assay_te])
    print(f"  Meta features: {X_tr.shape}")

    results = {}

    # A: CatBoost MAE
    print("\n--- A: CatBoost MAE ---")
    cat_mae_params = dict(
        iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        loss_function="MAE", eval_metric="MAE", random_seed=42,
        thread_count=4, allow_writing_files=False
    )
    r, oof, te_pred, ratio = cv_model(None, cat_mae_params, X_tr, y_tr, X_te, splits,
                                       "A: CatBoost MAE", is_catboost=True)
    results["A_catboost_mae"] = (r, oof, te_pred, ratio)

    # B: CatBoost MAE, more trees
    print("\n--- B: CatBoost MAE more trees ---")
    cat_mae_b = dict(
        iterations=1200, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
        loss_function="MAE", eval_metric="MAE", random_seed=42,
        thread_count=4, allow_writing_files=False
    )
    r, oof, te_pred, ratio = cv_model(None, cat_mae_b, X_tr, y_tr, X_te, splits,
                                       "B: CatBoost MAE 1200 trees", is_catboost=True)
    results["B_catboost_mae_1200"] = (r, oof, te_pred, ratio)

    # C: CatBoost MAE, deeper trees
    print("\n--- C: CatBoost MAE depth=8 ---")
    cat_mae_c = dict(
        iterations=600, learning_rate=0.04, depth=8, l2_leaf_reg=5.0,
        loss_function="MAE", eval_metric="MAE", random_seed=42,
        thread_count=4, allow_writing_files=False
    )
    r, oof, te_pred, ratio = cv_model(None, cat_mae_c, X_tr, y_tr, X_te, splits,
                                       "C: CatBoost MAE depth=8", is_catboost=True)
    results["C_catboost_mae_d8"] = (r, oof, te_pred, ratio)

    # D: sklearn GradientBoosting with MAE
    print("\n--- D: sklearn GBR (MAE loss) ---")
    gbr_params = dict(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        min_samples_leaf=5, subsample=0.8, loss="absolute_error",
        random_state=42
    )
    r, oof, te_pred, ratio = cv_model(None, gbr_params, X_tr, y_tr, X_te, splits,
                                       "D: sklearn GBR MAE", is_sklearn=True)
    results["D_gbr_mae"] = (r, oof, te_pred, ratio)

    # E: CatBoost Quantile 0.5 (equivalent to MAE at median)
    print("\n--- E: CatBoost Quantile 0.5 ---")
    cat_q50 = dict(
        iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        loss_function="Quantile:alpha=0.5", eval_metric="MAE",
        random_seed=42, thread_count=4, allow_writing_files=False
    )
    r, oof, te_pred, ratio = cv_model(None, cat_q50, X_tr, y_tr, X_te, splits,
                                       "E: CatBoost Quantile 0.5", is_catboost=True)
    results["E_catboost_q50"] = (r, oof, te_pred, ratio)

    # F: Ensemble of A+B+C CatBoost
    print("\n--- F: Ensemble top-3 CatBoost ---")
    top3 = sorted([(k, v) for k, v in results.items() if k.startswith(("A", "B", "C"))],
                  key=lambda x: x[1][0])[:3]
    ens_oof = np.mean([v[1] for _, v in top3], axis=0)
    ens_te  = np.mean([v[2] for _, v in top3], axis=0)
    r_ens = rae(y_tr, ens_oof)
    ratio_ens = ens_te.std() / ens_oof.std()
    results["F_catboost_ens"] = (r_ens, ens_oof, ens_te, ratio_ens)
    print(f"  [F: CatBoost ensemble top-3] RAE={r_ens:.4f}  ratio={ratio_ens:.2f}")

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:55s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid:
        valid = results
        print("WARNING: All configs below collapse threshold, using best anyway")
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, _ = valid[best_label]
    print(f"\nBest: {best_label}  OOF RAE={best_r:.4f}  (nb149 baseline: 0.3069)")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb156_catboost_mae.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb156_catboost_mae.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "156_catboost_mae.csv", index=False)
    print(f"\nSaved: submissions/156_catboost_mae.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
