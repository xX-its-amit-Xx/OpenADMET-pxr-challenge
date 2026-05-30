"""nb158 — Fix Test Collapse for Base-Only LGBM_MAE.

nb154 config A (110 base models, exclude meta-stacks) got OOF 0.3008 —
best ever — but ratio=0.57 (collapse, below 0.58 threshold).

Hypothesis: with 110 highly-correlated OOF features, LGBM_MAE overfits
feature selection on training → test predictions have lower variance.

Fix strategies:
  A1: More regularization (higher reg_lambda, min_child_samples=15)
  A2: Much fewer trees (n_estimators=300, lr=0.05) — less capacity
  A3: Constrained max_depth (depth-based tree instead of leaves)
  A4: Dropout regularization (DART booster)
  A5: Add explicit test-distribution anchor (blend A with equal-weight average)
  A6: Use raw nb149 approach but only with top-60 non-meta base models
  A7: Multi-seed on more regularized config

KEY TEST: can we get ratio >= 0.58 while keeping RAE < 0.3069?
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
}

LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
)


def load_base_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH, exclude_stems=None, top_k=None):
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
    if top_k is not None:
        results = results[:top_k]
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


def cv_lgbm_mae(cfg, X_tr, y_tr, X_te, splits, label):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(cfg, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr[va_idx])
    m_full = lgb.train(cfg, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_pred = m_full.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label:55s}] RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]", flush=True)
    return r, oof, te_pred, ratio


def main():
    print("=== nb158: Fix Test Collapse for Base-Only LGBM_MAE ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    # Load base-only models
    print("Loading base models (exclude meta-stacks)...")
    mods_all = load_base_oofs(n_tr, y_tr, exclude_stems=META_STEMS)
    print(f"  {len(mods_all)} base models loaded")
    oof_mat = np.column_stack([m["oof"] for m in mods_all])
    te_mat  = np.column_stack([m["te"]  for m in mods_all])
    X_tr_all = np.hstack([oof_mat, assay_oof])
    X_te_all  = np.hstack([te_mat, assay_te])

    # Load top-60 base models
    mods_60 = load_base_oofs(n_tr, y_tr, exclude_stems=META_STEMS, top_k=60)
    oof_60 = np.column_stack([m["oof"] for m in mods_60])
    te_60  = np.column_stack([m["te"]  for m in mods_60])
    X_tr_60 = np.hstack([oof_60, assay_oof])
    X_te_60  = np.hstack([te_60, assay_te])

    results = {}

    print("\n=== Testing collapse-fix strategies ===")

    # A1: More regularization (L2 + higher min_child)
    print("\n--- A1: High regularization ---")
    cfg_a1 = dict(n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=15,
                  subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05, reg_lambda=0.5,
                  objective="regression_l1", verbose=-1, n_jobs=4, random_state=42)
    r, oof, te_pred, ratio = cv_lgbm_mae(cfg_a1, X_tr_all, y_tr, X_te_all, splits, "A1: high_reg")
    results["A1_high_reg"] = (r, oof, te_pred, ratio)

    # A2: Fewer trees (300)
    print("\n--- A2: Fewer trees ---")
    cfg_a2 = dict(n_estimators=300, num_leaves=63, learning_rate=0.05, min_child_samples=5,
                  subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05,
                  objective="regression_l1", verbose=-1, n_jobs=4, random_state=42)
    r, oof, te_pred, ratio = cv_lgbm_mae(cfg_a2, X_tr_all, y_tr, X_te_all, splits, "A2: 300 trees")
    results["A2_300trees"] = (r, oof, te_pred, ratio)

    # A3: Max depth constraint (depth=6 instead of unlimited leaves)
    print("\n--- A3: Max depth=6 ---")
    cfg_a3 = dict(n_estimators=800, max_depth=6, learning_rate=0.03, min_child_samples=5,
                  subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05,
                  objective="regression_l1", verbose=-1, n_jobs=4, random_state=42)
    r, oof, te_pred, ratio = cv_lgbm_mae(cfg_a3, X_tr_all, y_tr, X_te_all, splits, "A3: depth=6")
    results["A3_depth6"] = (r, oof, te_pred, ratio)

    # A4: DART booster (dropout regularization)
    print("\n--- A4: DART ---")
    cfg_a4 = dict(n_estimators=600, num_leaves=63, learning_rate=0.05, min_child_samples=5,
                  subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05,
                  boosting_type="dart", drop_rate=0.1, skip_drop=0.5,
                  objective="regression_l1", verbose=-1, n_jobs=4, random_state=42)
    r, oof, te_pred, ratio = cv_lgbm_mae(cfg_a4, X_tr_all, y_tr, X_te_all, splits, "A4: DART")
    results["A4_dart"] = (r, oof, te_pred, ratio)

    # A5: Blend config A preds with equal-weight average (anchor to equal blend)
    print("\n--- A5: Anchored blend ---")
    equal_oof = oof_mat.mean(axis=1)
    equal_te  = te_mat.mean(axis=1)
    cfg_a5 = dict(n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
                  subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05,
                  objective="regression_l1", verbose=-1, n_jobs=4, random_state=42)
    _, oof_a5_raw, te_a5_raw, _ = cv_lgbm_mae(cfg_a5, X_tr_all, y_tr, X_te_all, splits, "A5_raw")
    for alpha in [0.85, 0.90, 0.95]:
        oof_blend = alpha * oof_a5_raw + (1-alpha) * equal_oof
        te_blend  = alpha * te_a5_raw  + (1-alpha) * equal_te
        r_b = rae(y_tr, oof_blend)
        ratio_b = te_blend.std() / oof_blend.std()
        flag = "PASS" if ratio_b >= COLLAPSE_THRESH else "FAIL"
        print(f"  A5 alpha={alpha:.2f}: RAE={r_b:.4f}  ratio={ratio_b:.2f}  [{flag}]", flush=True)
        results[f"A5_alpha{int(alpha*100)}"] = (r_b, oof_blend, te_blend, ratio_b)

    # A6: Top-60 base models (even cleaner subset)
    print("\n--- A6: Top-60 base models ---")
    cfg_a6 = dict(n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
                  subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05,
                  objective="regression_l1", verbose=-1, n_jobs=4, random_state=42)
    r, oof, te_pred, ratio = cv_lgbm_mae(cfg_a6, X_tr_60, y_tr, X_te_60, splits, "A6: top-60 base")
    results["A6_top60"] = (r, oof, te_pred, ratio)

    # A7: Multi-seed ensemble on A3 (depth=6) which might have better ratio
    if results.get("A3_depth6"):
        a3_r, _, _, a3_ratio = results["A3_depth6"]
        if a3_ratio >= COLLAPSE_THRESH:
            print("\n--- A7: Multi-seed on A3 (best passing config so far) ---")
            seed_oofs = []; seed_tes = []
            for seed in [42, 123, 456, 789, 1234]:
                cfg_s = {**cfg_a3, "random_state": seed}
                oof_s = np.full(n_tr, np.nan)
                for fold, (tr_idx, va_idx) in enumerate(splits):
                    m = lgb.train(cfg_s, lgb.Dataset(X_tr_all[tr_idx], label=y_tr[tr_idx]),
                                  callbacks=[lgb.log_evaluation(-1)])
                    oof_s[va_idx] = m.predict(X_tr_all[va_idx])
                m_f = lgb.train(cfg_s, lgb.Dataset(X_tr_all, label=y_tr),
                                callbacks=[lgb.log_evaluation(-1)])
                seed_oofs.append(oof_s); seed_tes.append(m_f.predict(X_te_all))
            oof_a7 = np.mean(seed_oofs, axis=0)
            te_a7  = np.mean(seed_tes, axis=0)
            r_a7 = rae(y_tr, oof_a7)
            ratio_a7 = te_a7.std() / oof_a7.std()
            flag = "PASS" if ratio_a7 >= COLLAPSE_THRESH else "FAIL"
            print(f"  [A7: multi-seed A3] RAE={r_a7:.4f}  ratio={ratio_a7:.2f}  [{flag}]")
            results["A7_multiseed_a3"] = (r_a7, oof_a7, te_a7, ratio_a7)

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:55s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid:
        valid = results
        print("WARNING: No config passed ratio threshold — using best anyway")
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBest: {best_label}  OOF RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb149: 0.3069, nb154_A: 0.3008 [FAIL])")

    if best_r < 0.3069:
        print("*** NEW BEST SINGLE MODEL! ***")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb158_collapse_fix.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb158_collapse_fix.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "158_collapse_fix.csv", index=False)
    print(f"\nSaved: submissions/158_collapse_fix.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
