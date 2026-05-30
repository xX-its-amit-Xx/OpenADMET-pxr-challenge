"""nb171 — CatBoost on Extended Mixed Pool.

Adds nb165 (multiseed_162c) and nb168 (multiseed_catboost) as additional
anchors on top of the original 6 anchors. Hypothesis: these new multi-seed
meta-stacks provide even more de-correlating signal.

Also tests CatBoost with newer anchors included.

Extended anchor set:
  Original 6: nb143, nb144, nb145, nb136, nb134, nb141
  New: nb165_multiseed_162c, nb168_multiseed_catboost (if available)

Tests:
  A: CatBoost depth=8, base+6 orig anchors (replicate nb156 on current pool)
  B: CatBoost depth=8, base+6+2 new anchors
  C: CatBoost depth=8, base+8 anchors, 1500 iters
  D: Multi-seed (5x) on best config
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
    "nb167_xgboost_mae", "nb168_multiseed_catboost", "nb169_rf_et_mae",
    "nb170_grand_v15", "nb171_catboost_mixed_extended",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
}

ANCHOR_STEMS_6 = [
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb136_xgb_meta", "nb134_grand_v9", "nb141_xgb_ablation",
]

ANCHOR_STEMS_EXTENDED = ANCHOR_STEMS_6 + [
    "nb165_multiseed_162c",
    "nb168_multiseed_catboost",
]

LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
)

CAT_D8 = dict(
    loss_function="MAE",
    eval_metric="MAE",
    iterations=600,
    learning_rate=0.04,
    depth=8,
    l2_leaf_reg=5.0,
    subsample=0.8,
    random_seed=42,
    thread_count=4,
    allow_writing_files=False,
    verbose=0,
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


def build_pool(base_mods, all_stem_map, anchor_list, assay_oof, assay_te):
    anchors = [all_stem_map[s] for s in anchor_list if s in all_stem_map]
    mods = base_mods + anchors
    oof_mat = np.column_stack([m["oof"] for m in mods])
    te_mat  = np.column_stack([m["te"]  for m in mods])
    X_tr = np.hstack([oof_mat, assay_oof])
    X_te  = np.hstack([te_mat, assay_te])
    return X_tr, X_te, anchors


def run_cat_cv(cfg, X_tr, y_tr, X_te, splits, label):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = CatBoostRegressor(**cfg)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)
    m_full = CatBoostRegressor(**cfg)
    m_full.fit(X_tr, y_tr)
    te_pred = m_full.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label:55s}] RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]", flush=True)
    return r, oof, te_pred, ratio


def main():
    print("=== nb171: CatBoost on Extended Mixed Pool ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    print("Loading all model OOFs...")
    all_mods = load_base_oofs(n_tr, y_tr, exclude_stems=set())
    all_stem_map = {m["stem"]: m for m in all_mods}
    base_mods = [m for m in all_mods if m["stem"] not in META_STEMS]
    base_mods.sort(key=lambda x: x["rae"])
    print(f"  {len(base_mods)} base models, {len(all_mods)} total")

    # Check which extended anchors are available
    avail_ext = [s for s in ANCHOR_STEMS_EXTENDED if s in all_stem_map]
    print(f"  Extended anchors available: {avail_ext}")

    results = {}

    # A: CatBoost depth=8, 6 original anchors (control)
    print("\n--- A: CatBoost d8, 6 orig anchors ---")
    X_tr_a, X_te_a, anch_a = build_pool(base_mods, all_stem_map, ANCHOR_STEMS_6, assay_oof, assay_te)
    print(f"  Pool shape: {X_tr_a.shape}")
    r_a, oof_a, te_a, ratio_a = run_cat_cv(CAT_D8, X_tr_a, y_tr, X_te_a, splits, "CB_d8_6anchors")
    results["A_d8_6anch"] = (r_a, oof_a, te_a, ratio_a)

    if len(avail_ext) > len(ANCHOR_STEMS_6):
        # B: CatBoost depth=8, extended anchors
        print(f"\n--- B: CatBoost d8, {len(avail_ext)} extended anchors ---")
        X_tr_b, X_te_b, anch_b = build_pool(base_mods, all_stem_map, avail_ext, assay_oof, assay_te)
        print(f"  Pool shape: {X_tr_b.shape}")
        r_b, oof_b, te_b, ratio_b = run_cat_cv(CAT_D8, X_tr_b, y_tr, X_te_b, splits, f"CB_d8_{len(avail_ext)}anchors")
        results["B_d8_ext"] = (r_b, oof_b, te_b, ratio_b)

        # C: CatBoost depth=8, 1500 iters, extended anchors
        print(f"\n--- C: CatBoost d8, 1500 iters, extended anchors ---")
        cfg_c = {**CAT_D8, "iterations": 1500}
        r_c, oof_c, te_c, ratio_c = run_cat_cv(cfg_c, X_tr_b, y_tr, X_te_b, splits, "CB_d8_1500iter_ext")
        results["C_d8_1500_ext"] = (r_c, oof_c, te_c, ratio_c)
    else:
        print(f"\n  NOTE: Extended anchors not yet available — running B/C on 6-anchor pool with more iters")
        cfg_c = {**CAT_D8, "iterations": 1500}
        r_c, oof_c, te_c, ratio_c = run_cat_cv(cfg_c, X_tr_a, y_tr, X_te_a, splits, "CB_d8_1500iter_6anch")
        results["C_d8_1500"] = (r_c, oof_c, te_c, ratio_c)

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:40s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    # Multi-seed on best passing
    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if valid:
        best_single = min(valid, key=lambda k: valid[k][0])
        best_r_s, best_oof_s, best_te_s, _ = valid[best_single]
        print(f"\nBest passing: {best_single}  RAE={best_r_s:.4f}")

        # Use the extended pool if B exists
        if "B_d8_ext" in results and len(avail_ext) > len(ANCHOR_STEMS_6):
            X_tr_best = X_tr_b; X_te_best = X_te_b
        else:
            X_tr_best = X_tr_a; X_te_best = X_te_a

        print(f"\n--- D: Multi-seed (5x) ---")
        seed_oofs = [best_oof_s]; seed_tes = [best_te_s]
        for seed in [123, 456, 789, 1234]:
            cfg_s = {**CAT_D8, "random_seed": seed}
            oof_s = np.full(n_tr, np.nan)
            for fold, (tr_idx, va_idx) in enumerate(splits):
                m = CatBoostRegressor(**cfg_s)
                m.fit(X_tr_best[tr_idx], y_tr[tr_idx])
                oof_s[va_idx] = m.predict(X_tr_best[va_idx])
            m_f = CatBoostRegressor(**cfg_s)
            m_f.fit(X_tr_best, y_tr)
            seed_oofs.append(oof_s); seed_tes.append(m_f.predict(X_te_best))
            print(f"  seed {seed}: RAE={rae(y_tr, oof_s):.4f}", flush=True)
        oof_ms = np.mean(seed_oofs, axis=0); te_ms = np.mean(seed_tes, axis=0)
        r_ms = rae(y_tr, oof_ms); ratio_ms = te_ms.std() / oof_ms.std()
        flag_ms = "PASS" if ratio_ms >= COLLAPSE_THRESH else "FAIL"
        print(f"  [D multi-seed] RAE={r_ms:.4f}  ratio={ratio_ms:.2f}  [{flag_ms}]")
        results["D_multiseed"] = (r_ms, oof_ms, te_ms, ratio_ms)

    all_valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not all_valid: all_valid = results
    final_label = min(all_valid, key=lambda k: all_valid[k][0])
    final_r, final_oof, final_te, final_ratio = all_valid[final_label]
    print(f"\nFINAL: {final_label}  RAE={final_r:.4f}  ratio={final_ratio:.2f}")
    print(f"(nb156 original CatBoost d8: 0.3083)")

    final_te_out = np.clip(final_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb171_catboost_extended.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb171_catboost_extended.npy",  final_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": final_te_out})
    sub.to_csv(SUBMISSIONS / "171_catboost_extended.csv", index=False)
    print(f"Saved: submissions/171_catboost_extended.csv  OOF RAE={final_r:.4f}")


if __name__ == "__main__":
    main()
