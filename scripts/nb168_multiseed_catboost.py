"""nb168 — Multi-seed CatBoost depth=8 on mixed pool.

nb156 CatBoost depth=8 achieves OOF 0.3083 (single seed=42) and becomes the
dominant model in grand v14 (weight=0.384). This notebook runs 10 seeds of
the same config to reduce variance, then tests a multi-seed ensemble.

Also tests:
  V1: 5 seeds, depth=8
  V2: 5 seeds, depth=8, more iterations (1500)
  V3: 5 seeds, depth=8, more leaves (num_leaves implicit via depth)
  V4: 10 seeds, depth=8 (seed ensemble)
  V5: Blend V1+V2+V3 (meta-ensemble of seed ensembles)
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
    "nb167_xgboost_mae",
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

CAT_BASE = dict(
    loss_function="MAE",
    eval_metric="MAE",
    iterations=600,
    learning_rate=0.04,
    depth=8,
    l2_leaf_reg=5.0,
    subsample=0.8,
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
    print("=== nb168: Multi-seed CatBoost depth=8 on Mixed Pool ===\n")

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

    all_mods = load_base_oofs(n_tr, y_tr, exclude_stems=set())
    stem_map = {m["stem"]: m for m in all_mods}
    anchors = [stem_map[s] for s in ANCHOR_STEMS if s in stem_map]
    print(f"  {len(anchors)} anchors: {[a['stem'] for a in anchors]}")

    mixed_mods = base_mods + anchors
    oof_mat = np.column_stack([m["oof"] for m in mixed_mods])
    te_mat  = np.column_stack([m["te"]  for m in mixed_mods])
    X_tr = np.hstack([oof_mat, assay_oof])
    X_te  = np.hstack([te_mat, assay_te])
    print(f"  Mixed pool shape: {X_tr.shape}")

    # V1: 5 seeds, depth=8, base config (600 iters)
    print("\n--- V1: 5 seeds, depth=8, 600 iters ---")
    SEEDS_5 = [42, 123, 456, 789, 1234]
    v1_oofs = []; v1_tes = []
    for seed in SEEDS_5:
        cfg = {**CAT_BASE, "random_seed": seed}
        r, oof_s, te_s, ratio_s = run_cat_cv(cfg, X_tr, y_tr, X_te, splits, f"cat_d8_seed{seed}")
        v1_oofs.append(oof_s); v1_tes.append(te_s)
    oof_v1 = np.mean(v1_oofs, axis=0); te_v1 = np.mean(v1_tes, axis=0)
    r_v1 = rae(y_tr, oof_v1); ratio_v1 = te_v1.std() / oof_v1.std()
    flag_v1 = "PASS" if ratio_v1 >= COLLAPSE_THRESH else "FAIL"
    print(f"  [V1 ensemble 5-seed 600iter] RAE={r_v1:.4f}  ratio={ratio_v1:.2f}  [{flag_v1}]")

    # V2: 5 seeds, depth=8, 1500 iters
    print("\n--- V2: 5 seeds, depth=8, 1500 iters ---")
    v2_oofs = []; v2_tes = []
    for seed in SEEDS_5:
        cfg = {**CAT_BASE, "iterations": 1500, "random_seed": seed}
        r, oof_s, te_s, ratio_s = run_cat_cv(cfg, X_tr, y_tr, X_te, splits, f"cat_d8_1500_seed{seed}")
        v2_oofs.append(oof_s); v2_tes.append(te_s)
    oof_v2 = np.mean(v2_oofs, axis=0); te_v2 = np.mean(v2_tes, axis=0)
    r_v2 = rae(y_tr, oof_v2); ratio_v2 = te_v2.std() / oof_v2.std()
    flag_v2 = "PASS" if ratio_v2 >= COLLAPSE_THRESH else "FAIL"
    print(f"  [V2 ensemble 5-seed 1500iter] RAE={r_v2:.4f}  ratio={ratio_v2:.2f}  [{flag_v2}]")

    # V3: 5 seeds, depth=6 (shallower for diversity)
    print("\n--- V3: 5 seeds, depth=6, 600 iters ---")
    v3_oofs = []; v3_tes = []
    for seed in SEEDS_5:
        cfg = {**CAT_BASE, "depth": 6, "random_seed": seed}
        r, oof_s, te_s, ratio_s = run_cat_cv(cfg, X_tr, y_tr, X_te, splits, f"cat_d6_seed{seed}")
        v3_oofs.append(oof_s); v3_tes.append(te_s)
    oof_v3 = np.mean(v3_oofs, axis=0); te_v3 = np.mean(v3_tes, axis=0)
    r_v3 = rae(y_tr, oof_v3); ratio_v3 = te_v3.std() / oof_v3.std()
    flag_v3 = "PASS" if ratio_v3 >= COLLAPSE_THRESH else "FAIL"
    print(f"  [V3 ensemble 5-seed d6 600iter] RAE={r_v3:.4f}  ratio={ratio_v3:.2f}  [{flag_v3}]")

    # V4: 10 seeds, depth=8 (more thorough seed avg)
    print("\n--- V4: 10 seeds, depth=8, 600 iters ---")
    SEEDS_10 = [42, 123, 456, 789, 1234, 2345, 3456, 4567, 5678, 6789]
    v4_oofs = list(v1_oofs)  # reuse V1 seeds
    v4_tes  = list(v1_tes)
    for seed in SEEDS_10[5:]:
        cfg = {**CAT_BASE, "random_seed": seed}
        r, oof_s, te_s, ratio_s = run_cat_cv(cfg, X_tr, y_tr, X_te, splits, f"cat_d8_seed{seed}")
        v4_oofs.append(oof_s); v4_tes.append(te_s)
    oof_v4 = np.mean(v4_oofs, axis=0); te_v4 = np.mean(v4_tes, axis=0)
    r_v4 = rae(y_tr, oof_v4); ratio_v4 = te_v4.std() / oof_v4.std()
    flag_v4 = "PASS" if ratio_v4 >= COLLAPSE_THRESH else "FAIL"
    print(f"  [V4 ensemble 10-seed] RAE={r_v4:.4f}  ratio={ratio_v4:.2f}  [{flag_v4}]")

    # V5: blend V1+V2+V3 ensembles
    oof_v5 = (oof_v1 + oof_v2 + oof_v3) / 3.0
    te_v5  = (te_v1  + te_v2  + te_v3)  / 3.0
    r_v5 = rae(y_tr, oof_v5); ratio_v5 = te_v5.std() / oof_v5.std()
    flag_v5 = "PASS" if ratio_v5 >= COLLAPSE_THRESH else "FAIL"
    print(f"\n  [V5 blend V1+V2+V3] RAE={r_v5:.4f}  ratio={ratio_v5:.2f}  [{flag_v5}]")

    results = {
        "V1_5seed_600": (r_v1, oof_v1, te_v1, ratio_v1),
        "V2_5seed_1500": (r_v2, oof_v2, te_v2, ratio_v2),
        "V3_5seed_d6": (r_v3, oof_v3, te_v3, ratio_v3),
        "V4_10seed": (r_v4, oof_v4, te_v4, ratio_v4),
        "V5_blend": (r_v5, oof_v5, te_v5, ratio_v5),
    }

    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:40s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid:
        valid = results
        print("\nWARN: no config passed — saving best-ratio config")
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBEST: {best_label}  RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb156 single seed CatBoost d8: 0.3083)")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb168_multiseed_catboost.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb168_multiseed_catboost.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "168_multiseed_catboost.csv", index=False)
    print(f"Saved: submissions/168_multiseed_catboost.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
