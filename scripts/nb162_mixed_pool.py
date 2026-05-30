"""nb162 — Mixed Pool: Base OOF + Best Meta-Stack Anchors.

Hypothesis: nb149 (113 models = base + some meta-stacks, ratio=0.58 PASS)
vs nb154-A (110 base-only models, ratio=0.57 FAIL).

The 3 meta-stack OOFs in nb149's pool (nb143, nb144, nb136?) act as
de-correlating anchors that prevent test prediction collapse.

Test: add ONLY the 2-5 best meta-stacks to base-only pool.
Candidates: nb149, nb143, nb144, nb145, nb136 (sorted by RAE).

Wait — nb149 is the model we're TRAINING, so we can't include its own
OOF as a feature. Instead, in the spirit of clean stacking:
  - Anchor set: nb143 (0.3143), nb144 (0.3126), nb145 (0.3125), nb136 (0.3334)
  - These are Level-2 outputs from XGBoost-based meta-stacks (not LGBM)
  - They provide orthogonal signal and may prevent LGBM_MAE collapse

Tests:
  A: base-only + nb143 + nb144 (best XGB meta-stacks)
  B: base-only + nb143 + nb144 + nb136
  C: base-only + nb143 + nb144 + nb145 + nb136 + nb134
  D: nb149 exact pool (reproduce, control)
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

# Pure base model stems to exclude (meta-stacks)
BASE_META_EXCLUDE = {
    "nb136_xgb_meta", "nb138_elnet_blend", "nb139_adaptive_blend",
    "nb140_xgb_lgbm_meta", "nb141_xgb_ablation", "nb142_xgb_calibrated",
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb146_pca_oof_meta", "nb147_oofrdkit_meta", "nb148_meta_disagreement",
    "nb149_meta_maeloss", "nb150_residual_ensemble", "nb151_grand_v11",
    "nb152_lgbm_mae_tuned", "nb153_grand_v12", "nb154_lgbm_mae_filtered",
    "nb155_grand_v13", "nb156_catboost_mae", "nb157_optuna_lgbm_mae",
    "nb158_collapse_fix", "nb159_variance_scaled_ensemble", "nb160_pca_meta",
    "nb161_neural_meta", "nb162_mixed_pool",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
}

# Anchor meta-stacks to selectively include
ANCHOR_STEMS = [
    "nb143_oofassay_meta",   # XGB+LGBM blend (0.3143)
    "nb144_grand_v10",       # SLSQP ensemble (0.3126)
    "nb145_level3_meta",     # SLSQP Level-3 (0.3125)
    "nb136_xgb_meta",        # XGBoost meta (0.3334)
    "nb134_grand_v9",        # Grand v9 (0.3303)
    "nb141_xgb_ablation",    # XGB ablation (0.3379)
]

LGBM_MAE = dict(
    n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
    subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05,
    objective="regression_l1", verbose=-1, n_jobs=4, random_state=42
)
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


def cv_lgbm_mae(X_tr, y_tr, X_te, splits, label):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM_MAE, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)
    m_full = lgb.train(LGBM_MAE, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_pred = m_full.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label:55s}] RAE={r:.4f}  n_feats={X_tr.shape[1]}  ratio={ratio:.2f}  [{flag}]", flush=True)
    return r, oof, te_pred, ratio


def main():
    print("=== nb162: Mixed Pool — Base OOF + Best Meta-Stack Anchors ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    print("Loading base models (exclude all meta-stacks)...")
    base_mods = load_base_oofs(n_tr, y_tr, exclude_stems=BASE_META_EXCLUDE)
    print(f"  {len(base_mods)} base models")
    oof_base = np.column_stack([m["oof"] for m in base_mods])
    te_base  = np.column_stack([m["te"]  for m in base_mods])

    print("Loading anchor meta-stacks...")
    anchors = {}
    for s in ANCHOR_STEMS:
        data = load_stem(s, n_tr)
        if data is not None:
            anchors[s] = data
            r = rae(y_tr, data["oof"])
            ratio = data["te"].std() / data["oof"].std()
            print(f"  {s:50s}  RAE={r:.4f}  ratio={ratio:.2f}")

    results = {}

    # A: base + nb143 + nb144
    print(f"\n--- A: base({len(base_mods)}) + nb143 + nb144 ---")
    anchors_a = [s for s in ["nb143_oofassay_meta", "nb144_grand_v10"] if s in anchors]
    oof_a = np.column_stack([oof_base] + [anchors[s]["oof"].reshape(-1,1) for s in anchors_a])
    te_a  = np.column_stack([te_base]  + [anchors[s]["te"].reshape(-1,1)  for s in anchors_a])
    X_tr = np.hstack([oof_a, assay_oof]); X_te = np.hstack([te_a, assay_te])
    r, oof, te_pred, ratio = cv_lgbm_mae(X_tr, y_tr, X_te, splits, "A: base+nb143+nb144")
    results["A_base_143_144"] = (r, oof, te_pred, ratio)

    # B: base + nb143 + nb144 + nb136
    print(f"\n--- B: base({len(base_mods)}) + nb143 + nb144 + nb136 ---")
    anchors_b = [s for s in ["nb143_oofassay_meta", "nb144_grand_v10", "nb136_xgb_meta"] if s in anchors]
    oof_b = np.column_stack([oof_base] + [anchors[s]["oof"].reshape(-1,1) for s in anchors_b])
    te_b  = np.column_stack([te_base]  + [anchors[s]["te"].reshape(-1,1)  for s in anchors_b])
    X_tr = np.hstack([oof_b, assay_oof]); X_te = np.hstack([te_b, assay_te])
    r, oof, te_pred, ratio = cv_lgbm_mae(X_tr, y_tr, X_te, splits, "B: base+nb143+nb144+nb136")
    results["B_base_143_144_136"] = (r, oof, te_pred, ratio)

    # C: base + all 6 anchor stems
    print(f"\n--- C: base({len(base_mods)}) + {len(anchors)} anchors ---")
    anchor_list = [s for s in ANCHOR_STEMS if s in anchors]
    oof_c = np.column_stack([oof_base] + [anchors[s]["oof"].reshape(-1,1) for s in anchor_list])
    te_c  = np.column_stack([te_base]  + [anchors[s]["te"].reshape(-1,1)  for s in anchor_list])
    X_tr = np.hstack([oof_c, assay_oof]); X_te = np.hstack([te_c, assay_te])
    r, oof, te_pred, ratio = cv_lgbm_mae(X_tr, y_tr, X_te, splits, f"C: base+{len(anchor_list)} anchors")
    results["C_base_all_anchors"] = (r, oof, te_pred, ratio)

    # D: top-5 anchors only (no base models, pure anchor meta-stack)
    print(f"\n--- D: {len(anchor_list)} anchors only (no base) + assay ---")
    anchor_list5 = anchor_list[:5]
    oof_d = np.column_stack([anchors[s]["oof"].reshape(-1,1) for s in anchor_list5])
    te_d  = np.column_stack([anchors[s]["te"].reshape(-1,1)  for s in anchor_list5])
    X_tr = np.hstack([oof_d, assay_oof]); X_te = np.hstack([te_d, assay_te])
    r, oof, te_pred, ratio = cv_lgbm_mae(X_tr, y_tr, X_te, splits, "D: anchors-only+assay")
    results["D_anchors_only"] = (r, oof, te_pred, ratio)

    # E: nb149 exact pool (all models with ratio >= 0.58, reproduce control)
    print(f"\n--- E: Full pool (all, ratio>=0.58) = nb149 approach ---")
    all_mods = load_base_oofs(n_tr, y_tr)  # no stem exclusion, just ratio filter
    print(f"  {len(all_mods)} models in full pool")
    oof_e = np.column_stack([m["oof"] for m in all_mods])
    te_e  = np.column_stack([m["te"]  for m in all_mods])
    X_tr = np.hstack([oof_e, assay_oof]); X_te = np.hstack([te_e, assay_te])
    r, oof, te_pred, ratio = cv_lgbm_mae(X_tr, y_tr, X_te, splits, f"E: full pool ({len(all_mods)} models)")
    results["E_full_pool"] = (r, oof, te_pred, ratio)

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
    print(f"(nb149: 0.3069, nb155: 0.3044)")

    if best_r < 0.3069:
        print("*** NEW BEST SINGLE MODEL! ***")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb162_mixed_pool.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb162_mixed_pool.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "162_mixed_pool.csv", index=False)
    print(f"\nSaved: submissions/162_mixed_pool.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
