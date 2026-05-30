"""nb160 — PCA-Reduced OOF Meta-Stack.

Problem: 110 base-only OOFs are highly correlated. LGBM_MAE overfits
the training combination → test predictions collapse (ratio=0.57).

Solution: PCA compress the 110 OOF vectors into K orthogonal components,
then train LGBM_MAE on K PCA components + 5 assay features.
Decorrelated input should prevent collapse while preserving signal.

Tests:
  K=20, 30, 40, 50 components (from base-only 110 OOFs)
  Also: K=20 from full 123 model pool (including meta-stacks)

Comparison baseline: nb149 (0.3069) and nb154_A (0.3008 collapsed).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import PCA

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
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
}

LGBM_MAE = dict(
    n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
    subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05,
    objective="regression_l1", verbose=-1, n_jobs=4, random_state=42
)
LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
)


def load_oofs(n_tr, y_tr, thresh=None, exclude_stems=None):
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
            if thresh is not None and ratio < thresh: continue
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


def cv_lgbm_mae_pca(oof_mat, te_mat, assay_oof, assay_te, y_tr, splits, k_pca, label):
    """PCA-compress oof_mat, fit LGBM_MAE, return OOF RAE and ratio."""
    n_tr = len(y_tr)
    oof_out = np.full(n_tr, np.nan)

    # Fit PCA on all training OOFs (no fold-specific PCA to avoid leakage)
    pca = PCA(n_components=min(k_pca, oof_mat.shape[1]))
    pca.fit(oof_mat)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA k={k_pca}: variance explained={var_explained:.3f}", flush=True)

    X_tr_pca = np.hstack([pca.transform(oof_mat), assay_oof])
    X_te_pca  = np.hstack([pca.transform(te_mat),  assay_te])

    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM_MAE, lgb.Dataset(X_tr_pca[tr_idx], label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof_out[va_idx] = m.predict(X_tr_pca[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof_out[va_idx]):.4f}", flush=True)
    m_full = lgb.train(LGBM_MAE, lgb.Dataset(X_tr_pca, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_pred = m_full.predict(X_te_pca)
    r = rae(y_tr, oof_out)
    ratio = te_pred.std() / oof_out.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label:55s}] RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]", flush=True)
    return r, oof_out, te_pred, ratio


def main():
    print("=== nb160: PCA-Reduced OOF Meta-Stack ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    # Base-only models (no meta-stacks, but keep all with ratio >= 0.58)
    print("Loading base models (exclude meta-stacks, ratio >= 0.58)...")
    mods_base = load_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH, exclude_stems=META_STEMS)
    print(f"  {len(mods_base)} base models loaded")
    oof_base = np.column_stack([m["oof"] for m in mods_base])
    te_base  = np.column_stack([m["te"]  for m in mods_base])

    # Full pool (all models with ratio >= 0.58)
    print("Loading full model pool (ratio >= 0.58)...")
    mods_full = load_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH)
    print(f"  {len(mods_full)} models in full pool")
    oof_full = np.column_stack([m["oof"] for m in mods_full])
    te_full  = np.column_stack([m["te"]  for m in mods_full])

    results = {}

    print("\n=== Base-only OOFs with PCA ===")
    for k in [20, 30, 40, 50]:
        if k > len(mods_base):
            print(f"  Skipping k={k} (only {len(mods_base)} models)")
            continue
        print(f"\n--- Base-only PCA k={k} ---")
        label = f"base_pca{k}"
        r, oof, te_pred, ratio = cv_lgbm_mae_pca(oof_base, te_base, assay_oof, assay_te,
                                                   y_tr, splits, k, label)
        results[label] = (r, oof, te_pred, ratio)

    print("\n=== Full pool OOFs with PCA ===")
    for k in [20, 30, 40]:
        if k > len(mods_full):
            break
        print(f"\n--- Full pool PCA k={k} ---")
        label = f"full_pca{k}"
        r, oof, te_pred, ratio = cv_lgbm_mae_pca(oof_full, te_full, assay_oof, assay_te,
                                                   y_tr, splits, k, label)
        results[label] = (r, oof, te_pred, ratio)

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:55s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid:
        valid = results
        print("WARNING: No config passed ratio threshold")
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBest: {best_label}  OOF RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb149: 0.3069, nb155: 0.3044)")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb160_pca_meta.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb160_pca_meta.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "160_pca_meta.csv", index=False)
    print(f"\nSaved: submissions/160_pca_meta.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
