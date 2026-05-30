"""nb146 — PCA-Reduced OOF Meta-Stack.

nb141 ablation showed: OOF+assay (116 feats) > OOF+str+assay (2381 feats).
Hypothesis: 111 OOF inputs are highly correlated (all predict pEC50 from same
training data). XGBoost handles this via colsample_bytree but could be improved
by decorrelating first.

PCA reduces 111 correlated OOF → K orthogonal components capturing most variance.
Then XGBoost on [PCA_OOF (K) + assay (5)] = K+5 features, all decorrelated.

Try K = 10, 20, 30, 50.
Also compare to nb143 baseline (OOF+assay, no PCA).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
COLLAPSE_THRESH = 0.58
LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
XGB_BEST = dict(
    n_estimators=800, max_depth=6, learning_rate=0.04, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.05, reg_lambda=0.5,
    tree_method="hist", device="cpu", verbosity=0, n_jobs=4, random_state=SEED
)


def load_model_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
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


def cv_xgb_pca(params, X_tr_raw, y_tr, X_te_raw, assay_tr, assay_te, splits, n_components, label):
    """CV with PCA fitted on TRAINING fold only (no leakage)."""
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Fit PCA on training fold
        scaler = StandardScaler()
        X_tr_fold = scaler.fit_transform(X_tr_raw[tr_idx])
        X_va_fold = scaler.transform(X_tr_raw[va_idx])

        pca = PCA(n_components=n_components, random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr_fold)
        X_va_pca = pca.transform(X_va_fold)

        X_tr_in = np.hstack([X_tr_pca, assay_tr[tr_idx]])
        X_va_in = np.hstack([X_va_pca, assay_tr[va_idx]])

        m = xgb.XGBRegressor(**params)
        m.fit(X_tr_in, y_tr[tr_idx], verbose=False)
        oof[va_idx] = m.predict(X_va_in)

    # Full PCA for test
    scaler_f = StandardScaler()
    X_tr_full = scaler_f.fit_transform(X_tr_raw)
    X_te_full  = scaler_f.transform(X_te_raw)
    pca_f = PCA(n_components=n_components, random_state=SEED)
    X_tr_pca_f = pca_f.fit_transform(X_tr_full)
    X_te_pca_f = pca_f.transform(X_te_full)

    X_tr_in_f = np.hstack([X_tr_pca_f, assay_tr])
    X_te_in_f = np.hstack([X_te_pca_f, assay_te])

    m_full = xgb.XGBRegressor(**params)
    m_full.fit(X_tr_in_f, y_tr, verbose=False)
    te_pred = m_full.predict(X_te_in_f)

    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    print(f"  [PCA-{n_components:3d} {label:30s}] RAE={r:.4f}  ratio={ratio:.2f}", flush=True)
    return r, oof, te_pred, ratio


def main():
    print("=== nb146: PCA-Reduced OOF Meta-Stack ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

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

    print("Computing structural features for aux models...")
    X_str = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))

    print("Loading OOF predictions...")
    models = load_model_oofs(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded")
    oof_all = np.column_stack([m["oof"] for m in models])
    te_all  = np.column_stack([m["te"]  for m in models])

    print("Aux OOF (assay features)...")
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

    # PCA variance analysis
    print("\nPCA variance analysis (on full OOF matrix):")
    scaler_a = StandardScaler()
    oof_scaled = scaler_a.fit_transform(oof_all)
    pca_analysis = PCA(random_state=SEED)
    pca_analysis.fit(oof_scaled)
    cumvar = np.cumsum(pca_analysis.explained_variance_ratio_)
    for k in [5, 10, 20, 30, 50]:
        print(f"  PC-{k:3d}: {cumvar[k-1]*100:.1f}% variance explained")

    # Sweep PCA components
    print(f"\nPCA-OOF + assay meta-stack (XGB best config):")
    results = {}
    for n_pca in [10, 20, 30, 50]:
        r, oof, te_pred, ratio = cv_xgb_pca(
            XGB_BEST, oof_all, y_tr, te_all, assay_oof, assay_te, splits, n_pca, "XGB_best")
        results[f"PCA-{n_pca}+assay+XGB"] = (r, oof, te_pred, ratio)

    # Also test with LGBM at best PCA size
    # Find best PCA size
    best_pca_k = min(results, key=lambda k: results[k][0])
    best_n_pca = int(best_pca_k.split("-")[1].split("+")[0])
    print(f"\nBest PCA size: {best_n_pca}  RAE={results[best_pca_k][0]:.4f}")

    # Comparison: OOF+assay (no PCA) with same XGB config
    print(f"\nBaseline OOF+assay (no PCA, {n_mod}+5 features):")
    X_direct_tr = np.hstack([oof_all, assay_oof])
    X_direct_te = np.hstack([te_all,  assay_te])
    oof_direct = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = xgb.XGBRegressor(**XGB_BEST)
        m.fit(X_direct_tr[tr_idx], y_tr[tr_idx], verbose=False)
        oof_direct[va_idx] = m.predict(X_direct_tr[va_idx])
    m_full = xgb.XGBRegressor(**XGB_BEST)
    m_full.fit(X_direct_tr, y_tr, verbose=False)
    te_direct = m_full.predict(X_direct_te)
    r_direct = rae(y_tr, oof_direct)
    ratio_direct = te_direct.std() / oof_direct.std()
    results["OOF+assay_direct"] = (r_direct, oof_direct, te_direct, ratio_direct)
    print(f"  RAE={r_direct:.4f}  ratio={ratio_direct:.2f}")

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:40s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, _ = valid[best_label]
    print(f"\nBest: {best_label}  OOF RAE={best_r:.4f}")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb146_pca_oof_meta.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb146_pca_oof_meta.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "146_pca_oof_meta.csv", index=False)
    print(f"\nSaved: submissions/146_pca_oof_meta.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
