"""nb149 — Meta-Stack with MAE/Quantile Loss Objectives.

Competition metric = RAE = MAE / baseline_MAE.
XGBoost default = MSE loss. Switching to MAE (quantile at 0.5) or Huber
loss directly optimizes the competition metric.

nb141 winner (Variant B: OOF+assay, 116 features):
  - XGBoost MSE:  OOF RAE=0.3186

This notebook tests:
  A: XGBoost MSE (baseline replica of nb141 Variant B)
  B: XGBoost MAE (reg:absoluteerror)
  C: XGBoost Huber (reg:pseudohubererror, delta=1.35)
  D: XGBoost Quantile50 (reg:quantileerror, quantile_alpha=0.5)
  E: LGBM MAE (objective='mape' is wrong; use objective='regression_l1')

Note: XGBoost MAE is slower (no exact gradient) but should generalize better
for the RAE metric.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
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

XGB_BASE = dict(
    n_estimators=800, max_depth=6, learning_rate=0.04, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.05, reg_lambda=0.5,
    tree_method="hist", device="cpu", verbosity=0, n_jobs=4, random_state=SEED
)

CONFIGS = [
    ("A_MSE",       {**XGB_BASE, "objective": "reg:squarederror"}),
    ("B_MAE",       {**XGB_BASE, "objective": "reg:absoluteerror", "n_estimators": 600}),
    ("C_Huber",     {**XGB_BASE, "objective": "reg:pseudohubererror"}),
    ("D_Quantile50",{**XGB_BASE, "objective": "reg:quantileerror",  "quantile_alpha": 0.5}),
]

LGBM_MAE = dict(
    n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
    subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05, objective="regression_l1",
    verbose=-1, n_jobs=4, random_state=SEED
)
LGBM_MSE = dict(
    n_estimators=800, num_leaves=63, learning_rate=0.03, min_child_samples=5,
    subsample=0.8, colsample_bytree=1.0, reg_alpha=0.05, objective="regression",
    verbose=-1, n_jobs=4, random_state=SEED
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


def cv_xgb(params, X_tr, y_tr, X_te, splits, label):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr[tr_idx], y_tr[tr_idx], verbose=False)
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)
    m_full = xgb.XGBRegressor(**params)
    m_full.fit(X_tr, y_tr, verbose=False)
    te_pred = m_full.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    print(f"  [{label:50s}] RAE={r:.4f}  ratio={ratio:.2f}", flush=True)
    return r, oof, te_pred, ratio


def cv_lgbm(cfg, X_tr, y_tr, X_te, splits, label):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(cfg, lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)
    m_full = lgb.train(cfg, lgb.Dataset(X_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_pred = m_full.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    print(f"  [{label:50s}] RAE={r:.4f}  ratio={ratio:.2f}", flush=True)
    return r, oof, te_pred, ratio


def main():
    print("=== nb149: Meta-Stack with MAE/Quantile Loss ===\n")

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

    X_meta_tr = np.hstack([oof_all, assay_oof])
    X_meta_te = np.hstack([te_all,  assay_te])
    print(f"Meta features: {X_meta_tr.shape}")

    results = {}
    print("\n=== XGBoost Loss Function Variants ===")
    for cfg_name, cfg in CONFIGS:
        print(f"\n--- {cfg_name} ---")
        r, oof, te_pred, ratio = cv_xgb(cfg, X_meta_tr, y_tr, X_meta_te, splits, cfg_name)
        results[cfg_name] = (r, oof, te_pred, ratio)

    print(f"\n--- E_LGBM_MAE ---")
    r, oof, te_pred, ratio = cv_lgbm(LGBM_MAE, X_meta_tr, y_tr, X_meta_te, splits, "E_LGBM_MAE")
    results["E_LGBM_MAE"] = (r, oof, te_pred, ratio)

    print(f"\n--- F_LGBM_MSE ---")
    r, oof, te_pred, ratio = cv_lgbm(LGBM_MSE, X_meta_tr, y_tr, X_meta_te, splits, "F_LGBM_MSE")
    results["F_LGBM_MSE"] = (r, oof, te_pred, ratio)

    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:40s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, _ = valid[best_label]
    print(f"\nBest: {best_label}  OOF RAE={best_r:.4f}")

    # Blend best with nb141-B baseline if available
    nb141_p = DATA_PROCESSED / "oof_nb141_xgb_ablation.npy"
    if nb141_p.exists():
        nb141_oof = np.load(nb141_p).astype(np.float64)
        if nb141_oof.ndim == 2: nb141_oof = nb141_oof[:, 0]
        print(f"\nBlend with nb141 (RAE={rae(y_tr, nb141_oof):.4f}):")
        for alpha in [0.3, 0.5, 0.7]:
            b = alpha * best_oof + (1-alpha) * nb141_oof
            print(f"  alpha(nb149)={alpha:.1f}: RAE={rae(y_tr, b):.4f}")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb149_meta_maeloss.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb149_meta_maeloss.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "149_meta_maeloss.csv", index=False)
    print(f"\nSaved: submissions/149_meta_maeloss.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
