"""nb140 — Deep XGBoost+LGBM Meta-Stack with Hyperparameter Search.

nb136 XGBoost meta-stack achieved OOF 0.3334. This notebook explores:
  1. XGBoost hyperparameter tuning (depth, learning rate, regularization)
  2. LGBM meta-stack with expanded features (same 105 OOF inputs as nb136)
  3. CatBoost meta-stack (different boosting algorithm)
  4. Best of all three + their pairwise blends

All use same input: [OOF_all (105) + structural (2265) + assay (5)] = 2375 features
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

# Multiple XGBoost configs to try
XGB_CONFIGS = [
    dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
         colsample_bytree=0.6, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
         tree_method="hist", device="cpu", verbosity=0, n_jobs=4, random_state=SEED),
    dict(n_estimators=700, max_depth=4, learning_rate=0.04, subsample=0.75,
         colsample_bytree=0.5, min_child_weight=8, reg_alpha=0.2, reg_lambda=2.0,
         tree_method="hist", device="cpu", verbosity=0, n_jobs=4, random_state=SEED),
    dict(n_estimators=1000, max_depth=3, learning_rate=0.03, subsample=0.8,
         colsample_bytree=0.7, min_child_weight=5, reg_alpha=0.05, reg_lambda=1.0,
         tree_method="hist", device="cpu", verbosity=0, n_jobs=4, random_state=SEED),
]

LGBM_META_CONFIGS = [
    dict(n_estimators=500, num_leaves=31, learning_rate=0.05, min_child_samples=10,
         subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, verbose=-1, n_jobs=4, random_state=SEED),
    dict(n_estimators=700, num_leaves=48, learning_rate=0.04, min_child_samples=8,
         subsample=0.75, colsample_bytree=0.5, reg_alpha=0.2, verbose=-1, n_jobs=4, random_state=SEED),
]


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


def full_metrics(y_true, y_pred, label=""):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return rae_v


def cv_model(model_class, params, X_tr, y_tr, X_te, splits, label):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = model_class(**params)
        if hasattr(m, 'fit'):
            kw = {}
            if isinstance(m, xgb.XGBRegressor):
                kw = {"verbose": False}
            m.fit(X_tr[tr_idx], y_tr[tr_idx], **kw)
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)
    m_full = model_class(**params)
    if hasattr(m_full, 'fit'):
        kw = {"verbose": False} if isinstance(m_full, xgb.XGBRegressor) else {}
        m_full.fit(X_tr, y_tr, **kw)
    te_pred = m_full.predict(X_te)
    full_metrics(y_tr, oof, label)
    ratio = te_pred.std() / oof.std()
    print(f"    Test: std={te_pred.std():.3f}  ratio={ratio:.2f}", flush=True)
    return oof, te_pred, ratio


def main():
    print("=== nb140: Deep XGBoost+LGBM Meta-Stack ===\n")

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

    print("Computing structural features...")
    X_str = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))

    print("Loading OOF predictions...")
    models = load_model_oofs(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded")

    oof_all = np.column_stack([m["oof"] for m in models])
    te_all  = np.column_stack([m["te"]  for m in models])

    print("Aux OOF...")
    oof_emax = np.full(n_tr, np.nan)
    oof_null = np.full(n_tr, np.nan)
    oof_sel  = np.full(n_tr, np.nan)
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

    X_meta_tr = np.hstack([oof_all, X_str, assay_oof])
    X_meta_te = np.hstack([te_all,  X_str_te, assay_te])
    print(f"Meta features: {X_meta_tr.shape}")

    results = {}

    # XGBoost configs
    for ci, cfg in enumerate(XGB_CONFIGS):
        label = f"XGBoost config_{ci}"
        print(f"\n=== {label} ===")
        oof, te_pred, ratio = cv_model(xgb.XGBRegressor, cfg,
                                        X_meta_tr, y_tr, X_meta_te, splits, label)
        results[label] = (rae(y_tr, oof), oof, te_pred, ratio)

    # LGBM meta configs
    for ci, cfg in enumerate(LGBM_META_CONFIGS):
        label = f"LGBM_meta config_{ci}"
        print(f"\n=== {label} ===")
        oof = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = lgb.train(cfg, lgb.Dataset(X_meta_tr[tr_idx], label=y_tr[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_meta_tr[va_idx])
            print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)
        m_full = lgb.train(cfg, lgb.Dataset(X_meta_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
        te_pred = m_full.predict(X_meta_te)
        full_metrics(y_tr, oof, label)
        ratio = te_pred.std() / oof.std()
        print(f"    Test: std={te_pred.std():.3f}  ratio={ratio:.2f}", flush=True)
        results[label] = (rae(y_tr, oof), oof, te_pred, ratio)

    # Find best
    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid:
        valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\n=== BEST: {best_label}  OOF RAE={best_r:.4f} ===")

    # Pairwise blends of top-3
    top3 = sorted(valid.items(), key=lambda x: x[1][0])[:3]
    print(f"\nPairwise blends of top-3:")
    for i in range(len(top3)):
        for j in range(i+1, len(top3)):
            lab_i, (r_i, oof_i, te_i, _) = top3[i]
            lab_j, (r_j, oof_j, te_j, _) = top3[j]
            # Sweep alpha
            for alpha in [0.3, 0.5, 0.7]:
                blend_oof = alpha * oof_i + (1-alpha) * oof_j
                r_blend = rae(y_tr, blend_oof)
                if r_blend < best_r:
                    best_r = r_blend
                    best_oof = blend_oof
                    best_te  = np.clip(alpha * te_i + (1-alpha) * te_j,
                                       y_tr.min() - 0.5, y_tr.max() + 0.5)
                    best_label = f"{lab_i}({alpha:.1f}) + {lab_j}({1-alpha:.1f})"
            mins = min(rae(y_tr, a*top3[i][1][1]+(1-a)*top3[j][1][1]) for a in [0.3,0.5,0.7])
            print(f"  {lab_i[:20]} + {lab_j[:20]}: alpha=sweep min={mins:.4f}")

    print(f"\nFINAL BEST: {best_label}  OOF RAE={best_r:.4f}")
    best_te = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio = best_te.std() / best_oof.std()
    print(f"Test: med={np.median(best_te):.2f}  std={best_te.std():.3f}  ratio={ratio:.2f}")

    np.save(DATA_PROCESSED / "oof_nb140_xgb_lgbm_meta.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb140_xgb_lgbm_meta.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "140_xgb_lgbm_meta.csv", index=False)
    print(f"\nSaved: submissions/140_xgb_lgbm_meta.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
