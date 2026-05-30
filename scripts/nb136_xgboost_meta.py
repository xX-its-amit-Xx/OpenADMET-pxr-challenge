"""nb136 — XGBoost Deep Meta-Stack.

Replicate the nb109 deep meta-stack architecture but using XGBoost instead
of LGBM as the level-2 learner. XGBoost uses exact splits (vs LGBM's
leaf-by-leaf), handles high-dimensional inputs differently, and provides
model diversity.

Also: use XGBoost gradient boosting on the counter_delta OOF features
(which nb129 showed are crucial for the best ensemble).

Input features (same as nb109):
  - All OOF predictions from non-collapsed models (up to 100)
  - Structural features (2265 dims)
  - Assay aux (5 dims): Emax, null, selectivity, has_null, log_Emax
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
XGB_MAIN = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
    tree_method="hist", device="cpu", verbosity=0, n_jobs=4
)


def load_model_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    """Load all non-collapsed OOF predictions."""
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


def main():
    print("=== nb136: XGBoost Deep Meta-Stack ===\n")

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
    print(f"  {len(models)} models loaded")
    for m in models[:8]:
        print(f"    {m['stem']:45s}  RAE={m['rae']:.4f}  ratio={m['ratio']:.2f}")

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

    # Full meta-stack: oof_all + structural + assay
    X_meta_tr = np.hstack([oof_all, X_str, assay_oof])
    X_meta_te = np.hstack([te_all,  X_str_te, assay_te])
    print(f"Meta-stack features: {X_meta_tr.shape}")

    print("\n=== XGBoost Meta-Stack (CV) ===")
    oof_xgb = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = xgb.XGBRegressor(**XGB_MAIN)
        m.fit(X_meta_tr[tr_idx], y_tr[tr_idx],
              eval_set=[(X_meta_tr[va_idx], y_tr[va_idx])],
              verbose=False)
        oof_xgb[va_idx] = m.predict(X_meta_tr[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof_xgb[va_idx]):.4f}", flush=True)

    m_full = xgb.XGBRegressor(**XGB_MAIN)
    m_full.fit(X_meta_tr, y_tr, verbose=False)
    te_xgb = m_full.predict(X_meta_te)
    te_xgb = np.clip(te_xgb, y_tr.min() - 0.5, y_tr.max() + 0.5)
    full_metrics(y_tr, oof_xgb, "XGBoost Meta-Stack")
    ratio = te_xgb.std() / oof_xgb.std()
    print(f"  Test: med={np.median(te_xgb):.2f}  std={te_xgb.std():.3f}  ratio={ratio:.2f}")

    # Blend XGBoost + LGBM meta (nb109)
    nb109_oof_p = DATA_PROCESSED / "oof_nb109_deep_meta_stack.npy"
    nb109_te_p  = DATA_PROCESSED / "te_nb109_deep_meta_stack.npy"
    if nb109_oof_p.exists():
        nb109_oof = np.load(nb109_oof_p); nb109_te = np.load(nb109_te_p)
        if nb109_oof.ndim == 2: nb109_oof = nb109_oof[:, 0]
        if nb109_te.ndim == 2:  nb109_te  = nb109_te[:, 0]
        blend_oof = 0.5 * oof_xgb + 0.5 * nb109_oof
        blend_te  = 0.5 * te_xgb + 0.5 * nb109_te
        full_metrics(y_tr, blend_oof, "XGB+LGBM meta blend (50/50)")
        b_ratio = blend_te.std() / blend_oof.std()
        print(f"  Blend test: std={blend_te.std():.3f}  ratio={b_ratio:.2f}")

    np.save(DATA_PROCESSED / "oof_nb136_xgb_meta.npy", oof_xgb)
    np.save(DATA_PROCESSED / "te_nb136_xgb_meta.npy",  te_xgb)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_xgb})
    sub.to_csv(SUBMISSIONS / "136_xgboost_meta.csv", index=False)
    print(f"\nSaved: submissions/136_xgboost_meta.csv")
    print(f"OOF RAE: {rae(y_tr, oof_xgb):.4f}")


if __name__ == "__main__":
    main()
