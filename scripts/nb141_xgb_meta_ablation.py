"""nb141 — XGBoost Meta-Stack Ablation Study.

nb136 XGBoost meta-stack achieved OOF 0.3334 using all 105 OOF + 2265 structural + 5 assay.
This notebook ablates the feature contributions:

  Variant A: OOF only (no structural, no assay) — 105 features
  Variant B: OOF + assay (no structural) — 110 features
  Variant C: OOF + structural + assay (nb136 baseline) — 2375 features
  Variant D: Top-20 OOF (by individual RAE) + structural + assay
  Variant E: XGBoost with DART (dropout regularization)

The ablation reveals which feature groups drive the improvement over LGBM meta-stack.
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
    n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.6, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", device="cpu", verbosity=0, n_jobs=4, random_state=SEED
)
XGB_DART = dict(
    n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.6, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    booster="dart", rate_drop=0.1, skip_drop=0.5,
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


def xgb_cv(params, X_tr, y_tr, X_te, splits, label):
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
    print(f"  [{label:55s}] RAE={r:.4f}  ratio={ratio:.2f}", flush=True)
    return r, oof, te_pred, ratio


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
    print("=== nb141: XGBoost Meta-Stack Ablation ===\n")

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
    oof_emax = np.full(n_tr, np.nan); oof_null = np.full(n_tr, np.nan); oof_sel = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=emax_log[tr_idx]), callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_str[va_idx])
        m_nl = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=null_imputed[tr_idx]), callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_str[va_idx])
        m_sl = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=selectivity[tr_idx]), callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx] = m_sl.predict(X_str[va_idx])

    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_str_te); te_null = m_nl_f.predict(X_str_te); te_sel = m_sl_f.predict(X_str_te)

    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null, np.log1p(np.clip(oof_emax, 0, None))])
    assay_te  = np.column_stack([te_emax, te_null, te_sel, np.zeros(len(X_str_te)), np.log1p(np.clip(te_emax, 0, None))])

    # Variants
    print(f"\n=== Variant A: OOF only ({n_mod} features) ===")
    r_a, oof_a, te_a, rat_a = xgb_cv(XGB_BASE, oof_all, y_tr, te_all, splits, "A: OOF only")

    print(f"\n=== Variant B: OOF + assay (no structural) ===")
    X_b_tr = np.hstack([oof_all, assay_oof]); X_b_te = np.hstack([te_all, assay_te])
    r_b, oof_b, te_b, rat_b = xgb_cv(XGB_BASE, X_b_tr, y_tr, X_b_te, splits, "B: OOF+assay")

    print(f"\n=== Variant C: OOF + structural + assay (nb136 baseline) ===")
    X_c_tr = np.hstack([oof_all, X_str, assay_oof]); X_c_te = np.hstack([te_all, X_str_te, assay_te])
    r_c, oof_c, te_c, rat_c = xgb_cv(XGB_BASE, X_c_tr, y_tr, X_c_te, splits, "C: OOF+str+assay")

    print(f"\n=== Variant D: Top-20 OOF + structural + assay ===")
    top20_oof = oof_all[:, :20]; top20_te = te_all[:, :20]
    X_d_tr = np.hstack([top20_oof, X_str, assay_oof]); X_d_te = np.hstack([top20_te, X_str_te, assay_te])
    r_d, oof_d, te_d, rat_d = xgb_cv(XGB_BASE, X_d_tr, y_tr, X_d_te, splits, "D: Top-20 OOF+str+assay")

    print(f"\n=== Variant E: XGBoost DART (dropout) ===")
    r_e, oof_e, te_e, rat_e = xgb_cv(XGB_DART, X_c_tr, y_tr, X_c_te, splits, "E: DART OOF+str+assay")

    # Summary
    print(f"\n=== Summary ===")
    variants = [
        ("A: OOF only", r_a, rat_a), ("B: OOF+assay", r_b, rat_b),
        ("C: OOF+str+assay", r_c, rat_c), ("D: Top-20 OOF+str+assay", r_d, rat_d),
        ("E: DART", r_e, rat_e)
    ]
    for vname, vr, vratio in variants:
        passed = "PASS" if vratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {vname:35s}  RAE={vr:.4f}  ratio={vratio:.2f}  [{passed}]")

    valid = [(n, r, oof, te) for (n, r, rat), oof, te in
             zip(variants, [oof_a, oof_b, oof_c, oof_d, oof_e],
                 [te_a, te_b, te_c, te_d, te_e]) if rat >= COLLAPSE_THRESH]
    if not valid:
        valid = [("C: best fallback", r_c, oof_c, te_c)]
    best_name, best_r_v, best_oof_v, best_te_v = min(valid, key=lambda x: x[1])
    print(f"\nBest variant: {best_name}  OOF RAE={best_r_v:.4f}")

    # Also blend best variant with nb139 adaptive
    nb139_oof_p = DATA_PROCESSED / "oof_nb139_adaptive_blend.npy"
    nb139_te_p  = DATA_PROCESSED / "te_nb139_adaptive_blend.npy"
    if nb139_oof_p.exists():
        nb139_oof = np.load(nb139_oof_p).astype(np.float64)
        nb139_te  = np.load(nb139_te_p).astype(np.float64)
        if nb139_oof.ndim == 2: nb139_oof = nb139_oof[:, 0]
        for alpha in [0.3, 0.5, 0.7]:
            blend_oof = alpha * best_oof_v + (1-alpha) * nb139_oof
            blend_te  = alpha * best_te_v  + (1-alpha) * nb139_te
            print(f"  Blend (alpha={alpha:.1f}): OOF RAE={rae(y_tr, blend_oof):.4f}")

    best_te_final = np.clip(best_te_v, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb141_xgb_ablation.npy", best_oof_v)
    np.save(DATA_PROCESSED / "te_nb141_xgb_ablation.npy",  best_te_final)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_final})
    sub.to_csv(SUBMISSIONS / "141_xgb_meta_ablation.csv", index=False)
    print(f"\nSaved: submissions/141_xgb_meta_ablation.csv  OOF RAE={best_r_v:.4f}")


if __name__ == "__main__":
    main()
