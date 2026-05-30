"""nb150 — Residual Ensemble (Boosting over Meta-Stack).

Instead of blending raw OOF predictions, train a second meta-stack on the
RESIDUALS of the best meta-stack (nb143 or nb141-B).

Rationale:
  - nb143 captures most of the signal from 113 OOF predictions
  - Residuals may be predictable from counter_delta, structural features, or
    specific base models that nb143 under-weights
  - XGBoost on residuals = gradient boosting over the meta-stack

Stage 1: get nb143 OOF residuals (y_tr - oof_nb143)
Stage 2: train a lightweight model on residuals using:
  - counter_delta OOF (orthogonal signal)
  - structural features (Morgan + RDKit)
  - assay features
  - individual base model residuals from worst-performing nb143 folds
Stage 3: final = nb143_oof + residual_correction
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
XGB_RESID = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.6, min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0,
    tree_method="hist", device="cpu", verbosity=0, n_jobs=4, random_state=SEED
)
LGBM_RESID = dict(
    n_estimators=200, num_leaves=16, learning_rate=0.05, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.6, reg_alpha=0.5, verbose=-1, n_jobs=4, random_state=SEED
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


def load_specific(stem, n_tr):
    """Load a specific model's OOF and test predictions."""
    oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
    if not oof_p.exists(): return None, None
    for te_pref in ("te_", "te_oof_"):
        te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
        if te_p.exists(): break
    else: return None, None
    oof = np.load(oof_p).astype(np.float64)
    te  = np.load(te_p).astype(np.float64)
    if oof.ndim == 2: oof = oof[:, 0]
    if te.ndim == 2:  te  = te[:, 0]
    if len(oof) != n_tr: return None, None
    return oof, te


def main():
    print("=== nb150: Residual Ensemble ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Priority: use nb143 if available, else nb141_xgb_ablation
    base_stem, base_label = None, None
    for candidate in ["nb143_oofassay_meta", "nb141_xgb_ablation"]:
        oof_b, te_b = load_specific(candidate, n_tr)
        if oof_b is not None:
            base_stem, base_label = candidate, candidate
            break

    if base_stem is None:
        print("No base meta-stack found. Run nb143 or nb141 first.")
        return

    print(f"Base meta-stack: {base_label}  RAE={rae(y_tr, oof_b):.4f}")
    residuals = y_tr - oof_b

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

    # Counter-delta OOF (key orthogonal signal)
    oof_cdelta, te_cdelta = load_specific("nb113_counter_delta", n_tr)

    print("Loading all OOF predictions (for residual features)...")
    models = load_model_oofs(n_tr, y_tr)
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

    print(f"Residual stats: mean={residuals.mean():.3f}  std={residuals.std():.3f}  "
          f"RAE(residuals vs. 0)={np.mean(np.abs(residuals))/np.mean(np.abs(y_tr-y_tr.mean())):.4f}")

    # Build residual feature matrix
    resid_feats_tr = [assay_oof, X_str]
    resid_feats_te = [assay_te, X_str_te]
    if oof_cdelta is not None:
        resid_feats_tr.append(oof_cdelta.reshape(-1, 1))
        resid_feats_te.append(te_cdelta.reshape(-1, 1))
        print(f"Counter-delta included (RAE={rae(y_tr, oof_cdelta):.4f})")
    # Add base model OOF as features for residuals
    resid_feats_tr.append(oof_b.reshape(-1, 1))
    resid_feats_te.append(te_b.reshape(-1, 1))

    X_resid_tr = np.hstack(resid_feats_tr)
    X_resid_te = np.hstack(resid_feats_te)
    print(f"Residual meta-feature matrix: {X_resid_tr.shape}")

    print("\nTraining residual corrector (XGBoost)...")
    resid_oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = xgb.XGBRegressor(**XGB_RESID)
        m.fit(X_resid_tr[tr_idx], residuals[tr_idx], verbose=False)
        resid_oof[va_idx] = m.predict(X_resid_tr[va_idx])
        print(f"    fold {fold+1}  resid_RAE={rae(y_tr[va_idx], oof_b[va_idx]+resid_oof[va_idx]):.4f}", flush=True)
    m_xgb_full = xgb.XGBRegressor(**XGB_RESID)
    m_xgb_full.fit(X_resid_tr, residuals, verbose=False)
    resid_te = m_xgb_full.predict(X_resid_te)

    oof_corrected_xgb = oof_b + resid_oof
    te_corrected_xgb  = te_b  + resid_te
    r_xgb = rae(y_tr, oof_corrected_xgb)
    print(f"  XGB corrected: RAE={r_xgb:.4f}  (base={rae(y_tr, oof_b):.4f})")

    print("\nTraining residual corrector (LGBM)...")
    resid_oof_lgb = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM_RESID, lgb.Dataset(X_resid_tr[tr_idx], label=residuals[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        resid_oof_lgb[va_idx] = m.predict(X_resid_tr[va_idx])
        print(f"    fold {fold+1}  resid_RAE={rae(y_tr[va_idx], oof_b[va_idx]+resid_oof_lgb[va_idx]):.4f}", flush=True)
    m_lgb_full = lgb.train(LGBM_RESID, lgb.Dataset(X_resid_tr, label=residuals), callbacks=[lgb.log_evaluation(-1)])
    resid_te_lgb = m_lgb_full.predict(X_resid_te)

    oof_corrected_lgb = oof_b + resid_oof_lgb
    te_corrected_lgb  = te_b  + resid_te_lgb
    r_lgb = rae(y_tr, oof_corrected_lgb)
    print(f"  LGB corrected: RAE={r_lgb:.4f}  (base={rae(y_tr, oof_b):.4f})")

    # Blend sweep (correction weight)
    print("\nCorrection weight sweep:")
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        b_xgb = oof_b + alpha * resid_oof
        b_lgb = oof_b + alpha * resid_oof_lgb
        print(f"  alpha={alpha:.1f}  XGB:{rae(y_tr, b_xgb):.4f}  LGB:{rae(y_tr, b_lgb):.4f}")

    # Best
    results = {
        "base": (rae(y_tr, oof_b), oof_b, te_b),
        "xgb_corrected": (r_xgb, oof_corrected_xgb, te_corrected_xgb),
        "lgb_corrected": (r_lgb, oof_corrected_lgb, te_corrected_lgb),
    }
    best_label = min(results, key=lambda k: results[k][0])
    best_r, best_oof, best_te = results[best_label]
    print(f"\nBest: {best_label}  RAE={best_r:.4f}")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb150_residual_ensemble.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb150_residual_ensemble.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "150_residual_ensemble.csv", index=False)
    print(f"\nSaved: submissions/150_residual_ensemble.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
