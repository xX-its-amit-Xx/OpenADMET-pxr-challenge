"""nb116 — Quantile Regression + Prediction Interval Calibration.

Key insight from error analysis: extreme compounds (pEC50 < 2.5 or > 6.0)
are dramatically mispredicted. Standard LGBM (MSE) gives wide prediction
intervals for these outliers. Quantile regression can:

1. Predict MEDIAN (more robust than mean for heavy-tailed distributions)
2. Predict 10th/90th percentiles → calibrated prediction intervals
3. For uncertain test compounds (wide interval): shrink toward median
4. For certain test compounds (narrow interval): trust the prediction

Strategy:
  - Train LGBM with quantile objective at alpha=0.5 (median), 0.1, 0.9
  - Augment with assay decomp meta-features (Emax/null/sel + OOF)
  - Final prediction: weighted average of median ± interval-based correction
  - Compounds with narrow interval (high confidence) → keep median prediction
  - Compounds with wide interval (low confidence) → use conservative estimate

This is orthogonal to all existing models (different objective function).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5

LGBM_QUANTILE_BASE = dict(
    objective="quantile",
    n_estimators=2000, num_leaves=64, learning_rate=0.03,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=600, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def main():
    print("=== nb116: Quantile Regression + Prediction Intervals ===\n")

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Assay features
    emax_col   = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw   = raw_train[emax_col].values.astype(np.float64)
    emax_log   = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_median  = np.nanmedian(pec50_null)
    null_imputed = np.where(np.isnan(pec50_null), null_median, pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    # Structural features
    print("Computing structural features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # Load meta-OOF (non-collapsed)
    meta_oofs, meta_tes = [], []
    for stem in ["nb107_assay_decomp", "nb109_deep_meta_stack",
                 "nb101_delta_base", "nb99_sc_bio_fp", "grand_v6b", "lgbm_tuned"]:
        of = DATA_PROCESSED / f"oof_{stem}.npy"
        tf = DATA_PROCESSED / f"te_{stem}.npy"
        if of.exists() and tf.exists():
            o = np.load(of); t = np.load(tf)
            if o.ndim == 2: o = o[:, 0]
            if t.ndim == 2: t = t[:, 0]
            if len(o) == n_tr:
                o = np.where(np.isfinite(o), o, np.nanmean(o))
                t = np.where(np.isfinite(t), t, np.nanmean(t))
                if t.std() / o.std() >= 0.58:
                    meta_oofs.append(o); meta_tes.append(t)
                    print(f"  Loaded meta-OOF: {stem} (RAE={rae(y_tr, o):.4f})")

    # Stage 1: OOF auxiliary predictors
    print("\nStage 1: auxiliary OOF predictors...")
    oof_emax = np.full(n_tr, np.nan)
    oof_null = np.full(n_tr, np.nan)
    oof_sel  = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_tr[va_idx])
        m_nl = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_tr[va_idx])
        m_sl = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx] = m_sl.predict(X_tr[va_idx])

    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    assay_oof_full = np.column_stack([
        oof_emax, oof_null, oof_sel, has_null,
        np.log1p(np.clip(oof_emax, 0, None)),
    ] + meta_oofs)
    assay_te = np.column_stack([
        te_emax, te_null, te_sel, np.zeros(len(X_te)),
        np.log1p(np.clip(te_emax, 0, None)),
    ] + meta_tes)

    X_tr_aug = np.hstack([X_tr, assay_oof_full])
    X_te_aug = np.hstack([X_te, assay_te])
    print(f"Augmented shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    # Stage 2: Quantile regression CV
    print("\n=== Stage 2: Quantile Regression CV (alpha=0.1, 0.5, 0.9) ===")
    quantiles = [0.1, 0.5, 0.9]
    oof_quantiles = {q: np.full(n_tr, np.nan) for q in quantiles}

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Recompute fold-level auxiliary features
        m_em2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
        m_nl2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
        m_sl2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])

        em_va = 10.0 ** m_em2.predict(X_tr[va_idx])
        nl_va = m_nl2.predict(X_tr[va_idx])
        sl_va = m_sl2.predict(X_tr[va_idx])

        va_assay = np.column_stack([
            em_va, nl_va, sl_va, has_null[va_idx],
            np.log1p(np.clip(em_va, 0, None)),
        ] + [o[va_idx] for o in meta_oofs])

        X_va_fold = np.hstack([X_tr[va_idx], va_assay])
        X_tr_fold = X_tr_aug[tr_idx]

        for q in quantiles:
            params_q = dict(LGBM_QUANTILE_BASE, alpha=q)
            m_q = lgb.train(
                params_q,
                lgb.Dataset(X_tr_fold, label=y_tr[tr_idx]),
                valid_sets=[lgb.Dataset(X_va_fold, label=y_tr[va_idx])],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)]
            )
            oof_quantiles[q][va_idx] = m_q.predict(X_va_fold)

        fold_rae = rae(y_tr[va_idx], oof_quantiles[0.5][va_idx])
        interval_width = np.mean(oof_quantiles[0.9][va_idx] - oof_quantiles[0.1][va_idx])
        print(f"  fold {fold+1}  Q50_RAE={fold_rae:.4f}  "
              f"avg_interval_width={interval_width:.3f}", flush=True)

    # Evaluate median quantile
    print("\nMedian quantile performance:")
    full_metrics(y_tr, oof_quantiles[0.5], "Q50 (median) quantile regression")

    # Compare with standard LGBM (nb107)
    oof_107 = np.load(DATA_PROCESSED / "oof_nb107_assay_decomp.npy")
    full_metrics(y_tr, oof_107, "nb107 AssayDecomp (reference)")

    # Interval analysis: calibration of prediction intervals
    interval_widths = oof_quantiles[0.9] - oof_quantiles[0.1]
    within_interval = ((y_tr >= oof_quantiles[0.1]) & (y_tr <= oof_quantiles[0.9]))
    print(f"\nInterval calibration:")
    print(f"  Expected 80% coverage: actual={100*within_interval.mean():.1f}%")
    print(f"  Mean interval width: {interval_widths.mean():.3f}")
    print(f"  Median interval width: {np.median(interval_widths):.3f}")

    # Uncertainty-weighted prediction
    # For high-uncertainty compounds (wide interval): shrink toward training median
    # For low-uncertainty compounds (narrow interval): keep Q50 prediction
    q50 = oof_quantiles[0.5]
    train_median = np.nanmedian(y_tr)
    confidence = 1.0 / (interval_widths + 0.1)  # higher confidence = narrower interval
    confidence_norm = confidence / np.nanmedian(confidence)  # normalize around 1.0

    print("\nSweeping uncertainty weighting strength:")
    best_r_uw, best_gamma = rae(y_tr, q50), 0.0
    for gamma in np.arange(0.0, 1.01, 0.1):
        shrinkage = np.clip(confidence_norm ** gamma, 0.5, 2.0)
        shrinkage = shrinkage / shrinkage.mean()
        oof_uw = train_median + (q50 - train_median) * np.clip(shrinkage, 0.5, 1.5)
        r_uw = rae(y_tr, oof_uw)
        if r_uw < best_r_uw:
            best_r_uw, best_gamma = r_uw, gamma
        if gamma in [0.0, 0.2, 0.5, 1.0]:
            print(f"  gamma={gamma:.1f}  RAE={r_uw:.4f}")
    print(f"  Best: gamma={best_gamma:.1f}  RAE={best_r_uw:.4f}")

    # Final test predictions
    print("\nTraining final quantile models...")
    te_quantiles = {}
    for q in quantiles:
        params_q = dict(LGBM_QUANTILE_BASE, alpha=q)
        m_q_final = lgb.train(
            params_q,
            lgb.Dataset(X_tr_aug, label=y_tr),
            callbacks=[lgb.log_evaluation(-1)]
        )
        te_quantiles[q] = m_q_final.predict(X_te_aug)

    te_q50 = np.clip(te_quantiles[0.5], y_tr.min() - 0.5, y_tr.max() + 0.5)
    te_intervals = te_quantiles[0.9] - te_quantiles[0.1]
    print(f"\nTest Q50: min={te_q50.min():.2f}  med={np.median(te_q50):.2f}  "
          f"max={te_q50.max():.2f}  std={te_q50.std():.3f}")
    print(f"Test interval width: mean={te_intervals.mean():.3f}  "
          f"median={np.median(te_intervals):.3f}")

    # Save quantile predictions as OOF features for ensembling
    oof_q10 = np.clip(oof_quantiles[0.1], y_tr.min() - 1, y_tr.max() + 1)
    oof_q50 = np.clip(oof_quantiles[0.5], y_tr.min() - 1, y_tr.max() + 1)
    oof_q90 = np.clip(oof_quantiles[0.9], y_tr.min() - 1, y_tr.max() + 1)

    np.save(DATA_PROCESSED / "oof_nb116_quantile_q50.npy", oof_q50)
    np.save(DATA_PROCESSED / "oof_nb116_quantile_width.npy", oof_q90 - oof_q10)
    np.save(DATA_PROCESSED / "te_nb116_quantile_q50.npy", te_q50)
    np.save(DATA_PROCESSED / "te_nb116_quantile_width.npy", te_intervals)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_q50})
    sub.to_csv(SUBMISSIONS / "116_quantile_regression_q50.csv", index=False)
    print(f"\nSaved: submissions/116_quantile_regression_q50.csv")
    print(f"OOF RAE (Q50) = {rae(y_tr, oof_q50):.4f}")


if __name__ == "__main__":
    main()
