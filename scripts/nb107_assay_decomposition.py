"""nb107 — Mechanistic Assay Decomposition.

The cell-based PXR reporter assay pEC50 is a composite of:
  binding affinity, intrinsic efficacy (Emax), cell permeability,
  and selectivity vs counter-assay.

This script decomposes pEC50 by:
  1. Training a selectivity predictor: pEC50 - pEC50_null (PXR vs artifacts)
  2. Using predicted selectivity as an additional input feature to main model
  3. Training a "structural binding component" model on pEC50 + Emax info
  4. Blending: pEC50_pred = f(structural, Emax_pred, selectivity_pred, delta_pred)

Key insight: the pEC50_null counter-assay tells us how much of the signal
is PXR-specific. High primary + low null = true PXR activation.
High primary + high null = likely artifact. This is the gating signal
that separates PXR mechanistic signal from assay noise.
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
from pxr.chem import bemis_murcko, morgan_fp_batch, compute_physchem
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
LGBM_PARAMS = dict(
    n_estimators=1000, num_leaves=64, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    r2    = 1 - np.sum((yt-yp)**2) / np.sum((yt-yt.mean())**2) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp); sp, _ = stats.spearmanr(yt, yp)
    if label:
        print(f"  [{label}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  R2={r2:.4f}  "
              f"r={pr:.4f}  rho={sp:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, R2=r2, Pearson=pr, Spearman=sp)


def main():
    print("=== nb107: Mechanistic Assay Decomposition ===")

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ── Extract assay component labels ───────────────────────────────────────
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log  = np.log10(np.clip(emax_raw, 0.05, 10.0))

    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)

    # Selectivity = pEC50_primary - pEC50_null (PXR-specific signal)
    # Impute missing null with median of observed
    null_median  = np.nanmedian(pec50_null)
    null_imputed = np.where(np.isnan(pec50_null), null_median, pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    print(f"Emax: mean={np.nanmean(emax_raw):.3f}  missing={np.isnan(emax_raw).sum()}")
    print(f"Selectivity: mean={selectivity.mean():.3f}  std={selectivity.std():.3f}")
    print(f"Has counter-assay: {has_null.sum():.0f} / {len(has_null)}")

    # ── Compute structural features ──────────────────────────────────────────
    print("\nComputing features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # ── Stage 1: OOF predictions for Emax, selectivity, pec50_null ──────────
    print("\n=== Stage 1: Auxiliary OOF predictors ===")
    oof_emax     = np.full(len(y_tr), np.nan)
    oof_null     = np.full(len(y_tr), np.nan)
    oof_sel      = np.full(len(y_tr), np.nan)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Emax predictor
        m_emax = lgb.train(LGBM_AUX,
                            lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                            callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_emax.predict(X_tr[va_idx])

        # pEC50_null predictor (imputed target)
        m_null = lgb.train(LGBM_AUX,
                            lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                            callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_null.predict(X_tr[va_idx])

        # Selectivity predictor
        m_sel = lgb.train(LGBM_AUX,
                           lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                           callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx] = m_sel.predict(X_tr[va_idx])

    # Print how well we predict auxiliary targets
    valid_null = has_null == 1
    if valid_null.sum() > 100:
        pr_null, _ = stats.pearsonr(pec50_null[valid_null], oof_null[valid_null])
        print(f"  Null pEC50 predictor: r={pr_null:.3f} on {valid_null.sum()} compounds with real null")
    pr_sel, _  = stats.spearmanr(selectivity, oof_sel)
    pr_emax, _ = stats.spearmanr(emax_raw, oof_emax)
    print(f"  Selectivity predictor: Spearman rho={pr_sel:.3f}")
    print(f"  Emax predictor: Spearman rho={pr_emax:.3f}")

    # ── Stage 2: Load Delta-ML OOF if available (best base model) ───────────
    print("\n=== Stage 2: Augmented features with assay components ===")
    delta_oof_path = DATA_PROCESSED / "oof_nb101_delta_base.npy"
    delta_te_path  = DATA_PROCESSED / "te_nb101_delta_base.npy"
    sc_oof_path    = DATA_PROCESSED / "oof_nb99_sc_bio_fp.npy"
    sc_te_path     = DATA_PROCESSED / "te_nb99_sc_bio_fp.npy"

    extra_oof_cols = []
    extra_te_cols  = []

    if delta_oof_path.exists():
        delta_oof = np.load(delta_oof_path)
        delta_te  = np.load(delta_te_path)
        extra_oof_cols.append(delta_oof[:, None])
        extra_te_cols.append(delta_te[:, None])
        print(f"  Delta-ML OOF loaded (RAE={rae(y_tr, delta_oof):.4f})")

    if sc_oof_path.exists():
        sc_oof = np.load(sc_oof_path)
        sc_te  = np.load(sc_te_path)
        extra_oof_cols.append(sc_oof[:, None])
        extra_te_cols.append(sc_te[:, None])
        print(f"  SC Bio-FP OOF loaded (RAE={rae(y_tr, sc_oof):.4f})")

    # Build augmented feature matrix:
    # [structural_features | OOF_Emax | OOF_null | OOF_selectivity | has_null | delta_oof? | sc_oof?]
    assay_oof = np.column_stack([
        oof_emax,          # predicted Emax
        oof_null,          # predicted pEC50_null
        oof_sel,           # predicted selectivity
        has_null,          # whether we have real counter-assay data (indicator)
        np.log1p(np.clip(oof_emax, 0, None)),  # log Emax
    ] + ([c[:, 0] for c in extra_oof_cols] if extra_oof_cols else []))

    # For test: predict assay components first
    print("\nPredicting test assay components...")
    m_emax_f  = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log),
                           callbacks=[lgb.log_evaluation(-1)])
    m_null_f  = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=null_imputed),
                           callbacks=[lgb.log_evaluation(-1)])
    m_sel_f   = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity),
                           callbacks=[lgb.log_evaluation(-1)])

    te_emax = 10.0 ** m_emax_f.predict(X_te)
    te_null = m_null_f.predict(X_te)
    te_sel  = m_sel_f.predict(X_te)

    assay_te = np.column_stack([
        te_emax, te_null, te_sel,
        np.zeros(len(X_te)),  # no real counter assay for test
        np.log1p(np.clip(te_emax, 0, None)),
    ] + ([c[:, 0] for c in extra_te_cols] if extra_te_cols else []))

    # Augmented feature matrix = structural + assay components
    X_tr_aug = np.hstack([X_tr, assay_oof])
    X_te_aug  = np.hstack([X_te, assay_te])
    print(f"Augmented feature shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    # ── Stage 3: Scaffold CV with augmented features ─────────────────────────
    print("\n=== Stage 3: Scaffold CV with assay-augmented features ===")
    oof_aug = np.full(len(y_tr), np.nan)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Need to recompute assay OOF for val using only fold-train data
        # (to avoid leakage through the delta/SC OOF columns)
        m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                            callbacks=[lgb.log_evaluation(-1)])
        m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                            callbacks=[lgb.log_evaluation(-1)])
        m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                            callbacks=[lgb.log_evaluation(-1)])

        em_va = 10.0 ** m_em_f.predict(X_tr[va_idx])
        nl_va = m_nl_f.predict(X_tr[va_idx])
        sl_va = m_sl_f.predict(X_tr[va_idx])

        # For fold-level augmented features, replace assay OOF with fold predictions
        X_tr_f_aug = X_tr_aug[tr_idx].copy()
        X_va_aug   = np.hstack([
            X_tr[va_idx],
            np.column_stack([
                em_va, nl_va, sl_va,
                has_null[va_idx],
                np.log1p(np.clip(em_va, 0, None)),
            ] + ([assay_oof[va_idx, 5 + k:5 + k + 1]
                  for k in range(len(extra_oof_cols))] if extra_oof_cols else []))
        ])

        m = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_tr_f_aug, label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_va_aug, label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        oof_aug[va_idx] = m.predict(X_va_aug)
        fold_rae = rae(y_tr[va_idx], oof_aug[va_idx])
        print(f"  fold {fold+1}  RAE={fold_rae:.4f}", flush=True)

    full_metrics(y_tr, oof_aug, "nb107_assay_decomposition")

    # ── Stage 4: Selectivity gating on high-confidence compounds ─────────────
    # Compounds predicted to have selectivity >= 2.0 get boosted predictions
    # Compounds predicted to have selectivity < 0.0 get shrunk toward mean
    print("\n=== Stage 4: Selectivity-informed adjustment ===")
    oof_gated = oof_aug.copy()
    high_sel = oof_sel >= 2.0
    low_sel  = oof_sel < 0.0
    oof_gated[low_sel] = 0.8 * oof_aug[low_sel] + 0.2 * np.nanmean(y_tr)
    print(f"  High selectivity (>=2.0): {high_sel.sum()} compounds")
    print(f"  Low selectivity (<0.0): {low_sel.sum()} compounds (shrunk)")
    full_metrics(y_tr, oof_gated, "nb107_with_sel_gating")

    # ── Final model ───────────────────────────────────────────────────────────
    print("\nTraining final model...")
    m_final = lgb.train(LGBM_PARAMS, lgb.Dataset(X_tr_aug, label=y_tr),
                         callbacks=[lgb.log_evaluation(-1)])
    te_preds = m_final.predict(X_te_aug)

    # Apply selectivity gating on test
    te_low_sel = te_sel < 0.0
    te_preds[te_low_sel] = 0.8 * te_preds[te_low_sel] + 0.2 * np.nanmean(y_tr)
    te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)

    print(f"  Test low-selectivity: {te_low_sel.sum()} compounds gated")
    print(f"  Test: min={te_preds.min():.2f}  med={np.median(te_preds):.2f}  "
          f"max={te_preds.max():.2f}")

    np.save(DATA_PROCESSED / "oof_nb107_assay_decomp.npy", oof_aug)
    np.save(DATA_PROCESSED / "te_nb107_assay_decomp.npy", te_preds)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    out = SUBMISSIONS / "107_assay_decomposition.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
