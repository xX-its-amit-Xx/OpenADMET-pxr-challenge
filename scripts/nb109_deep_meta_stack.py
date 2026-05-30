"""nb109 — Deep Meta-Stacking: all non-collapsed OOF models as Level-2 features.

Key insight from nb107/108 analysis:
  - grand_v8 (OOF 0.3088) and multi_template_delta (0.3266) have collapsed test
    predictions (test_std/oof_std < 0.5) — train-only feature leakage
  - nb107 assay decomp (0.3785) has reasonable test distribution (ratio 0.64)
  - 10+ diverse non-collapsed models exist with te_ files available

Strategy:
  Use ALL non-collapsed OOF models as meta-features in a Level-2 LGBM stacker,
  combined with structural features + assay decomp auxiliary features (Emax/null/sel).
  This is essentially nb107 v2 with 8 more diverse meta-feature columns.

Models used as meta-features (all with te_std/oof_std >= 0.60):
  - oof_nb107_assay_decomp (0.3785) - our best assay-decomp model
  - oof_nb103_seed_propagation (0.4196)
  - oof_nb99_sc_bio_fp (0.5249)
  - oof_grand_v6b (0.5281) - ensemble of ~30 models from session 1
  - oof_grand25 (0.5356)
  - oof_lgbm_tuned (0.5394)
  - oof_catboost (0.5594)
  - oof_multi_nr_transfer (0.5609) - multi-NR nuclear receptor transfer learning
  - oof_xgboost_dart (0.5635)
  - oof_chemprop_aux (0.5665) - GNN multi-task auxiliary
  - oof_nb101_delta_base (0.4164) - delta-ML (included despite slight collapse)
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
LGBM_PARAMS = dict(
    n_estimators=2000, num_leaves=64, learning_rate=0.03,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    r2    = 1 - np.sum((yt-yp)**2) / np.sum((yt-yt.mean())**2) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp); sp, _ = stats.spearmanr(yt, yp)
    if label:
        print(f"  [{label:50s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, R2=r2, Pearson=pr, Spearman=sp)


def load_meta(stem, n_tr):
    """Load OOF + test meta-feature arrays; return None if missing or wrong size."""
    # Try direct name first, then te_oof_ prefix
    for oof_prefix in ("oof_", ""):
        for te_prefix in ("te_", "te_oof_"):
            oof_f = DATA_PROCESSED / f"{oof_prefix}{stem}.npy"
            te_f  = DATA_PROCESSED / f"{te_prefix}{stem}.npy"
            if oof_f.exists() and te_f.exists():
                oof = np.load(oof_f)
                te  = np.load(te_f)
                if oof.ndim == 2: oof = oof[:, 0]
                if te.ndim == 2:  te  = te[:, 0]
                if len(oof) == n_tr:
                    return oof, te
    return None, None


def main():
    print("=== nb109: Deep Meta-Stacking ===\n")

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ── Assay aux features (same as nb107) ───────────────────────────────────
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

    # ── Structural features ───────────────────────────────────────────────────
    print("Computing structural features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # ── Load meta-feature candidates ─────────────────────────────────────────
    meta_candidates = [
        ("nb107_assay_decomp",      "AssayDecomp (our best)",      True),
        ("nb101_delta_base",         "Delta-ML",                    True),
        ("nb103_seed_propagation",   "Seed propagation",            True),
        ("nb99_sc_bio_fp",           "SC Bio-Fingerprint",          True),
        ("grand_v6b",                "Grand v6b ensemble",          False),
        ("grand25",                  "Grand25 ensemble",            False),
        ("lgbm_tuned",               "LGBM tuned",                  False),
        ("catboost",                 "CatBoost",                    False),
        ("multi_nr_transfer",        "Multi-NR transfer",           False),
        ("xgboost_dart",             "XGBoost DART",                False),
        ("chemprop_aux",             "Chemprop GNN aux",            False),
        ("nb97_pxr_features",        "PXR physics features",        True),
        ("grover_large",             "GROVER-large",                False),
    ]

    oof_metas, te_metas, meta_names = [], [], []
    print("\nLoading meta-features:")
    for stem, label, is_nb in meta_candidates:
        oof, te_p = load_meta(stem, n_tr)
        if oof is not None:
            valid = np.isfinite(oof); oof_f = oof.copy(); oof_f[~valid] = np.nanmean(oof)
            te_f_filled = te_p.copy(); te_f_filled[~np.isfinite(te_p)] = np.nanmean(te_p)
            r = rae(y_tr, oof_f)
            oof_std = np.nanstd(oof); te_std = np.nanstd(te_p)
            ratio = te_std / oof_std if oof_std > 0 else 0
            flag = "SKIP" if ratio < 0.50 else "OK"
            print(f"  {flag}  {r:.4f}  ratio={ratio:.2f}  {label}")
            if flag == "OK":
                oof_metas.append(oof_f)
                te_metas.append(te_f_filled)
                meta_names.append(label)
        else:
            print(f"  MISS                             {label}")

    M_tr = np.column_stack(oof_metas)   # (n_tr, n_meta)
    M_te = np.column_stack(te_metas)    # (n_te, n_meta)
    print(f"\nMeta-feature matrix: {M_tr.shape}")

    # ── Stage 1: OOF auxiliary predictors (Emax/null/sel) ────────────────────
    print("\n=== Stage 1: Auxiliary OOF predictors ===")
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

    pr_sel, _ = stats.spearmanr(selectivity, oof_sel)
    print(f"  Selectivity predictor rho={pr_sel:.3f}")

    # Predict assay components for test
    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log),
                       callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=null_imputed),
                       callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity),
                       callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    # ── Build augmented feature matrix ────────────────────────────────────────
    # OOF auxiliary features (using full-data predicted values, NOT fold-specific here)
    assay_oof_full = np.column_stack([
        oof_emax, oof_null, oof_sel, has_null,
        np.log1p(np.clip(oof_emax, 0, None)),
    ])
    assay_te = np.column_stack([
        te_emax, te_null, te_sel,
        np.zeros(len(X_te)),
        np.log1p(np.clip(te_emax, 0, None)),
    ])

    # Full augmented matrix: structural + assay aux + all meta OOF
    X_tr_full = np.hstack([X_tr, assay_oof_full, M_tr])
    X_te_full  = np.hstack([X_te, assay_te, M_te])
    print(f"\nFull augmented shape: train={X_tr_full.shape}  test={X_te_full.shape}")

    # ── Stage 2: Scaffold CV with full augmented features ────────────────────
    print("\n=== Stage 2: Scaffold CV with deep meta-stacking ===")
    oof_deep = np.full(n_tr, np.nan)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Recompute auxiliary fold-level features to avoid leakage
        m_em_f2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                             callbacks=[lgb.log_evaluation(-1)])
        m_nl_f2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                             callbacks=[lgb.log_evaluation(-1)])
        m_sl_f2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                             callbacks=[lgb.log_evaluation(-1)])

        em_va = 10.0 ** m_em_f2.predict(X_tr[va_idx])
        nl_va = m_nl_f2.predict(X_tr[va_idx])
        sl_va = m_sl_f2.predict(X_tr[va_idx])

        # Build fold-level val augmented features
        assay_va = np.column_stack([
            em_va, nl_va, sl_va, has_null[va_idx],
            np.log1p(np.clip(em_va, 0, None)),
        ])
        X_va_aug = np.hstack([X_tr[va_idx], assay_va, M_tr[va_idx]])
        X_tr_aug = X_tr_full[tr_idx]

        m = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_tr_aug, label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_va_aug, label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
        )
        oof_deep[va_idx] = m.predict(X_va_aug)
        fold_rae = rae(y_tr[va_idx], oof_deep[va_idx])
        print(f"  fold {fold+1}  RAE={fold_rae:.4f}", flush=True)

    full_metrics(y_tr, oof_deep, "nb109_deep_meta_stack")

    # Compare vs base models
    print("\n=== Comparison ===")
    for stem, label, _ in meta_candidates[:4]:
        oof_f_path = DATA_PROCESSED / f"oof_{stem}.npy"
        if oof_f_path.exists():
            oof_b = np.load(oof_f_path)
            if oof_b.ndim == 2: oof_b = oof_b[:, 0]
            if len(oof_b) == n_tr:
                full_metrics(y_tr, oof_b, label[:50])

    # ── Final model ───────────────────────────────────────────────────────────
    print("\nTraining final model on full training data...")
    m_final = lgb.train(
        dict(LGBM_PARAMS, n_estimators=1200),
        lgb.Dataset(X_tr_full, label=y_tr),
        callbacks=[lgb.log_evaluation(-1)]
    )
    te_preds = m_final.predict(X_te_full)
    te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)

    print(f"\nTest: min={te_preds.min():.2f}  med={np.median(te_preds):.2f}  "
          f"max={te_preds.max():.2f}  std={te_preds.std():.3f}")

    # Feature importance: which meta-models matter most?
    fi = m_final.feature_importance(importance_type="gain")
    n_struct = X_tr.shape[1] + 5  # structural + assay aux
    meta_fi = fi[n_struct:]
    print("\nMeta-feature importances (top):")
    for name, imp in sorted(zip(meta_names, meta_fi), key=lambda x: -x[1])[:8]:
        print(f"  {name:50s}  gain={imp:.1f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(DATA_PROCESSED / "oof_nb109_deep_meta_stack.npy", oof_deep)
    np.save(DATA_PROCESSED / "te_nb109_deep_meta_stack.npy", te_preds)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    out = SUBMISSIONS / "109_deep_meta_stack.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"OOF RAE = {rae(y_tr, oof_deep):.4f}")


if __name__ == "__main__":
    main()
