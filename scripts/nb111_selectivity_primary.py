"""nb111 — Selectivity-Primary Two-Stage Model.

Key mechanistic insight:
  pEC50 (cell-based reporter assay) = PXR binding selectivity + assay background

  Selectivity = pEC50_CRC - pEC50_null (counter-assay)
  This is the TRUE PXR-specific signal, uncorrupted by cellular artifacts.

  2,858/4,139 training compounds have real selectivity labels.
  Structural predictors should be BETTER at predicting selectivity than raw pEC50
  because selectivity removes compound-agnostic assay noise.

Strategy:
  Stage 1a: Predict pEC50_null (from structure + ALL compounds, including imputed)
  Stage 1b: Predict selectivity (from structure, using 2858 real + imputed rest)
  Stage 2:  pEC50 = predicted_selectivity + predicted_null
  Stage 3:  Blend with direct pEC50 prediction (nb107 assay decomp)
            to get best of mechanistic and empirical approaches

This is mechanistically distinct from nb107/108/109/110:
  - nb107: uses selectivity as a feature to predict pEC50 directly
  - nb111: uses selectivity as THE primary label, reconstructs pEC50
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
    n_estimators=1500, num_leaves=64, learning_rate=0.04,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_NULL = dict(
    n_estimators=800, num_leaves=48, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp); sp, _ = stats.spearmanr(yt, yp)
    if label:
        print(f"  [{label:50s}] RAE={rae_v:.4f}  r={pr:.4f}  rho={sp:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr, Spearman=sp)


def main():
    print("=== nb111: Selectivity-Primary Two-Stage Model ===\n")

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ── Assay components ─────────────────────────────────────────────────────
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))

    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    has_null    = np.isfinite(pec50_null)
    null_median = np.nanmedian(pec50_null)
    null_imputed = np.where(has_null, pec50_null, null_median)
    selectivity  = y_tr - null_imputed  # primary label for 2858 compounds
    has_null_f   = has_null.astype(np.float32)

    print(f"Compounds with real null: {has_null.sum()} / {n_tr}")
    print(f"Selectivity: mean={selectivity.mean():.3f}  std={selectivity.std():.3f}")
    print(f"True selectivity (real null only): mean={selectivity[has_null].mean():.3f}  "
          f"std={selectivity[has_null].std():.3f}")

    # ── Structural features ───────────────────────────────────────────────────
    print("\nComputing structural features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # ── Stage 1: OOF predictions for pEC50_null and selectivity ──────────────
    print("\n=== Stage 1: OOF null and selectivity predictors ===")
    oof_null_pred = np.full(n_tr, np.nan)
    oof_sel_pred  = np.full(n_tr, np.nan)
    oof_emax_pred = np.full(n_tr, np.nan)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Predict pEC50_null (all training)
        m_null = lgb.train(LGBM_NULL,
                           lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                           callbacks=[lgb.log_evaluation(-1)])
        oof_null_pred[va_idx] = m_null.predict(X_tr[va_idx])

        # Predict selectivity (all training, using imputed null for compounds without real data)
        m_sel = lgb.train(LGBM_NULL,
                          lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
        oof_sel_pred[va_idx] = m_sel.predict(X_tr[va_idx])

        # Predict Emax
        m_em = lgb.train(LGBM_NULL,
                         lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax_pred[va_idx] = 10.0 ** m_em.predict(X_tr[va_idx])

    # Evaluate Stage 1 predictions on real-null compounds
    rho_null, _ = stats.spearmanr(pec50_null[has_null], oof_null_pred[has_null])
    rho_sel, _  = stats.spearmanr(selectivity[has_null], oof_sel_pred[has_null])
    print(f"  Null predictor (real null only): rho={rho_null:.3f}")
    print(f"  Selectivity predictor (real null only): rho={rho_sel:.3f}")

    # Stage 1 reconstruction: pEC50 = sel + null
    oof_reconstructed = oof_sel_pred + oof_null_pred
    full_metrics(y_tr, oof_reconstructed, "Stage1: sel+null reconstruction")

    # ── Stage 2: Augmented features for direct pEC50 prediction ──────────────
    print("\n=== Stage 2: Augmented direct pEC50 prediction ===")

    # Load best OOF meta-features
    meta_oof_cols = []; meta_te_cols = []
    for stem in ["nb107_assay_decomp", "nb101_delta_base", "nb99_sc_bio_fp",
                 "grand_v6b", "lgbm_tuned", "nb103_seed_propagation"]:
        oof_f = DATA_PROCESSED / f"oof_{stem}.npy"
        te_f  = DATA_PROCESSED / f"te_{stem}.npy"
        if oof_f.exists() and te_f.exists():
            oof_a = np.load(oof_f)
            if oof_a.ndim == 2: oof_a = oof_a[:, 0]
            if len(oof_a) == n_tr:
                oof_filled = np.where(np.isfinite(oof_a), oof_a, np.nanmean(oof_a))
                te_a = np.load(te_f)
                if te_a.ndim == 2: te_a = te_a[:, 0]
                te_filled = np.where(np.isfinite(te_a), te_a, np.nanmean(te_a))
                meta_oof_cols.append(oof_filled)
                meta_te_cols.append(te_filled)

    # Full OOF augmented training matrix:
    # [structural | null_pred | sel_pred | emax_pred | sel+null | has_null | log_emax | meta_OOFs]
    assay_oof = np.column_stack([
        oof_null_pred,
        oof_sel_pred,
        oof_emax_pred,
        oof_reconstructed,        # the two-stage reconstruction
        has_null_f,
        np.log1p(np.clip(oof_emax_pred, 0, None)),
        oof_sel_pred * has_null_f, # true selectivity signal (0 for imputed)
    ])

    if meta_oof_cols:
        assay_oof = np.hstack([assay_oof, np.column_stack(meta_oof_cols)])

    # For test: run full-data Stage 1 predictors
    m_null_f = lgb.train(LGBM_NULL, lgb.Dataset(X_tr, label=null_imputed),
                         callbacks=[lgb.log_evaluation(-1)])
    m_sel_f  = lgb.train(LGBM_NULL, lgb.Dataset(X_tr, label=selectivity),
                         callbacks=[lgb.log_evaluation(-1)])
    m_em_f   = lgb.train(LGBM_NULL, lgb.Dataset(X_tr, label=emax_log),
                         callbacks=[lgb.log_evaluation(-1)])

    te_null_pred = m_null_f.predict(X_te)
    te_sel_pred  = m_sel_f.predict(X_te)
    te_emax_pred = 10.0 ** m_em_f.predict(X_te)
    te_recon     = te_sel_pred + te_null_pred

    assay_te = np.column_stack([
        te_null_pred, te_sel_pred, te_emax_pred, te_recon,
        np.zeros(len(X_te)),  # no real counter-assay for test
        np.log1p(np.clip(te_emax_pred, 0, None)),
        np.zeros(len(X_te)),  # no real selectivity for test
    ])
    if meta_te_cols:
        assay_te = np.hstack([assay_te, np.column_stack(meta_te_cols)])

    X_tr_aug = np.hstack([X_tr, assay_oof])
    X_te_aug = np.hstack([X_te, assay_te])
    print(f"Augmented shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    # ── Stage 3: Scaffold CV with augmented features ──────────────────────────
    print("\n=== Stage 3: Scaffold CV ===")
    oof_stage3 = np.full(n_tr, np.nan)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Recompute fold-level Stage 1 to avoid leakage
        m_nl2 = lgb.train(LGBM_NULL, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
        m_sl2 = lgb.train(LGBM_NULL, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
        m_em2 = lgb.train(LGBM_NULL, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])

        nl_va = m_nl2.predict(X_tr[va_idx])
        sl_va = m_sl2.predict(X_tr[va_idx])
        em_va = 10.0 ** m_em2.predict(X_tr[va_idx])
        rc_va = sl_va + nl_va

        # Build val augmented features (fold-level Stage 1)
        assay_va = np.column_stack([
            nl_va, sl_va, em_va, rc_va,
            has_null_f[va_idx],
            np.log1p(np.clip(em_va, 0, None)),
            sl_va * has_null_f[va_idx],
        ])
        if meta_oof_cols:
            assay_va = np.hstack([assay_va,
                np.column_stack([c[va_idx] for c in meta_oof_cols])])

        X_va_aug = np.hstack([X_tr[va_idx], assay_va])
        X_tr_aug_f = X_tr_aug[tr_idx]

        m = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_tr_aug_f, label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_va_aug, label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)]
        )
        oof_stage3[va_idx] = m.predict(X_va_aug)
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof_stage3[va_idx]):.4f}",
              flush=True)

    full_metrics(y_tr, oof_stage3, "nb111_selectivity_primary")

    # ── Compare with two-stage reconstruction ────────────────────────────────
    print("\n=== Blending Stage 3 with Stage 1 reconstruction ===")
    best_r, best_a = rae(y_tr, oof_stage3), 0.0
    for alpha in np.arange(0.0, 1.01, 0.1):
        blend = (1 - alpha) * oof_stage3 + alpha * oof_reconstructed
        r = rae(y_tr, blend)
        if r < best_r:
            best_r, best_a = r, alpha
        if alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
            print(f"  alpha(recon)={alpha:.1f}  RAE={r:.4f}")
    print(f"  Best: alpha(recon)={best_a:.1f}  RAE={best_r:.4f}")

    # ── Final model ───────────────────────────────────────────────────────────
    m_final = lgb.train(
        dict(LGBM_PARAMS, n_estimators=1000),
        lgb.Dataset(X_tr_aug, label=y_tr),
        callbacks=[lgb.log_evaluation(-1)]
    )
    te_preds = m_final.predict(X_te_aug)
    te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)

    print(f"\nTest: min={te_preds.min():.2f}  med={np.median(te_preds):.2f}  "
          f"max={te_preds.max():.2f}  std={te_preds.std():.3f}")

    # Blend Stage 1 reconstruction for test too
    te_preds_blended = np.clip(
        (1 - best_a) * te_preds + best_a * te_recon,
        y_tr.min() - 0.5, y_tr.max() + 0.5
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(DATA_PROCESSED / "oof_nb111_selectivity_primary.npy", oof_stage3)
    np.save(DATA_PROCESSED / "te_nb111_selectivity_primary.npy", te_preds)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    sub.to_csv(SUBMISSIONS / "111_selectivity_primary.csv", index=False)

    if best_a > 0:
        sub_b = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds_blended})
        sub_b.to_csv(SUBMISSIONS / "111_selectivity_recon_blend.csv", index=False)
        print(f"Saved: 111_selectivity_primary.csv (RAE={rae(y_tr, oof_stage3):.4f})")
        print(f"Saved: 111_selectivity_recon_blend.csv (OOF RAE={best_r:.4f})")
    else:
        print(f"Saved: 111_selectivity_primary.csv (RAE={rae(y_tr, oof_stage3):.4f})")


if __name__ == "__main__":
    main()
