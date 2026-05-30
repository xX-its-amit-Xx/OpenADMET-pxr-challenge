"""nb100 — Emax-Corrected Dual-Head Prediction.

The assay measures pEC50 at the EC50 point regardless of Emax.
Partial agonists (Emax < 1 vs positive control) have their pEC50 measured
at a lower absolute response level — systematically compressing apparent potency.

Strategy:
  1. Predict Emax.vs.pos.ctrl from structural features (Emax predictor)
  2. Apply pharmacological correction:
       pEC50_corrected = pEC50_raw + gamma * log10(Emax_vs_ctrl)
     where gamma is fitted by maximizing OOF RAE improvement
  3. Train on corrected targets, back-convert at test time

Also trains a selectivity predictor (pEC50 - pEC50_null) and uses
predicted selectivity to gate/shrink predictions for likely false positives.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats, optimize
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
LGBM_PARAMS = dict(
    n_estimators=1000, num_leaves=64, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_EMAX = dict(
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
    print("=== nb100: Emax-Corrected Dual-Head Prediction ===")
    # Load raw train CSV to get Emax columns
    raw_train = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Extract Emax column (Emax relative to positive control, dimensionless)
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    print(f"Emax stats: mean={np.nanmean(emax_raw):.3f}  "
          f"std={np.nanstd(emax_raw):.3f}  "
          f"missing={np.isnan(emax_raw).sum()}")

    # Clip Emax to reasonable range (super-agonists cap at 5x)
    emax_clipped = np.clip(emax_raw, 0.05, 5.0)

    # Counter-assay selectivity
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    selectivity = y_tr - np.where(np.isnan(pec50_null), y_tr.mean() - 2.0, pec50_null)
    print(f"Selectivity: mean={np.nanmean(selectivity):.2f}  "
          f"std={np.nanstd(selectivity):.2f}  "
          f">=1.5 for {(selectivity >= 1.5).mean()*100:.1f}%")

    print("Computing features...")
    X_all = impute(combined(tr["smiles"].tolist()))
    X_te  = impute(combined(te["smiles"].tolist()))

    # ── Step 1: Scaffold CV for OOF Emax predictions ───────────────────────────
    print("\n=== Emax predictor (OOF) ===")
    emax_target = np.log10(np.where(np.isnan(emax_clipped), 1.0, emax_clipped))
    emax_oof = np.zeros(len(y_tr), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(
            LGBM_EMAX,
            lgb.Dataset(X_all[tr_idx], label=emax_target[tr_idx]),
            callbacks=[lgb.log_evaluation(-1)]
        )
        emax_oof[va_idx] = m_em.predict(X_all[va_idx])
    emax_oof_linear = 10.0 ** np.clip(emax_oof, -1.0, 1.0)

    # ── Step 2: Find optimal gamma (Emax correction coefficient) ───────────────
    print("\n=== Fitting gamma (Emax correction) ===")
    log_emax_oof = np.log10(np.clip(emax_oof_linear, 0.1, 10.0))

    def neg_rae(gamma):
        corrected_target = y_tr - gamma * np.log10(np.clip(emax_clipped, 0.1, 10.0))
        # Use corrected target for OOF — run 1-fold quick sweep
        oof_corr = np.full(len(y_tr), np.nan)
        for tr_idx, va_idx in splits:
            corr_tr = corrected_target[tr_idx]
            m = lgb.train(
                {**LGBM_PARAMS, "n_estimators": 300},
                lgb.Dataset(X_all[tr_idx], label=corr_tr),
                callbacks=[lgb.log_evaluation(-1)]
            )
            raw_pred = m.predict(X_all[va_idx])
            # Back-convert: pEC50_pred = raw_pred + gamma * log10(Emax_oof)
            oof_corr[va_idx] = raw_pred + gamma * log_emax_oof[va_idx]
        valid = np.isfinite(oof_corr)
        return rae(y_tr[valid], oof_corr[valid])

    # Grid search gamma in [-0.5, 0.5]
    gammas = np.linspace(-0.5, 0.5, 21)
    rae_scores = []
    print("  Sweeping gamma:")
    for g in gammas:
        r = neg_rae(g)
        rae_scores.append(r)
        if abs(g) <= 0.15 or g in [-0.5, 0.5, 0.0]:
            print(f"    gamma={g:+.2f}  RAE={r:.4f}")

    best_gamma = gammas[np.argmin(rae_scores)]
    print(f"\n  Best gamma={best_gamma:+.3f}  RAE={min(rae_scores):.4f}")

    # ── Step 3: Full scaffold CV with best gamma ───────────────────────────────
    print(f"\n=== Full scaffold CV with gamma={best_gamma:+.3f} ===")
    corrected_target = y_tr - best_gamma * np.log10(np.clip(emax_clipped, 0.1, 10.0))
    oof_corr = np.full(len(y_tr), np.nan)
    oof_raw  = np.full(len(y_tr), np.nan)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Emax predictor for this fold
        m_em = lgb.train(
            LGBM_EMAX,
            lgb.Dataset(X_all[tr_idx], label=emax_target[tr_idx]),
            callbacks=[lgb.log_evaluation(-1)]
        )
        em_va = 10.0 ** np.clip(m_em.predict(X_all[va_idx]), -1.0, 1.0)

        # Main predictor on corrected target
        m = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_all[tr_idx], label=corrected_target[tr_idx]),
            valid_sets=[lgb.Dataset(X_all[va_idx], label=corrected_target[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        raw_pred = m.predict(X_all[va_idx])
        oof_corr[va_idx] = raw_pred + best_gamma * np.log10(np.clip(em_va, 0.1, 10.0))
        oof_raw[va_idx]  = raw_pred  # before back-conversion
        fold_rae = rae(y_tr[va_idx], oof_corr[va_idx])
        print(f"  fold {fold+1}  RAE={fold_rae:.4f}", flush=True)

    full_metrics(y_tr, oof_corr, "nb100_emax_corrected")

    # ── Step 4: Selectivity gating ────────────────────────────────────────────
    # Predict pEC50_null OOF
    print("\n=== Selectivity gating ===")
    # For compounds without null label, use mean
    null_target = np.where(np.isnan(pec50_null), np.nanmean(pec50_null), pec50_null)
    oof_null = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_null = lgb.train(
            {**LGBM_PARAMS, "n_estimators": 300},
            lgb.Dataset(X_all[tr_idx], label=null_target[tr_idx]),
            callbacks=[lgb.log_evaluation(-1)]
        )
        oof_null[va_idx] = m_null.predict(X_all[va_idx])

    oof_sel = oof_corr - oof_null
    # Shrink predictions where predicted selectivity < 0.5 (likely false positive)
    low_sel_mask = oof_sel < 0.5
    oof_gated = oof_corr.copy()
    oof_gated[low_sel_mask] = 0.7 * oof_corr[low_sel_mask] + 0.3 * np.nanmean(y_tr)
    full_metrics(y_tr, oof_gated, "nb100_emax_gated")
    print(f"  Low selectivity compounds: {low_sel_mask.sum()} / {len(y_tr)}")

    # ── Step 5: Final models ──────────────────────────────────────────────────
    print("\nTraining final models...")
    # Final Emax model
    m_em_final = lgb.train(LGBM_EMAX, lgb.Dataset(X_all, label=emax_target),
                           callbacks=[lgb.log_evaluation(-1)])
    em_te = 10.0 ** np.clip(m_em_final.predict(X_te), -1.0, 1.0)

    # Final null model
    m_null_final = lgb.train({**LGBM_PARAMS, "n_estimators": 300},
                              lgb.Dataset(X_all, label=null_target),
                              callbacks=[lgb.log_evaluation(-1)])
    null_te = m_null_final.predict(X_te)

    # Final main model
    m_final = lgb.train(LGBM_PARAMS, lgb.Dataset(X_all, label=corrected_target),
                        callbacks=[lgb.log_evaluation(-1)])
    raw_te = m_final.predict(X_te)
    te_preds = raw_te + best_gamma * np.log10(np.clip(em_te, 0.1, 10.0))

    # Selectivity gating on test
    sel_te = te_preds - null_te
    low_sel_te = sel_te < 0.5
    te_preds[low_sel_te] = 0.7 * te_preds[low_sel_te] + 0.3 * np.nanmean(y_tr)
    te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"  Test low-selectivity: {low_sel_te.sum()} compounds gated")

    np.save(DATA_PROCESSED / "oof_nb100_emax_corrected.npy", oof_gated)
    np.save(DATA_PROCESSED / "te_nb100_emax_corrected.npy", te_preds)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    out = SUBMISSIONS / "100_emax_correction.csv"
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(f"Test  min={te_preds.min():.2f}  med={np.median(te_preds):.2f}  max={te_preds.max():.2f}")
    print(f"Best gamma={best_gamma:+.3f}")


if __name__ == "__main__":
    main()
