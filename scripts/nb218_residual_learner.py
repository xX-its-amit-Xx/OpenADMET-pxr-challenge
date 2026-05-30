"""nb218 -- Residual learner on nb212 (current best ensemble).

Hypothesis: nb212 is the SLSQP optimum given the existing pool. Its residuals
(y_true - oof_nb212) are what the pool can't capture. If features correlate with
those residuals, a residual model can correct nb212's errors.

Approach:
1. Load nb212 OOF + test predictions
2. Compute fold-aware residuals on training set
3. Train LGBM (also XGBoost for diversity) to predict residuals from features
4. Output corrected predictions: nb212 + alpha * residual_pred for several alpha
5. Output the residual model OOF/test (for downstream ensembling)

The residual model can be added to the ensemble pool in nb219.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.featurize import combined as feat_combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
SEED = 42
COLLAPSE_THRESH = 0.58
PREV_BEST = 0.296172

t0 = time.time()


def main():
    print("=== nb218: Residual learner on nb212 ===\n", flush=True)

    tr_df = load_train()
    te_df = load_test()
    y_pec = tr_df["pec50"].values.astype(np.float64)
    n_tr = len(tr_df)

    # Load nb212
    oof_nb212 = np.load(DATA_PROCESSED / "oof_nb212_nb211_blend.npy").flatten().astype(np.float64)
    te_nb212  = np.load(DATA_PROCESSED / "te_nb212_nb211_blend.npy").flatten().astype(np.float64)

    base_rae = rae(y_pec, oof_nb212)
    base_ratio = te_nb212.std() / oof_nb212.std()
    print(f"nb212 baseline: RAE={base_rae:.6f}  ratio={base_ratio:.4f}\n", flush=True)

    # Compute residuals
    residuals = y_pec - oof_nb212
    print(f"Residuals: mean={residuals.mean():.4f}  std={residuals.std():.4f}", flush=True)
    print(f"           min={residuals.min():.3f}  max={residuals.max():.3f}", flush=True)

    scaffolds = tr_df["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Features
    print("\nComputing combined features...", flush=True)
    X_tr = impute(feat_combined(tr_df["smiles"].tolist())).astype(np.float32)
    X_te = impute(feat_combined(te_df["smiles"].tolist())).astype(np.float32)

    # Optional: include nb215 NNE features if they're saved (from a previous run)
    try:
        oof_nb215 = np.load(DATA_PROCESSED / "oof_nb215_chemist_features.npy").flatten()
        te_nb215  = np.load(DATA_PROCESSED / "te_nb215_chemist_features.npy").flatten()
        X_tr = np.hstack([X_tr, oof_nb215.reshape(-1, 1)])
        X_te = np.hstack([X_te, te_nb215.reshape(-1, 1)])
        print(f"  added nb215 OOF as feature ({oof_nb215.shape})", flush=True)
    except FileNotFoundError:
        print("  (nb215 features not yet available)", flush=True)

    print(f"  feature shape: {X_tr.shape} ({time.time()-t0:.0f}s)\n", flush=True)

    # ----- Residual LGBM (MAE) -----
    print("Training residual LGBM (MAE)...", flush=True)
    res_oof_lgbm = np.full(n_tr, np.nan)
    res_te_lgbm = np.zeros(len(te_df))
    for fi, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(
            n_estimators=1500, num_leaves=32, learning_rate=0.02,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=0.5, objective="regression_l1",
            random_state=SEED, verbose=-1,
        )
        m.fit(
            X_tr[tr_idx], residuals[tr_idx],
            eval_set=[(X_tr[va_idx], residuals[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        res_oof_lgbm[va_idx] = m.predict(X_tr[va_idx])
        res_te_lgbm += m.predict(X_te) / N_FOLDS
        print(f"  fold {fi+1}: best_iter={m.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    res_lgbm_mae = np.mean(np.abs(residuals - res_oof_lgbm))
    res_lgbm_explained = 1 - np.var(residuals - res_oof_lgbm) / np.var(residuals)
    print(f"  residual MAE: {res_lgbm_mae:.4f}  explained-var-ratio: {res_lgbm_explained:.4f}", flush=True)

    # ----- Residual XGBoost (MAE) -----
    print("\nTraining residual XGBoost (Pseudo-Huber)...", flush=True)
    res_oof_xgb = np.full(n_tr, np.nan)
    res_te_xgb = np.zeros(len(te_df))
    for fi, (tr_idx, va_idx) in enumerate(splits):
        m = xgb.XGBRegressor(
            n_estimators=1500, max_depth=5, learning_rate=0.02,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=0.5, objective="reg:pseudohubererror",
            random_state=SEED, verbosity=0, early_stopping_rounds=50,
        )
        m.fit(
            X_tr[tr_idx], residuals[tr_idx],
            eval_set=[(X_tr[va_idx], residuals[va_idx])],
            verbose=False,
        )
        res_oof_xgb[va_idx] = m.predict(X_tr[va_idx])
        res_te_xgb += m.predict(X_te) / N_FOLDS
        print(f"  fold {fi+1}: best_iter={m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)

    res_xgb_mae = np.mean(np.abs(residuals - res_oof_xgb))
    res_xgb_explained = 1 - np.var(residuals - res_oof_xgb) / np.var(residuals)
    print(f"  residual MAE: {res_xgb_mae:.4f}  explained-var-ratio: {res_xgb_explained:.4f}", flush=True)

    # ----- Average the two residual models -----
    res_oof = 0.5 * (res_oof_lgbm + res_oof_xgb)
    res_te = 0.5 * (res_te_lgbm + res_te_xgb)

    # ----- Test corrected predictions at multiple alpha values -----
    print("\n--- Corrected predictions: nb212 + alpha * residual_pred ---", flush=True)
    print(f"{'alpha':>6}  {'OOF RAE':>9}  {'ratio':>7}  {'flag':>4}", flush=True)
    best_alpha = 0.0; best_rae = base_rae; best_ratio = base_ratio
    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
        oof_corrected = oof_nb212 + alpha * res_oof
        te_corrected  = te_nb212  + alpha * res_te
        r = rae(y_pec, oof_corrected)
        ratio = te_corrected.std() / oof_corrected.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        marker = ""
        if r < best_rae and ratio >= COLLAPSE_THRESH:
            best_rae = r; best_alpha = alpha; best_ratio = ratio
            marker = " ***"
        print(f"  {alpha:>4.2f}   {r:>9.6f}  {ratio:>7.4f}  [{flag}]{marker}", flush=True)

    print(f"\nBest alpha: {best_alpha:.2f}  RAE={best_rae:.6f}  ratio={best_ratio:.4f}", flush=True)

    # Save the residual model OOF/test (raw, not added to nb212) for downstream stacking
    out_stem = "nb218_residual_blend"
    np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", res_oof)
    np.save(DATA_PROCESSED / f"te_{out_stem}.npy",  res_te)

    # If we found an improvement, also save the corrected submission
    if best_alpha > 0 and best_rae < PREV_BEST:
        oof_best = oof_nb212 + best_alpha * res_oof
        te_best  = te_nb212  + best_alpha * res_te
        np.save(DATA_PROCESSED / f"oof_nb218_corrected.npy", oof_best)
        np.save(DATA_PROCESSED / f"te_nb218_corrected.npy",  te_best)
        sub = pd.DataFrame({
            "SMILES": te_df["smiles"].values,
            "Molecule Name": te_df["name"].values,
            "pEC50": te_best,
        })
        sub.to_csv(SUBMISSIONS / f"nb218_corrected_a{best_alpha:.2f}.csv", index=False)
        print(f"\n*** SAVED CORRECTED nb218 (RAE={best_rae:.6f} beats {PREV_BEST}) ***", flush=True)
    else:
        print(f"\n(No improvement over nb212 baseline. Saved residual model for stacking.)", flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
