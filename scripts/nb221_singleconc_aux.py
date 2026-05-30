"""nb221 -- Single-concentration screen as auxiliary signal.

The single-concentration screen has 21,003 compounds with log2FC vs baseline +
FDR. That's 5x more data than the CRC training set. Most CRC compounds also
appear in single-conc, but ~8K compounds are single-conc-only.

Approach:
1. Load single-conc data; canonicalize SMILES; aggregate per-compound (mean log2FC
   if multiple measurements).
2. Train LGBM on single-conc to predict log2FC from morgan+rdkit features.
3. Use OOF + test predictions as a feature in a downstream pec50 LGBM.

The hypothesis: log2FC is correlated with pec50 (both measure PXR activation),
but trained on 5x more data, the log2FC model can learn structure-activity
patterns that the pec50 model can't see in 4K samples.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from pxr.data import load_train, load_test, load_single_conc
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, standardize_smiles
from pxr.featurize import combined as feat_combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
SEED = 42
COLLAPSE_THRESH = 0.58
PREV_BEST = 0.296172

t0 = time.time()


def main():
    print("=== nb221: Single-conc auxiliary signal ===\n", flush=True)

    tr_df = load_train()
    te_df = load_test()
    sc_df = load_single_conc()
    print(f"Train: {len(tr_df)}, Test: {len(te_df)}, Single-conc: {len(sc_df)}", flush=True)

    # Find log2FC column (it's something like 'log2FC' or 'log2_fold_change')
    print(f"Single-conc columns: {list(sc_df.columns)[:15]}", flush=True)

    # Try to identify the log2FC column
    log2fc_cols = [c for c in sc_df.columns if 'log2' in c.lower() or 'fold' in c.lower()]
    fdr_cols = [c for c in sc_df.columns if 'fdr' in c.lower() or 'p-value' in c.lower() or 'pvalue' in c.lower()]
    print(f"Candidate log2FC cols: {log2fc_cols}", flush=True)
    print(f"Candidate FDR cols: {fdr_cols}", flush=True)

    if not log2fc_cols:
        print("No log2FC column found, aborting.", flush=True)
        return

    log2fc_col = log2fc_cols[0]
    print(f"Using {log2fc_col}\n", flush=True)

    # Aggregate per-compound (mean log2FC across measurements, if multiple)
    # Standardize SMILES for matching
    print("Standardizing SMILES and aggregating...", flush=True)
    sc_df = sc_df.copy()
    sc_df["std_smiles"] = sc_df["smiles"].map(lambda s: standardize_smiles(s) if isinstance(s, str) else None)
    sc_df = sc_df.dropna(subset=["std_smiles", log2fc_col])
    sc_agg = sc_df.groupby("std_smiles")[log2fc_col].mean().reset_index()
    print(f"  unique compounds: {len(sc_agg)}", flush=True)

    # Compute features for unique single-conc compounds
    print(f"\nComputing features for single-conc ({len(sc_agg)} compounds)...", flush=True)
    X_sc = impute(feat_combined(sc_agg["std_smiles"].tolist())).astype(np.float32)
    y_sc = sc_agg[log2fc_col].values.astype(np.float64)
    print(f"  shape: {X_sc.shape} ({time.time()-t0:.0f}s)", flush=True)

    # Compute features for train/test
    print("\nComputing features for train/test...", flush=True)
    tr_smiles_std = [standardize_smiles(s) if isinstance(s, str) else s for s in tr_df["smiles"]]
    te_smiles_std = [standardize_smiles(s) if isinstance(s, str) else s for s in te_df["smiles"]]
    X_tr = impute(feat_combined(tr_smiles_std)).astype(np.float32)
    X_te = impute(feat_combined(te_smiles_std)).astype(np.float32)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Train LGBM on single-conc to predict log2FC
    print("\nTraining LGBM on single-conc (predict log2FC)...", flush=True)
    # Just use a simple holdout (no CV needed since this is for prediction features)
    # 5-fold CV would give us OOF for sc_agg's compounds, but we don't need those
    # We need predictions for tr/te compounds
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X_sc))
    sc_train_idx = perm[:int(0.9 * len(perm))]
    sc_val_idx = perm[int(0.9 * len(perm)):]

    m = lgb.LGBMRegressor(
        n_estimators=2000, num_leaves=64, learning_rate=0.03,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, objective="regression_l1",
        random_state=SEED, verbose=-1,
    )
    m.fit(
        X_sc[sc_train_idx], y_sc[sc_train_idx],
        eval_set=[(X_sc[sc_val_idx], y_sc[sc_val_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    val_mae = np.mean(np.abs(y_sc[sc_val_idx] - m.predict(X_sc[sc_val_idx])))
    val_corr = np.corrcoef(y_sc[sc_val_idx], m.predict(X_sc[sc_val_idx]))[0, 1]
    print(f"  Single-conc model: best_iter={m.best_iteration_}  val MAE={val_mae:.4f}  val corr={val_corr:.4f}", flush=True)

    # Predict log2FC for train and test
    log2fc_tr = m.predict(X_tr)
    log2fc_te = m.predict(X_te)
    print(f"  log2fc_tr: mean={log2fc_tr.mean():.3f}  std={log2fc_tr.std():.3f}", flush=True)
    print(f"  log2fc_te: mean={log2fc_te.mean():.3f}  std={log2fc_te.std():.3f}", flush=True)

    # Also: lookup REAL log2FC for compounds that overlap
    sc_lookup = dict(zip(sc_agg["std_smiles"], sc_agg[log2fc_col]))
    real_log2fc_tr = np.array([sc_lookup.get(s, np.nan) for s in tr_smiles_std])
    real_log2fc_te = np.array([sc_lookup.get(s, np.nan) for s in te_smiles_std])
    n_tr_overlap = (~np.isnan(real_log2fc_tr)).sum()
    n_te_overlap = (~np.isnan(real_log2fc_te)).sum()
    print(f"\n  Train overlap with single-conc: {n_tr_overlap}/{len(tr_df)}", flush=True)
    print(f"  Test overlap with single-conc:  {n_te_overlap}/{len(te_df)}", flush=True)

    # Hybrid: real where available, predicted otherwise
    log2fc_tr_hybrid = np.where(np.isnan(real_log2fc_tr), log2fc_tr, real_log2fc_tr)
    log2fc_te_hybrid = np.where(np.isnan(real_log2fc_te), log2fc_te, real_log2fc_te)

    # ---- Train MAIN pec50 model with log2FC as feature ----
    y_pec = tr_df["pec50"].values.astype(np.float64)
    n_tr = len(tr_df)
    scaffolds = tr_df["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    X_tr_full = np.hstack([X_tr, log2fc_tr_hybrid.reshape(-1, 1)]).astype(np.float32)
    X_te_full = np.hstack([X_te, log2fc_te_hybrid.reshape(-1, 1)]).astype(np.float32)

    print(f"\nTraining MAIN LGBM (pEC50) with single-conc feature...", flush=True)
    print(f"  feature shape: {X_tr_full.shape}", flush=True)

    oof = np.full(n_tr, np.nan)
    te_pred = np.zeros(len(te_df))
    for fi, (tr_idx, va_idx) in enumerate(splits):
        m2 = lgb.LGBMRegressor(
            n_estimators=2000, num_leaves=64, learning_rate=0.03,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, objective="regression_l1",
            random_state=SEED, verbose=-1,
        )
        m2.fit(
            X_tr_full[tr_idx], y_pec[tr_idx],
            eval_set=[(X_tr_full[va_idx], y_pec[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = m2.predict(X_tr_full[va_idx])
        te_pred += m2.predict(X_te_full) / N_FOLDS
        print(f"  fold {fi+1}/{N_FOLDS}: best_iter={m2.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    r = rae(y_pec, oof)
    ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***BEATS PREV BEST***" if (ratio >= COLLAPSE_THRESH and r < PREV_BEST) else ""
    print(f"\n=== nb221 OOF: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat} ===", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

    out_stem = "nb221_singleconc_aux"
    np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{out_stem}.npy", te_pred)
    sub = pd.DataFrame({
        "SMILES": te_df["smiles"].values,
        "Molecule Name": te_df["name"].values,
        "pEC50": te_pred,
    })
    sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
    print(f"Saved: {out_stem}", flush=True)


if __name__ == "__main__":
    main()
