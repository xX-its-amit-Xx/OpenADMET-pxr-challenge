"""nb960 -- Self-training with high-confidence pseudo-labels from SP-only compounds.

Hypothesis: the 21k single-concentration (SP) screen covers ~8.5k SMILES that
are absent from the 4139 CRC train set. If we use a bagged LGBM Huber on CRC
to assign pseudo-pEC50 + epistemic uncertainty to the SP-only compounds and
keep only the low-uncertainty rows, we expand training scaffold coverage with
minimal noise. Sample weight 0.4 prevents the (still noisier) pseudo-labels
from dominating the true CRC labels.

Pipeline:
  1) Load 4139 CRC train + 21003 SP rows. Reduce SP to unique SMILES, drop
     any that overlap with CRC by standardized SMILES.
  2) Featurize all (combined: Morgan + RDKit, imputed).
  3) 5-seed bagged LGBM Huber (alpha=1.0) trained on the 4139 CRC. Predict
     pec50 mean + std (epistemic) on SP-only.
  4) Keep SP-only rows with bagged_std < 0.3 -> pseudo-labels (expect ~5-10k).
  5) Retrain a single LGBM Huber on (CRC + pseudo) with sample_weight
     CRC=1.0, pseudo=0.4. Scaffold 5-fold CV on the CRC rows only to compute
     a clean OOF RAE; use the full retrained model to predict 513 test.
  6) in_RAE on 253 unblind. Save submission + artifacts.

Wall budget: < 10 min CPU.
Outputs:
  C:/pxr_artifacts/nb960/bagged_mean.npy      # (n_sp_only,)
  C:/pxr_artifacts/nb960/bagged_std.npy       # (n_sp_only,)
  C:/pxr_artifacts/nb960/pseudo_mask.npy      # bool (n_sp_only,)
  C:/pxr_artifacts/nb960/te_pred.npy          # (513,) float32
  C:/pxr_artifacts/nb960/summary.json
  data/processed/oof_nb960.npy                # (4139,) float32  (CV on CRC)
  data/processed/te_nb960.npy                 # (513,)
  submissions/nb960_pseudo_self_train.csv
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from pxr.data import load_train, load_test, load_single_conc
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, standardize_smiles
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb960"
ART = Path("C:/pxr_artifacts/nb960")
ART.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
N_SEEDS = 5
STD_THRESHOLD = 0.3
PSEUDO_WEIGHT = 0.4

BASE_PARAMS = dict(
    objective="huber",
    alpha=1.0,
    n_estimators=800,
    num_leaves=64,
    learning_rate=0.04,
    min_child_samples=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.1,
    verbose=-1,
    n_jobs=4,
)


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def main():
    t0 = time.time()
    print("=== nb960: self-training with high-confidence SP pseudo-labels ===")

    # --- Truth + indices for in_RAE ---
    unblind_idx = np.load(DATA_PROCESSED / "nb472_unblind_idx.npy")
    y_unblind = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    assert len(unblind_idx) == len(y_unblind) == 253

    # --- Load CRC train + test + SP screen ---
    tr = load_train()
    te = load_test()
    sp = load_single_conc()
    y_tr = tr["pec50"].values.astype(np.float64)
    print(f"CRC train={len(tr)}  test={len(te)}  SP rows={len(sp)}")

    # --- Reduce SP to unique standardized SMILES, drop CRC overlap ---
    print("Standardizing SMILES (canonical strings)...")
    tr_std = tr["smiles"].map(standardize_smiles).fillna("").values
    sp_std = sp["smiles"].map(standardize_smiles).fillna("").values
    crc_set = set(s for s in tr_std if s)

    sp_df = pd.DataFrame({"smiles": sp_std, "raw": sp["smiles"].values})
    sp_df = sp_df[sp_df["smiles"] != ""].drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    sp_only_df = sp_df[~sp_df["smiles"].isin(crc_set)].reset_index(drop=True)
    print(f"SP unique canonical={len(sp_df)}  SP-only (no CRC overlap)={len(sp_only_df)}")

    # --- Featurize ---
    print("Featurizing (Morgan + RDKit)...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    X_sp = impute(combined(sp_only_df["smiles"].tolist()))
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}  X_sp={X_sp.shape}  "
          f"mem~{(X_tr.nbytes+X_te.nbytes+X_sp.nbytes)/1e6:.1f}MB")

    # --- 5-seed bagged LGBM on full CRC: predict mean + std on SP-only ---
    print(f"Training {N_SEEDS}-seed bagged LGBM Huber on full CRC...")
    sp_preds = np.empty((N_SEEDS, len(X_sp)), dtype=np.float32)
    te_pre = np.empty((N_SEEDS, len(X_te)), dtype=np.float32)
    for s in range(N_SEEDS):
        params = dict(BASE_PARAMS, random_state=SEED + s, bagging_seed=SEED + s,
                      feature_fraction_seed=SEED + s)
        # rng-shuffled subsample via bagging
        m = lgb.train(
            params,
            lgb.Dataset(X_tr, label=y_tr),
            callbacks=[lgb.log_evaluation(-1)],
        )
        sp_preds[s] = m.predict(X_sp).astype(np.float32)
        te_pre[s] = m.predict(X_te).astype(np.float32)
        print(f"  seed {s}  trained  elapsed={time.time()-t0:.1f}s")

    bagged_mean = sp_preds.mean(axis=0)
    bagged_std = sp_preds.std(axis=0)
    te_pre_mean = te_pre.mean(axis=0)
    print(f"  bagged_std: median={np.median(bagged_std):.3f}  "
          f"p25={np.percentile(bagged_std,25):.3f}  p75={np.percentile(bagged_std,75):.3f}")

    # --- High-confidence pseudo-label selection ---
    mask = bagged_std < STD_THRESHOLD
    n_pseudo = int(mask.sum())
    print(f"Pseudo selected (std<{STD_THRESHOLD}): {n_pseudo}/{len(X_sp)} "
          f"({100*n_pseudo/len(X_sp):.1f}%)")

    np.save(ART / "bagged_mean.npy", bagged_mean)
    np.save(ART / "bagged_std.npy", bagged_std)
    np.save(ART / "pseudo_mask.npy", mask)

    X_pseudo = X_sp[mask]
    y_pseudo = bagged_mean[mask].astype(np.float64)

    # --- Retrain LGBM on (CRC + pseudo) with sample weight; scaffold CV on CRC ---
    print("Retraining LGBM on CRC + pseudo with sample weights (CRC=1.0, pseudo=0.4)...")
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    n_crc = len(X_tr)
    oof = np.full(n_crc, np.nan)
    retrain_params = dict(BASE_PARAMS, random_state=SEED, n_estimators=1000)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_fold = np.vstack([X_tr[tr_idx], X_pseudo])
        y_fold = np.concatenate([y_tr[tr_idx], y_pseudo])
        w_fold = np.concatenate([np.ones(len(tr_idx)), np.full(len(y_pseudo), PSEUDO_WEIGHT)])
        m = lgb.train(
            retrain_params,
            lgb.Dataset(X_fold, label=y_fold, weight=w_fold),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}  "
              f"elapsed={time.time()-t0:.1f}s")

    oof_rae = rae(y_tr, oof)
    print(f"OOF RAE (CRC scaffold-5-fold): {oof_rae:.4f}")

    # Final fit on ALL CRC + pseudo for test deploy
    print("Final fit on all CRC + pseudo for test deploy...")
    X_full = np.vstack([X_tr, X_pseudo])
    y_full = np.concatenate([y_tr, y_pseudo])
    w_full = np.concatenate([np.ones(n_crc), np.full(len(y_pseudo), PSEUDO_WEIGHT)])
    m_final = lgb.train(
        retrain_params,
        lgb.Dataset(X_full, label=y_full, weight=w_full),
        callbacks=[lgb.log_evaluation(-1)],
    )
    te_pred = np.clip(m_final.predict(X_te), y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"  te_pred: mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

    in_r = in_rae(y_unblind, te_pred[unblind_idx])
    in_r_base = in_rae(y_unblind, te_pre_mean[unblind_idx])
    print(f"in_RAE(253) self-train = {in_r:.4f}")
    print(f"in_RAE(253) baseline bagged (no pseudo) = {in_r_base:.4f}")

    # --- Save artifacts ---
    np.save(ART / "te_pred.npy", te_pred.astype(np.float32))
    np.save(DATA_PROCESSED / f"oof_{TAG}.npy", oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_pred.astype(np.float32))

    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_pred,
    })
    sub_path = SUBMISSIONS / "nb960_pseudo_self_train.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Wrote submission -> {sub_path}")

    summary = {
        "n_crc": int(n_crc),
        "n_sp_only_candidates": int(len(X_sp)),
        "n_pseudo_selected": int(n_pseudo),
        "std_threshold": STD_THRESHOLD,
        "pseudo_weight": PSEUDO_WEIGHT,
        "oof_rae_crc": float(oof_rae),
        "in_rae_253_self_train": float(in_r),
        "in_rae_253_baseline_bagged": float(in_r_base),
        "wall_time_s": float(time.time() - t0),
    }
    with open(ART / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary -> {ART/'summary.json'}")
    print(f"\n=== nb960 DONE in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
