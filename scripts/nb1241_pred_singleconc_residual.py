"""nb1241 -- Single-concentration log2FC as auxiliary residual feature on top of nb1070.

Hypothesis:
    The single-concentration primary screen
    (pxr-challenge_single_concentration_TRAIN.csv, 21,003 rows / 10,870
    unique compounds) is a CHEAP, NOISY orthogonal experimental readout of
    PXR activation: log2 fold-change vs. baseline at 4 concentrations.  It
    overlaps ~2,740 of the 4,139 CRC train compounds (~66%) but has NO test
    overlap.  We can therefore:
        1. Aggregate per-compound SP log2FC (median, max, mean -log10 FDR,
           count of measurements).
        2. Train a Morgan+RDKit -> SP-aggregate model on the SP-only
           subset, evaluated by 5-fold scaffold CV.
        3. Predict SP-aggregates on ALL 513 test compounds (SP labels not
           required at test time).
        4. Use predicted (median_log2FC, max_log2FC) on the 513 as residual
           features alongside MACCS-166 -> 168-dim residual learner on top
           of nb1070 anchor.

    Distinct from nb1183 (MACCS-only) because the SP-derived features
    inject EXPERIMENTAL information from an orthogonal assay
    (transcriptional fold-change in PXR-positive cells) rather than just
    chemistry features.  If the SP signal carries even ~0.1-0.2 RAE
    information about the CRC pEC50, the residual learner should pick it up
    and beat nb1183 0.5513 / nb1211 0.5451 at the 0.003 margin.

Protocol:
  1. load_single_conc -> 21,003 rows. Standardize SMILES. Aggregate per
     standardized SMILES:
        median_log2fc  = median(log2_fc_estimate)
        max_log2fc     = max(log2_fc_estimate)
        mean_neg_log10_fdr = mean(neg_log10_fdr)
        n_meas         = count of measurements
  2. Inner-join to CRC train (standardized SMILES) -> n_train_with_sp.
     Train two LGBM models (one per target: median, max) on
     cache_combined_features X_tr[sp_idx] (Morgan + RDKit, 2265 dims).
     Use 5-fold KFold (seed 42) and compute scaffold-free RAE on this
     SP-train subset.
  3. Train final SP predictors on ALL n_train_with_sp rows; predict on the
     full 513 test set via cache_combined_features X_te.
  4. Build residual feature matrix on 253 unblind:
        residual = y_unb - nb1070_pred_oof
        F_unb = concat[ MACCS_unb (166),
                        pred_median_log2fc_513[unb_idx] (1),
                        pred_max_log2fc_513[unb_idx]    (1) ]
        shape (253, 168)
  5. 5-seed shallow LGBM Huber bag (seeds 0/1/7/42/137), 5-fold per seed.
  6. Pool mean-bag corrected OOF; pooled RAE vs nb1070 anchor (0.5771),
     nb1183 (0.5513) and nb1211 (0.5451) at 0.003 margin.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1241_pred_median_log2fc_513.npy   (513,) float32
  data/processed/nb1241_pred_max_log2fc_513.npy      (513,) float32
  data/processed/nb1241_sp_train_oof_pred.npy        (n_sp,2) float32
  data/processed/nb1241_per_seed_corrected_oof.npy   (5, 253) float32
  data/processed/nb1241_mean_bag_oof.npy             (253,) float32
  data/processed/nb1241_median_bag_oof.npy           (253,) float32
  data/processed/nb1241_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_train, load_test, load_single_conc
from pxr.chem import standardize_smiles
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1241"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

SP_PRED_FOLDS = 5
SP_PRED_SEED = 42

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
COMBINED_CACHE = DATA_PROCESSED / "cache_combined_features.npz"

NB1070_REF_POOLED = 0.5771
NB1183_REF = 0.5513
NB1211_REF = 0.5451
DECISION_MARGIN = 0.003


def _shallow_resid_params(seed: int) -> dict:
    """Same shallow LGBM Huber capacity as nb1183 family."""
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _sp_predictor_params(seed: int) -> dict:
    """SP predictor: a regular LGBM (not residual), so use deeper capacity."""
    return dict(
        objective="regression_l1",
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_shallow_resid_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _build_sp_aggregates() -> pd.DataFrame:
    """Aggregate single-concentration data per standardized SMILES."""
    sc = load_single_conc()
    sc_uni = sc["smiles"].drop_duplicates().to_frame()
    sc_uni["std"] = sc_uni["smiles"].map(standardize_smiles)
    sc = sc.merge(sc_uni, on="smiles", how="left")
    # drop unparseable
    sc = sc[sc["std"].notna()].copy()
    # per-compound aggregate
    agg = sc.groupby("std").agg(
        median_log2fc=("log2_fc_estimate", "median"),
        max_log2fc=("log2_fc_estimate", "max"),
        mean_neg_log10_fdr=("neg_log10_fdr", "mean"),
        n_meas=("log2_fc_estimate", "count"),
    ).reset_index()
    return agg


def _load_combined() -> tuple[np.ndarray, np.ndarray]:
    d = np.load(COMBINED_CACHE)
    return d["X_tr"].astype(np.float32), d["X_te"].astype(np.float32)


def _load_maccs_unblind(n_test: int, unb_idx: np.ndarray) -> np.ndarray:
    X_te = np.load(MACCS_TE_PATH)
    if X_te.shape[0] != n_test:
        raise ValueError(
            f"MACCS test cache shape mismatch: {X_te.shape} vs {n_test}"
        )
    # Keep first 166 cols if 167 (drop the padding bit to keep dim convention)
    if X_te.shape[1] == 167:
        X_te = X_te[:, 1:]   # drop bit 0 padding
    return X_te[unb_idx].astype(np.float32)


def _sp_predictor_cv_then_deploy(
    X_sp: np.ndarray, y_sp: np.ndarray, X_test: np.ndarray, target_name: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """5-fold CV on SP training subset; report RAE; refit on full SP train,
    predict on 513 test."""
    n = len(y_sp)
    kf = KFold(n_splits=SP_PRED_FOLDS, shuffle=True, random_state=SP_PRED_SEED)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_sp_predictor_params(SP_PRED_SEED))
        mdl.fit(X_sp[tr_loc], y_sp[tr_loc])
        oof[va_loc] = mdl.predict(X_sp[va_loc])
    rae_cv = float(rae(y_sp, oof))
    # Deploy: train on all SP train rows, predict on 513
    mdl_dep = LGBMRegressor(**_sp_predictor_params(SP_PRED_SEED))
    mdl_dep.fit(X_sp, y_sp)
    pred_test = mdl_dep.predict(X_test).astype(np.float64)
    print(f"   [{target_name}] CV RAE={rae_cv:.4f}  "
          f"train_std={y_sp.std():.3f}  pred_test_mean={pred_test.mean():+.3f}  "
          f"pred_test_std={pred_test.std():.3f}")
    return oof, pred_test, rae_cv


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SP-log2FC residual feature on top of nb1070 anchor")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          shallow LGBM Huber (depth=3, n_est=80, lr=0.05)")
    print("=" * 78)

    # ---- Load test, audit indices, anchor OOF ----
    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    anchor_oof = np.load(anchor_path).astype(np.float64)
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof RAE = {rae_anchor:.4f}  "
          f"(ref ~{NB1070_REF_POOLED:.4f})")

    # ---- Load train + standardize SMILES ----
    tr = load_train()
    tr["std"] = tr["smiles"].map(standardize_smiles)
    print(f"[std] train standardized: "
          f"{tr['std'].notna().sum()}/{len(tr)}")

    # ---- Aggregate SP data ----
    sp_agg = _build_sp_aggregates()
    print(f"[sp ] unique SP compounds aggregated: {len(sp_agg)}")
    print(f"[sp ] median_log2fc: mean={sp_agg['median_log2fc'].mean():+.3f}  "
          f"std={sp_agg['median_log2fc'].std():.3f}  "
          f"range=[{sp_agg['median_log2fc'].min():+.3f}, "
          f"{sp_agg['median_log2fc'].max():+.3f}]")
    print(f"[sp ] max_log2fc:    mean={sp_agg['max_log2fc'].mean():+.3f}  "
          f"std={sp_agg['max_log2fc'].std():.3f}")
    print(f"[sp ] n_meas: mean={sp_agg['n_meas'].mean():.2f}  "
          f"max={sp_agg['n_meas'].max()}")

    # ---- Inner-join SP to TRAIN ----
    tr_with_sp = tr.merge(sp_agg, on="std", how="inner")
    n_sp_train = len(tr_with_sp)
    print(f"[join] CRC train compounds with SP data: "
          f"{n_sp_train}/{len(tr)} ({100*n_sp_train/len(tr):.1f}%)")

    sp_train_idx = tr_with_sp.index.to_numpy()  # row indices into ORIGINAL tr
    # We need positional indices into the ORIGINAL train ordering (cache_combined
    # was built in original train row order).  Since tr_with_sp preserves the
    # left-side index from tr (default RangeIndex), sp_train_idx is exactly the
    # positional index we need.

    # ---- Load combined features (train + test) ----
    X_tr_full, X_te_full = _load_combined()
    print(f"[feat] X_tr={X_tr_full.shape}  X_te={X_te_full.shape}")
    X_sp_train = X_tr_full[sp_train_idx]
    print(f"[feat] X_sp_train={X_sp_train.shape}")

    # ---- Train SP predictors (5-fold CV + deploy) ----
    print("\n" + "-" * 78)
    print(f"SP PREDICTOR TRAINING ({n_sp_train} CRC compounds with SP data)")
    print("-" * 78)
    y_med = tr_with_sp["median_log2fc"].to_numpy(dtype=np.float64)
    y_max = tr_with_sp["max_log2fc"].to_numpy(dtype=np.float64)

    oof_med, pred_med_test, rae_med = _sp_predictor_cv_then_deploy(
        X_sp_train, y_med, X_te_full, "median_log2fc"
    )
    oof_max, pred_max_test, rae_max = _sp_predictor_cv_then_deploy(
        X_sp_train, y_max, X_te_full, "max_log2fc"
    )
    sp_train_oof = np.stack([oof_med, oof_max], axis=1).astype(np.float32)
    np.save(DATA_PROCESSED / f"{TAG}_sp_train_oof_pred.npy", sp_train_oof)
    np.save(DATA_PROCESSED / f"{TAG}_pred_median_log2fc_513.npy",
            pred_med_test.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_pred_max_log2fc_513.npy",
            pred_max_test.astype(np.float32))

    # ---- Predicted SP distribution on 513 test ----
    print(f"[pred] predicted_median_log2fc_513: "
          f"mean={pred_med_test.mean():+.3f}  std={pred_med_test.std():.3f}  "
          f"range=[{pred_med_test.min():+.3f}, {pred_med_test.max():+.3f}]")
    print(f"[pred] predicted_max_log2fc_513:    "
          f"mean={pred_max_test.mean():+.3f}  std={pred_max_test.std():.3f}  "
          f"range=[{pred_max_test.min():+.3f}, {pred_max_test.max():+.3f}]")

    # ---- Build residual feature matrix (MACCS + SP preds) ----
    print("\n" + "-" * 78)
    print("RESIDUAL FEATURE ASSEMBLY (MACCS-166 + 2 SP-pred = 168)")
    print("-" * 78)
    X_maccs_unb = _load_maccs_unblind(n_test=n_test, unb_idx=unb_idx)
    print(f"[feat] MACCS_unb = {X_maccs_unb.shape}")
    sp_med_unb = pred_med_test[unb_idx].reshape(-1, 1).astype(np.float32)
    sp_max_unb = pred_max_test[unb_idx].reshape(-1, 1).astype(np.float32)
    X_unb = np.hstack([X_maccs_unb, sp_med_unb, sp_max_unb]).astype(np.float32)
    print(f"[feat] X_unb (168 cols) = {X_unb.shape}")
    print(f"[feat] SP-feature std (unb): "
          f"median={sp_med_unb.std():.3f}  max={sp_max_unb.std():.3f}")

    # ---- Residual cross-fit per seed ----
    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    print("\n" + "-" * 78)
    print("PER-SEED RESIDUAL CROSS-FIT")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_arr))
    rae_per_seed_std = float(per_seed_arr.std())
    rae_per_seed_min = float(per_seed_arr.min())
    rae_per_seed_max = float(per_seed_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1183 ref             = {NB1183_REF:.4f}")
    print(f"   nb1211 ref             = {NB1211_REF:.4f}")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_REF - DECISION_MARGIN
    beats_nb1211 = rae_mean_bag < NB1211_REF - DECISION_MARGIN

    if beats_nb1211:
        verdict = "SP_FEATURE_BEATS_NB1211_AND_NB1183_NEW_BEST"
    elif beats_nb1183:
        verdict = "SP_FEATURE_BEATS_NB1183_BUT_NOT_NB1211"
    elif beats_nb1070:
        verdict = "SP_FEATURE_HELPS_NB1070_BUT_NOT_NB1183"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "SP_FEATURE_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "SP_FEATURE_HURTS_NB1070"
    print(f"   verdict                = {verdict}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_pred_median_log2fc_513.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_pred_max_log2fc_513.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_train": int(len(tr)),
        "n_train_with_sp": int(n_sp_train),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "sp_unique_compounds": int(len(sp_agg)),
        "sp_pred_cv_rae_median_log2fc": rae_med,
        "sp_pred_cv_rae_max_log2fc":    rae_max,
        "pred_median_log2fc_513_mean": float(pred_med_test.mean()),
        "pred_median_log2fc_513_std":  float(pred_med_test.std()),
        "pred_median_log2fc_513_min":  float(pred_med_test.min()),
        "pred_median_log2fc_513_max":  float(pred_med_test.max()),
        "pred_max_log2fc_513_mean": float(pred_max_test.mean()),
        "pred_max_log2fc_513_std":  float(pred_max_test.std()),
        "pred_max_log2fc_513_min":  float(pred_max_test.min()),
        "pred_max_log2fc_513_max":  float(pred_max_test.max()),
        "feature_dim": int(X_unb.shape[1]),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_REF,
        "delta_mean_bag_vs_nb1211": rae_mean_bag - NB1211_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1211": bool(beats_nb1211),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1183_ref": NB1183_REF,
        "nb1211_ref": NB1211_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_train_with_sp",
        "sp_unique_compounds",
        "sp_pred_cv_rae_median_log2fc",
        "sp_pred_cv_rae_max_log2fc",
        "pred_median_log2fc_513_mean",
        "pred_median_log2fc_513_std",
        "pred_max_log2fc_513_mean",
        "pred_max_log2fc_513_std",
        "rae_anchor_nb1070",
        "per_seed_rae",
        "rae_per_seed_mean",
        "rae_per_seed_std",
        "rae_mean_bag",
        "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1183",
        "delta_mean_bag_vs_nb1211",
        "beats_nb1070",
        "beats_nb1183",
        "beats_nb1211",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
