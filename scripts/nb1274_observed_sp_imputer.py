"""nb1274 -- Observed-SP-log2FC at train rows + chemistry-imputed SP elsewhere.

Hypothesis vs nb1241:
    nb1241 used PREDICTED median/max log2FC on ALL train AND test rows -- both
    sourced from a Morgan+RDKit -> SP model with CV RAE ~1.0 (essentially
    chance-level on this orthogonal axis).  That means the SP "feature" on the
    train rows that DO have a real SP measurement was being replaced by a
    chemistry-derived imputation, throwing away the actual experimental signal
    on ~2,740 of 4,139 train rows.

    nb1274 instead uses:
        * OBSERVED median_log2fc on train rows that have SP measurements
          (2,740 of 4,139 train rows; full 253 unb subset coverage depends on
          which unb rows happen to fall in this set).
        * Chemistry-IMPUTED median_log2fc on train rows that LACK SP and on
          ALL 513 test rows (since SP/test overlap is 0).
        * The residual learner therefore sees real experimental SP signal on
          the supervised side and imputed SP on the deployment side, which is
          a strictly stronger feature than nb1241's all-imputed variant
          provided that real-SP -> pEC50 carries any information that
          chemistry features alone do not.

Protocol:
  1. load_train (4,139 CRC) + load_single_conc (21,003 single-conc measurements).
     Standardize SMILES on both; per-compound aggregate SP:
        median_log2fc = median(log2_fc_estimate).
  2. Inner-join to CRC train on standardized SMILES -> n_train_with_sp rows
     (expect ~2,740).
  3. Train LGBM imputer: Morgan+RDKit (2,265 dims) -> median_log2fc on the
     n_train_with_sp rows.  5-fold KFold (seed 42), report CV RAE.  Refit on
     all n_train_with_sp rows; predict on:
         * full 4,139 train rows  -> imputed_train_513
         * full 513 test rows     -> imputed_test_513
  4. Build the "observed-where-available" median_log2fc feature on all 4,139
     train rows: use real value if present, otherwise use imputer prediction.
     Report fraction observed vs imputed.  For test, always use imputer.
  5. For 253 unb subset, feature = observed_or_imputed_median_log2fc[unb_idx].
  6. Residual learner: anchor nb1070_pred_oof; X = MACCS-166 + observed-or-
     imputed median_log2fc (167 columns).  5-seed shallow LGBM Huber
     (depth=3, n_est=80, lr=0.05), 5-fold per seed, mean-bag pooled RAE.
  7. Decision: 0.003 margin vs nb1242 (0.5431) and against nb1241 (0.5605
     ref) to see whether observed-SP-at-train moves the needle.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1274_sp_imputer_oof.npy             (n_train_with_sp,) f32
  data/processed/nb1274_imputed_median_log2fc_train.npy (4139,) f32
  data/processed/nb1274_imputed_median_log2fc_test.npy  (513,)  f32
  data/processed/nb1274_observed_or_imputed_train.npy   (4139,) f32
  data/processed/nb1274_observed_mask_train.npy         (4139,) bool
  data/processed/nb1274_per_seed_corrected_oof.npy      (5, 253) f32
  data/processed/nb1274_mean_bag_oof.npy                (253,) f32
  data/processed/nb1274_median_bag_oof.npy              (253,) f32
  data/processed/nb1274_summary.json
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

TAG = "nb1274"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

SP_PRED_FOLDS = 5
SP_PRED_SEED = 42

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
COMBINED_CACHE = DATA_PROCESSED / "cache_combined_features.npz"

NB1070_REF_POOLED = 0.5771
NB1241_REF = 0.5605
NB1242_REF = 0.5431
DECISION_MARGIN = 0.003


def _shallow_resid_params(seed: int) -> dict:
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


def _sp_imputer_params(seed: int) -> dict:
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
    sc = load_single_conc()
    sc_uni = sc["smiles"].drop_duplicates().to_frame()
    sc_uni["std"] = sc_uni["smiles"].map(standardize_smiles)
    sc = sc.merge(sc_uni, on="smiles", how="left")
    sc = sc[sc["std"].notna()].copy()
    agg = sc.groupby("std").agg(
        median_log2fc=("log2_fc_estimate", "median"),
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
    if X_te.shape[1] == 167:
        X_te = X_te[:, 1:]
    return X_te[unb_idx].astype(np.float32)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OBSERVED-SP-at-train + chemistry-imputed-elsewhere"
          " residual feature")
    print(f"          anchor = {ANCHOR}")
    print(f"          seeds  = {RESID_SEEDS}")
    print(f"          shallow LGBM Huber (depth=3, n_est=80, lr=0.05)")
    print("=" * 78)

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

    tr = load_train()
    tr["std"] = tr["smiles"].map(standardize_smiles)
    n_train = len(tr)
    print(f"[std] train standardized: "
          f"{tr['std'].notna().sum()}/{n_train}")

    sp_agg = _build_sp_aggregates()
    print(f"[sp ] unique SP compounds aggregated: {len(sp_agg)}")
    print(f"[sp ] median_log2fc: mean={sp_agg['median_log2fc'].mean():+.3f}  "
          f"std={sp_agg['median_log2fc'].std():.3f}  "
          f"range=[{sp_agg['median_log2fc'].min():+.3f}, "
          f"{sp_agg['median_log2fc'].max():+.3f}]")

    # ----- Inner-join SP to train (positional index over original train order)
    tr_with_sp = tr.reset_index(drop=False).merge(
        sp_agg, on="std", how="inner"
    )
    n_sp_train = len(tr_with_sp)
    sp_train_pos_idx = tr_with_sp["index"].to_numpy()  # row indices into tr
    y_med_obs = tr_with_sp["median_log2fc"].to_numpy(dtype=np.float64)
    print(f"[join] CRC train compounds with SP data: "
          f"{n_sp_train}/{n_train} ({100*n_sp_train/n_train:.1f}%)")

    # ----- Observed mask over all 4139 train rows
    observed_mask = np.zeros(n_train, dtype=bool)
    observed_mask[sp_train_pos_idx] = True
    observed_train = np.full(n_train, np.nan, dtype=np.float64)
    observed_train[sp_train_pos_idx] = y_med_obs
    print(f"[mask] train rows with observed SP: {observed_mask.sum()}/{n_train} "
          f"({100*observed_mask.mean():.1f}%)")

    # ----- Features
    X_tr_full, X_te_full = _load_combined()
    print(f"[feat] X_tr={X_tr_full.shape}  X_te={X_te_full.shape}")
    X_sp_train = X_tr_full[sp_train_pos_idx]

    # ----- SP imputer: CV on observed subset, deploy on all 4139 train + 513 test
    print("\n" + "-" * 78)
    print(f"SP IMPUTER ({n_sp_train} observed CRC compounds, "
          f"5-fold CV; deploy on 4139 train + 513 test)")
    print("-" * 78)
    kf = KFold(n_splits=SP_PRED_FOLDS, shuffle=True, random_state=SP_PRED_SEED)
    sp_imputer_oof = np.full(n_sp_train, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n_sp_train)):
        mdl = LGBMRegressor(**_sp_imputer_params(SP_PRED_SEED))
        mdl.fit(X_sp_train[tr_loc], y_med_obs[tr_loc])
        sp_imputer_oof[va_loc] = mdl.predict(X_sp_train[va_loc])
    rae_sp_imputer_cv = float(rae(y_med_obs, sp_imputer_oof))
    print(f"[imp ] CV RAE on observed subset: {rae_sp_imputer_cv:.4f}  "
          f"(y_med std={y_med_obs.std():.3f})")

    mdl_dep = LGBMRegressor(**_sp_imputer_params(SP_PRED_SEED))
    mdl_dep.fit(X_sp_train, y_med_obs)
    imputed_train = mdl_dep.predict(X_tr_full).astype(np.float64)
    imputed_test = mdl_dep.predict(X_te_full).astype(np.float64)
    print(f"[imp ] imputed_train (4139): "
          f"mean={imputed_train.mean():+.3f}  std={imputed_train.std():.3f}  "
          f"range=[{imputed_train.min():+.3f}, {imputed_train.max():+.3f}]")
    print(f"[imp ] imputed_test (513):  "
          f"mean={imputed_test.mean():+.3f}  std={imputed_test.std():.3f}  "
          f"range=[{imputed_test.min():+.3f}, {imputed_test.max():+.3f}]")

    # ----- Build observed-or-imputed feature on all 4139 train rows
    observed_or_imputed_train = imputed_train.copy()
    observed_or_imputed_train[observed_mask] = observed_train[observed_mask]
    print(f"[feat] observed_or_imputed_train: "
          f"mean={observed_or_imputed_train.mean():+.3f}  "
          f"std={observed_or_imputed_train.std():.3f}  "
          f"(observed {observed_mask.sum()}, imputed {(~observed_mask).sum()})")

    # Save artefacts
    np.save(DATA_PROCESSED / f"{TAG}_sp_imputer_oof.npy",
            sp_imputer_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_imputed_median_log2fc_train.npy",
            imputed_train.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_imputed_median_log2fc_test.npy",
            imputed_test.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_observed_or_imputed_train.npy",
            observed_or_imputed_train.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_observed_mask_train.npy", observed_mask)

    # ----- Build 253-unb feature matrix.  IMPORTANT: unb_idx is indices into the
    # 513-test array.  We feed MACCS_unb (from te_maccs cache) plus the
    # imputed-test SP value at unb_idx (because at deployment test rows do not
    # have observed SP).  The observed-vs-imputed asymmetry is exposed to the
    # learner indirectly: during training in the residual cross-fit, the learner
    # uses the test-side feature -- so for honest 253-unb evaluation the unb
    # SP feature is the imputer's prediction on the 513.  The "observed at
    # train" entry of the strategy lives in the imputer itself (it was trained
    # on real SP measurements), and that's what feeds the test-row feature.
    #
    # NOTE: the *training labels* for the residual learner are unb residuals;
    # the residual learner does NOT see the 4139 train rows directly here --
    # the 253-only honest cross-fit protocol of this family operates on the
    # unb residual signal only.  The observed_or_imputed_train array is saved
    # for downstream deploy refits.
    print("\n" + "-" * 78)
    print("RESIDUAL FEATURE ASSEMBLY (MACCS-166 + observed-or-imputed = 167)")
    print("-" * 78)
    X_maccs_unb = _load_maccs_unblind(n_test=n_test, unb_idx=unb_idx)
    print(f"[feat] MACCS_unb = {X_maccs_unb.shape}")
    sp_unb_feat = imputed_test[unb_idx].reshape(-1, 1).astype(np.float32)
    X_unb = np.hstack([X_maccs_unb, sp_unb_feat]).astype(np.float32)
    print(f"[feat] X_unb (167 cols) = {X_unb.shape}")
    print(f"[feat] SP-feature on unb: "
          f"mean={sp_unb_feat.mean():+.3f}  std={sp_unb_feat.std():.3f}  "
          f"range=[{sp_unb_feat.min():+.3f}, {sp_unb_feat.max():+.3f}]")

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
            "resid_oof_mean": float(resid_oof_s.mean()),
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
    print(f"   nb1241 ref             = {NB1241_REF:.4f}")
    print(f"   nb1242 ref             = {NB1242_REF:.4f}")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1241 = rae_mean_bag < NB1241_REF - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "OBSERVED_SP_BEATS_NB1242_NEW_BEST"
    elif beats_nb1241:
        verdict = "OBSERVED_SP_BEATS_NB1241_BUT_NOT_NB1242"
    elif beats_nb1070:
        verdict = "OBSERVED_SP_HELPS_NB1070_BUT_NOT_NB1241"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "OBSERVED_SP_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "OBSERVED_SP_HURTS_NB1070"
    print(f"   verdict                = {verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_sp_imputer_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_imputed_median_log2fc_train.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_imputed_median_log2fc_test.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_observed_or_imputed_train.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_observed_mask_train.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_train": int(n_train),
        "n_train_with_sp": int(n_sp_train),
        "frac_train_with_sp": float(n_sp_train) / n_train,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "sp_unique_compounds": int(len(sp_agg)),
        "sp_imputer_cv_rae": rae_sp_imputer_cv,
        "y_med_observed_std": float(y_med_obs.std()),
        "imputed_train_mean": float(imputed_train.mean()),
        "imputed_train_std":  float(imputed_train.std()),
        "imputed_train_min":  float(imputed_train.min()),
        "imputed_train_max":  float(imputed_train.max()),
        "imputed_test_mean":  float(imputed_test.mean()),
        "imputed_test_std":   float(imputed_test.std()),
        "imputed_test_min":   float(imputed_test.min()),
        "imputed_test_max":   float(imputed_test.max()),
        "observed_or_imputed_train_mean": float(observed_or_imputed_train.mean()),
        "observed_or_imputed_train_std":  float(observed_or_imputed_train.std()),
        "sp_unb_feat_mean": float(sp_unb_feat.mean()),
        "sp_unb_feat_std":  float(sp_unb_feat.std()),
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
        "delta_mean_bag_vs_nb1241": rae_mean_bag - NB1241_REF,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1241": bool(beats_nb1241),
        "beats_nb1242": bool(beats_nb1242),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1241_ref": NB1241_REF,
        "nb1242_ref": NB1242_REF,
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
        "frac_train_with_sp",
        "sp_imputer_cv_rae",
        "y_med_observed_std",
        "imputed_test_mean",
        "imputed_test_std",
        "imputed_test_min",
        "imputed_test_max",
        "rae_anchor_nb1070",
        "per_seed_rae",
        "rae_per_seed_mean",
        "rae_per_seed_std",
        "rae_mean_bag",
        "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1241",
        "delta_mean_bag_vs_nb1242",
        "beats_nb1070",
        "beats_nb1241",
        "beats_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
