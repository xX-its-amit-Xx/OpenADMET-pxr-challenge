"""nb1160 -- Outer-seed bag of nb1153 (shallow LGBM Huber residual on nb1070
            using Mordred 1533 features).

Hypothesis test:
    nb1153 reported mean-bag pooled cross-fit RAE = 0.5640 (median-bag 0.5634)
    against nb1070 anchor (0.5771) using a SINGLE inner-seed set
    {0, 1, 7, 42, 137}. Per-seed spread inside nb1153 was 0.5640 .. 0.5892
    (std 0.0096). nb1143 had the same single-inner-seed-set design and
    nb1151 showed it was lucky -- outer-bag pulled it back. Does nb1153's
    0.5640 hold up under outer-seed re-rolling, or was it lucky too?

Protocol (mirrors nb1151):
    OUTER_SEEDS = {0, 1, 7, 42, 137}.
    For each outer seed o, derive inner seeds
        inner_seeds(o) = [o*1000 + s for s in (0, 1, 7, 42, 137)]
    so o=0 reproduces nb1153 verbatim (inner seeds 0,1,7,42,137 since
    0*1000+s=s). o>0 gives 5 disjoint inner-seed sets the nb1153 protocol
    has never seen.

    For each outer seed:
      * Run the full nb1153 protocol with its 5 derived inner seeds:
          - residual = y_unb - nb1070_oof
          - For each inner seed: KFold(5, shuffle=True, random_state=isd)
            shallow LGBM Huber (max_depth=3, num_leaves=7, n_est=80,
            lr=0.05, min_child_samples=20) on Mordred(1533) sliced to
            unblind rows; cross-fit residual OOF; pred_corr = anchor + resid_oof.
          - Inner mean-bag = mean over 5 inner seeds of pred_corr.
      * Record the pooled cross-fit RAE of that outer seed's inner mean-bag.

    Report: per-outer-seed pooled RAEs, mean / std / min / max, plus the
    row-level mean and median across the 5 outer-seed mean-bag OOFs.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1160_per_outer_mean_bag_oof.npy   (5, 253) float32
  data/processed/nb1160_outer_mean_oof.npy           (253,)   float32
  data/processed/nb1160_outer_median_oof.npy         (253,)   float32
  data/processed/nb1160_summary.json
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1160"
ANCHOR = "nb1070"

RESID_FOLDS = 5
INNER_SEED_BASE = [0, 1, 7, 42, 137]
OUTER_SEEDS = [0, 1, 7, 42, 137]

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5790
NB1130_MEAN_BAG_REF = 0.5673
NB1143_BAG_MEDIAN_REF = 0.5649
NB1153_MEAN_BAG_REF = 0.5640
NB1153_MEDIAN_BAG_REF = 0.5634


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- verbatim nb1153 hyperparams."""
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


def _residual_cross_fit_one_inner_seed(
    X: np.ndarray, residual: np.ndarray, inner_seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=inner_seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(inner_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred_unblind(n_test_expected: int,
                          unb_idx: np.ndarray) -> np.ndarray:
    """Load cached Mordred test matrix (513 x 1533) and slice to unblind rows."""
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}")
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m[unb_idx]


def run_one_outer_seed(outer_seed: int, X_unb: np.ndarray,
                       anchor_oof: np.ndarray, residual: np.ndarray,
                       y_unb: np.ndarray
                       ) -> tuple[float, np.ndarray, list[int], list[float]]:
    """nb1153 protocol with 5 derived inner seeds, mean inner bag."""
    inner_seeds = [outer_seed * 1000 + s for s in INNER_SEED_BASE]
    n = len(y_unb)
    inner_corr_stack = np.zeros((len(inner_seeds), n), dtype=np.float64)
    inner_per_seed_rae: list[float] = []
    for j, isd in enumerate(inner_seeds):
        resid_oof_j = _residual_cross_fit_one_inner_seed(X_unb, residual, isd)
        pred_corr_j = anchor_oof + resid_oof_j
        inner_corr_stack[j] = pred_corr_j
        inner_per_seed_rae.append(float(rae(y_unb, pred_corr_j)))
    mean_oof = inner_corr_stack.mean(axis=0)
    outer_mean_bag_rae = float(rae(y_unb, mean_oof))
    return outer_mean_bag_rae, mean_oof, inner_seeds, inner_per_seed_rae


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- outer-seed bag of nb1153 "
          f"(shallow LGBM Huber residual on nb1070, Mordred 1533)")
    print(f"          OUTER_SEEDS = {OUTER_SEEDS}")
    print(f"          inner_seeds(o) = [o*1000 + s for s in "
          f"{INNER_SEED_BASE}]")
    print(f"          LGBM: max_depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; required anchor OOF (run nb1070 first).")
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: {anchor_oof.shape} vs n_unb={n_unb}")
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    print(f"[feat] loading cached Mordred test matrix, slicing to "
          f"{n_unb} unblind rows ...")
    X_unb = _load_mordred_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat] X_unb shape = {X_unb.shape}  (Mordred only)")

    # ---- Per-outer-seed honest cross-fit + inner mean-bag ----
    print("\n" + "-" * 78)
    print("PER-OUTER-SEED nb1153 PROTOCOL (5 inner seeds -> inner mean-bag)")
    print("-" * 78)
    outer_oof_stack = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_rae: list[float] = []
    outer_records: list[dict] = []
    for i, o in enumerate(OUTER_SEEDS):
        outer_r, outer_mean_oof, inner_seeds, inner_rae = run_one_outer_seed(
            o, X_unb, anchor_oof, residual, y_unb)
        outer_oof_stack[i] = outer_mean_oof
        per_outer_rae.append(outer_r)
        outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "inner_per_seed_rae": inner_rae,
            "inner_rae_mean": float(np.mean(inner_rae)),
            "inner_rae_std": float(np.std(inner_rae)),
            "outer_mean_bag_rae": outer_r,
        })
        inner_rae_str = ",".join(f"{r:.4f}" for r in inner_rae)
        print(f"   outer {o:>3d}: inner_seeds={inner_seeds}")
        print(f"              inner_RAE=[{inner_rae_str}]  "
              f"(mean={np.mean(inner_rae):.4f}  std={np.std(inner_rae):.4f})")
        print(f"              outer mean-bag pooled RAE = {outer_r:.4f}")

    per_outer_arr = np.array(per_outer_rae)
    per_outer_mean = float(per_outer_arr.mean())
    per_outer_std = float(per_outer_arr.std())
    per_outer_min = float(per_outer_arr.min())
    per_outer_max = float(per_outer_arr.max())
    per_outer_median = float(np.median(per_outer_arr))
    print(f"\n[per-outer] pooled RAE  mean={per_outer_mean:.4f}  "
          f"std={per_outer_std:.4f}  "
          f"min={per_outer_min:.4f}  max={per_outer_max:.4f}")

    # ---- Row-level mean / median across the 5 outer-seed mean-bag OOFs ----
    outer_mean_oof = outer_oof_stack.mean(axis=0)
    outer_median_oof = np.median(outer_oof_stack, axis=0)
    outer_mean_rae = float(rae(y_unb, outer_mean_oof))
    outer_median_rae = float(rae(y_unb, outer_median_oof))
    print(f"[bag-of-bags] MEAN   across 5 outer mean-bag OOFs = "
          f"{outer_mean_rae:.4f}")
    print(f"[bag-of-bags] MEDIAN across 5 outer mean-bag OOFs = "
          f"{outer_median_rae:.4f}")

    # ---- Hypothesis verdict ----
    delta_vs_nb1153_mean = per_outer_mean - NB1153_MEAN_BAG_REF
    delta_vs_nb1070 = per_outer_mean - rae_anchor
    delta_vs_nb1130 = per_outer_mean - NB1130_MEAN_BAG_REF
    beats_nb1153 = per_outer_mean < NB1153_MEAN_BAG_REF - 0.003
    beats_nb1070 = per_outer_mean < rae_anchor - 0.003
    beats_nb1130 = per_outer_mean < NB1130_MEAN_BAG_REF - 0.003

    if abs(delta_vs_nb1153_mean) < 0.003:
        verdict = "NB1153_REPRODUCES_UNDER_OUTER_SEEDS"
    elif delta_vs_nb1153_mean > 0.003:
        verdict = "NB1153_WAS_LUCKY_SINGLE_INNER_SEED_SET"
    else:
        verdict = "NB1153_WAS_UNLUCKY_OUTER_BAG_IMPROVES"

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_mean_bag_oof.npy",
            outer_oof_stack.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_outer_mean_oof.npy",
            outer_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_outer_median_oof.npy",
            outer_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_outer_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_outer_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_outer_median_oof.npy'}")

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1070 anchor on 253             = {rae_anchor:.4f}  "
          f"(ref {NB1070_REF_POOLED:.4f})")
    print(f"   nb1130 mean-bag ref              = {NB1130_MEAN_BAG_REF:.4f}")
    print(f"   nb1143 bag_median ref            = {NB1143_BAG_MEDIAN_REF:.4f}")
    print(f"   nb1153 mean-bag ref (single seed-set)  = "
          f"{NB1153_MEAN_BAG_REF:.4f}")
    print(f"   nb1153 median-bag ref (single seed-set)= "
          f"{NB1153_MEDIAN_BAG_REF:.4f}")
    print(f"   nb1160 per-outer-seed RAE mean   = {per_outer_mean:.4f}  "
          f"std={per_outer_std:.4f}")
    print(f"   nb1160 per-outer-seed RAE median = {per_outer_median:.4f}")
    print(f"   nb1160 outer-MEAN bag-of-bags    = {outer_mean_rae:.4f}")
    print(f"   nb1160 outer-MEDIAN bag-of-bags  = {outer_median_rae:.4f}")
    print(f"   delta(per-outer mean vs nb1153)  = {delta_vs_nb1153_mean:+.4f}")
    print(f"   delta(per-outer mean vs nb1070)  = {delta_vs_nb1070:+.4f}")
    print(f"   delta(per-outer mean vs nb1130)  = {delta_vs_nb1130:+.4f}")
    print(f"   beats_nb1153 (>=0.003)           = {beats_nb1153}")
    print(f"   beats_nb1070 (>=0.003)           = {beats_nb1070}")
    print(f"   beats_nb1130 (>=0.003)           = {beats_nb1130}")
    print(f"   verdict                          = {verdict}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": "mordred_cached_nb1030",
        "n_unb": n_unb,
        "feature_dim": int(X_unb.shape[1]),
        "outer_seeds": OUTER_SEEDS,
        "inner_seed_base": INNER_SEED_BASE,
        "inner_seed_formula": "inner_seed = outer_seed * 1000 + base_seed",
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "rae_anchor_nb1070_on_253": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_outer_seed_rae": per_outer_rae,
        "per_outer_seed_rae_mean": per_outer_mean,
        "per_outer_seed_rae_median": per_outer_median,
        "per_outer_seed_rae_std": per_outer_std,
        "per_outer_seed_rae_min": per_outer_min,
        "per_outer_seed_rae_max": per_outer_max,
        "outer_bag_of_bags_mean_rae": outer_mean_rae,
        "outer_bag_of_bags_median_rae": outer_median_rae,
        "delta_per_outer_mean_vs_nb1153": delta_vs_nb1153_mean,
        "delta_per_outer_mean_vs_nb1070": delta_vs_nb1070,
        "delta_per_outer_mean_vs_nb1130": delta_vs_nb1130,
        "beats_nb1153": bool(beats_nb1153),
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1130": bool(beats_nb1130),
        "verdict": verdict,
        "outer_records": outer_records,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1130_mean_bag_ref": NB1130_MEAN_BAG_REF,
        "nb1143_bag_median_ref": NB1143_BAG_MEDIAN_REF,
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
        "nb1153_median_bag_ref": NB1153_MEDIAN_BAG_REF,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("rae_anchor_nb1070_on_253",
              "per_outer_seed_rae",
              "per_outer_seed_rae_mean",
              "per_outer_seed_rae_median",
              "per_outer_seed_rae_std",
              "outer_bag_of_bags_mean_rae",
              "outer_bag_of_bags_median_rae",
              "delta_per_outer_mean_vs_nb1153",
              "beats_nb1153", "beats_nb1070", "beats_nb1130",
              "verdict"):
        print(f"  {k}: {res.get(k)}")
