"""nb1200 -- Outer-bag VALIDATION of nb1183 (MACCS-167 residual standalone).

PRECEDENT
---------
nb1143 (0.5649) single-seed bag looked promising but was stress-tested by
nb1151 outer-bag and degraded to 0.5705 (lucky).  nb1181 triple naive-mean
0.5566 was stress-tested by nb1190 and was outer-bag better (pessimistic).
nb1183 MACCS-167 residual bag cross-fits to mean-bag pooled RAE 0.5513 -- this
script applies the same outer-bag stress test to verify the answer isn't a
KFold-seed artifact.

PROTOCOL
--------
For each OUTER seed o in {0, 1, 7, 42, 137}:
  inner_seeds(o) = [o * 1000 + s for s in {0, 1, 7, 42, 137}]
    => outer 0    -> inner [   0,    1,    7,   42,  137]   (reproduces nb1183 exactly)
       outer 1    -> inner [1000, 1001, 1007, 1042, 1137]
       outer 7    -> inner [7000, 7001, 7007, 7042, 7137]
       outer 42   -> inner [42000,42001,42007,42042,42137]
       outer 137  -> inner [137000,137001,137007,137042,137137]
  Rebuild the MACCS-167 residual mean-bag on the 253 unblind with those
  5 inner seeds, identical capacity to nb1183:
    shallow LGBM Huber (depth=3, leaves=7, n_est=80, lr=0.05,
    min_child_samples=20, alpha=1.0) on residual = y_unb - nb1070_pred_oof,
    5-fold KFold cross-fit per inner seed, mean over inner seeds.
  The o-th outer measurement = pooled RAE(y_unb, mean_bag_o).

SANITY CHECK
------------
Outer seed 0 must reproduce nb1183's 0.5513 within rounding (uses the same
inner seeds nb1183 used internally).

REPORT
------
Per-outer RAE list, mean, std, min, max.  Row-level bag-of-bags MEAN and
MEDIAN across the 5 outer-seed predictions, plus their pooled RAE.  Verdict
NB1183_REPRODUCES if per-outer mean is within 0.003 of 0.5513.

Outputs:
  data/processed/nb1200_per_outer_oof.npy    (5, 253) float32  mean-bag per outer seed
  data/processed/nb1200_bob_mean_oof.npy     (253,)   float32  bag-of-bags mean
  data/processed/nb1200_bob_median_oof.npy   (253,)   float32  bag-of-bags median
  data/processed/nb1200_summary.json
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

TAG = "nb1200"
ANCHOR = "nb1070"

# Outer / inner seed grid.
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASES = [0, 1, 7, 42, 137]   # inner_seed = outer * 1000 + base
RESID_FOLDS = 5

# MACCS cached features (nb1183 used these).
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"   # (4139, 167) uint8
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"   # (513,  167) uint8

# Reference (nb1183 mean_bag pooled RAE on 253 unblind).
NB1183_MEAN_BAG_REF = 0.5513
REPRO_MARGIN = 0.003


# -----------------------------------------------------------------------------
# Residual cross-fit primitive (identical capacity to nb1183).
# -----------------------------------------------------------------------------
def _lgbm_params(seed: int) -> dict:
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


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_maccs_unblind(n_test_expected: int, unb_idx: np.ndarray) -> np.ndarray:
    if not MACCS_TE_PATH.exists():
        raise FileNotFoundError(
            f"MACCS test cache missing: {MACCS_TE_PATH}"
        )
    X_te = np.load(MACCS_TE_PATH)
    if X_te.shape[0] != n_test_expected:
        raise ValueError(
            f"MACCS test cache shape mismatch: {X_te.shape} "
            f"vs n_test={n_test_expected}"
        )
    if X_te.shape[1] not in (166, 167):
        raise ValueError(
            f"MACCS test cache unexpected width: {X_te.shape[1]} "
            f"(expected 166 or 167)"
        )
    return X_te[unb_idx].astype(np.float32)


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG validation of nb1183 (MACCS-167 residual mean-bag) "
          f"under {len(OUTER_SEEDS)} outer seeds")
    print(f"          outer seeds = {OUTER_SEEDS}")
    print(f"          inner bases = {INNER_BASES}  (inner = outer*1000 + base)")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          feature  = MACCS-167 cached ({MACCS_TE_PATH})")
    print(f"          LGBM     = depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print(f"          reference (nb1183 mean_bag) = {NB1183_MEAN_BAG_REF:.4f}"
          f"   margin = {REPRO_MARGIN:.3f}")
    print("=" * 78)

    # ---- Load truth + anchor + features (once) ----
    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(f"{anchor_path} missing")
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: {anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    residual = y_unb - anchor_oof
    print(f"[load] {ANCHOR}_pred_oof.npy  pooled RAE = {rae_anchor:.4f}")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # Features (load once).
    print(f"[feat] loading cached MACCS-167 for {n_unb} unblind ...")
    X_unb = _load_maccs_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat] X_unb shape = {X_unb.shape}")
    print(f"[feat] bit density = {X_unb.mean():.4f}  "
          f"const cols = {int((X_unb.var(axis=0) == 0).sum())}/{X_unb.shape[1]}")

    # ---- Per-outer-seed rebuild ----
    print("\n" + "-" * 78)
    print("PER-OUTER REBUILD  (each: 1 component x 5 inner seeds x 5 folds = "
          f"{len(INNER_BASES) * RESID_FOLDS} LGBM fits)")
    print("-" * 78)

    per_outer_mean_bag = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_rae: list[float] = []
    per_outer_records: list[dict] = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 1000 + int(b) for b in INNER_BASES]
        # outer 0 -> [0, 1, 7, 42, 137] reproduces nb1183 exactly.

        bag = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        per_inner_rae: list[float] = []
        for j, s in enumerate(inner_seeds):
            resid_oof = _residual_cross_fit_one_seed(X_unb, residual, s)
            pred_corr_s = anchor_oof + resid_oof
            bag[j] = pred_corr_s
            per_inner_rae.append(float(rae(y_unb, pred_corr_s)))
        mean_bag_o = bag.mean(axis=0)
        rae_mean_bag_o = float(rae(y_unb, mean_bag_o))
        per_outer_mean_bag[oi] = mean_bag_o
        per_outer_rae.append(rae_mean_bag_o)

        # Did the outer-seed=0 case reproduce nb1183?
        nb1183_match_note = ""
        if o == 0:
            d = rae_mean_bag_o - NB1183_MEAN_BAG_REF
            nb1183_match_note = (
                f"  [REPRO_CHECK outer=0 vs nb1183 0.5513: d = {d:+.4f}, "
                f"|d|<0.003? {abs(d) < 0.003}]"
            )

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "per_inner_rae": per_inner_rae,
            "rae_mean_bag": rae_mean_bag_o,
            "delta_vs_nb1183_ref": rae_mean_bag_o - NB1183_MEAN_BAG_REF,
            "elapsed_sec": round(time.time() - t_outer, 1),
        })
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     per_inner_rae = "
              f"[{', '.join(f'{r:.4f}' for r in per_inner_rae)}]")
        print(f"     mean_bag_RAE  = {rae_mean_bag_o:.4f}  "
              f"(elapsed {time.time() - t_outer:.1f}s){nb1183_match_note}")

    # ---- Aggregate across outer seeds ----
    per_outer_rae_arr = np.array(per_outer_rae)
    outer_mean = float(per_outer_rae_arr.mean())
    outer_std = float(per_outer_rae_arr.std())
    outer_min = float(per_outer_rae_arr.min())
    outer_max = float(per_outer_rae_arr.max())

    # Row-level bag-of-bags: mean and median across outer-seed predictions.
    bob_mean_oof = per_outer_mean_bag.mean(axis=0)
    bob_median_oof = np.median(per_outer_mean_bag, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # Verdict.
    reproduces = abs(outer_mean - NB1183_MEAN_BAG_REF) <= REPRO_MARGIN
    outer0 = float(per_outer_rae_arr[0])
    outer0_reproduces = abs(outer0 - NB1183_MEAN_BAG_REF) <= REPRO_MARGIN

    if reproduces:
        verdict = "NB1183_REPRODUCES"
    elif outer_mean < NB1183_MEAN_BAG_REF - REPRO_MARGIN:
        verdict = "NB1183_PESSIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1183_LUCKY_OUTER_BAG_PULLS_UP"

    print("\n" + "=" * 78)
    print("OUTER-BAG AGGREGATIONS")
    print("=" * 78)
    print(f"   per-outer mean-bag RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae)}]")
    print(f"   per-outer mean   = {outer_mean:.4f}")
    print(f"   per-outer std    = {outer_std:.4f}")
    print(f"   per-outer min    = {outer_min:.4f}")
    print(f"   per-outer max    = {outer_max:.4f}")
    print(f"   bag-of-bags MEAN   row-level RAE = {rae_bob_mean:.4f}")
    print(f"   bag-of-bags MEDIAN row-level RAE = {rae_bob_median:.4f}")
    print(f"   nb1183 mean_bag reference       = {NB1183_MEAN_BAG_REF:.4f}")
    print(f"   delta(per-outer mean vs nb1183) = "
          f"{outer_mean - NB1183_MEAN_BAG_REF:+.4f}  (margin {REPRO_MARGIN:.3f})")
    print(f"   outer-seed=0 reproduces?  {outer0_reproduces}  "
          f"(outer0 RAE = {outer0:.4f}, d = {outer0 - NB1183_MEAN_BAG_REF:+.4f})")
    print(f"   VERDICT = {verdict}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_oof.npy",
            per_outer_mean_bag.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_outer_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": "maccs_cached_167",
        "maccs_cache_train": str(MACCS_TR_PATH),
        "maccs_cache_test": str(MACCS_TE_PATH),
        "n_unb": n_unb,
        "outer_seeds": OUTER_SEEDS,
        "inner_bases": INNER_BASES,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_outer_records": per_outer_records,
        "per_outer_rae": [float(x) for x in per_outer_rae],
        "outer_mean": outer_mean,
        "outer_std": outer_std,
        "outer_min": outer_min,
        "outer_max": outer_max,
        "rae_bag_of_bags_mean": rae_bob_mean,
        "rae_bag_of_bags_median": rae_bob_median,
        "nb1183_mean_bag_ref": NB1183_MEAN_BAG_REF,
        "delta_outer_mean_vs_nb1183": outer_mean - NB1183_MEAN_BAG_REF,
        "delta_outer0_vs_nb1183": outer0 - NB1183_MEAN_BAG_REF,
        "outer0_reproduces": bool(outer0_reproduces),
        "repro_margin": REPRO_MARGIN,
        "reproduces": bool(reproduces),
        "verdict": verdict,
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
    for k in ("rae_anchor_nb1070", "per_outer_rae",
              "outer_mean", "outer_std", "outer_min", "outer_max",
              "rae_bag_of_bags_mean", "rae_bag_of_bags_median",
              "delta_outer_mean_vs_nb1183", "delta_outer0_vs_nb1183",
              "outer0_reproduces", "reproduces", "verdict"):
        print(f"  {k}: {res.get(k)}")
