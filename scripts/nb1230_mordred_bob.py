"""nb1230 -- Outer-bag VALIDATION of nb1153 (MORDRED-1533 residual standalone),
parallel in spirit to nb1190 (triple-FP BoB) and nb1200 (MACCS BoB).

HYPOTHESIS
----------
Single-bag nb1153 cross-fits to mean-bag pooled RAE 0.5640 -- too weak to add
to the nb1211 (= nb1190 BoB mean + nb1200 BoB mean) naive-mean blend (0.5451).
But the FULL outer-bag -- 5 outer KFold seeds x 5 inner KFold seeds = 25 LGBM
fits -- should yield a more stable Mordred-only ensemble at ~0.555-0.560 RAE.
If that ensemble's residual is *orthogonal* enough to the nb1190 (Morgan +
RDKit + Mordred + AtomPair triple-mean) and nb1200 (MACCS-167) BoBs, the
triple-BoB naive mean (nb1190 + nb1200 + nb1230) may beat the nb1211 pair-mean
(0.5451) at the 0.003 margin.

NOTE: nb1153 (Mordred-only residual) is already a *subset* of the nb1190
triple-mean components -- nb1190's third component IS nb1153 at outer 0.
So the orthogonality question is whether *upweighting* Mordred (1x in nb1230
vs 1/3 weight inside nb1190) adds net signal or just doubles the same axis.
This script measures it directly.

PROTOCOL
--------
For each OUTER seed o in {0, 1, 7, 42, 137}:
  inner_seeds = [o * 1000 + s for s in {0, 1, 7, 42, 137}]
    => outer 0 -> inner [   0,    1,    7,   42,  137]  (reproduces nb1153 exactly)
       outer 1 -> inner [1000, 1001, 1007, 1042, 1137]
       outer 7 -> inner [7000, 7001, 7007, 7042, 7137]
       outer 42 -> inner [42000,42001,42007,42042,42137]
       outer 137 -> inner [137000,137001,137007,137042,137137]
  Rebuild Mordred-only residual mean-bag on the 253 unblind with those 5
  inner seeds, identical capacity to nb1153:
    shallow LGBM Huber (depth=3, leaves=7, n_est=80, lr=0.05,
    min_child_samples=20, alpha=1.0) on residual = y_unb - nb1070_pred_oof,
    5-fold KFold cross-fit per inner seed, mean over inner seeds.
  The o-th outer measurement = pooled RAE(y_unb, mean_bag_o).

SANITY CHECK
------------
Outer seed 0 must reproduce nb1153's 0.5640 within rounding (uses the same
inner seeds nb1153 used internally).

REPORT
------
Per-outer RAE list, mean, std, min, max.  Row-level bag-of-bags MEAN and
MEDIAN across the 5 outer-seed predictions, plus pooled RAE.  Verdict
NB1230_REPRODUCES_SINGLE_BAG if per-outer mean is within 0.003 of 0.5640.

THEN: 3-way naive mean blend of (nb1190_bob_mean + nb1200_bob_mean +
nb1230_bob_mean).  Pool RAE.  Verdict at 0.003 margin vs nb1211 naive mean
(0.5451).  Also try inverse-RAE-weighted blend.

Outputs:
  data/processed/nb1230_per_outer_oof.npy        (5, 253) float32  mean-bag per outer seed
  data/processed/nb1230_bob_mean_oof.npy         (253,)   float32  bag-of-bags mean
  data/processed/nb1230_bob_median_oof.npy       (253,)   float32  bag-of-bags median
  data/processed/nb1230_triple_bob_mean_oof.npy  (253,)   float32  triple-BoB naive mean
  data/processed/nb1230_triple_bob_invrae_oof.npy (253,)  float32  triple-BoB inverse-RAE weighted
  data/processed/nb1230_summary.json
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

TAG = "nb1230"
ANCHOR = "nb1070"

# Outer / inner seed grid.
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASES = [0, 1, 7, 42, 137]   # inner_seed = outer * 1000 + base
RESID_FOLDS = 5

# Mordred cached features (nb1153 used these).
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
MORDRED_TE_PATH = MORDRED_DIR / "X_mordred_test.npy"   # (513, 1533) float32

# Reference (nb1153 mean_bag pooled RAE on 253 unblind).
NB1153_MEAN_BAG_REF = 0.5640
REPRO_MARGIN = 0.003

# Triple-BoB blend reference (nb1211 naive mean of nb1190_mean + nb1200_mean).
NB1211_MEAN_REF = 0.5451
TRIPLE_BLEND_MARGIN = 0.003


# -----------------------------------------------------------------------------
# Residual cross-fit primitive (identical capacity to nb1153).
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


def _load_mordred_unblind(n_test_expected: int, unb_idx: np.ndarray) -> np.ndarray:
    """Load cached Mordred test matrix (513 x 1533), slice unblind, median-impute NaNs/Infs."""
    if not MORDRED_TE_PATH.exists():
        raise FileNotFoundError(
            f"Mordred test cache missing: {MORDRED_TE_PATH} (run nb1030 first)"
        )
    X_te = np.load(MORDRED_TE_PATH).astype(np.float32)
    if X_te.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test cache shape mismatch: {X_te.shape} "
            f"vs n_test={n_test_expected}"
        )
    if X_te.shape[1] != 1533:
        raise ValueError(
            f"Mordred test cache unexpected width: {X_te.shape[1]} (expected 1533)"
        )
    # Median-impute any non-finite values column-wise (matches nb1153 behavior).
    X_te = np.where(np.isfinite(X_te), X_te, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te[idx_r, idx_c] = col_med[idx_c]
    return X_te[unb_idx]


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG validation of nb1153 (MORDRED-1533 residual mean-bag) "
          f"under {len(OUTER_SEEDS)} outer seeds")
    print(f"          outer seeds = {OUTER_SEEDS}")
    print(f"          inner bases = {INNER_BASES}  (inner = outer*1000 + base)")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          feature  = Mordred-1533 cached ({MORDRED_TE_PATH})")
    print(f"          LGBM     = depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print(f"          reference (nb1153 mean_bag) = {NB1153_MEAN_BAG_REF:.4f}"
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
    print(f"[feat] loading cached Mordred-1533 for {n_unb} unblind ...")
    X_unb = _load_mordred_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat] X_unb shape = {X_unb.shape}")
    print(f"[feat] const cols = "
          f"{int((X_unb.var(axis=0) == 0).sum())}/{X_unb.shape[1]}")

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
        # outer 0 -> [0, 1, 7, 42, 137] reproduces nb1153 exactly.

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

        # Did the outer-seed=0 case reproduce nb1153?
        nb1153_match_note = ""
        if o == 0:
            d = rae_mean_bag_o - NB1153_MEAN_BAG_REF
            nb1153_match_note = (
                f"  [REPRO_CHECK outer=0 vs nb1153 {NB1153_MEAN_BAG_REF:.4f}: "
                f"d = {d:+.4f}, |d|<0.003? {abs(d) < 0.003}]"
            )

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "per_inner_rae": per_inner_rae,
            "rae_mean_bag": rae_mean_bag_o,
            "delta_vs_nb1153_ref": rae_mean_bag_o - NB1153_MEAN_BAG_REF,
            "elapsed_sec": round(time.time() - t_outer, 1),
        })
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     per_inner_rae = "
              f"[{', '.join(f'{r:.4f}' for r in per_inner_rae)}]")
        print(f"     mean_bag_RAE  = {rae_mean_bag_o:.4f}  "
              f"(elapsed {time.time() - t_outer:.1f}s){nb1153_match_note}")

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

    # Verdict on reproduction.
    reproduces = abs(outer_mean - NB1153_MEAN_BAG_REF) <= REPRO_MARGIN
    outer0 = float(per_outer_rae_arr[0])
    outer0_reproduces = abs(outer0 - NB1153_MEAN_BAG_REF) <= REPRO_MARGIN

    if reproduces:
        repro_verdict = "NB1230_REPRODUCES_SINGLE_BAG"
    elif outer_mean < NB1153_MEAN_BAG_REF - REPRO_MARGIN:
        repro_verdict = "NB1153_PESSIMISTIC_OUTER_BAG_BETTER"
    else:
        repro_verdict = "NB1153_LUCKY_OUTER_BAG_PULLS_UP"

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
    print(f"   nb1153 mean_bag reference       = {NB1153_MEAN_BAG_REF:.4f}")
    print(f"   delta(per-outer mean vs nb1153) = "
          f"{outer_mean - NB1153_MEAN_BAG_REF:+.4f}  (margin {REPRO_MARGIN:.3f})")
    print(f"   outer-seed=0 reproduces?  {outer0_reproduces}  "
          f"(outer0 RAE = {outer0:.4f}, d = {outer0 - NB1153_MEAN_BAG_REF:+.4f})")
    print(f"   REPRO VERDICT = {repro_verdict}")

    # ---- Save BoB artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_oof.npy",
            per_outer_mean_bag.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_outer_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    # ---- TRIPLE-BoB blend: nb1190_mean + nb1200_mean + nb1230_mean ----
    print("\n" + "=" * 78)
    print("TRIPLE-BoB BLEND  (nb1190_bob_mean + nb1200_bob_mean + nb1230_bob_mean)")
    print("=" * 78)

    nb1190_path = DATA_PROCESSED / "nb1190_bob_mean_oof.npy"
    nb1200_path = DATA_PROCESSED / "nb1200_bob_mean_oof.npy"
    if not nb1190_path.exists():
        raise FileNotFoundError(f"{nb1190_path} missing")
    if not nb1200_path.exists():
        raise FileNotFoundError(f"{nb1200_path} missing")
    nb1190_oof = np.load(nb1190_path).astype(np.float64)
    nb1200_oof = np.load(nb1200_path).astype(np.float64)
    if nb1190_oof.shape[0] != n_unb or nb1200_oof.shape[0] != n_unb:
        raise ValueError(
            f"nb1190/nb1200 shape mismatch: "
            f"nb1190={nb1190_oof.shape}, nb1200={nb1200_oof.shape}, n_unb={n_unb}"
        )

    rae_nb1190 = float(rae(y_unb, nb1190_oof))
    rae_nb1200 = float(rae(y_unb, nb1200_oof))
    rae_nb1230 = rae_bob_mean

    print(f"   standalone RAE:")
    print(f"     nb1190 BoB mean : {rae_nb1190:.4f}")
    print(f"     nb1200 BoB mean : {rae_nb1200:.4f}")
    print(f"     nb1230 BoB mean : {rae_nb1230:.4f}  (this script)")

    # Pearson diagnostics.
    pearson_1190_1230 = float(np.corrcoef(nb1190_oof, bob_mean_oof)[0, 1])
    pearson_1200_1230 = float(np.corrcoef(nb1200_oof, bob_mean_oof)[0, 1])
    pearson_1190_1200 = float(np.corrcoef(nb1190_oof, nb1200_oof)[0, 1])
    resid_corr_1190_1230 = float(
        np.corrcoef(nb1190_oof - y_unb, bob_mean_oof - y_unb)[0, 1]
    )
    resid_corr_1200_1230 = float(
        np.corrcoef(nb1200_oof - y_unb, bob_mean_oof - y_unb)[0, 1]
    )
    resid_corr_1190_1200 = float(
        np.corrcoef(nb1190_oof - y_unb, nb1200_oof - y_unb)[0, 1]
    )
    print(f"   pred Pearson:")
    print(f"     (nb1190, nb1200) = {pearson_1190_1200:.4f}")
    print(f"     (nb1190, nb1230) = {pearson_1190_1230:.4f}")
    print(f"     (nb1200, nb1230) = {pearson_1200_1230:.4f}")
    print(f"   resid Pearson:")
    print(f"     (nb1190, nb1200) = {resid_corr_1190_1200:.4f}")
    print(f"     (nb1190, nb1230) = {resid_corr_1190_1230:.4f}")
    print(f"     (nb1200, nb1230) = {resid_corr_1200_1230:.4f}")

    # 3-way naive mean blend.
    triple_mean_oof = (nb1190_oof + nb1200_oof + bob_mean_oof) / 3.0
    rae_triple_mean = float(rae(y_unb, triple_mean_oof))

    # Inverse-RAE weighted blend.
    inv_raes = np.array([1.0 / rae_nb1190, 1.0 / rae_nb1200, 1.0 / rae_nb1230])
    inv_weights = inv_raes / inv_raes.sum()
    triple_invrae_oof = (
        inv_weights[0] * nb1190_oof
        + inv_weights[1] * nb1200_oof
        + inv_weights[2] * bob_mean_oof
    )
    rae_triple_invrae = float(rae(y_unb, triple_invrae_oof))

    print(f"\n   3-way naive mean       RAE = {rae_triple_mean:.4f}  "
          f"(d vs nb1211 mean {NB1211_MEAN_REF:.4f} = "
          f"{rae_triple_mean - NB1211_MEAN_REF:+.4f})")
    print(f"   3-way inverse-RAE      RAE = {rae_triple_invrae:.4f}  "
          f"(weights = [nb1190={inv_weights[0]:.4f}, "
          f"nb1200={inv_weights[1]:.4f}, nb1230={inv_weights[2]:.4f}])")
    print(f"     (d vs nb1211 mean {NB1211_MEAN_REF:.4f} = "
          f"{rae_triple_invrae - NB1211_MEAN_REF:+.4f})")

    best_triple_rae = min(rae_triple_mean, rae_triple_invrae)
    best_triple_tag = (
        "naive_mean" if rae_triple_mean <= rae_triple_invrae else "inverse_rae"
    )
    beats_nb1211 = best_triple_rae < NB1211_MEAN_REF - TRIPLE_BLEND_MARGIN

    if beats_nb1211:
        triple_verdict = (
            f"TRIPLE_BOB_BEATS_NB1211 ({best_triple_tag} @ {best_triple_rae:.4f})"
        )
    elif abs(best_triple_rae - NB1211_MEAN_REF) < TRIPLE_BLEND_MARGIN:
        triple_verdict = (
            f"TRIPLE_BOB_FLAT_VS_NB1211 ({best_triple_tag} @ {best_triple_rae:.4f})"
        )
    else:
        triple_verdict = (
            f"TRIPLE_BOB_HURTS_VS_NB1211 ({best_triple_tag} @ {best_triple_rae:.4f})"
        )
    print(f"\n   best triple-BoB     = {best_triple_rae:.4f}  ({best_triple_tag})")
    print(f"   nb1211 mean ref     = {NB1211_MEAN_REF:.4f}")
    print(f"   beats nb1211 (>=0.003)? {beats_nb1211}")
    print(f"   TRIPLE-BoB VERDICT = {triple_verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_triple_bob_mean_oof.npy",
            triple_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_triple_bob_invrae_oof.npy",
            triple_invrae_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_triple_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_triple_bob_invrae_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": "mordred_cached_nb1030",
        "mordred_cache_test": str(MORDRED_TE_PATH),
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
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
        "delta_outer_mean_vs_nb1153": outer_mean - NB1153_MEAN_BAG_REF,
        "delta_outer0_vs_nb1153": outer0 - NB1153_MEAN_BAG_REF,
        "outer0_reproduces": bool(outer0_reproduces),
        "repro_margin": REPRO_MARGIN,
        "reproduces": bool(reproduces),
        "repro_verdict": repro_verdict,
        # Triple-BoB blend.
        "nb1190_bob_mean_rae": rae_nb1190,
        "nb1200_bob_mean_rae": rae_nb1200,
        "nb1230_bob_mean_rae": rae_nb1230,
        "pearson_pred_1190_1200": pearson_1190_1200,
        "pearson_pred_1190_1230": pearson_1190_1230,
        "pearson_pred_1200_1230": pearson_1200_1230,
        "pearson_resid_1190_1200": resid_corr_1190_1200,
        "pearson_resid_1190_1230": resid_corr_1190_1230,
        "pearson_resid_1200_1230": resid_corr_1200_1230,
        "triple_naive_mean_rae": rae_triple_mean,
        "triple_inverse_rae_rae": rae_triple_invrae,
        "triple_inverse_rae_weights": [float(x) for x in inv_weights],
        "best_triple_rae": best_triple_rae,
        "best_triple_tag": best_triple_tag,
        "nb1211_mean_ref": NB1211_MEAN_REF,
        "triple_blend_margin": TRIPLE_BLEND_MARGIN,
        "delta_best_triple_vs_nb1211": best_triple_rae - NB1211_MEAN_REF,
        "beats_nb1211": bool(beats_nb1211),
        "triple_verdict": triple_verdict,
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
              "delta_outer_mean_vs_nb1153", "delta_outer0_vs_nb1153",
              "outer0_reproduces", "reproduces", "repro_verdict",
              "nb1190_bob_mean_rae", "nb1200_bob_mean_rae",
              "nb1230_bob_mean_rae",
              "triple_naive_mean_rae", "triple_inverse_rae_rae",
              "best_triple_rae", "best_triple_tag",
              "delta_best_triple_vs_nb1211",
              "beats_nb1211", "triple_verdict"):
        print(f"  {k}: {res.get(k)}")
