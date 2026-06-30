"""nb1322 -- Quantile-regression LGBM residual calibration on nb1290 anchor.

Hypothesis:
    nb1290 (best_fixed_w blend, 0.35 * nb1190_bob_mean + 0.65 * nb1242_mean_bag,
    pooled RAE 0.5390) was built from LGBM components trained with the Huber
    loss (alpha=1.0).  Huber minimises a smoothed mixture of L1 and L2; the
    Relative-Absolute-Error metric we are scored on is pure L1.  A shallow
    quantile-regression LGBM at the median (q=0.5) targets the L1 conditional
    median directly.  Refitting only the residual correction with objective
    "quantile", alpha=0.5 may extract a few mRAE on top of the Huber-trained
    blend by aligning the loss function with the leaderboard metric.

Protocol per seed s in {0, 1, 7, 42, 137}:
  1. Anchor = nb1290 best_fixed_w blend, recomputed as
         anchor = 0.35 * nb1190_bob_mean_oof + 0.65 * nb1242_mean_bag_oof
     (matches data/processed/nb1290_summary.json -> best_fixed_w 0.5390).
  2. residual = y_unb - anchor
  3. KFold(n=5, shuffle=True, random_state=s) on 253 unblind rows.
  4. Shallow LGBM Quantile (objective="quantile", alpha=0.5, max_depth=2,
     num_leaves=3, n_est=40, lr=0.03, min_child_samples=20) on MACCS-167
     features sliced to the unblind rows.
  5. pred_corrected_s = anchor + residual_oof_s; pooled RAE.

Mean-bag pooled cross-fit RAE = RAE(y_unb, mean over seeds of pred_corr_s).
Verdict at 0.003 margin vs nb1290 (0.5390).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1322_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1322_mean_bag_oof.npy            (253,)   float32
  data/processed/nb1322_median_bag_oof.npy          (253,)   float32
  data/processed/nb1322_summary.json
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

TAG = "nb1322"
ANCHOR = "nb1290"  # virtual anchor = 0.35*nb1190 + 0.65*nb1242

# Anchor recipe (verified against nb1290_summary.json -> best_fixed_w 0.5390).
W_NB1190 = 0.35
W_NB1242 = 0.65
NB1190_OOF_PATH = DATA_PROCESSED / "nb1190_bob_mean_oof.npy"
NB1242_OOF_PATH = DATA_PROCESSED / "nb1242_mean_bag_oof.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"   # (4139, 167) uint8
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"   # (513, 167)  uint8

# Reference numbers (pooled RAE on 253 unblind).
NB1290_REF_POOLED = 0.5390  # best_fixed_w blend
DECISION_MARGIN = 0.003


def _lgbm_quantile_params(seed: int) -> dict:
    """Shallow LGBM Quantile at q=0.5 (median = L1 conditional optimum)."""
    return dict(
        objective="quantile",
        alpha=0.5,
        learning_rate=0.03,
        n_estimators=40,
        max_depth=2,
        num_leaves=3,
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
        mdl = LGBMRegressor(**_lgbm_quantile_params(seed))
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
    X_unb = X_te[unb_idx].astype(np.float32)
    return X_unb


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- QUANTILE-LGBM (q=0.5) residual calibration on nb1290 anchor")
    print(f"          anchor = {W_NB1190:.2f}*nb1190_bob_mean + "
          f"{W_NB1242:.2f}*nb1242_mean_bag")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          LGBM: objective=quantile, alpha=0.5, max_depth=2, "
          f"num_leaves=3, n_est=40, lr=0.03, min_child_samples=20")
    print(f"          features = cached MACCS keys ({MACCS_TE_PATH})")
    print(f"          verdict margin {DECISION_MARGIN} vs "
          f"nb1290 ({NB1290_REF_POOLED:.4f})")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Reconstruct nb1290 anchor as 0.35 nb1190 + 0.65 nb1242 ----
    for p in (NB1190_OOF_PATH, NB1242_OOF_PATH):
        if not p.exists():
            raise FileNotFoundError(f"missing component OOF: {p}")
    p1 = np.load(NB1190_OOF_PATH).astype(np.float64)
    p2 = np.load(NB1242_OOF_PATH).astype(np.float64)
    if p1.shape[0] != n_unb or p2.shape[0] != n_unb:
        raise ValueError(
            f"component shape mismatch: "
            f"nb1190={p1.shape}  nb1242={p2.shape}  n_unb={n_unb}"
        )
    anchor_oof = W_NB1190 * p1 + W_NB1242 * p2
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[anchor] recomputed nb1290 best_fixed_w blend"
          f"  pooled RAE = {rae_anchor:.4f}  "
          f"(ref {NB1290_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ---- Features ----
    print(f"[feat] loading cached MACCS test matrix, slicing to "
          f"{n_unb} unblind rows ...")
    X_unb = _load_maccs_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat] X_unb shape = {X_unb.shape}  (MACCS keys only)")
    print(f"[feat] bit density (unb) = {X_unb.mean():.4f}  "
          f"const cols = {int((X_unb.var(axis=0) == 0).sum())}/{X_unb.shape[1]}")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED QUANTILE-q05 RESIDUAL CROSS-FIT "
          f"(depth=2 ultra-shallow, MACCS {X_unb.shape[1]})")
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
            "delta_vs_nb1290": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1290 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

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
          f"(d_vs_nb1290 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1290 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1290 anchor          = {rae_anchor:.4f}  "
          f"(ref {NB1290_REF_POOLED:.4f})")

    beats_nb1290 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    flat_nb1290 = abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN

    if beats_nb1290:
        verdict = (f"QUANTILE_q05_RESIDUAL_BEATS_NB1290 "
                   f"(mean_bag @ {rae_mean_bag:.4f}, "
                   f"d_vs_nb1290 {rae_mean_bag - rae_anchor:+.4f})")
    elif flat_nb1290:
        verdict = (f"QUANTILE_q05_RESIDUAL_FLAT_VS_NB1290 "
                   f"(mean_bag @ {rae_mean_bag:.4f})")
    else:
        verdict = (f"QUANTILE_q05_RESIDUAL_HURTS_NB1290 "
                   f"(mean_bag @ {rae_mean_bag:.4f})")
    print(f"   verdict                = {verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_recipe": {
            "w_nb1190": W_NB1190,
            "w_nb1242": W_NB1242,
            "nb1190_oof_path": str(NB1190_OOF_PATH),
            "nb1242_oof_path": str(NB1242_OOF_PATH),
        },
        "feature_source": "maccs_cached_167",
        "maccs_cache_train": str(MACCS_TR_PATH),
        "maccs_cache_test": str(MACCS_TE_PATH),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "lgbm_objective": "quantile",
        "lgbm_alpha": 0.5,
        "lgbm_max_depth": 2,
        "lgbm_num_leaves": 3,
        "lgbm_n_estimators": 40,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 20,
        "rae_anchor_nb1290": rae_anchor,
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
        "delta_mean_bag_vs_nb1290": rae_mean_bag - rae_anchor,
        "delta_median_bag_vs_nb1290": rae_median_bag - rae_anchor,
        "beats_nb1290": bool(beats_nb1290),
        "flat_vs_nb1290": bool(flat_nb1290),
        "verdict": verdict,
        "nb1290_ref_pooled": NB1290_REF_POOLED,
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
    for k in ("rae_anchor_nb1290", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_nb1290",
              "delta_median_bag_vs_nb1290",
              "beats_nb1290", "flat_vs_nb1290", "verdict"):
        print(f"  {k}: {res.get(k)}")
