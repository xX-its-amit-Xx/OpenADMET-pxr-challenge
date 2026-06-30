"""nb1231 -- Alt-anchor MACCS BoB residual bag using chemprop_aux as anchor.

Hypothesis:
    nb1183 used nb1070 (per-quantile median bag from nb1014) as anchor; nb1070
    is itself a derived/calibrated prediction so its residual structure is
    highly compressed.  Anchoring directly on chemprop_aux (the raw stronger
    PRE-unblind deploy, in_RAE ~0.6216 -> predicted LB ~0.6246) may reveal
    residual structure the post-hoc-calibrated nb1070 anchor smoothed away.

    chemprop_aux is a full-model deploy (chemprop trained without unblind
    labels).  Its 253-slice is the LB-faithful estimate; we treat the slice as
    the anchor and let the MACCS residual learner correct it via 5-fold
    cross-fit on the 253 unblind rows.

Protocol per seed s in {0, 1, 7, 42, 137}:
  1. Anchor = chemprop_aux[unb_idx] (constant across seeds).
  2. residual = y_unb - chemprop_aux[unb_idx]
  3. KFold(n=5, shuffle=True, random_state=s) on 253 unblind rows.
  4. Shallow LGBM Huber (max_depth=3, num_leaves=7, n_est=80, lr=0.05,
     min_child_samples=20, alpha=1.0) on cached MACCS sliced to unblind.
  5. pred_corrected_s = chemprop_aux[unb] + residual_oof_s; pooled RAE.

Mean-bag pooled cross-fit RAE = RAE(y_unb, mean_seeds pred_corr_s).
Compare to nb1183 (0.5513, same MACCS residual on nb1070 anchor) at 0.003 margin.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1231_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1231_mean_bag_oof.npy            (253,)   float32
  data/processed/nb1231_median_bag_oof.npy          (253,)   float32
  data/processed/nb1231_summary.json
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

TAG = "nb1231"
ANCHOR = "chemprop_aux"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Cached MACCS test matrix (te_maccs166.npy was requested but the on-disk
# cache is te_maccs.npy with 167 cols; we use it as-is -- LGBM will down-weight
# const / dead bits).
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"   # (513, 167)  uint8

# Reference numbers (pooled RAE on 253 unblind).
NB1183_MEAN_BAG_REF = 0.5513   # MACCS residual on nb1070 anchor
CHEMPROP_AUX_REF_INRAE = 0.6216  # documented in_RAE on 253 unblind
DECISION_MARGIN = 0.003


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical capacity to nb1183 MACCS bag."""
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
    X_unb = X_te[unb_idx].astype(np.float32)
    return X_unb


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ALT-ANCHOR: shallow MACCS residual-LGBM bag on "
          f"chemprop_aux anchor, {len(RESID_SEEDS)} KFold seeds")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - chemprop_aux[unb_idx]")
    print(f"          features = cached MACCS keys ({MACCS_TE_PATH})")
    print(f"          LGBM: max_depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_path = DATA_PROCESSED / "te_chemprop_aux.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; required anchor deploy (chemprop_aux)."
        )
    te_anchor = np.load(anchor_path).astype(np.float64)
    if te_anchor.shape[0] != n_test:
        raise ValueError(
            f"{anchor_path} shape mismatch: {te_anchor.shape} "
            f"vs n_test={n_test}"
        )
    anchor_oof = te_anchor[unb_idx]  # 253-slice
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR} te shape={te_anchor.shape}; sliced to "
          f"unblind 253; in_RAE = {rae_anchor:.4f}  "
          f"(doc ref ~{CHEMPROP_AUX_REF_INRAE:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    print(f"[feat] loading cached MACCS test matrix, slicing to "
          f"{n_unb} unblind rows ...")
    X_unb = _load_maccs_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat] X_unb shape = {X_unb.shape}  (MACCS keys only)")
    print(f"[feat] bit density (unb) = {X_unb.mean():.4f}  "
          f"const cols = {int((X_unb.var(axis=0) == 0).sum())}/{X_unb.shape[1]}")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow, MACCS {X_unb.shape[1]})")
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
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
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
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_anchor = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1183 mean_bag ref    = {NB1183_MEAN_BAG_REF:.4f}  "
          f"(MACCS residual on nb1070 anchor)")

    beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_MEAN_BAG_REF - DECISION_MARGIN

    if beats_nb1183:
        verdict = "MACCS_RESID_ON_CHEMPROP_BEATS_NB1183_ALT_ANCHOR_WINS"
    elif abs(rae_mean_bag - NB1183_MEAN_BAG_REF) < DECISION_MARGIN:
        verdict = "MACCS_RESID_ON_CHEMPROP_TIES_NB1183_NO_NEW_SIGNAL"
    elif beats_anchor:
        verdict = "MACCS_RESID_HELPS_CHEMPROP_BUT_LOSES_TO_NB1183"
    else:
        verdict = "MACCS_RESID_FAILS_ON_CHEMPROP_ANCHOR"
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
        "anchor_source": "te_chemprop_aux.npy sliced to unblind 253",
        "feature_source": "maccs_cached_te_maccs.npy",
        "maccs_cache_test": str(MACCS_TE_PATH),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_chemprop_aux_inRAE": rae_anchor,
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
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_median_bag_vs_anchor": rae_median_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_MEAN_BAG_REF,
        "beats_anchor": bool(beats_anchor),
        "beats_nb1183": bool(beats_nb1183),
        "verdict": verdict,
        "nb1183_mean_bag_ref": NB1183_MEAN_BAG_REF,
        "chemprop_aux_ref_inRAE": CHEMPROP_AUX_REF_INRAE,
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
    for k in ("rae_anchor_chemprop_aux_inRAE", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_anchor",
              "delta_mean_bag_vs_nb1183",
              "beats_anchor", "beats_nb1183",
              "verdict"):
        print(f"  {k}: {res.get(k)}")
