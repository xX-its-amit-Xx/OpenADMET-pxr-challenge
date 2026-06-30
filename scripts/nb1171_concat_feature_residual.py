"""nb1171 -- Concatenated-feature residual: one shallow LGBM on
[Morgan+RDKit, Mordred] concat over nb1070 anchor.

Hypothesis:
    nb1130 fits a shallow residual LGBM on Morgan+RDKit (2265) over the
    nb1070 anchor; nb1153 fits the same shape on Mordred (1533).  Both
    operate at ~-0.012 RAE individually and a SLSQP blend extracts
    further gain.  But each model only sees its own feature manifold --
    cross-feature interactions (e.g. Mordred-Burden x ECFP4-bit) are
    invisible to either standalone.

    A single shallow LGBM Huber on the FULL concat (2265 + 1533 = 3798)
    has access to those interactions inside one tree.  If the cross-
    feature signal is real and not noise at n=253, the concat model
    should beat the blend ceiling (nb1153 0.5640 reference, nb1130 0.5673
    reference).  If the concat model is flat or worse, then the signal
    available at this anchor is fully captured by the separate-model
    blend and the interactions are noise at this sample size.

Protocol per seed s in {0, 1, 7, 42, 137}:
  1. Anchor = nb1070 pred_oof (constant across seeds).
  2. residual = y_unb - nb1070_oof
  3. KFold(n=5, shuffle=True, random_state=s) on 253 unblind.
  4. Shallow LGBM Huber (max_depth=3, num_leaves=7, n_est=80, lr=0.05,
     min_child_samples=20, alpha=1.0) on concat features
     [Morgan+RDKit (2265), Mordred (1533)] = (253, 3798); cross-fit
     residual OOF.
  5. pred_corrected_s = nb1070_oof + residual_oof_s; pooled RAE.

Then mean-bag pooled cross-fit RAE = RAE(y_unb, mean over seeds of pred_corr_s).
Verdict @ 0.003 margin vs nb1153 (0.5640).

Outputs:
  data/processed/nb1171_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1171_mean_bag_oof.npy            (253,) float32
  data/processed/nb1171_median_bag_oof.npy          (253,) float32
  data/processed/nb1171_summary.json
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

TAG = "nb1171"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
COMBINED_CACHE = DATA_PROCESSED / "cache_combined_features.npz"

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5790
NB1130_MEAN_BAG_REF = 0.5673   # Morgan+RDKit residual on nb1070
NB1153_MEAN_BAG_REF = 0.5640   # Mordred residual on nb1070
MARGIN = 0.003


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical capacity to nb1130 / nb1153."""
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


def _median_impute(X: np.ndarray) -> np.ndarray:
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_combined_unblind(n_test_expected: int, unb_idx: np.ndarray) -> np.ndarray:
    """Load cached Morgan+RDKit (513 x 2265) and slice to unblind rows."""
    if not COMBINED_CACHE.exists():
        raise FileNotFoundError(
            f"Combined feature cache missing -- {COMBINED_CACHE}"
        )
    z = np.load(COMBINED_CACHE)
    X_te_c = z["X_te"].astype(np.float32)
    if X_te_c.shape[0] != n_test_expected:
        raise ValueError(
            f"Combined test shape mismatch: {X_te_c.shape} vs "
            f"n_test={n_test_expected}"
        )
    X_te_c = _median_impute(X_te_c)
    return X_te_c[unb_idx]


def _load_mordred_unblind(n_test_expected: int, unb_idx: np.ndarray) -> np.ndarray:
    """Load cached Mordred (513 x 1533) and slice to unblind rows."""
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}"
        )
    X_te_m = _median_impute(X_te_m)
    return X_te_m[unb_idx]


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CONCAT-feature residual LGBM on nb1070 anchor")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          features = Morgan+RDKit (2265) || Mordred (1533) = 3798")
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
            f"{anchor_path} not found; required anchor OOF "
            f"(run nb1070 first)."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: "
            f"{anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    print(f"[feat] loading Morgan+RDKit cache + slicing to unblind ...")
    X_comb = _load_combined_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat]   Morgan+RDKit unblind shape = {X_comb.shape}")
    print(f"[feat] loading Mordred cache + slicing to unblind ...")
    X_mord = _load_mordred_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat]   Mordred       unblind shape = {X_mord.shape}")

    X_unb = np.concatenate([X_comb, X_mord], axis=1).astype(np.float32)
    print(f"[feat] CONCAT X_unb shape = {X_unb.shape}  "
          f"({X_comb.shape[1]} + {X_mord.shape[1]} = {X_unb.shape[1]})")

    # Final safety re-impute across the concat (covers any cross-cache
    # residual NaNs).
    if not np.all(np.isfinite(X_unb)):
        X_unb = _median_impute(X_unb)
        print("[feat] CONCAT had non-finite cells -> median-imputed")

    print("\n" + "-" * 78)
    print("PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow, CONCAT 3798)")
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
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1130 mean_bag ref    = {NB1130_MEAN_BAG_REF:.4f}  "
          f"(Morgan+RDKit residual)")
    print(f"   nb1153 mean_bag ref    = {NB1153_MEAN_BAG_REF:.4f}  "
          f"(Mordred-only residual)")

    beats_nb1070 = rae_mean_bag < rae_anchor - MARGIN
    beats_nb1130 = rae_mean_bag < NB1130_MEAN_BAG_REF - MARGIN
    beats_nb1153 = rae_mean_bag < NB1153_MEAN_BAG_REF - MARGIN

    if beats_nb1153:
        verdict = "CONCAT_BEATS_NB1153_CROSS_FEATURE_INTERACTIONS_REAL"
    elif beats_nb1130:
        verdict = "CONCAT_BEATS_NB1130_BUT_TIES_NB1153"
    elif beats_nb1070:
        verdict = "CONCAT_HELPS_NB1070_BUT_NOT_NB1153"
    elif abs(rae_mean_bag - NB1153_MEAN_BAG_REF) < MARGIN:
        verdict = "CONCAT_TIES_NB1153_NO_NEW_CROSS_FEATURE_SIGNAL"
    else:
        verdict = "CONCAT_HURTS_NB1153"
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
        "feature_source": "concat_morgan_rdkit_2265_plus_mordred_1533",
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "feature_dim_combined": int(X_comb.shape[1]),
        "feature_dim_mordred": int(X_mord.shape[1]),
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_alpha_huber": 1.0,
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
        "delta_mean_bag_vs_nb1130": rae_mean_bag - NB1130_MEAN_BAG_REF,
        "delta_mean_bag_vs_nb1153": rae_mean_bag - NB1153_MEAN_BAG_REF,
        "margin": MARGIN,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1130": bool(beats_nb1130),
        "beats_nb1153": bool(beats_nb1153),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1130_mean_bag_ref": NB1130_MEAN_BAG_REF,
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
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
    for k in ("rae_anchor_nb1070", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_nb1070",
              "delta_mean_bag_vs_nb1130",
              "delta_mean_bag_vs_nb1153",
              "beats_nb1070", "beats_nb1130", "beats_nb1153", "verdict"):
        print(f"  {k}: {res.get(k)}")
