"""nb1205 -- Concat-feature single residual on [MACCS, AtomPair, Mordred]
(166 + 2048 + 1533 = 3747) over nb1070 anchor.

Hypothesis:
    A single shallow LGBM with the concatenation of a curated substructure
    dictionary (MACCS-166), a 2-tuple-distance topological fingerprint
    (AtomPair-2048), and a continuous-valued descriptor family (Mordred-1533)
    may learn 3-way cross-feature splits (e.g., "if MACCS-bit-X = 1 AND
    AtomPair-bit-Y = 1 AND Mordred-feature-Z high") that no pairwise blend
    can capture. The combined input dimension is 3747; with a depth=3 tree
    and 80 estimators the model has roughly 80 splits, plenty for sparse
    cross-feature corrections if they exist.

Reference numbers (pooled RAE on 253 unblind):
    nb1070 anchor        ~ 0.5771
    nb1183 (MACCS-only)  = 0.5513  (current best single-family residual)
    nb1192 (mean ref)    = 0.5514

Verdict @ 0.003 margin vs nb1183 AND vs nb1192.

Protocol per seed s in {0, 1, 7, 42, 137}:
  1. Anchor = nb1070 pred_oof (constant across seeds).
  2. residual = y_unb - nb1070_oof
  3. KFold(n=5, shuffle=True, random_state=s) on 253 unblind rows.
  4. Shallow LGBM Huber (max_depth=3, num_leaves=7, n_est=80, lr=0.05,
     min_child_samples=20, alpha=1.0) on concat features
     [MACCS-166, AtomPair-2048, Mordred-1533] = (253, 3747); cross-fit
     residual OOF.
  5. pred_corrected_s = nb1070_oof + residual_oof_s; pooled RAE.

Then mean-bag pooled cross-fit RAE = RAE(y_unb, mean over seeds of pred_corr_s).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1205_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1205_mean_bag_oof.npy            (253,) float32
  data/processed/nb1205_median_bag_oof.npy          (253,) float32
  data/processed/nb1205_summary.json
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

TAG = "nb1205"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Feature caches
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"   # (513, 167) uint8
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"   # (513, 2048) uint8
ATOMPAIR_TE_2048_PATH = DATA_PROCESSED / "te_atompair2048.npy"  # alt
MORDRED_TE_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_test.npy")  # (513, 1533) f32

MACCS_DIM = 166        # spec wants 166 (drop bit 0 padding)
ATOMPAIR_DIM = 2048
MORDRED_DIM = 1533
CONCAT_DIM = MACCS_DIM + ATOMPAIR_DIM + MORDRED_DIM   # 3747

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5771
NB1183_REF_POOLED = 0.5513   # MACCS-only residual bag (current best)
NB1192_REF_POOLED = 0.5514   # mean reference per task spec
MARGIN = 0.003


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical capacity to nb1183 / nb1182 / nb1153."""
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


def _load_maccs_unblind(
    n_test_expected: int, unb_idx: np.ndarray
) -> np.ndarray:
    """Load MACCS test cache (513,167) and slice to 166 cols (drop bit 0)."""
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
    # Spec asks for MACCS-166; cache has 167. Drop bit 0 (padding) to land
    # on the standard 166-bit MACCS dictionary count.
    if X_te.shape[1] == 167:
        X_te = X_te[:, 1:]
    elif X_te.shape[1] == 166:
        pass
    else:
        raise ValueError(
            f"MACCS test cache unexpected width: {X_te.shape[1]} "
            f"(expected 166 or 167)"
        )
    return X_te[unb_idx].astype(np.float32)


def _load_atompair_unblind(
    n_test_expected: int, unb_idx: np.ndarray
) -> np.ndarray:
    """Load AtomPair test cache (513,2048) and slice to unblind."""
    if ATOMPAIR_TE_2048_PATH.exists():
        path = ATOMPAIR_TE_2048_PATH
    elif ATOMPAIR_TE_PATH.exists():
        path = ATOMPAIR_TE_PATH
    else:
        raise FileNotFoundError(
            f"AtomPair test cache missing: {ATOMPAIR_TE_PATH} "
            f"or {ATOMPAIR_TE_2048_PATH}"
        )
    X_te = np.load(path)
    if X_te.shape[0] != n_test_expected:
        raise ValueError(
            f"AtomPair test cache shape mismatch: {X_te.shape} "
            f"vs n_test={n_test_expected}  (path={path})"
        )
    if X_te.shape[1] != ATOMPAIR_DIM:
        raise ValueError(
            f"AtomPair test cache unexpected width: {X_te.shape[1]} "
            f"(expected {ATOMPAIR_DIM})"
        )
    print(f"[feat]   AtomPair source = cached: {path}")
    return X_te[unb_idx].astype(np.float32)


def _load_mordred_unblind(
    n_test_expected: int, unb_idx: np.ndarray
) -> np.ndarray:
    """Load Mordred test cache (513,1533) and slice to unblind."""
    if not MORDRED_TE_PATH.exists():
        raise FileNotFoundError(
            f"Mordred cache missing: {MORDRED_TE_PATH}"
        )
    X_te = np.load(MORDRED_TE_PATH).astype(np.float32)
    if X_te.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te.shape} vs "
            f"n_test={n_test_expected}"
        )
    if X_te.shape[1] != MORDRED_DIM:
        raise ValueError(
            f"Mordred test cache unexpected width: {X_te.shape[1]} "
            f"(expected {MORDRED_DIM})"
        )
    X_te = _median_impute(X_te)
    return X_te[unb_idx]


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CONCAT-feature residual LGBM on nb1070 anchor")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          features = MACCS ({MACCS_DIM}) || "
          f"AtomPair ({ATOMPAIR_DIM}) || Mordred ({MORDRED_DIM}) "
          f"= {CONCAT_DIM}")
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

    print(f"[feat] loading MACCS cache + slicing to unblind ...")
    X_maccs = _load_maccs_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat]   MACCS    unblind shape = {X_maccs.shape}  "
          f"(bit density = {X_maccs.mean():.4f}, "
          f"const cols = {int((X_maccs.var(axis=0) == 0).sum())}/"
          f"{X_maccs.shape[1]})")

    print(f"[feat] loading AtomPair cache + slicing to unblind ...")
    X_ap = _load_atompair_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat]   AtomPair unblind shape = {X_ap.shape}  "
          f"(bit density = {X_ap.mean():.4f}, "
          f"const cols = {int((X_ap.var(axis=0) == 0).sum())}/"
          f"{X_ap.shape[1]})")

    print(f"[feat] loading Mordred cache + slicing to unblind ...")
    X_mord = _load_mordred_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat]   Mordred  unblind shape = {X_mord.shape}  "
          f"(const cols = {int((X_mord.var(axis=0) == 0).sum())}/"
          f"{X_mord.shape[1]})")

    X_unb = np.concatenate([X_maccs, X_ap, X_mord], axis=1).astype(np.float32)
    print(f"[feat] CONCAT X_unb shape = {X_unb.shape}  "
          f"({X_maccs.shape[1]} + {X_ap.shape[1]} + {X_mord.shape[1]} "
          f"= {X_unb.shape[1]})")

    # Final safety re-impute across the concat (covers any cross-cache
    # residual NaNs from Mordred).
    if not np.all(np.isfinite(X_unb)):
        X_unb = _median_impute(X_unb)
        print("[feat] CONCAT had non-finite cells -> median-imputed")

    if X_unb.shape[1] != CONCAT_DIM:
        raise ValueError(
            f"CONCAT dim mismatch: {X_unb.shape[1]} vs expected {CONCAT_DIM}"
        )

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow, CONCAT {CONCAT_DIM})")
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
    print(f"   nb1183 ref             = {NB1183_REF_POOLED:.4f}  "
          f"(MACCS-only residual on nb1070)")
    print(f"   nb1192 ref             = {NB1192_REF_POOLED:.4f}  "
          f"(mean ref per task spec)")

    beats_nb1070 = rae_mean_bag < rae_anchor - MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_REF_POOLED - MARGIN
    beats_nb1192 = rae_mean_bag < NB1192_REF_POOLED - MARGIN

    if beats_nb1183 and beats_nb1192:
        verdict = "CONCAT_MACCS_AP_MORD_BEATS_BOTH_NB1183_AND_NB1192"
    elif beats_nb1183 and not beats_nb1192:
        verdict = "CONCAT_BEATS_NB1183_BUT_NOT_NB1192"
    elif beats_nb1192 and not beats_nb1183:
        verdict = "CONCAT_BEATS_NB1192_BUT_NOT_NB1183"
    elif beats_nb1070:
        verdict = "CONCAT_HELPS_NB1070_BUT_NOT_NB1183_OR_NB1192"
    elif abs(rae_mean_bag - NB1183_REF_POOLED) < MARGIN:
        verdict = "CONCAT_TIES_NB1183_NO_NEW_3WAY_CROSS_FEATURE_SIGNAL"
    else:
        verdict = "CONCAT_HURTS_NB1183"
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
        "feature_source": (
            f"concat_maccs_{MACCS_DIM}_atompair_{ATOMPAIR_DIM}_"
            f"mordred_{MORDRED_DIM}"
        ),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "feature_dim_maccs": int(X_maccs.shape[1]),
        "feature_dim_atompair": int(X_ap.shape[1]),
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
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_REF_POOLED,
        "delta_mean_bag_vs_nb1192": rae_mean_bag - NB1192_REF_POOLED,
        "margin": MARGIN,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1192": bool(beats_nb1192),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1183_ref_pooled": NB1183_REF_POOLED,
        "nb1192_ref_pooled": NB1192_REF_POOLED,
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
              "delta_mean_bag_vs_nb1183",
              "delta_mean_bag_vs_nb1192",
              "beats_nb1070", "beats_nb1183", "beats_nb1192",
              "verdict"):
        print(f"  {k}: {res.get(k)}")
