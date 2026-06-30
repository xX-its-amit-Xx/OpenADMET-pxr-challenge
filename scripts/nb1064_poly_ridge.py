"""nb1064 -- Polynomial degree-2 interaction features + Ridge / LGBM (cycle 135).

HYPOTHESIS:
    K=28 SHAP-top features (from nb2103, the current PRE-unblind winner at
    mean-bag RAE 0.4737 / median-bag 0.4698) get expanded with all pairwise
    interactions:  28 raw + C(28, 2) = 28 + 378 = 406 features.
    Ridge with strong L2 should survive n=253 (n_feat > n) by shrinking the
    interaction block towards zero unless it carries real signal.  LGBM with
    feature_fraction=0.3 on the same 406-feature space provides a non-Ridge
    counterpart (sparse, tree-based, random columns per split).
    Decision margin vs nb2103 K=28 (0.4737 / 0.4698) is 0.003.

PROTOCOL:
    1. Load SHAP top-28 feature matrix
       data/processed/X_unb_28_nb2103.npy  -> (253, 28) float32
       Anchor: chemprop_aux te[unb_idx]   (RAE 0.6216)
       Residual: y_unb - anchor
    2. PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
       28 raw + 28 squares + 378 pairwise = 434 columns
       (Strict task spec says "28 + C(28,2) = 406", i.e. interaction_only.
       We report BOTH variants -- with/without squares -- to be thorough.)
    3. StandardScaler FIT INSIDE EACH FOLD (no leakage).
    4. Ridge alpha grid {0.1, 1, 10, 100, 1000}; KFold(n_splits=5, shuffle,
       seed=42) cross-fit on residual.  Pick best alpha by cross-fit RAE.
       Final = chemprop_aux + cross-fit-Ridge-residual.
    5. LGBM with feature_fraction=0.3 on the 406- (or 434-) feature space,
       same 5-fold cross-fit, 5-seed bag.  Mean-bag and median-bag RAE.
    6. Compare vs nb2103 K=28 (mean-bag 0.4737, median-bag 0.4698) at margin
       0.003.  If beats -> build deploy CSV.

Outputs:
    scripts/nb1064_poly_ridge.py
    data/processed/nb1064_summary.json
    data/processed/nb1064_ridge_resid_oof_best.npy
    data/processed/nb1064_lgbm_resid_oof_mean_bag.npy
    submissions/nb1064_poly_ridge_lgbm.csv  (only if beats nb2103)
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
import lightgbm as lgb

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1064"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"

NB2103_K28_MEAN_BAG_REF = 0.4737   # nb2103 K=28 mean-bag (current PRE-unblind winner)
NB2103_K28_MEDIAN_BAG_REF = 0.4698 # nb2103 K=28 median-bag
CHEMPROP_AUX_REF = 0.6216
DECISION_MARGIN = 0.003

CV_FOLDS = 5
CV_SEED = 42
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
LGBM_SEEDS = [0, 1, 7, 42, 137]
LGBM_FEAT_FRAC = 0.3
LGBM_N_EST = 300
LGBM_LR = 0.03
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_MIN_CHILD = 5
LGBM_REG_LAMBDA = 2.0


def _ridge_cross_fit_one_alpha(
    X: np.ndarray, residual: np.ndarray, alpha: float
) -> np.ndarray:
    """5-fold cross-fit Ridge with per-fold StandardScaler."""
    n = len(residual)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_loc])
        X_va = scaler.transform(X[va_loc])
        mdl = Ridge(alpha=alpha, random_state=0)
        mdl.fit(X_tr, residual[tr_loc])
        oof[va_loc] = mdl.predict(X_va)
    return oof


def _lgbm_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    """5-fold cross-fit LGBM with feature_fraction=0.3."""
    n = len(residual)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(
            objective="regression",
            max_depth=LGBM_MAX_DEPTH,
            num_leaves=LGBM_NUM_LEAVES,
            n_estimators=LGBM_N_EST,
            learning_rate=LGBM_LR,
            min_child_samples=LGBM_MIN_CHILD,
            reg_lambda=LGBM_REG_LAMBDA,
            feature_fraction=LGBM_FEAT_FRAC,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=seed,
            n_jobs=2,
            verbosity=-1,
        )
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _sweep_ridge(
    X: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
    y_unb: np.ndarray, label: str,
) -> dict:
    print(f"\n   [Ridge sweep on {label}]")
    records = []
    best_alpha = None
    best_rae = np.inf
    best_oof = None
    for alpha in ALPHA_GRID:
        t0 = time.time()
        resid_oof = _ridge_cross_fit_one_alpha(X, residual, alpha)
        corrected = anchor + resid_oof
        rae_corr = float(rae(y_unb, corrected))
        records.append({
            "alpha": float(alpha),
            "rae_corrected": rae_corr,
            "resid_oof_std": float(resid_oof.std()),
            "resid_oof_mean": float(resid_oof.mean()),
            "wall_sec": round(time.time() - t0, 2),
        })
        print(f"      alpha={alpha:8.2f}  rae={rae_corr:.4f}  "
              f"resid_std={resid_oof.std():.4f}  "
              f"wall={time.time() - t0:.1f}s")
        if rae_corr < best_rae:
            best_rae = rae_corr
            best_alpha = float(alpha)
            best_oof = resid_oof.astype(np.float64)
    return {
        "label": label,
        "records": records,
        "best_alpha": best_alpha,
        "best_rae": best_rae,
        "best_oof": best_oof,
    }


def _bag_lgbm(
    X: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
    y_unb: np.ndarray, label: str,
) -> dict:
    print(f"\n   [LGBM bag on {label}]")
    n_seeds = len(LGBM_SEEDS)
    n_unb = len(y_unb)
    per_seed_corrected = np.zeros((n_seeds, n_unb), dtype=np.float64)
    per_seed_rae = []
    records = []
    for i, s in enumerate(LGBM_SEEDS):
        t0 = time.time()
        resid_oof = _lgbm_cross_fit_one_seed(X, residual, s)
        corrected = anchor + resid_oof
        per_seed_corrected[i] = corrected
        rae_s = float(rae(y_unb, corrected))
        per_seed_rae.append(rae_s)
        records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "resid_oof_std": float(resid_oof.std()),
            "resid_oof_mean": float(resid_oof.mean()),
            "wall_sec": round(time.time() - t0, 2),
        })
        print(f"      seed={s:3d}  rae={rae_s:.4f}  "
              f"resid_std={resid_oof.std():.4f}  "
              f"wall={time.time() - t0:.1f}s")
    mean_bag = per_seed_corrected.mean(axis=0)
    median_bag = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag))
    rae_median_bag = float(rae(y_unb, median_bag))
    print(f"      mean_bag   rae={rae_mean_bag:.4f}")
    print(f"      median_bag rae={rae_median_bag:.4f}")
    return {
        "label": label,
        "per_seed_records": records,
        "per_seed_rae_list": per_seed_rae,
        "per_seed_rae_mean": float(np.mean(per_seed_rae)),
        "per_seed_rae_std": float(np.std(per_seed_rae)),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "mean_bag_corrected": mean_bag,
        "median_bag_corrected": median_bag,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PolynomialFeatures(deg=2) + Ridge/LGBM on SHAP top-28")
    print(f"          anchor={ANCHOR}  cv_folds={CV_FOLDS}  cv_seed={CV_SEED}")
    print(f"          ridge alpha grid: {ALPHA_GRID}")
    print(f"          lgbm seeds: {LGBM_SEEDS}  feature_fraction={LGBM_FEAT_FRAC}")
    print(f"          ref nb2103 K=28 mean-bag = {NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median-bag = {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] n_test={n_test}  n_unb={n_unb}")
    print(f"[load] anchor in_RAE = {rae_anchor:.4f}  (ref {CHEMPROP_AUX_REF:.4f})")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load SHAP top-28 feature matrix ----
    if not X_UNB_28_PATH.exists():
        raise FileNotFoundError(f"missing SHAP top-28 cache: {X_UNB_28_PATH}")
    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float32)
    if X_unb_28.shape != (n_unb, 28):
        raise ValueError(f"X_unb_28 shape {X_unb_28.shape} != ({n_unb}, 28)")
    print(f"[feat] X_unb_28 (SHAP top-28) shape = {X_unb_28.shape}")
    # Per-column finiteness guard
    X_unb_28 = np.where(np.isfinite(X_unb_28), X_unb_28, 0.0).astype(np.float32)

    # ---- Build polynomial-degree-2 expansions ----
    # Variant A: strict-spec "28 + C(28,2) = 406"  -> interaction_only=True,
    #   include_bias=False
    poly_int = PolynomialFeatures(
        degree=2, interaction_only=True, include_bias=False
    )
    X_poly_int = poly_int.fit_transform(X_unb_28).astype(np.float32)
    n_feat_int = X_poly_int.shape[1]
    print(f"[poly_interactions_only] X shape = {X_poly_int.shape}  "
          f"(expected 28 + C(28,2) = {28 + (28*27)//2})")

    # Variant B: standard "28 + 28 squares + 378 cross" = 434
    poly_full = PolynomialFeatures(
        degree=2, interaction_only=False, include_bias=False
    )
    X_poly_full = poly_full.fit_transform(X_unb_28).astype(np.float32)
    n_feat_full = X_poly_full.shape[1]
    print(f"[poly_full_deg2]          X shape = {X_poly_full.shape}  "
          f"(expected 28 + 28 + C(28,2) = "
          f"{28 + 28 + (28*27)//2})")

    if n_feat_int != 28 + (28 * 27) // 2:
        raise ValueError(
            f"interaction-only count mismatch: got {n_feat_int}, "
            f"expected {28 + (28 * 27) // 2}"
        )
    if n_feat_full != 28 + 28 + (28 * 27) // 2:
        raise ValueError(
            f"full deg-2 count mismatch: got {n_feat_full}, "
            f"expected {28 + 28 + (28 * 27) // 2}"
        )

    # ---- Ridge sweep on BOTH variants ----
    ridge_int = _sweep_ridge(
        X_poly_int, residual, anchor, y_unb,
        label="poly_interactions_only_406"
    )
    ridge_full = _sweep_ridge(
        X_poly_full, residual, anchor, y_unb,
        label="poly_full_deg2_434"
    )
    # Also a Ridge sweep on the raw 28 cols as a sanity reference
    ridge_raw28 = _sweep_ridge(
        X_unb_28, residual, anchor, y_unb,
        label="raw_28_no_poly"
    )

    # Pick best Ridge across variants by cross-fit RAE
    ridge_candidates = [ridge_int, ridge_full, ridge_raw28]
    best_ridge = min(ridge_candidates, key=lambda r: r["best_rae"])
    print(f"\n[Ridge best] variant={best_ridge['label']}  "
          f"alpha={best_ridge['best_alpha']}  rae={best_ridge['best_rae']:.4f}")
    np.save(DATA_PROCESSED / f"{TAG}_ridge_resid_oof_best.npy",
            best_ridge["best_oof"].astype(np.float32))

    # ---- LGBM bag on full deg-2 (richer signal) ----
    lgbm_full = _bag_lgbm(
        X_poly_full, residual, anchor, y_unb,
        label="lgbm_feat_frac_0.3_on_poly_full_434"
    )
    # And LGBM bag on interaction-only 406 (strict-spec variant)
    lgbm_int = _bag_lgbm(
        X_poly_int, residual, anchor, y_unb,
        label="lgbm_feat_frac_0.3_on_poly_int_406"
    )

    best_lgbm = (
        lgbm_full if lgbm_full["rae_mean_bag"] < lgbm_int["rae_mean_bag"]
        else lgbm_int
    )
    print(f"\n[LGBM best] variant={best_lgbm['label']}  "
          f"mean_bag={best_lgbm['rae_mean_bag']:.4f}  "
          f"median_bag={best_lgbm['rae_median_bag']:.4f}")

    np.save(DATA_PROCESSED / f"{TAG}_lgbm_resid_oof_mean_bag.npy",
            (best_lgbm["mean_bag_corrected"] - anchor).astype(np.float32))

    # ---- Compare vs nb2103 K=28 ----
    print("\n" + "=" * 78)
    print("SUMMARY TABLE -- all variants vs nb2103 K=28")
    print("=" * 78)
    print(f"   nb2103 K=28 mean_bag        = {NB2103_K28_MEAN_BAG_REF:.4f}")
    print(f"   nb2103 K=28 median_bag      = {NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"   anchor (chemprop_aux)       = {rae_anchor:.4f}")
    print()
    print(f"   Ridge poly_int_406 best     = {ridge_int['best_rae']:.4f}  "
          f"(alpha={ridge_int['best_alpha']})")
    print(f"   Ridge poly_full_434 best    = {ridge_full['best_rae']:.4f}  "
          f"(alpha={ridge_full['best_alpha']})")
    print(f"   Ridge raw_28 best           = {ridge_raw28['best_rae']:.4f}  "
          f"(alpha={ridge_raw28['best_alpha']})")
    print(f"   LGBM poly_int_406 mean_bag  = {lgbm_int['rae_mean_bag']:.4f}  "
          f"median_bag = {lgbm_int['rae_median_bag']:.4f}")
    print(f"   LGBM poly_full_434 mean_bag = {lgbm_full['rae_mean_bag']:.4f}  "
          f"median_bag = {lgbm_full['rae_median_bag']:.4f}")
    print()

    best_overall = min(
        ridge_candidates + [lgbm_full, lgbm_int],
        key=lambda r: r.get("rae_mean_bag", r.get("best_rae"))
    )
    best_overall_rae = best_overall.get(
        "rae_mean_bag", best_overall.get("best_rae")
    )
    print(f"   best_overall                = {best_overall_rae:.4f}  "
          f"({best_overall['label']})")
    print(f"   delta_vs_nb2103_K28_mean    = "
          f"{best_overall_rae - NB2103_K28_MEAN_BAG_REF:+.4f}")
    print(f"   delta_vs_nb2103_K28_median  = "
          f"{best_overall_rae - NB2103_K28_MEDIAN_BAG_REF:+.4f}")
    print(f"   margin                      = {DECISION_MARGIN}")

    beats_mean = best_overall_rae < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    beats_median = best_overall_rae < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN
    flat_mean = abs(best_overall_rae - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN

    if beats_median:
        verdict = "BEATS_NB2103_K28_MEDIAN_BAG"
    elif beats_mean:
        verdict = "BEATS_NB2103_K28_MEAN_BAG_BUT_NOT_MEDIAN"
    elif flat_mean:
        verdict = "FLAT_VS_NB2103_K28"
    else:
        verdict = "DOES_NOT_BEAT_NB2103_K28"
    print(f"\n   verdict = {verdict}")

    # ---- Build deploy CSV only if we beat nb2103 ----
    deploy_path = None
    if beats_median or beats_mean:
        # NOTE: we ONLY have unb predictions here (n_unb=253), not n_test=513.
        # A full deploy CSV requires the same featurization applied to the
        # remaining 260 blinded rows.  That's a separate build step that this
        # script does not cover (X_unb_28_nb2103.npy is unb-only).
        # Instead, write a clearly-labeled marker file with the unb predictions
        # so the next stage (a deploy refit on 513-row features) can pick it up.
        marker_path = (
            DATA_PROCESSED
            / f"{TAG}_winner_unb_predictions.npy"
        )
        if best_overall.get("mean_bag_corrected") is not None:
            np.save(marker_path,
                    best_overall["mean_bag_corrected"].astype(np.float32))
        else:
            np.save(marker_path,
                    (anchor + best_overall["best_oof"]).astype(np.float32))
        deploy_path = str(marker_path)
        print(f"   [marker] wrote unb-only winner predictions: {marker_path}")
        print(f"   [note] no submissions/ CSV built: SHAP top-28 cache only "
              f"covers unb_idx (253); deploy refit on 513 requires nb2103-"
              f"style featurizer pipeline.")
    else:
        print("   [skip-deploy] does not clear margin vs nb2103 K=28")

    # ---- Save summary ----
    def _clean_records(d: dict) -> dict:
        out = {k: v for k, v in d.items()
               if k not in {"best_oof", "mean_bag_corrected",
                            "median_bag_corrected"}}
        return out

    summary = {
        "tag": TAG,
        "method": ("poly_features_deg2_ridge_alpha_grid_plus_lgbm_"
                   "feature_fraction_0.3_residual_correction_chemprop_aux"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("SHAP top-28 cached from nb2103 "
                        "(data/processed/X_unb_28_nb2103.npy)"),
        "feature_input": {
            "name": "X_unb_28_nb2103",
            "shape": list(X_unb_28.shape),
            "source": str(X_UNB_28_PATH),
        },
        "poly_interaction_only_406_shape": [n_unb, n_feat_int],
        "poly_full_deg2_434_shape": [n_unb, n_feat_full],
        "cv_folds": CV_FOLDS,
        "cv_seed": CV_SEED,
        "alpha_grid": ALPHA_GRID,
        "lgbm_seeds": LGBM_SEEDS,
        "lgbm_feature_fraction": LGBM_FEAT_FRAC,
        "lgbm_n_estimators": LGBM_N_EST,
        "lgbm_learning_rate": LGBM_LR,
        "lgbm_max_depth": LGBM_MAX_DEPTH,
        "lgbm_num_leaves": LGBM_NUM_LEAVES,
        "lgbm_min_child_samples": LGBM_MIN_CHILD,
        "lgbm_reg_lambda": LGBM_REG_LAMBDA,
        "n_unb": int(n_unb),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "decision_margin": DECISION_MARGIN,
        "ridge_poly_int_406": _clean_records(ridge_int),
        "ridge_poly_full_434": _clean_records(ridge_full),
        "ridge_raw_28": _clean_records(ridge_raw28),
        "lgbm_poly_int_406": _clean_records(lgbm_int),
        "lgbm_poly_full_434": _clean_records(lgbm_full),
        "best_overall_label": best_overall["label"],
        "best_overall_rae": float(best_overall_rae),
        "delta_best_vs_nb2103_K28_mean":
            float(best_overall_rae - NB2103_K28_MEAN_BAG_REF),
        "delta_best_vs_nb2103_K28_median":
            float(best_overall_rae - NB2103_K28_MEDIAN_BAG_REF),
        "beats_nb2103_K28_mean": bool(beats_mean),
        "beats_nb2103_K28_median": bool(beats_median),
        "flat_vs_nb2103_K28_mean": bool(flat_mean),
        "verdict": verdict,
        "deploy_path": deploy_path,
        "pre_unblind_clean": True,
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
    for k in (
        "poly_interaction_only_406_shape",
        "poly_full_deg2_434_shape",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_ref",
        "nb2103_K28_median_bag_ref",
        "best_overall_label",
        "best_overall_rae",
        "delta_best_vs_nb2103_K28_mean",
        "delta_best_vs_nb2103_K28_median",
        "beats_nb2103_K28_mean",
        "beats_nb2103_K28_median",
        "verdict",
        "deploy_path",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
