"""nb1174 -- Anchor switch: fit a THIRD residual on top of nb1161 (mean of
nb1130 + nb1153), using Morgan+RDKit features.

Hypothesis:
    nb1130 (Morgan+RDKit residual on nb1070) and nb1153 (Mordred residual
    on nb1070) have highly correlated residuals (Pearson ~0.9753), but
    AFTER both corrections have been blended into nb1161 (mean), the
    remaining error vector y - nb1161 has lower variance (std 0.6383 vs
    0.6666 for nb1070) and may still expose a small *structural* signal
    that a very shallow Morgan+RDKit LGBM can squeeze out.  This is a
    recursive-residual experiment: anchor switches from nb1070 (RAE 0.5771)
    to nb1161 mean (RAE 0.5600).

Protocol:
    1. Load anchor = nb1161_mean_oof.npy  (253,) -- pooled RAE 0.5600.
    2. residual_target = y_unb - nb1161_mean_oof.
    3. Features = cache_combined_features.npz['X_te'][unb_idx]
                  (Morgan 2048 + RDKit 217 = 2265 dims).
    4. 5-seed bag (0, 1, 7, 42, 137) of even SHALLOWER LGBM Huber:
            max_depth=2, num_leaves=4, n_estimators=60, learning_rate=0.04,
            min_child_samples=25, alpha=1.0
       (shallower than nb1130's spec because the target variance is smaller).
    5. 5-fold honest cross-fit per seed; pooled corrected OOF =
       anchor + residual_oof_seed.  Mean-bag across the 5 seeds.
    6. Verdict at 0.003 margin vs nb1161 mean (0.5600).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
    data/processed/nb1174_per_seed_corrected_oof.npy  (5, 253) float32
    data/processed/nb1174_mean_bag_oof.npy            (253,) float32
    data/processed/nb1174_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1174"
ANCHOR_TAG = "nb1161_mean"
PRIOR_ANCHOR_TAG = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Reference numbers (pooled RAE on 253 unblind).
NB1161_MEAN_REF = 0.5600
NB1070_REF = 0.5790
NB1130_MEAN_BAG_REF = 0.5673
NB1153_MEAN_BAG_REF = 0.5640

MARGIN = 0.003  # verdict tolerance


def _lgbm_params(seed: int) -> dict:
    """Even shallower than nb1130 -- recursive-residual target has smaller
    variance, so we further reduce capacity to avoid fitting noise."""
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.04,
        n_estimators=60,
        max_depth=2,
        num_leaves=4,
        min_child_samples=25,
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
    """Honest 5-fold cross-fit shallow LGBM Huber on residual; return OOF."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ANCHOR SWITCH: recursive residual on nb1161 mean")
    print(f"          anchor = {ANCHOR_TAG} (pooled RAE ~{NB1161_MEAN_REF:.4f})")
    print(f"          prior anchor = {PRIOR_ANCHOR_TAG} "
          f"(pooled RAE ~{NB1070_REF:.4f})")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - nb1161_mean_oof")
    print(f"          features = combined Morgan(2048)+RDKit(217) cached "
          f"(2265 dims)")
    print(f"          LGBM: max_depth=2, num_leaves=4, n_est=60, lr=0.04, "
          f"min_child=25, obj=huber(alpha=1.0)  [SHALLOWER than nb1130]")
    print("=" * 78)

    # ---- Truth + indices ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    n_unb = len(y_unb)
    print(f"[load] y_unb shape = {y_unb.shape}  unb_idx shape = {unb_idx.shape}")

    # ---- Anchor (nb1161 mean of nb1130 + nb1153) ----
    anchor_path = DATA_PROCESSED / "nb1161_mean_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; run nb1161 first."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: "
            f"{anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {anchor_path.name} shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref {NB1161_MEAN_REF:.4f})")

    # ---- Prior anchor (nb1070) -- for residual variance comparison ----
    nb1070_path = DATA_PROCESSED / f"{PRIOR_ANCHOR_TAG}_pred_oof.npy"
    if not nb1070_path.exists():
        raise FileNotFoundError(f"{nb1070_path} not found.")
    nb1070_oof = np.load(nb1070_path).astype(np.float64)
    rae_nb1070 = float(rae(y_unb, nb1070_oof))
    print(f"[load] {nb1070_path.name} pooled RAE = {rae_nb1070:.4f}  "
          f"(ref {NB1070_REF:.4f})")

    # ---- Residual targets + variance diagnostics ----
    residual = y_unb - anchor_oof          # target for the recursive LGBM
    residual_prior = y_unb - nb1070_oof    # for variance ratio diagnostic
    resid_std_anchor = float(residual.std())
    resid_std_prior = float(residual_prior.std())
    resid_var_ratio = resid_std_anchor / resid_std_prior if resid_std_prior > 0 \
        else float("nan")
    print(f"[resid] target  (y - nb1161_mean) : mean={residual.mean():+.4f}  "
          f"std={resid_std_anchor:.4f}  min={residual.min():+.4f}  "
          f"max={residual.max():+.4f}")
    print(f"[resid] prior   (y - nb1070_oof)  : mean={residual_prior.mean():+.4f}"
          f"  std={resid_std_prior:.4f}")
    print(f"[resid] variance ratio (nb1161-resid std / nb1070-resid std) "
          f"= {resid_var_ratio:.4f}")

    # ---- Load cached combined features and slice to unblind ----
    cache_path = DATA_PROCESSED / "cache_combined_features.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} not found.")
    z = np.load(cache_path)
    X_te = z["X_te"]   # (513, 2265) float32
    print(f"[feat] cache_combined_features.npz X_te shape = {X_te.shape}")
    X_unb = X_te[unb_idx].astype(np.float32, copy=False)
    # Median-impute (cached is already imputed in most paths, but be safe).
    if not np.all(np.isfinite(X_unb)):
        col_med = np.nanmedian(X_unb, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        bad = ~np.isfinite(X_unb)
        if bad.any():
            X_unb = np.where(bad, col_med[None, :], X_unb)
    print(f"[feat] X_unb (sliced) shape = {X_unb.shape}  "
          f"finite = {np.all(np.isfinite(X_unb))}")

    # ---- Per-seed cross-fit residual ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT ON nb1161 MEAN ANCHOR")
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
            "delta_vs_anchor": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1161_mean = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    # ---- Bag aggregations (mean only, per spec) ----
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATION (mean across seeds)")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1161_mean = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   nb1161 mean ref        = {NB1161_MEAN_REF:.4f}")
    print(f"   nb1130 mean bag ref    = {NB1130_MEAN_BAG_REF:.4f}")
    print(f"   nb1153 mean bag ref    = {NB1153_MEAN_BAG_REF:.4f}")
    print(f"   nb1070 anchor ref      = {NB1070_REF:.4f}")

    # ---- Verdict (0.003 margin per spec) ----
    beats_nb1161 = bool(rae_mean_bag < rae_anchor - MARGIN)
    ties_nb1161 = bool(abs(rae_mean_bag - rae_anchor) <= MARGIN)
    if beats_nb1161:
        verdict = "RECURSIVE_RESIDUAL_BEATS_NB1161_ANCHOR"
    elif ties_nb1161:
        verdict = "RECURSIVE_RESIDUAL_FLAT_VS_NB1161_ANCHOR"
    else:
        verdict = "RECURSIVE_RESIDUAL_HURTS_VS_NB1161_ANCHOR"
    print(f"   verdict                = {verdict}")

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR_TAG,
        "prior_anchor": PRIOR_ANCHOR_TAG,
        "n_unb": int(n_unb),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "feature_source": "cache_combined_features.npz[X_te][unb_idx]",
        "lgbm_params": _lgbm_params(0),
        "rae_anchor_nb1161_mean": rae_anchor,
        "rae_prior_anchor_nb1070": rae_nb1070,
        "resid_std_anchor": resid_std_anchor,
        "resid_std_prior": resid_std_prior,
        "resid_variance_ratio_nb1161_over_nb1070": resid_var_ratio,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "delta_mean_bag_vs_nb1161_anchor": rae_mean_bag - rae_anchor,
        "beats_nb1161": beats_nb1161,
        "margin": MARGIN,
        "verdict": verdict,
        "nb1161_mean_ref": NB1161_MEAN_REF,
        "nb1130_mean_bag_ref": NB1130_MEAN_BAG_REF,
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
        "nb1070_ref": NB1070_REF,
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
    for k in ("rae_anchor_nb1161_mean", "rae_prior_anchor_nb1070",
              "resid_std_anchor", "resid_std_prior",
              "resid_variance_ratio_nb1161_over_nb1070",
              "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag",
              "delta_mean_bag_vs_nb1161_anchor",
              "beats_nb1161", "verdict"):
        print(f"  {k}: {res.get(k)}")
