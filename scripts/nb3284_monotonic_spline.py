"""nb3284 -- Monotonic cubic-spline calibration on top of nb3090.

NEW PARADIGM:
    Every prior post-hoc recalibration on this anchor family used either a
    scalar rank-stretch (df=1) or IsotonicRegression -- a piecewise-CONSTANT
    monotone step function that places a knot at (almost) every training
    point.  Isotonic is non-smooth and, at n=253, the step grid can overfit
    local wiggles of the (anchor -> truth) map.

    nb3284 replaces the step function with a SMOOTH MONOTONE CUBIC SPLINE
    (scipy.interpolate.PchipInterpolator).  PCHIP (Piecewise Cubic Hermite
    Interpolating Polynomial) is shape-preserving: it is monotone on any
    interval where the control points are monotone and never overshoots,
    so it stays a valid recalibration map while being C1-smooth.

    To control degrees of freedom we do NOT fit PCHIP on every training
    point.  Instead, per fold we:
        1. sort fold-train by the anchor value nb3090,
        2. cut into ~10 quantile bins,
        3. take the per-bin MEDIAN of (anchor, y) as ~10 control points,
        4. enforce strict monotone increase of the control x's (and of the
           control y's, since the map must be non-decreasing), then
        5. fit PchipInterpolator on those ~10 (x_med, y_med) knots.
    The ~10 bin-median control points give the spline ~10 effective df vs
    isotonic's ~N -- smoother, less overfit.  Val anchors outside the
    control-x envelope are CLAMPED to the boundary (PCHIP does not
    extrapolate sensibly).

PROTOCOL:
    - Anchor = nb3090_pred_oof.npy  (q-cut finer winner, 15-seed mean 0.4472).
    - 5-fold SCAFFOLD CV via scaffold_kfold_indices, repeated over
      kf_seeds {1216..1230} (15 fresh seeds, decision-grade dispersion).
    - Per (kf_seed, fold): build ~10 bin-median knots on (anchor[tr], y[tr]),
      fit PCHIP, predict anchor[va] (clamped).  Pool the 5 fold-val vecs into
      a 253 spline_oof; pooled RAE per seed.
    - per-fold-mean = mean over the 15 per-seed pooled RAEs.
    - Deploy: refit the bin-median PCHIP on the FULL 253 anchor
      (nb3090_pred_oof -> y_unb); transform te_nb3090 (513) with the same
      clamp; save te_nb3284.  Deploy pred_oof = mean of the 15 per-seed
      spline_oof vectors.

GATE:
    per_fold_mean < 0.4423  ->  "BETTER"
    else                    ->  "FAIL"

Outputs:
    scripts/nb3284_monotonic_spline.py
    data/processed/nb3284_summary.json
    data/processed/nb3284_pred_oof.npy   (253,) float32
    data/processed/te_nb3284.npy         (513,) float32
    submissions/nb3284_monotonic_spline.csv  (on any non-FAIL)
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
import pandas as pd
from scipy.interpolate import PchipInterpolator

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3284"

# ---- Anchor: nb3090 winner ----
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb3090.npy"

# ---- Diagnostic anchor ----
CHEMPROP_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ---- Spline knot config ----
N_BINS = 10            # ~10 quantile bins -> ~10 control points
Y_CLAMP_MIN = 3.0      # pEC50 envelope clamp on spline output
Y_CLAMP_MAX = 8.0

# ---- CV eval ----
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 fresh seeds {1216..1230}

# ---- Gate ----
GATE_BETTER = 0.4423   # per-fold-mean strictly below -> BETTER

# ---- Refs ----
NB3090_REF = 0.4472
NB3080_REF = 0.4475
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


def _bin_median_knots(x: np.ndarray, y: np.ndarray, n_bins: int):
    """Sort by x, cut into ~n_bins quantile bins, return per-bin (median x,
    median y) control points with strictly-increasing x and non-decreasing y.

    Returns (kx, ky) suitable for PchipInterpolator.  Falls back gracefully
    when there are fewer distinct quantile edges than requested (small fold).
    """
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]
    n = len(xs)
    # quantile bin edges on the sorted anchor
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(xs, qs)
    # assign each point to a bin in [0, n_bins-1]
    bin_id = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, n_bins - 1)

    kx_list, ky_list = [], []
    for b in range(n_bins):
        m = bin_id == b
        if not m.any():
            continue
        kx_list.append(float(np.median(xs[m])))
        ky_list.append(float(np.median(ys[m])))
    kx = np.asarray(kx_list, dtype=np.float64)
    ky = np.asarray(ky_list, dtype=np.float64)

    # Collapse duplicate / non-increasing control x's (PCHIP needs strictly
    # increasing x).  When two bin medians share an x, merge their y by mean.
    keep_x, keep_y = [], []
    for cx, cy in zip(kx, ky):
        if keep_x and cx <= keep_x[-1]:
            # merge into previous control point (mean y), keep x as-is
            keep_y[-1] = 0.5 * (keep_y[-1] + cy)
            continue
        keep_x.append(cx)
        keep_y.append(cy)
    kx = np.asarray(keep_x, dtype=np.float64)
    ky = np.asarray(keep_y, dtype=np.float64)

    # Enforce non-decreasing y (the recalibration map must be monotone up).
    ky = np.maximum.accumulate(ky)
    return kx, ky


def calibrate_fold(anchor_tr: np.ndarray, y_tr: np.ndarray,
                   anchor_va: np.ndarray, n_bins: int):
    """Fit a bin-median monotone PCHIP spline on (anchor_tr, y_tr) and apply
    to anchor_va with boundary clamping + pEC50 envelope clamp.

    Returns (pred_va, n_knots).  If fewer than 2 distinct control points are
    available (degenerate tiny fold) it falls back to the identity map so the
    fold still contributes.
    """
    kx, ky = _bin_median_knots(anchor_tr, y_tr, n_bins)
    if len(kx) < 2:
        # degenerate: not enough distinct control points -> identity
        pred = np.clip(anchor_va, Y_CLAMP_MIN, Y_CLAMP_MAX)
        return pred.astype(np.float64), int(len(kx))

    spline = PchipInterpolator(kx, ky, extrapolate=False)
    # clamp val anchors to the control-x envelope (no extrapolation)
    xv = np.clip(anchor_va, kx[0], kx[-1])
    pred = spline(xv)
    # PchipInterpolator with extrapolate=False returns nan exactly at points
    # outside [kx[0], kx[-1]]; clamping above prevents that, but guard anyway.
    pred = np.where(np.isfinite(pred), pred, np.clip(anchor_va, kx[0], kx[-1]))
    pred = np.clip(pred, Y_CLAMP_MIN, Y_CLAMP_MAX)
    return pred.astype(np.float64), int(len(kx))


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- monotonic cubic-spline (PCHIP) calibration on nb3090")
    print(f"          anchor = nb3090_pred_oof.npy (q-cut finer, ref {NB3090_REF})")
    print(f"          ~{N_BINS} bin-median control points -> smooth monotone spline")
    print(f"          y envelope clamp [{Y_CLAMP_MIN}, {Y_CLAMP_MAX}]")
    print(f"          scaffold CV n_folds={N_FOLDS}  {len(KF_SEEDS)} seeds "
          f"{KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"          gate BETTER strict < {GATE_BETTER}")
    print("=" * 78)

    # ---- Load test + truth + scaffolds ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Load anchor (nb3090) ----
    if not ANCHOR_OOF_PATH.exists():
        raise FileNotFoundError(f"missing anchor OOF: {ANCHOR_OOF_PATH}")
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing anchor te: {ANCHOR_TE_PATH}")
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if anchor_oof.shape != (n_unb,):
        raise ValueError(f"anchor_oof shape {anchor_oof.shape} != ({n_unb},)")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"anchor_te shape {anchor_te.shape} != ({n_test},)")
    rae_anchor_oof = float(rae(y_unb, anchor_oof))
    print(f"[load] nb3090 anchor_oof RAE = {rae_anchor_oof:.4f} (ref {NB3090_REF:.4f})")
    print(f"       anchor_oof mean={anchor_oof.mean():.3f}  "
          f"std={anchor_oof.std():.3f}  (truth_std {y_unb.std():.3f})")
    print(f"       anchor_te  mean={anchor_te.mean():.3f}  std={anchor_te.std():.3f}")

    if CHEMPROP_AUX_TE_PATH.exists():
        chemprop_te = np.load(CHEMPROP_AUX_TE_PATH).astype(np.float64)
        rae_chemprop = float(rae(y_unb, chemprop_te[unb_idx]))
        print(f"[diag] chemprop_aux te[unb_idx] in_RAE = {rae_chemprop:.4f} "
              f"(ref {CHEMPROP_AUX_REF:.4f})")

    # ============================================================
    # STEP 1: per-seed scaffold cross-fit monotone-spline calibration
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STEP 1: scaffold {N_FOLDS}-fold CV  PCHIP bin-median spline  "
          f"({len(KF_SEEDS)} seeds)")
    print("-" * 78)

    per_seed_results = []
    per_seed_pooled = []
    per_seed_oof_blobs = []

    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        spline_oof = np.full(n_unb, np.nan, dtype=np.float64)
        per_fold_knots = []
        for k, (tr, va) in enumerate(splits):
            pred_va, n_knots = calibrate_fold(
                anchor_oof[tr], y_unb[tr], anchor_oof[va], N_BINS
            )
            spline_oof[va] = pred_va
            per_fold_knots.append(n_knots)

        if np.isnan(spline_oof).any():
            raise RuntimeError(
                "scaffold splits did not cover all rows; check protocol"
            )

        pooled = float(rae(y_unb, spline_oof))
        per_seed_results.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "per_fold_knots": per_fold_knots,
            "mean_knots": float(np.mean(per_fold_knots)),
        })
        per_seed_pooled.append(pooled)
        per_seed_oof_blobs.append(spline_oof.copy())
        print(f"   kf_seed={kf_seed:5d}  pooled={pooled:.4f}  "
              f"knots={per_fold_knots}")

    per_fold_mean = float(np.mean(per_seed_pooled))
    std_rae = float(np.std(per_seed_pooled))
    min_rae = float(np.min(per_seed_pooled))
    max_rae = float(np.max(per_seed_pooled))
    print(f"\n[eval] per-fold-mean (mean over {len(KF_SEEDS)} seeds) = "
          f"{per_fold_mean:.4f} +/- {std_rae:.4f}  "
          f"[min {min_rae:.4f}, max {max_rae:.4f}]")

    # ---- Compose deploy pred_oof = mean of per-seed spline_oof vectors ----
    pred_oof_avg = np.mean(np.stack(per_seed_oof_blobs, axis=0), axis=0)
    pred_oof_avg_rae = float(rae(y_unb, pred_oof_avg))
    print(f"[eval] avg-of-seeds pooled RAE on 253 = {pred_oof_avg_rae:.4f}")

    # ============================================================
    # STEP 2: gate
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 2: gate")
    print("-" * 78)
    verdict = "BETTER" if per_fold_mean < GATE_BETTER else "FAIL"
    print(f"[gate] per_fold_mean={per_fold_mean:.4f}  "
          f"(strict < {GATE_BETTER} -> BETTER)  -> {verdict}")

    # ============================================================
    # STEP 3: deploy (bin-median PCHIP on full 253; transform te_nb3090 513)
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 3: deploy (PCHIP bin-median on full 253)")
    print("-" * 78)
    kx_d, ky_d = _bin_median_knots(anchor_oof, y_unb, N_BINS)
    n_knots_deploy = int(len(kx_d))
    if n_knots_deploy < 2:
        raise RuntimeError("deploy spline degenerate (<2 control points)")
    spline_d = PchipInterpolator(kx_d, ky_d, extrapolate=False)

    def _apply(a):
        xv = np.clip(a, kx_d[0], kx_d[-1])
        p = spline_d(xv)
        p = np.where(np.isfinite(p), p, xv)
        return np.clip(p, Y_CLAMP_MIN, Y_CLAMP_MAX).astype(np.float64)

    in_oof_deploy = _apply(anchor_oof)
    in_rae_deploy = float(rae(y_unb, in_oof_deploy))
    te_final = _apply(anchor_te)
    te_unb_in = float(rae(y_unb, te_final[unb_idx]))
    print(f"   deploy n_knots            = {n_knots_deploy}")
    print(f"   deploy control x          = "
          f"[{kx_d[0]:.3f} .. {kx_d[-1]:.3f}]")
    print(f"   deploy control y          = "
          f"[{ky_d[0]:.3f} .. {ky_d[-1]:.3f}]")
    print(f"   deploy in-sample RAE      = {in_rae_deploy:.4f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")
    print(f"   te_final mean={te_final.mean():.3f}  std={te_final.std():.3f}  "
          f"(nb3090 te was {anchor_te.mean():.3f}/{anchor_te.std():.3f})")

    # ============================================================
    # STEP 4: save artifacts
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 4: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_avg.astype(np.float32))
    np.save(te_path, te_final.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_monotonic_spline.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_final.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip submission] verdict=FAIL")

    delta_vs_nb3090 = per_fold_mean - NB3090_REF
    delta_vs_nb3080 = per_fold_mean - NB3080_REF
    delta_vs_nb2171 = per_fold_mean - NB2171_REF
    print(f"\n   delta vs nb3090 winner ({NB3090_REF:.4f}) = {delta_vs_nb3090:+.4f}")
    print(f"   delta vs nb3080 ({NB3080_REF:.4f}) = {delta_vs_nb3080:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_REF:.4f}) = {delta_vs_nb2171:+.4f}")

    # ---- summary JSON ----
    summary = {
        "tag": TAG,
        "parent_tag": "nb3090",
        "method": "monotonic_pchip_spline_bin_median_calibration",
        "paradigm": "post_hoc_smooth_monotone_spline_vs_isotonic_step",
        "anchor": "nb3090 q-cut finer (15-seed mean 0.4472)",
        "anchor_pre_unblind": True,
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_oof_rae": rae_anchor_oof,
        "n_bins": N_BINS,
        "y_clamp_min": Y_CLAMP_MIN,
        "y_clamp_max": Y_CLAMP_MAX,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed_results,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_fold_mean": per_fold_mean,
        "mean_rae": per_fold_mean,
        "std_rae": std_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "pred_oof_avg_rae": pred_oof_avg_rae,
        "deploy_n_knots": n_knots_deploy,
        "deploy_control_x": kx_d.tolist(),
        "deploy_control_y": ky_d.tolist(),
        "in_sample_deploy_rae": in_rae_deploy,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_final.mean()),
        "te_std": float(te_final.std()),
        "anchor_te_mean": float(anchor_te.mean()),
        "anchor_te_std": float(anchor_te.std()),
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "nb3090_ref": NB3090_REF,
        "delta_vs_nb3090": delta_vs_nb3090,
        "nb3080_ref": NB3080_REF,
        "delta_vs_nb3080": delta_vs_nb3080,
        "nb2171_ref": NB2171_REF,
        "delta_vs_nb2171": delta_vs_nb2171,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv) if verdict != "FAIL" else None),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor              = nb3090 (oof_RAE {rae_anchor_oof:.4f})")
    print(f"   per-fold-mean       = {per_fold_mean:.4f} +/- {std_rae:.4f}")
    print(f"   deploy n_knots      = {n_knots_deploy}")
    print(f"   verdict             = {verdict}")
    print(f"   delta vs nb3090     = {delta_vs_nb3090:+.4f}")
    print(f"   delta vs nb2171     = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] in-RAE  = {te_unb_in:.4f}")
    print(f"   wall                = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean",
        "std_rae",
        "deploy_n_knots",
        "verdict",
        "delta_vs_nb3090",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
