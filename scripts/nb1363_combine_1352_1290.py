"""nb1363 -- Combine nb1352 (SHAP-pruned MACCS) + nb1290 (best-w 2-way).

Hypothesis:
    nb1352 (mean 0.5323 / median 0.5315, SHAP-pruned MACCS residual on
    nb1070 anchor) and nb1290 (0.5390, best fixed-w blend of nb1190 BoB +
    nb1242 ChEMBL-feat residual) draw on similar feature signal but differ
    in mechanism (residual on different anchor / kNN-feat vs MACCS-pruned).
    Pairing may yield small variance reduction.

Protocol:
  1. Load nb1352_mean_bag_oof.npy AND nb1352_median_bag_oof.npy.
  2. Load nb1290_bestw_oof.npy.
  3. Compute pairwise Pearson (pred + residual) for {1352_mean, 1352_median, 1290}.
  4. Grid search w in {0.0..1.0 step 0.05} for w*nb1352 + (1-w)*nb1290,
     run for BOTH mean and median nb1352 variants.
  5. 5-fold cross-fit SLSQP (2-way) for BOTH mean and median variants.
  6. Verdict at 0.003 margin vs nb1352 median (0.5315).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1363_bestw_mean_oof.npy     (253,) float32  -- best fixed-w (mean variant)
  data/processed/nb1363_bestw_median_oof.npy   (253,) float32  -- best fixed-w (median variant)
  data/processed/nb1363_slsqp_mean_oof.npy     (253,) float32  -- SLSQP cross-fit (mean variant)
  data/processed/nb1363_slsqp_median_oof.npy   (253,) float32  -- SLSQP cross-fit (median variant)
  data/processed/nb1363_summary.json
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1363"
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Reference numbers (pooled RAE on 253 unblind).
NB1352_MEAN_REF = 0.5323
NB1352_MEDIAN_REF = 0.5315  # primary baseline to beat
NB1290_REF = 0.5390
MARGIN = 0.003


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Argmin MSE over the K-simplex (w_i >= 0, sum w_i = 1)."""
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w: np.ndarray) -> float:
        pred = P_tr @ w
        diff = y_tr - pred
        return float(np.mean(diff * diff))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        _loss, w0, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        return np.full(K, 1.0 / K)
    return w / s


def _slsqp_cross_fit(P: np.ndarray, y: np.ndarray,
                     n_splits: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_blend_weights(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w
        fold_records.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "weights": [float(x) for x in w],
        })
    return oof, fold_records


def _grid_search(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray,
                 name_a: str, name_b: str) -> tuple[list[dict], float, float, np.ndarray]:
    """w*p_a + (1-w)*p_b for w in 0..1 step 0.05. Returns (grid, best_w_a, best_rae, best_oof)."""
    grid = []
    best_w = None
    best_rae = float("inf")
    best_oof = None
    for w in np.arange(0.0, 1.0 + 1e-9, 0.05):
        w = float(round(w, 4))
        blend = w * p_a + (1.0 - w) * p_b
        r = float(rae(y, blend))
        grid.append({f"w_{name_a}": w, f"w_{name_b}": float(round(1.0 - w, 4)),
                     "rae": r})
        marker = ""
        if r < best_rae:
            best_rae = r
            best_w = w
            best_oof = blend
            marker = "  <-- best so far"
        print(f"   w[{name_a}]={w:.2f}  w[{name_b}]={1.0 - w:.2f}  "
              f"RAE={r:.4f}{marker}")
    return grid, best_w, best_rae, best_oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 2-way blend nb1352 (mean/median) + nb1290 (best-w)")
    print(f"          grid w in 0..1 step 0.05  +  5-fold SLSQP cross-fit")
    print(f"          verdict margin {MARGIN} vs nb1352 median ({NB1352_MEDIAN_REF:.4f})")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = {
        "nb1352_mean":   DATA_PROCESSED / "nb1352_mean_bag_oof.npy",
        "nb1352_median": DATA_PROCESSED / "nb1352_median_bag_oof.npy",
        "nb1290":        DATA_PROCESSED / "nb1290_bestw_oof.npy",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({k})")

    preds = {k: np.load(p).astype(np.float64) for k, p in paths.items()}
    for k, v in preds.items():
        if v.shape[0] != n_unb:
            raise ValueError(f"shape mismatch: {k}={v.shape}, n_unb={n_unb}")

    p_mean   = preds["nb1352_mean"]
    p_median = preds["nb1352_median"]
    p_1290   = preds["nb1290"]

    standalone_rae = {
        "nb1352_mean":   float(rae(y_unb, p_mean)),
        "nb1352_median": float(rae(y_unb, p_median)),
        "nb1290":        float(rae(y_unb, p_1290)),
    }
    print("\n[load] standalone pooled RAE on 253 unblind:")
    print(f"   nb1352_mean   : {standalone_rae['nb1352_mean']:.4f}  (ref {NB1352_MEAN_REF:.4f})")
    print(f"   nb1352_median : {standalone_rae['nb1352_median']:.4f}  (ref {NB1352_MEDIAN_REF:.4f})")
    print(f"   nb1290_bestw  : {standalone_rae['nb1290']:.4f}  (ref {NB1290_REF:.4f})")

    # Pairwise Pearson (pred + residual).
    def _pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
        ra = a - y_unb
        rb = b - y_unb
        return {
            "pred_pearson":      float(np.corrcoef(a, b)[0, 1]),
            "residual_pearson":  float(np.corrcoef(ra, rb)[0, 1]),
            "residual_std_a":    float(ra.std()),
            "residual_std_b":    float(rb.std()),
        }

    pair_stats = {
        "mean_vs_median": _pair_stats(p_mean, p_median),
        "mean_vs_1290":   _pair_stats(p_mean, p_1290),
        "median_vs_1290": _pair_stats(p_median, p_1290),
    }
    print("\n[diag] pairwise correlation:")
    for pair, st in pair_stats.items():
        print(f"   {pair:18s}  pred_r={st['pred_pearson']:.4f}  "
              f"resid_r={st['residual_pearson']:.4f}  "
              f"resid_std=({st['residual_std_a']:.4f}, {st['residual_std_b']:.4f})")

    # ---- Grid search: nb1352_mean variant ----
    print("\n" + "-" * 78)
    print("  BLOCK A: fixed-w grid -- w*nb1352_mean + (1-w)*nb1290")
    print("-" * 78)
    grid_mean, best_w_mean, best_rae_mean, best_oof_mean = _grid_search(
        p_mean, p_1290, y_unb, "nb1352_mean", "nb1290")
    print(f"   best fixed-w (mean variant): w[nb1352_mean]={best_w_mean:.4f}  "
          f"w[nb1290]={1.0 - best_w_mean:.4f}  RAE={best_rae_mean:.4f}")

    # ---- Grid search: nb1352_median variant ----
    print("\n" + "-" * 78)
    print("  BLOCK B: fixed-w grid -- w*nb1352_median + (1-w)*nb1290")
    print("-" * 78)
    grid_median, best_w_median, best_rae_median, best_oof_median = _grid_search(
        p_median, p_1290, y_unb, "nb1352_median", "nb1290")
    print(f"   best fixed-w (median variant): w[nb1352_median]={best_w_median:.4f}  "
          f"w[nb1290]={1.0 - best_w_median:.4f}  RAE={best_rae_median:.4f}")

    # ---- SLSQP cross-fit: mean variant ----
    print("\n" + "-" * 78)
    print("  BLOCK C: 2-way SLSQP cross-fit -- (nb1352_mean, nb1290)")
    print("-" * 78)
    P_mean_pair = np.column_stack([p_mean, p_1290])
    slsqp_oof_mean, fold_records_mean = _slsqp_cross_fit(
        P_mean_pair, y_unb, SLSQP_FOLDS, SLSQP_SEED)
    rae_slsqp_mean = float(rae(y_unb, slsqp_oof_mean))
    fold_w_mean = np.array([r["weights"] for r in fold_records_mean])
    mean_w_mean = fold_w_mean.mean(axis=0)
    for rec in fold_records_mean:
        w = rec["weights"]
        print(f"     fold {rec['fold']}: w[nb1352_mean]={w[0]:.4f}  w[nb1290]={w[1]:.4f}")
    print(f"   mean weights:  w[nb1352_mean]={mean_w_mean[0]:.4f}  "
          f"w[nb1290]={mean_w_mean[1]:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit, mean) = {rae_slsqp_mean:.4f}")

    w_full_mean = _slsqp_blend_weights(P_mean_pair, y_unb)
    p_full_mean = P_mean_pair @ w_full_mean
    rae_full_mean = float(rae(y_unb, p_full_mean))
    print(f"   in-sample SLSQP: w[nb1352_mean]={w_full_mean[0]:.4f}  "
          f"w[nb1290]={w_full_mean[1]:.4f}   RAE={rae_full_mean:.4f}")

    # ---- SLSQP cross-fit: median variant ----
    print("\n" + "-" * 78)
    print("  BLOCK D: 2-way SLSQP cross-fit -- (nb1352_median, nb1290)")
    print("-" * 78)
    P_median_pair = np.column_stack([p_median, p_1290])
    slsqp_oof_median, fold_records_median = _slsqp_cross_fit(
        P_median_pair, y_unb, SLSQP_FOLDS, SLSQP_SEED)
    rae_slsqp_median = float(rae(y_unb, slsqp_oof_median))
    fold_w_median = np.array([r["weights"] for r in fold_records_median])
    mean_w_median = fold_w_median.mean(axis=0)
    for rec in fold_records_median:
        w = rec["weights"]
        print(f"     fold {rec['fold']}: w[nb1352_median]={w[0]:.4f}  w[nb1290]={w[1]:.4f}")
    print(f"   mean weights:  w[nb1352_median]={mean_w_median[0]:.4f}  "
          f"w[nb1290]={mean_w_median[1]:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit, median) = {rae_slsqp_median:.4f}")

    w_full_median = _slsqp_blend_weights(P_median_pair, y_unb)
    p_full_median = P_median_pair @ w_full_median
    rae_full_median = float(rae(y_unb, p_full_median))
    print(f"   in-sample SLSQP: w[nb1352_median]={w_full_median[0]:.4f}  "
          f"w[nb1290]={w_full_median[1]:.4f}   RAE={rae_full_median:.4f}")

    # ---- Verdict ----
    candidates = {
        "bestw_mean":      best_rae_mean,
        "bestw_median":    best_rae_median,
        "slsqp_mean":      rae_slsqp_mean,
        "slsqp_median":    rae_slsqp_median,
    }
    best_blend_tag = min(candidates, key=candidates.get)
    best_blend_rae = candidates[best_blend_tag]

    best_standalone_tag = min(standalone_rae, key=standalone_rae.get)
    best_standalone = standalone_rae[best_standalone_tag]

    # Baseline to beat: nb1352 median.
    beats_nb1352 = best_blend_rae < NB1352_MEDIAN_REF - MARGIN
    flat_nb1352 = abs(best_blend_rae - NB1352_MEDIAN_REF) < MARGIN

    if beats_nb1352:
        verdict = (f"COMBINE_1352_1290_BEATS_NB1352 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")
    elif flat_nb1352:
        verdict = (f"COMBINE_1352_1290_FLAT_VS_NB1352 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")
    else:
        verdict = (f"COMBINE_1352_1290_HURTS_VS_NB1352 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1352_mean   standalone : {standalone_rae['nb1352_mean']:.4f}")
    print(f"   nb1352_median standalone : {standalone_rae['nb1352_median']:.4f}  "
          f"(BASELINE)")
    print(f"   nb1290        standalone : {standalone_rae['nb1290']:.4f}")
    print()
    print(f"   candidate pooled RAE table:")
    for tag, val in sorted(candidates.items(), key=lambda kv: kv[1]):
        print(f"     {tag:18s} = {val:.4f}")
    print()
    print(f"   best blend                    : {best_blend_rae:.4f}  ({best_blend_tag})")
    print(f"   delta vs nb1352 median (0.5315): {best_blend_rae - NB1352_MEDIAN_REF:+.4f}")
    print(f"   beats_nb1352 (>= {MARGIN})       : {beats_nb1352}")
    print(f"   verdict                       : {verdict}")

    # Persist canonical artifacts.
    np.save(DATA_PROCESSED / f"{TAG}_bestw_mean_oof.npy",
            best_oof_mean.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bestw_median_oof.npy",
            best_oof_median.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_slsqp_mean_oof.npy",
            slsqp_oof_mean.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_slsqp_median_oof.npy",
            slsqp_oof_median.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bestw_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bestw_median_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_slsqp_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_slsqp_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "components": ["nb1352_mean", "nb1352_median", "nb1290_bestw"],
        "standalone_rae": standalone_rae,
        "pair_stats": pair_stats,
        # Grid (mean variant).
        "grid_mean_results": grid_mean,
        "best_fixed_w_nb1352_mean": best_w_mean,
        "best_fixed_w_nb1290_for_mean": float(1.0 - best_w_mean),
        "rae_best_fixed_w_mean": best_rae_mean,
        # Grid (median variant).
        "grid_median_results": grid_median,
        "best_fixed_w_nb1352_median": best_w_median,
        "best_fixed_w_nb1290_for_median": float(1.0 - best_w_median),
        "rae_best_fixed_w_median": best_rae_median,
        # SLSQP cross-fit (mean variant).
        "slsqp_mean_fold_records": fold_records_mean,
        "slsqp_mean_mean_fold_weights": [float(x) for x in mean_w_mean],
        "slsqp_mean_in_sample_weights": [float(x) for x in w_full_mean],
        "slsqp_mean_in_sample_rae": rae_full_mean,
        "rae_slsqp_cross_fit_mean": rae_slsqp_mean,
        # SLSQP cross-fit (median variant).
        "slsqp_median_fold_records": fold_records_median,
        "slsqp_median_mean_fold_weights": [float(x) for x in mean_w_median],
        "slsqp_median_in_sample_weights": [float(x) for x in w_full_median],
        "slsqp_median_in_sample_rae": rae_full_median,
        "rae_slsqp_cross_fit_median": rae_slsqp_median,
        # Verdict.
        "candidate_rae_table": candidates,
        "best_blend_tag": best_blend_tag,
        "best_blend_rae": best_blend_rae,
        "best_standalone_tag": best_standalone_tag,
        "best_standalone_rae": best_standalone,
        "nb1352_mean_ref":   NB1352_MEAN_REF,
        "nb1352_median_ref": NB1352_MEDIAN_REF,
        "nb1290_ref":        NB1290_REF,
        "delta_best_vs_nb1352_median": best_blend_rae - NB1352_MEDIAN_REF,
        "delta_best_vs_nb1352_mean":   best_blend_rae - NB1352_MEAN_REF,
        "delta_best_vs_nb1290":        best_blend_rae - NB1290_REF,
        "beats_nb1352":  bool(beats_nb1352),
        "flat_vs_nb1352": bool(flat_nb1352),
        "margin": MARGIN,
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
    for k in ("standalone_rae",
              "pair_stats",
              "best_fixed_w_nb1352_mean",   "rae_best_fixed_w_mean",
              "best_fixed_w_nb1352_median", "rae_best_fixed_w_median",
              "slsqp_mean_mean_fold_weights",   "rae_slsqp_cross_fit_mean",
              "slsqp_median_mean_fold_weights", "rae_slsqp_cross_fit_median",
              "candidate_rae_table",
              "best_blend_tag", "best_blend_rae",
              "delta_best_vs_nb1352_median",
              "beats_nb1352", "verdict"):
        print(f"  {k}: {res.get(k)}")
