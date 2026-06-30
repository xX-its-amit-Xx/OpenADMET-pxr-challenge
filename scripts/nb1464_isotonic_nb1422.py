"""nb1464 -- Isotonic monotone calibration on nb1422 bob_median OOF.

Hypothesis:
    nb1323 isotonic on nb1290 hurt by +0.024. nb1422 is a DIFFERENT predictor
    (3-way pruned BoB median blend, RAE 0.5016) -- may respond differently to
    monotone isotonic calibration. Apply the same recipe to the new floor.

Protocol:
  1. Load nb1422_bob_median_oof (0.5016).
  2. 5-fold cross-fit IsotonicRegression(out_of_bounds='clip'). Per-fold map.
  3. 5-seed bag (KFold seed in {42, 43, 44, 45, 46}).
  4. Pool mean-bag RAE; verdict at 0.003 margin vs nb1422 (0.5016).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1464_mean_bag_oof.npy   (253,) float32 -- mean across 5 seeds
  data/processed/nb1464_summary.json
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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1464"
N_FOLDS = 5
SEEDS = [42, 43, 44, 45, 46]

# nb1422 bob_median reference (3-way pruned BoB blend).
NB1422_REF = 0.5016
MARGIN = 0.003


def _isotonic_cross_fit(p: np.ndarray, y: np.ndarray,
                        n_folds: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """Per-fold IsotonicRegression(y_min, y_max, out_of_bounds='clip')."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    n_thresholds: list[int] = []
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        iso = IsotonicRegression(y_min=y_min, y_max=y_max,
                                 increasing=True, out_of_bounds="clip")
        iso.fit(p[tr_loc], y[tr_loc])
        oof[va_loc] = iso.predict(p[va_loc])
        n_thresholds.append(int(len(iso.X_thresholds_)))
    return oof, n_thresholds


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Isotonic calibration on nb1422 bob_median OOF "
          f"(5-fold cross-fit, 5-seed bag)")
    print(f"          verdict margin {MARGIN} vs nb1422 ({NB1422_REF:.4f})")
    print("=" * 78)

    # ---- Load anchors ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    p_path = DATA_PROCESSED / "nb1422_bob_median_oof.npy"
    if not p_path.exists():
        raise FileNotFoundError(f"{p_path} not found")
    p_oof = np.load(p_path).astype(np.float64)
    if p_oof.shape[0] != n_unb:
        raise ValueError(f"shape mismatch: nb1422_bob_median_oof={p_oof.shape}, "
                         f"n_unb={n_unb}")

    nb1422_rae = float(rae(y_unb, p_oof))
    print(f"\n[load] nb1422 bob_median OOF standalone RAE = {nb1422_rae:.4f}  "
          f"(ref {NB1422_REF:.4f})")
    print(f"[diag] pred_std = {p_oof.std():.4f}  truth_std = {y_unb.std():.4f}  "
          f"ratio = {p_oof.std()/y_unb.std():.4f}")

    # ---- 5-seed bag of 5-fold isotonic cross-fits ----
    print("\n" + "-" * 78)
    print("  BLOCK: 5-seed bag of 5-fold isotonic cross-fits")
    print("-" * 78)

    per_seed_oofs: list[np.ndarray] = []
    per_seed_rae: list[float] = []
    per_seed_thresholds: list[list[int]] = []
    for seed in SEEDS:
        oof_s, n_thr = _isotonic_cross_fit(p_oof, y_unb, N_FOLDS, seed)
        r = float(rae(y_unb, oof_s))
        per_seed_oofs.append(oof_s)
        per_seed_rae.append(r)
        per_seed_thresholds.append(n_thr)
        print(f"   seed {seed}: pooled RAE = {r:.4f}   "
              f"n_thresholds per fold = {n_thr}   "
              f"avg = {float(np.mean(n_thr)):.1f}")

    # ---- Mean-bag pooled RAE ----
    stacked = np.column_stack(per_seed_oofs)
    mean_bag_oof = stacked.mean(axis=1)
    median_bag_oof = np.median(stacked, axis=1)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    print(f"\n[bag] mean-bag pooled RAE   = {rae_mean_bag:.4f}")
    print(f"[bag] median-bag pooled RAE = {rae_median_bag:.4f}")

    # ---- Avg breakpoints across all 25 folds ----
    flat_thr = [t for seed_thr in per_seed_thresholds for t in seed_thr]
    avg_n_thresholds = float(np.mean(flat_thr))
    print(f"[diag] isotonic curve avg n_thresholds (across 25 folds) = "
          f"{avg_n_thresholds:.1f}")
    print(f"[diag] calibrated pred_std (mean-bag) = {mean_bag_oof.std():.4f}  "
          f"(was {p_oof.std():.4f}, truth {y_unb.std():.4f})")

    # ---- Verdict ----
    beats_nb1422 = rae_mean_bag < NB1422_REF - MARGIN
    flat_nb1422 = abs(rae_mean_bag - NB1422_REF) < MARGIN
    delta_vs_nb1422 = rae_mean_bag - NB1422_REF

    if beats_nb1422:
        verdict = (f"ISOTONIC_NB1422_BEATS_NB1422 "
                   f"(mean_bag @ {rae_mean_bag:.4f})")
    elif flat_nb1422:
        verdict = (f"ISOTONIC_NB1422_FLAT_VS_NB1422 "
                   f"(mean_bag @ {rae_mean_bag:.4f})")
    else:
        verdict = (f"ISOTONIC_NB1422_HURTS_VS_NB1422 "
                   f"(mean_bag @ {rae_mean_bag:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1422 standalone (anchor)  : {nb1422_rae:.4f}  "
          f"(ref {NB1422_REF:.4f})")
    print(f"   per-seed pooled RAE         : "
          f"{['%.4f' % r for r in per_seed_rae]}")
    print(f"   per-seed mean +/- std       : {np.mean(per_seed_rae):.4f}  "
          f"+/- {np.std(per_seed_rae):.4f}")
    print(f"   mean-bag pooled RAE         : {rae_mean_bag:.4f}")
    print(f"   median-bag pooled RAE       : {rae_median_bag:.4f}")
    print(f"   delta vs nb1422             : {delta_vs_nb1422:+.4f}")
    print(f"   isotonic avg n_thresholds   : {avg_n_thresholds:.1f}")
    print(f"   beats_nb1422 (>= {MARGIN})    : {beats_nb1422}")
    print(f"   verdict                     : {verdict}")

    # ---- Persist artifacts ----
    out_oof = DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy"
    np.save(out_oof, mean_bag_oof.astype(np.float32))
    print(f"\n[save] {out_oof}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "anchor": "nb1422_bob_median_oof",
        "anchor_rae_standalone": nb1422_rae,
        "anchor_pred_std": float(p_oof.std()),
        "truth_std": float(y_unb.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_rae_mean": float(np.mean(per_seed_rae)),
        "per_seed_rae_std": float(np.std(per_seed_rae)),
        "per_seed_n_thresholds": per_seed_thresholds,
        "avg_n_thresholds": avg_n_thresholds,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "calibrated_mean_bag_std": float(mean_bag_oof.std()),
        "nb1422_ref": NB1422_REF,
        "delta_vs_nb1422": delta_vs_nb1422,
        "beats_nb1422": bool(beats_nb1422),
        "flat_vs_nb1422": bool(flat_nb1422),
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
    for k in ("anchor_rae_standalone",
              "anchor_pred_std", "truth_std",
              "per_seed_rae",
              "per_seed_rae_mean", "per_seed_rae_std",
              "avg_n_thresholds",
              "rae_mean_bag", "rae_median_bag",
              "calibrated_mean_bag_std",
              "delta_vs_nb1422",
              "beats_nb1422", "verdict"):
        print(f"  {k}: {res.get(k)}")
