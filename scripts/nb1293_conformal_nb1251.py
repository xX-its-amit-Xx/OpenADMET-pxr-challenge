"""nb1293 -- Conformal calibration on nb1251 OOF residuals.

Hypothesis:
    nb1251 best fixed-w blend (0.55*nb1242 + 0.45*nb1211) has pooled RAE 0.5394
    on the 253 unblind set.  The OOF residual distribution may be asymmetric
    (skewed) so a 1-parameter mean shift -- estimated cross-fit via a 5-fold
    split-conformal protocol -- could nudge the pooled RAE down.

    A 2-parameter sim-scaled variant additionally rescales the per-row mean
    correction by a function of `top1_sim` so rows further from training
    receive a more conservative (larger) correction.

    Both calibrators are tested at 0.003 RAE margin vs nb1251 (0.5394).

Protocol:
  1. Load nb1251 best-fixed-w OOF (nb1251_bestw_oof.npy).
  2. Compute residuals r = y_unb - pred. Report mean, median, std, skew.
  3. Split-conformal mean-shift: per fold, compute mean(r_train) and add it
     to held-out predictions; pool RAE.
  4. Sim-scaled mean-shift: per fold, fit shift = a + b * z(sim_top1)
     with OLS on (r_train, sim_top1_train); apply to held-out rows.
  5. Verdict at 0.003 margin vs nb1251 (0.5394).

Outputs:
  data/processed/nb1293_shift_oof.npy       (253,) float32
  data/processed/nb1293_sim_scaled_oof.npy  (253,) float32
  data/processed/nb1293_summary.json
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
from scipy import stats as sp_stats
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1293"
N_FOLDS = 5
SEED = 42

NB1251_REF = 0.5394
MARGIN = 0.003


def _conformal_shift_cv(pred: np.ndarray, y: np.ndarray,
                        n_splits: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    """Per-fold: shift = mean(y_tr - pred_tr); apply to validation rows.

    Cross-fit so each row's correction is computed from data that
    excludes it -- this avoids the in-sample-overfit bias.
    """
    n = len(y)
    corrected = np.full(n, np.nan, dtype=np.float64)
    records: list[dict] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        r_tr = y[tr_loc] - pred[tr_loc]
        shift = float(r_tr.mean())
        corrected[va_loc] = pred[va_loc] + shift
        records.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "shift": shift,
        })
    return corrected, records


def _conformal_sim_scaled_cv(pred: np.ndarray, y: np.ndarray,
                             sim: np.ndarray,
                             n_splits: int, seed: int
                             ) -> tuple[np.ndarray, list[dict]]:
    """Per-fold: fit shift = a + b * z(sim) via OLS on training residuals.

    z(sim) is standardized within the training fold to avoid leakage.
    Apply learned (a, b) plus training-fold standardization stats to
    the held-out rows.
    """
    n = len(y)
    corrected = np.full(n, np.nan, dtype=np.float64)
    records: list[dict] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        r_tr = y[tr_loc] - pred[tr_loc]
        s_tr = sim[tr_loc]
        mu_s = float(s_tr.mean())
        sd_s = float(s_tr.std())
        if sd_s < 1e-12:
            z_tr = np.zeros_like(s_tr)
            z_va = np.zeros_like(sim[va_loc])
        else:
            z_tr = (s_tr - mu_s) / sd_s
            z_va = (sim[va_loc] - mu_s) / sd_s

        # OLS: r = a + b * z + eps
        X = np.column_stack([np.ones_like(z_tr), z_tr])
        coef, *_ = np.linalg.lstsq(X, r_tr, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        shift_va = a + b * z_va
        corrected[va_loc] = pred[va_loc] + shift_va
        records.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "a": a,
            "b": b,
            "mu_sim_tr": mu_s,
            "sd_sim_tr": sd_s,
        })
    return corrected, records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- conformal calibration on nb1251 OOF residuals")
    print(f"          1-param mean shift  +  2-param sim-scaled shift")
    print(f"          5-fold split-conformal cross-fit on 253 unblind")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    n_unb = len(y_unb)

    # --- Load nb1251 best-fixed-w OOF predictions ---
    pred_path = DATA_PROCESSED / "nb1251_bestw_oof.npy"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} not found -- run nb1251 first")
    pred = np.load(pred_path).astype(np.float64)
    if pred.shape[0] != n_unb:
        raise ValueError(f"shape mismatch: pred={pred.shape}, n_unb={n_unb}")

    # Sanity check: pooled RAE should match nb1251 best-fixed-w (0.5394).
    rae_base = float(rae(y_unb, pred))
    print(f"\n[load] nb1251 best-fixed-w OOF  RAE = {rae_base:.4f}  "
          f"(ref {NB1251_REF:.4f})")

    # --- Sim-to-train top-1 on the 253 unblind ---
    diff_df = pd.read_parquet(DATA_PROCESSED / "test_difficulty.parquet")
    sim_full = diff_df["top1_sim"].values.astype(np.float64)
    if sim_full.shape[0] != 513:
        raise ValueError(f"test_difficulty rows = {sim_full.shape[0]}, expected 513")
    sim_unb = sim_full[unb_idx]
    print(f"[load] sim_to_train_top1[unb_idx]  n={len(sim_unb)}  "
          f"min={sim_unb.min():.3f}  max={sim_unb.max():.3f}  "
          f"mean={sim_unb.mean():.3f}  median={np.median(sim_unb):.3f}")

    # --- Residual distribution diagnostic ---
    r = y_unb - pred
    r_mean = float(r.mean())
    r_median = float(np.median(r))
    r_std = float(r.std())
    r_skew = float(sp_stats.skew(r))
    r_kurt = float(sp_stats.kurtosis(r))
    r_q05 = float(np.quantile(r, 0.05))
    r_q25 = float(np.quantile(r, 0.25))
    r_q75 = float(np.quantile(r, 0.75))
    r_q95 = float(np.quantile(r, 0.95))
    print("\n" + "-" * 78)
    print("  BLOCK: residual distribution (y - pred) on 253 unblind")
    print("-" * 78)
    print(f"   mean    = {r_mean:+.4f}")
    print(f"   median  = {r_median:+.4f}")
    print(f"   std     = {r_std:.4f}")
    print(f"   skew    = {r_skew:+.4f}   (positive => right tail heavier)")
    print(f"   kurtosis= {r_kurt:+.4f}")
    print(f"   q05 / q25 / q75 / q95 = "
          f"{r_q05:+.4f} / {r_q25:+.4f} / {r_q75:+.4f} / {r_q95:+.4f}")

    # --- 1-parameter mean-shift conformal cross-fit ---
    print("\n" + "-" * 78)
    print("  BLOCK: 1-param split-conformal mean shift (cross-fit)")
    print("-" * 78)
    shift_oof, shift_records = _conformal_shift_cv(pred, y_unb, N_FOLDS, SEED)
    rae_shift = float(rae(y_unb, shift_oof))
    shifts = [r["shift"] for r in shift_records]
    print(f"   per-fold shifts: " +
          "  ".join(f"f{r['fold']}={r['shift']:+.4f}" for r in shift_records))
    print(f"   mean shift across folds = {np.mean(shifts):+.4f}")
    print(f"   pooled RAE(shift)      = {rae_shift:.4f}   "
          f"(delta {rae_shift - rae_base:+.4f})")

    # --- 2-parameter sim-scaled conformal cross-fit ---
    print("\n" + "-" * 78)
    print("  BLOCK: 2-param sim-scaled conformal shift (cross-fit)")
    print("-" * 78)
    sim_oof, sim_records = _conformal_sim_scaled_cv(
        pred, y_unb, sim_unb, N_FOLDS, SEED,
    )
    rae_sim = float(rae(y_unb, sim_oof))
    print(f"   per-fold (a, b): " +
          "  ".join(f"f{r['fold']}=(a={r['a']:+.4f},b={r['b']:+.4f})"
                   for r in sim_records))
    print(f"   pooled RAE(sim_scaled) = {rae_sim:.4f}   "
          f"(delta {rae_sim - rae_base:+.4f})")

    # --- Verdict ---
    candidates = {
        "shift":      rae_shift,
        "sim_scaled": rae_sim,
    }
    best_tag = min(candidates, key=candidates.get)
    best_rae_val = candidates[best_tag]
    beats_nb1251 = best_rae_val < rae_base - MARGIN
    flat_nb1251 = abs(best_rae_val - rae_base) < MARGIN

    if beats_nb1251:
        verdict = (f"CONFORMAL_BEATS_NB1251 "
                   f"({best_tag} @ {best_rae_val:.4f})")
    elif flat_nb1251:
        verdict = (f"CONFORMAL_FLAT_VS_NB1251 "
                   f"({best_tag} @ {best_rae_val:.4f})")
    else:
        verdict = (f"CONFORMAL_HURTS_VS_NB1251 "
                   f"({best_tag} @ {best_rae_val:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1251 base                : {rae_base:.4f}  (ref {NB1251_REF:.4f})")
    print(f"   conformal shift            : {rae_shift:.4f}  "
          f"({rae_shift - rae_base:+.4f})")
    print(f"   conformal sim-scaled       : {rae_sim:.4f}  "
          f"({rae_sim - rae_base:+.4f})")
    print(f"   best conformal             : {best_rae_val:.4f}  ({best_tag})")
    print(f"   delta vs nb1251 (0.5394)   : {best_rae_val - rae_base:+.4f}")
    print(f"   beats_nb1251 (>= {MARGIN})    : {beats_nb1251}")
    print(f"   verdict                    : {verdict}")

    # Persist artifacts.
    np.save(DATA_PROCESSED / f"{TAG}_shift_oof.npy",
            shift_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_sim_scaled_oof.npy",
            sim_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_shift_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_sim_scaled_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "anchor": "nb1251_bestw_oof",
        "rae_anchor_base": rae_base,
        "nb1251_ref": NB1251_REF,
        "residual_distribution": {
            "mean":    r_mean,
            "median":  r_median,
            "std":     r_std,
            "skew":    r_skew,
            "kurtosis": r_kurt,
            "q05": r_q05,
            "q25": r_q25,
            "q75": r_q75,
            "q95": r_q95,
        },
        "sim_unb_stats": {
            "min":    float(sim_unb.min()),
            "max":    float(sim_unb.max()),
            "mean":   float(sim_unb.mean()),
            "median": float(np.median(sim_unb)),
        },
        "shift_records":      shift_records,
        "sim_scaled_records": sim_records,
        "rae_shift":      rae_shift,
        "rae_sim_scaled": rae_sim,
        "candidate_rae_table": candidates,
        "best_conformal_tag": best_tag,
        "best_conformal_rae": best_rae_val,
        "delta_best_vs_nb1251": best_rae_val - rae_base,
        "beats_nb1251":   bool(beats_nb1251),
        "flat_vs_nb1251": bool(flat_nb1251),
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
    for k in ("rae_anchor_base",
              "residual_distribution",
              "rae_shift", "rae_sim_scaled",
              "candidate_rae_table",
              "best_conformal_tag", "best_conformal_rae",
              "delta_best_vs_nb1251",
              "beats_nb1251", "verdict"):
        print(f"  {k}: {res.get(k)}")
