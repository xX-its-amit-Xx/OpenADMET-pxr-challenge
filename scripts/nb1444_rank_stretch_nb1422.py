"""nb1444 -- RANK STRETCH grid on nb1422 BoB MEDIAN (the new floor).

CONTEXT: rank-stretch failed on nb1290 (nb1330). Test if nb1422 (which is a
3-way blend of pruned learners) responds differently.

Recipe: stretched[i] = mu + s * (pred[i] - mu), with mu = mean(pred).

Protocol:
  1. Load nb1422 BoB median OOF (expected pooled RAE 0.5016).
  2. Compute pred_std vs truth_std on the 253.
  3. In-sample grid: s in {0.95, 1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15,
     1.20, 1.25}. Pool RAE per s.
  4. 5-fold cross-fit s grid: per training fold, pick best s on training
     rows, apply to held-out rows.
  5. Verdict at 0.003 margin vs nb1422 (0.5016).

Outputs:
  data/processed/nb1444_summary.json
  data/processed/nb1444_best_s_oof.npy   -- in-sample stretched OOF (253,)
  data/processed/nb1444_cf_oof.npy       -- cross-fit stretched OOF (253,)
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

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1444"
SEED = 0
N_FOLDS = 5
STRETCH_GRID = [0.95, 1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.20, 1.25]
NB1422_REF = 0.5016
MARGIN = 0.003


def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def _best_s_on(p_tr: np.ndarray, y_tr: np.ndarray, grid) -> tuple[float, float, float]:
    """Pick s minimising RAE on (p_tr, y_tr); mu fit from p_tr. Returns (s*, mu, best_rae)."""
    mu = float(p_tr.mean())
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = _stretch(p_tr, mu, s)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, mu, best_r


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RANK STRETCH grid on nb1422 BoB MEDIAN (new floor)")
    print(f"          grid={STRETCH_GRID}")
    print(f"          margin={MARGIN} vs nb1422 (ref {NB1422_REF:.4f})")
    print("=" * 78)

    # ---- Load truth ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] unblind y: n={n_unb}  mean={y_unb.mean():.4f}  "
          f"std={y_unb.std():.4f}")

    # ---- Load nb1422 BoB median OOF ----
    p_path = DATA_PROCESSED / "nb1422_bob_median_oof.npy"
    if not p_path.exists():
        raise FileNotFoundError(f"{p_path} not found")
    p_oof = np.load(p_path).astype(np.float64)
    assert p_oof.shape == (n_unb,), f"shape mismatch: {p_oof.shape}"

    mu_global = float(p_oof.mean())
    sigma_p = float(p_oof.std())
    sigma_y = float(y_unb.std())
    var_match_s = sigma_y / max(sigma_p, 1e-9)

    rae_nb1422 = float(rae(y_unb, p_oof))
    print(f"\n[diag] nb1422 OOF pooled RAE = {rae_nb1422:.4f}  "
          f"(ref {NB1422_REF:.4f})")
    print(f"[diag] pred_mean (mu)      = {mu_global:.4f}")
    print(f"[diag] pred_std  (sigma_p) = {sigma_p:.4f}")
    print(f"[diag] truth_std (sigma_y) = {sigma_y:.4f}")
    print(f"[diag] ratio sigma_p/sigma_y = {sigma_p / sigma_y:.4f}  "
          f"(<1 = compressed)")
    print(f"[diag] variance-match s    = {var_match_s:.4f}")

    # Verify reference
    if abs(rae_nb1422 - NB1422_REF) > 5e-4:
        print(f"  WARN: rae_nb1422 ({rae_nb1422:.4f}) "
              f"differs from reference {NB1422_REF:.4f} by "
              f">5e-4 ({rae_nb1422 - NB1422_REF:+.4f}). Proceeding with actual.")

    # =================================================================
    # (A) In-sample grid (mu = global mean of all 253)
    # =================================================================
    print("\n" + "-" * 78)
    print("(A) IN-SAMPLE GRID (mu fit on all 253; s sweep)")
    print("-" * 78)
    grid_in_sample = []
    best_s_is = 1.0
    best_rae_is = float("inf")
    best_oof_is = p_oof.copy()
    for s in STRETCH_GRID:
        pred = _stretch(p_oof, mu_global, s)
        r = float(rae(y_unb, pred))
        marker = ""
        if r < best_rae_is:
            best_rae_is = r
            best_s_is = float(s)
            best_oof_is = pred.copy()
            marker = "  <-- best so far"
        grid_in_sample.append({"s": float(s), "rae": r})
        print(f"   s={s:.2f}  RAE={r:.4f}{marker}")
    print(f"   best in-sample: s*={best_s_is:.2f}  RAE={best_rae_is:.4f}")

    # =================================================================
    # (B) 5-fold cross-fit: per training fold pick best s
    # =================================================================
    print("\n" + "-" * 78)
    print(f"(B) 5-FOLD CROSS-FIT GRID  seed={SEED}")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_cf = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        s_star, mu_tr, rae_tr = _best_s_on(p_oof[tr_i], y_unb[tr_i], STRETCH_GRID)
        oof_cf[va_i] = _stretch(p_oof[va_i], mu_tr, s_star)
        r_va = float(rae(y_unb[va_i], oof_cf[va_i]))
        fold_records.append({
            "fold": int(fold),
            "n_tr": int(len(tr_i)),
            "n_va": int(len(va_i)),
            "s_star": float(s_star),
            "mu_tr": float(mu_tr),
            "rae_tr": float(rae_tr),
            "rae_va": float(r_va),
        })
        print(f"   fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"s*={s_star:.2f}  mu_tr={mu_tr:.4f}  "
              f"RAE_tr={rae_tr:.4f}  RAE_va={r_va:.4f}")
    assert not np.any(np.isnan(oof_cf)), "cross-fit oof has NaNs"
    rae_cf = float(rae(y_unb, oof_cf))
    fold_per_va_raes = [r["rae_va"] for r in fold_records]
    fold_s_stars = [r["s_star"] for r in fold_records]
    print(f"\n   pooled CROSS-FIT RAE = {rae_cf:.4f}  "
          f"(per-fold va: {min(fold_per_va_raes):.4f}..{max(fold_per_va_raes):.4f})")
    print(f"   per-fold s*: {fold_s_stars}")

    # =================================================================
    # Verdict
    # =================================================================
    candidates = {
        "nb1422_passthrough_s1.00": rae_nb1422,
        "in_sample_best":           best_rae_is,
        "cross_fit":                rae_cf,
    }
    best_tag = min(candidates, key=candidates.get)
    best_rae_candidate = candidates[best_tag]

    # Honest evaluation: only the cross-fit value matters for LB.
    delta_cf_vs_nb1422 = rae_cf - rae_nb1422
    delta_is_vs_nb1422 = best_rae_is - rae_nb1422

    beats_nb1422_cf = rae_cf < rae_nb1422 - MARGIN
    flat_nb1422_cf = abs(rae_cf - rae_nb1422) < MARGIN
    beats_nb1422_is = best_rae_is < rae_nb1422 - MARGIN

    if beats_nb1422_cf:
        verdict = (f"STRETCH_NB1422_BEATS_NB1422_CROSSFIT  "
                   f"(cf RAE {rae_cf:.4f}, delta {delta_cf_vs_nb1422:+.4f})")
    elif flat_nb1422_cf:
        verdict = (f"STRETCH_NB1422_FLAT_VS_NB1422_CROSSFIT  "
                   f"(cf RAE {rae_cf:.4f}, delta {delta_cf_vs_nb1422:+.4f})")
    else:
        verdict = (f"STRETCH_NB1422_HURTS_VS_NB1422_CROSSFIT  "
                   f"(cf RAE {rae_cf:.4f}, delta {delta_cf_vs_nb1422:+.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1422 standalone (s=1.00) : {rae_nb1422:.4f}  "
          f"(ref {NB1422_REF:.4f})")
    print(f"   best in-sample s={best_s_is:.2f} : {best_rae_is:.4f}  "
          f"(delta {delta_is_vs_nb1422:+.4f})")
    print(f"   5-fold cross-fit           : {rae_cf:.4f}  "
          f"(delta {delta_cf_vs_nb1422:+.4f})")
    print(f"   margin                     : {MARGIN}")
    print(f"   beats nb1422 cross-fit     : {beats_nb1422_cf}")
    print(f"   beats nb1422 in-sample     : {beats_nb1422_is}")
    print(f"   verdict                    : {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_s_oof.npy",
            best_oof_is.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_cf_oof.npy",
            oof_cf.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_s_oof.npy'}  (in-sample stretched)")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_cf_oof.npy'}      (cross-fit stretched)")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "nb1422_ref": NB1422_REF,
        "margin": MARGIN,
        "rae_nb1422_actual": rae_nb1422,
        "pred_mean": mu_global,
        "pred_std": sigma_p,
        "truth_std": sigma_y,
        "pred_std_over_truth_std": sigma_p / sigma_y,
        "var_match_s": var_match_s,
        "in_sample_grid": grid_in_sample,
        "in_sample_best_s": float(best_s_is),
        "in_sample_best_rae": float(best_rae_is),
        "fold_records": fold_records,
        "cf_per_fold_s": [float(x) for x in fold_s_stars],
        "cf_per_fold_va_rae": [float(x) for x in fold_per_va_raes],
        "rae_cross_fit": float(rae_cf),
        "candidate_rae_table": candidates,
        "best_candidate_tag": best_tag,
        "best_candidate_rae": float(best_rae_candidate),
        "delta_cf_vs_nb1422": float(delta_cf_vs_nb1422),
        "delta_in_sample_vs_nb1422": float(delta_is_vs_nb1422),
        "beats_nb1422_cross_fit": bool(beats_nb1422_cf),
        "beats_nb1422_in_sample": bool(beats_nb1422_is),
        "flat_vs_nb1422_cross_fit": bool(flat_nb1422_cf),
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
    for k in (
        "rae_nb1422_actual",
        "pred_mean", "pred_std", "truth_std",
        "pred_std_over_truth_std", "var_match_s",
        "in_sample_best_s", "in_sample_best_rae",
        "cf_per_fold_s", "rae_cross_fit",
        "delta_cf_vs_nb1422", "delta_in_sample_vs_nb1422",
        "beats_nb1422_cross_fit", "beats_nb1422_in_sample", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
