"""
nb1942 — Blend nb1934 (LGBM Huber 10-outer BoB) + nb1861 (LGBM MSE 25-bag pooled).

Protocol:
1. Load nb1934_bob_mean_oof.npy (0.5017) and nb1861_pooled_25bag_oof.npy (0.5013).
2. Pearson correlation.
3. Grid w in {0.0..1.0 step 0.05}.
4. SLSQP cross-fit (5-fold).
5. Verdict at 0.003 threshold vs nb1861 (0.5013).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

REPO = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = REPO / "data" / "processed"


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    return float(num / den)


def load_unblind_truth() -> np.ndarray:
    """Load the 253 unblind truth values used by other ladder candidates."""
    for cand in [
        "_audit_unblind_y.npy",
        "unblind_truth.npy",
        "y_unblind_true.npy",
        "y_unblind.npy",
        "unblind_y.npy",
    ]:
        p = PROC / cand
        if p.exists():
            return np.load(p)
    raise FileNotFoundError("No unblind truth array found in data/processed/")


def main() -> dict:
    p1 = PROC / "nb1934_bob_mean_oof.npy"
    p2 = PROC / "nb1861_pooled_25bag_oof.npy"

    a = np.load(p1).astype(float)
    b = np.load(p2).astype(float)
    print(f"nb1934 BoB mean OOF: shape={a.shape}, mean={a.mean():.4f}, std={a.std():.4f}")
    print(f"nb1861 pooled 25bag OOF: shape={b.shape}, mean={b.mean():.4f}, std={b.std():.4f}")

    y = load_unblind_truth().astype(float)
    if y.shape[0] != a.shape[0]:
        raise ValueError(f"truth shape {y.shape} != oof shape {a.shape}")
    print(f"truth: shape={y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")

    r1 = rae(y, a)
    r2 = rae(y, b)
    print(f"\nStandalone RAE:")
    print(f"  nb1934 (Huber BoB): {r1:.4f}")
    print(f"  nb1861 (MSE 25bag): {r2:.4f}")

    pear = pearsonr(a, b)
    print(f"\nPearson(nb1934, nb1861): r={pear.statistic:.4f}  p={pear.pvalue:.2e}")

    # ---------- Grid search ----------
    ws = np.arange(0.0, 1.0 + 1e-9, 0.05)
    grid = []
    for w in ws:
        blend = w * a + (1.0 - w) * b
        grid.append((float(w), rae(y, blend)))
    grid_best = min(grid, key=lambda t: t[1])
    print(f"\nGrid sweep (w on nb1934, 1-w on nb1861):")
    for w, r in grid:
        marker = " <-- best" if (w, r) == grid_best else ""
        print(f"  w={w:.2f} -> RAE={r:.4f}{marker}")

    # ---------- SLSQP cross-fit (5-fold) ----------
    def neg_rae(w_vec, A, B, Y):
        w = float(np.clip(w_vec[0], 0.0, 1.0))
        return rae(Y, w * A + (1.0 - w) * B)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    blend_oof = np.zeros_like(y)
    fold_ws = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(a)):
        res = minimize(
            neg_rae,
            x0=[0.5],
            args=(a[tr_idx], b[tr_idx], y[tr_idx]),
            method="SLSQP",
            bounds=[(0.0, 1.0)],
            options={"ftol": 1e-8, "maxiter": 200},
        )
        w_fold = float(np.clip(res.x[0], 0.0, 1.0))
        fold_ws.append(w_fold)
        blend_oof[va_idx] = w_fold * a[va_idx] + (1.0 - w_fold) * b[va_idx]
        print(f"  fold {fold}: w(nb1934)={w_fold:.4f}, train RAE={res.fun:.4f}")

    slsqp_rae = rae(y, blend_oof)
    print(f"\nSLSQP cross-fit RAE: {slsqp_rae:.4f}")
    print(f"Mean fold weight on nb1934: {np.mean(fold_ws):.4f}")

    # ---------- Verdict ----------
    baseline = r2  # nb1861
    delta_grid = grid_best[1] - baseline
    delta_slsqp = slsqp_rae - baseline
    threshold = -0.003  # improvement of >= 0.003

    print("\n========== VERDICT ==========")
    print(f"Baseline nb1861 RAE: {baseline:.4f}")
    print(f"Grid best  RAE: {grid_best[1]:.4f}  (delta {delta_grid:+.4f})")
    print(f"SLSQP CF   RAE: {slsqp_rae:.4f}  (delta {delta_slsqp:+.4f})")
    print(f"Threshold: improve by >= 0.003")

    verdict_grid = "PASS" if delta_grid <= threshold else "FAIL"
    verdict_slsqp = "PASS" if delta_slsqp <= threshold else "FAIL"
    print(f"Grid verdict:  {verdict_grid}")
    print(f"SLSQP verdict: {verdict_slsqp}")

    summary = {
        "nb1934_standalone_rae": r1,
        "nb1861_standalone_rae": r2,
        "pearson_r": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "grid": grid,
        "grid_best_w_nb1934": grid_best[0],
        "grid_best_rae": grid_best[1],
        "slsqp_fold_weights_nb1934": fold_ws,
        "slsqp_mean_w_nb1934": float(np.mean(fold_ws)),
        "slsqp_crossfit_rae": slsqp_rae,
        "baseline_nb1861_rae": baseline,
        "delta_grid_vs_baseline": delta_grid,
        "delta_slsqp_vs_baseline": delta_slsqp,
        "threshold_improvement": 0.003,
        "verdict_grid": verdict_grid,
        "verdict_slsqp": verdict_slsqp,
    }

    out_json = PROC / "nb1942_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    np.save(PROC / "nb1942_blend_oof.npy", blend_oof)
    print(f"Wrote {PROC / 'nb1942_blend_oof.npy'}")

    return summary


if __name__ == "__main__":
    main()
