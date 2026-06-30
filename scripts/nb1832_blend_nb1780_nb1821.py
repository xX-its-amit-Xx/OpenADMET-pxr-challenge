"""
nb1832 — Blend nb1780 BoB + nb1821 25-bag pooled.

Both anchors reach 0.5025-0.5032 on the 253 unblind.
We test simple convex blends and SLSQP cross-fit weighting.

Verdict gate: 0.003 margin vs nb1821 (0.5025).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

A_PATH = PROC / "nb1780_bob_mean_oof.npy"
B_PATH = PROC / "nb1821_pooled_25bag_oof.npy"
TRUTH_PATH = PROC / "_audit_unblind_y.npy"

ANCHOR_RAE = 0.5025  # nb1821
MARGIN = 0.003


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_true - y_pred).sum()
    den = np.abs(y_true - y_true.mean()).sum()
    return float(num / den)


def find_truth() -> np.ndarray:
    candidates = [
        TRUTH_PATH,
        PROC / "unblind_253_truth.npy",
        PROC / "y_unblind_253.npy",
        PROC / "y_unblind.npy",
        PROC / "truth_unblind_253.npy",
    ]
    for p in candidates:
        if p.exists():
            arr = np.load(p)
            if arr.shape[0] == 253:
                return arr.astype(float)
    # Fallback: search for any *unblind*truth* or *truth*unblind* npy of length 253
    for p in PROC.glob("*.npy"):
        try:
            arr = np.load(p)
        except Exception:
            continue
        if arr.ndim == 1 and arr.shape[0] == 253 and "truth" in p.name.lower():
            return arr.astype(float)
    raise FileNotFoundError(
        "Could not locate 253-row unblind truth array under data/processed."
    )


def main() -> None:
    # --- Load ---
    a = np.load(A_PATH).astype(float).reshape(-1)
    b = np.load(B_PATH).astype(float).reshape(-1)
    y = find_truth().reshape(-1)

    assert a.shape == b.shape == y.shape, (
        f"shape mismatch: a={a.shape} b={b.shape} y={y.shape}"
    )
    n = a.shape[0]

    # --- Standalone RAEs ---
    rae_a = rae(y, a)
    rae_b = rae(y, b)

    # --- Pearson ---
    pearson = float(np.corrcoef(a, b)[0, 1])
    resid_a = a - y
    resid_b = b - y
    pearson_resid = float(np.corrcoef(resid_a, resid_b)[0, 1])

    # --- W grid ---
    grid_results = []
    for w in np.arange(0.0, 1.0001, 0.05):
        p = w * a + (1.0 - w) * b
        grid_results.append({"w_nb1780": round(float(w), 2), "rae": rae(y, p)})
    grid_df = pd.DataFrame(grid_results)
    best_row = grid_df.loc[grid_df["rae"].idxmin()]
    best_w = float(best_row["w_nb1780"])
    best_rae = float(best_row["rae"])

    # --- SLSQP cross-fit (5-fold) ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pred_cf = np.full(n, np.nan, dtype=float)
    fold_weights = []
    for fold_i, (tr, te) in enumerate(kf.split(np.arange(n))):
        Atr = np.column_stack([a[tr], b[tr]])
        ytr = y[tr]

        def loss(w: np.ndarray) -> float:
            p = Atr @ w
            return float(np.mean(np.abs(ytr - p)))

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bnds = [(0.0, 1.0), (0.0, 1.0)]
        w0 = np.array([0.5, 0.5])
        res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons)
        w_hat = res.x
        fold_weights.append(w_hat.tolist())
        pred_cf[te] = a[te] * w_hat[0] + b[te] * w_hat[1]

    rae_slsqp_cf = rae(y, pred_cf)

    # SLSQP in-sample for reference
    def loss_all(w: np.ndarray) -> float:
        p = a * w[0] + b * w[1]
        return float(np.mean(np.abs(y - p)))

    res_all = minimize(
        loss_all,
        np.array([0.5, 0.5]),
        method="SLSQP",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},),
    )
    w_in = res_all.x.tolist()
    rae_slsqp_in = rae(y, a * w_in[0] + b * w_in[1])

    # --- Verdict ---
    best_overall = min(best_rae, rae_slsqp_cf)
    delta = ANCHOR_RAE - best_overall
    pass_gate = bool(delta >= MARGIN)
    verdict = "PASS" if pass_gate else "FAIL"

    # --- Report ---
    print("=" * 72)
    print("nb1832 — Blend nb1780 BoB x nb1821 25-bag pooled")
    print("=" * 72)
    print(f"n                       : {n}")
    print(f"RAE nb1780 (BoB mean)   : {rae_a:.4f}")
    print(f"RAE nb1821 (pooled-25)  : {rae_b:.4f}")
    print(f"Pearson(a, b)           : {pearson:.4f}")
    print(f"Pearson(resid_a,resid_b): {pearson_resid:.4f}")
    print("-" * 72)
    print("W grid (w = weight on nb1780, (1-w) on nb1821):")
    for r in grid_results:
        marker = " <-- BEST" if (
            r["w_nb1780"] == best_w and abs(r["rae"] - best_rae) < 1e-12
        ) else ""
        print(f"  w={r['w_nb1780']:.2f}  RAE={r['rae']:.4f}{marker}")
    print("-" * 72)
    print(f"BEST GRID  w={best_w:.2f}  RAE={best_rae:.4f}")
    print(f"SLSQP in-sample  w={w_in}  RAE={rae_slsqp_in:.4f}")
    print(f"SLSQP cross-fit (5-fold)  RAE={rae_slsqp_cf:.4f}")
    print(f"Per-fold weights        : {fold_weights}")
    print("-" * 72)
    print(f"Anchor (nb1821) RAE     : {ANCHOR_RAE:.4f}")
    print(f"Best blend RAE          : {best_overall:.4f}")
    print(f"Delta (anchor - blend)  : {delta:+.4f}  (margin {MARGIN})")
    print(f"VERDICT                 : {verdict}")
    print("=" * 72)

    # --- Save artifacts ---
    summary = {
        "n": n,
        "rae_nb1780": rae_a,
        "rae_nb1821": rae_b,
        "pearson": pearson,
        "pearson_residual": pearson_resid,
        "grid": grid_results,
        "best_grid_w_nb1780": best_w,
        "best_grid_rae": best_rae,
        "slsqp_insample_w": w_in,
        "slsqp_insample_rae": rae_slsqp_in,
        "slsqp_crossfit_rae": rae_slsqp_cf,
        "slsqp_fold_weights": fold_weights,
        "anchor_rae": ANCHOR_RAE,
        "margin": MARGIN,
        "delta": delta,
        "verdict": verdict,
    }
    out_json = PROC / "nb1832_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    np.save(PROC / "nb1832_blend_oof.npy", a * w_in[0] + b * w_in[1])
    print(f"Saved: {out_json}")
    print(f"Saved: {PROC / 'nb1832_blend_oof.npy'}")


if __name__ == "__main__":
    main()
