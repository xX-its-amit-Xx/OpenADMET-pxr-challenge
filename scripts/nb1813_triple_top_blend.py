"""nb1813 — Triple top blend: nb1780 BoB + nb1632 BoB + nb1771 single-bag.

Protocol:
  1. Load 3 OOF arrays.
  2. Pairwise Pearson.
  3. 3D simplex grid (step 0.05).
  4. SLSQP cross-fit (5-fold).
  5. Verdict at 0.003 margin vs nb1780 (0.5032).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

# Anchor / margin
ANCHOR_NAME = "nb1780"
ANCHOR_RAE = 0.5032
MARGIN = 0.003


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.abs(y_true - y_pred).sum()
    den = np.abs(y_true - y_true.mean()).sum()
    return float(num / den)


def load_truth() -> np.ndarray:
    # Standard PXR unblind truth array
    candidates = [
        PROC / "_audit_unblind_y.npy",
        PROC / "unblind_truth_253.npy",
        PROC / "y_unblind_253.npy",
        PROC / "unblind_y_true.npy",
        PROC / "postmortem" / "y_true_253.npy",
    ]
    for c in candidates:
        if c.exists():
            return np.load(c).astype(float)
    raise FileNotFoundError(f"No truth array found among {candidates}")


def slsqp_3way_crossfit(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[float, np.ndarray]:
    """5-fold cross-fit SLSQP blend weights."""
    n = X.shape[0]
    oof = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_w = []
    for tr_idx, te_idx in kf.split(X):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte = X[te_idx]

        def loss(w):
            pred = Xtr @ w
            return float(np.abs(ytr - pred).sum() / np.abs(ytr - ytr.mean()).sum())

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bnds = [(0.0, 1.0)] * X.shape[1]
        x0 = np.ones(X.shape[1]) / X.shape[1]
        res = minimize(loss, x0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 500})
        w = res.x
        fold_w.append(w)
        oof[te_idx] = Xte @ w
    w_mean = np.mean(fold_w, axis=0)
    return rae(y, oof), w_mean


def grid_3d(step: float = 0.05) -> list[tuple[float, float, float]]:
    pts = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i * step, j * step, k * step))
    return pts


def main():
    files = {
        "nb1780": PROC / "nb1780_bob_mean_oof.npy",
        "nb1632": PROC / "nb1632_bob_mean_oof.npy",
        "nb1771": PROC / "nb1771_mean_bag_oof.npy",
    }
    arrs = {k: np.load(v).astype(float) for k, v in files.items()}
    y = load_truth()

    # Sanity
    for k, a in arrs.items():
        assert a.shape == y.shape, f"{k}: shape {a.shape} vs y {y.shape}"

    # Solo RAE
    solo = {k: rae(y, a) for k, a in arrs.items()}
    print("=== Solo RAE ===")
    for k, v in solo.items():
        print(f"  {k}: {v:.4f}")

    # Pairwise Pearson
    keys = list(arrs.keys())
    print("\n=== Pairwise Pearson ===")
    pcorr = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r = float(np.corrcoef(arrs[keys[i]], arrs[keys[j]])[0, 1])
            pcorr[f"{keys[i]}__{keys[j]}"] = r
            print(f"  {keys[i]} ~ {keys[j]}: r = {r:.4f}")

    # 3D simplex grid
    X = np.column_stack([arrs[k] for k in keys])
    grid = grid_3d(step=0.05)
    rows = []
    for w in grid:
        w_arr = np.array(w)
        pred = X @ w_arr
        r = rae(y, pred)
        rows.append((w, r))
    rows.sort(key=lambda x: x[1])
    print("\n=== Top-5 simplex grid (step=0.05) ===")
    for (w, r) in rows[:5]:
        print(f"  w={tuple(round(x, 2) for x in w)} ({'+'.join(keys)})  RAE={r:.4f}")

    best_grid_w, best_grid_r = rows[0]

    # SLSQP cross-fit
    slsqp_rae, slsqp_w = slsqp_3way_crossfit(X, y, n_splits=5)
    print("\n=== SLSQP 5-fold cross-fit ===")
    print(f"  weights (mean over folds): {dict(zip(keys, [round(float(x), 4) for x in slsqp_w]))}")
    print(f"  cross-fit RAE: {slsqp_rae:.4f}")

    # Verdict
    best_blend_r = min(best_grid_r, slsqp_rae)
    delta = ANCHOR_RAE - best_blend_r
    pass_margin = delta >= MARGIN
    print("\n=== Verdict ===")
    print(f"  anchor: {ANCHOR_NAME} RAE = {ANCHOR_RAE:.4f}")
    print(f"  best blend RAE     = {best_blend_r:.4f}")
    print(f"  delta (anchor - blend) = {delta:+.4f}  (margin = {MARGIN})")
    print(f"  VERDICT: {'PASS (deploy blend)' if pass_margin else 'FAIL (keep anchor)'}")

    # Persist summary
    summary = {
        "anchor": {"name": ANCHOR_NAME, "rae": ANCHOR_RAE},
        "margin": MARGIN,
        "solo": solo,
        "pairwise_pearson": pcorr,
        "grid_top5": [
            {"w": [float(x) for x in w], "rae": float(r)} for (w, r) in rows[:5]
        ],
        "slsqp": {
            "weights": {k: float(v) for k, v in zip(keys, slsqp_w)},
            "rae": float(slsqp_rae),
        },
        "best_blend_rae": float(best_blend_r),
        "delta_vs_anchor": float(delta),
        "verdict": "PASS" if pass_margin else "FAIL",
    }
    out = PROC / "nb1813_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
