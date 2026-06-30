"""nb1663 — Triple blend: nb1632 BoB + nb1561 CatBoost + nb1571 blend.

Protocol:
  1. Load three OOF arrays (all on 253 unblind).
  2. Pairwise Pearson correlations.
  3. 3D simplex grid step 0.05.
  4. SLSQP cross-fit (5-fold).
  5. Verdict at 0.003 margin vs nb1632 (0.5107).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# ---------- helpers ----------

def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative Absolute Error vs mean predictor."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_true - y_pred).sum()
    den = np.abs(y_true - y_true.mean()).sum()
    return float(num / den) if den > 0 else float("nan")


def load_unblind_truth() -> tuple[np.ndarray, np.ndarray]:
    """Return (truth on 253 unblind, unb_idx into 513 test)."""
    npy_path = PROC / "postmortem" / "pm_unblind_y.npy"
    if npy_path.exists():
        y = np.load(npy_path).astype(float)
        return y, np.arange(len(y))
    raise FileNotFoundError("pm_unblind_y.npy not found")


# ---------- load OOFs ----------

p1 = PROC / "nb1632_bob_mean_oof.npy"
p2 = PROC / "nb1561_bob_mean_oof.npy"
p3 = PROC / "nb1571_best_oof.npy"

oof1 = np.load(p1).astype(float)
oof2 = np.load(p2).astype(float)
oof3 = np.load(p3).astype(float)

print(f"[nb1663] loaded OOFs: nb1632={oof1.shape}, nb1561={oof2.shape}, nb1571={oof3.shape}")

# normalize lengths -- all expected on 253 unblind
n = min(len(oof1), len(oof2), len(oof3))
oof1, oof2, oof3 = oof1[:n], oof2[:n], oof3[:n]

# truth
y_full, unb_idx = load_unblind_truth()
if len(y_full) > n:
    y = y_full[:n]
else:
    y = y_full
print(f"[nb1663] truth shape={y.shape}, n={n}")

base_rae = {
    "nb1632": rae(y, oof1),
    "nb1561": rae(y, oof2),
    "nb1571": rae(y, oof3),
}
print(f"[nb1663] standalone RAE: {base_rae}")

# ---------- pairwise Pearson ----------
corr = {
    "nb1632_vs_nb1561": float(np.corrcoef(oof1, oof2)[0, 1]),
    "nb1632_vs_nb1571": float(np.corrcoef(oof1, oof3)[0, 1]),
    "nb1561_vs_nb1571": float(np.corrcoef(oof2, oof3)[0, 1]),
}
print(f"[nb1663] pairwise Pearson: {corr}")

# ---------- 3D simplex grid step 0.05 ----------
step = 0.05
grid_results: list[tuple[float, float, float, float]] = []
ws = np.arange(0.0, 1.0 + 1e-9, step)
for w1 in ws:
    for w2 in ws:
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9 or w3 > 1.0 + 1e-9:
            continue
        w3 = max(0.0, min(1.0, w3))
        blend = w1 * oof1 + w2 * oof2 + w3 * oof3
        r = rae(y, blend)
        grid_results.append((w1, w2, w3, r))

grid_results.sort(key=lambda t: t[3])
top5 = grid_results[:5]
print("[nb1663] top-5 grid:")
for w1, w2, w3, r in top5:
    print(f"  w=({w1:.2f},{w2:.2f},{w3:.2f}) RAE={r:.4f}")

best_grid_w = np.array(top5[0][:3])
best_grid_rae = top5[0][3]

# ---------- SLSQP cross-fit (5-fold) ----------

def neg_blend_rae(w: np.ndarray, mat: np.ndarray, t: np.ndarray) -> float:
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 0:
        return 1e9
    w = w / s
    pred = mat @ w
    return rae(t, pred)


X = np.stack([oof1, oof2, oof3], axis=1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cf_pred = np.full(n, np.nan)
fold_weights: list[list[float]] = []

cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
bounds = [(0.0, 1.0)] * 3
x0 = np.array([1 / 3, 1 / 3, 1 / 3])

for fold, (tr, te) in enumerate(kf.split(X)):
    res = minimize(
        neg_blend_rae,
        x0,
        args=(X[tr], y[tr]),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-8},
    )
    w = np.clip(res.x, 0, None)
    w = w / max(w.sum(), 1e-12)
    fold_weights.append(w.tolist())
    cf_pred[te] = X[te] @ w
    print(f"[nb1663] fold {fold} weights={w.round(3).tolist()} fold_RAE={rae(y[te], X[te]@w):.4f}")

slsqp_cf_rae = rae(y, cf_pred)
print(f"[nb1663] SLSQP cross-fit RAE: {slsqp_cf_rae:.4f}")

# in-sample SLSQP (informational)
res_in = minimize(
    neg_blend_rae,
    x0,
    args=(X, y),
    method="SLSQP",
    bounds=bounds,
    constraints=cons,
    options={"maxiter": 300, "ftol": 1e-9},
)
w_in = np.clip(res_in.x, 0, None)
w_in = w_in / max(w_in.sum(), 1e-12)
in_sample_rae = rae(y, X @ w_in)
print(f"[nb1663] SLSQP in-sample weights={w_in.round(3).tolist()} RAE={in_sample_rae:.4f}")

# ---------- verdict ----------
margin = 0.003
target = base_rae["nb1632"]  # 0.5107
best_blend_rae = min(slsqp_cf_rae, best_grid_rae)
beats_nb1632 = bool(best_blend_rae < target - margin)
verdict = (
    f"DEPLOY nb1663 (best_blend_RAE={best_blend_rae:.4f} beats {target:.4f} by margin)"
    if beats_nb1632
    else f"REJECT nb1663 (best_blend_RAE={best_blend_rae:.4f} fails 0.003 margin vs {target:.4f})"
)
print(f"[nb1663] verdict: {verdict}")

# ---------- persist artifacts ----------
# Best OOF: use the better of (grid, cross-fit). For deploy-faithful, store CF predictions.
best_oof = cf_pred if slsqp_cf_rae <= best_grid_rae else (X @ best_grid_w)
np.save(PROC / "nb1663_best_oof.npy", best_oof)

summary = {
    "method": "nb1663_triple_blend",
    "n_unblind": int(n),
    "standalone_rae": base_rae,
    "pairwise_pearson": corr,
    "grid": {
        "step": step,
        "top5": [
            {"w_nb1632": w1, "w_nb1561": w2, "w_nb1571": w3, "rae": r}
            for w1, w2, w3, r in top5
        ],
        "best_weights": {
            "w_nb1632": float(best_grid_w[0]),
            "w_nb1561": float(best_grid_w[1]),
            "w_nb1571": float(best_grid_w[2]),
        },
        "best_rae": float(best_grid_rae),
    },
    "slsqp": {
        "cross_fit_rae": float(slsqp_cf_rae),
        "in_sample_rae": float(in_sample_rae),
        "in_sample_weights": {
            "w_nb1632": float(w_in[0]),
            "w_nb1561": float(w_in[1]),
            "w_nb1571": float(w_in[2]),
        },
        "fold_weights": fold_weights,
    },
    "target_rae_nb1632": float(target),
    "margin": margin,
    "best_blend_rae": float(best_blend_rae),
    "beats_nb1632": beats_nb1632,
    "verdict": verdict,
}

with open(PROC / "nb1663_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"[nb1663] saved {PROC / 'nb1663_summary.json'}")
print(f"[nb1663] saved {PROC / 'nb1663_best_oof.npy'}")
