"""nb1682 — PRE+POST orthogonal blend: nb1632 BoB (PRE-unblind) + nb1471 (POST-unblind).

CAVEAT: nb1471 is POST-unblind. LB transfer risk applies if w_post > 0.

Protocol:
  1. Load nb1632_bob_mean_oof.npy (0.5107 PRE) and nb1471_best_oof.npy (0.4995 POST).
  2. Pearson.
  3. Grid w in {0.0..1.0 step 0.05} where w = weight on nb1471 (POST).
  4. 5-fold cross-fit SLSQP.
  5. Verdict at 0.003 margin vs nb1632 (0.5107).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_true - y_pred).sum()
    den = np.abs(y_true - y_true.mean()).sum()
    return float(num / den) if den > 0 else float("nan")


def load_unblind_truth() -> np.ndarray:
    npy_path = PROC / "postmortem" / "pm_unblind_y.npy"
    if npy_path.exists():
        return np.load(npy_path).astype(float)
    raise FileNotFoundError("pm_unblind_y.npy not found")


# ---------- load OOFs ----------
p_pre = PROC / "nb1632_bob_mean_oof.npy"
p_post = PROC / "nb1471_best_oof.npy"

oof_pre = np.load(p_pre).astype(float)
oof_post = np.load(p_post).astype(float)
print(f"[nb1682] loaded nb1632 (PRE)={oof_pre.shape}, nb1471 (POST)={oof_post.shape}")

n = min(len(oof_pre), len(oof_post))
oof_pre, oof_post = oof_pre[:n], oof_post[:n]

y_full = load_unblind_truth()
y = y_full[:n] if len(y_full) > n else y_full
print(f"[nb1682] truth shape={y.shape}, n={n}")

rae_pre = rae(y, oof_pre)
rae_post = rae(y, oof_post)
print(f"[nb1682] standalone: nb1632 PRE={rae_pre:.4f}, nb1471 POST={rae_post:.4f}")

# ---------- Pearson ----------
pearson = float(np.corrcoef(oof_pre, oof_post)[0, 1])
residuals_pre = oof_pre - y
residuals_post = oof_post - y
resid_pearson = float(np.corrcoef(residuals_pre, residuals_post)[0, 1])
print(f"[nb1682] Pearson pred={pearson:.4f}, residual={resid_pearson:.4f}")

# ---------- 1D grid: w = weight on nb1471 (POST) ----------
step = 0.05
grid = []
ws = np.arange(0.0, 1.0 + 1e-9, step)
for w in ws:
    blend = (1.0 - w) * oof_pre + w * oof_post
    r = rae(y, blend)
    grid.append({"w_post_nb1471": float(w), "w_pre_nb1632": float(1.0 - w), "rae": r})

grid_sorted = sorted(grid, key=lambda d: d["rae"])
best_grid = grid_sorted[0]
print(f"[nb1682] grid best: w_post={best_grid['w_post_nb1471']:.2f}, RAE={best_grid['rae']:.4f}")
print("[nb1682] grid top-5:")
for r in grid_sorted[:5]:
    print(f"  w_post={r['w_post_nb1471']:.2f} w_pre={r['w_pre_nb1632']:.2f} RAE={r['rae']:.4f}")

# ---------- 5-fold cross-fit SLSQP ----------

def neg_blend_rae(w: np.ndarray, mat: np.ndarray, t: np.ndarray) -> float:
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 0:
        return 1e9
    w = w / s
    return rae(t, mat @ w)


X = np.stack([oof_pre, oof_post], axis=1)  # cols: [PRE nb1632, POST nb1471]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cf_pred = np.full(n, np.nan)
fold_weights = []
cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
bounds = [(0.0, 1.0)] * 2
x0 = np.array([0.5, 0.5])

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
    print(f"[nb1682] fold {fold}: w=[pre={w[0]:.3f}, post={w[1]:.3f}] fold_RAE={rae(y[te], X[te]@w):.4f}")

slsqp_cf_rae = rae(y, cf_pred)
print(f"[nb1682] SLSQP cross-fit RAE: {slsqp_cf_rae:.4f}")

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
print(f"[nb1682] SLSQP in-sample: w=[pre={w_in[0]:.3f}, post={w_in[1]:.3f}] RAE={in_sample_rae:.4f}")

# ---------- verdict ----------
margin = 0.003
target = 0.5107  # nb1632 reference
best_blend_rae = min(slsqp_cf_rae, best_grid["rae"])
beats_nb1632 = bool(best_blend_rae < target - margin)
verdict = (
    f"DEPLOY nb1682 (best_blend_RAE={best_blend_rae:.4f} beats {target:.4f} by margin) — REGIME CAVEAT: w_post>0 carries POST-unblind LB transfer risk"
    if beats_nb1632
    else f"REJECT nb1682 (best_blend_RAE={best_blend_rae:.4f} fails 0.003 margin vs {target:.4f})"
)
print(f"[nb1682] verdict: {verdict}")

# ---------- persist ----------
best_oof = cf_pred if slsqp_cf_rae <= best_grid["rae"] else (
    (1.0 - best_grid["w_post_nb1471"]) * oof_pre + best_grid["w_post_nb1471"] * oof_post
)
np.save(PROC / "nb1682_best_oof.npy", best_oof)

summary = {
    "method": "nb1682_PRE_POST_blend",
    "n_unblind": int(n),
    "components": {
        "nb1632_PRE_unblind": {"file": str(p_pre.name), "rae": rae_pre},
        "nb1471_POST_unblind": {"file": str(p_post.name), "rae": rae_post},
    },
    "regime_caveat": "nb1471 is POST-unblind; any w_post>0 carries LB transfer risk",
    "pred_pearson": pearson,
    "residual_pearson": resid_pearson,
    "grid": {
        "step": step,
        "table": grid,
        "top5": grid_sorted[:5],
        "best": best_grid,
    },
    "slsqp": {
        "cross_fit_rae": float(slsqp_cf_rae),
        "in_sample_rae": float(in_sample_rae),
        "in_sample_weights": {"w_pre_nb1632": float(w_in[0]), "w_post_nb1471": float(w_in[1])},
        "fold_weights": fold_weights,
    },
    "target_rae_nb1632": float(target),
    "margin": margin,
    "best_blend_rae": float(best_blend_rae),
    "beats_nb1632": beats_nb1632,
    "verdict": verdict,
}

with open(PROC / "nb1682_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"[nb1682] saved {PROC / 'nb1682_summary.json'}")
print(f"[nb1682] saved {PROC / 'nb1682_best_oof.npy'}")
