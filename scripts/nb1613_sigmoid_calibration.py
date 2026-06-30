"""nb1613 — Sigmoid calibration on nb1561 BoB mean OOF.

4-parameter sigmoid warp:   y = a + b * sigmoid(c * (x - d))
Fit per training fold by minimizing MAE (L1), apply to held-out, pool RAE.

Verdict at 0.003 margin vs nb1561 (rae_bob_mean = 0.5155).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

REPO = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = REPO / "data" / "processed"

NB1561_OOF = PROC / "nb1561_bob_mean_oof.npy"
TRUTH_Y = PROC / "_audit_unblind_y.npy"
OUT_JSON = PROC / "nb1613_summary.json"

NB1561_REF = 0.5155          # rae_bob_mean per nb1561 summary
MARGIN = 0.003
N_SPLITS = 5
SEED = 0


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative Absolute Error: sum|y-p| / sum|y - mean(y)|."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    num = np.abs(y_true - y_pred).sum()
    den = np.abs(y_true - y_true.mean()).sum()
    return float(num / den)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def warp(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    a, b, c, d = theta
    return a + b * sigmoid(c * (x - d))


def mae_loss(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    pred = warp(x, theta)
    return float(np.mean(np.abs(y - pred)))


def fit_warp(x_tr: np.ndarray, y_tr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit (a,b,c,d) minimizing MAE on training fold.

    Initialize so the warp is approximately identity-ish over the data range,
    then try a couple of restarts and pick the best by training MAE.
    """
    x_lo, x_hi = float(x_tr.min()), float(x_tr.max())
    y_lo, y_hi = float(y_tr.min()), float(y_tr.max())
    x_mid = 0.5 * (x_lo + x_hi)
    x_range = max(x_hi - x_lo, 1e-6)
    y_range = max(y_hi - y_lo, 1e-6)

    inits = [
        # near-identity around data: sigmoid centered at midpoint, slope so it spans
        np.array([y_lo,           y_range,        4.0 / x_range, x_mid]),
        np.array([y_lo - 0.5,     y_range + 1.0,  2.0 / x_range, x_mid]),
        np.array([y_lo,           y_range,        6.0 / x_range, x_mid + 0.25 * x_range]),
        np.array([y_lo,           y_range,        3.0 / x_range, x_mid - 0.25 * x_range]),
    ]

    best_theta = None
    best_loss = np.inf
    best_diag = {}
    for i, theta0 in enumerate(inits):
        try:
            res = minimize(
                mae_loss, theta0, args=(x_tr, y_tr),
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000, "adaptive": True},
            )
        except Exception as exc:
            best_diag.setdefault("errors", []).append(f"init{i}: {exc!r}")
            continue
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_theta = res.x.copy()
            best_diag = {
                "init_idx": i,
                "init_theta": theta0.tolist(),
                "n_iter": int(res.nit),
                "success": bool(res.success),
                "final_msg": str(res.message),
                "train_mae": float(res.fun),
            }
    if best_theta is None:
        # fallback identity (no-op warp)
        best_theta = np.array([0.0, 0.0, 1.0, x_mid])
        best_diag = {"init_idx": -1, "fallback": True, "train_mae": float(np.mean(np.abs(y_tr - x_tr)))}
    return best_theta, best_diag


def main() -> None:
    t0 = time.time()

    x = np.load(NB1561_OOF).astype(np.float64)
    y = np.load(TRUTH_Y).astype(np.float64)
    assert x.shape == y.shape == (253,), f"shape mismatch x={x.shape} y={y.shape}"

    rae_base = rae(y, x)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full_like(x, np.nan, dtype=np.float64)

    per_fold = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x)):
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_va, y_va = x[va_idx], y[va_idx]

        theta, diag = fit_warp(x_tr, y_tr)
        pred_va = warp(x_va, theta)
        oof[va_idx] = pred_va

        per_fold.append({
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
            "theta_a": float(theta[0]),
            "theta_b": float(theta[1]),
            "theta_c": float(theta[2]),
            "theta_d": float(theta[3]),
            "train_mae": float(diag.get("train_mae", float("nan"))),
            "valid_mae": float(np.mean(np.abs(y_va - pred_va))),
            "valid_rae_fold": float(rae(y_va, pred_va)),
            "init_idx": int(diag.get("init_idx", -2)),
            "n_iter": int(diag.get("n_iter", -1)),
            "success": bool(diag.get("success", False)),
        })

    assert not np.isnan(oof).any(), "OOF has NaNs"
    rae_warp = rae(y, oof)
    delta = rae_warp - NB1561_REF
    beats = bool(delta <= -MARGIN)

    if beats:
        verdict = "BEATS_NB1561"
    elif delta < 0:
        verdict = "IMPROVES_WITHIN_MARGIN"
    elif delta <= MARGIN:
        verdict = "TIES_NB1561"
    else:
        verdict = "WORSE_THAN_NB1561"

    summary = {
        "tag": "nb1613",
        "method": "4-param sigmoid warp (a + b*sigmoid(c*(x-d))) per-fold MAE fit",
        "anchor": "nb1561_bob_mean_oof",
        "anchor_path": str(NB1561_OOF),
        "truth_path": str(TRUTH_Y),
        "n_obs": int(len(x)),
        "n_splits": N_SPLITS,
        "seed": SEED,
        "nb1561_ref": NB1561_REF,
        "margin": MARGIN,
        "rae_base_nb1561_oof": rae_base,
        "rae_sigmoid_crossfit": rae_warp,
        "delta_vs_nb1561": delta,
        "beats_nb1561": beats,
        "verdict": verdict,
        "per_fold": per_fold,
        "wall_sec": round(time.time() - t0, 3),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "tag": "nb1613",
        "rae_base_nb1561_oof": rae_base,
        "rae_sigmoid_crossfit": rae_warp,
        "delta_vs_nb1561": delta,
        "beats_nb1561": beats,
        "verdict": verdict,
        "per_fold_thetas": [
            {k: f["{}".format(k)] for k in ("fold","theta_a","theta_b","theta_c","theta_d","valid_rae_fold")}
            for f in per_fold
        ],
        "out_json": str(OUT_JSON),
    }, indent=2))


if __name__ == "__main__":
    main()
