"""nb1972 — Blend nb1951 (== nb1941 pooled 50-bag OOF) + nb1934 BoB mean (top 2 BoB candidates).

Note: nb1951 is a deploy-only refit of nb1941 (no separate 253-OOF artifact); per protocol
fallback, the nb1941 pooled 50-bag OOF (0.4990) is used as the nb1951 OOF surrogate.

Protocol:
1. Load nb1941_pooled_50bag_oof.npy (0.4990) and nb1934_bob_mean_oof.npy (0.5017).
2. Pearson correlation between the two OOFs.
3. Grid search w in {0.0..1.0 step 0.05}.
4. 5-fold cross-fit SLSQP convex blend.
5. Verdict at 0.003 margin vs nb1941 (0.4990).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true))))


def load_truth() -> np.ndarray:
    candidates = [
        PROC / "postmortem" / "pm_unblind_y.npy",
        PROC / "postmortem" / "y_unblind_253.npy",
        PROC / "y_unblind_253.npy",
        PROC / "postmortem" / "y_true_253.npy",
    ]
    for c in candidates:
        if c.exists():
            return np.load(c).astype(float)
    csv = PROC / "unblind_253.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        for col in ("pec50", "pEC50", "y", "truth"):
            if col in df.columns:
                return df[col].to_numpy(dtype=float)
    raise FileNotFoundError("Could not find 253-unblind truth vector.")


def slsqp_blend(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    k = P.shape[1]

    def obj(w):
        return rae(y, P @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * k
    w0 = np.full(k, 1.0 / k)
    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    return res.x


def cross_fit_slsqp(y: np.ndarray, P: np.ndarray, n_splits: int = 5, seed: int = 0) -> tuple[float, list[np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    ws: list[np.ndarray] = []
    for tr, va in kf.split(y):
        w = slsqp_blend(y[tr], P[tr])
        oof[va] = P[va] @ w
        ws.append(w)
    return rae(y, oof), ws


def main() -> dict:
    # nb1951 OOF surrogate = nb1941 pooled 50-bag OOF (nb1951 is deploy refit of nb1941).
    p_nb1951 = np.load(PROC / "nb1941_pooled_50bag_oof.npy").astype(float)
    p_nb1934 = np.load(PROC / "nb1934_bob_mean_oof.npy").astype(float)
    y = load_truth()

    assert p_nb1951.shape == p_nb1934.shape == y.shape, \
        f"shape mismatch: nb1951 {p_nb1951.shape}, nb1934 {p_nb1934.shape}, y {y.shape}"

    rae_nb1951 = rae(y, p_nb1951)
    rae_nb1934 = rae(y, p_nb1934)
    r, _ = pearsonr(p_nb1951, p_nb1934)

    # Grid: w on nb1951.
    grid = []
    best = {"w_nb1951": None, "rae": np.inf}
    for w in np.round(np.arange(0.0, 1.0001, 0.05), 4):
        pred = w * p_nb1951 + (1.0 - w) * p_nb1934
        r_ = rae(y, pred)
        grid.append({"w_nb1951": float(w), "rae": float(r_)})
        if r_ < best["rae"]:
            best = {"w_nb1951": float(w), "rae": float(r_)}

    # 5-fold cross-fit SLSQP.
    P = np.column_stack([p_nb1951, p_nb1934])
    slsqp_rae, ws = cross_fit_slsqp(y, P, n_splits=5, seed=0)
    w_mean = np.mean(ws, axis=0)

    # Anchor for promotion gate = nb1941 (0.4990) — this is the nb1951-surrogate value.
    anchor_rae = rae_nb1951
    margin = 0.003
    delta_grid = best["rae"] - anchor_rae
    verdict_grid = "PROMOTE" if (anchor_rae - best["rae"]) >= margin else "REJECT"
    delta_slsqp = slsqp_rae - anchor_rae
    verdict_slsqp = "PROMOTE" if (anchor_rae - slsqp_rae) >= margin else "REJECT"

    summary = {
        "tag": "nb1972",
        "anchors": {
            "nb1951_oof_source": "nb1941_pooled_50bag_oof.npy (nb1951 deploy-only refit, no separate OOF)",
            "nb1934_oof_source": "nb1934_bob_mean_oof.npy",
        },
        "n": int(y.size),
        "rae_nb1951": rae_nb1951,
        "rae_nb1934": rae_nb1934,
        "pearson": float(r),
        "grid": grid,
        "best_grid": best,
        "delta_grid_vs_nb1951": float(delta_grid),
        "verdict_grid_at_0.003": verdict_grid,
        "slsqp_5fold_rae": float(slsqp_rae),
        "slsqp_mean_weights": {"nb1951": float(w_mean[0]), "nb1934": float(w_mean[1])},
        "slsqp_fold_weights": [{"nb1951": float(w[0]), "nb1934": float(w[1])} for w in ws],
        "delta_slsqp_vs_nb1951": float(delta_slsqp),
        "verdict_slsqp_at_0.003": verdict_slsqp,
    }

    out = PROC / "nb1972_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k != "grid"}, indent=2))
    print("\nGRID:")
    for row in grid:
        print(f"  w_nb1951={row['w_nb1951']:.2f}  RAE={row['rae']:.6f}")
    return summary


if __name__ == "__main__":
    main()
