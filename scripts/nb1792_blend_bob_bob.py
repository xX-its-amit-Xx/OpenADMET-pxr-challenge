"""
nb1792 — Blend nb1780 BoB MEAN + nb1632 BoB MEAN.

PROTOCOL:
1. Load nb1780_bob_mean_oof.npy (0.5032) and nb1632_bob_mean_oof.npy (0.5107).
2. Pairwise Pearson.
3. Grid w in {0.0..1.0 step 0.05}.
4. 5-fold cross-fit SLSQP.
5. Verdict at 0.003 margin vs nb1780 (0.5032).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

ROOT = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

NB1780_OOF = PROC / "nb1780_bob_mean_oof.npy"
NB1632_OOF = PROC / "nb1632_bob_mean_oof.npy"
TRUTH_PATH = PROC / "_audit_unblind_y.npy"
BASELINE_NB1780 = 0.5032
MARGIN = 0.003


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    return float(num / den)


def load_truth() -> np.ndarray:
    if TRUTH_PATH.exists():
        return np.load(TRUTH_PATH)
    # Fallback paths used in this repo
    for cand in [
        PROC / "unblind_253_truth.npy",
        PROC / "truth_253.npy",
        PROC / "y_unblind_253.npy",
    ]:
        if cand.exists():
            return np.load(cand)
    raise FileNotFoundError("Truth file for 253 unblind not found.")


def slsqp_two_way(p1: np.ndarray, p2: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Returns (w1, w2) constrained to sum=1, both >=0, minimizing RAE."""

    def obj(w):
        pred = w[0] * p1 + w[1] * p2
        return rae(y, pred)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bnds = ((0.0, 1.0), (0.0, 1.0))
    x0 = np.array([0.5, 0.5])
    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    w = res.x / res.x.sum()
    return float(w[0]), float(w[1])


def kfold_cross_fit_slsqp(p1: np.ndarray, p2: np.ndarray, y: np.ndarray,
                          n_splits: int = 5, seed: int = 42) -> tuple[float, np.ndarray]:
    """Cross-fitted SLSQP: weights fit on train fold, applied to held-out fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(y, dtype=float)
    weights = []
    for tr_idx, te_idx in kf.split(p1):
        w1, w2 = slsqp_two_way(p1[tr_idx], p2[tr_idx], y[tr_idx])
        weights.append((w1, w2))
        oof[te_idx] = w1 * p1[te_idx] + w2 * p2[te_idx]
    return rae(y, oof), np.array(weights)


def main() -> dict:
    p_nb1780 = np.load(NB1780_OOF)
    p_nb1632 = np.load(NB1632_OOF)
    y = load_truth()

    # Align lengths (defensive)
    n = min(len(p_nb1780), len(p_nb1632), len(y))
    p_nb1780 = p_nb1780[:n]
    p_nb1632 = p_nb1632[:n]
    y = y[:n]

    # Sanity-check standalone RAEs
    rae_nb1780 = rae(y, p_nb1780)
    rae_nb1632 = rae(y, p_nb1632)

    # 1) Pearson
    r, p_val = pearsonr(p_nb1780, p_nb1632)

    # 2) Grid sweep w * nb1780 + (1-w) * nb1632
    grid = np.arange(0.0, 1.0 + 1e-9, 0.05)
    grid_rows = []
    for w in grid:
        pred = w * p_nb1780 + (1.0 - w) * p_nb1632
        grid_rows.append((float(w), rae(y, pred)))
    best_w, best_rae = min(grid_rows, key=lambda r_: r_[1])

    # 3) 5-fold cross-fit SLSQP
    slsqp_rae, fold_weights = kfold_cross_fit_slsqp(p_nb1780, p_nb1632, y, n_splits=5, seed=42)
    # In-sample SLSQP for reference
    w_in1, w_in2 = slsqp_two_way(p_nb1780, p_nb1632, y)
    insample_slsqp_rae = rae(y, w_in1 * p_nb1780 + w_in2 * p_nb1632)

    best_overall = min(best_rae, slsqp_rae)
    beats_nb1780 = bool(best_overall < BASELINE_NB1780 - MARGIN)

    if beats_nb1780:
        verdict = "PROMOTE: blend beats nb1780 by >=0.003 RAE"
    elif best_overall < BASELINE_NB1780:
        verdict = "MARGINAL: blend improves but inside 0.003 noise band -- keep nb1780 as PRIMARY"
    else:
        verdict = "REJECT: blend does not improve over nb1780"

    out = {
        "n": int(n),
        "rae_nb1780_standalone": rae_nb1780,
        "rae_nb1632_standalone": rae_nb1632,
        "pearson_r": float(r),
        "pearson_p": float(p_val),
        "grid": [{"w_nb1780": w, "rae": rr} for w, rr in grid_rows],
        "best_grid_w_nb1780": float(best_w),
        "best_grid_rae": float(best_rae),
        "slsqp_in_sample_w_nb1780": float(w_in1),
        "slsqp_in_sample_w_nb1632": float(w_in2),
        "slsqp_in_sample_rae": float(insample_slsqp_rae),
        "slsqp_crossfit_rae": float(slsqp_rae),
        "slsqp_fold_weights": fold_weights.tolist(),
        "baseline_nb1780": BASELINE_NB1780,
        "margin": MARGIN,
        "beats_nb1780": beats_nb1780,
        "verdict": verdict,
    }

    summary_path = PROC / "nb1792_summary.json"
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)

    # Print report
    print("=" * 72)
    print("nb1792 — Blend nb1780 BoB MEAN + nb1632 BoB MEAN")
    print("=" * 72)
    print(f"n compounds       : {n}")
    print(f"nb1780 standalone : RAE {rae_nb1780:.4f} (reported 0.5032)")
    print(f"nb1632 standalone : RAE {rae_nb1632:.4f} (reported 0.5107)")
    print(f"Pearson r         : {r:.4f}  (p={p_val:.2e})")
    print("-" * 72)
    print("Grid w * nb1780 + (1-w) * nb1632:")
    print(f"  {'w_nb1780':>10s}  {'RAE':>8s}")
    for w, rr in grid_rows:
        marker = "  <-- best" if (w, rr) == (best_w, best_rae) else ""
        print(f"  {w:10.2f}  {rr:8.4f}{marker}")
    print("-" * 72)
    print(f"Best grid         : w_nb1780={best_w:.2f}  RAE={best_rae:.4f}")
    print(f"SLSQP in-sample   : w_nb1780={w_in1:.4f}  w_nb1632={w_in2:.4f}  RAE={insample_slsqp_rae:.4f}")
    print(f"SLSQP 5-fold xfit : RAE={slsqp_rae:.4f}")
    print(f"Baseline nb1780   : {BASELINE_NB1780:.4f}  (margin {MARGIN})")
    print(f"beats_nb1780      : {beats_nb1780}")
    print(f"VERDICT           : {verdict}")
    print(f"Summary           : {summary_path}")
    return out


if __name__ == "__main__":
    main()
