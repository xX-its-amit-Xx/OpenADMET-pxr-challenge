"""nb1454 - Triple blend nb1441 + nb1422 + nb1411 at the sub-0.50 frontier.

Components (all aligned on the 253 unblind compounds):
  P1: nb1441_mean_bag_oof         (CatBoost residual on top of nb1070; ~0.5051)
  P2: nb1422_bob_mean_oof         (best-of-bag mean over 5 outer seeds of nb1411 blend; ~0.5022)
  P3: nb1411 naive 1/3            (1/3 each of nb1373 + nb1352 + nb1364; ~0.5037)

Steps:
  1. Load three components + unblind truth.
  2. Pairwise Pearson.
  3. 3D simplex grid (step 0.05) -> top-5 + best RAE.
  4. 5-fold cross-fit SLSQP (per-fold convex weight) -> RAE.
  5. Naive 1/3 mean -> RAE.
  6. Verdict at decision_margin 0.003 vs nb1422 bob_median (0.5016) and vs nb1441 blend (0.4990).

Outputs:
  data/processed/nb1454_summary.json
  data/processed/nb1454_best_oof.npy
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

ROOT = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / "src"))
from pxr.eval import rae  # noqa: E402


def grid_simplex(step: float):
    """Yield convex weights (w1, w2, w3) on a simplex with grid step."""
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            yield (i / n, j / n, k / n)


def slsqp_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convex non-negative weights summing to 1 minimising RAE on (P,y)."""

    def obj(w):
        return rae(y, P @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * P.shape[1]
    w0 = np.ones(P.shape[1]) / P.shape[1]
    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 400, "ftol": 1e-10})
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum()
    return w


def slsqp_crossfit(P: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 0):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred = np.zeros_like(y, dtype=float)
    fold_ws = []
    for tr_idx, te_idx in kf.split(P):
        w = slsqp_weights(P[tr_idx], y[tr_idx])
        pred[te_idx] = P[te_idx] @ w
        fold_ws.append(w.tolist())
    return pred, fold_ws


def main() -> int:
    t0 = time.time()

    y = np.load(PROC / "_audit_unblind_y.npy")
    p1 = np.load(PROC / "nb1441_mean_bag_oof.npy")           # 0.5051
    p2 = np.load(PROC / "nb1422_bob_mean_oof.npy")           # 0.5022
    # Build nb1411 naive 1/3 component to match the 0.5037 target
    n73 = np.load(PROC / "nb1373_mean_bag_oof.npy")
    n52 = np.load(PROC / "nb1352_mean_bag_oof.npy")
    n64 = np.load(PROC / "nb1364_mean_bag_oof.npy")
    p3 = (n73 + n52 + n64) / 3.0

    assert y.shape == p1.shape == p2.shape == p3.shape == (253,)

    rae_p1 = float(rae(y, p1))
    rae_p2 = float(rae(y, p2))
    rae_p3 = float(rae(y, p3))

    # ---- pairwise Pearson on predictions
    P = np.stack([p1, p2, p3], axis=1)
    corr = np.corrcoef(P.T).tolist()

    # ---- 3D simplex grid step 0.05
    grid_step = 0.05
    grid_records = []
    for w in grid_simplex(grid_step):
        wa = np.array(w)
        r = float(rae(y, P @ wa))
        grid_records.append({"w": list(w), "rae": r})
    grid_records.sort(key=lambda d: d["rae"])
    grid_top5 = grid_records[:5]
    grid_best = grid_records[0]

    # ---- SLSQP in-sample
    w_in = slsqp_weights(P, y)
    rae_slsqp_in = float(rae(y, P @ w_in))

    # ---- SLSQP 5-fold cross-fit
    pred_cf, fold_ws = slsqp_crossfit(P, y, n_folds=5, seed=0)
    rae_slsqp_cf = float(rae(y, pred_cf))
    w_cf_mean = np.mean(fold_ws, axis=0).tolist()
    w_cf_std = np.std(fold_ws, axis=0).tolist()

    # ---- naive 1/3
    naive_w = np.array([1 / 3, 1 / 3, 1 / 3])
    naive_pred = P @ naive_w
    rae_naive = float(rae(y, naive_pred))

    # ---- pick the best-honest = SLSQP cross-fit (avoid in-sample overfit)
    pick_label = "slsqp_crossfit"
    pick_rae = rae_slsqp_cf
    pick_pred = pred_cf

    # also surface in-sample best (grid_best, slsqp_in, naive)
    candidates = {
        "slsqp_crossfit": rae_slsqp_cf,
        "slsqp_insample": rae_slsqp_in,
        "grid_best": grid_best["rae"],
        "naive_third": rae_naive,
    }

    # Reference markers
    NB1422_MEDIAN = 0.5016
    NB1441_BLEND_REF = 0.4990
    NB1422_MEAN = 0.5022
    decision_margin = 0.003

    beats_nb1422_median = pick_rae < (NB1422_MEDIAN - decision_margin)
    flat_nb1422_median = abs(pick_rae - NB1422_MEDIAN) <= decision_margin
    beats_nb1441_blend = pick_rae < (NB1441_BLEND_REF - decision_margin)
    flat_nb1441_blend = abs(pick_rae - NB1441_BLEND_REF) <= decision_margin

    if beats_nb1441_blend:
        verdict = "TRIPLE_BEATS_NB1441_BLEND"
    elif flat_nb1441_blend and beats_nb1422_median:
        verdict = "TRIPLE_BEATS_NB1422_MEDIAN_FLAT_VS_NB1441_BLEND"
    elif beats_nb1422_median:
        verdict = "TRIPLE_BEATS_NB1422_MEDIAN"
    elif flat_nb1422_median:
        verdict = "TRIPLE_FLAT_VS_NB1422_MEDIAN"
    else:
        verdict = "TRIPLE_LOSES_VS_NB1422_MEDIAN"

    summary = {
        "tag": "nb1454",
        "components": ["nb1441_mean_bag", "nb1422_bob_mean", "nb1411_naive_third"],
        "component_rae": {
            "nb1441_mean_bag": rae_p1,
            "nb1422_bob_mean": rae_p2,
            "nb1411_naive_third": rae_p3,
        },
        "n_unb": int(y.shape[0]),
        "pred_pearson": corr,
        "grid_step": grid_step,
        "n_grid_pts": len(grid_records),
        "grid_top5": grid_top5,
        "grid_best": grid_best,
        "slsqp_insample_w": w_in.tolist(),
        "slsqp_insample_rae": rae_slsqp_in,
        "slsqp_crossfit_folds": 5,
        "slsqp_crossfit_seed": 0,
        "slsqp_crossfit_fold_w": fold_ws,
        "slsqp_crossfit_fold_w_mean": w_cf_mean,
        "slsqp_crossfit_fold_w_std": w_cf_std,
        "slsqp_crossfit_rae": rae_slsqp_cf,
        "naive_third_w": naive_w.tolist(),
        "naive_third_rae": rae_naive,
        "candidates_rae": candidates,
        "pick_label": pick_label,
        "pick_rae": pick_rae,
        "ref_nb1422_bob_mean": NB1422_MEAN,
        "ref_nb1422_bob_median": NB1422_MEDIAN,
        "ref_nb1441_blend": NB1441_BLEND_REF,
        "decision_margin": decision_margin,
        "delta_pick_vs_nb1422_median": pick_rae - NB1422_MEDIAN,
        "delta_pick_vs_nb1441_blend": pick_rae - NB1441_BLEND_REF,
        "beats_nb1422_median": bool(beats_nb1422_median),
        "flat_nb1422_median": bool(flat_nb1422_median),
        "beats_nb1441_blend": bool(beats_nb1441_blend),
        "flat_nb1441_blend": bool(flat_nb1441_blend),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }

    out_summary = PROC / "nb1454_summary.json"
    out_oof = PROC / "nb1454_best_oof.npy"
    with out_summary.open("w") as fh:
        json.dump(summary, fh, indent=2)
    np.save(out_oof, pick_pred)

    # console report
    print(json.dumps({
        "pred_pearson": corr,
        "component_rae": summary["component_rae"],
        "grid_top5": grid_top5,
        "slsqp_in_w": w_in.tolist(),
        "slsqp_in_rae": rae_slsqp_in,
        "slsqp_cf_w_mean": w_cf_mean,
        "slsqp_cf_rae": rae_slsqp_cf,
        "naive_third_rae": rae_naive,
        "pick": (pick_label, pick_rae),
        "beats_nb1422_median": beats_nb1422_median,
        "beats_nb1441_blend": beats_nb1441_blend,
        "verdict": verdict,
        "wall_sec": summary["wall_sec"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
