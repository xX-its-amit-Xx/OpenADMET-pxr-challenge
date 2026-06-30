"""nb1431 -- BoB-component 3-way blend.

Combines the OUTER-BAGGED MEAN versions of three pruned-feature residual
learners:
    A = nb1381 (AtomPair top-30 SHAP-pruned + ChEMBL residual) BoB-mean
    B = nb1361 (MACCS    top-20 SHAP-pruned + ChEMBL residual) BoB-mean
    C = nb1423 (Mordred  top-20 SHAP-pruned + ChEMBL residual) BoB-mean

Each component is itself a 5-outer x 5-inner bagged predictor on the 253
unblind compounds.  Hypothesis: replacing single-bag components with their
outer-bag means reduces seed variance in the 3-way blend versus nb1422
(median across the 5 outer bags of the 3-way naive mean), whose pooled BoB
MEDIAN RAE is 0.5016.

PROTOCOL:
    1. Load three (253,) BoB-mean OOFs.
    2. Pairwise Pearson correlations.
    3. 3D simplex grid step 0.05; report top-5 (w_A, w_B, w_C, RAE).
    4. SLSQP cross-fit (5-fold simplex optimization, no leakage).
    5. Naive 1/3 mean.
    6. Verdict at +/- 0.003 margin vs nb1422 (0.5016).

Outputs:
    scripts/nb1431_bob_component_3way.py   (this file)
    data/processed/nb1431_summary.json
    data/processed/nb1431_best_oof.npy     (253,) float32
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from itertools import product
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1431"

COMP_A_PATH = DATA_PROCESSED / "nb1381_bob_mean_oof.npy"
COMP_B_PATH = DATA_PROCESSED / "nb1361_bob_mean_oof.npy"
COMP_C_PATH = DATA_PROCESSED / "nb1423_bob_mean_oof.npy"

COMP_LABELS = ("nb1381_AP", "nb1361_MACCS", "nb1423_MORD")

NB1422_REF = 0.5016
REPRODUCE_MARGIN = 0.003

GRID_STEP = 0.05
SLSQP_FOLDS = 5
SLSQP_SEED = 0
SLSQP_N_STARTS = 9


def _simplex_grid(step: float) -> np.ndarray:
    """All (w_A, w_B, w_C) on simplex {w >= 0, sum w == 1} with given step."""
    n = int(round(1.0 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            if k < 0:
                continue
            pts.append((i / n, j / n, k / n))
    arr = np.asarray(pts, dtype=np.float64)
    s = arr.sum(axis=1)
    arr = arr / s[:, None]
    return arr


def _rae_blend(w: np.ndarray, P: np.ndarray, y: np.ndarray) -> float:
    pred = P @ w
    return float(rae(y, pred))


def _slsqp_one_fit(P: np.ndarray, y: np.ndarray,
                   x0: np.ndarray) -> tuple[np.ndarray, float]:
    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * P.shape[1]
    res = minimize(
        fun=lambda w: _rae_blend(w, P, y),
        x0=x0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-8, "disp": False},
    )
    w = np.clip(res.x, 0.0, 1.0)
    if w.sum() <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    return w, float(res.fun)


def _slsqp_cross_fit(P: np.ndarray, y: np.ndarray,
                     n_folds: int, seed: int,
                     n_starts: int) -> tuple[np.ndarray, list]:
    """Cross-fit SLSQP: fit weights on train fold, evaluate held-out preds."""
    n, k = P.shape
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(n, dtype=np.float64)
    fold_weights = []

    # Build a set of seed starting points.
    starts = [np.ones(k) / k]
    for i in range(k):
        e = np.full(k, (1 - 0.8) / (k - 1))
        e[i] = 0.8
        starts.append(e)
    rng = np.random.default_rng(seed)
    while len(starts) < n_starts:
        v = rng.dirichlet(np.ones(k))
        starts.append(v)
    starts = starts[:n_starts]

    for fi, (tr, va) in enumerate(kf.split(np.arange(n))):
        best_w = None
        best_fun = np.inf
        for x0 in starts:
            w_try, fun_try = _slsqp_one_fit(P[tr], y[tr], x0)
            if fun_try < best_fun:
                best_w = w_try
                best_fun = fun_try
        oof[va] = P[va] @ best_w
        fold_weights.append({
            "fold": fi,
            "weights": [float(x) for x in best_w],
            "train_rae": best_fun,
        })
    return oof, fold_weights


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- BoB-component 3-way blend (AtomPair + MACCS + Mordred)")
    print(f"         components       = {COMP_LABELS}")
    print(f"         nb1422 ref       = {NB1422_REF:.4f}  (median pooled RAE)")
    print(f"         margin           = {REPRODUCE_MARGIN}")
    print(f"         grid step        = {GRID_STEP}")
    print(f"         SLSQP folds      = {SLSQP_FOLDS}  starts={SLSQP_N_STARTS}")
    print("=" * 78)

    # ---- Load truth + 3 BoB-mean OOFs ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape = ({n_unb},)")

    paths = (COMP_A_PATH, COMP_B_PATH, COMP_C_PATH)
    P_list = []
    for label, p in zip(COMP_LABELS, paths):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing.")
        v = np.load(p).astype(np.float64)
        if v.shape != (n_unb,):
            raise ValueError(f"{label} shape {v.shape} != ({n_unb},)")
        P_list.append(v)
        print(f"[load] {label:14s} -> {p.name}   shape={v.shape}")

    P = np.stack(P_list, axis=1)  # (n_unb, 3)

    # ---- Standalone component RAEs ----
    standalone_rae = {}
    print("\n" + "-" * 78)
    print("STANDALONE COMPONENT RAE")
    print("-" * 78)
    for li, lab in enumerate(COMP_LABELS):
        r = float(rae(y_unb, P[:, li]))
        standalone_rae[lab] = r
        print(f"   {lab:14s} RAE = {r:.4f}")

    # ---- Pairwise Pearson correlations ----
    print("\n" + "-" * 78)
    print("PAIRWISE PEARSON CORRELATIONS")
    print("-" * 78)
    corr = np.corrcoef(P, rowvar=False)
    corr_records = []
    for i in range(3):
        for j in range(i + 1, 3):
            r = float(corr[i, j])
            corr_records.append({
                "a": COMP_LABELS[i], "b": COMP_LABELS[j], "pearson": r
            })
            print(f"   {COMP_LABELS[i]:14s} <-> {COMP_LABELS[j]:14s}  "
                  f"r = {r:+.4f}")

    # ---- 3D simplex grid ----
    print("\n" + "-" * 78)
    print(f"3D SIMPLEX GRID  step={GRID_STEP}")
    print("-" * 78)
    grid = _simplex_grid(GRID_STEP)
    grid_preds = P @ grid.T  # (n_unb, n_grid)
    grid_rae = np.empty(grid.shape[0], dtype=np.float64)
    for gi in range(grid.shape[0]):
        grid_rae[gi] = float(rae(y_unb, grid_preds[:, gi]))
    order = np.argsort(grid_rae)
    top5_records = []
    print("   top-5 (w_A, w_B, w_C)  ->  in-sample pooled RAE")
    for k in range(5):
        gi = order[k]
        w = grid[gi]
        top5_records.append({
            "rank": k + 1,
            "w_A_nb1381_AP": float(w[0]),
            "w_B_nb1361_MACCS": float(w[1]),
            "w_C_nb1423_MORD": float(w[2]),
            "rae": float(grid_rae[gi]),
        })
        print(f"   #{k+1}  w = ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})  "
              f"RAE = {grid_rae[gi]:.4f}")
    best_grid_idx = int(order[0])
    w_grid_best = grid[best_grid_idx]
    rae_grid_best = float(grid_rae[best_grid_idx])

    # ---- SLSQP cross-fit ----
    print("\n" + "-" * 78)
    print(f"SLSQP CROSS-FIT  (k={SLSQP_FOLDS}, seed={SLSQP_SEED}, "
          f"n_starts={SLSQP_N_STARTS})")
    print("-" * 78)
    slsqp_oof, fold_weights = _slsqp_cross_fit(
        P, y_unb, SLSQP_FOLDS, SLSQP_SEED, SLSQP_N_STARTS
    )
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    # Median of fold weights for a single representative.
    fold_w_arr = np.array([fw["weights"] for fw in fold_weights])
    median_weights = np.median(fold_w_arr, axis=0)
    median_weights = median_weights / median_weights.sum()
    for fw in fold_weights:
        w = fw["weights"]
        print(f"   fold {fw['fold']}  w = "
              f"({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f})  "
              f"train_rae = {fw['train_rae']:.4f}")
    print(f"   SLSQP cross-fit pooled RAE = {rae_slsqp:.4f}")
    print(f"   median fold weights        = "
          f"({median_weights[0]:.3f}, {median_weights[1]:.3f}, "
          f"{median_weights[2]:.3f})")

    # ---- Naive 1/3 mean ----
    print("\n" + "-" * 78)
    print("NAIVE 1/3 MEAN")
    print("-" * 78)
    naive_oof = P.mean(axis=1)
    rae_naive = float(rae(y_unb, naive_oof))
    print(f"   naive 1/3 pooled RAE = {rae_naive:.4f}")

    # ---- Choose 'best' deployable OOF: prefer SLSQP cross-fit (honest)
    candidates = {
        "slsqp_cross_fit": (slsqp_oof, rae_slsqp),
        "naive_third": (naive_oof, rae_naive),
        "grid_best_in_sample": (
            grid_preds[:, best_grid_idx].copy(), rae_grid_best
        ),
    }
    best_label = min(candidates.keys(), key=lambda k: candidates[k][1])
    best_oof, best_rae = candidates[best_label]
    print("\n" + "-" * 78)
    print(f"BEST DEPLOYABLE OOF = {best_label}   RAE = {best_rae:.4f}")
    print("-" * 78)

    # ---- Verdict vs nb1422 ----
    delta_slsqp = rae_slsqp - NB1422_REF
    delta_naive = rae_naive - NB1422_REF
    delta_best = best_rae - NB1422_REF
    beats_nb1422 = bool(best_rae < NB1422_REF - REPRODUCE_MARGIN)
    ties_nb1422 = bool(abs(delta_best) <= REPRODUCE_MARGIN)
    if beats_nb1422:
        verdict = "BEATS_NB1422"
    elif ties_nb1422:
        verdict = "TIES_NB1422"
    else:
        verdict = "LOSES_TO_NB1422"
    print(f"\n   delta SLSQP  vs nb1422 ({NB1422_REF:.4f}) = {delta_slsqp:+.4f}")
    print(f"   delta naive  vs nb1422 ({NB1422_REF:.4f}) = {delta_naive:+.4f}")
    print(f"   delta BEST   vs nb1422 ({NB1422_REF:.4f}) = {delta_best:+.4f}")
    print(f"   beats_nb1422 ({REPRODUCE_MARGIN} margin) = {beats_nb1422}")
    print(f"   verdict = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy",
            best_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "components": list(COMP_LABELS),
        "component_paths": [str(p.name) for p in paths],
        "standalone_rae": standalone_rae,
        "pairwise_pearson": corr_records,
        "grid_step": GRID_STEP,
        "grid_top5": top5_records,
        "grid_best": {
            "weights": [float(x) for x in w_grid_best],
            "rae": rae_grid_best,
        },
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "slsqp_n_starts": SLSQP_N_STARTS,
        "slsqp_fold_weights": fold_weights,
        "slsqp_median_weights": [float(x) for x in median_weights],
        "slsqp_cross_fit_rae": rae_slsqp,
        "naive_third_rae": rae_naive,
        "best_label": best_label,
        "best_rae": best_rae,
        "nb1422_ref": NB1422_REF,
        "reproduce_margin": REPRODUCE_MARGIN,
        "delta_slsqp_vs_nb1422": delta_slsqp,
        "delta_naive_vs_nb1422": delta_naive,
        "delta_best_vs_nb1422": delta_best,
        "beats_nb1422": beats_nb1422,
        "ties_nb1422": ties_nb1422,
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
        "n_unb", "components", "standalone_rae", "pairwise_pearson",
        "grid_top5", "grid_best",
        "slsqp_cross_fit_rae", "slsqp_median_weights",
        "naive_third_rae",
        "best_label", "best_rae",
        "delta_slsqp_vs_nb1422", "delta_naive_vs_nb1422",
        "delta_best_vs_nb1422",
        "beats_nb1422", "ties_nb1422", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
