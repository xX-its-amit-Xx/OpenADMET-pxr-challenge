"""nb1394 -- Triple-blend nb1373 (AtomPair-30) + nb1352 (MACCS-20) + nb1372 (dual MACCS+Mordred).

Hypothesis:
    Three SHAP-pruned learners with different feature axes:
        nb1373 -- AtomPair-30 + ChEMBL  (mean_bag 0.5095)  [topology/distance]
        nb1352 -- MACCS-20 + ChEMBL     (mean_bag 0.5323)  [substructure]
        nb1372 -- MACCS-20 + Mordred-30 + ChEMBL (mean_bag 0.5207)  [structure+descriptor]
    Even with high pairwise correlation, a 3-way simplex blend may extract
    residual orthogonal signal vs nb1373 standalone.

Protocol:
    1. Load three mean_bag_oof.npy arrays (253,).
    2. Compute pairwise Pearson.
    3. 3D simplex grid search (step 0.05) over (w1, w2, w3) with w_i >= 0,
       sum == 1.  Top-5 tuples by pooled RAE.
    4. 5-fold cross-fit SLSQP simplex optimization (fold-out weights ->
       held-out fold prediction; assemble cross-fit OOF).
    5. Naive 1/3 mean baseline.
    6. Verdict at 0.003 margin vs nb1373 (0.5095).

Outputs:
    scripts/nb1394_triple_pruned_blend.py             (this file)
    data/processed/nb1394_summary.json
    data/processed/nb1394_best_oof.npy                (253,) float32
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1394"

NB1373_OOF = DATA_PROCESSED / "nb1373_mean_bag_oof.npy"
NB1352_OOF = DATA_PROCESSED / "nb1352_mean_bag_oof.npy"
NB1372_OOF = DATA_PROCESSED / "nb1372_mean_bag_oof.npy"

NB1373_REF = 0.5095
NB1352_REF = 0.5323
NB1372_REF = 0.5207
DECISION_MARGIN = 0.003

GRID_STEP = 0.05
CROSS_FIT_FOLDS = 5
CROSS_FIT_SEED = 42


def _simplex_grid(step: float) -> np.ndarray:
    """Generate all 3-tuples (w1,w2,w3) on the simplex with given step."""
    n = int(round(1.0 / step)) + 1
    vals = np.linspace(0.0, 1.0, n)
    pts = []
    for i, w1 in enumerate(vals):
        for j, w2 in enumerate(vals):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            if w3 < 0.0:
                w3 = 0.0
            pts.append((w1, w2, w3))
    return np.array(pts, dtype=np.float64)


def _slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find non-negative simplex weights minimizing RAE(P @ w, y).

    P: (n, k) per-source preds, y: (n,) truth.
    """
    k = P.shape[1]

    def obj(w):
        pred = P @ w
        return rae(y, pred)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, 1.0)] * k
    w0 = np.full(k, 1.0 / k)
    res = minimize(
        obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )
    w = res.x
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0.0:
        w = np.full(k, 1.0 / k)
    else:
        w = w / s
    return w


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- TRIPLE BLEND nb1373 + nb1352 + nb1372")
    print(f"          grid step = {GRID_STEP}  cross-fit folds = {CROSS_FIT_FOLDS}")
    print(f"          ref nb1373 = {NB1373_REF:.4f}  margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load OOFs + truth ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    oof_1373 = np.load(NB1373_OOF).astype(np.float64)
    oof_1352 = np.load(NB1352_OOF).astype(np.float64)
    oof_1372 = np.load(NB1372_OOF).astype(np.float64)
    assert oof_1373.shape == (n_unb,)
    assert oof_1352.shape == (n_unb,)
    assert oof_1372.shape == (n_unb,)
    print(f"[load] n_unb = {n_unb}")

    rae_1373 = float(rae(y_unb, oof_1373))
    rae_1352 = float(rae(y_unb, oof_1352))
    rae_1372 = float(rae(y_unb, oof_1372))
    print(f"[base] nb1373 RAE = {rae_1373:.4f}  (ref {NB1373_REF:.4f})")
    print(f"[base] nb1352 RAE = {rae_1352:.4f}  (ref {NB1352_REF:.4f})")
    print(f"[base] nb1372 RAE = {rae_1372:.4f}  (ref {NB1372_REF:.4f})")

    # ---- Pairwise Pearson ----
    pearson_1373_1352 = float(np.corrcoef(oof_1373, oof_1352)[0, 1])
    pearson_1373_1372 = float(np.corrcoef(oof_1373, oof_1372)[0, 1])
    pearson_1352_1372 = float(np.corrcoef(oof_1352, oof_1372)[0, 1])
    print("\n[pearson] pairwise correlations")
    print(f"   nb1373 vs nb1352 = {pearson_1373_1352:.4f}")
    print(f"   nb1373 vs nb1372 = {pearson_1373_1372:.4f}")
    print(f"   nb1352 vs nb1372 = {pearson_1352_1372:.4f}")

    # ---- Stack: cols are [nb1373, nb1352, nb1372] ----
    P = np.column_stack([oof_1373, oof_1352, oof_1372])  # (253, 3)
    src_names = ["nb1373", "nb1352", "nb1372"]

    # ---- 3D simplex grid search ----
    grid = _simplex_grid(GRID_STEP)
    n_pts = len(grid)
    print(f"\n[grid] simplex points (step={GRID_STEP}): {n_pts}")
    grid_preds = P @ grid.T  # (253, n_pts)
    grid_rae = np.array([float(rae(y_unb, grid_preds[:, i]))
                         for i in range(n_pts)])
    order = np.argsort(grid_rae)
    top5_idx = order[:5]
    print("\n[grid] top-5 tuples (w_nb1373, w_nb1352, w_nb1372) -> RAE")
    top5_records = []
    for rank, idx in enumerate(top5_idx, start=1):
        w = grid[idx]
        r = grid_rae[idx]
        print(f"   #{rank}: w=({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}) "
              f"RAE = {r:.4f}  d_vs_nb1373 = {r - NB1373_REF:+.4f}")
        top5_records.append({
            "rank": rank,
            "w_nb1373": float(w[0]),
            "w_nb1352": float(w[1]),
            "w_nb1372": float(w[2]),
            "rae": float(r),
            "delta_vs_nb1373": float(r - NB1373_REF),
        })
    best_grid_idx = int(top5_idx[0])
    best_grid_w = grid[best_grid_idx].copy()
    best_grid_rae = float(grid_rae[best_grid_idx])
    best_grid_oof = P @ best_grid_w

    # ---- In-sample SLSQP simplex ----
    w_slsqp_in = _slsqp_simplex(P, y_unb)
    oof_slsqp_in = P @ w_slsqp_in
    rae_slsqp_in = float(rae(y_unb, oof_slsqp_in))
    print("\n[slsqp-in] in-sample SLSQP simplex")
    print(f"   w = ({w_slsqp_in[0]:.4f}, {w_slsqp_in[1]:.4f}, "
          f"{w_slsqp_in[2]:.4f})")
    print(f"   RAE = {rae_slsqp_in:.4f}  d_vs_nb1373 = "
          f"{rae_slsqp_in - NB1373_REF:+.4f}")

    # ---- 5-fold cross-fit SLSQP ----
    kf = KFold(n_splits=CROSS_FIT_FOLDS, shuffle=True, random_state=CROSS_FIT_SEED)
    oof_slsqp_cf = np.full(n_unb, np.nan, dtype=np.float64)
    fold_weights = []
    for fold_i, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w_fold = _slsqp_simplex(P[tr_loc], y_unb[tr_loc])
        oof_slsqp_cf[va_loc] = P[va_loc] @ w_fold
        fold_weights.append(w_fold.tolist())
    rae_slsqp_cf = float(rae(y_unb, oof_slsqp_cf))
    print("\n[slsqp-cf] 5-fold cross-fit SLSQP")
    for i, w in enumerate(fold_weights):
        print(f"   fold {i}: w = ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})")
    print(f"   cross-fit RAE = {rae_slsqp_cf:.4f}  d_vs_nb1373 = "
          f"{rae_slsqp_cf - NB1373_REF:+.4f}")

    # ---- Naive 1/3 mean ----
    w_naive = np.full(3, 1.0 / 3.0)
    oof_naive = P @ w_naive
    rae_naive = float(rae(y_unb, oof_naive))
    print("\n[naive] 1/3 mean")
    print(f"   RAE = {rae_naive:.4f}  d_vs_nb1373 = "
          f"{rae_naive - NB1373_REF:+.4f}")

    # ---- Best of all 3 candidate aggregations ----
    candidates = {
        "grid_best": (best_grid_w.tolist(), best_grid_rae, best_grid_oof),
        "slsqp_in_sample": (w_slsqp_in.tolist(), rae_slsqp_in, oof_slsqp_in),
        "slsqp_cross_fit": (None, rae_slsqp_cf, oof_slsqp_cf),
        "naive_mean": (w_naive.tolist(), rae_naive, oof_naive),
    }
    # honest-best: cross-fit if it beats nb1373; otherwise grid (which IS
    # in-sample but matches Bayesian-optimal-on-tiny-grid spirit).
    # Recommended save = cross-fit OOF (honest).
    best_oof = oof_slsqp_cf
    best_rae = rae_slsqp_cf
    best_w = None  # cross-fit weights vary per fold

    # ---- Verdict (use cross-fit SLSQP as honest measure) ----
    beats_nb1373_grid = best_grid_rae < NB1373_REF - DECISION_MARGIN
    beats_nb1373_slsqp_in = rae_slsqp_in < NB1373_REF - DECISION_MARGIN
    beats_nb1373_slsqp_cf = rae_slsqp_cf < NB1373_REF - DECISION_MARGIN
    beats_nb1373_naive = rae_naive < NB1373_REF - DECISION_MARGIN

    if beats_nb1373_slsqp_cf:
        verdict = "TRIPLE_BLEND_CROSSFIT_BEATS_NB1373_NEW_PRIMARY_CANDIDATE"
    elif abs(rae_slsqp_cf - NB1373_REF) < DECISION_MARGIN:
        verdict = "TRIPLE_BLEND_CROSSFIT_FLAT_VS_NB1373"
    elif beats_nb1373_grid:
        verdict = "TRIPLE_BLEND_GRID_BEATS_NB1373_BUT_CROSSFIT_DOES_NOT"
    else:
        verdict = "TRIPLE_BLEND_HURTS_NB1373"

    print("\n" + "-" * 78)
    print(f"VERDICT: {verdict}")
    print("-" * 78)
    print(f"   beats_nb1373_grid          = {beats_nb1373_grid}")
    print(f"   beats_nb1373_slsqp_in      = {beats_nb1373_slsqp_in}")
    print(f"   beats_nb1373_slsqp_cf      = {beats_nb1373_slsqp_cf}")
    print(f"   beats_nb1373_naive         = {beats_nb1373_naive}")

    # ---- Save ----
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy",
            best_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}  (cross-fit SLSQP)")

    summary = {
        "tag": TAG,
        "sources": src_names,
        "n_unb": n_unb,
        "base_rae": {
            "nb1373": rae_1373,
            "nb1352": rae_1352,
            "nb1372": rae_1372,
        },
        "pearson": {
            "nb1373_vs_nb1352": pearson_1373_1352,
            "nb1373_vs_nb1372": pearson_1373_1372,
            "nb1352_vs_nb1372": pearson_1352_1372,
        },
        "grid_step": GRID_STEP,
        "grid_n_pts": int(n_pts),
        "grid_top5": top5_records,
        "grid_best_w": best_grid_w.tolist(),
        "grid_best_rae": best_grid_rae,
        "slsqp_in_sample_w": w_slsqp_in.tolist(),
        "slsqp_in_sample_rae": rae_slsqp_in,
        "slsqp_cross_fit_fold_weights": fold_weights,
        "slsqp_cross_fit_rae": rae_slsqp_cf,
        "naive_mean_rae": rae_naive,
        "delta_grid_vs_nb1373": best_grid_rae - NB1373_REF,
        "delta_slsqp_in_vs_nb1373": rae_slsqp_in - NB1373_REF,
        "delta_slsqp_cf_vs_nb1373": rae_slsqp_cf - NB1373_REF,
        "delta_naive_vs_nb1373": rae_naive - NB1373_REF,
        "beats_nb1373_grid": bool(beats_nb1373_grid),
        "beats_nb1373_slsqp_in": bool(beats_nb1373_slsqp_in),
        "beats_nb1373_slsqp_cf": bool(beats_nb1373_slsqp_cf),
        "beats_nb1373_naive": bool(beats_nb1373_naive),
        "best_oof_source": "slsqp_cross_fit",
        "best_oof_rae": best_rae,
        "verdict": verdict,
        "nb1373_ref": NB1373_REF,
        "nb1352_ref": NB1352_REF,
        "nb1372_ref": NB1372_REF,
        "decision_margin": DECISION_MARGIN,
        "cross_fit_folds": CROSS_FIT_FOLDS,
        "cross_fit_seed": CROSS_FIT_SEED,
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
        "sources", "n_unb",
        "base_rae", "pearson",
        "grid_step", "grid_n_pts",
        "grid_best_w", "grid_best_rae",
        "slsqp_in_sample_w", "slsqp_in_sample_rae",
        "slsqp_cross_fit_rae", "naive_mean_rae",
        "delta_grid_vs_nb1373", "delta_slsqp_in_vs_nb1373",
        "delta_slsqp_cf_vs_nb1373", "delta_naive_vs_nb1373",
        "beats_nb1373_grid", "beats_nb1373_slsqp_in",
        "beats_nb1373_slsqp_cf", "beats_nb1373_naive",
        "best_oof_source", "best_oof_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
