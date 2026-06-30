"""nb1474 -- PRE-unblind 2-way blend.

Components:
    A = nb1460_mean_bag_oof.npy   (chemprop_aux anchor + AtomPair-pruned + ChEMBL
                                    residual mean-bag, RAE 0.5550 on 253 unblind)
    B = te_chemprop_aux[unb_idx]  (raw PRE-unblind chemprop_aux anchor,
                                    in_RAE 0.6216 on 253)

Both are PRE-unblind (trained on 4139-only universe -- chemprop_aux is the
backbone, nb1460 is the residual learner anchored to chemprop_aux but its
features and ChEMBL pool are NOT contaminated by the 253 unblind labels).

Protocol:
    1.  Load A, B, y_unb, unb_idx.
    2.  Pearson(A, B), and RAE(A), RAE(B).
    3.  Grid w in {0.0, 0.05, ..., 1.0} for  blend = w*A + (1-w)*B.
        Pool RAE on all 253 (this is in-sample for any single w but
        the grid is 1-D so the optimism is bounded; we cross-fit
        below).
    4.  5-fold KFold cross-fit: per fold pick w* on the 4/5 training
        rows by SLSQP, evaluate on held-out 1/5; concatenate predictions
        and report pooled RAE (LB-honest).
    5.  Verdict at 0.003 margin vs nb1460 standalone (0.5550).
    6.  PRE-unblind LB estimate = best_cross_fit_RAE + 0.003.

Outputs:
    data/processed/nb1474_summary.json
    data/processed/nb1474_best_oof.npy   (253,) float32  -- cross-fit blend
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
from sklearn.model_selection import KFold
from scipy.optimize import minimize

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1474"
DECISION_MARGIN = 0.003
NB1460_REF = 0.5550
CHEMPROP_AUX_REF = 0.6216

W_GRID = np.round(np.arange(0.0, 1.000001, 0.05), 4)
N_SPLITS = 5
SEED = 0


def _slsqp_w(A_tr: np.ndarray, B_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """Find w in [0,1] minimizing pooled RAE of w*A + (1-w)*B on training fold."""
    def loss(w_vec):
        w = float(w_vec[0])
        pred = w * A_tr + (1.0 - w) * B_tr
        return float(rae(y_tr, pred))

    best_w = 0.5
    best_l = float("inf")
    # multi-start to avoid local minima of RAE (piecewise abs)
    for w0 in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        try:
            res = minimize(
                loss,
                x0=np.array([w0], dtype=float),
                method="SLSQP",
                bounds=[(0.0, 1.0)],
                options={"maxiter": 200, "ftol": 1e-9},
            )
            if res.fun < best_l:
                best_l = float(res.fun)
                best_w = float(np.clip(res.x[0], 0.0, 1.0))
        except Exception:
            continue
    return best_w


def main() -> None:
    t0 = time.time()

    # ---- load inputs --------------------------------------------------------
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb = {n_unb}")

    A = np.load(DATA_PROCESSED / "nb1460_mean_bag_oof.npy").astype(np.float64)
    te_chemprop_aux = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    B = te_chemprop_aux[unb_idx]

    assert A.shape == (n_unb,), f"A shape mismatch: {A.shape}"
    assert B.shape == (n_unb,), f"B shape mismatch: {B.shape}"

    rae_A = float(rae(y_unb, A))
    rae_B = float(rae(y_unb, B))
    print(f"[A] nb1460_mean_bag       RAE = {rae_A:.4f}  (expected ~{NB1460_REF})")
    print(f"[B] chemprop_aux te[unb]  RAE = {rae_B:.4f}  (expected ~{CHEMPROP_AUX_REF})")

    # ---- Pearson ------------------------------------------------------------
    pearson = float(np.corrcoef(A, B)[0, 1])
    print(f"[corr] Pearson(A, B) = {pearson:.4f}")

    # ---- w-grid (pooled RAE on all 253, in-sample for picking w*) ----------
    rows = []
    best_grid_w = 0.0
    best_grid_rae = float("inf")
    for w in W_GRID:
        pred = w * A + (1.0 - w) * B
        r = float(rae(y_unb, pred))
        rows.append({"w": float(w), "rae": r})
        if r < best_grid_rae:
            best_grid_rae = r
            best_grid_w = float(w)
    print("[grid] w-sweep (top 12 + extremes):")
    print(f"        w=0.00  -> RAE {rows[0]['rae']:.4f}")
    sorted_rows = sorted(rows, key=lambda r: r["rae"])[:12]
    for r in sorted_rows:
        print(f"        w={r['w']:.2f}  -> RAE {r['rae']:.4f}")
    print(f"        w=1.00  -> RAE {rows[-1]['rae']:.4f}")
    print(f"[grid] best w = {best_grid_w:.2f}  RAE = {best_grid_rae:.4f}")

    # ---- 5-fold cross-fit SLSQP --------------------------------------------
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cf_pred = np.zeros(n_unb, dtype=np.float64)
    fold_ws = []
    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(np.arange(n_unb))):
        w_star = _slsqp_w(A[tr_idx], B[tr_idx], y_unb[tr_idx])
        cf_pred[te_idx] = w_star * A[te_idx] + (1.0 - w_star) * B[te_idx]
        fold_ws.append(w_star)
        print(f"   [fold {fold_i}] w* = {w_star:.4f}")
    rae_cross_fit = float(rae(y_unb, cf_pred))
    mean_w = float(np.mean(fold_ws))
    print(f"[slsqp] cross-fit RAE = {rae_cross_fit:.4f}   mean(w*) = {mean_w:.4f}")

    # ---- verdict ------------------------------------------------------------
    delta_vs_nb1460 = rae_cross_fit - NB1460_REF
    beats_nb1460 = rae_cross_fit < (NB1460_REF - DECISION_MARGIN)
    predicted_lb = rae_cross_fit + DECISION_MARGIN

    if beats_nb1460:
        verdict = "BEATS_NB1460_AT_MARGIN"
    elif rae_cross_fit < NB1460_REF:
        verdict = "BEATS_NB1460_WITHIN_NOISE"
    elif abs(delta_vs_nb1460) <= DECISION_MARGIN:
        verdict = "EQUIVALENT_TO_NB1460"
    else:
        verdict = "WORSE_THAN_NB1460"

    print(f"[verdict] {verdict}")
    print(f"   delta vs nb1460 (0.5550): {delta_vs_nb1460:+.4f}")
    print(f"   PRE-unblind LB estimate : {predicted_lb:.4f}")

    # ---- save ---------------------------------------------------------------
    out_oof = DATA_PROCESSED / f"{TAG}_best_oof.npy"
    np.save(out_oof, cf_pred.astype(np.float32))
    print(f"[save] {out_oof}")

    summary = {
        "tag": TAG,
        "n_unb": int(n_unb),
        "anchor_A": "nb1460_mean_bag_oof",
        "anchor_B": "te_chemprop_aux[unb_idx]",
        "regime": "PRE_unblind_both",
        "rae_A_nb1460": rae_A,
        "rae_B_chemprop_aux": rae_B,
        "pearson_A_B": pearson,
        "w_grid": [float(x) for x in W_GRID],
        "grid_rows": rows,
        "best_grid_w": best_grid_w,
        "best_grid_rae": best_grid_rae,
        "n_splits": N_SPLITS,
        "fold_ws": fold_ws,
        "mean_w_cross_fit": mean_w,
        "rae_cross_fit_slsqp": rae_cross_fit,
        "delta_vs_nb1460": delta_vs_nb1460,
        "beats_nb1460_at_margin": bool(beats_nb1460),
        "decision_margin": DECISION_MARGIN,
        "nb1460_ref": NB1460_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "predicted_lb": predicted_lb,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = DATA_PROCESSED / f"{TAG}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] {out_json}")
    print(f"[done] wall {summary['wall_sec']}s")


if __name__ == "__main__":
    main()
