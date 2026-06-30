"""nb1751 -- 4-way ensemble of BoB-validated candidates.

Components (all 253-row honest cross-fit OOFs, all BoB-validated):
    p1 = nb1632 bag-of-bags mean   (RAE 0.5107 -- best PRIMARY)
    p2 = nb1561 bag-of-bags mean
    p3 = nb1500 bag-of-bags mean
    p4 = nb1521 bag-of-bags mean

Protocol:
  1. Load 4 BoB-mean OOFs + _audit_unblind_y.
  2. Pairwise Pearson on predictions and residuals.
  3. 4D simplex GRID search, step 0.05 (w_i>=0, sum_i w_i == 1).
  4. 5-fold SLSQP cross-fit on the 4-simplex.
  5. Naive 1/4 mean.
  6. Verdict at 0.003 margin vs nb1632 (0.5107).

Outputs:
  data/processed/nb1751_grid_oof.npy
  data/processed/nb1751_slsqp_oof.npy
  data/processed/nb1751_mean_oof.npy
  data/processed/nb1751_summary.json
"""
from __future__ import annotations

import itertools
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

TAG = "nb1751"
SLSQP_FOLDS = 5
SLSQP_SEED = 42
GRID_STEP = 0.05

NB1632_BOB_REF = 0.5107
MARGIN = 0.003

COMP_TAGS = ["nb1632", "nb1561", "nb1500", "nb1521"]
COMP_FILES = [f"{t}_bob_mean_oof.npy" for t in COMP_TAGS]


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w: np.ndarray) -> float:
        r = y_tr - P_tr @ w
        return float(np.mean(r * r))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        _loss, w0, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        return np.full(K, 1.0 / K)
    return w / s


def _slsqp_cross_fit(P: np.ndarray, y: np.ndarray,
                     n_splits: int, seed: int
                     ) -> tuple[np.ndarray, list[dict]]:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_blend_weights(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w
        fold_records.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "weights": [float(x) for x in w],
        })
    return oof, fold_records


def _simplex_grid(step: float, K: int = 4) -> np.ndarray:
    """Enumerate K-simplex grid with mesh `step`. Returns (M, K) array."""
    n = int(round(1.0 / step))
    pts = []
    for combo in itertools.product(range(n + 1), repeat=K - 1):
        s = sum(combo)
        if s <= n:
            last = n - s
            pts.append([c / n for c in combo] + [last / n])
    return np.array(pts, dtype=np.float64)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 4-way BoB ensemble  "
          f"nb1632 + nb1561 + nb1500 + nb1521")
    print(f"         grid step {GRID_STEP}  +  5-fold SLSQP cross-fit  "
          f"+  naive mean")
    print(f"         target margin {MARGIN:.3f} vs nb1632 BoB "
          f"{NB1632_BOB_REF:.4f}")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = [DATA_PROCESSED / fn for fn in COMP_FILES]
    for tag, p in zip(COMP_TAGS, paths):
        if not p.exists():
            raise FileNotFoundError(f"{tag}: {p}")

    preds_list = [np.load(p).astype(np.float64) for p in paths]
    for tag, arr in zip(COMP_TAGS, preds_list):
        if arr.shape[0] != n_unb:
            raise ValueError(
                f"shape mismatch: {tag}={arr.shape}, n_unb={n_unb}")

    p1, p2, p3, p4 = preds_list
    rae_p1 = float(rae(y_unb, p1))
    rae_p2 = float(rae(y_unb, p2))
    rae_p3 = float(rae(y_unb, p3))
    rae_p4 = float(rae(y_unb, p4))
    standalone = {
        "nb1632": rae_p1,
        "nb1561": rae_p2,
        "nb1500": rae_p3,
        "nb1521": rae_p4,
    }
    print(f"[load] nb1632 BoB mean : RAE = {rae_p1:.4f}  "
          f"(ref {NB1632_BOB_REF:.4f})")
    print(f"[load] nb1561 BoB mean : RAE = {rae_p2:.4f}")
    print(f"[load] nb1500 BoB mean : RAE = {rae_p3:.4f}")
    print(f"[load] nb1521 BoB mean : RAE = {rae_p4:.4f}")

    # Pairwise correlations (preds + residuals).
    e1, e2, e3, e4 = p1 - y_unb, p2 - y_unb, p3 - y_unb, p4 - y_unb
    preds = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}
    resids = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}
    corr_pred = {}
    corr_resid = {}
    keys = ["p1", "p2", "p3", "p4"]
    rkeys = ["e1", "e2", "e3", "e4"]
    name_for_p = {"p1": "nb1632", "p2": "nb1561",
                  "p3": "nb1500", "p4": "nb1521"}
    for i in range(4):
        for j in range(i + 1, 4):
            corr_pred[
                f"{name_for_p[keys[i]]}_{name_for_p[keys[j]]}"
            ] = float(np.corrcoef(preds[keys[i]], preds[keys[j]])[0, 1])
            corr_resid[
                f"{name_for_p[keys[i]]}_{name_for_p[keys[j]]}"
            ] = float(np.corrcoef(resids[rkeys[i]], resids[rkeys[j]])[0, 1])
    print("[diag] pred  corr:",
          {k: f"{v:.4f}" for k, v in corr_pred.items()})
    print("[diag] resid corr:",
          {k: f"{v:.4f}" for k, v in corr_resid.items()})

    P = np.column_stack([p1, p2, p3, p4])  # (253, 4)

    # (A) 4D simplex grid search, step 0.05.
    print("\n" + "-" * 78)
    print(f"(A) 4D simplex grid search  (step {GRID_STEP}, K=4)")
    print("-" * 78)
    W = _simplex_grid(GRID_STEP, K=4)
    print(f"   grid size = {len(W)} points")
    preds_grid = W @ P.T            # (M, 253)
    err = preds_grid - y_unb[None, :]
    abs_err = np.abs(err).sum(axis=1)
    denom = float(np.abs(y_unb - np.mean(y_unb)).sum())
    raes_grid = abs_err / denom
    order = np.argsort(raes_grid)[:5]
    top5 = []
    for rank, idx in enumerate(order, start=1):
        w = W[idx]
        r = float(raes_grid[idx])
        top5.append({
            "rank": int(rank),
            "weights": {
                "nb1632": float(w[0]),
                "nb1561": float(w[1]),
                "nb1500": float(w[2]),
                "nb1521": float(w[3]),
            },
            "rae": r,
        })
        print(f"   rank {rank}: "
              f"w1632={w[0]:.2f}  w1561={w[1]:.2f}  "
              f"w1500={w[2]:.2f}  w1521={w[3]:.2f}  "
              f"RAE = {r:.4f}")
    best_grid_w = W[order[0]]
    grid_oof = (P @ best_grid_w).astype(np.float64)
    rae_grid = float(rae(y_unb, grid_oof))
    print(f"   best grid RAE (recheck) = {rae_grid:.4f}")
    print(f"     d_vs_nb1632 = {rae_grid - NB1632_BOB_REF:+.4f}")

    # (B) 5-fold SLSQP cross-fit.
    print("\n" + "-" * 78)
    print("(B) 5-fold SLSQP cross-fit blend  (simplex: w>=0, sum==1, K=4)")
    print("-" * 78)
    slsqp_oof, fold_records = _slsqp_cross_fit(
        P, y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED
    )
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    print("   per-fold weights:")
    for rec in fold_records:
        w = rec["weights"]
        print(f"     fold {rec['fold']}: "
              f"w1632={w[0]:.4f}  w1561={w[1]:.4f}  "
              f"w1500={w[2]:.4f}  w1521={w[3]:.4f}  "
              f"(n_tr={rec['n_tr']}, n_va={rec['n_va']})")
    fold_weights = np.array([r["weights"] for r in fold_records])
    mean_w = fold_weights.mean(axis=0)
    std_w = fold_weights.std(axis=0)
    print(f"   mean weights:  w1632={mean_w[0]:.4f}  "
          f"w1561={mean_w[1]:.4f}  w1500={mean_w[2]:.4f}  "
          f"w1521={mean_w[3]:.4f}")
    print(f"   std  weights:  w1632={std_w[0]:.4f}  "
          f"w1561={std_w[1]:.4f}  w1500={std_w[2]:.4f}  "
          f"w1521={std_w[3]:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}")
    print(f"     d_vs_nb1632 = {rae_slsqp - NB1632_BOB_REF:+.4f}")

    # In-sample SLSQP diagnostic.
    w_full = _slsqp_blend_weights(P, y_unb)
    p_full = P @ w_full
    rae_full = float(rae(y_unb, p_full))
    print(f"   in-sample SLSQP (diagnostic): "
          f"w1632={w_full[0]:.4f}  w1561={w_full[1]:.4f}  "
          f"w1500={w_full[2]:.4f}  w1521={w_full[3]:.4f}  "
          f"RAE = {rae_full:.4f}")

    # (C) Naive 1/4 mean.
    print("\n" + "-" * 78)
    print("(C) Naive 1/4 mean")
    print("-" * 78)
    mean_oof = (p1 + p2 + p3 + p4) / 4.0
    rae_mean = float(rae(y_unb, mean_oof))
    print(f"   pooled RAE(mean) = {rae_mean:.4f}  "
          f"(d_vs_nb1632 = {rae_mean - NB1632_BOB_REF:+.4f})")

    # Verdict.
    cand_rae = {
        "grid": rae_grid,
        "slsqp": rae_slsqp,
        "mean": rae_mean,
    }
    best_variant = min(cand_rae, key=cand_rae.get)
    best_rae = cand_rae[best_variant]
    beats_nb1632 = best_rae < NB1632_BOB_REF - MARGIN
    flat_vs_nb1632 = abs(best_rae - NB1632_BOB_REF) < MARGIN

    if beats_nb1632:
        verdict = f"4BOB_BEATS_NB1632_BY_MARGIN  (best={best_variant})"
    elif flat_vs_nb1632:
        verdict = f"4BOB_FLAT_VS_NB1632  (best={best_variant})"
    else:
        verdict = f"4BOB_HURTS_VS_NB1632  (best={best_variant})"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1632 BoB mean standalone : {rae_p1:.4f}")
    print(f"   nb1561 BoB mean standalone : {rae_p2:.4f}")
    print(f"   nb1500 BoB mean standalone : {rae_p3:.4f}")
    print(f"   nb1521 BoB mean standalone : {rae_p4:.4f}")
    print(f"   4-way grid (step {GRID_STEP}) : {rae_grid:.4f}")
    print(f"   4-way SLSQP cross-fit      : {rae_slsqp:.4f}")
    print(f"   4-way naive 1/4 mean       : {rae_mean:.4f}")
    print(f"   best variant               : {best_variant}  ({best_rae:.4f})")
    print(f"   margin vs nb1632 BoB       : "
          f"{best_rae - NB1632_BOB_REF:+.4f} (gate {MARGIN:.3f})")
    print(f"   beats_nb1632               : {beats_nb1632}")
    print(f"   verdict                    : {verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_grid_oof.npy", grid_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_slsqp_oof.npy",
            slsqp_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_oof.npy", mean_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_grid_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_slsqp_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "grid_step": GRID_STEP,
        "components": COMP_FILES,
        "component_tags": COMP_TAGS,
        "rae_standalone": standalone,
        "pred_corr": corr_pred,
        "residual_corr": corr_resid,
        "top5_grid": top5,
        "best_grid_weights": {
            "nb1632": float(best_grid_w[0]),
            "nb1561": float(best_grid_w[1]),
            "nb1500": float(best_grid_w[2]),
            "nb1521": float(best_grid_w[3]),
        },
        "rae_grid": rae_grid,
        "rae_slsqp_cross_fit": rae_slsqp,
        "rae_naive_mean": rae_mean,
        "in_sample_slsqp_weights": {
            "nb1632": float(w_full[0]),
            "nb1561": float(w_full[1]),
            "nb1500": float(w_full[2]),
            "nb1521": float(w_full[3]),
        },
        "in_sample_slsqp_rae": rae_full,
        "fold_records_slsqp": fold_records,
        "mean_fold_weights_slsqp": {
            "nb1632": float(mean_w[0]),
            "nb1561": float(mean_w[1]),
            "nb1500": float(mean_w[2]),
            "nb1521": float(mean_w[3]),
        },
        "std_fold_weights_slsqp": {
            "nb1632": float(std_w[0]),
            "nb1561": float(std_w[1]),
            "nb1500": float(std_w[2]),
            "nb1521": float(std_w[3]),
        },
        "best_variant": best_variant,
        "best_rae": best_rae,
        "delta_best_vs_nb1632": best_rae - NB1632_BOB_REF,
        "beats_nb1632": bool(beats_nb1632),
        "flat_vs_nb1632": bool(flat_vs_nb1632),
        "verdict": verdict,
        "nb1632_bob_ref": NB1632_BOB_REF,
        "margin": MARGIN,
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
    for k in ("rae_standalone",
              "pred_corr", "residual_corr",
              "top5_grid", "best_grid_weights",
              "rae_grid", "rae_slsqp_cross_fit", "rae_naive_mean",
              "in_sample_slsqp_weights",
              "mean_fold_weights_slsqp", "std_fold_weights_slsqp",
              "best_variant", "best_rae",
              "delta_best_vs_nb1632",
              "beats_nb1632", "verdict"):
        print(f"  {k}: {res.get(k)}")
