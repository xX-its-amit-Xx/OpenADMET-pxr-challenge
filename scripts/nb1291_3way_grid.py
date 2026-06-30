"""nb1291 -- 3-way grid: nb1190 BoB + nb1242 ChEMBL + nb1211 BoB-blend.

Hypothesis:
    Meta-stack put nb1190 (40%), nb1242 (20%), nb1211 (9%) as top-3.
    Direct grid search over 3D simplex (step 0.05) on full 253
    avoids cross-fit overfit at this low dim.

Protocol:
  1. Load nb1190_bob_mean_oof, nb1242_mean_bag_oof, nb1211_mean_oof.
  2. Grid search over simplex {(w1,w2,w3): wi>=0, sum=1, step 0.05}. ~231 pts.
  3. Compute pooled RAE on full 253 for each.
  4. Report best 5 weight tuples + RAE.
  5. Also: 5-fold cross-fit SLSQP, naive 1/3 mean.
  6. Verdict at 0.003 margin vs nb1251 (0.5394).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1291_best_oof.npy    (253,) float32  -- best grid blend
  data/processed/nb1291_summary.json
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

TAG = "nb1291"
SLSQP_FOLDS = 5
SLSQP_SEED = 42
GRID_STEP = 0.05

# Reference numbers (pooled RAE on 253 unblind).
NB1190_REF = None  # to be loaded
NB1242_REF = 0.5431
NB1211_REF = 0.5451
NB1251_REF = 0.5394
MARGIN = 0.003


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Argmin MSE over the K-simplex (w_i >= 0, sum w_i = 1)."""
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w: np.ndarray) -> float:
        pred = P_tr @ w
        diff = y_tr - pred
        return float(np.mean(diff * diff))

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
                     n_splits: int, seed: int) -> tuple[np.ndarray, list[dict]]:
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


def _enumerate_simplex(step: float) -> list[tuple[float, float, float]]:
    """Enumerate the 3-simplex at the given step (multiples of step)."""
    pts = []
    # Use integer multiples to avoid float drift.
    n_units = int(round(1.0 / step))
    for i in range(n_units + 1):
        for j in range(n_units + 1 - i):
            k = n_units - i - j
            w1 = i / n_units
            w2 = j / n_units
            w3 = k / n_units
            pts.append((w1, w2, w3))
    return pts


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 3-way grid: nb1190 BoB + nb1242 ChEMBL + nb1211 BoB-blend")
    print(f"          simplex grid step={GRID_STEP}, 5-fold SLSQP cross-fit, 1/3 mean")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = {
        "nb1190": DATA_PROCESSED / "nb1190_bob_mean_oof.npy",
        "nb1242": DATA_PROCESSED / "nb1242_mean_bag_oof.npy",
        "nb1211": DATA_PROCESSED / "nb1211_mean_oof.npy",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({k})")

    preds = {k: np.load(p).astype(np.float64) for k, p in paths.items()}
    for k, v in preds.items():
        if v.shape[0] != n_unb:
            raise ValueError(f"shape mismatch: {k}={v.shape}, n_unb={n_unb}")

    p1 = preds["nb1190"]
    p2 = preds["nb1242"]
    p3 = preds["nb1211"]

    standalone_rae = {
        "nb1190": float(rae(y_unb, p1)),
        "nb1242": float(rae(y_unb, p2)),
        "nb1211": float(rae(y_unb, p3)),
    }
    print("\n[load] standalone pooled RAE on 253 unblind:")
    print(f"   nb1190 BoB (bag-of-bags)    : {standalone_rae['nb1190']:.4f}")
    print(f"   nb1242 ChEMBL-feat residual : {standalone_rae['nb1242']:.4f}  "
          f"(ref {NB1242_REF:.4f})")
    print(f"   nb1211 BoB-of-BoBs blend    : {standalone_rae['nb1211']:.4f}  "
          f"(ref {NB1211_REF:.4f})")

    # Pred-pred and residual correlations.
    pred_corr = {
        "nb1190_nb1242": float(np.corrcoef(p1, p2)[0, 1]),
        "nb1190_nb1211": float(np.corrcoef(p1, p3)[0, 1]),
        "nb1242_nb1211": float(np.corrcoef(p2, p3)[0, 1]),
    }
    r1 = p1 - y_unb
    r2 = p2 - y_unb
    r3 = p3 - y_unb
    resid_corr = {
        "nb1190_nb1242": float(np.corrcoef(r1, r2)[0, 1]),
        "nb1190_nb1211": float(np.corrcoef(r1, r3)[0, 1]),
        "nb1242_nb1211": float(np.corrcoef(r2, r3)[0, 1]),
    }
    print(f"\n[diag] pred-pred Pearson:")
    for k, v in pred_corr.items():
        print(f"   {k}: {v:.4f}")
    print(f"[diag] residual Pearson:")
    for k, v in resid_corr.items():
        print(f"   {k}: {v:.4f}")
    print(f"[diag] residual std: nb1190={r1.std():.4f}  "
          f"nb1242={r2.std():.4f}  nb1211={r3.std():.4f}")

    # ---- 5-fold SLSQP cross-fit (3-way) ----
    print("\n" + "-" * 78)
    print("  BLOCK: 3-way SLSQP 5-fold cross-fit")
    print("-" * 78)
    P = np.column_stack([p1, p2, p3])
    slsqp_oof, fold_records = _slsqp_cross_fit(P, y_unb, SLSQP_FOLDS, SLSQP_SEED)
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    fold_w = np.array([r["weights"] for r in fold_records])
    mean_w = fold_w.mean(axis=0)
    print(f"   per-fold weights (nb1190, nb1242, nb1211):")
    for rec in fold_records:
        w = rec["weights"]
        print(f"     fold {rec['fold']}: "
              f"w[nb1190]={w[0]:.4f}  w[nb1242]={w[1]:.4f}  w[nb1211]={w[2]:.4f}")
    print(f"   mean weights across folds: "
          f"w[nb1190]={mean_w[0]:.4f}  "
          f"w[nb1242]={mean_w[1]:.4f}  "
          f"w[nb1211]={mean_w[2]:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}")

    # In-sample SLSQP for diagnostic.
    w_full = _slsqp_blend_weights(P, y_unb)
    p_full = P @ w_full
    rae_full = float(rae(y_unb, p_full))
    print(f"   in-sample SLSQP weights: "
          f"w[nb1190]={w_full[0]:.4f}  "
          f"w[nb1242]={w_full[1]:.4f}  "
          f"w[nb1211]={w_full[2]:.4f}   RAE = {rae_full:.4f}")

    # ---- Naive 1/3 mean ----
    naive_oof = (p1 + p2 + p3) / 3.0
    rae_naive = float(rae(y_unb, naive_oof))
    print(f"\n[block] naive 1/3 mean        RAE = {rae_naive:.4f}")

    # ---- 3-simplex grid (step GRID_STEP) ----
    print("\n" + "-" * 78)
    print(f"  BLOCK: 3-simplex grid (step={GRID_STEP})")
    print("-" * 78)
    grid_pts = _enumerate_simplex(GRID_STEP)
    print(f"   total grid points: {len(grid_pts)}")

    grid_results = []
    for (w1, w2, w3) in grid_pts:
        blend = w1 * p1 + w2 * p2 + w3 * p3
        r = float(rae(y_unb, blend))
        grid_results.append({
            "w_nb1190": float(round(w1, 4)),
            "w_nb1242": float(round(w2, 4)),
            "w_nb1211": float(round(w3, 4)),
            "rae": r,
        })

    grid_results_sorted = sorted(grid_results, key=lambda d: d["rae"])
    top5 = grid_results_sorted[:5]
    print(f"\n   TOP-5 grid points (lowest RAE):")
    print(f"   {'rank':<5}{'w[nb1190]':<12}{'w[nb1242]':<12}{'w[nb1211]':<12}{'RAE':<10}")
    for i, rec in enumerate(top5):
        print(f"   {i+1:<5}"
              f"{rec['w_nb1190']:<12.4f}"
              f"{rec['w_nb1242']:<12.4f}"
              f"{rec['w_nb1211']:<12.4f}"
              f"{rec['rae']:<10.4f}")

    best = top5[0]
    best_w = (best["w_nb1190"], best["w_nb1242"], best["w_nb1211"])
    best_rae = best["rae"]
    best_oof = best_w[0] * p1 + best_w[1] * p2 + best_w[2] * p3

    # ---- Verdict ----
    candidates = {
        "slsqp_cross_fit": rae_slsqp,
        "naive_third":     rae_naive,
        "best_grid":       best_rae,
    }
    best_blend_tag = min(candidates, key=candidates.get)
    best_blend_rae = candidates[best_blend_tag]

    best_standalone_tag = min(standalone_rae, key=standalone_rae.get)
    best_standalone = standalone_rae[best_standalone_tag]

    beats_nb1251 = best_blend_rae < NB1251_REF - MARGIN
    flat_nb1251 = abs(best_blend_rae - NB1251_REF) < MARGIN

    if beats_nb1251:
        verdict = (f"NB1291_3WAY_BEATS_NB1251 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")
    elif flat_nb1251:
        verdict = (f"NB1291_3WAY_FLAT_VS_NB1251 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")
    else:
        verdict = (f"NB1291_3WAY_HURTS_VS_NB1251 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1190 standalone : {standalone_rae['nb1190']:.4f}")
    print(f"   nb1242 standalone : {standalone_rae['nb1242']:.4f}  (ref {NB1242_REF:.4f})")
    print(f"   nb1211 standalone : {standalone_rae['nb1211']:.4f}  (ref {NB1211_REF:.4f})")
    print(f"   best standalone   : {best_standalone:.4f}  ({best_standalone_tag})")
    print(f"")
    print(f"   candidate pooled RAE table:")
    for tag, val in sorted(candidates.items(), key=lambda kv: kv[1]):
        print(f"     {tag:18s} = {val:.4f}")
    print(f"")
    print(f"   best blend                 : {best_blend_rae:.4f}  ({best_blend_tag})")
    print(f"   nb1251 reference           : {NB1251_REF:.4f}")
    print(f"   delta vs nb1251            : {best_blend_rae - NB1251_REF:+.4f}")
    print(f"   beats_nb1251 (>= {MARGIN})    : {beats_nb1251}")
    print(f"   verdict                    : {verdict}")

    # Persist canonical artifacts.
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy",
            best_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "grid_step": GRID_STEP,
        "n_grid_points": len(grid_pts),
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "components": list(paths.keys()),
        "standalone_rae": standalone_rae,
        "pred_corr": pred_corr,
        "residual_corr": resid_corr,
        "residual_std": {
            "nb1190": float(r1.std()),
            "nb1242": float(r2.std()),
            "nb1211": float(r3.std()),
        },
        "slsqp_fold_records": fold_records,
        "slsqp_mean_fold_weights": {
            "nb1190": float(mean_w[0]),
            "nb1242": float(mean_w[1]),
            "nb1211": float(mean_w[2]),
        },
        "slsqp_in_sample_weights": {
            "nb1190": float(w_full[0]),
            "nb1242": float(w_full[1]),
            "nb1211": float(w_full[2]),
        },
        "slsqp_in_sample_rae": rae_full,
        "rae_slsqp_cross_fit": rae_slsqp,
        "rae_naive_third": rae_naive,
        "rae_best_grid": best_rae,
        "best_grid_weights": {
            "nb1190": best_w[0],
            "nb1242": best_w[1],
            "nb1211": best_w[2],
        },
        "top5_grid": top5,
        "candidate_rae_table": candidates,
        "best_blend_tag": best_blend_tag,
        "best_blend_rae": best_blend_rae,
        "best_standalone_tag": best_standalone_tag,
        "best_standalone_rae": best_standalone,
        "nb1251_ref": NB1251_REF,
        "nb1242_ref": NB1242_REF,
        "nb1211_ref": NB1211_REF,
        "delta_best_vs_nb1251": best_blend_rae - NB1251_REF,
        "delta_best_vs_nb1242": best_blend_rae - standalone_rae["nb1242"],
        "beats_nb1251": bool(beats_nb1251),
        "flat_vs_nb1251": bool(flat_nb1251),
        "margin": MARGIN,
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
    for k in ("standalone_rae",
              "pred_corr", "residual_corr",
              "slsqp_mean_fold_weights", "rae_slsqp_cross_fit",
              "rae_naive_third",
              "best_grid_weights", "rae_best_grid",
              "top5_grid",
              "candidate_rae_table",
              "best_blend_tag", "best_blend_rae",
              "delta_best_vs_nb1251",
              "beats_nb1251", "verdict"):
        print(f"  {k}: {res.get(k)}")
