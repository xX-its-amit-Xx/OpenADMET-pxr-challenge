"""nb1350 -- Blend CatBoost median (nb1341) + nb1242 + nb1190.

Question:
    Does CatBoost-residual add anything to the nb1290 manifold (nb1190+nb1242)?
    nb1341 ties nb1242 standalone (mean_bag 0.5420, median_bag 0.5395) and is
    99.0% correlated with nb1242. If CatBoost-median draws on a different
    axis than nb1242 mean_bag, a 3-way blend could break the 0.5390 ceiling
    set by nb1290 best_fixed_w (w_nb1190=0.35, w_nb1242=0.65).

Inputs (on 253 unblind):
    - nb1341 CatBoost-residual MEDIAN bag (recomputed from per_seed array)
    - nb1242 ChEMBL-feature residual mean_bag       (0.5431)
    - nb1190 BoB-of-Bags mean                       (0.5499)

Protocol:
    1. Load per-seed CatBoost OOF -> median across seeds (target == nb1341
       reported rae_median_bag 0.5395). Compare with mean_bag (0.5420) and
       confirm shape.
    2. Pairwise pred Pearson + residual Pearson among all 3.
    3. 3-simplex grid step 0.05 -> RAE per tuple; report top-5.
    4. 5-fold cross-fit SLSQP 3-way (mirror nb1290 SLSQP_SEED=42).
    5. Naive 1/3 mean as baseline.
    6. Compare best 3-way to nb1290 best_fixed_w (0.5390).

Outputs:
    data/processed/nb1350_summary.json
    data/processed/nb1350_best_oof.npy   (253,) float32 -- whichever blend
                                                          wins by pooled RAE
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

TAG = "nb1350"
SLSQP_FOLDS = 5
SLSQP_SEED = 42
GRID_STEP = 0.05

NB1190_REF = 0.5499
NB1242_REF = 0.5431
NB1341_MEAN_REF = 0.5420
NB1341_MEDIAN_REF = 0.5395
NB1290_BESTW_REF = 0.5390
MARGIN = 0.003


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
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


def _enumerate_simplex(step: float, K: int) -> list[tuple[float, ...]]:
    """All (w_0, ..., w_{K-1}) on simplex with each w_i in {0, step, 2*step,...,1}."""
    n_steps = int(round(1.0 / step))
    out: list[tuple[float, ...]] = []
    # Iterate over (i, j) for K=3 only
    if K != 3:
        raise NotImplementedError("only K=3 grid implemented")
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            out.append((i * step, j * step, k * step))
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 3-way blend  CatBoost-median (nb1341) + nb1242 + nb1190")
    print(f"          target: beat nb1290 best_fixed_w ({NB1290_BESTW_REF:.4f})")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    # ---- Load CatBoost per-seed and compute median bag ----
    cat_per_seed_path = DATA_PROCESSED / "nb1341_per_seed_corrected_oof.npy"
    cat_mean_path = DATA_PROCESSED / "nb1341_mean_bag_oof.npy"
    if not cat_per_seed_path.exists():
        raise FileNotFoundError(cat_per_seed_path)

    per_seed = np.load(cat_per_seed_path).astype(np.float64)   # (S, N)
    if per_seed.ndim != 2 or per_seed.shape[1] != n_unb:
        raise ValueError(f"unexpected per-seed shape: {per_seed.shape}")
    cat_median = np.median(per_seed, axis=0)
    cat_mean = np.load(cat_mean_path).astype(np.float64)

    rae_cat_median = float(rae(y_unb, cat_median))
    rae_cat_mean = float(rae(y_unb, cat_mean))
    print(f"\n[load] CatBoost per-seed OOF shape = {per_seed.shape}")
    print(f"       median-bag pooled RAE  = {rae_cat_median:.4f}  "
          f"(ref {NB1341_MEDIAN_REF:.4f})")
    print(f"       mean-bag pooled RAE    = {rae_cat_mean:.4f}  "
          f"(ref {NB1341_MEAN_REF:.4f})")

    # Pick median (per spec)
    p_cat = cat_median

    # ---- Load nb1242 and nb1190 ----
    p_1242 = np.load(DATA_PROCESSED / "nb1242_mean_bag_oof.npy").astype(np.float64)
    p_1190 = np.load(DATA_PROCESSED / "nb1190_bob_mean_oof.npy").astype(np.float64)

    standalone_rae = {
        "nb1341_median": rae_cat_median,
        "nb1242": float(rae(y_unb, p_1242)),
        "nb1190": float(rae(y_unb, p_1190)),
    }
    print("\n[load] standalone pooled RAE on 253 unblind:")
    print(f"   nb1341_median (CatBoost-residual median bag): "
          f"{standalone_rae['nb1341_median']:.4f}")
    print(f"   nb1242 (ChEMBL-feat residual mean bag)      : "
          f"{standalone_rae['nb1242']:.4f}  (ref {NB1242_REF:.4f})")
    print(f"   nb1190 (BoB-of-Bags mean)                   : "
          f"{standalone_rae['nb1190']:.4f}  (ref {NB1190_REF:.4f})")

    # ---- Pairwise Pearson (pred and residual) ----
    print("\n" + "-" * 78)
    print("  BLOCK: pairwise Pearson (pred + residual)")
    print("-" * 78)
    preds = {"nb1341": p_cat, "nb1242": p_1242, "nb1190": p_1190}
    resids = {k: v - y_unb for k, v in preds.items()}
    names = list(preds.keys())

    pred_corr = {}
    resid_corr = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pc = float(np.corrcoef(preds[a], preds[b])[0, 1])
            rc = float(np.corrcoef(resids[a], resids[b])[0, 1])
            pred_corr[f"{a}__{b}"] = pc
            resid_corr[f"{a}__{b}"] = rc
            print(f"   {a:10s} vs {b:10s}  pred r={pc:+.4f}   resid r={rc:+.4f}")
    resid_std = {k: float(v.std()) for k, v in resids.items()}
    for k, v in resid_std.items():
        print(f"   resid_std[{k}] = {v:.4f}")

    # ---- Grid search 3-simplex (step 0.05) ----
    print("\n" + "-" * 78)
    print(f"  BLOCK: 3-simplex grid step {GRID_STEP}")
    print("-" * 78)
    P = np.column_stack([p_cat, p_1242, p_1190])  # cols: nb1341, nb1242, nb1190
    tuples = _enumerate_simplex(GRID_STEP, 3)
    grid_records = []
    for (w1, w2, w3) in tuples:
        blend = w1 * p_cat + w2 * p_1242 + w3 * p_1190
        r = float(rae(y_unb, blend))
        grid_records.append({
            "w_nb1341": float(round(w1, 4)),
            "w_nb1242": float(round(w2, 4)),
            "w_nb1190": float(round(w3, 4)),
            "rae": r,
        })
    grid_records.sort(key=lambda d: d["rae"])
    top5 = grid_records[:5]
    print(f"   evaluated {len(grid_records)} tuples")
    print("   top-5 tuples:")
    for rec in top5:
        print(f"     w_cat={rec['w_nb1341']:.2f}  w_1242={rec['w_nb1242']:.2f}  "
              f"w_1190={rec['w_nb1190']:.2f}  RAE={rec['rae']:.4f}")

    best_grid = top5[0]
    best_grid_oof = (best_grid["w_nb1341"] * p_cat
                     + best_grid["w_nb1242"] * p_1242
                     + best_grid["w_nb1190"] * p_1190)
    rae_best_grid = best_grid["rae"]

    # Naive 1/3
    naive_oof = (p_cat + p_1242 + p_1190) / 3.0
    rae_naive = float(rae(y_unb, naive_oof))
    print(f"\n[block] naive 1/3 mean  RAE = {rae_naive:.4f}")

    # ---- 3-way SLSQP cross-fit ----
    print("\n" + "-" * 78)
    print(f"  BLOCK: 3-way SLSQP cross-fit ({SLSQP_FOLDS}-fold, seed {SLSQP_SEED})")
    print("-" * 78)
    slsqp_oof, fold_records = _slsqp_cross_fit(P, y_unb, SLSQP_FOLDS, SLSQP_SEED)
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    fold_w = np.array([r["weights"] for r in fold_records])
    mean_w = fold_w.mean(axis=0)
    print("   per-fold weights (nb1341, nb1242, nb1190):")
    for rec in fold_records:
        w = rec["weights"]
        print(f"     fold {rec['fold']}:  w_cat={w[0]:.4f}  w_1242={w[1]:.4f}  "
              f"w_1190={w[2]:.4f}")
    print(f"   mean fold weights:  w_cat={mean_w[0]:.4f}  "
          f"w_1242={mean_w[1]:.4f}  w_1190={mean_w[2]:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}")

    w_full = _slsqp_blend_weights(P, y_unb)
    p_full = P @ w_full
    rae_full = float(rae(y_unb, p_full))
    print(f"   in-sample SLSQP weights: w_cat={w_full[0]:.4f}  "
          f"w_1242={w_full[1]:.4f}  w_1190={w_full[2]:.4f}   "
          f"RAE = {rae_full:.4f}")

    # ---- Verdict ----
    candidates = {
        "grid_best":       rae_best_grid,
        "slsqp_cross_fit": rae_slsqp,
        "naive_mean":      rae_naive,
    }
    best_blend_tag = min(candidates, key=candidates.get)
    best_blend_rae = candidates[best_blend_tag]
    if best_blend_tag == "grid_best":
        best_oof = best_grid_oof
    elif best_blend_tag == "slsqp_cross_fit":
        best_oof = slsqp_oof
    else:
        best_oof = naive_oof

    beats_nb1290 = best_blend_rae < NB1290_BESTW_REF - MARGIN
    flat_nb1290 = abs(best_blend_rae - NB1290_BESTW_REF) < MARGIN

    if beats_nb1290:
        verdict = (f"CATBOOST_BLEND_BEATS_NB1290 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")
    elif flat_nb1290:
        verdict = (f"CATBOOST_BLEND_FLAT_VS_NB1290 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")
    else:
        verdict = (f"CATBOOST_BLEND_HURTS_VS_NB1290 "
                   f"({best_blend_tag} @ {best_blend_rae:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1290 best_fixed_w ref     : {NB1290_BESTW_REF:.4f}")
    print(f"   3-way grid best            : {rae_best_grid:.4f}  "
          f"(w_cat={best_grid['w_nb1341']:.2f}, "
          f"w_1242={best_grid['w_nb1242']:.2f}, "
          f"w_1190={best_grid['w_nb1190']:.2f})")
    print(f"   3-way SLSQP cross-fit      : {rae_slsqp:.4f}")
    print(f"   3-way naive 1/3 mean       : {rae_naive:.4f}")
    print(f"   best 3-way blend           : {best_blend_rae:.4f}  ({best_blend_tag})")
    print(f"   delta vs nb1290 (0.5390)   : {best_blend_rae - NB1290_BESTW_REF:+.4f}")
    print(f"   beats_nb1290 (>= {MARGIN})  : {beats_nb1290}")
    print(f"   verdict                    : {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy", best_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "grid_step": GRID_STEP,
        "components": ["nb1341_median", "nb1242", "nb1190"],
        "standalone_rae": standalone_rae,
        "rae_cat_mean_check": rae_cat_mean,
        "rae_cat_median_check": rae_cat_median,
        "pred_corr": pred_corr,
        "residual_corr": resid_corr,
        "resid_std": resid_std,
        "grid_top5": top5,
        "best_grid_weights": {
            "w_nb1341": best_grid["w_nb1341"],
            "w_nb1242": best_grid["w_nb1242"],
            "w_nb1190": best_grid["w_nb1190"],
        },
        "rae_best_grid": rae_best_grid,
        "rae_naive_mean": rae_naive,
        "slsqp_fold_records": fold_records,
        "slsqp_mean_fold_weights": {
            "w_nb1341": float(mean_w[0]),
            "w_nb1242": float(mean_w[1]),
            "w_nb1190": float(mean_w[2]),
        },
        "slsqp_in_sample_weights": {
            "w_nb1341": float(w_full[0]),
            "w_nb1242": float(w_full[1]),
            "w_nb1190": float(w_full[2]),
        },
        "slsqp_in_sample_rae": rae_full,
        "rae_slsqp_cross_fit": rae_slsqp,
        "candidate_rae_table": candidates,
        "best_blend_tag": best_blend_tag,
        "best_blend_rae": best_blend_rae,
        "nb1290_bestw_ref": NB1290_BESTW_REF,
        "delta_best_vs_nb1290": best_blend_rae - NB1290_BESTW_REF,
        "beats_nb1290": bool(beats_nb1290),
        "flat_vs_nb1290": bool(flat_nb1290),
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
              "grid_top5",
              "best_grid_weights", "rae_best_grid",
              "slsqp_mean_fold_weights", "rae_slsqp_cross_fit",
              "rae_naive_mean",
              "candidate_rae_table",
              "best_blend_tag", "best_blend_rae",
              "delta_best_vs_nb1290",
              "beats_nb1290", "verdict"):
        print(f"  {k}: {res.get(k)}")
