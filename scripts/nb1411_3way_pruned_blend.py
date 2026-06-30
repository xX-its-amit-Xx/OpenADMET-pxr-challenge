"""nb1411 -- 3-way SHAP-pruned residual-learner blend (Avalon dropped).

Combine 3 SHAP-pruned residual learners from different fingerprint /
descriptor spaces.  Avalon (nb1392) is DROPPED because in the nb1401
4-way grid + SLSQP it landed at 0% weight.

    A nb1373  AtomPair-30   mean_bag RAE 0.5095   (current PRIMARY-1)
    B nb1352  MACCS-20      mean_bag RAE 0.5323
    C nb1364  Mordred-30    mean_bag RAE 0.5242

HYPOTHESIS:  nb1401 assigned Avalon 0% in both grid + SLSQP -- the real
diversification signal lives in AtomPair + MACCS + Mordred.  Directly
exploring the 3D simplex at FINER granularity (step 0.05) may reveal a
weight tuple that beats both:
  - nb1373 standalone (0.5095)         (margin 0.003)
  - nb1391 2-way best-w  (0.5076)      (the existing 2-way ceiling)

PROTOCOL:
    1. Load 3 *_mean_bag_oof.npy arrays (n_unb,).
    2. Load truth y_unb (_audit_unblind_y.npy) + sanity-check shapes.
    3. Pairwise pred Pearson + residual Pearson (vs truth).
    4. 3D simplex GRID search at step 0.05  (231 grid pts).
    5. Honest 5-fold cross-fit SLSQP convex blend.
    6. Naive 1/3 mean blend.
    7. Verdict at 0.003 RAE margin vs nb1373 (0.5095) AND vs nb1391 (0.5076).

Outputs:
    scripts/nb1411_3way_pruned_blend.py            (this file)
    data/processed/nb1411_summary.json
    data/processed/nb1411_best_oof.npy              (n_unb,) float32
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
from sklearn.model_selection import KFold
from scipy.optimize import minimize

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1411"

CANDIDATES = [
    ("nb1373", "AtomPair-30",  0.5095),  # PRIMARY-1
    ("nb1352", "MACCS-20",     0.5323),
    ("nb1364", "Mordred-30",   0.5242),
]

REF_RAE_NB1373 = 0.5095        # nb1373 mean_bag standalone
REF_RAE_NB1391 = 0.5076        # nb1391 2-way best-w (mean_mean grid)
DECISION_MARGIN = 0.003

GRID_STEP = 0.05
SLSQP_FOLDS = 5
SLSQP_SEED = 0


def _load_oofs() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    preds = []
    meta = []
    for tag, family, ref in CANDIDATES:
        p = DATA_PROCESSED / f"{tag}_mean_bag_oof.npy"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
        arr = np.load(p).astype(np.float64)
        if arr.shape[0] != n_unb:
            raise ValueError(f"{tag} shape {arr.shape} vs n_unb {n_unb}")
        r = float(rae(y_unb, arr))
        preds.append(arr)
        meta.append({"tag": tag, "family": family, "ref_rae": ref,
                     "reproduced_rae": r,
                     "delta_vs_ref": r - ref})
        print(f"   [load] {tag}  {family:<12s}  ref={ref:.4f}  "
              f"reproduced={r:.4f}  d={r-ref:+.4f}")
    P = np.stack(preds, axis=1)  # (n_unb, 3)
    return y_unb, P, meta


def _pearson_matrices(y: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = P - y.reshape(-1, 1)
    pc = np.corrcoef(P.T)
    rc = np.corrcoef(R.T)
    return pc, rc


def _simplex_grid(n_components: int, step: float):
    """Yield all weight tuples with sum=1, step granularity, w_i >= 0."""
    k = int(round(1.0 / step))
    for combo in product(range(k + 1), repeat=n_components):
        if sum(combo) == k:
            yield tuple(c / k for c in combo)


def _grid_search(y: np.ndarray, P: np.ndarray, step: float) -> list[dict]:
    rows = []
    for w in _simplex_grid(P.shape[1], step):
        pred = P @ np.array(w)
        r = float(rae(y, pred))
        rows.append({"w": list(w), "rae": r})
    rows.sort(key=lambda r: r["rae"])
    return rows


def _slsqp_blend(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    n_c = P.shape[1]
    w0 = np.full(n_c, 1.0 / n_c)

    def loss(w):
        pred = P @ w
        return float(rae(y, pred))

    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bnds = [(0.0, 1.0)] * n_c
    res = minimize(loss, w0, method="SLSQP", bounds=bnds,
                   constraints=cons, options={"maxiter": 400, "ftol": 1e-9})
    if not res.success:
        return w0
    return res.x


def _slsqp_crossfit(y: np.ndarray, P: np.ndarray,
                    n_folds: int = SLSQP_FOLDS,
                    seed: int = SLSQP_SEED) -> tuple[np.ndarray, list[np.ndarray]]:
    n = len(y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred_oof = np.full(n, np.nan, dtype=np.float64)
    fold_w = []
    for tr_loc, va_loc in kf.split(np.arange(n)):
        w_tr = _slsqp_blend(y[tr_loc], P[tr_loc])
        pred_oof[va_loc] = P[va_loc] @ w_tr
        fold_w.append(w_tr)
    return pred_oof, fold_w


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 3-way SHAP-pruned residual-learner blend (Avalon dropped)")
    print("           anchors: nb1373 (AtomPair) + nb1352 (MACCS) + "
          "nb1364 (Mordred)")
    print(f"           ref_nb1373 = {REF_RAE_NB1373:.4f}  "
          f"ref_nb1391 = {REF_RAE_NB1391:.4f}  margin = {DECISION_MARGIN}")
    print("=" * 78)

    y_unb, P, meta = _load_oofs()
    n_unb, n_c = P.shape
    print(f"[load] n_unb = {n_unb}   n_components = {n_c}")

    # ---- Pairwise correlations ----
    pred_corr, resid_corr = _pearson_matrices(y_unb, P)
    print("\n" + "-" * 78)
    print("PAIRWISE PEARSON (predictions)")
    print("-" * 78)
    tags = [c[0] for c in CANDIDATES]
    print("        " + "  ".join(f"{t:>8s}" for t in tags))
    for i, t in enumerate(tags):
        print(f"{t:>6s}  " + "  ".join(
            f"{pred_corr[i, j]:>8.4f}" for j in range(n_c)
        ))
    print("\n" + "-" * 78)
    print("PAIRWISE PEARSON (residuals y - pred)")
    print("-" * 78)
    print("        " + "  ".join(f"{t:>8s}" for t in tags))
    for i, t in enumerate(tags):
        print(f"{t:>6s}  " + "  ".join(
            f"{resid_corr[i, j]:>8.4f}" for j in range(n_c)
        ))

    # ---- 3D simplex grid (step 0.05) ----
    print("\n" + "-" * 78)
    print(f"3D SIMPLEX GRID  (step = {GRID_STEP})")
    print("-" * 78)
    grid_rows = _grid_search(y_unb, P, GRID_STEP)
    n_grid = len(grid_rows)
    print(f"   grid pts evaluated = {n_grid}")
    print(f"   top 5 tuples (w order = nb1373, nb1352, nb1364):")
    for r in grid_rows[:5]:
        w = r["w"]
        print(f"     w=({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})  "
              f"RAE={r['rae']:.4f}")

    best_grid = grid_rows[0]
    best_grid_w = np.array(best_grid["w"])
    best_grid_oof = P @ best_grid_w
    best_grid_rae = float(best_grid["rae"])

    # ---- Naive 1/3 mean ----
    naive_w = np.full(n_c, 1.0 / n_c)
    naive_oof = P @ naive_w
    naive_rae = float(rae(y_unb, naive_oof))
    print(f"\n   naive 1/3 mean RAE = {naive_rae:.4f}")

    # ---- In-sample SLSQP ----
    insample_w = _slsqp_blend(y_unb, P)
    insample_oof = P @ insample_w
    insample_rae = float(rae(y_unb, insample_oof))
    print(f"   in-sample SLSQP  w = {np.round(insample_w, 4).tolist()}  "
          f"RAE = {insample_rae:.4f}")

    # ---- 5-fold cross-fit SLSQP ----
    crossfit_oof, fold_w = _slsqp_crossfit(y_unb, P)
    crossfit_rae = float(rae(y_unb, crossfit_oof))
    fold_w_arr = np.stack(fold_w, axis=0)
    fold_w_mean = fold_w_arr.mean(axis=0)
    fold_w_std = fold_w_arr.std(axis=0)
    print(f"\n   5-fold cross-fit SLSQP RAE = {crossfit_rae:.4f}")
    print(f"     mean fold w = {np.round(fold_w_mean, 4).tolist()}")
    print(f"     std  fold w = {np.round(fold_w_std, 4).tolist()}")

    # ---- Pick BEST_OOF for export ----
    candidates_rae = {
        "grid_best": (best_grid_rae, best_grid_oof, best_grid_w.tolist()),
        "naive_third": (naive_rae, naive_oof, naive_w.tolist()),
        "slsqp_insample": (insample_rae, insample_oof,
                           insample_w.tolist()),
        "slsqp_crossfit": (crossfit_rae, crossfit_oof,
                           fold_w_mean.tolist()),
    }
    # Honest pick = cross-fit (most LB-faithful)
    pick_label = "slsqp_crossfit"
    pick_rae, pick_oof, pick_w = candidates_rae[pick_label]

    beats_nb1373 = pick_rae < REF_RAE_NB1373 - DECISION_MARGIN
    flat_vs_nb1373 = abs(pick_rae - REF_RAE_NB1373) < DECISION_MARGIN
    beats_nb1391 = pick_rae < REF_RAE_NB1391 - DECISION_MARGIN
    flat_vs_nb1391 = abs(pick_rae - REF_RAE_NB1391) < DECISION_MARGIN

    # also report grid_best beats flags
    grid_beats_nb1373 = best_grid_rae < REF_RAE_NB1373 - DECISION_MARGIN
    grid_beats_nb1391 = best_grid_rae < REF_RAE_NB1391 - DECISION_MARGIN

    if beats_nb1391:
        verdict = "TRI_BLEND_BEATS_NB1391"
    elif beats_nb1373:
        verdict = "TRI_BLEND_BEATS_NB1373_FLAT_NB1391"
    elif flat_vs_nb1373:
        verdict = "TRI_BLEND_FLAT_VS_NB1373"
    else:
        verdict = "TRI_BLEND_WORSE_THAN_NB1373"

    print("\n" + "=" * 78)
    print(f"FINAL  pick={pick_label}  RAE={pick_rae:.4f}")
    print(f"       d_vs_nb1373={pick_rae - REF_RAE_NB1373:+.4f}  "
          f"d_vs_nb1391={pick_rae - REF_RAE_NB1391:+.4f}")
    print(f"       beats_nb1373={beats_nb1373}  "
          f"flat_vs_nb1373={flat_vs_nb1373}")
    print(f"       beats_nb1391={beats_nb1391}  "
          f"flat_vs_nb1391={flat_vs_nb1391}")
    print(f"       (grid_best={best_grid_rae:.4f}  "
          f"beats_nb1373={grid_beats_nb1373}  "
          f"beats_nb1391={grid_beats_nb1391})")
    print(f"       verdict={verdict}")
    print("=" * 78)

    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy",
            pick_oof.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}  "
          f"(pick={pick_label})")

    summary = {
        "tag": TAG,
        "candidates": [
            {"tag": tag, "family": family, "ref_rae": ref}
            for (tag, family, ref) in CANDIDATES
        ],
        "candidate_meta": meta,
        "n_unb": int(n_unb),
        "n_components": int(n_c),
        "ref_rae_nb1373": REF_RAE_NB1373,
        "ref_rae_nb1391": REF_RAE_NB1391,
        "decision_margin": DECISION_MARGIN,
        "pred_pearson": pred_corr.tolist(),
        "resid_pearson": resid_corr.tolist(),
        "grid_step": GRID_STEP,
        "n_grid_pts": int(n_grid),
        "grid_top5": grid_rows[:5],
        "grid_best_rae": best_grid_rae,
        "grid_best_w": best_grid_w.tolist(),
        "grid_beats_nb1373": bool(grid_beats_nb1373),
        "grid_beats_nb1391": bool(grid_beats_nb1391),
        "naive_third_w": naive_w.tolist(),
        "naive_third_rae": naive_rae,
        "slsqp_insample_w": insample_w.tolist(),
        "slsqp_insample_rae": insample_rae,
        "slsqp_crossfit_folds": SLSQP_FOLDS,
        "slsqp_crossfit_seed": SLSQP_SEED,
        "slsqp_crossfit_fold_w": fold_w_arr.tolist(),
        "slsqp_crossfit_fold_w_mean": fold_w_mean.tolist(),
        "slsqp_crossfit_fold_w_std": fold_w_std.tolist(),
        "slsqp_crossfit_rae": crossfit_rae,
        "pick_label": pick_label,
        "pick_rae": pick_rae,
        "pick_w": pick_w,
        "delta_pick_vs_nb1373": pick_rae - REF_RAE_NB1373,
        "delta_pick_vs_nb1391": pick_rae - REF_RAE_NB1391,
        "beats_nb1373": bool(beats_nb1373),
        "flat_vs_nb1373": bool(flat_vs_nb1373),
        "beats_nb1391": bool(beats_nb1391),
        "flat_vs_nb1391": bool(flat_vs_nb1391),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_unb", "n_components",
        "ref_rae_nb1373", "ref_rae_nb1391",
        "n_grid_pts",
        "grid_best_rae", "grid_best_w",
        "naive_third_rae",
        "slsqp_insample_rae", "slsqp_insample_w",
        "slsqp_crossfit_rae",
        "slsqp_crossfit_fold_w_mean",
        "pick_label", "pick_rae",
        "delta_pick_vs_nb1373", "delta_pick_vs_nb1391",
        "beats_nb1373", "flat_vs_nb1373",
        "beats_nb1391", "flat_vs_nb1391",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
