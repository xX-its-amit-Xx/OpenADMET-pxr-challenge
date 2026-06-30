"""nb1421 -- Upgraded 3-way blend: AtomPair + MACCS + Mordred-K20.

Drop-in replacement for nb1411's 3-way blend, swapping out nb1364
(Mordred-K30, RAE 0.5242) for nb1412 (Mordred-K20, RAE 0.5180).

    A nb1373  AtomPair-30   mean_bag      RAE 0.5095   (current PRIMARY-1)
    B nb1352  MACCS-20      mean_bag      RAE 0.5323
    C nb1412  Mordred-K20   best_K (=20)  RAE 0.5180   (UPGRADE vs nb1364 0.5242)

HYPOTHESIS:  nb1411 hit cross-fit RAE 0.5045 with nb1364 (Mordred-30).
nb1412 (K=20) is 0.006 stronger than nb1364 (K=30) as a standalone
residual learner.  Upgrading the Mordred slot should push the 3-way
toward ~0.501 cross-fit.

PROTOCOL  (mirrors nb1411 exactly):
    1. Load 3 OOFs.  Sanity-check shapes vs y_unb.
    2. Pairwise pred Pearson + residual Pearson.
    3. 3D simplex GRID search at step 0.05  (231 grid pts).
    4. Naive 1/3 mean blend.
    5. In-sample SLSQP convex blend.
    6. Honest 5-fold cross-fit SLSQP convex blend.
    7. Verdict at 0.003 RAE margin vs nb1411 cross-fit (0.5045)
       AND nb1411 naive-third (0.5037).

Outputs:
    scripts/nb1421_upgraded_3way.py            (this file)
    data/processed/nb1421_summary.json
    data/processed/nb1421_best_oof.npy          (n_unb,) float32
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

TAG = "nb1421"

# (tag, family, oof_filename, ref_rae)
CANDIDATES = [
    ("nb1373", "AtomPair-30",  "nb1373_mean_bag_oof.npy", 0.5095),  # PRIMARY-1
    ("nb1352", "MACCS-20",     "nb1352_mean_bag_oof.npy", 0.5323),
    ("nb1412", "Mordred-K20",  "nb1412_best_K_oof.npy",   0.5180),  # UPGRADE
]

REF_RAE_NB1411_CROSSFIT = 0.5045   # nb1411 5-fold SLSQP cross-fit
REF_RAE_NB1411_NAIVE    = 0.5037   # nb1411 naive 1/3 mean
DECISION_MARGIN = 0.003

GRID_STEP = 0.05
SLSQP_FOLDS = 5
SLSQP_SEED = 0


def _load_oofs() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    preds = []
    meta = []
    for tag, family, fname, ref in CANDIDATES:
        p = DATA_PROCESSED / fname
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
        arr = np.load(p).astype(np.float64)
        if arr.shape[0] != n_unb:
            raise ValueError(f"{tag} shape {arr.shape} vs n_unb {n_unb}")
        r = float(rae(y_unb, arr))
        preds.append(arr)
        meta.append({"tag": tag, "family": family,
                     "oof_file": fname,
                     "ref_rae": ref,
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
    print(f"{TAG} -- Upgraded 3-way blend (AtomPair + MACCS + Mordred-K20)")
    print("           anchors: nb1373 + nb1352 + nb1412 (UPGRADE from nb1364)")
    print(f"           ref_nb1411_crossfit = {REF_RAE_NB1411_CROSSFIT:.4f}  "
          f"ref_nb1411_naive = {REF_RAE_NB1411_NAIVE:.4f}  "
          f"margin = {DECISION_MARGIN}")
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
    print(f"   top 5 tuples (w order = nb1373, nb1352, nb1412):")
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

    beats_nb1411_crossfit = pick_rae < REF_RAE_NB1411_CROSSFIT - DECISION_MARGIN
    flat_vs_nb1411_crossfit = abs(pick_rae - REF_RAE_NB1411_CROSSFIT) < DECISION_MARGIN
    beats_nb1411_naive = pick_rae < REF_RAE_NB1411_NAIVE - DECISION_MARGIN
    flat_vs_nb1411_naive = abs(pick_rae - REF_RAE_NB1411_NAIVE) < DECISION_MARGIN

    # also report grid_best beats flags
    grid_beats_nb1411_crossfit = best_grid_rae < REF_RAE_NB1411_CROSSFIT - DECISION_MARGIN
    grid_beats_nb1411_naive = best_grid_rae < REF_RAE_NB1411_NAIVE - DECISION_MARGIN

    # Also a single overall beats_nb1411 flag (vs whichever nb1411 number
    # is the relevant ceiling -- use the cross-fit since this pick is
    # cross-fit and that is the LB-faithful comparison).
    beats_nb1411 = beats_nb1411_crossfit

    if beats_nb1411_naive:
        verdict = "UPGRADED_TRI_BEATS_NB1411_NAIVE"
    elif beats_nb1411_crossfit:
        verdict = "UPGRADED_TRI_BEATS_NB1411_CROSSFIT_FLAT_NAIVE"
    elif flat_vs_nb1411_crossfit:
        verdict = "UPGRADED_TRI_FLAT_VS_NB1411"
    else:
        verdict = "UPGRADED_TRI_WORSE_THAN_NB1411"

    print("\n" + "=" * 78)
    print(f"FINAL  pick={pick_label}  RAE={pick_rae:.4f}")
    print(f"       d_vs_nb1411_crossfit={pick_rae - REF_RAE_NB1411_CROSSFIT:+.4f}  "
          f"d_vs_nb1411_naive={pick_rae - REF_RAE_NB1411_NAIVE:+.4f}")
    print(f"       beats_nb1411_crossfit={beats_nb1411_crossfit}  "
          f"flat_vs_nb1411_crossfit={flat_vs_nb1411_crossfit}")
    print(f"       beats_nb1411_naive={beats_nb1411_naive}  "
          f"flat_vs_nb1411_naive={flat_vs_nb1411_naive}")
    print(f"       (grid_best={best_grid_rae:.4f}  "
          f"beats_nb1411_crossfit={grid_beats_nb1411_crossfit}  "
          f"beats_nb1411_naive={grid_beats_nb1411_naive})")
    print(f"       verdict={verdict}")
    print("=" * 78)

    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy",
            pick_oof.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}  "
          f"(pick={pick_label})")

    summary = {
        "tag": TAG,
        "candidates": [
            {"tag": tag, "family": family, "oof_file": fname, "ref_rae": ref}
            for (tag, family, fname, ref) in CANDIDATES
        ],
        "candidate_meta": meta,
        "n_unb": int(n_unb),
        "n_components": int(n_c),
        "ref_rae_nb1411_crossfit": REF_RAE_NB1411_CROSSFIT,
        "ref_rae_nb1411_naive": REF_RAE_NB1411_NAIVE,
        "decision_margin": DECISION_MARGIN,
        "pred_pearson": pred_corr.tolist(),
        "resid_pearson": resid_corr.tolist(),
        "grid_step": GRID_STEP,
        "n_grid_pts": int(n_grid),
        "grid_top5": grid_rows[:5],
        "grid_best_rae": best_grid_rae,
        "grid_best_w": best_grid_w.tolist(),
        "grid_beats_nb1411_crossfit": bool(grid_beats_nb1411_crossfit),
        "grid_beats_nb1411_naive": bool(grid_beats_nb1411_naive),
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
        "delta_pick_vs_nb1411_crossfit": pick_rae - REF_RAE_NB1411_CROSSFIT,
        "delta_pick_vs_nb1411_naive": pick_rae - REF_RAE_NB1411_NAIVE,
        "beats_nb1411": bool(beats_nb1411),
        "beats_nb1411_crossfit": bool(beats_nb1411_crossfit),
        "flat_vs_nb1411_crossfit": bool(flat_vs_nb1411_crossfit),
        "beats_nb1411_naive": bool(beats_nb1411_naive),
        "flat_vs_nb1411_naive": bool(flat_vs_nb1411_naive),
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
        "ref_rae_nb1411_crossfit", "ref_rae_nb1411_naive",
        "n_grid_pts",
        "grid_best_rae", "grid_best_w",
        "naive_third_rae",
        "slsqp_insample_rae", "slsqp_insample_w",
        "slsqp_crossfit_rae",
        "slsqp_crossfit_fold_w_mean",
        "pick_label", "pick_rae",
        "delta_pick_vs_nb1411_crossfit", "delta_pick_vs_nb1411_naive",
        "beats_nb1411",
        "beats_nb1411_crossfit", "flat_vs_nb1411_crossfit",
        "beats_nb1411_naive", "flat_vs_nb1411_naive",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
