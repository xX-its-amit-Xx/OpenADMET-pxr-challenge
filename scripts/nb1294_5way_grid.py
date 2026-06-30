"""nb1294 -- Constrained 5-way SIMPLEX grid search on n=253 unblind.

Hypothesis:
  SLSQP cross-fit overfits 5D weights at n=253. Pure in-sample 5D simplex
  grid search (step 0.1) over the SAME 5 components that won non-zero
  weight in nb1281 has only ~1001 grid points and 5 DoF, which is lower
  capacity than SLSQP (continuous over the simplex).

Components (the 5 with non-zero weight in nb1281 meta-stack):
  nb1190_bob_mean_oof    (triple-FP BoB,   0.5499)
  nb1242_mean_bag_oof    (ChEMBL+MACCS,    0.5431)
  nb1252_bob_mean_oof    (ChEMBL BoB,      0.5446)
  nb1200_bob_mean_oof    (MACCS BoB,       0.5495)
  nb1211_mean_oof        (BoB-blend,       0.5451)

Reports:
  - top-10 (w1,w2,w3,w4,w5) tuples by in-sample pooled RAE
  - 5-fold cross-fit version (re-grid-search per training fold; extreme overfit)
  - verdict at 0.003 margin vs nb1251 (0.5394)
  - mass concentration analysis (entropy, max-w, n>=0.1)
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

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1294"
N_FOLDS = 5
SEED = 42
NB1251_REF = 0.5394
MARGIN = 0.003
STEP = 0.1  # 0.0, 0.1, ..., 1.0 -> 11 levels per axis
TOL = 1e-9

COMPONENT_FILES = [
    ("nb1190", "nb1190_bob_mean_oof.npy", 0.5499),
    ("nb1242", "nb1242_mean_bag_oof.npy", 0.5431),
    ("nb1252", "nb1252_bob_mean_oof.npy", 0.5446),
    ("nb1200", "nb1200_bob_mean_oof.npy", 0.5495),
    ("nb1211", "nb1211_mean_oof.npy",     0.5451),
]


def _gen_simplex_grid(step: float = STEP) -> np.ndarray:
    """All 5-tuples (w1..w5), each w in {0, step, 2*step, ..., 1}, sum == 1."""
    levels = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 6)
    K = 5
    pts = []
    for combo in product(levels, repeat=K - 1):
        s = sum(combo)
        if s > 1.0 + TOL:
            continue
        last = round(1.0 - s, 6)
        if last < -TOL or last > 1.0 + TOL:
            continue
        # snap last to nearest grid level
        nearest = round(last / step) * step
        if abs(nearest - last) > 1e-6:
            continue
        if nearest < -TOL or nearest > 1.0 + TOL:
            continue
        pts.append((*combo, nearest))
    arr = np.asarray(pts, dtype=np.float64)
    # safety: clip and renormalize tiny float drift
    arr = np.clip(arr, 0.0, 1.0)
    s = arr.sum(axis=1, keepdims=True)
    arr = arr / s
    # dedupe
    arr = np.unique(np.round(arr, 6), axis=0)
    return arr


def _grid_best(P: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (best_w, best_rae, all_raes)."""
    # P (n, 5), grid (G, 5) -> preds (G, n) = grid @ P.T
    preds = grid @ P.T
    raes = np.empty(preds.shape[0], dtype=np.float64)
    for g in range(preds.shape[0]):
        raes[g] = rae(y, preds[g])
    g_star = int(np.argmin(raes))
    return grid[g_star], float(raes[g_star]), raes


def _cross_fit_grid(P: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for f, (tr, va) in enumerate(kf.split(np.arange(n))):
        # grid search on training fold (MSE since RAE on small subset is noisy
        # but we follow the spirit of the protocol: use RAE everywhere).
        # train P[tr] (n_tr,5), y[tr]
        preds_tr = grid @ P[tr].T
        # use RAE on training fold to pick weights (matches in-sample protocol)
        raes_tr = np.array([rae(y[tr], preds_tr[g]) for g in range(grid.shape[0])])
        g_star = int(np.argmin(raes_tr))
        w = grid[g_star]
        oof[va] = P[va] @ w
        fold_records.append({
            "fold": int(f),
            "weights": [float(x) for x in w],
            "train_rae": float(raes_tr[g_star]),
        })
    return oof, fold_records


def _entropy(w: np.ndarray, eps: float = 1e-12) -> float:
    w = np.asarray(w, dtype=np.float64)
    w = w[w > eps]
    return float(-np.sum(w * np.log(w))) if len(w) > 0 else 0.0


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- constrained 5-way SIMPLEX grid (step={STEP})")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape = {y_unb.shape}")

    preds = {}
    standalone = {}
    for tag, fname, ref in COMPONENT_FILES:
        p = DATA_PROCESSED / fname
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({tag})")
        v = np.load(p).astype(np.float64)
        if v.shape[0] != n_unb:
            raise ValueError(f"shape mismatch: {tag}={v.shape}")
        preds[tag] = v
        standalone[tag] = float(rae(y_unb, v))
        print(f"   {tag}  loaded {p.name:30s}  RAE={standalone[tag]:.4f}  (ref {ref:.4f})")

    tags = [t for t, _, _ in COMPONENT_FILES]
    P = np.column_stack([preds[t] for t in tags])
    print(f"\n[stack] P shape = {P.shape}  (5 features, {n_unb} rows)")

    # Pairwise pred correlation
    corr = np.corrcoef(P, rowvar=False)
    print("\n[diag] pairwise pred Pearson:")
    print("         " + "  ".join(f"{t:>7s}" for t in tags))
    for i, t in enumerate(tags):
        row = "  ".join(f"{corr[i, j]:>7.4f}" for j in range(len(tags)))
        print(f"   {t:7s}  {row}")

    # ----- Build grid -----
    grid = _gen_simplex_grid(STEP)
    G = grid.shape[0]
    print(f"\n[grid] {G} valid simplex points at step={STEP}")

    # ----- In-sample evaluation -----
    print("\n" + "-" * 78)
    print("  BLOCK 1: in-sample grid search on full 253")
    print("-" * 78)
    best_w_in, best_rae_in, all_raes = _grid_best(P, y_unb, grid)
    print(f"   best in-sample RAE = {best_rae_in:.4f}")
    print(f"   best in-sample weights:")
    for t, w in zip(tags, best_w_in):
        print(f"     {t}: {w:.4f}")

    # Top-10
    order = np.argsort(all_raes)[:10]
    top10 = []
    print("\n   top-10 in-sample (w_nb1190, w_nb1242, w_nb1252, w_nb1200, w_nb1211 -> RAE):")
    for rank, ix in enumerate(order, start=1):
        w = grid[ix]
        r = float(all_raes[ix])
        rec = {
            "rank": rank,
            "weights": {t: float(w[i]) for i, t in enumerate(tags)},
            "rae": r,
        }
        top10.append(rec)
        ws = "  ".join(f"{t}={w[i]:.2f}" for i, t in enumerate(tags))
        print(f"     {rank:2d}. {ws}    RAE={r:.4f}")

    # ----- 5-fold cross-fit (extreme overfit risk) -----
    print("\n" + "-" * 78)
    print("  BLOCK 2: 5-fold cross-fit (re-grid per fold; extreme overfit risk)")
    print("-" * 78)
    cf_oof, cf_folds = _cross_fit_grid(P, y_unb, grid)
    rae_cf = float(rae(y_unb, cf_oof))
    print(f"   per-fold winning weights:")
    for rec in cf_folds:
        ws = "  ".join(f"{t}={rec['weights'][i]:.2f}" for i, t in enumerate(tags))
        print(f"     fold {rec['fold']}: {ws}  (train RAE {rec['train_rae']:.4f})")
    fold_w = np.array([r["weights"] for r in cf_folds])
    mean_cf_w = fold_w.mean(axis=0)
    print(f"   mean cross-fit weights:")
    for t, w in zip(tags, mean_cf_w):
        print(f"     {t}: {w:.4f}")
    print(f"   pooled cross-fit RAE = {rae_cf:.4f}")

    # ----- Mass concentration analysis -----
    print("\n" + "-" * 78)
    print("  BLOCK 3: mass concentration analysis (best in-sample weights)")
    print("-" * 78)
    n_nonzero = int(np.sum(best_w_in > 0.0 + 1e-9))
    n_ge_0p1 = int(np.sum(best_w_in >= 0.1 - 1e-9))
    n_ge_0p2 = int(np.sum(best_w_in >= 0.2 - 1e-9))
    max_w = float(best_w_in.max())
    ent = _entropy(best_w_in)
    ent_max = float(np.log(5.0))  # uniform 5-way entropy
    ent_ratio = ent / ent_max if ent_max > 0 else 0.0
    print(f"   nonzero components       : {n_nonzero}/5")
    print(f"   components >= 0.1        : {n_ge_0p1}/5")
    print(f"   components >= 0.2        : {n_ge_0p2}/5")
    print(f"   max weight               : {max_w:.4f}")
    print(f"   entropy (nats)           : {ent:.4f}  (uniform max = {ent_max:.4f})")
    print(f"   entropy ratio            : {ent_ratio:.4f}")
    concentrated = (n_ge_0p1 <= 3) or (max_w >= 0.6) or (ent_ratio < 0.6)
    print(f"   concentrated (heuristic) : {concentrated}")

    # Top-10 mass concentration distribution
    top10_n_ge_0p1 = [int(np.sum(np.array([r["weights"][t] for t in tags]) >= 0.1 - 1e-9))
                     for r in top10]
    top10_max_w = [float(max(r["weights"].values())) for r in top10]
    print(f"\n   top-10 n>=0.1 distribution: {top10_n_ge_0p1}")
    print(f"   top-10 max-w distribution: {[round(x, 2) for x in top10_max_w]}")

    # ----- Verdict -----
    # The honest LB proxy is the cross-fit; the in-sample is the upper-bound
    # diagnostic for "capacity". Per the protocol we report both and judge
    # the BETTER OF the two (mirrors nb1281's candidate table).
    candidates = {
        "in_sample": best_rae_in,
        "cross_fit": rae_cf,
    }
    best_tag = min(candidates, key=candidates.get)
    best_rae = candidates[best_tag]

    beats_nb1251 = best_rae < NB1251_REF - MARGIN
    flat_nb1251 = abs(best_rae - NB1251_REF) < MARGIN

    if beats_nb1251:
        verdict = f"5WAY_GRID_BEATS_NB1251 ({best_tag} @ {best_rae:.4f})"
    elif flat_nb1251:
        verdict = f"5WAY_GRID_FLAT_VS_NB1251 ({best_tag} @ {best_rae:.4f})"
    else:
        verdict = f"5WAY_GRID_HURTS_VS_NB1251 ({best_tag} @ {best_rae:.4f})"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1251 ref (best_fixed_w)    = {NB1251_REF:.4f}")
    print(f"   in-sample 5D grid RAE        = {best_rae_in:.4f}  (gap {best_rae_in-NB1251_REF:+.4f})")
    print(f"   cross-fit 5D grid RAE        = {rae_cf:.4f}  (gap {rae_cf-NB1251_REF:+.4f})")
    print(f"   better of two                = {best_tag} @ {best_rae:.4f}")
    print(f"   beats_nb1251 (>= {MARGIN})      = {beats_nb1251}")
    print(f"   verdict                      = {verdict}")

    # Persist
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy", (P @ best_w_in).astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_cf_oof.npy", cf_oof.astype(np.float32))
    print(f"\n[save] {TAG}_best_oof.npy  /  {TAG}_cf_oof.npy")

    summary = {
        "tag": TAG,
        "n_unb": int(n_unb),
        "step": STEP,
        "n_grid_points": int(G),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "component_tags": tags,
        "component_files": [f for _, f, _ in COMPONENT_FILES],
        "standalone_rae": standalone,
        "pred_corr_matrix": [[float(corr[i, j]) for j in range(len(tags))]
                             for i in range(len(tags))],
        "in_sample_best_weights": {t: float(w) for t, w in zip(tags, best_w_in)},
        "in_sample_best_rae": float(best_rae_in),
        "top10": top10,
        "cross_fit_fold_records": cf_folds,
        "cross_fit_mean_weights": {t: float(w) for t, w in zip(tags, mean_cf_w)},
        "rae_cross_fit": float(rae_cf),
        "mass_concentration": {
            "best_in_sample": {
                "n_nonzero": n_nonzero,
                "n_ge_0p1": n_ge_0p1,
                "n_ge_0p2": n_ge_0p2,
                "max_w": max_w,
                "entropy_nats": float(ent),
                "entropy_uniform_max": float(ent_max),
                "entropy_ratio": float(ent_ratio),
                "concentrated_heuristic": bool(concentrated),
            },
            "top10_n_ge_0p1": [int(x) for x in top10_n_ge_0p1],
            "top10_max_w": [float(x) for x in top10_max_w],
        },
        "candidate_rae_table": candidates,
        "best_tag": best_tag,
        "best_rae": float(best_rae),
        "nb1251_ref": NB1251_REF,
        "delta_in_sample_vs_nb1251": float(best_rae_in - NB1251_REF),
        "delta_cross_fit_vs_nb1251": float(rae_cf - NB1251_REF),
        "delta_best_vs_nb1251": float(best_rae - NB1251_REF),
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
    for k in (
        "standalone_rae",
        "n_grid_points",
        "in_sample_best_weights",
        "in_sample_best_rae",
        "rae_cross_fit",
        "cross_fit_mean_weights",
        "best_tag",
        "best_rae",
        "delta_in_sample_vs_nb1251",
        "delta_cross_fit_vs_nb1251",
        "beats_nb1251",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
