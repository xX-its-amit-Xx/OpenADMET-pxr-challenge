"""nb127 — Exhaustive 3-Way Ensemble Search.

The Optuna k=3 ensemble (nb119) found OOF 0.3706 using 300 trials on 79 models.
C(79, 3) = 79,002 combinations. Optuna can't enumerate all combinations in 300 trials.

This script does an exhaustive search:
  1. Start with equal-weight 3-way blends over all C(79,3) combinations
  2. For each promising combination (top-100), optimize weights via grid search
  3. Report the best weighted 3-way blend

Also searches k=4 and k=5 (using only top-20 models for computational feasibility):
  - C(20, 4) = 4,845 combinations
  - C(20, 5) = 15,504 combinations

This is purely deterministic and comprehensive — no randomness.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
COLLAPSE_THRESH = 0.58
TOP_MODELS_K4 = 25  # use top-25 for k=4 search


def load_all_models(n_tr, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists(): break
        else:
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te  = np.where(np.isfinite(te), te,   np.nanmean(te))
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
            r = rae(y_tr, oof)
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results


def best_weighted_blend(oof_mat, y_tr, n_grid=11):
    """Find best convex combination for up to 5 columns via grid search."""
    k = oof_mat.shape[1]
    if k == 2:
        best_r, best_w = 1.0, None
        for a in np.linspace(0, 1, n_grid):
            pred = a * oof_mat[:, 0] + (1 - a) * oof_mat[:, 1]
            r = rae(y_tr, pred)
            if r < best_r:
                best_r, best_w = r, np.array([a, 1 - a])
        return best_r, best_w
    # For k >= 3: use uniform weights as starting point + local search
    uniform = np.ones(k) / k
    pred_u = oof_mat @ uniform
    best_r = rae(y_tr, pred_u)
    best_w = uniform.copy()
    # Perturb each weight
    for i in range(k):
        for delta in np.linspace(-0.4, 0.4, 9):
            w = uniform.copy()
            w[i] += delta
            w = np.clip(w, 0, 1)
            w /= w.sum()
            pred = oof_mat @ w
            r = rae(y_tr, pred)
            if r < best_r:
                best_r, best_w = r, w.copy()
    return best_r, best_w


def main():
    global y_tr
    print("=== nb127: Exhaustive 3-Way Ensemble Search ===\n")

    from pxr.data import load_train, load_test
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)

    print("Loading models...")
    models = load_all_models(n_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded")
    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    stems = [m["stem"] for m in models]

    # === k=3 exhaustive search (equal weights) ===
    print(f"\nk=3 exhaustive search (C({n_mod},3)={n_mod*(n_mod-1)*(n_mod-2)//6:,} combos)...")
    top100 = []
    for i, j, k in combinations(range(n_mod), 3):
        pred = (oof_mat[:, i] + oof_mat[:, j] + oof_mat[:, k]) / 3
        r = rae(y_tr, pred)
        top100.append((r, i, j, k))
    top100.sort(key=lambda x: x[0])
    print(f"  Equal-weight best: {top100[0][0]:.4f}  "
          f"({stems[top100[0][1]]}, {stems[top100[0][2]]}, {stems[top100[0][3]]})")

    # Optimize weights for top-200 equal-weight combos
    print("  Optimizing weights for top-200 combos...")
    best_3_r, best_3_combo, best_3_w = top100[0][0], top100[0][1:], None
    for entry in top100[:200]:
        r_eq, i, j, k = entry
        sub = oof_mat[:, [i, j, k]]
        r_opt, w_opt = best_weighted_blend(sub, y_tr)
        if r_opt < best_3_r:
            best_3_r = r_opt
            best_3_combo = (i, j, k)
            best_3_w = w_opt
    i3, j3, k3 = best_3_combo
    print(f"  Best weighted 3-way: RAE={best_3_r:.4f}")
    print(f"    {stems[i3]} ({best_3_w[0] if best_3_w is not None else '1/3':.3f})")
    print(f"    {stems[j3]} ({best_3_w[1] if best_3_w is not None else '1/3':.3f})")
    print(f"    {stems[k3]} ({best_3_w[2] if best_3_w is not None else '1/3':.3f})")

    # === k=4 exhaustive (top-25 models) ===
    top_n = min(TOP_MODELS_K4, n_mod)
    n_k4 = top_n * (top_n - 1) * (top_n - 2) * (top_n - 3) // 24
    print(f"\nk=4 exhaustive search (top-{top_n} models, C={n_k4:,} combos)...")
    top50_4 = []
    for i, j, k, l in combinations(range(top_n), 4):
        pred = (oof_mat[:, i] + oof_mat[:, j] + oof_mat[:, k] + oof_mat[:, l]) / 4
        r = rae(y_tr, pred)
        top50_4.append((r, i, j, k, l))
    top50_4.sort(key=lambda x: x[0])
    print(f"  Equal-weight best 4-way: {top50_4[0][0]:.4f}")

    # Optimize top-100 k=4 combos
    best_4_r, best_4_combo, best_4_w = top50_4[0][0], top50_4[0][1:], None
    for entry in top50_4[:100]:
        r_eq, *idx = entry
        sub = oof_mat[:, list(idx)]
        r_opt, w_opt = best_weighted_blend(sub, y_tr)
        if r_opt < best_4_r:
            best_4_r = r_opt
            best_4_combo = tuple(idx)
            best_4_w = w_opt
    idx4 = best_4_combo
    print(f"  Best weighted 4-way: RAE={best_4_r:.4f}")
    for ii, ww in zip(idx4, best_4_w if best_4_w is not None else [0.25]*4):
        print(f"    {stems[ii]} ({ww:.3f})")

    # === Best of k=3 and k=4 ===
    if best_3_r <= best_4_r:
        best_oof = oof_mat[:, list(best_3_combo)] @ best_3_w
        best_te  = te_mat[:,  list(best_3_combo)] @ best_3_w
        best_r   = best_3_r
        print(f"\nBest: k=3  OOF RAE={best_r:.4f}")
    else:
        best_oof = oof_mat[:, list(best_4_combo)] @ best_4_w
        best_te  = te_mat[:,  list(best_4_combo)] @ best_4_w
        best_r   = best_4_r
        print(f"\nBest: k=4  OOF RAE={best_r:.4f}")

    best_te = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"Test: med={np.median(best_te):.2f}  std={best_te.std():.3f}  "
          f"ratio={best_te.std()/best_oof.std():.2f}")

    np.save(DATA_PROCESSED / "oof_nb127_exhaustive_blend.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb127_exhaustive_blend.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "127_exhaustive_3way.csv", index=False)
    print(f"\nSaved: submissions/127_exhaustive_3way.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
