"""nb144 — Grand Ensemble v10 (All models through nb143+).

Improvements over nb134 (Grand v9):
  - Adds nb140, nb141, nb142, nb143 to the model pool
  - k=2 exhaustive search (nb143 alone may blend well with few models)
  - k=3/4/5/6 with expanded top_n for k=4+
  - Anchored search: fix nb143 (best meta-stack) and search best 1/2/3 complements
  - SLSQP + perturbation for top-200 combos at each k

Expected pool: ~115+ models. nb143 should be best single model (~0.3186).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats, optimize
from itertools import combinations

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
COLLAPSE_THRESH = 0.58


def load_all_models(n_tr, y_tr, thresh=COLLAPSE_THRESH):
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
            te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
            r = rae(y_tr, oof)
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results


def optimize_weights_slsqp(oof_mat, y_tr, w0=None):
    k = oof_mat.shape[1]
    if w0 is None:
        w0 = np.ones(k) / k

    def objective(w):
        pred = oof_mat @ w
        mae = np.mean(np.abs(y_tr - pred))
        baseline = np.mean(np.abs(y_tr - y_tr.mean()))
        return mae / baseline

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * k
    res = optimize.minimize(objective, w0, method="SLSQP",
                            bounds=bounds, constraints=constraints,
                            options={"ftol": 1e-8, "maxiter": 500})
    return res.fun, res.x


def perturb_optimize(oof_mat, y_tr, n_iters=20):
    k = oof_mat.shape[1]
    w = np.ones(k) / k
    best_r = rae(y_tr, oof_mat @ w)
    for _ in range(n_iters):
        for i in range(k):
            for delta in np.linspace(-0.5, 0.5, 21):
                w2 = w.copy(); w2[i] += delta
                w2 = np.clip(w2, 0, 1)
                if w2.sum() < 1e-6: continue
                w2 /= w2.sum()
                r = rae(y_tr, oof_mat @ w2)
                if r < best_r:
                    best_r, w = r, w2.copy()
    r2, w2 = optimize_weights_slsqp(oof_mat, y_tr, w)
    if r2 < best_r:
        best_r, w = r2, w2
    return best_r, w


def search_k(oof_mat, y_tr, stems, k, top_n, top_combos=200, label=""):
    n_mod = min(top_n, oof_mat.shape[1])
    from math import comb
    n_comb = comb(n_mod, k)
    print(f"\n{label} exhaustive (top-{n_mod}, C={n_comb:,})...", flush=True)
    top_combos_eq = []
    for idx in combinations(range(n_mod), k):
        pred = oof_mat[:, list(idx)].mean(axis=1)
        r = rae(y_tr, pred)
        top_combos_eq.append((r, *idx))
    top_combos_eq.sort(key=lambda x: x[0])
    print(f"  Equal-weight best {label}: {top_combos_eq[0][0]:.4f}")

    best_r, best_combo, best_w = top_combos_eq[0][0], top_combos_eq[0][1:], None
    for entry in top_combos_eq[:top_combos]:
        r_eq, *idx_list = entry
        sub = oof_mat[:, list(idx_list)]
        r_opt, w_opt = perturb_optimize(sub, y_tr)
        if r_opt < best_r:
            best_r = r_opt; best_combo = tuple(idx_list); best_w = w_opt
    print(f"  Optimized best {label}: {best_r:.4f}")
    return best_r, best_combo, best_w


def anchored_search(oof_mat, y_tr, stems, anchor_idx, complement_top_n, k_complements, label=""):
    """Fix anchor model, search best k_complements companions from top-complement_top_n."""
    n_mod = min(complement_top_n, oof_mat.shape[1])
    others = [i for i in range(n_mod) if i != anchor_idx]
    from math import comb
    n_comb = comb(len(others), k_complements)
    print(f"\n{label} anchored search (anchor={stems[anchor_idx][:30]}, C={n_comb:,})...", flush=True)

    combos_eq = []
    for comp in combinations(others, k_complements):
        idx = [anchor_idx] + list(comp)
        pred = oof_mat[:, idx].mean(axis=1)
        r = rae(y_tr, pred)
        combos_eq.append((r, *idx))
    combos_eq.sort(key=lambda x: x[0])
    print(f"  Equal-weight best: {combos_eq[0][0]:.4f}")

    best_r, best_combo, best_w = combos_eq[0][0], combos_eq[0][1:], None
    for entry in combos_eq[:200]:
        r_eq, *idx_list = entry
        sub = oof_mat[:, list(idx_list)]
        r_opt, w_opt = perturb_optimize(sub, y_tr)
        if r_opt < best_r:
            best_r = r_opt; best_combo = tuple(idx_list); best_w = w_opt
    print(f"  Optimized best: {best_r:.4f}")
    return best_r, best_combo, best_w


def main():
    print("=== nb144: Grand Ensemble v10 ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)

    models = load_all_models(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded (threshold={COLLAPSE_THRESH})")
    for m in models[:20]:
        print(f"    {m['stem']:55s}  RAE={m['rae']:.4f}  ratio={m['ratio']:.2f}")

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    stems   = [m["stem"] for m in models]

    # k=2: exhaustive over top-30
    best_2_r, best_2_combo, best_2_w = search_k(
        oof_mat, y_tr, stems, k=2, top_n=min(30, n_mod), top_combos=200, label="k=2")

    # k=3: exhaustive over all models
    best_3_r, best_3_combo, best_3_w = search_k(
        oof_mat, y_tr, stems, k=3, top_n=n_mod, top_combos=200, label="k=3")

    # k=4: exhaustive over top-35
    best_4_r, best_4_combo, best_4_w = search_k(
        oof_mat, y_tr, stems, k=4, top_n=min(35, n_mod), top_combos=200, label="k=4")

    # k=5: exhaustive over top-25
    best_5_r, best_5_combo, best_5_w = search_k(
        oof_mat, y_tr, stems, k=5, top_n=min(25, n_mod), top_combos=100, label="k=5")

    # k=6: exhaustive over top-18
    best_6_r, best_6_combo, best_6_w = search_k(
        oof_mat, y_tr, stems, k=6, top_n=min(18, n_mod), top_combos=50, label="k=6")

    # Anchored search: fix best model (index 0), find best 1/2/3 complements
    best_anc2_r, best_anc2_combo, best_anc2_w = anchored_search(
        oof_mat, y_tr, stems, anchor_idx=0, complement_top_n=min(40, n_mod),
        k_complements=1, label="anchored k=1+1")

    best_anc3_r, best_anc3_combo, best_anc3_w = anchored_search(
        oof_mat, y_tr, stems, anchor_idx=0, complement_top_n=min(30, n_mod),
        k_complements=2, label="anchored k=1+2")

    candidates = [
        ("k=2", best_2_r, best_2_combo, best_2_w),
        ("k=3", best_3_r, best_3_combo, best_3_w),
        ("k=4", best_4_r, best_4_combo, best_4_w),
        ("k=5", best_5_r, best_5_combo, best_5_w),
        ("k=6", best_6_r, best_6_combo, best_6_w),
        ("anchored k=2", best_anc2_r, best_anc2_combo, best_anc2_w),
        ("anchored k=3", best_anc3_r, best_anc3_combo, best_anc3_w),
    ]
    best_label, best_r, best_combo, best_w = min(candidates, key=lambda x: x[1])
    print(f"\n=== BEST: {best_label}  OOF RAE={best_r:.4f} ===")
    safe_w = best_w if best_w is not None else [1/len(best_combo)]*len(best_combo)
    for ii, ww in zip(best_combo, safe_w):
        print(f"  {stems[ii]:55s}  w={ww:.3f}")

    best_oof = oof_mat[:, list(best_combo)] @ np.array(safe_w)
    best_te  = te_mat[:,  list(best_combo)] @ np.array(safe_w)
    best_te  = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio = best_te.std() / best_oof.std()
    print(f"Test: med={np.median(best_te):.2f}  std={best_te.std():.3f}  "
          f"max={best_te.max():.2f}  ratio={ratio:.2f}")

    np.save(DATA_PROCESSED / "oof_nb144_grand_v10.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb144_grand_v10.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "144_grand_ensemble_v10.csv", index=False)
    print(f"\nSaved: submissions/144_grand_ensemble_v10.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
