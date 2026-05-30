"""nb155 — Grand Ensemble v13 (After nb154 filtered LGBM_MAE).

Run after nb154 saves its OOF. Pool now includes:
  nb154_lgbm_mae_filtered (best filtered LGBM_MAE, expected < 0.3069)
  nb153_grand_v12 (k=5 ensemble, 0.3055)
  nb149_meta_maeloss (0.3069)
  etc.

Uses same k=2-5 exhaustive + perturb+SLSQP strategy.
Also adds forward selection (Caruana method) as an additional search strategy.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats, optimize
from itertools import combinations
from math import comb

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
    if w0 is None: w0 = np.ones(k) / k
    def objective(w):
        return np.mean(np.abs(y_tr - oof_mat @ w)) / np.mean(np.abs(y_tr - y_tr.mean()))
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * k
    res = optimize.minimize(objective, w0, method="SLSQP", bounds=bounds,
                            constraints=constraints, options={"ftol": 1e-9, "maxiter": 1000})
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


def caruana_forward_selection(oof_mat, y_tr, n_iters=200, n_bag=20, bag_frac=0.5):
    """Caruana forward selection with replacement (bagged variant for stability)."""
    n_tr, n_mod = oof_mat.shape
    counts = np.zeros(n_mod, dtype=np.int32)
    best_r = rae(y_tr, np.mean(oof_mat, axis=1))  # start with equal blend
    ensemble = np.zeros(n_tr)
    current_n = 0

    print(f"\nCaruana forward selection ({n_iters} iters, {n_mod} models)...", flush=True)
    for iteration in range(n_iters):
        best_add = -1
        for i in range(n_mod):
            # Add model i once more (with replacement)
            candidate = (ensemble * current_n + oof_mat[:, i]) / (current_n + 1)
            r = rae(y_tr, candidate)
            if r < best_r:
                best_r = r
                best_add = i
        if best_add == -1:
            print(f"  Converged at iter {iteration+1}, RAE={best_r:.4f}")
            break
        counts[best_add] += 1
        current_n += 1
        ensemble = (ensemble * (current_n - 1) + oof_mat[:, best_add]) / current_n
        if iteration % 20 == 0:
            print(f"  iter {iteration+1}: added #{best_add}, RAE={best_r:.4f}", flush=True)

    w = counts / counts.sum()
    print(f"  Caruana final RAE={best_r:.4f}, non-zero models: {(counts > 0).sum()}")
    return best_r, w, counts


def main():
    print("=== nb155: Grand Ensemble v13 ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)

    models = load_all_models(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded (threshold={COLLAPSE_THRESH})")
    for m in models[:25]:
        print(f"    {m['stem']:55s}  RAE={m['rae']:.4f}  ratio={m['ratio']:.2f}")

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    stems   = [m["stem"] for m in models]

    # Standard k=2-5 exhaustive
    best_2_r, best_2_combo, best_2_w = search_k(
        oof_mat, y_tr, stems, k=2, top_n=min(20, n_mod), top_combos=200, label="k=2")
    best_3_r, best_3_combo, best_3_w = search_k(
        oof_mat, y_tr, stems, k=3, top_n=n_mod, top_combos=200, label="k=3")
    best_4_r, best_4_combo, best_4_w = search_k(
        oof_mat, y_tr, stems, k=4, top_n=min(30, n_mod), top_combos=200, label="k=4")
    best_5_r, best_5_combo, best_5_w = search_k(
        oof_mat, y_tr, stems, k=5, top_n=min(20, n_mod), top_combos=100, label="k=5")

    # Caruana forward selection
    car_r, car_w, car_counts = caruana_forward_selection(oof_mat, y_tr, n_iters=300)
    car_oof = oof_mat @ car_w
    car_te  = te_mat  @ car_w

    candidates = [
        ("k=2", best_2_r, best_2_combo, best_2_w),
        ("k=3", best_3_r, best_3_combo, best_3_w),
        ("k=4", best_4_r, best_4_combo, best_4_w),
        ("k=5", best_5_r, best_5_combo, best_5_w),
        ("caruana", car_r, None, car_w),
    ]
    best_label, best_r, best_combo, best_w_final = min(candidates, key=lambda x: x[1])
    print(f"\n=== BEST: {best_label}  OOF RAE={best_r:.4f} ===")

    if best_label == "caruana":
        best_oof = car_oof
        best_te  = car_te
        top_idxs = np.where(car_counts > 0)[0]
        for ii in top_idxs:
            print(f"  {stems[ii]:55s}  count={car_counts[ii]}  w={car_w[ii]:.3f}")
    else:
        safe_w = best_w_final if best_w_final is not None else [1/len(best_combo)]*len(best_combo)
        for ii, ww in zip(best_combo, safe_w):
            print(f"  {stems[ii]:55s}  w={ww:.3f}")
        best_oof = oof_mat[:, list(best_combo)] @ np.array(safe_w)
        best_te  = te_mat[:,  list(best_combo)] @ np.array(safe_w)

    best_te = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio = best_te.std() / best_oof.std()
    print(f"Test: med={np.median(best_te):.2f}  std={best_te.std():.3f}  ratio={ratio:.2f}")

    np.save(DATA_PROCESSED / "oof_nb155_grand_v13.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb155_grand_v13.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "155_grand_v13.csv", index=False)
    print(f"\nSaved: submissions/155_grand_v13.csv  OOF RAE={best_r:.4f}")
    print(f"(nb153 best was 0.3055, nb151 best was 0.3056)")


if __name__ == "__main__":
    main()
