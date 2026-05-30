"""nb170 — Grand Ensemble v15.

Run after nb157 (Optuna LGBM), nb165 (multiseed nb162C), nb166 (CatBoost v2),
nb167 (XGBoost MAE), nb168 (multiseed CatBoost), nb169 (RF/ET MAE) complete.

Uses same search strategy as nb164 (grand v14):
  - k=2,3,4,5 exhaustive + optimized weights
  - Caruana forward selection (300 iters)
  - Anchored search with best known anchors

Current best to beat: OOF RAE = 0.3013 (nb164 grand v14)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import optimize
from itertools import combinations
from math import comb

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
COLLAPSE_THRESH = 0.58

# Exclude grand-ensemble and derived outputs to avoid circular stacking
EXCLUDE_DERIVED = {
    "nb164_grand_v14", "nb170_grand_v15",
    "nb172_bootstrap_ensemble", "nb173_softmax_sweep",
    "nb175_bayes_blend", "nb174_top10_lgbm",
    "nb108_grand_v2", "nb112_grand_v3", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
    "nb134_grand_v9", "nb144_grand_v10", "nb151_grand_v11",
    "nb153_grand_v12", "nb155_grand_v13",
}


def load_all_models(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in EXCLUDE_DERIVED:
            continue
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


def anchored_search(oof_mat, y_tr, stems, anchor_idx, k_add, top_n=None):
    n_mod = oof_mat.shape[1] if top_n is None else min(top_n, oof_mat.shape[1])
    non_anchors = [i for i in range(n_mod) if i not in anchor_idx]
    print(f"\nAnchored search: anchors={[stems[i] for i in anchor_idx]}, adding k={k_add}...")

    best_r = 1e9; best_combo = None
    for add_idx in combinations(non_anchors, k_add):
        full_idx = list(anchor_idx) + list(add_idx)
        sub = oof_mat[:, full_idx]
        pred = sub.mean(axis=1)
        r = rae(y_tr, pred)
        if r < best_r:
            best_r = r; best_combo = tuple(full_idx)

    best_w = None
    if best_combo:
        sub = oof_mat[:, list(best_combo)]
        r_opt, w_opt = perturb_optimize(sub, y_tr)
        if r_opt < best_r:
            best_r = r_opt; best_w = w_opt
    print(f"  Anchored best: {best_r:.4f}")
    return best_r, best_combo, best_w


def caruana_forward_selection(oof_mat, y_tr, n_iters=300):
    n_tr, n_mod = oof_mat.shape
    counts = np.zeros(n_mod, dtype=np.int32)
    best_r = rae(y_tr, np.mean(oof_mat, axis=1))
    ensemble = np.zeros(n_tr); current_n = 0
    print(f"\nCaruana forward selection ({n_iters} iters, {n_mod} models)...", flush=True)
    for iteration in range(n_iters):
        best_add = -1
        for i in range(n_mod):
            candidate = (ensemble * current_n + oof_mat[:, i]) / (current_n + 1)
            r = rae(y_tr, candidate)
            if r < best_r:
                best_r = r; best_add = i
        if best_add == -1:
            print(f"  Converged at iter {iteration+1}, RAE={best_r:.4f}")
            break
        counts[best_add] += 1; current_n += 1
        ensemble = (ensemble * (current_n - 1) + oof_mat[:, best_add]) / current_n
        if iteration % 50 == 0:
            print(f"  iter {iteration+1}: RAE={best_r:.4f}", flush=True)
    w = counts / counts.sum()
    print(f"  Caruana final RAE={best_r:.4f}, non-zero: {(counts>0).sum()}")
    return best_r, w, counts


def main():
    print("=== nb170: Grand Ensemble v15 ===\n")
    print("Target: beat OOF RAE 0.3013 (nb164 grand v14)\n")

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

    # k=2 search (top 20)
    best_2_r, best_2_combo, best_2_w = search_k(
        oof_mat, y_tr, stems, k=2, top_n=min(20, n_mod), top_combos=200, label="k=2")
    # k=3 search (all)
    best_3_r, best_3_combo, best_3_w = search_k(
        oof_mat, y_tr, stems, k=3, top_n=n_mod, top_combos=200, label="k=3")
    # k=4 search (top 30)
    best_4_r, best_4_combo, best_4_w = search_k(
        oof_mat, y_tr, stems, k=4, top_n=min(30, n_mod), top_combos=200, label="k=4")
    # k=5 search (top 25)
    best_5_r, best_5_combo, best_5_w = search_k(
        oof_mat, y_tr, stems, k=5, top_n=min(25, n_mod), top_combos=150, label="k=5")
    # k=6 search (top 20)
    best_6_r, best_6_combo, best_6_w = search_k(
        oof_mat, y_tr, stems, k=6, top_n=min(20, n_mod), top_combos=100, label="k=6")

    # Anchored searches: fix the three best models as anchors
    anchor_priority = [
        "nb156_catboost_mae", "nb154_lgbm_mae_filtered", "nb149_meta_maeloss",
        "nb162_mixed_pool", "nb168_multiseed_catboost", "nb165_multiseed_162c",
        "nb157_optuna_lgbm_mae", "nb166_catboost_v2",
    ]
    anchor_idx = []
    for ac in anchor_priority[:3]:
        if ac in stems and stems.index(ac) not in anchor_idx:
            anchor_idx.append(stems.index(ac))
    if len(anchor_idx) >= 2:
        anch_1_r, anch_1_combo, anch_1_w = anchored_search(oof_mat, y_tr, stems, anchor_idx[:2], k_add=1)
        anch_2_r, anch_2_combo, anch_2_w = anchored_search(oof_mat, y_tr, stems, anchor_idx[:2], k_add=2)
        anch_3_r, anch_3_combo, anch_3_w = anchored_search(oof_mat, y_tr, stems, anchor_idx[:2], k_add=3, top_n=min(30, n_mod))
    else:
        anch_1_r = anch_2_r = anch_3_r = 1e9
        anch_1_combo = anch_2_combo = anch_3_combo = None
        anch_1_w = anch_2_w = anch_3_w = None

    # Caruana
    car_r, car_w, car_counts = caruana_forward_selection(oof_mat, y_tr, n_iters=300)
    car_oof = oof_mat @ car_w
    car_te  = te_mat  @ car_w

    candidates = [
        ("k=2", best_2_r, best_2_combo, best_2_w),
        ("k=3", best_3_r, best_3_combo, best_3_w),
        ("k=4", best_4_r, best_4_combo, best_4_w),
        ("k=5", best_5_r, best_5_combo, best_5_w),
        ("k=6", best_6_r, best_6_combo, best_6_w),
        ("caruana", car_r, None, car_w),
    ]
    if anch_1_combo: candidates.append(("anchored+1", anch_1_r, anch_1_combo, anch_1_w))
    if anch_2_combo: candidates.append(("anchored+2", anch_2_r, anch_2_combo, anch_2_w))
    if anch_3_combo: candidates.append(("anchored+3", anch_3_r, anch_3_combo, anch_3_w))

    best_label, best_r, best_combo, best_w_final = min(candidates, key=lambda x: x[1])
    print(f"\n=== BEST: {best_label}  OOF RAE={best_r:.4f} ===")

    if best_label == "caruana":
        best_oof = car_oof; best_te = car_te
        for ii in np.where(car_counts > 0)[0]:
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

    if best_r < 0.3013:
        print("*** NEW BEST ENSEMBLE! (beat 0.3013) ***")
    elif best_r < 0.3044:
        print("*** Improvement over nb155 (0.3044) ***")

    np.save(DATA_PROCESSED / "oof_nb170_grand_v15.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb170_grand_v15.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "170_grand_v15.csv", index=False)
    print(f"\nSaved: submissions/170_grand_v15.csv  OOF RAE={best_r:.4f}")
    print(f"(nb164 grand v14 best: 0.3013)")


if __name__ == "__main__":
    main()
