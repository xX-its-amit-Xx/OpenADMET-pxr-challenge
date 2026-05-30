"""nb153 — Grand Ensemble v12 (All models through nb152+).

After nb152 (LGBM_MAE tuned): run exhaustive search with new best model pool.
Expected new models: nb152_lgbm_mae_tuned, nb141_xgb_ablation, nb132_diverse_seed.

Strategy: k=2-5 exhaustive + anchored search (fix nb149/nb152 as anchor).
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


def main():
    print("=== nb153: Grand Ensemble v12 ===\n")

    from pxr.data import load_train, load_test
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

    # k=2: exhaustive over top-20
    best_2_r, best_2_combo, best_2_w = search_k(
        oof_mat, y_tr, stems, k=2, top_n=min(20, n_mod), top_combos=200, label="k=2")

    # k=3: exhaustive over all models
    best_3_r, best_3_combo, best_3_w = search_k(
        oof_mat, y_tr, stems, k=3, top_n=n_mod, top_combos=200, label="k=3")

    # k=4: exhaustive over top-30
    best_4_r, best_4_combo, best_4_w = search_k(
        oof_mat, y_tr, stems, k=4, top_n=min(30, n_mod), top_combos=200, label="k=4")

    # k=5: exhaustive over top-20
    best_5_r, best_5_combo, best_5_w = search_k(
        oof_mat, y_tr, stems, k=5, top_n=min(20, n_mod), top_combos=100, label="k=5")

    candidates = [
        ("k=2", best_2_r, best_2_combo, best_2_w),
        ("k=3", best_3_r, best_3_combo, best_3_w),
        ("k=4", best_4_r, best_4_combo, best_4_w),
        ("k=5", best_5_r, best_5_combo, best_5_w),
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
    print(f"Test: med={np.median(best_te):.2f}  std={best_te.std():.3f}  ratio={ratio:.2f}")

    np.save(DATA_PROCESSED / "oof_nb153_grand_v12.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb153_grand_v12.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "153_grand_ensemble_v12.csv", index=False)
    print(f"\nSaved: submissions/153_grand_ensemble_v12.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
