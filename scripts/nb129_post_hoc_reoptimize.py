"""nb129 — Post-hoc Weight Re-optimization on Expanded Model Set.

After nb120-nb128 complete, re-run the exhaustive 3/4-way search on the
expanded pool of models. This serves as the definitive final ensemble search.

Key additions since nb127:
  - nb120_huber variants (huber_0_5, huber_1_0, huber_2_0, mape)
  - nb121_test_sim_weighted
  - nb122_nonlinear_meta
  - nb123 LAD variants
  - nb124_scaffold_specific
  - nb125 variants (2way, ridge, enet)
  - nb126_classifier_conditioned
  - nb128 RF/ET augmented

Strategy:
  1. Exhaustive k=3 over all non-collapsed models (including new ones)
  2. Exhaustive k=4 over top-30 models
  3. Exhaustive k=5 over top-20 models
  4. Optimize weights for top-100 combos at each k
  5. Final submission from best overall
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
COLLAPSE_THRESH = 0.58


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
            te  = np.where(np.isfinite(te), te, np.nanmean(te))
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
            r = rae(y_tr, oof)
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results


def perturb_optimize(oof_mat, y_tr, n_iters=30):
    """Local search for optimal weights via coordinate-wise perturbation."""
    k = oof_mat.shape[1]
    w = np.ones(k) / k
    best_r = rae(y_tr, oof_mat @ w)
    for _ in range(n_iters):
        for i in range(k):
            for delta in np.linspace(-0.5, 0.5, 21):
                w2 = w.copy()
                w2[i] += delta
                w2 = np.clip(w2, 0, 1)
                if w2.sum() < 1e-6: continue
                w2 /= w2.sum()
                r = rae(y_tr, oof_mat @ w2)
                if r < best_r:
                    best_r, w = r, w2.copy()
    return best_r, w


def main():
    global y_tr
    print("=== nb129: Post-hoc Weight Re-optimization ===\n")

    from pxr.data import load_train, load_test
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)

    models = load_all_models(n_tr)
    n_mod = len(models)
    print(f"  {n_mod} models available")
    for m in models[:12]:
        print(f"    {m['stem']:50s}  RAE={m['rae']:.4f}  ratio={m['ratio']:.2f}")

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    stems   = [m["stem"] for m in models]

    # k=3: exhaustive
    n_k3 = n_mod * (n_mod - 1) * (n_mod - 2) // 6
    print(f"\nk=3 exhaustive (C={n_k3:,})...")
    top200_3 = []
    for i, j, k in combinations(range(n_mod), 3):
        pred = (oof_mat[:, i] + oof_mat[:, j] + oof_mat[:, k]) / 3
        r = rae(y_tr, pred)
        top200_3.append((r, i, j, k))
    top200_3.sort(key=lambda x: x[0])
    print(f"  Equal-weight best k=3: {top200_3[0][0]:.4f}  "
          f"({stems[top200_3[0][1]][:30]}, ...)")

    best_3_r, best_3_combo, best_3_w = top200_3[0][0], top200_3[0][1:], None
    for entry in top200_3[:200]:
        r_eq, i, j, k = entry
        sub = oof_mat[:, [i, j, k]]
        r_opt, w_opt = perturb_optimize(sub, y_tr)
        if r_opt < best_3_r:
            best_3_r = r_opt; best_3_combo = (i, j, k); best_3_w = w_opt
    print(f"  Optimized best k=3: {best_3_r:.4f}")
    for ii, ww in zip(best_3_combo, best_3_w if best_3_w is not None else [1/3]*3):
        print(f"    {stems[ii]:45s}  w={ww:.3f}")

    # k=4: exhaustive on top-30
    top_n4 = min(30, n_mod)
    n_k4 = top_n4 * (top_n4 - 1) * (top_n4 - 2) * (top_n4 - 3) // 24
    print(f"\nk=4 exhaustive (top-{top_n4}, C={n_k4:,})...")
    top200_4 = []
    for i, j, k, l in combinations(range(top_n4), 4):
        pred = (oof_mat[:, i] + oof_mat[:, j] + oof_mat[:, k] + oof_mat[:, l]) / 4
        r = rae(y_tr, pred)
        top200_4.append((r, i, j, k, l))
    top200_4.sort(key=lambda x: x[0])
    print(f"  Equal-weight best k=4: {top200_4[0][0]:.4f}")

    best_4_r, best_4_combo, best_4_w = top200_4[0][0], top200_4[0][1:], None
    for entry in top200_4[:200]:
        r_eq, *idx = entry
        sub = oof_mat[:, list(idx)]
        r_opt, w_opt = perturb_optimize(sub, y_tr)
        if r_opt < best_4_r:
            best_4_r = r_opt; best_4_combo = tuple(idx); best_4_w = w_opt
    print(f"  Optimized best k=4: {best_4_r:.4f}")

    # k=5: exhaustive on top-20
    top_n5 = min(20, n_mod)
    n_k5 = top_n5 * (top_n5 - 1) * (top_n5 - 2) * (top_n5 - 3) * (top_n5 - 4) // 120
    print(f"\nk=5 exhaustive (top-{top_n5}, C={n_k5:,})...")
    top100_5 = []
    for idx in combinations(range(top_n5), 5):
        pred = oof_mat[:, list(idx)].mean(axis=1)
        r = rae(y_tr, pred)
        top100_5.append((r, *idx))
    top100_5.sort(key=lambda x: x[0])
    print(f"  Equal-weight best k=5: {top100_5[0][0]:.4f}")

    best_5_r, best_5_combo, best_5_w = top100_5[0][0], top100_5[0][1:], None
    for entry in top100_5[:100]:
        r_eq, *idx = entry
        sub = oof_mat[:, list(idx)]
        r_opt, w_opt = perturb_optimize(sub, y_tr)
        if r_opt < best_5_r:
            best_5_r = r_opt; best_5_combo = tuple(idx); best_5_w = w_opt
    print(f"  Optimized best k=5: {best_5_r:.4f}")

    # Final best
    candidates = [
        ("k=3", best_3_r, best_3_combo, best_3_w),
        ("k=4", best_4_r, best_4_combo, best_4_w),
        ("k=5", best_5_r, best_5_combo, best_5_w),
    ]
    best_label, best_r, best_combo, best_w = min(candidates, key=lambda x: x[1])
    print(f"\n=== BEST: {best_label}  OOF RAE={best_r:.4f} ===")
    for ii, ww in zip(best_combo, best_w if best_w is not None else [1/len(best_combo)]*len(best_combo)):
        print(f"  {stems[ii]:50s}  w={ww:.3f}")

    best_oof = oof_mat[:, list(best_combo)] @ best_w
    best_te  = te_mat[:,  list(best_combo)] @ best_w
    best_te  = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"Test: med={np.median(best_te):.2f}  std={best_te.std():.3f}  "
          f"max={best_te.max():.2f}  ratio={best_te.std()/best_oof.std():.2f}")

    np.save(DATA_PROCESSED / "oof_nb129_final_blend.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb129_final_blend.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "129_post_hoc_reoptimize.csv", index=False)
    print(f"\nSaved: submissions/129_post_hoc_reoptimize.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
