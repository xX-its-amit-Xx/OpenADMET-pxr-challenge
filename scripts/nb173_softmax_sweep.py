"""nb173 — Fine-grained softmax temperature sweep.

nb172 found that softmax RAE-weighting of top-20 models at T=0.01 achieves
OOF RAE 0.3037 (vs. grand v14 optimized SLSQP 0.3013) without any training.

This notebook sweeps T more finely, tries different top-N values, and also
tests inverse-rank weighting as an alternative to softmax.

Key insight: simpler methods are less overfit to OOF selection bias.

Strategies:
  A: T sweep (0.001 to 0.1) on top-20
  B: Different top-N (5, 10, 15, 20, 30, all) at best T
  C: Inverse-rank weighting (1/rank)
  D: Harmonic rank weighting (1/(rank+constant))
  E: Best found × grand v14 blend
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
COLLAPSE_THRESH = 0.58

EXCLUDE_SELF = {
    "nb172_bootstrap_ensemble", "nb173_softmax_sweep",
    "nb164_grand_v14", "nb170_grand_v15",
}


def load_all_models(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in EXCLUDE_SELF: continue
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


def softmax_blend(oof_mat, te_mat, raes, T):
    inv_rae = 1.0 / raes
    log_w = inv_rae / T - (inv_rae / T).max()
    w = np.exp(log_w); w /= w.sum()
    oof = oof_mat @ w; te = te_mat @ w
    return oof, te, w


def main():
    print("=== nb173: Softmax Temperature Sweep ===\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)

    models = load_all_models(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded")
    for m in models[:25]:
        print(f"    {m['stem']:55s}  RAE={m['rae']:.4f}")

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    stems   = [m["stem"] for m in models]
    raes    = np.array([m["rae"] for m in models])

    results = {}

    # --- A: T sweep on top-20 ---
    print(f"\n--- A: Temperature sweep, top-20 ---")
    TOP_N = 20
    T_vals = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10]
    best_T_r = 1e9; best_T = None
    for T in T_vals:
        oof_t, te_t, w_t = softmax_blend(oof_mat[:, :TOP_N], te_mat[:, :TOP_N], raes[:TOP_N], T)
        r_t = rae(y_tr, oof_t); ratio_t = te_t.std() / oof_t.std()
        flag = "PASS" if ratio_t >= COLLAPSE_THRESH else "FAIL"
        print(f"  T={T:.3f}  RAE={r_t:.4f}  ratio={ratio_t:.2f}  [{flag}]")
        if r_t < best_T_r and ratio_t >= COLLAPSE_THRESH:
            best_T_r = r_t; best_T = T
            results[f"A_T{T}"] = (r_t, oof_t, te_t, ratio_t)

    print(f"  Best T={best_T}  RAE={best_T_r:.4f}")
    if best_T is None: best_T = 0.01

    # --- B: top-N sweep at best T ---
    print(f"\n--- B: top-N sweep at T={best_T} ---")
    best_N_r = 1e9; best_N = None
    for N in [5, 10, 15, 20, 30, min(40, n_mod), n_mod]:
        if N > n_mod: continue
        oof_n, te_n, w_n = softmax_blend(oof_mat[:, :N], te_mat[:, :N], raes[:N], best_T)
        r_n = rae(y_tr, oof_n); ratio_n = te_n.std() / oof_n.std()
        flag = "PASS" if ratio_n >= COLLAPSE_THRESH else "FAIL"
        print(f"  N={N:3d}  RAE={r_n:.4f}  ratio={ratio_n:.2f}  [{flag}]")
        if r_n < best_N_r and ratio_n >= COLLAPSE_THRESH:
            best_N_r = r_n; best_N = N
            results[f"B_N{N}"] = (r_n, oof_n, te_n, ratio_n)

    print(f"  Best N={best_N}  RAE={best_N_r:.4f}")
    if best_N is None: best_N = 20

    # --- C: Inverse-rank weighting (1/rank) ---
    print(f"\n--- C: Inverse-rank weighting ---")
    for N in [10, 20, 30]:
        if N > n_mod: continue
        ranks = np.arange(1, N + 1, dtype=float)
        w_r = 1.0 / ranks; w_r /= w_r.sum()
        oof_c = oof_mat[:, :N] @ w_r; te_c = te_mat[:, :N] @ w_r
        r_c = rae(y_tr, oof_c); ratio_c = te_c.std() / oof_c.std()
        flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
        print(f"  N={N}  RAE={r_c:.4f}  ratio={ratio_c:.2f}  [{flag}]")
        if r_c < results.get("C_best", (1e9,))[0] and ratio_c >= COLLAPSE_THRESH:
            results["C_inv_rank"] = (r_c, oof_c, te_c, ratio_c)

    # --- D: Harmonic rank (1/(rank+1)) ---
    print(f"\n--- D: Harmonic weighting (1/(rank+1)) ---")
    for N in [10, 20, 30]:
        if N > n_mod: continue
        ranks = np.arange(1, N + 1, dtype=float)
        w_h = 1.0 / (ranks + 1); w_h /= w_h.sum()
        oof_h = oof_mat[:, :N] @ w_h; te_h = te_mat[:, :N] @ w_h
        r_h = rae(y_tr, oof_h); ratio_h = te_h.std() / oof_h.std()
        flag = "PASS" if ratio_h >= COLLAPSE_THRESH else "FAIL"
        print(f"  N={N}  RAE={r_h:.4f}  ratio={ratio_h:.2f}  [{flag}]")
        if r_h < results.get("D_best", (1e9,))[0] and ratio_h >= COLLAPSE_THRESH:
            results["D_harm_rank"] = (r_h, oof_h, te_h, ratio_h)

    # --- E: Blend best-found with grand v14 ---
    gv14_oof_path = DATA_PROCESSED / "oof_nb164_grand_v14.npy"
    gv14_te_path  = DATA_PROCESSED / "te_nb164_grand_v14.npy"
    if gv14_oof_path.exists() and gv14_te_path.exists():
        print(f"\n--- E: Blend best softmax with grand v14 ---")
        oof_gv14 = np.load(gv14_oof_path).astype(np.float64)
        te_gv14  = np.load(gv14_te_path).astype(np.float64)
        r_gv14 = rae(y_tr, oof_gv14)
        print(f"  grand v14 OOF RAE={r_gv14:.4f}")

        # Find best so far
        if results:
            best_key = min(results, key=lambda k: results[k][0])
            best_r_cur, best_oof_cur, best_te_cur, _ = results[best_key]
            for alpha in [0.1, 0.2, 0.3, 0.5, 0.7]:
                oof_e = (1 - alpha) * oof_gv14 + alpha * best_oof_cur
                te_e  = (1 - alpha) * te_gv14  + alpha * best_te_cur
                r_e = rae(y_tr, oof_e); ratio_e = te_e.std() / oof_e.std()
                flag = "PASS" if ratio_e >= COLLAPSE_THRESH else "FAIL"
                print(f"  alpha={alpha:.1f} (gv14*(1-a)+softmax*a)  RAE={r_e:.4f}  ratio={ratio_e:.2f}  [{flag}]")
                if r_e < results.get("E_blend_best", (1e9,))[0] and ratio_e >= COLLAPSE_THRESH:
                    results[f"E_blend_a{alpha}"] = (r_e, oof_e, te_e, ratio_e)

    # --- Summary ---
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:35s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBEST: {best_label}  RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb164 grand v14: 0.3013, nb172 softmax T=0.01: 0.3037)")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb173_softmax_sweep.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb173_softmax_sweep.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "173_softmax_sweep.csv", index=False)
    print(f"Saved: submissions/173_softmax_sweep.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
