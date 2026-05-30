"""nb175 — Bayesian Model Averaging blend.

nb173 showed that blend(grand_v14*0.3 + softmax_top10*0.7) = 0.3006.
This is a 2-component blend. Can we push further?

Strategy: treat each model's OOF RAE as a proxy for its "likelihood" and
use BMA-style weighting: w_i ∝ exp(-beta * RAE_i).

Also tests combining grand_v14 with multiple new model outputs.

Tests:
  A: BMA of all passing models (beta sweep)
  B: blend(grand_v14, nb165_multiseed, alpha sweep)
  C: blend(grand_v14, nb167_xgboost, alpha sweep)
  D: 3-way blend: grand_v14 + nb165 + nb167
  E: full grid search over all best models
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import optimize
from itertools import product

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
COLLAPSE_THRESH = 0.58

EXCLUDE_DERIVED = {
    "nb164_grand_v14", "nb170_grand_v15", "nb175_bayes_blend",
    "nb172_bootstrap_ensemble", "nb173_softmax_sweep",
    "nb108_grand_v2", "nb112_grand_v3", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
    "nb134_grand_v9", "nb144_grand_v10", "nb151_grand_v11",
    "nb153_grand_v12", "nb155_grand_v13",
}


def load_all_models(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in EXCLUDE_DERIVED: continue
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


def load_single(stem, n_tr):
    p = DATA_PROCESSED / f"oof_{stem}.npy"
    if not p.exists(): return None, None
    for te_pref in ("te_", "te_oof_"):
        te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
        if te_p.exists(): break
    else:
        return None, None
    oof = np.load(p).astype(np.float64)
    te  = np.load(te_p).astype(np.float64)
    if oof.ndim == 2: oof = oof[:, 0]
    if te.ndim == 2:  te  = te[:, 0]
    if len(oof) != n_tr: return None, None
    return oof, te


def main():
    print("=== nb175: Bayesian Model Averaging blend ===\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr); n_te = len(te_df)

    # Load base models (excluding derived)
    models = load_all_models(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded (excluding derived ensembles)")
    for m in models[:20]:
        print(f"    {m['stem']:55s}  RAE={m['rae']:.4f}")

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    stems   = [m["stem"] for m in models]
    raes    = np.array([m["rae"] for m in models])

    # Load grand_v14 as reference
    gv14_oof, gv14_te = load_single("nb164_grand_v14", n_tr)
    if gv14_oof is None:
        print("WARNING: grand_v14 OOF not found")
        gv14_oof = oof_mat[:, 0]; gv14_te = te_mat[:, 0]

    results = {}

    # --- A: BMA on top-20 (beta sweep) ---
    print(f"\n--- A: BMA on top-20 (beta sweep) ---")
    TOP_N = min(20, n_mod)
    for beta in [10, 20, 30, 50, 100, 200]:
        log_w = -beta * raes[:TOP_N]
        log_w -= log_w.max()
        w = np.exp(log_w); w /= w.sum()
        oof_a = oof_mat[:, :TOP_N] @ w; te_a = te_mat[:, :TOP_N] @ w
        r_a = rae(y_tr, oof_a); ratio_a = te_a.std() / oof_a.std()
        flag = "PASS" if ratio_a >= COLLAPSE_THRESH else "FAIL"
        print(f"  beta={beta:3d}  RAE={r_a:.4f}  ratio={ratio_a:.2f}  [{flag}]")
        if r_a < results.get("A_best", (1e9,))[0] and ratio_a >= COLLAPSE_THRESH:
            results["A_bma"] = (r_a, oof_a, te_a, ratio_a)

    # --- B: blend grand_v14 + nb165 multiseed ---
    nb165_oof, nb165_te = load_single("nb165_multiseed_162c", n_tr)
    if nb165_oof is not None:
        print(f"\n--- B: blend grand_v14 + nb165_multiseed ---")
        r_gv14 = rae(y_tr, gv14_oof); r_165 = rae(y_tr, nb165_oof)
        print(f"  gv14 RAE={r_gv14:.4f}, nb165 RAE={r_165:.4f}")
        for alpha in np.arange(0.1, 1.0, 0.1):
            oof_b = (1-alpha)*gv14_oof + alpha*nb165_oof
            te_b  = (1-alpha)*gv14_te  + alpha*nb165_te
            r_b = rae(y_tr, oof_b); ratio_b = te_b.std() / oof_b.std()
            flag = "PASS" if ratio_b >= COLLAPSE_THRESH else "FAIL"
            if alpha in [0.2, 0.4, 0.6, 0.8] or r_b < results.get("B_best", (1e9,))[0]:
                print(f"  alpha={alpha:.1f}  RAE={r_b:.4f}  ratio={ratio_b:.2f}  [{flag}]")
            if r_b < results.get("B_best", (1e9,))[0] and ratio_b >= COLLAPSE_THRESH:
                results[f"B_nb165_a{alpha:.1f}"] = (r_b, oof_b, te_b, ratio_b)

    # --- C: blend grand_v14 + nb167 XGBoost ---
    nb167_oof, nb167_te = load_single("nb167_xgboost_mae", n_tr)
    if nb167_oof is not None:
        print(f"\n--- C: blend grand_v14 + nb167_xgboost ---")
        r_167 = rae(y_tr, nb167_oof)
        print(f"  gv14 RAE={rae(y_tr, gv14_oof):.4f}, nb167 RAE={r_167:.4f}")
        for alpha in np.arange(0.1, 1.0, 0.1):
            oof_c = (1-alpha)*gv14_oof + alpha*nb167_oof
            te_c  = (1-alpha)*gv14_te  + alpha*nb167_te
            r_c = rae(y_tr, oof_c); ratio_c = te_c.std() / oof_c.std()
            flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
            if alpha in [0.2, 0.4, 0.6, 0.8] or r_c < results.get("C_best", (1e9,))[0]:
                print(f"  alpha={alpha:.1f}  RAE={r_c:.4f}  ratio={ratio_c:.2f}  [{flag}]")
            if r_c < results.get("C_best", (1e9,))[0] and ratio_c >= COLLAPSE_THRESH:
                results[f"C_nb167_a{alpha:.1f}"] = (r_c, oof_c, te_c, ratio_c)

    # --- D: 3-way: grand_v14 + nb165 + nb167 ---
    if nb165_oof is not None and nb167_oof is not None:
        print(f"\n--- D: 3-way grid grand_v14 + nb165 + nb167 ---")
        best_D_r = 1e9
        for w0, w1, w2 in [(0.3,0.4,0.3),(0.2,0.5,0.3),(0.2,0.4,0.4),(0.3,0.3,0.4),(0.1,0.6,0.3),(0.1,0.4,0.5)]:
            oof_d = w0*gv14_oof + w1*nb165_oof + w2*nb167_oof
            te_d  = w0*gv14_te  + w1*nb165_te  + w2*nb167_te
            r_d = rae(y_tr, oof_d); ratio_d = te_d.std() / oof_d.std()
            flag = "PASS" if ratio_d >= COLLAPSE_THRESH else "FAIL"
            print(f"  w=({w0:.1f},{w1:.1f},{w2:.1f})  RAE={r_d:.4f}  ratio={ratio_d:.2f}  [{flag}]")
            if r_d < best_D_r and ratio_d >= COLLAPSE_THRESH:
                best_D_r = r_d; results["D_3way"] = (r_d, oof_d, te_d, ratio_d)

    # --- E: SLSQP on just top-10 models (no structural features) ---
    print(f"\n--- E: SLSQP weight optimization on top-10 ---")
    TOP10 = min(10, n_mod)
    oof10 = oof_mat[:, :TOP10]; te10 = te_mat[:, :TOP10]
    def obj(w):
        pred = oof10 @ w
        return np.mean(np.abs(y_tr - pred)) / np.mean(np.abs(y_tr - y_tr.mean()))
    res = optimize.minimize(
        obj, np.ones(TOP10)/TOP10, method="SLSQP",
        bounds=[(0,1)]*TOP10,
        constraints=[{"type":"eq","fun": lambda w: w.sum()-1}],
        options={"ftol":1e-9,"maxiter":1000}
    )
    oof_e = oof10 @ res.x; te_e = te10 @ res.x
    r_e = rae(y_tr, oof_e); ratio_e = te_e.std() / oof_e.std()
    flag_e = "PASS" if ratio_e >= COLLAPSE_THRESH else "FAIL"
    print(f"  SLSQP top-10  RAE={r_e:.4f}  ratio={ratio_e:.2f}  [{flag_e}]")
    for i, (s, w) in enumerate(zip(stems[:TOP10], res.x)):
        if w > 0.01:
            print(f"    {s[:50]:50s}  w={w:.3f}")
    if ratio_e >= COLLAPSE_THRESH:
        results["E_slsqp_top10"] = (r_e, oof_e, te_e, ratio_e)

    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:35s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBEST: {best_label}  RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb173 best: 0.3006, nb164 grand v14: 0.3013)")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb175_bayes_blend.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb175_bayes_blend.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "175_bayes_blend.csv", index=False)
    print(f"Saved: submissions/175_bayes_blend.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
