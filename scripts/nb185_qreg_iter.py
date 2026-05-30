"""nb185 -- Iterate QReg poly: apply poly trick to 7-model OOFs.

7-model SLSQP (nb183 + 6 originals): RAE=0.299711 ratio=0.5801 PASS (NEW BEST)

Can we iterate? QReg poly on the 7 models captures nonlinear interactions
between nb183 (poly-10) and the 6 linear SLSQP components.

Tests:
  A: QReg poly (degree=2) on 7-model OOFs, fine alpha sweep
  B: QReg poly (degree=3) on 6 original OOFs only
  C: QReg poly (degree=2) on 7 models + extended features (mean, std, range)
  D: 8-model SLSQP: 7 models + best nb185 result
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58
SEED = 42

SIX_MODELS = [
    "nb167_xgboost_mae",
    "nb156_catboost_mae",
    "nb154_lgbm_mae_filtered",
    "nb162_mixed_pool",
    "nb165_multiseed_162c",
    "nb149_meta_maeloss",
]


def load_oof(stem, n_tr):
    p = DATA_PROCESSED / f"oof_{stem}.npy"
    if not p.exists():
        return None, None
    for pref in ("te_", "te_oof_"):
        te_p = DATA_PROCESSED / f"{pref}{stem}.npy"
        if te_p.exists():
            break
    else:
        return None, None
    oof = np.load(p).astype(np.float64).flatten()
    te  = np.load(te_p).astype(np.float64).flatten()
    if len(oof) != n_tr:
        return None, None
    oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
    te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
    return oof, te


def qreg_sweep(X_tr, y_tr, X_te, splits, label, alphas):
    n_tr = len(y_tr)
    best_r, best_alpha, best_oof, best_te = 1e9, None, None, None
    for alpha in alphas:
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof_s[va_idx] = m.predict(X_tr[va_idx])
        m_f = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        m_f.fit(X_tr, y_tr)
        te_s = m_f.predict(X_te)
        r_s = rae(y_tr, oof_s); ratio_s = te_s.std()/oof_s.std()
        flag = "PASS" if ratio_s >= COLLAPSE_THRESH else "FAIL"
        beat = " ***BEATS 0.2997!***" if (ratio_s >= COLLAPSE_THRESH and r_s < 0.2997) else ""
        print(f"  [{label}] alpha={alpha:.5f}  RAE={r_s:.6f}  ratio={ratio_s:.4f}  [{flag}]{beat}")
        if ratio_s >= COLLAPSE_THRESH and r_s < best_r:
            best_r, best_alpha, best_oof, best_te = r_s, alpha, oof_s.copy(), te_s.copy()
    return best_r, best_oof, best_te, best_alpha


def slsqp_multistart(X_tr, y_tr, X_te, n_starts, label):
    k = X_tr.shape[1]
    mean_pred = y_tr.mean()
    def obj(w): return np.mean(np.abs(y_tr - X_tr@w)) / np.mean(np.abs(y_tr - mean_pred))
    best_r, best_w = 1e9, None
    for _ in range(n_starts):
        w0 = np.random.dirichlet(np.ones(k))
        r = minimize(obj, w0, method='SLSQP',
                     bounds=[(0,1)]*k,
                     constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                     options={'ftol':1e-12, 'maxiter':2000})
        if r.fun < best_r:
            best_r, best_w = r.fun, r.x
    oof_b = X_tr@best_w; te_b = X_te@best_w
    ratio = te_b.std()/oof_b.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label}] RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]  w_max={best_w.max():.4f}")
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb185: Iterate QReg poly on 7-model OOFs ===\n")
    print("Reference: 7-model SLSQP = 0.299711  ratio=0.5801\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load 6 originals + nb183
    oofs6, tes6 = [], []
    for stem in SIX_MODELS:
        oof, te = load_oof(stem, n_tr)
        if oof is not None:
            oofs6.append(oof); tes6.append(te)
    X6_tr = np.column_stack(oofs6); X6_te = np.column_stack(tes6)

    oof183, te183 = load_oof("nb183_qreg_poly10", n_tr)
    if oof183 is None:
        print("ERROR: nb183 not found!")
        return

    X7_tr = np.column_stack([X6_tr, oof183.reshape(-1,1)])
    X7_te = np.column_stack([X6_te, te183.reshape(-1,1)])
    print(f"7-model matrix: {X7_tr.shape}")

    results = {}

    # --- A: QReg poly-2 on 7 models ---
    print("--- A: QReg(0.5) degree-2 poly on 7 models (7->28 features) ---")
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X7p2_tr = poly2.fit_transform(X7_tr); X7p2_te = poly2.transform(X7_te)
    print(f"  Features: {X7p2_tr.shape}")
    alphas_a = [0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.002, 0.003, 0.005, 0.01]
    r_a, oof_a, te_a, alpha_a = qreg_sweep(X7p2_tr, y_tr, X7p2_te, splits, "QReg_poly2_7", alphas_a)
    if oof_a is not None:
        results["A_poly2_7"] = (r_a, oof_a, te_a, te_a.std()/oof_a.std())

    # --- B: QReg poly-3 on 6 original models (degree=3) ---
    print("\n--- B: QReg(0.5) degree-3 poly on 6 originals (6->83 features) ---")
    poly3 = PolynomialFeatures(degree=3, include_bias=False)
    X6p3_tr = poly3.fit_transform(X6_tr); X6p3_te = poly3.transform(X6_te)
    print(f"  Features: {X6p3_tr.shape}")
    alphas_b = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02]
    r_b, oof_b, te_b, alpha_b = qreg_sweep(X6p3_tr, y_tr, X6p3_te, splits, "QReg_poly3_6", alphas_b)
    if oof_b is not None:
        results["B_poly3_6"] = (r_b, oof_b, te_b, te_b.std()/oof_b.std())

    # --- C: QReg poly-2 on 7 models + stats features ---
    print("\n--- C: QReg(0.5) poly-2 on 7 + mean/std/range features ---")
    X7ext_tr = np.hstack([X7_tr, X7_tr.mean(1,keepdims=True),
                          X7_tr.std(1,keepdims=True),
                          (X7_tr.max(1,keepdims=True)-X7_tr.min(1,keepdims=True))])
    X7ext_te = np.hstack([X7_te, X7_te.mean(1,keepdims=True),
                          X7_te.std(1,keepdims=True),
                          (X7_te.max(1,keepdims=True)-X7_te.min(1,keepdims=True))])
    poly2c = PolynomialFeatures(degree=2, include_bias=False)
    X7extp2_tr = poly2c.fit_transform(X7ext_tr); X7extp2_te = poly2c.transform(X7ext_te)
    print(f"  Features: {X7extp2_tr.shape}")
    alphas_c = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]
    r_c, oof_c, te_c, alpha_c = qreg_sweep(X7extp2_tr, y_tr, X7extp2_te, splits, "QReg_poly2_7ext", alphas_c)
    if oof_c is not None:
        results["C_poly2_7ext"] = (r_c, oof_c, te_c, te_c.std()/oof_c.std())

    # --- D: 8-model SLSQP: 7 models + best nb185 ---
    if results:
        best_key = min(results, key=lambda k: results[k][0])
        r_best, oof_best, te_best, ratio_best = results[best_key]
        print(f"\n--- D: 8-model SLSQP: 7 models + best ({best_key}, RAE={r_best:.6f}) ---")
        X8_tr = np.column_stack([X7_tr, oof_best.reshape(-1,1)])
        X8_te = np.column_stack([X7_te, te_best.reshape(-1,1)])
        r_d, oof_d, te_d, ratio_d = slsqp_multistart(X8_tr, y_tr, X8_te, 200, "8model_SLSQP")
        if ratio_d >= COLLAPSE_THRESH:
            results["D_8model"] = (r_d, oof_d, te_d, ratio_d)
            if r_d < 0.2997:
                print(f"  *** BEATS 0.2997! ***")

    # === Summary ===
    print(f"\n=== Summary ===")
    print(f"  7-model SLSQP (nb184):   RAE=0.299711  ratio=0.5801  PASS")
    clean = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    for k, (r, _, _, ratio) in sorted(clean.items(), key=lambda x: x[1][0]):
        beat = " *** NEW BEST ***" if r < 0.2997 else ("*** beats 0.299711 ***" if r < 0.299711 else "")
        print(f"  {k:30s}  RAE={r:.6f}  ratio={ratio:.4f}  PASS  {beat}")

    if not clean:
        print("  No new models pass collapse check below 0.299711.")
        return

    best_r, best_oof, best_te, best_ratio = min(clean.values(), key=lambda x: x[0])
    if best_r >= 0.299711:
        print(f"\n7-model SLSQP (0.299711) still best.")
        return

    best_te_clip = np.clip(best_te, y_tr.min()-0.5, y_tr.max()+0.5)
    np.save(DATA_PROCESSED / "oof_nb185_qreg_iter.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb185_qreg_iter.npy",  best_te_clip)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_clip})
    sub.to_csv(SUBMISSIONS / "185_qreg_iter.csv", index=False)
    print(f"\nSaved: submissions/185_qreg_iter.csv  OOF RAE={best_r:.6f}")
    print(f"*** NEW BEST: {best_r:.6f} vs 0.299711 ***")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
