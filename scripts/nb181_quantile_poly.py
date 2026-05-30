"""nb181 — QuantileRegressor (MAE-direct) on polynomial features of top-6 SLSQP models.

Key finding from nb180:
  - Linear SLSQP: RAE=0.3001  ratio=0.580  (reference)
  - Poly(degree=2)+Ridge(alpha=10): RAE=0.3025  ratio=0.598  (PASS but worse RAE)
  - Poly+Ridge ratio=0.598 > linear ratio=0.580 — polynomial features boost ratio.

Ridge optimizes MSE, not MAE. QuantileRegressor(quantile=0.5) directly minimizes MAE.
Hypothesis: same polynomial features + MAE-direct optimization → RAE < 0.3001, ratio ~0.598.

Also tests:
  A: QuantileRegressor(0.5) on 6 raw features — should approximate SLSQP
  B: QuantileRegressor(0.5) on degree-2 poly (27 features)
  C: QuantileRegressor(0.5) on degree-2 poly of top-10 models
  D: Blend linear+poly_quantile to combine strengths
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, Ridge
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

# Top-10 for experiment C
TOP10_MODELS = [
    "nb167_xgboost_mae",
    "nb156_catboost_mae",
    "nb154_lgbm_mae_filtered",
    "nb162_mixed_pool",
    "nb165_multiseed_162c",
    "nb149_meta_maeloss",
    "nb166_catboost_v2",
    "nb168_multiseed_catboost",
    "nb169_rf_et_mae",
    "nb171_catboost_extended",
]


def load_models(stems, n_tr):
    oofs, tes, loaded = [], [], []
    for stem in stems:
        p = DATA_PROCESSED / f"oof_{stem}.npy"
        if not p.exists():
            continue
        for pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{pref}{stem}.npy"
            if te_p.exists():
                break
        else:
            continue
        oof = np.load(p).astype(np.float64).flatten()
        te  = np.load(te_p).astype(np.float64).flatten()
        if len(oof) != n_tr:
            continue
        oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
        te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
        oofs.append(oof); tes.append(te); loaded.append(stem)
    return np.column_stack(oofs), np.column_stack(tes), loaded


def cv_quantile(X_tr, y_tr, X_te, splits, label, alpha_best=None):
    """CV for QuantileRegressor, sweeping alpha if not provided."""
    n_tr = len(y_tr)

    # Sweep alpha if not given
    alphas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    if alpha_best is not None:
        alphas = [alpha_best]

    best_r, best_alpha = 1e9, alphas[0]
    for alpha in alphas:
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof_s[va_idx] = m.predict(X_tr[va_idx])
        r = rae(y_tr, oof_s)
        if r < best_r:
            best_r, best_alpha = r, alpha

    # Final model with best alpha
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = QuantileRegressor(quantile=0.5, alpha=best_alpha, solver="highs")
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx])
    m_f = QuantileRegressor(quantile=0.5, alpha=best_alpha, solver="highs")
    m_f.fit(X_tr, y_tr)
    te_pred = m_f.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label}] alpha={best_alpha}  RAE={r:.4f}  ratio={ratio:.3f}  [{flag}]")
    return r, oof, te_pred, ratio


def main():
    print("=== nb181: QuantileRegressor on polynomial features ===\n")
    print("Goal: poly features boost ratio (0.598 > 0.580); MAE-direct optimizer")
    print("closes the RAE gap (0.3025 -> < 0.3001)?\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load 6-model features
    X6_tr, X6_te, loaded6 = load_models(SIX_MODELS, n_tr)
    print(f"Loaded {len(loaded6)}/6 models")

    # Load top-10 features
    X10_tr, X10_te, loaded10 = load_models(TOP10_MODELS, n_tr)
    print(f"Loaded {len(loaded10)}/10 extended models")

    # Linear SLSQP reference
    def obj(w): return np.mean(np.abs(y_tr - X6_tr@w))/np.mean(np.abs(y_tr-y_tr.mean()))
    res = minimize(obj, np.ones(6)/6, method='SLSQP',
                   bounds=[(0,1)]*6,
                   constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                   options={'ftol':1e-10})
    oof_lin = X6_tr@res.x; te_lin = X6_te@res.x
    r_lin = res.fun; ratio_lin = te_lin.std()/oof_lin.std()
    print(f"Linear SLSQP reference: RAE={r_lin:.6f}  ratio={ratio_lin:.3f}\n")

    results = {}

    # --- A: QuantileRegressor on 6 raw features ---
    print("--- A: QuantileRegressor(0.5) on 6 raw features ---")
    r_a, oof_a, te_a, ratio_a = cv_quantile(X6_tr, y_tr, X6_te, splits, "QReg_6raw")
    if ratio_a >= COLLAPSE_THRESH:
        results["A_qreg_6raw"] = (r_a, oof_a, te_a, ratio_a)

    # --- B: QuantileRegressor on degree-2 polynomial of 6 features ---
    print("\n--- B: QuantileRegressor(0.5) on degree-2 poly (6->27 features) ---")
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X6p2_tr = poly2.fit_transform(X6_tr)
    X6p2_te = poly2.transform(X6_te)
    print(f"  Poly features: {X6p2_tr.shape}")
    r_b, oof_b, te_b, ratio_b = cv_quantile(X6p2_tr, y_tr, X6p2_te, splits, "QReg_poly2_6")
    if ratio_b >= COLLAPSE_THRESH:
        results["B_qreg_poly2_6"] = (r_b, oof_b, te_b, ratio_b)

    # --- C: QuantileRegressor on degree-2 poly of top-10 features ---
    if X10_tr.shape[1] >= 8:
        print(f"\n--- C: QuantileRegressor(0.5) on degree-2 poly of top-{X10_tr.shape[1]} features ---")
        poly2_10 = PolynomialFeatures(degree=2, include_bias=False)
        X10p2_tr = poly2_10.fit_transform(X10_tr)
        X10p2_te = poly2_10.transform(X10_te)
        print(f"  Poly features: {X10p2_tr.shape}")
        r_c, oof_c, te_c, ratio_c = cv_quantile(X10p2_tr, y_tr, X10p2_te, splits, "QReg_poly2_10")
        if ratio_c >= COLLAPSE_THRESH:
            results["C_qreg_poly2_10"] = (r_c, oof_c, te_c, ratio_c)
    else:
        r_c = ratio_c = None

    # --- D: Blend linear SLSQP + best nonlinear ---
    # Try blending linear with best poly-quantile result
    print("\n--- D: Blend linear + poly-QuantileReg ---")
    for key in ["B_qreg_poly2_6", "A_qreg_6raw"]:
        if key in results:
            r_nl, oof_nl, te_nl, ratio_nl = results[key]
            print(f"  Blending linear(0.3001) with {key}(RAE={r_nl:.4f})")
            for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                oof_d = alpha * oof_lin + (1-alpha) * oof_nl
                te_d  = alpha * te_lin  + (1-alpha) * te_nl
                r_d = rae(y_tr, oof_d); ratio_d = te_d.std()/oof_d.std()
                flag = "PASS" if ratio_d >= COLLAPSE_THRESH else "FAIL"
                print(f"  alpha={alpha:.1f}  RAE={r_d:.4f}  ratio={ratio_d:.3f}  [{flag}]")
                if ratio_d >= COLLAPSE_THRESH and r_d < results.get("D_best_r", 1e9):
                    results["D_best_r"] = r_d
                    results["D_blend"] = (r_d, oof_d, te_d, ratio_d)
            break

    # --- E: Ridge with lower alpha on poly features (find the knee) ---
    print("\n--- E: Ridge alpha sweep on degree-2 poly (find lower-alpha knee) ---")
    best_r_e, best_alpha_e, best_oof_e, best_te_e = 1e9, None, None, None
    for alpha_r in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        oof_e = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = Ridge(alpha=alpha_r)
            m.fit(X6p2_tr[tr_idx], y_tr[tr_idx])
            oof_e[va_idx] = m.predict(X6p2_tr[va_idx])
        m_f = Ridge(alpha=alpha_r); m_f.fit(X6p2_tr, y_tr)
        te_e = m_f.predict(X6p2_te)
        r_e = rae(y_tr, oof_e); ratio_e = te_e.std()/oof_e.std()
        flag = "PASS" if ratio_e >= COLLAPSE_THRESH else "FAIL"
        print(f"  Ridge alpha={alpha_r:6.3f}  RAE={r_e:.4f}  ratio={ratio_e:.3f}  [{flag}]")
        if ratio_e >= COLLAPSE_THRESH and r_e < best_r_e:
            best_r_e, best_alpha_e = r_e, alpha_r
            best_oof_e, best_te_e = oof_e.copy(), te_e.copy()
    if best_alpha_e is not None:
        ratio_e_best = best_te_e.std() / best_oof_e.std()
        results["E_ridge_poly"] = (best_r_e, best_oof_e, best_te_e, ratio_e_best)

    # === Summary ===
    print(f"\n=== Summary ===")
    print(f"  Linear SLSQP reference:  RAE=0.300101  ratio=0.580  PASS")
    clean = {k: v for k, v in results.items() if not k.endswith("_r") and v[3] >= COLLAPSE_THRESH}
    for k, (r, _, _, ratio) in sorted(clean.items(), key=lambda x: x[1][0]):
        better = "*** BEATS 0.3001! ***" if r < 0.3001 else ""
        print(f"  {k:30s}  RAE={r:.6f}  ratio={ratio:.3f}  PASS  {better}")

    if not clean:
        print("  No models passed collapse check better than linear!")
        print("  Using linear blend as output.")
        best_oof, best_te = oof_lin, te_lin
        best_r = r_lin
    else:
        best_key = min(clean, key=lambda k: clean[k][0])
        best_r, best_oof, best_te, best_ratio = clean[best_key]
        print(f"\nBEST: {best_key}  RAE={best_r:.6f}")
        if best_r < 0.3001:
            print("*** NEW BEST — BEATS LINEAR BLEND! ***")
        else:
            print("Linear blend wins.")

    # Save OOF/TE for ensemble pool
    np.save(DATA_PROCESSED / "oof_nb181_quantile_poly.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb181_quantile_poly.npy", best_te)

    best_te_clip = np.clip(best_te, y_tr.min()-0.5, y_tr.max()+0.5)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_clip})
    sub.to_csv(SUBMISSIONS / "181_quantile_poly.csv", index=False)
    print(f"Saved: submissions/181_quantile_poly.csv  OOF RAE={best_r:.6f}")


if __name__ == "__main__":
    main()
