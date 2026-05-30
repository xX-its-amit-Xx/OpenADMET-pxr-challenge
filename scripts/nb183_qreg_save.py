"""nb183 -- Save QReg poly-10 alpha=0.0012 winner from nb182.

nb182 found:
  QReg(0.5, alpha=0.0012) on degree-2 poly of top-10 model OOFs
  RAE=0.300023  ratio=0.5800  PASS  -- beats linear blend (0.300101)!

This script:
1. Trains the winning model
2. Saves OOF + TE predictions
3. Generates submission CSV
4. Reruns a quick blend check: linear + qreg-poly-10 at various ratios
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

SIX_MODELS = [
    "nb167_xgboost_mae",
    "nb156_catboost_mae",
    "nb154_lgbm_mae_filtered",
    "nb162_mixed_pool",
    "nb165_multiseed_162c",
    "nb149_meta_maeloss",
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


def main():
    print("=== nb183: Save QReg poly-10 winner + blend check ===\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    X10_tr, X10_te, loaded10 = load_models(TOP10_MODELS, n_tr)
    X6_tr,  X6_te,  loaded6  = load_models(SIX_MODELS, n_tr)
    print(f"Top-10 loaded: {len(loaded10)}, Top-6 loaded: {len(loaded6)}")

    # Polynomial features
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X10p2_tr = poly2.fit_transform(X10_tr)
    X10p2_te = poly2.transform(X10_te)
    print(f"Poly features: {X10p2_tr.shape}")

    # Linear SLSQP reference
    def obj(w): return np.mean(np.abs(y_tr - X6_tr@w))/np.mean(np.abs(y_tr-y_tr.mean()))
    res = minimize(obj, np.ones(6)/6, method='SLSQP',
                   bounds=[(0,1)]*6,
                   constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                   options={'ftol':1e-10})
    oof_lin = X6_tr@res.x; te_lin = X6_te@res.x
    r_lin = res.fun; ratio_lin = te_lin.std()/oof_lin.std()
    print(f"Linear SLSQP: RAE={r_lin:.6f}  ratio={ratio_lin:.4f}\n")

    # --- Train and save winning model: alpha=0.0012 ---
    print("--- Training winning QReg poly-10 (alpha=0.0012) ---")
    alpha_win = 0.0012
    oof_win = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = QuantileRegressor(quantile=0.5, alpha=alpha_win, solver="highs")
        m.fit(X10p2_tr[tr_idx], y_tr[tr_idx])
        oof_win[va_idx] = m.predict(X10p2_tr[va_idx])
    m_f = QuantileRegressor(quantile=0.5, alpha=alpha_win, solver="highs")
    m_f.fit(X10p2_tr, y_tr)
    te_win = m_f.predict(X10p2_te)
    r_win = rae(y_tr, oof_win)
    ratio_win = te_win.std() / oof_win.std()
    flag_win = "PASS" if ratio_win >= COLLAPSE_THRESH else "FAIL"
    print(f"  [QReg_poly10_a0012] RAE={r_win:.6f}  ratio={ratio_win:.6f}  [{flag_win}]")

    # Save OOF and TE for pool
    np.save(DATA_PROCESSED / "oof_nb183_qreg_poly10.npy", oof_win)
    np.save(DATA_PROCESSED / "te_nb183_qreg_poly10.npy",  te_win)
    print(f"  Saved OOF/TE predictions to data/processed/")

    # --- Also test alpha=0.0015 (second winner) ---
    print("\n--- Also training alpha=0.0015 ---")
    alpha_2 = 0.0015
    oof_2 = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = QuantileRegressor(quantile=0.5, alpha=alpha_2, solver="highs")
        m.fit(X10p2_tr[tr_idx], y_tr[tr_idx])
        oof_2[va_idx] = m.predict(X10p2_tr[va_idx])
    m_f2 = QuantileRegressor(quantile=0.5, alpha=alpha_2, solver="highs")
    m_f2.fit(X10p2_tr, y_tr)
    te_2 = m_f2.predict(X10p2_te)
    r_2 = rae(y_tr, oof_2); ratio_2 = te_2.std()/oof_2.std()
    flag_2 = "PASS" if ratio_2 >= COLLAPSE_THRESH else "FAIL"
    print(f"  [QReg_poly10_a0015] RAE={r_2:.6f}  ratio={ratio_2:.6f}  [{flag_2}]")

    # --- Blend winner with linear ---
    print("\n--- Blend QReg_poly10(a=0.0012) with linear SLSQP ---")
    best_blend_r, best_blend_oof, best_blend_te, best_blend_ratio = 1e9, None, None, None
    for alpha_b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        oof_b = alpha_b * oof_lin + (1-alpha_b) * oof_win
        te_b  = alpha_b * te_lin  + (1-alpha_b) * te_win
        r_b = rae(y_tr, oof_b); ratio_b = te_b.std()/oof_b.std()
        flag_b = "PASS" if ratio_b >= COLLAPSE_THRESH else "FAIL"
        beat = " ***BEATS 0.3001!***" if (ratio_b >= COLLAPSE_THRESH and r_b < 0.3001) else ""
        print(f"  alpha_lin={alpha_b:.1f}  RAE={r_b:.6f}  ratio={ratio_b:.4f}  [{flag_b}]{beat}")
        if ratio_b >= COLLAPSE_THRESH and r_b < best_blend_r:
            best_blend_r = r_b
            best_blend_oof, best_blend_te, best_blend_ratio = oof_b, te_b, ratio_b

    # --- Multi-start SLSQP over 7 models: 6 SLSQP models + QReg_poly10 ---
    print("\n--- SLSQP over 7 models: 6-model pool + QReg_poly10 winner ---")
    # Build 7-column OOF matrix
    X6_plus = np.column_stack([X6_tr, oof_win])  # (n_tr, 7)
    X6_te_plus = np.column_stack([X6_te, te_win])  # (n_te, 7)

    def obj7(w): return np.mean(np.abs(y_tr - X6_plus@w))/np.mean(np.abs(y_tr-y_tr.mean()))
    best_r7, best_w7, best_oof7, best_te7 = 1e9, None, None, None
    for _ in range(50):
        w0 = np.random.dirichlet(np.ones(7))
        r7 = minimize(obj7, w0, method='SLSQP',
                      bounds=[(0,1)]*7,
                      constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                      options={'ftol':1e-12, 'maxiter':1000})
        if r7.fun < best_r7:
            best_r7, best_w7 = r7.fun, r7.x
    oof7 = X6_plus@best_w7; te7 = X6_te_plus@best_w7
    ratio7 = te7.std()/oof7.std()
    flag7 = "PASS" if ratio7 >= COLLAPSE_THRESH else "FAIL"
    print(f"  7-model SLSQP: RAE={best_r7:.6f}  ratio={ratio7:.4f}  [{flag7}]")
    print(f"  Weights: {dict(zip(['nb167','nb156','nb154','nb162','nb165','nb149','nb183'], best_w7.round(4)))}")
    if ratio7 >= COLLAPSE_THRESH:
        beat = "***BEATS 0.3001!***" if best_r7 < 0.3001 else "(not better)"
        print(f"  {beat}")

    # === Summary ===
    print(f"\n=== Summary ===")
    print(f"  Linear SLSQP (6-model):  RAE=0.300101  ratio=0.580  PASS")
    print(f"  QReg poly-10 a=0.0012:   RAE={r_win:.6f}  ratio={ratio_win:.4f}  [{flag_win}]")
    print(f"  QReg poly-10 a=0.0015:   RAE={r_2:.6f}  ratio={ratio_2:.4f}  [{flag_2}]")
    print(f"  7-model SLSQP:           RAE={best_r7:.6f}  ratio={ratio7:.4f}  [{flag7}]")

    # Determine best model to save as submission
    candidates = []
    if ratio_win >= COLLAPSE_THRESH:
        candidates.append((r_win, oof_win, te_win, "QReg_poly10_a0012"))
    if ratio_2 >= COLLAPSE_THRESH:
        candidates.append((r_2, oof_2, te_2, "QReg_poly10_a0015"))
    if ratio7 >= COLLAPSE_THRESH:
        candidates.append((best_r7, oof7, te7, "7model_SLSQP"))
    if best_blend_oof is not None:
        candidates.append((best_blend_r, best_blend_oof, best_blend_te, "blend_lin_qreg"))

    if not candidates:
        print("\nNo new models pass collapse check.")
        return

    best_label = min(candidates, key=lambda x: x[0])
    best_r_final, best_oof_final, best_te_final, label = best_label
    best_te_clip = np.clip(best_te_final, y_tr.min()-0.5, y_tr.max()+0.5)

    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_clip})
    sub.to_csv(SUBMISSIONS / "183_qreg_poly10.csv", index=False)
    print(f"\nBEST: {label}  RAE={best_r_final:.6f}")
    if best_r_final < 0.3001:
        print("*** NEW BEST -- BEATS LINEAR BLEND (0.3001)! ***")
    print(f"Saved: submissions/183_qreg_poly10.csv")


if __name__ == "__main__":
    np.random.seed(42)
    main()
