"""nb182 -- Fine-tune QReg poly-10 alpha to cross ratio=0.580 threshold.

nb181 experiment C: QReg(0.5, alpha=0.001) on degree-2 poly of top-10 models
  RAE=0.3000  ratio=0.579  FAIL  (just 0.001 below threshold)

Key observation: higher alpha -> higher ratio, worse RAE.
This experiment sweeps alpha finely to find where ratio crosses 0.580
while RAE is still < 0.3001.

Also: test blending C predictions with linear blend to get composite ratio >= 0.580.
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


def qreg_cv(X_tr, y_tr, X_te, splits, alpha):
    n_tr = len(y_tr)
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx])
    m_f = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
    m_f.fit(X_tr, y_tr)
    te_pred = m_f.predict(X_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    return r, oof, te_pred, ratio


def main():
    print("=== nb182: QReg poly-10 alpha sweep -- find ratio=0.580 crossover ===\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load models
    X10_tr, X10_te, loaded10 = load_models(TOP10_MODELS, n_tr)
    X6_tr,  X6_te,  loaded6  = load_models(SIX_MODELS, n_tr)
    print(f"Top-10 models loaded: {len(loaded10)}")

    # Poly features of top-10
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
    print(f"Linear SLSQP reference: RAE={r_lin:.6f}  ratio={ratio_lin:.3f}\n")

    # --- A: Fine alpha sweep for QReg poly-10 ---
    print("--- A: QReg poly-10 fine alpha sweep ---")
    alphas = [0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.002, 0.003,
              0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]
    results_a = []
    best_pass_a = None
    for alpha in alphas:
        r, oof, te, ratio = qreg_cv(X10p2_tr, y_tr, X10p2_te, splits, alpha)
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        beat = " *** BEATS 0.3001! ***" if (ratio >= COLLAPSE_THRESH and r < 0.3001) else ""
        print(f"  alpha={alpha:.4f}  RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
        results_a.append((alpha, r, oof, te, ratio))
        if ratio >= COLLAPSE_THRESH and (best_pass_a is None or r < best_pass_a[1]):
            best_pass_a = (alpha, r, oof, te, ratio)

    # --- B: QReg poly-9 (drop nb169_rf_et_mae, most different algorithm) ---
    print("\n--- B: QReg poly-9 (drop nb169_rf_et_mae) ---")
    top9 = [m for m in TOP10_MODELS if m != "nb169_rf_et_mae"]
    X9_tr, X9_te, _ = load_models(top9, n_tr)
    X9p2_tr = poly2.fit_transform(X9_tr)
    X9p2_te = poly2.transform(X9_te)
    print(f"  Poly features: {X9p2_tr.shape}")
    for alpha in [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]:
        r, oof, te, ratio = qreg_cv(X9p2_tr, y_tr, X9p2_te, splits, alpha)
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        beat = " *** BEATS 0.3001! ***" if (ratio >= COLLAPSE_THRESH and r < 0.3001) else ""
        print(f"  [poly9] alpha={alpha:.4f}  RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")

    # --- C: QReg poly-8 (drop nb169 + nb171) ---
    print("\n--- C: QReg poly-8 (drop nb169, nb171) ---")
    top8 = [m for m in TOP10_MODELS if m not in ("nb169_rf_et_mae", "nb171_catboost_extended")]
    X8_tr, X8_te, _ = load_models(top8, n_tr)
    X8p2_tr = poly2.fit_transform(X8_tr)
    X8p2_te = poly2.transform(X8_te)
    print(f"  Poly features: {X8p2_tr.shape}")
    for alpha in [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]:
        r, oof, te, ratio = qreg_cv(X8p2_tr, y_tr, X8p2_te, splits, alpha)
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        beat = " *** BEATS 0.3001! ***" if (ratio >= COLLAPSE_THRESH and r < 0.3001) else ""
        print(f"  [poly8] alpha={alpha:.4f}  RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")

    # --- D: Blend QReg-poly-10 (best from A) with linear SLSQP ---
    print("\n--- D: Blend QReg-poly-10 (best failing) with linear SLSQP ---")
    # Use alpha=0.001 result (RAE=0.3000, ratio=0.579) from nb181
    r_c, oof_c, te_c, ratio_c = qreg_cv(X10p2_tr, y_tr, X10p2_te, splits, 0.001)
    print(f"  QReg-poly-10 alpha=0.001: RAE={r_c:.6f}  ratio={ratio_c:.4f}")
    best_d = None
    for alpha_blend in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        oof_d = alpha_blend * te_lin + (1-alpha_blend) * oof_c
        # Wait - should blend OOFs with OOFs and TEs with TEs
        oof_blend = alpha_blend * oof_lin + (1-alpha_blend) * oof_c
        te_blend  = alpha_blend * te_lin  + (1-alpha_blend) * te_c
        r_d = rae(y_tr, oof_blend); ratio_d = te_blend.std()/oof_blend.std()
        flag = "PASS" if ratio_d >= COLLAPSE_THRESH else "FAIL"
        beat = " *** BEATS 0.3001! ***" if (ratio_d >= COLLAPSE_THRESH and r_d < 0.3001) else ""
        print(f"  lin_alpha={alpha_blend:.1f}  RAE={r_d:.6f}  ratio={ratio_d:.4f}  [{flag}]{beat}")
        if ratio_d >= COLLAPSE_THRESH and (best_d is None or r_d < best_d[0]):
            best_d = (r_d, oof_blend, te_blend, ratio_d)

    # === Summary ===
    print(f"\n=== Summary ===")
    print(f"  Linear reference:  RAE=0.300101  ratio=0.580  PASS")

    best_result = None
    if best_pass_a is not None:
        alpha_a, r_a, oof_a, te_a, ratio_a = best_pass_a
        print(f"  Best QReg-poly-10: alpha={alpha_a}  RAE={r_a:.6f}  ratio={ratio_a:.4f}  PASS")
        if r_a < 0.3001:
            print("  *** BEATS LINEAR BLEND! ***")
            best_result = (r_a, oof_a, te_a, ratio_a)

    if best_d is not None:
        r_d, oof_d, te_d, ratio_d = best_d
        print(f"  Best blend:        RAE={r_d:.6f}  ratio={ratio_d:.4f}  PASS")
        if r_d < 0.3001 and (best_result is None or r_d < best_result[0]):
            print("  *** BEATS LINEAR BLEND! ***")
            best_result = (r_d, oof_d, te_d, ratio_d)

    if best_result is None:
        print("  No improvement over 0.3001")
        return

    best_r, best_oof, best_te, best_ratio = best_result
    best_te_clip = np.clip(best_te, y_tr.min()-0.5, y_tr.max()+0.5)
    np.save(DATA_PROCESSED / "oof_nb182_qreg_alpha.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb182_qreg_alpha.npy",  best_te_clip)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_clip})
    sub.to_csv(SUBMISSIONS / "182_qreg_alpha.csv", index=False)
    print(f"\nSaved: submissions/182_qreg_alpha.csv  OOF RAE={best_r:.6f}")


if __name__ == "__main__":
    main()
