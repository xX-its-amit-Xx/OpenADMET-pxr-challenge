"""nb188 -- Refine diverse QReg poly: fine alpha sweep + SLSQP expansion.

nb187 breakthrough:
  A: diverse-10 (nb183 + greedy diverse)  RAE=0.299246  ratio=0.5826  PASS  NEW BEST
  D: hybrid-10 (5 RAE + 5 diverse)        RAE=0.299294  ratio=0.5830  PASS  NEW BEST

This script:
  A: Fine alpha sweep around 0.003 for diverse-10
  B: Diverse-12 and diverse-8 variants
  C: 8-model SLSQP: 6 original + nb183 + nb187 (diverse)
  D: SLSQP over [6 original, nb183, nb187, nb187_diverse12]
  E: QReg poly on diverse-20 (bigger feature space)
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

META_STEMS = {
    "nb136_xgb_meta", "nb138_elnet_blend", "nb139_adaptive_blend",
    "nb140_xgb_lgbm_meta", "nb141_xgb_ablation", "nb142_xgb_calibrated",
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb146_pca_oof_meta", "nb147_oofrdkit_meta", "nb148_meta_disagreement",
    "nb149_meta_maeloss", "nb150_residual_ensemble", "nb151_grand_v11",
    "nb152_lgbm_mae_tuned", "nb153_grand_v12", "nb154_lgbm_mae_filtered",
    "nb155_grand_v13", "nb156_catboost_mae", "nb157_optuna_lgbm_mae",
    "nb158_collapse_fix", "nb159_variance_scaled_ensemble", "nb160_pca_meta",
    "nb161_neural_meta", "nb162_mixed_pool", "nb163_lgbm_colsample_low",
    "nb163_lgbm_colsample",
    "nb164_grand_v14", "nb165_multiseed_162c", "nb166_catboost_v2",
    "nb167_xgboost_mae",
    "nb168_multiseed_catboost", "nb169_rf_et_mae",
    "nb170_grand_v15", "nb171_catboost_extended",
    "nb172_bootstrap_ensemble", "nb173_softmax_sweep",
    "nb174_top10_lgbm", "nb175_bayes_blend", "nb176_optuna_weights",
    "nb177_xgb_histgb", "nb178_xgb_10seed", "nb179_xgb_collapse_fix",
    "nb180_nonlinear_6model", "nb181_quantile_poly",
    "nb182_qreg_alpha", "nb183_qreg_poly10", "nb184_grand_v16",
    "nb185_qreg_iter", "nb186_sc_blend",
    "nb187_diversity_qreg",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
    "nb108_grand_v2", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
}

SIX_MODELS = [
    "nb167_xgboost_mae", "nb156_catboost_mae", "nb154_lgbm_mae_filtered",
    "nb162_mixed_pool", "nb165_multiseed_162c", "nb149_meta_maeloss",
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


def load_pool(n_tr):
    oofs, tes, stems = [], [], []
    for f in sorted(DATA_PROCESSED.glob("oof_nb*.npy")):
        stem = f.stem[4:]
        if any(stem.startswith(ms) or stem == ms for ms in META_STEMS):
            continue
        oof, te = load_oof(stem, n_tr)
        if oof is None:
            continue
        oofs.append(oof); tes.append(te); stems.append(stem)
    return oofs, tes, stems


def greedy_diversity_select(oofs, k, seed_idx=0):
    corr = np.corrcoef(np.column_stack(oofs).T)
    selected = [seed_idx]
    remaining = list(range(len(oofs)))
    remaining.remove(seed_idx)
    while len(selected) < k and remaining:
        avg_corrs = [(np.mean([abs(corr[i, j]) for j in selected]), i) for i in remaining]
        avg_corrs.sort(key=lambda x: x[0])
        best_i = avg_corrs[0][1]
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


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
    r = rae(y_tr, oof); ratio = te_pred.std()/oof.std()
    return r, oof, te_pred, ratio


def qreg_sweep(X_tr, y_tr, X_te, splits, label, alphas):
    best_r, best_alpha, best_oof, best_te = 1e9, None, None, None
    for alpha in alphas:
        r, oof, te, ratio = qreg_cv(X_tr, y_tr, X_te, splits, alpha)
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and r < 0.299246) else ""
        print(f"  [{label}] alpha={alpha:.5f}  RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
        if ratio >= COLLAPSE_THRESH and r < best_r:
            best_r, best_alpha, best_oof, best_te = r, alpha, oof, te
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
    beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and best_r < 0.299246) else ""
    print(f"  [{label}] RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
    print(f"    weights: {dict(zip([f'w{i}' for i in range(k)], best_w.round(4)))}")
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb188: Refine diverse QReg poly ===\n")
    print("Reference: nb187 diverse-10 = 0.299246  ratio=0.5826")
    print("Reference: nb184 7-model SLSQP = 0.299711  ratio=0.5801\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load pool
    oofs, tes, stems = load_pool(n_tr)
    raes = [rae(y_tr, oof) for oof in oofs]
    print(f"Pool: {len(stems)} pure base models")

    # Add nb183 as pool anchor
    oof183, te183 = load_oof("nb183_qreg_poly10", n_tr)
    if oof183 is not None:
        oofs.insert(0, oof183); tes.insert(0, te183)
        stems.insert(0, "nb183_qreg_poly10"); raes.insert(0, rae(y_tr, oof183))
    nb183_idx = 0

    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    results = {}

    # Get diverse-10 indices (same as nb187)
    div10_idx = greedy_diversity_select(oofs, k=10, seed_idx=nb183_idx)
    print(f"Diverse-10: {[stems[i] for i in div10_idx]}")
    X_div10_tr = np.column_stack([oofs[i] for i in div10_idx])
    X_div10_te = np.column_stack([tes[i] for i in div10_idx])
    X_div10p2_tr = poly2.fit_transform(X_div10_tr)
    X_div10p2_te = poly2.transform(X_div10_te)
    print(f"Poly features: {X_div10p2_tr.shape}\n")

    # --- A: Fine alpha sweep for diverse-10 ---
    print("--- A: Fine alpha sweep around 0.003 for diverse-10 ---")
    alphas_a = [0.0020, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.006, 0.007]
    r_a, oof_a, te_a, alpha_a = qreg_sweep(X_div10p2_tr, y_tr, X_div10p2_te, splits, "div10_fine", alphas_a)
    if oof_a is not None:
        results["A_div10_fine"] = (r_a, oof_a, te_a, te_a.std()/oof_a.std())

    # --- B: Diverse-8 and diverse-12 ---
    print("\n--- B: Diverse-8 (fewer models) ---")
    div8_idx = greedy_diversity_select(oofs, k=8, seed_idx=nb183_idx)
    X_div8_tr = np.column_stack([oofs[i] for i in div8_idx])
    X_div8_te = np.column_stack([tes[i] for i in div8_idx])
    X_div8p2_tr = poly2.fit_transform(X_div8_tr)
    X_div8p2_te = poly2.transform(X_div8_te)
    print(f"  Poly features: {X_div8p2_tr.shape}")
    alphas_b8 = [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01]
    r_b8, oof_b8, te_b8, alpha_b8 = qreg_sweep(X_div8p2_tr, y_tr, X_div8p2_te, splits, "div8", alphas_b8)
    if oof_b8 is not None:
        results["B_div8"] = (r_b8, oof_b8, te_b8, te_b8.std()/oof_b8.std())

    print("\n--- B2: Diverse-12 ---")
    div12_idx = greedy_diversity_select(oofs, k=12, seed_idx=nb183_idx)
    X_div12_tr = np.column_stack([oofs[i] for i in div12_idx])
    X_div12_te = np.column_stack([tes[i] for i in div12_idx])
    X_div12p2_tr = poly2.fit_transform(X_div12_tr)
    X_div12p2_te = poly2.transform(X_div12_te)
    print(f"  Poly features: {X_div12p2_tr.shape}")
    alphas_b12 = [0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.015]
    r_b12, oof_b12, te_b12, alpha_b12 = qreg_sweep(X_div12p2_tr, y_tr, X_div12p2_te, splits, "div12", alphas_b12)
    if oof_b12 is not None:
        results["B_div12"] = (r_b12, oof_b12, te_b12, te_b12.std()/oof_b12.std())

    # Save nb187 result (diverse-10 at alpha=0.003) — ensure it's saved properly
    best_div10_r, best_div10_oof, best_div10_te, _ = qreg_cv(
        X_div10p2_tr, y_tr, X_div10p2_te, splits, 0.003
    )
    np.save(DATA_PROCESSED / "oof_nb187_diversity_qreg.npy", best_div10_oof)
    np.save(DATA_PROCESSED / "te_nb187_diversity_qreg.npy",  best_div10_te)
    print(f"\nnb187 confirmed: RAE={best_div10_r:.6f}  ratio={best_div10_te.std()/best_div10_oof.std():.4f}")

    # --- C: 8-model SLSQP: 6 original + nb183 + nb187 ---
    print("\n--- C: 8-model SLSQP: 6 original + nb183 + nb187 (500 starts) ---")
    oofs6, tes6 = [], []
    for stem in SIX_MODELS:
        oof, te = load_oof(stem, n_tr)
        if oof is not None:
            oofs6.append(oof); tes6.append(te)
    X6_tr = np.column_stack(oofs6); X6_te = np.column_stack(tes6)

    oof187 = best_div10_oof; te187 = best_div10_te
    X8_tr = np.column_stack([X6_tr, oof183.reshape(-1,1), oof187.reshape(-1,1)])
    X8_te = np.column_stack([X6_te, te183.reshape(-1,1), te187.reshape(-1,1)])
    r_c, oof_c, te_c, ratio_c = slsqp_multistart(X8_tr, y_tr, X8_te, 500, "8model_SLSQP")
    if ratio_c >= COLLAPSE_THRESH:
        results["C_8model"] = (r_c, oof_c, te_c, ratio_c)

    # --- D: SLSQP over [nb183, nb187, 6-model linear blend] = 3 models ---
    print("\n--- D: 3-model SLSQP: [nb183, nb187, 6-linear-blend] (200 starts) ---")
    # Linear blend (treated as a model)
    def obj6(w): return np.mean(np.abs(y_tr - X6_tr@w))/np.mean(np.abs(y_tr-y_tr.mean()))
    res6 = minimize(obj6, np.ones(6)/6, method='SLSQP',
                    bounds=[(0,1)]*6,
                    constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                    options={'ftol':1e-10})
    oof6_lin = X6_tr@res6.x; te6_lin = X6_te@res6.x

    X3_tr = np.column_stack([oof183, oof187, oof6_lin])
    X3_te = np.column_stack([te183, te187, te6_lin])
    r_d, oof_d, te_d, ratio_d = slsqp_multistart(X3_tr, y_tr, X3_te, 200, "3model_nb183+nb187+lin")
    if ratio_d >= COLLAPSE_THRESH:
        results["D_3model"] = (r_d, oof_d, te_d, ratio_d)

    # === Summary ===
    print(f"\n=== Summary ===")
    print(f"  nb183 standalone:        RAE=0.300023  ratio=0.580")
    print(f"  nb184 7-model SLSQP:     RAE=0.299711  ratio=0.5801")
    print(f"  nb187 diverse-10:        RAE=0.299246  ratio=0.5826  PASS  ***PREV BEST***")
    clean = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    for k, (r, _, _, ratio) in sorted(clean.items(), key=lambda x: x[1][0]):
        beat = " ***NEW BEST***" if r < 0.299246 else ""
        print(f"  {k:30s}  RAE={r:.6f}  ratio={ratio:.4f}  PASS  {beat}")

    if not clean:
        print("  nb187 (0.299246) still best.")
        return

    best_r, best_oof, best_te, best_ratio = min(clean.values(), key=lambda x: x[0])
    if best_r >= 0.299246:
        print(f"\nnb187 (0.299246) still best — no further improvement.")
        return

    best_te_clip = np.clip(best_te, y_tr.min()-0.5, y_tr.max()+0.5)
    np.save(DATA_PROCESSED / "oof_nb188_diverse_refine.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb188_diverse_refine.npy",  best_te_clip)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_clip})
    sub.to_csv(SUBMISSIONS / "188_diverse_refine.csv", index=False)
    print(f"\nSaved: submissions/188_diverse_refine.csv  OOF RAE={best_r:.6f}")
    print(f"*** NEW BEST: {best_r:.6f} ***")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
