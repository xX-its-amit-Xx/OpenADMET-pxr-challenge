"""nb189 -- Iterate diverse QReg poly: anchor at nb187, level-3 stacking.

nb188 results:
  8-model SLSQP (6 orig + nb183 + nb187): RAE=0.298519  ratio=0.5815  PASS  NEW BEST
  Weights: nb187=0.6319, nb167=0.1422, nb156=0.0926, nb165=0.059, nb154=0.0531
           nb149=0.0209, nb162=0.0003, nb183=0.0 (zero!)

Key insight: nb187 dominates, nb183 gets zero weight (nb187 subsumes nb183).
Next: create nb189 = QReg poly diverse-10 anchored at nb187.

Tests:
  A: QReg poly diverse-10 seeded at nb187 (level-3 meta-learner)
  B: 9-model SLSQP: 8 models + nb189
  C: Also try 7-model SLSQP: [nb187, nb6 original] (confirm nb183=0 in 8-model)
  D: Fine-tune nb189 alpha
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
    "nb187_diversity_qreg", "nb188_diverse_refine",
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


def qreg_sweep(X_tr, y_tr, X_te, splits, label, alphas):
    n_tr = len(y_tr)
    best_r, best_alpha, best_oof, best_te = 1e9, None, None, None
    for alpha in alphas:
        oof = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof[va_idx] = m.predict(X_tr[va_idx])
        m_f = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        m_f.fit(X_tr, y_tr)
        te = m_f.predict(X_te)
        r = rae(y_tr, oof); ratio = te.std()/oof.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        beat = " ***BEATS 0.298519***" if (ratio >= COLLAPSE_THRESH and r < 0.298519) else ""
        print(f"  [{label}] alpha={alpha:.5f}  RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
        if ratio >= COLLAPSE_THRESH and r < best_r:
            best_r, best_alpha, best_oof, best_te = r, alpha, oof, te
    return best_r, best_oof, best_te, best_alpha


def slsqp_multistart(X_tr, y_tr, X_te, n_starts, label, model_names=None):
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
    beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and best_r < 0.298519) else ""
    print(f"  [{label}] RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
    if model_names:
        for name, w in zip(model_names, best_w):
            if w > 0.001:
                print(f"    {name}: {w:.4f}")
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb189: Iterate diverse QReg poly (anchor=nb187) ===\n")
    print("nb188 best: 8-model SLSQP = 0.298519  ratio=0.5815")
    print("Goal: beat 0.298519\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load pool (excludes meta-stems)
    oofs, tes, stems = load_pool(n_tr)
    print(f"Pool: {len(stems)} pure base models")

    # Load nb183 and nb187 (manually add to pool)
    oof183, te183 = load_oof("nb183_qreg_poly10", n_tr)
    oof187, te187 = load_oof("nb187_diversity_qreg", n_tr)

    if oof183 is None or oof187 is None:
        print("ERROR: nb183 or nb187 not found!")
        return

    r187 = rae(y_tr, oof187)
    print(f"nb183: RAE={rae(y_tr, oof183):.4f}")
    print(f"nb187: RAE={r187:.4f}")

    # Build pool with nb187 as anchor (insert at position 0)
    oofs_with187 = [oof187] + oofs
    tes_with187  = [te187]  + tes
    stems_with187 = ["nb187_diversity_qreg"] + stems

    # Load 6 original models
    oofs6, tes6, names6 = [], [], []
    for stem in SIX_MODELS:
        oof, te = load_oof(stem, n_tr)
        if oof is not None:
            oofs6.append(oof); tes6.append(te); names6.append(stem)
    X6_tr = np.column_stack(oofs6); X6_te = np.column_stack(tes6)

    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    results = {}

    # --- A: QReg poly diverse-10 seeded at nb187 ---
    print("\n--- A: QReg poly diverse-10 seeded at nb187 ---")
    div10_idx_187 = greedy_diversity_select(oofs_with187, k=10, seed_idx=0)
    print(f"  Selected: {[stems_with187[i] for i in div10_idx_187]}")
    X_d10_tr = np.column_stack([oofs_with187[i] for i in div10_idx_187])
    X_d10_te = np.column_stack([tes_with187[i] for i in div10_idx_187])
    X_d10p2_tr = poly2.fit_transform(X_d10_tr)
    X_d10p2_te = poly2.transform(X_d10_te)
    print(f"  Poly features: {X_d10p2_tr.shape}")
    alphas_a = [0.002, 0.0025, 0.003, 0.004, 0.005, 0.007, 0.01]
    r_a, oof_a, te_a, alpha_a = qreg_sweep(X_d10p2_tr, y_tr, X_d10p2_te, splits, "nb189_div10", alphas_a)
    if oof_a is not None:
        results["A_nb189"] = (r_a, oof_a, te_a, te_a.std()/oof_a.std())
        print(f"  Best: alpha={alpha_a}  RAE={r_a:.6f}")

    # --- B: 7-model SLSQP: [6 orig + nb187] (confirm nb183=0 effect) ---
    print("\n--- B: 7-model SLSQP [6 orig + nb187] (200 starts) ---")
    X7_tr = np.column_stack([X6_tr, oof187.reshape(-1,1)])
    X7_te = np.column_stack([X6_te, te187.reshape(-1,1)])
    r_b, oof_b, te_b, ratio_b = slsqp_multistart(
        X7_tr, y_tr, X7_te, 200, "7model_6orig+nb187",
        model_names=names6 + ["nb187"]
    )
    if ratio_b >= COLLAPSE_THRESH:
        results["B_7model_nb187"] = (r_b, oof_b, te_b, ratio_b)

    # --- C: 9-model SLSQP: 8 models + nb189 ---
    if oof_a is not None:
        print(f"\n--- C: 9-model SLSQP: 8 models + nb189 (RAE={r_a:.4f}) (500 starts) ---")
        oofs_8 = oofs6 + [oof183, oof187]
        tes_8  = tes6  + [te183, te187]
        names_8 = names6 + ["nb183", "nb187"]
        X9_tr = np.column_stack(oofs_8 + [oof_a.reshape(-1,1)])
        X9_te = np.column_stack(tes_8  + [te_a.reshape(-1,1)])
        r_c, oof_c, te_c, ratio_c = slsqp_multistart(
            X9_tr, y_tr, X9_te, 500, "9model_8+nb189",
            model_names=names_8 + ["nb189"]
        )
        if ratio_c >= COLLAPSE_THRESH:
            results["C_9model"] = (r_c, oof_c, te_c, ratio_c)

    # --- D: Multi-diverse: pool with nb183, nb187 both as candidates ---
    print(f"\n--- D: Multi-candidate pool: diverse-10 from [nb183, nb187, pool] (200 starts) ---")
    oofs_full = [oof183, oof187] + oofs
    tes_full  = [te183, te187]   + tes
    stems_full = ["nb183", "nb187"] + stems
    # Greedy select 10 starting from nb187 (index 1)
    div10_full = greedy_diversity_select(oofs_full, k=10, seed_idx=1)
    print(f"  Diverse-10: {[stems_full[i] for i in div10_full]}")
    X_df_tr = np.column_stack([oofs_full[i] for i in div10_full])
    X_df_te = np.column_stack([tes_full[i] for i in div10_full])
    X_dfp2_tr = poly2.fit_transform(X_df_tr)
    X_dfp2_te = poly2.transform(X_df_te)
    alphas_d = [0.002, 0.003, 0.004, 0.005, 0.007, 0.01]
    r_d, oof_d, te_d, alpha_d = qreg_sweep(X_dfp2_tr, y_tr, X_dfp2_te, splits, "div10_multi", alphas_d)
    if oof_d is not None:
        results["D_div10_multi"] = (r_d, oof_d, te_d, te_d.std()/oof_d.std())

    # === Summary ===
    print(f"\n=== Summary ===")
    print(f"  nb188 8-model SLSQP:   RAE=0.298519  ratio=0.5815  PASS  (PREV BEST)")
    clean = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    for k, (r, _, _, ratio) in sorted(clean.items(), key=lambda x: x[1][0]):
        beat = " ***NEW BEST***" if r < 0.298519 else ""
        print(f"  {k:30s}  RAE={r:.6f}  ratio={ratio:.4f}  PASS  {beat}")

    if not clean:
        print("  nb188 (0.298519) still best.")
        return

    best_r, best_oof, best_te, best_ratio = min(clean.values(), key=lambda x: x[0])
    if best_r >= 0.298519:
        print(f"\nnb188 (0.298519) still best.")
        return

    best_te_clip = np.clip(best_te, y_tr.min()-0.5, y_tr.max()+0.5)
    np.save(DATA_PROCESSED / "oof_nb189_iterate_diverse.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb189_iterate_diverse.npy",  best_te_clip)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_clip})
    sub.to_csv(SUBMISSIONS / "189_iterate_diverse.csv", index=False)
    print(f"\nSaved: submissions/189_iterate_diverse.csv  OOF RAE={best_r:.6f}")
    print(f"*** NEW BEST: {best_r:.6f} ***")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
