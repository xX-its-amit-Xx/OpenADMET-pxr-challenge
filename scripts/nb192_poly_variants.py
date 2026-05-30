"""nb192 -- Polynomial variants for QReg stacker.

nb188 best: 0.298519  ratio=0.5815
nb187 QReg poly-2 Pearson-diverse-10: 0.299246  ratio=0.5826

Hypothesis: More flexible polynomial spaces (degree-3, interaction-only) or
larger diverse model sets may find better QReg solutions than degree-2.

Experiments:
  A: QReg poly-3 on Pearson-diverse-10 (degree 3 = 285 features vs 65)
  B: QReg interaction-only poly-2 on diverse-10 (45 features vs 65)
  C: QReg poly-2 on diverse-15 set
  D: QReg poly-2 on diverse-20 set
  E: Best result -> 9-model SLSQP with nb188's 8-pool
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures

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
    "nb185_qreg_iter", "nb186_single_conc_lookup",
    "nb187_diversity_qreg", "nb188_diverse_refine", "nb189_iterate_diverse",
    "nb190_random_diverse_search", "nb191_lgbm_qstack",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
    "nb108_grand_v2", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
}

NB188_POOL = [
    "nb167_xgboost_mae",
    "nb156_catboost_mae",
    "nb154_lgbm_mae_filtered",
    "nb162_mixed_pool",
    "nb165_multiseed_162c",
    "nb149_meta_maeloss",
    "nb183_qreg_poly10",
    "nb187_diversity_qreg",
]


def load_pool(n_tr):
    oofs, tes, stems = [], [], []
    for f in sorted(DATA_PROCESSED.glob("oof_nb*.npy")):
        stem = f.stem[4:]
        if any(stem.startswith(ms) or stem == ms for ms in META_STEMS):
            continue
        for pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{pref}{stem}.npy"
            if te_p.exists():
                break
        else:
            continue
        oof = np.load(f).astype(np.float64).flatten()
        te  = np.load(te_p).astype(np.float64).flatten()
        if len(oof) != n_tr:
            continue
        oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
        te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
        oofs.append(oof); tes.append(te); stems.append(stem)
    return oofs, tes, stems


def greedy_diversity(oofs, k, seed_idx=0):
    X = np.column_stack(oofs)
    corr = np.abs(np.corrcoef(X.T))
    selected = [seed_idx]
    remaining = list(range(len(oofs)))
    remaining.remove(seed_idx)
    while len(selected) < k and remaining:
        avg_corrs = [(np.mean([corr[i, j] for j in selected]), i) for i in remaining]
        avg_corrs.sort(key=lambda x: x[0])
        selected.append(avg_corrs[0][1])
        remaining.remove(avg_corrs[0][1])
    return selected


def qreg_sweep(X_tr, y_tr, X_te, splits, label, alphas):
    n_tr = len(y_tr)
    best_r, best_alpha, best_oof, best_te, best_ratio = 1e9, None, None, None, 0
    for alpha in alphas:
        oof_s = np.full(n_tr, np.nan)
        for _, (tr_idx, va_idx) in enumerate(splits):
            m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof_s[va_idx] = m.predict(X_tr[va_idx])
        m_f = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        m_f.fit(X_tr, y_tr)
        te_s = m_f.predict(X_te)
        r_s = rae(y_tr, oof_s)
        ratio_s = te_s.std() / oof_s.std()
        flag = "PASS" if ratio_s >= COLLAPSE_THRESH else "FAIL"
        beat = " ***NEW BEST***" if (ratio_s >= COLLAPSE_THRESH and r_s < 0.298519) else ""
        print(f"  [{label}] alpha={alpha:.5f}  RAE={r_s:.6f}  ratio={ratio_s:.4f}  [{flag}]{beat}")
        if ratio_s >= COLLAPSE_THRESH and r_s < best_r:
            best_r, best_alpha = r_s, alpha
            best_oof, best_te, best_ratio = oof_s.copy(), te_s.copy(), ratio_s
    return best_r, best_oof, best_te, best_ratio, best_alpha


def slsqp_blend(pool_stems, nb192_oof, nb192_te, y_tr, n_tr, n_starts=200):
    oofs_list, tes_list, names = [], [], []
    for stem in pool_stems:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        te_p  = DATA_PROCESSED / f"te_{stem}.npy"
        if not oof_p.exists():
            print(f"  Missing {oof_p}")
            continue
        oof_m = np.load(oof_p).astype(np.float64).flatten()
        te_m  = np.load(te_p).astype(np.float64).flatten()
        oof_m = np.where(np.isfinite(oof_m), oof_m, np.nanmean(oof_m))
        te_m  = np.where(np.isfinite(te_m),  te_m,  np.nanmean(te_m))
        oofs_list.append(oof_m)
        tes_list.append(te_m)
        names.append(stem)
    oofs_list.append(nb192_oof)
    tes_list.append(nb192_te)
    names.append("nb192")

    X_tr = np.column_stack(oofs_list)
    X_te = np.column_stack(tes_list)
    n_m = len(oofs_list)
    best_r, best_w = 1e9, None

    def neg_rae(w):
        return rae(y_tr, X_tr @ w)

    def neg_rae_grad(w):
        pred = X_tr @ w
        res = pred - y_tr
        mae_mean = np.abs(y_tr - y_tr.mean()).mean()
        sign = np.sign(res)
        return X_tr.T @ sign / (n_tr * mae_mean)

    rng = np.random.default_rng(SEED)
    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(n_m))
        res = minimize(neg_rae, w0, jac=neg_rae_grad,
                       method="SLSQP",
                       bounds=[(0, 1)] * n_m,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                       options={"maxiter": 500, "ftol": 1e-10})
        if res.fun < best_r:
            best_r, best_w = res.fun, res.x

    oof_b = X_tr @ best_w
    te_b  = X_te @ best_w
    ratio = te_b.std() / oof_b.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{n_m}-model SLSQP] RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]")
    for i, nm in enumerate(names):
        if best_w[i] > 0.001:
            print(f"    {nm:50s}  w={best_w[i]:.4f}")
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb192: Polynomial variants for QReg stacker ===\n")
    print("nb188 best: 0.298519  ratio=0.5815")
    print("nb187 QReg poly-2 Pearson-diverse-10: 0.299246  ratio=0.5826\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    oofs, tes, stems = load_pool(n_tr)
    print(f"Pool: {len(stems)} pure base models")

    # Insert nb183 anchor (same as nb187)
    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists():
        oof183 = np.load(oof183_p).astype(np.float64).flatten()
        te183  = np.load(te183_p).astype(np.float64).flatten()
        oof183 = np.where(np.isfinite(oof183), oof183, np.nanmean(oof183))
        te183  = np.where(np.isfinite(te183),  te183,  np.nanmean(te183))
        oofs.insert(0, oof183); tes.insert(0, te183)
        stems.insert(0, "nb183_qreg_poly10")
        print(f"Added nb183 as anchor")

    nb183_idx = 0
    results = {}

    # Pearson-diverse-10 (same set as nb187)
    div10_idx = greedy_diversity(oofs, k=10, seed_idx=nb183_idx)
    print(f"\nPearson-diverse-10: {[stems[i] for i in div10_idx]}")
    X10_tr = np.column_stack([oofs[i] for i in div10_idx])
    X10_te = np.column_stack([tes[i] for i in div10_idx])

    # --- A: degree-3 polynomial ---
    print("\n--- A: QReg poly-3 on Pearson-diverse-10 ---")
    poly3 = PolynomialFeatures(degree=3, include_bias=False)
    X3_tr = poly3.fit_transform(X10_tr)
    X3_te = poly3.transform(X10_te)
    print(f"  Poly-3 features: {X3_tr.shape}")
    alphas_a = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
    r_a, oof_a, te_a, ratio_a, alpha_a = qreg_sweep(X3_tr, y_tr, X3_te, splits, "poly3", alphas_a)
    if oof_a is not None:
        results["A_poly3"] = (r_a, oof_a, te_a, ratio_a)
        print(f"  Best A: RAE={r_a:.6f}  ratio={ratio_a:.4f}  alpha={alpha_a}")

    # --- B: interaction-only poly-2 ---
    print("\n--- B: QReg interaction-only poly-2 on Pearson-diverse-10 ---")
    poly2_int = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X2i_tr = poly2_int.fit_transform(X10_tr)
    X2i_te = poly2_int.transform(X10_te)
    print(f"  Interaction-only poly-2 features: {X2i_tr.shape}")
    # Fewer features → can use lower alpha → potentially better RAE
    alphas_b = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]
    r_b, oof_b, te_b, ratio_b, alpha_b = qreg_sweep(X2i_tr, y_tr, X2i_te, splits, "poly2_int", alphas_b)
    if oof_b is not None:
        results["B_int"] = (r_b, oof_b, te_b, ratio_b)
        print(f"  Best B: RAE={r_b:.6f}  ratio={ratio_b:.4f}  alpha={alpha_b}")

    # --- C: poly-2 on diverse-15 ---
    print("\n--- C: QReg poly-2 on Pearson-diverse-15 ---")
    div15_idx = greedy_diversity(oofs, k=15, seed_idx=nb183_idx)
    print(f"  Last 5: {[stems[i] for i in div15_idx[10:]]}")
    X15_tr = np.column_stack([oofs[i] for i in div15_idx])
    X15_te = np.column_stack([tes[i] for i in div15_idx])
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X15p2_tr = poly2.fit_transform(X15_tr)
    X15p2_te = poly2.transform(X15_te)
    print(f"  Poly-2 features: {X15p2_tr.shape}")
    alphas_c = [0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]
    r_c, oof_c, te_c, ratio_c, alpha_c = qreg_sweep(X15p2_tr, y_tr, X15p2_te, splits, "div15_poly2", alphas_c)
    if oof_c is not None:
        results["C_div15"] = (r_c, oof_c, te_c, ratio_c)
        print(f"  Best C: RAE={r_c:.6f}  ratio={ratio_c:.4f}  alpha={alpha_c}")

    # --- D: poly-2 on diverse-20 ---
    print("\n--- D: QReg poly-2 on Pearson-diverse-20 ---")
    div20_idx = greedy_diversity(oofs, k=20, seed_idx=nb183_idx)
    print(f"  Last 5: {[stems[i] for i in div20_idx[15:]]}")
    X20_tr = np.column_stack([oofs[i] for i in div20_idx])
    X20_te = np.column_stack([tes[i] for i in div20_idx])
    X20p2_tr = poly2.fit_transform(X20_tr)
    X20p2_te = poly2.transform(X20_te)
    print(f"  Poly-2 features: {X20p2_tr.shape}")
    alphas_d = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    r_d, oof_d, te_d, ratio_d, alpha_d = qreg_sweep(X20p2_tr, y_tr, X20p2_te, splits, "div20_poly2", alphas_d)
    if oof_d is not None:
        results["D_div20"] = (r_d, oof_d, te_d, ratio_d)
        print(f"  Best D: RAE={r_d:.6f}  ratio={ratio_d:.4f}  alpha={alpha_d}")

    # --- Summary ---
    print("\n=== Summary ===")
    best_oof, best_te, best_rae = None, None, 1e9
    for k_str, (r, oof_s, te_s, ratio) in results.items():
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k_str}: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]")
        if ratio >= COLLAPSE_THRESH and r < best_rae:
            best_rae, best_oof, best_te = r, oof_s, te_s

    if best_oof is None:
        print("\nNo result improved over nb188 (0.298519). No file saved.")
        print("Conclusion: polynomial space exhausted; consider new base models.")
        return

    print(f"\nBest: RAE={best_rae:.6f}")
    np.save(DATA_PROCESSED / "oof_nb192_poly_variant.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb192_poly_variant.npy",  best_te)

    # --- E: 9-model SLSQP ---
    print("\n--- E: 9-model SLSQP (nb188 pool + nb192) with 200 starts ---")
    r_slsqp, oof_slsqp, te_slsqp, ratio_slsqp = slsqp_blend(
        NB188_POOL, best_oof, best_te, y_tr, n_tr, n_starts=200)

    if ratio_slsqp >= COLLAPSE_THRESH and r_slsqp < 0.298519:
        final_oof, final_te, final_rae = oof_slsqp, te_slsqp, r_slsqp
        print(f"  SLSQP improves: {r_slsqp:.6f}  ***NEW BEST***")
    elif ratio_slsqp >= COLLAPSE_THRESH:
        final_oof, final_te, final_rae = oof_slsqp, te_slsqp, r_slsqp
        print(f"  SLSQP: {r_slsqp:.6f} (no improvement over 0.298519)")
    else:
        final_oof, final_te, final_rae = best_oof, best_te, best_rae
        print(f"  SLSQP collapsed, using stacker alone: {best_rae:.6f}")

    sub = pd.DataFrame({"Molecule Name": te_df["molecule_name"], "pEC50": final_te})
    out_path = SUBMISSIONS / "192_poly_variant.csv"
    sub.to_csv(out_path, index=False)
    np.save(DATA_PROCESSED / "oof_nb192_final.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb192_final.npy",  final_te)
    print(f"\nSaved: {out_path}  (RAE={final_rae:.6f})")

    print("\n=== Final comparison ===")
    print(f"  nb188 (prev best): 0.298519")
    print(f"  nb192 best:        {final_rae:.6f}")


if __name__ == "__main__":
    main()
