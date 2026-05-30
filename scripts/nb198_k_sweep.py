"""nb198 -- k-sweep for diverse model sets + random seeds.

nb197 best: 0.297639  ratio=0.5800  (91-model constrained SLSQP)

Key finding: div15(alpha=0.00171) = RAE=0.297249 is the best individual OOF.
Div15 k=15 appears to be the sweet spot.

Approach:
  A: Fine k-sweep: div-k for k in [10,11,12,13,14,15,16,17,18,19,20]
     at optimal alpha for each k → find best k
  B: 5 random diversity seeds for k=15 (different starting models)
  C: 2000-start constrained SLSQP on best k + best seeds combined pool
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
    "nb192_poly_variant", "nb193_div_fine",
    "nb194_constrained", "nb195_expanded", "nb196_fine_div15",
    "nb197_dense_grid",
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


def qreg_oof(X_tr, y_tr, X_te, splits, alpha):
    n_tr = len(y_tr)
    oof_s = np.full(n_tr, np.nan)
    for _, (tr_idx, va_idx) in enumerate(splits):
        m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_s[va_idx] = m.predict(X_tr[va_idx])
    m_f = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
    m_f.fit(X_tr, y_tr)
    return oof_s, m_f.predict(X_te)


def find_best_alpha_for_k(base_oofs, base_tes, y_tr, splits, poly2, k, seed_idx=0, n_alpha=15):
    """Find alpha that minimizes OOF RAE for div-k."""
    div_idx = greedy_diversity(base_oofs, k=k, seed_idx=seed_idx)
    Xk_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div_idx]))
    Xk_te = poly2.transform(np.column_stack([base_tes[i] for i in div_idx]))
    # Coarse sweep first
    alphas_coarse = np.logspace(-4, -2, 20)
    best_r, best_alpha = 1e9, None
    for alpha in alphas_coarse:
        oof_c, te_c = qreg_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
        r_c = rae(y_tr, oof_c)
        if r_c < best_r:
            best_r, best_alpha = r_c, alpha
    # Fine sweep around best
    alphas_fine = np.linspace(max(best_alpha*0.5, 1e-5), min(best_alpha*2, 0.05), n_alpha)
    for alpha in alphas_fine:
        oof_c, te_c = qreg_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
        r_c = rae(y_tr, oof_c)
        ratio_c = te_c.std() / oof_c.std()
        if r_c < best_r:
            best_r, best_alpha = r_c, alpha
    return best_r, best_alpha, Xk_tr, Xk_te


def constrained_slsqp(X_tr, X_te, y_tr, n_starts=1500, prev_best=0.297639):
    n_m = X_tr.shape[1]
    n_tr = len(y_tr)
    best_r, best_w = 1e9, None

    def obj(w): return rae(y_tr, X_tr @ w)
    def obj_grad(w):
        pred = X_tr @ w
        mae_mean = np.abs(y_tr - y_tr.mean()).mean()
        return X_tr.T @ np.sign(pred - y_tr) / (n_tr * mae_mean)
    def ratio_con(w):
        pred_tr = X_tr @ w; pred_te = X_te @ w
        std_tr = pred_tr.std()
        if std_tr < 1e-9: return -1.0
        return pred_te.std() / std_tr - COLLAPSE_THRESH

    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1},
        {"type": "ineq", "fun": ratio_con},
    ]
    bounds = [(0, 1)] * n_m
    rng = np.random.default_rng(SEED)

    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(n_m))
        res = minimize(obj, w0, jac=obj_grad, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-11})
        if res.success and ratio_con(res.x) >= -1e-6:
            r = obj(res.x)
            if r < best_r:
                best_r, best_w = r, res.x.copy()

    if best_w is None:
        return 1e9, None, None, 0

    oof_b = X_tr @ best_w; te_b = X_te @ best_w
    ratio = te_b.std() / oof_b.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and best_r < prev_best) else ""
    print(f"  n={n_m}  RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
    top_idx = np.argsort(best_w)[::-1]
    for i in top_idx[:8]:
        if best_w[i] > 0.01:
            print(f"    model_{i}  w={best_w[i]:.4f}")
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb198: k-sweep for diverse sets + random seeds ===\n")
    print("nb197 best: 0.297639  ratio=0.5800")
    print("div15(0.00171) = 0.297249 is best individual OOF\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    base_oofs, base_tes, base_stems = load_pool(n_tr)
    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists():
        oof183 = np.load(oof183_p).astype(np.float64).flatten()
        te183  = np.load(te183_p).astype(np.float64).flatten()
        oof183 = np.where(np.isfinite(oof183), oof183, np.nanmean(oof183))
        te183  = np.where(np.isfinite(te183),  te183,  np.nanmean(te183))
        base_oofs.insert(0, oof183); base_tes.insert(0, te183)
        base_stems.insert(0, "nb183_qreg_poly10")
    print(f"Base pool: {len(base_oofs)} models (seed=nb183 at idx 0)")

    poly2 = PolynomialFeatures(degree=2, include_bias=False)

    # --- A: k-sweep ---
    print("--- A: k-sweep: find best k (optimal alpha for each) ---")
    k_values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    k_results = {}
    for k in k_values:
        best_r_k, best_alpha_k, _, _ = find_best_alpha_for_k(
            base_oofs, base_tes, y_tr, splits, poly2, k, seed_idx=0)
        print(f"  k={k:2d}  best_alpha={best_alpha_k:.5f}  best_RAE={best_r_k:.6f}")
        k_results[k] = (best_r_k, best_alpha_k)

    best_k = min(k_results, key=lambda k: k_results[k][0])
    print(f"\nBest k={best_k} with RAE={k_results[best_k][0]:.6f}")

    # --- B: Random seeds for k=best_k and k=15 ---
    print(f"\n--- B: Random seeds for k={best_k} and k=15 ---")
    seed_indices = [0]  # nb183 is the default
    # Add a few random seeds from the base pool
    rng_seed = np.random.default_rng(SEED)
    extra_seeds = rng_seed.choice(len(base_oofs), size=5, replace=False).tolist()
    seed_indices.extend(extra_seeds)
    print(f"  Seeds to try: {seed_indices}")

    all_cands_oofs, all_cands_tes = [], []

    for k in sorted({best_k, 15}):
        best_alpha_k = k_results.get(k, (None, 0.0017))[1]
        # Fine alpha range around best
        alphas_k = sorted(set([round(a, 6) for a in
            list(np.linspace(max(best_alpha_k*0.4, 1e-5), min(best_alpha_k*2.5, 0.005), 12))
            + [0.009, 0.015, 0.02, 0.03]]))  # include high-ratio providers
        for seed_idx in seed_indices:
            div_idx = greedy_diversity(base_oofs, k=k, seed_idx=seed_idx)
            Xk_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div_idx]))
            Xk_te = poly2.transform(np.column_stack([base_tes[i] for i in div_idx]))
            for alpha in alphas_k:
                oof_c, te_c = qreg_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
                r_c = rae(y_tr, oof_c)
                ratio_c = te_c.std() / oof_c.std()
                flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
                print(f"  k={k} seed={seed_idx} a={alpha:.5f}  RAE={r_c:.6f}  ratio={ratio_c:.4f}  [{flag}]")
                all_cands_oofs.append(oof_c)
                all_cands_tes.append(te_c)

    print(f"\nTotal candidates: {len(all_cands_oofs)}")

    # Load nb188 pool
    pool_oofs_tr, pool_oofs_te = [], []
    for stem in NB188_POOL:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        te_p  = DATA_PROCESSED / f"te_{stem}.npy"
        if not oof_p.exists(): continue
        oof_m = np.load(oof_p).astype(np.float64).flatten()
        te_m  = np.load(te_p).astype(np.float64).flatten()
        oof_m = np.where(np.isfinite(oof_m), oof_m, np.nanmean(oof_m))
        te_m  = np.where(np.isfinite(te_m),  te_m,  np.nanmean(te_m))
        pool_oofs_tr.append(oof_m); pool_oofs_te.append(te_m)

    # --- C: 2000-start constrained SLSQP ---
    print(f"\n--- C: 2000-start constrained SLSQP on nb188 + {len(all_cands_oofs)} candidates ---")
    X_tr = np.column_stack(pool_oofs_tr + all_cands_oofs)
    X_te = np.column_stack(pool_oofs_te + all_cands_tes)
    r_c, oof_c, te_c, ratio_c = constrained_slsqp(
        X_tr, X_te, y_tr, n_starts=2000, prev_best=0.297639)

    print("\n=== Summary ===")
    print(f"  nb197 (prev best): RAE=0.297639  ratio=0.5800  [PASS]")
    if oof_c is not None:
        flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
        beat = " ***NEW BEST***" if (ratio_c >= COLLAPSE_THRESH and r_c < 0.297639) else ""
        print(f"  nb198:             RAE={r_c:.6f}  ratio={ratio_c:.4f}  [{flag}]{beat}")

    if oof_c is not None and ratio_c >= COLLAPSE_THRESH and r_c < 0.297639:
        print(f"\n*** NEW BEST: RAE={r_c:.6f} ***")
        np.save(DATA_PROCESSED / "oof_nb198_k_sweep.npy", oof_c)
        np.save(DATA_PROCESSED / "te_nb198_k_sweep.npy",  te_c)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_c})
        out_path = SUBMISSIONS / "198_k_sweep.csv"
        sub.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    else:
        print("\nNo improvement over nb197 (0.297639).")


if __name__ == "__main__":
    main()
