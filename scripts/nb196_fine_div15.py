"""nb196 -- Very fine div-15 alpha sweep + 1000-start constrained SLSQP.

nb195 best: 0.298080  ratio=0.5800  (30-model constrained SLSQP)

Key findings from nb195:
  - div15(0.001): RAE=0.297775 ratio=0.5608 FAIL (close to div15(0.002)=0.297307)
  - div20(0.003): RAE=0.298508 ratio=0.5559 FAIL (useful weight in SLSQP)
  - The improvement from 494 to 500 starts is marginal -- more starts needed

Strategy:
  A: Very fine div-15 alpha sweep (0.0005-0.002) to find global min
  B: Add div-25 candidates
  C: 1000-start constrained SLSQP on combined pool (35+ candidates)
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
    "nb194_constrained", "nb195_expanded",
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


def constrained_slsqp(X_tr, X_te, y_tr, n_starts=1000, label="slsqp", prev_best=0.298080):
    n_m = X_tr.shape[1]
    n_tr = len(y_tr)
    best_r, best_w = 1e9, None

    def obj(w):
        return rae(y_tr, X_tr @ w)

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
        res = minimize(obj, w0, jac=obj_grad,
                       method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-11})
        if res.success and ratio_con(res.x) >= -1e-6:
            r = obj(res.x)
            if r < best_r:
                best_r, best_w = r, res.x.copy()

    if best_w is None:
        print(f"  [{label}] No feasible solution in {n_starts} starts")
        return 1e9, None, None, 0

    oof_b = X_tr @ best_w
    te_b  = X_te @ best_w
    ratio = te_b.std() / oof_b.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and best_r < prev_best) else ""
    print(f"  [{label}] n={n_m}  RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
    top_idx = np.argsort(best_w)[::-1]
    for i in top_idx[:10]:
        if best_w[i] > 0.005:
            print(f"    model_{i}  w={best_w[i]:.4f}")
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb196: Fine div-15 sweep + 1000-start constrained SLSQP ===\n")
    print("nb195 best: 0.298080  ratio=0.5800\n")

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

    div10_idx = greedy_diversity(base_oofs, k=10, seed_idx=0)
    div15_idx = greedy_diversity(base_oofs, k=15, seed_idx=0)
    div20_idx = greedy_diversity(base_oofs, k=20, seed_idx=0)
    div25_idx = greedy_diversity(base_oofs, k=25, seed_idx=0)

    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X10_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div10_idx]))
    X10_te = poly2.transform(np.column_stack([base_tes[i] for i in div10_idx]))
    X15_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div15_idx]))
    X15_te = poly2.transform(np.column_stack([base_tes[i] for i in div15_idx]))
    X20_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div20_idx]))
    X20_te = poly2.transform(np.column_stack([base_tes[i] for i in div20_idx]))
    X25_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div25_idx]))
    X25_te = poly2.transform(np.column_stack([base_tes[i] for i in div25_idx]))

    print(f"div-25 last 5: {[base_stems[i] for i in div25_idx[20:]]}")
    print(f"Poly-2 features: div10={X10_tr.shape[1]}, div15={X15_tr.shape[1]}, div20={X20_tr.shape[1]}, div25={X25_tr.shape[1]}")

    # Fine sweep: all alphas across all div-k settings
    print("\nBuilding comprehensive candidate grid...")
    cand_oofs_tr, cand_oofs_te, cand_names = [], [], []

    alpha_grid = {
        "div10": [0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.002, 0.0025, 0.003],
        "div15": [0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],
        "div20": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],
        "div25": [0.005, 0.008, 0.01, 0.015, 0.02],
    }
    X_map = {
        "div10": (X10_tr, X10_te),
        "div15": (X15_tr, X15_te),
        "div20": (X20_tr, X20_te),
        "div25": (X25_tr, X25_te),
    }

    for div_key, alphas in alpha_grid.items():
        Xt, Xte_arr = X_map[div_key]
        for alpha in alphas:
            oof_c, te_c = qreg_oof(Xt, y_tr, Xte_arr, splits, alpha)
            r_c = rae(y_tr, oof_c)
            ratio_c = te_c.std() / oof_c.std()
            flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
            print(f"  {div_key} a={alpha:.4f}  RAE={r_c:.6f}  ratio={ratio_c:.4f}  [{flag}]")
            cand_oofs_tr.append(oof_c)
            cand_oofs_te.append(te_c)
            cand_names.append(f"{div_key}_a{alpha:.4f}")

    print(f"\nTotal candidates: {len(cand_names)}")

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

    # --- A: Full pool, 1000 starts ---
    print("\n--- A: 1000-start constrained SLSQP on nb188 + all candidates ---")
    all_oofs_tr = pool_oofs_tr + cand_oofs_tr
    all_oofs_te = pool_oofs_te + cand_oofs_te
    X_a_tr = np.column_stack(all_oofs_tr)
    X_a_te = np.column_stack(all_oofs_te)
    r_a, oof_a, te_a, ratio_a = constrained_slsqp(
        X_a_tr, X_a_te, y_tr, n_starts=1000, label="full_1000", prev_best=0.298080)

    # --- B: Best-performing candidates only (top by OOF RAE) ---
    best_by_rae = sorted(zip(
        [rae(y_tr, o) for o in cand_oofs_tr], cand_oofs_tr, cand_oofs_te, cand_names
    ), key=lambda x: x[0])
    top20_oofs = [x[1] for x in best_by_rae[:20]]
    top20_tes  = [x[2] for x in best_by_rae[:20]]
    top20_names = [x[3] for x in best_by_rae[:20]]
    print(f"\n--- B: 500-start SLSQP on nb188 + top-20 candidates by RAE ---")
    print(f"  Top 5 RAEs: {[f'{x[0]:.4f}' for x in best_by_rae[:5]]}")
    X_b_tr = np.column_stack(pool_oofs_tr + top20_oofs)
    X_b_te = np.column_stack(pool_oofs_te + top20_tes)
    r_b, oof_b, te_b, ratio_b = constrained_slsqp(
        X_b_tr, X_b_te, y_tr, n_starts=500, label="top20_rae", prev_best=0.298080)

    print("\n=== Summary ===")
    print(f"  nb195 (prev best): RAE=0.298080  ratio=0.5800  [PASS]")
    candidates_final = [
        (r_a, oof_a, te_a, ratio_a, "A_full_1000"),
        (r_b, oof_b, te_b, ratio_b, "B_top20"),
    ]
    best_rae, best_oof, best_te = 0.298080, None, None
    for r, oof_s, te_s, ratio, nm in candidates_final:
        if oof_s is not None:
            flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
            beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and r < 0.298080) else ""
            print(f"  {nm}: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
            if ratio >= COLLAPSE_THRESH and r < best_rae:
                best_rae, best_oof, best_te = r, oof_s, te_s

    if best_oof is None:
        print("\nNo improvement over nb195 (0.298080).")
        return

    print(f"\n*** NEW BEST: RAE={best_rae:.6f} ***")
    np.save(DATA_PROCESSED / "oof_nb196_fine_div15.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb196_fine_div15.npy",  best_te)

    sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": best_te})
    out_path = SUBMISSIONS / "196_fine_div15.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
