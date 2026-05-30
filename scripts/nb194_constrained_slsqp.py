"""nb194 -- Constrained SLSQP: force ratio >= 0.58 in the optimization.

nb188 best:           0.298519  ratio=0.5815
nb193 experiment C:   0.298177  ratio=0.5797  [FAIL] -- 0.0003 below threshold!

KEY INSIGHT: The unconstrained SLSQP finds a better RAE (0.298177) but ratio=0.5797.
If we add ratio >= 0.58 as a hard constraint, the constrained optimum should be
close to 0.298177 (since it only needs to move ratio by 0.0003).

Constrained form:
  min  RAE(w)
  s.t. sum(w) = 1
       w_i >= 0
       std(X_te @ w) / std(X_tr @ w) >= 0.58

Experiments:
  A: Constrained SLSQP on 10-model pool (nb193 exp C pool)
  B: Constrained SLSQP on 11-model pool (add div15(0.002) and div20(0.005))
  C: Constrained SLSQP on full expanded pool (all candidates)
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


def qreg_train_full(X_tr, y_tr, X_te, splits, alpha):
    n_tr = len(y_tr)
    oof_s = np.full(n_tr, np.nan)
    for _, (tr_idx, va_idx) in enumerate(splits):
        m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_s[va_idx] = m.predict(X_tr[va_idx])
    m_f = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
    m_f.fit(X_tr, y_tr)
    return oof_s, m_f.predict(X_te)


def constrained_slsqp(X_tr, X_te, y_tr, n_starts=300, label="constrained_slsqp"):
    """SLSQP with ratio >= 0.58 as hard constraint."""
    n_m = X_tr.shape[1]
    n_tr = len(y_tr)
    best_r, best_w = 1e9, None

    def obj(w):
        return rae(y_tr, X_tr @ w)

    def obj_grad(w):
        pred = X_tr @ w
        mae_mean = np.abs(y_tr - y_tr.mean()).mean()
        return X_tr.T @ np.sign(pred - y_tr) / (n_tr * mae_mean)

    def ratio_constraint(w):
        pred_tr = X_tr @ w
        pred_te = X_te @ w
        std_tr = pred_tr.std()
        std_te = pred_te.std()
        if std_tr < 1e-9:
            return -1.0
        return std_te / std_tr - COLLAPSE_THRESH

    constraints = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1},
        {"type": "ineq", "fun": ratio_constraint},
    ]
    bounds = [(0, 1)] * n_m
    rng = np.random.default_rng(SEED)

    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(n_m))
        # Optionally start near feasible: scale w0 toward high-ratio models
        res = minimize(obj, w0, jac=obj_grad,
                       method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-11})
        if res.success and ratio_constraint(res.x) >= -1e-6:
            r = obj(res.x)
            if r < best_r:
                best_r, best_w = r, res.x.copy()

    if best_w is None:
        print(f"  [{label}] No feasible solution found in {n_starts} starts")
        return 1e9, None, None, 0

    oof_b = X_tr @ best_w
    te_b  = X_te @ best_w
    ratio = te_b.std() / oof_b.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL (numerics)"
    beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and best_r < 0.298519) else ""
    print(f"  [{label}] RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
    for i in range(n_m):
        if best_w[i] > 0.005:
            print(f"    model_{i}  w={best_w[i]:.4f}")
    return best_r, oof_b, te_b, ratio


def load_nb188_pool_arrays(n_tr):
    oofs_tr, oofs_te = [], []
    for stem in NB188_POOL:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        te_p  = DATA_PROCESSED / f"te_{stem}.npy"
        if not oof_p.exists():
            continue
        oof_m = np.load(oof_p).astype(np.float64).flatten()
        te_m  = np.load(te_p).astype(np.float64).flatten()
        oof_m = np.where(np.isfinite(oof_m), oof_m, np.nanmean(oof_m))
        te_m  = np.where(np.isfinite(te_m),  te_m,  np.nanmean(te_m))
        oofs_tr.append(oof_m); oofs_te.append(te_m)
    return oofs_tr, oofs_te


def main():
    print("=== nb194: Constrained SLSQP (force ratio >= 0.58) ===\n")
    print("nb188 best: 0.298519  ratio=0.5815")
    print("nb193 C (unconstrained): 0.298177  ratio=0.5797  [FAIL]")
    print("nb193 D (unconstrained+div20): 0.297617  ratio=0.5683  [FAIL]")
    print("nb193 E (unconstrained+div15): 0.296489  ratio=0.5659  [FAIL]\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load nb188 pool
    pool_oofs_tr, pool_oofs_te = load_nb188_pool_arrays(n_tr)
    print(f"nb188 pool: {len(pool_oofs_tr)} models loaded")

    # Build diverse sets (need to compute QReg OOFs for div15 and div20 candidates)
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

    div15_idx = greedy_diversity(base_oofs, k=15, seed_idx=0)
    div20_idx = greedy_diversity(base_oofs, k=20, seed_idx=0)

    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X15_base_tr = np.column_stack([base_oofs[i] for i in div15_idx])
    X15_base_te = np.column_stack([base_tes[i] for i in div15_idx])
    X20_base_tr = np.column_stack([base_oofs[i] for i in div20_idx])
    X20_base_te = np.column_stack([base_tes[i] for i in div20_idx])

    X15p2_tr = poly2.fit_transform(X15_base_tr)
    X15p2_te = poly2.transform(X15_base_te)
    X20p2_tr = poly2.fit_transform(X20_base_tr)
    X20p2_te = poly2.transform(X20_base_te)

    print("\nTraining QReg OOF predictions for candidates...")
    # div15 alpha=0.002 (best OOF RAE=0.297307)
    print("  Computing div15(alpha=0.002) OOF...")
    oof_div15_002, te_div15_002 = qreg_train_full(X15p2_tr, y_tr, X15p2_te, splits, 0.002)
    print(f"    div15(0.002): RAE={rae(y_tr, oof_div15_002):.6f}  ratio={te_div15_002.std()/oof_div15_002.std():.4f}")

    # div15 alpha=0.009 (first passing)
    print("  Computing div15(alpha=0.009) OOF...")
    oof_div15_009, te_div15_009 = qreg_train_full(X15p2_tr, y_tr, X15p2_te, splits, 0.009)
    print(f"    div15(0.009): RAE={rae(y_tr, oof_div15_009):.6f}  ratio={te_div15_009.std()/oof_div15_009.std():.4f}")

    # div20 alpha=0.005 (best OOF RAE)
    print("  Computing div20(alpha=0.005) OOF...")
    oof_div20_005, te_div20_005 = qreg_train_full(X20p2_tr, y_tr, X20p2_te, splits, 0.005)
    print(f"    div20(0.005): RAE={rae(y_tr, oof_div20_005):.6f}  ratio={te_div20_005.std()/oof_div20_005.std():.4f}")

    # div20 alpha=0.009 (first passing)
    print("  Computing div20(alpha=0.009) OOF...")
    oof_div20_009, te_div20_009 = qreg_train_full(X20p2_tr, y_tr, X20p2_te, splits, 0.009)
    print(f"    div20(0.009): RAE={rae(y_tr, oof_div20_009):.6f}  ratio={te_div20_009.std()/oof_div20_009.std():.4f}")

    results = {}

    # --- A: Constrained SLSQP on 10-model pool (nb188 + div20(0.009) + div15(0.009)) ---
    print("\n--- A: Constrained SLSQP on 10-model pool (nb188 + div20(0.009) + div15(0.009)) ---")
    oofs_a = pool_oofs_tr + [oof_div20_009, oof_div15_009]
    tes_a  = pool_oofs_te + [te_div20_009, te_div15_009]
    X_a_tr = np.column_stack(oofs_a)
    X_a_te = np.column_stack(tes_a)
    r_a, oof_a, te_a, ratio_a = constrained_slsqp(X_a_tr, X_a_te, y_tr, n_starts=300, label="10model_A")
    if oof_a is not None:
        results["A"] = (r_a, oof_a, te_a, ratio_a)

    # --- B: Constrained SLSQP: nb188 + div20(0.005) [failing] ---
    print("\n--- B: Constrained SLSQP: nb188 + div20(0.005) ---")
    oofs_b = pool_oofs_tr + [oof_div20_005]
    tes_b  = pool_oofs_te + [te_div20_005]
    X_b_tr = np.column_stack(oofs_b)
    X_b_te = np.column_stack(tes_b)
    r_b, oof_b, te_b, ratio_b = constrained_slsqp(X_b_tr, X_b_te, y_tr, n_starts=300, label="9model_div20_005")
    if oof_b is not None:
        results["B"] = (r_b, oof_b, te_b, ratio_b)

    # --- C: Constrained SLSQP: nb188 + div15(0.002) [failing] ---
    print("\n--- C: Constrained SLSQP: nb188 + div15(0.002) ---")
    oofs_c = pool_oofs_tr + [oof_div15_002]
    tes_c  = pool_oofs_te + [te_div15_002]
    X_c_tr = np.column_stack(oofs_c)
    X_c_te = np.column_stack(tes_c)
    r_c, oof_c, te_c, ratio_c = constrained_slsqp(X_c_tr, X_c_te, y_tr, n_starts=300, label="9model_div15_002")
    if oof_c is not None:
        results["C"] = (r_c, oof_c, te_c, ratio_c)

    # --- D: Constrained SLSQP: nb188 + all 4 candidates ---
    print("\n--- D: Constrained SLSQP: nb188 + all 4 QReg candidates ---")
    oofs_d = pool_oofs_tr + [oof_div15_002, oof_div15_009, oof_div20_005, oof_div20_009]
    tes_d  = pool_oofs_te + [te_div15_002, te_div15_009, te_div20_005, te_div20_009]
    X_d_tr = np.column_stack(oofs_d)
    X_d_te = np.column_stack(tes_d)
    r_d, oof_d, te_d, ratio_d = constrained_slsqp(X_d_tr, X_d_te, y_tr, n_starts=300, label="12model_all")
    if oof_d is not None:
        results["D"] = (r_d, oof_d, te_d, ratio_d)

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  nb188 (prev best): RAE=0.298519  ratio=0.5815  [PASS]")
    best_rae, best_oof, best_te = 0.298519, None, None
    for k_str, (r, oof_s, te_s, ratio) in results.items():
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and r < 0.298519) else ""
        print(f"  {k_str}: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
        if ratio >= COLLAPSE_THRESH and r < best_rae:
            best_rae, best_oof, best_te = r, oof_s, te_s

    if best_oof is None:
        print("\nNo improvement over nb188 (0.298519).")
        return

    print(f"\n*** NEW BEST: RAE={best_rae:.6f} ***")
    np.save(DATA_PROCESSED / "oof_nb194_constrained.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb194_constrained.npy",  best_te)

    sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": best_te})
    out_path = SUBMISSIONS / "194_constrained_slsqp.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
