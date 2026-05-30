"""nb201 -- Chemprop-enhanced constrained SLSQP.

Run AFTER nb93 Kaggle GPU completes and OOF predictions are copied to data/processed/.

Setup:
  cp submissions/kaggle_nb93/oof_nb93_chemprop_large_gpu.npy data/processed/
  cp submissions/kaggle_nb93/te_nb93_chemprop_large_gpu.npy data/processed/

What changes vs nb197/nb199:
  - nb93_chemprop_large_gpu is now in the base pool (automatically loaded)
  - GNN predictions are structurally different from LGBM/XGB/QReg (graph vs tabular)
  - The diversity selection will likely include Chemprop early (low correlation with others)
  - Polynomial interactions Chemprop × LGBM are new feature combinations

Also includes:
  - External PXR models as ratio reservoirs (from nb199 approach, ratio ~0.87-0.98)
  - 3000 starts for SLSQP to be more thorough

Expected: Breaking past 0.297639 if Chemprop has OOF RAE ~0.48-0.52 and ratio >= 0.58.
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
from lightgbm import LGBMRegressor

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58
SEED = 42
PREV_BEST = 0.297639

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
    "nb197_dense_grid", "nb198_k_sweep", "nb199_external_pxr",
    "nb200_multi_seed", "nb201_chemprop_slsqp",
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


def load_base_pool(n_tr):
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


def constrained_slsqp(X_tr, X_te, y_tr, n_starts=3000, prev_best=PREV_BEST):
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
    for i in top_idx[:12]:
        if best_w[i] > 0.005:
            print(f"    model_{i}  w={best_w[i]:.4f}")
    return best_r, oof_b, te_b, ratio


def load_external_models(X_tr_int, X_te_int, y_tr_int):
    """Load or retrain external-only PXR models for ratio reservoir."""
    ext_data_dir = DATA_PROCESSED.parent / "external"
    ch_p = ext_data_dir / "chembl_pxr_all_types.parquet"
    bd_p = ext_data_dir / "bindingdb_pxr_direct.parquet"
    if not ch_p.exists() or not bd_p.exists():
        print("External data not found, skipping external models")
        return [], []

    df_ch = pd.read_parquet(ch_p)
    df_bd = pd.read_parquet(bd_p)
    df_ch_f = df_ch[df_ch["measurement_type"].isin(["EC50", "IC50", "AC50"])].copy()
    df_bd_f = df_bd[df_bd["standard_type"].isin(["EC50", "IC50"])].copy()
    ext_df = pd.concat([df_ch_f[["smiles", "pec50"]], df_bd_f[["smiles", "pec50"]]],
                        ignore_index=True).dropna().drop_duplicates("smiles")
    print(f"External PXR: {len(ext_df)} compounds")

    X_ext = impute(combined(ext_df["smiles"].tolist()))
    y_ext = ext_df["pec50"].values.astype(np.float64)

    ext_oofs, ext_tes = [], []
    for name, params in [
        ("ext_lgbm", dict(n_estimators=500, num_leaves=64, learning_rate=0.05)),
        ("ext_lgbm_large", dict(n_estimators=1000, num_leaves=128, learning_rate=0.02,
                                min_child_samples=5)),
    ]:
        m = LGBMRegressor(**params, verbose=-1, random_state=SEED)
        m.fit(X_ext, y_ext)
        tr_pred = m.predict(X_tr_int)
        te_pred = m.predict(X_te_int)
        ratio = te_pred.std() / tr_pred.std()
        rae_val = rae(y_tr_int, tr_pred)
        print(f"  {name}: RAE_tr={rae_val:.4f}  ratio={ratio:.4f}")
        ext_oofs.append(tr_pred)
        ext_tes.append(te_pred)
    return ext_oofs, ext_tes


def main():
    print("=== nb201: Chemprop-enhanced constrained SLSQP ===\n")
    print(f"nb197 best: {PREV_BEST}  ratio=0.5800\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load base pool (includes nb93_chemprop_large_gpu if .npy files exist)
    print("Loading base pool...")
    base_oofs, base_tes, base_stems = load_base_pool(n_tr)
    # Prepend nb183
    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists():
        oof183 = np.load(oof183_p).astype(np.float64).flatten()
        te183  = np.load(te183_p).astype(np.float64).flatten()
        oof183 = np.where(np.isfinite(oof183), oof183, np.nanmean(oof183))
        te183  = np.where(np.isfinite(te183),  te183,  np.nanmean(te183))
        base_oofs.insert(0, oof183); base_tes.insert(0, te183)
        base_stems.insert(0, "nb183_qreg_poly10")
    print(f"Base pool: {len(base_oofs)} models")

    # Check if Chemprop is in pool
    chemprop_in_pool = any("chemprop" in s for s in base_stems)
    if chemprop_in_pool:
        chemprop_stem = next(s for s in base_stems if "chemprop" in s)
        chemprop_idx = base_stems.index(chemprop_stem)
        chemprop_oof = base_oofs[chemprop_idx]
        chemprop_te  = base_tes[chemprop_idx]
        ratio_cp = chemprop_te.std() / chemprop_oof.std()
        rae_cp = rae(y_tr, chemprop_oof)
        print(f"\nChemprop found: {chemprop_stem}")
        print(f"  OOF RAE={rae_cp:.4f}  ratio={ratio_cp:.4f}")
    else:
        print("\nWARNING: Chemprop not found in pool!")
        print("Run: cp submissions/kaggle_nb93/oof_nb93_chemprop_large_gpu.npy data/processed/")
        print("     cp submissions/kaggle_nb93/te_nb93_chemprop_large_gpu.npy data/processed/")
        print("Continuing without Chemprop (will give same result as nb197)")

    # nb188 anchor pool
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

    # Also add Chemprop to pool directly (as anchor)
    if chemprop_in_pool:
        pool_oofs_tr.append(chemprop_oof)
        pool_oofs_te.append(chemprop_te)
        print(f"nb188 + Chemprop anchor pool: {len(pool_oofs_tr)} models")
    else:
        print(f"nb188 pool: {len(pool_oofs_tr)} models")

    # Load external models (ratio reservoirs)
    print("\nLoading external PXR models...")
    X_tr_feat = impute(combined(tr["smiles"].tolist()))
    X_te_feat = impute(combined(te_df["smiles"].tolist()))
    ext_oofs, ext_tes = load_external_models(X_tr_feat, X_te_feat, y_tr)

    # Build QReg candidates
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    alpha_dense = sorted(set([round(a, 5) for a in
        list(np.linspace(0.0005, 0.004, 30)) + [0.005, 0.006, 0.007, 0.008, 0.009]]))
    alpha_high_ratio = [0.01, 0.015, 0.02, 0.03]
    all_alphas = alpha_dense + alpha_high_ratio

    print(f"\nBuilding QReg candidates ({len(all_alphas)} alphas, div15/20/25)...")
    all_cand_oofs, all_cand_tes = [], []

    for k in [15, 20, 25]:
        div_idx = greedy_diversity(base_oofs, k=k, seed_idx=0)
        alphas_k = all_alphas if k in [15, 20] else [0.008, 0.01, 0.015, 0.02, 0.03]
        Xk_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div_idx]))
        Xk_te = poly2.transform(np.column_stack([base_tes[i] for i in div_idx]))
        for alpha in alphas_k:
            oof_c, te_c = qreg_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
            all_cand_oofs.append(oof_c)
            all_cand_tes.append(te_c)
    print(f"QReg candidates: {len(all_cand_oofs)}")

    # If Chemprop is available, also try diversity seeded FROM Chemprop
    if chemprop_in_pool:
        print(f"\nBuilding Chemprop-seeded QReg candidates (seed_idx={chemprop_idx})...")
        for k in [15, 20]:
            div_idx = greedy_diversity(base_oofs, k=k, seed_idx=chemprop_idx)
            Xk_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div_idx]))
            Xk_te = poly2.transform(np.column_stack([base_tes[i] for i in div_idx]))
            for alpha in alpha_dense + alpha_high_ratio:
                oof_c, te_c = qreg_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
                all_cand_oofs.append(oof_c)
                all_cand_tes.append(te_c)
        print(f"Total candidates: {len(all_cand_oofs)}")

    # === SLSQP experiments ===

    # A: Without Chemprop, without external (baseline, sanity check vs nb197)
    print(f"\n--- A: Baseline (nb188 + QReg) ---")
    X_a_tr = np.column_stack(pool_oofs_tr[:8] + all_cand_oofs[:len(alpha_dense)+len(alpha_high_ratio)])
    X_a_te = np.column_stack(pool_oofs_te[:8] + all_cand_tes[:len(alpha_dense)+len(alpha_high_ratio)])
    # Use 1000 starts for baseline (just a sanity check)
    r_a, oof_a, te_a, ratio_a = constrained_slsqp(X_a_tr, X_a_te, y_tr,
                                                    n_starts=1000, prev_best=1e9)

    # B: Full pool (nb188 + Chemprop + all QReg candidates + external)
    print(f"\n--- B: Full pool (3000 starts) ---")
    X_b_tr = np.column_stack(pool_oofs_tr + all_cand_oofs + ext_oofs)
    X_b_te = np.column_stack(pool_oofs_te + all_cand_tes  + ext_tes)
    r_b, oof_b, te_b, ratio_b = constrained_slsqp(X_b_tr, X_b_te, y_tr,
                                                    n_starts=3000, prev_best=PREV_BEST)

    # C: Without external (Chemprop only as new element)
    print(f"\n--- C: nb188 + Chemprop + QReg (no external, 2000 starts) ---")
    X_c_tr = np.column_stack(pool_oofs_tr + all_cand_oofs)
    X_c_te = np.column_stack(pool_oofs_te + all_cand_tes)
    r_c, oof_c, te_c, ratio_c = constrained_slsqp(X_c_tr, X_c_te, y_tr,
                                                    n_starts=2000, prev_best=PREV_BEST)

    # Summary
    print("\n=== Summary ===")
    print(f"  nb197 (prev best): RAE={PREV_BEST}  ratio=0.5800  [PASS]")
    print(f"  A_baseline:   RAE={r_a:.6f}  ratio={ratio_a:.4f}  "
          f"[{'PASS' if ratio_a>=0.58 else 'FAIL'}]")

    candidates_final = [
        (r_b, oof_b, te_b, ratio_b, "B_full"),
        (r_c, oof_c, te_c, ratio_c, "C_no_ext"),
    ]
    best_rae, best_oof, best_te = PREV_BEST, None, None
    for r, oof_s, te_s, ratio, nm in candidates_final:
        if oof_s is not None:
            flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
            beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and r < PREV_BEST) else ""
            print(f"  {nm}: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
            if ratio >= COLLAPSE_THRESH and r < best_rae:
                best_rae, best_oof, best_te = r, oof_s, te_s

    if best_oof is None:
        print(f"\nNo improvement over nb197 ({PREV_BEST}).")
        return

    print(f"\n*** NEW BEST: RAE={best_rae:.6f} ***")
    np.save(DATA_PROCESSED / "oof_nb201_chemprop_slsqp.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb201_chemprop_slsqp.npy",  best_te)

    sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": best_te})
    out_path = SUBMISSIONS / "201_chemprop_slsqp.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
