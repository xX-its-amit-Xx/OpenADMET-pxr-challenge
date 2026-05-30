"""nb199 -- External PXR data as ratio reservoir in constrained SLSQP.

nb197 best: 0.297639  ratio=0.5800

Hypothesis:
  External-only models (trained on ChEMBL + BindingDB PXR EC50/IC50 data)
  predict on internal train/test with ratio ~1.0, making them superior
  ratio reservoirs vs div25(alpha=0.02) (ratio=0.6351). This lets the
  constrained SLSQP allocate more weight to the low-ratio good-OOF-RAE
  div15 candidates, potentially beating 0.297639.

Approach A: External-only predictions
  Train QReg / LGBM on ChEMBL + BindingDB PXR data.
  Predict on internal train/test (no internal labels used in training).
  Expected ratio: ~1.0 (same chemical space, no fold artifacts).

Approach B: External-augmented OOF
  Scaffold CV: train on external_PXR + (internal folds K-1), predict fold K.
  True OOF predictions from a richer training set.
  Expected: slightly different OOF predictions + potentially different ratio.

Both are added to nb197's pool and run through constrained SLSQP (2000 starts).
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


def constrained_slsqp(X_tr, X_te, y_tr, n_starts=2000, prev_best=PREV_BEST):
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


def build_qreg_candidates(base_oofs, base_tes, base_stems, y_tr, splits, poly2,
                           div_k_list, alpha_list):
    cand_oofs, cand_tes, cand_names = [], [], []
    for k in div_k_list:
        div_idx = greedy_diversity(base_oofs, k=k, seed_idx=0)
        Xk_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div_idx]))
        Xk_te = poly2.transform(np.column_stack([base_tes[i] for i in div_idx]))
        for alpha in alpha_list:
            oof_c, te_c = qreg_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
            r_c = rae(y_tr, oof_c)
            ratio_c = te_c.std() / oof_c.std()
            flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
            cand_oofs.append(oof_c)
            cand_tes.append(te_c)
            cand_names.append(f"div{k}_a{alpha:.5f}")
    return cand_oofs, cand_tes, cand_names


def load_external_pxr():
    """Load and combine ChEMBL + BindingDB PXR EC50/IC50/AC50 data."""
    df_ch = pd.read_parquet(DATA_PROCESSED.parent / "external" / "chembl_pxr_all_types.parquet")
    df_bd = pd.read_parquet(DATA_PROCESSED.parent / "external" / "bindingdb_pxr_direct.parquet")
    df_ch_f = df_ch[df_ch["measurement_type"].isin(["EC50", "IC50", "AC50"])].copy()
    df_bd_f = df_bd[df_bd["standard_type"].isin(["EC50", "IC50"])].copy()
    ext_df = pd.concat([
        df_ch_f[["smiles", "pec50"]],
        df_bd_f[["smiles", "pec50"]].rename(columns={"smiles": "smiles"}),
    ], ignore_index=True)
    ext_df = ext_df.dropna(subset=["smiles", "pec50"]).drop_duplicates("smiles")
    print(f"External PXR: {len(ext_df)} compounds (ChEMBL {len(df_ch_f)} + BindingDB {len(df_bd_f)}, dedup)")
    return ext_df


def train_external_models(ext_df, X_tr_int, X_te_int, y_tr_int, splits):
    """Train models on external data only; predict on internal train/test."""
    X_ext = impute(combined(ext_df["smiles"].tolist()))
    y_ext = ext_df["pec50"].values.astype(np.float64)
    print(f"External features: {X_ext.shape}")

    ext_preds = {}

    # A1: LGBM on external data (standard)
    print("  Training: LGBM on external only...", flush=True)
    lgbm_ext = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                              verbose=-1, random_state=SEED)
    lgbm_ext.fit(X_ext, y_ext)
    tr_pred = lgbm_ext.predict(X_tr_int)
    te_pred = lgbm_ext.predict(X_te_int)
    ratio_a1 = te_pred.std() / tr_pred.std()
    rae_a1 = rae(y_tr_int, tr_pred)
    print(f"    A1 LGBM-ext: RAE_tr={rae_a1:.6f}  ratio={ratio_a1:.4f}")
    ext_preds["ext_lgbm"] = (tr_pred, te_pred)

    # A2: LGBM on external, more trees, lower LR
    print("  Training: LGBM (large) on external only...", flush=True)
    lgbm_ext2 = LGBMRegressor(n_estimators=1000, num_leaves=128, learning_rate=0.02,
                               min_child_samples=5, verbose=-1, random_state=SEED+1)
    lgbm_ext2.fit(X_ext, y_ext)
    tr_pred2 = lgbm_ext2.predict(X_tr_int)
    te_pred2 = lgbm_ext2.predict(X_te_int)
    ratio_a2 = te_pred2.std() / tr_pred2.std()
    rae_a2 = rae(y_tr_int, tr_pred2)
    print(f"    A2 LGBM-large-ext: RAE_tr={rae_a2:.6f}  ratio={ratio_a2:.4f}")
    ext_preds["ext_lgbm_large"] = (tr_pred2, te_pred2)

    # A3: LGBM with MAE objective on external data
    print("  Training: LGBM-MAE on external only...", flush=True)
    lgbm_ext3 = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                               objective="quantile", alpha=0.5,
                               verbose=-1, random_state=SEED+2)
    lgbm_ext3.fit(X_ext, y_ext)
    tr_pred3 = lgbm_ext3.predict(X_tr_int)
    te_pred3 = lgbm_ext3.predict(X_te_int)
    ratio_a3 = te_pred3.std() / tr_pred3.std()
    rae_a3 = rae(y_tr_int, tr_pred3)
    print(f"    A3 LGBM-MAE-ext: RAE_tr={rae_a3:.6f}  ratio={ratio_a3:.4f}")
    ext_preds["ext_lgbm_mae"] = (tr_pred3, te_pred3)

    return ext_preds


def train_augmented_oof(ext_df, X_tr_int, X_te_int, y_tr_int, splits):
    """Scaffold-CV OOF from LGBM trained on external + internal (each fold)."""
    X_ext = impute(combined(ext_df["smiles"].tolist()))
    y_ext = ext_df["pec50"].values.astype(np.float64)
    n_tr = len(y_tr_int)
    oof_aug = np.full(n_tr, np.nan)

    print("  Training: augmented OOF (external + internal fold)...", flush=True)
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        # Combine external + internal training fold
        X_fold_tr = np.vstack([X_ext, X_tr_int[tr_idx]])
        y_fold_tr = np.concatenate([y_ext, y_tr_int[tr_idx]])
        m = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                          verbose=-1, random_state=SEED + fold_i)
        m.fit(X_fold_tr, y_fold_tr)
        oof_aug[va_idx] = m.predict(X_tr_int[va_idx])

    # Full model for test predictions
    X_full = np.vstack([X_ext, X_tr_int])
    y_full = np.concatenate([y_ext, y_tr_int])
    m_full = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                           verbose=-1, random_state=SEED)
    m_full.fit(X_full, y_full)
    te_aug = m_full.predict(X_te_int)

    ratio_aug = te_aug.std() / oof_aug.std()
    rae_aug = rae(y_tr_int, oof_aug)
    print(f"    B1 augmented-OOF: RAE_oof={rae_aug:.6f}  ratio={ratio_aug:.4f}")
    return oof_aug, te_aug, rae_aug, ratio_aug


def main():
    print("=== nb199: External PXR data as ratio reservoir ===\n")
    print(f"nb197 best: {PREV_BEST}  ratio=0.5800\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Internal features for external model predictions
    print("Featurizing internal train/test for external models...")
    X_tr_feat = impute(combined(tr["smiles"].tolist()))
    X_te_feat = impute(combined(te_df["smiles"].tolist()))

    # Load external PXR data
    print("\n--- Loading external PXR data ---")
    ext_df = load_external_pxr()

    # --- A: External-only model predictions ---
    print("\n--- A: External-only models ---")
    ext_preds = train_external_models(ext_df, X_tr_feat, X_te_feat, y_tr, splits)

    # --- B: External-augmented OOF ---
    print("\n--- B: External-augmented OOF ---")
    oof_aug, te_aug, rae_aug, ratio_aug = train_augmented_oof(
        ext_df, X_tr_feat, X_te_feat, y_tr, splits)

    # --- Rebuild nb197 pool ---
    print("\n--- Rebuilding nb197 pool (base models + QReg candidates) ---")
    base_oofs, base_tes, base_stems = load_base_pool(n_tr)
    # Prepend nb183 if available
    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists():
        oof183 = np.load(oof183_p).astype(np.float64).flatten()
        te183  = np.load(te183_p).astype(np.float64).flatten()
        oof183 = np.where(np.isfinite(oof183), oof183, np.nanmean(oof183))
        te183  = np.where(np.isfinite(te183),  te183,  np.nanmean(te183))
        base_oofs.insert(0, oof183); base_tes.insert(0, te183)
        base_stems.insert(0, "nb183_qreg_poly10")

    poly2 = PolynomialFeatures(degree=2, include_bias=False)

    # nb188 pool (8 base models)
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

    print(f"nb188 base models loaded: {len(pool_oofs_tr)}")

    # QReg candidates (same as nb197 dense grid)
    alpha_dense = sorted(set([round(a, 5) for a in
        list(np.linspace(0.0005, 0.004, 30)) + [0.005, 0.006, 0.007, 0.008, 0.009]]))
    alpha_high_ratio = [0.01, 0.015, 0.02, 0.03]
    all_alphas = alpha_dense + alpha_high_ratio

    print(f"Building QReg candidates (div15, div20, div25 x {len(all_alphas)} alphas)...")
    cand_oofs, cand_tes, cand_names = build_qreg_candidates(
        base_oofs, base_tes, base_stems, y_tr, splits, poly2,
        div_k_list=[15, 20], alpha_list=all_alphas)
    # div25 (ratio reservoirs)
    cand25_oofs, cand25_tes, cand25_names = build_qreg_candidates(
        base_oofs, base_tes, base_stems, y_tr, splits, poly2,
        div_k_list=[25], alpha_list=[0.008, 0.01, 0.015, 0.02, 0.03])
    all_cands_oofs = cand_oofs + cand25_oofs
    all_cands_tes  = cand_tes  + cand25_tes
    all_cands_names = cand_names + cand25_names
    print(f"QReg candidates built: {len(all_cands_oofs)}")

    # --- C: nb197 baseline (no external) ---
    print("\n--- C: nb197 baseline (sanity check, 1000 starts) ---")
    X_c_tr = np.column_stack(pool_oofs_tr + all_cands_oofs)
    X_c_te = np.column_stack(pool_oofs_te + all_cands_tes)
    r_c, oof_c, te_c, ratio_c = constrained_slsqp(
        X_c_tr, X_c_te, y_tr, n_starts=1000, prev_best=1e9)  # compare to nb197

    # --- D: + External-only models ---
    print("\n--- D: + External-only models (2000 starts) ---")
    ext_oofs = [v[0] for v in ext_preds.values()]
    ext_tes  = [v[1] for v in ext_preds.values()]
    X_d_tr = np.column_stack(pool_oofs_tr + all_cands_oofs + ext_oofs)
    X_d_te = np.column_stack(pool_oofs_te + all_cands_tes  + ext_tes)
    r_d, oof_d, te_d, ratio_d = constrained_slsqp(
        X_d_tr, X_d_te, y_tr, n_starts=2000, prev_best=PREV_BEST)

    # --- E: + External-augmented OOF ---
    print("\n--- E: + External-augmented OOF only (2000 starts) ---")
    X_e_tr = np.column_stack(pool_oofs_tr + all_cands_oofs + [oof_aug])
    X_e_te = np.column_stack(pool_oofs_te + all_cands_tes  + [te_aug])
    r_e, oof_e, te_e, ratio_e = constrained_slsqp(
        X_e_tr, X_e_te, y_tr, n_starts=2000, prev_best=PREV_BEST)

    # --- F: All external (both A and B) ---
    print("\n--- F: All external (external-only + augmented OOF, 2000 starts) ---")
    X_f_tr = np.column_stack(pool_oofs_tr + all_cands_oofs + ext_oofs + [oof_aug])
    X_f_te = np.column_stack(pool_oofs_te + all_cands_tes  + ext_tes  + [te_aug])
    r_f, oof_f, te_f, ratio_f = constrained_slsqp(
        X_f_tr, X_f_te, y_tr, n_starts=2000, prev_best=PREV_BEST)

    print("\n=== Summary ===")
    print(f"  nb197 (prev best): RAE={PREV_BEST}  ratio=0.5800  [PASS]")
    print(f"  C_baseline:   RAE={r_c:.6f}  ratio={ratio_c:.4f}  "
          f"[{'PASS' if ratio_c>=0.58 else 'FAIL'}]")

    candidates_final = [
        (r_d, oof_d, te_d, ratio_d, "D_ext_only"),
        (r_e, oof_e, te_e, ratio_e, "E_ext_aug"),
        (r_f, oof_f, te_f, ratio_f, "F_all_ext"),
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
    np.save(DATA_PROCESSED / "oof_nb199_external_pxr.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb199_external_pxr.npy",  best_te)

    sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": best_te})
    out_path = SUBMISSIONS / "199_external_pxr.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
