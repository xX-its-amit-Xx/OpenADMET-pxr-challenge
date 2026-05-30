"""nb202 -- Fast external reservoir + Ridge candidates for constrained SLSQP.

Key speedup over nb199: Ridge regression (closed-form, ~1s per fit) instead of
QuantileRegressor (LP solver, ~30-120s per fit). Ridge candidates have a
similar ratio vs RAE tradeoff to QReg candidates — high alpha = smoother
predictions = higher test/train ratio.

External LGBM models (ratio ~0.87-0.98) act as ratio reservoirs, allowing
SLSQP to allocate less weight to poor-RAE high-ratio candidates.

Expected runtime: ~20 minutes (vs nb199's 3+ hours).

nb197 best: 0.297639  ratio=0.5800
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
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

t0 = time.time()

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
    "nb200_multi_seed", "nb201_chemprop_slsqp", "nb202_ridge_external",
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


def ridge_oof(X_tr, y_tr, X_te, splits, alpha):
    """Ridge regression OOF predictions."""
    n_tr = len(y_tr)
    oof_s = np.full(n_tr, np.nan)
    for _, (tr_idx, va_idx) in enumerate(splits):
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_s[va_idx] = m.predict(X_tr[va_idx])
    m_f = Ridge(alpha=alpha, fit_intercept=True)
    m_f.fit(X_tr, y_tr)
    return oof_s, m_f.predict(X_te)


def constrained_slsqp(X_tr, X_te, y_tr, n_starts=2000, prev_best=PREV_BEST, tag=""):
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
        print(f"  [{tag}] No feasible solution found", flush=True)
        return 1e9, None, None, 0

    oof_b = X_tr @ best_w; te_b = X_te @ best_w
    ratio = te_b.std() / oof_b.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and best_r < prev_best) else ""
    print(f"  [{tag}] n={n_m}  RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}", flush=True)
    top_idx = np.argsort(best_w)[::-1]
    for i in top_idx[:10]:
        if best_w[i] > 0.005:
            print(f"    model_{i}  w={best_w[i]:.4f}", flush=True)
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb202: Ridge candidates + External reservoir ===\n", flush=True)
    print(f"nb197 best: {PREV_BEST}  ratio=0.5800\n", flush=True)

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # --- Load external PXR data ---
    print("Loading external PXR data...", flush=True)
    df_ch = pd.read_parquet(DATA_PROCESSED.parent / "external" / "chembl_pxr_all_types.parquet")
    df_bd = pd.read_parquet(DATA_PROCESSED.parent / "external" / "bindingdb_pxr_direct.parquet")
    df_ch_f = df_ch[df_ch["measurement_type"].isin(["EC50", "IC50", "AC50"])].copy()
    df_bd_f = df_bd[df_bd["standard_type"].isin(["EC50", "IC50"])].copy()
    ext_df = pd.concat([df_ch_f[["smiles", "pec50"]], df_bd_f[["smiles", "pec50"]]], ignore_index=True)
    ext_df = ext_df.dropna(subset=["smiles", "pec50"]).drop_duplicates("smiles")
    print(f"External PXR: {len(ext_df)} compounds", flush=True)

    # --- Featurize ---
    print("Featurizing internal train/test...", flush=True)
    X_tr_feat = impute(combined(tr["smiles"].tolist()))
    X_te_feat = impute(combined(te_df["smiles"].tolist()))
    print("Featurizing external compounds...", flush=True)
    X_ext = impute(combined(ext_df["smiles"].tolist()))
    y_ext = ext_df["pec50"].values.astype(np.float64)
    print(f"Done. {time.time()-t0:.0f}s elapsed", flush=True)

    # --- A: Train external LGBM models ---
    print("\n--- A: External-only LGBM models ---", flush=True)
    ext_preds = {}

    # A1: Standard LGBM
    print("  A1: LGBM (standard)...", flush=True)
    m = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, verbose=-1, random_state=SEED)
    m.fit(X_ext, y_ext)
    tr_p = m.predict(X_tr_feat); te_p = m.predict(X_te_feat)
    r1 = te_p.std() / tr_p.std()
    print(f"    RAE_tr={rae(y_tr, tr_p):.4f}  ratio={r1:.4f}", flush=True)
    ext_preds["ext_lgbm"] = (tr_p, te_p)

    # A2: Large LGBM
    print("  A2: LGBM (large)...", flush=True)
    m2 = LGBMRegressor(n_estimators=1000, num_leaves=128, learning_rate=0.02,
                       min_child_samples=5, verbose=-1, random_state=SEED+1)
    m2.fit(X_ext, y_ext)
    tr_p2 = m2.predict(X_tr_feat); te_p2 = m2.predict(X_te_feat)
    r2 = te_p2.std() / tr_p2.std()
    print(f"    RAE_tr={rae(y_tr, tr_p2):.4f}  ratio={r2:.4f}", flush=True)
    ext_preds["ext_lgbm_large"] = (tr_p2, te_p2)

    # A3: MAE objective
    print("  A3: LGBM (MAE)...", flush=True)
    m3 = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                       objective="quantile", alpha=0.5, verbose=-1, random_state=SEED+2)
    m3.fit(X_ext, y_ext)
    tr_p3 = m3.predict(X_tr_feat); te_p3 = m3.predict(X_te_feat)
    r3 = te_p3.std() / tr_p3.std()
    print(f"    RAE_tr={rae(y_tr, tr_p3):.4f}  ratio={r3:.4f}", flush=True)
    ext_preds["ext_lgbm_mae"] = (tr_p3, te_p3)
    print(f"  Done. {time.time()-t0:.0f}s elapsed", flush=True)

    # --- B: Load base pool ---
    print("\n--- B: Loading base pool ---", flush=True)
    base_oofs, base_tes, base_stems = load_base_pool(n_tr)

    # Add nb183 if not in pool
    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists() and not any(s == "nb183_qreg_poly10" for s in base_stems):
        o183 = np.load(oof183_p).astype(np.float64).flatten()
        t183 = np.load(te183_p).astype(np.float64).flatten()
        o183 = np.where(np.isfinite(o183), o183, np.nanmean(o183))
        t183 = np.where(np.isfinite(t183), t183, np.nanmean(t183))
        base_oofs.insert(0, o183); base_tes.insert(0, t183)
        base_stems.insert(0, "nb183_qreg_poly10")

    print(f"Base pool: {len(base_oofs)} models", flush=True)

    # NB188_POOL subset
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
    print(f"NB188 base models: {len(pool_oofs_tr)}", flush=True)

    # --- C: Build Ridge candidates (fast!) ---
    print("\n--- C: Building Ridge candidates (div15, div20, div25) ---", flush=True)
    poly2 = PolynomialFeatures(degree=2, include_bias=False)

    # Key alpha values: cover the same range as nb197 QReg but sparser
    # Low alpha = low ratio (but good RAE), high alpha = high ratio (but poor RAE)
    alpha_sparse = sorted(set([
        0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
        1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0
    ]))

    cand_oofs, cand_tes, cand_names = [], [], []
    for k in [15, 20, 25]:
        t_k = time.time()
        div_idx = greedy_diversity(base_oofs, k=k, seed_idx=0)
        Xk_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div_idx]))
        Xk_te = poly2.transform(np.column_stack([base_tes[i] for i in div_idx]))
        for alpha in alpha_sparse:
            oof_c, te_c = ridge_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
            r_c = rae(y_tr, oof_c)
            ratio_c = te_c.std() / oof_c.std()
            flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
            print(f"  div{k} a={alpha:.4f}  RAE={r_c:.4f}  ratio={ratio_c:.4f}  [{flag}]", flush=True)
            cand_oofs.append(oof_c); cand_tes.append(te_c)
            cand_names.append(f"ridge_div{k}_a{alpha}")
        print(f"  div{k} done ({len(alpha_sparse)} alphas, {time.time()-t_k:.0f}s)", flush=True)

    print(f"Total Ridge candidates: {len(cand_oofs)}", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)

    # External model OOF/te arrays
    ext_oofs = [v[0] for v in ext_preds.values()]
    ext_tes  = [v[1] for v in ext_preds.values()]

    # --- D: Baseline (NB188 + Ridge, no external) ---
    print("\n--- D: Baseline (NB188 base + Ridge candidates, 1500 starts) ---", flush=True)
    X_d_tr = np.column_stack(pool_oofs_tr + cand_oofs)
    X_d_te = np.column_stack(pool_oofs_te + cand_tes)
    r_d, oof_d, te_d, ratio_d = constrained_slsqp(
        X_d_tr, X_d_te, y_tr, n_starts=1500, prev_best=1e9, tag="D_baseline")

    # --- E: + External models ---
    print("\n--- E: + External LGBM models (2000 starts) ---", flush=True)
    X_e_tr = np.column_stack(pool_oofs_tr + cand_oofs + ext_oofs)
    X_e_te = np.column_stack(pool_oofs_te + cand_tes  + ext_tes)
    r_e, oof_e, te_e, ratio_e = constrained_slsqp(
        X_e_tr, X_e_te, y_tr, n_starts=2000, prev_best=PREV_BEST, tag="E_ext")

    # --- F: Full base pool + Ridge + external (broader pool) ---
    print("\n--- F: Full base pool + Ridge + external (2000 starts) ---", flush=True)
    X_f_tr = np.column_stack(base_oofs + cand_oofs + ext_oofs)
    X_f_te = np.column_stack(base_tes  + cand_tes  + ext_tes)
    r_f, oof_f, te_f, ratio_f = constrained_slsqp(
        X_f_tr, X_f_te, y_tr, n_starts=2000, prev_best=PREV_BEST, tag="F_full")

    print(f"\n=== Summary ({time.time()-t0:.0f}s total) ===", flush=True)
    print(f"  nb197 (prev best): RAE={PREV_BEST}  ratio=0.5800  [PASS]", flush=True)
    print(f"  D baseline (Ridge, no ext): RAE={r_d:.6f}  ratio={ratio_d:.4f}", flush=True)

    candidates = [
        (r_e, oof_e, te_e, ratio_e, "E_ext_lgbm"),
        (r_f, oof_f, te_f, ratio_f, "F_full_pool"),
    ]
    best_rae, best_oof, best_te, best_name = PREV_BEST, None, None, None
    for r, oof_s, te_s, ratio, nm in candidates:
        if oof_s is not None:
            flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
            beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and r < PREV_BEST) else ""
            print(f"  {nm}: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}", flush=True)
            if ratio >= COLLAPSE_THRESH and r < best_rae:
                best_rae, best_oof, best_te, best_name = r, oof_s, te_s, nm

    if best_oof is not None and best_rae < PREV_BEST:
        print(f"\n*** New best: {best_name}  RAE={best_rae:.6f} ***", flush=True)
        te_df2 = load_test()
        sub = pd.DataFrame({"Molecule Name": te_df2["name"].values, "pEC50": best_te})
        assert len(sub) == 513 and sub["pEC50"].notna().all()
        out_stem = f"nb202_ridge_ext"
        oof_path = DATA_PROCESSED / f"oof_{out_stem}.npy"
        te_path  = DATA_PROCESSED / f"te_{out_stem}.npy"
        sub_path = SUBMISSIONS / f"{out_stem}.csv"
        np.save(oof_path, best_oof)
        np.save(te_path,  best_te)
        sub.to_csv(sub_path, index=False)
        print(f"Saved: {sub_path}", flush=True)
        print(f"Test: min={best_te.min():.3f}  med={np.median(best_te):.3f}  max={best_te.max():.3f}", flush=True)
    else:
        # Save D baseline (Ridge only) as a checkpoint
        if oof_d is not None:
            out_stem = "nb202_ridge_baseline"
            np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof_d)
            np.save(DATA_PROCESSED / f"te_{out_stem}.npy",  te_d)
            sub_d = pd.DataFrame({"Molecule Name": load_test()["name"].values, "pEC50": te_d})
            sub_d.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
            print(f"Saved baseline (D): {SUBMISSIONS/f'{out_stem}.csv'}  RAE={r_d:.6f}", flush=True)
        print("No improvement over nb197.  External reservoir did not help with Ridge candidates.", flush=True)
        print("Next: wait for nb199 (QReg candidates) or nb93 Chemprop.", flush=True)


if __name__ == "__main__":
    main()
