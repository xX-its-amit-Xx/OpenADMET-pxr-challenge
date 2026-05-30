"""nb203 -- QReg div25+ at low alpha: the gap in nb197's search.

nb197 tested div25 only at alpha=[0.008, 0.01, 0.015, 0.02, 0.03].
nb202 showed Ridge div25 at alpha=0.001 has ratio=0.6187 (PASS!) while
Ridge div15/div20 fail at all alphas.

Key insight: div25 selection naturally produces high ratio regardless of
regularization (diversity drives ratio, not regularization). QReg should
show the same pattern.

If QReg div25 at alpha=0.001-0.007 gives ratio >= 0.58 AND RAE < 0.300
(QReg's L1 loss makes it better aligned with the MAE metric than Ridge),
we might find candidates with RAE < 0.297639 at ratio >= 0.58.

This is a targeted fill of the gap in nb197's div25 alpha grid.
Also test div30 and div35 if base pool is large enough.

Runtime: ~30-90 min depending on QReg LP solve time (14-21 QReg candidates).
"""
import os, sys, warnings, time
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
    "nb203_div25_low_alpha",
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

    for i in range(n_starts):
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
    print("=== nb203: QReg div25/30/35 at low alpha (nb197's gap) ===\n", flush=True)
    print(f"nb197 best: {PREV_BEST}  ratio=0.5800", flush=True)
    print(f"nb197 tested div25 at [0.008, 0.01, 0.015, 0.02, 0.03] only", flush=True)
    print(f"Hypothesis: div25 at alpha=0.001-0.007 gives ratio>=0.58 + better RAE\n", flush=True)

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load base pool
    print("Loading base pool...", flush=True)
    base_oofs, base_tes, base_stems = load_base_pool(n_tr)

    # Add nb183 if not already in pool
    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists() and not any(s == "nb183_qreg_poly10" for s in base_stems):
        o183 = np.load(oof183_p).astype(np.float64).flatten()
        t183 = np.load(te183_p).astype(np.float64).flatten()
        o183 = np.where(np.isfinite(o183), o183, np.nanmean(o183))
        t183 = np.where(np.isfinite(t183), t183, np.nanmean(t183))
        base_oofs.insert(0, o183); base_tes.insert(0, t183)
        base_stems.insert(0, "nb183_qreg_poly10")

    n_pool = len(base_oofs)
    print(f"Base pool: {n_pool} models", flush=True)

    # Load NB188_POOL explicitly
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
    print(f"NB188 pool models: {len(pool_oofs_tr)}", flush=True)

    poly2 = PolynomialFeatures(degree=2, include_bias=False)

    # --- Targeted div25/30/35 at low alpha ---
    # The key gap: nb197 only tested div25 at alpha>=0.008
    # From Ridge analysis: div25 at alpha=0.001 gives ratio=0.6187 (PASS!)
    # QReg at same diversity should give similar or higher ratio

    # Alpha grid: low alphas (0.001-0.007, gap in nb197) + sparse high for reference
    alpha_low  = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
    alpha_ref  = [0.008, 0.009, 0.01]  # nb197 tested these for div25

    all_alphas = alpha_low + alpha_ref

    k_values = [25]  # Primary target; add 30, 35 if pool is large enough
    if n_pool >= 30: k_values.append(30)
    if n_pool >= 35: k_values.append(35)
    print(f"Testing div-k for k={k_values}", flush=True)
    print(f"Alpha range: {all_alphas}\n", flush=True)

    all_cand_oofs, all_cand_tes, all_cand_names = [], [], []

    for k in k_values:
        div_idx = greedy_diversity(base_oofs, k=k, seed_idx=0)
        Xk_tr_raw = np.column_stack([base_oofs[i] for i in div_idx])
        Xk_te_raw = np.column_stack([base_tes[i] for i in div_idx])
        Xk_tr = poly2.fit_transform(Xk_tr_raw)
        Xk_te = poly2.transform(Xk_te_raw)
        n_poly_feats = Xk_tr.shape[1]
        print(f"div{k}: {n_poly_feats} poly2 features, "
              f"LP size ~{int(n_tr*0.8) * (n_poly_feats+1):,}", flush=True)

        t_k = time.time()
        for alpha in all_alphas:
            t_a = time.time()
            oof_c, te_c = qreg_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
            r_c = rae(y_tr, oof_c)
            ratio_c = te_c.std() / oof_c.std()
            flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
            gap = "" if alpha < 0.008 else " [nb197]"
            print(f"  div{k} a={alpha:.4f}  RAE={r_c:.6f}  ratio={ratio_c:.4f}  [{flag}]{gap}  ({time.time()-t_a:.0f}s)", flush=True)
            all_cand_oofs.append(oof_c)
            all_cand_tes.append(te_c)
            all_cand_names.append(f"div{k}_a{alpha:.4f}")
        print(f"  div{k} done in {time.time()-t_k:.0f}s", flush=True)

    print(f"\nTotal new candidates: {len(all_cand_oofs)}", flush=True)
    print(f"Elapsed so far: {time.time()-t0:.0f}s\n", flush=True)

    # Quick check: how many pass individually?
    passing = [(nm, rae(y_tr, oof), te.std()/oof.std())
               for nm, oof, te in zip(all_cand_names, all_cand_oofs, all_cand_tes)
               if te.std()/oof.std() >= COLLAPSE_THRESH]
    print(f"Candidates passing ratio individually: {len(passing)}", flush=True)
    for nm, r, ratio in sorted(passing, key=lambda x: x[1]):
        print(f"  {nm}: RAE={r:.6f}  ratio={ratio:.4f}", flush=True)

    # --- SLSQP: NB188 base + new QReg candidates (same as nb197 approach) ---
    print(f"\n--- SLSQP: NB188 + QReg div25+ low-alpha (2000 starts) ---", flush=True)
    X_tr = np.column_stack(pool_oofs_tr + all_cand_oofs)
    X_te = np.column_stack(pool_oofs_te + all_cand_tes)
    r_best, oof_best, te_best, ratio_best = constrained_slsqp(
        X_tr, X_te, y_tr, n_starts=2000, prev_best=PREV_BEST, tag="div25+_low_alpha")

    # Also run with nb197's div25 candidates appended (for context)
    print(f"\n--- Also: full div25 alpha grid (low + ref, 2000 starts) ---", flush=True)
    print(f"  (Same as above but confirming nb197 reference points)", flush=True)

    print(f"\n=== Summary ({time.time()-t0:.0f}s total) ===", flush=True)
    print(f"  nb197 (prev best): RAE={PREV_BEST}  ratio=0.5800  [PASS]", flush=True)
    flag = "PASS" if ratio_best >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio_best >= COLLAPSE_THRESH and r_best < PREV_BEST) else ""
    print(f"  nb203 (div25+, low alpha): RAE={r_best:.6f}  ratio={ratio_best:.4f}  [{flag}]{beat}", flush=True)

    if oof_best is not None and r_best < PREV_BEST and ratio_best >= COLLAPSE_THRESH:
        te_out = load_test()
        sub = pd.DataFrame({"Molecule Name": te_out["name"].values, "pEC50": te_best})
        assert len(sub) == 513 and sub["pEC50"].notna().all()
        out_stem = "nb203_div25_low_alpha"
        np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof_best)
        np.save(DATA_PROCESSED / f"te_{out_stem}.npy",  te_best)
        sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
        print(f"Saved: {SUBMISSIONS/f'{out_stem}.csv'}", flush=True)
        print(f"Test: min={te_best.min():.3f}  med={np.median(te_best):.3f}  max={te_best.max():.3f}", flush=True)
    elif r_best >= PREV_BEST or r_best == 1e9:
        # Save best low-alpha candidate as checkpoint
        best_passing = None
        for nm, oof_c, te_c in zip(all_cand_names, all_cand_oofs, all_cand_tes):
            ratio_c = te_c.std() / oof_c.std()
            r_c = rae(y_tr, oof_c)
            if ratio_c >= COLLAPSE_THRESH:
                if best_passing is None or r_c < best_passing[0]:
                    best_passing = (r_c, oof_c, te_c, nm)
        if best_passing is not None:
            r_c, oof_c, te_c, nm = best_passing
            print(f"\nBest individual passing: {nm}  RAE={r_c:.6f}", flush=True)
            np.save(DATA_PROCESSED / "oof_nb203_best_cand.npy", oof_c)
            np.save(DATA_PROCESSED / "te_nb203_best_cand.npy",  te_c)


if __name__ == "__main__":
    main()
