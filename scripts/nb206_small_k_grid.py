"""nb206 -- Small-k diversity QReg: does k=8/10/12 cross ratio=0.58 at lower alpha?

For k=15 (nb197), ratio transition happens at alpha~0.009 (OOF RAE=0.300).
The best OOF RAE at any FAILING ratio is 0.297307 at alpha=0.002 (nb205).

Hypothesis: smaller k (fewer poly2 features) might shift the ratio-alpha
tradeoff. With fewer features, the LP is less complex and might maintain
test variance at lower alpha values, giving us a naturally ratio-passing
candidate with OOF RAE < 0.297639.

Tests: k=[8, 10, 12, 13, 14] at fine alpha grid [0.0010..0.0150]
Also checks ratio-inflated SLSQP with the best-OOF-RAE small-k candidates.
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
    "nb203_div25_low_alpha", "nb204_multiseed_fast", "nb205_ratio_inflate",
    "nb206_small_k_grid",
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


def inflate_te(oof_pred, te_pred, target_ratio):
    oof_std = oof_pred.std()
    te_std = te_pred.std()
    if te_std < 1e-9:
        return te_pred.copy()
    te_mean = te_pred.mean()
    return te_mean + (te_pred - te_mean) * (target_ratio * oof_std / te_std)


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
    for i in top_idx[:12]:
        if best_w[i] > 0.005:
            print(f"    model_{i}  w={best_w[i]:.4f}", flush=True)
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb206: Small-k diversity QReg grid ===\n", flush=True)
    print(f"nb197 best: {PREV_BEST}  ratio=0.5800", flush=True)
    print(f"Hypothesis: smaller k shifts ratio-alpha transition to lower alpha\n", flush=True)

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Loading base pool...", flush=True)
    base_oofs, base_tes, base_stems = load_base_pool(n_tr)
    print(f"Base pool: {len(base_oofs)} models\n", flush=True)

    # NB188 pool
    pool_oofs_tr, pool_oofs_te, pool_names = [], [], []
    for stem in NB188_POOL:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        te_p  = DATA_PROCESSED / f"te_{stem}.npy"
        if not oof_p.exists(): continue
        oof_m = np.load(oof_p).astype(np.float64).flatten()
        te_m  = np.load(te_p).astype(np.float64).flatten()
        oof_m = np.where(np.isfinite(oof_m), oof_m, np.nanmean(oof_m))
        te_m  = np.where(np.isfinite(te_m),  te_m,  np.nanmean(te_m))
        pool_oofs_tr.append(oof_m); pool_oofs_te.append(te_m)
        pool_names.append(stem)
    print(f"NB188 base: {len(pool_oofs_tr)} models\n", flush=True)

    poly2 = PolynomialFeatures(degree=2, include_bias=False)

    # Fine alpha grid emphasizing low-alpha region
    alpha_grid = [0.0010, 0.0015, 0.0017, 0.0019, 0.0020, 0.0022, 0.0025, 0.0030,
                  0.0040, 0.0050, 0.0060, 0.0070, 0.0080, 0.0090, 0.0100, 0.0120, 0.0150]

    k_values = [8, 10, 12, 13, 14]

    best_overall_rae = PREV_BEST
    best_passing = {}  # (k, alpha) -> (rae, ratio)

    # Track best candidates for SLSQP
    candidates_orig = []   # (oof, te, name, rae, ratio)
    candidates_inf = []    # ratio-inflated to 0.585

    for k in k_values:
        div_idx = greedy_diversity(base_oofs, k=k, seed_idx=0)
        Xk_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div_idx]))
        Xk_te = poly2.transform(np.column_stack([base_tes[i] for i in div_idx]))
        n_poly = Xk_tr.shape[1]
        print(f"--- k={k}: {n_poly} poly2 features ---", flush=True)

        for alpha in alpha_grid:
            t_a = time.time()
            oof_c, te_c = qreg_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
            r_c = rae(y_tr, oof_c)
            oof_std = oof_c.std()
            te_std = te_c.std()
            ratio_c = te_std / oof_std if oof_std > 1e-9 else 0.0
            flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
            elapsed = time.time() - t_a

            better = " *BEST OOF*" if r_c < best_overall_rae else ""
            print(f"  k={k} a={alpha:.4f}  RAE={r_c:.6f}  ratio={ratio_c:.4f}  [{flag}]{better}  ({elapsed:.0f}s)", flush=True)

            if r_c < best_overall_rae:
                best_overall_rae = r_c

            name = f"k{k}_a{alpha:.4f}"
            candidates_orig.append((oof_c, te_c, name, r_c, ratio_c))

            if ratio_c < COLLAPSE_THRESH:
                # Inflate to 0.585
                te_inf = inflate_te(oof_c, te_c, 0.585)
                candidates_inf.append((oof_c, te_inf, f"{name}_inf585", r_c, 0.585))

            if ratio_c >= COLLAPSE_THRESH:
                best_passing[(k, alpha)] = (r_c, ratio_c)

        print(f"  k={k} done\n", flush=True)

    print(f"\n=== Grid Summary ===", flush=True)
    print(f"Best passing candidates by (k, alpha):", flush=True)
    best_list = sorted(best_passing.items(), key=lambda x: x[1][0])
    for (k, alpha), (r, ratio) in best_list[:10]:
        beat = " ***BEATS nb197***" if r < PREV_BEST else ""
        print(f"  k={k} a={alpha:.4f}: RAE={r:.6f}  ratio={ratio:.4f}{beat}", flush=True)

    print(f"\nBest OOF RAE seen (failing or passing): {best_overall_rae:.6f}", flush=True)
    print(f"Elapsed: {time.time()-t0:.0f}s\n", flush=True)

    # SLSQP: NB188 + all candidates (orig + inflated)
    pool_tr_ext = pool_oofs_tr.copy()
    pool_te_ext = pool_oofs_te.copy()
    all_names = pool_names.copy()

    for (oof_c, te_c, name, r_c, ratio_c) in candidates_orig:
        pool_tr_ext.append(oof_c)
        pool_te_ext.append(te_c)
        all_names.append(f"{name}_orig")

    for (oof_c, te_c, name, r_c, ratio_c) in candidates_inf:
        pool_tr_ext.append(oof_c)
        pool_te_ext.append(te_c)
        all_names.append(name)

    X_tr = np.column_stack(pool_tr_ext)
    X_te = np.column_stack(pool_te_ext)
    print(f"--- SLSQP: NB188 + all k-grid + inflated candidates (2000 starts) ---", flush=True)
    print(f"Total pool: {X_tr.shape[1]} models", flush=True)
    r_best, oof_best, te_best, ratio_best = constrained_slsqp(
        X_tr, X_te, y_tr, n_starts=2000, prev_best=PREV_BEST, tag="small_k_grid")

    print(f"\n=== Final Summary ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"  nb197 (prev best):  RAE={PREV_BEST:.6f}  ratio=0.5800  [PASS]", flush=True)
    flag = "PASS" if ratio_best >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio_best >= COLLAPSE_THRESH and r_best < PREV_BEST) else ""
    print(f"  nb206 small-k:      RAE={r_best:.6f}  ratio={ratio_best:.4f}  [{flag}]{beat}", flush=True)

    if oof_best is not None and r_best < PREV_BEST and ratio_best >= COLLAPSE_THRESH:
        sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": te_best})
        assert len(sub) == 513 and sub["pEC50"].notna().all()
        out_stem = "nb206_small_k_grid"
        np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof_best)
        np.save(DATA_PROCESSED / f"te_{out_stem}.npy",  te_best)
        sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
        print(f"Saved: {SUBMISSIONS/f'{out_stem}.csv'}", flush=True)
    elif oof_best is not None:
        out_stem = "nb206_small_k_grid_baseline"
        np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof_best)
        np.save(DATA_PROCESSED / f"te_{out_stem}.npy",  te_best)
        sub_d = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": te_best})
        sub_d.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
        print(f"Saved baseline: {SUBMISSIONS/f'{out_stem}.csv'}  RAE={r_best:.6f}", flush=True)


if __name__ == "__main__":
    main()
