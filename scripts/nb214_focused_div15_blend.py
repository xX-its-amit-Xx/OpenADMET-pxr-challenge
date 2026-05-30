"""nb214 -- Focused SLSQP: best div15 QReg candidates + nb212 pool.

Rebuilds div15 QReg (a=0.0013-0.0021, 9 alphas) using the same 47-model
base pool and div15 greedy-diversity subset as nb207, then blends with
the nb212 pool (nb211, nb205, nb157, nb167, nb169).

~50 models vs nb207's 274 -> SLSQP ratio-constraint Jacobian is 5x faster.
Uses nb212 weights as a warm start to exploit the known good solution.
Target: RAE < 0.296172 (nb212 current best).
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
PREV_BEST = 0.296172   # nb212 current best

t0 = time.time()

# Same exclusion list as nb207 to reproduce the same base pool
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
    "nb206_small_k_grid", "nb207_fine_alpha_min", "nb208_chemprop_blend",
    "nb209_diversity_inflated", "nb210_grand_v17",
    "nb211_div15_chemprop_blend", "nb212_nb211_blend",
    "nb213_chemberta_mtr", "nb213_chemberta_zinc",
    "nb213_zinc_lgbm", "nb213_zinc_mlp",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
    "nb108_grand_v2", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
}

# nb212 pool: models explicitly included in the best blend
NB212_POOL = [
    "nb211_div15_chemprop_blend",
    "nb205_ratio_inflate",
    "nb157_optuna_lgbm_mae",
    "nb167_xgboost_mae",
    "nb169_rf_et_mae",
]
# nb212 weights (ordered as NB212_POOL)
NB212_WEIGHTS = [0.5276, 0.2686, 0.1383, 0.0379, 0.0277]


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


def constrained_slsqp(X_tr, X_te, y_tr, n_starts=5000, prev_best=PREV_BEST, tag="",
                      warm_weight_vecs=None):
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

    def _try_start(w0):
        nonlocal best_r, best_w
        res = minimize(obj, w0, jac=obj_grad, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-8})
        if res.success and ratio_con(res.x) >= -1e-6:
            r = obj(res.x)
            if r < best_r:
                best_r, best_w = r, res.x.copy()

    # Warm starts: known good weight vectors
    if warm_weight_vecs is not None:
        for w0 in warm_weight_vecs:
            if len(w0) == n_m:
                _try_start(w0)

    # Targeted: put most weight on individual high-ratio models
    for i in range(n_m):
        for main_w in [0.5, 0.7, 0.9]:
            w0 = np.ones(n_m) * (1.0 - main_w) / max(n_m - 1, 1)
            w0[i] = main_w
            w0 = np.clip(w0, 0, 1); w0 /= w0.sum()
            _try_start(w0)

    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(n_m))
        _try_start(w0)

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
    print("=== nb214: Focused div15 QReg + nb212 pool SLSQP ===\n", flush=True)
    print(f"nb212 best: {PREV_BEST}  ratio=0.5800\n", flush=True)

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Loading base pool (same as nb207)...", flush=True)
    base_oofs, base_tes, base_stems = load_base_pool(n_tr)

    # Add nb183 back (excluded from META_STEMS but needed for greedy diversity)
    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists() and not any(s == "nb183_qreg_poly10" for s in base_stems):
        o183 = np.load(oof183_p).astype(np.float64).flatten()
        t183 = np.load(te183_p).astype(np.float64).flatten()
        o183 = np.where(np.isfinite(o183), o183, np.nanmean(o183))
        t183 = np.where(np.isfinite(t183), t183, np.nanmean(t183))
        if len(o183) == n_tr:
            base_oofs.insert(0, o183); base_tes.insert(0, t183)
            base_stems.insert(0, "nb183_qreg_poly10")
    print(f"Base pool: {len(base_oofs)} models\n", flush=True)

    # --- Build div15 QReg candidates ---
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    div15_idx = greedy_diversity(base_oofs, k=15, seed_idx=0)
    X15_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div15_idx]))
    X15_te = poly2.transform(np.column_stack([base_tes[i] for i in div15_idx]))
    print(f"--- div15: {X15_tr.shape[1]} poly2 features, {len(div15_idx)} models ---", flush=True)

    # Fine alpha range around the minimum (a=0.0017 was best in nb207)
    alpha_range = [0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016,
                   0.0017, 0.0018, 0.0019, 0.0020, 0.0021, 0.0022]
    inflate_targets = [0.580, 0.582, 0.585, 0.590]

    pool_oofs_tr, pool_oofs_te, pool_names = [], [], []
    inf_warm_vecs = []   # weight vectors for warm-starting on best inflated candidates

    best_raw_rae = 1e9
    for alpha in alpha_range:
        t_a = time.time()
        oof_c, te_c = qreg_oof(X15_tr, y_tr, X15_te, splits, alpha)
        r_c = rae(y_tr, oof_c)
        ratio_c = te_c.std() / oof_c.std() if oof_c.std() > 1e-9 else 0.0
        flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
        note = ""
        if r_c < best_raw_rae:
            best_raw_rae = r_c
            note = f" *best so far*"
        print(f"  div15 a={alpha:.4f}  RAE={r_c:.6f}  ratio={ratio_c:.4f}  [{flag}]{note}  ({time.time()-t_a:.0f}s)", flush=True)

        pool_oofs_tr.append(oof_c); pool_oofs_te.append(te_c)
        pool_names.append(f"div15_a{alpha:.4f}_orig")

        if ratio_c < COLLAPSE_THRESH:
            inf_idxs = []
            for t_r in inflate_targets:
                te_inf = inflate_te(oof_c, te_c, t_r)
                pool_oofs_tr.append(oof_c); pool_oofs_te.append(te_inf)
                inf_idxs.append(len(pool_oofs_tr) - 1)
                pool_names.append(f"div15_a{alpha:.4f}_inf{t_r:.3f}")
            # Warm start: high weight on the best inflated version of good candidates
            if r_c < 0.2976:   # only for the very best alphas
                for inf_i in inf_idxs:
                    n_pool_so_far = len(pool_oofs_tr)
                    # will be extended later; store partial index for now
                    inf_warm_vecs.append(inf_idxs)

    print(f"\nDiv15 QReg done ({time.time()-t0:.0f}s). Best raw RAE: {best_raw_rae:.6f}", flush=True)
    n_div15 = len(pool_oofs_tr)
    print(f"div15 pool size: {n_div15}", flush=True)

    # --- Add nb212 pool models ---
    print("\nLoading nb212 pool models...", flush=True)
    nb212_start_idx = len(pool_oofs_tr)
    nb212_loaded_weights = []
    for stem, w212 in zip(NB212_POOL, NB212_WEIGHTS):
        op = DATA_PROCESSED / f"oof_{stem}.npy"
        tp = DATA_PROCESSED / f"te_{stem}.npy"
        if not op.exists():
            print(f"  MISSING: {stem}", flush=True); continue
        oof_m = np.load(op).astype(np.float64).flatten()
        te_m  = np.load(tp).astype(np.float64).flatten()
        oof_m = np.where(np.isfinite(oof_m), oof_m, np.nanmean(oof_m))
        te_m  = np.where(np.isfinite(te_m),  te_m,  np.nanmean(te_m))
        r_m = rae(y_tr, oof_m)
        ratio_m = te_m.std() / oof_m.std()
        print(f"  {stem}: RAE={r_m:.6f}  ratio={ratio_m:.4f}", flush=True)
        pool_oofs_tr.append(oof_m); pool_oofs_te.append(te_m)
        pool_names.append(stem)
        nb212_loaded_weights.append(w212)

    n_total = len(pool_oofs_tr)
    print(f"\nTotal pool: {n_total} models", flush=True)

    X_tr = np.column_stack(pool_oofs_tr)
    X_te = np.column_stack(pool_oofs_te)

    # --- Build warm starts ---
    warm_vecs = []

    # nb212 weights mapped onto current pool (zeros for div15 candidates)
    if nb212_loaded_weights and len(nb212_loaded_weights) == len(NB212_POOL):
        w_nb212 = np.zeros(n_total)
        for j, w in enumerate(nb212_loaded_weights):
            idx = nb212_start_idx + j
            if idx < n_total:
                w_nb212[idx] = w
        if w_nb212.sum() > 0:
            w_nb212 /= w_nb212.sum()
            warm_vecs.append(w_nb212)
            print("Added nb212 weights as warm start.", flush=True)

    # High-weight on best div15 inflated + nb205
    nb205_idx = None
    for j, nm in enumerate(pool_names):
        if nm == "nb205_ratio_inflate":
            nb205_idx = j; break
    if nb205_idx is not None:
        best_div15_alpha_idx = alpha_range.index(0.0017)  # best alpha
        # Find inflated indices for this alpha
        for j, nm in enumerate(pool_names[:n_div15]):
            if f"div15_a0.0017_inf0.582" in nm:
                w0 = np.zeros(n_total)
                w0[j] = 0.6; w0[nb205_idx] = 0.4
                w0 /= w0.sum()
                warm_vecs.append(w0)
                break

    print(f"\n--- SLSQP: {n_total} models, 5000 starts + {len(warm_vecs)} warm starts ---", flush=True)
    r_best, oof_best, te_best, ratio_best = constrained_slsqp(
        X_tr, X_te, y_tr, n_starts=5000, prev_best=PREV_BEST, tag="nb214",
        warm_weight_vecs=warm_vecs)

    flag = "PASS" if ratio_best >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio_best >= COLLAPSE_THRESH and r_best < PREV_BEST) else ""
    print(f"\n=== Summary ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"  nb212 (prev best): RAE={PREV_BEST:.6f}  ratio=0.5800  [PASS]", flush=True)
    print(f"  nb214 SLSQP:       RAE={r_best:.6f}  ratio={ratio_best:.4f}  [{flag}]{beat}", flush=True)
    if pool_names:
        print("\nPool names:", flush=True)
        for i, nm in enumerate(pool_names):
            print(f"  {i:3d}  {nm}", flush=True)

    if oof_best is not None:
        out_stem = "nb214_div15_nb212_blend"
        np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof_best)
        np.save(DATA_PROCESSED / f"te_{out_stem}.npy",  te_best)
        smiles = te_df["smiles"].values if "smiles" in te_df.columns else te_df.iloc[:,0].values
        names  = te_df["name"].values if "name" in te_df.columns else te_df.iloc[:,1].values
        sub = pd.DataFrame({"SMILES": smiles, "Molecule Name": names, "pEC50": te_best})
        assert len(sub) == 513 and sub["pEC50"].notna().all()
        sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
        if ratio_best >= COLLAPSE_THRESH and r_best < PREV_BEST:
            print(f"Saved NEW BEST: {SUBMISSIONS/f'{out_stem}.csv'}", flush=True)
        else:
            print(f"Saved baseline: {SUBMISSIONS/f'{out_stem}.csv'}  (did not beat nb212)", flush=True)


if __name__ == "__main__":
    main()
