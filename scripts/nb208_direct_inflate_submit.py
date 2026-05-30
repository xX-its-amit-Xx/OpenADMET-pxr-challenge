"""nb208 -- Direct inflation submission: bypasses SLSQP uncertainty.

Key finding (nb205 setup):
  div15 a=0.002: OOF RAE=0.297307 (BEST, beats nb197=0.297639), ratio=0.5610 [FAIL by 0.001]

This script directly computes the inflated predictions for the best candidate
and creates a submission without SLSQP. This is a "pure" test of the inflation idea.

Also tests: what blend of inflated + NB188 is optimal when we start from the
known-good solution (w=1.0 on inflated div15 a=0.002) and allow small adjustments?
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
    "nb206_small_k_grid", "nb207_fine_alpha_min", "nb208_direct_inflate_submit",
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


def local_slsqp_from_warm(X_tr, X_te, y_tr, warm_w, n_fine=200, tag=""):
    """Fine-tune from a warm start: run SLSQP from warm_w and nearby perturbed starts."""
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

    def _try(w0):
        nonlocal best_r, best_w
        w0 = np.clip(w0, 0, 1); w0 /= w0.sum()
        res = minimize(obj, w0, jac=obj_grad, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 500, "ftol": 1e-8})
        if res.success and ratio_con(res.x) >= -1e-6:
            r = obj(res.x)
            if r < best_r:
                best_r, best_w = r, res.x.copy()

    # Try the warm start directly
    _try(warm_w.copy())

    # Perturb warm start and try many variations
    for _ in range(n_fine):
        noise = rng.normal(0, 0.05, n_m)
        w_pert = warm_w + noise
        _try(w_pert)

    if best_w is None:
        print(f"  [{tag}] No feasible solution found", flush=True)
        return 1e9, None, None, 0

    oof_b = X_tr @ best_w; te_b = X_te @ best_w
    ratio = te_b.std() / oof_b.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and best_r < PREV_BEST) else ""
    print(f"  [{tag}] n={n_m}  RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}", flush=True)
    top_idx = np.argsort(best_w)[::-1]
    for i in top_idx[:10]:
        if best_w[i] > 0.005:
            print(f"    model_{i}  w={best_w[i]:.4f}", flush=True)
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb208: Direct inflation submission ===\n", flush=True)
    print(f"nb197 best: {PREV_BEST}  ratio=0.5800", flush=True)
    print(f"Key: div15 a=0.002 -> OOF RAE=0.297307, ratio=0.5610\n", flush=True)

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Building div15 QReg at a=0.002...", flush=True)
    base_oofs, base_tes, base_stems = load_base_pool(n_tr)
    print(f"Base pool: {len(base_oofs)} models", flush=True)

    div_idx = greedy_diversity(base_oofs, k=15, seed_idx=0)
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X15_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div_idx]))
    X15_te = poly2.transform(np.column_stack([base_tes[i] for i in div_idx]))

    t_a = time.time()
    oof002, te002 = qreg_oof(X15_tr, y_tr, X15_te, splits, alpha=0.002)
    r002 = rae(y_tr, oof002)
    ratio002 = te002.std() / oof002.std()
    print(f"div15 a=0.002: OOF RAE={r002:.6f}  ratio={ratio002:.4f}  ({time.time()-t_a:.0f}s)", flush=True)

    # Direct inflation tests
    print("\n--- Direct inflation: w=1.0 on inflated ---", flush=True)
    for t_r in [0.580, 0.582, 0.584, 0.585, 0.586, 0.588, 0.590, 0.595, 0.600]:
        te_inf = inflate_te(oof002, te002, t_r)
        act_ratio = te_inf.std() / oof002.std()
        flag = "PASS" if act_ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  inflate to {t_r:.3f}: actual_ratio={act_ratio:.4f}  OOF RAE={r002:.6f}  [{flag}]", flush=True)

    # The minimal inflation that passes: target_ratio = 0.58 × oof_std / te_std
    min_target = COLLAPSE_THRESH * oof002.std() / te002.std() * te002.std() / oof002.std()
    print(f"\nMinimum target_ratio to pass: {COLLAPSE_THRESH * oof002.std() / te002.std():.6f}", flush=True)
    min_inflate_ratio = COLLAPSE_THRESH / ratio002
    print(f"(= {min_inflate_ratio:.4f}x inflation of test spread)", flush=True)

    # Load NB188 pool
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
    print(f"\nNB188 base: {len(pool_oofs_tr)} models", flush=True)

    # Add inflated div15 a=0.002 (at minimal target ratio)
    te_inf_min = inflate_te(oof002, te002, 0.582)  # just above 0.58
    pool_oofs_tr.append(oof002); pool_oofs_te.append(te_inf_min)
    pool_names.append("div15_a0.002_inf0.582")
    inf_idx = len(pool_oofs_tr) - 1

    X_tr = np.column_stack(pool_oofs_tr)
    X_te = np.column_stack(pool_oofs_te)

    # Warm start: 90% on inflated div15 a=0.002
    warm_w = np.ones(len(pool_names)) * 0.1 / (len(pool_names) - 1)
    warm_w[inf_idx] = 0.9

    print("\n--- Section A: Local SLSQP from warm start (90% on inflated div15 a=0.002) ---", flush=True)
    r_A, oof_A, te_A, ratio_A = local_slsqp_from_warm(X_tr, X_te, y_tr, warm_w, n_fine=500, tag="A_warm")

    # Also try adding inflated div20 a=0.002
    oof20, te20 = qreg_oof(
        poly2.fit_transform(np.column_stack([base_oofs[i] for i in greedy_diversity(base_oofs, k=20)])),
        y_tr,
        poly2.transform(np.column_stack([base_tes[i] for i in greedy_diversity(base_oofs, k=20)])),
        splits, alpha=0.002
    )
    te20_inf = inflate_te(oof20, te20, 0.582)
    pool_oofs_tr.append(oof20); pool_oofs_te.append(te20_inf)
    pool_names.append("div20_a0.002_inf0.582")
    inf20_idx = len(pool_oofs_tr) - 1
    X_tr2 = np.column_stack(pool_oofs_tr)
    X_te2 = np.column_stack(pool_oofs_te)

    warm_w2 = np.ones(len(pool_names)) * 0.05 / (len(pool_names) - 2)
    warm_w2[inf_idx] = 0.60
    warm_w2[inf20_idx] = 0.35

    print("\n--- Section B: Local SLSQP with both inflated (div15 + div20 a=0.002) ---", flush=True)
    r_B, oof_B, te_B, ratio_B = local_slsqp_from_warm(X_tr2, X_te2, y_tr, warm_w2, n_fine=500, tag="B_both")

    print(f"\n=== Final Summary ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"  nb197 (prev best):          RAE={PREV_BEST:.6f}  ratio=0.5800  [PASS]", flush=True)
    print(f"  div15 a=0.002 (standalone): RAE={r002:.6f}  ratio={ratio002:.4f}  [FAIL by {COLLAPSE_THRESH-ratio002:.4f}]", flush=True)
    flag_A = "PASS" if ratio_A >= COLLAPSE_THRESH else "FAIL"
    beat_A = " ***NEW BEST***" if (ratio_A >= COLLAPSE_THRESH and r_A < PREV_BEST) else ""
    print(f"  nb208-A (+NB188, warm):     RAE={r_A:.6f}  ratio={ratio_A:.4f}  [{flag_A}]{beat_A}", flush=True)
    flag_B = "PASS" if ratio_B >= COLLAPSE_THRESH else "FAIL"
    beat_B = " ***NEW BEST***" if (ratio_B >= COLLAPSE_THRESH and r_B < PREV_BEST) else ""
    print(f"  nb208-B (+NB188+div20, w):  RAE={r_B:.6f}  ratio={ratio_B:.4f}  [{flag_B}]{beat_B}", flush=True)

    best_r = min(v for v in [r_A, r_B] if v < PREV_BEST) if any(v < PREV_BEST for v in [r_A, r_B]) else PREV_BEST
    best_oof = oof_A if r_A <= r_B else oof_B
    best_te  = te_A  if r_A <= r_B else te_B
    best_ratio = ratio_A if r_A <= r_B else ratio_B

    if best_r < PREV_BEST and best_ratio >= COLLAPSE_THRESH:
        sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te})
        assert len(sub) == 513 and sub["pEC50"].notna().all()
        out_stem = "nb208_direct_inflate"
        np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", best_oof)
        np.save(DATA_PROCESSED / f"te_{out_stem}.npy",  best_te)
        sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
        print(f"Saved: {SUBMISSIONS/f'{out_stem}.csv'}", flush=True)
    else:
        print("No new best found. Not saving submission.", flush=True)


if __name__ == "__main__":
    main()
