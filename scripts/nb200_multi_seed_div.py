"""nb200 -- Multi-seed greedy diversity for constrained SLSQP.

nb197 best: 0.297639  ratio=0.5800  (seed_idx=0 only, k=15)

Unexplored space: nb197 only used seed_idx=0 (nb183) for greedy diversity.
Different seeds give different sets of 15 diverse models with potentially
different ratio and OOF RAE characteristics.

Key question: does any seed produce a div-15 candidate where the first-passing
alpha (ratio >= 0.58) gives OOF RAE < 0.297639?

Approach:
  For each seed_idx in 0..24:
    Build div-15 QReg poly candidates at 25 alpha values
    Record (best OOF RAE, ratio at that alpha) and (OOF RAE at first passing alpha)

  Pool all candidates from all seeds + nb188 base models.
  Run constrained SLSQP (2000 starts) on the combined pool.

  Also try div-12, div-13 at seed_idx variations to broaden search.
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


def main():
    print("=== nb200: Multi-seed greedy diversity ===\n")
    print(f"nb197 best: {PREV_BEST}  ratio=0.5800\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load base pool
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
    print(f"Base pool size: {len(base_oofs)} models\n")

    poly2 = PolynomialFeatures(degree=2, include_bias=False)

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

    # Alpha grid
    # Dense around the known sweet spot (0.001-0.003), plus ratio-provider region
    alpha_grid = sorted(set([round(a, 5) for a in
        list(np.linspace(0.0008, 0.0035, 20)) +
        [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03]]))

    # --- Multi-seed sweep ---
    n_seeds = min(25, len(base_oofs))  # try up to 25 seeds
    print(f"=== Sweeping {n_seeds} seeds x {len(alpha_grid)} alphas at k=15 ===")
    print("(Printing best alpha per seed and first-passing alpha)")
    print()

    all_cand_oofs, all_cand_tes, all_cand_names = [], [], []
    seed_summaries = []

    for seed_idx in range(n_seeds):
        div_idx = greedy_diversity(base_oofs, k=15, seed_idx=seed_idx)
        Xk_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div_idx]))
        Xk_te = poly2.transform(np.column_stack([base_tes[i] for i in div_idx]))

        best_rae_this = 1e9
        best_alpha_this = None
        best_ratio_this = 0
        first_pass_rae = None
        first_pass_alpha = None

        for alpha in alpha_grid:
            oof_c, te_c = qreg_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
            r_c = rae(y_tr, oof_c)
            ratio_c = te_c.std() / oof_c.std()

            # Keep all candidates for the pool
            all_cand_oofs.append(oof_c)
            all_cand_tes.append(te_c)
            all_cand_names.append(f"s{seed_idx}_a{alpha:.5f}")

            if r_c < best_rae_this:
                best_rae_this = r_c
                best_alpha_this = alpha
                best_ratio_this = ratio_c

            if first_pass_rae is None and ratio_c >= COLLAPSE_THRESH:
                first_pass_rae = r_c
                first_pass_alpha = alpha

        flag_best = "PASS" if best_ratio_this >= COLLAPSE_THRESH else "FAIL"
        pass_str = (f"first_pass: a={first_pass_alpha:.5f} RAE={first_pass_rae:.6f}"
                    if first_pass_rae else "no passing alpha in grid")
        print(f"  seed={seed_idx:2d} stem={base_stems[seed_idx][:25]:25s} "
              f"best_RAE={best_rae_this:.6f} ratio={best_ratio_this:.4f} [{flag_best}] | "
              f"{pass_str}")
        seed_summaries.append((seed_idx, best_rae_this, best_ratio_this,
                                first_pass_rae, first_pass_alpha))

    print(f"\nTotal candidates collected: {len(all_cand_oofs)}")

    # Find any single candidate with both good RAE and passing ratio
    print("\n--- Candidates with ratio >= 0.58 and RAE < 0.300 ---")
    for nm, oof_s, te_s in zip(all_cand_names, all_cand_oofs, all_cand_tes):
        r_c = rae(y_tr, oof_s)
        ratio_c = te_s.std() / oof_s.std()
        if ratio_c >= COLLAPSE_THRESH and r_c < 0.300:
            print(f"  {nm}: RAE={r_c:.6f} ratio={ratio_c:.4f} [PASS]")

    # --- SLSQP with full multi-seed pool ---
    print(f"\n--- Constrained SLSQP: nb188 + all multi-seed candidates (2000 starts) ---")
    X_full_tr = np.column_stack(pool_oofs_tr + all_cand_oofs)
    X_full_te = np.column_stack(pool_oofs_te + all_cand_tes)
    r_full, oof_full, te_full, ratio_full = constrained_slsqp(
        X_full_tr, X_full_te, y_tr, n_starts=2000, prev_best=PREV_BEST)

    # Also try: top-200 by OOF RAE only (filter redundant candidates)
    print(f"\n--- Constrained SLSQP: nb188 + top-150 by OOF RAE (2000 starts) ---")
    rae_scores = [rae(y_tr, o) for o in all_cand_oofs]
    top150_idx = np.argsort(rae_scores)[:150]
    top_oofs = [all_cand_oofs[i] for i in top150_idx]
    top_tes  = [all_cand_tes[i] for i in top150_idx]
    X_top_tr = np.column_stack(pool_oofs_tr + top_oofs)
    X_top_te = np.column_stack(pool_oofs_te + top_tes)
    r_top, oof_top, te_top, ratio_top = constrained_slsqp(
        X_top_tr, X_top_te, y_tr, n_starts=2000, prev_best=PREV_BEST)

    # Also try: candidates with ratio >= 0.57 (near-passing) + high-ratio seeds
    print(f"\n--- Constrained SLSQP: nb188 + near-pass + high-ratio (2000 starts) ---")
    sel_oofs, sel_tes = [], []
    for o, t in zip(all_cand_oofs, all_cand_tes):
        r_c = rae(y_tr, o)
        ratio_c = t.std() / o.std()
        # Include if: good RAE (< 0.302) OR high ratio (> 0.62)
        if r_c < 0.302 or ratio_c > 0.62:
            sel_oofs.append(o); sel_tes.append(t)
    print(f"  Selected {len(sel_oofs)} candidates (good RAE or high ratio)")
    if sel_oofs:
        X_sel_tr = np.column_stack(pool_oofs_tr + sel_oofs)
        X_sel_te = np.column_stack(pool_oofs_te + sel_tes)
        r_sel, oof_sel, te_sel, ratio_sel = constrained_slsqp(
            X_sel_tr, X_sel_te, y_tr, n_starts=2000, prev_best=PREV_BEST)
    else:
        r_sel, oof_sel, te_sel, ratio_sel = 1e9, None, None, 0

    print("\n=== Summary ===")
    print(f"  nb197 (prev best): RAE={PREV_BEST}  ratio=0.5800  [PASS]")
    candidates_final = [
        (r_full, oof_full, te_full, ratio_full, "full_multiseed"),
        (r_top,  oof_top,  te_top,  ratio_top,  "top150_multiseed"),
        (r_sel,  oof_sel,  te_sel,  ratio_sel,  "nearpass_multiseed"),
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
    np.save(DATA_PROCESSED / "oof_nb200_multi_seed.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb200_multi_seed.npy",  best_te)

    sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": best_te})
    out_path = SUBMISSIONS / "200_multi_seed_div.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
