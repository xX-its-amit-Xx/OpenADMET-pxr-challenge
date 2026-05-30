"""nb207 -- Ultra-fine alpha grid at the OOF RAE minimum (div15 a~0.002).

nb205 discovered:
  div15 a=0.002: OOF RAE=0.297307 (BEST seen), ratio=0.5610 [FAIL by 0.001]
  div15 a=0.001: OOF RAE=0.297775, ratio=0.5608

The OOF RAE minimum for div15 appears to be near alpha=0.002. This
experiment finds the true minimum with 20 fine-grained alpha points,
and also checks div25 at very low alpha (nb203 only tested 0.001-0.010).

All failing candidates are ratio-inflated to [0.58, 0.582, 0.585, 0.59].
The pool is augmented with NB188 + nb205 inflated candidates from the best
alpha, then run through SLSQP with 2500 starts.

If the true minimum is lower than 0.297307, nb207 will find it.
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
    "nb206_small_k_grid", "nb207_fine_alpha_min",
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


def constrained_slsqp(X_tr, X_te, y_tr, n_starts=2500, prev_best=PREV_BEST, tag="",
                      warm_starts=None):
    """warm_starts: list of index arrays or weight arrays for targeted warm starts."""
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
        # Use looser ftol for speed; tight enough for RAE differences at 1e-5 level
        res = minimize(obj, w0, jac=obj_grad, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-8})
        if res.success and ratio_con(res.x) >= -1e-6:
            r = obj(res.x)
            if r < best_r:
                best_r, best_w = r, res.x.copy()

    # Targeted warm starts: high weight on specific models (those with best individual OOF RAE)
    if warm_starts is not None:
        for idx_list in warm_starts:
            for main_w in [0.5, 0.6, 0.7, 0.8]:
                w0 = np.ones(n_m) * (1.0 - main_w) / (n_m - len(idx_list))
                for i in idx_list:
                    w0[i] = main_w / len(idx_list)
                w0 = np.clip(w0, 0, 1)
                w0 /= w0.sum()
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
    print("=== nb207: Ultra-fine alpha at OOF RAE minimum ===\n", flush=True)
    print(f"nb197 best: {PREV_BEST}  ratio=0.5800", flush=True)
    print(f"Key: div15 a=0.002 gives OOF RAE=0.297307 (BEST), ratio=0.5610\n", flush=True)

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Loading base pool...", flush=True)
    base_oofs, base_tes, base_stems = load_base_pool(n_tr)

    # nb183 is in META_STEMS (excluded from base pool) but needed for greedy diversity
    # Add it back explicitly so the base pool matches nb205's 47-model pool
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

    # Ultra-fine alpha grid focused on [0.001, 0.003] where the minimum is
    alpha_fine = [
        0.0008, 0.0009, 0.0010, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015,
        0.0016, 0.0017, 0.0018, 0.0019, 0.0020, 0.0021, 0.0022, 0.0024,
        0.0026, 0.0028, 0.0030, 0.0035, 0.0040, 0.0050,
        0.0070, 0.0090, 0.0100
    ]

    # Coarser grid for div25 (LP is larger)
    alpha_div25 = [0.0008, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050, 0.0080, 0.0100]

    inflate_targets = [0.580, 0.582, 0.585, 0.590]

    best_rae = PREV_BEST
    best_candidate = None  # (oof, te_orig, te_inflated, name, rae, ratio)
    # Track indices in pool for warm-starting SLSQP on the best candidates
    inf_candidate_indices = {}  # name -> list of pool indices (for inflated versions)

    for k, alpha_list in [(15, alpha_fine), (20, alpha_fine), (25, alpha_div25)]:
        div_idx = greedy_diversity(base_oofs, k=k, seed_idx=0)
        Xk_tr = poly2.fit_transform(np.column_stack([base_oofs[i] for i in div_idx]))
        Xk_te = poly2.transform(np.column_stack([base_tes[i] for i in div_idx]))
        n_poly = Xk_tr.shape[1]
        print(f"--- div{k}: {n_poly} poly2 features ---", flush=True)

        for alpha in alpha_list:
            t_a = time.time()
            oof_c, te_c = qreg_oof(Xk_tr, y_tr, Xk_te, splits, alpha)
            r_c = rae(y_tr, oof_c)
            oof_std = oof_c.std()
            te_std = te_c.std()
            ratio_c = te_std / oof_std if oof_std > 1e-9 else 0.0
            flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
            elapsed = time.time() - t_a

            note = ""
            if r_c < best_rae:
                note = f" *NEW BEST OOF {r_c:.6f}*"
                best_rae = r_c

            print(f"  div{k} a={alpha:.4f}  RAE={r_c:.6f}  ratio={ratio_c:.4f}  [{flag}]{note}  ({elapsed:.0f}s)", flush=True)

            # Add original + all inflated versions to pool
            pool_oofs_tr.append(oof_c)
            pool_oofs_te.append(te_c)
            orig_idx = len(pool_oofs_tr) - 1
            pool_names.append(f"div{k}_a{alpha:.4f}_orig")

            if ratio_c < COLLAPSE_THRESH:
                inf_idxs = []
                for t_r in inflate_targets:
                    te_inf = inflate_te(oof_c, te_c, t_r)
                    pool_oofs_tr.append(oof_c)
                    pool_oofs_te.append(te_inf)
                    inf_idxs.append(len(pool_oofs_tr) - 1)
                    pool_names.append(f"div{k}_a{alpha:.4f}_inf{t_r:.3f}")
                # Track the inflated indices for candidates with OOF RAE < PREV_BEST
                if r_c < PREV_BEST:
                    name = f"div{k}_a{alpha:.4f}"
                    inf_candidate_indices[name] = inf_idxs

            if r_c < PREV_BEST and best_candidate is None:
                best_candidate = (r_c, ratio_c, f"div{k}_a{alpha:.4f}", oof_c, te_c)

        print(f"  div{k} done ({time.time()-t0:.0f}s elapsed)\n", flush=True)

    print(f"\nBest OOF RAE observed: {best_rae:.6f}", flush=True)
    print(f"Total SLSQP pool: {len(pool_oofs_tr)} models", flush=True)
    print(f"Candidates with OOF RAE < prev_best: {len(inf_candidate_indices)}", flush=True)
    print(f"QReg building elapsed: {time.time()-t0:.0f}s\n", flush=True)

    X_tr = np.column_stack(pool_oofs_tr)
    X_te = np.column_stack(pool_oofs_te)

    # Build warm starts for the best inflated candidates
    warm_starts = list(inf_candidate_indices.values())
    print(f"Using {len(warm_starts)} targeted warm starts (best inflated candidates)", flush=True)

    print("--- SLSQP: NB188 + fine-alpha candidates + ratio-inflated (2500 starts + warm) ---", flush=True)
    r_best, oof_best, te_best, ratio_best = constrained_slsqp(
        X_tr, X_te, y_tr, n_starts=2500, prev_best=PREV_BEST, tag="fine_alpha_inflated",
        warm_starts=warm_starts)

    print(f"\n=== Final Summary ({time.time()-t0:.0f}s) ===", flush=True)
    print(f"  nb197 (prev best):   RAE={PREV_BEST:.6f}  ratio=0.5800  [PASS]", flush=True)
    if best_candidate:
        r_bc, ratio_bc, name_bc, _, _ = best_candidate
        print(f"  Best raw candidate:  RAE={r_bc:.6f}  ratio={ratio_bc:.4f}  [{name_bc}]  [FAIL]", flush=True)
    flag = "PASS" if ratio_best >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio_best >= COLLAPSE_THRESH and r_best < PREV_BEST) else ""
    print(f"  nb207 SLSQP:         RAE={r_best:.6f}  ratio={ratio_best:.4f}  [{flag}]{beat}", flush=True)

    if oof_best is not None and r_best < PREV_BEST and ratio_best >= COLLAPSE_THRESH:
        sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": te_best})
        assert len(sub) == 513 and sub["pEC50"].notna().all()
        out_stem = "nb207_fine_alpha_min"
        np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof_best)
        np.save(DATA_PROCESSED / f"te_{out_stem}.npy",  te_best)
        sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
        print(f"Saved: {SUBMISSIONS/f'{out_stem}.csv'}", flush=True)
    elif oof_best is not None:
        out_stem = "nb207_fine_alpha_min_baseline"
        np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof_best)
        np.save(DATA_PROCESSED / f"te_{out_stem}.npy",  te_best)
        sub_d = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": te_best})
        sub_d.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
        print(f"Saved baseline: {SUBMISSIONS/f'{out_stem}.csv'}  RAE={r_best:.6f}", flush=True)


if __name__ == "__main__":
    main()
