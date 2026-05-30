"""nb190 -- Random search over diverse-10 model sets for QReg poly.

nb189 showed that iterating the greedy diverse-10 anchor hits ceiling at 0.298519.
New idea: random search over diverse-10 sets — different greedy starts might find
better diagnostic model combinations.

Approach:
  - For each of 30 random starting models from the pool:
    - Greedy diverse-10 selection seeded from that model
    - QReg poly at alpha in {0.002, 0.0025, 0.003}
    - Track best RAE
  - Take the best nb190 result and add to SLSQP pool
  - 9-model SLSQP: [6 orig + nb187 + nb190_best]

Also: try centered polynomial features (subtract mean before poly transform)
and rank-correlation-based diversity.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize
from scipy.stats import spearmanr

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
    "nb185_qreg_iter", "nb186_sc_blend",
    "nb187_diversity_qreg", "nb188_diverse_refine", "nb189_iterate_diverse",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
    "nb108_grand_v2", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
}

SIX_MODELS = [
    "nb167_xgboost_mae", "nb156_catboost_mae", "nb154_lgbm_mae_filtered",
    "nb162_mixed_pool", "nb165_multiseed_162c", "nb149_meta_maeloss",
]


def load_oof(stem, n_tr):
    p = DATA_PROCESSED / f"oof_{stem}.npy"
    if not p.exists():
        return None, None
    for pref in ("te_", "te_oof_"):
        te_p = DATA_PROCESSED / f"{pref}{stem}.npy"
        if te_p.exists():
            break
    else:
        return None, None
    oof = np.load(p).astype(np.float64).flatten()
    te  = np.load(te_p).astype(np.float64).flatten()
    if len(oof) != n_tr:
        return None, None
    oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
    te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
    return oof, te


def load_pool(n_tr):
    oofs, tes, stems = [], [], []
    for f in sorted(DATA_PROCESSED.glob("oof_nb*.npy")):
        stem = f.stem[4:]
        if any(stem.startswith(ms) or stem == ms for ms in META_STEMS):
            continue
        oof, te = load_oof(stem, n_tr)
        if oof is None:
            continue
        oofs.append(oof); tes.append(te); stems.append(stem)
    return oofs, tes, stems


def greedy_diversity_select_from(oofs, k, seed_idx, use_spearman=False):
    X = np.column_stack(oofs)
    if use_spearman:
        n = X.shape[1]
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                r, _ = spearmanr(X[:,i], X[:,j])
                corr[i,j] = corr[j,i] = abs(r)
    else:
        corr = np.abs(np.corrcoef(X.T))

    selected = [seed_idx]
    remaining = list(range(len(oofs)))
    remaining.remove(seed_idx)
    while len(selected) < k and remaining:
        avg_corrs = [(np.mean([corr[i, j] for j in selected]), i) for i in remaining]
        avg_corrs.sort(key=lambda x: x[0])
        best_i = avg_corrs[0][1]
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


def qreg_eval(X_tr, y_tr, X_te, splits, alpha, centered=False):
    n_tr = len(y_tr)
    if centered:
        mu_tr = X_tr.mean(axis=1, keepdims=True)
        mu_te = X_te.mean(axis=1, keepdims=True)
        X_tr = X_tr - mu_tr
        X_te = X_te - mu_te
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx])
    m_f = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
    m_f.fit(X_tr, y_tr)
    te = m_f.predict(X_te)
    r = rae(y_tr, oof); ratio = te.std()/oof.std()
    return r, oof, te, ratio


def slsqp_multistart(X_tr, y_tr, X_te, n_starts, label):
    k = X_tr.shape[1]
    mean_pred = y_tr.mean()
    def obj(w): return np.mean(np.abs(y_tr - X_tr@w)) / np.mean(np.abs(y_tr - mean_pred))
    best_r, best_w = 1e9, None
    for _ in range(n_starts):
        w0 = np.random.dirichlet(np.ones(k))
        r = minimize(obj, w0, method='SLSQP',
                     bounds=[(0,1)]*k,
                     constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                     options={'ftol':1e-12, 'maxiter':2000})
        if r.fun < best_r:
            best_r, best_w = r.fun, r.x
    oof_b = X_tr@best_w; te_b = X_te@best_w
    ratio = te_b.std()/oof_b.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and best_r < 0.298519) else ""
    print(f"  [{label}] RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb190: Random search over diverse-10 model sets ===\n")
    print("nb188 best: 0.298519  ratio=0.5815")
    print("Goal: find diverse-10 set with better QReg poly than nb187 (0.299246)\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load pool
    oofs, tes, stems = load_pool(n_tr)
    raes = [rae(y_tr, oof) for oof in oofs]
    print(f"Pool: {len(stems)} models")

    # Add nb183 and nb187 to pool as anchors
    oof183, te183 = load_oof("nb183_qreg_poly10", n_tr)
    oof187, te187 = load_oof("nb187_diversity_qreg", n_tr)
    nb183_idx = None; nb187_idx = None
    if oof183 is not None:
        oofs.insert(0, oof183); tes.insert(0, te183)
        stems.insert(0, "nb183"); raes.insert(0, rae(y_tr, oof183))
        nb183_idx = 0
    if oof187 is not None:
        oofs.insert(0, oof187); tes.insert(0, te187)
        stems.insert(0, "nb187"); raes.insert(0, rae(y_tr, oof187))
        nb187_idx = 0
        if nb183_idx is not None:
            nb183_idx = 1

    # Load 6 originals
    oofs6, tes6, names6 = [], [], []
    for stem in SIX_MODELS:
        oof, te = load_oof(stem, n_tr)
        if oof is not None:
            oofs6.append(oof); tes6.append(te); names6.append(stem)
    X6_tr = np.column_stack(oofs6); X6_te = np.column_stack(tes6)

    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    ALPHAS = [0.002, 0.0025, 0.003, 0.004, 0.005]
    n_pool = len(stems)

    # --- A: Random search -- try 30 different seed indices ---
    print(f"--- A: Random search (30 seed indices, alpha in {ALPHAS[:3]}) ---")
    # Only consider seed_idx where we have reasonable individual RAE
    candidate_seeds = list(range(n_pool))
    np.random.shuffle(candidate_seeds)
    candidate_seeds = candidate_seeds[:30]  # Try 30 random seeds

    all_results = []
    for seed_idx in candidate_seeds:
        div10_idx = greedy_diversity_select_from(oofs, k=10, seed_idx=seed_idx)
        X_d10_tr = np.column_stack([oofs[i] for i in div10_idx])
        X_d10_te = np.column_stack([tes[i] for i in div10_idx])
        X_d10p2_tr = poly2.fit_transform(X_d10_tr)
        X_d10p2_te = poly2.transform(X_d10_te)
        # Quick eval: use alpha=0.003 as proxy
        r, oof_s, te_s, ratio = qreg_eval(X_d10p2_tr, y_tr, X_d10p2_te, splits, 0.003)
        if ratio >= COLLAPSE_THRESH:
            all_results.append((r, oof_s, te_s, ratio, seed_idx, div10_idx, 0.003))

    # Sort by RAE and show top-5
    all_results.sort(key=lambda x: x[0])
    print(f"  {len(all_results)} sets passed collapse check (alpha=0.003)")
    print(f"  Best 5:")
    for r, _, _, ratio, seed_idx, div10_idx, alpha in all_results[:5]:
        print(f"    seed={stems[seed_idx]:25s}  RAE={r:.6f}  ratio={ratio:.4f}")

    # Fine-tune the best seed
    best_global_r = all_results[0][0] if all_results else 1e9
    best_global_oof = best_global_te = None
    if all_results:
        best_seed_data = all_results[0]
        r_best0, _, _, _, seed_best, div10_best, _ = best_seed_data
        print(f"\n  Fine-tuning best seed ({stems[seed_best]}) with extended alpha sweep...")
        X_best_tr = np.column_stack([oofs[i] for i in div10_best])
        X_best_te = np.column_stack([tes[i] for i in div10_best])
        X_bestp2_tr = poly2.fit_transform(X_best_tr)
        X_bestp2_te = poly2.transform(X_best_te)
        for alpha in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.007]:
            r, oof_s, te_s, ratio = qreg_eval(X_bestp2_tr, y_tr, X_bestp2_te, splits, alpha)
            flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
            beat = " ***BEATS nb187***" if (ratio >= COLLAPSE_THRESH and r < 0.299246) else ""
            print(f"    alpha={alpha:.4f}  RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
            if ratio >= COLLAPSE_THRESH and r < best_global_r:
                best_global_r, best_global_oof, best_global_te = r, oof_s, te_s

    # --- B: Centered polynomial features on nb183-seeded diverse-10 ---
    print(f"\n--- B: Centered poly on nb183-seeded diverse-10 ---")
    if nb183_idx is not None:
        div10_183 = greedy_diversity_select_from(oofs, k=10, seed_idx=nb183_idx)
        X_c_tr = np.column_stack([oofs[i] for i in div10_183])
        X_c_te = np.column_stack([tes[i] for i in div10_183])
        # Center before poly
        mu_tr = X_c_tr.mean(axis=1, keepdims=True)
        mu_te = X_c_te.mean(axis=1, keepdims=True)
        Xc_tr = X_c_tr - mu_tr; Xc_te = X_c_te - mu_te
        Xcp2_tr = poly2.fit_transform(Xc_tr); Xcp2_te = poly2.transform(Xc_te)
        for alpha in [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]:
            r, oof_s, te_s, ratio = qreg_eval(Xcp2_tr, y_tr, Xcp2_te, splits, alpha)
            flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
            beat = " ***BEATS nb187***" if (ratio >= COLLAPSE_THRESH and r < 0.299246) else ""
            print(f"  [centered] alpha={alpha:.4f}  RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
            if ratio >= COLLAPSE_THRESH and r < best_global_r:
                best_global_r, best_global_oof, best_global_te = r, oof_s, te_s

    # --- C: Spearman-rank diversity ---
    print(f"\n--- C: Spearman-rank diverse-10 seeded from nb183 ---")
    if nb183_idx is not None:
        div10_sp = greedy_diversity_select_from(oofs, k=10, seed_idx=nb183_idx, use_spearman=True)
        print(f"  Spearman diverse: {[stems[i] for i in div10_sp]}")
        X_sp_tr = np.column_stack([oofs[i] for i in div10_sp])
        X_sp_te = np.column_stack([tes[i] for i in div10_sp])
        Xsp2_tr = poly2.fit_transform(X_sp_tr); Xsp2_te = poly2.transform(X_sp_te)
        for alpha in [0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]:
            r, oof_s, te_s, ratio = qreg_eval(Xsp2_tr, y_tr, Xsp2_te, splits, alpha)
            flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
            beat = " ***BEATS nb187***" if (ratio >= COLLAPSE_THRESH and r < 0.299246) else ""
            print(f"  [spearman] alpha={alpha:.4f}  RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
            if ratio >= COLLAPSE_THRESH and r < best_global_r:
                best_global_r, best_global_oof, best_global_te = r, oof_s, te_s

    # --- D: 9-model SLSQP if we found a better nb190 ---
    if best_global_oof is not None and best_global_r < 0.299246:
        print(f"\n--- D: 9-model SLSQP: [6 orig + nb187 + nb190_best] (300 starts) ---")
        X9_tr = np.column_stack([X6_tr, oof187.reshape(-1,1), best_global_oof.reshape(-1,1)])
        X9_te = np.column_stack([X6_te, te187.reshape(-1,1), best_global_te.reshape(-1,1)])
        r_d, oof_d, te_d, ratio_d = slsqp_multistart(X9_tr, y_tr, X9_te, 300, "9model_slsqp")
        if ratio_d >= COLLAPSE_THRESH:
            if r_d < 0.298519:
                best_global_r = r_d; best_global_oof = oof_d; best_global_te = te_d
    else:
        # Still run SLSQP with nb190_best from A even if not better than nb187
        if all_results:
            print(f"\n--- D: 9-model SLSQP with best A result (RAE={all_results[0][0]:.4f}) ---")
            best_oof_a, best_te_a = all_results[0][1], all_results[0][2]
            X9_tr = np.column_stack([X6_tr, oof187.reshape(-1,1), best_oof_a.reshape(-1,1)])
            X9_te = np.column_stack([X6_te, te187.reshape(-1,1), best_te_a.reshape(-1,1)])
            r_d, oof_d, te_d, ratio_d = slsqp_multistart(X9_tr, y_tr, X9_te, 300, "9model_slsqp")
            if ratio_d >= COLLAPSE_THRESH and r_d < 0.298519:
                best_global_r = r_d; best_global_oof = oof_d; best_global_te = te_d
                print(f"  *** NEW BEST: {r_d:.6f} ***")

    # === Summary ===
    print(f"\n=== Summary ===")
    print(f"  nb187 diverse-10 (greedy from nb183): RAE=0.299246  ratio=0.5826")
    print(f"  nb188 8-model SLSQP:                  RAE=0.298519  ratio=0.5815")
    if best_global_r < 0.299246:
        print(f"  nb190 IMPROVEMENT:  RAE={best_global_r:.6f}")
        print(f"  *** BEATS nb187! ***")
    elif best_global_r < 0.298519:
        print(f"  nb190:  RAE={best_global_r:.6f}  *** BEATS nb188! ***")
    else:
        print(f"  nb190: No improvement over nb188 ({best_global_r:.6f} if found)")
        print(f"  Conclusion: 0.298519 appears to be the ceiling for this approach.")
        return

    if best_global_oof is not None:
        best_te_clip = np.clip(best_global_te, y_tr.min()-0.5, y_tr.max()+0.5)
        np.save(DATA_PROCESSED / "oof_nb190_random_diverse.npy", best_global_oof)
        np.save(DATA_PROCESSED / "te_nb190_random_diverse.npy",  best_te_clip)
        sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_clip})
        sub.to_csv(SUBMISSIONS / "190_random_diverse.csv", index=False)
        print(f"\nSaved: submissions/190_random_diverse.csv  OOF RAE={best_global_r:.6f}")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
