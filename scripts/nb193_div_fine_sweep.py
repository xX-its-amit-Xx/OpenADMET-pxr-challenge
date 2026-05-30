"""nb193 -- Fine sweep on diverse-15/20 QReg poly and SLSQP with failing models.

nb188 best: 0.298519  ratio=0.5815
nb192 finding: div20 at alpha=0.005 gives RAE=0.298973 ratio=0.5632 (FAIL).
The crossover for ratio=0.58 is between alpha=0.005 and alpha=0.010.

KEY INSIGHT: In SLSQP, a constituent model doesn't need to pass collapse alone.
Only the FINAL BLEND needs ratio >= 0.58. Adding div20(alpha=0.005) to the
9-model SLSQP pool could improve over 0.298519 if the blend is stable.

Experiments:
  A: Fine alpha sweep for div-20 poly-2 (find crossover)
  B: Fine alpha sweep for div-15 poly-2 (find crossover)
  C: 10-model SLSQP: nb188 8-pool + div20(best) + div15(best)
  D: Diverse sets with different k and fine alpha
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
    "nb192_poly_variant",
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


def load_pool(n_tr):
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


def qreg_train(X_tr, y_tr, X_te, splits, alpha):
    n_tr = len(y_tr)
    oof_s = np.full(n_tr, np.nan)
    for _, (tr_idx, va_idx) in enumerate(splits):
        m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_s[va_idx] = m.predict(X_tr[va_idx])
    m_f = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
    m_f.fit(X_tr, y_tr)
    te_s = m_f.predict(X_te)
    r_s = rae(y_tr, oof_s)
    ratio_s = te_s.std() / oof_s.std()
    return r_s, oof_s, te_s, ratio_s


def qreg_sweep(X_tr, y_tr, X_te, splits, label, alphas, prev_best=0.298519):
    best_r, best_alpha, best_oof, best_te, best_ratio = 1e9, None, None, None, 0
    for alpha in alphas:
        r_s, oof_s, te_s, ratio_s = qreg_train(X_tr, y_tr, X_te, splits, alpha)
        flag = "PASS" if ratio_s >= COLLAPSE_THRESH else "FAIL"
        beat = " ***NEW BEST***" if (ratio_s >= COLLAPSE_THRESH and r_s < prev_best) else ""
        print(f"  [{label}] alpha={alpha:.6f}  RAE={r_s:.6f}  ratio={ratio_s:.4f}  [{flag}]{beat}")
        if ratio_s >= COLLAPSE_THRESH and r_s < best_r:
            best_r, best_alpha = r_s, alpha
            best_oof, best_te, best_ratio = oof_s.copy(), te_s.copy(), ratio_s
    return best_r, best_oof, best_te, best_ratio, best_alpha


def slsqp_blend(pool_stems, extra_oofs_tr, extra_oofs_te, extra_names, y_tr, n_starts=300):
    all_oofs_tr, all_oofs_te, all_names = [], [], []
    for stem in pool_stems:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        te_p  = DATA_PROCESSED / f"te_{stem}.npy"
        if not oof_p.exists():
            continue
        oof_m = np.load(oof_p).astype(np.float64).flatten()
        te_m  = np.load(te_p).astype(np.float64).flatten()
        oof_m = np.where(np.isfinite(oof_m), oof_m, np.nanmean(oof_m))
        te_m  = np.where(np.isfinite(te_m),  te_m,  np.nanmean(te_m))
        all_oofs_tr.append(oof_m); all_oofs_te.append(te_m); all_names.append(stem)

    for oof_e, te_e, nm in zip(extra_oofs_tr, extra_oofs_te, extra_names):
        all_oofs_tr.append(oof_e); all_oofs_te.append(te_e); all_names.append(nm)

    X_tr = np.column_stack(all_oofs_tr)
    X_te = np.column_stack(all_oofs_te)
    n_m = len(all_oofs_tr)
    n_tr = len(y_tr)
    best_r, best_w = 1e9, None

    def neg_rae(w):
        return rae(y_tr, X_tr @ w)

    def neg_rae_grad(w):
        pred = X_tr @ w
        mae_mean = np.abs(y_tr - y_tr.mean()).mean()
        return X_tr.T @ np.sign(pred - y_tr) / (n_tr * mae_mean)

    rng = np.random.default_rng(SEED)
    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(n_m))
        res = minimize(neg_rae, w0, jac=neg_rae_grad,
                       method="SLSQP",
                       bounds=[(0, 1)] * n_m,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                       options={"maxiter": 500, "ftol": 1e-10})
        if res.fun < best_r:
            best_r, best_w = res.fun, res.x

    oof_b = X_tr @ best_w
    te_b  = X_te @ best_w
    ratio = te_b.std() / oof_b.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio >= COLLAPSE_THRESH and best_r < 0.298519) else ""
    print(f"  [{n_m}-model SLSQP] RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]{beat}")
    for i, nm in enumerate(all_names):
        if best_w[i] > 0.001:
            print(f"    {nm:50s}  w={best_w[i]:.4f}")
    return best_r, oof_b, te_b, ratio


def main():
    print("=== nb193: Fine sweep on diverse-15/20 QReg poly, SLSQP with failing models ===\n")
    print("nb188 best: 0.298519  ratio=0.5815")
    print("nb192 div20(a=0.005): 0.298973 ratio=0.5632 [FAIL]")
    print("nb192 div15(a=0.002): 0.297307 ratio=0.5610 [FAIL] — best OOF ever!\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    oofs, tes, stems = load_pool(n_tr)
    print(f"Pool: {len(stems)} pure base models")

    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists():
        oof183 = np.load(oof183_p).astype(np.float64).flatten()
        te183  = np.load(te183_p).astype(np.float64).flatten()
        oof183 = np.where(np.isfinite(oof183), oof183, np.nanmean(oof183))
        te183  = np.where(np.isfinite(te183),  te183,  np.nanmean(te183))
        oofs.insert(0, oof183); tes.insert(0, te183)
        stems.insert(0, "nb183_qreg_poly10")

    nb183_idx = 0
    poly2 = PolynomialFeatures(degree=2, include_bias=False)

    # Build diverse sets
    div15_idx = greedy_diversity(oofs, k=15, seed_idx=nb183_idx)
    div20_idx = greedy_diversity(oofs, k=20, seed_idx=nb183_idx)
    print(f"\ndiv15 last 5: {[stems[i] for i in div15_idx[10:]]}")
    print(f"div20 last 5: {[stems[i] for i in div20_idx[15:]]}")

    X15_tr = poly2.fit_transform(np.column_stack([oofs[i] for i in div15_idx]))
    X15_te = poly2.transform(np.column_stack([tes[i] for i in div15_idx]))
    X20_tr = poly2.fit_transform(np.column_stack([oofs[i] for i in div20_idx]))
    X20_te = poly2.transform(np.column_stack([tes[i] for i in div20_idx]))

    # --- A: Fine sweep div20 ---
    print("\n--- A: Fine sweep div-20 poly-2 ---")
    alphas_a = [0.0050, 0.0055, 0.0060, 0.0065, 0.0070, 0.0075, 0.0080, 0.0090, 0.0095, 0.0100]
    r_a, oof_a, te_a, ratio_a, alpha_a = qreg_sweep(X20_tr, y_tr, X20_te, splits, "div20", alphas_a)
    all_div20 = {}
    for alpha in alphas_a:
        r_s, oof_s, te_s, ratio_s = qreg_train(X20_tr, y_tr, X20_te, splits, alpha)
        all_div20[alpha] = (r_s, oof_s, te_s, ratio_s)

    # --- B: Fine sweep div15 ---
    print("\n--- B: Fine sweep div-15 poly-2 ---")
    alphas_b = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012]
    r_b, oof_b, te_b, ratio_b, alpha_b = qreg_sweep(X15_tr, y_tr, X15_te, splits, "div15", alphas_b)
    all_div15 = {}
    for alpha in alphas_b:
        r_s, oof_s, te_s, ratio_s = qreg_train(X15_tr, y_tr, X15_te, splits, alpha)
        all_div15[alpha] = (r_s, oof_s, te_s, ratio_s)

    # --- C: 10-model SLSQP with best passing div20 ---
    print("\n--- C: 10-model SLSQP: nb188 8-pool + best div20 + best div15 ---")
    extra_oofs_tr, extra_oofs_te, extra_names = [], [], []
    if oof_a is not None:
        extra_oofs_tr.append(oof_a); extra_oofs_te.append(te_a)
        extra_names.append(f"div20_a{alpha_a:.4f}")
    if oof_b is not None:
        extra_oofs_tr.append(oof_b); extra_oofs_te.append(te_b)
        extra_names.append(f"div15_a{alpha_b:.4f}")

    if extra_oofs_tr:
        r_c, oof_c, te_c, ratio_c = slsqp_blend(
            NB188_POOL, extra_oofs_tr, extra_oofs_te, extra_names, y_tr, n_starts=300)
    else:
        r_c, ratio_c = 1e9, 0

    # --- D: SLSQP with FAILING div20 models (low alpha) ---
    # Key: even if individual model fails, SLSQP blend might still pass
    print("\n--- D: SLSQP with failing div20(a=0.005) as extra model ---")
    r20_005, oof20_005, te20_005, ratio20_005 = all_div20.get(0.005, (None, None, None, None))
    if oof20_005 is not None:
        print(f"  div20(a=0.005): RAE={r20_005:.6f}  ratio={ratio20_005:.4f}  (fails alone)")
        r_d, oof_d, te_d, ratio_d = slsqp_blend(
            NB188_POOL, [oof20_005], [te20_005], ["div20_a0.005"], y_tr, n_starts=200)
    else:
        r_d, ratio_d = 1e9, 0

    # --- E: SLSQP with div15(a=0.002) as extra ---
    print("\n--- E: SLSQP with failing div15(a=0.002) as extra model ---")
    r15_002, oof15_002, te15_002, ratio15_002 = all_div15.get(0.002, (None, None, None, None))
    if oof15_002 is not None:
        print(f"  div15(a=0.002): RAE={r15_002:.6f}  ratio={ratio15_002:.4f}  (fails alone)")
        r_e, oof_e, te_e, ratio_e = slsqp_blend(
            NB188_POOL, [oof15_002], [te15_002], ["div15_a0.002"], y_tr, n_starts=200)
    else:
        r_e, ratio_e = 1e9, 0

    # --- F: SLSQP with BOTH failing low-alpha models ---
    print("\n--- F: SLSQP with nb188 pool + div20(0.005) + div15(0.002) ---")
    if oof20_005 is not None and oof15_002 is not None:
        r_f, oof_f, te_f, ratio_f = slsqp_blend(
            NB188_POOL,
            [oof20_005, oof15_002],
            [te20_005, te15_002],
            ["div20_a0.005", "div15_a0.002"],
            y_tr, n_starts=300)
    else:
        r_f, ratio_f = 1e9, 0

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  nb188 (prev best):      RAE=0.298519  ratio=0.5815  [PASS]")
    print(f"  A: div20 fine sweep:    RAE={r_a:.6f}  ratio={ratio_a:.4f}  {'[PASS]' if ratio_a >= COLLAPSE_THRESH else '[FAIL]'}")
    print(f"  B: div15 fine sweep:    RAE={r_b:.6f}  ratio={ratio_b:.4f}  {'[PASS]' if ratio_b >= COLLAPSE_THRESH else '[FAIL]'}")
    print(f"  C: 10-model SLSQP:      RAE={r_c:.6f}  ratio={ratio_c:.4f}  {'[PASS]' if ratio_c >= COLLAPSE_THRESH else '[FAIL]'}")
    print(f"  D: SLSQP+div20(0.005):  RAE={r_d:.6f}  ratio={ratio_d:.4f}  {'[PASS]' if ratio_d >= COLLAPSE_THRESH else '[FAIL]'}")
    print(f"  E: SLSQP+div15(0.002):  RAE={r_e:.6f}  ratio={ratio_e:.4f}  {'[PASS]' if ratio_e >= COLLAPSE_THRESH else '[FAIL]'}")
    print(f"  F: SLSQP+both:          RAE={r_f:.6f}  ratio={ratio_f:.4f}  {'[PASS]' if ratio_f >= COLLAPSE_THRESH else '[FAIL]'}")

    # Find the best passing result
    candidates = [
        (r_a, oof_a, te_a, ratio_a, "div20_fine"),
        (r_b, oof_b, te_b, ratio_b, "div15_fine"),
        (r_c, oof_c, te_c, ratio_c, "10model_slsqp"),
        (r_d, oof_d, te_d, ratio_d, "slsqp_div20_005"),
        (r_e, oof_e, te_e, ratio_e, "slsqp_div15_002"),
        (r_f, oof_f, te_f, ratio_f, "slsqp_both"),
    ]
    best_rae, best_oof, best_te, best_name = 0.298519, None, None, None
    for r, oof_s, te_s, ratio, nm in candidates:
        if oof_s is not None and ratio >= COLLAPSE_THRESH and r < best_rae:
            best_rae, best_oof, best_te, best_name = r, oof_s, te_s, nm

    if best_oof is None:
        print("\nNo improvement over nb188 (0.298519).")
        print("Polynomial QReg approach has been exhaustively explored.")
        return

    print(f"\n*** NEW BEST: {best_name}  RAE={best_rae:.6f} ***")
    np.save(DATA_PROCESSED / "oof_nb193_div_fine.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb193_div_fine.npy",  best_te)

    sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": best_te})
    out_path = SUBMISSIONS / "193_div_fine.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
