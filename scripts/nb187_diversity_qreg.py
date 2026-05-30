"""nb187 -- Diversity-based model selection for QReg poly.

nb183 used top-10 models by OOF RAE for QReg poly (degree=2).
Hypothesis: selecting MORE DIVERSE models (lower OOF correlation) gives
polynomial interactions with more new signal, potentially beating 0.299711.

Approach:
  1. Compute pairwise OOF correlations for all pool models
  2. Greedy diversity selection: start with nb183 (best), greedily add model
     with lowest average correlation to existing set
  3. Apply QReg poly on this diverse-10 set
  4. Compare to top-10-by-RAE (nb183 result: 0.300023)

Also tests:
  B: QReg poly on ALL pool models (top-30 or all 47, with high alpha)
  C: QReg poly on diverse-15 set (more models, more interactions)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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
    "nb185_qreg_iter",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
    "nb108_grand_v2", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
}


def load_pool(n_tr):
    oofs, tes, stems, raes = [], [], [], []
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


def greedy_diversity_select(oofs, stems, raes, k, seed_idx=0):
    """Greedy selection: start with seed_idx, add model with lowest avg correlation."""
    n = len(oofs)
    # Correlation matrix
    X = np.column_stack(oofs)
    corr = np.corrcoef(X.T)  # (n, n)

    selected = [seed_idx]
    remaining = list(range(n))
    remaining.remove(seed_idx)

    while len(selected) < k and remaining:
        # For each remaining, compute avg correlation to selected set
        avg_corrs = []
        for i in remaining:
            avg_c = np.mean([abs(corr[i, j]) for j in selected])
            avg_corrs.append((avg_c, i))
        # Pick the one with lowest avg correlation (most diverse)
        avg_corrs.sort(key=lambda x: x[0])
        best_i = avg_corrs[0][1]
        selected.append(best_i)
        remaining.remove(best_i)

    return selected


def qreg_sweep(X_tr, y_tr, X_te, splits, label, alphas):
    n_tr = len(y_tr)
    best_r, best_alpha, best_oof, best_te = 1e9, None, None, None
    for alpha in alphas:
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof_s[va_idx] = m.predict(X_tr[va_idx])
        m_f = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")
        m_f.fit(X_tr, y_tr)
        te_s = m_f.predict(X_te)
        r_s = rae(y_tr, oof_s); ratio_s = te_s.std()/oof_s.std()
        flag = "PASS" if ratio_s >= COLLAPSE_THRESH else "FAIL"
        beat = " ***NEW BEST***" if (ratio_s >= COLLAPSE_THRESH and r_s < 0.299711) else ""
        print(f"  [{label}] alpha={alpha:.5f}  RAE={r_s:.6f}  ratio={ratio_s:.4f}  [{flag}]{beat}")
        if ratio_s >= COLLAPSE_THRESH and r_s < best_r:
            best_r, best_alpha, best_oof, best_te = r_s, alpha, oof_s.copy(), te_s.copy()
    return best_r, best_oof, best_te, best_alpha


def main():
    print("=== nb187: Diversity-based model selection for QReg poly ===\n")
    print("Reference: nb183 QReg poly-10 (by RAE) = 0.300023  ratio=0.580")
    print("Reference: nb184 7-model SLSQP = 0.299711  ratio=0.5801\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Load pool
    oofs, tes, stems = load_pool(n_tr)
    raes = [rae(y_tr, oof) for oof in oofs]
    print(f"Pool: {len(stems)} pure base models")

    # Also include nb183 as a candidate (it's in META_STEMS but we want it)
    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists():
        oof183 = np.load(oof183_p).astype(np.float64).flatten()
        te183  = np.load(te183_p).astype(np.float64).flatten()
        oof183 = np.where(np.isfinite(oof183), oof183, np.nanmean(oof183))
        te183  = np.where(np.isfinite(te183),  te183,  np.nanmean(te183))
        oofs.insert(0, oof183); tes.insert(0, te183)
        stems.insert(0, "nb183_qreg_poly10"); raes.insert(0, rae(y_tr, oof183))
        print(f"Added nb183 (RAE={raes[0]:.4f}) as pool anchor")

    # Sort by RAE
    sorted_idx = np.argsort(raes)
    print(f"Top-10 by RAE:")
    for i in sorted_idx[:10]:
        print(f"  {stems[i]:50s}  RAE={raes[i]:.4f}")

    # Greedy diversity selection (seeded from nb183 = index 0 if included)
    nb183_idx = 0  # nb183 is first (inserted above)

    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    alphas = [0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.002, 0.003, 0.005]
    results = {}

    # --- A: Diverse-10 selection (seeded from nb183) ---
    print(f"\n--- A: Greedy diverse-10 (seed=nb183) ---")
    div10_idx = greedy_diversity_select(oofs, stems, raes, k=10, seed_idx=nb183_idx)
    print(f"  Selected: {[stems[i] for i in div10_idx]}")
    print(f"  RAEs:     {[f'{raes[i]:.4f}' for i in div10_idx]}")
    X_div10_tr = np.column_stack([oofs[i] for i in div10_idx])
    X_div10_te = np.column_stack([tes[i] for i in div10_idx])
    X_div10p2_tr = poly2.fit_transform(X_div10_tr)
    X_div10p2_te = poly2.transform(X_div10_te)
    print(f"  Poly features: {X_div10p2_tr.shape}")
    r_a, oof_a, te_a, alpha_a = qreg_sweep(X_div10p2_tr, y_tr, X_div10p2_te, splits, "diverse10", alphas)
    if oof_a is not None:
        results["A_diverse10"] = (r_a, oof_a, te_a, te_a.std()/oof_a.std())

    # --- B: Diverse-15 selection ---
    print(f"\n--- B: Greedy diverse-15 (seed=nb183) ---")
    if len(oofs) >= 15:
        div15_idx = greedy_diversity_select(oofs, stems, raes, k=15, seed_idx=nb183_idx)
        print(f"  Last 5 added: {[stems[i] for i in div15_idx[10:]]}")
        X_div15_tr = np.column_stack([oofs[i] for i in div15_idx])
        X_div15_te = np.column_stack([tes[i] for i in div15_idx])
        X_div15p2_tr = poly2.fit_transform(X_div15_tr)
        X_div15p2_te = poly2.transform(X_div15_te)
        print(f"  Poly features: {X_div15p2_tr.shape}")
        alphas_b = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02]
        r_b, oof_b, te_b, alpha_b = qreg_sweep(X_div15p2_tr, y_tr, X_div15p2_te, splits, "diverse15", alphas_b)
        if oof_b is not None:
            results["B_diverse15"] = (r_b, oof_b, te_b, te_b.std()/oof_b.std())

    # --- C: Top-10 by RAE but excluding nb183 (to see if diversity helps vs pure RAE selection) ---
    print(f"\n--- C: Top-10 by RAE (excluding nb183) ---")
    top10_noNB183 = [i for i in sorted_idx if stems[i] != "nb183_qreg_poly10"][:10]
    print(f"  {[stems[i] for i in top10_noNB183]}")
    X_t10_tr = np.column_stack([oofs[i] for i in top10_noNB183])
    X_t10_te = np.column_stack([tes[i] for i in top10_noNB183])
    X_t10p2_tr = poly2.fit_transform(X_t10_tr)
    X_t10p2_te = poly2.transform(X_t10_te)
    print(f"  Poly features: {X_t10p2_tr.shape}")
    r_c, oof_c, te_c, alpha_c = qreg_sweep(X_t10p2_tr, y_tr, X_t10p2_te, splits, "top10_noNB183", alphas)
    if oof_c is not None:
        results["C_top10_noNB183"] = (r_c, oof_c, te_c, te_c.std()/oof_c.std())

    # --- D: Mix of diverse + top RAE ---
    print(f"\n--- D: Hybrid set (5 best RAE + 5 most diverse from remaining) ---")
    top5_rae = list(sorted_idx[:5])
    remaining_stems = [i for i in range(len(stems)) if i not in top5_rae]
    # From remaining, greedily select 5 most diverse from top5
    X_top5 = np.column_stack([oofs[i] for i in top5_rae])
    corr_all = np.corrcoef(np.column_stack(oofs).T)
    div5_from_remaining = []
    rem = list(remaining_stems)
    for _ in range(5):
        if not rem: break
        current_set = top5_rae + div5_from_remaining
        avg_corrs = [(np.mean([abs(corr_all[i, j]) for j in current_set]), i) for i in rem]
        avg_corrs.sort(key=lambda x: x[0])
        best = avg_corrs[0][1]
        div5_from_remaining.append(best)
        rem.remove(best)
    hybrid10 = top5_rae + div5_from_remaining
    print(f"  Top-5 RAE: {[stems[i] for i in top5_rae]}")
    print(f"  Diverse-5: {[stems[i] for i in div5_from_remaining]}")
    X_hyb_tr = np.column_stack([oofs[i] for i in hybrid10])
    X_hyb_te = np.column_stack([tes[i] for i in hybrid10])
    X_hybp2_tr = poly2.fit_transform(X_hyb_tr)
    X_hybp2_te = poly2.transform(X_hyb_te)
    print(f"  Poly features: {X_hybp2_tr.shape}")
    r_d, oof_d, te_d, alpha_d = qreg_sweep(X_hybp2_tr, y_tr, X_hybp2_te, splits, "hybrid10", alphas)
    if oof_d is not None:
        results["D_hybrid10"] = (r_d, oof_d, te_d, te_d.std()/oof_d.std())

    # === Summary ===
    print(f"\n=== Summary ===")
    print(f"  nb183 (top-10 by RAE):   RAE=0.300023  ratio=0.5800  PASS")
    print(f"  nb184 (7-model SLSQP):   RAE=0.299711  ratio=0.5801  PASS")
    clean = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    for k, (r, _, _, ratio) in sorted(clean.items(), key=lambda x: x[1][0]):
        beat = " ***NEW BEST***" if r < 0.299711 else ("***beats 0.300023***" if r < 0.300023 else "")
        print(f"  {k:30s}  RAE={r:.6f}  ratio={ratio:.4f}  PASS  {beat}")

    if not clean:
        print("  No diversity-based model beats existing results.")
        print("  Conclusion: top-10 by RAE (nb183) is already near-optimal selection.")
        return

    best_r, best_oof, best_te, best_ratio = min(clean.values(), key=lambda x: x[0])
    if best_r >= 0.299711:
        print(f"\n7-model SLSQP (0.299711) still best.")
        return

    best_te_clip = np.clip(best_te, y_tr.min()-0.5, y_tr.max()+0.5)
    np.save(DATA_PROCESSED / "oof_nb187_diversity_qreg.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb187_diversity_qreg.npy",  best_te_clip)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_clip})
    sub.to_csv(SUBMISSIONS / "187_diversity_qreg.csv", index=False)
    print(f"\nSaved: submissions/187_diversity_qreg.csv  OOF RAE={best_r:.6f}")
    print(f"*** NEW BEST: {best_r:.6f} ***")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
