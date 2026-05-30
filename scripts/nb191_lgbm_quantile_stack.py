"""nb191 -- LGBM quantile stacker on diverse OOF polynomial features.

nb188 best: 0.298519  ratio=0.5815
Goal: beat 0.298519 using LGBM(quantile, alpha=0.5) instead of QReg.

Hypothesis: LGBM(quantile) minimizes MAE like QReg but can learn nonlinear
interactions more efficiently (tree splits on poly features) and may
naturally maintain test variance better than regularized QReg.

Experiments:
  A: LGBM quantile on poly-2 of Pearson-diverse-10 (nb187's set)
  B: LGBM quantile on poly-2 of Spearman-diverse-10 (nb190's set C)
  C: LGBM quantile on raw linear diverse-10 OOFs (no poly)
  D: HistGradientBoosting quantile on poly-2 diverse-10
  E: Best result -> 9-model SLSQP with nb188's 8-model pool
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

import lightgbm as lgb

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
    "nb190_random_diverse_search",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
    "nb108_grand_v2", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
}

# nb188 8-model SLSQP pool (for final ensemble)
NB188_POOL = [
    ("nb167_xgboost_mae",    0.1422),
    ("nb156_catboost_mae",   0.0926),
    ("nb154_lgbm_mae_filtered", 0.0531),
    ("nb162_mixed_pool",     0.0003),
    ("nb165_multiseed_162c", 0.059),
    ("nb149_meta_maeloss",   0.0209),
    ("nb183_qreg_poly10",    0.0),
    ("nb187_diversity_qreg", 0.6319),
]


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


def greedy_diversity_select_pearson(oofs, k, seed_idx=0):
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


def greedy_diversity_select_spearman(oofs, k, seed_idx=0):
    X = np.column_stack(oofs)
    n = X.shape[1]
    # Pre-rank
    X_r = np.apply_along_axis(rankdata, 0, X)
    corr = np.abs(np.corrcoef(X_r.T))
    selected = [seed_idx]
    remaining = list(range(n))
    remaining.remove(seed_idx)
    while len(selected) < k and remaining:
        avg_corrs = [(np.mean([corr[i, j] for j in selected]), i) for i in remaining]
        avg_corrs.sort(key=lambda x: x[0])
        selected.append(avg_corrs[0][1])
        remaining.remove(avg_corrs[0][1])
    return selected


def lgbm_quantile_cv(X_tr, y_tr, X_te, splits, label,
                     n_est=200, max_depth=4, reg_lambda=10.0, lr=0.05,
                     min_child_samples=20, subsample=0.8):
    n_tr = len(y_tr)
    oof_s = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5,
            n_estimators=n_est, max_depth=max_depth,
            learning_rate=lr, reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            subsample=subsample, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1,
        )
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_s[va_idx] = m.predict(X_tr[va_idx])
    m_f = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5,
        n_estimators=n_est, max_depth=max_depth,
        learning_rate=lr, reg_lambda=reg_lambda,
        min_child_samples=min_child_samples,
        subsample=subsample, colsample_bytree=0.8,
        random_state=SEED, verbosity=-1,
    )
    m_f.fit(X_tr, y_tr)
    te_s = m_f.predict(X_te)
    r_s = rae(y_tr, oof_s)
    ratio_s = te_s.std() / oof_s.std()
    flag = "PASS" if ratio_s >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio_s >= COLLAPSE_THRESH and r_s < 0.298519) else ""
    print(f"  [{label}] RAE={r_s:.6f}  ratio={ratio_s:.4f}  [{flag}]{beat}")
    return r_s, oof_s, te_s, ratio_s


def histgb_quantile_cv(X_tr, y_tr, X_te, splits, label,
                       n_est=200, max_depth=4, lr=0.05, reg_lambda=10.0):
    n_tr = len(y_tr)
    oof_s = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.5,
            max_iter=n_est, max_depth=max_depth,
            learning_rate=lr, l2_regularization=reg_lambda,
            random_state=SEED,
        )
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_s[va_idx] = m.predict(X_tr[va_idx])
    m_f = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.5,
        max_iter=n_est, max_depth=max_depth,
        learning_rate=lr, l2_regularization=reg_lambda,
        random_state=SEED,
    )
    m_f.fit(X_tr, y_tr)
    te_s = m_f.predict(X_te)
    r_s = rae(y_tr, oof_s)
    ratio_s = te_s.std() / oof_s.std()
    flag = "PASS" if ratio_s >= COLLAPSE_THRESH else "FAIL"
    beat = " ***NEW BEST***" if (ratio_s >= COLLAPSE_THRESH and r_s < 0.298519) else ""
    print(f"  [{label}] RAE={r_s:.6f}  ratio={ratio_s:.4f}  [{flag}]{beat}")
    return r_s, oof_s, te_s, ratio_s


def slsqp_blend(oofs_tr, oofs_te, y_tr, n_starts=100, label="slsqp"):
    n_m = len(oofs_tr)
    X_tr = np.column_stack(oofs_tr)
    X_te = np.column_stack(oofs_te)
    best_r, best_w = 1e9, None

    def neg_rae(w):
        pred = X_tr @ w
        return rae(y_tr, pred)

    def neg_rae_grad(w):
        pred = X_tr @ w
        res = pred - y_tr
        mae_mean = np.abs(y_tr - y_tr.mean()).mean()
        sign = np.sign(res)
        return X_tr.T @ sign / (len(y_tr) * mae_mean)

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

    oof_blend = X_tr @ best_w
    te_blend  = X_te @ best_w
    ratio = te_blend.std() / oof_blend.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    print(f"  [{label}] RAE={best_r:.6f}  ratio={ratio:.4f}  [{flag}]")
    for i, (stem, _) in enumerate(NB188_POOL):
        print(f"    {stem:50s}  w={best_w[i]:.4f}")
    return best_r, oof_blend, te_blend, best_w


def main():
    print("=== nb191: LGBM quantile stacker on diverse OOF poly features ===\n")
    print("nb188 best: 0.298519  ratio=0.5815")
    print("nb187 QReg poly-10 (Pearson-diverse): 0.299246  ratio=0.5826\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    oofs, tes, stems = load_pool(n_tr)
    raes_all = [rae(y_tr, oof) for oof in oofs]
    print(f"Pool: {len(stems)} pure base models")

    # Insert nb183 anchor
    oof183_p = DATA_PROCESSED / "oof_nb183_qreg_poly10.npy"
    te183_p  = DATA_PROCESSED / "te_nb183_qreg_poly10.npy"
    if oof183_p.exists():
        oof183 = np.load(oof183_p).astype(np.float64).flatten()
        te183  = np.load(te183_p).astype(np.float64).flatten()
        oof183 = np.where(np.isfinite(oof183), oof183, np.nanmean(oof183))
        te183  = np.where(np.isfinite(te183),  te183,  np.nanmean(te183))
        oofs.insert(0, oof183); tes.insert(0, te183)
        stems.insert(0, "nb183_qreg_poly10")
        raes_all.insert(0, rae(y_tr, oof183))
        print(f"Added nb183 (RAE={raes_all[0]:.4f}) as pool anchor")

    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    results = {}

    # --- A: LGBM quantile on poly-2 of Pearson-diverse-10 ---
    print("\n--- A: LGBM quantile on poly-2, Pearson-diverse-10 (nb187 set) ---")
    nb183_idx = 0
    div10_P = greedy_diversity_select_pearson(oofs, k=10, seed_idx=nb183_idx)
    print(f"  Selected (Pearson): {[stems[i] for i in div10_P]}")
    X_p_tr = poly2.fit_transform(np.column_stack([oofs[i] for i in div10_P]))
    X_p_te = poly2.transform(np.column_stack([tes[i] for i in div10_P]))
    print(f"  Poly features: {X_p_tr.shape}")
    best_ra, best_oof_a, best_te_a, best_ratio_a = 1e9, None, None, 0
    for reg_l in [1, 5, 10, 20, 50]:
        r, oof_s, te_s, ratio = lgbm_quantile_cv(
            X_p_tr, y_tr, X_p_te, splits, f"lgbm_poly10_lam{reg_l}", reg_lambda=reg_l)
        if ratio >= COLLAPSE_THRESH and r < best_ra:
            best_ra, best_oof_a, best_te_a, best_ratio_a = r, oof_s, te_s, ratio
    if best_oof_a is not None:
        results["A"] = (best_ra, best_oof_a, best_te_a, best_ratio_a)
        print(f"  Best A: RAE={best_ra:.6f}  ratio={best_ratio_a:.4f}")

    # --- B: LGBM quantile on poly-2 of Spearman-diverse-10 ---
    print("\n--- B: LGBM quantile on poly-2, Spearman-diverse-10 ---")
    div10_S = greedy_diversity_select_spearman(oofs, k=10, seed_idx=nb183_idx)
    print(f"  Selected (Spearman): {[stems[i] for i in div10_S]}")
    X_s_tr = poly2.fit_transform(np.column_stack([oofs[i] for i in div10_S]))
    X_s_te = poly2.transform(np.column_stack([tes[i] for i in div10_S]))
    print(f"  Poly features: {X_s_tr.shape}")
    best_rb, best_oof_b, best_te_b, best_ratio_b = 1e9, None, None, 0
    for reg_l in [1, 5, 10, 20, 50]:
        r, oof_s, te_s, ratio = lgbm_quantile_cv(
            X_s_tr, y_tr, X_s_te, splits, f"lgbm_spear10_lam{reg_l}", reg_lambda=reg_l)
        if ratio >= COLLAPSE_THRESH and r < best_rb:
            best_rb, best_oof_b, best_te_b, best_ratio_b = r, oof_s, te_s, ratio
    if best_oof_b is not None:
        results["B"] = (best_rb, best_oof_b, best_te_b, best_ratio_b)
        print(f"  Best B: RAE={best_rb:.6f}  ratio={best_ratio_b:.4f}")

    # --- C: LGBM quantile on raw linear diverse-10 (no poly) ---
    print("\n--- C: LGBM quantile on raw (linear) Pearson-diverse-10 ---")
    X_lin_tr = np.column_stack([oofs[i] for i in div10_P])
    X_lin_te = np.column_stack([tes[i] for i in div10_P])
    best_rc, best_oof_c, best_te_c, best_ratio_c = 1e9, None, None, 0
    for reg_l in [0.1, 1, 5, 10]:
        r, oof_s, te_s, ratio = lgbm_quantile_cv(
            X_lin_tr, y_tr, X_lin_te, splits, f"lgbm_lin10_lam{reg_l}", reg_lambda=reg_l)
        if ratio >= COLLAPSE_THRESH and r < best_rc:
            best_rc, best_oof_c, best_te_c, best_ratio_c = r, oof_s, te_s, ratio
    if best_oof_c is not None:
        results["C"] = (best_rc, best_oof_c, best_te_c, best_ratio_c)
        print(f"  Best C: RAE={best_rc:.6f}  ratio={best_ratio_c:.4f}")

    # --- D: HistGBM quantile on poly-2 Pearson-diverse-10 ---
    print("\n--- D: HistGBM quantile on poly-2, Pearson-diverse-10 ---")
    best_rd, best_oof_d, best_te_d, best_ratio_d = 1e9, None, None, 0
    for reg_l in [1, 5, 10, 30]:
        r, oof_s, te_s, ratio = histgb_quantile_cv(
            X_p_tr, y_tr, X_p_te, splits, f"histgb_poly10_lam{reg_l}", reg_lambda=reg_l)
        if ratio >= COLLAPSE_THRESH and r < best_rd:
            best_rd, best_oof_d, best_te_d, best_ratio_d = r, oof_s, te_s, ratio
    if best_oof_d is not None:
        results["D"] = (best_rd, best_oof_d, best_te_d, best_ratio_d)
        print(f"  Best D: RAE={best_rd:.6f}  ratio={best_ratio_d:.4f}")

    # --- E: Deeper LGBM sweep (more n_est, different depths) ---
    print("\n--- E: LGBM quantile, Pearson-diverse-10, depth/nest sweep ---")
    best_re, best_oof_e, best_te_e, best_ratio_e = 1e9, None, None, 0
    for n_est in [100, 300, 500]:
        for depth in [3, 5, 6]:
            r, oof_s, te_s, ratio = lgbm_quantile_cv(
                X_p_tr, y_tr, X_p_te, splits,
                f"lgbm_poly10_n{n_est}_d{depth}", n_est=n_est, max_depth=depth, reg_lambda=20)
            if ratio >= COLLAPSE_THRESH and r < best_re:
                best_re, best_oof_e, best_te_e, best_ratio_e = r, oof_s, te_s, ratio
    if best_oof_e is not None:
        results["E"] = (best_re, best_oof_e, best_te_e, best_ratio_e)
        print(f"  Best E: RAE={best_re:.6f}  ratio={best_ratio_e:.4f}")

    # --- Summary & pick best ---
    print("\n=== Summary ===")
    overall_best_r, overall_best_oof, overall_best_te = 1e9, None, None
    for k, (r, oof_s, te_s, ratio) in results.items():
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k}: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]")
        if ratio >= COLLAPSE_THRESH and r < overall_best_r:
            overall_best_r, overall_best_oof, overall_best_te = r, oof_s, te_s

    if overall_best_oof is None:
        print("\nNo experiment passed collapse check — no improvement over nb188.")
        return

    print(f"\nBest this run: RAE={overall_best_r:.6f}")
    if overall_best_r >= 0.298519:
        print("No improvement over nb188 (0.298519). Saving as candidate anyway.")

    # Save nb191 predictions
    oof191 = overall_best_oof
    te191  = overall_best_te
    np.save(DATA_PROCESSED / "oof_nb191_lgbm_qstack.npy", oof191)
    np.save(DATA_PROCESSED / "te_nb191_lgbm_qstack.npy",  te191)
    print(f"Saved: oof_nb191_lgbm_qstack.npy  RAE={rae(y_tr, oof191):.6f}")

    # --- F: 9-model SLSQP (nb188 pool + nb191) ---
    print("\n--- F: 9-model SLSQP (nb188 8-pool + nb191) with 200 starts ---")
    oofs9_tr, oofs9_te = [], []
    for stem, _ in NB188_POOL:
        oof_p = DATA_PROCESSED / f"oof_{stem}.npy"
        te_p  = DATA_PROCESSED / f"te_{stem}.npy"
        if not oof_p.exists():
            print(f"  Missing {oof_p}, skipping")
            continue
        oof_m = np.load(oof_p).astype(np.float64).flatten()
        te_m  = np.load(te_p).astype(np.float64).flatten()
        oof_m = np.where(np.isfinite(oof_m), oof_m, np.nanmean(oof_m))
        te_m  = np.where(np.isfinite(te_m),  te_m,  np.nanmean(te_m))
        oofs9_tr.append(oof_m)
        oofs9_te.append(te_m)
    oofs9_tr.append(oof191)
    oofs9_te.append(te191)
    print(f"  9-model pool: {len(oofs9_tr)} models")
    r_slsqp, oof_slsqp, te_slsqp, w_slsqp = slsqp_blend(oofs9_tr, oofs9_te, y_tr, n_starts=200, label="9model_slsqp")

    if te_slsqp.std() / oof_slsqp.std() >= COLLAPSE_THRESH:
        oof191_final = oof_slsqp
        te191_final  = te_slsqp
        final_rae    = r_slsqp
    else:
        oof191_final = oof191
        te191_final  = te191
        final_rae    = overall_best_r

    # Save submission
    sub = pd.DataFrame({"Molecule Name": te_df["molecule_name"], "pEC50": te191_final})
    out_path = SUBMISSIONS / "191_lgbm_qstack.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}  (final RAE={final_rae:.6f})")

    np.save(DATA_PROCESSED / "oof_nb191_lgbm_qstack_final.npy", oof191_final)
    np.save(DATA_PROCESSED / "te_nb191_lgbm_qstack_final.npy",  te191_final)

    print("\n=== Final ===")
    print(f"  nb188 (prev best): 0.298519")
    print(f"  nb191 best stacker: {overall_best_r:.6f}")
    print(f"  nb191 9-model SLSQP: {final_rae:.6f}")


if __name__ == "__main__":
    main()
