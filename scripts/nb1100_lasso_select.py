"""nb1100 -- LASSO L1 feature selection (different paradigm from SHAP).

HYPOTHESIS:
    nb2103 K=28 (top-28 SHAP, on 117-col 5-way K-tuned matrix) gave
    mean_bag=0.4737 / median_bag=0.4698. SHAP is a tree-attribution
    feature ranking; LASSO is an L1-regularized LINEAR coefficient
    ranking. Different paradigm: SHAP captures non-linear tree splits;
    LASSO captures linear-projection importance only. If the LASSO
    top-28 substantially overlaps SHAP top-28, the SHAP K=28 verdict
    is paradigm-robust; if it disagrees but matches or beats SHAP RAE,
    LASSO opens an orthogonal feature subset.

PROTOCOL:
    1. Load 117-col 5-way K-tuned feature matrix (chemprop_aux residual
       substrate, identical to nb2103) using the nb1086 helper.
    2. Standardize features (StandardScaler -- LASSO requires scaling).
    3. Fit LassoCV(cv=5, alphas=np.logspace(-3, 1, 50)) on the 253
       residual = y_unb - chemprop_aux te[unb_idx].
    4. Extract non-zero coefficients; rank by |coef|; select top-28.
    5. Compute overlap (Jaccard, intersection count) vs SHAP top-28
       indices recovered from nb2063_shap_importance_full117.npy.
    6. Run the SAME LGBM(MSE) 5-seed bag (seeds 0, 1, 7, 42, 137) with
       5-fold KFold cross-fit per seed on the LASSO top-28 cols.
    7. Compare mean-bag vs nb2103 K=28 (0.4737/0.4698) at decision
       margin 0.005 (Bonferroni: 2-paradigm comparison -> 0.005 = 0.0025*2).
    8. If best mean-bag passes, run fresh-seed verification on
       kf_seeds {1001, 1002, ..., 1010}; record per-seed RAE
       distribution; promote if fresh_mean_bag <= nb2103 K=28
       median_bag - 0.003.
    9. If reproducible: build deploy CSV (5-seed bag of LGBM-MSE fit on
       ALL 253 unb -> predict residual on 513 -> te = anchor + mean_resid).

OUTPUTS:
    scripts/nb1100_lasso_select.py
    data/processed/nb1100_summary.json
    data/processed/nb1100_lasso_coef_full117.npy
    data/processed/nb1100_lasso_top28_idx.npy
    data/processed/nb1100_mean_bag_oof.npy
    data/processed/nb1100_median_bag_oof.npy
    submissions/nb1100_lasso_top28.csv          (only if promoted)
    data/processed/te_nb1100.npy                (only if promoted)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import importlib.util
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1100"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

# Reuse nb1086 helper to build identical 117-col matrix
NB1086_PATH = Path(__file__).parent / "nb1086_recursive_add.py"

TOP_K = 28
ORIG_SEEDS = [0, 1, 7, 42, 137]
FRESH_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
RESID_FOLDS = 5

# LassoCV grid
LASSO_ALPHAS = np.logspace(-3, 1, 50)
LASSO_CV = 5

# References (PRE-unblind path)
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
# Bonferroni decision margin (2-paradigm comparison)
DECISION_MARGIN = 0.005
PROMOTE_DELTA = 0.003

SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)


def _lgbm_params(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb2063/nb2081/nb2091/nb2103."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LASSO L1 feature selection (different paradigm from SHAP)")
    print(f"        TOP_K   = {TOP_K}    (matches nb2103 K=28)")
    print(f"        ref nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"        decision_margin = {DECISION_MARGIN} (Bonferroni 2-paradigm)")
    print("=" * 78)

    # ---- Reuse nb1086 helper to build identical 117-col matrix ----
    spec = importlib.util.spec_from_file_location("nb1086_mod", NB1086_PATH)
    nb1086_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb1086_mod)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "name" in te.columns:
        mol_names = te["name"].astype(str).tolist()
    else:
        raise KeyError("no name column on test set")
    test_mols = [standardize(s) for s in test_smiles]
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor_unb
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Rebuild 117-col matrix (calls nb1086 helper) ----
    print("\n" + "-" * 78)
    print("REBUILD 117-COL MATRIX (nb1086 helper)")
    print("-" * 78)
    X_te_117, X_unb_117, feat_names, feat_family, n_pool = \
        nb1086_mod._build_full_117_matrices(
            test_smiles, test_mols, n_test, unb_idx
        )
    feat_dim = X_unb_117.shape[1]
    print(f"   X_te_117  = {X_te_117.shape}")
    print(f"   X_unb_117 = {X_unb_117.shape}")
    print(f"   feat_dim  = {feat_dim}")

    # ---- LASSO L1 paradigm: standardize + LassoCV ----
    print("\n" + "-" * 78)
    print("LASSO L1 FEATURE SELECTION")
    print("-" * 78)
    scaler = StandardScaler()
    X_unb_std = scaler.fit_transform(X_unb_117).astype(np.float64)
    print(f"   X_unb standardized: mean~{X_unb_std.mean():+.6f}  "
          f"std~{X_unb_std.std():.4f}")

    t_lasso = time.time()
    lasso = LassoCV(
        cv=LASSO_CV,
        alphas=LASSO_ALPHAS,
        max_iter=20000,
        tol=1e-5,
        n_jobs=2,
        random_state=0,
        selection="cyclic",
    )
    lasso.fit(X_unb_std, residual)
    print(f"   LassoCV fit wall = {time.time() - t_lasso:.1f}s")
    print(f"   best alpha = {lasso.alpha_:.6f}  "
          f"(grid range [{LASSO_ALPHAS.min():.4f}, {LASSO_ALPHAS.max():.4f}])")
    n_alphas_below_best = int(np.sum(LASSO_ALPHAS < lasso.alpha_))
    print(f"   alpha rank in grid = {n_alphas_below_best}/{len(LASSO_ALPHAS)}")

    coef = lasso.coef_.astype(np.float64)
    abs_coef = np.abs(coef)
    n_nonzero = int(np.sum(abs_coef > 1e-12))
    print(f"   non-zero coefficients: {n_nonzero} / {feat_dim}")

    # Save coefficient vector
    np.save(DATA_PROCESSED / f"{TAG}_lasso_coef_full117.npy",
            coef.astype(np.float32))
    print(f"   [save] {DATA_PROCESSED / f'{TAG}_lasso_coef_full117.npy'}")

    # ---- Top-28 by |coef| ----
    if n_nonzero >= TOP_K:
        lasso_rank = np.argsort(-abs_coef).astype(np.int32)
        top_k_idx = lasso_rank[:TOP_K].astype(np.int32)
        print(f"   selecting top-{TOP_K} from {n_nonzero} non-zero")
    else:
        # Fall back: pad with zero-coef features by abs_coef rank
        lasso_rank = np.argsort(-abs_coef).astype(np.int32)
        top_k_idx = lasso_rank[:TOP_K].astype(np.int32)
        print(f"   only {n_nonzero} non-zero (< TOP_K={TOP_K}); "
              f"padding with smallest-|coef| ranked features")

    fam_counts: dict[str, int] = {}
    for i in top_k_idx:
        fam = feat_family[int(i)]
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    print(f"   LASSO top-{TOP_K} family breakdown: {fam_counts}")
    print(f"   LASSO top-10 indices: {top_k_idx[:10].tolist()}")
    print(f"   LASSO top-10 names:   "
          f"{[feat_names[int(i)] for i in top_k_idx[:10]]}")
    print(f"   LASSO top-10 |coef|:  "
          f"{[round(float(abs_coef[int(i)]), 4) for i in top_k_idx[:10]]}")

    np.save(DATA_PROCESSED / f"{TAG}_lasso_top28_idx.npy", top_k_idx)
    print(f"   [save] {DATA_PROCESSED / f'{TAG}_lasso_top28_idx.npy'}")

    # ---- SHAP top-28 from nb2063 importance ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    shap_imp = np.load(NB2063_SHAP_IMP).astype(np.float32)
    shap_rank = np.argsort(-shap_imp).astype(np.int32)
    shap_top28 = shap_rank[:TOP_K].astype(np.int32)
    shap_fam_counts: dict[str, int] = {}
    for i in shap_top28:
        fam = feat_family[int(i)]
        shap_fam_counts[fam] = shap_fam_counts.get(fam, 0) + 1
    print(f"\n   SHAP top-{TOP_K} family breakdown: {shap_fam_counts}")
    print(f"   SHAP top-10 indices: {shap_top28[:10].tolist()}")

    # ---- Overlap analysis ----
    lasso_set = set(int(i) for i in top_k_idx)
    shap_set = set(int(i) for i in shap_top28)
    inter = lasso_set & shap_set
    union = lasso_set | shap_set
    jaccard = len(inter) / max(len(union), 1)
    print("\n" + "-" * 78)
    print("LASSO vs SHAP TOP-28 OVERLAP")
    print("-" * 78)
    print(f"   intersection = {len(inter)} / {TOP_K}  ({len(inter)/TOP_K*100:.1f}%)")
    print(f"   union        = {len(union)}")
    print(f"   Jaccard      = {jaccard:.4f}")
    overlap_idx_sorted = sorted(int(i) for i in inter)
    print(f"   overlap idx (sorted): {overlap_idx_sorted}")

    # ---- LGBM(MSE) 5-seed bag, 5-fold cross-fit on LASSO top-28 ----
    print("\n" + "-" * 78)
    print(f"LGBM(MSE) 5-seed bag x 5-fold cross-fit on LASSO top-{TOP_K}")
    print("-" * 78)
    X_unb_top = X_unb_117[:, top_k_idx].astype(np.float32)
    print(f"   X_unb_top = {X_unb_top.shape}")

    per_seed_corrected = np.zeros((len(ORIG_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae = []
    per_seed_records = []
    for i, s in enumerate(ORIG_SEEDS):
        ts = time.time()
        oof_s = _residual_cross_fit_one_seed(X_unb_top, residual, s)
        pred_s = anchor_unb + oof_s
        per_seed_corrected[i] = pred_s
        rs = float(rae(y_unb, pred_s))
        per_seed_rae.append(rs)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rs,
            "delta_vs_chemprop_aux": rs - rae_anchor,
            "resid_oof_std": float(oof_s.std()),
            "resid_oof_mean": float(oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:4d}: rae_corr = {rs:.4f}  "
              f"(d_anchor = {rs - rae_anchor:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_arr.mean())
    rae_per_seed_std = float(per_seed_arr.std())

    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"   [save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"   [save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    delta_mean = rae_mean_bag - NB2103_K28_MEAN_BAG_REF
    delta_median = rae_median_bag - NB2103_K28_MEDIAN_BAG_REF

    print("\n" + "=" * 78)
    print("LASSO TOP-28 vs nb2103 K=28 (SHAP top-28)")
    print("=" * 78)
    print(f"   LASSO  mean_bag   = {rae_mean_bag:.4f}    "
          f"per_seed_mean = {rae_per_seed_mean:.4f}    "
          f"per_seed_std = {rae_per_seed_std:.4f}")
    print(f"   LASSO  median_bag = {rae_median_bag:.4f}")
    print(f"   nb2103 mean_bag   = {NB2103_K28_MEAN_BAG_REF:.4f}")
    print(f"   nb2103 median_bag = {NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"   delta(mean_bag)   = {delta_mean:+.4f}  "
          f"(margin {DECISION_MARGIN:.3f})")
    print(f"   delta(median_bag) = {delta_median:+.4f}")

    beats_nb2103 = rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    flat_nb2103 = abs(delta_mean) < DECISION_MARGIN
    if beats_nb2103:
        primary_verdict = "LASSO_BEATS_NB2103_K28"
    elif flat_nb2103:
        primary_verdict = "LASSO_FLAT_VS_NB2103_K28"
    else:
        primary_verdict = "LASSO_WORSE_THAN_NB2103_K28"
    print(f"   PRIMARY verdict   = {primary_verdict}")

    # ---- Fresh-seed verification if best mean-bag passes ----
    fresh_block = None
    deploy_block = None
    pass_for_fresh = (rae_mean_bag <= NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
                      or rae_mean_bag <= NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN)
    if pass_for_fresh:
        print("\n" + "-" * 78)
        print(f"FRESH-SEED VERIFICATION {FRESH_SEEDS}")
        print("-" * 78)
        fresh_corr = np.zeros((len(FRESH_SEEDS), n_unb), dtype=np.float64)
        fresh_rae = []
        for i, s in enumerate(FRESH_SEEDS):
            oof_s = _residual_cross_fit_one_seed(X_unb_top, residual, s)
            pred_s = anchor_unb + oof_s
            fresh_corr[i] = pred_s
            rs = float(rae(y_unb, pred_s))
            fresh_rae.append(rs)
            print(f"   fresh seed={s:4d}: rae = {rs:.4f}")
        fresh_mean_bag = float(rae(y_unb, fresh_corr.mean(axis=0)))
        fresh_median_bag = float(rae(y_unb, np.median(fresh_corr, axis=0)))
        fresh_arr = np.array(fresh_rae)
        fresh_per_seed_mean = float(fresh_arr.mean())
        fresh_per_seed_std = float(fresh_arr.std())
        fresh_per_seed_min = float(fresh_arr.min())
        fresh_per_seed_max = float(fresh_arr.max())

        promote_threshold = NB2103_K28_MEDIAN_BAG_REF - PROMOTE_DELTA
        reproducible = fresh_mean_bag <= promote_threshold
        if reproducible:
            fresh_verdict = (
                f"PROMOTE_REPRODUCIBLE_fresh_mean_bag={fresh_mean_bag:.4f}_"
                f"<={promote_threshold:.4f}"
            )
        else:
            fresh_verdict = (
                f"REJECT_LUCKY_SEED_fresh_mean_bag={fresh_mean_bag:.4f}_"
                f">{promote_threshold:.4f}"
            )
        print(f"\n   fresh mean_bag    = {fresh_mean_bag:.4f}")
        print(f"   fresh median_bag  = {fresh_median_bag:.4f}")
        print(f"   fresh per-seed mu = {fresh_per_seed_mean:.4f}  "
              f"std = {fresh_per_seed_std:.4f}  "
              f"[min {fresh_per_seed_min:.4f}, max {fresh_per_seed_max:.4f}]")
        print(f"   promote threshold = {promote_threshold:.4f}  "
              f"(nb2103 K=28 median_bag - {PROMOTE_DELTA:.3f})")
        print(f"   FRESH verdict     = {fresh_verdict}")
        fresh_block = {
            "fresh_seeds": FRESH_SEEDS,
            "fresh_per_seed_rae": fresh_rae,
            "fresh_per_seed_mean": fresh_per_seed_mean,
            "fresh_per_seed_std": fresh_per_seed_std,
            "fresh_per_seed_min": fresh_per_seed_min,
            "fresh_per_seed_max": fresh_per_seed_max,
            "fresh_mean_bag": fresh_mean_bag,
            "fresh_median_bag": fresh_median_bag,
            "promote_threshold": promote_threshold,
            "promote_delta": PROMOTE_DELTA,
            "delta_fresh_mean_vs_nb2103_median": (
                fresh_mean_bag - NB2103_K28_MEDIAN_BAG_REF
            ),
            "reproducible": bool(reproducible),
            "fresh_verdict": fresh_verdict,
        }

        # ---- If reproducible: build deploy CSV ----
        if reproducible:
            print("\n" + "-" * 78)
            print(f"DEPLOY: {len(ORIG_SEEDS)} LGBM(MSE) on ALL 253 unb -> "
                  f"predict 513")
            print("-" * 78)
            X_te_top = X_te_117[:, top_k_idx].astype(np.float32)
            all_resid_513 = np.zeros((len(ORIG_SEEDS), n_test), dtype=np.float64)
            for k, s in enumerate(ORIG_SEEDS):
                t_in = time.time()
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_unb_top, residual)
                resid_513 = mdl.predict(X_te_top)
                all_resid_513[k] = resid_513
                print(f"   fit {k+1}/{len(ORIG_SEEDS)}  seed={s:4d}  "
                      f"resid_mean={resid_513.mean():+.4f}  "
                      f"resid_std={resid_513.std():.4f}  "
                      f"wall={time.time() - t_in:.1f}s")
            mean_resid_513 = all_resid_513.mean(axis=0)
            te_nb1100 = te_anchor_513 + mean_resid_513
            in_pred_unb = te_nb1100[unb_idx]
            rae_in_unb = float(rae(y_unb, in_pred_unb))
            print(f"\n   in-sample te[unb_idx] RAE = {rae_in_unb:.4f}  "
                  f"(anchor {rae_anchor:.4f})")

            df_sub = pd.DataFrame({
                "SMILES": test_smiles,
                "Molecule Name": mol_names,
                "pEC50": te_nb1100.astype(np.float32),
            })
            if len(df_sub) != 513:
                raise ValueError(f"submission rows {len(df_sub)} != 513")
            sub_path = SUBMISSIONS_DIR / f"{TAG}_lasso_top28.csv"
            df_sub.to_csv(sub_path, index=False)
            print(f"   [save] submission CSV: {sub_path}")
            te_path = DATA_PROCESSED / f"te_{TAG}.npy"
            np.save(te_path, te_nb1100.astype(np.float32))
            print(f"   [save] te artifact:    {te_path}")
            deploy_block = {
                "deployed": True,
                "submission_path": str(sub_path),
                "te_path": str(te_path),
                "te_mean": float(te_nb1100.mean()),
                "te_std": float(te_nb1100.std()),
                "te_min": float(te_nb1100.min()),
                "te_max": float(te_nb1100.max()),
                "in_sample_rae_unb": rae_in_unb,
            }
        else:
            deploy_block = {"deployed": False,
                            "reason": "fresh-seed not reproducible"}
    else:
        print("\n   [skip] LASSO mean_bag does not pass margin vs nb2103 "
              "K=28; fresh-seed verification and deploy SKIPPED")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "method": ("lasso_l1_feature_selection_top28_then_lgbm_mse_"
                   "5seed_bag_on_117col_paradigm_compare_with_shap_K28"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "feat_dim_full": int(feat_dim),
        "n_chembl_pool": int(n_pool),
        "top_k": int(TOP_K),
        "decision_margin": DECISION_MARGIN,
        "decision_margin_note": ("0.005 = Bonferroni 2-paradigm "
                                  "(0.0025 base * 2)"),
        "promote_delta": PROMOTE_DELTA,
        "resid_folds": RESID_FOLDS,
        "orig_seeds": ORIG_SEEDS,
        "fresh_seeds": FRESH_SEEDS,
        "lasso_alphas_grid": LASSO_ALPHAS.tolist(),
        "lasso_alpha_best": float(lasso.alpha_),
        "lasso_alpha_rank_in_grid": n_alphas_below_best,
        "lasso_alpha_grid_len": int(len(LASSO_ALPHAS)),
        "lasso_n_nonzero": n_nonzero,
        "lasso_top28_idx_in_117": top_k_idx.tolist(),
        "lasso_top28_names": [feat_names[int(i)] for i in top_k_idx],
        "lasso_top28_families": [feat_family[int(i)] for i in top_k_idx],
        "lasso_top28_abs_coef": [float(abs_coef[int(i)]) for i in top_k_idx],
        "lasso_top28_family_counts": fam_counts,
        "shap_top28_idx_in_117": shap_top28.tolist(),
        "shap_top28_family_counts": shap_fam_counts,
        "overlap_count": int(len(inter)),
        "overlap_jaccard": float(jaccard),
        "overlap_idx_sorted": overlap_idx_sorted,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "per_seed_records": per_seed_records,
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb2103_K28": delta_mean,
        "delta_median_bag_vs_nb2103_K28_median": delta_median,
        "beats_nb2103_K28": bool(beats_nb2103),
        "flat_vs_nb2103_K28": bool(flat_nb2103),
        "primary_verdict": primary_verdict,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "fresh_block": fresh_block,
        "deploy_block": deploy_block,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "lasso_alpha_best",
        "lasso_n_nonzero",
        "overlap_count",
        "overlap_jaccard",
        "rae_anchor_chemprop_aux",
        "rae_mean_bag",
        "rae_median_bag",
        "delta_mean_bag_vs_nb2103_K28",
        "primary_verdict",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    if res.get("fresh_block"):
        fb = res["fresh_block"]
        print("\n==== FRESH-SEED VERIFICATION ====")
        print(f"  fresh_mean_bag   : {fb['fresh_mean_bag']:.4f}")
        print(f"  fresh_median_bag : {fb['fresh_median_bag']:.4f}")
        print(f"  fresh_per_seed_mu: {fb['fresh_per_seed_mean']:.4f}")
        print(f"  fresh_per_seed_sd: {fb['fresh_per_seed_std']:.4f}")
        print(f"  reproducible     : {fb['reproducible']}")
        print(f"  fresh_verdict    : {fb['fresh_verdict']}")
    if res.get("deploy_block"):
        db = res["deploy_block"]
        print("\n==== DEPLOY ====")
        for k, v in db.items():
            print(f"  {k}: {v}")
