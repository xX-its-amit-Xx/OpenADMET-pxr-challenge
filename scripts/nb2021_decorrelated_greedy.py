"""nb2021 -- SHAP-decorrelated greedy single-feature add beyond K=28.

HYPOTHESIS:
    nb1086 already swept all 89 (= 117 - 28) candidate adds at K=29 and found
    the best by mean-bag (AtomPair_bit_1313, +0.0052 gain). nb1086_verify
    then re-tested on FRESH kf_seeds (1001-1010) and per memory
    `feedback_stack_overfitting`, gains past ~5 components often fail
    reproducibility — feature add at K=29+ usually adds noise.

    This script tightens the candidate pool with a DECORRELATION FILTER
    before greedy add: only candidates whose max |Pearson| with the
    existing top-28 SHAP set is < 0.50 are eligible. The intuition: if a
    candidate is already highly correlated with one of the SHAP-28 columns,
    it cannot carry genuinely orthogonal signal — any small gain is noise.

PROTOCOL:
    1. Rebuild the 117-col 5-way K-tuned feature matrix (same helper as
       nb1086_recursive_add._build_full_117_matrices).
    2. Identify top-28 SHAP indices = argsort(-nb2063_shap_importance)[:28].
    3. For each of the 89 non-SHAP candidates, compute Pearson correlation
       with every top-28 column on the 253-unb matrix.
    4. Filter to candidates with max |Pearson with top-28| < DECORR_THRESH
       (= 0.50). Call this the "decorrelated pool".
    5. Greedy add: for each decorrelated candidate, fit K=29 LGBM(MSE)
       5-seed bag (0, 1, 7, 42, 137), 5-fold cross-fit on
       residual = y_unb - chemprop_aux te[unb_idx], compute mean-bag RAE.
    6. Pick the candidate with best mean-bag RAE.
    7. Verify the winner on 10 FRESH kf_seeds (1001-1010) to avoid the
       nb1086 lucky-seed trap.
    8. If gain >= 0.003 reproducibly (fresh-seed mean-bag RAE
       <= nb2103 K=28 mean-bag - 0.003 = 0.4707), build deploy K=29 CSV.

Outputs:
    scripts/nb2021_decorrelated_greedy.py
    data/processed/nb2021_summary.json
    submissions/nb2021_decorrelated_K29.csv  (only if fresh-seed gain >= 0.003)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
import importlib.util
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Reuse the 117-col matrix builder from nb1086 (identical signature)
nb1086_path = Path(__file__).parent / "nb1086_recursive_add.py"
spec = importlib.util.spec_from_file_location("nb1086_mod", nb1086_path)
nb1086_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb1086_mod)

from pxr.chem import standardize
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2021"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

TOP_K_SHAP = 28
K_ADDED = 29
RESID_FOLDS = 5
ORIG_SEEDS = [0, 1, 7, 42, 137]
FRESH_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]

# Decision thresholds
DECORR_THRESH = 0.50          # max |Pearson with top-28| must be below this
DECISION_MARGIN = 0.003       # promotion margin on fresh-seed verification

# Reference numbers from nb2103 (PRE-unblind 5-seed cross-fit on K=28)
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
CHEMPROP_AUX_REF = 0.6216

SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)


def _lgbm_params(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb2063/nb2103/nb1086."""
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


def _bag_rae(X: np.ndarray, residual: np.ndarray, anchor_unb: np.ndarray,
             y_unb: np.ndarray, seeds: list[int]) -> tuple[float, float, list[float]]:
    """Returns (mean_bag_rae, median_bag_rae, per_seed_rae)."""
    n = len(y_unb)
    per_seed_corr = np.zeros((len(seeds), n), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(seeds):
        oof_s = _residual_cross_fit_one_seed(X, residual, s)
        pred_s = anchor_unb + oof_s
        per_seed_corr[i] = pred_s
        per_seed_rae.append(float(rae(y_unb, pred_s)))
    mean_bag = float(rae(y_unb, per_seed_corr.mean(axis=0)))
    median_bag = float(rae(y_unb, np.median(per_seed_corr, axis=0)))
    return mean_bag, median_bag, per_seed_rae


def _pearson_matrix(X: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation. Columns with zero variance return 0 corr."""
    Xc = X.astype(np.float64)
    mu = Xc.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=0, keepdims=True)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (Xc - mu) / sd_safe
    # Zero-out zero-variance columns so they correlate 0 with everything
    zero_var = (sd <= 1e-12).ravel()
    if zero_var.any():
        Z[:, zero_var] = 0.0
    C = (Z.T @ Z) / Xc.shape[0]
    return C


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SHAP-DECORRELATED greedy single-feature add beyond K=28")
    print(f"          anchor={ANCHOR}  decorr_thresh={DECORR_THRESH:.2f}  "
          f"K_added={K_ADDED}")
    print(f"          orig seeds  : {ORIG_SEEDS}")
    print(f"          fresh seeds : {FRESH_SEEDS}")
    print(f"          ref nb2103 K=28 mean_bag = {NB2103_K28_MEAN_BAG_REF:.4f}"
          f" / median_bag = {NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"          promotion threshold (fresh mean-bag) "
          f"= {NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:.4f}"
          f" (= nb2103 K=28 - {DECISION_MARGIN:.3f})")
    print("=" * 78)

    # ---- Load SHAP importance + top-28 indices ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    shap_imp = np.load(NB2063_SHAP_IMP).astype(np.float32)
    if shap_imp.shape[0] != 117:
        raise ValueError(f"SHAP importance length {shap_imp.shape[0]} != 117")
    full_rank_order = np.argsort(-shap_imp).astype(np.int32)
    top28_idx = full_rank_order[:TOP_K_SHAP].astype(np.int32)
    remaining_idx = np.array(
        [i for i in range(117) if i not in set(top28_idx.tolist())],
        dtype=np.int32,
    )
    print(f"[shap] top-28 indices (head 10): {top28_idx[:10].tolist()}")
    print(f"[shap] {len(remaining_idx)} candidate adds (heads): "
          f"{remaining_idx[:10].tolist()} ...")

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

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor_unb
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Build full 117-col matrices (reuse nb1086 helper) ----
    print("\n" + "-" * 78)
    print("BUILD 117-COL 5-WAY K-TUNED MATRIX (reusing nb1086 builder)")
    print("-" * 78)
    X_te_117, X_unb_117, feat_names, feat_family, n_pool = \
        nb1086_mod._build_full_117_matrices(
            test_smiles, test_mols, n_test, unb_idx
        )
    print(f"   X_te_117  = {X_te_117.shape}")
    print(f"   X_unb_117 = {X_unb_117.shape}")

    # ---- Compute Pearson decorrelation filter ----
    print("\n" + "-" * 78)
    print(f"DECORRELATION FILTER: max |Pearson with top-28| < "
          f"{DECORR_THRESH:.2f}")
    print("-" * 78)
    C_full = _pearson_matrix(X_unb_117.astype(np.float32))
    # Each row of C_sub: candidate's correlations with the 28 top-SHAP columns
    C_cand_to_top = np.abs(C_full[remaining_idx][:, top28_idx])
    max_corr_per_cand = C_cand_to_top.max(axis=1)
    # Best-matched top-28 column index (for diagnostics)
    best_match_top28_arg = C_cand_to_top.argmax(axis=1)
    best_match_top28_idx = top28_idx[best_match_top28_arg]

    decorr_mask = max_corr_per_cand < DECORR_THRESH
    decorr_cand_idx = remaining_idx[decorr_mask]
    print(f"   89 raw candidates  ->  {int(decorr_mask.sum())} decorrelated "
          f"(threshold {DECORR_THRESH:.2f})")
    print(f"   max |Pearson| stats over 89 cands: "
          f"min={max_corr_per_cand.min():.3f}  "
          f"median={np.median(max_corr_per_cand):.3f}  "
          f"max={max_corr_per_cand.max():.3f}")
    if len(decorr_cand_idx) == 0:
        print("   [WARN] decorrelated pool is empty -- nothing to add.")
    # Save per-candidate diagnostic
    decorr_records: list[dict] = []
    for ki, j in enumerate(remaining_idx):
        decorr_records.append({
            "add_feat_idx_in_117": int(j),
            "add_feat_name": feat_names[int(j)],
            "add_feat_family": feat_family[int(j)],
            "shap_rank_in_117": int(np.where(full_rank_order == j)[0][0]),
            "shap_importance": float(shap_imp[int(j)]),
            "max_abs_pearson_with_top28": float(max_corr_per_cand[ki]),
            "best_match_top28_idx_in_117": int(best_match_top28_idx[ki]),
            "best_match_top28_name": feat_names[int(best_match_top28_idx[ki])],
            "passes_decorr": bool(decorr_mask[ki]),
        })

    # ---- Sanity: K=28 baseline ----
    print("\n" + "-" * 78)
    print(f"SANITY: re-run K=28 baseline on rebuilt X_unb_117 (orig seeds)")
    print("-" * 78)
    X_unb_28 = X_unb_117[:, top28_idx].astype(np.float32)
    base_mean_bag, base_median_bag, base_per_seed_rae = _bag_rae(
        X_unb_28, residual, anchor_unb, y_unb, ORIG_SEEDS
    )
    for i, s in enumerate(ORIG_SEEDS):
        print(f"   K=28 seed={s:4d}: rae = {base_per_seed_rae[i]:.4f}")
    print(f"   K=28 mean-bag   = {base_mean_bag:.4f}  "
          f"(ref {NB2103_K28_MEAN_BAG_REF:.4f}, "
          f"delta {base_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"   K=28 median-bag = {base_median_bag:.4f}  "
          f"(ref {NB2103_K28_MEDIAN_BAG_REF:.4f}, "
          f"delta {base_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")

    # ---- Greedy add from decorrelated pool (orig seeds) ----
    print("\n" + "-" * 78)
    print(f"GREEDY ADD: {len(decorr_cand_idx)} decorrelated K=29 candidates "
          f"(orig seeds)")
    print("-" * 78)
    per_add_records: list[dict] = []
    for i_add, j in enumerate(decorr_cand_idx):
        ts = time.time()
        cols = np.concatenate(
            [top28_idx, np.array([j], dtype=np.int32)]
        ).astype(np.int32)
        X_unb_K29 = X_unb_117[:, cols].astype(np.float32)
        mean_bag_j, median_bag_j, per_seed_j = _bag_rae(
            X_unb_K29, residual, anchor_unb, y_unb, ORIG_SEEDS
        )
        gain_mean = NB2103_K28_MEAN_BAG_REF - mean_bag_j
        gain_med = NB2103_K28_MEDIAN_BAG_REF - median_bag_j
        rec = {
            "i_add": int(i_add),
            "add_feat_idx_in_117": int(j),
            "add_feat_name": feat_names[int(j)],
            "add_feat_family": feat_family[int(j)],
            "shap_importance": float(shap_imp[int(j)]),
            "shap_rank_in_117": int(np.where(full_rank_order == j)[0][0]),
            "max_abs_pearson_with_top28": float(
                max_corr_per_cand[np.where(remaining_idx == j)[0][0]]
            ),
            "per_seed_rae_orig": per_seed_j,
            "rae_mean_bag_orig": mean_bag_j,
            "rae_median_bag_orig": median_bag_j,
            "gain_mean_bag_vs_K28": gain_mean,
            "gain_median_bag_vs_K28": gain_med,
            "wall_sec": round(time.time() - ts, 2),
        }
        per_add_records.append(rec)
        if i_add < 5 or i_add % 5 == 0 or gain_mean > 0.003:
            print(f"   add#{i_add:3d}  feat_idx={int(j):3d}  "
                  f"name={feat_names[int(j)]:<28s}  "
                  f"|r|_max={float(max_corr_per_cand[np.where(remaining_idx == j)[0][0]]):.3f}  "
                  f"mean_bag={mean_bag_j:.4f}  "
                  f"gain={gain_mean:+.4f}  ({time.time() - ts:.1f}s)")

    # ---- Rank by gain_mean_bag ----
    sorted_by_gain_mean = sorted(
        per_add_records, key=lambda r: -r["gain_mean_bag_vs_K28"]
    )
    print("\n" + "=" * 78)
    print(f"DECORRELATED RANKING by gain_mean_bag_vs_K28 (top 10)")
    print("=" * 78)
    print(f"   {'rank':>4s}  {'feat_idx':>8s}  {'name':<28s}  "
          f"{'|r|_max':>7s}  {'mean_bag':>10s}  {'gain':>9s}  {'shap_rk':>7s}")
    for ri, r in enumerate(sorted_by_gain_mean[:10]):
        print(f"   {ri + 1:>4d}  {r['add_feat_idx_in_117']:>8d}  "
              f"{r['add_feat_name']:<28s}  "
              f"{r['max_abs_pearson_with_top28']:>7.3f}  "
              f"{r['rae_mean_bag_orig']:>10.4f}  "
              f"{r['gain_mean_bag_vs_K28']:>+9.4f}  "
              f"{r['shap_rank_in_117']:>7d}")

    # ---- Pick winner ----
    if len(sorted_by_gain_mean) == 0:
        print("\n   No decorrelated candidates -- aborting.")
        winner = None
        fresh_mean_bag = None
        fresh_median_bag = None
        fresh_per_seed_rae = None
        promote = False
        verdict = "NO_DECORRELATED_CANDIDATES"
        do_deploy = False
        deploy_csv_path = None
        deploy_in_rae_unb = None
        deploy_stats = None
    else:
        winner = sorted_by_gain_mean[0]
        winner_j = int(winner["add_feat_idx_in_117"])
        print(f"\n   WINNER (orig-seed): idx={winner_j}  "
              f"name={winner['add_feat_name']}  "
              f"|r|_max={winner['max_abs_pearson_with_top28']:.3f}  "
              f"orig mean_bag={winner['rae_mean_bag_orig']:.4f}  "
              f"gain={winner['gain_mean_bag_vs_K28']:+.4f}")

        # ---- FRESH-SEED VERIFICATION ----
        print("\n" + "-" * 78)
        print(f"FRESH-SEED VERIFICATION: 10 kf_seeds 1001-1010 (anti-lucky-seed)")
        print("-" * 78)
        cols = np.concatenate(
            [top28_idx, np.array([winner_j], dtype=np.int32)]
        ).astype(np.int32)
        X_unb_K29_win = X_unb_117[:, cols].astype(np.float32)
        fresh_mean_bag, fresh_median_bag, fresh_per_seed_rae = _bag_rae(
            X_unb_K29_win, residual, anchor_unb, y_unb, FRESH_SEEDS
        )
        for i, s in enumerate(FRESH_SEEDS):
            print(f"   fresh seed={s:4d}: rae = {fresh_per_seed_rae[i]:.4f}")
        print(f"   fresh mean-bag   = {fresh_mean_bag:.4f}")
        print(f"   fresh median-bag = {fresh_median_bag:.4f}")
        print(f"   nb2103 K=28 mean = {NB2103_K28_MEAN_BAG_REF:.4f}")
        fresh_gain_mean = NB2103_K28_MEAN_BAG_REF - fresh_mean_bag
        fresh_gain_med = NB2103_K28_MEDIAN_BAG_REF - fresh_median_bag
        print(f"   fresh gain (mean-bag)   = {fresh_gain_mean:+.4f}")
        print(f"   fresh gain (median-bag) = {fresh_gain_med:+.4f}")

        promote = fresh_gain_mean >= DECISION_MARGIN
        if promote:
            verdict = (
                f"PROMOTE_DECORR_REPRODUCIBLE_idx={winner_j}_"
                f"name={winner['add_feat_name']}_"
                f"fresh_gain_mean={fresh_gain_mean:+.4f}_"
                f">=margin_{DECISION_MARGIN:.3f}"
            )
            print(f"\n   VERDICT: PROMOTE (fresh gain {fresh_gain_mean:+.4f} "
                  f">= {DECISION_MARGIN:.3f})")
        else:
            verdict = (
                f"REJECT_DECORR_NOT_REPRODUCIBLE_idx={winner_j}_"
                f"name={winner['add_feat_name']}_"
                f"fresh_gain_mean={fresh_gain_mean:+.4f}_"
                f"<margin_{DECISION_MARGIN:.3f}"
            )
            print(f"\n   VERDICT: REJECT (fresh gain {fresh_gain_mean:+.4f} "
                  f"< {DECISION_MARGIN:.3f}) -- lucky-seed artifact")

        # ---- Deploy if promoted ----
        do_deploy = promote
        deploy_csv_path = None
        deploy_in_rae_unb = None
        deploy_stats = None
        if do_deploy:
            print("\n" + "-" * 78)
            print(f"DEPLOY: K=29 (top-28 SHAP + {winner['add_feat_name']}), "
                  f"5x5=25-bag fit-on-all-253 / predict-513")
            print("-" * 78)
            X_te_K29 = X_te_117[:, cols].astype(np.float32)
            X_unb_K29 = X_unb_K29_win
            outer = [0, 1, 7, 42, 137]
            inner = [0, 1, 7, 42, 137]
            n_total = len(outer) * len(inner)
            all_resid_513 = np.zeros((n_total, n_test), dtype=np.float64)
            k_global = 0
            for o in outer:
                for s in inner:
                    seed = o * 1000 + s
                    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
                    mdl.fit(X_unb_K29, residual)
                    all_resid_513[k_global] = mdl.predict(X_te_K29)
                    k_global += 1
            median_resid_513 = np.median(all_resid_513, axis=0)
            te_deploy = te_anchor_513 + median_resid_513
            deploy_in_rae_unb = float(rae(y_unb, te_deploy[unb_idx]))
            df_sub = pd.DataFrame({
                "SMILES": test_smiles,
                "Molecule Name": mol_names,
                "pEC50": te_deploy.astype(np.float32),
            })
            if len(df_sub) != n_test:
                raise ValueError(f"submission rows {len(df_sub)} != {n_test}")
            deploy_csv_path = SUBMISSIONS_DIR / "nb2021_decorrelated_K29.csv"
            df_sub.to_csv(deploy_csv_path, index=False)
            deploy_stats = {
                "te_mean": float(te_deploy.mean()),
                "te_std": float(te_deploy.std()),
                "te_min": float(te_deploy.min()),
                "te_max": float(te_deploy.max()),
                "median_resid_mean": float(median_resid_513.mean()),
                "median_resid_std": float(median_resid_513.std()),
            }
            print(f"   [save] {deploy_csv_path}  ({len(df_sub)} rows)")
            print(f"   in-sample RAE on unb_idx (median bag) = "
                  f"{deploy_in_rae_unb:.4f}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": (
            "shap_decorrelated_greedy_single_feature_add_beyond_top28_SHAP_"
            "117col_5way_with_fresh_seed_verification"
        ),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": (
            "117-col 5-way K-tuned matrix (AtomPair-25 / MACCS-20 / "
            "Mordred-20 / ChempropEmbed-20 / Avalon-30 + ChEMBL kNN-2), "
            "top-28 SHAP set from nb2063 + 1 candidate add at K=29, "
            "decorrelation-filtered"
        ),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "feat_dim_full": 117,
        "K_anchor": TOP_K_SHAP,
        "K_added": K_ADDED,
        "decorr_thresh": DECORR_THRESH,
        "decision_margin_fresh_seed": DECISION_MARGIN,
        "n_raw_candidates": int(len(remaining_idx)),
        "n_decorrelated_candidates": int(len(decorr_cand_idx)),
        "n_chembl_pool": int(n_pool),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "resid_seeds_orig": ORIG_SEEDS,
        "resid_seeds_fresh": FRESH_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "top28_idx_in_117": top28_idx.tolist(),
        "remaining_idx_in_117": remaining_idx.tolist(),
        "decorrelated_idx_in_117": decorr_cand_idx.tolist(),
        "decorr_records_all89": decorr_records,
        "k28_baseline_mean_bag_rebuilt": base_mean_bag,
        "k28_baseline_median_bag_rebuilt": base_median_bag,
        "k28_baseline_per_seed_rae": base_per_seed_rae,
        "k28_baseline_mean_bag_ref_nb2103": NB2103_K28_MEAN_BAG_REF,
        "k28_baseline_median_bag_ref_nb2103": NB2103_K28_MEDIAN_BAG_REF,
        "per_add_records_decorr": per_add_records,
        "top_10_by_gain_mean_bag_decorr": [
            {
                "rank": ri + 1,
                "add_feat_idx_in_117": r["add_feat_idx_in_117"],
                "add_feat_name": r["add_feat_name"],
                "add_feat_family": r["add_feat_family"],
                "shap_rank_in_117": r["shap_rank_in_117"],
                "max_abs_pearson_with_top28": r["max_abs_pearson_with_top28"],
                "rae_mean_bag_orig": r["rae_mean_bag_orig"],
                "rae_median_bag_orig": r["rae_median_bag_orig"],
                "gain_mean_bag_vs_K28": r["gain_mean_bag_vs_K28"],
                "gain_median_bag_vs_K28": r["gain_median_bag_vs_K28"],
            }
            for ri, r in enumerate(sorted_by_gain_mean[:10])
        ],
        "winner_orig_seed": (
            None if winner is None else {
                "add_feat_idx_in_117": int(winner["add_feat_idx_in_117"]),
                "add_feat_name": winner["add_feat_name"],
                "add_feat_family": winner["add_feat_family"],
                "shap_rank_in_117": winner["shap_rank_in_117"],
                "max_abs_pearson_with_top28": winner[
                    "max_abs_pearson_with_top28"
                ],
                "rae_mean_bag_orig": winner["rae_mean_bag_orig"],
                "rae_median_bag_orig": winner["rae_median_bag_orig"],
                "gain_mean_bag_vs_K28": winner["gain_mean_bag_vs_K28"],
                "per_seed_rae_orig": winner["per_seed_rae_orig"],
            }
        ),
        "fresh_seed_verification": (
            None if winner is None else {
                "fresh_seeds": FRESH_SEEDS,
                "fresh_per_seed_rae": fresh_per_seed_rae,
                "fresh_mean_bag": fresh_mean_bag,
                "fresh_median_bag": fresh_median_bag,
                "fresh_gain_mean_bag_vs_nb2103_K28": (
                    NB2103_K28_MEAN_BAG_REF - fresh_mean_bag
                ),
                "fresh_gain_median_bag_vs_nb2103_K28": (
                    NB2103_K28_MEDIAN_BAG_REF - fresh_median_bag
                ),
                "promote_threshold_mean_bag": (
                    NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
                ),
                "promote": bool(promote),
            }
        ),
        "deploy": bool(do_deploy),
        "deploy_csv_path": str(deploy_csv_path) if deploy_csv_path else None,
        "deploy_in_rae_unb": deploy_in_rae_unb,
        "deploy_stats": deploy_stats,
        "verdict": verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
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
        "K_anchor", "K_added", "decorr_thresh",
        "n_raw_candidates", "n_decorrelated_candidates",
        "rae_anchor_chemprop_aux",
        "k28_baseline_mean_bag_rebuilt",
        "k28_baseline_median_bag_rebuilt",
        "winner_orig_seed",
        "fresh_seed_verification",
        "deploy", "deploy_csv_path", "deploy_in_rae_unb",
        "verdict",
    ):
        v = res.get(k)
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"      {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
