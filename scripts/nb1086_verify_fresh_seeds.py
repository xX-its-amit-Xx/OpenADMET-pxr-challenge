"""nb1086 verify -- fresh kf_seeds reproducibility test for K=29 add.

Re-runs the K=29 (top-28 SHAP + AtomPair_bit_1313 at idx 15) on 10 FRESH
kf_seeds (1001-1010) to check whether the +0.0052 gain vs nb2103 K=28
(mean-bag) was real or a lucky-seed artifact of seeds {0,1,7,42,137}.

PROTOCOL:
    1. Rebuild X_unb_117 using the SAME helper from nb1086 main script.
    2. Slice to K=29 = top-28 indices + AtomPair_bit_1313 (idx 15).
    3. For each kf_seed in 1001-1010:
         5-fold KFold(shuffle=True, random_state=seed) cross-fit
         LGBM(MSE) on residual = y_unb - chemprop_aux te[unb_idx]
         pred = anchor_unb + oof
         rae = RAE(y_unb, pred)
    4. Compare distribution mean to nb2112 (=0.4698 median-bag) - 0.003.
    5. Verdict: PROMOTE if mean(fresh) <= 0.4698 - 0.003 = 0.4668
                REJECT otherwise (lucky-seed).
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

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Reuse the helper from the main nb1086 script
import importlib.util
nb1086_path = Path(__file__).parent / "nb1086_recursive_add.py"
spec = importlib.util.spec_from_file_location("nb1086", nb1086_path)
nb1086_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb1086_mod)

from pxr.chem import standardize
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1086_verify"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

TOP_K_SHAP = 28
BEST_ADD_IDX = 15  # AtomPair_bit_1313
ORIG_SEEDS = [0, 1, 7, 42, 137]
FRESH_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
RESID_FOLDS = 5

NB2112_MEDIAN_BAG = 0.4697789387514748
NB2112_MEAN_BAG = 0.4736869992362601
PROMOTE_DELTA = 0.003  # mean(fresh) <= nb2112_median - 0.003 = 0.4668


def _lgbm_params(seed: int) -> dict:
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
    print(f"{TAG} -- fresh kf_seeds reproducibility test for K=29 best add")
    print(f"          best_add_idx_in_117={BEST_ADD_IDX} (AtomPair_bit_1313)")
    print(f"          original seeds  : {ORIG_SEEDS}")
    print(f"          fresh    seeds  : {FRESH_SEEDS}")
    print(f"          nb2112 median_bag baseline = {NB2112_MEDIAN_BAG:.4f}")
    print(f"          promote threshold = {NB2112_MEDIAN_BAG - PROMOTE_DELTA:.4f}"
          f" (= nb2112 - {PROMOTE_DELTA:.3f})")
    print("=" * 78)

    # ---- Load SHAP importance + top-28 indices (must match nb1086) ----
    shap_imp = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp).astype(np.int32)
    top28_idx = full_rank_order[:TOP_K_SHAP].astype(np.int32)
    print(f"[shap] top-28 indices (head 10): {top28_idx[:10].tolist()}")

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    test_mols = [standardize(s) for s in test_smiles]
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor_unb

    # ---- Rebuild 117-col matrix (calls nb1086 helper) ----
    print("\n" + "-" * 78)
    print("REBUILD 117-COL MATRIX")
    print("-" * 78)
    X_te_117, X_unb_117, feat_names, feat_family, n_pool = \
        nb1086_mod._build_full_117_matrices(
            test_smiles, test_mols, n_test, unb_idx
        )
    print(f"   X_unb_117 shape = {X_unb_117.shape}")
    print(f"   add_feat_name at idx {BEST_ADD_IDX}: {feat_names[BEST_ADD_IDX]}")

    # ---- Build K=29 = top-28 + add idx 15 ----
    cols = np.concatenate(
        [top28_idx, np.array([BEST_ADD_IDX], dtype=np.int32)]
    ).astype(np.int32)
    X_unb_K29 = X_unb_117[:, cols].astype(np.float32)
    print(f"   X_unb_K29 shape = {X_unb_K29.shape}")

    # ---- Re-run ORIGINAL seeds (must match nb1086 exactly) ----
    print("\n" + "-" * 78)
    print("RE-RUN ORIGINAL SEEDS (sanity check vs nb1086 0.4685 mean-bag)")
    print("-" * 78)
    orig_per_seed_corr = np.zeros((len(ORIG_SEEDS), n_unb), dtype=np.float64)
    orig_per_seed_rae = []
    for i, s in enumerate(ORIG_SEEDS):
        oof_s = _residual_cross_fit_one_seed(X_unb_K29, residual, s)
        pred_s = anchor_unb + oof_s
        orig_per_seed_corr[i] = pred_s
        rs = float(rae(y_unb, pred_s))
        orig_per_seed_rae.append(rs)
        print(f"   orig seed={s:4d}: rae = {rs:.4f}")
    orig_mean_bag = float(rae(y_unb, orig_per_seed_corr.mean(axis=0)))
    orig_median_bag = float(rae(y_unb, np.median(orig_per_seed_corr, axis=0)))
    print(f"   orig mean-bag   = {orig_mean_bag:.4f}  (nb1086 ref = 0.4685)")
    print(f"   orig median-bag = {orig_median_bag:.4f}  (nb1086 ref = 0.4669)")

    # ---- FRESH seeds 1001-1010 ----
    print("\n" + "-" * 78)
    print("FRESH SEEDS 1001-1010 (reproducibility test)")
    print("-" * 78)
    fresh_per_seed_corr = np.zeros((len(FRESH_SEEDS), n_unb), dtype=np.float64)
    fresh_per_seed_rae = []
    for i, s in enumerate(FRESH_SEEDS):
        oof_s = _residual_cross_fit_one_seed(X_unb_K29, residual, s)
        pred_s = anchor_unb + oof_s
        fresh_per_seed_corr[i] = pred_s
        rs = float(rae(y_unb, pred_s))
        fresh_per_seed_rae.append(rs)
        print(f"   fresh seed={s:4d}: rae = {rs:.4f}")
    fresh_mean_bag = float(rae(y_unb, fresh_per_seed_corr.mean(axis=0)))
    fresh_median_bag = float(rae(y_unb, np.median(fresh_per_seed_corr, axis=0)))
    fresh_per_seed_mean = float(np.mean(fresh_per_seed_rae))
    fresh_per_seed_std = float(np.std(fresh_per_seed_rae))
    fresh_per_seed_min = float(np.min(fresh_per_seed_rae))
    fresh_per_seed_max = float(np.max(fresh_per_seed_rae))

    print("\n" + "=" * 78)
    print("FRESH-SEED DISTRIBUTION")
    print("=" * 78)
    print(f"   per-seed mean   = {fresh_per_seed_mean:.4f}")
    print(f"   per-seed std    = {fresh_per_seed_std:.4f}")
    print(f"   per-seed min    = {fresh_per_seed_min:.4f}")
    print(f"   per-seed max    = {fresh_per_seed_max:.4f}")
    print(f"   mean-bag        = {fresh_mean_bag:.4f}")
    print(f"   median-bag      = {fresh_median_bag:.4f}")
    print(f"   nb2112 median   = {NB2112_MEDIAN_BAG:.4f}")
    print(f"   nb2112 mean     = {NB2112_MEAN_BAG:.4f}")
    print(f"   delta (fresh mean-bag - nb2112 median): "
          f"{fresh_mean_bag - NB2112_MEDIAN_BAG:+.4f}")
    print(f"   delta (fresh mean-bag - nb2112 mean):   "
          f"{fresh_mean_bag - NB2112_MEAN_BAG:+.4f}")

    # ---- Verdict ----
    promote_threshold = NB2112_MEDIAN_BAG - PROMOTE_DELTA
    promote = fresh_mean_bag <= promote_threshold
    if promote:
        verdict = (
            f"PROMOTE_REPRODUCIBLE_fresh_mean_bag={fresh_mean_bag:.4f}_"
            f"<={promote_threshold:.4f}"
        )
        print(f"\n   VERDICT: PROMOTE (fresh mean-bag {fresh_mean_bag:.4f} "
              f"<= threshold {promote_threshold:.4f})")
    else:
        verdict = (
            f"REJECT_LUCKY_SEED_fresh_mean_bag={fresh_mean_bag:.4f}_"
            f">{promote_threshold:.4f}"
        )
        print(f"\n   VERDICT: REJECT (fresh mean-bag {fresh_mean_bag:.4f} "
              f"> threshold {promote_threshold:.4f})")

    summary = {
        "tag": TAG,
        "method": "fresh_kf_seeds_reproducibility_K29_AtomPair_bit_1313",
        "best_add_idx_in_117": int(BEST_ADD_IDX),
        "best_add_name": feat_names[BEST_ADD_IDX],
        "best_add_family": feat_family[BEST_ADD_IDX],
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "rae_anchor_chemprop_aux": rae_anchor,
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "resid_folds": RESID_FOLDS,
        "orig_seeds": ORIG_SEEDS,
        "orig_per_seed_rae": orig_per_seed_rae,
        "orig_mean_bag": orig_mean_bag,
        "orig_median_bag": orig_median_bag,
        "fresh_seeds": FRESH_SEEDS,
        "fresh_per_seed_rae": fresh_per_seed_rae,
        "fresh_per_seed_mean": fresh_per_seed_mean,
        "fresh_per_seed_std": fresh_per_seed_std,
        "fresh_per_seed_min": fresh_per_seed_min,
        "fresh_per_seed_max": fresh_per_seed_max,
        "fresh_mean_bag": fresh_mean_bag,
        "fresh_median_bag": fresh_median_bag,
        "nb2112_mean_bag_ref": NB2112_MEAN_BAG,
        "nb2112_median_bag_ref": NB2112_MEDIAN_BAG,
        "promote_threshold": promote_threshold,
        "promote_delta": PROMOTE_DELTA,
        "delta_fresh_mean_vs_nb2112_median": (
            fresh_mean_bag - NB2112_MEDIAN_BAG
        ),
        "delta_fresh_mean_vs_nb2112_mean": (
            fresh_mean_bag - NB2112_MEAN_BAG
        ),
        "promote": bool(promote),
        "verdict": verdict,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / "nb1086_verify_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
