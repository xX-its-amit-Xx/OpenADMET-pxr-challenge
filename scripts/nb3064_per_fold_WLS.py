"""nb3064 -- Per-fold weighted (ordinary) least squares on (K18, K19) with intercept.

NEW PARADIGM:
    Drop the SLSQP simplex constraint (w>=0, sum w=1) and instead fit an
    UNCONSTRAINED 2-feature linear regression of y on (K18_pred, K19_pred)
    with intercept per fold. This allows:
        - negative coefficients (de-emphasize anchors)
        - sum != 1 (variance rescale absorbed into coefficients)
        - intercept (bias absorbed; not constrained to mean(y))

    Rationale: the simplex paradigm closed (nb3022 stretch s=1.000,
    nb3021 isotonic, nb2200 stretch on deep30 all REJECTED). A 2-feature
    OLS has 3 free parameters per fold (b0, b1, b2) vs simplex's 1 free
    parameter (w in [0,1]), so it has STRICTLY MORE capacity. The question
    is whether n=~200 per fold-train can support 3 params without overfit.

PROTOCOL:
    Anchors:
        K18: nb2960 deep-30 fresh-seed OOFs + te arrays
        K19: nb3000 deep-30 fresh-seed OOFs + te arrays
    Outer CV: 5-fold scaffold split, 15 fresh kf_seeds {1081..1095}
    Per fold:
        - sklearn LinearRegression(fit_intercept=True) on fold-train (K18,K19)->y
        - Apply to fold-val
    Per seed: pooled RAE on the 5 outer-val folds (covering all 253)
    Reported gate metric = MEAN pooled RAE across the 15 seeds.

    Deploy:
        - Refit OLS on FULL 253 -> single (b0, b1, b2)
        - Apply to (513, 2) stacked te arrays -> te_nb3064

GATE:
    mean_pooled_rae < 0.4509  -> "BETTER_THAN_NB3030"
    else                      -> "FAIL"

References:
    nb3030 wide-seed verify nb3020 K23-only deep30      = 0.4509  (current bar)
    nb3002 per-fold simplex K18,K19 deep30 (15 seeds)   ~ ladder PRIMARY-1 band
    nb3022 per-fold stretch on nb3002 (REJECTED)        = 0.4682
    nb2171 prior post-hoc-blend ceiling                 = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3064_summary.json
    data/processed/nb3064_pred_oof.npy  (253,) float32 -- per-fold OLS OOF
                                         (taken from the first seed = 1081)
    data/processed/te_nb3064.npy        (513,) float32 -- deploy te
    submissions/nb3064_per_fold_wls_K18_K19.csv  (only if verdict != "FAIL")
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
from rdkit import RDLogger
from sklearn.linear_model import LinearRegression

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3064"
PARENT_TAG = "nb2960+nb3000"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K19"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K19": DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K19": DATA_PROCESSED / "te_nb3000_K19.npy",
}
K_DEPTH = {"K18": "deep30", "K19": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1081, 1096))  # 15 fresh seeds {1081..1095}

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB3030 = 0.4509

# -- References ----------------------------------------------------------------
REF_NB3030 = 0.4509
REF_NB2171 = 0.4682


def _ols_fit_predict(P_tr: np.ndarray, y_tr: np.ndarray,
                     P_va: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Fit LinearRegression(fit_intercept=True) on P_tr -> y_tr, predict P_va.

    Returns (val_pred, coefs, intercept, train_rae, train_R2)
    where coefs = (b1, b2) for (K18, K19).
    """
    model = LinearRegression(fit_intercept=True)
    model.fit(P_tr, y_tr)
    b0 = float(model.intercept_)
    b = model.coef_.astype(np.float64)
    train_pred = model.predict(P_tr)
    train_rae = float(rae(y_tr, train_pred))
    # train R^2
    ss_res = float(np.sum((y_tr - train_pred) ** 2))
    ss_tot = float(np.sum((y_tr - y_tr.mean()) ** 2))
    train_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    val_pred = model.predict(P_va)
    return val_pred, b, b0, train_rae, train_r2


def _run_one_seed(kf_seed: int, P_unb: np.ndarray, y_unb: np.ndarray,
                  unb_scaffolds: list[str]) -> tuple[float, list[dict], np.ndarray, list[dict]]:
    """Per-fold OLS with one kf_seed. Returns (pooled_rae, fold_records, oof_blend, fold_coefs)."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    fold_coefs = []
    K = P_unb.shape[1]
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        val_pred, b, b0, train_rae, train_r2 = _ols_fit_predict(
            P_unb[tr_loc], y_unb[tr_loc], P_unb[va_loc],
        )
        oof_blend[va_loc] = val_pred
        r_val = float(rae(y_unb[va_loc], val_pred))
        coef_dict = {"intercept": round(b0, 4),
                     **{K_LABELS[k]: round(float(b[k]), 4) for k in range(K)}}
        fold_coefs.append(coef_dict)
        fold_records.append({
            "fold": int(fold_i),
            "n_train": int(len(tr_loc)),
            "n_val": int(len(va_loc)),
            "intercept": round(b0, 4),
            "coefs": {K_LABELS[k]: round(float(b[k]), 4) for k in range(K)},
            "sum_coefs": round(float(b.sum()), 4),
            "train_rae": round(train_rae, 4),
            "train_R2": round(train_r2, 4),
            "val_rae": round(r_val, 4),
        })
    if np.isnan(oof_blend).any():
        raise RuntimeError(f"scaffold splits did not cover all 253 rows (kf_seed={kf_seed})")
    pooled_rae = float(rae(y_unb, oof_blend))
    return pooled_rae, fold_records, oof_blend, fold_coefs


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold OLS LinearRegression (intercept+2 coefs) on {K_LABELS}")
    print(f"          parents: nb2960 (K18 deep-30) + nb3000 (K19 deep-30)")
    print(f"          paradigm: drop simplex constraint, use unconstrained OLS with intercept")
    print(f"          outer CV: {N_FOLDS}-fold scaffold, {len(KF_SEEDS)} fresh seeds {KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"          gate: <{GATE_BETTER_THAN_NB3030} BETTER_THAN_NB3030 else FAIL")
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load K-anchor OOFs + te arrays --------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K-anchor OOFs and te arrays (both deep-30)")
    print("-" * 78)
    oof_cols, te_cols = [], []
    per_K_full_rae = {}
    for k in K_LABELS:
        oof = np.load(OOF_PATHS[k]).astype(np.float64)
        te_arr = np.load(TE_PATHS[k]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{k} OOF shape {oof.shape} != ({n_unb},)")
        if te_arr.shape != (n_test,):
            raise ValueError(f"{k} te shape {te_arr.shape} != ({n_test},)")
        oof_cols.append(oof)
        te_cols.append(te_arr)
        r = float(rae(y_unb, oof))
        per_K_full_rae[k] = round(r, 4)
        print(f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

    K = len(K_LABELS)
    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # Pair-wise correlation
    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n  OOF correlation matrix:")
    print(f"        {'  '.join([f'{k:>6s}' for k in K_LABELS])}")
    for i, ki in enumerate(K_LABELS):
        row = "  ".join([f"{corr_mat[i, j]:6.3f}" for j in range(K)])
        print(f"   {ki:>6s}  {row}")

    # -- Build scaffolds ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique scaffolds = {n_unique_scaf}")

    # -- Per-seed per-fold OLS (15 seeds) ------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: outer CV with per-fold OLS LinearRegression, 15 fresh kf_seeds")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_fold_records = {}
    per_seed_fold_coefs = {}
    first_seed_oof_blend = None
    for seed in KF_SEEDS:
        p_rae, fold_recs, oof_blend, fold_cs = _run_one_seed(
            seed, P_unb, y_unb, unb_scaffolds,
        )
        per_seed_pooled.append(p_rae)
        per_seed_fold_records[str(seed)] = fold_recs
        per_seed_fold_coefs[str(seed)] = fold_cs
        if first_seed_oof_blend is None:
            first_seed_oof_blend = oof_blend
        mean_val = float(np.mean([r["val_rae"] for r in fold_recs]))
        print(f"   seed={seed}  pooled={p_rae:.4f}  per-fold mean={mean_val:.4f}")

    arr_pooled = np.asarray(per_seed_pooled)
    mean_pooled = float(arr_pooled.mean())
    std_pooled = float(arr_pooled.std(ddof=1))
    min_pooled = float(arr_pooled.min())
    max_pooled = float(arr_pooled.max())
    print(f"\n   POOLED-OUTER-VAL RAE over {len(KF_SEEDS)} seeds:")
    print(f"     mean = {mean_pooled:.4f}")
    print(f"     std  = {std_pooled:.4f}")
    print(f"     min  = {min_pooled:.4f}")
    print(f"     max  = {max_pooled:.4f}")

    # -- Aggregate per-fold coefficients across folds and seeds --------------
    all_intercepts, all_b_K18, all_b_K19 = [], [], []
    for seed_key, clist in per_seed_fold_coefs.items():
        for c in clist:
            all_intercepts.append(c["intercept"])
            all_b_K18.append(c["K18"])
            all_b_K19.append(c["K19"])
    mean_b0 = float(np.mean(all_intercepts))
    mean_b_K18 = float(np.mean(all_b_K18))
    mean_b_K19 = float(np.mean(all_b_K19))
    sd_b0 = float(np.std(all_intercepts, ddof=1))
    sd_b_K18 = float(np.std(all_b_K18, ddof=1))
    sd_b_K19 = float(np.std(all_b_K19, ddof=1))
    print(f"\n   mean +/- sd coefficients across {len(all_intercepts)} (seed,fold) cells:")
    print(f"     intercept = {mean_b0:+.4f} +/- {sd_b0:.4f}")
    print(f"     b[K18]    = {mean_b_K18:+.4f} +/- {sd_b_K18:.4f}")
    print(f"     b[K19]    = {mean_b_K19:+.4f} +/- {sd_b_K19:.4f}")
    print(f"     sum b     = {mean_b_K18 + mean_b_K19:+.4f}")

    # -- Deploy: single-pool OLS on FULL 253 ---------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: deploy OLS LinearRegression on FULL 253")
    print("-" * 78)
    full_model = LinearRegression(fit_intercept=True)
    full_model.fit(P_unb, y_unb)
    full_b0 = float(full_model.intercept_)
    full_b = full_model.coef_.astype(np.float64)
    full_pred_unb = full_model.predict(P_unb)
    r_full = float(rae(y_unb, full_pred_unb))
    ss_res = float(np.sum((y_unb - full_pred_unb) ** 2))
    ss_tot = float(np.sum((y_unb - y_unb.mean()) ** 2))
    full_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    print(f"   in-sample RAE = {r_full:.4f}  R^2 = {full_r2:.4f}")
    print(f"     intercept = {full_b0:+.4f}")
    for k in range(K):
        print(f"     b[{K_LABELS[k]:>4s}]   = {full_b[k]:+.4f}")
    print(f"     sum b     = {full_b.sum():+.4f}")

    te_pred = (full_model.predict(P_te)).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te(full-pool) mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"in-sample unb RAE={te_unb_in_rae:.4f}")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: GATE on mean pooled outer-val RAE across 15 seeds")
    print("-" * 78)
    if mean_pooled < GATE_BETTER_THAN_NB3030:
        verdict = "BETTER_THAN_NB3030"
    else:
        verdict = "FAIL"
    delta_vs_nb3030 = mean_pooled - REF_NB3030
    delta_vs_nb2171 = mean_pooled - REF_NB2171
    print(f"   mean_pooled_rae          = {mean_pooled:.4f} (std {std_pooled:.4f})")
    print(f"   delta vs nb3030 (0.4509) = {delta_vs_nb3030:+.4f}")
    print(f"   delta vs nb2171 (0.4682) = {delta_vs_nb2171:+.4f}")
    print(f"   verdict                  = {verdict}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, first_seed_oof_blend.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}  (single-seed OOF, kf_seed={KF_SEEDS[0]})")
    print(f"   [save] {te_path}   (deploy from FULL-253 OLS)")

    sub_csv = SUBMISSIONS / f"{TAG}_per_fold_wls_K18_K19.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "per_fold_OLS_LinearRegression_K18_K19_intercept_15seed",
        "paradigm": "unconstrained_2feature_OLS_with_intercept",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": K_LABELS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "per_seed_pooled_rae": [round(r, 5) for r in per_seed_pooled],
        "per_seed_fold_records": per_seed_fold_records,
        "per_seed_fold_coefs": per_seed_fold_coefs,
        "pooled_rae_mean": round(mean_pooled, 5),
        "pooled_rae_std": round(std_pooled, 5),
        "pooled_rae_min": round(min_pooled, 5),
        "pooled_rae_max": round(max_pooled, 5),
        "mean_coefs_across_all_cells": {
            "intercept": round(mean_b0, 4),
            "K18": round(mean_b_K18, 4),
            "K19": round(mean_b_K19, 4),
        },
        "sd_coefs_across_all_cells": {
            "intercept": round(sd_b0, 4),
            "K18": round(sd_b_K18, 4),
            "K19": round(sd_b_K19, 4),
        },
        "full_pool_ols": {
            "intercept": round(full_b0, 4),
            "K18": round(float(full_b[0]), 4),
            "K19": round(float(full_b[1]), 4),
            "sum_coefs": round(float(full_b.sum()), 4),
            "rae_in_sample": round(float(r_full), 4),
            "R2_in_sample": round(full_r2, 4),
        },
        "te_unb_in_sample_rae_full_pool": round(te_unb_in_rae, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "mean_rae": mean_pooled,
        "ref_nb3030": REF_NB3030,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3030": delta_vs_nb3030,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_better_than_nb3030": GATE_BETTER_THAN_NB3030,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-K full-OOF RAE       = "
          + ", ".join([f"{k}={v:.4f}" for k, v in per_K_full_rae.items()]))
    print(f"   pooled outer-val RAE     = {mean_pooled:.4f} +/- {std_pooled:.4f} "
          f"(15 seeds)")
    print(f"   min/max pooled RAE       = {min_pooled:.4f} / {max_pooled:.4f}")
    print(f"   mean-cell coefs          = "
          f"b0={mean_b0:+.3f}, K18={mean_b_K18:+.3f}, K19={mean_b_K19:+.3f}")
    print(f"   full-pool coefs          = "
          f"b0={full_b0:+.3f}, K18={full_b[0]:+.3f}, K19={full_b[1]:+.3f}")
    print(f"   te[unb_idx] in-sample    = {te_unb_in_rae:.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean",
        "pooled_rae_std",
        "pooled_rae_min",
        "pooled_rae_max",
        "mean_coefs_across_all_cells",
        "full_pool_ols",
        "te_unb_in_sample_rae_full_pool",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
