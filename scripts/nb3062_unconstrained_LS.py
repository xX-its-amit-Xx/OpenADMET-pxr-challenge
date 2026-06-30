"""nb3062 -- Unconstrained least squares on K18/K19/K23 deep-30 anchors.

NEW PARADIGM:
    Relax simplex positivity constraint. Allow negative weights (anti-correlated
    subtraction) but keep sum(w)=1 constraint (Lagrange via normal equations
    on (P - P[:,-1:]) reduction; SLSQP polish with sum=1, no bounds).

    Hypothesis: simplex (w>=0) is overly restrictive when anchors are highly
    correlated and one anchor has a systematic bias that another anchor can
    counter-balance via subtraction.

PROTOCOL:
    3 anchors: K18, K19, K23 (all deep-30, all PRE-unblind, all chemprop_aux +
    residual LGBM on K-feature slices of nb2231 117-col matrix).

    Per fold:
      1. Lagrange closed-form: minimize ||P w - y||^2 s.t. sum(w)=1
         -> w* = A^-1 P^T y + lambda * A^-1 1 where A = P^T P,
            lambda = (1 - 1^T A^-1 P^T y) / (1^T A^-1 1)
      2. SLSQP polish (RAE loss, eq-constraint sum(w)=1, NO bounds)
         multi-start from Lagrange w*.

    15 fresh kf_seeds {1081..1095}.

GATE: mean < 0.4509 -> "BETTER_THAN_NB3030"; else "FAIL".

References:
    nb3030 wide-seed mean (simplex) = 0.4509  <- gate
    nb3020 single-kf=1001 simplex   = 0.4501
    nb3001 wide-seed simplex        = 0.4511
    nb2171 ceiling deep-30          = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy
    data/processed/nb3020_K23_30seed_oof.npy
    data/processed/te_nb3020_K23.npy

Outputs:
    data/processed/nb3062_summary.json
    data/processed/nb3062_pred_oof.npy
    data/processed/te_nb3062.npy
    submissions/nb3062_unconstrained_LS.csv  (only on BETTER_THAN_NB3030)
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
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3062"
PARENT_TAG = "nb3030"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K19", "K23"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K19": DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",
    "K23": DATA_PROCESSED / "nb3020_K23_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K19": DATA_PROCESSED / "te_nb3000_K19.npy",
    "K23": DATA_PROCESSED / "te_nb3020_K23.npy",
}
K_DEPTH = {"K18": "deep30", "K19": "deep30", "K23": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1081, 1096))   # 15 fresh seeds {1081..1095}
N_STARTS_FOLD = 8
N_STARTS_FULL = 12
EXTREME_W_THRESHOLD = 5.0            # |w| > 5 flagged as numerically extreme

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4509                 # mean < this -> BETTER_THAN_NB3030

# -- References ----------------------------------------------------------------
REF_NB3030_WIDE_MEAN = 0.4509
REF_NB3020_SINGLE_KF = 0.4501
REF_NB3001_WIDE_MEAN = 0.4511
REF_NB2171 = 0.4682


def _lagrange_ls_sum1(P: np.ndarray, y: np.ndarray,
                      ridge: float = 1e-8) -> np.ndarray:
    """Closed-form: minimize ||P w - y||^2 s.t. sum(w)=1, w in R^K (no positivity).

    Using Lagrangian L = ||Pw - y||^2 + 2*mu*(sum(w) - 1):
        dL/dw = 2 P^T(Pw - y) + 2*mu*1 = 0
        => w = (P^T P)^-1 (P^T y - mu * 1)
    Impose sum(w)=1:
        mu = (1^T (P^T P)^-1 P^T y - 1) / (1^T (P^T P)^-1 1)
    """
    K = P.shape[1]
    PtP = P.T @ P + ridge * np.eye(K)
    Pty = P.T @ y
    inv = np.linalg.inv(PtP)
    a = inv @ Pty           # (K,)
    b = inv @ np.ones(K)    # (K,)
    mu = (a.sum() - 1.0) / b.sum()
    w = a - mu * b
    # numerical normalization (Lagrange should give sum=1, but float drift)
    w = w / w.sum()
    return w


def _unconstrained_ls_sum1(P: np.ndarray, y: np.ndarray,
                           n_starts: int = 8, seed: int = 0,
                           ) -> tuple[np.ndarray, float]:
    """Minimize RAE(y, P @ w) over {w in R^K : sum(w)=1} (NO positivity).

    Multi-start: warm-start from Lagrange closed-form + random perturbations
    (Gaussian noise around the Lagrange solution + the simplex centroid).
    """
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w: np.ndarray) -> float:
        return float(rae(y, P @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    # No bounds; weights allowed negative or > 1, only sum=1 constraint.

    # Starting points
    starts = []
    try:
        w_lag = _lagrange_ls_sum1(P, y)
        starts.append(w_lag)
    except np.linalg.LinAlgError:
        w_lag = np.full(K, 1.0 / K)
    starts.append(np.full(K, 1.0 / K))               # simplex centroid
    for _ in range(max(0, n_starts - 2)):
        # Gaussian-perturbed Lagrange start, re-normalized to sum=1
        pert = w_lag + rng.normal(scale=0.3, size=K)
        pert = pert - (pert.sum() - 1.0) / K
        starts.append(pert)

    best_w, best_r = None, np.inf
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="SLSQP", constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-10})
            w = res.x.astype(np.float64)
            s = float(w.sum())
            if not np.isfinite(s) or s == 0.0:
                continue
            w = w / s   # enforce sum=1 exactly
            r = float(rae(y, P @ w))
            if r < best_r:
                best_r, best_w = r, w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_r = float(rae(y, P @ best_w))
    return best_w, best_r


def _run_one_seed(P_unb: np.ndarray, y_unb: np.ndarray,
                  unb_scaffolds: list[str], kf_seed: int) -> dict:
    """Run per-fold unconstrained LS pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    K = P_unb.shape[1]
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_w_stack = []
    any_negative = False
    any_extreme = False
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        w, _r_train = _unconstrained_ls_sum1(
            P_unb[tr_loc], y_unb[tr_loc],
            n_starts=N_STARTS_FOLD,
            seed=kf_seed * 11 + fold_i,
        )
        val_pred = P_unb[va_loc] @ w
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_w_stack.append(w)
        if (w < 0).any():
            any_negative = True
        if np.abs(w).max() > EXTREME_W_THRESHOLD:
            any_extreme = True

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    w_mean = np.stack(fold_w_stack, axis=0).mean(axis=0)
    # mean-of-fold weights should sum to ~1; renormalize for storage
    if abs(w_mean.sum()) > 1e-8:
        w_mean = w_mean / w_mean.sum()
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "mean_fold_weights": w_mean.tolist(),
        "any_fold_negative": any_negative,
        "any_fold_extreme": any_extreme,
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Unconstrained LS (sum=1, w can be negative) on "
          f"{K_LABELS} deep-30")
    print(f"          15 fresh kf_seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          gate: mean < {GATE_BETTER} -> BETTER_THAN_NB3030")
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

    # -- Load anchors --------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load 3 deep-30 K-anchor OOFs and te arrays")
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

    P_unb = np.column_stack(oof_cols)  # (253, 3)
    P_te = np.column_stack(te_cols)    # (513, 3)
    K = len(K_LABELS)

    # Pair-wise correlations
    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n  OOF correlation matrix (all deep-30):")
    print(f"        {'  '.join([f'{k:>6s}' for k in K_LABELS])}")
    for i, ki in enumerate(K_LABELS):
        row = "  ".join([f"{corr_mat[i, j]:6.3f}" for j in range(K)])
        print(f"   {ki:>6s}  {row}")

    # -- Scaffolds -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # Lagrange closed-form on full pool for orientation
    w_lag_full = _lagrange_ls_sum1(P_unb, y_unb)
    r_lag_full = float(rae(y_unb, P_unb @ w_lag_full))
    print(f"\n   Lagrange-LS full-pool weights = "
          + ", ".join(f"{K_LABELS[k]}={w_lag_full[k]:+.4f}" for k in range(K)))
    print(f"   Lagrange-LS full-pool RAE     = {r_lag_full:.4f}")

    # -- Wide-seed sweep -----------------------------------------------------
    print("\n" + "-" * 78)
    print(f"WIDE-SEED SWEEP: {len(KF_SEEDS)} fresh kf_seeds")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    w_mean_stack = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        w_mean_stack.append(res["mean_fold_weights"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "mean_fold_weights": {
                K_LABELS[k]: round(float(res["mean_fold_weights"][k]), 4)
                for k in range(K)
            },
            "any_fold_negative": res["any_fold_negative"],
            "any_fold_extreme": res["any_fold_extreme"],
        })
        wstr = ", ".join(f"{K_LABELS[k]}={res['mean_fold_weights'][k]:+.3f}"
                         for k in range(K))
        print(f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
              f"neg={res['any_fold_negative']}  "
              f"extreme={res['any_fold_extreme']}  "
              f"w=[{wstr}]  wall={time.time()-ts:.2f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1))
    sem = std_rae / np.sqrt(len(arr))
    t_mult = 2.145
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))

    print("\n" + "-" * 78)
    print("AGGREGATE (15 fresh seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   sem     = {sem:.4f}")
    print(f"   95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median  = {median_rae:.4f}")
    print(f"   5/95p   = [{p5:.4f}, {p95:.4f}]")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")

    print(f"\n   nb3030 wide-seed ref (simplex) = {REF_NB3030_WIDE_MEAN:.4f}")
    print(f"   delta vs nb3030                = {mean_rae - REF_NB3030_WIDE_MEAN:+.4f}")
    print(f"   nb3020 single-kf=1001 simplex  = {REF_NB3020_SINGLE_KF:.4f}")

    # Mean-of-seed mean-of-fold weights
    w_seed_mean = np.asarray(w_mean_stack).mean(axis=0)
    if abs(w_seed_mean.sum()) > 1e-8:
        w_seed_mean = w_seed_mean / w_seed_mean.sum()
    print(f"\n   mean-of-seed mean-of-fold weights = "
          + ", ".join(f"{K_LABELS[k]}={w_seed_mean[k]:+.4f}" for k in range(K)))

    # -- Deploy: full-pool unconstrained LS ---------------------------------
    w_full, r_full = _unconstrained_ls_sum1(
        P_unb, y_unb, n_starts=N_STARTS_FULL, seed=0,
    )
    full_pool_weights = {K_LABELS[k]: round(float(w_full[k]), 4) for k in range(K)}
    full_pool_negative = bool((w_full < 0).any())
    full_pool_extreme = bool(np.abs(w_full).max() > EXTREME_W_THRESHOLD)
    te_pred = (P_te @ w_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"\n   full-pool unconstrained LS weights = "
          + ", ".join(f"{K_LABELS[k]}={w_full[k]:+.4f}" for k in range(K)))
    print(f"   full-pool in-sample RAE            = {r_full:.4f}")
    print(f"   te[unb] in-sample RAE              = {te_unb_in_rae:.4f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[len(arr) // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER_THAN_NB3030"
        ladder_action = (
            f"PROMOTE candidate. Unconstrained-LS wide-seed mean {mean_rae:.4f} "
            f"beats nb3030 simplex ceiling {REF_NB3030_WIDE_MEAN:.4f} by "
            f"{mean_rae - REF_NB3030_WIDE_MEAN:+.4f}. Negative-weight paradigm "
            "validated; deploy as new PRIMARY-1 candidate."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. Unconstrained-LS wide-seed mean {mean_rae:.4f} >= "
            f"nb3030 simplex ceiling {REF_NB3030_WIDE_MEAN:.4f}. "
            "Negative-weight relaxation did not improve over simplex; keep "
            "prior PRIMARY-1."
        )

    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_unconstrained_LS.csv"
    if verdict == "BETTER_THAN_NB3030":
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
        "method": "unconstrained_LS_sum1_negative_allowed_K18_K19_K23_deep30",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": K_LABELS,
        "lagrange_full_pool_weights": {
            K_LABELS[k]: round(float(w_lag_full[k]), 4) for k in range(K)
        },
        "lagrange_full_pool_rae": round(r_lag_full, 4),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_starts_fold": N_STARTS_FOLD,
        "n_starts_full": N_STARTS_FULL,
        "extreme_w_threshold": EXTREME_W_THRESHOLD,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "p5_rae": round(p5, 4),
        "p95_rae": round(p95, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "ref_nb3030_wide_mean": REF_NB3030_WIDE_MEAN,
        "ref_nb3020_single_kf": REF_NB3020_SINGLE_KF,
        "ref_nb3001_wide_mean": REF_NB3001_WIDE_MEAN,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3030": round(mean_rae - REF_NB3030_WIDE_MEAN, 4),
        "mean_of_seed_mean_fold_weights": {
            K_LABELS[k]: round(float(w_seed_mean[k]), 4) for k in range(K)
        },
        "full_pool_weights": full_pool_weights,
        "full_pool_rae_in_sample": round(float(r_full), 4),
        "full_pool_negative": full_pool_negative,
        "full_pool_extreme": full_pool_extreme,
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER_THAN_NB3030" else None,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (15 seeds)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3030       = {mean_rae - REF_NB3030_WIDE_MEAN:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   ladder action         = {ladder_action}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3030",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  full_pool_weights: {res.get('full_pool_weights')}")
    print(f"  lagrange_full_pool_weights: {res.get('lagrange_full_pool_weights')}")
