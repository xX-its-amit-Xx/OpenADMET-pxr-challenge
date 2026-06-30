"""nb3084 -- Per-row learned w via Ridge on rank features over {K18, K19}
            deep-30 anchors.

NEW PARADIGM:
    Prior work (nb3030/nb3063/nb3070/nb3073/nb3080) used hard-split or smooth
    sigmoid per-row blends where w_K18(row) was a hand-crafted function of
    rank(K18) (median split, q40 cut, sigmoid steepness sweep). This script
    LEARNS w_K18 per row via Ridge regression on (rank_K18, rank_K19) features
    from fold-train data; per-row proxy "optimal_w" is the value in [0, 1]
    that minimizes |y - (w*K18 + (1-w)*K19)| for that training row.

PROTOCOL (per kf_seed, 5-fold scaffold split, anchors LOADED no rebuild):
    Per outer fold:
        a) Rank K18 and K19 predictions within fold-train; convert to U[0,1]
           quantile features `rank_K18`, `rank_K19`.
        b) Per fold-train row compute proxy optimal_w =
               argmin_{w in [0,1]} |y - (w*K18 + (1-w)*K19)|
           closed form: w_star = clip((y - K19) / (K18 - K19), 0, 1) when
           |K18 - K19| > eps else 0.5 (ambiguous; both anchors equal).
        c) Fit Ridge(alpha=1.0) on fold-train: features (rank_K18, rank_K19),
           target optimal_w. (Two-feature linear model; df=3 = intercept + 2
           slopes.)
        d) For fold-val: rank val K18 and K19 *jointly with* fold-train (use
           empirical CDF from fold-train ranks; out-of-range val rows are
           clipped to [0, 1]). Apply Ridge -> w_hat_val per row. Clip to
           [0, 1]. Blend: pred = w_hat_val * K18_val + (1 - w_hat_val) *
           K19_val.
        e) Stitch into oof_blend (253,); pooled_rae across 5 outer folds.
    Repeat for 15 FRESH kf_seeds {1111..1125}. Report mean +/- std + 95% CI.

GATE (on 15-seed mean):
    mean < 0.4477 -> "BETTER_THAN_NB3070"
        (beats prior wide-seed quantile-conditional verification)
    else        -> "FAIL"
        (paradigm does not improve over hand-crafted hard-split)

References:
    nb3073 5-seed best combo mean   = 0.4470 +/- 0.0009 (UNDER-DISPERSED)
    nb3070 15-seed verify of nb3063 = (pending; gate threshold 0.4477)
    nb3063 5-seed mean              = 0.4477
    nb3030 15-seed wide-mean        = 0.4509
    nb3002 single-kf=1001           = 0.4501
    nb2960 K18 deep-30 OOF          = 0.4536
    nb3000 K19 deep-30 OOF          = 0.4607
    nb2171 prior post-hoc top       = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3084_summary.json
    data/processed/nb3084_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3084.npy         (513,) float32 -- deploy te
    submissions/nb3084_learned_per_row_w.csv  (only on BETTER_THAN_NB3070)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.linear_model import Ridge

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3084"
PARENT_TAG = "nb3070"

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
KF_SEEDS = list(range(1111, 1126))  # 15 FRESH seeds {1111..1125}

# -- Ridge config --------------------------------------------------------------
RIDGE_ALPHA = 1.0
W_EPS = 1e-6  # threshold for |K18 - K19| ambiguity

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4477  # mean < this -> BETTER_THAN_NB3070

# -- References ----------------------------------------------------------------
REF_NB3073 = 0.4470
REF_NB3070_GATE = 0.4477
REF_NB3063 = 0.4477
REF_NB3030 = 0.4509
REF_NB3002 = 0.4501
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _rank_u01(x: np.ndarray) -> np.ndarray:
    """Empirical CDF U[0,1] rank within the input array."""
    n = len(x)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return (ranks + 1.0) / (n + 1.0)  # (0, 1) avoiding 0 and 1


def _rank_against_reference(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Compute U[0,1] rank of x against the sorted reference distribution.

    Equivalent to empirical CDF of ref evaluated at x; clipped to (0, 1).
    """
    ref_sorted = np.sort(ref)
    pos = np.searchsorted(ref_sorted, x, side="right").astype(np.float64)
    n_ref = float(len(ref_sorted))
    return np.clip(pos / (n_ref + 1.0), 1.0 / (n_ref + 1.0), n_ref / (n_ref + 1.0))


def _optimal_w_proxy(y: np.ndarray, k18: np.ndarray, k19: np.ndarray) -> np.ndarray:
    """Per-row proxy optimal w in [0, 1] minimizing |y - (w*K18 + (1-w)*K19)|."""
    denom = k18 - k19
    safe = np.abs(denom) > W_EPS
    out = np.full_like(y, 0.5, dtype=np.float64)
    out[safe] = np.clip((y[safe] - k19[safe]) / denom[safe], 0.0, 1.0)
    return out


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run learned per-row w pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_w_means = []
    fold_w_stds = []
    fold_ridge_coefs = []
    fold_ridge_intercepts = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # -- fold-train features -----------------------------------------
        tr_k18 = P_unb[tr_loc, 0]
        tr_k19 = P_unb[tr_loc, 1]
        tr_y = y_unb[tr_loc]
        rank_tr_k18 = _rank_u01(tr_k18)
        rank_tr_k19 = _rank_u01(tr_k19)
        X_tr = np.column_stack([rank_tr_k18, rank_tr_k19])
        w_tr = _optimal_w_proxy(tr_y, tr_k18, tr_k19)

        # -- fit Ridge ---------------------------------------------------
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_tr, w_tr)
        fold_ridge_coefs.append(ridge.coef_.tolist())
        fold_ridge_intercepts.append(float(ridge.intercept_))

        # -- fold-val features (rank against fold-train CDF) -------------
        va_k18 = P_unb[va_loc, 0]
        va_k19 = P_unb[va_loc, 1]
        rank_va_k18 = _rank_against_reference(va_k18, tr_k18)
        rank_va_k19 = _rank_against_reference(va_k19, tr_k19)
        X_va = np.column_stack([rank_va_k18, rank_va_k19])
        w_hat_va = np.clip(ridge.predict(X_va), 0.0, 1.0)
        val_pred = w_hat_va * va_k18 + (1.0 - w_hat_va) * va_k19
        oof_blend[va_loc] = val_pred

        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_w_means.append(float(w_hat_va.mean()))
        fold_w_stds.append(float(w_hat_va.std(ddof=0)))

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_w_mean_mean": float(np.mean(fold_w_means)),
        "fold_w_std_mean": float(np.mean(fold_w_stds)),
        "fold_ridge_coef_means": [
            float(np.mean([c[i] for c in fold_ridge_coefs])) for i in range(2)
        ],
        "fold_ridge_intercept_mean": float(np.mean(fold_ridge_intercepts)),
        "oof": oof_blend,
    }


def _deploy_ridge_and_te(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    P_te: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit Ridge on FULL 253 unblind set, apply to 513 test rows."""
    unb_k18 = P_unb[:, 0]
    unb_k19 = P_unb[:, 1]
    rank_unb_k18 = _rank_u01(unb_k18)
    rank_unb_k19 = _rank_u01(unb_k19)
    X_unb = np.column_stack([rank_unb_k18, rank_unb_k19])
    w_unb = _optimal_w_proxy(y_unb, unb_k18, unb_k19)
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_unb, w_unb)

    te_k18 = P_te[:, 0]
    te_k19 = P_te[:, 1]
    rank_te_k18 = _rank_against_reference(te_k18, unb_k18)
    rank_te_k19 = _rank_against_reference(te_k19, unb_k19)
    X_te = np.column_stack([rank_te_k18, rank_te_k19])
    w_hat_te = np.clip(ridge.predict(X_te), 0.0, 1.0)
    te_pred = w_hat_te * te_k18 + (1.0 - w_hat_te) * te_k19
    deploy_info = {
        "ridge_coef": [float(c) for c in ridge.coef_],
        "ridge_intercept": float(ridge.intercept_),
        "w_hat_te_mean": float(w_hat_te.mean()),
        "w_hat_te_std": float(w_hat_te.std(ddof=0)),
        "w_hat_te_min": float(w_hat_te.min()),
        "w_hat_te_max": float(w_hat_te.max()),
    }
    return te_pred.astype(np.float64), deploy_info


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LEARNED per-row w via Ridge on rank features over "
          f"{K_LABELS} deep-30")
    print(f"          features = (rank_K18, rank_K19); target = "
          "argmin_w |y - (w*K18 + (1-w)*K19)|")
    print(f"          ridge alpha = {RIDGE_ALPHA}")
    print(f"          kf_seeds = {len(KF_SEEDS)} fresh "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          gate: mean < {GATE_BETTER:.4f} -> BETTER_THAN_NB3070; "
          "else FAIL")
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

    # -- Load K18, K19 deep-30 anchor OOFs + te arrays ------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18, K19 deep-30 OOFs and te arrays")
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

    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    corr = float(np.corrcoef(P_unb.T)[0, 1])
    print(f"   pairwise corr({K_LABELS[0]}, {K_LABELS[1]}) = {corr:.4f}")

    # -- Scaffolds ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Multi-seed sweep -----------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_w_mean_mean": round(res["fold_w_mean_mean"], 4),
            "fold_w_std_mean": round(res["fold_w_std_mean"], 4),
            "fold_ridge_coef_means": [round(c, 4) for c in res["fold_ridge_coef_means"]],
            "fold_ridge_intercept_mean": round(res["fold_ridge_intercept_mean"], 4),
        })
        print(f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
              f"w_mean={res['fold_w_mean_mean']:.3f}  "
              f"w_std={res['fold_w_std_mean']:.3f}  "
              f"wall={time.time()-ts:.2f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    # 95% CI via t-multiplier (n=15, df=14, t~2.145)
    t_mult = 2.145
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   sem     = {sem:.4f}")
    print(f"   95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median  = {median_rae:.4f}")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"\n   ref nb3070 gate              = {REF_NB3070_GATE:.4f}")
    print(f"   delta vs nb3070 gate         = {mean_rae - REF_NB3070_GATE:+.4f}")
    print(f"   ref nb3030 wide-seed ceiling = {REF_NB3030:.4f}")
    print(f"   delta vs nb3030              = {mean_rae - REF_NB3030:+.4f}")

    # -- Deploy: Ridge fit on full 253, applied to 513 ------------------------
    te_pred_raw, deploy_info = _deploy_ridge_and_te(P_unb, y_unb, P_te)
    te_pred = np.clip(te_pred_raw, 3.0, 9.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"\n   deploy ridge coef    = {deploy_info['ridge_coef']}")
    print(f"   deploy ridge intercept = {deploy_info['ridge_intercept']:.4f}")
    print(f"   te w_hat mean={deploy_info['w_hat_te_mean']:.3f}  "
          f"std={deploy_info['w_hat_te_std']:.3f}  "
          f"range=[{deploy_info['w_hat_te_min']:.3f}, "
          f"{deploy_info['w_hat_te_max']:.3f}]")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER_THAN_NB3070"
        ladder_action = (
            f"PROMOTE. nb3084 learned per-row Ridge blend wide-seed 15-mean "
            f"{mean_rae:.4f} beats nb3070 gate {GATE_BETTER:.4f} "
            f"({mean_rae - GATE_BETTER:+.4f}). Learned-w paradigm "
            "extracts orthogonal signal vs hand-crafted hard-split. "
            "Consider as new PRIMARY-1 candidate after wide-seed audit."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REPORT. nb3084 learned per-row Ridge blend wide-seed 15-mean "
            f"{mean_rae:.4f} fails to beat nb3070 gate {GATE_BETTER:.4f} "
            f"({mean_rae - GATE_BETTER:+.4f}). Ridge on (rank_K18, rank_K19) "
            "either reproduces median-split behavior or overfits the proxy "
            "optimal_w target. Closes the learned-per-row-w axis on the "
            "(K18, K19) anchor pool; keep nb3030 / prior PRIMARY-1."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_learned_per_row_w.csv"
    if verdict == "BETTER_THAN_NB3070":
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
        "method": "learned_per_row_w_ridge_on_rank_features_K18_K19_deep30",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "ridge_alpha": RIDGE_ALPHA,
        "w_eps": W_EPS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "ref_nb3073": REF_NB3073,
        "ref_nb3070_gate": REF_NB3070_GATE,
        "ref_nb3063": REF_NB3063,
        "ref_nb3030": REF_NB3030,
        "ref_nb3002": REF_NB3002,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3070_gate": round(mean_rae - REF_NB3070_GATE, 4),
        "delta_vs_nb3030": round(mean_rae - REF_NB3030, 4),
        "deploy_ridge_coef": deploy_info["ridge_coef"],
        "deploy_ridge_intercept": round(deploy_info["ridge_intercept"], 4),
        "deploy_w_hat_te_mean": round(deploy_info["w_hat_te_mean"], 4),
        "deploy_w_hat_te_std": round(deploy_info["w_hat_te_std"], 4),
        "deploy_w_hat_te_min": round(deploy_info["w_hat_te_min"], 4),
        "deploy_w_hat_te_max": round(deploy_info["w_hat_te_max"], 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER_THAN_NB3070" else None,
        "gate_better_than_nb3070": GATE_BETTER,
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
    print(f"   mean_rae ({n_s} seeds) = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3070 gate  = {mean_rae - REF_NB3070_GATE:+.4f}")
    print(f"   delta vs nb3030      = {mean_rae - REF_NB3030:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3070_gate", "delta_vs_nb3030",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
