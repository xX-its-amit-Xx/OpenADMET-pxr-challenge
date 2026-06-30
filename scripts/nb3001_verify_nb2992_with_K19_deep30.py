"""nb3001 -- 15-seed wide-seed verification of nb2992 per-fold SLSQP simplex
            {K18, K19, K20} using K=19 DEEP-30 (nb3000) instead of 5-seed (nb2631).

CONTEXT (cycle 246+):
    nb2992 reported pooled_outer_val_RAE = 0.4479 on single kf_seed=1001
    using {K18 deep-30, K19 5-seed, K20 deep-30} (BETTER_THAN_NB2982 0.4505).
    Per cycle-160 deep-30 rule and cycle-246 finding that deep-30 is
    LOAD-BEARING for K-pyramid OOFs, K=19 5-seed was an honesty leak.

    nb3000 rebuilt K=19 with 30 fresh seeds {3001..3030}, bag-mean
    RAE = 0.4607 (-0.0018 vs nb2631 5-seed 0.4625). Now we must
    re-verify nb2992 using the deep-30 K=19 anchor under 15 FRESH
    kf_seeds {1036..1050} (excludes nb2982 single-fit 1001 and
    nb2990 wide-seed 1021-1035) before any promotion decision.

PROTOCOL:
    Anchors (all deep-30):
        K18 = nb2960_K18_30seed_oof.npy + nb2960_K18_30seed_te.npy
        K19 = nb3000_K19_30seed_oof.npy + te_nb3000_K19.npy        (NEW deep-30)
        K20 = nb2960_K20_30seed_oof.npy + nb2960_K20_30seed_te.npy
    Outer CV: 5-fold scaffold split on 253 unblind, kf_seeds {1036..1050}
    Per fold:
        - SLSQP minimize fold-train RAE on simplex (w >= 0, sum w = 1)
        - 8 multi-starts (uniform + 7 Dirichlet draws)
        - Apply per-fold weights to held-out fold-val slice
    Pooled outer-val RAE per kf_seed -> mean +/- std across 15 fresh seeds.

    Deploy:
        - Refit SLSQP on FULL 253 -> single weight vector
        - Apply to (513, 3) stacked te arrays -> te_nb3001

GATE (on 15-seed mean):
    mean < 0.4518  -> "VERIFIED_NEW_PRIMARY1"
                       (beats current PRIMARY-1 nb2982 by margin; replace it)
    mean < 0.4570  -> "VERIFIED_MARGINAL"
                       (inside PROMOTE gate but not decisively better)
    shift > +0.005 vs single-kf -> "LUCKY_SEED_TRAP"
                       (nb2992 single-kf=1001 0.4479 was a fortunate seed)
    else            -> "FAIL_OR_REPORT"

References:
    nb2992 single-kf=1001 (K19 5-seed)   = 0.4479 (BETTER_THAN_NB2982)
    nb2982 single-kf=1001 (K18+K20)      = 0.4505 (current PRIMARY-1)
    nb2980 wide-seed K18+K20             = 0.4535 +/- 0.0018
    nb2960 K18 deep-30                   = 0.4536
    nb3000 K19 deep-30                   = 0.4607 (NEW)
    nb2960 K20 deep-30                   = 0.4625
    nb2171 prior ceiling                 = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy
    data/processed/nb2960_K20_30seed_oof.npy
    data/processed/nb2960_K20_30seed_te.npy

Outputs:
    data/processed/nb3001_summary.json
    data/processed/nb3001_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3001.npy         (513,) float32 -- full-pool deploy te
    submissions/nb3001_wide_seed_verify_nb2992_K19_deep30.csv  (only if VERIFIED_*)
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

TAG = "nb3001"
PARENT_TAG = "nb2992"

# -- Inputs (ALL deep-30) ------------------------------------------------------
K_LABELS = ["K18", "K19", "K20"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K19": DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",   # deep-30 from nb3000
    "K20": DATA_PROCESSED / "nb2960_K20_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K19": DATA_PROCESSED / "te_nb3000_K19.npy",           # deep-30 from nb3000
    "K20": DATA_PROCESSED / "nb2960_K20_30seed_te.npy",
}
K_DEPTH = {"K18": "deep30", "K19": "deep30", "K20": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1036, 1051))   # 15 fresh seeds {1036..1050}
                                     #   NOT 1001 (nb2982/nb2992 single-fit)
                                     #   NOT 1006-1020 (nb2980 verify nb2973)
                                     #   NOT 1021-1035 (nb2990 verify nb2982)
N_STARTS_FOLD = 8
N_STARTS_FULL = 12
DEGEN_MAX_W = 0.85

# -- Gates ---------------------------------------------------------------------
GATE_NEW_PRIMARY1 = 0.4518            # mean < this -> VERIFIED_NEW_PRIMARY1
GATE_MARGINAL = 0.4570                # mean < this -> VERIFIED_MARGINAL
LUCKY_SHIFT_THRESHOLD = 0.005         # shift > this -> LUCKY_SEED_TRAP

# -- References ----------------------------------------------------------------
REF_NB2992_SINGLE_KF = 0.4479         # nb2992 single-kf=1001 (K19 5-seed)
REF_NB2982_SINGLE_KF = 0.4505         # nb2982 single-kf=1001 (current PRIMARY-1)
REF_NB2980_WIDE_MEAN = 0.4535         # nb2980 wide-seed K18+K20
REF_NB2980_WIDE_STD = 0.0018
REF_K18_DEEP30 = 0.4536
REF_K19_DEEP30 = 0.4607               # NEW from nb3000
REF_K19_5SEED = 0.4625
REF_K20_DEEP30 = 0.4625
REF_NB2171 = 0.4682


def _simplex_slsqp(P: np.ndarray, y: np.ndarray, n_starts: int = 8,
                   seed: int = 0) -> tuple[np.ndarray, float]:
    """Minimize RAE(y, P @ w) over the simplex (w>=0, sum w=1) with multi-start."""
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w: np.ndarray) -> float:
        return float(rae(y, P @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K

    starts = [np.full(K, 1.0 / K)]
    for _ in range(max(0, n_starts - 1)):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w, best_r = None, np.inf
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="SLSQP", bounds=bnds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-9})
            w = np.clip(res.x, 0.0, 1.0)
            s = float(w.sum())
            if s <= 0.0:
                continue
            w = w / s
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
    """Run nb2992 per-fold SLSQP simplex pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    K = P_unb.shape[1]
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_w_stack = []
    any_degen = False
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        w, _r_train = _simplex_slsqp(
            P_unb[tr_loc], y_unb[tr_loc],
            n_starts=N_STARTS_FOLD,
            seed=kf_seed * 11 + fold_i,
        )
        val_pred = P_unb[va_loc] @ w
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_w_stack.append(w)
        if w.max() > DEGEN_MAX_W:
            any_degen = True

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    w_mean = np.stack(fold_w_stack, axis=0).mean(axis=0)
    w_mean = w_mean / w_mean.sum()
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "mean_fold_weights": w_mean.tolist(),
        "any_fold_degenerate": any_degen,
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 15-seed wide-seed verify of {PARENT_TAG} per-fold SLSQP "
          f"{K_LABELS}")
    print(f"          K19 ANCHOR UPGRADED: 5-seed (nb2631) -> deep-30 (nb3000)")
    print(f"          kf_seeds = {KF_SEEDS}")
    print(f"          (fresh, EXCLUDES 1001 nb2982/nb2992 single, 1006-1020 nb2980, "
          f"1021-1035 nb2990)")
    print(f"          gates: <{GATE_NEW_PRIMARY1} VERIFIED_NEW_PRIMARY1 / "
          f"<{GATE_MARGINAL} VERIFIED_MARGINAL / "
          f">+{LUCKY_SHIFT_THRESHOLD:.3f} LUCKY_SEED_TRAP")
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

    # -- Load deep-30 K-anchor OOFs + te arrays -------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load 3 K-anchor deep-30 OOFs + te arrays")
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
    P_unb = np.column_stack(oof_cols)  # (253, 3)
    P_te = np.column_stack(te_cols)    # (513, 3)

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # Pair-wise correlations
    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n  OOF correlation matrix:")
    print(f"        {'  '.join([f'{k:>6s}' for k in K_LABELS])}")
    for i, ki in enumerate(K_LABELS):
        row = "  ".join([f"{corr_mat[i, j]:6.3f}" for j in range(K)])
        print(f"   {ki:>6s}  {row}")

    # -- Scaffolds (kf_seed-independent) -------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique scaffolds = {n_unique_scaf}")

    # -- Wide-seed sweep ------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: WIDE-SEED SWEEP -- {len(KF_SEEDS)} fresh kf_seeds "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
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
            "any_fold_degenerate": res["any_fold_degenerate"],
        })
        print(f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
              f"degen={res['any_fold_degenerate']}  "
              f"w=[K18={res['mean_fold_weights'][0]:.3f}, "
              f"K19={res['mean_fold_weights'][1]:.3f}, "
              f"K20={res['mean_fold_weights'][2]:.3f}]  "
              f"wall={time.time()-ts:.1f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1))
    sem = std_rae / np.sqrt(len(arr))
    # 95% CI via t-multiplier (n=15, df=14, t~2.145)
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

    shift_vs_single = mean_rae - REF_NB2992_SINGLE_KF
    print(f"\n   nb2992 single-kf=1001    = {REF_NB2992_SINGLE_KF:.4f} (K19 5-seed)")
    print(f"   shift (mean - single)    = {shift_vs_single:+.4f}")
    print(f"   nb2982 single-kf=1001    = {REF_NB2982_SINGLE_KF:.4f}")
    print(f"   nb2980 wide-seed K18+K20 = {REF_NB2980_WIDE_MEAN:.4f} "
          f"+/- {REF_NB2980_WIDE_STD:.4f}")
    print(f"   K19 deep-30 (nb3000)     = {REF_K19_DEEP30:.4f}")
    print(f"   K19 5-seed (nb2631)      = {REF_K19_5SEED:.4f}")

    # Mean-of-seed mean-of-fold weights (deploy proxy)
    w_seed_mean = np.asarray(w_mean_stack).mean(axis=0)
    w_seed_mean = w_seed_mean / w_seed_mean.sum()
    print(f"\n   mean-of-seed mean-of-fold weights = "
          + ", ".join(f"{K_LABELS[k]}={w_seed_mean[k]:.4f}" for k in range(K)))

    # -- Deploy: SLSQP on FULL 253 (kf-seed-independent) ----------------------
    print("\n" + "-" * 78)
    print("STEP 4: deploy SLSQP on FULL 253")
    print("-" * 78)
    w_full, r_full = _simplex_slsqp(P_unb, y_unb, n_starts=N_STARTS_FULL, seed=0)
    full_pool_weights = {K_LABELS[k]: round(float(w_full[k]), 4) for k in range(K)}
    full_pool_degen = bool(w_full.max() > DEGEN_MAX_W)
    print(f"   in-sample RAE = {r_full:.4f}  max_w={w_full.max():.4f}  "
          f"degen={full_pool_degen}")
    for k in range(K):
        flag = " (zeroed)" if w_full[k] < 1e-6 else ""
        print(f"     w[{K_LABELS[k]:6s}] = {w_full[k]:+.4f}{flag}")

    te_pred = (P_te @ w_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"te[unb] in-sample RAE={te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[len(arr) // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: GATE")
    print("-" * 78)
    if shift_vs_single > LUCKY_SHIFT_THRESHOLD:
        verdict = "LUCKY_SEED_TRAP"
        ladder_action = (
            f"REJECT nb2992 promotion. nb2992 single-kf=1001 {REF_NB2992_SINGLE_KF:.4f} "
            f"was a fortunate seed; wide-seed mean shift {shift_vs_single:+.4f} "
            f"exceeds +{LUCKY_SHIFT_THRESHOLD:.3f} tolerance. "
            "Keep nb2982 as PRIMARY-1 (0.4505 single-kf, 0.4535 wide-seed verified)."
        )
    elif mean_rae < GATE_NEW_PRIMARY1:
        verdict = "VERIFIED_NEW_PRIMARY1"
        ladder_action = (
            f"PROMOTE nb3001 to PRIMARY-1 (wide-seed {mean_rae:.4f} beats "
            f"nb2982 single-kf {REF_NB2982_SINGLE_KF:.4f} and wide-seed gate "
            f"{GATE_NEW_PRIMARY1:.4f}). Demote nb2982 to PRIMARY-2."
        )
    elif mean_rae < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
        ladder_action = (
            f"KEEP nb2982 as PRIMARY-1. nb3001 wide-seed {mean_rae:.4f} is "
            f"inside PROMOTE gate {GATE_MARGINAL:.4f} but not strictly better "
            f"than current PRIMARY-1 {REF_NB2982_SINGLE_KF:.4f}; tag nb3001 as alternate."
        )
    else:
        verdict = "FAIL_OR_REPORT"
        ladder_action = (
            f"REJECT nb2992 promotion. Wide-seed mean {mean_rae:.4f} above "
            f"PROMOTE gate {GATE_MARGINAL:.4f}. Keep nb2982 as PRIMARY-1."
        )

    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_wide_seed_verify_{PARENT_TAG}_K19_deep30.csv"
    promote_verdicts = {"VERIFIED_NEW_PRIMARY1", "VERIFIED_MARGINAL"}
    if verdict in promote_verdicts:
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
        "method": "wide_seed_15_verify_per_fold_slsqp_K18_K19_K20_with_K19_deep30",
        "paradigm": "verify_nb2992_with_K19_deep30_anchor_upgrade",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "K19_upgrade": "5seed_nb2631 -> deep30_nb3000",
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": K_LABELS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_starts_fold": N_STARTS_FOLD,
        "n_starts_full": N_STARTS_FULL,
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
        "ref_nb2992_single_kf": REF_NB2992_SINGLE_KF,
        "ref_nb2982_single_kf": REF_NB2982_SINGLE_KF,
        "ref_nb2980_wide_mean": REF_NB2980_WIDE_MEAN,
        "ref_nb2980_wide_std": REF_NB2980_WIDE_STD,
        "ref_K18_deep30": REF_K18_DEEP30,
        "ref_K19_deep30": REF_K19_DEEP30,
        "ref_K19_5seed": REF_K19_5SEED,
        "ref_K20_deep30": REF_K20_DEEP30,
        "ref_nb2171": REF_NB2171,
        "shift_mean_vs_single_kf": round(shift_vs_single, 4),
        "delta_vs_nb2982_single": round(mean_rae - REF_NB2982_SINGLE_KF, 4),
        "delta_vs_nb2980_wide": round(mean_rae - REF_NB2980_WIDE_MEAN, 4),
        "mean_of_seed_mean_fold_weights": {
            K_LABELS[k]: round(float(w_seed_mean[k]), 4) for k in range(K)
        },
        "full_pool_slsqp": {
            "weights": full_pool_weights,
            "rae_in_sample": round(float(r_full), 4),
            "max_w": round(float(w_full.max()), 4),
            "degenerate": full_pool_degen,
        },
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict in promote_verdicts else None,
        "gate_new_primary1": GATE_NEW_PRIMARY1,
        "gate_marginal": GATE_MARGINAL,
        "lucky_shift_threshold": LUCKY_SHIFT_THRESHOLD,
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
    print(f"   per-K full-OOF RAE      = "
          + ", ".join([f"{k}={v:.4f}" for k, v in per_K_full_rae.items()]))
    print(f"   mean_rae (15 seeds)     = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   shift vs nb2992 kf=1001 = {shift_vs_single:+.4f}")
    print(f"   delta vs nb2982 single  = {mean_rae - REF_NB2982_SINGLE_KF:+.4f}")
    print(f"   delta vs nb2980 wide    = {mean_rae - REF_NB2980_WIDE_MEAN:+.4f}")
    print(f"   full-pool weights       = {full_pool_weights}")
    print(f"   te[unb] in-sample       = {te_unb_in_rae:.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   ladder action           = {ladder_action}")
    print(f"   wall                    = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "shift_mean_vs_single_kf", "delta_vs_nb2982_single",
        "delta_vs_nb2980_wide", "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  full_pool_slsqp: {res.get('full_pool_slsqp')}")
    print(f"  mean_of_seed_mean_fold_weights: "
          f"{res.get('mean_of_seed_mean_fold_weights')}")
