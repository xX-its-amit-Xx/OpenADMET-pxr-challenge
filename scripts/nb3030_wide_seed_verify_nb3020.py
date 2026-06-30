"""nb3030 -- 15-seed wide-seed verification of nb3020 per-fold SLSQP simplex
            on 3 K-anchors {K18, K19, K23} all at deep-30.

CONTEXT:
    nb3020 reported pooled outer-val RAE 0.4501 on single kf_seed=1001
    ("BETTER_THAN_NB3001", -0.0010 vs nb3001 wide-seed mean 0.4511). Per
    cycle-160/245 wide-seed protocol, must verify with >=15 fresh seeds
    before promote.

    nb3020 paradigm: rebuilt K=23 at deep-30 (30 fresh resid seeds
    3001..3030, identical recipe to nb3000/nb2960/nb3014), then per-fold
    SLSQP simplex on {K18, K19, K23} all deep-30. All 3 anchors are
    PRE-unblind (chemprop_aux anchor + residual LGBM on K-feature slices of
    nb2231 117-col matrix); no POST-unblind contamination chain risk.

PROTOCOL:
    Re-run nb3020 per-fold SLSQP simplex step at 15 FRESH kf_seeds
    {1051..1065} (NEW seeds, not used in any prior nb29xx/nb30xx wide-seed
    verify or single-fit). Per kf_seed: 5-fold scaffold split, SLSQP simplex
    per fold on {K18, K19, K23} deep-30 cached OOFs.
    Report mean +/- std + 95% CI across the 15 fresh seeds.

    Anchors LOADED from caches (NO rebuild):
        K18: data/processed/nb2960_K18_30seed_oof.npy / _te.npy
        K19: data/processed/nb3000_K19_30seed_oof.npy / te_nb3000_K19.npy
        K23: data/processed/nb3020_K23_30seed_oof.npy / te_nb3020_K23.npy

GATE (on 15-seed mean):
    mean < 0.4511 -> "VERIFIED_NEW_PRIMARY1"
        (nb3020 beats nb3001 wide-seed-verified ceiling 0.4511, new top)
    mean < 0.4518 -> "VERIFIED_MARGINAL"
        (still inside nb3003 single-seed reference but not new PRIMARY-1)
    shift > +0.005 vs single-kf -> "LUCKY_SEED_TRAP"
        (nb3020 single-kf=1001 was a fortunate seed; reject promotion)
    else -> "FAIL_OR_REPORT"

References:
    nb3020 single-kf=1001 pooled RAE  = 0.4501   <- parent
    nb3001 15-seed wide-mean          = 0.4511   <- current best ceiling
    nb3003 5-anchor single-seed       = 0.4518   <- PROMOTE band
    nb2982 single-kf=1001             = 0.4505   (2-anchor K18,K20)
    nb2990 15-seed wide-mean (verify) = ?        (same protocol as this)
    nb2960 K18 deep-30                = 0.4536
    nb3000 K19 deep-30                = 0.4607
    nb3020 K23 deep-30                = 0.4XXX   (rebuild)
    nb2171 ceiling deep-30            = 0.4682

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
    data/processed/nb3030_summary.json
    data/processed/nb3030_pred_oof.npy   (253,) float32 -- median-seed OOF (always)
    data/processed/te_nb3030.npy         (513,) float32 -- full-pool deploy te (always)
    submissions/nb3030_wide_seed_verify_nb3020.csv  (only on promote verdicts)
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

TAG = "nb3030"
PARENT_TAG = "nb3020"

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
KF_SEEDS = list(range(1051, 1066))   # 15 fresh seeds {1051..1065}
                                     #   NOT 1001 (nb3020 single-fit)
                                     #   NOT 1006-1050 (prior wide-seed verifies)
N_STARTS_FOLD = 8
N_STARTS_FULL = 12
DEGEN_MAX_W = 0.85

# -- Gates ---------------------------------------------------------------------
GATE_NEW_PRIMARY1 = 0.4511            # mean < this -> VERIFIED_NEW_PRIMARY1
GATE_MARGINAL = 0.4518                # mean < this -> VERIFIED_MARGINAL
LUCKY_SHIFT_THRESHOLD = 0.005         # shift > this -> LUCKY_SEED_TRAP

# -- References ----------------------------------------------------------------
REF_NB3020_SINGLE_KF = 0.4501
REF_NB3001_WIDE_MEAN = 0.4511
REF_NB3003_SINGLE_KF = 0.4518
REF_NB2982_SINGLE_KF = 0.4505
REF_K18 = 0.4536
REF_K19 = 0.4607
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
    """Run nb3020 per-fold SLSQP simplex pipeline at a single kf_seed."""
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
    print(f"          all 3 anchors deep-30 (PRE-unblind chemprop_aux + resid)")
    print(f"          kf_seeds = {KF_SEEDS}")
    print(f"          (fresh, EXCLUDES 1001 from nb3020 single-fit and "
          f"all prior wide-seed seeds 1006-1050)")
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
    print("STEP 1: load 3 K-anchor deep-30 OOFs and te arrays")
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

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # Pair-wise correlations
    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n  OOF correlation matrix (all deep-30):")
    print(f"        {'  '.join([f'{k:>6s}' for k in K_LABELS])}")
    for i, ki in enumerate(K_LABELS):
        row = "  ".join([f"{corr_mat[i, j]:6.3f}" for j in range(K)])
        print(f"   {ki:>6s}  {row}")

    # -- Scaffolds (kf_seed independent) -------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Wide-seed sweep ------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"WIDE-SEED SWEEP: {len(KF_SEEDS)} fresh kf_seeds "
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
        wstr = ", ".join(f"{K_LABELS[k]}={res['mean_fold_weights'][k]:.3f}"
                         for k in range(K))
        print(f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
              f"degen={res['any_fold_degenerate']}  "
              f"w=[{wstr}]  wall={time.time()-ts:.2f}s")

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

    shift_vs_single = mean_rae - REF_NB3020_SINGLE_KF
    print(f"\n   nb3020 single-kf=1001     = {REF_NB3020_SINGLE_KF:.4f}")
    print(f"   shift (mean - single)     = {shift_vs_single:+.4f}")
    print(f"   nb3001 wide-seed ref      = {REF_NB3001_WIDE_MEAN:.4f}")
    print(f"   nb3003 single-seed ref    = {REF_NB3003_SINGLE_KF:.4f}")

    # Mean-of-seed mean-of-fold weights (deploy proxy)
    w_seed_mean = np.asarray(w_mean_stack).mean(axis=0)
    w_seed_mean = w_seed_mean / w_seed_mean.sum()
    print(f"\n   mean-of-seed mean-of-fold weights = "
          + ", ".join(f"{K_LABELS[k]}={w_seed_mean[k]:.4f}" for k in range(K)))

    # -- Deploy: SLSQP on FULL 253 (kf-seed-independent) ----------------------
    w_full, r_full = _simplex_slsqp(P_unb, y_unb, n_starts=N_STARTS_FULL, seed=0)
    full_pool_weights = {K_LABELS[k]: round(float(w_full[k]), 4) for k in range(K)}
    full_pool_degen = bool(w_full.max() > DEGEN_MAX_W)
    te_pred = (P_te @ w_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"\n   full-pool SLSQP weights   = "
          + ", ".join(f"{K_LABELS[k]}={w_full[k]:.4f}" for k in range(K)))
    print(f"   full-pool in-sample RAE   = {r_full:.4f}")
    print(f"   te[unb] in-sample RAE     = {te_unb_in_rae:.4f}")
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
    if shift_vs_single > LUCKY_SHIFT_THRESHOLD:
        verdict = "LUCKY_SEED_TRAP"
        ladder_action = (
            f"REJECT nb3020 promotion. nb3020 single-kf=1001 0.4501 was a "
            f"fortunate seed; wide-seed mean {mean_rae:.4f} shift "
            f"{shift_vs_single:+.4f} exceeds +{LUCKY_SHIFT_THRESHOLD:.3f} tolerance. "
            "Keep prior PRIMARY-1."
        )
    elif mean_rae < GATE_NEW_PRIMARY1:
        verdict = "VERIFIED_NEW_PRIMARY1"
        ladder_action = (
            f"PROMOTE nb3020 to PRIMARY-1 (wide-seed {mean_rae:.4f} beats "
            f"nb3001 ceiling {REF_NB3001_WIDE_MEAN:.4f}). "
            "Demote prior PRIMARY-1 to PRIMARY-2."
        )
    elif mean_rae < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
        ladder_action = (
            f"KEEP prior PRIMARY-1. nb3020 wide-seed {mean_rae:.4f} is inside "
            f"PROMOTE gate {GATE_MARGINAL} but not strictly better than "
            f"nb3001 ceiling {REF_NB3001_WIDE_MEAN}; tag as alternate."
        )
    else:
        verdict = "FAIL_OR_REPORT"
        ladder_action = (
            f"REJECT nb3020 promotion. Wide-seed mean {mean_rae:.4f} above "
            f"PROMOTE gate {GATE_MARGINAL}. Keep prior PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_wide_seed_verify_{PARENT_TAG}.csv"
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
        "method": "wide_seed_15_verify_per_fold_slsqp_K18_K19_K23_all_deep30",
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
        "ref_nb3020_single_kf": REF_NB3020_SINGLE_KF,
        "ref_nb3001_wide_mean": REF_NB3001_WIDE_MEAN,
        "ref_nb3003_single_kf": REF_NB3003_SINGLE_KF,
        "ref_nb2982_single_kf": REF_NB2982_SINGLE_KF,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "shift_mean_vs_single_kf": round(shift_vs_single, 4),
        "delta_vs_nb3001_ceiling": round(mean_rae - REF_NB3001_WIDE_MEAN, 4),
        "delta_vs_nb3003_marginal": round(mean_rae - REF_NB3003_SINGLE_KF, 4),
        "mean_of_seed_mean_fold_weights": {
            K_LABELS[k]: round(float(w_seed_mean[k]), 4) for k in range(K)
        },
        "full_pool_weights": full_pool_weights,
        "full_pool_rae_in_sample": round(float(r_full), 4),
        "full_pool_degenerate": full_pool_degen,
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
    print(f"   mean_rae (15 seeds)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   shift vs nb3020=1001  = {shift_vs_single:+.4f}")
    print(f"   delta vs nb3001 P1    = {mean_rae - REF_NB3001_WIDE_MEAN:+.4f}")
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
        "shift_mean_vs_single_kf", "delta_vs_nb3001_ceiling",
        "delta_vs_nb3003_marginal",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  full_pool_weights: {res.get('full_pool_weights')}")
