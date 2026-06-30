"""nb2980 -- 15-seed wide-seed verification of nb2973 per-fold SLSQP simplex 4K.

CONTEXT:
    nb2973 reported pooled outer-val RAE 0.4539 on single kf_seed=1001 (PROMOTE).
    Cycle 244 nb2970 showed nb2961 single-kf 0.4567 shifted to 0.4573 +/- 0.0025
    under 15 fresh seeds. Lucky-seed risk is real on n=253 outer-CV gates.

PROTOCOL:
    Re-run nb2973 pipeline (per-fold SLSQP simplex on 4 K-anchors) with 15 FRESH
    kf_seeds {1006..1020} (NOT including 1001 used in nb2973 single-fit).
    Per kf_seed: 5-fold scaffold split, SLSQP simplex per fold, pool RAE.
    Report mean +/- std + 95% CI across the 15 fresh seeds.

GATE (on 15-seed mean):
    mean < 0.4570  -> "VERIFIED_PROMOTE_PRIMARY1" (nb2973 stays PRIMARY-1)
    mean < 0.4598  -> "VERIFIED_MARGINAL"          (demote to PRIMARY-2)
    shift > +0.005 -> "LUCKY_SEED_TRAP"            (revert nb2973 promotion)

References:
    nb2973 single-kf=1001 pooled RAE        = 0.4539 (reported PROMOTE)
    nb2960 equal_K(K18,K24,K28)             = 0.4567
    nb2961 single-kf=1001                   = 0.4567 -> wide-seed 0.4573 +/- 0.0025
    nb2171 ceiling deep-30                  = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K{18,20,24,28}_30seed_oof.npy
    data/processed/nb2960_K{18,20,24,28}_30seed_te.npy

Outputs:
    data/processed/nb2980_summary.json
    data/processed/nb2980_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb2980.npy         (513,) float32 -- mean-weight deploy te
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
from pxr.paths import DATA_PROCESSED

TAG = "nb2980"
PARENT_TAG = "nb2973"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K20", "K24", "K28"]
OOF_PATHS = {k: DATA_PROCESSED / f"nb2960_{k}_30seed_oof.npy" for k in K_LABELS}
TE_PATHS = {k: DATA_PROCESSED / f"nb2960_{k}_30seed_te.npy" for k in K_LABELS}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1006, 1021))   # 15 fresh seeds {1006..1020}, EXCLUDES 1001
N_STARTS_FOLD = 8
N_STARTS_FULL = 12
DEGEN_MAX_W = 0.85

# -- Gates ---------------------------------------------------------------------
GATE_PROMOTE = 0.4570        # mean < this -> VERIFIED_PROMOTE_PRIMARY1
GATE_MARGINAL = 0.4598       # mean < this -> VERIFIED_MARGINAL
LUCKY_SHIFT_THRESHOLD = 0.005

# -- References ----------------------------------------------------------------
REF_NB2973_SINGLE_KF = 0.4539
REF_NB2961_WIDE_MEAN = 0.4573
REF_NB2961_WIDE_STD = 0.0025
REF_K18 = 0.4536
REF_EQUAL_K_18_24_28 = 0.4567
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
    """Run nb2973 per-fold SLSQP simplex pipeline at a single kf_seed."""
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
    print(f"{TAG} -- 15-seed wide-seed verify of {PARENT_TAG} per-fold SLSQP 4K")
    print(f"          kf_seeds = {KF_SEEDS}  (fresh, EXCLUDES 1001 from nb2973)")
    print(f"          gates: <{GATE_PROMOTE} VERIFIED_PROMOTE / "
          f"<{GATE_MARGINAL} VERIFIED_MARGINAL / "
          f">+{LUCKY_SHIFT_THRESHOLD:.3f} LUCKY_SEED_TRAP")
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load deep-30 K-anchor OOFs + te arrays -------------------------------
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
        per_K_full_rae[k] = round(float(rae(y_unb, oof)), 4)
        print(f"   {k}: oof_RAE = {per_K_full_rae[k]:.4f}")

    P_unb = np.column_stack(oof_cols)  # (253, 4)
    P_te = np.column_stack(te_cols)    # (513, 4)
    K = len(K_LABELS)

    # -- Scaffolds (kf_seed independent) -------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaf] n_unique_scaffolds = {n_unique_scaf}")

    # -- Wide-seed sweep ------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"WIDE-SEED SWEEP: {len(KF_SEEDS)} fresh kf_seeds")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    w_mean_stack = []
    oof_stack = []
    for s in KF_SEEDS:
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
              f"degen={res['any_fold_degenerate']}")

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

    shift_vs_single = mean_rae - REF_NB2973_SINGLE_KF
    print(f"\n   nb2973 single-kf=1001     = {REF_NB2973_SINGLE_KF:.4f}")
    print(f"   shift (mean - single)     = {shift_vs_single:+.4f}")
    print(f"   nb2961 wide-seed ref      = {REF_NB2961_WIDE_MEAN:.4f} "
          f"+/- {REF_NB2961_WIDE_STD:.4f}")

    # Mean-of-seed mean-of-fold weights (deploy proxy)
    w_seed_mean = np.asarray(w_mean_stack).mean(axis=0)
    w_seed_mean = w_seed_mean / w_seed_mean.sum()
    print(f"\n   mean-of-seed mean-of-fold weights = "
          + ", ".join(f"{K_LABELS[k]}={w_seed_mean[k]:.4f}" for k in range(K)))

    # -- Deploy: SLSQP on FULL 253 (kf-seed-independent) ----------------------
    w_full, r_full = _simplex_slsqp(P_unb, y_unb, n_starts=N_STARTS_FULL, seed=0)
    te_pred = (P_te @ w_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"\n   full-pool SLSQP weights   = "
          + ", ".join(f"{K_LABELS[k]}={w_full[k]:.4f}" for k in range(K)))
    print(f"   full-pool in-sample RAE   = {r_full:.4f}")
    print(f"   te[unb] in-sample RAE     = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[len(arr) // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    if shift_vs_single > LUCKY_SHIFT_THRESHOLD:
        verdict = "LUCKY_SEED_TRAP"
        ladder_action = (
            "REVERT nb2973 promotion. Restore prior PRIMARY-1 "
            "(nb2943 or nb2240 per ladder log)."
        )
    elif mean_rae < GATE_PROMOTE:
        verdict = "VERIFIED_PROMOTE_PRIMARY1"
        ladder_action = "KEEP nb2973 as PRIMARY-1. Ladder unchanged."
    elif mean_rae < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
        ladder_action = (
            "DEMOTE nb2973 to PRIMARY-2; restore prior PRIMARY-1 "
            "(nb2943 or nb2240)."
        )
    else:
        verdict = "LUCKY_SEED_TRAP"
        ladder_action = (
            "REVERT nb2973 promotion. Restore prior PRIMARY-1 "
            "(nb2943 or nb2240 per ladder log)."
        )

    print("\n" + "-" * 78)
    print(f"VERDICT: {verdict}")
    print(f"LADDER ACTION: {ladder_action}")
    print("-" * 78)

    # -- Save -----------------------------------------------------------------
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "wide_seed_15_verify_per_fold_slsqp_4K",
        "anchor_pool": K_LABELS,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
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
        "ref_nb2973_single_kf": REF_NB2973_SINGLE_KF,
        "shift_mean_vs_single_kf": round(shift_vs_single, 4),
        "ref_nb2961_wide_mean": REF_NB2961_WIDE_MEAN,
        "ref_nb2961_wide_std": REF_NB2961_WIDE_STD,
        "ref_K18_deep30": REF_K18,
        "ref_equal_K_18_24_28": REF_EQUAL_K_18_24_28,
        "ref_nb2171": REF_NB2171,
        "mean_of_seed_mean_fold_weights": {
            K_LABELS[k]: round(float(w_seed_mean[k]), 4) for k in range(K)
        },
        "full_pool_weights": {
            K_LABELS[k]: round(float(w_full[k]), 4) for k in range(K)
        },
        "full_pool_rae_in_sample": round(float(r_full), 4),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "gate_promote": GATE_PROMOTE,
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
    print(f"   shift vs nb2973=1001  = {shift_vs_single:+.4f}")
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
        "shift_mean_vs_single_kf", "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
