"""nb3080 -- 15-seed wide-seed verification of nb3073 best combo
            quantile-conditional hard-split blend on {K18, K19} deep-30.

CONTEXT:
    nb3073 swept (q_cut, w_K18_low, w_K18_high) over 36 combos x 5 kf_seeds
    {1051..1055} and reported best combo (q_cut=0.4, w_K18_low=0.9,
    w_K18_high=0.5) at 5-seed mean **0.4470 +/- 0.0009**. This beats both
    nb3030 ceiling 0.4509 and nb3063 5-seed 0.4477. However, 5-seed std
    0.0009 is under-dispersed (cf. cycle-160 nb2060 5-seed std 0.00087 ->
    30-seed std 0.00408 = 4.7x). Cycle-160 rule + cycle-149 nb1191 lucky-seed
    re-verify rule: 5-seed gate-A pass at exceptional magnitudes triggers
    wide-seed (>=15) re-verification before promote.

PROTOCOL (per kf_seed, 5-fold scaffold split, anchors LOADED no rebuild):
    1. Compute fold-train K18 q_cut=0.4 quantile threshold q (per fold).
    2. Per fold-val row:
         pred_K18 <= q -> blend(0.9 K18, 0.1 K19)
         pred_K18 >  q -> blend(0.5 K18, 0.5 K19)
    3. Stitch into oof_blend (253,); pooled_rae across 5 outer folds.
    Repeat for 15 FRESH kf_seeds {1111..1125} (NEW, not used in any prior
    wide-seed verify or single-fit). Report mean +/- std + 95% CI.

GATE (on 15-seed mean):
    mean < 0.4477 -> "VERIFIED_NEW_PRIMARY1"
        (beats nb3070/nb3063 5-seed 0.4477 -> new top)
    mean < 0.4509 -> "VERIFIED_MARGINAL"
        (still inside PROMOTE band, beats nb3030 wide-seed 0.4509)
    mean - 0.4470 > +0.005 -> "LUCKY_SEED_TRAP"
        (5-seed kf {1051..1055} was a fortunate batch; reject promotion)
    else -> "FAIL_OR_REPORT"

References:
    nb3073 5-seed best combo mean    = 0.4470 +/- 0.0009 (UNDER-DISPERSED)
    nb3070 15-seed verify of nb3063  = (pending result; gate ref)
    nb3063 5-seed mean               = 0.4477
    nb3030 15-seed wide-mean         = 0.4509   <- ceiling to beat
    nb3002 single-kf=1001            = 0.4501
    nb2960 K18 deep-30 OOF           = 0.4536
    nb3000 K19 deep-30 OOF           = 0.4607
    nb2171 prior post-hoc top        = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3080_summary.json
    data/processed/nb3080_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3080.npy         (513,) float32 -- deploy te
    submissions/nb3080_verify_nb3073.csv  (only on PROMOTE verdicts)
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3080"
PARENT_TAG = "nb3073"

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

# -- Best combo from nb3073 (FIXED) --------------------------------------------
Q_CUT = 0.4
W_K18_LOW = 0.9   # w_K19_low = 1 - 0.9 = 0.1
W_K18_HIGH = 0.5  # w_K19_high = 1 - 0.5 = 0.5
W_K19_LOW = 1.0 - W_K18_LOW
W_K19_HIGH = 1.0 - W_K18_HIGH

# -- Gates ---------------------------------------------------------------------
GATE_VERIFIED_NEW_PRIMARY1 = 0.4477  # mean < this -> beats nb3063/nb3070 5-seed
GATE_VERIFIED_MARGINAL = 0.4509      # mean < this -> beats nb3030 ceiling
LUCKY_SEED_SHIFT = 0.005             # mean - parent > this -> LUCKY_SEED_TRAP

# -- References ----------------------------------------------------------------
REF_NB3073_5SEED = 0.4470
REF_NB3073_5SEED_STD = 0.0009
REF_NB3063_5SEED = 0.4477
REF_NB3030 = 0.4509
REF_NB3002 = 0.4501
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _blend_quantile_conditional(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q_thr: float,
) -> np.ndarray:
    """Per-row hard-split blend matching nb3073 best combo.

    rows with p_k18 <= q_thr -> (W_K18_LOW=0.9, W_K19_LOW=0.1)
    rows with p_k18 >  q_thr -> (W_K18_HIGH=0.5, W_K19_HIGH=0.5)
    """
    low_mask = p_k18 <= q_thr
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = (
        W_K18_LOW * p_k18[low_mask] + W_K19_LOW * p_k19[low_mask]
    )
    out[~low_mask] = (
        W_K18_HIGH * p_k18[~low_mask] + W_K19_HIGH * p_k19[~low_mask]
    )
    return out


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run quantile-conditional blend pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_q_thrs = []
    fold_high_share = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # Per-fold q_cut quantile from fold-train K18 preds
        q_thr = float(np.quantile(P_unb[tr_loc, 0], Q_CUT))
        fold_q_thrs.append(q_thr)
        val_p_k18 = P_unb[va_loc, 0]
        val_p_k19 = P_unb[va_loc, 1]
        val_pred = _blend_quantile_conditional(val_p_k18, val_p_k19, q_thr)
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_high_share.append(float(np.mean(val_p_k18 > q_thr)))

    if np.isnan(oof_blend).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_q_thr_mean": float(np.mean(fold_q_thrs)),
        "fold_q_thr_std": float(np.std(fold_q_thrs, ddof=1)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- WIDE-SEED VERIFY of {PARENT_TAG} best-combo "
        f"quantile-conditional blend on {K_LABELS} deep-30"
    )
    print(
        f"          fixed combo: q_cut={Q_CUT}, w_K18_low={W_K18_LOW} "
        f"(w_K19_low={W_K19_LOW}), w_K18_high={W_K18_HIGH} "
        f"(w_K19_high={W_K19_HIGH})"
    )
    print(
        f"          per-row weights: low (K18<=q): "
        f"(K18={W_K18_LOW}, K19={W_K19_LOW})"
    )
    print(
        f"                           high (K18 >q): "
        f"(K18={W_K18_HIGH}, K19={W_K19_HIGH})"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          {PARENT_TAG} 5-seed reference = "
        f"{REF_NB3073_5SEED:.4f} +/- {REF_NB3073_5SEED_STD:.4f}  "
        f"(UNDER-DISPERSED)"
    )
    print(
        f"          gate1: mean < {GATE_VERIFIED_NEW_PRIMARY1:.4f} -> "
        f"VERIFIED_NEW_PRIMARY1"
    )
    print(
        f"          gate2: mean < {GATE_VERIFIED_MARGINAL:.4f} -> "
        f"VERIFIED_MARGINAL"
    )
    print(
        f"          gate3: shift > +{LUCKY_SEED_SHIFT:.3f} -> LUCKY_SEED_TRAP"
    )
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )
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
        print(
            f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
            f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}"
        )

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
    print(
        f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
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
            "fold_q_thr_mean": round(res["fold_q_thr_mean"], 4),
            "fold_q_thr_std": round(res["fold_q_thr_std"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"q_thr_mean={res['fold_q_thr_mean']:.3f}  "
            f"high_share={res['fold_high_share_mean']:.2f}  "
            f"wall={time.time()-ts:.2f}s"
        )

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

    # Under-dispersion ratio vs nb3073 5-seed std
    ud_ratio = (
        std_rae / REF_NB3073_5SEED_STD
        if REF_NB3073_5SEED_STD > 0
        else float("nan")
    )

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   sem     = {sem:.4f}")
    print(f"   95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median  = {median_rae:.4f}")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(
        f"\n   ref {PARENT_TAG} 5-seed mean       = "
        f"{REF_NB3073_5SEED:.4f} +/- {REF_NB3073_5SEED_STD:.4f}"
    )
    print(
        f"   shift vs {PARENT_TAG} 5-seed mean = "
        f"{mean_rae - REF_NB3073_5SEED:+.4f}"
    )
    print(
        f"   under-dispersion ratio 15/5     = {ud_ratio:.2f}x "
        f"(>2.0 = under-dispersed in 5-seed)"
    )
    print(f"   ref nb3063 5-seed               = {REF_NB3063_5SEED:.4f}")
    print(f"   ref nb3030 wide-seed ceiling    = {REF_NB3030:.4f}")
    print(f"   delta vs nb3030                 = {mean_rae - REF_NB3030:+.4f}")

    # -- Deploy: q_thr from FULL 253 K18 OOF, then blend te -----------------
    deploy_q_thr = float(np.quantile(P_unb[:, 0], Q_CUT))
    te_k18 = P_te[:, 0]
    te_k19 = P_te[:, 1]
    te_pred = _blend_quantile_conditional(
        te_k18, te_k19, deploy_q_thr,
    ).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    te_low_share = float(np.mean(te_k18 <= deploy_q_thr))
    print(
        f"\n   deploy q_thr (full K18 OOF q{Q_CUT}) = {deploy_q_thr:.4f}"
    )
    print(f"   te(513) low-half share = {te_low_share:.3f}")
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
    shift = mean_rae - REF_NB3073_5SEED
    if shift > LUCKY_SEED_SHIFT:
        verdict = "LUCKY_SEED_TRAP"
        ladder_action = (
            f"REJECT. nb3080 15-seed mean {mean_rae:.4f} shifts "
            f"{shift:+.4f} vs nb3073 5-seed {REF_NB3073_5SEED:.4f} "
            f"(> +{LUCKY_SEED_SHIFT:.3f} lucky-seed threshold). The 5-seed "
            f"std={REF_NB3073_5SEED_STD:.4f} was under-dispersed "
            f"(15-seed std={std_rae:.4f}, ratio {ud_ratio:.1f}x). "
            f"nb3073 best combo NOT a stable improvement over nb3030. "
            "Keep nb3030 / prior PRIMARY-1."
        )
    elif mean_rae < GATE_VERIFIED_NEW_PRIMARY1:
        verdict = "VERIFIED_NEW_PRIMARY1"
        ladder_action = (
            f"PROMOTE. nb3080 wide-seed 15-mean {mean_rae:.4f} beats "
            f"nb3063/nb3070 5-seed {GATE_VERIFIED_NEW_PRIMARY1:.4f} "
            f"({mean_rae - GATE_VERIFIED_NEW_PRIMARY1:+.4f}) and "
            f"nb3030 ceiling {REF_NB3030:.4f} "
            f"({mean_rae - REF_NB3030:+.4f}). Wide-seed shift vs nb3073 "
            f"5-seed = {shift:+.4f} (within tolerance). nb3073 "
            f"best combo (q_cut={Q_CUT}, w_K18_low={W_K18_LOW}, "
            f"w_K18_high={W_K18_HIGH}) VERIFIED as new PRIMARY-1 candidate."
        )
    elif mean_rae < GATE_VERIFIED_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
        ladder_action = (
            f"PROMOTE-MARGINAL. nb3080 wide-seed 15-mean {mean_rae:.4f} "
            f"inside PROMOTE band (< {GATE_VERIFIED_MARGINAL:.4f}) beats "
            f"nb3030 ceiling {REF_NB3030:.4f} "
            f"({mean_rae - REF_NB3030:+.4f}) but not new PRIMARY-1 gate "
            f"{GATE_VERIFIED_NEW_PRIMARY1:.4f}. Keep prior PRIMARY-1, "
            f"nb3073 best combo as alternate candidate."
        )
    else:
        verdict = "FAIL_OR_REPORT"
        ladder_action = (
            f"REPORT. nb3080 wide-seed 15-mean {mean_rae:.4f} not inside "
            f"PROMOTE band ({GATE_VERIFIED_MARGINAL:.4f}). Shift vs nb3073 "
            f"5-seed = {shift:+.4f}. Under-dispersion ratio "
            f"{ud_ratio:.1f}x (15-seed std={std_rae:.4f} vs 5-seed "
            f"{REF_NB3073_5SEED_STD:.4f}). Keep nb3030 / prior PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_verify_nb3073.csv"
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
        "method": (
            "wide_seed_verify_quantile_conditional_hard_split_blend_"
            "K18_K19_deep30_best_combo"
        ),
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "best_combo": {
            "q_cut": Q_CUT,
            "w_K18_low": W_K18_LOW,
            "w_K19_low": W_K19_LOW,
            "w_K18_high": W_K18_HIGH,
            "w_K19_high": W_K19_HIGH,
        },
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
        "ref_parent_5seed_mean": REF_NB3073_5SEED,
        "ref_parent_5seed_std": REF_NB3073_5SEED_STD,
        "shift_vs_parent_5seed": round(shift, 4),
        "under_dispersion_ratio_15_over_5": round(ud_ratio, 2),
        "ref_nb3063_5seed": REF_NB3063_5SEED,
        "ref_nb3030": REF_NB3030,
        "ref_nb3002": REF_NB3002,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3030": round(mean_rae - REF_NB3030, 4),
        "deploy_q_thr": round(deploy_q_thr, 4),
        "te_low_share": round(te_low_share, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (
            str(sub_csv) if verdict in promote_verdicts else None
        ),
        "gate_verified_new_primary1": GATE_VERIFIED_NEW_PRIMARY1,
        "gate_verified_marginal": GATE_VERIFIED_MARGINAL,
        "gate_lucky_seed_shift": LUCKY_SEED_SHIFT,
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
    print(f"   mean_rae ({n_s} seeds)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   shift vs nb3073 5sd   = {shift:+.4f}")
    print(f"   under-disp ratio      = {ud_ratio:.2f}x")
    print(f"   delta vs nb3030       = {mean_rae - REF_NB3030:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "shift_vs_parent_5seed", "under_dispersion_ratio_15_over_5",
        "delta_vs_nb3030",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  best_combo: {res.get('best_combo')}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
