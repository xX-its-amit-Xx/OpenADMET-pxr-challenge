"""nb3240 -- Sigmoid soft-clip on nb3090 with SLOPE=1.0 (lighter compression vs nb3234).

NEW PARADIGM: lighter sigmoid compression.

    nb3234 used SLOPE=1.5 on the sigmoid soft-clip (sigmoid stretches center past
    identity then asymptotes). nb3240 substitutes SLOPE=1.0, which leaves the
    local derivative at raw=mid equal to 1 (identity-passthrough at the central
    mass), and only compresses the tails via the tanh asymptote.

    Soft-clip transform (per fold-train calibrated):
        mid   = mean(fold_train_y)
        half  = (q95(fold_train_y) - q05(fold_train_y)) / 2
        slope = 1.0
        pred  = mid + half * tanh(slope * (raw - mid) / half)

    Key properties vs nb3234 (slope=1.5):
      - Local derivative at raw=mid = 1.0 (identity at the central mass)
      - Same asymptote (mid +- half) -- tail compression is identical
      - Smooth, monotone, rank-preserving
      - LIGHTER central compression: rows close to mid are passed through almost
        unchanged, only the tails get squashed

    Hypothesis: if nb3234 slope=1.5 was over-compressing the center (effective
    shrink toward mid) and that hurt rank-correlated rows near the central mass,
    slope=1.0 should recover the parent rank at the center while still capturing
    the tail variance gain. If slope=1.5 was the right amount, slope=1.0 will
    be slightly worse.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    pred_base = nb3090_pred_oof  (253,)  -- q35-quantile-conditional blend
    Per outer fold:
        a) Compute (mid, half) from fold-train y ONLY.
           mid  = mean(y[fold_train])
           half = (q95(y[fold_train]) - q05(y[fold_train])) / 2
        b) Soft-clip fold-val with slope=1.0:
           val_pred = mid + half * tanh(1.0 * (pred_base[fold_val] - mid) / half)
        c) Stitch into oof_soft; record per-fold val RAE.
    Repeat for 15 FRESH kf_seeds {1216..1230}.

GATE (on 15-seed PER-FOLD-MEAN):
    pf_mean < 0.4424 -> "BETTER"
    else             -> "FAIL"

References:
    nb3090 q35 fine-cut blend           = 0.4472  <- parent anchor
    nb3173 learned-clip on nb3080       = 0.4437  (clip-operator ceiling)
    nb3190 learned-clip on nb3090       = 0.4422  (q35 hard-clip target)
    nb3234 sigmoid slope=1.5 on nb3090  = sibling reference
    nb2171 prior post-hoc PRIMARY-1     = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy
    data/processed/te_nb3090.npy

Outputs:
    data/processed/nb3240_summary.json
    data/processed/nb3240_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3240.npy         (513,) float32 -- deploy te
    submissions/nb3240_sigmoid_slope1.csv  (only on BETTER)
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

TAG = "nb3240"
PARENT_TAG = "nb3090"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3090.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Sigmoid soft-clip parameters ----------------------------------------------
# pred = mid + (range/2) * tanh(SLOPE * (raw - mid) / (range/2))
#   mid   = mean(fold_train_y)
#   range = q95(fold_train_y) - q05(fold_train_y)
SLOPE = 1.0  # nb3240: lighter compression vs nb3234 (slope=1.5)
Q_LOW = 0.05
Q_HIGH = 0.95

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4424     # per-fold-mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_PARENT_NB3090 = 0.4472
REF_NB3173 = 0.4437
REF_NB3190 = 0.4422
REF_NB3234_SLOPE15 = None  # sibling reference (slope=1.5), filled at runtime if available
REF_NB2171 = 0.4682


def _soft_clip(
    raw: np.ndarray,
    mid: float,
    half: float,
    slope: float,
) -> np.ndarray:
    """Sigmoid (tanh) soft-clip preserving monotonic rank.

    pred = mid + half * tanh(slope * (raw - mid) / half)

    Asymptotes to (mid - half, mid + half). At raw = mid returns mid.
    Local derivative at raw = mid is `slope` (so slope < 1 shrinks center,
    slope > 1 stretches center then asymptotes; slope = 1 is identity at center).
    """
    if half <= 0:
        return np.full_like(raw, mid, dtype=np.float64)
    z = (raw - mid) / half
    return mid + half * np.tanh(slope * z)


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run sigmoid soft-clip pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_soft = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_mids = []
    fold_halfs = []
    fold_los = []
    fold_his = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        y_tr = y_unb[tr_loc]
        mid = float(np.mean(y_tr))
        q05 = float(np.quantile(y_tr, Q_LOW))
        q95 = float(np.quantile(y_tr, Q_HIGH))
        rng = q95 - q05
        half = rng / 2.0
        fold_mids.append(mid)
        fold_halfs.append(half)
        fold_los.append(mid - half)
        fold_his.append(mid + half)
        val_pred = _soft_clip(pred_base[va_loc], mid, half, SLOPE)
        oof_soft[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))

    if np.isnan(oof_soft).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_soft))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_mids": fold_mids,
        "fold_halfs": fold_halfs,
        "fold_lo_asymp_mean": float(np.mean(fold_los)),
        "fold_hi_asymp_mean": float(np.mean(fold_his)),
        "oof": oof_soft,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- SIGMOID soft-clip on {PARENT_TAG} "
        f"(LIGHTER compression vs nb3234)"
    )
    print(f"          SLOPE = {SLOPE}  (Q_LOW={Q_LOW}, Q_HIGH={Q_HIGH})")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          honest gate metric = PER-FOLD-MEAN"
    )
    print(
        f"          gate: pf_mean < {GATE_BETTER:.4f} -> BETTER, else FAIL"
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

    # -- Load nb3090 anchor pred_oof + te -------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {PARENT_TAG} pred_oof + te (q35 hard-split blend)")
    print("-" * 78)
    pred_base = np.load(PRED_OOF_PATH).astype(np.float64)
    te_base = np.load(TE_PATH).astype(np.float64)
    if pred_base.shape != (n_unb,):
        raise ValueError(
            f"{PARENT_TAG} pred_oof shape {pred_base.shape} != ({n_unb},)"
        )
    if te_base.shape != (n_test,):
        raise ValueError(
            f"{PARENT_TAG} te shape {te_base.shape} != ({n_test},)"
        )
    full_oof_rae = float(rae(y_unb, pred_base))
    print(
        f"   pred_base: oof_RAE={full_oof_rae:.4f}  "
        f"mean={pred_base.mean():.3f}  std={pred_base.std():.3f}  "
        f"min={pred_base.min():.3f}  max={pred_base.max():.3f}"
    )
    print(
        f"   te_base:   mean={te_base.mean():.3f}  std={te_base.std():.3f}  "
        f"min={te_base.min():.3f}  max={te_base.max():.3f}"
    )

    # Leak sanity on parent
    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN parent: {leak_eq:.1%} rows == truth -- possible leak")

    # Truth stats
    print(
        f"   y_unb stats: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
        f"min={y_unb.min():.3f}  max={y_unb.max():.3f}"
    )

    # -- Sibling reference (nb3234 slope=1.5) --------------------------------
    nb3234_pf_mean = None
    nb3234_summary_path = DATA_PROCESSED / "nb3234_summary.json"
    if nb3234_summary_path.exists():
        try:
            with open(nb3234_summary_path, "r", encoding="utf-8") as f:
                nb3234_sum = json.load(f)
            nb3234_pf_mean = float(nb3234_sum.get("per_fold_mean_rae_mean"))
            print(
                f"   sibling nb3234 (slope=1.5) pf_mean = {nb3234_pf_mean:.4f}"
            )
        except Exception as e:
            print(f"   [warn] could not read nb3234 summary: {e}")

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
    per_fold_means = []
    oof_stack = []
    all_fold_mids = []
    all_fold_halfs = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_fold_mids.extend(res["fold_mids"])
        all_fold_halfs.extend(res["fold_halfs"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_mids": [round(v, 4) for v in res["fold_mids"]],
            "fold_halfs": [round(v, 4) for v in res["fold_halfs"]],
            "fold_lo_asymp_mean": round(res["fold_lo_asymp_mean"], 4),
            "fold_hi_asymp_mean": round(res["fold_hi_asymp_mean"], 4),
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"mid_mean={np.mean(res['fold_mids']):.3f}  "
            f"half_mean={np.mean(res['fold_halfs']):.3f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    arr_pf = np.asarray(per_fold_means, dtype=np.float64)
    pf_mean = float(arr_pf.mean())
    pf_std = float(arr_pf.std(ddof=1)) if n_s > 1 else 0.0
    pf_sem = pf_std / np.sqrt(n_s) if n_s > 1 else 0.0
    pf_ci_low = pf_mean - t_mult * pf_sem
    pf_ci_high = pf_mean + t_mult * pf_sem
    pf_median = float(np.median(arr_pf))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   POOLED RAE:")
    print(f"     mean   = {mean_rae:.4f}")
    print(f"     std    = {std_rae:.4f}")
    print(f"     sem    = {sem:.4f}")
    print(f"     95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"     median = {median_rae:.4f}")
    print(f"     min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   PER-FOLD-MEAN RAE (HONEST GATE METRIC):")
    print(f"     mean   = {pf_mean:.4f}")
    print(f"     std    = {pf_std:.4f}")
    print(f"     sem    = {pf_sem:.4f}")
    print(f"     95% CI = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"     median = {pf_median:.4f}")
    print(f"     min/max = [{arr_pf.min():.4f}, {arr_pf.max():.4f}]")
    print(
        f"\n   ref {PARENT_TAG} pooled baseline = {REF_PARENT_NB3090:.4f}"
    )
    print(
        f"   delta vs {PARENT_TAG} (pooled)   = "
        f"{mean_rae - REF_PARENT_NB3090:+.4f}"
    )
    print(
        f"   delta vs {PARENT_TAG} (pf_mean)  = "
        f"{pf_mean - REF_PARENT_NB3090:+.4f}"
    )
    print(
        f"   ref nb3190 hard-clip on nb3090 = {REF_NB3190:.4f}  "
        f"(rank-preserving sensitivity)"
    )
    print(
        f"   delta vs nb3190 (pf_mean)      = "
        f"{pf_mean - REF_NB3190:+.4f}"
    )
    if nb3234_pf_mean is not None:
        print(
            f"   ref nb3234 sibling (slope=1.5) = {nb3234_pf_mean:.4f}"
        )
        print(
            f"   delta vs nb3234 sibling        = "
            f"{pf_mean - nb3234_pf_mean:+.4f}"
        )
    print(
        f"   ref nb3173 clip-operator   = {REF_NB3173:.4f}"
    )
    print(f"   ref nb2171 prior PRIMARY-1 = {REF_NB2171:.4f}")
    print(f"   gain vs nb2171 (pf_mean)        = {REF_NB2171 - pf_mean:+.4f}")

    all_mids_arr = np.asarray(all_fold_mids, dtype=np.float64)
    all_halfs_arr = np.asarray(all_fold_halfs, dtype=np.float64)
    print(
        f"\n   75-fold mid stats:  mean={all_mids_arr.mean():.4f}  "
        f"std={all_mids_arr.std(ddof=1):.4f}  "
        f"min={all_mids_arr.min():.4f}  max={all_mids_arr.max():.4f}"
    )
    print(
        f"   75-fold half stats: mean={all_halfs_arr.mean():.4f}  "
        f"std={all_halfs_arr.std(ddof=1):.4f}  "
        f"min={all_halfs_arr.min():.4f}  max={all_halfs_arr.max():.4f}"
    )

    # -- Deploy: calibrate (mid, half) on FULL 253 y -------------------------
    deploy_mid = float(np.mean(y_unb))
    deploy_q05 = float(np.quantile(y_unb, Q_LOW))
    deploy_q95 = float(np.quantile(y_unb, Q_HIGH))
    deploy_half = (deploy_q95 - deploy_q05) / 2.0
    te_pred = _soft_clip(te_base, deploy_mid, deploy_half, SLOPE).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    n_te_below_mid = int(np.sum(te_base < deploy_mid))
    n_te_above_mid = int(np.sum(te_base > deploy_mid))
    print(
        f"\n   deploy (mid, half) = ({deploy_mid:.4f}, {deploy_half:.4f})  "
        f"asymptotes [{deploy_mid - deploy_half:.4f}, "
        f"{deploy_mid + deploy_half:.4f}]"
    )
    print(
        f"   te(513) below_mid={n_te_below_mid}  above_mid={n_te_above_mid}"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage (median over per-fold-mean -- honest metric)
    med_seed_idx = int(np.argsort(arr_pf)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"   median seed = {median_seed} "
        f"(pf_mean={arr_pf[med_seed_idx]:.4f}, pooled={arr[med_seed_idx]:.4f})"
    )

    # -- Gate (on PER-FOLD-MEAN per task) ------------------------------------
    print("\n" + "-" * 78)
    print("GATE (honest metric = PER-FOLD-MEAN)")
    print("-" * 78)
    sib_clause = (
        f" Sibling nb3234 (slope=1.5) pf_mean = {nb3234_pf_mean:.4f}; "
        f"delta = {pf_mean - nb3234_pf_mean:+.4f}."
        if nb3234_pf_mean is not None else ""
    )
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3240 15-seed PER-FOLD-MEAN {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f} "
            f"({pf_mean - GATE_BETTER:+.4f}). Sigmoid soft-clip with LIGHTER "
            f"slope={SLOPE} (identity passthrough at central mass, tail-only "
            f"compression via tanh asymptote to q05/q95) on q35 anchor nb3090 "
            f"({REF_PARENT_NB3090:.4f}) -> {pf_mean:.4f} = "
            f"{REF_PARENT_NB3090 - pf_mean:.4f} RAE reduction.{sib_clause} "
            f"Confirms lighter compression preserves central rank order while "
            f"still capturing tail variance gain. Re-verify with deep-30 before "
            f"PRIMARY-1 swap. anchor_pre_unblind=True (parent nb3090 built on "
            f"PRE-clean K18/K19 deep-30 OOFs)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3240 15-seed PER-FOLD-MEAN {pf_mean:.4f} fails BETTER "
            f"gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). Delta vs "
            f"parent nb3090 (pf_mean) = {pf_mean - REF_PARENT_NB3090:+.4f}, "
            f"delta vs hard-clip nb3190 = {pf_mean - REF_NB3190:+.4f}.{sib_clause} "
            f"Lighter slope=1.0 (identity at center) either under-compresses the "
            f"tails (tanh shoulder asymptotes too slowly without center stretch) "
            f"or removes the central-mass shrink that nb3234 slope=1.5 used to "
            f"down-weight high-confidence-but-wrong rows. If pf_mean is at or "
            f"below the parent ({REF_PARENT_NB3090:.4f}), slope=1.0 is at least "
            f"rank-preserving non-degrading; the lighter-compression axis is "
            f"closed at this slope value. If pf_mean > parent, sigmoid family on "
            f"this anchor is closed altogether. Closes the lighter-compression "
            f"sensitivity axis at slope=1.0, q05/q95 asymptote."
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

    sub_csv = SUBMISSIONS / f"{TAG}_sigmoid_slope1.csv"
    if verdict == "BETTER":
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
            "sigmoid_tanh_soft_clip_per_fold_calibrated_mid_mean_y_half_q95_q05_"
            "slope1_on_nb3090_q35_quantile_conditional_blend"
        ),
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,  # nb3090 built on PRE-clean K18/K19 deep-30
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "slope": SLOPE,
        "q_low_asymp": Q_LOW,
        "q_high_asymp": Q_HIGH,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_val_rae_means_array": [
            round(float(v), 4) for v in per_fold_means
        ],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "per_fold_mean_rae_mean": round(pf_mean, 4),
        "per_fold_mean_rae_std": round(pf_std, 4),
        "per_fold_mean_rae_sem": round(pf_sem, 4),
        "per_fold_mean_rae_ci95_low": round(pf_ci_low, 4),
        "per_fold_mean_rae_ci95_high": round(pf_ci_high, 4),
        "per_fold_mean_rae_median": round(pf_median, 4),
        "per_fold_mean_rae_min": round(float(arr_pf.min()), 4),
        "per_fold_mean_rae_max": round(float(arr_pf.max()), 4),
        "honest_metric": "per_fold_mean",
        "fold_mid_mean": round(float(all_mids_arr.mean()), 4),
        "fold_mid_std": round(float(all_mids_arr.std(ddof=1)), 4),
        "fold_half_mean": round(float(all_halfs_arr.mean()), 4),
        "fold_half_std": round(float(all_halfs_arr.std(ddof=1)), 4),
        "ref_parent_nb3090": REF_PARENT_NB3090,
        "delta_vs_parent_pooled": round(mean_rae - REF_PARENT_NB3090, 4),
        "delta_vs_parent_pf_mean": round(pf_mean - REF_PARENT_NB3090, 4),
        "ref_nb3190_hard_clip": REF_NB3190,
        "delta_vs_nb3190_pf_mean": round(pf_mean - REF_NB3190, 4),
        "ref_nb3234_sibling_slope15_pf_mean": (
            round(nb3234_pf_mean, 4) if nb3234_pf_mean is not None else None
        ),
        "delta_vs_nb3234_sibling_pf_mean": (
            round(pf_mean - nb3234_pf_mean, 4)
            if nb3234_pf_mean is not None else None
        ),
        "ref_nb3173": REF_NB3173,
        "ref_nb2171": REF_NB2171,
        "gain_vs_nb2171_pf_mean": round(REF_NB2171 - pf_mean, 4),
        "deploy_mid": round(deploy_mid, 4),
        "deploy_half": round(deploy_half, 4),
        "deploy_lo_asymp": round(deploy_mid - deploy_half, 4),
        "deploy_hi_asymp": round(deploy_mid + deploy_half, 4),
        "n_te_below_mid": n_te_below_mid,
        "n_te_above_mid": n_te_above_mid,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "gate_better": GATE_BETTER,
        "gate_metric": "per_fold_mean",
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
    print(f"   pf_mean ({n_s} seeds) = {pf_mean:.4f} +/- {pf_std:.4f}")
    print(f"   pf_mean 95% CI       = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   pooled mean          = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   delta vs nb3090 (pf) = {pf_mean - REF_PARENT_NB3090:+.4f}")
    print(f"   delta vs nb3190 (pf) = {pf_mean - REF_NB3190:+.4f}")
    if nb3234_pf_mean is not None:
        print(
            f"   delta vs nb3234 (pf) = {pf_mean - nb3234_pf_mean:+.4f}"
        )
    print(f"   gain vs nb2171  (pf) = {REF_NB2171 - pf_mean:+.4f}")
    print(f"   verdict              = {verdict}")
    print(f"   wall                 = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean_rae_mean", "per_fold_mean_rae_std",
        "per_fold_mean_rae_ci95_low", "per_fold_mean_rae_ci95_high",
        "mean_rae", "std_rae",
        "delta_vs_parent_pf_mean", "delta_vs_nb3190_pf_mean",
        "delta_vs_nb3234_sibling_pf_mean",
        "deploy_mid", "deploy_half",
        "deploy_lo_asymp", "deploy_hi_asymp",
        "n_te_below_mid", "n_te_above_mid",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
