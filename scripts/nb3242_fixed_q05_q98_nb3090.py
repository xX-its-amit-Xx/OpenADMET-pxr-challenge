"""nb3242 -- FIXED (q05, q98) clip on nb3090 anchor (skip per-fold grid search).

NEW PARADIGM:
    nb3190 ran an inner-grid search per fold to LEARN the best (q_low, q_high)
    on fold-train. Across 75 folds (15 seeds * 5 folds) the modal pick was
    (q05, q98) at 70-72 / 75 folds (>=93%). The per-fold grid burns
    inner-train degrees of freedom and adds variance for almost no
    informational gain when the modal pick is this concentrated.

    nb3242 SKIPS the per-fold search and uses the modal (q05, q98) directly.
    The expected outcome is a tighter, lower-variance estimate of essentially
    the same operator: same anchor (nb3090), same clip semantics, but no
    selection noise. If the operator was truly worth its variance, nb3242
    matches or beats nb3190's 0.4426 mean; if the per-fold search was just
    noise, the fixed version wins by removing it.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    pred_base = nb3090_pred_oof  (253,)
    Per outer fold:
        lo = quantile(y[fold_train], 0.05)
        hi = quantile(y[fold_train], 0.98)
        val_pred = np.clip(pred_base[fold_val], lo, hi)
    Stitch into oof_clip; pooled RAE across 5 folds.
    Repeat for 15 FRESH kf_seeds {1216..1230}.

GATE (per-fold-mean across 15 seeds):
    mean < 0.4424 -> "BETTER"   (beats nb3190 0.4426 modal-grid mean)
    else          -> "FAIL"

References:
    nb3190 learned-clip on nb3090 = 0.4426  <- modal-grid prior
    nb3090 best combo 15-seed     = 0.4472  <- parent anchor
    nb3173 learned-clip on nb3080 = 0.4437
    nb3170 fixed q05/q95 on nb3080 = 0.4437
    nb3080 wide-seed verify       = 0.4475
    nb3030 wide-seed SLSQP        = 0.4509
    nb2960 K18 deep-30 OOF        = 0.4536
    nb2171 prior post-hoc top     = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy
    data/processed/te_nb3090.npy

Outputs:
    data/processed/nb3242_summary.json
    data/processed/nb3242_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3242.npy         (513,) float32 -- deploy te
    submissions/nb3242_fixed_q05_q98_nb3090.csv  (only on BETTER)
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

TAG = "nb3242"
PARENT_TAG = "nb3090"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3090.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Fixed (q_low, q_high) -- modal pick from nb3190 (70-72 / 75 folds) -------
Q_LOW = 0.05
Q_HIGH = 0.98

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4424

# -- References ----------------------------------------------------------------
REF_PARENT_NB3090 = 0.4472
REF_NB3190 = 0.4426
REF_NB3173 = 0.4437
REF_NB3170_FIXED = 0.4437
REF_NB3080 = 0.4475
REF_NB3030 = 0.4509
REF_K18 = 0.4536
REF_NB2171 = 0.4682


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Apply fixed (q05, q98) clip per fold at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_lo = []
    fold_hi = []
    fold_clipped_lo = []
    fold_clipped_hi = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        lo = float(np.quantile(y_unb[tr_loc], Q_LOW))
        hi = float(np.quantile(y_unb[tr_loc], Q_HIGH))
        fold_lo.append(lo)
        fold_hi.append(hi)
        val_pred = pred_base[va_loc]
        n_lo = int(np.sum(val_pred < lo))
        n_hi = int(np.sum(val_pred > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)
        clipped = np.clip(val_pred, lo, hi)
        oof_clip[va_loc] = clipped
        fold_val_raes.append(float(rae(y_unb[va_loc], clipped)))

    if np.isnan(oof_clip).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_clip))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "n_clipped_lo": int(np.sum(fold_clipped_lo)),
        "n_clipped_hi": int(np.sum(fold_clipped_hi)),
        "oof": oof_clip,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- FIXED (q{int(Q_LOW*100):02d}, q{int(Q_HIGH*100):02d}) clip "
        f"on {PARENT_TAG} pred_oof (no per-fold grid search)"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER, "
        f"else FAIL"
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
    print(f"STEP 1: load {PARENT_TAG} pred_oof + te")
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
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"per_fold_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"(lo,hi)=({res['fold_lo_mean']:.3f},{res['fold_hi_mean']:.3f})  "
            f"clipped(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    # Per-fold-mean is the GATE metric (per protocol)
    arr_pfm = np.asarray(per_fold_means, dtype=np.float64)
    arr_pooled = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr_pfm)

    # Per-fold-mean stats (the gate metric)
    pfm_mean = float(arr_pfm.mean())
    pfm_std = float(arr_pfm.std(ddof=1)) if n_s > 1 else 0.0
    pfm_sem = pfm_std / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    pfm_ci_low = pfm_mean - t_mult * pfm_sem
    pfm_ci_high = pfm_mean + t_mult * pfm_sem
    pfm_median = float(np.median(arr_pfm))

    # Pooled stats (reference)
    pooled_mean = float(arr_pooled.mean())
    pooled_std = float(arr_pooled.std(ddof=1)) if n_s > 1 else 0.0
    pooled_sem = pooled_std / np.sqrt(n_s) if n_s > 1 else 0.0
    pooled_ci_low = pooled_mean - t_mult * pooled_sem
    pooled_ci_high = pooled_mean + t_mult * pooled_sem
    pooled_median = float(np.median(arr_pooled))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print("   PER-FOLD-MEAN (gate metric):")
    print(f"     mean    = {pfm_mean:.4f}")
    print(f"     std     = {pfm_std:.4f}")
    print(f"     sem     = {pfm_sem:.4f}")
    print(f"     95% CI  = [{pfm_ci_low:.4f}, {pfm_ci_high:.4f}]")
    print(f"     median  = {pfm_median:.4f}")
    print(f"     min/max = [{arr_pfm.min():.4f}, {arr_pfm.max():.4f}]")
    print("   POOLED RAE (reference):")
    print(f"     mean    = {pooled_mean:.4f}")
    print(f"     std     = {pooled_std:.4f}")
    print(f"     95% CI  = [{pooled_ci_low:.4f}, {pooled_ci_high:.4f}]")
    print(f"     median  = {pooled_median:.4f}")
    print(
        f"\n   ref {PARENT_TAG} 15-seed mean     = "
        f"{REF_PARENT_NB3090:.4f}"
    )
    print(
        f"   delta vs {PARENT_TAG}             = "
        f"{pfm_mean - REF_PARENT_NB3090:+.4f}"
    )
    print(
        f"   ref nb3190 modal-grid mean   = {REF_NB3190:.4f}"
    )
    print(
        f"   delta vs nb3190              = "
        f"{pfm_mean - REF_NB3190:+.4f}"
    )

    # -- Deploy: apply fixed (q05, q98) of FULL 253 y to te -------------------
    deploy_lo = float(np.quantile(y_unb, Q_LOW))
    deploy_hi = float(np.quantile(y_unb, Q_HIGH))
    te_pred = np.clip(te_base, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_base < deploy_lo))
    n_te_hi = int(np.sum(te_base > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy clip = (q{Q_LOW:.2f}, q{Q_HIGH:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f}) from full 253 y"
    )
    print(
        f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513  "
        f"total={n_te_lo + n_te_hi}/513"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage (by per-fold-mean ranking)
    med_seed_idx = int(np.argsort(arr_pfm)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"   median seed = {median_seed} "
        f"(per_fold_mean={arr_pfm[med_seed_idx]:.4f})"
    )

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (per-fold-mean)")
    print("-" * 78)
    if pfm_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3242 per-fold-mean {pfm_mean:.4f} beats "
            f"gate {GATE_BETTER:.4f} ({pfm_mean - GATE_BETTER:+.4f}). Fixed "
            f"(q{Q_LOW:.2f}, q{Q_HIGH:.2f}) on nb3090 anchor beats the "
            f"per-fold grid search (nb3190 {REF_NB3190:.4f}). The inner-grid "
            f"DOF was noise -- modal pick {Q_LOW:.2f}/{Q_HIGH:.2f} was correct "
            f"in 70-72/75 folds (>=93%) and a fixed application both lowers "
            f"variance and improves the central estimate. Re-verify with "
            f"deep-30 before PRIMARY-1 swap."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3242 per-fold-mean {pfm_mean:.4f} fails gate "
            f"{GATE_BETTER:.4f} (delta {pfm_mean - GATE_BETTER:+.4f}). Delta vs "
            f"parent nb3090 = {pfm_mean - REF_PARENT_NB3090:+.4f}, delta vs "
            f"nb3190 modal-grid = {pfm_mean - REF_NB3190:+.4f}. The fixed "
            f"modal pick does not beat the learned-clip's per-fold "
            f"adaptation; the small minority of folds where the search picked "
            f"a non-modal value were carrying real signal, or the operator on "
            f"nb3090 is already at its ceiling. Keep nb3090 / nb3190 on the "
            f"ladder."
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

    sub_csv = SUBMISSIONS / f"{TAG}_fixed_q05_q98_nb3090.csv"
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
            "fixed_q05_q98_clip_on_nb3090_pred_oof_no_inner_grid"
        ),
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "q_low_fixed": Q_LOW,
        "q_high_fixed": Q_HIGH,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_metric": {
            "mean": round(pfm_mean, 4),
            "std": round(pfm_std, 4),
            "sem": round(pfm_sem, 4),
            "ci95_low": round(pfm_ci_low, 4),
            "ci95_high": round(pfm_ci_high, 4),
            "median": round(pfm_median, 4),
            "min": round(float(arr_pfm.min()), 4),
            "max": round(float(arr_pfm.max()), 4),
        },
        "pooled_rae_metric": {
            "mean": round(pooled_mean, 4),
            "std": round(pooled_std, 4),
            "sem": round(pooled_sem, 4),
            "ci95_low": round(pooled_ci_low, 4),
            "ci95_high": round(pooled_ci_high, 4),
            "median": round(pooled_median, 4),
            "min": round(float(arr_pooled.min()), 4),
            "max": round(float(arr_pooled.max()), 4),
        },
        "gate_metric_name": "per_fold_mean",
        "mean_rae": round(pfm_mean, 4),
        "std_rae": round(pfm_std, 4),
        "ref_parent_nb3090": REF_PARENT_NB3090,
        "delta_vs_parent": round(pfm_mean - REF_PARENT_NB3090, 4),
        "ref_nb3190": REF_NB3190,
        "delta_vs_nb3190": round(pfm_mean - REF_NB3190, 4),
        "ref_nb3173": REF_NB3173,
        "ref_nb3170_fixed": REF_NB3170_FIXED,
        "ref_nb3080": REF_NB3080,
        "ref_nb3030": REF_NB3030,
        "ref_K18_deep30": REF_K18,
        "ref_nb2171": REF_NB2171,
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (
            str(sub_csv) if verdict == "BETTER" else None
        ),
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
    print(
        f"   per_fold_mean ({n_s} seeds) = {pfm_mean:.4f} +/- {pfm_std:.4f}"
    )
    print(f"   95% CI                = [{pfm_ci_low:.4f}, {pfm_ci_high:.4f}]")
    print(f"   pooled_rae mean       = {pooled_mean:.4f}")
    print(f"   delta vs nb3090       = {pfm_mean - REF_PARENT_NB3090:+.4f}")
    print(f"   delta vs nb3190       = {pfm_mean - REF_NB3190:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae",
        "delta_vs_parent", "delta_vs_nb3190",
        "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
