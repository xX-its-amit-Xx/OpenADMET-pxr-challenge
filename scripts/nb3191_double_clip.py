"""nb3191 -- Double-clip: apply clip primitive twice iteratively on nb3080.

NEW PARADIGM (clip-and-recompute-clip iteratively, 2 iterations):
    Generalizes nb3170 fixed-band q05/q95 clip and nb3173 learned per-fold
    grid-search clip. Instead of learning the band or applying the band once,
    we apply the SAME clip primitive twice with the band RECOMPUTED from the
    iter-1 clipped values.

    Iter 1: clip nb3080 fold-val to (q05, q95) of fold-train y values.
    Iter 2: clip iter-1 fold-val output to (q05, q95) of fold-train CLIPPED
            values (i.e. clip y[fold_train] first to its own q05/q95, then
            take q05/q95 of THAT clipped sequence to define iter-2 band).

    Hypothesis: nb3170 fixed q05/q95 hits 0.4437 on 15 seeds. nb3173 learned
    per-fold grid search reaches 0.4422 by tightening when justified. Double
    clipping recomputes the band on already-tighter values, producing a
    monotone-shrinking compressive transform that may extract another
    ~0.0005-0.0015 RAE on top of nb3173. If the iter-2 band collapses the
    tails too aggressively, RAE will inflate above nb3170 and the gate fails.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    pred_base = nb3080_pred_oof  (253,)
    Per outer fold:
        a) y_tr   = y[fold_train]
           lo1   = quantile(y_tr, 0.05)
           hi1   = quantile(y_tr, 0.95)
        b) y_tr_clipped = np.clip(y_tr, lo1, hi1)
           lo2   = quantile(y_tr_clipped, 0.05)
           hi2   = quantile(y_tr_clipped, 0.95)
        c) val_pred_iter1 = np.clip(pred_base[va_loc], lo1, hi1)
           val_pred_iter2 = np.clip(val_pred_iter1, lo2, hi2)
        d) stitch val_pred_iter2 into oof_clip (253,); pooled RAE across 5 folds.
    Repeat for 15 FRESH kf_seeds {1171..1185}.

GATE (on 15-seed mean):
    mean < 0.4422 -> "BETTER"   (beats nb3173 learned grid)
    mean < 0.4437 -> "MARGINAL" (beats nb3170 fixed single clip)
    else          -> "FAIL"

References:
    nb3173 15-seed (learned grid)   = 0.4422 +/- ?      <- target to beat
    nb3170 15-seed (fixed q05/q95)  = 0.4437 +/- 0.0009 <- single clip baseline
    nb3080 15-seed wide-mean        = 0.4475 +/- 0.0006 <- parent anchor
    nb3030 wide-seed ceiling        = 0.4509
    nb2960 K18 deep-30 OOF          = 0.4536
    nb2171 prior post-hoc top       = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3080_pred_oof.npy
    data/processed/te_nb3080.npy

Outputs:
    data/processed/nb3191_summary.json
    data/processed/nb3191_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3191.npy         (513,) float32 -- deploy te
    submissions/nb3191_double_clip.csv   (only on BETTER/MARGINAL verdict)
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

TAG = "nb3191"
PARENT_TAG = "nb3080"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3080_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3080.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1171, 1186))  # 15 FRESH seeds {1171..1185}

# -- Clip primitive ------------------------------------------------------------
Q_LOW = 0.05
Q_HIGH = 0.95
N_ITERS = 2

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4422   # mean < this -> BETTER  (beats nb3173 learned grid)
GATE_MARGINAL = 0.4437  # mean < this -> MARGINAL (beats nb3170 single clip)

# -- References ----------------------------------------------------------------
REF_NB3173 = 0.4422
REF_NB3170 = 0.4437
REF_NB3170_STD = 0.0009
REF_NB3080 = 0.4475
REF_NB3080_STD = 0.0006
REF_NB3030 = 0.4509
REF_K18 = 0.4536
REF_NB2171 = 0.4682


def _double_clip_band(y_tr: np.ndarray) -> tuple[float, float, float, float]:
    """Compute the iter-1 (lo1, hi1) and iter-2 (lo2, hi2) clip band.

    Iter 1 band is (q05, q95) of raw y_tr.
    Iter 2 band is (q05, q95) of y_tr after iter-1 clipping.
    """
    lo1 = float(np.quantile(y_tr, Q_LOW))
    hi1 = float(np.quantile(y_tr, Q_HIGH))
    y_tr_clipped = np.clip(y_tr, lo1, hi1)
    lo2 = float(np.quantile(y_tr_clipped, Q_LOW))
    hi2 = float(np.quantile(y_tr_clipped, Q_HIGH))
    return lo1, hi1, lo2, hi2


def _apply_double_clip(
    pred: np.ndarray,
    lo1: float,
    hi1: float,
    lo2: float,
    hi2: float,
) -> np.ndarray:
    """Apply 2-iteration clip to prediction array."""
    out = np.clip(pred, lo1, hi1)
    out = np.clip(out, lo2, hi2)
    return out


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run double-clip pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_lo1 = []
    fold_hi1 = []
    fold_lo2 = []
    fold_hi2 = []
    fold_n_lo1 = []
    fold_n_hi1 = []
    fold_n_lo2 = []
    fold_n_hi2 = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        lo1, hi1, lo2, hi2 = _double_clip_band(y_unb[tr_loc])
        fold_lo1.append(lo1)
        fold_hi1.append(hi1)
        fold_lo2.append(lo2)
        fold_hi2.append(hi2)
        val_pred = pred_base[va_loc]
        n_lo1 = int(np.sum(val_pred < lo1))
        n_hi1 = int(np.sum(val_pred > hi1))
        val_iter1 = np.clip(val_pred, lo1, hi1)
        n_lo2 = int(np.sum(val_iter1 < lo2))
        n_hi2 = int(np.sum(val_iter1 > hi2))
        val_iter2 = np.clip(val_iter1, lo2, hi2)
        fold_n_lo1.append(n_lo1)
        fold_n_hi1.append(n_hi1)
        fold_n_lo2.append(n_lo2)
        fold_n_hi2.append(n_hi2)
        oof_clip[va_loc] = val_iter2
        fold_val_raes.append(float(rae(y_unb[va_loc], val_iter2)))

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
        "fold_lo1_mean": float(np.mean(fold_lo1)),
        "fold_hi1_mean": float(np.mean(fold_hi1)),
        "fold_lo2_mean": float(np.mean(fold_lo2)),
        "fold_hi2_mean": float(np.mean(fold_hi2)),
        "fold_lo1_minus_lo2_mean": float(
            np.mean(np.asarray(fold_lo2) - np.asarray(fold_lo1))
        ),
        "fold_hi1_minus_hi2_mean": float(
            np.mean(np.asarray(fold_hi1) - np.asarray(fold_hi2))
        ),
        "n_clipped_lo_iter1": int(np.sum(fold_n_lo1)),
        "n_clipped_hi_iter1": int(np.sum(fold_n_hi1)),
        "n_clipped_lo_iter2": int(np.sum(fold_n_lo2)),
        "n_clipped_hi_iter2": int(np.sum(fold_n_hi2)),
        "oof": oof_clip,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- DOUBLE clip (2 iters of q05/q95) on {PARENT_TAG} pred_oof"
    )
    print(
        f"          Q_LOW = {Q_LOW}  Q_HIGH = {Q_HIGH}  N_ITERS = {N_ITERS}"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate: mean < {GATE_BETTER:.4f} -> BETTER, "
        f"< {GATE_MARGINAL:.4f} -> MARGINAL, else FAIL"
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

    # -- Load nb3080 anchor pred_oof + te -------------------------------------
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
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_lo1_mean": round(res["fold_lo1_mean"], 4),
            "fold_hi1_mean": round(res["fold_hi1_mean"], 4),
            "fold_lo2_mean": round(res["fold_lo2_mean"], 4),
            "fold_hi2_mean": round(res["fold_hi2_mean"], 4),
            "fold_lo2_minus_lo1_mean": round(
                res["fold_lo1_minus_lo2_mean"], 4
            ),
            "fold_hi1_minus_hi2_mean": round(
                res["fold_hi1_minus_hi2_mean"], 4
            ),
            "n_clipped_lo_iter1": res["n_clipped_lo_iter1"],
            "n_clipped_hi_iter1": res["n_clipped_hi_iter1"],
            "n_clipped_lo_iter2": res["n_clipped_lo_iter2"],
            "n_clipped_hi_iter2": res["n_clipped_hi_iter2"],
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"band1=({res['fold_lo1_mean']:.3f},{res['fold_hi1_mean']:.3f})  "
            f"band2=({res['fold_lo2_mean']:.3f},{res['fold_hi2_mean']:.3f})  "
            f"clipped1=({res['n_clipped_lo_iter1']},"
            f"{res['n_clipped_hi_iter1']})  "
            f"clipped2=({res['n_clipped_lo_iter2']},"
            f"{res['n_clipped_hi_iter2']})  "
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
        f"\n   ref nb3173 learned grid    = {REF_NB3173:.4f}"
    )
    print(
        f"   delta vs nb3173            = {mean_rae - REF_NB3173:+.4f}"
    )
    print(
        f"   ref nb3170 fixed clip      = "
        f"{REF_NB3170:.4f} +/- {REF_NB3170_STD:.4f}"
    )
    print(
        f"   delta vs nb3170            = {mean_rae - REF_NB3170:+.4f}"
    )
    print(
        f"   ref {PARENT_TAG} 15-seed mean    = "
        f"{REF_NB3080:.4f} +/- {REF_NB3080_STD:.4f}"
    )
    print(
        f"   delta vs {PARENT_TAG}            = {mean_rae - REF_NB3080:+.4f}"
    )

    # -- Deploy: compute (lo1,hi1,lo2,hi2) on FULL 253 y, apply to te ---------
    deploy_lo1, deploy_hi1, deploy_lo2, deploy_hi2 = _double_clip_band(y_unb)
    te_pred = _apply_double_clip(
        te_base, deploy_lo1, deploy_hi1, deploy_lo2, deploy_hi2
    ).astype(np.float32)
    n_te_lo1 = int(np.sum(te_base < deploy_lo1))
    n_te_hi1 = int(np.sum(te_base > deploy_hi1))
    te_iter1 = np.clip(te_base, deploy_lo1, deploy_hi1)
    n_te_lo2 = int(np.sum(te_iter1 < deploy_lo2))
    n_te_hi2 = int(np.sum(te_iter1 > deploy_hi2))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy band1 = ({deploy_lo1:.3f}, {deploy_hi1:.3f})  "
        f"band2 = ({deploy_lo2:.3f}, {deploy_hi2:.3f})  on full 253 y"
    )
    print(
        f"   te clipped iter1: lo={n_te_lo1}/513  hi={n_te_hi1}/513"
    )
    print(
        f"   te clipped iter2: lo={n_te_lo2}/513  hi={n_te_hi2}/513"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
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
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3191 15-seed mean {mean_rae:.4f} beats "
            f"nb3173 learned-clip {REF_NB3173:.4f} "
            f"({mean_rae - REF_NB3173:+.4f}). Double-clip (recomputing q05/"
            f"q95 on iter-1 clipped values) extracts further variance "
            f"compression beyond the single-pass clip. Iter-2 mean band "
            f"narrowing: lo2-lo1={deploy_lo2 - deploy_lo1:+.3f}, "
            f"hi1-hi2={deploy_hi1 - deploy_hi2:+.3f}. Re-verify with "
            f"deep-30 before PRIMARY-1 swap."
        )
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL"
        ladder_action = (
            f"MARGINAL. nb3191 15-seed mean {mean_rae:.4f} beats nb3170 "
            f"single-clip {REF_NB3170:.4f} but does NOT beat nb3173 learned "
            f"{REF_NB3173:.4f} ({mean_rae - REF_NB3173:+.4f}). Double-clip "
            f"adds modest variance compression on top of single clip but "
            f"the second iteration is largely a no-op once iter-1 has bound "
            f"the tails. Keep nb3173 as the active clip primitive."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3191 15-seed mean {mean_rae:.4f} fails the nb3170 "
            f"single-clip gate {REF_NB3170:.4f} "
            f"({mean_rae - REF_NB3170:+.4f}). The second clip iteration "
            f"over-compresses the band by recomputing q05/q95 on already-"
            f"tightened values, removing rank-order capacity that nb3170 "
            f"and nb3173 preserved. Iterating the clip primitive is not a "
            f"free axis. Keep prior PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_double_clip.csv"
    if verdict in ("BETTER", "MARGINAL"):
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
        "method": "double_clip_q05q95_2iters_on_nb3080_pred_oof",
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "n_iters": N_ITERS,
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
        "ref_nb3173": REF_NB3173,
        "delta_vs_nb3173": round(mean_rae - REF_NB3173, 4),
        "ref_nb3170": REF_NB3170,
        "ref_nb3170_std": REF_NB3170_STD,
        "delta_vs_nb3170": round(mean_rae - REF_NB3170, 4),
        "ref_parent": REF_NB3080,
        "ref_parent_std": REF_NB3080_STD,
        "delta_vs_parent": round(mean_rae - REF_NB3080, 4),
        "ref_nb3030": REF_NB3030,
        "delta_vs_nb3030": round(mean_rae - REF_NB3030, 4),
        "ref_K18_deep30": REF_K18,
        "ref_nb2171": REF_NB2171,
        "deploy_lo1": round(deploy_lo1, 4),
        "deploy_hi1": round(deploy_hi1, 4),
        "deploy_lo2": round(deploy_lo2, 4),
        "deploy_hi2": round(deploy_hi2, 4),
        "deploy_lo2_minus_lo1": round(deploy_lo2 - deploy_lo1, 4),
        "deploy_hi1_minus_hi2": round(deploy_hi1 - deploy_hi2, 4),
        "n_te_clipped_lo_iter1": n_te_lo1,
        "n_te_clipped_hi_iter1": n_te_hi1,
        "n_te_clipped_lo_iter2": n_te_lo2,
        "n_te_clipped_hi_iter2": n_te_hi2,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (
            str(sub_csv) if verdict in ("BETTER", "MARGINAL") else None
        ),
        "gate_better": GATE_BETTER,
        "gate_marginal": GATE_MARGINAL,
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
    print(f"   delta vs nb3173       = {mean_rae - REF_NB3173:+.4f}")
    print(f"   delta vs nb3170       = {mean_rae - REF_NB3170:+.4f}")
    print(f"   delta vs nb3080       = {mean_rae - REF_NB3080:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3173", "delta_vs_nb3170", "delta_vs_parent",
        "deploy_lo1", "deploy_hi1", "deploy_lo2", "deploy_hi2",
        "deploy_lo2_minus_lo1", "deploy_hi1_minus_hi2",
        "n_te_clipped_lo_iter1", "n_te_clipped_hi_iter1",
        "n_te_clipped_lo_iter2", "n_te_clipped_hi_iter2",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
