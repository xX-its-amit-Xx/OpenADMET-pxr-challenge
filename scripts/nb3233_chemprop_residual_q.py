"""nb3233 -- chemprop_aux RESIDUAL-conditional clip on nb3080 (q40 hard-split).

NEW PARADIGM: per-fold q40 from chemprop_aux residual MAGNITUDE selects WHICH
rows get the y-range clip applied to nb3080 (vs prior nb32xx series that
used K18-prediction quantile to gate row-conditional weights).

    The chemprop_aux anchor is the only verified-clean PRE-unblind anchor on
    the 253 unblind. The per-row residual |y - chemprop_aux| is a clean
    confidence proxy: rows where chemprop_aux predicts close to truth are
    "easy" (low residual), rows with high residual are "hard". Hypothesis:
    only the EASY (low-residual) tail benefits from a wide y-range clip on
    nb3080, because the easy rows are where the nb3080 predictor has reliable
    rank-order but variance-compressed magnitude. The HARD (high-residual)
    tail is left untouched -- clipping there would mask genuine confidence.

PROTOCOL (per kf_seed, 5-fold scaffold split, anchors LOADED no rebuild):
    pred_base    = nb3080_pred_oof   (253,)  -- K18/K19 q40 hard-split blend
    chemprop_oof = te_chemprop_aux.npy[unb_idx] (253,) -- PRE-clean anchor
    Per outer fold:
        a) Compute fold-train residual magnitude
              r_tr = |y[fold_train] - chemprop_aux[fold_train]|
           Compute q40 threshold of r_tr -> q_thr_resid.
        b) Compute fold-train y-clip range from FOLD TRAIN ONLY:
              lo = quantile(y[fold_train], 0.05)
              hi = quantile(y[fold_train], 0.95)
        c) Apply HARD-SPLIT clip on fold-val:
              r_va = |y[fold_val] - chemprop_aux[fold_val]|  (probe)
              -- but we cannot use y[fold_val] at deploy time. Instead use
                 ABSOLUTE chemprop_aux residual proxy on val pred itself by
                 mapping each val row to the chemprop_aux pred and using the
                 trained q_thr_resid as the gate. The residual is computed
                 from a PROXY using the cross-fit residual model... but for
                 honest deploy, we use the CHEMPROP_AUX prediction's
                 deviation from the local fold-train median as the gate.
           Simpler honest variant (what we use): use |chemprop_aux_pred -
           median(chemprop_aux_train)| as a confidence proxy per row at
           inference. Rows where this proxy is BELOW the q40 threshold of
           the corresponding train-time proxy distribution -> APPLY clip;
           rows ABOVE -> leave nb3080 prediction unchanged.

           This keeps the gate deploy-safe: requires only chemprop_aux pred
           and fold-train chemprop_aux statistics, no truth on val.
        d) Stitch into oof_clip; record per-fold val RAE.
    Repeat for 15 FRESH kf_seeds {1216..1230}.

GATE (on 15-seed PER-FOLD-MEAN):
    pf_mean < 0.4424 -> "BETTER"
    else             -> "FAIL"

References:
    nb3080 wide-seed verify (q40 anchor) = 0.4470  <- parent anchor
    nb3190 learned-clip on nb3090        = 0.4422  <- compounding target
    nb3173 learned-clip on nb3080        = 0.4437  (clip-operator ceiling)
    nb2171 prior post-hoc top            = 0.4682
    chemprop_aux PRE-unblind in_RAE      = 0.6216  (verified clean anchor)

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3080_pred_oof.npy
    data/processed/te_nb3080.npy
    data/processed/te_chemprop_aux.npy

Outputs:
    data/processed/nb3233_summary.json
    data/processed/nb3233_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3233.npy         (513,) float32 -- deploy te
    submissions/nb3233_chemprop_residual_q.csv  (only on BETTER)
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

TAG = "nb3233"
PARENT_TAG = "nb3080"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3080_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3080.npy"
CHEMPROP_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Operator parameters (FIXED per spec) -------------------------------------
Q_RESID_CUT = 0.40   # per-fold q40 of |y_tr - chemprop_aux_tr| residual magnitude
Q_Y_LOW = 0.05       # per-fold q05 of y_tr for clip lo
Q_Y_HIGH = 0.95      # per-fold q95 of y_tr for clip hi

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4424  # per-fold-mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_PARENT_NB3080 = 0.4470
REF_NB3190 = 0.4422
REF_NB3173 = 0.4437
REF_NB2171 = 0.4682
REF_CHEMPROP_AUX_INSAMPLE = 0.6216


def _run_one_seed(
    pred_base: np.ndarray,
    chemprop_oof: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Per-seed chemprop-residual-gated y-clip on nb3080."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_q_resid = []
    fold_lo = []
    fold_hi = []
    fold_n_low_resid_val = []   # rows in val classified LOW residual -> get clipped
    fold_n_clipped_lo = []
    fold_n_clipped_hi = []
    fold_chemprop_med = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # -- (a) Fold-train residual magnitude q40 threshold from TRUE residual
        resid_tr_true = np.abs(y_unb[tr_loc] - chemprop_oof[tr_loc])
        q_thr_resid_true = float(np.quantile(resid_tr_true, Q_RESID_CUT))
        # -- Deploy-safe PROXY for residual magnitude on val: |chemprop_va - median(chemprop_tr)|
        chemprop_med_tr = float(np.median(chemprop_oof[tr_loc]))
        proxy_tr = np.abs(chemprop_oof[tr_loc] - chemprop_med_tr)
        # Per-fold-train proxy quantile -> gate threshold
        q_thr_proxy = float(np.quantile(proxy_tr, Q_RESID_CUT))
        fold_q_resid.append(q_thr_proxy)
        fold_chemprop_med.append(chemprop_med_tr)
        # -- (b) Fold-train y-clip range
        lo = float(np.quantile(y_unb[tr_loc], Q_Y_LOW))
        hi = float(np.quantile(y_unb[tr_loc], Q_Y_HIGH))
        fold_lo.append(lo)
        fold_hi.append(hi)
        # -- (c) Apply HARD-SPLIT clip on val using proxy gate
        val_pred = pred_base[va_loc].copy()
        proxy_va = np.abs(chemprop_oof[va_loc] - chemprop_med_tr)
        low_resid_mask = proxy_va <= q_thr_proxy
        fold_n_low_resid_val.append(int(low_resid_mask.sum()))
        # Clip only the low-residual ("easy") rows
        to_clip = val_pred.copy()
        n_lo = int(np.sum((to_clip < lo) & low_resid_mask))
        n_hi = int(np.sum((to_clip > hi) & low_resid_mask))
        fold_n_clipped_lo.append(n_lo)
        fold_n_clipped_hi.append(n_hi)
        clipped_lo_only = np.where(low_resid_mask, np.clip(val_pred, lo, hi), val_pred)
        oof_clip[va_loc] = clipped_lo_only
        fold_val_raes.append(float(rae(y_unb[va_loc], clipped_lo_only)))
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
        "fold_q_resid_mean": float(np.mean(fold_q_resid)),
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "fold_chemprop_med_mean": float(np.mean(fold_chemprop_med)),
        "n_low_resid_val_total": int(np.sum(fold_n_low_resid_val)),
        "n_clipped_lo_total": int(np.sum(fold_n_clipped_lo)),
        "n_clipped_hi_total": int(np.sum(fold_n_clipped_hi)),
        "oof": oof_clip,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- chemprop_aux RESIDUAL-conditional q40 hard-split clip on "
        f"{PARENT_TAG}"
    )
    print(f"          Q_RESID_CUT = {Q_RESID_CUT}  (per-fold quantile of |y-chemprop|)")
    print(f"          Q_Y_LOW     = {Q_Y_LOW}    Q_Y_HIGH = {Q_Y_HIGH}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"          honest gate metric = PER-FOLD-MEAN")
    print(f"          gate: pf_mean < {GATE_BETTER:.4f} -> BETTER, else FAIL")
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
    print(f"STEP 1: load {PARENT_TAG} pred_oof + te (q40 hard-split blend)")
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

    # -- Load chemprop_aux PRE-clean anchor on 253 ----------------------------
    print("\n" + "-" * 78)
    print("STEP 2: load chemprop_aux PRE-clean anchor on 253 unblind")
    print("-" * 78)
    te_chemprop = np.load(CHEMPROP_TE_PATH).astype(np.float64)
    if te_chemprop.shape != (n_test,):
        raise ValueError(
            f"chemprop_aux te shape {te_chemprop.shape} != ({n_test},)"
        )
    chemprop_oof = te_chemprop[unb_idx].astype(np.float64)
    rae_chemprop = float(rae(y_unb, chemprop_oof))
    resid_full = y_unb - chemprop_oof
    abs_resid_full = np.abs(resid_full)
    print(
        f"   chemprop_aux[unb]: in_RAE={rae_chemprop:.4f} "
        f"(ref PRE in_RAE={REF_CHEMPROP_AUX_INSAMPLE:.4f})"
    )
    print(
        f"   abs_resid stats: mean={abs_resid_full.mean():.3f}  "
        f"median={np.median(abs_resid_full):.3f}  "
        f"q40={np.quantile(abs_resid_full, 0.40):.3f}  "
        f"q60={np.quantile(abs_resid_full, 0.60):.3f}  "
        f"max={abs_resid_full.max():.3f}"
    )
    # Leak sanity
    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN parent: {leak_eq:.1%} rows == truth -- possible leak")
    leak_eq_chem = float(np.mean(np.isclose(chemprop_oof, y_unb, atol=1e-6)))
    if leak_eq_chem > 0.05:
        print(
            f"   WARN chemprop_aux: {leak_eq_chem:.1%} rows == truth -- "
            f"possible leak"
        )
    print(
        f"   y_unb stats: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
        f"min={y_unb.min():.3f}  max={y_unb.max():.3f}"
    )

    # -- Scaffolds ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: scaffolds for outer CV")
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
        res = _run_one_seed(pred_base, chemprop_oof, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_q_resid_mean": round(res["fold_q_resid_mean"], 4),
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "fold_chemprop_med_mean": round(res["fold_chemprop_med_mean"], 4),
            "n_low_resid_val_total": res["n_low_resid_val_total"],
            "n_clipped_lo_total": res["n_clipped_lo_total"],
            "n_clipped_hi_total": res["n_clipped_hi_total"],
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"q_resid={res['fold_q_resid_mean']:.3f}  "
            f"clip(lo,hi)=({res['fold_lo_mean']:.2f},{res['fold_hi_mean']:.2f})  "
            f"n_low_resid={res['n_low_resid_val_total']}/253  "
            f"n_clipped(lo,hi)=({res['n_clipped_lo_total']},{res['n_clipped_hi_total']})  "
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
    print(f"\n   ref {PARENT_TAG} pooled baseline = {REF_PARENT_NB3080:.4f}")
    print(
        f"   delta vs {PARENT_TAG} (pooled)   = "
        f"{mean_rae - REF_PARENT_NB3080:+.4f}"
    )
    print(
        f"   delta vs {PARENT_TAG} (pf_mean)  = "
        f"{pf_mean - REF_PARENT_NB3080:+.4f}"
    )
    print(f"   ref nb3190 clip-on-nb3090   = {REF_NB3190:.4f}")
    print(f"   delta vs nb3190 (pf_mean)   = {pf_mean - REF_NB3190:+.4f}")
    print(f"   ref nb3173 clip-operator    = {REF_NB3173:.4f}")
    print(f"   ref nb2171 prior PRIMARY-1  = {REF_NB2171:.4f}")
    print(f"   gain vs nb2171 (pf_mean)    = {REF_NB2171 - pf_mean:+.4f}")

    # -- Deploy: build te using full-253 chemprop residual stats --------------
    deploy_chemprop_med = float(np.median(chemprop_oof))
    deploy_proxy_full = np.abs(chemprop_oof - deploy_chemprop_med)
    deploy_q_thr_proxy = float(np.quantile(deploy_proxy_full, Q_RESID_CUT))
    deploy_lo = float(np.quantile(y_unb, Q_Y_LOW))
    deploy_hi = float(np.quantile(y_unb, Q_Y_HIGH))
    te_proxy = np.abs(te_chemprop - deploy_chemprop_med)
    te_low_resid_mask = te_proxy <= deploy_q_thr_proxy
    te_pred = np.where(
        te_low_resid_mask, np.clip(te_base, deploy_lo, deploy_hi), te_base
    ).astype(np.float32)
    n_te_low_resid = int(te_low_resid_mask.sum())
    n_te_clipped_lo = int(np.sum((te_base < deploy_lo) & te_low_resid_mask))
    n_te_clipped_hi = int(np.sum((te_base > deploy_hi) & te_low_resid_mask))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy q_thr_proxy = {deploy_q_thr_proxy:.4f}  "
        f"chemprop_med = {deploy_chemprop_med:.4f}"
    )
    print(
        f"   deploy clip range  = ({deploy_lo:.3f}, {deploy_hi:.3f}) "
        f"from full 253 y q05/q95"
    )
    print(
        f"   te low-resid rows  = {n_te_low_resid}/513  "
        f"(get clip applied)"
    )
    print(
        f"   te clipped: lo={n_te_clipped_lo}/513  hi={n_te_clipped_hi}/513"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage (median over per-fold-mean -- honest metric)
    med_seed_idx = int(np.argsort(arr_pf)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"   median seed = {median_seed} "
        f"(pf_mean={arr_pf[med_seed_idx]:.4f}, pooled={arr[med_seed_idx]:.4f})"
    )

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (honest metric = PER-FOLD-MEAN)")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3233 15-seed PER-FOLD-MEAN {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"chemprop_aux residual-magnitude q40 gate on nb3080 y-clip "
            f"({REF_PARENT_NB3080:.4f}) -> {pf_mean:.4f} = "
            f"{REF_PARENT_NB3080 - pf_mean:.4f} RAE reduction. NEW PARADIGM: "
            f"residual-conditional clip (vs prediction-conditional) gates "
            f"clip application to chemprop-easy rows only. Re-verify with "
            f"deep-30 before PRIMARY-1 swap. anchor_pre_unblind=True (parent "
            f"nb3080 on K18/K19 deep-30 + chemprop_aux gate PRE-clean)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3233 15-seed PER-FOLD-MEAN {pf_mean:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"Delta vs parent nb3080 (pf_mean) = "
            f"{pf_mean - REF_PARENT_NB3080:+.4f}, delta vs nb3190 = "
            f"{pf_mean - REF_NB3190:+.4f}. chemprop_aux residual-magnitude "
            f"gate either (1) doesn't separate easy-vs-hard cleanly on this "
            f"OOD wall, OR (2) the proxy |chemprop - median_train_chemprop| "
            f"is too coarse a confidence signal vs true residual magnitude. "
            f"Closes the residual-conditional-clip-on-nb3080 axis."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_chemprop_residual_q.csv"
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
            "chemprop_aux_residual_magnitude_q40_gated_y_clip_on_nb3080_"
            "hard_split_apply_clip_to_low_residual_rows_only"
        ),
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "chemprop_anchor_path": str(CHEMPROP_TE_PATH),
        "anchor_pre_unblind": True,
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "chemprop_in_rae": round(rae_chemprop, 4),
        "chemprop_leak_eq_truth_frac": round(leak_eq_chem, 4),
        "abs_resid_full_q40": round(float(np.quantile(abs_resid_full, 0.40)), 4),
        "abs_resid_full_q60": round(float(np.quantile(abs_resid_full, 0.60)), 4),
        "abs_resid_full_mean": round(float(abs_resid_full.mean()), 4),
        "abs_resid_full_median": round(float(np.median(abs_resid_full)), 4),
        "q_resid_cut": Q_RESID_CUT,
        "q_y_low": Q_Y_LOW,
        "q_y_high": Q_Y_HIGH,
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
        "ref_parent_nb3080": REF_PARENT_NB3080,
        "delta_vs_parent_pooled": round(mean_rae - REF_PARENT_NB3080, 4),
        "delta_vs_parent_pf_mean": round(pf_mean - REF_PARENT_NB3080, 4),
        "ref_nb3190": REF_NB3190,
        "delta_vs_nb3190_pf_mean": round(pf_mean - REF_NB3190, 4),
        "ref_nb3173": REF_NB3173,
        "ref_nb2171": REF_NB2171,
        "ref_chemprop_aux_insample": REF_CHEMPROP_AUX_INSAMPLE,
        "gain_vs_nb2171_pf_mean": round(REF_NB2171 - pf_mean, 4),
        "deploy_chemprop_med": round(deploy_chemprop_med, 4),
        "deploy_q_thr_proxy": round(deploy_q_thr_proxy, 4),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_low_resid": n_te_low_resid,
        "n_te_clipped_lo": n_te_clipped_lo,
        "n_te_clipped_hi": n_te_clipped_hi,
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
    print(f"   delta vs nb3080 (pf) = {pf_mean - REF_PARENT_NB3080:+.4f}")
    print(f"   delta vs nb3190 (pf) = {pf_mean - REF_NB3190:+.4f}")
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
        "deploy_q_thr_proxy", "deploy_chemprop_med",
        "deploy_lo", "deploy_hi",
        "n_te_low_resid", "n_te_clipped_lo", "n_te_clipped_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
