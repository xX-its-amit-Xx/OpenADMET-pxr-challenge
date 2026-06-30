"""nb3303 -- Per-scaffold-frequency-bin residual correction on nb3200 anchor.

NEW PARADIGM (substrate change, not operator change):
    The dominant Phase-1 failure mode (pm06 / feedback_phase2_prescriptions) is
    F2: novel-scaffold INACTIVES over-predicted (+1.23 RAE prize on the tail).
    78% of the 253 unblind rows (197/253) are novel-scaffold (scaf_train_freq==0).
    If the anchor carries a SYSTEMATIC per-bin residual shift (e.g. over-predicts
    novel-scaffold rows but is calibrated on common-scaffold rows), a per-bin
    additive correction learned ON FOLD-TRAIN and applied to FOLD-VAL can remove
    that bias honestly -- this is a scaffold-frequency-conditioned mean-shift, a
    different axis than the post-hoc clip/stretch operators (nb3173/nb3200 ceiling)
    which act on the global value distribution, not on scaffold-support strata.

    Scaffold-train-frequency = how many of the 4139 TRAIN compounds share each
    test compound's Bemis-Murcko scaffold. Bins:
        novel  : freq == 0   (197/253 unblind)  <- F2 tail lives here
        rare   : freq 1-2     (26/253)
        common : freq >= 3    (30/253)

PROTOCOL (mirror nb3200 deep-seed structure):
    anchor = nb3200 pred_oof (253,) / te_nb3200 (513,)  [PRE-unblind clean, RAE 0.4416]
    For each kf_seed in {1216..1230} (15 FRESH seeds, disjoint from all prior):
        scaffold_kfold_indices(n_splits=5, shuffle=True, seed=kf_seed)
        For each fold:
            a) On FOLD-TRAIN: per-bin residual mean d_b = mean(y - anchor) over
               rows in bin b. James-Stein shrink each d_b toward the global
               fold-train residual mean d_global:
                   shrink_b = 1 - (K-3) * sigma2_b / (n_b * (d_b - d_global)^2)
               (clamped to [0,1]); d_b_shrunk = d_global + shrink_b*(d_b - d_global)
            b) Apply per-bin shift to FOLD-VAL rows by their OWN bin:
                   corrected[val_row] = anchor[val_row] + d_b_shrunk[bin(val_row)]
        per-fold RAE on (y_val, corrected_val); per-fold-mean = mean over 5 folds.
    Aggregate per-fold-mean across 15 seeds (mean +/- std).

GATE (per task):
    per-fold-mean < 0.4423 -> "BETTER"
    else                    -> "FAIL"

DEPLOY:
    Fit per-bin shrunk shifts on FULL 253 (y - anchor) and apply to the 513-test
    anchor te_nb3200 by each test compound's scaffold-train-frequency bin.

Inputs:
    data/processed/_audit_unblind_idx.npy   (253,) int   -> rows into 513-test
    data/processed/_audit_unblind_y.npy     (253,) float -> truth
    data/processed/nb3200_pred_oof.npy       (253,) float -> anchor OOF
    data/processed/te_nb3200.npy             (513,) float -> anchor deploy te

Outputs:
    data/processed/nb3303_summary.json
    data/processed/nb3303_pred_oof.npy   (253,) float32 -- median-seed corrected OOF
    data/processed/te_nb3303.npy         (513,) float32 -- deploy corrected te
    submissions/nb3303_scaffold_bin_residual.csv  (only on BETTER)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3303"
ANCHOR_TAG = "nb3200"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / f"{ANCHOR_TAG}_pred_oof.npy"
TE_PATH = DATA_PROCESSED / f"te_{ANCHOR_TAG}.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Bin definition ------------------------------------------------------------
# bin id: 0 = novel (freq==0), 1 = rare (1-2), 2 = common (3+)
BIN_NAMES = {0: "novel(freq==0)", 1: "rare(1-2)", 2: "common(3+)"}
N_BINS = 3


def _scaf_freq_bin(freq: int) -> int:
    """Map scaffold-train-frequency to bin id."""
    if freq == 0:
        return 0
    if freq <= 2:
        return 1
    return 2


# -- Gates (per task) ----------------------------------------------------------
GATE_BETTER = 0.4423  # per-fold-mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_ANCHOR_NB3200 = 0.4416  # nb3200 pred_oof full RAE
REF_NB3173 = 0.4437  # prior learned-clip ceiling
REF_NB2171 = 0.4682  # post-hoc-blend ceiling on nb730 anchor


def _james_stein_bin_shifts(
    resid_tr: np.ndarray,
    bins_tr: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Per-bin residual mean, James-Stein shrunk toward the global residual mean.

    James-Stein positive-part estimator per bin:
        d_b_shrunk = d_global + max(0, 1 - (K-3)*s2_b / (n_b*(d_b-d_global)^2)) * (d_b - d_global)
    where s2_b is the within-bin variance of residuals, n_b the bin count, and
    K=N_BINS. Shrinks unreliable (high-variance / low-n / near-global) bin means
    back to the global residual; preserves large, well-supported deviations
    (the F2 novel-scaffold over-prediction bias).

    Returns shifts[N_BINS] and a diagnostics dict.
    """
    d_global = float(resid_tr.mean())
    shifts = np.full(N_BINS, d_global, dtype=np.float64)
    diag = {}
    for b in range(N_BINS):
        m = bins_tr == b
        n_b = int(m.sum())
        if n_b == 0:
            diag[b] = {"n": 0, "d_raw": None, "shrink": None, "d_shrunk": d_global}
            continue
        rb = resid_tr[m]
        d_b = float(rb.mean())
        # within-bin variance (ddof=1 if possible, else 0)
        s2_b = float(rb.var(ddof=1)) if n_b > 1 else 0.0
        gap = d_b - d_global
        if N_BINS > 3 and abs(gap) > 1e-12 and n_b > 0:
            shrink = 1.0 - (N_BINS - 3) * s2_b / (n_b * gap * gap)
        else:
            # K=3 -> (K-3)=0 -> classic JS shrink factor vanishes; fall back to
            # an empirical-Bayes shrink that still pulls low-n/high-var bins in:
            #   shrink = gap^2 / (gap^2 + s2_b/n_b)   (reliability weight)
            denom = gap * gap + (s2_b / n_b if n_b > 0 else 0.0)
            shrink = (gap * gap) / denom if denom > 1e-12 else 0.0
        shrink = float(np.clip(shrink, 0.0, 1.0))
        d_shrunk = d_global + shrink * gap
        shifts[b] = d_shrunk
        diag[b] = {
            "n": n_b,
            "d_raw": round(d_b, 4),
            "s2": round(s2_b, 4),
            "shrink": round(shrink, 4),
            "d_shrunk": round(d_shrunk, 4),
        }
    diag["global"] = round(d_global, 4)
    return shifts, diag


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    bins_unb: np.ndarray,
    kf_seed: int,
) -> dict:
    """Per-bin residual correction at a single kf_seed; returns per-fold stats."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_corr = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_base_raes = []
    fold_shift_records = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        resid_tr = y_unb[tr_loc] - pred_base[tr_loc]
        bins_tr = bins_unb[tr_loc]
        shifts, diag = _james_stein_bin_shifts(resid_tr, bins_tr)
        # apply per-val-row by its own bin
        bins_va = bins_unb[va_loc]
        corrected = pred_base[va_loc] + shifts[bins_va]
        oof_corr[va_loc] = corrected
        fold_val_raes.append(float(rae(y_unb[va_loc], corrected)))
        fold_base_raes.append(float(rae(y_unb[va_loc], pred_base[va_loc])))
        fold_shift_records.append({
            "fold": fold_i,
            "shifts": [round(float(s), 4) for s in shifts],
            "diag": diag,
        })

    if np.isnan(oof_corr).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_corr))
    per_fold_mean = float(np.mean(fold_val_raes))
    per_fold_base_mean = float(np.mean(fold_base_raes))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_mean": per_fold_mean,
        "per_fold_std": float(np.std(fold_val_raes, ddof=1)),
        "per_fold_base_mean": per_fold_base_mean,
        "fold_val_raes": [round(v, 4) for v in fold_val_raes],
        "fold_shift_records": fold_shift_records,
        "oof": oof_corr,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PER-SCAFFOLD-FREQ-BIN RESIDUAL CORRECTION on {ANCHOR_TAG}")
    print(f"          bins: novel(freq==0) / rare(1-2) / common(3+)")
    print(f"          per-fold-train JamesStein-shrunk residual mean per bin")
    print(f"          kf_seeds = {len(KF_SEEDS)} fresh "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          GATE: per-fold-mean < {GATE_BETTER:.4f} -> BETTER, else FAIL")
    print("=" * 78)

    # -- Load test, train, truth ---------------------------------------------
    te = load_test()
    tr = load_train()
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
    print(f"[load] n_test={n_test}  n_train={len(tr)}  n_unb={n_unb}")

    # -- Load nb3200 anchor ---------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {ANCHOR_TAG} anchor pred_oof + te")
    print("-" * 78)
    pred_base = np.load(PRED_OOF_PATH).astype(np.float64)
    te_base = np.load(TE_PATH).astype(np.float64)
    if pred_base.shape != (n_unb,):
        raise ValueError(f"{ANCHOR_TAG} pred_oof shape {pred_base.shape} != ({n_unb},)")
    if te_base.shape != (n_test,):
        raise ValueError(f"{ANCHOR_TAG} te shape {te_base.shape} != ({n_test},)")
    full_oof_rae = float(rae(y_unb, pred_base))
    print(f"   anchor pred_oof: RAE={full_oof_rae:.4f}  "
          f"mean={pred_base.mean():.3f} std={pred_base.std():.3f}")
    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN anchor: {leak_eq:.1%} rows == truth -- possible leak")

    # -- Scaffolds + frequency bins ------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: Bemis-Murcko scaffolds + train-frequency bins")
    print("-" * 78)
    tr_scaffolds = [bemis_murcko(s) or "" for s in tr["smiles"].tolist()]
    te_scaffolds_full = [bemis_murcko(s) or "" for s in te_smiles]
    tr_counts = Counter([s for s in tr_scaffolds if s])

    unb_scaffolds = [te_scaffolds_full[i] for i in unb_idx]
    unb_freq = np.array(
        [tr_counts.get(s, 0) if s else 0 for s in unb_scaffolds], dtype=np.int64
    )
    bins_unb = np.array([_scaf_freq_bin(f) for f in unb_freq], dtype=np.int64)

    te_freq_full = np.array(
        [tr_counts.get(s, 0) if s else 0 for s in te_scaffolds_full], dtype=np.int64
    )
    bins_te = np.array([_scaf_freq_bin(f) for f in te_freq_full], dtype=np.int64)

    n_unique_scaf = len({s for s in unb_scaffolds if s})
    bin_counts_unb = {BIN_NAMES[b]: int((bins_unb == b).sum()) for b in range(N_BINS)}
    bin_counts_te = {BIN_NAMES[b]: int((bins_te == b).sum()) for b in range(N_BINS)}
    print(f"   n_unique_scaffolds(unb) = {n_unique_scaf}")
    print(f"   unb 253 bin counts  = {bin_counts_unb}")
    print(f"   test 513 bin counts = {bin_counts_te}")

    # Per-bin anchor bias on FULL 253 (diagnostic of F2 over-prediction)
    print("\n   per-bin anchor residual (y - anchor) on full 253:")
    for b in range(N_BINS):
        m = bins_unb == b
        if m.sum() == 0:
            continue
        r = y_unb[m] - pred_base[m]
        rb = float(rae(y_unb[m], pred_base[m])) if m.sum() > 1 else np.nan
        print(f"     {BIN_NAMES[b]:16s} n={int(m.sum()):3d}  "
              f"mean_resid={r.mean():+.4f}  (neg => anchor OVER-predicts)  "
              f"bin_RAE={rb:.4f}")

    # -- Multi-seed sweep -----------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print("-" * 78)
    seed_records = []
    per_fold_means = []
    pooled_raes = []
    per_fold_base_means = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, bins_unb, s)
        per_fold_means.append(res["per_fold_mean"])
        pooled_raes.append(res["pooled_rae"])
        per_fold_base_means.append(res["per_fold_base_mean"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_mean": round(res["per_fold_mean"], 4),
            "per_fold_std": round(res["per_fold_std"], 4),
            "per_fold_base_mean": round(res["per_fold_base_mean"], 4),
            "fold_val_raes": res["fold_val_raes"],
        })
        print(f"   kf={s}: per_fold_mean={res['per_fold_mean']:.4f}  "
              f"(base {res['per_fold_base_mean']:.4f})  "
              f"pooled={res['pooled_rae']:.4f}  wall={time.time()-ts:.2f}s")

    pfm = np.asarray(per_fold_means, dtype=np.float64)
    base_pfm = np.asarray(per_fold_base_means, dtype=np.float64)
    n_s = len(pfm)
    mean_pfm = float(pfm.mean())
    std_pfm = float(pfm.std(ddof=1)) if n_s > 1 else 0.0
    sem_pfm = std_pfm / np.sqrt(n_s) if n_s > 1 else 0.0
    # df=14 two-sided 95% t = 2.1448
    t_mult = 2.1448
    ci_low = mean_pfm - t_mult * sem_pfm
    ci_high = mean_pfm + t_mult * sem_pfm
    median_pfm = float(np.median(pfm))
    mean_base_pfm = float(base_pfm.mean())
    delta_vs_base = mean_pfm - mean_base_pfm

    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_pooled = float(pooled_arr.mean())

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds, per-fold-mean)")
    print("-" * 78)
    print(f"   per-fold-mean      = {mean_pfm:.4f} +/- {std_pfm:.4f}")
    print(f"   sem                = {sem_pfm:.4f}")
    print(f"   95% CI (df=14)     = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median             = {median_pfm:.4f}")
    print(f"   min/max            = [{pfm.min():.4f}, {pfm.max():.4f}]")
    print(f"   base per-fold-mean = {mean_base_pfm:.4f}  (uncorrected anchor)")
    print(f"   delta vs base      = {delta_vs_base:+.4f}  (neg => correction helps)")
    print(f"   mean pooled RAE    = {mean_pooled:.4f}")
    print(f"\n   ref anchor nb3200  = {REF_ANCHOR_NB3200:.4f}")
    print(f"   ref nb3173 ceiling = {REF_NB3173:.4f}")
    print(f"   gate BETTER        = {GATE_BETTER:.4f}")

    # -- Deploy: fit per-bin shrunk shifts on FULL 253, apply to 513 te -------
    print("\n" + "-" * 78)
    print("DEPLOY: fit per-bin shifts on full 253; apply to 513-test anchor")
    print("-" * 78)
    resid_full = y_unb - pred_base
    deploy_shifts, deploy_diag = _james_stein_bin_shifts(resid_full, bins_unb)
    te_pred = (te_base + deploy_shifts[bins_te]).astype(np.float32)
    # clip to a sane pEC50 range to avoid runaway shifts on empty/odd bins
    te_pred = np.clip(te_pred, 1.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy per-bin shifts = "
          f"{{novel:{deploy_shifts[0]:+.4f}, rare:{deploy_shifts[1]:+.4f}, "
          f"common:{deploy_shifts[2]:+.4f}}}  (global {deploy_diag['global']:+.4f})")
    print(f"   deploy diag = {deploy_diag}")
    print(f"   te(513): mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"min={te_pred.min():.3f} max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}  "
          f"(anchor was {rae(y_unb, te_base[unb_idx]):.4f})")

    # Median-(per-fold-mean)-seed OOF for storage
    med_seed_idx = int(np.argsort(pfm)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} "
          f"(per_fold_mean={pfm[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_pfm < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3303 per-bin residual correction reaches "
            f"per-fold-mean {mean_pfm:.4f} +/- {std_pfm:.4f} (15 seeds), clearing "
            f"the BETTER gate {GATE_BETTER:.4f} on the nb3200 anchor "
            f"(base {mean_base_pfm:.4f}, delta {delta_vs_base:+.4f}). This is a "
            f"SUBSTRATE-axis move (scaffold-frequency-conditioned mean-shift), "
            f"orthogonal to the post-hoc clip/stretch ceiling (nb3173 0.4437). "
            f"Deploy per-bin shifts novel={deploy_shifts[0]:+.4f}/"
            f"rare={deploy_shifts[1]:+.4f}/common={deploy_shifts[2]:+.4f}. "
            f"Recommend deep-30 re-verification before PRIMARY-1 promotion "
            f"(cycle-160 rule)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3303 per-bin residual correction per-fold-mean "
            f"{mean_pfm:.4f} +/- {std_pfm:.4f} >= BETTER gate {GATE_BETTER:.4f} "
            f"(base {mean_base_pfm:.4f}, delta {delta_vs_base:+.4f}). The nb3200 "
            f"anchor's per-bin residual bias is too small / too noisy at n=253 "
            f"(novel bin only carries mean_resid near the global) for a "
            f"3-bin additive shift to beat the learned-clip ceiling. The F2 "
            f"novel-scaffold over-prediction is already largely absorbed by the "
            f"nb3090->nb3200 clip pipeline; a coarse scaffold-frequency stratum "
            f"shift adds no honest signal. Keep nb3200/nb3190 line PRIMARY; "
            f"do not promote nb3303."
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

    sub_csv = SUBMISSIONS / f"{TAG}_scaffold_bin_residual.csv"
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
        "anchor_tag": ANCHOR_TAG,
        "method": "per_scaffold_freq_bin_james_stein_residual_shift",
        "paradigm": (
            "scaffold-frequency-conditioned additive residual correction "
            "(F2 novel-scaffold over-prediction); substrate axis, orthogonal "
            "to post-hoc clip/stretch operators"
        ),
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "anchor_full_oof_rae": round(full_oof_rae, 4),
        "anchor_leak_eq_truth_frac": round(leak_eq, 4),
        "bins": {str(b): BIN_NAMES[b] for b in range(N_BINS)},
        "bin_counts_unb": bin_counts_unb,
        "bin_counts_te": bin_counts_te,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_per_fold_mean": round(mean_pfm, 4),
        "std_per_fold_mean": round(std_pfm, 4),
        "sem_per_fold_mean": round(sem_pfm, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_per_fold_mean": round(median_pfm, 4),
        "min_per_fold_mean": round(float(pfm.min()), 4),
        "max_per_fold_mean": round(float(pfm.max()), 4),
        "mean_base_per_fold_mean": round(mean_base_pfm, 4),
        "delta_vs_base": round(delta_vs_base, 4),
        "mean_pooled_rae": round(mean_pooled, 4),
        "ref_anchor_nb3200": REF_ANCHOR_NB3200,
        "ref_nb3173": REF_NB3173,
        "ref_nb2171": REF_NB2171,
        "deploy_shifts": {
            "novel": round(float(deploy_shifts[0]), 4),
            "rare": round(float(deploy_shifts[1]), 4),
            "common": round(float(deploy_shifts[2]), 4),
        },
        "deploy_diag": deploy_diag,
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
    print(f"   per-fold-mean ({n_s} seeds) = {mean_pfm:.4f} +/- {std_pfm:.4f}")
    print(f"   95% CI                  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   base (uncorrected)      = {mean_base_pfm:.4f}")
    print(f"   delta vs base           = {delta_vs_base:+.4f}")
    print(f"   gate BETTER             = {GATE_BETTER:.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   wall                    = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_per_fold_mean", "std_per_fold_mean", "ci95_low", "ci95_high",
        "mean_base_per_fold_mean", "delta_vs_base", "mean_pooled_rae",
        "deploy_shifts", "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
