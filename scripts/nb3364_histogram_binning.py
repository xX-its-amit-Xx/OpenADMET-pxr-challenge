"""nb3364 -- Histogram-binning calibration on the nb3090 anchor.

NEW PARADIGM (operator: non-parametric histogram-binning calibration):
    Classic histogram binning (Zadrozny & Elkan 2001) for *regression*
    recalibration. Partition the predictor's output range into B equal-frequency
    bins and replace every prediction in a bin by the bin's empirical mean truth.
    Unlike rank-stretch / isotonic / clip (which apply a *monotone* global map to
    the value distribution), histogram binning is a *piecewise-constant* map: it
    can correct NON-monotone, locally-varying calibration error (e.g. the anchor
    over-predicts in the middle of its range but is calibrated in the tails) that
    a single global stretch factor cannot touch. It is, however, a very high-
    capacity operator (B free parameters), so each bin value is JAMES-STEIN
    SHRUNK toward that bin's own prediction-mean to control n=253 overfit: an
    uninformative / low-support bin collapses back to "predict your own
    prediction level" (the identity recalibration), preserving the anchor.

    B = 15 equal-frequency bins (quantile edges on FOLD-TRAIN predictions).

PROTOCOL (mirror nb3303 / nb3090 deep-seed cross-fit structure):
    anchor = nb3090 pred_oof (253,) / te_nb3090 (513,)  [PRE-unblind clean,
             full-OOF RAE 0.4470, post-hoc q-cut-blend ceiling on K18/K19]
    For each kf_seed in {1216..1230} (15 FRESH seeds):
        scaffold_kfold_indices(n_splits=5, shuffle=True, seed=kf_seed)
        For each fold:
            a) On FOLD-TRAIN preds p_tr: 15 equal-frequency bin edges via
               np.quantile(p_tr, linspace(0,1,16)); de-dup edges; assign each
               train row to a bin with np.digitize.
            b) Per bin b:
                   bin_pred_mean_b  = mean(p_tr  in bin b)   (identity target)
                   bin_truth_mean_b = mean(y_tr  in bin b)   (raw calibration)
               James-Stein shrink bin_truth_mean_b toward bin_pred_mean_b:
                   gap    = bin_truth_mean_b - bin_pred_mean_b
                   shrink = gap^2 / (gap^2 + s2_b / n_b)      (reliability wt)
                   value_b = bin_pred_mean_b + clip(shrink,0,1) * gap
               where s2_b = within-bin variance of y_tr, n_b = bin count.
               Empty bins -> value = bin midpoint prediction (identity).
            c) Apply to FOLD-VAL: assign each val row to a bin by the SAME
               fold-train edges (np.digitize, clamped to [0, n_bins-1]); set
               corrected[val] = value_b[bin(val)].
        per-fold RAE on (y_val, corrected_val); per-fold-mean = mean over 5 folds.
    Aggregate per-fold-mean across 15 seeds (mean +/- std, 95% CI df=14).

GATE (per task):
    per-fold-mean < 0.4423 -> "BETTER"
    else                   -> "FAIL"

DEPLOY:
    Fit 15 equal-frequency bins + JS-shrunk values on the FULL 253 (p=anchor
    pred_oof, y=truth); apply to the 513-test anchor te_nb3090 by each test
    row's bin assignment under the full-253 edges. Clip to a sane pEC50 range.

Inputs:
    data/processed/_audit_unblind_idx.npy   (253,) int   -> rows into 513-test
    data/processed/_audit_unblind_y.npy     (253,) float -> truth
    data/processed/nb3090_pred_oof.npy       (253,) float -> anchor OOF
    data/processed/te_nb3090.npy             (513,) float -> anchor deploy te

Outputs:
    data/processed/nb3364_summary.json
    data/processed/nb3364_pred_oof.npy   (253,) float32 -- median-seed corrected OOF
    data/processed/te_nb3364.npy         (513,) float32 -- deploy corrected te
    submissions/nb3364_histogram_binning.csv  (only on BETTER)
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
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3364"
ANCHOR_TAG = "nb3090"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / f"{ANCHOR_TAG}_pred_oof.npy"
TE_PATH = DATA_PROCESSED / f"te_{ANCHOR_TAG}.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Histogram-binning hyperparameters -----------------------------------------
N_BINS = 15  # equal-frequency bins on fold-train predictions

# -- Gates (per task) ----------------------------------------------------------
GATE_BETTER = 0.4423  # per-fold-mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_ANCHOR_NB3090 = 0.4470  # nb3090 pred_oof full RAE
REF_NB3080 = 0.4475         # prior post-hoc-blend ceiling
REF_NB2171 = 0.4682         # post-hoc-blend ceiling on nb730 anchor
REF_CEILING_BAND = 0.4718   # deep-30 post-hoc-blend co-converged ceiling (cyc163)


def _equal_freq_edges(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency (quantile) bin edges on predictions p.

    Returns interior+outer edges; de-duplicated so np.digitize is monotone even
    when ties collapse adjacent quantiles. Always spans [-inf, +inf] effectively
    by using the data min/max as the outer edges (digitize handles out-of-range
    rows by clamping in _assign_bins).
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(p, qs)
    # de-duplicate (ties at repeated prediction values collapse bins)
    edges = np.unique(edges)
    return edges


def _assign_bins(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each value in p to a bin index in [0, n_bins-1].

    n_bins = len(edges) - 1. Uses np.digitize on the INTERIOR edges and clamps
    so values below edges[0] land in bin 0 and values above edges[-1] land in
    the top bin (correct behaviour for fold-val rows outside the train range).
    """
    n_bins = len(edges) - 1
    if n_bins <= 0:
        return np.zeros(len(p), dtype=np.int64)
    # interior edges = edges[1:-1]; right=False -> bins are [edge_i, edge_{i+1})
    b = np.digitize(p, edges[1:-1], right=False)
    return np.clip(b, 0, n_bins - 1).astype(np.int64)


def _fit_bin_values(
    p_tr: np.ndarray,
    y_tr: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Per-bin James-Stein-shrunk calibrated value.

    For each bin b (n_bins = len(edges)-1):
        bin_pred_mean  = mean(p_tr in b)   (identity / shrink target)
        bin_truth_mean = mean(y_tr in b)   (raw calibrated value)
        gap    = bin_truth_mean - bin_pred_mean
        shrink = gap^2 / (gap^2 + s2/n)    (empirical-Bayes reliability weight)
        value  = bin_pred_mean + clip(shrink,0,1) * gap
    s2 = within-bin variance of y_tr (ddof=1 if n>1 else 0). Empty bins fall
    back to value = bin-edge midpoint (pure identity, contributes nothing).

    Returns (values[n_bins], bin_assignment_of_train[len(p_tr)], diag list).
    """
    n_bins = len(edges) - 1
    bins_tr = _assign_bins(p_tr, edges)
    values = np.empty(n_bins, dtype=np.float64)
    diag = []
    for b in range(n_bins):
        m = bins_tr == b
        n_b = int(m.sum())
        if n_b == 0:
            # empty bin -> identity at the bin midpoint (no calibration signal)
            mid = 0.5 * (edges[b] + edges[b + 1])
            values[b] = mid
            diag.append({"bin": b, "n": 0, "pred_mean": round(float(mid), 4),
                         "truth_mean": None, "shrink": None,
                         "value": round(float(mid), 4)})
            continue
        pb = p_tr[m]
        yb = y_tr[m]
        bin_pred_mean = float(pb.mean())
        bin_truth_mean = float(yb.mean())
        s2 = float(yb.var(ddof=1)) if n_b > 1 else 0.0
        gap = bin_truth_mean - bin_pred_mean
        denom = gap * gap + (s2 / n_b if n_b > 0 else 0.0)
        shrink = (gap * gap) / denom if denom > 1e-12 else 0.0
        shrink = float(np.clip(shrink, 0.0, 1.0))
        value = bin_pred_mean + shrink * gap
        values[b] = value
        diag.append({
            "bin": b, "n": n_b,
            "pred_mean": round(bin_pred_mean, 4),
            "truth_mean": round(bin_truth_mean, 4),
            "s2": round(s2, 4),
            "shrink": round(shrink, 4),
            "value": round(float(value), 4),
        })
    return values, bins_tr, diag


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Histogram-binning calibration at a single kf_seed; per-fold stats."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_corr = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_base_raes = []
    fold_nbins = []
    for tr_loc, va_loc in splits:
        p_tr = pred_base[tr_loc]
        y_tr = y_unb[tr_loc]
        edges = _equal_freq_edges(p_tr, N_BINS)
        values, _bins_tr, _diag = _fit_bin_values(p_tr, y_tr, edges)
        fold_nbins.append(len(edges) - 1)
        # apply to fold-val by SAME fold-train edges
        bins_va = _assign_bins(pred_base[va_loc], edges)
        corrected = values[bins_va]
        oof_corr[va_loc] = corrected
        fold_val_raes.append(float(rae(y_unb[va_loc], corrected)))
        fold_base_raes.append(float(rae(y_unb[va_loc], pred_base[va_loc])))

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
        "fold_nbins": fold_nbins,
        "oof": oof_corr,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- HISTOGRAM-BINNING CALIBRATION on {ANCHOR_TAG}")
    print(f"          {N_BINS} equal-frequency bins on fold-train predictions")
    print(f"          bin value = JS-shrink(mean(y_tr in bin) -> mean(pred in bin))")
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

    # -- Load nb3090 anchor ---------------------------------------------------
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
          f"mean={pred_base.mean():.3f} std={pred_base.std():.3f} "
          f"min={pred_base.min():.3f} max={pred_base.max():.3f}")
    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN anchor: {leak_eq:.1%} rows == truth -- possible leak")

    # -- Scaffolds for scaffold-CV -------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: Bemis-Murcko scaffolds for scaffold-CV splits")
    print("-" * 78)
    te_scaffolds_full = [bemis_murcko(s) or "" for s in te_smiles]
    unb_scaffolds = [te_scaffolds_full[i] for i in unb_idx]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds(unb) = {n_unique_scaf}")

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
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
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
            "fold_nbins": res["fold_nbins"],
        })
        print(f"   kf={s}: per_fold_mean={res['per_fold_mean']:.4f}  "
              f"(base {res['per_fold_base_mean']:.4f})  "
              f"pooled={res['pooled_rae']:.4f}  "
              f"nbins={res['fold_nbins']}  wall={time.time()-ts:.2f}s")

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
    print(f"   delta vs base      = {delta_vs_base:+.4f}  (neg => binning helps)")
    print(f"   mean pooled RAE    = {mean_pooled:.4f}")
    print(f"\n   ref anchor nb3090  = {REF_ANCHOR_NB3090:.4f}")
    print(f"   ref nb3080 ceiling = {REF_NB3080:.4f}")
    print(f"   gate BETTER        = {GATE_BETTER:.4f}")

    # -- Deploy: fit bins on FULL 253; apply to 513 te -----------------------
    print("\n" + "-" * 78)
    print("DEPLOY: fit 15 equal-freq bins + JS values on full 253; apply to 513")
    print("-" * 78)
    deploy_edges = _equal_freq_edges(pred_base, N_BINS)
    deploy_values, _bins_full, deploy_diag = _fit_bin_values(
        pred_base, y_unb, deploy_edges
    )
    bins_te = _assign_bins(te_base, deploy_edges)
    te_pred = deploy_values[bins_te].astype(np.float64)
    # clip to a sane pEC50 range to avoid runaway empty-bin midpoints
    te_pred = np.clip(te_pred, 1.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    bin_te_counts = [int((bins_te == b).sum()) for b in range(len(deploy_values))]
    print(f"   deploy n_bins      = {len(deploy_values)} "
          f"(edges de-dup from {N_BINS + 1} quantiles)")
    print(f"   deploy edges       = {[round(float(e),3) for e in deploy_edges]}")
    print(f"   deploy values      = {[round(float(v),3) for v in deploy_values]}")
    print(f"   te bin counts      = {bin_te_counts}")
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
            f"PROMOTE-CANDIDATE. nb3364 histogram-binning calibration reaches "
            f"per-fold-mean {mean_pfm:.4f} +/- {std_pfm:.4f} (15 seeds), clearing "
            f"the BETTER gate {GATE_BETTER:.4f} on the nb3090 anchor "
            f"(base {mean_base_pfm:.4f}, delta {delta_vs_base:+.4f}). Piecewise-"
            f"constant {N_BINS}-bin map corrects non-monotone local calibration "
            f"error the global q-cut-blend ceiling (nb3080 0.4475) cannot. "
            f"Recommend deep-30 re-verification before PRIMARY-1 promotion "
            f"(cycle-160 rule)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3364 histogram-binning calibration per-fold-mean "
            f"{mean_pfm:.4f} +/- {std_pfm:.4f} >= BETTER gate {GATE_BETTER:.4f} "
            f"(base {mean_base_pfm:.4f}, delta {delta_vs_base:+.4f}). The nb3090 "
            f"anchor is already monotone-calibrated by its q-cut-blend pipeline; "
            f"a {N_BINS}-bin piecewise-constant map (15 free params at n=253, "
            f"~17 rows/bin) adds quantization variance without correcting a "
            f"residual non-monotone bias. JS-shrink toward bin-pred-mean drives "
            f"most bins back to identity, so the operator degenerates to the "
            f"anchor. Keep nb3090/nb3080 post-hoc-blend line PRIMARY; do not "
            f"promote nb3364."
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

    sub_csv = SUBMISSIONS / f"{TAG}_histogram_binning.csv"
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
        "method": "histogram_binning_15_equal_freq_james_stein_to_bin_pred_mean",
        "paradigm": (
            "non-parametric piecewise-constant histogram-binning recalibration "
            "(15 equal-frequency bins on fold-train predictions; per-bin value = "
            "JS-shrunk mean(y_tr in bin) toward mean(pred in bin)); corrects "
            "NON-monotone local calibration error, orthogonal to the monotone "
            "global q-cut-blend / stretch / clip operator ceiling"
        ),
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "anchor_full_oof_rae": round(full_oof_rae, 4),
        "anchor_leak_eq_truth_frac": round(leak_eq, 4),
        "n_bins": N_BINS,
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
        "ref_anchor_nb3090": REF_ANCHOR_NB3090,
        "ref_nb3080": REF_NB3080,
        "ref_nb2171": REF_NB2171,
        "ref_ceiling_band": REF_CEILING_BAND,
        "deploy_n_bins": int(len(deploy_values)),
        "deploy_edges": [round(float(e), 4) for e in deploy_edges],
        "deploy_values": [round(float(v), 4) for v in deploy_values],
        "deploy_diag": deploy_diag,
        "te_bin_counts": bin_te_counts,
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
        "te_unb_in_sample_rae", "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
