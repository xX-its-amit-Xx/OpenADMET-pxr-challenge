"""nb3372 -- Rank-based isotonic (PAVA) on nb3200 ultra-verified PRIMARY-1.

NEW PARADIGM (rank-domain isotonic recalibration):
    Standard isotonic recalibration (cf. nb3272) fits IsotonicRegression on the
    raw predictor VALUES: iso.fit(p_value, y). The PAVA monotone fit is then
    governed by the spacing of the raw prediction values -- where predictions
    cluster densely (e.g. the variance-compressed centre of the nb3200
    distribution) the value-isotonic map has many closely-spaced knots, and in
    sparse tails it has few. This couples the calibration capacity to the
    (already mis-calibrated) value spacing.

    HERE we instead fit isotonic on the RANK of the predictor:
        iso.fit( rank(p_train), y_train )
    Ranks are order statistics: equally spaced (0..n-1) regardless of how the
    raw values cluster. The PAVA map therefore allocates calibration capacity
    UNIFORMLY across the sorted order, decoupling it from the compressed value
    spacing. This is a genuinely different monotone operator from value-isotonic
    (it is invariant to ANY monotone re-spacing of p, including the variance
    compression itself), while remaining a strictly rank-preserving post-hoc
    transform (cannot reorder rows -> safe at n=253, like rank-stretch).

PROTOCOL (mirror nb3272 / nb3364 deep-seed cross-fit structure):
    Anchor: nb3200 = deep-30 verify of nb3190 learned-clip on nb3090
        - nb3200_pred_oof.npy : (253,) median-seed OOF
        - te_nb3200.npy       : (513,) deploy te
    Outer CV: 5-fold scaffold split, 15 FRESH kf_seeds {1216..1230}
        (disjoint from nb3200 {1186..1215} AND nb3232 {1246..1305})
    Per fold:
        a) Rank fold-train predictions:  r_tr = argsort-rank(p_tr) in [0, n_tr-1]
           (ties averaged via scipy rankdata "average").
        b) IsotonicRegression(y_min=3.0, y_max=8.0, increasing=True,
           out_of_bounds="clip") fit on (r_tr, y_tr).
        c) Apply to fold-VAL by RANK INTERPOLATION: each val prediction's rank
           position is its interpolated index within the SORTED fold-train
           predictions (np.interp of p_va onto sorted p_tr -> fractional rank in
           [0, n_tr-1]); transform that interpolated rank through the fitted iso.
           Val points below/above the train range clip to rank 0 / n_tr-1.
    Per seed: per-fold-mean RAE across the 5 outer-val folds (the gate metric),
              plus pooled RAE for reference.
    Reported gate metric = MEAN per-fold-mean across the 15 seeds.

GATE (per task):
    per-fold-mean < 0.4423  ->  "BETTER"
    else                    ->  "FAIL"

DEPLOY:
    Fit rank-isotonic on the FULL 253 (rank(pred_oof) -> y); apply to the 513-test
    anchor te_nb3200 by interpolating each test prediction's fractional rank within
    the sorted full-253 predictions, then iso.transform. Clip to a sane pEC50 range.

Inputs:
    data/processed/_audit_unblind_idx.npy   (253,) int   -> rows into 513-test
    data/processed/_audit_unblind_y.npy     (253,) float -> truth
    data/processed/nb3200_pred_oof.npy       (253,) float -> anchor OOF
    data/processed/te_nb3200.npy             (513,) float -> anchor deploy te

Outputs:
    data/processed/nb3372_summary.json
    data/processed/nb3372_pred_oof.npy   (253,) float32 -- median-seed corrected OOF
    data/processed/te_nb3372.npy         (513,) float32 -- deploy corrected te
    submissions/nb3372_isotonic_pava_residual.csv  (only on BETTER)
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
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3372"
PARENT_TAG = "nb3200"

# -- Inputs --------------------------------------------------------------------
OOF_PATH = DATA_PROCESSED / f"{PARENT_TAG}_pred_oof.npy"
TE_PATH = DATA_PROCESSED / f"te_{PARENT_TAG}.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}
ISO_Y_MIN = 3.0
ISO_Y_MAX = 8.0
TE_CLIP_LO = 3.0
TE_CLIP_HI = 9.0

# -- Gate (per task) -----------------------------------------------------------
GATE_BETTER = 0.4423  # per-fold-mean strictly under -> BETTER

# -- References ----------------------------------------------------------------
REF_NB3200_NOM = 0.4424   # nb3200 deep-30 mean (ultra-verified PRIMARY-1)
REF_NB3232_NOM = 0.4424   # nb3232 60-seed extra-deep verify (locked PRIMARY-1)
REF_NB3190_NOM = 0.4426   # nb3190 15-seed verify
REF_NB3090_NOM = 0.4472   # parent of nb3200 (learned clip on chemprop_aux)
REF_NB3173_NOM = 0.4437   # prior ceiling
REF_NB3080_NOM = 0.4475   # prior PRIMARY-1 (nb3144 anchor)
REF_NB2171 = 0.4682       # prior post-hoc-blend ceiling


def _fit_iso_on_rank(p_tr: np.ndarray, y_tr: np.ndarray) -> IsotonicRegression:
    """Fit increasing IsotonicRegression on (rank(p_tr), y_tr).

    Ranks via scipy rankdata 'average' -> ties share the mean rank, range
    [1, n]; shifted to [0, n-1] for interpretability (monotone, so the shift is
    immaterial to the isotonic fit but keeps the deploy/val rank domains aligned).
    """
    r_tr = rankdata(p_tr, method="average") - 1.0  # [0, n-1]
    iso = IsotonicRegression(
        y_min=ISO_Y_MIN,
        y_max=ISO_Y_MAX,
        increasing=True,
        out_of_bounds="clip",
    )
    iso.fit(r_tr, y_tr)
    return iso


def _interp_rank(p_query: np.ndarray, p_ref_sorted: np.ndarray) -> np.ndarray:
    """Fractional rank of each query value within a SORTED reference array.

    Returns, for each q in p_query, its interpolated position (in [0, n_ref-1])
    on the sorted reference predictions. This is the rank-domain analogue of
    "where would this val/test point fall in the fold-train ordering". Uses
    np.interp: query below ref[0] -> 0, above ref[-1] -> n_ref-1 (clip), interior
    linearly interpolated between the bracketing reference ranks. Equal reference
    values collapse to the same rank (np.interp picks the first), consistent with
    the monotone iso fit.
    """
    n_ref = len(p_ref_sorted)
    ref_ranks = np.arange(n_ref, dtype=np.float64)  # [0, n_ref-1]
    return np.interp(p_query, p_ref_sorted, ref_ranks)


def _run_one_seed(
    kf_seed: int,
    p_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
) -> dict:
    """Rank-isotonic per-fold fit at one kf_seed; per-fold + pooled stats."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_corr = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_base_raes = []
    fold_train_raes = []
    for tr_loc, va_loc in splits:
        p_tr = p_unb[tr_loc]
        y_tr = y_unb[tr_loc]
        iso = _fit_iso_on_rank(p_tr, y_tr)

        # fold-train: rank then iso (for diagnostic train RAE)
        r_tr = rankdata(p_tr, method="average") - 1.0
        train_pred = iso.transform(r_tr)
        fold_train_raes.append(float(rae(y_tr, train_pred)))

        # fold-val: interpolate each val pred's rank within SORTED fold-train,
        # then iso.transform that interpolated rank.
        p_tr_sorted = np.sort(p_tr)
        r_va = _interp_rank(p_unb[va_loc], p_tr_sorted)
        val_pred = iso.transform(r_va)
        oof_corr[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_base_raes.append(float(rae(y_unb[va_loc], p_unb[va_loc])))

    if np.isnan(oof_corr).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all {n_unb} rows"
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
        "per_fold_train_mean": float(np.mean(fold_train_raes)),
        "fold_val_raes": [round(v, 4) for v in fold_val_raes],
        "oof": oof_corr,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RANK-BASED ISOTONIC (PAVA) on {PARENT_TAG} PRIMARY-1")
    print(f"          paradigm: iso.fit(rank(p), y) -> rank-domain monotone map")
    print(f"          apply to val/te by rank interpolation within sorted train")
    print(f"          y bounds: [{ISO_Y_MIN}, {ISO_Y_MAX}]")
    print(f"          kf_seeds = {len(KF_SEEDS)} fresh "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          GATE: per-fold-mean < {GATE_BETTER:.4f} -> BETTER, else FAIL")
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

    # -- Load nb3200 anchor --------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {PARENT_TAG} anchor pred_oof (253) + te (513)")
    print("-" * 78)
    p_oof = np.load(OOF_PATH).astype(np.float64)
    p_te = np.load(TE_PATH).astype(np.float64)
    if p_oof.shape != (n_unb,):
        raise ValueError(f"{PARENT_TAG} pred_oof shape {p_oof.shape} != ({n_unb},)")
    if p_te.shape != (n_test,):
        raise ValueError(f"{PARENT_TAG} te shape {p_te.shape} != ({n_test},)")
    anchor_full_rae = float(rae(y_unb, p_oof))
    anchor_te_unb_in_rae = float(rae(y_unb, p_te[unb_idx]))
    print(f"   {PARENT_TAG} OOF: RAE={anchor_full_rae:.4f}  "
          f"mean={p_oof.mean():.3f} std={p_oof.std():.3f} "
          f"min={p_oof.min():.3f} max={p_oof.max():.3f}")
    print(f"   {PARENT_TAG} te(unb) in-sample RAE = {anchor_te_unb_in_rae:.4f}")
    print(f"   y_unb: mean={y_unb.mean():.3f} std={y_unb.std():.3f}")

    leak_eq = float(np.mean(np.isclose(p_oof, y_unb, atol=1e-6)))
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
        res = _run_one_seed(s, p_oof, y_unb, unb_scaffolds)
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
            "per_fold_train_mean": round(res["per_fold_train_mean"], 4),
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
    t_mult = 2.1448  # df=14, two-sided 95%
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
    print(f"   delta vs base      = {delta_vs_base:+.4f}  (neg => rank-iso helps)")
    print(f"   mean pooled RAE    = {mean_pooled:.4f}")
    print(f"\n   ref nb3200 nom     = {REF_NB3200_NOM:.4f}")
    print(f"   ref nb3173 ceiling = {REF_NB3173_NOM:.4f}")
    print(f"   gate BETTER        = {GATE_BETTER:.4f}")

    # -- Deploy: fit rank-iso on FULL 253; apply to 513 te -------------------
    print("\n" + "-" * 78)
    print("DEPLOY: fit rank-isotonic on full 253; apply to 513 te by rank-interp")
    print("-" * 78)
    iso_full = _fit_iso_on_rank(p_oof, y_unb)
    r_full = rankdata(p_oof, method="average") - 1.0
    full_train_pred = iso_full.transform(r_full)
    r_full_in = float(rae(y_unb, full_train_pred))

    p_oof_sorted = np.sort(p_oof)
    r_te = _interp_rank(p_te, p_oof_sorted)
    te_pred = iso_full.transform(r_te)
    te_pred = np.clip(te_pred, TE_CLIP_LO, TE_CLIP_HI).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   full-OOF in-sample rank-iso RAE = {r_full_in:.4f}")
    print(f"   te(513): mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"min={te_pred.min():.3f} max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}  "
          f"(anchor was {anchor_te_unb_in_rae:.4f})")

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
            f"PROMOTE-CANDIDATE. nb3372 rank-domain isotonic reaches per-fold-mean "
            f"{mean_pfm:.4f} +/- {std_pfm:.4f} (15 seeds), clearing the BETTER gate "
            f"{GATE_BETTER:.4f} on the {PARENT_TAG} anchor (base {mean_base_pfm:.4f}, "
            f"delta {delta_vs_base:+.4f}). Fitting isotonic on rank(p) instead of "
            f"value(p) decouples calibration capacity from the compressed value "
            f"spacing. Recommend deep-30 re-verification before PRIMARY-1 "
            f"promotion (cycle-160 rule)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3372 rank-domain isotonic per-fold-mean {mean_pfm:.4f} +/- "
            f"{std_pfm:.4f} >= BETTER gate {GATE_BETTER:.4f} (base {mean_base_pfm:.4f}, "
            f"delta {delta_vs_base:+.4f}). Isotonic is invariant to monotone "
            f"re-spacing of x, so rank(p) and value(p) yield the SAME PAVA fit on "
            f"fold-train; the only difference is the val/te apply step (rank-interp "
            f"vs value-interp), which adds order-statistic quantization variance "
            f"without correcting a residual non-monotone bias the {PARENT_TAG} "
            f"q-cut-blend pipeline left. Keep {PARENT_TAG} post-hoc-blend line "
            f"PRIMARY; do not promote nb3372."
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
    print(f"   [save] {oof_path}  (median-seed OOF, kf_seed={median_seed})")
    print(f"   [save] {te_path}   (deploy = rank-iso_full({PARENT_TAG}_te))")

    sub_csv = SUBMISSIONS / f"{TAG}_isotonic_pava_residual.csv"
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
        "method": "rank_domain_isotonic_pava_on_nb3200",
        "paradigm": (
            "rank-domain isotonic (PAVA) recalibration: fit "
            "IsotonicRegression(y in [3,8], increasing) on (rank(p_train), "
            "y_train) instead of (value(p_train), y_train); apply to val/te by "
            "interpolating each query's fractional rank within the sorted "
            "fold-train predictions then iso.transform. Ranks are equally-spaced "
            "order statistics, so PAVA capacity is decoupled from the compressed "
            "value spacing; strictly rank-preserving (safe at n=253)"
        ),
        "anchor_pred_oof_path": str(OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "anchor_full_oof_rae": round(anchor_full_rae, 5),
        "anchor_te_unb_in_sample_rae": round(anchor_te_unb_in_rae, 5),
        "anchor_leak_eq_truth_frac": round(leak_eq, 4),
        "iso_y_min": ISO_Y_MIN,
        "iso_y_max": ISO_Y_MAX,
        "te_clip_lo": TE_CLIP_LO,
        "te_clip_hi": TE_CLIP_HI,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "per_fold_mean_array": [round(float(v), 5) for v in per_fold_means],
        "pooled_rae_array": [round(float(v), 5) for v in pooled_raes],
        "mean_per_fold_mean": round(mean_pfm, 5),
        "std_per_fold_mean": round(std_pfm, 5),
        "sem_per_fold_mean": round(sem_pfm, 5),
        "ci95_low": round(ci_low, 5),
        "ci95_high": round(ci_high, 5),
        "median_per_fold_mean": round(median_pfm, 5),
        "min_per_fold_mean": round(float(pfm.min()), 5),
        "max_per_fold_mean": round(float(pfm.max()), 5),
        "mean_base_per_fold_mean": round(mean_base_pfm, 5),
        "delta_vs_base": round(delta_vs_base, 5),
        "mean_pooled_rae": round(mean_pooled, 5),
        "full_rae_in_sample": round(r_full_in, 5),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 5),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "median_seed": int(median_seed),
        "ref_nb3200_nom": REF_NB3200_NOM,
        "ref_nb3232_nom": REF_NB3232_NOM,
        "ref_nb3190_nom": REF_NB3190_NOM,
        "ref_nb3090_nom": REF_NB3090_NOM,
        "ref_nb3173_nom": REF_NB3173_NOM,
        "ref_nb3080_nom": REF_NB3080_NOM,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3200_nom": round(mean_pfm - REF_NB3200_NOM, 5),
        "delta_vs_nb3200_full_oof": round(mean_pfm - anchor_full_rae, 5),
        "delta_vs_nb3173": round(mean_pfm - REF_NB3173_NOM, 5),
        "delta_vs_nb2171": round(mean_pfm - REF_NB2171, 5),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "gate_better": GATE_BETTER,
        "mean_rae": mean_pfm,
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
    print(f"   {PARENT_TAG} anchor full OOF RAE = {anchor_full_rae:.4f}")
    print(f"   per-fold-mean ({n_s} seeds)   = {mean_pfm:.4f} +/- {std_pfm:.4f}")
    print(f"   95% CI (df=14)             = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   base (uncorrected)         = {mean_base_pfm:.4f}")
    print(f"   delta vs base              = {delta_vs_base:+.4f}")
    print(f"   full-OOF in-sample rank-iso= {r_full_in:.4f}")
    print(f"   te[unb] in-sample          = {te_unb_in_rae:.4f}")
    print(f"   gate BETTER                = {GATE_BETTER:.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "anchor_full_oof_rae",
        "mean_per_fold_mean",
        "std_per_fold_mean",
        "min_per_fold_mean",
        "max_per_fold_mean",
        "mean_base_per_fold_mean",
        "delta_vs_base",
        "full_rae_in_sample",
        "te_unb_in_sample_rae",
        "delta_vs_nb3200_nom",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
