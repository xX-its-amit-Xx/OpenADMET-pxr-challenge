"""nb3042 -- 15-seed wide-seed verification of nb3033 LOG-SPACE per-fold
            simplex (geometric-mean blend) on K18+K19 deep-30.

CONTEXT:
    nb3033 reported 5-seed pooled outer-val RAE 0.4504 (BETTER vs gate 0.4511)
    on kf_seeds {1051..1055}. n_seeds=5 is hypothesis-generation tier per
    cycle-160/245 wide-seed protocol; gate decisions REQUIRE >=15 fresh
    seeds before promote. nb3033's reported std=0.00111 looks suspiciously
    under-dispersed (similar to nb2060 5-seed 0.00087 -> 30-seed 0.00408
    4.7x under-dispersion).

    nb3033 paradigm: pred = exp(w*log(K18) + (1-w)*log(K19)) i.e. geometric
    mean in pEC50 space. Per-fold golden-section over w in [0,1]. Both
    anchors at deep-30 PRE-unblind (chemprop_aux + residual LGBM); no
    POST-unblind contamination chain risk.

PROTOCOL:
    Re-run nb3033 per-fold golden-section log-blend at 15 FRESH kf_seeds
    {1066..1080} (NEW seeds: NOT 1051-1055 from nb3033 single-fit,
    NOT 1051-1065 from nb3030 wide-seed verify of nb3020,
    NOT 1006-1050 from prior wide-seed verifies).
    Per kf_seed: 5-fold scaffold split, per-fold 1D golden-section over
    w in [0,1] on log-blended K18/K19 deep-30 cached OOFs.
    Report mean +/- std + 95% CI across the 15 fresh seeds.

    Anchors LOADED from caches (NO rebuild):
        K18: data/processed/nb2960_K18_30seed_oof.npy / _te.npy
        K19: data/processed/nb3000_K19_30seed_oof.npy / te_nb3000_K19.npy

GATE (same paradigm as nb3040 / nb3030 wide-seed verify protocol):
    mean < 0.4511 -> "VERIFIED_NEW_PRIMARY1"
        (nb3033 beats nb3001 wide-seed-verified ceiling 0.4511, new top)
    mean < 0.4518 -> "VERIFIED_MARGINAL"
        (still inside nb3003 single-seed reference but not new PRIMARY-1)
    shift > +0.005 vs single-kf -> "LUCKY_SEED_TRAP"
        (nb3033 5-seed mean 0.4504 was a fortunate batch; reject promotion)
    else -> "FAIL_OR_REPORT"

References:
    nb3033 5-seed pooled mean         = 0.4504   <- parent (verify target)
    nb3033 5-seed pooled std          = 0.00111  (suspect under-dispersed)
    nb3001 15-seed wide-mean          = 0.4511   <- current best ceiling
    nb3003 5-anchor single-seed       = 0.4518   <- PROMOTE band
    nb2982 single-kf=1001             = 0.4505   (2-anchor K18,K20 linear)
    nb2992 3-anchor linear simplex    = 0.4479
    nb2960 K18 deep-30                = 0.4536
    nb3000 K19 deep-30                = 0.4607
    nb2171 ceiling deep-30            = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3042_summary.json
    data/processed/nb3042_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3042.npy         (513,) float32 -- full-pool deploy te
    submissions/nb3042_verify_nb3033.csv  (only on promote verdicts)
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3042"
PARENT_TAG = "nb3033"

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
KF_SEEDS = list(range(1066, 1081))   # 15 fresh seeds {1066..1080}
                                     #   NOT 1001 (nb3020 single-fit)
                                     #   NOT 1051..1055 (nb3033 single-fit)
                                     #   NOT 1051..1065 (nb3030 wide-seed verify)
                                     #   NOT 1006-1050 (prior wide-seed verifies)

# -- Golden-section tuning -----------------------------------------------------
GS_TOL = 1e-6
GS_MAX_ITER = 200
PHI = (np.sqrt(5.0) - 1.0) / 2.0  # 0.6180339...

# Floor for log() to avoid log(0); pEC50 typically lives in [3, 9] anyway.
LOG_FLOOR = 1e-6

# -- Gates (mirror nb3030 / nb3040 wide-seed verify protocol) ------------------
GATE_NEW_PRIMARY1 = 0.4511            # mean < this -> VERIFIED_NEW_PRIMARY1
GATE_MARGINAL = 0.4518                # mean < this -> VERIFIED_MARGINAL
LUCKY_SHIFT_THRESHOLD = 0.005         # shift > this -> LUCKY_SEED_TRAP

# -- References ----------------------------------------------------------------
REF_NB3033_5SEED_MEAN = 0.4504
REF_NB3033_5SEED_STD = 0.00111
REF_NB3001_WIDE_MEAN = 0.4511
REF_NB3003_SINGLE_KF = 0.4518
REF_NB2982_SINGLE_KF = 0.4505
REF_NB2992 = 0.4479
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _log_blend(w: float, P_log: np.ndarray) -> np.ndarray:
    """Geometric mean blend: pred = exp(w*log(K18) + (1-w)*log(K19))."""
    return np.exp(w * P_log[:, 0] + (1.0 - w) * P_log[:, 1])


def _golden_section(P_log: np.ndarray, y: np.ndarray,
                    lo: float = 0.0, hi: float = 1.0,
                    tol: float = GS_TOL,
                    max_iter: int = GS_MAX_ITER) -> tuple[float, float]:
    """Minimize RAE(y, exp(w*log(K18)+(1-w)*log(K19))) over w in [lo, hi].

    Identical implementation to nb3033 _golden_section.
    """
    a, b = float(lo), float(hi)

    def loss(w: float) -> float:
        return float(rae(y, _log_blend(w, P_log)))

    # Endpoint values (so we never miss boundary minima at w=0 or w=1)
    la = loss(a)
    lb = loss(b)

    c = b - PHI * (b - a)
    d = a + PHI * (b - a)
    lc = loss(c)
    ld = loss(d)
    for _ in range(max_iter):
        if (b - a) < tol:
            break
        if lc < ld:
            b, lb = d, ld
            d, ld = c, lc
            c = b - PHI * (b - a)
            lc = loss(c)
        else:
            a, la = c, lc
            c, lc = d, ld
            d = a + PHI * (b - a)
            ld = loss(d)
    # candidate minima: interior bracket midpoint + both endpoints
    interior = 0.5 * (a + b)
    li = loss(interior)
    cands = [(a, la), (b, lb), (interior, li)]
    w_best, l_best = min(cands, key=lambda t: t[1])
    return float(w_best), float(l_best)


def _run_one_seed(kf_seed: int, P_log_unb: np.ndarray, y_unb: np.ndarray,
                  unb_scaffolds: list[str]) -> dict:
    """Run nb3033 per-fold golden-section log-blend at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_w_list = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        w, _r_train = _golden_section(P_log_unb[tr_loc], y_unb[tr_loc])
        val_pred = _log_blend(w, P_log_unb[va_loc])
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_w_list.append(w)
    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_w_K18_list": [float(w) for w in fold_w_list],
        "mean_w_K18": float(np.mean(fold_w_list)),
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 15-seed wide-seed verify of {PARENT_TAG} per-fold "
          f"LOG-SPACE simplex {K_LABELS}")
    print(f"          paradigm: pred = exp(w*log(K18) + (1-w)*log(K19))")
    print(f"          both anchors deep-30 (PRE-unblind chemprop_aux + resid)")
    print(f"          kf_seeds = {KF_SEEDS}")
    print(f"          (fresh, EXCLUDES 1051..1055 nb3033 single-fit and "
          f"1051..1065 nb3030 wide-seed verify)")
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
    print("STEP 1: load 2 K-anchor deep-30 OOFs and te arrays")
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
        print(f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE={r:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}  "
              f"oof_min={oof.min():.3f}")

    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)

    # Log-transform (with floor for safety). pEC50 ~ [3,9], well > 0.
    n_floor_unb = int((P_unb < LOG_FLOOR).sum())
    n_floor_te = int((P_te < LOG_FLOOR).sum())
    if n_floor_unb or n_floor_te:
        print(f"   WARN floored vals: unb={n_floor_unb}, te={n_floor_te}")
    P_log_unb = np.log(np.maximum(P_unb, LOG_FLOOR))
    P_log_te = np.log(np.maximum(P_te, LOG_FLOOR))

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # Correlation in linear and log space
    corr_lin = float(np.corrcoef(P_unb.T)[0, 1])
    corr_log = float(np.corrcoef(P_log_unb.T)[0, 1])
    print(f"\n   OOF correlation (linear)  = {corr_lin:.4f}")
    print(f"   OOF correlation (log)     = {corr_log:.4f}")

    # Reference: pure-log endpoints + geomean
    rae_w1 = float(rae(y_unb, _log_blend(1.0, P_log_unb)))
    rae_w0 = float(rae(y_unb, _log_blend(0.0, P_log_unb)))
    rae_w_half = float(rae(y_unb, _log_blend(0.5, P_log_unb)))
    print(f"   ref RAE @ w=1.0 (=K18)    = {rae_w1:.4f}")
    print(f"   ref RAE @ w=0.0 (=K19)    = {rae_w0:.4f}")
    print(f"   ref RAE @ w=0.5 (geomean) = {rae_w_half:.4f}")

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
    w_K18_seed_means = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(s, P_log_unb, y_unb, unb_scaffolds)
        pooled_raes.append(res["pooled_rae"])
        w_K18_seed_means.append(res["mean_w_K18"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_w_K18_list": [round(w, 4) for w in res["fold_w_K18_list"]],
            "mean_w_K18": round(res["mean_w_K18"], 4),
            "mean_w_K19": round(1.0 - res["mean_w_K18"], 4),
        })
        print(f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
              f"mean_w_K18={res['mean_w_K18']:.3f}  "
              f"wall={time.time()-ts:.2f}s")

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

    shift_vs_5seed = mean_rae - REF_NB3033_5SEED_MEAN
    disp_ratio = std_rae / REF_NB3033_5SEED_STD if REF_NB3033_5SEED_STD > 0 else float("nan")
    print(f"\n   nb3033 5-seed mean        = {REF_NB3033_5SEED_MEAN:.4f}")
    print(f"   nb3033 5-seed std         = {REF_NB3033_5SEED_STD:.5f}")
    print(f"   shift (15-seed - 5-seed)  = {shift_vs_5seed:+.4f}")
    print(f"   dispersion ratio (15/5)   = {disp_ratio:.2f}x")
    print(f"   nb3001 wide-seed ref      = {REF_NB3001_WIDE_MEAN:.4f}")
    print(f"   nb3003 single-seed ref    = {REF_NB3003_SINGLE_KF:.4f}")

    mean_w_K18 = float(np.mean(w_K18_seed_means))
    print(f"\n   mean-of-seed mean-of-fold w_K18 = {mean_w_K18:.4f}")
    print(f"   mean-of-seed mean-of-fold w_K19 = {1.0 - mean_w_K18:.4f}")

    # -- Deploy: golden-section on FULL 253 -> single global w ---------------
    w_full, r_full = _golden_section(P_log_unb, y_unb)
    print(f"\n   full-pool golden-section: w_K18={w_full:.4f}  "
          f"w_K19={1.0-w_full:.4f}")
    print(f"   full-pool in-sample RAE = {r_full:.4f}")
    te_pred = _log_blend(w_full, P_log_te).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te[unb] in-sample RAE   = {te_unb_in_rae:.4f}")
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
    if shift_vs_5seed > LUCKY_SHIFT_THRESHOLD:
        verdict = "LUCKY_SEED_TRAP"
        ladder_action = (
            f"REJECT nb3033 promotion. nb3033 5-seed mean "
            f"{REF_NB3033_5SEED_MEAN:.4f} was a fortunate batch; 15-seed "
            f"wide-seed mean {mean_rae:.4f} shift {shift_vs_5seed:+.4f} "
            f"exceeds +{LUCKY_SHIFT_THRESHOLD:.3f} tolerance. Keep prior PRIMARY-1."
        )
    elif mean_rae < GATE_NEW_PRIMARY1:
        verdict = "VERIFIED_NEW_PRIMARY1"
        ladder_action = (
            f"PROMOTE nb3033 (log-space simplex K18+K19) to PRIMARY-1 "
            f"(wide-seed {mean_rae:.4f} beats nb3001 ceiling "
            f"{REF_NB3001_WIDE_MEAN:.4f}). Demote prior PRIMARY-1 to PRIMARY-2."
        )
    elif mean_rae < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
        ladder_action = (
            f"KEEP prior PRIMARY-1. nb3033 wide-seed {mean_rae:.4f} is inside "
            f"PROMOTE gate {GATE_MARGINAL} but not strictly better than "
            f"nb3001 ceiling {REF_NB3001_WIDE_MEAN}; tag as alternate."
        )
    else:
        verdict = "FAIL_OR_REPORT"
        ladder_action = (
            f"REJECT nb3033 promotion. Wide-seed mean {mean_rae:.4f} above "
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

    sub_csv = SUBMISSIONS / f"{TAG}_verify_{PARENT_TAG}.csv"
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
        "method": "wide_seed_15_verify_log_space_per_fold_golden_section_K18_K19_deep30",
        "paradigm": "log_space_simplex_geometric_mean_vs_linear",
        "blend_formula": "exp(w*log(K18) + (1-w)*log(K19))",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_corr_linear": round(corr_lin, 4),
        "oof_corr_log": round(corr_log, 4),
        "ref_rae_w1_K18_only": round(rae_w1, 4),
        "ref_rae_w0_K19_only": round(rae_w0, 4),
        "ref_rae_w_half_geomean": round(rae_w_half, 4),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "gs_tol": GS_TOL,
        "gs_max_iter": GS_MAX_ITER,
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
        "p5_rae": round(p5, 4),
        "p95_rae": round(p95, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "ref_nb3033_5seed_mean": REF_NB3033_5SEED_MEAN,
        "ref_nb3033_5seed_std": REF_NB3033_5SEED_STD,
        "ref_nb3001_wide_mean": REF_NB3001_WIDE_MEAN,
        "ref_nb3003_single_kf": REF_NB3003_SINGLE_KF,
        "ref_nb2982_single_kf": REF_NB2982_SINGLE_KF,
        "ref_nb2992": REF_NB2992,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "shift_15seed_vs_5seed": round(shift_vs_5seed, 4),
        "dispersion_ratio_15_over_5": round(disp_ratio, 2),
        "delta_vs_nb3001_ceiling": round(mean_rae - REF_NB3001_WIDE_MEAN, 4),
        "delta_vs_nb3003_marginal": round(mean_rae - REF_NB3003_SINGLE_KF, 4),
        "mean_w_K18_across_seed_means": round(mean_w_K18, 4),
        "mean_w_K19_across_seed_means": round(1.0 - mean_w_K18, 4),
        "full_pool_golden_section": {
            "w_K18": round(float(w_full), 4),
            "w_K19": round(float(1.0 - w_full), 4),
            "rae_in_sample": round(float(r_full), 4),
        },
        "te_unb_in_sample_rae_full_pool": round(te_unb_in_rae, 4),
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
    print(f"   shift vs nb3033 5-seed= {shift_vs_5seed:+.4f}")
    print(f"   dispersion ratio      = {disp_ratio:.2f}x")
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
        "shift_15seed_vs_5seed", "dispersion_ratio_15_over_5",
        "delta_vs_nb3001_ceiling", "delta_vs_nb3003_marginal",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  full_pool_golden_section: {res.get('full_pool_golden_section')}")
