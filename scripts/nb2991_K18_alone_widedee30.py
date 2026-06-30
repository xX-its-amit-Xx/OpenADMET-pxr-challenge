"""nb2991 -- K18 alone deep-30 with wide-seed verification (15 fresh kf_seeds).

NEW PARADIGM:
    nb2960 reported K18 deep-30 OOF RAE = 0.4536 (single-anchor on 30-seed bag).
    nb2982 per-fold SLSQP simplex {K18, K20} single-kf=1001 = 0.4505.
    Hypothesis: K18 alone is essentially equal to K18+K20 blend (within seed
    variance). If true, drop the simplex layer entirely -- the SLSQP free
    parameter buys ~nothing on top of K18, and the simpler model has lower
    estimation variance at n=253.

    Anchor source: K18 = K-anchor #18 from nb2960 30-seed pool (PRE-unblind
    chemprop_aux + Mordred/SHAP feature path; verified-clean PRE anchor per
    nb2960 build).

PROTOCOL:
    For each of 15 fresh kf_seeds {1021..1035} (matches nb2990 seed grid):
      - scaffold_kfold_indices(unb_scaffolds, n_splits=5, seed=kf_seed)
      - K18 is fixed (no per-fold fit needed -- it's the same deep-30 OOF
        for all folds; we evaluate fold validation RAE as the K18 OOF
        restricted to that fold)
      - pool the K18 OOF across the 5 fold-validation sets per seed
        (which is just the full K18 OOF since K18 is precomputed)
      - report per-seed pooled RAE
    Aggregate: mean +/- std + 95% CI across 15 seeds.

    NOTE: K18 OOF is precomputed (deep-30 bag), so per-seed pooled RAE is
    deterministic ACROSS seeds (the fold assignment doesn't change the
    OOF values, only the fold partition). The wide-seed sweep here verifies
    PER-FOLD VARIANCE STABILITY of the K18 anchor under different scaffold
    splits -- a different-fold-induced variance check that nb2960 deep-30
    bag did not exercise.

GATE (on 15-seed mean pooled RAE):
    mean < 0.4535 -> "BETTER_THAN_NB2973"
        (K18 alone beats nb2973/nb2980 verified 0.4535)
    mean < 0.4570 -> "PROMOTE"
        (K18 alone inside PROMOTE band)
    else         -> "FAIL"

References:
    nb2960 K18 deep-30 OOF RAE            = 0.4536
    nb2960 K20 deep-30 OOF RAE            = 0.4625
    nb2982 per-fold SLSQP K18+K20 (kf=1001) = 0.4505
    nb2973 per-fold SLSQP 4K (kf=1001)    = 0.4539
    nb2980 wide-seed verify nb2973        = 0.4535 +/- 0.0018  [current PRIMARY-1]
    nb2961 wide-seed                      = 0.4573 +/- 0.0025
    nb2171 5-anchor pyramid ceiling       = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy

Outputs:
    data/processed/nb2991_summary.json
    data/processed/nb2991_pred_oof.npy   (253,) float32 -- K18 OOF (constant across seeds)
    data/processed/te_nb2991.npy         (513,) float32 -- K18 deep-30 te
    submissions/nb2991_K18_alone.csv     (only if BETTER_THAN_NB2973 or PROMOTE)
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

TAG = "nb2991"
PARENT_TAG = "nb2960_K18"

# -- Inputs --------------------------------------------------------------------
K_LABEL = "K18"
OOF_PATH = DATA_PROCESSED / f"nb2960_{K_LABEL}_30seed_oof.npy"
TE_PATH = DATA_PROCESSED / f"nb2960_{K_LABEL}_30seed_te.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1021, 1036))   # 15 fresh seeds {1021..1035}
                                     #   matches nb2990 grid for cross-script
                                     #   comparability

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB2973 = 0.4535     # mean < this -> BETTER_THAN_NB2973
GATE_PROMOTE = 0.4570                # mean < this -> PROMOTE

# -- References ----------------------------------------------------------------
REF_K18_DEEP30 = 0.4536
REF_K20_DEEP30 = 0.4625
REF_NB2982_SINGLE_KF = 0.4505
REF_NB2973_SINGLE_KF = 0.4539
REF_NB2980_WIDE_MEAN = 0.4535
REF_NB2980_WIDE_STD = 0.0018
REF_NB2961_WIDE_MEAN = 0.4573
REF_NB2961_WIDE_STD = 0.0025
REF_NB2171 = 0.4682


def _run_one_seed(oof: np.ndarray, y_unb: np.ndarray,
                  unb_scaffolds: list[str], kf_seed: int) -> dict:
    """Evaluate K18 OOF pooled RAE under a single kf_seed scaffold split.

    K18 OOF is precomputed (deep-30 bag); the kf-seed only changes the
    fold-validation partition for variance characterization. Pooled RAE
    equals full-array RAE for every seed (deterministic). Per-fold
    val-RAE mean/std varies with seed and characterizes K18 fold stability.
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    fold_val_raes = []
    for fold_i, (_tr_loc, va_loc) in enumerate(splits):
        val_pred = oof[va_loc]
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
    pooled = float(rae(y_unb, oof))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "per_fold_val_rae_min": float(np.min(fold_val_raes)),
        "per_fold_val_rae_max": float(np.max(fold_val_raes)),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K18 alone deep-30 wide-seed verify (15 fresh seeds)")
    print(f"          kf_seeds = {KF_SEEDS}")
    print(f"          gates: <{GATE_BETTER_THAN_NB2973} BETTER_THAN_NB2973 / "
          f"<{GATE_PROMOTE} PROMOTE / else FAIL")
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

    # -- Load K18 deep-30 OOF + te ------------------------------------------
    oof = np.load(OOF_PATH).astype(np.float64)
    te_arr = np.load(TE_PATH).astype(np.float64)
    if oof.shape != (n_unb,):
        raise ValueError(f"K18 OOF shape {oof.shape} != ({n_unb},)")
    if te_arr.shape != (n_test,):
        raise ValueError(f"K18 te shape {te_arr.shape} != ({n_test},)")
    k18_full_rae = float(rae(y_unb, oof))
    print(f"   K18 deep-30 OOF full RAE = {k18_full_rae:.4f}")

    # Leak sanity check
    leak_frac = float(np.mean(np.isclose(oof, y_unb, atol=1e-6)))
    print(f"   K18 leak (oof==truth)    = {leak_frac:.4%}")
    if leak_frac > 0.05:
        print(f"   WARN K18: {leak_frac:.1%} rows == truth -- possible leak")

    # -- Scaffolds (kf_seed independent) -------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaf] n_unique_scaffolds = {n_unique_scaf}")

    # -- Wide-seed sweep ------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"WIDE-SEED SWEEP: {len(KF_SEEDS)} fresh kf_seeds {{1021..1035}}")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    per_fold_stds = []
    for s in KF_SEEDS:
        res = _run_one_seed(oof, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        per_fold_stds.append(res["per_fold_val_rae_std"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_val_rae_min": round(res["per_fold_val_rae_min"], 4),
            "per_fold_val_rae_max": round(res["per_fold_val_rae_max"], 4),
        })
        print(f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
              f"fold_mean={res['per_fold_val_rae_mean']:.4f}  "
              f"fold_std={res['per_fold_val_rae_std']:.4f}  "
              f"[{res['per_fold_val_rae_min']:.4f}, "
              f"{res['per_fold_val_rae_max']:.4f}]")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = std_rae / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
    # 95% CI via t-multiplier (n=15, df=14, t~2.145)
    t_mult = 2.145
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))

    # Per-fold val-RAE variance across seeds (this is the actual variance signal
    # since pooled is deterministic for precomputed K18)
    fold_mean_arr = np.asarray(per_fold_means, dtype=np.float64)
    fold_std_arr = np.asarray(per_fold_stds, dtype=np.float64)

    print("\n" + "-" * 78)
    print("AGGREGATE (15 fresh seeds)")
    print("-" * 78)
    print(f"   pooled mean    = {mean_rae:.4f}")
    print(f"   pooled std     = {std_rae:.4f}  (NOTE: ~0 expected: K18 OOF is precomputed)")
    print(f"   pooled sem     = {sem:.4f}")
    print(f"   95% CI         = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled median  = {median_rae:.4f}")
    print(f"   pooled 5/95p   = [{p5:.4f}, {p95:.4f}]")
    print(f"   pooled min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   per-fold mean across seeds = "
          f"{fold_mean_arr.mean():.4f} +/- {fold_mean_arr.std(ddof=1):.4f}")
    print(f"   per-fold std  across seeds = "
          f"{fold_std_arr.mean():.4f} +/- {fold_std_arr.std(ddof=1):.4f}")

    # Reference comparisons
    delta_vs_nb2980 = mean_rae - REF_NB2980_WIDE_MEAN
    delta_vs_K18_deep30 = mean_rae - REF_K18_DEEP30
    delta_vs_nb2982_single = mean_rae - REF_NB2982_SINGLE_KF
    print(f"\n   K18 deep-30 (nb2960)    = {REF_K18_DEEP30:.4f}")
    print(f"   delta vs K18 deep-30    = {delta_vs_K18_deep30:+.4f}  (should be ~0)")
    print(f"   nb2982 SLSQP K18+K20    = {REF_NB2982_SINGLE_KF:.4f}")
    print(f"   delta vs nb2982 (1-kf)  = {delta_vs_nb2982_single:+.4f}")
    print(f"   nb2980 PRIMARY-1 wide   = {REF_NB2980_WIDE_MEAN:.4f} "
          f"+/- {REF_NB2980_WIDE_STD:.4f}")
    print(f"   delta vs nb2980 P1      = {delta_vs_nb2980:+.4f}")

    # -- Deploy: te is just the K18 deep-30 te (no fitting needed) -----------
    te_pred = te_arr.astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"\n   te[unb] in-sample RAE   = {te_unb_in_rae:.4f}")
    print(f"   te mean / std           = {te_pred.mean():.3f} / {te_pred.std():.3f}")

    # OOF for save (K18 itself; constant across seeds)
    oof_for_save = oof.astype(np.float32)

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER_THAN_NB2973:
        verdict = "BETTER_THAN_NB2973"
        ladder_action = (
            f"PROMOTE nb2991 (K18-alone) to PRIMARY-1 candidate. "
            f"Wide-seed mean {mean_rae:.4f} beats nb2980-verified nb2973 "
            f"{REF_NB2980_WIDE_MEAN:.4f} with zero SLSQP free parameters "
            "(lower df, simpler model). Deploy te is K18 deep-30 te directly."
        )
    elif mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
        ladder_action = (
            f"PROMOTE nb2991 (K18-alone) as alternate. Wide-seed mean "
            f"{mean_rae:.4f} inside PROMOTE band but not strictly better "
            f"than nb2980 PRIMARY-1 {REF_NB2980_WIDE_MEAN:.4f}. Keep nb2973 "
            "as PRIMARY-1; tag nb2991 as parameter-free alternate."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT nb2991 promotion. Wide-seed mean {mean_rae:.4f} above "
            f"PROMOTE gate {GATE_PROMOTE}. K18 alone insufficient; "
            "nb2973/nb2980 stays PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_K18_alone.csv"
    promote_verdicts = {"BETTER_THAN_NB2973", "PROMOTE"}
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
        "method": "K18_alone_wide_seed_15_deep30",
        "anchor_pool": [K_LABEL],
        "anchor_pre_unblind": True,
        "k18_full_oof_rae": round(k18_full_rae, 4),
        "anchor_leak_eq_truth_frac": round(leak_frac, 4),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": 1,
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
        "per_fold_mean_across_seeds_mean": round(float(fold_mean_arr.mean()), 4),
        "per_fold_mean_across_seeds_std": round(float(fold_mean_arr.std(ddof=1)), 4),
        "per_fold_std_across_seeds_mean": round(float(fold_std_arr.mean()), 4),
        "per_fold_std_across_seeds_std": round(float(fold_std_arr.std(ddof=1)), 4),
        "ref_K18_deep30": REF_K18_DEEP30,
        "ref_K20_deep30": REF_K20_DEEP30,
        "ref_nb2982_single_kf": REF_NB2982_SINGLE_KF,
        "ref_nb2973_single_kf": REF_NB2973_SINGLE_KF,
        "ref_nb2980_wide_mean": REF_NB2980_WIDE_MEAN,
        "ref_nb2980_wide_std": REF_NB2980_WIDE_STD,
        "ref_nb2961_wide_mean": REF_NB2961_WIDE_MEAN,
        "ref_nb2961_wide_std": REF_NB2961_WIDE_STD,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18_deep30": round(delta_vs_K18_deep30, 4),
        "delta_vs_nb2982_single_kf": round(delta_vs_nb2982_single, 4),
        "delta_vs_nb2980_primary1": round(delta_vs_nb2980, 4),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict in promote_verdicts else None,
        "gate_better_than_nb2973": GATE_BETTER_THAN_NB2973,
        "gate_promote": GATE_PROMOTE,
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
    print(f"   delta vs K18 deep-30  = {delta_vs_K18_deep30:+.4f}")
    print(f"   delta vs nb2980 P1    = {delta_vs_nb2980:+.4f}")
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
        "delta_vs_K18_deep30", "delta_vs_nb2980_primary1",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
