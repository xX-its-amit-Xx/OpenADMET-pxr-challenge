"""nb3171 -- Tighter clip (q03, q97) on nb3080 predictions.

NEW PARADIGM:
    More aggressive tail-cap. nb3161 used (q05, q95) and improved over nb3080
    parent. Push further: tighten the band to (q03, q97) of fold-train y. If
    the OOD compression is still present, a tighter band continues to tame
    the tails; if nb3161 already captured the benefit, this will regress.

    Hypothesis: nb3080 deep-30 blend over-predicts high tail / under-predicts
    low tail (variance compression by training-range extrapolation). Tighter
    quantile clip than nb3161 explores whether the tail-cap benefit is
    monotone in clip aggressiveness on this anchor.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    pred_base = nb3080_pred_oof  (253,) -- prior wide-seed blend
    Per outer fold:
        a) lo = quantile(y[fold_train], 0.03)
           hi = quantile(y[fold_train], 0.97)
        b) val_pred = np.clip(pred_base[va_loc], lo, hi)
        c) stitch into oof_clip (253,); pooled_rae across 5 folds.
    Repeat for 15 FRESH kf_seeds {1156..1170}.

GATE (on 15-seed mean):
    mean < 0.4437 (nb3161) -> "BETTER"
    mean < 0.4475          -> "MARGINAL" (between nb3161 and nb3080 parent)
    else                   -> "FAIL"

References:
    nb3161 15-seed mean             = 0.4437 +/- ?      <- parent (q05/q95)
    nb3080 15-seed wide-mean        = 0.4475 +/- 0.0006
    nb3030 15-seed wide-mean        = 0.4509
    nb2960 K18 deep-30 OOF          = 0.4536
    nb2171 prior post-hoc top       = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3080_pred_oof.npy
    data/processed/te_nb3080.npy

Outputs:
    data/processed/nb3171_summary.json
    data/processed/nb3171_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3171.npy         (513,) float32 -- deploy te
    submissions/nb3171_tighter_clip.csv  (only on BETTER or MARGINAL verdict)
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

TAG = "nb3171"
PARENT_TAG = "nb3080"
PRIOR_TAG = "nb3161"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3080_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3080.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1156, 1171))  # 15 FRESH seeds {1156..1170}

# -- Clip quantiles (TIGHTER than nb3161's 0.05/0.95) --------------------------
Q_LOW = 0.03
Q_HIGH = 0.97

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4437   # mean < this -> BETTER (beats nb3161 q05/q95)
GATE_MARGINAL = 0.4475 # mean < this -> MARGINAL (still beats nb3080 parent)

# -- References ----------------------------------------------------------------
REF_NB3161 = 0.4437
REF_NB3080 = 0.4475
REF_NB3080_STD = 0.0006
REF_NB3030 = 0.4509
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run per-fold quantile clip pipeline at a single kf_seed.

    pred_base : (253,) parent nb3080 OOF predictions (constant across seeds).
    y_unb     : (253,) truth labels.
    """
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
        f"{TAG} -- TIGHTER per-fold clip (q{Q_LOW:.2f}, q{Q_HIGH:.2f}) on "
        f"{PARENT_TAG} pred_oof"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          {PRIOR_TAG} (q05/q95) reference = {REF_NB3161:.4f}"
    )
    print(
        f"          {PARENT_TAG} parent reference   = "
        f"{REF_NB3080:.4f} +/- {REF_NB3080_STD:.4f}"
    )
    print(
        f"          gate: mean < {GATE_BETTER:.4f} -> BETTER; "
        f"< {GATE_MARGINAL:.4f} -> MARGINAL; else FAIL"
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

    # Truth stats
    y_full_lo = float(np.quantile(y_unb, Q_LOW))
    y_full_hi = float(np.quantile(y_unb, Q_HIGH))
    print(
        f"   y_unb stats: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
        f"min={y_unb.min():.3f}  max={y_unb.max():.3f}"
    )
    print(
        f"   y_unb full quantiles: q{Q_LOW:.2f}={y_full_lo:.3f}  "
        f"q{Q_HIGH:.2f}={y_full_hi:.3f}"
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
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"lo={res['fold_lo_mean']:.2f}  hi={res['fold_hi_mean']:.2f}  "
            f"clipped(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
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
        f"\n   ref {PRIOR_TAG} (q05/q95) 15-seed mean = {REF_NB3161:.4f}"
    )
    print(
        f"   delta vs {PRIOR_TAG}              = "
        f"{mean_rae - REF_NB3161:+.4f}"
    )
    print(
        f"   ref {PARENT_TAG} parent 15-seed mean   = "
        f"{REF_NB3080:.4f} +/- {REF_NB3080_STD:.4f}"
    )
    print(
        f"   delta vs {PARENT_TAG}              = "
        f"{mean_rae - REF_NB3080:+.4f}"
    )
    print(f"   ref nb3030 wide-seed ceiling     = {REF_NB3030:.4f}")
    print(f"   delta vs nb3030                  = {mean_rae - REF_NB3030:+.4f}")

    # -- Deploy: clip te to (q03, q97) of FULL 253 y --------------------------
    deploy_lo = float(np.quantile(y_unb, Q_LOW))
    deploy_hi = float(np.quantile(y_unb, Q_HIGH))
    te_pred = np.clip(te_base, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_base < deploy_lo))
    n_te_hi = int(np.sum(te_base > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy clip band = ({deploy_lo:.3f}, {deploy_hi:.3f}) "
        f"from full 253 y"
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
            f"PROMOTE-CANDIDATE. nb3171 15-seed mean {mean_rae:.4f} beats "
            f"nb3161 (q05/q95) {REF_NB3161:.4f} ({mean_rae - REF_NB3161:+.4f}) "
            f"and nb3080 parent {REF_NB3080:.4f} "
            f"({mean_rae - REF_NB3080:+.4f}). Tighter (q{Q_LOW:.2f}, "
            f"q{Q_HIGH:.2f}) tail-cap extracts real RAE gain. Consider "
            f"as candidate; re-verify with deep-30 before PRIMARY-1 swap."
        )
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL"
        ladder_action = (
            f"MARGINAL. nb3171 15-seed mean {mean_rae:.4f} beats nb3080 "
            f"parent {REF_NB3080:.4f} ({mean_rae - REF_NB3080:+.4f}) but "
            f"does NOT beat nb3161 (q05/q95) {REF_NB3161:.4f} "
            f"({mean_rae - REF_NB3161:+.4f}). Tighter band over-clips; "
            f"nb3161 remains the better operating point. Keep nb3161 / "
            f"prior PRIMARY-1."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3171 15-seed mean {mean_rae:.4f} does NOT beat "
            f"nb3080 parent {REF_NB3080:.4f} ({mean_rae - REF_NB3080:+.4f}) "
            f"nor nb3161 {REF_NB3161:.4f} ({mean_rae - REF_NB3161:+.4f}). "
            f"Per-fold y-range clip at (q{Q_LOW:.2f}, q{Q_HIGH:.2f}) is "
            f"too aggressive on this anchor (over-clips). Keep nb3161 / "
            f"prior PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_tighter_clip.csv"
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
        "prior_tag": PRIOR_TAG,
        "method": (
            "per_fold_y_range_clip_nb3080_pred_oof_q03_q97_tighter"
        ),
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "y_full_q_low": round(y_full_lo, 4),
        "y_full_q_high": round(y_full_hi, 4),
        "y_full_min": float(y_unb.min()),
        "y_full_max": float(y_unb.max()),
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
        "ref_prior_nb3161": REF_NB3161,
        "delta_vs_nb3161": round(mean_rae - REF_NB3161, 4),
        "ref_parent": REF_NB3080,
        "ref_parent_std": REF_NB3080_STD,
        "delta_vs_parent": round(mean_rae - REF_NB3080, 4),
        "ref_nb3030": REF_NB3030,
        "delta_vs_nb3030": round(mean_rae - REF_NB3030, 4),
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
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
    print(f"   delta vs nb3161       = {mean_rae - REF_NB3161:+.4f}")
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
        "delta_vs_nb3161", "delta_vs_parent", "delta_vs_nb3030",
        "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
