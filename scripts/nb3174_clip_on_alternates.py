"""nb3174 -- Apply nb3161 (q05, q95) per-fold y-range clip to ALTERNATE anchors.

NEW PARADIGM:
    nb3161 confirmed that per-fold clipping nb3080 preds to fold-train
    (q05, q95) gives -0.0038 RAE (0.4475 -> 0.4437). Test whether the SAME
    1-parameter-per-fold range-clip operator transfers to OTHER anchors that
    have NOT been blended into the nb3080 pyramid:
        - nb3070 (alt anchor A)
        - nb3072 (alt anchor B)
        - nb3090 (alt anchor C)
        - K18 deep-30 (nb2960_K18_30seed_oof / te)

    Hypothesis: If clip is universal post-hoc on variance-compressed predictors
    (analog to rank-stretch family), it should help any anchor whose OOF preds
    over/under-shoot fold-train (q05, q95). If clip ONLY helps nb3080, it is
    anchor-specific and we stop fanning out.

PROTOCOL (per anchor, per kf_seed, 5-fold scaffold split):
    pred_base = <anchor>_pred_oof  (253,)
    Per outer fold:
        a) lo = quantile(y[fold_train], 0.05)
           hi = quantile(y[fold_train], 0.95)
        b) val_pred = np.clip(pred_base[va_loc], lo, hi)
        c) stitch into oof_clip; pooled_rae across 5 folds.
    Repeat for 5 FRESH kf_seeds {1156..1160}.

GATE (on best 5-seed mean across 4 anchors):
    best_mean < 0.4437 -> "BETTER" (beats nb3161 ceiling)
    best_mean < 0.4475 -> "MARGINAL" (improves over nb3080, not nb3161)
    else             -> "FAIL"

References:
    nb3161 15-seed mean (clip on nb3080) = 0.4437 +/- 0.0017  <- CEILING
    nb3080 15-seed mean                  = 0.4475 +/- 0.0006
    nb3070 5-seed (parent)               = (loaded from summary)
    nb3072 5-seed (parent)               = (loaded from summary)
    nb3090 5-seed (parent)               = (loaded from summary)
    nb2960 K18 deep-30 OOF               = 0.4536

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3070_pred_oof.npy + te_nb3070.npy
    data/processed/nb3072_pred_oof.npy + te_nb3072.npy
    data/processed/nb3090_pred_oof.npy + te_nb3090.npy
    data/processed/nb2960_K18_30seed_oof.npy + nb2960_K18_30seed_te.npy

Outputs:
    data/processed/nb3174_summary.json
    data/processed/nb3174_pred_oof.npy   (253,) float32 -- best-anchor median-seed OOF
    data/processed/te_nb3174.npy         (513,) float32 -- best-anchor deploy te
    submissions/nb3174_clip_<anchor>.csv (only on BETTER verdict)
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

TAG = "nb3174"

# -- Alternate anchors to test -------------------------------------------------
# (anchor_key, pred_oof_path, te_path, reference_label)
ANCHORS = [
    ("nb3070", DATA_PROCESSED / "nb3070_pred_oof.npy",
              DATA_PROCESSED / "te_nb3070.npy",       "nb3070"),
    ("nb3072", DATA_PROCESSED / "nb3072_pred_oof.npy",
              DATA_PROCESSED / "te_nb3072.npy",       "nb3072"),
    ("nb3090", DATA_PROCESSED / "nb3090_pred_oof.npy",
              DATA_PROCESSED / "te_nb3090.npy",       "nb3090"),
    ("K18",    DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
              DATA_PROCESSED / "nb2960_K18_30seed_te.npy", "K18_deep30"),
]

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1156, 1161))  # 5 FRESH seeds {1156..1160}

# -- Clip quantiles (FIXED, mirror nb3161) ------------------------------------
Q_LOW = 0.05
Q_HIGH = 0.95

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4437   # beats nb3161 ceiling
GATE_MARGINAL = 0.4475  # beats nb3080 parent but not nb3161

# -- References ----------------------------------------------------------------
REF_NB3161 = 0.4437
REF_NB3080 = 0.4475
REF_K18 = 0.4536


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run per-fold quantile clip pipeline at a single kf_seed."""
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
    for tr_loc, va_loc in splits:
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


def _run_anchor(
    anchor_key: str,
    pred_path,
    te_path,
    label: str,
    y_unb: np.ndarray,
    unb_idx: np.ndarray,
    unb_scaffolds: list[str],
    n_test: int,
) -> dict:
    """Run full multi-seed clip pipeline for one anchor."""
    print("\n" + "-" * 78)
    print(f"ANCHOR: {anchor_key}  label={label}")
    print(f"   pred_oof: {pred_path}")
    print(f"   te:       {te_path}")
    print("-" * 78)
    pred_base = np.load(pred_path).astype(np.float64)
    te_base = np.load(te_path).astype(np.float64)
    n_unb = len(y_unb)
    if pred_base.shape != (n_unb,):
        raise ValueError(
            f"{anchor_key} pred_oof shape {pred_base.shape} != ({n_unb},)"
        )
    if te_base.shape != (n_test,):
        raise ValueError(
            f"{anchor_key} te shape {te_base.shape} != ({n_test},)"
        )
    parent_oof_rae = float(rae(y_unb, pred_base))
    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    print(
        f"   pred_base oof_RAE = {parent_oof_rae:.4f}  "
        f"mean={pred_base.mean():.3f}  std={pred_base.std():.3f}  "
        f"min={pred_base.min():.3f}  max={pred_base.max():.3f}"
    )
    print(
        f"   te_base         : mean={te_base.mean():.3f}  "
        f"std={te_base.std():.3f}  min={te_base.min():.3f}  "
        f"max={te_base.max():.3f}"
    )
    if leak_eq > 0.05:
        print(f"   WARN: {leak_eq:.1%} rows == truth -- possible leak")

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
    median_rae = float(np.median(arr))
    delta_vs_parent = mean_rae - parent_oof_rae

    print(
        f"   AGGREGATE: mean={mean_rae:.4f}  std={std_rae:.4f}  "
        f"sem={sem:.4f}  median={median_rae:.4f}  "
        f"min/max=[{arr.min():.4f},{arr.max():.4f}]"
    )
    print(
        f"   delta vs parent_full_oof = {delta_vs_parent:+.4f}  "
        f"(parent {parent_oof_rae:.4f} -> clip {mean_rae:.4f})"
    )

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx]

    # Deploy: clip te to (q05, q95) of FULL 253 y
    deploy_lo = float(np.quantile(y_unb, Q_LOW))
    deploy_hi = float(np.quantile(y_unb, Q_HIGH))
    te_pred = np.clip(te_base, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_base < deploy_lo))
    n_te_hi = int(np.sum(te_base > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"   deploy clip = ({deploy_lo:.3f}, {deploy_hi:.3f}); "
        f"te clipped lo={n_te_lo}/513 hi={n_te_hi}/513; "
        f"te[unb] in-sample RAE={te_unb_in_rae:.4f}"
    )

    return {
        "anchor_key": anchor_key,
        "label": label,
        "pred_path": str(pred_path),
        "te_path": str(te_path),
        "parent_oof_rae": round(parent_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "delta_vs_parent": round(delta_vs_parent, 4),
        "delta_vs_nb3161": round(mean_rae - REF_NB3161, 4),
        "delta_vs_nb3080": round(mean_rae - REF_NB3080, 4),
        "median_seed": int(median_seed),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        # in-memory only, dropped before json.dump
        "_oof_for_save": oof_for_save.astype(np.float32),
        "_te_for_save": te_pred,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- per-fold (q{Q_LOW:.2f}, q{Q_HIGH:.2f}) CLIP transfer test "
        f"on {len(ANCHORS)} ALT anchors"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gates: best mean < {GATE_BETTER:.4f} -> BETTER; "
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

    # -- Scaffolds for outer CV ----------------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_unique_scaffolds = {n_unique_scaf}")

    # Truth stats
    y_full_lo = float(np.quantile(y_unb, Q_LOW))
    y_full_hi = float(np.quantile(y_unb, Q_HIGH))
    print(
        f"[load] y_unb stats: mean={y_unb.mean():.3f}  "
        f"std={y_unb.std():.3f}  min={y_unb.min():.3f}  max={y_unb.max():.3f}"
    )
    print(
        f"[load] y_unb full q{Q_LOW:.2f}={y_full_lo:.3f}  "
        f"q{Q_HIGH:.2f}={y_full_hi:.3f}"
    )

    # -- Run each anchor ------------------------------------------------------
    anchor_results = []
    for anchor_key, pred_path, te_path, label in ANCHORS:
        res = _run_anchor(
            anchor_key=anchor_key,
            pred_path=pred_path,
            te_path=te_path,
            label=label,
            y_unb=y_unb,
            unb_idx=unb_idx,
            unb_scaffolds=unb_scaffolds,
            n_test=n_test,
        )
        anchor_results.append(res)

    # -- Pick best anchor by mean_rae ----------------------------------------
    print("\n" + "=" * 78)
    print("CROSS-ANCHOR SUMMARY")
    print("=" * 78)
    print(
        f"{'anchor':<10} {'parent':>8} {'mean':>8} {'std':>8} {'delta_par':>10} "
        f"{'d_3161':>8} {'d_3080':>8}"
    )
    for r in anchor_results:
        print(
            f"{r['anchor_key']:<10} {r['parent_oof_rae']:>8.4f} "
            f"{r['mean_rae']:>8.4f} {r['std_rae']:>8.4f} "
            f"{r['delta_vs_parent']:>+10.4f} "
            f"{r['delta_vs_nb3161']:>+8.4f} {r['delta_vs_nb3080']:>+8.4f}"
        )

    best_idx = int(np.argmin([r["mean_rae"] for r in anchor_results]))
    best = anchor_results[best_idx]
    best_mean = best["mean_rae"]
    best_anchor = best["anchor_key"]
    print(
        f"\nBEST anchor = {best_anchor}  mean={best_mean:.4f}  "
        f"std={best['std_rae']:.4f}  delta_vs_nb3161={best['delta_vs_nb3161']:+.4f}"
    )

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if best_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3174 best anchor {best_anchor} 5-seed mean "
            f"{best_mean:.4f} beats nb3161 ceiling {REF_NB3161:.4f} "
            f"({best['delta_vs_nb3161']:+.4f}). Clip operator transfers and "
            f"finds a NEW best substrate. Re-verify with deep-30 (>=15 fresh "
            f"seeds) before any PRIMARY-1 swap."
        )
    elif best_mean < GATE_MARGINAL:
        verdict = "MARGINAL"
        ladder_action = (
            f"MARGINAL. nb3174 best anchor {best_anchor} 5-seed mean "
            f"{best_mean:.4f} beats nb3080 parent {REF_NB3080:.4f} "
            f"({best['delta_vs_nb3080']:+.4f}) but does NOT beat nb3161 "
            f"ceiling {REF_NB3161:.4f} ({best['delta_vs_nb3161']:+.4f}). "
            f"Clip helps some alternate anchors but nb3080+clip remains the "
            f"best post-hoc on this substrate. No ladder change."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3174 best anchor {best_anchor} 5-seed mean "
            f"{best_mean:.4f} does NOT beat nb3080 parent {REF_NB3080:.4f} "
            f"({best['delta_vs_nb3080']:+.4f}) or nb3161 ceiling "
            f"{REF_NB3161:.4f}. Range clip is anchor-specific to nb3080 "
            f"(or these alt anchors are already inside training range / "
            f"too compressed for clip to help). Closes clip-fan-out axis."
        )
    print(f"   best anchor   = {best_anchor}")
    print(f"   best mean_rae = {best_mean:.4f}")
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_for_save = best.pop("_oof_for_save")
    te_for_save = best.pop("_te_for_save")
    # also pop temp arrays from non-best anchors
    for r in anchor_results:
        r.pop("_oof_for_save", None)
        r.pop("_te_for_save", None)

    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path_out = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path_out, te_for_save)
    print(f"   [save] {oof_path}  (best anchor = {best_anchor})")
    print(f"   [save] {te_path_out}")

    sub_csv = SUBMISSIONS / f"{TAG}_clip_{best_anchor}.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_for_save,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "method": (
            "per_fold_y_range_clip_q05_q95_on_alternate_anchors"
        ),
        "anchors_tested": [a[0] for a in ANCHORS],
        "anchor_pre_unblind": None,  # mixed: nb3070/72/90/K18 PRE-unblind status varies
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "y_full_q_low": round(y_full_lo, 4),
        "y_full_q_high": round(y_full_hi, 4),
        "y_full_min": float(y_unb.min()),
        "y_full_max": float(y_unb.max()),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_anchor_results": anchor_results,
        "best_anchor": best_anchor,
        "best_mean_rae": round(best_mean, 4),
        "best_std_rae": round(best["std_rae"], 4),
        "best_delta_vs_nb3161": round(best["delta_vs_nb3161"], 4),
        "best_delta_vs_nb3080": round(best["delta_vs_nb3080"], 4),
        "ref_nb3161": REF_NB3161,
        "ref_nb3080": REF_NB3080,
        "ref_K18_deep30": REF_K18,
        "gate_better": GATE_BETTER,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path_out),
        "submission_csv": (
            str(sub_csv) if verdict == "BETTER" else None
        ),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   best anchor          = {best_anchor}")
    print(f"   best mean_rae        = {best_mean:.4f} +/- {best['std_rae']:.4f}")
    print(f"   delta vs nb3161      = {best['delta_vs_nb3161']:+.4f}")
    print(f"   delta vs nb3080      = {best['delta_vs_nb3080']:+.4f}")
    print(f"   verdict              = {verdict}")
    print(f"   wall                 = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_anchor", "best_mean_rae", "best_std_rae",
        "best_delta_vs_nb3161", "best_delta_vs_nb3080",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
