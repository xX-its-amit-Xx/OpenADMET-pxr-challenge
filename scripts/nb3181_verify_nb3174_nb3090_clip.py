"""nb3181 -- 15-seed wide-seed VERIFY of nb3174 (nb3090 + per-fold q05/q95 clip).

CONTEXT
    nb3174 anchor=nb3090 5-seed mean = 0.4427 +/- 0.0013 (kf_seeds 1156..1160)
    nb3161 anchor=nb3080 15-seed mean = 0.4437 +/- 0.0017  <- prior CEILING
    nb3080 15-seed mean               = 0.4475

    Per project rules (cycle-149 nb1191 wide-seed precedent + cycle-160
    deep-30 standard): 5-seed gate-A passes at exceptional magnitudes
    (>+1.5 sigma) require >=15-seed re-verification BEFORE promote.
    nb3174 best (nb3090) sits 0.4427 vs 0.4437 ceiling -- exactly the
    case the rule covers. Re-verify on FRESH seeds {1171..1185}.

PROTOCOL
    pred_base = nb3090_pred_oof  (253,)
    te_base   = te_nb3090.npy    (513,)
    For each kf_seed in {1171..1185}:
        scaffold_kfold_indices(unb_scaffolds, n_splits=5, shuffle=True, seed=s)
        Per fold:
            lo = quantile(y[tr_loc], 0.05)
            hi = quantile(y[tr_loc], 0.95)
            oof_clip[va_loc] = clip(pred_base[va_loc], lo, hi)
        pooled_rae across 5 folds.
    Aggregate: mean, std, sem, median across 15 seeds.

DEPLOY
    deploy_lo = quantile(y_unb FULL 253, 0.05)
    deploy_hi = quantile(y_unb FULL 253, 0.95)
    te_clipped = clip(te_nb3090, deploy_lo, deploy_hi)
    Submission CSV columns: SMILES, Molecule Name, pEC50.

GATES
    mean < 0.4437 -> "VERIFIED_NEW_PRIMARY1"  (beats nb3161 deep-15 ceiling)
    mean < 0.4475 -> "MARGINAL"               (beats nb3080 parent only)
    else          -> "FAIL"

INPUTS
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy
    data/processed/te_nb3090.npy

OUTPUTS
    data/processed/nb3181_summary.json
    data/processed/nb3181_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3181.npy         (513,) float32 -- deploy clip
    submissions/nb3181_deploy_nb3174_clip_nb3090.csv  (always written)
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

TAG = "nb3181"

# -- Anchor under verification -------------------------------------------------
ANCHOR_KEY = "nb3090"
PRED_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3090.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1171, 1186))  # 15 FRESH seeds {1171..1185}

# -- Clip quantiles (FIXED, mirror nb3161/nb3174) -----------------------------
Q_LOW = 0.05
Q_HIGH = 0.95

# -- Gates ---------------------------------------------------------------------
GATE_VERIFIED = 0.4437   # beats nb3161 deep-15 ceiling
GATE_MARGINAL = 0.4475   # beats nb3080 parent but not nb3161

# -- References ----------------------------------------------------------------
REF_NB3174_5SEED = 0.4427   # the value being verified
REF_NB3161_15SEED = 0.4437  # prior ceiling
REF_NB3080_15SEED = 0.4475  # parent
REF_NB3090_PARENT = 0.4470  # nb3090 parent oof_rae (per nb3174 summary)


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Single-seed per-fold q05/q95 clip pipeline. Pooled RAE across 5 folds."""
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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- WIDE-SEED VERIFY nb3174: anchor={ANCHOR_KEY} + per-fold "
        f"(q{Q_LOW:.2f},q{Q_HIGH:.2f}) CLIP"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} FRESH "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gates: mean < {GATE_VERIFIED:.4f} -> VERIFIED_NEW_PRIMARY1; "
        f"< {GATE_MARGINAL:.4f} -> MARGINAL; else FAIL"
    )
    print(
        f"          ref: nb3174 5-seed = {REF_NB3174_5SEED:.4f} "
        f"(value under verification)"
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

    # -- Load anchor ---------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"ANCHOR: {ANCHOR_KEY}")
    print(f"   pred_oof: {PRED_PATH}")
    print(f"   te:       {TE_PATH}")
    print("-" * 78)
    pred_base = np.load(PRED_PATH).astype(np.float64)
    te_base = np.load(TE_PATH).astype(np.float64)
    if pred_base.shape != (n_unb,):
        raise ValueError(
            f"{ANCHOR_KEY} pred_oof shape {pred_base.shape} != ({n_unb},)"
        )
    if te_base.shape != (n_test,):
        raise ValueError(
            f"{ANCHOR_KEY} te shape {te_base.shape} != ({n_test},)"
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

    # -- Run 15 fresh seeds --------------------------------------------------
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
    min_rae = float(arr.min())
    max_rae = float(arr.max())

    print("\n" + "-" * 78)
    print("AGGREGATE (15 fresh seeds)")
    print("-" * 78)
    print(
        f"   mean    = {mean_rae:.4f}  std = {std_rae:.4f}  sem = {sem:.4f}"
    )
    print(
        f"   median  = {median_rae:.4f}  min = {min_rae:.4f}  max = {max_rae:.4f}"
    )
    print(
        f"   delta vs nb3174 5-seed  ({REF_NB3174_5SEED:.4f}) = "
        f"{mean_rae - REF_NB3174_5SEED:+.4f}"
    )
    print(
        f"   delta vs nb3161 15-seed ({REF_NB3161_15SEED:.4f}) = "
        f"{mean_rae - REF_NB3161_15SEED:+.4f}"
    )
    print(
        f"   delta vs nb3080 15-seed ({REF_NB3080_15SEED:.4f}) = "
        f"{mean_rae - REF_NB3080_15SEED:+.4f}"
    )
    print(
        f"   delta vs nb3090 parent  ({REF_NB3090_PARENT:.4f}) = "
        f"{mean_rae - REF_NB3090_PARENT:+.4f}"
    )

    # Median-seed OOF (for storage; rank-stable)
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)

    # -- Deploy clip on FULL 253 truth quantiles -----------------------------
    deploy_lo = float(np.quantile(y_unb, Q_LOW))
    deploy_hi = float(np.quantile(y_unb, Q_HIGH))
    te_pred = np.clip(te_base, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_base < deploy_lo))
    n_te_hi = int(np.sum(te_base > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print("\n" + "-" * 78)
    print("DEPLOY (te clip on full-253 truth quantiles)")
    print("-" * 78)
    print(
        f"   deploy clip = ({deploy_lo:.3f}, {deploy_hi:.3f}); "
        f"te clipped lo={n_te_lo}/513 hi={n_te_hi}/513"
    )
    print(
        f"   te stats:   mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(
        f"   te[unb] IN-SAMPLE RAE = {te_unb_in_rae:.4f}  "
        f"(expected << pred_oof RAE; deploy-refit on FULL 253 truth)"
    )

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_VERIFIED:
        verdict = "VERIFIED_NEW_PRIMARY1"
        ladder_action = (
            f"VERIFIED_NEW_PRIMARY1. nb3181 15-seed mean {mean_rae:.4f} +/- "
            f"{std_rae:.4f} beats nb3161 deep-15 ceiling {REF_NB3161_15SEED:.4f} "
            f"({mean_rae - REF_NB3161_15SEED:+.4f}). nb3174 5-seed result "
            f"({REF_NB3174_5SEED:.4f}) REPRODUCES on 15 fresh seeds. "
            f"Promote nb3181 (nb3090 + per-fold q05/q95 clip) to PRIMARY-1; "
            f"demote nb3161 (nb3080+clip) to alt. Cycle-149 wide-seed precedent "
            f"satisfied. Predicted LB under +0.0045 PRE-unblind delta calibration "
            f"= ~{mean_rae + 0.0045:.4f}."
        )
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL"
        ladder_action = (
            f"MARGINAL. nb3181 15-seed mean {mean_rae:.4f} beats nb3080 parent "
            f"{REF_NB3080_15SEED:.4f} ({mean_rae - REF_NB3080_15SEED:+.4f}) but "
            f"does NOT beat nb3161 deep-15 ceiling {REF_NB3161_15SEED:.4f} "
            f"({mean_rae - REF_NB3161_15SEED:+.4f}). nb3174 5-seed "
            f"{REF_NB3174_5SEED:.4f} was a lucky-batch shift -- expected "
            f"cycle-149 dynamic where 5-seed mean reverts to the true band on "
            f"15-seed re-verification. Keep nb3161 PRIMARY-1, nb3181 as alt."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"FAIL. nb3181 15-seed mean {mean_rae:.4f} does NOT beat nb3080 "
            f"parent {REF_NB3080_15SEED:.4f} or nb3161 ceiling "
            f"{REF_NB3161_15SEED:.4f}. nb3174 5-seed {REF_NB3174_5SEED:.4f} "
            f"was lucky-seed noise (nb1086 pattern). Close clip-on-nb3090 axis; "
            f"keep nb3161 PRIMARY-1."
        )
    print(f"   mean_rae      = {mean_rae:.4f}  std = {std_rae:.4f}")
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path_out = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path_out, te_pred)
    print(f"   [save] {oof_path}  (median-seed OOF, seed={median_seed})")
    print(f"   [save] {te_path_out}")

    sub_csv = SUBMISSIONS / f"{TAG}_deploy_nb3174_clip_{ANCHOR_KEY}.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": (
            f"wide_seed_verify_nb3174__{ANCHOR_KEY}_per_fold_clip_q05_q95"
        ),
        "anchor_key": ANCHOR_KEY,
        "pred_path": str(PRED_PATH),
        "te_path": str(TE_PATH),
        "parent_oof_rae": round(parent_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "anchor_pre_unblind": None,  # nb3090 PRE/POST status: see nb3090 summary
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "y_full_q_low": round(y_full_lo, 4),
        "y_full_q_high": round(y_full_hi, 4),
        "y_full_min": float(y_unb.min()),
        "y_full_max": float(y_unb.max()),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(min_rae, 4),
        "max_rae": round(max_rae, 4),
        "median_seed": int(median_seed),
        "delta_vs_nb3174_5seed": round(mean_rae - REF_NB3174_5SEED, 4),
        "delta_vs_nb3161_15seed": round(mean_rae - REF_NB3161_15SEED, 4),
        "delta_vs_nb3080_15seed": round(mean_rae - REF_NB3080_15SEED, 4),
        "delta_vs_nb3090_parent": round(mean_rae - REF_NB3090_PARENT, 4),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "ref_nb3174_5seed": REF_NB3174_5SEED,
        "ref_nb3161_15seed": REF_NB3161_15SEED,
        "ref_nb3080_15seed": REF_NB3080_15SEED,
        "ref_nb3090_parent": REF_NB3090_PARENT,
        "gate_verified": GATE_VERIFIED,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path_out),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor               = {ANCHOR_KEY}")
    print(f"   n_seeds              = {len(KF_SEEDS)}  ({KF_SEEDS[0]}..{KF_SEEDS[-1]})")
    print(f"   mean_rae             = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   delta vs nb3174      = {mean_rae - REF_NB3174_5SEED:+.4f}")
    print(f"   delta vs nb3161      = {mean_rae - REF_NB3161_15SEED:+.4f}")
    print(f"   verdict              = {verdict}")
    print(f"   wall                 = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "anchor_key", "n_seeds", "mean_rae", "std_rae", "median_rae",
        "delta_vs_nb3174_5seed", "delta_vs_nb3161_15seed",
        "delta_vs_nb3080_15seed", "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
