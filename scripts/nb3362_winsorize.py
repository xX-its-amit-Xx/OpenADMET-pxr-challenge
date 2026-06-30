"""nb3362 -- Per-fold WINSORIZE (vs clip) on nb3090 pred_oof.

NEW PARADIGM -- WINSORIZE on the PREDICTION distribution:
    Winsorization replaces the extreme tails of a distribution with the
    boundary VALUES at chosen quantiles. Here the boundaries are computed
    from the *PREDICTION* distribution of fold-train -- NOT the y / truth
    distribution. This is the subtle but load-bearing difference from
    nb3161-family y-range clipping:

      * nb3161 (y-range clip):  lo = quantile(y[tr],   q05)
                                hi = quantile(y[tr],   q95)
      * nb3362 (winsorize):     lo = quantile(pred[tr], q05)
                                hi = quantile(pred[tr], q95)

    Then fold-val predictions are np.clip()-ped to (lo, hi). Because the
    anchor nb3090 is a variance-COMPRESSED blend (pred_std ~0.90 vs
    truth_std ~1.0 on novel-scaffold OOD), pegging the extreme 5% of
    predictions to the prediction-quantile boundary tames over-confident
    tail rows whose magnitude the blend cannot justify, without referencing
    truth at all (truth-blind, so the deploy band transfers honestly).

    Hypothesis: prediction-quantile winsorization is a strictly post-hoc,
    truth-blind, 1-parameter-per-side transform that may shave RAE on the
    over-extended tails of nb3090 -> potential push under the 0.4423 gate.

PROTOCOL (per kf_seed, 5-fold scaffold split, anchor LOADED no rebuild):
    pred_base = nb3090_pred_oof  (253,)  -- finer-grid q-cut best blend
    Per outer fold:
        a) lo = quantile(pred_base[fold_train], 0.05)   (PREDICTION dist)
           hi = quantile(pred_base[fold_train], 0.95)   (PREDICTION dist)
        b) val_pred = np.clip(pred_base[fold_val], lo, hi)
        c) stitch into oof_wins (253,); fold-val RAE recorded.
    Honest gate metric = PER-FOLD-MEAN (mean of the 5 fold-val RAEs),
    averaged over 15 FRESH kf_seeds {1216..1230} (disjoint from nb3090
    grid {1126..1140} and all prior post-hoc verifications).

GATE (on 15-seed mean of the per-fold-mean metric):
    pf_mean < 0.4423 -> "BETTER"
        (prediction-quantile winsorize extracts real gain past the
         nb3190 0.4422 / nb3173 0.4437 post-hoc band on this anchor)
    else            -> "FAIL"
        (winsorize is NEUTRAL or HURTS; nb3090 prediction tails are
         already well-calibrated -- keep prior PRIMARY-1)

References:
    nb3090 best-combo 15-seed mean  = 0.4472 +/- 0.0008  <- parent anchor
    nb3090 full-253 OOF RAE         = 0.4470
    nb3190 learned-clip post-hoc    = 0.4422
    nb3173 post-hoc ref             = 0.4437
    nb3080 wide-seed ceiling        = 0.4475
    nb2960 K18 deep-30 OOF          = 0.4536
    nb3000 K19 deep-30 OOF          = 0.4607
    nb2171 prior post-hoc top       = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy
    data/processed/te_nb3090.npy

Outputs:
    data/processed/nb3362_summary.json
    data/processed/nb3362_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3362.npy         (513,) float32 -- deploy te
    submissions/nb3362_winsorize.csv     (only on BETTER verdict)
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

TAG = "nb3362"
PARENT_TAG = "nb3090"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3090.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Winsorize quantiles (FIXED, on the PREDICTION distribution) ---------------
Q_LOW = 0.05
Q_HIGH = 0.95

# -- Gate ----------------------------------------------------------------------
GATE_BETTER = 0.4423  # per-fold-mean < this -> BETTER, else FAIL

# -- References ----------------------------------------------------------------
REF_NB3090 = 0.4472
REF_NB3090_STD = 0.0008
REF_NB3090_FULL_OOF = 0.4470
REF_NB3190 = 0.4422
REF_NB3173 = 0.4437
REF_NB3080 = 0.4475
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run per-fold prediction-quantile winsorize at a single kf_seed.

    pred_base : (253,) parent nb3090 OOF predictions (constant across seeds).
    y_unb     : (253,) truth labels (used ONLY for RAE scoring, never for the
                winsorize boundaries -- boundaries come from pred_base[tr]).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_wins = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_lo = []
    fold_hi = []
    fold_clipped_lo = []
    fold_clipped_hi = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # WINSORIZE boundaries from the PREDICTION distribution of fold-train.
        lo = float(np.quantile(pred_base[tr_loc], Q_LOW))
        hi = float(np.quantile(pred_base[tr_loc], Q_HIGH))
        fold_lo.append(lo)
        fold_hi.append(hi)
        val_pred = pred_base[va_loc]
        n_lo = int(np.sum(val_pred < lo))
        n_hi = int(np.sum(val_pred > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)
        winsorized = np.clip(val_pred, lo, hi)
        oof_wins[va_loc] = winsorized
        fold_val_raes.append(float(rae(y_unb[va_loc], winsorized)))

    if np.isnan(oof_wins).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_wins))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "n_clipped_lo": int(np.sum(fold_clipped_lo)),
        "n_clipped_hi": int(np.sum(fold_clipped_hi)),
        "oof": oof_wins,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- WINSORIZE per-fold on PREDICTION quantiles "
        f"(q{Q_LOW:.2f}, q{Q_HIGH:.2f}) of {PARENT_TAG} pred_oof"
    )
    print(
        f"          (boundaries from PREDICTION dist of fold-train, "
        f"NOT y -- subtle diff vs nb3161 y-range clip)"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}  (disjoint from {PARENT_TAG} grid)"
    )
    print(
        f"          {PARENT_TAG} reference = "
        f"{REF_NB3090:.4f} +/- {REF_NB3090_STD:.4f}"
    )
    print(
        f"          honest gate metric = PER-FOLD-MEAN"
    )
    print(
        f"          gate: pf_mean < {GATE_BETTER:.4f} -> BETTER, else FAIL"
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

    # -- Load nb3090 anchor pred_oof + te -------------------------------------
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

    # Prediction-quantile + truth stats (full 253, reference only)
    pred_full_lo = float(np.quantile(pred_base, Q_LOW))
    pred_full_hi = float(np.quantile(pred_base, Q_HIGH))
    print(
        f"   y_unb stats: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
        f"min={y_unb.min():.3f}  max={y_unb.max():.3f}"
    )
    print(
        f"   pred_base full quantiles: q{Q_LOW:.2f}={pred_full_lo:.3f}  "
        f"q{Q_HIGH:.2f}={pred_full_hi:.3f}  (PREDICTION distribution)"
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
    pf_means = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        pf_means.append(res["per_fold_val_rae_mean"])
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
            f"   kf={s}: pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"pooled={res['pooled_rae']:.4f}  "
            f"lo={res['fold_lo_mean']:.2f}  hi={res['fold_hi_mean']:.2f}  "
            f"wins(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    # Honest gate metric = PER-FOLD-MEAN across seeds
    pf_arr = np.asarray(pf_means, dtype=np.float64)
    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(pf_arr)
    mean_rae = float(pf_arr.mean())          # gate metric
    std_rae = float(pf_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(pf_arr))
    pooled_mean = float(pooled_arr.mean())

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds) -- gate metric = PER-FOLD-MEAN")
    print("-" * 78)
    print(f"   pf_mean (gate)  = {mean_rae:.4f}")
    print(f"   std             = {std_rae:.4f}")
    print(f"   sem             = {sem:.4f}")
    print(f"   95% CI          = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median          = {median_rae:.4f}")
    print(f"   min/max         = [{pf_arr.min():.4f}, {pf_arr.max():.4f}]")
    print(f"   pooled_mean     = {pooled_mean:.4f}  (reference, not gated)")
    print(
        f"\n   ref {PARENT_TAG} 15-seed mean    = "
        f"{REF_NB3090:.4f} +/- {REF_NB3090_STD:.4f}"
    )
    print(
        f"   delta vs {PARENT_TAG}            = "
        f"{mean_rae - REF_NB3090:+.4f}"
    )
    print(f"   ref nb3190 post-hoc ceiling    = {REF_NB3190:.4f}")
    print(f"   delta vs nb3190                = {mean_rae - REF_NB3190:+.4f}")

    # -- Deploy: winsorize te to (q05, q95) of FULL 253 PREDICTION dist -------
    deploy_lo = float(np.quantile(pred_base, Q_LOW))
    deploy_hi = float(np.quantile(pred_base, Q_HIGH))
    te_pred = np.clip(te_base, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_base < deploy_lo))
    n_te_hi = int(np.sum(te_base > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy winsorize band = ({deploy_lo:.3f}, {deploy_hi:.3f}) "
        f"from full 253 PREDICTION quantiles"
    )
    print(
        f"   te winsorized: lo={n_te_lo}/513  hi={n_te_hi}/513  "
        f"total={n_te_lo + n_te_hi}/513"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage (median by gate metric pf_mean)
    med_seed_idx = int(np.argsort(pf_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (pf_mean={pf_arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3362 15-seed per-fold-mean {mean_rae:.4f} "
            f"beats gate {GATE_BETTER:.4f} and parent nb3090 "
            f"{REF_NB3090:.4f} ({mean_rae - REF_NB3090:+.4f}). Per-fold "
            f"prediction-quantile winsorize (q{Q_LOW:.2f}, q{Q_HIGH:.2f}) "
            f"yields real gain on top of nb3090 -- truth-blind, so deploy "
            f"band transfers honestly. Re-verify with deep-30 before any "
            f"PRIMARY-1 swap (cycle-160 rule)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3362 15-seed per-fold-mean {mean_rae:.4f} does NOT "
            f"beat gate {GATE_BETTER:.4f} (delta vs parent nb3090 "
            f"{mean_rae - REF_NB3090:+.4f}). Per-fold prediction-quantile "
            f"winsorize at (q{Q_LOW:.2f}, q{Q_HIGH:.2f}) is NEUTRAL or HURTS "
            f"on this anchor (nb3090 prediction tails already calibrated; "
            f"30-seed averaging in the K18/K19 deep ensembles already did "
            f"the variance compression that post-hoc winsorize targets). "
            f"Keep nb3090 / prior PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_winsorize.csv"
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
            "per_fold_winsorize_prediction_quantiles_nb3090_pred_oof_q05_q95"
        ),
        "paradigm_note": (
            "winsorize boundaries from the PREDICTION distribution of "
            "fold-train (quantile(pred[tr], q)), NOT the y distribution "
            "(subtle diff vs nb3161 y-range clip); truth-blind transform"
        ),
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "boundary_source": "prediction_distribution_fold_train",
        "pred_full_q_low": round(pred_full_lo, 4),
        "pred_full_q_high": round(pred_full_hi, 4),
        "pred_full_min": float(pred_base.min()),
        "pred_full_max": float(pred_base.max()),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "gate_metric": "per_fold_mean",
        "seed_records": seed_records,
        "pf_mean_array": [round(float(v), 4) for v in pf_means],
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),          # gate metric (per-fold-mean)
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(pf_arr.min()), 4),
        "max_rae": round(float(pf_arr.max()), 4),
        "pooled_mean_rae": round(pooled_mean, 4),
        "ref_parent": REF_NB3090,
        "ref_parent_std": REF_NB3090_STD,
        "delta_vs_parent": round(mean_rae - REF_NB3090, 4),
        "ref_nb3090_full_oof": REF_NB3090_FULL_OOF,
        "ref_nb3190": REF_NB3190,
        "delta_vs_nb3190": round(mean_rae - REF_NB3190, 4),
        "ref_nb3173": REF_NB3173,
        "ref_nb3080": REF_NB3080,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_winsorized_lo": n_te_lo,
        "n_te_winsorized_hi": n_te_hi,
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
    print(
        f"   pf_mean ({n_s} seeds, gate) = {mean_rae:.4f} +/- {std_rae:.4f}"
    )
    print(f"   95% CI                   = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled_mean (ref)        = {pooled_mean:.4f}")
    print(f"   delta vs nb3090          = {mean_rae - REF_NB3090:+.4f}")
    print(f"   delta vs nb3190          = {mean_rae - REF_NB3190:+.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "pooled_mean_rae", "delta_vs_parent", "delta_vs_nb3190",
        "deploy_lo", "deploy_hi",
        "n_te_winsorized_lo", "n_te_winsorized_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
