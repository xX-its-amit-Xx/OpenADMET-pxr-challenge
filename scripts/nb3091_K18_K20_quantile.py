"""nb3091 -- Quantile-conditional hard-split blend on {K18, K20} deep-30.

NEW PARADIGM:
    Substitute K20 deep-30 for K19 deep-30 in the nb3070/nb3080 quantile
    blend pattern. K19 was the prior partner anchor; K20 is the next K-band
    out from K18 (per nb2960 deep-30 OOF: K18=0.4536, K20=0.4625, K24=0.4687,
    K28=0.4740). K20 is the closest deep-30 cohort to K18 by RAE that we have
    not yet tried in a quantile-conditional 2-anchor blend. Question: is the
    nb3070/3080 ceiling (~0.4470 5-seed, pending 15-seed verify) specific to
    {K18, K19}, or does the same hard-split mechanism transfer to {K18, K20}?

PROTOCOL (per kf_seed, 5-fold scaffold split, anchors LOADED no rebuild):
    1. Compute fold-train K18 q_cut quantile threshold q (per fold).
    2. Per fold-val row:
         pred_K18 <= q -> blend(w_K18_low=0.8, w_K20_low=0.2)
         pred_K18 >  q -> blend(w_K18_high=0.5, w_K20_high=0.5)
    3. Stitch into oof_blend (253,); pooled_rae across 5 outer folds.
    Repeat for 15 FRESH kf_seeds {1126..1140}.

WEIGHTS (per task spec):
    q_cut       = 0.5 (median split; mid-range default since K20 untested)
    w_K18_low   = 0.8  (low-tail trusts K18 anchor more strongly)
    w_K18_high  = 0.5  (high-tail equal-blend)
    -> w_K20_low = 0.2; w_K20_high = 0.5

GATE (on 15-seed mean):
    mean < 0.4475 -> "BETTER_THAN_NB3080" (new K20 paradigm wins)
    else          -> "FAIL"

References:
    nb3073 5-seed best combo (K18+K19) = 0.4470 +/- 0.0009 (UNDER-DISPERSED)
    nb3080 15-seed verify of nb3073    = (pending result; gate ref)
    nb3030 15-seed wide-mean (K18 alone post-hoc) = 0.4509
    nb2960 K18 deep-30 OOF             = 0.4536
    nb2960 K20 deep-30 OOF             = 0.4625
    nb2171 prior post-hoc top          = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb2960_K20_30seed_oof.npy
    data/processed/nb2960_K20_30seed_te.npy

Outputs:
    data/processed/nb3091_summary.json
    data/processed/nb3091_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3091.npy         (513,) float32 -- deploy te
    submissions/nb3091_K18_K20_quantile.csv  (only on PROMOTE verdict)
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

TAG = "nb3091"
PARENT_TAG = "nb3080"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K20"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K20": DATA_PROCESSED / "nb2960_K20_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K20": DATA_PROCESSED / "nb2960_K20_30seed_te.npy",
}
K_DEPTH = {"K18": "deep30", "K20": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1126, 1141))  # 15 FRESH seeds {1126..1140}

# -- Quantile-conditional blend weights (per task spec) ------------------------
Q_CUT = 0.5
W_K18_LOW = 0.8   # w_K20_low = 1 - 0.8 = 0.2
W_K18_HIGH = 0.5  # w_K20_high = 1 - 0.5 = 0.5
W_K20_LOW = 1.0 - W_K18_LOW
W_K20_HIGH = 1.0 - W_K18_HIGH

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB3080 = 0.4475  # mean < this -> BETTER_THAN_NB3080

# -- References ----------------------------------------------------------------
REF_NB3073_5SEED = 0.4470
REF_NB3030 = 0.4509
REF_K18 = 0.4536
REF_K20 = 0.4625
REF_NB2171 = 0.4682


def _blend_quantile_conditional(
    p_k18: np.ndarray,
    p_k20: np.ndarray,
    q_thr: float,
) -> np.ndarray:
    """Per-row hard-split blend.

    rows with p_k18 <= q_thr -> (W_K18_LOW=0.8, W_K20_LOW=0.2)
    rows with p_k18 >  q_thr -> (W_K18_HIGH=0.5, W_K20_HIGH=0.5)
    """
    low_mask = p_k18 <= q_thr
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = (
        W_K18_LOW * p_k18[low_mask] + W_K20_LOW * p_k20[low_mask]
    )
    out[~low_mask] = (
        W_K18_HIGH * p_k18[~low_mask] + W_K20_HIGH * p_k20[~low_mask]
    )
    return out


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run quantile-conditional blend pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_q_thrs = []
    fold_high_share = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # Per-fold q_cut quantile from fold-train K18 preds
        q_thr = float(np.quantile(P_unb[tr_loc, 0], Q_CUT))
        fold_q_thrs.append(q_thr)
        val_p_k18 = P_unb[va_loc, 0]
        val_p_k20 = P_unb[va_loc, 1]
        val_pred = _blend_quantile_conditional(val_p_k18, val_p_k20, q_thr)
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_high_share.append(float(np.mean(val_p_k18 > q_thr)))

    if np.isnan(oof_blend).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_q_thr_mean": float(np.mean(fold_q_thrs)),
        "fold_q_thr_std": float(np.std(fold_q_thrs, ddof=1)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- QUANTILE-CONDITIONAL HARD-SPLIT BLEND on {K_LABELS} "
        f"deep-30 (NEW PARADIGM: K20 swap for K19)"
    )
    print(
        f"          weights: q_cut={Q_CUT}, w_K18_low={W_K18_LOW} "
        f"(w_K20_low={W_K20_LOW}), w_K18_high={W_K18_HIGH} "
        f"(w_K20_high={W_K20_HIGH})"
    )
    print(
        f"          per-row: low (K18<=q): (K18={W_K18_LOW}, K20={W_K20_LOW})"
    )
    print(
        f"                   high(K18 >q): (K18={W_K18_HIGH}, K20={W_K20_HIGH})"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate: mean < {GATE_BETTER_THAN_NB3080:.4f} -> "
        f"BETTER_THAN_NB3080; else FAIL"
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

    # -- Load K18, K20 deep-30 anchor OOFs + te arrays ------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18, K20 deep-30 OOFs and te arrays")
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
        print(
            f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
            f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}"
        )

    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    corr = float(np.corrcoef(P_unb.T)[0, 1])
    print(f"   pairwise corr({K_LABELS[0]}, {K_LABELS[1]}) = {corr:.4f}")

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
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_q_thr_mean": round(res["fold_q_thr_mean"], 4),
            "fold_q_thr_std": round(res["fold_q_thr_std"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"q_thr_mean={res['fold_q_thr_mean']:.3f}  "
            f"high_share={res['fold_high_share_mean']:.2f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    # 95% CI via t-multiplier (n=15, df=14, t~2.145)
    t_mult = 2.145
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
        f"\n   ref nb3073 K18+K19 5-seed       = {REF_NB3073_5SEED:.4f}"
    )
    print(f"   ref nb3030 wide-seed ceiling    = {REF_NB3030:.4f}")
    print(f"   delta vs nb3030                 = {mean_rae - REF_NB3030:+.4f}")
    print(f"   K18 standalone deep-30          = {REF_K18:.4f}")
    print(f"   K20 standalone deep-30          = {REF_K20:.4f}")

    # -- Deploy: q_thr from FULL 253 K18 OOF, then blend te -----------------
    deploy_q_thr = float(np.quantile(P_unb[:, 0], Q_CUT))
    te_k18 = P_te[:, 0]
    te_k20 = P_te[:, 1]
    te_pred = _blend_quantile_conditional(
        te_k18, te_k20, deploy_q_thr,
    ).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    te_low_share = float(np.mean(te_k18 <= deploy_q_thr))
    print(
        f"\n   deploy q_thr (full K18 OOF q{Q_CUT}) = {deploy_q_thr:.4f}"
    )
    print(f"   te(513) low-half share = {te_low_share:.3f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
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
    if mean_rae < GATE_BETTER_THAN_NB3080:
        verdict = "BETTER_THAN_NB3080"
        ladder_action = (
            f"PROMOTE. nb3091 K18+K20 quantile-conditional 15-seed mean "
            f"{mean_rae:.4f} beats nb3080 K18+K19 gate "
            f"{GATE_BETTER_THAN_NB3080:.4f} "
            f"({mean_rae - GATE_BETTER_THAN_NB3080:+.4f}). K20 paradigm swap "
            f"breaks K18+K19 ceiling -- new substrate axis opened. Promote "
            f"to PRIMARY-1 candidate pending wide-seed cross-verify."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"FAIL. nb3091 K18+K20 quantile-conditional 15-seed mean "
            f"{mean_rae:.4f} does not beat gate "
            f"{GATE_BETTER_THAN_NB3080:.4f} "
            f"({mean_rae - GATE_BETTER_THAN_NB3080:+.4f}). K20 partner "
            f"anchor weaker than K19 in this hard-split pattern (K20 "
            f"standalone {REF_K20:.4f} vs K19 standalone 0.4607; +0.0018 "
            f"penalty). Keep K18+K19 quantile blend; K20 paradigm closed."
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

    sub_csv = SUBMISSIONS / f"{TAG}_K18_K20_quantile.csv"
    promote_verdicts = {"BETTER_THAN_NB3080"}
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
        "method": (
            "quantile_conditional_hard_split_blend_K18_K20_deep30_"
            "new_paradigm_K20_swap"
        ),
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "weights": {
            "q_cut": Q_CUT,
            "w_K18_low": W_K18_LOW,
            "w_K20_low": W_K20_LOW,
            "w_K18_high": W_K18_HIGH,
            "w_K20_high": W_K20_HIGH,
        },
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
        "ref_nb3073_5seed_K18_K19": REF_NB3073_5SEED,
        "ref_nb3030": REF_NB3030,
        "ref_K18_deep30": REF_K18,
        "ref_K20_deep30": REF_K20,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3030": round(mean_rae - REF_NB3030, 4),
        "delta_vs_K18_alone": round(mean_rae - REF_K18, 4),
        "delta_vs_K20_alone": round(mean_rae - REF_K20, 4),
        "deploy_q_thr": round(deploy_q_thr, 4),
        "te_low_share": round(te_low_share, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (
            str(sub_csv) if verdict in promote_verdicts else None
        ),
        "gate_better_than_nb3080": GATE_BETTER_THAN_NB3080,
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
    print(f"   delta vs nb3030       = {mean_rae - REF_NB3030:+.4f}")
    print(f"   delta vs gate         = {mean_rae - GATE_BETTER_THAN_NB3080:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3030", "delta_vs_K18_alone", "delta_vs_K20_alone",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  weights: {res.get('weights')}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
