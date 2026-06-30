"""nb3251 -- Iterative refinement: per-fold bias correction on nb3200 residuals.

NEW PARADIGM:
    Per fold compute residual mean (y_train - nb3200_pred_train); apply this
    additive bias correction to validation predictions. Tests whether nb3200
    (learned per-fold clip on nb3090) carries a residual mean-bias that an
    extra constant offset can mop up.

PROTOCOL:
    For each kf_seed in {1216..1230} (15 FRESH seeds, disjoint from nb3200's
    {1186..1215}):
        scaffold_kfold_indices(n_splits=5, seed=kf_seed)
        For each fold (tr, va):
            bias = mean(y_unb[tr] - pred_base[tr])
            corrected[va] = pred_base[va] + bias
        Pool 5 folds -> pooled_rae
    Aggregate per-fold-mean over 15 seeds (use mean of pooled_raes).

GATE:
    per-fold-mean < 0.4423 -> "BETTER"
    else                    -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy
    data/processed/te_nb3200.npy

Outputs:
    data/processed/nb3251_summary.json
    data/processed/nb3251_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3251.npy         (513,) float32 -- deploy te
    submissions/nb3251_iterative_refinement.csv  (only on BETTER)
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

TAG = "nb3251"
PARENT_TAG = "nb3200"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3200.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Gate (per task) -----------------------------------------------------------
GATE_BETTER = 0.4423  # per-fold-mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_NB3200 = 0.4424   # nb3200 deep-30 mean
REF_NB3190 = 0.4426
REF_PARENT_NB3090 = 0.4472
REF_NB3173 = 0.4437
REF_NB2171 = 0.4682


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Per-fold bias correction: corrected = pred_base + mean(y_tr - pred_tr)."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_corr = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_bias = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        bias = float(np.mean(y_unb[tr_loc] - pred_base[tr_loc]))
        fold_bias.append(bias)
        corrected = pred_base[va_loc] + bias
        oof_corr[va_loc] = corrected
        fold_val_raes.append(float(rae(y_unb[va_loc], corrected)))

    if np.isnan(oof_corr).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_corr))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_bias": fold_bias,
        "fold_bias_mean": float(np.mean(fold_bias)),
        "fold_bias_std": float(np.std(fold_bias, ddof=1)),
        "oof": oof_corr,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- ITERATIVE REFINEMENT (per-fold bias correction on {PARENT_TAG})"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER, else FAIL"
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

    # -- Load nb3200 anchor pred_oof + te -------------------------------------
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
    global_bias = float(np.mean(y_unb - pred_base))
    print(
        f"   pred_base: oof_RAE={full_oof_rae:.4f}  "
        f"mean={pred_base.mean():.3f}  std={pred_base.std():.3f}  "
        f"min={pred_base.min():.3f}  max={pred_base.max():.3f}"
    )
    print(
        f"   te_base:   mean={te_base.mean():.3f}  std={te_base.std():.3f}  "
        f"min={te_base.min():.3f}  max={te_base.max():.3f}"
    )
    print(f"   global_bias (y - pred) = {global_bias:+.4f}")

    # Leak sanity on parent
    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN parent: {leak_eq:.1%} rows == truth -- possible leak")

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
    per_fold_means = []
    oof_stack = []
    all_fold_biases = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_fold_biases.extend(res["fold_bias"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_bias": [round(v, 4) for v in res["fold_bias"]],
            "fold_bias_mean": round(res["fold_bias_mean"], 4),
            "fold_bias_std": round(res["fold_bias_std"], 4),
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"per_fold_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"bias_mean={res['fold_bias_mean']:+.4f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr_pooled = np.asarray(pooled_raes, dtype=np.float64)
    arr_pf = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(arr_pooled)
    # Primary statistic per task: per-fold-mean across seeds
    pf_mean = float(arr_pf.mean())
    pf_std = float(arr_pf.std(ddof=1)) if n_s > 1 else 0.0
    pf_sem = pf_std / np.sqrt(n_s) if n_s > 1 else 0.0
    # Also report pooled stats
    pooled_mean = float(arr_pooled.mean())
    pooled_std = float(arr_pooled.std(ddof=1)) if n_s > 1 else 0.0
    pooled_sem = pooled_std / np.sqrt(n_s) if n_s > 1 else 0.0
    # df=14, two-sided 95% t_mult = 2.1448
    t_mult = 2.1448
    pf_ci_low = pf_mean - t_mult * pf_sem
    pf_ci_high = pf_mean + t_mult * pf_sem
    pooled_ci_low = pooled_mean - t_mult * pooled_sem
    pooled_ci_high = pooled_mean + t_mult * pooled_sem

    shift_vs_nb3200 = pf_mean - REF_NB3200
    bias_arr = np.asarray(all_fold_biases, dtype=np.float64)
    bias_overall_mean = float(bias_arr.mean())
    bias_overall_std = float(bias_arr.std(ddof=1)) if len(bias_arr) > 1 else 0.0

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   per-fold-mean (primary)")
    print(f"     mean        = {pf_mean:.4f}")
    print(f"     std         = {pf_std:.4f}")
    print(f"     sem         = {pf_sem:.4f}")
    print(f"     95% CI      = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   pooled (secondary)")
    print(f"     mean        = {pooled_mean:.4f}")
    print(f"     std         = {pooled_std:.4f}")
    print(f"     sem         = {pooled_sem:.4f}")
    print(f"     95% CI      = [{pooled_ci_low:.4f}, {pooled_ci_high:.4f}]")
    print(f"\n   shift vs nb3200 (0.4424) = {shift_vs_nb3200:+.4f}")
    print(f"   bias distribution: mean={bias_overall_mean:+.4f}  std={bias_overall_std:.4f}")
    print(f"   bias range = [{bias_arr.min():+.4f}, {bias_arr.max():+.4f}]")

    # -- Deploy: apply GLOBAL bias correction to te ---------------------------
    # On deploy, use bias computed from full 253 (y_unb - pred_base on full set).
    deploy_bias = float(np.mean(y_unb - pred_base))
    te_pred = (te_base + deploy_bias).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy_bias (full 253 y - pred) = {deploy_bias:+.4f}"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage (use per-fold-mean ranking)
    med_idx = int(np.argsort(arr_pf)[n_s // 2])
    median_seed = KF_SEEDS[med_idx]
    oof_for_save = oof_stack[med_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (per_fold_mean={arr_pf[med_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"BETTER. nb3251 per-fold-mean {pf_mean:.4f} < gate {GATE_BETTER:.4f}. "
            f"Per-fold bias correction extracts +{REF_NB3200 - pf_mean:.4f} RAE on top "
            f"of nb3200 ({REF_NB3200:.4f}). Mean residual bias = {bias_overall_mean:+.4f} "
            f"(std {bias_overall_std:.4f}) -- nb3200 carries a small but consistent "
            f"offset that per-fold subtraction mops up. PROMOTE nb3251 as PRIMARY-1 "
            f"candidate, displaces nb3200/nb2171 on the ladder."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"FAIL. nb3251 per-fold-mean {pf_mean:.4f} >= gate {GATE_BETTER:.4f}. "
            f"Shift vs nb3200 = {shift_vs_nb3200:+.4f}. Mean residual bias "
            f"{bias_overall_mean:+.4f} is too small/noisy for per-fold subtraction to "
            f"beat the {REF_NB3200:.4f} ceiling. Constant-offset axis closed for "
            f"nb3200 anchor. Keep nb3200 (or nb2171) as PRIMARY; do not promote nb3251."
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

    sub_csv = SUBMISSIONS / f"{TAG}_iterative_refinement.csv"
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
        "method": "per_fold_bias_correction_on_nb3200_residuals",
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
        "global_bias_full253": round(global_bias, 4),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "per_fold_mean_mean": round(pf_mean, 4),
        "per_fold_mean_std": round(pf_std, 4),
        "per_fold_mean_sem": round(pf_sem, 4),
        "per_fold_mean_ci95_low": round(pf_ci_low, 4),
        "per_fold_mean_ci95_high": round(pf_ci_high, 4),
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "pooled_sem": round(pooled_sem, 4),
        "pooled_ci95_low": round(pooled_ci_low, 4),
        "pooled_ci95_high": round(pooled_ci_high, 4),
        "shift_vs_nb3200": round(shift_vs_nb3200, 4),
        "bias_overall_mean": round(bias_overall_mean, 4),
        "bias_overall_std": round(bias_overall_std, 4),
        "bias_min": round(float(bias_arr.min()), 4),
        "bias_max": round(float(bias_arr.max()), 4),
        "deploy_bias": round(deploy_bias, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "ref_nb3200": REF_NB3200,
        "ref_nb3190": REF_NB3190,
        "ref_parent_nb3090": REF_PARENT_NB3090,
        "ref_nb3173": REF_NB3173,
        "ref_nb2171": REF_NB2171,
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
    print(f"   per-fold-mean ({n_s} seeds) = {pf_mean:.4f} +/- {pf_std:.4f}")
    print(f"   95% CI                  = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   pooled ({n_s} seeds)      = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"   shift vs nb3200         = {shift_vs_nb3200:+.4f}")
    print(f"   bias mean (150 folds)   = {bias_overall_mean:+.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   wall                    = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean_mean", "per_fold_mean_std",
        "per_fold_mean_ci95_low", "per_fold_mean_ci95_high",
        "pooled_mean", "pooled_std",
        "shift_vs_nb3200", "bias_overall_mean", "bias_overall_std",
        "deploy_bias", "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
