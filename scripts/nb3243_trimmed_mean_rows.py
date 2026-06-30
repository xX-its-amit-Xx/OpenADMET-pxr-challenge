"""nb3243 -- Per-row trimmed mean of 5 best-clip variants.

NEW PARADIGM:
    Per row of the 253 unblind (and 513 test), stack the predictions from the
    5 best-clip-variant anchors:
        {nb3190, nb3173, nb3174, nb3170, nb3200}
    and take the TRIMMED MEAN: drop the per-row min and per-row max, then
    average the remaining 3. This is a parameter-free row-wise robust mean
    that suppresses the single outlier-anchor on each row.

    Operator-only (no fold-fit weights, no SLSQP, no learned-clip):
        for row i in 0..N-1:
            v = [p_nb3190[i], p_nb3173[i], p_nb3174[i], p_nb3170[i], p_nb3200[i]]
            trimmed[i] = mean(sorted(v)[1:4])  # drop min + max, mean of 3 mid

    This is different from:
        - SLSQP simplex blends (nb3231/3214/3215) -- those FIT weights per-fold
        - equal-weight mean -- this DROPS the per-row tails first
        - rank-trim-mean -- this trims on VALUES, not rank-orders

    Hypothesis: if the 5 best-clip variants have correlated central mass
    (the clip operator squeezes them into a narrow band ~0.4422-0.4437) but
    different per-row "lucky" outlier picks, dropping the per-row extremes
    cancels the variance-of-extremes without paying the SLSQP overfit tax.

PROTOCOL (per kf_seed, 5-fold scaffold split on 253 unblind):
    Build trimmed_oof from the 5 anchor pred_oofs directly (no model fit).
    Per outer fold:
        val_pred = trimmed_oof[va_loc]     # operator-only; tr_loc unused
        fold_rae = rae(y_unb[va_loc], val_pred)
    Stitch all fold-val into oof; pooled and per-fold-mean.
    Repeat for 15 FRESH kf_seeds {1216..1230}.

    Because the trimmed-mean has no fitted parameters, every kf_seed yields
    the SAME pooled OOF; what varies across kf_seeds is the per-fold
    decomposition (which rows land in which fold). Per-fold-mean is the
    reported gate metric (consistent with nb3231 protocol).

GATE (on 15-seed per-fold-mean):
    per_fold_mean < 0.4424 -> "BETTER"
    else                   -> "FAIL"

Deploy te applies the same trimmed-mean operator directly to the 5 anchor te
arrays (513,5) -> (513,) -- parameter-free, identical structure to OOF path.

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3190_pred_oof.npy   data/processed/te_nb3190.npy
    data/processed/nb3173_pred_oof.npy   data/processed/te_nb3173.npy
    data/processed/nb3174_pred_oof.npy   data/processed/te_nb3174.npy
    data/processed/nb3170_pred_oof.npy   data/processed/te_nb3170.npy
    data/processed/nb3200_pred_oof.npy   data/processed/te_nb3200.npy

Outputs:
    data/processed/nb3243_summary.json
    data/processed/nb3243_pred_oof.npy   (253,) float32 -- trimmed-mean OOF
    data/processed/te_nb3243.npy         (513,) float32 -- trimmed-mean te
    submissions/nb3243_trimmed_mean_rows.csv  (only on BETTER)
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

TAG = "nb3243"

# -- Anchors (5 best clip variants) -------------------------------------------
ANCHOR_NAMES = ["nb3190", "nb3173", "nb3174", "nb3170", "nb3200"]
ANCHOR_OOF_PATHS = {n: DATA_PROCESSED / f"{n}_pred_oof.npy" for n in ANCHOR_NAMES}
ANCHOR_TE_PATHS = {n: DATA_PROCESSED / f"te_{n}.npy" for n in ANCHOR_NAMES}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4424  # per-fold-mean < this -> BETTER

# -- References (PRE-unblind ceiling band) ------------------------------------
REF_NB3173 = 0.4422
REF_NB3200 = 0.4424
REF_NB3190 = 0.4426
REF_NB3174 = 0.4427
REF_NB3170 = 0.4437
REF_NB3080 = 0.4475
REF_NB3090 = 0.4472
REF_NB2171 = 0.4682


def _trimmed_mean_rows(P: np.ndarray) -> np.ndarray:
    """Per-row trimmed mean of 5 columns: drop min + max, mean of remaining 3.

    Parameters
    ----------
    P : (N, 5) float64

    Returns
    -------
    out : (N,) float64 -- per-row mean of middle-3 values
    """
    if P.shape[1] != 5:
        raise ValueError(f"trimmed mean requires 5 columns, got {P.shape[1]}")
    # Sort per row, take middle-3 (indices 1, 2, 3 of sorted axis=1)
    sorted_P = np.sort(P, axis=1)
    middle = sorted_P[:, 1:4]
    return middle.mean(axis=1)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-row trimmed mean of 5 best-clip variants")
    print(f"          anchors  = {ANCHOR_NAMES}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate: per_fold_mean < {GATE_BETTER:.4f} -> BETTER, "
        f"else FAIL"
    )
    print("=" * 78)

    # -- Load test, truth, unblind idx ---------------------------------------
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

    # -- Load anchors ---------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load 5 clip-variant anchors (pred_oof + te)")
    print("-" * 78)
    P_list = []
    te_list = []
    indiv_rae = {}
    leak_eq = {}
    for name in ANCHOR_NAMES:
        oof = np.load(ANCHOR_OOF_PATHS[name]).astype(np.float64)
        te_a = np.load(ANCHOR_TE_PATHS[name]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(
                f"{name} pred_oof shape {oof.shape} != ({n_unb},)"
            )
        if te_a.shape != (n_test,):
            raise ValueError(
                f"{name} te shape {te_a.shape} != ({n_test},)"
            )
        r = float(rae(y_unb, oof))
        leak = float(np.mean(np.isclose(oof, y_unb, atol=1e-6)))
        indiv_rae[name] = round(r, 4)
        leak_eq[name] = round(leak, 4)
        P_list.append(oof)
        te_list.append(te_a)
        print(
            f"   {name}: oof_RAE={r:.4f}  "
            f"oof mean={oof.mean():.3f} std={oof.std():.3f}  "
            f"leak_eq={leak:.2%}  "
            f"te mean={te_a.mean():.3f} std={te_a.std():.3f}"
        )
        if leak > 0.05:
            print(f"   WARN {name}: {leak:.1%} rows == truth -- possible leak")
    P = np.stack(P_list, axis=1)  # (253, 5)
    TE = np.stack(te_list, axis=1)  # (513, 5)

    # Cross-anchor correlation
    corr_mat = np.corrcoef(P.T)
    print("\n   cross-anchor Pearson corr (OOF):")
    header = "          " + "  ".join(f"{n:>8}" for n in ANCHOR_NAMES)
    print(header)
    for i, ni in enumerate(ANCHOR_NAMES):
        cells = "  ".join(
            f"{corr_mat[i, j]:8.3f}" for j in range(len(ANCHOR_NAMES))
        )
        print(f"     {ni:>8}: {cells}")
    mean_offdiag = float(
        corr_mat[np.triu_indices_from(corr_mat, k=1)].mean()
    )
    print(f"   mean off-diagonal Pearson corr = {mean_offdiag:.4f}")

    # Equal-weight baseline (mean of all 5, no trimming)
    eq_oof = P.mean(axis=1)
    eq_oof_rae = float(rae(y_unb, eq_oof))
    print(
        f"\n   equal-weight mean (all 5) OOF RAE = {eq_oof_rae:.4f}  "
        f"(reference for trim delta)"
    )

    # -- Build trimmed-mean OOF (parameter-free) ------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: build per-row trimmed-mean OOF (drop min+max, mean of mid-3)")
    print("-" * 78)
    trimmed_oof = _trimmed_mean_rows(P)
    trimmed_oof_rae = float(rae(y_unb, trimmed_oof))
    print(
        f"   trimmed_oof: rae={trimmed_oof_rae:.4f}  "
        f"mean={trimmed_oof.mean():.3f}  std={trimmed_oof.std():.3f}  "
        f"min={trimmed_oof.min():.3f}  max={trimmed_oof.max():.3f}"
    )
    print(
        f"   delta vs equal-weight (full mean) = "
        f"{trimmed_oof_rae - eq_oof_rae:+.4f}"
    )
    # Count per-row which anchor was the dropped-min and dropped-max
    sorted_idx = np.argsort(P, axis=1)
    drop_min_idx = sorted_idx[:, 0]
    drop_max_idx = sorted_idx[:, -1]
    print(
        f"   dropped-min anchor count = "
        f"{ {ANCHOR_NAMES[i]: int(np.sum(drop_min_idx == i)) for i in range(5)} }"
    )
    print(
        f"   dropped-max anchor count = "
        f"{ {ANCHOR_NAMES[i]: int(np.sum(drop_max_idx == i)) for i in range(5)} }"
    )

    # -- Scaffolds ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Per-seed evaluation (operator is parameter-free, so OOF is identical
    #    across seeds; what changes is per-fold decomposition) ---------------
    print("\n" + "-" * 78)
    print(
        f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes: list[float] = []
    per_fold_means: list[float] = []
    for s in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=s,
        )
        oof_pred = np.full(n_unb, np.nan, dtype=np.float64)
        fold_val_raes: list[float] = []
        for fold_i, (tr_loc, va_loc) in enumerate(splits):
            val_pred = trimmed_oof[va_loc]
            oof_pred[va_loc] = val_pred
            fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        if np.isnan(oof_pred).any():
            raise RuntimeError(
                f"kf_seed={s}: scaffold splits did not cover all rows"
            )
        pooled = float(rae(y_unb, oof_pred))
        pf_mean = float(np.mean(fold_val_raes))
        pf_std = float(np.std(fold_val_raes, ddof=1))
        pooled_raes.append(pooled)
        per_fold_means.append(pf_mean)
        seed_records.append({
            "kf_seed": int(s),
            "pooled_rae": round(pooled, 4),
            "per_fold_val_rae_mean": round(pf_mean, 4),
            "per_fold_val_rae_std": round(pf_std, 4),
            "fold_val_raes": [round(v, 4) for v in fold_val_raes],
        })
        print(
            f"   kf={s}: pooled_RAE={pooled:.4f}  "
            f"perfold_mean={pf_mean:.4f}  perfold_std={pf_std:.4f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    pf_arr = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(pooled_arr)

    mean_pooled = float(pooled_arr.mean())
    std_pooled = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem_pooled = std_pooled / np.sqrt(n_s) if n_s > 1 else 0.0

    mean_pf = float(pf_arr.mean())
    std_pf = float(pf_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem_pf = std_pf / np.sqrt(n_s) if n_s > 1 else 0.0

    t_mult = 2.145  # df=14, two-sided 95%
    ci_low_pf = mean_pf - t_mult * sem_pf
    ci_high_pf = mean_pf + t_mult * sem_pf
    median_pf = float(np.median(pf_arr))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled  mean    = {mean_pooled:.4f}  std = {std_pooled:.4f}")
    print(f"   perfold mean    = {mean_pf:.4f}  std = {std_pf:.4f}")
    print(f"   perfold sem     = {sem_pf:.4f}")
    print(f"   perfold 95% CI  = [{ci_low_pf:.4f}, {ci_high_pf:.4f}]")
    print(f"   perfold median  = {median_pf:.4f}")
    print(f"   perfold min/max = [{pf_arr.min():.4f}, {pf_arr.max():.4f}]")

    print(
        f"\n   indiv anchor RAE (pooled, no CV) = {indiv_rae}"
    )
    print(
        f"   equal-weight mean (no trim) OOF  = {eq_oof_rae:.4f}"
    )
    print(
        f"   trimmed_oof in-sample (no CV)    = {trimmed_oof_rae:.4f}"
    )
    print(
        f"\n   ref nb3173 (best single clip)   = {REF_NB3173:.4f}"
    )
    print(
        f"   delta vs nb3173 (perfold mean)   = "
        f"{mean_pf - REF_NB3173:+.4f}"
    )
    print(
        f"   ref nb3200                       = {REF_NB3200:.4f}"
    )
    print(
        f"   delta vs nb3200 (perfold mean)   = "
        f"{mean_pf - REF_NB3200:+.4f}"
    )

    # -- Deploy te (parameter-free) -------------------------------------------
    print("\n" + "-" * 78)
    print("DEPLOY te (per-row trimmed mean of (513, 5) anchor te arrays)")
    print("-" * 78)
    te_pred = _trimmed_mean_rows(TE).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")
    # Sanity: te[unb] computed from te should match trimmed_oof on the same rows
    # (operator is identical, but indices may differ); we don't assert here.

    # Median-seed OOF for storage -- since operator is parameter-free, all
    # seeds produce identical pooled OOF. We use seed[0] for storage.
    oof_for_save = trimmed_oof.astype(np.float32)

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_pf < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3243 15-seed per-fold-mean "
            f"{mean_pf:.4f} beats BETTER gate {GATE_BETTER:.4f} "
            f"({mean_pf - GATE_BETTER:+.4f}) using a parameter-free per-row "
            f"trimmed mean (drop min + max, average of middle 3) over the 5 "
            f"best clip variants {ANCHOR_NAMES}. Operator carries no fitted "
            f"weights -- result is identical regardless of fold-fit. Beats "
            f"equal-weight mean {eq_oof_rae:.4f} ({mean_pf - eq_oof_rae:+.4f}) "
            f"and best single clip nb3173 {REF_NB3173:.4f} "
            f"({mean_pf - REF_NB3173:+.4f}). Re-verify with deep-30 before "
            f"any PRIMARY-1 swap. Mean off-diag corr = {mean_offdiag:.4f}."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3243 15-seed per-fold-mean {mean_pf:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f} ({mean_pf - GATE_BETTER:+.4f}). "
            f"Per-row trimmed mean of 5 best-clip variants does not extract "
            f"residual orthogonality beyond the central ceiling band -- the "
            f"5 anchors share too much correlated signal (mean off-diag corr "
            f"{mean_offdiag:.4f}), so trimming the per-row extremes only "
            f"shifts within the noise floor. Delta vs nb3173 {REF_NB3173:.4f} "
            f"= {mean_pf - REF_NB3173:+.4f}, vs equal-weight {eq_oof_rae:.4f} "
            f"= {mean_pf - eq_oof_rae:+.4f}. Keep nb3173/nb3200 on the "
            f"ladder; close the row-wise-trim-of-clip-variants axis."
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

    sub_csv = SUBMISSIONS / f"{TAG}_trimmed_mean_rows.csv"
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
        "method": "per_row_trimmed_mean_of_5_clip_variants_drop_min_max_mean_of_3",
        "anchors": ANCHOR_NAMES,
        "anchor_oof_paths": {k: str(v) for k, v in ANCHOR_OOF_PATHS.items()},
        "anchor_te_paths": {k: str(v) for k, v in ANCHOR_TE_PATHS.items()},
        "anchor_pre_unblind": True,
        "indiv_oof_rae": indiv_rae,
        "leak_eq_truth_frac": leak_eq,
        "cross_anchor_corr_oof": [
            [round(float(corr_mat[i, j]), 4) for j in range(len(ANCHOR_NAMES))]
            for i in range(len(ANCHOR_NAMES))
        ],
        "mean_off_diag_corr": round(mean_offdiag, 4),
        "equal_weight_oof_rae": round(eq_oof_rae, 4),
        "trimmed_oof_in_sample_rae": round(trimmed_oof_rae, 4),
        "drop_min_anchor_count": {
            ANCHOR_NAMES[i]: int(np.sum(drop_min_idx == i)) for i in range(5)
        },
        "drop_max_anchor_count": {
            ANCHOR_NAMES[i]: int(np.sum(drop_max_idx == i)) for i in range(5)
        },
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "mean_pooled_rae": round(mean_pooled, 4),
        "std_pooled_rae": round(std_pooled, 4),
        "sem_pooled_rae": round(sem_pooled, 4),
        "mean_per_fold_rae": round(mean_pf, 4),
        "std_per_fold_rae": round(std_pf, 4),
        "sem_per_fold_rae": round(sem_pf, 4),
        "ci95_per_fold_low": round(ci_low_pf, 4),
        "ci95_per_fold_high": round(ci_high_pf, 4),
        "median_per_fold_rae": round(median_pf, 4),
        "min_per_fold_rae": round(float(pf_arr.min()), 4),
        "max_per_fold_rae": round(float(pf_arr.max()), 4),
        "ref_nb3173_best_single_clip": REF_NB3173,
        "delta_vs_nb3173_perfold_mean": round(mean_pf - REF_NB3173, 4),
        "delta_vs_equal_weight_perfold_mean": round(mean_pf - eq_oof_rae, 4),
        "ref_nb3200": REF_NB3200,
        "ref_nb3190": REF_NB3190,
        "ref_nb3174": REF_NB3174,
        "ref_nb3170": REF_NB3170,
        "ref_nb3080": REF_NB3080,
        "ref_nb3090": REF_NB3090,
        "ref_nb2171": REF_NB2171,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv) if verdict == "BETTER" else None),
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
        f"   per_fold_mean ({n_s} seeds) = {mean_pf:.4f} +/- {std_pf:.4f}"
    )
    print(f"   95% CI                = [{ci_low_pf:.4f}, {ci_high_pf:.4f}]")
    print(f"   delta vs nb3173       = {mean_pf - REF_NB3173:+.4f}")
    print(f"   delta vs equal-weight = {mean_pf - eq_oof_rae:+.4f}")
    print(f"   trimmed in-sample     = {trimmed_oof_rae:.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_per_fold_rae", "std_per_fold_rae",
        "ci95_per_fold_low", "ci95_per_fold_high",
        "delta_vs_nb3173_perfold_mean",
        "delta_vs_equal_weight_perfold_mean",
        "equal_weight_oof_rae",
        "trimmed_oof_in_sample_rae",
        "indiv_oof_rae", "mean_off_diag_corr",
        "drop_min_anchor_count", "drop_max_anchor_count",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
