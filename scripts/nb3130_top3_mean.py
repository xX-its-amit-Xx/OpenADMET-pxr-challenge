"""nb3130 -- Equal-weight mean of top-3 PROMOTE candidates (nb3070, nb3072, nb3080).

NEW PARADIGM:
    nb3070, nb3072, nb3080 all cluster in 0.4475-0.4477 deep-30 wide-seed band on
    {K18, K19} quantile-conditional blends. Hypothesis: equal-weight mean of the
    three OOF/te arrays averages-out per-config noise, possibly breaking under
    the cluster mean.

PROTOCOL:
    1. Load three pred_oof arrays (all shape (253,)) from nb3070, nb3072, nb3080.
    2. Load three te arrays (all shape (513,)) from same trio.
    3. oof_mean = (oof_a + oof_b + oof_c) / 3
       te_mean  = (te_a  + te_b  + te_c ) / 3
    4. Wide-seed sweep: re-evaluate the FIXED equal-weight blend under 15 fresh
       kf_seeds {1141..1155} via 5-fold scaffold CV. Since the blend itself is
       parameter-free, each kf_seed only changes which row goes in which val
       fold; the pooled OOF over all rows is invariant in kf_seed (a property of
       fixed averaging). We still loop seeds for diagnostic reproducibility +
       per-fold dispersion stats.

GATE (on 15-seed mean):
    mean < 0.4475 -> "BETTER" (PROMOTE candidate)
    else          -> "FAIL"

References:
    nb3070 (wide-seed verify of nb3063, q50 = full-OOF median)        = 0.4477
    nb3072 (per-fold TRAIN q50 quantile-conditional blend)            = 0.4477
    nb3080 (wide-seed verify of nb3073 best-combo q_cut=0.4, w=0.9/0.1)= 0.4475
    -> cluster mean ~ 0.4476

Inputs:
    data/processed/_audit_unblind_idx.npy   (253,)
    data/processed/_audit_unblind_y.npy     (253,)
    data/processed/nb3070_pred_oof.npy      (253,)
    data/processed/nb3072_pred_oof.npy      (253,)
    data/processed/nb3080_pred_oof.npy      (253,)
    data/processed/te_nb3070.npy            (513,)
    data/processed/te_nb3072.npy            (513,)
    data/processed/te_nb3080.npy            (513,)

Outputs:
    data/processed/nb3130_summary.json
    data/processed/nb3130_pred_oof.npy      (253,) float32 -- equal-weight mean
    data/processed/te_nb3130.npy            (513,) float32 -- equal-weight mean
    submissions/nb3130_top3_mean.csv        (only on BETTER verdict)
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

TAG = "nb3130"
PARENT_TAGS = ["nb3070", "nb3072", "nb3080"]

# -- Inputs --------------------------------------------------------------------
OOF_PATHS = {
    "nb3070": DATA_PROCESSED / "nb3070_pred_oof.npy",
    "nb3072": DATA_PROCESSED / "nb3072_pred_oof.npy",
    "nb3080": DATA_PROCESSED / "nb3080_pred_oof.npy",
}
TE_PATHS = {
    "nb3070": DATA_PROCESSED / "te_nb3070.npy",
    "nb3072": DATA_PROCESSED / "te_nb3072.npy",
    "nb3080": DATA_PROCESSED / "te_nb3080.npy",
}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1141, 1156))  # 15 FRESH seeds {1141..1155}

# -- Gate ---------------------------------------------------------------------
GATE_BETTER = 0.4475

# -- References ---------------------------------------------------------------
REF_NB3070 = 0.4477
REF_NB3072 = 0.4477
REF_NB3080 = 0.4475
REF_CLUSTER_MEAN = 0.4476
REF_NB3030 = 0.4509


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- equal-weight mean of top-3 PROMOTE candidates "
          f"{PARENT_TAGS}")
    print(f"          kf_seeds = {len(KF_SEEDS)} fresh "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          gate: mean < {GATE_BETTER:.4f} -> BETTER else FAIL")
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

    # -- Load 3 OOF + 3 te arrays --------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load 3 OOFs and 3 te arrays")
    print("-" * 78)
    oof_cols, te_cols = [], []
    per_parent_full_rae = {}
    for tag in PARENT_TAGS:
        oof = np.load(OOF_PATHS[tag]).astype(np.float64)
        te_arr = np.load(TE_PATHS[tag]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{tag} OOF shape {oof.shape} != ({n_unb},)")
        if te_arr.shape != (n_test,):
            raise ValueError(f"{tag} te shape {te_arr.shape} != ({n_test},)")
        oof_cols.append(oof)
        te_cols.append(te_arr)
        r = float(rae(y_unb, oof))
        per_parent_full_rae[tag] = round(r, 4)
        print(f"   {tag}: oof_RAE = {r:.4f}  te_mean={te_arr.mean():.3f}  "
              f"te_std={te_arr.std():.3f}")

    P_unb = np.column_stack(oof_cols)  # (253, 3)
    P_te = np.column_stack(te_cols)    # (513, 3)

    # Leak sanity
    leak_flags = {}
    for i, tag in enumerate(PARENT_TAGS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[tag] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {tag}: {frac:.1%} rows == truth -- possible leak")

    # Pairwise correlations among the 3 OOFs
    corr_mat = np.corrcoef(P_unb.T)
    pair_corrs = {
        f"{PARENT_TAGS[0]}_{PARENT_TAGS[1]}": round(float(corr_mat[0, 1]), 4),
        f"{PARENT_TAGS[0]}_{PARENT_TAGS[2]}": round(float(corr_mat[0, 2]), 4),
        f"{PARENT_TAGS[1]}_{PARENT_TAGS[2]}": round(float(corr_mat[1, 2]), 4),
    }
    print(f"   pairwise corr: {pair_corrs}")

    # -- Equal-weight mean ---------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: build equal-weight mean (1/3 each)")
    print("-" * 78)
    oof_mean = P_unb.mean(axis=1)  # (253,)
    te_mean = P_te.mean(axis=1)    # (513,)
    full_oof_rae = float(rae(y_unb, oof_mean))
    print(f"   full-OOF equal-mean RAE on 253 = {full_oof_rae:.4f}")
    print(f"   te_mean: mean={te_mean.mean():.3f}  std={te_mean.std():.3f}  "
          f"min={te_mean.min():.3f}  max={te_mean.max():.3f}")

    # -- Scaffolds -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Multi-seed sweep ----------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    for s in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=s,
        )
        # Blend is parameter-free; pooled OOF == oof_mean for every seed.
        # We still stitch fold-val predictions for the per-fold rae stats.
        oof_seed = np.full(n_unb, np.nan, dtype=np.float64)
        fold_val_raes = []
        for fold_i, (tr_loc, va_loc) in enumerate(splits):
            val_pred = oof_mean[va_loc]
            oof_seed[va_loc] = val_pred
            fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        if np.isnan(oof_seed).any():
            raise RuntimeError(
                f"kf_seed={s}: scaffold splits did not cover all rows"
            )
        pooled = float(rae(y_unb, oof_seed))
        pooled_raes.append(pooled)
        seed_records.append({
            "kf_seed": int(s),
            "pooled_rae": round(pooled, 4),
            "per_fold_val_rae_mean": round(float(np.mean(fold_val_raes)), 4),
            "per_fold_val_rae_std": round(float(np.std(fold_val_raes, ddof=1)), 4),
        })
        print(f"   kf={s}: pooled_RAE={pooled:.4f}  "
              f"per_fold_mean={np.mean(fold_val_raes):.4f}  "
              f"per_fold_std={np.std(fold_val_raes, ddof=1):.4f}  "
              f"wall={time.time()-ts:.2f}s")

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
    print(f"\n   ref nb3070 (15-seed)         = {REF_NB3070:.4f}")
    print(f"   ref nb3072 (15-seed)         = {REF_NB3072:.4f}")
    print(f"   ref nb3080 (15-seed)         = {REF_NB3080:.4f}")
    print(f"   ref cluster naive avg        = {REF_CLUSTER_MEAN:.4f}")
    print(f"   ref nb3030 wide-seed ceiling = {REF_NB3030:.4f}")
    print(f"   delta vs cluster_mean        = "
          f"{mean_rae - REF_CLUSTER_MEAN:+.4f}")
    print(f"   delta vs nb3030              = {mean_rae - REF_NB3030:+.4f}")

    # te[unb] in-sample sanity (parameter-free blend so identical to full_oof_rae)
    te_pred_f32 = np.clip(te_mean, 3.0, 9.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred_f32[unb_idx]))
    print(f"\n   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE. nb3130 equal-weight mean of {PARENT_TAGS} "
            f"15-seed mean {mean_rae:.4f} < gate {GATE_BETTER:.4f}. "
            f"Beats cluster naive avg {REF_CLUSTER_MEAN:.4f} by "
            f"{mean_rae - REF_CLUSTER_MEAN:+.4f}. New PRIMARY-1 candidate."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"FAIL. nb3130 equal-weight mean of {PARENT_TAGS} "
            f"15-seed mean {mean_rae:.4f} >= gate {GATE_BETTER:.4f}. "
            f"Shift vs cluster naive avg {REF_CLUSTER_MEAN:.4f} = "
            f"{mean_rae - REF_CLUSTER_MEAN:+.4f}. Keep prior PRIMARY-1."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_for_save = oof_mean.astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred_f32)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_top3_mean.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred_f32,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tags": PARENT_TAGS,
        "method": "equal_weight_mean_of_top3_promote_candidates",
        "weights": {tag: 1.0 / 3.0 for tag in PARENT_TAGS},
        "anchor_pre_unblind": True,
        "per_parent_full_oof_rae": per_parent_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": pair_corrs,
        "full_oof_equal_mean_rae": round(full_oof_rae, 4),
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
        "ref_nb3070": REF_NB3070,
        "ref_nb3072": REF_NB3072,
        "ref_nb3080": REF_NB3080,
        "ref_cluster_mean": REF_CLUSTER_MEAN,
        "ref_nb3030": REF_NB3030,
        "delta_vs_cluster_mean": round(mean_rae - REF_CLUSTER_MEAN, 4),
        "delta_vs_nb3030": round(mean_rae - REF_NB3030, 4),
        "te_mean": float(te_pred_f32.mean()),
        "te_std": float(te_pred_f32.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
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
    print(f"   mean_rae ({n_s} seeds)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs cluster mean = {mean_rae - REF_CLUSTER_MEAN:+.4f}")
    print(f"   delta vs nb3030       = {mean_rae - REF_NB3030:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_cluster_mean", "delta_vs_nb3030",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_parent_full_oof_rae: {res.get('per_parent_full_oof_rae')}")
    print(f"  oof_pairwise_corr: {res.get('oof_pairwise_corr')}")
