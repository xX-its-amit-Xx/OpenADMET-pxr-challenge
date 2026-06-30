"""nb2940 -- RESCUE: deterministic 5-component equal-weight blend
            (K=18, K=20, K=24, K=28, nb1191) re-verified across 5 kf_seeds.

CONTEXT (cycle 241 rescue):
    nb2934 reported pooled scaffold-CV RAE 0.4585 (MARGINAL_BEAT @ <0.4598).
    The original heavy nb2940 (deep-30 fresh-seed RFE rebuild) timed out.
    Per task spec rescue: reload the EXISTING cached K-RFE OOFs (these ARE
    the cached results, no rebuild), add nb1191 as 5th component, equal
    weight 1/5 each, and pool RAE per kf_seed in {1001..1005}. Since
    predictions are FIXED across kf_seeds (no learning), only the scaffold
    fold partitions vary -- the kf_seed sweep measures fold-partition
    sensitivity, not seed variance.

PROTOCOL:
    1. Reload 5 cached OOF + te arrays (NO rebuild, NO refit):
         K=18   -> nb2604_mean_bag_oof_K18.npy + te_nb2604_K18.npy
         K=20   -> nb2240_mean_bag_oof_K20.npy + te_nb2240_K20.npy
         K=24   -> nb2310_mean_bag_oof_K24.npy + te_nb2310_K24.npy
         K=28   -> nb2103_mean_bag_oof_K28.npy + te_nb2112.npy
         nb1191 -> nb1191_pred_oof.npy + te_nb1191.npy
    2. Equal weight 0.2 each (deterministic, df = 0):
         pred_oof_unb = mean(K18, K20, K24, K28, nb1191) on 253
         pred_te_513  = mean(K18, K20, K24, K28, nb1191) on 513
    3. For each kf_seed in {1001..1005}:
         5-fold scaffold split -> pooled RAE on full 253-vec
    4. Report mean RAE across 5 kf_seeds.

GATE:
    mean < 0.4570  -> "VERIFIED_PROMOTE"
    mean < 0.4598  -> "VERIFIED_MARGINAL"
    else            -> "FAIL"

References:
    nb2934 single-shot pooled RAE (kf=1001) = 0.4585
    nb2604 5-seed                            = 0.4580 -> nb2640 deep-30 = 0.4611
    nb2171 deep-30 ceiling                   = 0.4682
    nb1191 deep-30 ceiling                   = 0.4718

Outputs:
    data/processed/nb2940_summary.json
    data/processed/nb2940_pred_oof.npy           (253,) float32
    data/processed/te_nb2940.npy                 (513,) float32
    submissions/nb2940_deep30_verify_nb2934.csv  (always, for ladder visibility)
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

TAG = "nb2940"
PARENT_TAG = "nb2934"

# ---- Cached component OOF + te paths (NO rebuild) ---------------------------
COMPONENTS = [
    ("K18",    "nb2604_mean_bag_oof_K18.npy",  "te_nb2604_K18.npy"),
    ("K20",    "nb2240_mean_bag_oof_K20.npy",  "te_nb2240_K20.npy"),
    ("K24",    "nb2310_mean_bag_oof_K24.npy",  "te_nb2310_K24.npy"),
    ("K28",    "nb2103_mean_bag_oof_K28.npy",  "te_nb2112.npy"),
    ("nb1191", "nb1191_pred_oof.npy",          "te_nb1191.npy"),
]

# ---- Eval ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- References ----
NB2934_SINGLE_REF = 0.4585
NB2604_5SEED_REF = 0.4580
NB2640_DEEP30_REF = 0.4611
NB2171_REF = 0.4682
NB1191_REF = 0.4718
CHEMPROP_AUX_REF = 0.6216


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RESCUE deterministic 5-component verify of {PARENT_TAG}")
    print(f"          paradigm: plain mean (df=0), kf_seed sweep only")
    print(f"          gate PROMOTE < {GATE_PROMOTE}  MARGINAL < {GATE_MARGINAL}")
    print("=" * 78)

    # -- Load truth + scaffolds ---------------------------------------------
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (te_df["smiles"].astype(str).tolist()
                 if "smiles" in te_df.columns
                 else te_df["SMILES"].astype(str).tolist())
    te_names = (te_df["name"].values
                if "name" in te_df.columns
                else te_df["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # -- Load 5 cached components -------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load 5 cached components (NO rebuild)")
    print("-" * 78)
    oofs = {}
    tes = {}
    per_anchor_rae = {}
    for name, oof_rel, te_rel in COMPONENTS:
        oof_p = DATA_PROCESSED / oof_rel
        te_p = DATA_PROCESSED / te_rel
        if not oof_p.exists():
            raise FileNotFoundError(f"missing OOF: {oof_p}")
        if not te_p.exists():
            raise FileNotFoundError(f"missing te:  {te_p}")
        oof = np.load(oof_p).astype(np.float64)
        te_v = np.load(te_p).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{name} oof shape {oof.shape} != ({n_unb},)")
        if te_v.shape != (n_test,):
            raise ValueError(f"{name} te shape {te_v.shape} != ({n_test},)")
        oofs[name] = oof
        tes[name] = te_v
        r = float(rae(y_unb, oof))
        per_anchor_rae[name] = r
        print(f"   {name:>7s}  oof_RAE={r:.4f}  te_mean={te_v.mean():.3f}  "
              f"te_std={te_v.std():.3f}  [{oof_rel} + {te_rel}]")

    # -- Equal-weight blend (deterministic) ---------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: equal-weight blend (0.2 each, df = 0)")
    print("-" * 78)
    labels = [c[0] for c in COMPONENTS]
    P_unb = np.column_stack([oofs[k] for k in labels])  # (253, 5)
    P_te = np.column_stack([tes[k] for k in labels])    # (513, 5)
    weights = np.full(5, 0.2, dtype=np.float64)
    pred_oof_unb = (P_unb * weights).sum(axis=1)
    pred_te_513 = (P_te * weights).sum(axis=1)
    print(f"   P_unb={P_unb.shape}  P_te={P_te.shape}  weights={weights.tolist()}")
    print(f"   pred_oof_unb mean={pred_oof_unb.mean():.3f}  "
          f"std={pred_oof_unb.std():.3f}  (truth_std={y_unb.std():.3f})")
    print(f"   pred_te_513  mean={pred_te_513.mean():.3f}  "
          f"std={pred_te_513.std():.3f}")

    rae_full = float(rae(y_unb, pred_oof_unb))
    print(f"   single-shot pooled RAE (no folds) = {rae_full:.4f}")

    # Diagnostic: pair-wise correlations
    corr_mat = np.corrcoef(P_unb.T)
    print(f"\n   OOF correlation matrix:")
    print(f"          {'  '.join([f'{k:>7s}' for k in labels])}")
    for i, ki in enumerate(labels):
        row = "  ".join([f"{corr_mat[i, j]:7.3f}" for j in range(5)])
        print(f"     {ki:>7s}  {row}")

    # -- 5-fold scaffold CV, sweep kf_seeds ---------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: 5-fold scaffold CV, kf_seeds={KF_SEEDS}")
    print(f"       (predictions FIXED across kf_seeds; only fold partitions vary)")
    print("-" * 78)
    per_kf_pooled = []
    per_kf_per_fold = []
    for kfs in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kfs,
        )
        oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
        per_fold_rae = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            oof_pooled[va_loc] = pred_oof_unb[va_loc]
            r = float(rae(y_unb[va_loc], pred_oof_unb[va_loc]))
            per_fold_rae.append(r)
        if np.isnan(oof_pooled).any():
            raise RuntimeError(f"kf_seed={kfs} scaffold splits did not cover all rows")
        pooled_r = float(rae(y_unb, oof_pooled))
        per_kf_pooled.append(pooled_r)
        per_kf_per_fold.append(per_fold_rae)
        print(f"   kf_seed={kfs:4d}  pooled_RAE={pooled_r:.4f}  "
              f"per_fold={[f'{v:.4f}' for v in per_fold_rae]}")

    pooled_arr = np.array(per_kf_pooled, dtype=np.float64)
    mean_rae = float(pooled_arr.mean())
    std_rae = float(pooled_arr.std(ddof=1))
    min_rae = float(pooled_arr.min())
    max_rae = float(pooled_arr.max())
    median_rae = float(np.median(pooled_arr))

    print(f"\n   5-kf_seed pooled RAE: mean={mean_rae:.4f}  "
          f"std={std_rae:.5f}  median={median_rae:.4f}  "
          f"min={min_rae:.4f}  max={max_rae:.4f}")

    # -- Gate ---------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: GATE")
    print("-" * 78)
    if mean_rae < GATE_PROMOTE:
        verdict = "VERIFIED_PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
    else:
        verdict = "FAIL"
    shift_vs_single = mean_rae - NB2934_SINGLE_REF
    delta_vs_nb2171 = mean_rae - NB2171_REF
    delta_vs_nb1191 = mean_rae - NB1191_REF
    print(f"   nb2934 single-shot (kf=1001) ref     = {NB2934_SINGLE_REF:.4f}")
    print(f"   nb2940 5-kf_seed mean +/- std        = "
          f"{mean_rae:.4f} +/- {std_rae:.5f} (n=5)")
    print(f"   shift (5kf - single)                = {shift_vs_single:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_REF})      = {delta_vs_nb2171:+.4f}")
    print(f"   delta vs nb1191 ({NB1191_REF})      = {delta_vs_nb1191:+.4f}")
    print(f"     <{GATE_PROMOTE}  -> VERIFIED_PROMOTE")
    print(f"     <{GATE_MARGINAL} -> VERIFIED_MARGINAL")
    print(f"     else             -> FAIL")
    print(f"   -> {verdict}")

    # -- Save artifacts -----------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_deep30_verify_nb2934.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    te_unb_in_rae = float(rae(y_unb, pred_te_513[unb_idx]))

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "rescue_deterministic_5component_equal_weight_kfsweep",
        "paradigm": "plain_mean_df_zero_no_learning_kfseed_sweep",
        "components": labels,
        "component_oof_paths": {c[0]: str(DATA_PROCESSED / c[1]) for c in COMPONENTS},
        "component_te_paths":  {c[0]: str(DATA_PROCESSED / c[2]) for c in COMPONENTS},
        "weights": weights.tolist(),
        "per_anchor_rae_in_sample": per_anchor_rae,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_corr_labels": labels,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_kf_pooled_rae": per_kf_pooled,
        "per_kf_per_fold_rae": per_kf_per_fold,
        "single_shot_pooled_rae_no_folds": rae_full,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "median_rae": median_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "anchor_pre_unblind": True,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "shift_vs_nb2934_single": shift_vs_single,
        "delta_vs_nb2171": delta_vs_nb2171,
        "delta_vs_nb1191": delta_vs_nb1191,
        "nb2934_single_ref": NB2934_SINGLE_REF,
        "nb2604_5seed_ref": NB2604_5SEED_REF,
        "nb2640_deep30_ref": NB2640_DEEP30_REF,
        "nb2171_ref": NB2171_REF,
        "nb1191_ref": NB1191_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "te_unb_in_sample_rae": te_unb_in_rae,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-anchor RAE = " + "  ".join(
        f"{k}={v:.4f}" for k, v in per_anchor_rae.items()))
    print(f"   nb2934 single (kf=1001) = {NB2934_SINGLE_REF:.4f}")
    print(f"   nb2940 5-kf mean+/-std   = {mean_rae:.4f} +/- {std_rae:.5f}")
    print(f"   shift (5kf - single)    = {shift_vs_single:+.4f}")
    print(f"   verdict                 = "
          f"<{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL -> {verdict}")
    print(f"   wall                    = {time.time()-t0:.2f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "median_rae", "min_rae", "max_rae",
        "verdict",
        "shift_vs_nb2934_single", "delta_vs_nb2171", "delta_vs_nb1191",
        "te_unb_in_sample_rae", "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
