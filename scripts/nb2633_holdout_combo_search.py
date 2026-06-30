"""nb2633 -- Hold-out validated K-combo search (avoid selection bias).

NEW PARADIGM:
    nb2621 evaluated 372 equal-weight K-combos on the SAME 253 unblind rows
    used to select the winner -> selection bias.  Reported best
    in-sample mean_rae 0.4552 is a lower bound on TRUE generalisation.

    Here we do K-fold cross-validation OVER the combo search itself:
        Outer 5-fold scaffold CV on 253 unblind.
        Within each outer fold:
            - search 372 equal-weight K-combos using ONLY outer-train rows
            - pick best combo by outer-train RAE
            - apply that combo on outer-val rows
        Pool outer-val predictions -> honest selection-validated RAE.

    Each outer fold can pick a DIFFERENT winner; that's the whole point.

PROTOCOL:
    1. Load 9 cached K-RFE OOF arrays (anchors).
    2. Load 253 unblind y + scaffolds.
    3. Outer scaffold 5-fold (averaged over a few seeds for stability).
    4. For each outer fold:
         a. enumerate 372 equal-weight subsets size 2..5
         b. rank combos by RAE on outer-train rows
         c. pick best, apply on outer-val rows
    5. Pool outer-val predictions (each row predicted exactly once per seed).
    6. Mean over kf_seeds -> outer_val_rae.
    7. Deploy submission: equal-weight mean over MODE selected combo across folds
       (most-frequently-chosen combo across fold-seed pairs), and te projection.

GATE:
    outer-val mean_rae < 0.4570 -> PROMOTE_HOLDOUT_VALIDATED
    outer-val mean_rae < 0.4580 -> MARGINAL
    else                        -> SELECTION_BIAS_CONFIRMED

Outputs:
    data/processed/nb2633_summary.json
    data/processed/nb2633_pred_oof.npy   (253,) float32  outer-val pooled
    data/processed/te_nb2633.npy         (513,) float32  mode-combo te
    submissions/nb2633_holdout_combo_search.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2633"

# ---- 9 cached K-RFE anchors (OOF on 253 unblind + te on 513) ----
CACHE_OOF = {
    14: DATA_PROCESSED / "nb2611_mean_bag_oof_K14.npy",
    16: DATA_PROCESSED / "nb2611_mean_bag_oof_K16.npy",
    18: DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy",
    20: DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
    22: DATA_PROCESSED / "nb2261_mean_bag_oof_K22.npy",
    24: DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy",
    28: DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy",
    32: DATA_PROCESSED / "nb2611_mean_bag_oof_K32.npy",
    36: DATA_PROCESSED / "nb2611_mean_bag_oof_K36.npy",
}
CACHE_TE = {
    14: DATA_PROCESSED / "te_nb2611_K14.npy",
    16: DATA_PROCESSED / "te_nb2611_K16.npy",
    18: DATA_PROCESSED / "te_nb2604_K18.npy",
    20: DATA_PROCESSED / "te_nb2240_K20.npy",
    22: DATA_PROCESSED / "te_nb2261_K22.npy",
    24: DATA_PROCESSED / "te_nb2310_K24.npy",
    28: DATA_PROCESSED / "te_nb2112.npy",
    32: DATA_PROCESSED / "te_nb2611_K32.npy",
    36: DATA_PROCESSED / "te_nb2611_K36.npy",
}

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

MIN_R = 2
MAX_R = 5
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]  # average over 5 seeds for outer split stability

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4580

# ---- Refs ----
NB2621_INSAMPLE_REF = 0.4552  # nb2621 in-sample best (selection biased)
NB2604_REF = 0.4580
NB2611_REF = 0.4580
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- hold-out validated K-combo search")
    print(f"          9 anchors x C(9,2..5)=372 combos searched WITHIN each outer fold")
    print(f"          outer 5-fold scaffold CV averaged over {len(KF_SEEDS)} seeds")
    print(f"          gate PROMOTE < {GATE_PROMOTE}, MARGINAL < {GATE_MARGINAL}")
    print(f"          compare to nb2621 in-sample {NB2621_INSAMPLE_REF}")
    print("=" * 78)

    # ---- Load truth + scaffolds ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    rae_anchor = float(rae(y_unb, te_anchor_513[unb_idx]))
    print(f"[load] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f}")

    # ---- Load 9 K anchors ----
    print("\n" + "-" * 78)
    print("STEP 1: load 9 K-RFE anchors")
    print("-" * 78)
    K_list = sorted(CACHE_OOF.keys())
    K_oof = {}
    K_te = {}
    per_K_rae = {}
    for K in K_list:
        if not CACHE_OOF[K].exists():
            raise FileNotFoundError(f"missing OOF cache for K={K}: {CACHE_OOF[K]}")
        if not CACHE_TE[K].exists():
            raise FileNotFoundError(f"missing te cache for K={K}: {CACHE_TE[K]}")
        K_oof[K] = np.load(CACHE_OOF[K]).astype(np.float64)
        K_te[K] = np.load(CACHE_TE[K]).astype(np.float64)
        if K_oof[K].shape != (n_unb,):
            raise ValueError(f"K={K} OOF shape {K_oof[K].shape} != ({n_unb},)")
        if K_te[K].shape != (n_test,):
            raise ValueError(f"K={K} te shape {K_te[K].shape} != ({n_test},)")
        per_K_rae[K] = float(rae(y_unb, K_oof[K]))
        print(f"   K={K:2d}  full_oof_RAE={per_K_rae[K]:.4f}  "
              f"te_mean={K_te[K].mean():.3f}  te_std={K_te[K].std():.3f}")

    # ---- Pre-stack: (n_unb, 9) and (n_test, 9) ----
    K_oof_stack = np.column_stack([K_oof[k] for k in K_list])   # (253, 9)
    K_te_stack = np.column_stack([K_te[k] for k in K_list])      # (513, 9)
    K_idx = {k: i for i, k in enumerate(K_list)}

    # ---- Enumerate ALL combos once (column index tuples into the 9 anchors) ----
    all_combos = []
    for r in range(MIN_R, MAX_R + 1):
        for c in combinations(range(len(K_list)), r):
            all_combos.append(c)
    n_combos = len(all_combos)
    print(f"[combos] enumerated {n_combos} (r {MIN_R}..{MAX_R})")

    # ---- Outer CV loop ----
    print("\n" + "-" * 78)
    print(f"STEP 2: outer 5-fold scaffold CV  ({len(KF_SEEDS)} seeds x {N_FOLDS} folds = {len(KF_SEEDS)*N_FOLDS} outer searches)")
    print("-" * 78)

    per_seed_outer_rae = []
    per_seed_pooled_oof = []
    per_fold_selected_combos = []  # list of dicts across all seed*fold
    combo_choice_counter = Counter()  # frequency of selected combos across folds

    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        pooled = np.full(n_unb, np.nan, dtype=np.float64)
        per_fold_rae = []
        for fold_i, (tr_loc, va_loc) in enumerate(splits):
            y_tr = y_unb[tr_loc]
            # Search 372 combos on outer-TRAIN rows only
            best_rae = np.inf
            best_combo = None
            best_pred_val = None
            for combo in all_combos:
                cols = np.array(combo, dtype=np.intp)
                # Equal-weight mean across selected anchors
                pred_tr = K_oof_stack[np.ix_(tr_loc, cols)].mean(axis=1)
                r = float(rae(y_tr, pred_tr))
                if r < best_rae:
                    best_rae = r
                    best_combo = combo
                    best_pred_val = K_oof_stack[np.ix_(va_loc, cols)].mean(axis=1)
            pooled[va_loc] = best_pred_val
            fold_val_rae = float(rae(y_unb[va_loc], best_pred_val))
            per_fold_rae.append(fold_val_rae)
            selected_Ks = [K_list[i] for i in best_combo]
            per_fold_selected_combos.append({
                "kf_seed": int(kf_seed),
                "fold": int(fold_i),
                "selected_Ks": selected_Ks,
                "r": int(len(selected_Ks)),
                "tr_rae": float(best_rae),
                "val_rae": fold_val_rae,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
            })
            combo_choice_counter[tuple(selected_Ks)] += 1

        # Per-seed pooled
        rae_seed = float(rae(y_unb, pooled))
        per_seed_outer_rae.append(rae_seed)
        per_seed_pooled_oof.append(pooled)
        per_fold_summary = "  ".join([f"f{i}={r:.4f}" for i, r in enumerate(per_fold_rae)])
        print(f"   seed={kf_seed}  pooled_outer_val_RAE={rae_seed:.4f}   per_fold: {per_fold_summary}")

    per_seed_outer_rae_arr = np.asarray(per_seed_outer_rae, dtype=np.float64)
    outer_val_mean_rae = float(per_seed_outer_rae_arr.mean())
    outer_val_std_rae = float(per_seed_outer_rae_arr.std(ddof=1))
    print(f"\n[outer] mean over {len(KF_SEEDS)} seeds = {outer_val_mean_rae:.4f}  "
          f"std = {outer_val_std_rae:.4f}")

    # ---- Mean over seeds: average pooled predictions for stability ----
    pred_oof_outer = np.mean(np.column_stack(per_seed_pooled_oof), axis=1)
    pooled_avg_rae = float(rae(y_unb, pred_oof_outer))
    print(f"[outer] mean-of-pooled RAE          = {pooled_avg_rae:.4f}")

    # ---- Choose deploy combo: MODE across all seed*fold picks ----
    print("\n" + "-" * 78)
    print(f"STEP 3: deploy combo selection -- MODE across {len(per_fold_selected_combos)} picks")
    print("-" * 78)
    top_combos = combo_choice_counter.most_common(10)
    print(f"   top 10 most-selected combos:")
    for cks, cnt in top_combos:
        print(f"     {cnt:3d}x  Ks={list(cks)}  (r={len(cks)})")

    deploy_Ks = list(top_combos[0][0])
    deploy_cols = np.array([K_idx[k] for k in deploy_Ks], dtype=np.intp)
    pred_te_deploy = K_te_stack[:, deploy_cols].mean(axis=1)
    pred_oof_deploy_all = K_oof_stack[:, deploy_cols].mean(axis=1)
    rae_deploy_insample = float(rae(y_unb, pred_oof_deploy_all))
    print(f"\n   deploy_Ks (MODE) = {deploy_Ks}  (r={len(deploy_Ks)})")
    print(f"   deploy_combo full-data in-sample RAE = {rae_deploy_insample:.4f}  "
          f"(this number is selection-biased, NOT honest)")
    print(f"   pred_te mean={pred_te_deploy.mean():.3f}  std={pred_te_deploy.std():.3f}")

    # ---- Gate ----
    if outer_val_mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE_HOLDOUT_VALIDATED"
    elif outer_val_mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL"
    else:
        verdict = "SELECTION_BIAS_CONFIRMED"
    delta_vs_nb2621_insample = outer_val_mean_rae - NB2621_INSAMPLE_REF
    print(f"\n[gate] outer_val_mean_rae={outer_val_mean_rae:.4f}  -> {verdict}")
    print(f"       delta vs nb2621 in-sample = {delta_vs_nb2621_insample:+.4f}")

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 4: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_outer.astype(np.float32))
    np.save(te_path, pred_te_deploy.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_holdout_combo_search.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    te_unb_in = float(rae(y_unb, pred_te_deploy[unb_idx]))
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")

    summary = {
        "tag": TAG,
        "method": "outer_5fold_scaffold_cv_over_K_combo_search",
        "paradigm": "selection_validated_no_in_sample_leak",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "all_Ks": K_list,
        "per_K_full_oof_rae": {str(k): per_K_rae[k] for k in K_list},
        "min_r": MIN_R,
        "max_r": MAX_R,
        "n_combos_searched_per_fold": n_combos,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "per_seed_outer_val_rae": per_seed_outer_rae,
        "outer_val_mean_rae": outer_val_mean_rae,
        "outer_val_std_rae": outer_val_std_rae,
        "outer_pooled_avg_rae": pooled_avg_rae,
        "per_fold_selected_combos": per_fold_selected_combos,
        "top10_mode_combos": [
            {"selected_Ks": list(cks), "count": int(cnt)} for cks, cnt in top_combos
        ],
        "deploy_Ks": deploy_Ks,
        "deploy_combo_full_data_insample_rae": rae_deploy_insample,
        "deploy_te_unb_insample_rae": te_unb_in,
        "te_mean": float(pred_te_deploy.mean()),
        "te_std": float(pred_te_deploy.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "delta_vs_nb2621_insample": delta_vs_nb2621_insample,
        "nb2621_insample_ref": NB2621_INSAMPLE_REF,
        "nb2604_ref": NB2604_REF,
        "nb2611_ref": NB2611_REF,
        "nb2171_ref": NB2171_REF,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   outer_val_mean_rae        = {outer_val_mean_rae:.4f} +/- {outer_val_std_rae:.4f}")
    print(f"   pooled_avg_rae            = {pooled_avg_rae:.4f}")
    print(f"   nb2621 in-sample ref      = {NB2621_INSAMPLE_REF}")
    print(f"   delta vs nb2621 in-sample = {delta_vs_nb2621_insample:+.4f}")
    print(f"   verdict                   = {verdict}")
    print(f"   deploy_Ks (MODE)          = {deploy_Ks}")
    print(f"   deploy in-sample RAE      = {rae_deploy_insample:.4f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")
    print(f"   wall                      = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "outer_val_mean_rae",
        "outer_val_std_rae",
        "verdict",
        "delta_vs_nb2621_insample",
        "deploy_Ks",
        "deploy_te_unb_insample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
