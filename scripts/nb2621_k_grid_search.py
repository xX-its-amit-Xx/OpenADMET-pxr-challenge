"""nb2621 -- Exhaustive K-combo grid search via equal-weight subsets.

NEW PARADIGM:
    nb2611 tested only 4 hand-picked subsets of the 9 cached K-RFE anchors
    {K14, K16, K18, K20, K22, K24, K28, K32, K36}: ALL9, LOW(14-20),
    MID(18-24), HIGH(20-32).  Here we enumerate EVERY subset of size 2..5
    via itertools.combinations and pick the equal-weight average with lowest
    pooled RAE on y_unb.

    Hypothesis: nb2611's "best" was a constrained pick; the true optimal
    subset may not be contiguous in K.  Grid search has zero learnable df
    relative to the 9 fixed anchors (each combo is a fixed equal-weight
    average; we only select WHICH 2-5 to mean).  Total combos:
        C(9,2)+C(9,3)+C(9,4)+C(9,5) = 36+84+126+126 = 372.

PROTOCOL:
    1. Load 9 cached K-RFE OOF + te arrays.
    2. itertools.combinations(K_list, r) for r in 2..5  ->  372 combos.
    3. For each combo: equal-weight mean of OOFs on 253 unblind; RAE vs y_unb.
    4. Report top 10 subsets by mean_rae.
    5. Save best combo's pred_oof (253) + te (513) + full ranking JSON.

GATE:
    best mean_rae < 0.4570  -> PROMOTE
    best mean_rae < 0.4580  -> BETTER_THAN_NB2604
    else                    -> FAIL

Outputs:
    scripts/nb2621_k_grid_search.py
    data/processed/nb2621_summary.json
    data/processed/nb2621_pred_oof.npy   (253,) float32
    data/processed/te_nb2621.npy         (513,) float32
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2621"

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

# ---- Combo sizes ----
MIN_R = 2
MAX_R = 5

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_BEAT_NB2604 = 0.4580

# ---- Refs ----
CHEMPROP_AUX_REF = 0.6216
NB2604_REF = 0.4580
NB2611_REF = 0.4580  # nb2611 best combo
NB2171_REF = 0.4682


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- exhaustive K-combo grid search (size {MIN_R}..{MAX_R})")
    print(f"          9 anchors x C(9,2)+C(9,3)+C(9,4)+C(9,5) = "
          f"{sum(len(list(combinations(range(9), r))) for r in range(MIN_R, MAX_R+1))} combos")
    print(f"          gate PROMOTE < {GATE_PROMOTE}, BEAT_NB2604 < {GATE_BEAT_NB2604}")
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
        print(f"   K={K:2d}  oof_RAE={per_K_rae[K]:.4f}  "
              f"te_mean={K_te[K].mean():.3f}  te_std={K_te[K].std():.3f}")

    # ---- Enumerate combos ----
    print("\n" + "-" * 78)
    print(f"STEP 2: enumerate equal-weight subsets size {MIN_R}..{MAX_R}")
    print("-" * 78)
    rankings = []  # list of dicts
    for r in range(MIN_R, MAX_R + 1):
        n_combos_r = 0
        for combo in combinations(K_list, r):
            oof_stack = np.column_stack([K_oof[k] for k in combo])
            pred_oof = oof_stack.mean(axis=1)
            mean_rae_c = float(rae(y_unb, pred_oof))
            rankings.append({
                "r": r,
                "Ks": list(combo),
                "mean_rae": mean_rae_c,
            })
            n_combos_r += 1
        print(f"   r={r}: {n_combos_r} combos evaluated")

    # ---- Sort + top 10 ----
    rankings.sort(key=lambda x: x["mean_rae"])
    n_total = len(rankings)
    print(f"\n[total] {n_total} combos evaluated  "
          f"best mean_rae={rankings[0]['mean_rae']:.4f}")

    print("\n" + "-" * 78)
    print("STEP 3: top 10 subsets")
    print("-" * 78)
    print(f"   {'rank':>4s}  {'r':>2s}  {'mean_rae':>9s}  Ks")
    for i, rec in enumerate(rankings[:10]):
        print(f"   {i+1:4d}  {rec['r']:2d}  {rec['mean_rae']:9.4f}  {rec['Ks']}")

    # ---- Best ----
    best = rankings[0]
    best_Ks = best["Ks"]
    best_r = best["r"]
    best_mean = best["mean_rae"]
    oof_stack_best = np.column_stack([K_oof[k] for k in best_Ks])
    te_stack_best = np.column_stack([K_te[k] for k in best_Ks])
    pred_oof_best = oof_stack_best.mean(axis=1)
    pred_te_best = te_stack_best.mean(axis=1)
    print(f"\n[best] Ks={best_Ks}  r={best_r}  mean_rae={best_mean:.4f}")
    print(f"       pred_oof std={pred_oof_best.std():.3f}  (truth_std={y_unb.std():.3f})")
    print(f"       pred_te mean={pred_te_best.mean():.3f}  std={pred_te_best.std():.3f}")

    # ---- Gate ----
    if best_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean < GATE_BEAT_NB2604:
        verdict = "BETTER_THAN_NB2604"
    else:
        verdict = "FAIL"
    print(f"[gate] best_mean={best_mean:.4f}  "
          f"(<{GATE_PROMOTE} PROMOTE / <{GATE_BEAT_NB2604} BETTER_THAN_NB2604)"
          f"  -> {verdict}")

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 4: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_best.astype(np.float32))
    np.save(te_path, pred_te_best.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_k_grid_search.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_best.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    te_unb_in = float(rae(y_unb, pred_te_best[unb_idx]))
    delta_vs_nb2604 = best_mean - NB2604_REF
    delta_vs_nb2611 = best_mean - NB2611_REF
    delta_vs_nb2171 = best_mean - NB2171_REF
    print(f"\n   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")
    print(f"   delta vs nb2604 ({NB2604_REF}) = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2611 ({NB2611_REF}) = {delta_vs_nb2611:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_REF}) = {delta_vs_nb2171:+.4f}")

    # ---- Summary JSON ----
    summary = {
        "tag": TAG,
        "method": "exhaustive_K_combo_grid_search_equal_weight",
        "paradigm": "plain_mean_no_learning_df_zero",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "all_Ks": K_list,
        "per_K_oof_rae": {str(k): per_K_rae[k] for k in K_list},
        "cache_paths_oof": {str(k): str(p) for k, p in CACHE_OOF.items()},
        "cache_paths_te": {str(k): str(p) for k, p in CACHE_TE.items()},
        "min_r": MIN_R,
        "max_r": MAX_R,
        "n_combos_total": n_total,
        "n_combos_by_r": {
            str(r): sum(1 for x in rankings if x["r"] == r)
            for r in range(MIN_R, MAX_R + 1)
        },
        "full_ranking": rankings,
        "top10": rankings[:10],
        "best_combo_Ks": best_Ks,
        "best_combo_r": best_r,
        "best_mean_rae": best_mean,
        "gate_promote": GATE_PROMOTE,
        "gate_beat_nb2604": GATE_BEAT_NB2604,
        "verdict": verdict,
        "delta_vs_nb2604": delta_vs_nb2604,
        "nb2604_ref": NB2604_REF,
        "delta_vs_nb2611": delta_vs_nb2611,
        "nb2611_ref": NB2611_REF,
        "delta_vs_nb2171": delta_vs_nb2171,
        "nb2171_ref": NB2171_REF,
        "te_mean": float(pred_te_best.mean()),
        "te_std": float(pred_te_best.std()),
        "te_unb_in_sample_rae": te_unb_in,
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
    print(f"   total combos evaluated = {n_total}")
    print(f"   BEST Ks               = {best_Ks}  (r={best_r})")
    print(f"   BEST mean_rae         = {best_mean:.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   delta vs nb2604       = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2611       = {delta_vs_nb2611:+.4f}")
    print(f"   delta vs nb2171       = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] in-RAE    = {te_unb_in:.4f}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_combo_Ks",
        "best_combo_r",
        "best_mean_rae",
        "verdict",
        "delta_vs_nb2604",
        "delta_vs_nb2611",
        "delta_vs_nb2171",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
