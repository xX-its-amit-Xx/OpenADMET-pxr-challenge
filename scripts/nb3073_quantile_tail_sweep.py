"""nb3073 -- 2-D sweep of (q_cut, w_K18_low, w_K18_high) for nb3063-style blend.

NEW PARADIGM:
    nb3063 fixed q_cut=0.5 with (w_K18_low=0.8, w_K18_high=0.5). This script
    sweeps q_cut and tail w_K18 values to find the optimal hard-split combo.

    Grid:
      q_cut       in {0.4, 0.5, 0.6, 0.7}              -- 4 vals
      w_K18_low   in {0.7, 0.8, 0.9}                   -- 3 vals (w_K19_low = 1-w)
      w_K18_high  in {0.4, 0.5, 0.6}                   -- 3 vals (w_K19_high = 1-w)

    36 combos. Per combo: 5 fresh seeds {1051..1055}, pooled scaffold-CV RAE,
    report mean. Best combo wins.

PROTOCOL (per combo, per kf_seed, 5-fold scaffold split):
    1. Compute fold-train K18 q_cut quantile threshold q.
    2. For fold-val rows where pred_K18 <= q -> use (w_K18_low, 1-w_K18_low).
       For fold-val rows where pred_K18 >  q -> use (w_K18_high, 1-w_K18_high).
    3. Stitch into oof_blend (253,); pooled_rae across the 5 outer folds.
    Repeat for 5 kf_seeds; report mean per combo.

GATE (on best combo's 5-seed mean):
    mean < 0.4509 -> "BETTER_THAN_NB3030"
    else          -> "FAIL"

References:
    nb2960 K18 deep-30 OOF        = 0.4536
    nb3000 K19 deep-30 OOF        = 0.4607
    nb3002 SLSQP per-fold K18,K19 = 0.4501 (single-kf=1001)
    nb3030 wide-seed verify       = 0.4509 (15-seed) <- CEILING TO BEAT
    nb3063 quantile-conditional   = (q=0.5, w_low=0.8, w_high=0.5) baseline
    nb2171 prior post-hoc ceiling = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3073_summary.json
    data/processed/nb3073_pred_oof.npy  (253,) float32 -- best combo median-seed OOF
    data/processed/te_nb3073.npy        (513,) float32 -- best combo deploy te
    submissions/nb3073_quantile_tail_sweep.csv (only on BETTER_THAN_NB3030 verdict)
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

TAG = "nb3073"
PARENT_TAG = "nb3063"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K19"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K19": DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K19": DATA_PROCESSED / "te_nb3000_K19.npy",
}
K_DEPTH = {"K18": "deep30", "K19": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1051, 1056))  # 5 fresh seeds {1051..1055}

# -- Sweep grid ----------------------------------------------------------------
Q_CUTS = [0.4, 0.5, 0.6, 0.7]
W_K18_LOWS = [0.7, 0.8, 0.9]
W_K18_HIGHS = [0.4, 0.5, 0.6]

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB3030 = 0.4509

# -- References ----------------------------------------------------------------
REF_NB3030 = 0.4509
REF_NB3063 = 0.4509  # nb3063 ceiling reference (same gate)
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _blend_quantile_conditional(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q_thr: float,
    w_low: float,
    w_high: float,
) -> np.ndarray:
    """Per-row hard-split blend.

    rows with p_k18 <= q_thr -> (w_low * p_k18 + (1-w_low) * p_k19)
    rows with p_k18 >  q_thr -> (w_high * p_k18 + (1-w_high) * p_k19)
    """
    low_mask = p_k18 <= q_thr
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = w_low * p_k18[low_mask] + (1.0 - w_low) * p_k19[low_mask]
    out[~low_mask] = w_high * p_k18[~low_mask] + (1.0 - w_high) * p_k19[~low_mask]
    return out


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
    q_cut: float,
    w_low: float,
    w_high: float,
) -> dict:
    """Run quantile-conditional blend pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # Compute fold-train K18 q_cut quantile (cut threshold)
        q_thr = float(np.quantile(P_unb[tr_loc, 0], q_cut))
        val_p_k18 = P_unb[va_loc, 0]
        val_p_k19 = P_unb[va_loc, 1]
        val_pred = _blend_quantile_conditional(
            val_p_k18, val_p_k19, q_thr, w_low, w_high,
        )
        oof_blend[va_loc] = val_pred

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "oof": oof_blend,
    }


def _eval_combo(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    q_cut: float,
    w_low: float,
    w_high: float,
) -> dict:
    """Evaluate one (q_cut, w_low, w_high) combo across all KF_SEEDS."""
    pooled_raes = []
    oof_stack = []
    for s in KF_SEEDS:
        res = _run_one_seed(
            P_unb, y_unb, unb_scaffolds, s, q_cut, w_low, w_high,
        )
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "q_cut": q_cut,
        "w_low": w_low,
        "w_high": w_high,
        "pooled_raes": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "min_rae": float(arr.min()),
        "max_rae": float(arr.max()),
        "oof_stack": oof_stack,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 2-D sweep (q_cut, w_K18_low, w_K18_high) for nb3063-style "
          f"hard-split blend")
    print(f"          Q_CUTS       = {Q_CUTS}")
    print(f"          W_K18_LOWS   = {W_K18_LOWS}")
    print(f"          W_K18_HIGHS  = {W_K18_HIGHS}")
    print(f"          combos       = {len(Q_CUTS)*len(W_K18_LOWS)*len(W_K18_HIGHS)}")
    print(f"          kf_seeds     = {KF_SEEDS}")
    print(f"          gate: best_mean < {GATE_BETTER_THAN_NB3030} -> BETTER_THAN_NB3030")
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

    # -- Load K18, K19 deep-30 anchor OOFs + te arrays ------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18, K19 deep-30 OOFs and te arrays")
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
        print(f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

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

    # -- Scaffolds (kf_seed independent) -------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- 2-D sweep -----------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(Q_CUTS)*len(W_K18_LOWS)*len(W_K18_HIGHS)} combos x "
          f"{len(KF_SEEDS)} seeds")
    print("-" * 78)
    combo_records = []
    best_combo = None
    best_mean = float("inf")
    best_oof_stack = None
    n_combos = len(Q_CUTS) * len(W_K18_LOWS) * len(W_K18_HIGHS)
    combo_i = 0
    for q_cut in Q_CUTS:
        for w_low in W_K18_LOWS:
            for w_high in W_K18_HIGHS:
                combo_i += 1
                ts = time.time()
                res = _eval_combo(
                    P_unb, y_unb, unb_scaffolds, q_cut, w_low, w_high,
                )
                record = {
                    "combo_idx": combo_i,
                    "q_cut": q_cut,
                    "w_K18_low": w_low,
                    "w_K18_high": w_high,
                    "pooled_raes": res["pooled_raes"],
                    "mean_rae": round(res["mean_rae"], 4),
                    "std_rae": round(res["std_rae"], 4),
                    "min_rae": round(res["min_rae"], 4),
                    "max_rae": round(res["max_rae"], 4),
                }
                combo_records.append(record)
                if res["mean_rae"] < best_mean:
                    best_mean = res["mean_rae"]
                    best_combo = record.copy()
                    best_oof_stack = res["oof_stack"]
                print(f"   [{combo_i:>2d}/{n_combos}] q={q_cut} "
                      f"w_low={w_low} w_high={w_high}  "
                      f"mean={res['mean_rae']:.4f}  std={res['std_rae']:.4f}  "
                      f"wall={time.time()-ts:.2f}s")

    # -- Best combo summary ---------------------------------------------------
    print("\n" + "-" * 78)
    print("BEST COMBO")
    print("-" * 78)
    print(f"   q_cut      = {best_combo['q_cut']}")
    print(f"   w_K18_low  = {best_combo['w_K18_low']}")
    print(f"   w_K18_high = {best_combo['w_K18_high']}")
    print(f"   mean_rae   = {best_combo['mean_rae']:.4f}")
    print(f"   std_rae    = {best_combo['std_rae']:.4f}")
    print(f"   min/max    = [{best_combo['min_rae']:.4f}, "
          f"{best_combo['max_rae']:.4f}]")
    print(f"   ref nb3030 = {REF_NB3030:.4f}")
    print(f"   delta      = {best_combo['mean_rae'] - REF_NB3030:+.4f}")

    # -- Deploy te for best combo --------------------------------------------
    deploy_q_thr = float(np.quantile(P_unb[:, 0], best_combo["q_cut"]))
    te_pred = _blend_quantile_conditional(
        P_te[:, 0], P_te[:, 1], deploy_q_thr,
        best_combo["w_K18_low"], best_combo["w_K18_high"],
    ).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    te_low_share = float(np.mean(P_te[:, 0] <= deploy_q_thr))
    print(f"\n   deploy q_thr (full K18 OOF q{best_combo['q_cut']}) = "
          f"{deploy_q_thr:.4f}")
    print(f"   te(513) low-half share = {te_low_share:.3f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage (best combo)
    pooled_arr = np.asarray(
        [r for r in best_combo["pooled_raes"]], dtype=np.float64,
    )
    med_seed_idx = int(np.argsort(pooled_arr)[len(pooled_arr) // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = best_oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={pooled_arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if best_mean < GATE_BETTER_THAN_NB3030:
        verdict = "BETTER_THAN_NB3030"
        ladder_action = (
            f"PROMOTE candidate. nb3073 2-D sweep best combo "
            f"(q={best_combo['q_cut']}, "
            f"w_low={best_combo['w_K18_low']}, "
            f"w_high={best_combo['w_K18_high']}) "
            f"mean {best_mean:.4f} beats nb3030 ceiling {REF_NB3030:.4f} "
            f"({best_mean - REF_NB3030:+.4f}). "
            "Re-verify with wide-seed (>=15) before deploy."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3073 2-D sweep best combo "
            f"(q={best_combo['q_cut']}, "
            f"w_low={best_combo['w_K18_low']}, "
            f"w_high={best_combo['w_K18_high']}) "
            f"mean {best_mean:.4f} not better than nb3030 ceiling "
            f"{REF_NB3030:.4f} ({best_mean - REF_NB3030:+.4f}). "
            "Hard-split quantile blending does not break the global-w "
            "simplex ceiling on this anchor pair under any tested combo. "
            "Keep nb3030 / prior PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_quantile_tail_sweep.csv"
    if verdict == "BETTER_THAN_NB3030":
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
        "method": "2D_quantile_tail_sweep_K18_K19_deep30",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "q_cuts": Q_CUTS,
        "w_K18_lows": W_K18_LOWS,
        "w_K18_highs": W_K18_HIGHS,
        "n_combos": n_combos,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "combo_records": combo_records,
        "best_combo": {
            "q_cut": best_combo["q_cut"],
            "w_K18_low": best_combo["w_K18_low"],
            "w_K18_high": best_combo["w_K18_high"],
            "mean_rae": round(best_mean, 4),
            "std_rae": round(best_combo["std_rae"], 4),
            "min_rae": round(best_combo["min_rae"], 4),
            "max_rae": round(best_combo["max_rae"], 4),
            "pooled_raes": best_combo["pooled_raes"],
        },
        "best_mean_rae": round(best_mean, 4),
        "ref_nb3030": REF_NB3030,
        "ref_nb3063": REF_NB3063,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3030": round(best_mean - REF_NB3030, 4),
        "deploy_q_thr": round(deploy_q_thr, 4),
        "te_low_share": round(te_low_share, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER_THAN_NB3030" else None,
        "gate_better_than_nb3030": GATE_BETTER_THAN_NB3030,
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
    print(f"   best combo (q, w_low, w_high) = "
          f"({best_combo['q_cut']}, {best_combo['w_K18_low']}, "
          f"{best_combo['w_K18_high']})")
    print(f"   best mean_rae (5 seeds)       = {best_mean:.4f}")
    print(f"   delta vs nb3030               = {best_mean - REF_NB3030:+.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_mean_rae", "delta_vs_nb3030",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  best_combo: {res.get('best_combo')}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
