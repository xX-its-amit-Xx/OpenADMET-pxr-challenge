"""nb3103 -- 4-region quantile-conditional blend on {K18, K19} deep-30.

NEW PARADIGM: extend nb3081 3-region split (q33/q66) to 4-region split (q25/q50/q75)
with descending w_K18 weights {0.95, 0.80, 0.55, 0.35}. Hypothesis: finer
partitioning of the K18 predicted-activity axis lets us trust K18 more on the
deepest "low-activity" tail (where K18 OOF RAE is best calibrated) and lean
harder on K19 as predicted activity rises into the K18 variance-compressed tail.

PROTOCOL (per kf_seed, 5-fold scaffold split, anchors LOADED no rebuild):
    1. Compute fold-train K18 q25, q50, q75 from K18_pred.
    2. Per fold-val row, partition by K18 predicted activity:
        pred_K18 <= q25                 -> w_K18 = 0.95, w_K19 = 0.05
        q25 < pred_K18 <= q50           -> w_K18 = 0.80, w_K19 = 0.20
        q50 < pred_K18 <= q75           -> w_K18 = 0.55, w_K19 = 0.45
        pred_K18 >  q75                 -> w_K18 = 0.35, w_K19 = 0.65
    3. Stitch into oof_blend (253,); pooled_rae across 5 outer folds.
    Repeat for 15 FRESH kf_seeds {1141..1155}. Report mean +/- std + 95% CI.

GATE (on 15-seed mean):
    mean < 0.4472 -> "BETTER_THAN_NB3090"
    else          -> "FAIL"

References:
    nb2960 K18 deep-30 OOF        = 0.4536
    nb3000 K19 deep-30 OOF        = 0.4607
    nb3030 wide-seed SLSQP        = 0.4509
    nb3063 quantile-conditional   = 0.4509-band (single-seed)
    nb3070 wide-seed q50 split    = ~0.4477 (15 seeds)
    nb3073 36-combo 5-seed best   = (q=0.4, w_low=0.9, w_high=0.5) mean 0.4445 (5 seeds)
    nb3080 wide-seed verify nb3073 = 0.4475 (15 seeds)
    nb3081 3-region q33/q66 split = ?       (parent paradigm)
    nb3090 finer 2-region grid    = 0.4472 (15 seeds) <- CEILING TO BEAT
    nb2171 prior post-hoc top     = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3103_summary.json
    data/processed/nb3103_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3103.npy         (513,) float32 -- deploy te
    submissions/nb3103_4region_quantile.csv  (only on BETTER_THAN_NB3090)
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

TAG = "nb3103"
PARENT_TAG = "nb3090"

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
KF_SEEDS = list(range(1141, 1156))  # 15 fresh seeds {1141..1155}

# -- 4-region split ------------------------------------------------------------
# Quantile cuts (per fold-train K18 predictions)
Q_CUTS = [0.25, 0.50, 0.75]
# Descending w_K18 weights per region (R1 .. R4)
W_K18 = [0.95, 0.80, 0.55, 0.35]
W_K19 = [1.0 - w for w in W_K18]
REGION_LABELS = [
    "R1 (<=q25)",
    "R2 (q25-q50)",
    "R3 (q50-q75)",
    "R4 (>q75)",
]

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB3090 = 0.4472

# -- References ----------------------------------------------------------------
REF_NB3090 = 0.4472
REF_NB3080 = 0.4475
REF_NB3030 = 0.4509
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _blend_4region(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q25: float,
    q50: float,
    q75: float,
    w_k18: list[float],
) -> np.ndarray:
    """Per-row 4-region hard-split blend on K18 predicted activity.

    R1 (p_k18 <= q25):       w_K18 = w_k18[0]
    R2 (q25 < p_k18 <= q50): w_K18 = w_k18[1]
    R3 (q50 < p_k18 <= q75): w_K18 = w_k18[2]
    R4 (p_k18 > q75):        w_K18 = w_k18[3]
    """
    out = np.empty_like(p_k18, dtype=np.float64)
    r1 = p_k18 <= q25
    r2 = (p_k18 > q25) & (p_k18 <= q50)
    r3 = (p_k18 > q50) & (p_k18 <= q75)
    r4 = p_k18 > q75
    out[r1] = w_k18[0] * p_k18[r1] + (1.0 - w_k18[0]) * p_k19[r1]
    out[r2] = w_k18[1] * p_k18[r2] + (1.0 - w_k18[1]) * p_k19[r2]
    out[r3] = w_k18[2] * p_k18[r3] + (1.0 - w_k18[2]) * p_k19[r3]
    out[r4] = w_k18[3] * p_k18[r4] + (1.0 - w_k18[3]) * p_k19[r4]
    return out


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run 4-region quantile-conditional blend pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    region_counts = np.zeros(4, dtype=np.int64)
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        q25 = float(np.quantile(P_unb[tr_loc, 0], 0.25))
        q50 = float(np.quantile(P_unb[tr_loc, 0], 0.50))
        q75 = float(np.quantile(P_unb[tr_loc, 0], 0.75))
        val_p_k18 = P_unb[va_loc, 0]
        val_p_k19 = P_unb[va_loc, 1]
        val_pred = _blend_4region(
            val_p_k18, val_p_k19, q25, q50, q75, W_K18,
        )
        oof_blend[va_loc] = val_pred
        # Track region assignment counts
        region_counts[0] += int(np.sum(val_p_k18 <= q25))
        region_counts[1] += int(np.sum((val_p_k18 > q25) & (val_p_k18 <= q50)))
        region_counts[2] += int(np.sum((val_p_k18 > q50) & (val_p_k18 <= q75)))
        region_counts[3] += int(np.sum(val_p_k18 > q75))

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "oof": oof_blend,
        "region_counts": region_counts.tolist(),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 4-region quantile-conditional blend on K18, K19 deep-30")
    print(f"          Q_CUTS    = {Q_CUTS}")
    print(f"          W_K18     = {W_K18}  (descending)")
    print(f"          W_K19     = {W_K19}")
    print(f"          regions   = {REGION_LABELS}")
    print(f"          kf_seeds  = {KF_SEEDS}")
    print(f"          gate: mean < {GATE_BETTER_THAN_NB3090} -> BETTER_THAN_NB3090")
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

    # -- Wide-seed evaluation ------------------------------------------------
    print("\n" + "-" * 78)
    print(f"WIDE-SEED EVAL: 15 kf_seeds {KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print("-" * 78)
    seed_results = []
    pooled_raes = []
    oof_stack = []
    region_counts_total = np.zeros(4, dtype=np.int64)
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        region_counts_total += np.asarray(res["region_counts"], dtype=np.int64)
        seed_results.append({
            "kf_seed": int(s),
            "pooled_rae": round(float(res["pooled_rae"]), 4),
            "region_counts": res["region_counts"],
        })
        print(f"   [seed {s}] pooled_rae = {res['pooled_rae']:.4f}  "
              f"regions = {res['region_counts']}  wall={time.time()-ts:.2f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se = std_rae / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    ci95 = (mean_rae - 1.96 * se, mean_rae + 1.96 * se)
    min_rae = float(arr.min())
    max_rae = float(arr.max())

    print("\n" + "-" * 78)
    print("WIDE-SEED SUMMARY")
    print("-" * 78)
    print(f"   n_seeds        = {len(KF_SEEDS)}")
    print(f"   mean_rae       = {mean_rae:.4f}")
    print(f"   std_rae        = {std_rae:.4f}")
    print(f"   se             = {se:.4f}")
    print(f"   95% CI         = [{ci95[0]:.4f}, {ci95[1]:.4f}]")
    print(f"   min/max        = [{min_rae:.4f}, {max_rae:.4f}]")
    print(f"   ref nb3090     = {REF_NB3090:.4f}")
    print(f"   delta vs nb3090 = {mean_rae - REF_NB3090:+.4f}")
    print(f"   region totals  = {region_counts_total.tolist()} "
          f"(R1+R2+R3+R4 over all folds*seeds)")

    # -- Deploy te (full-OOF K18 quantiles) ---------------------------------
    deploy_q25 = float(np.quantile(P_unb[:, 0], 0.25))
    deploy_q50 = float(np.quantile(P_unb[:, 0], 0.50))
    deploy_q75 = float(np.quantile(P_unb[:, 0], 0.75))
    te_pred = _blend_4region(
        P_te[:, 0], P_te[:, 1], deploy_q25, deploy_q50, deploy_q75, W_K18,
    ).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))

    # te region shares
    te_r1 = float(np.mean(P_te[:, 0] <= deploy_q25))
    te_r2 = float(np.mean((P_te[:, 0] > deploy_q25) & (P_te[:, 0] <= deploy_q50)))
    te_r3 = float(np.mean((P_te[:, 0] > deploy_q50) & (P_te[:, 0] <= deploy_q75)))
    te_r4 = float(np.mean(P_te[:, 0] > deploy_q75))

    print(f"\n   deploy q25/q50/q75 (full K18 OOF) = "
          f"{deploy_q25:.4f} / {deploy_q50:.4f} / {deploy_q75:.4f}")
    print(f"   te(513) region shares R1/R2/R3/R4 = "
          f"{te_r1:.3f} / {te_r2:.3f} / {te_r3:.3f} / {te_r4:.3f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    med_seed_idx = int(np.argsort(pooled_arr)[len(pooled_arr) // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={pooled_arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER_THAN_NB3090:
        verdict = "BETTER_THAN_NB3090"
        ladder_action = (
            f"PROMOTE candidate. nb3103 4-region quantile blend "
            f"(q25/q50/q75 with w_K18 = {W_K18}) "
            f"15-seed mean {mean_rae:.4f} beats nb3090 ceiling "
            f"{REF_NB3090:.4f} ({mean_rae - REF_NB3090:+.4f}). "
            "Decision-grade dispersion at 15 seeds; consider deploy. "
            "Note: deep-30 standard for final gate decisions per cycle-160 rule."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3103 4-region quantile blend "
            f"(q25/q50/q75 with w_K18 = {W_K18}) "
            f"15-seed mean {mean_rae:.4f} not better than nb3090 ceiling "
            f"{REF_NB3090:.4f} ({mean_rae - REF_NB3090:+.4f}). "
            "Finer 4-region partition with prescribed descending w_K18 weights "
            "does not break the 2-region quantile-conditional ceiling on the "
            "{K18, K19} deep-30 anchor pair. Keep nb3090 / prior PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_4region_quantile.csv"
    if verdict == "BETTER_THAN_NB3090":
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
        "method": "4region_quantile_q25_q50_q75_K18_K19_deep30",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "q_cuts": Q_CUTS,
        "w_K18": W_K18,
        "w_K19": W_K19,
        "region_labels": REGION_LABELS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_results": seed_results,
        "pooled_raes": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "se": round(se, 4),
        "ci95": [round(ci95[0], 4), round(ci95[1], 4)],
        "min_rae": round(min_rae, 4),
        "max_rae": round(max_rae, 4),
        "region_counts_total": region_counts_total.tolist(),
        "ref_nb3090": REF_NB3090,
        "ref_nb3080": REF_NB3080,
        "ref_nb3030": REF_NB3030,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3090": round(mean_rae - REF_NB3090, 4),
        "deploy_q25": round(deploy_q25, 4),
        "deploy_q50": round(deploy_q50, 4),
        "deploy_q75": round(deploy_q75, 4),
        "te_region_shares": {
            "R1": round(te_r1, 4),
            "R2": round(te_r2, 4),
            "R3": round(te_r3, 4),
            "R4": round(te_r4, 4),
        },
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER_THAN_NB3090" else None,
        "gate_better_than_nb3090": GATE_BETTER_THAN_NB3090,
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
    print(f"   4-region split with w_K18 = {W_K18}")
    print(f"   15-seed mean_rae          = {mean_rae:.4f}")
    print(f"   std / se                  = {std_rae:.4f} / {se:.4f}")
    print(f"   95% CI                    = [{ci95[0]:.4f}, {ci95[1]:.4f}]")
    print(f"   delta vs nb3090           = {mean_rae - REF_NB3090:+.4f}")
    print(f"   verdict                   = {verdict}")
    print(f"   wall                      = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95", "delta_vs_nb3090",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  region_counts_total: {res.get('region_counts_total')}")
    print(f"  te_region_shares: {res.get('te_region_shares')}")
