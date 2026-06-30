"""nb3092 -- Sigmoid SLOPE SWEEP on nb3082 paradigm.

NEW PARADIGM (vs nb3082 fixed slope=10):
    nb3082 froze SIGMOID_GAIN=10 and verified BETTER_THAN_NB3070 at mean
    0.4475 (matching nb3080 ceiling). This script sweeps slope across
    {3, 5, 8, 12, 20} on the same value-anchored K18-q50/iqr schedule:

        z      = (K18_pred - q50) / iqr           (fold-train computed)
        w_K18  = 0.9 - 0.4 * sigmoid(slope * z)

    Slope interpretation:
        slope=3   -- very smooth, near-linear transition over +/-1 IQR
        slope=5   -- gentle smooth
        slope=8   -- moderate (close to nb3082's 10)
        slope=12  -- crisper than nb3082
        slope=20  -- near hard-split at q50 (approaches nb3070 paradigm)

PROTOCOL (per slope, per kf_seed, 5-fold scaffold split):
    1. Fold-train: q50, iqr of K18 predictions.
    2. Fold-val:   z = (K18_val - q50) / iqr.
    3. w_K18 = 0.9 - 0.4 * sigmoid(slope * z).
    4. pred = w_K18 * K18 + (1 - w_K18) * K19.
    5. Pooled RAE across the 5 outer folds.
    Repeat for 15 fresh kf_seeds {1126..1140}; report mean +/- std + 95% CI.

GATE (on best-slope 15-seed mean):
    mean < 0.4475 -> "BETTER_THAN_NB3080"
    else          -> "FAIL"

Outputs:
    data/processed/nb3092_summary.json   -- full sweep + best slope
    data/processed/nb3092_pred_oof.npy   -- (253,) float32 median-seed OOF at best slope
    data/processed/te_nb3092.npy         -- (513,) float32 deploy te at best slope
    submissions/nb3092_sigmoid_slope_sweep.csv -- only on PROMOTE verdict
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

TAG = "nb3092"
PARENT_TAG = "nb3082"

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
KF_SEEDS = list(range(1126, 1141))  # 15 fresh seeds {1126..1140}

# -- Sigmoid schedule ----------------------------------------------------------
W_BASE_HI = 0.90
W_SPREAD = 0.40
SLOPE_SWEEP = [3.0, 5.0, 8.0, 12.0, 20.0]

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB3080 = 0.4475

# -- References ----------------------------------------------------------------
REF_NB3080 = 0.4475
REF_NB3082 = 0.4475
REF_NB3070 = 0.4477
REF_NB3030 = 0.4509
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e_z = np.exp(z[neg])
    out[neg] = e_z / (1.0 + e_z)
    return out


def _blend_sigmoid(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q50: float,
    iqr: float,
    slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row sigmoid blend keyed on K18 prediction value."""
    if iqr <= 0:
        iqr = 1.0
    z = (p_k18 - q50) / iqr
    w_k18 = W_BASE_HI - W_SPREAD * _sigmoid(slope * z)
    blended = w_k18 * p_k18 + (1.0 - w_k18) * p_k19
    return blended, w_k18


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
    slope: float,
) -> dict:
    """Run sigmoid blend pipeline at a single kf_seed and slope."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_w_means = []
    for tr_loc, va_loc in splits:
        train_p_k18 = P_unb[tr_loc, 0]
        q50 = float(np.median(train_p_k18))
        q25, q75 = np.percentile(train_p_k18, [25.0, 75.0])
        iqr = float(q75 - q25) if (q75 - q25) > 0 else 1.0

        val_p_k18 = P_unb[va_loc, 0]
        val_p_k19 = P_unb[va_loc, 1]
        val_pred, val_w = _blend_sigmoid(val_p_k18, val_p_k19, q50, iqr, slope)
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_w_means.append(float(val_w.mean()))

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed} slope={slope}: splits didn't cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "slope": float(slope),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "fold_w_k18_mean": float(np.mean(fold_w_means)),
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Sigmoid SLOPE SWEEP on nb3082 paradigm")
    print(f"          w_K18 = {W_BASE_HI} - {W_SPREAD} * sigmoid(slope * z)")
    print(f"          z = (K18_pred - q50) / iqr  (fold-train computed)")
    print(f"          slopes  = {SLOPE_SWEEP}")
    print(f"          kf_seeds = {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}} (n={len(KF_SEEDS)})")
    print(f"          gate: best mean < {GATE_BETTER_THAN_NB3080} -> BETTER_THAN_NB3080")
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

    # -- Multi-slope sweep ----------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(SLOPE_SWEEP)} slopes x {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    per_slope_results = {}
    per_slope_oofs = {}
    for slope in SLOPE_SWEEP:
        ts = time.time()
        pooled_raes = []
        seed_records = []
        oof_stack = []
        for s in KF_SEEDS:
            res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s, slope)
            pooled_raes.append(res["pooled_rae"])
            oof_stack.append(res["oof"])
            seed_records.append({
                "kf_seed": res["kf_seed"],
                "pooled_rae": round(res["pooled_rae"], 4),
                "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
                "fold_w_k18_mean": round(res["fold_w_k18_mean"], 4),
            })
        arr = np.asarray(pooled_raes, dtype=np.float64)
        n_seeds = len(arr)
        mean_rae = float(arr.mean())
        std_rae = float(arr.std(ddof=1)) if n_seeds > 1 else 0.0
        sem = std_rae / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
        t_mult = 2.145
        ci_low = mean_rae - t_mult * sem
        ci_high = mean_rae + t_mult * sem
        median_rae = float(np.median(arr))
        med_seed_idx = int(np.argsort(arr)[n_seeds // 2])
        median_seed = KF_SEEDS[med_seed_idx]

        per_slope_results[float(slope)] = {
            "slope": float(slope),
            "mean_rae": round(mean_rae, 4),
            "std_rae": round(std_rae, 4),
            "sem_rae": round(sem, 4),
            "ci95_low": round(ci_low, 4),
            "ci95_high": round(ci_high, 4),
            "median_rae": round(median_rae, 4),
            "min_rae": round(float(arr.min()), 4),
            "max_rae": round(float(arr.max()), 4),
            "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
            "median_seed": int(median_seed),
            "seed_records": seed_records,
        }
        per_slope_oofs[float(slope)] = oof_stack[med_seed_idx].astype(np.float32)

        print(f"   slope={slope:5.1f}: mean={mean_rae:.4f} +/- {std_rae:.4f}  "
              f"CI=[{ci_low:.4f}, {ci_high:.4f}]  med={median_rae:.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  "
              f"wall={time.time()-ts:.2f}s")

    # -- Pick best slope by mean_rae -----------------------------------------
    slopes_sorted = sorted(per_slope_results.items(), key=lambda kv: kv[1]["mean_rae"])
    best_slope, best_rec = slopes_sorted[0]
    best_mean = best_rec["mean_rae"]
    best_std = best_rec["std_rae"]
    best_ci_low = best_rec["ci95_low"]
    best_ci_high = best_rec["ci95_high"]
    print("\n" + "-" * 78)
    print("BEST SLOPE")
    print("-" * 78)
    print(f"   best slope = {best_slope}")
    print(f"   mean       = {best_mean:.4f}")
    print(f"   std        = {best_std:.4f}")
    print(f"   95% CI     = [{best_ci_low:.4f}, {best_ci_high:.4f}]")
    print(f"   delta vs nb3080 ceiling {REF_NB3080} = {best_mean - REF_NB3080:+.4f}")

    # -- Deploy at best slope ------------------------------------------------
    te_k18 = P_te[:, 0]
    te_k19 = P_te[:, 1]
    te_q50 = float(np.median(te_k18))
    te_q25, te_q75 = np.percentile(te_k18, [25.0, 75.0])
    te_iqr = float(te_q75 - te_q25) if (te_q75 - te_q25) > 0 else 1.0
    te_pred_raw, te_w_k18 = _blend_sigmoid(te_k18, te_k19, te_q50, te_iqr, best_slope)
    te_pred = np.clip(te_pred_raw.astype(np.float32), 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"\n   deploy q50={te_q50:.3f}  iqr={te_iqr:.3f}  slope={best_slope}")
    print(f"   deploy w_K18: mean={te_w_k18.mean():.3f}  "
          f"[{te_w_k18.min():.3f}, {te_w_k18.max():.3f}]")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    oof_for_save = per_slope_oofs[best_slope]

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if best_mean < GATE_BETTER_THAN_NB3080:
        verdict = "BETTER_THAN_NB3080"
        ladder_action = (
            f"PROMOTE candidate. nb3092 slope={best_slope} mean "
            f"{best_mean:.4f} beats nb3080 ceiling {REF_NB3080:.4f} "
            f"({best_mean - REF_NB3080:+.4f}). Sigmoid-slope sweep extracts "
            "additional gain over fixed slope=10; re-verify with wider seed "
            "sweep (deep-30) before deploy."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3092 best slope={best_slope} mean {best_mean:.4f} "
            f"not better than nb3080 ceiling {REF_NB3080:.4f} "
            f"({best_mean - REF_NB3080:+.4f}). Sigmoid-slope axis is closed "
            "for this anchor pair. Keep nb3080 PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_sigmoid_slope_sweep.csv"
    if verdict == "BETTER_THAN_NB3080":
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
        "method": "sigmoid_slope_sweep_value_quantile_blend_K18_K19_deep30",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "schedule": {
            "form": "w_K18 = W_BASE_HI - W_SPREAD * sigmoid(slope * z)",
            "z_form": "z = (K18_pred - q50) / iqr",
            "W_BASE_HI": W_BASE_HI,
            "W_SPREAD": W_SPREAD,
            "slope_sweep": SLOPE_SWEEP,
        },
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_slope_results": {str(k): v for k, v in per_slope_results.items()},
        "slope_sorted_by_mean": [
            {"slope": float(s), "mean_rae": rec["mean_rae"],
             "std_rae": rec["std_rae"], "ci95_low": rec["ci95_low"],
             "ci95_high": rec["ci95_high"]}
            for s, rec in slopes_sorted
        ],
        "best_slope": float(best_slope),
        "best_mean_rae": round(best_mean, 4),
        "best_std_rae": round(best_std, 4),
        "best_ci95_low": round(best_ci_low, 4),
        "best_ci95_high": round(best_ci_high, 4),
        "best_median_rae": best_rec["median_rae"],
        "best_min_rae": best_rec["min_rae"],
        "best_max_rae": best_rec["max_rae"],
        "best_median_seed": best_rec["median_seed"],
        "ref_nb3080": REF_NB3080,
        "ref_nb3082": REF_NB3082,
        "ref_nb3070": REF_NB3070,
        "ref_nb3030": REF_NB3030,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3080": round(best_mean - REF_NB3080, 4),
        "deploy_slope": float(best_slope),
        "deploy_q50": te_q50,
        "deploy_iqr": te_iqr,
        "deploy_w_k18_mean": float(te_w_k18.mean()),
        "deploy_w_k18_min": float(te_w_k18.min()),
        "deploy_w_k18_max": float(te_w_k18.max()),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER_THAN_NB3080" else None,
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
    print(f"   best slope            = {best_slope}")
    print(f"   best mean_rae         = {best_mean:.4f} +/- {best_std:.4f}")
    print(f"   best 95% CI           = [{best_ci_low:.4f}, {best_ci_high:.4f}]")
    print(f"   delta vs nb3080       = {best_mean - REF_NB3080:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_slope", "best_mean_rae", "best_std_rae",
        "best_ci95_low", "best_ci95_high",
        "delta_vs_nb3080",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  slope_sorted_by_mean: {res.get('slope_sorted_by_mean')}")
