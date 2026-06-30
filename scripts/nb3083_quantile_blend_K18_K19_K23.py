"""nb3083 -- Quantile-conditional blend with K=23 as 3rd option (high-pred regime).

NEW PARADIGM:
    Use K23 (different K-pyramid) in the high-pred regime instead of K19. The
    per-fold q50 cutoff is computed EXCLUSIVELY from fold-TRAIN K18 OOF
    predictions (clean cross-fit). Low-half stays on the (K18, K19) pair to
    preserve nb3072-style behavior on inactives; high-half swaps K19 -> K23
    to test whether K23 carries additional information in the active tail.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    1. Inside each outer fold: q50 = median(K18_pred[fold_train])
       (split threshold is a function of fold-TRAIN ONLY).
    2. Apply to fold-VAL rows:
         row_pred_K18 <= q50  -> w = 0.8*K18 + 0.2*K19
         row_pred_K18 >  q50  -> w = 0.5*K18 + 0.5*K23     <- K23 in high regime
    3. Stitch the 5 fold-val predictions into oof_blend (253,);
       pooled_rae across folds.
    Repeat for 15 fresh kf_seeds {1111..1125}.

    Deploy:
        - Compute q50 = median of K18 OOF on the FULL 253 (deploy proxy).
        - For te (513): rows where te_K18 <= q50 -> (0.8*K18 + 0.2*K19);
          else (0.5*K18 + 0.5*K23). Apply, clip [3, 9], save te_nb3083.

GATE (on 15-seed mean):
    mean < 0.4477 -> "BETTER_THAN_NB3070"
    else          -> "FAIL"

References:
    nb2960 K18 deep-30 OOF                = 0.4536
    nb3000 K19 deep-30 OOF                = 0.4607
    nb3020 K23 deep-30 OOF                = ?    (loaded at runtime)
    nb3030 wide-seed 15-mean ceiling      = 0.4509
    nb3070 wide-seed verify 15-mean       = 0.4477  <- gate
    nb3072 fold-TRAIN q50 K18+K19 verify  = ?
    nb2171 prior post-hoc ceiling         = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy
    data/processed/nb3020_K23_30seed_oof.npy
    data/processed/te_nb3020_K23.npy

Outputs:
    data/processed/nb3083_summary.json
    data/processed/nb3083_pred_oof.npy  (253,) float32 -- median-seed OOF
    data/processed/te_nb3083.npy        (513,) float32 -- deploy te
    submissions/nb3083_quantile_blend_K18_K19_K23.csv (only on BETTER_THAN_NB3070)
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

TAG = "nb3083"
PARENT_TAG = "nb3072"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K19", "K23"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K19": DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",
    "K23": DATA_PROCESSED / "nb3020_K23_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K19": DATA_PROCESSED / "te_nb3000_K19.npy",
    "K23": DATA_PROCESSED / "te_nb3020_K23.npy",
}
K_DEPTH = {"K18": "deep30", "K19": "deep30", "K23": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1111, 1126))  # 15 fresh seeds {1111..1125}

# -- Quantile-conditional weights ---------------------------------------------
# Low regime: K18+K19 (preserve nb3072 inactive-tail behavior)
W_LOW_K18, W_LOW_K19 = 0.8, 0.2
# High regime: K18+K23  (NEW: swap K19 -> K23 in active tail)
W_HIGH_K18, W_HIGH_K23 = 0.5, 0.5
QUANTILE_CUT = 0.5  # median split, fold-train only

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB3070 = 0.4477

# -- References ----------------------------------------------------------------
REF_NB3030 = 0.4509
REF_NB3070 = 0.4477
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _blend_quantile_conditional(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    p_k23: np.ndarray,
    q50: float,
) -> np.ndarray:
    """Per-row hard-split blend by q50 cutoff (fold-train-derived).

    Low: 0.8*K18 + 0.2*K19
    High: 0.5*K18 + 0.5*K23
    """
    low_mask = p_k18 <= q50
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = W_LOW_K18 * p_k18[low_mask] + W_LOW_K19 * p_k19[low_mask]
    out[~low_mask] = (
        W_HIGH_K18 * p_k18[~low_mask] + W_HIGH_K23 * p_k23[~low_mask]
    )
    return out


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run quantile-conditional blend at one kf_seed.

    q50 is computed PER-FOLD from fold-TRAIN K18 OOF only.
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_q50s = []
    fold_high_share = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # *** q50 from fold-TRAIN ONLY -- no fold-val info leakage ***
        q50 = float(np.median(P_unb[tr_loc, 0]))
        fold_q50s.append(q50)
        val_p_k18 = P_unb[va_loc, 0]
        val_p_k19 = P_unb[va_loc, 1]
        val_p_k23 = P_unb[va_loc, 2]
        val_pred = _blend_quantile_conditional(
            val_p_k18, val_p_k19, val_p_k23, q50,
        )
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_high_share.append(float(np.mean(val_p_k18 > q50)))

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_q50_mean": float(np.mean(fold_q50s)),
        "fold_q50_std": float(np.std(fold_q50s, ddof=1)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Quantile-conditional blend (K23 in high regime)")
    print(f"          low (K18<=q50): (K18={W_LOW_K18}, K19={W_LOW_K19})")
    print(f"          high (K18 >q50): (K18={W_HIGH_K18}, K23={W_HIGH_K23})")
    print(f"          kf_seeds = {KF_SEEDS[0]}..{KF_SEEDS[-1]} (n={len(KF_SEEDS)})")
    print(f"          gate: mean < {GATE_BETTER_THAN_NB3070} -> BETTER_THAN_NB3070")
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

    # -- Load K18, K19, K23 deep-30 anchor OOFs + te arrays ------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18, K19, K23 deep-30 OOFs and te arrays")
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

    P_unb = np.column_stack(oof_cols)  # (253, 3)
    P_te = np.column_stack(te_cols)    # (513, 3)

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # Pairwise corrs (3 pairs)
    corr_mat = np.corrcoef(P_unb.T)
    pair_corrs = {
        "K18_K19": round(float(corr_mat[0, 1]), 4),
        "K18_K23": round(float(corr_mat[0, 2]), 4),
        "K19_K23": round(float(corr_mat[1, 2]), 4),
    }
    print(f"   pairwise corrs: {pair_corrs}")

    # -- Scaffolds (kf_seed independent) -------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Multi-seed sweep -----------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(KF_SEEDS)} kf_seeds "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}  (fold-TRAIN q50 only)")
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
            "fold_q50_mean": round(res["fold_q50_mean"], 4),
            "fold_q50_std": round(res["fold_q50_std"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
        })
        print(f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
              f"q50_mean={res['fold_q50_mean']:.3f}  "
              f"high_share={res['fold_high_share_mean']:.2f}  "
              f"wall={time.time()-ts:.2f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = std_rae / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    # 95% CI via t-multiplier (n=15, df=14, t~2.145)
    t_mult = 2.145
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({len(KF_SEEDS)} seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   sem     = {sem:.4f}")
    print(f"   95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median  = {median_rae:.4f}")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"\n   ref nb3070 wide-seed verify  = {REF_NB3070:.4f}")
    print(f"   ref nb3030 wide-seed ceiling = {REF_NB3030:.4f}")
    print(f"   delta vs nb3070              = {mean_rae - REF_NB3070:+.4f}")
    print(f"   delta vs nb3030              = {mean_rae - REF_NB3030:+.4f}")

    # -- Deploy: q50 from FULL 253 K18 OOF, then blend te --------------------
    deploy_q50 = float(np.median(P_unb[:, 0]))
    te_k18 = P_te[:, 0]
    te_k19 = P_te[:, 1]
    te_k23 = P_te[:, 2]
    te_pred = _blend_quantile_conditional(
        te_k18, te_k19, te_k23, deploy_q50,
    ).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    te_low_share = float(np.mean(te_k18 <= deploy_q50))
    print(f"\n   deploy q50 (full K18 OOF median) = {deploy_q50:.4f}")
    print(f"   te(513) low-half share = {te_low_share:.3f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[len(arr) // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER_THAN_NB3070:
        verdict = "BETTER_THAN_NB3070"
        ladder_action = (
            f"PROMOTE candidate. nb3083 K23-high-regime quantile blend 15-mean "
            f"{mean_rae:.4f} beats nb3070 ceiling {REF_NB3070:.4f} "
            f"({mean_rae - REF_NB3070:+.4f}). K23 carries information in the "
            "active tail beyond what K19 provides."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3083 K23-high-regime 15-mean {mean_rae:.4f} "
            f"not better than nb3070 ceiling {REF_NB3070:.4f} "
            f"({mean_rae - REF_NB3070:+.4f}). Keep nb3070 / prior PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_quantile_blend_K18_K19_K23.csv"
    if verdict == "BETTER_THAN_NB3070":
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
        "method": "per_fold_TRAIN_q50_quantile_conditional_blend_K18_K19_K23_deep30",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": pair_corrs,
        "w_low": {"K18": W_LOW_K18, "K19": W_LOW_K19},
        "w_high": {"K18": W_HIGH_K18, "K23": W_HIGH_K23},
        "quantile_cut": QUANTILE_CUT,
        "q50_source": "fold_train_only",
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
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
        "ref_nb3030": REF_NB3030,
        "ref_nb3070": REF_NB3070,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3070": round(mean_rae - REF_NB3070, 4),
        "delta_vs_nb3030": round(mean_rae - REF_NB3030, 4),
        "deploy_q50": round(deploy_q50, 4),
        "te_low_share": round(te_low_share, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER_THAN_NB3070" else None,
        "gate_better_than_nb3070": GATE_BETTER_THAN_NB3070,
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
    print(f"   mean_rae ({len(KF_SEEDS)} seeds) = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3070       = {mean_rae - REF_NB3070:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3070", "delta_vs_nb3030",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
