"""nb3102 -- Sigmoid blend anchored at q=0.35 (not q50) on {K18, K19} deep-30.

NEW PARADIGM (vs nb3082 sigmoid centered at q50):
    nb3082 centered the sigmoid at the *median* (q50) of fold-train K18
    predictions. The nb3090 finer q_cut sweep showed the *hard-split*
    optimum lives at q_cut=0.35 (NOT q50): best combo
    (q=0.35, w_low=0.95, w_high=0.40) with mean RAE 0.4472.

    nb3102 hypothesis: a SMOOTH sigmoid centered at the q=0.35 cut point
    (matching nb3090's optimum location) with asymptotes that match the
    nb3090 winning weights (low->0.95, high->0.40) will preserve the
    bias-axis location nb3090 found AND inherit the smoothness benefit
    nb3082's sigmoid offered, breaking the 0.4472 hard-split ceiling
    without re-introducing the q50-mislocation cost.

    Per-fold schedule (q35, iqr computed on the fold-TRAIN K18 predictions):
        z = (K18_pred - q35) / iqr
        w_K18 = 0.95 - 0.55 * sigmoid(8 * z)
        # K18_pred << q35 (z->-inf): w_K18 -> 0.95, w_K19 -> 0.05
        # K18_pred == q35 (z=0):     w_K18 = 0.675, w_K19 = 0.325
        # K18_pred >> q35 (z->+inf): w_K18 -> 0.40, w_K19 -> 0.60

PROTOCOL (per kf_seed, 5-fold scaffold split):
    1. In fold-train: compute q35, iqr of K18 predictions.
    2. In fold-val: compute z = (K18_val - q35) / iqr.
    3. w_K18 = 0.95 - 0.55 * sigmoid(8 * z).
    4. pred = w_K18 * K18 + (1 - w_K18) * K19, stitched into oof_blend (253,).
    5. Pooled RAE across the 5 outer folds.
    Repeat for 15 fresh kf_seeds {1141..1155}; report mean +/- std + 95% CI.

    Deploy:
        - Compute q35, iqr from K18 te predictions (full 513).
        - z = (K18_te - q35) / iqr, w_K18 = 0.95 - 0.55 * sigmoid(8 * z).
        - Blend, clip [3, 9].

GATE (on 15-seed mean):
    mean < 0.4472 -> "BETTER_THAN_NB3090"
    else          -> "FAIL"

References:
    nb2960 K18 deep-30 OOF        = 0.4536
    nb3000 K19 deep-30 OOF        = 0.4607
    nb3030 wide-seed verify       = 0.4509
    nb3070 hard-split q50         = 0.4477
    nb3082 sigmoid centered q50   = ?
    nb3090 finer q_cut q=0.35     = 0.4472 <- CEILING TO BEAT
    nb2171 prior post-hoc ceiling = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3102_summary.json
    data/processed/nb3102_pred_oof.npy  (253,) float32 -- median-seed OOF
    data/processed/te_nb3102.npy        (513,) float32 -- deploy te
    submissions/nb3102_sigmoid_q35_anchored.csv (only on PROMOTE verdict)
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

TAG = "nb3102"
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

# -- Sigmoid schedule (anchored at q=0.35) -------------------------------------
# w_K18 = W_BASE_HI - W_SPREAD * sigmoid(SIGMOID_GAIN * z)
# z = (K18_pred - q35) / iqr  (fold-train computed q35, iqr)
# Asymptotes chosen to match nb3090 best: low w=0.95, high w=0.40
Q_ANCHOR = 0.35      # quantile location of sigmoid center
W_BASE_HI = 0.95     # asymptote at z -> -inf (low K18, K18 trusted more)
W_SPREAD = 0.55      # range so asymptote at z -> +inf is 0.40
SIGMOID_GAIN = 8.0   # steepness factor

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB3090 = 0.4472  # mean < this -> BETTER_THAN_NB3090

# -- References ----------------------------------------------------------------
REF_NB3090 = 0.4472
REF_NB3082 = None   # populated if available
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


def _blend_sigmoid_q35(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q35: float,
    iqr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row sigmoid blend keyed on K18 prediction, anchored at q35.

    w_K18 = W_BASE_HI - W_SPREAD * sigmoid(SIGMOID_GAIN * (p_k18 - q35) / iqr)

    Returns (blended_pred, w_K18_per_row).
    """
    if iqr <= 0:
        iqr = 1.0
    z = (p_k18 - q35) / iqr
    w_k18 = W_BASE_HI - W_SPREAD * _sigmoid(SIGMOID_GAIN * z)
    blended = w_k18 * p_k18 + (1.0 - w_k18) * p_k19
    return blended, w_k18


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run sigmoid-q35 blend pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_w_means = []
    fold_w_mins = []
    fold_w_maxs = []
    fold_q35s = []
    fold_iqrs = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        train_p_k18 = P_unb[tr_loc, 0]
        q35 = float(np.percentile(train_p_k18, Q_ANCHOR * 100.0))
        q25, q75 = np.percentile(train_p_k18, [25.0, 75.0])
        iqr = float(q75 - q25) if (q75 - q25) > 0 else 1.0

        val_p_k18 = P_unb[va_loc, 0]
        val_p_k19 = P_unb[va_loc, 1]
        val_pred, val_w = _blend_sigmoid_q35(val_p_k18, val_p_k19, q35, iqr)
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_w_means.append(float(val_w.mean()))
        fold_w_mins.append(float(val_w.min()))
        fold_w_maxs.append(float(val_w.max()))
        fold_q35s.append(q35)
        fold_iqrs.append(iqr)

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_w_k18_mean": float(np.mean(fold_w_means)),
        "fold_w_k18_min": float(np.min(fold_w_mins)),
        "fold_w_k18_max": float(np.max(fold_w_maxs)),
        "fold_q35_mean": float(np.mean(fold_q35s)),
        "fold_iqr_mean": float(np.mean(fold_iqrs)),
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Sigmoid blend anchored at q={Q_ANCHOR} on {K_LABELS} deep-30")
    print(f"          w_K18 = {W_BASE_HI} - {W_SPREAD} * sigmoid({SIGMOID_GAIN} * z)")
    print(f"          z = (K18_pred - q35) / iqr  (fold-train computed)")
    print(f"          asymptotes: z->-inf w_K18={W_BASE_HI:.2f}; "
          f"z=0 w_K18={W_BASE_HI - 0.5*W_SPREAD:.3f}; "
          f"z->+inf w_K18={W_BASE_HI - W_SPREAD:.2f}")
    print(f"          kf_seeds = {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}} (n={len(KF_SEEDS)})")
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

    # Correlation
    corr = float(np.corrcoef(P_unb.T)[0, 1])
    print(f"   pairwise corr({K_LABELS[0]}, {K_LABELS[1]}) = {corr:.4f}")

    # -- Tail diagnostic: where does each anchor do better? ------------------
    p_k18 = P_unb[:, 0]
    p_k19 = P_unb[:, 1]
    full_q35 = float(np.percentile(p_k18, Q_ANCHOR * 100.0))
    q25_full, q75_full = np.percentile(p_k18, [25.0, 75.0])
    full_iqr = float(q75_full - q25_full)
    low_mask_full = p_k18 <= full_q35
    rae_k18_low = float(rae(y_unb[low_mask_full], p_k18[low_mask_full]))
    rae_k19_low = float(rae(y_unb[low_mask_full], p_k19[low_mask_full]))
    rae_k18_high = float(rae(y_unb[~low_mask_full], p_k18[~low_mask_full]))
    rae_k19_high = float(rae(y_unb[~low_mask_full], p_k19[~low_mask_full]))
    print(f"\n   full-pool K18 q35 = {full_q35:.4f}  iqr = {full_iqr:.4f}")
    print(f"   below-q35 (n={low_mask_full.sum()}): "
          f"K18 RAE={rae_k18_low:.4f}, K19 RAE={rae_k19_low:.4f}")
    print(f"   above-q35 (n={(~low_mask_full).sum()}): "
          f"K18 RAE={rae_k18_high:.4f}, K19 RAE={rae_k19_high:.4f}")

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
    print(f"SWEEP: {len(KF_SEEDS)} kf_seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
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
            "fold_w_k18_mean": round(res["fold_w_k18_mean"], 4),
            "fold_w_k18_min": round(res["fold_w_k18_min"], 4),
            "fold_w_k18_max": round(res["fold_w_k18_max"], 4),
            "fold_q35_mean": round(res["fold_q35_mean"], 4),
            "fold_iqr_mean": round(res["fold_iqr_mean"], 4),
        })
        print(f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
              f"w_K18_mean={res['fold_w_k18_mean']:.3f}  "
              f"[{res['fold_w_k18_min']:.3f}, {res['fold_w_k18_max']:.3f}]  "
              f"q35={res['fold_q35_mean']:.3f} iqr={res['fold_iqr_mean']:.3f}  "
              f"wall={time.time()-ts:.2f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_seeds = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_seeds > 1 else 0.0
    sem = std_rae / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
    # 95% CI via t-multiplier (n=15, df=14, t~2.145)
    t_mult = 2.145
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_seeds} seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   sem     = {sem:.4f}")
    print(f"   95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median  = {median_rae:.4f}")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"\n   ref nb3090 q=0.35 ceiling     = {REF_NB3090:.4f}")
    print(f"   delta vs nb3090               = {mean_rae - REF_NB3090:+.4f}")

    # -- Deploy: use full-513 K18 q35/iqr for sigmoid centering --------------
    te_k18 = P_te[:, 0]
    te_k19 = P_te[:, 1]
    te_q35 = float(np.percentile(te_k18, Q_ANCHOR * 100.0))
    te_q25, te_q75 = np.percentile(te_k18, [25.0, 75.0])
    te_iqr = float(te_q75 - te_q25) if (te_q75 - te_q25) > 0 else 1.0
    te_pred_raw, te_w_k18 = _blend_sigmoid_q35(te_k18, te_k19, te_q35, te_iqr)
    te_pred = np.clip(te_pred_raw.astype(np.float32), 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"\n   deploy q35={te_q35:.3f}  iqr={te_iqr:.3f}")
    print(f"   deploy w_K18: mean={te_w_k18.mean():.3f}  "
          f"[{te_w_k18.min():.3f}, {te_w_k18.max():.3f}]")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_seeds // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER_THAN_NB3090:
        verdict = "BETTER_THAN_NB3090"
        ladder_action = (
            f"PROMOTE candidate. nb3102 sigmoid-q35-anchored mean "
            f"{mean_rae:.4f} beats nb3090 ceiling {REF_NB3090:.4f} "
            f"({mean_rae - REF_NB3090:+.4f}). "
            "Re-centering sigmoid at q=0.35 (matching nb3090 hard-split "
            "optimum location) extracts gain over q50-centered nb3082; "
            "re-verify with wider seed sweep before deploy."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3102 sigmoid-q35-anchored mean {mean_rae:.4f} "
            f"not better than nb3090 ceiling {REF_NB3090:.4f} "
            f"({mean_rae - REF_NB3090:+.4f}). "
            "Re-centering sigmoid at q=0.35 does not extract smoothness "
            "gain over hard-split at same location. Keep nb3090 PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_sigmoid_q35_anchored.csv"
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
        "method": "sigmoid_q35_anchored_value_blend_K18_K19_deep30",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "tail_diagnostic": {
            "full_q35": round(full_q35, 4),
            "full_iqr": round(full_iqr, 4),
            "below_q35_K18_rae": round(rae_k18_low, 4),
            "below_q35_K19_rae": round(rae_k19_low, 4),
            "above_q35_K18_rae": round(rae_k18_high, 4),
            "above_q35_K19_rae": round(rae_k19_high, 4),
        },
        "schedule": {
            "form": "w_K18 = W_BASE_HI - W_SPREAD * sigmoid(SIGMOID_GAIN * z)",
            "z_form": "z = (K18_pred - q35) / iqr",
            "Q_ANCHOR": Q_ANCHOR,
            "W_BASE_HI": W_BASE_HI,
            "W_SPREAD": W_SPREAD,
            "SIGMOID_GAIN": SIGMOID_GAIN,
            "w_K18_asymp_low": round(W_BASE_HI, 4),
            "w_K18_at_q35": round(W_BASE_HI - 0.5 * W_SPREAD, 4),
            "w_K18_asymp_high": round(W_BASE_HI - W_SPREAD, 4),
        },
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_seeds,
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
        "ref_nb3090": REF_NB3090,
        "ref_nb3070": REF_NB3070,
        "ref_nb3030": REF_NB3030,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3090": round(mean_rae - REF_NB3090, 4),
        "deploy_q35": te_q35,
        "deploy_iqr": te_iqr,
        "deploy_w_k18_mean": float(te_w_k18.mean()),
        "deploy_w_k18_min": float(te_w_k18.min()),
        "deploy_w_k18_max": float(te_w_k18.max()),
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
    print(f"   mean_rae ({n_seeds} seeds) = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3090       = {mean_rae - REF_NB3090:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3090",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  schedule: {res.get('schedule')}")
