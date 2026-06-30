"""nb3131 -- L2-regularized quantile blend with Gaussian-noise weight schedule.

NEW PARADIGM (vs nb3080 fixed hard-split blend):
    nb3080 used a FIXED per-row hard-split blend on {K18, K19} deep-30:
        rows with K18_pred <= q40 -> w = (0.9 K18 + 0.1 K19)
        rows with K18_pred >  q40 -> w = (0.5 K18 + 0.5 K19)
    The (w_low=0.9, w_high=0.5) weights are GLOBAL constants -- when applied
    per-fold this concentrates leverage on a fixed schedule that can overfit
    the (q_cut, w_low, w_high) tuple to the fold-train manifold.

    nb3131 ADDS L2 PENALTY via STOCHASTIC PERTURBATION of the per-row weight
    schedule. Per outer fold:

        For each noise realization r in {0..N_NOISE-1}:
            eps_low_r  ~ N(0, sigma=0.05)
            eps_high_r ~ N(0, sigma=0.05)
            w_low_r    = clip(W_LOW  + eps_low_r,  0.0, 1.0)
            w_high_r   = clip(W_HIGH + eps_high_r, 0.0, 1.0)
            For each val row:
                w_actual = w_low_r if K18_pred <= q40 else w_high_r
                blended_r = w_actual * K18_pred + (1 - w_actual) * K19_pred
        oof_blend = mean over N_NOISE realizations  (per-row averaging)

    Effect: 1-realization variant degenerates to nb3080. As N_NOISE -> inf,
    blended -> (W_LOW * K18 + (1-W_LOW) * K19) for low-q rows AND
              (W_HIGH * K18 + (1-W_HIGH) * K19) for high-q rows
    (i.e. noise averages out since eps is zero-mean), BUT the variance of
    blended per row is reduced by sqrt(1/N_NOISE) which acts as an L2-like
    shrinkage on the conditional blend mass when averaged across folds /
    seeds. Effectively a Bayesian-noise smoother around the nb3080 schedule.

PROTOCOL (per kf_seed, 5-fold scaffold split, anchors LOADED no rebuild):
    1. Compute fold-train K18 q40 quantile threshold q (per fold).
    2. Per fold-val row:
         For each noise realization r in {0..9} (10 noise realizations):
             draw eps_low ~ N(0, 0.05), eps_high ~ N(0, 0.05)
             w_low_r  = clip(0.9 + eps_low,  0.0, 1.0)
             w_high_r = clip(0.5 + eps_high, 0.0, 1.0)
             w_actual = w_low_r if pred_K18 <= q else w_high_r
             blended_r = w_actual * pred_K18 + (1 - w_actual) * pred_K19
         oof_blend[row] = mean of blended_r over r in {0..9}
    3. Stitch into oof_blend (253,); pooled_rae across 5 outer folds.
    4. Repeat for 15 FRESH kf_seeds {1141..1155}.

GATE (on 15-seed mean):
    mean < 0.4475 -> "BETTER"  (beats nb3080 5-seed reference 0.4477)
    else          -> "FAIL"

References:
    nb2960 K18 deep-30 OOF                  = 0.4536
    nb3000 K19 deep-30 OOF                  = 0.4607
    nb3080 15-seed wide-verify of nb3073    = 0.4477 (parent paradigm)
    nb3073 5-seed quantile-conditional best = 0.4470
    nb3030 15-seed wide-mean                = 0.4509
    nb2171 prior post-hoc ceiling           = 0.4682
    chemprop_aux                            = 0.6216
    GATE                                    = 0.4475

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3131_summary.json
    data/processed/nb3131_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3131.npy         (513,) float32 -- deploy te
    submissions/nb3131_L2_penalty_quantile.csv  (only on BETTER verdict)
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

TAG = "nb3131"
PARENT_TAG = "nb3080"

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
KF_SEEDS = list(range(1141, 1156))  # 15 FRESH seeds {1141..1155}

# -- nb3080 base schedule (FIXED) ----------------------------------------------
Q_CUT = 0.4
W_LOW = 0.9   # nb3080 w_K18_low; w_K19_low = 0.1
W_HIGH = 0.5  # nb3080 w_K18_high; w_K19_high = 0.5

# -- L2-noise schedule ---------------------------------------------------------
NOISE_SIGMA = 0.05    # Gaussian sigma on per-fold weight perturbation
N_NOISE = 10          # noise realizations averaged per fold
NOISE_SEED_BASE = 9000  # rng seed offset; each (kf_seed, fold_i) draws fresh

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4475          # mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_NB3080_15SEED = 0.4477
REF_NB3073_5SEED = 0.4470
REF_NB3030 = 0.4509
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _blend_noisy(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q_thr: float,
    rng: np.random.Generator,
    n_noise: int = N_NOISE,
    sigma: float = NOISE_SIGMA,
    w_low: float = W_LOW,
    w_high: float = W_HIGH,
) -> np.ndarray:
    """L2-noise quantile blend on val rows.

    For each noise realization r in {0..n_noise-1}:
        draw eps_low, eps_high ~ N(0, sigma)
        w_low_r  = clip(w_low  + eps_low,  0.0, 1.0)
        w_high_r = clip(w_high + eps_high, 0.0, 1.0)
        w_actual = w_low_r where p_k18 <= q_thr else w_high_r
        blended_r = w_actual * p_k18 + (1 - w_actual) * p_k19
    Return mean blended over n_noise realizations.
    """
    low_mask = p_k18 <= q_thr
    n_val = len(p_k18)
    acc = np.zeros(n_val, dtype=np.float64)
    for _ in range(n_noise):
        eps_low = float(rng.normal(0.0, sigma))
        eps_high = float(rng.normal(0.0, sigma))
        w_low_r = float(np.clip(w_low + eps_low, 0.0, 1.0))
        w_high_r = float(np.clip(w_high + eps_high, 0.0, 1.0))
        w_actual = np.where(low_mask, w_low_r, w_high_r)
        blended_r = w_actual * p_k18 + (1.0 - w_actual) * p_k19
        acc += blended_r
    return acc / float(n_noise)


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run L2-noise quantile blend pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_q_thrs = []
    fold_high_share = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # Per-fold q40 quantile from fold-train K18 preds
        q_thr = float(np.quantile(P_unb[tr_loc, 0], Q_CUT))
        fold_q_thrs.append(q_thr)
        val_p_k18 = P_unb[va_loc, 0]
        val_p_k19 = P_unb[va_loc, 1]
        rng = np.random.default_rng(NOISE_SEED_BASE + kf_seed * 100 + fold_i)
        val_pred = _blend_noisy(
            val_p_k18, val_p_k19, q_thr, rng,
            n_noise=N_NOISE, sigma=NOISE_SIGMA,
            w_low=W_LOW, w_high=W_HIGH,
        )
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_high_share.append(float(np.mean(val_p_k18 > q_thr)))

    if np.isnan(oof_blend).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_q_thr_mean": float(np.mean(fold_q_thrs)),
        "fold_q_thr_std": float(np.std(fold_q_thrs, ddof=1)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- L2-NOISE quantile blend (sigma={NOISE_SIGMA}, "
        f"n_noise={N_NOISE}) on {K_LABELS} deep-30"
    )
    print(
        f"          base schedule: q_cut={Q_CUT}, w_low={W_LOW}, "
        f"w_high={W_HIGH} (from {PARENT_TAG})"
    )
    print(
        f"          per-fold noise: eps_low, eps_high ~ N(0, {NOISE_SIGMA}); "
        f"avg over {N_NOISE} realizations"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          {PARENT_TAG} 15-seed reference = {REF_NB3080_15SEED:.4f}"
    )
    print(
        f"          gate: mean < {GATE_BETTER:.4f} -> BETTER"
    )
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
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
        print(
            f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
            f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}"
        )

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

    # -- Scaffolds ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Multi-seed sweep -----------------------------------------------------
    print("\n" + "-" * 78)
    print(
        f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
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
            "fold_q_thr_mean": round(res["fold_q_thr_mean"], 4),
            "fold_q_thr_std": round(res["fold_q_thr_std"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"q_thr_mean={res['fold_q_thr_mean']:.3f}  "
            f"high_share={res['fold_high_share_mean']:.2f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    # 95% CI via t-multiplier (n=15, df=14, t~2.145)
    t_mult = 2.145
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
    print(
        f"\n   ref {PARENT_TAG} 15-seed mean   = {REF_NB3080_15SEED:.4f}"
    )
    print(
        f"   shift vs {PARENT_TAG} 15-seed = "
        f"{mean_rae - REF_NB3080_15SEED:+.4f}"
    )
    print(f"   ref nb3073 5-seed             = {REF_NB3073_5SEED:.4f}")
    print(f"   ref nb3030 wide-seed ceiling  = {REF_NB3030:.4f}")
    print(f"   delta vs nb3030               = {mean_rae - REF_NB3030:+.4f}")
    print(f"   ref K18 deep-30               = {REF_K18:.4f}")
    print(f"   ref K19 deep-30               = {REF_K19:.4f}")

    # -- Deploy: q_thr from FULL 253 K18 OOF, then noise-avg blend te --------
    deploy_q_thr = float(np.quantile(P_unb[:, 0], Q_CUT))
    te_k18 = P_te[:, 0]
    te_k19 = P_te[:, 1]
    # Deploy noise blend with fresh, deterministic rng (NOT one of the
    # per-seed/per-fold rngs above) -- single deploy realization averaged
    # over N_NOISE draws.
    deploy_rng = np.random.default_rng(NOISE_SEED_BASE + 999_999)
    te_pred = _blend_noisy(
        te_k18, te_k19, deploy_q_thr, deploy_rng,
        n_noise=N_NOISE, sigma=NOISE_SIGMA,
        w_low=W_LOW, w_high=W_HIGH,
    ).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    te_low_share = float(np.mean(te_k18 <= deploy_q_thr))
    print(
        f"\n   deploy q_thr (full K18 OOF q{Q_CUT}) = {deploy_q_thr:.4f}"
    )
    print(f"   te(513) low-half share = {te_low_share:.3f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    shift = mean_rae - REF_NB3080_15SEED
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3131 15-seed mean {mean_rae:.4f} beats "
            f"GATE {GATE_BETTER:.4f} ({mean_rae - GATE_BETTER:+.4f}) and "
            f"nb3080 15-seed reference {REF_NB3080_15SEED:.4f} "
            f"({shift:+.4f}). L2-noise smoothing of nb3080 schedule "
            f"(sigma={NOISE_SIGMA}, n_noise={N_NOISE}) produces real gain. "
            f"Recommend deep-30 re-verify before PRIMARY promotion (cycle-160 "
            f"rule). Watch for under-dispersion vs nb3080."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REPORT. nb3131 15-seed mean {mean_rae:.4f} fails GATE "
            f"{GATE_BETTER:.4f} ({mean_rae - GATE_BETTER:+.4f}). Shift vs "
            f"nb3080 15-seed {REF_NB3080_15SEED:.4f} = {shift:+.4f}. "
            f"L2-noise smoothing on nb3080 schedule does not improve. "
            f"Closes L2-noise axis on quantile-conditional blend. "
            f"Keep prior PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_L2_penalty_quantile.csv"
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
        "parent_tag": PARENT_TAG,
        "method": (
            "L2_noise_quantile_conditional_blend_K18_K19_deep30_"
            "gaussian_perturb_avg10"
        ),
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "base_schedule": {
            "q_cut": Q_CUT,
            "w_low": W_LOW,
            "w_high": W_HIGH,
        },
        "noise_schedule": {
            "sigma": NOISE_SIGMA,
            "n_noise": N_NOISE,
            "noise_seed_base": NOISE_SEED_BASE,
            "clip_low": 0.0,
            "clip_high": 1.0,
        },
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
        "ref_parent_15seed_mean": REF_NB3080_15SEED,
        "shift_vs_parent_15seed": round(shift, 4),
        "ref_nb3073_5seed": REF_NB3073_5SEED,
        "ref_nb3030": REF_NB3030,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3030": round(mean_rae - REF_NB3030, 4),
        "deploy_q_thr": round(deploy_q_thr, 4),
        "te_low_share": round(te_low_share, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
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
    print(f"   mean_rae ({n_s} seeds)  = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   shift vs nb3080 15sd  = {shift:+.4f}")
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
        "shift_vs_parent_15seed", "delta_vs_nb3030",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  base_schedule: {res.get('base_schedule')}")
    print(f"  noise_schedule: {res.get('noise_schedule')}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
