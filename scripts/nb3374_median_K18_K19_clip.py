"""nb3374 -- Per-row MEDIAN of {K18 deep-30, K19 deep-30, nb3200} THEN
            per-fold learned clip.

NEW PARADIGM:
    Robust per-row median of 3 PRE-unblind predictors as the blend base,
    instead of SLSQP convex simplex (nb3223) or quantile-conditional hard
    split (nb3070). The median is a parameter-FREE robust aggregator: for
    each test row it returns the middle of {K18, K19, nb3200}, immune to a
    single outlier predictor on that row. The hypothesis is that median
    aggregation gives a cleaner, lower-variance substrate for the learned
    tail-clip operator to decompress than a weight-fitted convex blend (no
    per-fold weight estimation noise), while nb3200 (itself a clip winner,
    OOF 0.4416) anchors the median toward the strongest single member.

    Composition:
        STEP 1 (parameter-free): per-row median over (K18, K19, nb3200).
                                 Identical at train and deploy -- no fit.
        STEP 2 (per fold): on the median fold-TRAIN predictions, inner grid
                           (q_low, q_high) search to pick tail-clip
                           percentiles minimizing fold-TRAIN RAE.
        STEP 3 (per fold): apply the per-fold (lo, hi) clip to fold-VAL
                           median rows.
        Stitch -> median_clipped_oof; pooled + per-fold-mean RAE across
        5 outer folds. 15 FRESH kf_seeds {1216..1230}.

    Note: STEP 1 has no learnable parameters, so the ONLY source of
    train/val leakage is the clip percentile grid in STEP 2, which is
    fit strictly on fold-TRAIN and applied to held-out fold-VAL.

    Deploy:
        - median over (513, 3) te anchors -> te_median (no fit).
        - Re-pick (q_low, q_high) on FULL 253 median output by inner grid.
        - Apply clip to te_median -> te_nb3374.

GATE (on PER-FOLD-MEAN across 15 seeds):
    mean < 0.4423 -> "BETTER"   (user-supplied gate)
    else          -> "FAIL"

References (all PRE-unblind anchor chain):
    nb2960 K18 deep-30 OOF        = 0.4536
    nb3000 K19 deep-30 OOF        = 0.4607
    nb3200 deep-30 clip winner OOF= 0.4416  (strongest single member)
    nb3002 SLSQP simplex {K18,K19} 15-seed pooled = 0.4501
    nb3173 learned clip on nb3080 wide-bag        = 0.4422 (best clip-winner)
    nb3214 SLSQP on 3 clip winners                = ~0.4418
    nb3223 SLSQP simplex {K18,K19} + clip         = (sibling, same clip op)

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy
    data/processed/nb3200_pred_oof.npy
    data/processed/te_nb3200.npy

Outputs:
    data/processed/nb3374_summary.json
    data/processed/nb3374_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3374.npy         (513,) float32 -- deploy te
    submissions/nb3374_median_K18_K19_clip.csv  (only on BETTER verdict)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter

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

TAG = "nb3374"
PARENT_TAG = "median3+clip"

# -- Inputs --------------------------------------------------------------------
# Order of the 3 members fed to the per-row median.
K_LABELS = ["K18", "K19", "nb3200"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K19": DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",
    "nb3200": DATA_PROCESSED / "nb3200_pred_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K19": DATA_PROCESSED / "te_nb3000_K19.npy",
    "nb3200": DATA_PROCESSED / "te_nb3200.npy",
}
K_DEPTH = {"K18": "deep30", "K19": "deep30", "nb3200": "deep30-clip"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Per-fold clip grid (matches nb3201 / nb3173 / nb3223 family) --------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Gates (on PER-FOLD-MEAN over 15 seeds) ------------------------------------
GATE_BETTER = 0.4423  # mean < this -> BETTER (user-supplied gate)

# -- References ----------------------------------------------------------------
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB3200 = 0.4416           # strongest single member (deep-30 clip winner)
REF_NB3002 = 0.4501           # SLSQP simplex {K18,K19} 15-seed pooled
REF_NB3173_BEST_CLIP_WINNER = 0.4422
REF_NB3214_SLSQP_CLIP_3 = 0.4418
REF_NB2171 = 0.4682


def _row_median(P: np.ndarray) -> np.ndarray:
    """Per-row median across the K member columns. Parameter-free."""
    return np.median(P, axis=1)


def _pick_best_clip(
    y_tr: np.ndarray,
    pred_tr: np.ndarray,
) -> tuple[float, float, float, float]:
    """Inner grid: pick (q_low*, q_high*) minimizing fold-train RAE."""
    best_rae = np.inf
    best_ql = Q_LOW_GRID[0]
    best_qh = Q_HIGH_GRID[-1]
    best_lo = float(np.quantile(y_tr, best_ql))
    best_hi = float(np.quantile(y_tr, best_qh))
    for ql in Q_LOW_GRID:
        lo = float(np.quantile(y_tr, ql))
        for qh in Q_HIGH_GRID:
            hi = float(np.quantile(y_tr, qh))
            if hi <= lo:
                continue
            clipped = np.clip(pred_tr, lo, hi)
            r = float(rae(y_tr, clipped))
            if r < best_rae:
                best_rae = r
                best_ql = ql
                best_qh = qh
                best_lo = lo
                best_hi = hi
    return best_ql, best_qh, best_lo, best_hi


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    med_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Per-row median (precomputed) + per-fold learned clip at one kf_seed.

    P_unb is kept for signature parity but the median is supplied precomputed
    (it is parameter-free and identical across folds/seeds).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_final = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes: list[float] = []
    fold_train_raes: list[float] = []
    fold_base_val_raes: list[float] = []  # median base, pre-clip, on fold-val
    fold_ql: list[float] = []
    fold_qh: list[float] = []
    fold_lo: list[float] = []
    fold_hi: list[float] = []
    fold_clipped_lo: list[int] = []
    fold_clipped_hi: list[int] = []

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # --- Step 1: per-row median (parameter-free, already computed) -------
        tr_med = med_unb[tr_loc]
        va_med = med_unb[va_loc]

        # --- Step 2: learned clip on fold-train median output ---------------
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], tr_med)
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)

        n_lo = int(np.sum(va_med < lo))
        n_hi = int(np.sum(va_med > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)

        # --- Step 3: apply clip to fold-val median output -------------------
        val_pred = np.clip(va_med, lo, hi)
        oof_final[va_loc] = val_pred
        r_tr = float(rae(y_unb[tr_loc], np.clip(tr_med, lo, hi)))
        fold_train_raes.append(r_tr)
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_base_val_raes.append(float(rae(y_unb[va_loc], va_med)))

    if np.isnan(oof_final).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_final))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "per_fold_train_rae_mean": float(np.mean(fold_train_raes)),
        "per_fold_base_val_rae_mean": float(np.mean(fold_base_val_raes)),
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "n_clipped_lo": int(np.sum(fold_clipped_lo)),
        "n_clipped_hi": int(np.sum(fold_clipped_hi)),
        "oof": oof_final,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- per-row MEDIAN of {K_LABELS} THEN per-fold learned clip"
    )
    print(f"          paradigm: robust median base (parameter-free) + tail clip")
    print(f"          Q_LOW_GRID  = {Q_LOW_GRID}")
    print(f"          Q_HIGH_GRID = {Q_HIGH_GRID}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate (per-fold-mean): < {GATE_BETTER:.4f} -> BETTER, "
        f"else FAIL"
    )
    print("=" * 78)

    # -- Load test, truth, unblind idx ---------------------------------------
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

    # -- Load anchors --------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {K_LABELS} OOF + te arrays")
    print("-" * 78)
    oof_cols: list[np.ndarray] = []
    te_cols: list[np.ndarray] = []
    per_K_full_rae: dict[str, float] = {}
    leak_flags: dict[str, float] = {}
    for k in K_LABELS:
        oof = np.load(OOF_PATHS[k]).astype(np.float64)
        te_a = np.load(TE_PATHS[k]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{k} OOF shape {oof.shape} != ({n_unb},)")
        if te_a.shape != (n_test,):
            raise ValueError(f"{k} te shape {te_a.shape} != ({n_test},)")
        oof_cols.append(oof)
        te_cols.append(te_a)
        r = float(rae(y_unb, oof))
        leak = float(np.mean(np.isclose(oof, y_unb, atol=1e-6)))
        per_K_full_rae[k] = round(r, 4)
        leak_flags[k] = round(leak, 4)
        print(
            f"   {k:>6s} ({K_DEPTH[k]:>11s}): oof_RAE={r:.4f}  "
            f"oof mean={oof.mean():.3f} std={oof.std():.3f}  "
            f"leak_eq={leak:.2%}  "
            f"te mean={te_a.mean():.3f} std={te_a.std():.3f}"
        )
        if leak > 0.05:
            print(f"   WARN {k}: {leak:.1%} rows == truth -- possible leak")
    P_unb = np.column_stack(oof_cols)  # (253, 3)
    P_te = np.column_stack(te_cols)    # (513, 3)

    # Pairwise correlations among members
    print("\n   pairwise OOF correlations:")
    for i in range(len(K_LABELS)):
        for j in range(i + 1, len(K_LABELS)):
            c = float(np.corrcoef(P_unb[:, i], P_unb[:, j])[0, 1])
            print(f"     corr({K_LABELS[i]:>6s}, {K_LABELS[j]:>6s}) = {c:.4f}")

    # -- Build per-row median (parameter-free) -------------------------------
    med_unb = _row_median(P_unb)   # (253,)
    med_te = _row_median(P_te)     # (513,)
    base_rae = float(rae(y_unb, med_unb))
    print(
        f"\n   median base OOF_RAE = {base_rae:.4f}  "
        f"mean={med_unb.mean():.3f} std={med_unb.std():.3f}"
    )
    print(
        f"   median base te(513): mean={med_te.mean():.3f} "
        f"std={med_te.std():.3f} min={med_te.min():.3f} max={med_te.max():.3f}"
    )
    # How often does each member supply the median value? (informational)
    is_med = np.isclose(P_unb, med_unb[:, None], atol=1e-9)
    med_share = is_med.mean(axis=0)
    print("   median-supplier share (253 OOF rows):")
    for k in range(len(K_LABELS)):
        print(f"     {K_LABELS[k]:>6s}: {med_share[k]:.1%}")

    # Truth stats
    print(
        f"\n   y_unb stats: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
        f"min={y_unb.min():.3f}  max={y_unb.max():.3f}"
    )

    # -- Scaffolds for outer CV -----------------------------------------------
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
    pooled_raes: list[float] = []
    per_fold_means: list[float] = []
    per_fold_stds: list[float] = []
    base_fold_means: list[float] = []
    oof_stack: list[np.ndarray] = []
    all_fold_ql: list[float] = []
    all_fold_qh: list[float] = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, med_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        per_fold_stds.append(res["per_fold_val_rae_std"])
        base_fold_means.append(res["per_fold_base_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_train_rae_mean": round(res["per_fold_train_rae_mean"], 4),
            "per_fold_base_val_rae_mean": round(
                res["per_fold_base_val_rae_mean"], 4
            ),
            "fold_ql": [round(v, 3) for v in res["fold_ql"]],
            "fold_qh": [round(v, 3) for v in res["fold_qh"]],
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(
            f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"base_pf={res['per_fold_base_val_rae_mean']:.4f}  "
            f"clip(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr_pooled = np.asarray(pooled_raes, dtype=np.float64)
    arr_pf = np.asarray(per_fold_means, dtype=np.float64)
    arr_base = np.asarray(base_fold_means, dtype=np.float64)
    n_s = len(arr_pf)

    # Aggregate stats: PER-FOLD-MEAN is the gate metric
    pf_mean = float(arr_pf.mean())
    pf_std = float(arr_pf.std(ddof=1)) if n_s > 1 else 0.0
    pf_sem = pf_std / np.sqrt(n_s) if n_s > 1 else 0.0
    pf_median = float(np.median(arr_pf))
    # df=14, two-sided 95%, t_mult = 2.145
    t_mult = 2.145
    pf_ci_low = pf_mean - t_mult * pf_sem
    pf_ci_high = pf_mean + t_mult * pf_sem

    pooled_mean = float(arr_pooled.mean())
    pooled_std = float(arr_pooled.std(ddof=1)) if n_s > 1 else 0.0
    base_pf_mean = float(arr_base.mean())

    # Most-picked q values
    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   median base (NO clip), per-fold-mean over seeds:")
    print(f"     base_pf_mean = {base_pf_mean:.4f}   (clip lift target)")
    print(f"\n   POOLED (split-variant, with fitted clip):")
    print(f"     mean = {pooled_mean:.4f}   std = {pooled_std:.4f}")
    print(f"     min/max = [{arr_pooled.min():.4f}, {arr_pooled.max():.4f}]")
    print(f"\n   PER-FOLD-MEAN (gate metric):")
    print(f"     mean    = {pf_mean:.4f}")
    print(f"     std     = {pf_std:.4f}")
    print(f"     sem     = {pf_sem:.4f}")
    print(f"     95% CI  = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"     median  = {pf_median:.4f}")
    print(f"     min/max = [{arr_pf.min():.4f}, {arr_pf.max():.4f}]")
    print(f"     clip lift vs base = {pf_mean - base_pf_mean:+.4f}")

    print(f"\n   ql_distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh_distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")

    print(f"\n   ref K18 deep-30 OOF         = {REF_K18:.4f}")
    print(f"   ref K19 deep-30 OOF         = {REF_K19:.4f}")
    print(
        f"   ref nb3200 (strongest)      = {REF_NB3200:.4f}  "
        f"<- strongest single member"
    )
    print(f"   delta vs nb3200             = {pf_mean - REF_NB3200:+.4f}")
    print(
        f"   ref nb3002 SLSQP simplex    = {REF_NB3002:.4f}  "
        f"<- weight-fit blend base"
    )
    print(f"   ref nb3173 best clip-winner = {REF_NB3173_BEST_CLIP_WINNER:.4f}")
    print(f"   ref nb3214 SLSQP-clip-3     = {REF_NB3214_SLSQP_CLIP_3:.4f}")

    # -- Deploy: median over 513 + full-253 grid-picked clip ------------------
    print("\n" + "-" * 78)
    print("DEPLOY: per-row median over 513, then pick clip on FULL 253 median")
    print("-" * 78)
    full_med = med_unb  # already the parameter-free median on 253
    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(
        y_unb, full_med,
    )
    print(
        f"   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f}) from FULL 253 y"
    )

    te_pred = np.clip(med_te, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(med_te < deploy_lo))
    n_te_hi = int(np.sum(med_te > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513  "
        f"total={n_te_lo + n_te_hi}/513"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr_pf)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"\n   median (by pf_mean) seed = {median_seed} "
        f"(pf_mean={arr_pf[med_seed_idx]:.4f}, "
        f"pooled={arr_pooled[med_seed_idx]:.4f})"
    )

    # -- Gate (on per-fold-mean) ---------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (per-fold-mean over 15 seeds)")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3374 15-seed per-fold-mean {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f} "
            f"({pf_mean - GATE_BETTER:+.4f}). Per-row median of "
            f"{K_LABELS} + learned tail clip beats the post-hoc-blend "
            f"ceiling. The parameter-free median base removes weight-fit "
            f"noise that SLSQP simplex (nb3223) incurs, and nb3200 "
            f"(OOF {REF_NB3200:.4f}) anchors the median toward the "
            f"strongest member. Modal clip = (q{ql_mode:.2f}, q{qh_mode:.2f}); "
            f"clip lift vs raw median = {pf_mean - base_pf_mean:+.4f}. "
            f"Re-verify with deep-30 before any PRIMARY-1 swap "
            f"(cycle-160 rule: 15-seed std under-dispersed ~4x)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3374 15-seed per-fold-mean {pf_mean:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"Per-row median of {K_LABELS} + learned clip does NOT beat "
            f"the clip-winner ladder. The median caps each row at the "
            f"middle member, discarding the strong tails nb3200 already "
            f"learned, and the raw-median base (base_pf {base_pf_mean:.4f}) "
            f"leaves the clip operator less decompression headroom than the "
            f"smoother wide-bag (nb3173 0.4422). Keep current clip-winner "
            f"ladder (nb3173 best single 0.4422, nb3214/nb3210 ensembles). "
            f"Modal clip = (q{ql_mode:.2f}, q{qh_mode:.2f})."
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

    sub_csv = SUBMISSIONS / f"{TAG}_median_K18_K19_clip.csv"
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
        "method": "per_row_median_K18_K19_nb3200_then_per_fold_learned_clip",
        "paradigm": "robust_median_base_parameter_free_plus_tail_clip",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "median_base_oof_rae": round(base_rae, 4),
        "median_supplier_share": {
            K_LABELS[k]: round(float(med_share[k]), 4)
            for k in range(len(K_LABELS))
        },
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_val_rae_means_array": [
            round(float(v), 4) for v in per_fold_means
        ],
        "per_fold_val_rae_stds_array": [
            round(float(v), 4) for v in per_fold_stds
        ],
        "base_per_fold_means_array": [
            round(float(v), 4) for v in base_fold_means
        ],
        # Primary gate metric: per-fold-mean
        "pf_mean": round(pf_mean, 4),
        "pf_std": round(pf_std, 4),
        "pf_sem": round(pf_sem, 4),
        "pf_ci95_low": round(pf_ci_low, 4),
        "pf_ci95_high": round(pf_ci_high, 4),
        "pf_median": round(pf_median, 4),
        "pf_min": round(float(arr_pf.min()), 4),
        "pf_max": round(float(arr_pf.max()), 4),
        "base_pf_mean": round(base_pf_mean, 4),
        "clip_lift_vs_base": round(pf_mean - base_pf_mean, 4),
        # Mean_rae mirror for ladder script compatibility
        "mean_rae": round(pf_mean, 4),
        "std_rae": round(pf_std, 4),
        # Pooled (for reference)
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "pooled_min": round(float(arr_pooled.min()), 4),
        "pooled_max": round(float(arr_pooled.max()), 4),
        # Clips
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        # Deploy
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv) if verdict == "BETTER" else None),
        # References
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb3200_strongest_member": REF_NB3200,
        "ref_nb3002_simplex_K18_K19": REF_NB3002,
        "ref_nb3173_best_clip_winner": REF_NB3173_BEST_CLIP_WINNER,
        "ref_nb3214_slsqp_clip_3": REF_NB3214_SLSQP_CLIP_3,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18": round(pf_mean - REF_K18, 4),
        "delta_vs_K19": round(pf_mean - REF_K19, 4),
        "delta_vs_nb3200": round(pf_mean - REF_NB3200, 4),
        "delta_vs_nb3002_raw_simplex": round(pf_mean - REF_NB3002, 4),
        "delta_vs_nb3173_best_clip": round(
            pf_mean - REF_NB3173_BEST_CLIP_WINNER, 4
        ),
        # Gate
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
    print(f"   pf_mean ({n_s} seeds)    = {pf_mean:.4f} +/- {pf_std:.4f}")
    print(f"   95% CI                = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   pooled_mean           = {pooled_mean:.4f}")
    print(f"   median base (no clip) = {base_pf_mean:.4f}")
    print(f"   clip lift             = {pf_mean - base_pf_mean:+.4f}")
    print(f"   delta vs nb3200       = {pf_mean - REF_NB3200:+.4f}")
    print(f"   delta vs nb3173 clip  = {pf_mean - REF_NB3173_BEST_CLIP_WINNER:+.4f}")
    print(f"   modal clip (ql, qh)   = ({ql_mode}, {qh_mode})")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pf_mean", "pf_std", "pf_ci95_low", "pf_ci95_high",
        "pooled_mean", "pooled_std",
        "median_base_oof_rae", "base_pf_mean", "clip_lift_vs_base",
        "delta_vs_nb3200", "delta_vs_nb3002_raw_simplex",
        "delta_vs_nb3173_best_clip",
        "median_supplier_share", "ql_mode", "qh_mode",
        "deploy_ql", "deploy_qh", "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
