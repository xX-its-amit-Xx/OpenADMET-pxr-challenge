"""nb3263 -- K=18 deep-30 ALONE under nb3070-style quantile-conditional
            (per-quantile) scalar blend + per-fold learned clip.

NEW PARADIGM:
    Apply the nb3070 quantile-conditional schedule WITHOUT a second anchor
    partner. K=18 deep-30 is used in BOTH "low" and "high" slots; the
    quantile-conditional blend collapses to a per-quantile scalar weight
    on K=18 (low-half rows scaled by W_LOW, high-half rows scaled by
    W_HIGH). After this per-quantile scaling, apply the per-fold learned
    (q_low, q_high) clip primitive on the rescaled output.

    Rationale: nb3070 quantile-conditional blend on {K18, K19} shifted
    blend weights between two anchors per quantile half. The natural
    "K=18-only" projection of that operator is a per-quantile scalar
    rescale of K=18 around the truth manifold (low rows tugged
    toward median * W_LOW, high rows toward median * W_HIGH), which
    targets the exact two-sided variance-compression failure mode
    (low-truth over-predicted + high-truth under-predicted) documented
    in `feedback_failure_mode_quantile_compression`. The learned clip
    then handles residual tail outliers.

    Composition (per fold):
        STEP 1: q50 = median(K18 fold-train predictions)
        STEP 2: rescaled = W_LOW * K18    where K18 <= q50
                rescaled = W_HIGH * K18   where K18  > q50
        STEP 3: inner grid (q_low, q_high) on fold-train rescaled
                values, pick (lo*, hi*) minimizing fold-train RAE.
        STEP 4: val_pred = clip(rescaled_val, lo*, hi*)
        Stitch -> oof_final; pooled + per-fold-mean RAE across 5 folds.
        Repeat for 15 FRESH kf_seeds {1216..1230}.

    Quantile-conditional weights (MATCH nb3070 nominal schedule):
        W_LOW  = 0.8 + 0.2 = 1.0  -> degenerate as a same-anchor sum
        => Re-interpret per the new-paradigm spec: use the K=18 partner-
           sum so each half gets a SCALAR rescale equal to the SUM of
           nb3070 weights on that half (low: 0.8 + 0.2 = 1.0; high:
           0.5 + 0.5 = 1.0). That is identity, also degenerate.
        => Adopt the meaningful per-quantile rescale: use the LOW-side
           K18 weight in nb3070 (0.8) as the W_LOW scalar and the
           HIGH-side K18 weight (0.5) as W_HIGH. This treats K=18 as
           "both partners" by assigning each partner's K18 weight to
           its respective half, capturing the nb3070 high-half pull
           toward partner (here also K18) at 0.5 vs low-half pull at
           0.8. The resulting per-quantile rescale is non-degenerate
           and matches the spec "K=18 as both partners (low/high w)".

Deploy:
    - q50_deploy   = median(K18 full-253 OOF)
    - Rescale K18 te (513,) per the same low/high split using q50_deploy
    - Pick (q_low, q_high) on FULL 253 rescaled output by inner grid.
    - Apply clip; final te_nb3263 (513,) float32 clipped to [3, 9].

GATE (on per-fold-mean across 15 seeds):
    mean < 0.4423 -> "BETTER"
    else          -> "FAIL"

References (all PRE-unblind anchor chain):
    nb2960 K=18 deep-30 OOF        = 0.4536
    nb3201 learned clip on K18      = 0.4437  (best single-anchor clip)
    nb3070 wide-seed q-cond {K18,K19} = 0.4509
    nb3173 best clip-winner          = 0.4422
    nb3214 SLSQP on 3 clip winners   = ~0.4418
    nb3223 SLSQP {K18,K19} + clip    = 0.4424
    nb3250 SLSQP {K18,K23} + clip    = (sibling, parallel paradigm)
    nb2171 prior post-hoc top        = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy

Outputs:
    data/processed/nb3263_summary.json
    data/processed/nb3263_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3263.npy         (513,) float32 -- deploy te
    submissions/nb3263_K18_quantile_clip.csv  (only on BETTER verdict)
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

TAG = "nb3263"
PARENT_TAG = "nb3070_quantile_conditional"

# -- Inputs --------------------------------------------------------------------
K_LABEL = "K18"
OOF_PATH = DATA_PROCESSED / f"nb2960_{K_LABEL}_30seed_oof.npy"
TE_PATH = DATA_PROCESSED / f"nb2960_{K_LABEL}_30seed_te.npy"
K_DEPTH = "deep30"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Quantile-conditional per-quantile scalar weights -------------------------
# (mapped from nb3070: K18-side weight on each half)
# low-half rows (K18_pred <= q50):  scalar = W_LOW
# high-half rows (K18_pred >  q50): scalar = W_HIGH
W_LOW = 0.8   # nb3070 K18 low-side weight (rows where K18 under-predicts truth)
W_HIGH = 0.5  # nb3070 K18 high-side weight
QUANTILE_CUT = 0.5  # median split

# -- Per-fold learned-clip grid (matches nb3201 / nb3173 / nb3223 family) -----
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Output range clip (matches nb3070 deploy stage) --------------------------
TE_RANGE_LO = 3.0
TE_RANGE_HI = 9.0

# -- Gates (on PER-FOLD-MEAN over 15 seeds) -----------------------------------
GATE_BETTER = 0.4423  # mean < this -> BETTER (user-supplied gate)

# -- References ---------------------------------------------------------------
REF_K18 = 0.4536
REF_NB3070 = 0.4509            # quantile-conditional verify {K18, K19}
REF_NB3201_LEARNED_CLIP_K18 = 0.4437
REF_NB3173_BEST_CLIP_WINNER = 0.4422
REF_NB3214_SLSQP_CLIP_3 = 0.4418
REF_NB3223 = 0.4424            # SLSQP simplex {K18,K19} + clip
REF_NB2171 = 0.4682


def _quantile_rescale(p_k18: np.ndarray, q50: float) -> np.ndarray:
    """nb3070-style per-quantile scalar rescale using K=18 as both partners.

    Rows with p_k18 <= q50 -> p_k18 * W_LOW
    Rows with p_k18 >  q50 -> p_k18 * W_HIGH
    """
    low_mask = p_k18 <= q50
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = W_LOW * p_k18[low_mask]
    out[~low_mask] = W_HIGH * p_k18[~low_mask]
    return out


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
    p_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Quantile-conditional rescale + learned clip at one kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_final = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes: list[float] = []
    fold_train_raes: list[float] = []
    fold_q50s: list[float] = []
    fold_ql: list[float] = []
    fold_qh: list[float] = []
    fold_lo: list[float] = []
    fold_hi: list[float] = []
    fold_clipped_lo: list[int] = []
    fold_clipped_hi: list[int] = []
    fold_high_share: list[float] = []
    fold_rescale_train_raes: list[float] = []

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # --- Step 1: q50 from fold-train K18 predictions --------------------
        q50 = float(np.median(p_unb[tr_loc]))
        fold_q50s.append(q50)

        # --- Step 2: per-quantile scalar rescale ----------------------------
        tr_rescaled = _quantile_rescale(p_unb[tr_loc], q50)
        va_rescaled = _quantile_rescale(p_unb[va_loc], q50)
        fold_rescale_train_raes.append(float(rae(y_unb[tr_loc], tr_rescaled)))

        # --- Step 3: learned clip on fold-train rescaled --------------------
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], tr_rescaled)
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)

        n_lo = int(np.sum(va_rescaled < lo))
        n_hi = int(np.sum(va_rescaled > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)
        fold_high_share.append(float(np.mean(p_unb[va_loc] > q50)))

        # --- Step 4: apply clip to fold-val rescaled ------------------------
        val_pred = np.clip(va_rescaled, lo, hi)
        oof_final[va_loc] = val_pred
        r_tr = float(rae(y_unb[tr_loc], np.clip(tr_rescaled, lo, hi)))
        fold_train_raes.append(r_tr)
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))

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
        "per_fold_rescale_train_rae_mean": float(np.mean(fold_rescale_train_raes)),
        "fold_q50_mean": float(np.mean(fold_q50s)),
        "fold_q50_std": float(np.std(fold_q50s, ddof=1)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
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
        f"{TAG} -- K={K_LABEL} deep-30 ALONE: nb3070-style quantile-"
        f"conditional rescale + per-fold learned clip"
    )
    print(
        f"          per-quantile scalars: W_LOW={W_LOW}  W_HIGH={W_HIGH}  "
        f"cut=q{QUANTILE_CUT:.2f}"
    )
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

    # -- Load anchor ----------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {K_LABEL} deep-30 OOF + te arrays")
    print("-" * 78)
    p_unb = np.load(OOF_PATH).astype(np.float64)
    p_te = np.load(TE_PATH).astype(np.float64)
    if p_unb.shape != (n_unb,):
        raise ValueError(f"{K_LABEL} OOF shape {p_unb.shape} != ({n_unb},)")
    if p_te.shape != (n_test,):
        raise ValueError(f"{K_LABEL} te shape {p_te.shape} != ({n_test},)")
    base_rae = float(rae(y_unb, p_unb))
    leak = float(np.mean(np.isclose(p_unb, y_unb, atol=1e-6)))
    print(
        f"   {K_LABEL} ({K_DEPTH:>6s}): oof_RAE={base_rae:.4f}  "
        f"oof mean={p_unb.mean():.3f} std={p_unb.std():.3f}  "
        f"leak_eq={leak:.2%}  "
        f"te mean={p_te.mean():.3f} std={p_te.std():.3f}"
    )
    if leak > 0.05:
        print(f"   WARN {K_LABEL}: {leak:.1%} rows == truth -- possible leak")
    print(
        f"   y_unb stats: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
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
    oof_stack: list[np.ndarray] = []
    all_fold_ql: list[float] = []
    all_fold_qh: list[float] = []
    all_fold_q50: list[float] = []

    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(p_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        per_fold_stds.append(res["per_fold_val_rae_std"])
        oof_stack.append(res["oof"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        all_fold_q50.append(res["fold_q50_mean"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_train_rae_mean": round(res["per_fold_train_rae_mean"], 4),
            "per_fold_rescale_train_rae_mean": round(
                res["per_fold_rescale_train_rae_mean"], 4
            ),
            "fold_q50_mean": round(res["fold_q50_mean"], 4),
            "fold_q50_std": round(res["fold_q50_std"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
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
            f"q50={res['fold_q50_mean']:.3f}  "
            f"hi_share={res['fold_high_share_mean']:.2f}  "
            f"clip(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr_pooled = np.asarray(pooled_raes, dtype=np.float64)
    arr_pf = np.asarray(per_fold_means, dtype=np.float64)
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

    # Most-picked q values
    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]
    q50_grand_mean = float(np.mean(all_fold_q50))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   POOLED (split-variant, with rescale+clip):")
    print(f"     mean = {pooled_mean:.4f}   std = {pooled_std:.4f}")
    print(f"     min/max = [{arr_pooled.min():.4f}, {arr_pooled.max():.4f}]")
    print(f"\n   PER-FOLD-MEAN (gate metric):")
    print(f"     mean    = {pf_mean:.4f}")
    print(f"     std     = {pf_std:.4f}")
    print(f"     sem     = {pf_sem:.4f}")
    print(f"     95% CI  = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"     median  = {pf_median:.4f}")
    print(f"     min/max = [{arr_pf.min():.4f}, {arr_pf.max():.4f}]")

    print(f"\n   ql_distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh_distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")
    print(f"   grand-mean q50 across seeds = {q50_grand_mean:.4f}")

    print(f"\n   ref {K_LABEL} deep-30 OOF        = {REF_K18:.4f}")
    print(f"   ref nb3201 clip K18 alone   = {REF_NB3201_LEARNED_CLIP_K18:.4f}")
    print(f"   ref nb3070 q-cond {{K18,K19}} = {REF_NB3070:.4f}")
    print(f"   ref nb3173 best clip-winner = {REF_NB3173_BEST_CLIP_WINNER:.4f}")
    print(f"   ref nb3214 SLSQP-clip-3     = {REF_NB3214_SLSQP_CLIP_3:.4f}")
    print(f"   ref nb3223 {{K18,K19}}+clip   = {REF_NB3223:.4f}")
    print(
        f"   delta vs nb3201 (clip K18)  = "
        f"{pf_mean - REF_NB3201_LEARNED_CLIP_K18:+.4f}"
    )
    print(
        f"   delta vs nb3173 best clip    = "
        f"{pf_mean - REF_NB3173_BEST_CLIP_WINNER:+.4f}"
    )

    # -- Deploy: full-253 q50, then learned clip on rescaled full-253 ---------
    print("\n" + "-" * 78)
    print(
        "DEPLOY: q50 from FULL 253 K18 OOF -> rescale te(513) -> "
        "pick clip on FULL 253 rescaled"
    )
    print("-" * 78)
    deploy_q50 = float(np.median(p_unb))
    full_rescaled = _quantile_rescale(p_unb, deploy_q50)
    te_rescaled = _quantile_rescale(p_te, deploy_q50)
    full_rescale_rae = float(rae(y_unb, full_rescaled))
    print(f"   deploy q50 (full 253 K18 OOF median) = {deploy_q50:.4f}")
    print(f"   full-253 rescaled in-sample RAE      = {full_rescale_rae:.4f}")

    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(
        y_unb, full_rescaled,
    )
    print(
        f"   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f}) from FULL 253 y"
    )

    te_pred_pre_range = np.clip(te_rescaled, deploy_lo, deploy_hi)
    te_pred = np.clip(te_pred_pre_range, TE_RANGE_LO, TE_RANGE_HI).astype(np.float32)
    n_te_lo = int(np.sum(te_rescaled < deploy_lo))
    n_te_hi = int(np.sum(te_rescaled > deploy_hi))
    te_low_share = float(np.mean(p_te <= deploy_q50))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"   te low-half share (p_te <= q50) = {te_low_share:.3f}"
    )
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
            f"PROMOTE-CANDIDATE. nb3263 15-seed per-fold-mean {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f} "
            f"({pf_mean - GATE_BETTER:+.4f}). nb3070-style quantile-"
            f"conditional rescale of K=18 alone (W_LOW={W_LOW}, "
            f"W_HIGH={W_HIGH}) + per-fold learned clip beats the existing "
            f"clip-winner ladder. Modal clip = (q{ql_mode:.2f}, "
            f"q{qh_mode:.2f}). Mechanism: per-quantile decompression of "
            f"K=18 tails targets the documented two-sided variance-"
            f"compression failure mode without a second anchor. Re-verify "
            f"with deep-30 before any PRIMARY-1 swap; cycle-160 deep-30 "
            f"rule mandatory for gate-grade decisions."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3263 15-seed per-fold-mean {pf_mean:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"Applying the nb3070 quantile-conditional schedule to K=18 "
            f"alone (W_LOW={W_LOW}, W_HIGH={W_HIGH}) does not beat the "
            f"current clip-winner ceiling. Either (a) the per-quantile "
            f"K18-only scalar rescale does not target the failure mode "
            f"correctly when there is no second-anchor decorrelation, or "
            f"(b) the learned clip absorbs all of the gain that the q-cond "
            f"rescale would have added. Keep nb3173 (0.4422) / nb3201 "
            f"(0.4437) on ladder. Modal clip = (q{ql_mode:.2f}, "
            f"q{qh_mode:.2f})."
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

    sub_csv = SUBMISSIONS / f"{TAG}_K18_quantile_clip.csv"
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
        "method": "K18_deep30_nb3070_style_quantile_conditional_rescale_plus_per_fold_learned_clip",
        "paradigm": "K18_as_both_partners_low_high_w_then_learned_clip",
        "anchor": K_LABEL,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "anchor_oof_path": str(OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_oof_rae": round(base_rae, 4),
        "anchor_leak_eq_truth_frac": round(leak, 4),
        "w_low": W_LOW,
        "w_high": W_HIGH,
        "quantile_cut": QUANTILE_CUT,
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "te_range_lo": TE_RANGE_LO,
        "te_range_hi": TE_RANGE_HI,
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
        # Primary gate metric: per-fold-mean
        "pf_mean": round(pf_mean, 4),
        "pf_std": round(pf_std, 4),
        "pf_sem": round(pf_sem, 4),
        "pf_ci95_low": round(pf_ci_low, 4),
        "pf_ci95_high": round(pf_ci_high, 4),
        "pf_median": round(pf_median, 4),
        "pf_min": round(float(arr_pf.min()), 4),
        "pf_max": round(float(arr_pf.max()), 4),
        # Mean_rae mirror for ladder script compatibility
        "mean_rae": round(pf_mean, 4),
        "std_rae": round(pf_std, 4),
        # Pooled (for reference)
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "pooled_min": round(float(arr_pooled.min()), 4),
        "pooled_max": round(float(arr_pooled.max()), 4),
        # Clip stats
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "q50_grand_mean_across_seeds": round(q50_grand_mean, 4),
        # Deploy
        "deploy_q50": round(deploy_q50, 4),
        "deploy_rescale_in_sample_rae": round(full_rescale_rae, 4),
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_low_share": round(te_low_share, 4),
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
        "ref_nb3070_quantile_cond_K18_K19": REF_NB3070,
        "ref_nb3201_learned_clip_K18": REF_NB3201_LEARNED_CLIP_K18,
        "ref_nb3173_best_clip_winner": REF_NB3173_BEST_CLIP_WINNER,
        "ref_nb3214_slsqp_clip_3": REF_NB3214_SLSQP_CLIP_3,
        "ref_nb3223_pair_clip": REF_NB3223,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18": round(pf_mean - REF_K18, 4),
        "delta_vs_nb3070": round(pf_mean - REF_NB3070, 4),
        "delta_vs_nb3201_clip_K18": round(
            pf_mean - REF_NB3201_LEARNED_CLIP_K18, 4
        ),
        "delta_vs_nb3173_best_clip": round(
            pf_mean - REF_NB3173_BEST_CLIP_WINNER, 4
        ),
        "delta_vs_nb3223_pair_clip": round(pf_mean - REF_NB3223, 4),
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
    print(
        f"   delta vs nb3173 clip   = "
        f"{pf_mean - REF_NB3173_BEST_CLIP_WINNER:+.4f}"
    )
    print(
        f"   delta vs nb3201 K18    = "
        f"{pf_mean - REF_NB3201_LEARNED_CLIP_K18:+.4f}"
    )
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
        "delta_vs_nb3173_best_clip", "delta_vs_nb3201_clip_K18",
        "delta_vs_K18", "delta_vs_nb3070",
        "ql_mode", "qh_mode",
        "deploy_q50", "deploy_ql", "deploy_qh",
        "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
