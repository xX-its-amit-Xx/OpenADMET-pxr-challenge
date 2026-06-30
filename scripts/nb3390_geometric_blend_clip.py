"""nb3390 -- Geometric-mean (log-space) quantile-conditional blend {K18, K19} + clip.

NEW PARADIGM: log-space quantile-conditional blend then clip (vs arithmetic nb3070).

    nb3070 / nb3063 build the q50 hard-split blend in ARITHMETIC space:
        out = w_low*K18 + w_low2*K19   (low half)
        out = w_high*K18 + w_high2*K19 (high half)
    nb3220 then bolts a learned per-fold clip on top of nb3070 (q50).

    nb3390 replaces the arithmetic mean with a GEOMETRIC mean computed in
    log-space, and shifts the cut to q35 with asymmetric fixed weights that
    lean harder on K18 in the inactive (low-pred) half and raise K19 in the
    active (high-pred) tail:

        low  (K18 <= q35): out = exp(0.95*log K18 + 0.05*log K19)
        high (K18 >  q35): out = exp(0.40*log K18 + 0.60*log K19)

    The geometric mean is the multiplicative analogue of the arithmetic
    blend; on pEC50 (already a log-concentration) it pulls the consensus
    toward the smaller of the two anchors and compresses upper-tail
    disagreement differently than the arithmetic mean. A learned per-fold
    clip (same operator as nb3220) is then applied. All K18/K19 OOF/te
    values are strictly positive (min 1.5), so log is well-defined with no
    clamping.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    pred anchors = K18 deep-30 OOF, K19 deep-30 OOF  (253,)
    Per outer fold:
        a) q35 = quantile(K18[fold_train], 0.35)  -- cut threshold
        b) Geometric blend on fold-val:
             low_mask  = K18_val <= q35 -> exp(0.95*logK18 + 0.05*logK19)
             high_mask = K18_val >  q35 -> exp(0.40*logK18 + 0.60*logK19)
        c) Inner learned-clip search on fold-train ONLY (after applying the
           SAME geometric blend to fold-train with the fold-train q35):
             for q_low in {0.01, 0.05, 0.10}:
               for q_high in {0.95, 0.98, 0.99}:
                 lo = quantile(y[fold_train], q_low)
                 hi = quantile(y[fold_train], q_high)
                 minimize fold-train RAE of clip(blend_tr, lo, hi)
           Pick (q_low*, q_high*) -> (lo*, hi*).
        d) val_pred = clip(blend_val, lo*, hi*); stitch into oof; record
           per-fold val RAE.
    Repeat for 15 FRESH kf_seeds {1216..1230}; honest gate = PER-FOLD-MEAN.

    Deploy:
        - q35 = quantile(K18 OOF on FULL 253, 0.35) as the cut proxy.
        - Geometric blend on te (513).
        - Learned clip picked on FULL 253 y by the same inner search.
        - Clip te, save te_nb3390.

GATE (on 15-seed PER-FOLD-MEAN):
    pf_mean < 0.4423 -> "BETTER"
    else             -> "FAIL"

References:
    nb2960 K18 deep-30 OOF        = 0.4536
    nb3000 K19 deep-30 OOF        = 0.4607
    nb3030 wide-seed simplex      = 0.4509
    nb3070 q50 arithmetic blend   = 0.4477 (pooled)
    nb3190 learned-clip on nb3090 = 0.4422 (q35 arith + clip ref)
    nb3220 learned-clip on nb3070 = (q50 arith + clip sibling)
    nb2171 prior post-hoc top     = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3390_summary.json
    data/processed/nb3390_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3390.npy         (513,) float32 -- deploy te
    submissions/nb3390_geometric_blend_clip.csv  (only on BETTER)
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

TAG = "nb3390"
PARENT_TAG = "nb3070"

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
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Geometric (log-space) quantile-conditional weights ------------------------
QUANTILE_CUT = 0.35  # q35 split on K18
# low half  (K18 <= q35): heavy K18
W_LOW_K18, W_LOW_K19 = 0.95, 0.05
# high half (K18 >  q35): raise K19
W_HIGH_K18, W_HIGH_K19 = 0.40, 0.60

# -- Per-fold learned clip grid (per task) -------------------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.95, 0.98, 0.99]

# -- Gate (per task) -----------------------------------------------------------
GATE_BETTER = 0.4423  # per-fold-mean < this -> BETTER

# -- References ----------------------------------------------------------------
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB3030 = 0.4509
REF_NB3070 = 0.4477
REF_NB3190 = 0.4422
REF_NB2171 = 0.4682


def _geom_blend(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q35: float,
) -> np.ndarray:
    """Geometric-mean (log-space) quantile-conditional hard-split blend.

    rows with p_k18 <= q35 -> exp(W_LOW_K18*log K18 + W_LOW_K19*log K19)
    rows with p_k18 >  q35 -> exp(W_HIGH_K18*log K18 + W_HIGH_K19*log K19)

    All anchor values are strictly positive (min ~1.5), so log is safe.
    """
    log18 = np.log(p_k18)
    log19 = np.log(p_k19)
    low_mask = p_k18 <= q35
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = np.exp(
        W_LOW_K18 * log18[low_mask] + W_LOW_K19 * log19[low_mask]
    )
    out[~low_mask] = np.exp(
        W_HIGH_K18 * log18[~low_mask] + W_HIGH_K19 * log19[~low_mask]
    )
    return out


def _pick_best_clip(
    y_tr: np.ndarray,
    pred_tr: np.ndarray,
) -> tuple[float, float, float, float]:
    """Inner grid search: pick (q_low*, q_high*) minimizing fold-train RAE."""
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
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run geometric-blend + learned-clip pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_q35s = []
    fold_high_share = []
    fold_ql = []
    fold_qh = []
    fold_lo = []
    fold_hi = []
    fold_clipped_lo = []
    fold_clipped_hi = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # q35 cut from fold-train K18
        q35 = float(np.quantile(P_unb[tr_loc, 0], QUANTILE_CUT))
        fold_q35s.append(q35)

        # Geometric blend on fold-train (with fold-train q35) for clip fitting
        blend_tr = _geom_blend(P_unb[tr_loc, 0], P_unb[tr_loc, 1], q35)
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], blend_tr)
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)

        # Geometric blend on fold-val (with fold-train q35), then clip
        val_p_k18 = P_unb[va_loc, 0]
        val_p_k19 = P_unb[va_loc, 1]
        blend_val = _geom_blend(val_p_k18, val_p_k19, q35)
        fold_high_share.append(float(np.mean(val_p_k18 > q35)))
        n_lo = int(np.sum(blend_val < lo))
        n_hi = int(np.sum(blend_val > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)
        clipped = np.clip(blend_val, lo, hi)
        oof_clip[va_loc] = clipped
        fold_val_raes.append(float(rae(y_unb[va_loc], clipped)))

    if np.isnan(oof_clip).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_clip))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_q35_mean": float(np.mean(fold_q35s)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "n_clipped_lo": int(np.sum(fold_clipped_lo)),
        "n_clipped_hi": int(np.sum(fold_clipped_hi)),
        "oof": oof_clip,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- GEOMETRIC-mean (log-space) quantile-conditional blend "
        f"{K_LABELS} deep-30 + learned clip"
    )
    print(
        f"          cut = K18 q{int(QUANTILE_CUT * 100)}  "
        f"low (K18<=q35): exp({W_LOW_K18}*logK18 + {W_LOW_K19}*logK19)"
    )
    print(
        f"                           "
        f"high(K18 >q35): exp({W_HIGH_K18}*logK18 + {W_HIGH_K19}*logK19)"
    )
    print(f"          Q_LOW_GRID  = {Q_LOW_GRID}")
    print(f"          Q_HIGH_GRID = {Q_HIGH_GRID}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"          honest gate metric = PER-FOLD-MEAN")
    print(f"          gate: pf_mean < {GATE_BETTER:.4f} -> BETTER, else FAIL")
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
        # log safety: strictly positive
        if oof.min() <= 0 or te_arr.min() <= 0:
            raise ValueError(
                f"{k} has non-positive values (oof_min={oof.min():.3f}, "
                f"te_min={te_arr.min():.3f}); geometric blend undefined"
            )
        oof_cols.append(oof)
        te_cols.append(te_arr)
        r = float(rae(y_unb, oof))
        per_K_full_rae[k] = round(r, 4)
        print(
            f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
            f"oof[min,max]=[{oof.min():.3f},{oof.max():.3f}]  "
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

    # -- Geometric vs arithmetic sanity on full pool -------------------------
    full_q35 = float(np.quantile(P_unb[:, 0], QUANTILE_CUT))
    geom_full = _geom_blend(P_unb[:, 0], P_unb[:, 1], full_q35)
    # arithmetic analogue with same weights/cut for reference
    low_full = P_unb[:, 0] <= full_q35
    arith_full = np.empty(n_unb)
    arith_full[low_full] = (
        W_LOW_K18 * P_unb[low_full, 0] + W_LOW_K19 * P_unb[low_full, 1]
    )
    arith_full[~low_full] = (
        W_HIGH_K18 * P_unb[~low_full, 0] + W_HIGH_K19 * P_unb[~low_full, 1]
    )
    geom_full_rae = float(rae(y_unb, geom_full))
    arith_full_rae = float(rae(y_unb, arith_full))
    print(
        f"\n   full-pool K18 q35 = {full_q35:.4f}  "
        f"(low={low_full.sum()}, high={(~low_full).sum()})"
    )
    print(
        f"   full-pool geom-blend RAE  = {geom_full_rae:.4f}  "
        f"(arith same-weights = {arith_full_rae:.4f}, "
        f"delta={geom_full_rae - arith_full_rae:+.4f})"
    )
    print(
        f"   geom-arith max abs diff   = "
        f"{np.max(np.abs(geom_full - arith_full)):.4f}"
    )

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
    per_fold_means = []
    oof_stack = []
    all_fold_ql = []
    all_fold_qh = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_q35_mean": round(res["fold_q35_mean"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
            "fold_ql": [round(v, 3) for v in res["fold_ql"]],
            "fold_qh": [round(v, 3) for v in res["fold_qh"]],
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"q35={res['fold_q35_mean']:.3f}  "
            f"high_share={res['fold_high_share_mean']:.2f}  "
            f"clipped(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    arr_pf = np.asarray(per_fold_means, dtype=np.float64)
    pf_mean = float(arr_pf.mean())
    pf_std = float(arr_pf.std(ddof=1)) if n_s > 1 else 0.0
    pf_sem = pf_std / np.sqrt(n_s) if n_s > 1 else 0.0
    pf_ci_low = pf_mean - t_mult * pf_sem
    pf_ci_high = pf_mean + t_mult * pf_sem
    pf_median = float(np.median(arr_pf))

    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   POOLED RAE:")
    print(f"     mean   = {mean_rae:.4f}")
    print(f"     std    = {std_rae:.4f}")
    print(f"     95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"     median = {median_rae:.4f}")
    print(f"     min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   PER-FOLD-MEAN RAE (HONEST GATE METRIC):")
    print(f"     mean   = {pf_mean:.4f}")
    print(f"     std    = {pf_std:.4f}")
    print(f"     sem    = {pf_sem:.4f}")
    print(f"     95% CI = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"     median = {pf_median:.4f}")
    print(f"     min/max = [{arr_pf.min():.4f}, {arr_pf.max():.4f}]")
    print(f"\n   ref nb3070 (q50 arith pooled) = {REF_NB3070:.4f}")
    print(f"   delta vs nb3070 (pf_mean)     = {pf_mean - REF_NB3070:+.4f}")
    print(f"   ref nb3190 (q35 arith + clip) = {REF_NB3190:.4f}")
    print(f"   delta vs nb3190 (pf_mean)     = {pf_mean - REF_NB3190:+.4f}")
    print(f"   ref nb2171 prior PRIMARY-1    = {REF_NB2171:.4f}")
    print(f"   gain vs nb2171 (pf_mean)      = {REF_NB2171 - pf_mean:+.4f}")
    print(f"\n   ql_distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh_distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")

    # -- Deploy: q35 from FULL 253 K18 OOF; geom blend te; learned clip on 253 -
    deploy_q35 = float(np.quantile(P_unb[:, 0], QUANTILE_CUT))
    blend_full_oof = _geom_blend(P_unb[:, 0], P_unb[:, 1], deploy_q35)
    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(
        y_unb, blend_full_oof
    )
    te_k18 = P_te[:, 0]
    te_k19 = P_te[:, 1]
    blend_te = _geom_blend(te_k18, te_k19, deploy_q35)
    te_pred = np.clip(blend_te, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(blend_te < deploy_lo))
    n_te_hi = int(np.sum(blend_te > deploy_hi))
    te_low_share = float(np.mean(te_k18 <= deploy_q35))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy q35 (full K18 OOF) = {deploy_q35:.4f}  "
        f"te low-half share = {te_low_share:.3f}"
    )
    print(
        f"   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f})"
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

    # Median-seed OOF for storage (median over per-fold-mean -- honest metric)
    med_seed_idx = int(np.argsort(arr_pf)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"   median seed = {median_seed} "
        f"(pf_mean={arr_pf[med_seed_idx]:.4f}, pooled={arr[med_seed_idx]:.4f})"
    )

    # -- Gate (on PER-FOLD-MEAN) ---------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (honest metric = PER-FOLD-MEAN)")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3390 15-seed PER-FOLD-MEAN {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"Geometric (log-space) q35 quantile-conditional blend + learned "
            f"clip beats the arithmetic-blend siblings nb3070 ({REF_NB3070:.4f}) "
            f"and matches/beats the q35-arith clip ref nb3190 ({REF_NB3190:.4f}) "
            f"by {REF_NB3190 - pf_mean:+.4f}. Multiplicative consensus on pEC50 "
            f"compresses upper-tail anchor disagreement differently than the "
            f"arithmetic mean. Modal clip pick (q{ql_mode}, q{qh_mode}). "
            f"Re-verify with deep-30 before any PRIMARY swap. "
            f"anchor_pre_unblind=True (K18/K19 deep-30 OOFs are PRE-clean)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3390 15-seed PER-FOLD-MEAN {pf_mean:.4f} fails BETTER "
            f"gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"Delta vs nb3070 (q50 arith) = {pf_mean - REF_NB3070:+.4f}, "
            f"delta vs nb3190 (q35 arith+clip) = {pf_mean - REF_NB3190:+.4f}. "
            f"Geometric mean in log-space does not beat the arithmetic "
            f"quantile-conditional blend on this anchor pair: on pEC50 (already "
            f"log-concentration) the multiplicative consensus pulls toward the "
            f"smaller anchor and the fixed 0.95/0.05 + 0.40/0.60 asymmetry plus "
            f"q35 cut does not align with the conditional bias structure better "
            f"than the verified arithmetic 0.8/0.2 + 0.5/0.5 q50 split. Closes "
            f"the geometric-blend axis; keep nb3070 / nb3190 arithmetic ladder."
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

    sub_csv = SUBMISSIONS / f"{TAG}_geometric_blend_clip.csv"
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
            "geometric_mean_logspace_quantile_conditional_q35_hard_split_blend_"
            "K18_K19_deep30_plus_learned_clip"
        ),
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "blend_space": "geometric_log",
        "quantile_cut": QUANTILE_CUT,
        "w_low": {"K18": W_LOW_K18, "K19": W_LOW_K19},
        "w_high": {"K18": W_HIGH_K18, "K19": W_HIGH_K19},
        "full_q35": round(full_q35, 4),
        "full_pool_geom_rae": round(geom_full_rae, 4),
        "full_pool_arith_same_weights_rae": round(arith_full_rae, 4),
        "full_pool_geom_minus_arith": round(geom_full_rae - arith_full_rae, 4),
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
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "per_fold_mean_rae_mean": round(pf_mean, 4),
        "per_fold_mean_rae_std": round(pf_std, 4),
        "per_fold_mean_rae_sem": round(pf_sem, 4),
        "per_fold_mean_rae_ci95_low": round(pf_ci_low, 4),
        "per_fold_mean_rae_ci95_high": round(pf_ci_high, 4),
        "per_fold_mean_rae_median": round(pf_median, 4),
        "per_fold_mean_rae_min": round(float(arr_pf.min()), 4),
        "per_fold_mean_rae_max": round(float(arr_pf.max()), 4),
        "honest_metric": "per_fold_mean",
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb3030": REF_NB3030,
        "ref_nb3070": REF_NB3070,
        "ref_nb3190": REF_NB3190,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3070_pf_mean": round(pf_mean - REF_NB3070, 4),
        "delta_vs_nb3190_pf_mean": round(pf_mean - REF_NB3190, 4),
        "gain_vs_nb2171_pf_mean": round(REF_NB2171 - pf_mean, 4),
        "deploy_q35": round(deploy_q35, 4),
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "te_low_share": round(te_low_share, 4),
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
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "gate_better": GATE_BETTER,
        "gate_metric": "per_fold_mean",
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
    print(f"   pf_mean ({n_s} seeds) = {pf_mean:.4f} +/- {pf_std:.4f}")
    print(f"   pf_mean 95% CI       = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   pooled mean          = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   delta vs nb3070 (pf) = {pf_mean - REF_NB3070:+.4f}")
    print(f"   delta vs nb3190 (pf) = {pf_mean - REF_NB3190:+.4f}")
    print(f"   gain vs nb2171  (pf) = {REF_NB2171 - pf_mean:+.4f}")
    print(f"   verdict              = {verdict}")
    print(f"   wall                 = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean_rae_mean", "per_fold_mean_rae_std",
        "per_fold_mean_rae_ci95_low", "per_fold_mean_rae_ci95_high",
        "mean_rae", "std_rae",
        "delta_vs_nb3070_pf_mean", "delta_vs_nb3190_pf_mean",
        "full_pool_geom_minus_arith",
        "ql_mode", "qh_mode",
        "deploy_q35", "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
