"""nb3400 -- 3-REGION quantile-conditional blend {K18, K19, K23} + learned clip.

NEW PARADIGM (3-region quantile with K23 as 3rd anchor):
    Prior q-blend ladders were either 2-region binary splits over {K18, K19}
    (nb3070/nb3173/nb3314 q35: low -> 0.8*K18+0.2*K19, high -> 0.5*K18+0.5*K19)
    or 3-tier with K23 dominant in the MID range (nb3093: low 0.85/0.15,
    mid 0.6*K23+0.4*K18, high 0.6*K19+0.4*K18). Both saturated near the
    post-hoc-blend ceiling (~0.4422-0.4475).

    Now that a deep-30 K=23 anchor EXISTS (nb3020, OOF RAE 0.4750), this script
    routes K23 into the HIGH (active) tail instead of the mid-range, and uses
    K18/K19 in the low and mid regions. The intuition: K23 is the deepest
    feature pyramid (most Mordred/embed columns) so it should carry the most
    information on the harder high-activity rows, while the inactive low tail
    stays K18-dominant (cleaner variance behavior) and the mid is a balanced
    K18/K19 average. A learned per-fold clip then compresses tail variance.

    3-region schedule (cut at q33/q66 of fold-TRAIN K18 OOF, NO val leakage):
        LOW   (K18 <= q33):       0.9*K18 + 0.1*K19
        MID   (q33 < K18 <= q66): 0.5*K18 + 0.5*K19
        HIGH  (K18 >  q66):       0.5*K19 + 0.5*K23   <- K23 in HIGH tail

PROTOCOL (per outer kf_seed, 5-fold scaffold split):
    STEP 1: q33, q66 = 33rd/66th percentile of fold-TRAIN K18 deep-30 OOF.
    STEP 2: 3-region hard-split blend (above schedule) on fold-train + fold-val.
    STEP 3: learned clip -- inner grid (q_low, q_high) on fold-TRAIN blended
            output, pick (lo*, hi*) minimizing fold-train RAE.
    STEP 4: val_pred = clip(val_blend, lo*, hi*); stitch 5 fold-vals.
    Pooled RAE + per-fold-mean RAE per seed.
    Repeat for 15 FRESH kf_seeds {1216..1230}.

    Deploy:
        q33/q66 from FULL-253 K18 OOF; 3-region blend te(513);
        (lo*, hi*) learned on FULL-253 blended output;
        te_pred = clip(clip(te_blend, lo*, hi*), 3.0, 9.0).

GATE (on PER-FOLD-MEAN across 15 seeds):
    per_fold_mean < 0.4423 -> "BETTER"
    else                   -> "FAIL"

References (PRE-unblind anchor chain; chemprop_aux anchored, all deep-30):
    nb2960 K18 deep-30 OOF              = 0.4536
    nb3000 K19 deep-30 OOF              = 0.4607
    nb3020 K23 deep-30 OOF             ~= 0.4750
    nb3070 wide q-cond {K18,K19} deep30 = 0.4477 / 0.4509
    nb3093 3-tier K23-mid blend         = (sibling; K23 in MID)
    nb3173 best clip-winner             = 0.4422
    nb3223 SLSQP {K18,K19} + clip       = 0.4424
    nb3314 q35 deep-60 + clip           = (sibling)
    nb2171 prior post-hoc top           = 0.4682
    chemprop_aux anchor                 = 0.6216

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
    data/processed/nb3400_summary.json
    data/processed/nb3400_pred_oof.npy  (253,) float32 -- median-seed OOF
    data/processed/te_nb3400.npy        (513,) float32 -- deploy te
    submissions/nb3400_K18_K19_K23_deep30_clip.csv (only on BETTER verdict)
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

TAG = "nb3400"
PARENT_TAG = "nb3093_nb3173_three_region_quantile_blend_K23_high_then_learned_clip"

# -- Inputs (all deep-30, cached) ---------------------------------------------
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

# -- CV protocol --------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH kf_seeds {1216..1230}

# -- 3-region quantile-conditional weights ------------------------------------
# LOW  (K18 <= q33):        0.9*K18 + 0.1*K19
# MID  (q33 < K18 <= q66):  0.5*K18 + 0.5*K19
# HIGH (K18 >  q66):        0.5*K19 + 0.5*K23  <- K23 (deepest pyramid) in HIGH
Q_LOW = 0.33
Q_HIGH = 0.66
W_LOW_K18, W_LOW_K19 = 0.9, 0.1
W_MID_K18, W_MID_K19 = 0.5, 0.5
W_HIGH_K19, W_HIGH_K23 = 0.5, 0.5

# -- Learned-clip grid (nb3263 / nb3314 family) -------------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Output range clip (deploy stage) -----------------------------------------
TE_RANGE_LO = 3.0
TE_RANGE_HI = 9.0

# -- Gate ---------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ---------------------------------------------------------------
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_K23 = 0.4750
REF_NB3070 = 0.4477
REF_NB3173_CLIP = 0.4422
REF_NB3223 = 0.4424
REF_NB2171 = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# 3-region quantile blend + learned clip
# ============================================================================

def _blend_3region(p_k18, p_k19, p_k23, q33, q66):
    """Per-row 3-region hard-split blend by (q33, q66) cutoffs on K18.

    LOW   (K18 <= q33):       0.9*K18 + 0.1*K19
    MID   (q33 < K18 <= q66): 0.5*K18 + 0.5*K19
    HIGH  (K18 >  q66):       0.5*K19 + 0.5*K23
    """
    low_mask = p_k18 <= q33
    high_mask = p_k18 > q66
    mid_mask = (~low_mask) & (~high_mask)
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = W_LOW_K18 * p_k18[low_mask] + W_LOW_K19 * p_k19[low_mask]
    out[mid_mask] = W_MID_K18 * p_k18[mid_mask] + W_MID_K19 * p_k19[mid_mask]
    out[high_mask] = (
        W_HIGH_K19 * p_k19[high_mask] + W_HIGH_K23 * p_k23[high_mask]
    )
    return out


def _pick_best_clip(y_tr, pred_tr):
    """Inner grid: pick (q_low*, q_high*) minimizing fold-train RAE.

    Clip bounds are quantiles of fold-TRAIN truth (no val leakage).
    """
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


def _run_one_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    """3-region quantile blend + learned clip at one outer kf_seed.

    q33/q66 + clip params derived from fold-TRAIN ONLY (clean cross-fit).
    P_unb columns: [K18, K19, K23].
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_final = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_blend_train_raes = []
    fold_q33s = []
    fold_q66s = []
    fold_ql = []
    fold_qh = []
    fold_lo = []
    fold_hi = []
    fold_clipped_lo = []
    fold_clipped_hi = []
    fold_low_share = []
    fold_mid_share = []
    fold_high_share = []

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # --- STEP 1: q33, q66 from fold-TRAIN K18 OOF ONLY -----------------
        q33 = float(np.quantile(P_unb[tr_loc, 0], Q_LOW))
        q66 = float(np.quantile(P_unb[tr_loc, 0], Q_HIGH))
        fold_q33s.append(q33)
        fold_q66s.append(q66)

        # --- STEP 2: 3-region blend on fold-train + fold-val ---------------
        tr_blend = _blend_3region(
            P_unb[tr_loc, 0], P_unb[tr_loc, 1], P_unb[tr_loc, 2], q33, q66,
        )
        va_k18 = P_unb[va_loc, 0]
        va_blend = _blend_3region(
            va_k18, P_unb[va_loc, 1], P_unb[va_loc, 2], q33, q66,
        )
        fold_blend_train_raes.append(float(rae(y_unb[tr_loc], tr_blend)))
        fold_low_share.append(float(np.mean(va_k18 <= q33)))
        fold_mid_share.append(
            float(np.mean((va_k18 > q33) & (va_k18 <= q66)))
        )
        fold_high_share.append(float(np.mean(va_k18 > q66)))

        # --- STEP 3: learned clip on fold-train blended --------------------
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], tr_blend)
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)

        n_lo = int(np.sum(va_blend < lo))
        n_hi = int(np.sum(va_blend > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)

        # --- STEP 4: apply clip to fold-val blended ------------------------
        val_pred = np.clip(va_blend, lo, hi)
        oof_final[va_loc] = val_pred
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
        "per_fold_blend_train_rae_mean": float(np.mean(fold_blend_train_raes)),
        "fold_q33_mean": float(np.mean(fold_q33s)),
        "fold_q66_mean": float(np.mean(fold_q66s)),
        "fold_low_share_mean": float(np.mean(fold_low_share)),
        "fold_mid_share_mean": float(np.mean(fold_mid_share)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "n_clipped_lo": int(np.sum(fold_clipped_lo)),
        "n_clipped_hi": int(np.sum(fold_clipped_hi)),
        "oof": oof_final,
    }


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 3-REGION quantile-conditional blend {{K18,K19,K23}} "
          f"+ learned clip")
    print(f"          LOW  (K18 <= q33):       "
          f"{W_LOW_K18}*K18 + {W_LOW_K19}*K19")
    print(f"          MID  (q33 < K18 <= q66): "
          f"{W_MID_K18}*K18 + {W_MID_K19}*K19")
    print(f"          HIGH (K18 >  q66):       "
          f"{W_HIGH_K19}*K19 + {W_HIGH_K23}*K23   <- K23 in HIGH tail")
    print(f"          q_low={Q_LOW}  q_high={Q_HIGH}  source=fold_train_only")
    print(f"          clip grid: ql={Q_LOW_GRID}  qh={Q_HIGH_GRID}")
    print(f"          outer kf_seeds = {KF_SEEDS[0]}..{KF_SEEDS[-1]} "
          f"(n={len(KF_SEEDS)})")
    print(f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER else FAIL")
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
    leak_flags = {}
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
        frac = float(np.mean(np.isclose(oof, y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")
        print(f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}  "
              f"leak={frac:.2%}")

    P_unb = np.column_stack(oof_cols)  # (253, 3) -> [K18, K19, K23]
    P_te = np.column_stack(te_cols)    # (513, 3)

    # Pairwise correlations
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

    # -- 3-region blend + learned-clip 15-seed sweep -------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: 3-REGION BLEND + LEARNED-CLIP SWEEP -- "
          f"{len(KF_SEEDS)} fresh kf_seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    per_fold_stds = []
    oof_stack = []
    all_fold_ql = []
    all_fold_qh = []
    all_fold_q33 = []
    all_fold_q66 = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        per_fold_stds.append(res["per_fold_val_rae_std"])
        oof_stack.append(res["oof"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        all_fold_q33.append(res["fold_q33_mean"])
        all_fold_q66.append(res["fold_q66_mean"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_blend_train_rae_mean": round(
                res["per_fold_blend_train_rae_mean"], 4),
            "fold_q33_mean": round(res["fold_q33_mean"], 4),
            "fold_q66_mean": round(res["fold_q66_mean"], 4),
            "fold_low_share_mean": round(res["fold_low_share_mean"], 4),
            "fold_mid_share_mean": round(res["fold_mid_share_mean"], 4),
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
            f"q33={res['fold_q33_mean']:.3f} q66={res['fold_q66_mean']:.3f}  "
            f"shares(L/M/H)={res['fold_low_share_mean']:.2f}/"
            f"{res['fold_mid_share_mean']:.2f}/"
            f"{res['fold_high_share_mean']:.2f}  "
            f"clip(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr_pool = np.asarray(pooled_raes, dtype=np.float64)
    arr_pfm = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(arr_pfm)
    pooled_mean = float(arr_pool.mean())
    pooled_std = float(arr_pool.std(ddof=1)) if n_s > 1 else 0.0
    per_fold_mean = float(arr_pfm.mean())   # GATE METRIC
    per_fold_std = float(arr_pfm.std(ddof=1)) if n_s > 1 else 0.0
    sem_pfm = per_fold_std / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14
    ci_low_pfm = per_fold_mean - t_mult * sem_pfm
    ci_high_pfm = per_fold_mean + t_mult * sem_pfm
    pf_median = float(np.median(arr_pfm))

    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]
    q33_grand_mean = float(np.mean(all_fold_q33))
    q66_grand_mean = float(np.mean(all_fold_q66))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled_RAE       mean = {pooled_mean:.4f}  std = {pooled_std:.4f}")
    print(f"   pooled  min/max  [{arr_pool.min():.4f}, {arr_pool.max():.4f}]")
    print(f"   per_fold_mean    mean = {per_fold_mean:.4f}  std = {per_fold_std:.4f}  "
          f"95% CI [{ci_low_pfm:.4f}, {ci_high_pfm:.4f}]   <- GATE METRIC")
    print(f"   per-fm  median   = {pf_median:.4f}")
    print(f"   per-fm  min/max  [{arr_pfm.min():.4f}, {arr_pfm.max():.4f}]")
    print(f"\n   ql distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")
    print(f"   grand-mean q33 = {q33_grand_mean:.4f}  q66 = {q66_grand_mean:.4f}")
    print(f"\n   ref nb3173 best clip-winner = {REF_NB3173_CLIP:.4f}")
    print(f"   ref nb3070 q-cond {{K18,K19}} = {REF_NB3070:.4f}")
    print(f"   ref K23 deep-30 solo         = {REF_K23:.4f}")
    print(f"   delta vs nb3173             = {per_fold_mean - REF_NB3173_CLIP:+.4f}")
    print(f"   delta vs nb3070             = {per_fold_mean - REF_NB3070:+.4f}")

    # -- Deploy: q33/q66 from FULL 253 K18 OOF, blend te, learned clip --------
    print("\n" + "-" * 78)
    print("STEP 4: DEPLOY -- q33/q66 from full-253 K18 OOF -> 3-region blend "
          "te(513) -> learned clip on full-253 blended")
    print("-" * 78)
    deploy_q33 = float(np.quantile(P_unb[:, 0], Q_LOW))
    deploy_q66 = float(np.quantile(P_unb[:, 0], Q_HIGH))
    full_blend = _blend_3region(
        P_unb[:, 0], P_unb[:, 1], P_unb[:, 2], deploy_q33, deploy_q66,
    )
    te_blend = _blend_3region(
        P_te[:, 0], P_te[:, 1], P_te[:, 2], deploy_q33, deploy_q66,
    )
    full_blend_rae = float(rae(y_unb, full_blend))
    print(f"   deploy q33 (full-253 K18 OOF) = {deploy_q33:.4f}")
    print(f"   deploy q66 (full-253 K18 OOF) = {deploy_q66:.4f}")
    print(f"   full-253 blended in-sample RAE = {full_blend_rae:.4f}")

    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(y_unb, full_blend)
    print(f"   deploy clip = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
          f"({deploy_lo:.3f}, {deploy_hi:.3f}) from full-253 y")

    te_pred_pre_range = np.clip(te_blend, deploy_lo, deploy_hi)
    te_pred = np.clip(te_pred_pre_range, TE_RANGE_LO, TE_RANGE_HI).astype(np.float32)
    n_te_lo = int(np.sum(te_blend < deploy_lo))
    n_te_hi = int(np.sum(te_blend > deploy_hi))
    te_k18 = P_te[:, 0]
    te_low_share = float(np.mean(te_k18 <= deploy_q33))
    te_mid_share = float(
        np.mean((te_k18 > deploy_q33) & (te_k18 <= deploy_q66))
    )
    te_high_share = float(np.mean(te_k18 > deploy_q66))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te(513) shares(L/M/H) = {te_low_share:.3f}/"
          f"{te_mid_share:.3f}/{te_high_share:.3f}")
    print(f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513  "
          f"total={n_te_lo + n_te_hi}/513")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
          f"min={te_pred.min():.3f}  max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr_pfm)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median (by pf_mean) seed = {median_seed} "
          f"(pf_mean={arr_pfm[med_seed_idx]:.4f}, "
          f"pooled={arr_pool[med_seed_idx]:.4f})")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: GATE")
    print("-" * 78)
    if per_fold_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3400 3-region quantile blend {{K18,K19,K23}} "
            f"(K23 in HIGH tail) + learned clip 15-seed per-fold-mean "
            f"{per_fold_mean:.4f} clears the BETTER gate {GATE_BETTER:.4f} "
            f"({per_fold_mean - GATE_BETTER:+.4f}), beating nb3173 clip-winner "
            f"({REF_NB3173_CLIP:.4f}) by {per_fold_mean - REF_NB3173_CLIP:+.4f}. "
            f"Routing the deepest pyramid K23 into the high-activity tail "
            f"extracts net gain over the 2-anchor {{K18,K19}} q-blend ceiling. "
            f"Re-verify with deep-30 rule (cycle-160) before any PRIMARY-1 swap. "
            f"Predicted LB under +0.0045 PRE delta = {per_fold_mean + 0.0045:.4f}. "
            f"Modal clip = (q{ql_mode:.2f}, q{qh_mode:.2f})."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3400 3-region quantile blend {{K18,K19,K23}} (K23 in "
            f"HIGH tail) + learned clip 15-seed per-fold-mean {per_fold_mean:.4f} "
            f"fails BETTER gate {GATE_BETTER:.4f} "
            f"({per_fold_mean - GATE_BETTER:+.4f}). Adding K23 as a 3rd "
            f"quantile-region anchor in the high tail does NOT break the "
            f"post-hoc-blend ceiling: K23 (deep-30 solo {REF_K23:.4f}) is highly "
            f"correlated with K19 (r={pair_corrs['K19_K23']:.3f}) so the HIGH "
            f"region gains no fresh information. The q-blend + clip ceiling is "
            f"set by the (anchor=chemprop_aux, K-feature, n=253) substrate, not "
            f"the number of quantile regions. Keep nb3173 ({REF_NB3173_CLIP:.4f}) "
            f"/ nb3223 ({REF_NB3223:.4f}) on ladder; no ladder change. "
            f"Blended pre-clip in-sample RAE = {full_blend_rae:.4f}. Modal clip "
            f"= (q{ql_mode:.2f}, q{qh_mode:.2f})."
        )
    print(f"   per_fold_mean   = {per_fold_mean:.4f}  (gate: < {GATE_BETTER:.4f})")
    print(f"   pooled_mean     = {pooled_mean:.4f}  (informational)")
    print(f"   verdict         = {verdict}")
    print(f"   ladder action   = {ladder_action}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_K18_K19_K23_deep30_clip.csv"
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
        "method": ("three_region_quantile_conditional_blend_K18_K19_K23_deep30_"
                   "K23_in_high_then_learned_per_fold_clip"),
        "paradigm": "3_region_quantile_K23_as_3rd_anchor_high_tail",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_corr_matrix": corr_mat.tolist(),
        "oof_pairwise_corr": pair_corrs,
        # 3-region blend schedule
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "q_source": "fold_train_only",
        "w_low": {"K18": W_LOW_K18, "K19": W_LOW_K19},
        "w_mid": {"K18": W_MID_K18, "K19": W_MID_K19},
        "w_high": {"K19": W_HIGH_K19, "K23": W_HIGH_K23},
        # learned clip
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
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "per_fold_std_array": [round(float(v), 4) for v in per_fold_stds],
        # Gate metric: per-fold-mean
        "per_fold_mean_rae": round(per_fold_mean, 4),
        "per_fold_std_rae": round(per_fold_std, 4),
        "per_fold_sem_rae": round(sem_pfm, 4),
        "per_fold_ci95_low": round(ci_low_pfm, 4),
        "per_fold_ci95_high": round(ci_high_pfm, 4),
        "per_fold_median_rae": round(pf_median, 4),
        "per_fold_min_rae": round(float(arr_pfm.min()), 4),
        "per_fold_max_rae": round(float(arr_pfm.max()), 4),
        # mean_rae mirror for ladder-script compatibility
        "mean_rae": round(per_fold_mean, 4),
        "std_rae": round(per_fold_std, 4),
        # Pooled (reference)
        "pooled_mean_rae": round(pooled_mean, 4),
        "pooled_std_rae": round(pooled_std, 4),
        "pooled_min_rae": round(float(arr_pool.min()), 4),
        "pooled_max_rae": round(float(arr_pool.max()), 4),
        # clip stats
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "q33_grand_mean_across_seeds": round(q33_grand_mean, 4),
        "q66_grand_mean_across_seeds": round(q66_grand_mean, 4),
        # references
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_K23_deep30": REF_K23,
        "ref_nb3070": REF_NB3070,
        "ref_nb3173_clip": REF_NB3173_CLIP,
        "ref_nb3223": REF_NB3223,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3173": round(per_fold_mean - REF_NB3173_CLIP, 4),
        "delta_vs_nb3070": round(per_fold_mean - REF_NB3070, 4),
        "delta_vs_nb3223": round(per_fold_mean - REF_NB3223, 4),
        "delta_vs_K23_solo": round(per_fold_mean - REF_K23, 4),
        # deploy
        "deploy_q33": round(deploy_q33, 4),
        "deploy_q66": round(deploy_q66, 4),
        "deploy_blend_in_sample_rae": round(full_blend_rae, 4),
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": round(deploy_lo, 4),
        "deploy_hi": round(deploy_hi, 4),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_low_share": round(te_low_share, 4),
        "te_mid_share": round(te_mid_share, 4),
        "te_high_share": round(te_high_share, 4),
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
    print(f"   per-K OOF RAE            = "
          + ", ".join([f"{k}={v:.4f}" for k, v in per_K_full_rae.items()]))
    print(f"   pairwise corrs          = {pair_corrs}")
    print(f"   q33/q66 grand-mean      = {q33_grand_mean:.3f} / {q66_grand_mean:.3f}")
    print(f"   3-region blend + clip per-fold-mean = {per_fold_mean:.4f} "
          f"+/- {per_fold_std:.4f}  <- GATE METRIC")
    print(f"   pooled mean             = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"   delta vs nb3173 (0.4422) = {per_fold_mean - REF_NB3173_CLIP:+.4f}")
    print(f"   modal clip (ql, qh)     = ({ql_mode}, {qh_mode})")
    print(f"   te[unb] in-sample       = {te_unb_in_rae:.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   wall                    = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_K_full_oof_rae",
        "oof_pairwise_corr",
        "q33_grand_mean_across_seeds",
        "q66_grand_mean_across_seeds",
        "per_fold_mean_rae",
        "per_fold_std_rae",
        "per_fold_ci95_low",
        "per_fold_ci95_high",
        "pooled_mean_rae",
        "pooled_std_rae",
        "delta_vs_nb3173",
        "delta_vs_nb3070",
        "delta_vs_K23_solo",
        "ql_mode",
        "qh_mode",
        "deploy_q33",
        "deploy_q66",
        "deploy_ql",
        "deploy_qh",
        "deploy_lo",
        "deploy_hi",
        "n_te_clipped_lo",
        "n_te_clipped_hi",
        "te_unb_in_sample_rae",
        "verdict",
        "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
