"""nb3312 -- Quantile-blend {K18, K20} q35 + learned clip (alt to {K18, K19}).

NEW PARADIGM: try the K20 partner anchor in the quantile-conditional + learned-
    clip pipeline that nb3190 ran on the {K18, K19} q35 blend.

    nb3090 found the q35 quantile-conditional blend on {K18, K19} (q_cut=0.35,
    w_K18_low=0.95, w_K18_high=0.40) -> 15-seed mean 0.4472. nb3190 then applied
    a learned per-fold clip operator on top of that blend -> 0.4426 (MARGINAL),
    i.e. clip compounded ~-0.0046 RAE on the {K18, K19} anchor. Separately,
    nb3091 swapped K20 for K19 in the q50 hard-split blend and FAILED (0.4489 vs
    0.4472): K20 standalone deep-30 (0.4625) is a weaker partner than K19
    (0.4607). Open question this script answers: can the learned-clip operator
    RESCUE the weaker K20 partner inside the SAME q35 + clip pipeline, landing at
    or below the nb3173 clip-operator ceiling (0.4422)?

    The K18/K20/K23 name reflects the clip-on-K-pair family this entry joins
    (cf. nb3250 K18+K23, nb3224 K18+K19+K23). This entry is the {K18, K20} q35
    quantile-blend + learned-clip member.

PIPELINE (per kf_seed, 5-fold scaffold split, anchors LOADED no rebuild):
    Per outer fold:
      1. q35 threshold q = quantile(K18[fold_train], 0.35)  (fold-train only).
      2. Build quantile-conditional {K18, K20} blend on fold-train AND fold-val:
           K18 <= q -> w_K18_low =0.95  (w_K20_low =0.05)
           K18 >  q -> w_K18_high=0.40  (w_K20_high=0.60)
      3. Inner grid search on the BLENDED fold-train vs y_tr:
           for q_low in {0.01, 0.05, 0.10}:
             for q_high in {0.90, 0.95, 0.98, 0.99}:
               lo=quantile(y_tr,q_low); hi=quantile(y_tr,q_high)
               minimize rae(y_tr, clip(blend_tr, lo, hi))
         Pick (q_low*, q_high*) minimizing fold-train RAE.
      4. Apply learned (lo*, hi*) to BLENDED fold-val -> oof; record per-fold RAE.
    Repeat for 15 FRESH kf_seeds {1216..1230}.

GATE (honest metric = 15-seed PER-FOLD-MEAN, per task):
    pf_mean < 0.4423 -> "BETTER"
    else             -> "FAIL"

WEIGHTS (nb3090 q35 winning combo, transplanted to K20 partner):
    q_cut       = 0.35
    w_K18_low   = 0.95  (w_K20_low  = 0.05)
    w_K18_high  = 0.40  (w_K20_high = 0.60)

References:
    nb3090 q35 {K18,K19} 15-seed best  = 0.4472  (blend-only, K19 partner)
    nb3190 learned-clip on nb3090      = 0.4426  (MARGINAL, K19 partner+clip)
    nb3173 learned-clip operator ceil  = 0.4422  (gate-defining)
    nb3091 q50 {K18,K20} blend         = 0.4489  (FAIL, K20 weaker partner)
    nb2960 K18 deep-30 OOF             = 0.4536
    nb2960 K20 deep-30 OOF             = 0.4625
    nb2171 prior post-hoc top          = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb2960_K20_30seed_oof.npy
    data/processed/nb2960_K20_30seed_te.npy

Outputs:
    data/processed/nb3312_summary.json
    data/processed/nb3312_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3312.npy         (513,) float32 -- deploy te
    submissions/nb3312_clip_on_K18_K20_K23.csv  (only on BETTER verdict)
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

TAG = "nb3312"
PARENT_TAG = "nb3091"  # K18+K20 q-blend lineage

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K20"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K20": DATA_PROCESSED / "nb2960_K20_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K20": DATA_PROCESSED / "nb2960_K20_30seed_te.npy",
}
K_DEPTH = {"K18": "deep30", "K20": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- q35 quantile-conditional blend weights (nb3090 winning combo) -------------
Q_CUT = 0.35
W_K18_LOW = 0.95   # w_K20_low  = 1 - 0.95 = 0.05
W_K18_HIGH = 0.40  # w_K20_high = 1 - 0.40 = 0.60
W_K20_LOW = 1.0 - W_K18_LOW
W_K20_HIGH = 1.0 - W_K18_HIGH

# -- Learned-clip inner grid (nb3190 prescription) -----------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98, 0.99]

# -- Gate (honest metric = per-fold-mean, per task) ----------------------------
GATE_BETTER = 0.4423  # pf_mean < this -> BETTER; else FAIL

# -- References ----------------------------------------------------------------
REF_NB3090 = 0.4472     # q35 {K18,K19} blend-only
REF_NB3190 = 0.4426     # learned-clip on nb3090 ({K18,K19}+clip)
REF_NB3173 = 0.4422     # clip-operator ceiling
REF_NB3091 = 0.4489     # q50 {K18,K20} blend (FAIL)
REF_K18 = 0.4536
REF_K20 = 0.4625
REF_NB2171 = 0.4682


def _blend_quantile_conditional(
    p_k18: np.ndarray,
    p_k20: np.ndarray,
    q_thr: float,
) -> np.ndarray:
    """Per-row hard-split q35 blend on {K18, K20}.

    rows with p_k18 <= q_thr -> (W_K18_LOW=0.95, W_K20_LOW=0.05)
    rows with p_k18 >  q_thr -> (W_K18_HIGH=0.40, W_K20_HIGH=0.60)
    """
    low_mask = p_k18 <= q_thr
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = (
        W_K18_LOW * p_k18[low_mask] + W_K20_LOW * p_k20[low_mask]
    )
    out[~low_mask] = (
        W_K18_HIGH * p_k18[~low_mask] + W_K20_HIGH * p_k20[~low_mask]
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
    """Run q35 blend + learned-clip pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_q_thrs = []
    fold_ql = []
    fold_qh = []
    fold_lo = []
    fold_hi = []
    fold_high_share = []
    fold_clipped_lo = []
    fold_clipped_hi = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # 1. q35 threshold from fold-train K18
        q_thr = float(np.quantile(P_unb[tr_loc, 0], Q_CUT))
        fold_q_thrs.append(q_thr)

        # 2. q35 {K18, K20} blend on fold-train AND fold-val
        blend_tr = _blend_quantile_conditional(
            P_unb[tr_loc, 0], P_unb[tr_loc, 1], q_thr,
        )
        blend_va = _blend_quantile_conditional(
            P_unb[va_loc, 0], P_unb[va_loc, 1], q_thr,
        )
        fold_high_share.append(float(np.mean(P_unb[va_loc, 0] > q_thr)))

        # 3. learn clip on BLENDED fold-train vs y_tr
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], blend_tr)
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)

        # 4. apply learned clip to BLENDED fold-val
        n_lo = int(np.sum(blend_va < lo))
        n_hi = int(np.sum(blend_va > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)
        clipped = np.clip(blend_va, lo, hi)
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
        "fold_q_thr_mean": float(np.mean(fold_q_thrs)),
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "fold_lo_mean": float(np.mean(fold_lo)),
        "fold_hi_mean": float(np.mean(fold_hi)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "n_clipped_lo": int(np.sum(fold_clipped_lo)),
        "n_clipped_hi": int(np.sum(fold_clipped_hi)),
        "oof": oof_clip,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- q35 QUANTILE-BLEND {{{K_LABELS[0]},{K_LABELS[1]}}} + "
        f"LEARNED CLIP (NEW PARADIGM: K20 partner vs K19)"
    )
    print(
        f"          blend: q_cut={Q_CUT}, w_K18_low={W_K18_LOW} "
        f"(w_K20_low={W_K20_LOW}), w_K18_high={W_K18_HIGH} "
        f"(w_K20_high={W_K20_HIGH})"
    )
    print(f"          clip grid: Q_LOW={Q_LOW_GRID}  Q_HIGH={Q_HIGH_GRID}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"          honest gate metric = PER-FOLD-MEAN")
    print(f"          gate: pf_mean < {GATE_BETTER:.4f} -> BETTER; else FAIL")
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

    # -- Load K18, K20 deep-30 anchor OOFs + te arrays ------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18, K20 deep-30 OOFs and te arrays")
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
    print(
        f"   y_unb stats: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
        f"min={y_unb.min():.3f}  max={y_unb.max():.3f}"
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
            "fold_q_thr_mean": round(res["fold_q_thr_mean"], 4),
            "fold_ql": [round(v, 3) for v in res["fold_ql"]],
            "fold_qh": [round(v, 3) for v in res["fold_qh"]],
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(
            f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"q_thr={res['fold_q_thr_mean']:.3f}  "
            f"ql={res['fold_ql']}  qh={res['fold_qh']}  "
            f"clip(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
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

    # Most-picked q values across all 5*15=75 folds
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
    print(
        f"\n   ref nb3090 q35 {{K18,K19}} blend = {REF_NB3090:.4f}"
    )
    print(
        f"   ref nb3190 clip-on-nb3090       = {REF_NB3190:.4f} "
        f"({{K18,K19}}+clip)"
    )
    print(f"   delta vs nb3190 (pf_mean)       = {pf_mean - REF_NB3190:+.4f}")
    print(f"   ref nb3173 clip-operator ceil   = {REF_NB3173:.4f}")
    print(f"   delta vs nb3173 (pf_mean)       = {pf_mean - REF_NB3173:+.4f}")
    print(
        f"   ref nb3091 q50 {{K18,K20}} blend = {REF_NB3091:.4f} "
        f"(blend-only K20)"
    )
    print(f"   K18 standalone deep-30          = {REF_K18:.4f}")
    print(f"   K20 standalone deep-30          = {REF_K20:.4f}")
    print(f"   gain vs nb2171 (pf_mean)        = {REF_NB2171 - pf_mean:+.4f}")
    print(f"\n   ql_distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh_distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")

    # -- Deploy: full-253 blend, then learn clip on full-253, apply to te ----
    deploy_q_thr = float(np.quantile(P_unb[:, 0], Q_CUT))
    blend_unb_full = _blend_quantile_conditional(
        P_unb[:, 0], P_unb[:, 1], deploy_q_thr,
    )
    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(
        y_unb, blend_unb_full,
    )
    te_blend = _blend_quantile_conditional(
        P_te[:, 0], P_te[:, 1], deploy_q_thr,
    )
    te_pred = np.clip(te_blend, deploy_lo, deploy_hi).astype(np.float32)
    # Final safety bound to physical pEC50 range (matches nb3090/nb3091)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    n_te_lo = int(np.sum(te_blend < deploy_lo))
    n_te_hi = int(np.sum(te_blend > deploy_hi))
    te_low_share = float(np.mean(P_te[:, 0] <= deploy_q_thr))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy q_thr (full K18 OOF q{Q_CUT}) = {deploy_q_thr:.4f}"
    )
    print(f"   te(513) low-half share = {te_low_share:.3f}")
    print(
        f"   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f}) from full 253 y"
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

    # -- Gate (on PER-FOLD-MEAN per task) ------------------------------------
    print("\n" + "-" * 78)
    print("GATE (honest metric = PER-FOLD-MEAN)")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3312 15-seed PER-FOLD-MEAN {pf_mean:.4f} "
            f"clears gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). The "
            f"q35 {{K18,K20}} quantile blend + learned-clip RESCUES the weaker "
            f"K20 partner (blend-only nb3091 was 0.4489 FAIL): the learned-clip "
            f"operator compensates, landing at/under the nb3173 clip-operator "
            f"ceiling {REF_NB3173:.4f} and matching the {{K18,K19}} clip-sibling "
            f"nb3190 {REF_NB3190:.4f} (delta {pf_mean - REF_NB3190:+.4f}). K20 "
            f"opens a second viable partner axis for the q35+clip pipeline. "
            f"Modal clip pick (q{ql_mode}, q{qh_mode}). Re-verify with deep-30 "
            f"before any PRIMARY-1 swap. anchor_pre_unblind=True (K18/K20 are "
            f"PRE-clean deep-30 OOFs)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3312 15-seed PER-FOLD-MEAN {pf_mean:.4f} fails gate "
            f"{GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). Delta vs the "
            f"{{K18,K19}} clip-sibling nb3190 ({REF_NB3190:.4f}) = "
            f"{pf_mean - REF_NB3190:+.4f}; delta vs clip-operator ceiling "
            f"nb3173 ({REF_NB3173:.4f}) = {pf_mean - REF_NB3173:+.4f}. The "
            f"learned-clip operator does NOT rescue the weaker K20 partner "
            f"(K20 deep-30 {REF_K20:.4f} vs K19 0.4607): the blend-stage deficit "
            f"that sank nb3091 ({REF_NB3091:.4f}) carries through clip. The q35+"
            f"clip compounding path is K19-partner-specific; the K20 partner "
            f"axis is closed for the quantile-blend+clip family. Keep nb3190 / "
            f"nb3173 ({{K18,K19}}) on the ladder."
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

    sub_csv = SUBMISSIONS / f"{TAG}_clip_on_K18_K20_K23.csv"
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
            "q35_quantile_conditional_blend_K18_K20_deep30_plus_per_fold_"
            "learned_clip_new_paradigm_K20_partner"
        ),
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "blend_weights": {
            "q_cut": Q_CUT,
            "w_K18_low": W_K18_LOW,
            "w_K20_low": W_K20_LOW,
            "w_K18_high": W_K18_HIGH,
            "w_K20_high": W_K20_HIGH,
        },
        "clip_q_low_grid": Q_LOW_GRID,
        "clip_q_high_grid": Q_HIGH_GRID,
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
        "ref_nb3090_q35_K18_K19": REF_NB3090,
        "ref_nb3190_clip_K18_K19": REF_NB3190,
        "delta_vs_nb3190_pf_mean": round(pf_mean - REF_NB3190, 4),
        "ref_nb3173_clip_ceiling": REF_NB3173,
        "delta_vs_nb3173_pf_mean": round(pf_mean - REF_NB3173, 4),
        "ref_nb3091_q50_K18_K20": REF_NB3091,
        "ref_K18_deep30": REF_K18,
        "ref_K20_deep30": REF_K20,
        "ref_nb2171": REF_NB2171,
        "gain_vs_nb2171_pf_mean": round(REF_NB2171 - pf_mean, 4),
        "deploy_q_thr": round(deploy_q_thr, 4),
        "te_low_share": round(te_low_share, 4),
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
    print(f"   delta vs nb3190 (pf) = {pf_mean - REF_NB3190:+.4f}")
    print(f"   delta vs nb3173 (pf) = {pf_mean - REF_NB3173:+.4f}")
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
        "delta_vs_nb3190_pf_mean", "delta_vs_nb3173_pf_mean",
        "ql_mode", "qh_mode",
        "deploy_q_thr", "deploy_ql", "deploy_qh", "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  blend_weights: {res.get('blend_weights')}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
