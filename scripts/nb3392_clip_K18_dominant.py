"""nb3392 -- q35 quantile-conditional blend {K18, K19} deep-30 with K18
            DOMINANT in BOTH regimes (w_low=0.95, w_high=0.70) + learned clip.

NEW PARADIGM (K18-dominant everywhere):
    Every prior q-cond ladder rung made the HIGH (active) tail K19-heavy:
        nb3070/nb3314  high = 0.5*K18 + 0.5*K19
        nb3090 best    high = 0.40*K18 + 0.60*K19   (K19-dominant high tail)
        nb3093         high = 0.40*K18 + 0.60*K19   (K19 anchors actives)
    The implicit thesis was "K19 carries the active-tail signal". This script
    INVERTS that thesis: keep K18 DOMINANT in BOTH regimes --

        LOW  (K18 <= q35):  0.95*K18 + 0.05*K19   (K18-only-low; near-pure K18)
        HIGH (K18 >  q35):  0.70*K18 + 0.30*K19   (still K18-heavy)

    K18 (deep-30 OOF 0.4536) is the stronger single anchor than K19 (0.4607),
    so a K18-dominant schedule keeps the blend anchored on the stronger member
    everywhere and only lets K19 nudge a small minority share. The learned
    per-fold tail clip then decompresses the variance-compressed tails. The
    question: does staying near-pure-K18 (with a light K19 correction) + clip
    beat the K19-heavy-high q-cond + clip ceiling on this anchor pair?

    Anchors (deep-30, cached -- no rebuild; matches nb3374 inputs):
        K18-deep30  -- nb2960 cache (OOF 0.4536)
        K19-deep30  -- nb3000 cache (OOF 0.4607)

    Composite (per outer fold, fold-TRAIN-derived q35 + clip -- no val leak):
        STEP 1: q35 = 35th percentile of fold-TRAIN K18-deep30 OOF preds.
        STEP 2: quantile-conditional K18-dominant blend (hard split):
                  row K18 <= q35 -> 0.95*K18 + 0.05*K19   (LOW / inactive)
                  row K18  > q35 -> 0.70*K18 + 0.30*K19   (HIGH / active)
        STEP 3: learned clip -- inner grid (q_low, q_high) on fold-TRAIN
                blended output, pick (lo*, hi*) minimizing fold-train RAE.
        STEP 4: val_blend = blend(K18_val, K19_val, q35);
                val_pred  = clip(val_blend, lo*, hi*).
        Stitch 5 fold-vals -> oof_final; pooled + per-fold-mean RAE.
        Repeat for 15 FRESH kf_seeds {1216..1230}.

    Deploy:
        q35_deploy = 35th percentile of FULL-253 K18-deep30 OOF.
        te_blend   = blend(te_K18, te_K19, q35_deploy).
        (lo*, hi*) = learned clip on FULL-253 blended output.
        te_pred    = clip(clip(te_blend, lo*, hi*), 3.0, 9.0).

GATE (on PER-FOLD-MEAN across 15 seeds):
    mean < 0.4423 -> "BETTER"
    else          -> "FAIL"

References (PRE-unblind anchor chain; chemprop_aux anchored):
    nb2960 K18 deep-30 OOF                 = 0.4536  (stronger anchor)
    nb3000 K19 deep-30 OOF                 = 0.4607
    nb3090 q-cond best (K19-heavy high)    = 0.4475-band (BETTER_THAN_NB3080)
    nb3070 wide q-cond {K18,K19} deep30    = 0.4477 / 0.4509
    nb3173 best clip-winner                = 0.4422
    nb3200 deep-30 clip winner OOF         = 0.4416 (strongest single member)
    nb3223 SLSQP {K18,K19} + clip          = 0.4424
    nb3314 q35 blend deep60 + clip (FAIL)  = 0.4706
    nb3374 median3 + clip (FAIL)           = 0.4534
    nb2171 prior post-hoc top              = 0.4682
    chemprop_aux anchor                    = 0.6216

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3392_summary.json
    data/processed/nb3392_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3392.npy         (513,) float32 -- deploy te
    submissions/nb3392_clip_K18_dominant.csv  (only on BETTER verdict)
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

TAG = "nb3392"
PARENT_TAG = "q35_K18_dominant_blend_then_learned_clip_deep30"

# -- Inputs (deep-30 cached anchors; identical to nb3374) ----------------------
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

# -- q35 quantile-conditional blend weights (K18 DOMINANT EVERYWHERE) ----------
QUANTILE_CUT = 0.35   # q35 split on K18 prediction
# LOW  (K18 <= q35):  0.95*K18 + 0.05*K19   (K18-only-low; near-pure K18)
# HIGH (K18 >  q35):  0.70*K18 + 0.30*K19   (still K18-heavy)
W_LOW_K18, W_LOW_K19 = 0.95, 0.05
W_HIGH_K18, W_HIGH_K19 = 0.70, 0.30

# -- Learned-clip grid (nb3173 / nb3223 / nb3374 family) -----------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Output range clip (matches nb3070 / nb3314 / nb3374 deploy stage) ---------
TE_RANGE_LO = 3.0
TE_RANGE_HI = 9.0

# -- Gate ----------------------------------------------------------------------
GATE_BETTER = 0.4423  # per-fold-mean < this -> BETTER (user-supplied gate)

# -- References ----------------------------------------------------------------
REF_K18 = 0.4536            # stronger single anchor
REF_K19 = 0.4607
REF_NB3090 = 0.4475         # q-cond best (K19-heavy high tail)
REF_NB3070 = 0.4477
REF_NB3173_CLIP = 0.4422    # best clip-winner
REF_NB3200 = 0.4416         # strongest single member (deep-30 clip winner)
REF_NB3223 = 0.4424
REF_NB3314_DEEP60 = 0.4706
REF_NB3374_MEDIAN = 0.4534
REF_NB2171 = 0.4682
CHEMPROP_AUX_REF = 0.6216


def _blend_quantile_conditional(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q_thr: float,
) -> np.ndarray:
    """Per-row hard-split blend by q_thr (fold-train-derived), K18 DOMINANT.

    rows with p_k18 <= q_thr -> 0.95*K18 + 0.05*K19   (low / inactive)
    rows with p_k18 >  q_thr -> 0.70*K18 + 0.30*K19   (high / active)
    """
    low_mask = p_k18 <= q_thr
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = W_LOW_K18 * p_k18[low_mask] + W_LOW_K19 * p_k19[low_mask]
    out[~low_mask] = W_HIGH_K18 * p_k18[~low_mask] + W_HIGH_K19 * p_k19[~low_mask]
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
    p_k18_unb: np.ndarray,
    p_k19_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """q35 K18-dominant blend + learned clip at one outer kf_seed.

    q35 + clip params derived from fold-TRAIN ONLY (clean cross-fit).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_final = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes: list[float] = []
    fold_blend_train_raes: list[float] = []
    fold_base_val_raes: list[float] = []  # blend base, pre-clip, on fold-val
    fold_q35s: list[float] = []
    fold_ql: list[float] = []
    fold_qh: list[float] = []
    fold_lo: list[float] = []
    fold_hi: list[float] = []
    fold_clipped_lo: list[int] = []
    fold_clipped_hi: list[int] = []
    fold_high_share: list[float] = []

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # --- STEP 1: q35 from fold-TRAIN K18-deep30 preds ONLY --------------
        q35 = float(np.quantile(p_k18_unb[tr_loc], QUANTILE_CUT))
        fold_q35s.append(q35)

        # --- STEP 2: quantile-conditional K18-dominant blend ----------------
        tr_blend = _blend_quantile_conditional(
            p_k18_unb[tr_loc], p_k19_unb[tr_loc], q35
        )
        va_blend = _blend_quantile_conditional(
            p_k18_unb[va_loc], p_k19_unb[va_loc], q35
        )
        fold_blend_train_raes.append(float(rae(y_unb[tr_loc], tr_blend)))
        fold_base_val_raes.append(float(rae(y_unb[va_loc], va_blend)))
        fold_high_share.append(float(np.mean(p_k18_unb[va_loc] > q35)))

        # --- STEP 3: learned clip on fold-train blended ---------------------
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], tr_blend)
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)

        n_lo = int(np.sum(va_blend < lo))
        n_hi = int(np.sum(va_blend > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)

        # --- STEP 4: apply clip to fold-val blended -------------------------
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
        "per_fold_base_val_rae_mean": float(np.mean(fold_base_val_raes)),
        "fold_q35_mean": float(np.mean(fold_q35s)),
        "fold_q35_std": float(np.std(fold_q35s, ddof=1)),
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
        f"{TAG} -- q{int(QUANTILE_CUT*100)} K18-DOMINANT blend {{K18,K19}} "
        f"deep-30 + learned per-fold clip"
    )
    print(
        f"          LOW  (K18 <= q35):  {W_LOW_K18}*K18 + {W_LOW_K19}*K19  "
        "(K18-only-low)"
    )
    print(
        f"          HIGH (K18 >  q35):  {W_HIGH_K18}*K18 + {W_HIGH_K19}*K19  "
        "(K18-heavy high)"
    )
    print(f"          clip grid: ql={Q_LOW_GRID}  qh={Q_HIGH_GRID}")
    print(
        f"          outer kf_seeds = {KF_SEEDS[0]}..{KF_SEEDS[-1]} "
        f"(n={len(KF_SEEDS)})"
    )
    print(f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER else FAIL")
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

    # -- Load deep-30 anchors -------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18, K19 deep-30 OOF + te arrays")
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
            f"   {k:>4s} ({K_DEPTH[k]:>6s}): oof_RAE={r:.4f}  "
            f"oof mean={oof.mean():.3f} std={oof.std():.3f}  "
            f"leak_eq={leak:.2%}  "
            f"te mean={te_a.mean():.3f} std={te_a.std():.3f}"
        )
        if leak > 0.05:
            print(f"   WARN {k}: {leak:.1%} rows == truth -- possible leak")
    p_k18_unb = oof_cols[0]
    p_k19_unb = oof_cols[1]
    p_k18_te = te_cols[0]
    p_k19_te = te_cols[1]

    pair_corr = float(np.corrcoef(p_k18_unb, p_k19_unb)[0, 1])
    print(f"   K18 vs K19 OOF pearson = {pair_corr:.4f}")
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

    # -- q35 K18-dominant blend + learned-clip 15-seed sweep ------------------
    print("\n" + "-" * 78)
    print(
        f"STEP 3: WIDE-SEED q{int(QUANTILE_CUT*100)} K18-DOMINANT BLEND + "
        f"LEARNED-CLIP SWEEP -- {len(KF_SEEDS)} fresh kf_seeds "
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
    all_fold_q35: list[float] = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(p_k18_unb, p_k19_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        per_fold_stds.append(res["per_fold_val_rae_std"])
        base_fold_means.append(res["per_fold_base_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        all_fold_q35.append(res["fold_q35_mean"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_blend_train_rae_mean": round(
                res["per_fold_blend_train_rae_mean"], 4),
            "per_fold_base_val_rae_mean": round(
                res["per_fold_base_val_rae_mean"], 4),
            "fold_q35_mean": round(res["fold_q35_mean"], 4),
            "fold_q35_std": round(res["fold_q35_std"], 4),
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
            f"base_pf={res['per_fold_base_val_rae_mean']:.4f}  "
            f"q35={res['fold_q35_mean']:.3f}  "
            f"hi_share={res['fold_high_share_mean']:.2f}  "
            f"clip(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr_pool = np.asarray(pooled_raes, dtype=np.float64)
    arr_pfm = np.asarray(per_fold_means, dtype=np.float64)
    arr_base = np.asarray(base_fold_means, dtype=np.float64)
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
    base_pf_mean = float(arr_base.mean())

    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]
    q35_grand_mean = float(np.mean(all_fold_q35))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled_RAE       mean = {pooled_mean:.4f}  std = {pooled_std:.4f}")
    print(f"   pooled  min/max  [{arr_pool.min():.4f}, {arr_pool.max():.4f}]")
    print(f"   base_pf_mean (pre-clip) = {base_pf_mean:.4f}  (clip lift target)")
    print(
        f"   per_fold_mean    mean = {per_fold_mean:.4f}  std = {per_fold_std:.4f}  "
        f"95% CI [{ci_low_pfm:.4f}, {ci_high_pfm:.4f}]   <- GATE METRIC"
    )
    print(f"   per-fm  median   = {pf_median:.4f}")
    print(f"   per-fm  min/max  [{arr_pfm.min():.4f}, {arr_pfm.max():.4f}]")
    print(f"   clip lift vs base = {per_fold_mean - base_pf_mean:+.4f}")
    print(f"\n   ql distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")
    print(f"   grand-mean q35 across seeds = {q35_grand_mean:.4f}")
    print(f"\n   ref nb3173 best clip-winner = {REF_NB3173_CLIP:.4f}")
    print(f"   ref nb3090 q-cond (K19-high) = {REF_NB3090:.4f}")
    print(f"   ref K18 deep-30 (stronger)   = {REF_K18:.4f}")
    print(f"   delta vs nb3173             = {per_fold_mean - REF_NB3173_CLIP:+.4f}")
    print(f"   delta vs nb3090             = {per_fold_mean - REF_NB3090:+.4f}")
    print(f"   delta vs K18 deep-30        = {per_fold_mean - REF_K18:+.4f}")

    # -- Deploy: q35 from FULL 253 K18, blend te, learned clip ----------------
    print("\n" + "-" * 78)
    print("STEP 4: DEPLOY -- q35 from full-253 K18-deep30 -> blend te(513) "
          "-> learned clip on full-253 blended")
    print("-" * 78)
    deploy_q35 = float(np.quantile(p_k18_unb, QUANTILE_CUT))
    full_blend = _blend_quantile_conditional(p_k18_unb, p_k19_unb, deploy_q35)
    te_blend = _blend_quantile_conditional(p_k18_te, p_k19_te, deploy_q35)
    full_blend_rae = float(rae(y_unb, full_blend))
    print(f"   deploy q35 (full-253 K18-deep30) = {deploy_q35:.4f}")
    print(f"   full-253 blended in-sample RAE   = {full_blend_rae:.4f}")

    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(y_unb, full_blend)
    print(
        f"   deploy clip = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f}) from full-253 y"
    )

    te_pred_pre_range = np.clip(te_blend, deploy_lo, deploy_hi)
    te_pred = np.clip(te_pred_pre_range, TE_RANGE_LO, TE_RANGE_HI).astype(np.float32)
    n_te_lo = int(np.sum(te_blend < deploy_lo))
    n_te_hi = int(np.sum(te_blend > deploy_hi))
    te_low_share = float(np.mean(p_k18_te <= deploy_q35))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te low-half share (te_K18 <= q35) = {te_low_share:.3f}")
    print(
        f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513  "
        f"total={n_te_lo + n_te_hi}/513"
    )
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage (by pf_mean)
    med_seed_idx = int(np.argsort(arr_pfm)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"   median (by pf_mean) seed = {median_seed} "
        f"(pf_mean={arr_pfm[med_seed_idx]:.4f}, "
        f"pooled={arr_pool[med_seed_idx]:.4f})"
    )

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: GATE (per-fold-mean over 15 seeds)")
    print("-" * 78)
    if per_fold_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3392 q{int(QUANTILE_CUT*100)} K18-DOMINANT "
            f"blend {{K18,K19}} deep-30 (low={W_LOW_K18}*K18+{W_LOW_K19}*K19, "
            f"high={W_HIGH_K18}*K18+{W_HIGH_K19}*K19) + learned clip 15-seed "
            f"per-fold-mean {per_fold_mean:.4f} clears the BETTER gate "
            f"{GATE_BETTER:.4f} ({per_fold_mean - GATE_BETTER:+.4f}), beating "
            f"nb3173 clip-winner ({REF_NB3173_CLIP:.4f}) by "
            f"{per_fold_mean - REF_NB3173_CLIP:+.4f}. INVERTING the active-tail "
            f"thesis WORKS: staying near-pure-K18 (the stronger anchor, "
            f"{REF_K18:.4f} vs K19 {REF_K19:.4f}) in BOTH regimes with only a "
            f"light K19 correction + clip beats the K19-heavy-high q-cond "
            f"schedule (nb3090 {REF_NB3090:.4f}). Re-verify deep-30 cycle-160 "
            f"rule (15-seed std under-dispersed ~4x) with deep-30+ seeds before "
            f"any PRIMARY-1 swap. Predicted LB under +0.0045 PRE delta = "
            f"{per_fold_mean + 0.0045:.4f}. Modal clip = "
            f"(q{ql_mode:.2f}, q{qh_mode:.2f}); clip lift vs base = "
            f"{per_fold_mean - base_pf_mean:+.4f}."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3392 q{int(QUANTILE_CUT*100)} K18-DOMINANT blend "
            f"{{K18,K19}} deep-30 + learned clip 15-seed per-fold-mean "
            f"{per_fold_mean:.4f} fails BETTER gate {GATE_BETTER:.4f} "
            f"({per_fold_mean - GATE_BETTER:+.4f}). Inverting the active-tail "
            f"thesis to K18-DOMINANT everywhere (low={W_LOW_K18}*K18, "
            f"high={W_HIGH_K18}*K18) does NOT beat the clip-winner ladder: "
            f"near-pure-K18 collapses the blend toward the single K18 anchor "
            f"(deep-30 {REF_K18:.4f}), and the small K19 share cannot supply "
            f"the active-tail correction that the K19-heavy-high schedule "
            f"(nb3090 {REF_NB3090:.4f}) and 50/50-high (nb3070 {REF_NB3070:.4f}) "
            f"provide. The q-cond+clip ceiling is set by the (anchor="
            f"chemprop_aux, K=18/19, n=253) substrate and the BALANCE of the "
            f"two anchors, not by maximizing the stronger member's weight. Keep "
            f"nb3173 ({REF_NB3173_CLIP:.4f}) / nb3200 ({REF_NB3200:.4f}) / "
            f"nb3223 ({REF_NB3223:.4f}) on ladder; no ladder change. blended "
            f"pre-clip in-sample RAE = {full_blend_rae:.4f}. Modal clip = "
            f"(q{ql_mode:.2f}, q{qh_mode:.2f})."
        )
    print(f"   per_fold_mean   = {per_fold_mean:.4f}  (gate: < {GATE_BETTER:.4f})")
    print(f"   pooled_mean     = {pooled_mean:.4f}  (informational)")
    print(f"   verdict         = {verdict}")
    print(f"   ladder action   = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_clip_K18_dominant.csv"
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
        "method": ("q35_K18_dominant_quantile_conditional_blend_K18_K19_deep30_"
                   "then_learned_per_fold_clip"),
        "paradigm": "K18_dominant_everywhere_inverts_K19_heavy_active_tail_thesis",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "anchor_pair_oof_pearson_K18_K19": round(pair_corr, 4),
        # q35 blend (K18 dominant)
        "quantile_cut": QUANTILE_CUT,
        "q_source": "fold_train_only",
        "w_low": {"K18": W_LOW_K18, "K19": W_LOW_K19},
        "w_high": {"K18": W_HIGH_K18, "K19": W_HIGH_K19},
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
        "base_per_fold_means_array": [round(float(v), 4) for v in base_fold_means],
        # Gate metric: per-fold-mean
        "per_fold_mean_rae": round(per_fold_mean, 4),
        "per_fold_std_rae": round(per_fold_std, 4),
        "per_fold_sem_rae": round(sem_pfm, 4),
        "per_fold_ci95_low": round(ci_low_pfm, 4),
        "per_fold_ci95_high": round(ci_high_pfm, 4),
        "per_fold_median_rae": round(pf_median, 4),
        "per_fold_min_rae": round(float(arr_pfm.min()), 4),
        "per_fold_max_rae": round(float(arr_pfm.max()), 4),
        "base_pf_mean": round(base_pf_mean, 4),
        "clip_lift_vs_base": round(per_fold_mean - base_pf_mean, 4),
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
        "q35_grand_mean_across_seeds": round(q35_grand_mean, 4),
        # references
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb3090": REF_NB3090,
        "ref_nb3070": REF_NB3070,
        "ref_nb3173_clip": REF_NB3173_CLIP,
        "ref_nb3200": REF_NB3200,
        "ref_nb3223": REF_NB3223,
        "ref_nb3314_deep60": REF_NB3314_DEEP60,
        "ref_nb3374_median": REF_NB3374_MEDIAN,
        "ref_nb2171": REF_NB2171,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_nb3173": round(per_fold_mean - REF_NB3173_CLIP, 4),
        "delta_vs_nb3090": round(per_fold_mean - REF_NB3090, 4),
        "delta_vs_nb3070": round(per_fold_mean - REF_NB3070, 4),
        "delta_vs_nb3223": round(per_fold_mean - REF_NB3223, 4),
        "delta_vs_K18_deep30": round(per_fold_mean - REF_K18, 4),
        # deploy
        "deploy_q35": round(deploy_q35, 4),
        "deploy_blend_in_sample_rae": round(full_blend_rae, 4),
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
    print(f"   q35 K18-dominant blend + clip per-fold-mean = {per_fold_mean:.4f} "
          f"+/- {per_fold_std:.4f}  <- GATE METRIC")
    print(f"   95% CI                                      = "
          f"[{ci_low_pfm:.4f}, {ci_high_pfm:.4f}]")
    print(f"   base (pre-clip) per-fold-mean               = {base_pf_mean:.4f}")
    print(f"   clip lift                                   = "
          f"{per_fold_mean - base_pf_mean:+.4f}")
    print(f"   pooled mean                                 = {pooled_mean:.4f} "
          f"+/- {pooled_std:.4f}")
    print(f"   q35 grand-mean                              = {q35_grand_mean:.4f}")
    print(f"   delta vs nb3173 clip (0.4422)               = "
          f"{per_fold_mean - REF_NB3173_CLIP:+.4f}")
    print(f"   delta vs nb3090 (K19-heavy 0.4475)          = "
          f"{per_fold_mean - REF_NB3090:+.4f}")
    print(f"   modal clip (ql, qh)                         = ({ql_mode}, {qh_mode})")
    print(f"   te[unb] in-sample                           = {te_unb_in_rae:.4f}")
    print(f"   verdict                                     = {verdict}")
    print(f"   wall                                        = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean_rae",
        "per_fold_std_rae",
        "per_fold_ci95_low",
        "per_fold_ci95_high",
        "base_pf_mean",
        "clip_lift_vs_base",
        "pooled_mean_rae",
        "pooled_std_rae",
        "delta_vs_nb3173",
        "delta_vs_nb3090",
        "delta_vs_K18_deep30",
        "ql_mode",
        "qh_mode",
        "deploy_q35",
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
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
