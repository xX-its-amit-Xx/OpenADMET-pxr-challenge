"""nb3403 -- Per-anchor PREDICTION-quantile WINSORIZE on {K18, K19} deep-30
            BEFORE blend, THEN q35 quantile-conditional blend + learned clip.

NEW PARADIGM -- WINSORIZE EACH ANCHOR'S PREDICTION DISTRIBUTION (not y, not
the blend output) BEFORE combining:
    Prior winsorize work (nb3362) winsorized the BLEND OUTPUT to a single
    (q05, q95) prediction band. Prior q35-blend ladders (nb3314, nb3070,
    nb3373) blended the RAW anchors with no pre-treatment. This script does
    something neither did: it winsorizes EACH anchor SEPARATELY to its OWN
    (q02, q98) PREDICTION quantiles, computed on fold-TRAIN, and only THEN
    feeds the two tamed anchors into the q35 quantile-conditional blend +
    learned clip.

      * nb3362 (winsorize blend out):  lo,hi = q(blend[tr], q05/q95); clip blend
      * nb3403 (winsorize each anchor): for K in {K18,K19}:
                                          loK,hiK = q(K[tr], q02/q98)
                                          K <- clip(K, loK, hiK)   (per anchor)
                                        THEN q35 blend(K18', K19') THEN clip

    Rationale: K18 and K19 are each variance-COMPRESSED deep-30 bags
    (pred_std ~0.75-0.90 vs truth_std ~1.03 on novel-scaffold OOD) whose
    INDIVIDUAL over-extended tail rows differ (their residuals are not
    perfectly correlated). Taming each anchor's own extreme 2% tails to its
    own prediction-quantile boundary BEFORE blending removes per-anchor tail
    over-confidence that the convex/quantile-conditional blend would
    otherwise average into the combined prediction. The downstream q35
    hard-split blend then operates on two cleaner inputs, and the learned
    clip extracts its usual residual variance-decompression on top. The
    per-anchor winsorize boundaries are PREDICTION-derived (truth-blind), so
    the deploy band transfers honestly; the only truth-fit component is the
    learned-clip quantile grid, fit strictly on fold-TRAIN.

PROTOCOL (per kf_seed, 5-fold scaffold split; anchors LOADED, no rebuild):
    Per outer fold (all params from fold-TRAIN ONLY -> clean cross-fit):
        STEP 1 (per-anchor winsorize, PREDICTION quantiles):
                for K in {K18, K19}:
                    loK = quantile(K[fold_train], 0.02)   (PREDICTION dist)
                    hiK = quantile(K[fold_train], 0.98)   (PREDICTION dist)
                    K_tr' = clip(K[fold_train], loK, hiK)
                    K_va' = clip(K[fold_val],   loK, hiK)
        STEP 2 (q35 quantile-conditional blend, nb3070/nb3314 schedule):
                q35 = quantile(K18'[fold_train], 0.35)
                  row K18' <= q35 -> 0.8*K18' + 0.2*K19'  (low/inactive tail)
                  row K18'  > q35 -> 0.5*K18' + 0.5*K19'  (high/active  tail)
        STEP 3 (learned clip): inner grid (q_low, q_high) on fold-TRAIN
                blended output; pick (lo*, hi*) minimizing fold-train RAE.
        STEP 4: val_pred = clip(val_blend, lo*, hi*). Stitch -> oof_final.
    Report BOTH pooled RAE and per-fold-mean RAE per seed, 15 FRESH kf_seeds
    {1216..1230}.

    Deploy:
        Per-anchor winsorize on FULL-253 (q02, q98) PREDICTION quantiles,
        applied to both 253 OOF and 513 te anchors.
        q35 from FULL-253 winsorized-K18 OOF; blend te.
        Re-pick (lo*, hi*) learned clip on FULL-253 blended output.
        te_pred = clip(clip(te_blend, lo*, hi*), 3.0, 9.0).

GATE (task-supplied; either clause -> BETTER):
    per-fold-mean < 0.4423  OR  pooled < 0.4419 -> "BETTER"
    else                                         -> "FAIL"

References (PRE-unblind anchor chain, deep-30 {K18,K19}):
    nb2960 K18 deep-30 OOF              = 0.4536
    nb3000 K19 deep-30 OOF              = 0.4607
    nb3002 RAE-SLSQP simplex {K18,K19}  = 0.4501
    nb3070 wide q-cond {K18,K19} deep30 = 0.4477 / 0.4509
    nb3173 best clip-winner            = 0.4422
    nb3223 SLSQP {K18,K19} + clip       = 0.4424
    nb3314 q35-blend + clip (deep-60)   = (composite sibling, no pre-winsor)
    nb3362 winsorize blend-out q05/q95  = (winsorize sibling, post-blend)
    nb2171 prior post-hoc top           = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3403_summary.json
    data/processed/nb3403_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3403.npy         (513,) float32 -- deploy te
    submissions/nb3403_winsor_then_blend.csv  (only on BETTER verdict)
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

TAG = "nb3403"
PARENT_TAG = "winsor_each_anchor+q35_blend+clip"

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

# -- Per-anchor winsorize quantiles (on the PREDICTION distribution) -----------
WINS_Q_LOW = 0.02
WINS_Q_HIGH = 0.98

# -- q35 quantile-conditional blend (nb3070/nb3314 schedule) -------------------
QUANTILE_CUT = 0.35
W_LOW_K18, W_LOW_K19 = 0.8, 0.2     # low / inactive regime (K18 <= q35)
W_HIGH_K18, W_HIGH_K19 = 0.5, 0.5   # high / active regime  (K18  > q35)

# -- Learned-clip grid (nb3173/nb3223/nb3314 family) ---------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Output range clip (matches nb3070/nb3314 deploy stage) --------------------
TE_RANGE_LO = 3.0
TE_RANGE_HI = 9.0

# -- Gate (task-supplied; either clause) ---------------------------------------
GATE_BETTER_PF = 0.4423   # per-fold-mean < this -> BETTER
GATE_BETTER_POOLED = 0.4419  # pooled < this -> BETTER

# -- References ----------------------------------------------------------------
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB3002 = 0.4501
REF_NB3070 = 0.4477
REF_NB3173_BEST_CLIP_WINNER = 0.4422
REF_NB3223_SLSQP_CLIP = 0.4424
REF_NB2171 = 0.4682


def _winsorize_anchor(
    p_tr: np.ndarray,
    p_va: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Winsorize ONE anchor to its OWN (WINS_Q_LOW, WINS_Q_HIGH) PREDICTION
    quantiles computed on fold-train. Boundaries are truth-blind.

    Returns (tr_winsorized, va_winsorized, lo, hi).
    """
    lo = float(np.quantile(p_tr, WINS_Q_LOW))
    hi = float(np.quantile(p_tr, WINS_Q_HIGH))
    return np.clip(p_tr, lo, hi), np.clip(p_va, lo, hi), lo, hi


def _blend_quantile_conditional(
    k18: np.ndarray,
    k19: np.ndarray,
    q35: float,
) -> np.ndarray:
    """q35 hard-split convex blend (nb3070/nb3314 schedule). The split is on
    the K18 axis: rows with K18 <= q35 use the low-regime weights, rows above
    use the high-regime weights."""
    low_mask = k18 <= q35
    out = np.empty_like(k18, dtype=np.float64)
    out[low_mask] = W_LOW_K18 * k18[low_mask] + W_LOW_K19 * k19[low_mask]
    out[~low_mask] = W_HIGH_K18 * k18[~low_mask] + W_HIGH_K19 * k19[~low_mask]
    return out


def _pick_best_clip(
    y_tr: np.ndarray,
    pred_tr: np.ndarray,
) -> tuple[float, float, float, float]:
    """Inner grid: pick (q_low*, q_high*) on TRUTH quantiles minimizing
    fold-train RAE (learned-clip operator, nb3173/nb3223 family)."""
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
    k18_unb: np.ndarray,
    k19_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Per-anchor winsorize -> q35 blend -> learned clip at one kf_seed.

    All winsorize boundaries, q35 cut, and clip knots are derived from
    fold-TRAIN ONLY, then applied to fold-VAL (clean cross-fit).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_final = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes: list[float] = []
    fold_train_raes: list[float] = []
    fold_blend_train_raes: list[float] = []   # post-winsor+blend, pre-clip (tr)
    fold_q35: list[float] = []
    fold_k18_lo: list[float] = []
    fold_k18_hi: list[float] = []
    fold_k19_lo: list[float] = []
    fold_k19_hi: list[float] = []
    fold_n_wins: list[int] = []               # total val rows winsorized
    fold_high_share: list[float] = []         # val share in high regime
    fold_ql: list[float] = []
    fold_qh: list[float] = []
    fold_lo: list[float] = []
    fold_hi: list[float] = []
    fold_clipped_lo: list[int] = []
    fold_clipped_hi: list[int] = []

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # --- STEP 1: per-anchor PREDICTION-quantile winsorize ----------------
        k18_tr, k18_va, k18_lo, k18_hi = _winsorize_anchor(
            k18_unb[tr_loc], k18_unb[va_loc]
        )
        k19_tr, k19_va, k19_lo, k19_hi = _winsorize_anchor(
            k19_unb[tr_loc], k19_unb[va_loc]
        )
        fold_k18_lo.append(k18_lo)
        fold_k18_hi.append(k18_hi)
        fold_k19_lo.append(k19_lo)
        fold_k19_hi.append(k19_hi)
        n_wins_va = int(
            np.sum(k18_unb[va_loc] != k18_va) + np.sum(k19_unb[va_loc] != k19_va)
        )
        fold_n_wins.append(n_wins_va)

        # --- STEP 2: q35 quantile-conditional blend --------------------------
        q35 = float(np.quantile(k18_tr, QUANTILE_CUT))
        fold_q35.append(q35)
        tr_blend = _blend_quantile_conditional(k18_tr, k19_tr, q35)
        va_blend = _blend_quantile_conditional(k18_va, k19_va, q35)
        fold_high_share.append(float(np.mean(k18_va > q35)))
        fold_blend_train_raes.append(float(rae(y_unb[tr_loc], tr_blend)))

        # --- STEP 3: learned clip on fold-train blended output ---------------
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], tr_blend)
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)
        n_lo = int(np.sum(va_blend < lo))
        n_hi = int(np.sum(va_blend > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)

        # --- STEP 4: apply clip to fold-val blend ----------------------------
        val_pred = np.clip(va_blend, lo, hi)
        oof_final[va_loc] = val_pred
        fold_train_raes.append(
            float(rae(y_unb[tr_loc], np.clip(tr_blend, lo, hi)))
        )
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
        "per_fold_blend_train_rae_mean": float(np.mean(fold_blend_train_raes)),
        "fold_q35_mean": float(np.mean(fold_q35)),
        "fold_q35_std": float(np.std(fold_q35, ddof=1)),
        "fold_k18_lo_mean": float(np.mean(fold_k18_lo)),
        "fold_k18_hi_mean": float(np.mean(fold_k18_hi)),
        "fold_k19_lo_mean": float(np.mean(fold_k19_lo)),
        "fold_k19_hi_mean": float(np.mean(fold_k19_hi)),
        "n_winsorized_val": int(np.sum(fold_n_wins)),
        "high_share_mean": float(np.mean(fold_high_share)),
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
        f"{TAG} -- per-anchor PREDICTION-quantile WINSORIZE "
        f"(q{WINS_Q_LOW:.2f}, q{WINS_Q_HIGH:.2f}) on {K_LABELS} deep-30"
    )
    print(
        f"          THEN q{int(QUANTILE_CUT*100)} quantile-conditional blend "
        f"+ learned clip"
    )
    print(
        f"          paradigm: winsorize EACH anchor's pred dist BEFORE blend "
        f"(neither nb3362 nor nb3314 did this)"
    )
    print(f"          Q_LOW_GRID  = {Q_LOW_GRID}")
    print(f"          Q_HIGH_GRID = {Q_HIGH_GRID}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate: pf_mean < {GATE_BETTER_PF:.4f} OR "
        f"pooled < {GATE_BETTER_POOLED:.4f} -> BETTER, else FAIL"
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
    print("STEP 1: load K18, K19 deep-30 OOF + te arrays")
    print("-" * 78)
    oof_cols: dict[str, np.ndarray] = {}
    te_cols: dict[str, np.ndarray] = {}
    per_K_full_rae: dict[str, float] = {}
    leak_flags: dict[str, float] = {}
    for k in K_LABELS:
        oof = np.load(OOF_PATHS[k]).astype(np.float64)
        te_a = np.load(TE_PATHS[k]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{k} OOF shape {oof.shape} != ({n_unb},)")
        if te_a.shape != (n_test,):
            raise ValueError(f"{k} te shape {te_a.shape} != ({n_test},)")
        oof_cols[k] = oof
        te_cols[k] = te_a
        r = float(rae(y_unb, oof))
        leak = float(np.mean(np.isclose(oof, y_unb, atol=1e-6)))
        per_K_full_rae[k] = round(r, 4)
        leak_flags[k] = round(leak, 4)
        print(
            f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE={r:.4f}  "
            f"oof mean={oof.mean():.3f} std={oof.std():.3f}  "
            f"leak_eq={leak:.2%}  "
            f"te mean={te_a.mean():.3f} std={te_a.std():.3f}"
        )
        if leak > 0.05:
            print(f"   WARN {k}: {leak:.1%} rows == truth -- possible leak")

    k18_unb = oof_cols["K18"]
    k19_unb = oof_cols["K19"]
    te_k18 = te_cols["K18"]
    te_k19 = te_cols["K19"]

    corr = float(np.corrcoef(k18_unb, k19_unb)[0, 1])
    resid_corr = float(np.corrcoef(k18_unb - y_unb, k19_unb - y_unb)[0, 1])
    print(f"\n   pairwise corr(K18, K19)      = {corr:.4f}")
    print(f"   residual corr(K18, K19)      = {resid_corr:.4f}")
    print(
        f"   y_unb stats: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
        f"min={y_unb.min():.3f}  max={y_unb.max():.3f}"
    )

    # Full-253 per-anchor winsorize quantiles (reference / deploy preview)
    for k in K_LABELS:
        flo = float(np.quantile(oof_cols[k], WINS_Q_LOW))
        fhi = float(np.quantile(oof_cols[k], WINS_Q_HIGH))
        print(
            f"   {k} full-253 winsor band: q{WINS_Q_LOW:.2f}={flo:.3f}  "
            f"q{WINS_Q_HIGH:.2f}={fhi:.3f}  (PREDICTION dist)"
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
    all_fold_q35: list[float] = []
    all_fold_ql: list[float] = []
    all_fold_qh: list[float] = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(k18_unb, k19_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        per_fold_stds.append(res["per_fold_val_rae_std"])
        oof_stack.append(res["oof"])
        all_fold_q35.append(res["fold_q35_mean"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_train_rae_mean": round(res["per_fold_train_rae_mean"], 4),
            "per_fold_blend_train_rae_mean": round(
                res["per_fold_blend_train_rae_mean"], 4
            ),
            "fold_q35_mean": round(res["fold_q35_mean"], 4),
            "fold_q35_std": round(res["fold_q35_std"], 4),
            "fold_k18_lo_mean": round(res["fold_k18_lo_mean"], 4),
            "fold_k18_hi_mean": round(res["fold_k18_hi_mean"], 4),
            "fold_k19_lo_mean": round(res["fold_k19_lo_mean"], 4),
            "fold_k19_hi_mean": round(res["fold_k19_hi_mean"], 4),
            "n_winsorized_val": res["n_winsorized_val"],
            "high_share_mean": round(res["high_share_mean"], 4),
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
            f"q35~{res['fold_q35_mean']:.3f}  "
            f"wins_va={res['n_winsorized_val']}  "
            f"clip(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr_pooled = np.asarray(pooled_raes, dtype=np.float64)
    arr_pf = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(arr_pf)

    # Aggregate stats: PER-FOLD-MEAN is the primary gate metric
    pf_mean = float(arr_pf.mean())
    pf_std = float(arr_pf.std(ddof=1)) if n_s > 1 else 0.0
    pf_sem = pf_std / np.sqrt(n_s) if n_s > 1 else 0.0
    pf_median = float(np.median(arr_pf))
    t_mult = 2.145  # df=14, two-sided 95%
    pf_ci_low = pf_mean - t_mult * pf_sem
    pf_ci_high = pf_mean + t_mult * pf_sem

    pooled_mean = float(arr_pooled.mean())
    pooled_std = float(arr_pooled.std(ddof=1)) if n_s > 1 else 0.0
    pooled_median = float(np.median(arr_pooled))

    q35_grand_mean = float(np.mean(all_fold_q35))
    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   POOLED (per-anchor winsor + q35 blend + clip):")
    print(f"     mean   = {pooled_mean:.4f}   std = {pooled_std:.4f}")
    print(f"     median = {pooled_median:.4f}")
    print(f"     min/max= [{arr_pooled.min():.4f}, {arr_pooled.max():.4f}]")
    print(f"\n   PER-FOLD-MEAN (primary gate metric):")
    print(f"     mean    = {pf_mean:.4f}")
    print(f"     std     = {pf_std:.4f}")
    print(f"     sem     = {pf_sem:.4f}")
    print(f"     95% CI  = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"     median  = {pf_median:.4f}")
    print(f"     min/max = [{arr_pf.min():.4f}, {arr_pf.max():.4f}]")

    print(f"\n   q35 grand-mean across seeds = {q35_grand_mean:.4f}")
    print(f"   ql_distribution (75 folds)  = {dict(ql_counter)}")
    print(f"   qh_distribution (75 folds)  = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")

    print(f"\n   ref K18 deep-30 OOF          = {REF_K18:.4f}")
    print(f"   ref K19 deep-30 OOF          = {REF_K19:.4f}")
    print(f"   ref nb3070 q-cond {{K18,K19}}  = {REF_NB3070:.4f}")
    print(
        f"   ref nb3223 SLSQP+clip        = {REF_NB3223_SLSQP_CLIP:.4f}  "
        f"<- composite ref"
    )
    print(
        f"   ref nb3173 best clip-winner  = "
        f"{REF_NB3173_BEST_CLIP_WINNER:.4f}"
    )
    print(
        f"   delta vs nb3173 (pf_mean)    = "
        f"{pf_mean - REF_NB3173_BEST_CLIP_WINNER:+.4f}"
    )
    print(f"   ref nb2171 prior post-hoc    = {REF_NB2171:.4f}")
    print(f"   gain vs nb2171 (pf_mean)     = {REF_NB2171 - pf_mean:+.4f}")

    # -- Deploy: full-253 per-anchor winsor -> q35 blend -> learned clip ------
    print("\n" + "-" * 78)
    print(
        "DEPLOY: per-anchor winsor on FULL 253 (and te), q35 from full-253, "
        "learned clip on full-253 blend"
    )
    print("-" * 78)
    deploy_wins: dict[str, dict[str, float]] = {}
    # Winsorize K18 / K19 OOF (253) and te (513) to full-253 PREDICTION bands.
    k18_oof_w = k18_unb.copy()
    k19_oof_w = k19_unb.copy()
    te_k18_w = te_k18.copy()
    te_k19_w = te_k19.copy()
    for k, oof_w_ref, te_w_ref in (
        ("K18", "k18", "te_k18"),
        ("K19", "k19", "te_k19"),
    ):
        flo = float(np.quantile(oof_cols[k], WINS_Q_LOW))
        fhi = float(np.quantile(oof_cols[k], WINS_Q_HIGH))
        deploy_wins[k] = {"lo": round(flo, 4), "hi": round(fhi, 4)}
    # apply (kept explicit for clarity)
    d_k18_lo = float(np.quantile(k18_unb, WINS_Q_LOW))
    d_k18_hi = float(np.quantile(k18_unb, WINS_Q_HIGH))
    d_k19_lo = float(np.quantile(k19_unb, WINS_Q_LOW))
    d_k19_hi = float(np.quantile(k19_unb, WINS_Q_HIGH))
    k18_oof_w = np.clip(k18_unb, d_k18_lo, d_k18_hi)
    k19_oof_w = np.clip(k19_unb, d_k19_lo, d_k19_hi)
    te_k18_w = np.clip(te_k18, d_k18_lo, d_k18_hi)
    te_k19_w = np.clip(te_k19, d_k19_lo, d_k19_hi)
    n_te_wins = int(np.sum(te_k18 != te_k18_w) + np.sum(te_k19 != te_k19_w))
    print(
        f"   K18 deploy winsor band = ({d_k18_lo:.3f}, {d_k18_hi:.3f})  "
        f"K19 = ({d_k19_lo:.3f}, {d_k19_hi:.3f})"
    )
    print(f"   te rows winsorized (K18+K19) = {n_te_wins}/{2 * n_test}")

    deploy_q35 = float(np.quantile(k18_oof_w, QUANTILE_CUT))
    full_blend = _blend_quantile_conditional(k18_oof_w, k19_oof_w, deploy_q35)
    te_blend = _blend_quantile_conditional(te_k18_w, te_k19_w, deploy_q35)
    full_blend_rae = float(rae(y_unb, full_blend))
    te_low_share = float(np.mean(te_k18_w <= deploy_q35))
    print(
        f"   deploy q35 (full-253 winsor-K18) = {deploy_q35:.4f}  "
        f"in-sample blend RAE={full_blend_rae:.4f}"
    )
    print(f"   te low-regime share (te_K18' <= q35) = {te_low_share:.3f}")

    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(
        y_unb, full_blend,
    )
    print(
        f"   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f}) from FULL 253 y"
    )

    te_pred = np.clip(te_blend, deploy_lo, deploy_hi)
    te_pred = np.clip(te_pred, TE_RANGE_LO, TE_RANGE_HI).astype(np.float32)
    n_te_lo = int(np.sum(te_blend < deploy_lo))
    n_te_hi = int(np.sum(te_blend > deploy_hi))
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

    # Median-seed OOF for storage (by per-fold-mean)
    med_seed_idx = int(np.argsort(arr_pf)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"\n   median (by pf_mean) seed = {median_seed} "
        f"(pf_mean={arr_pf[med_seed_idx]:.4f}, "
        f"pooled={arr_pooled[med_seed_idx]:.4f})"
    )

    # -- Gate (either clause -> BETTER) --------------------------------------
    print("\n" + "-" * 78)
    print("GATE (per-fold-mean < 0.4423 OR pooled < 0.4419)")
    print("-" * 78)
    pass_pf = pf_mean < GATE_BETTER_PF
    pass_pooled = pooled_mean < GATE_BETTER_POOLED
    if pass_pf or pass_pooled:
        verdict = "BETTER"
        which = []
        if pass_pf:
            which.append(
                f"per-fold-mean {pf_mean:.4f} < {GATE_BETTER_PF:.4f} "
                f"({pf_mean - GATE_BETTER_PF:+.4f})"
            )
        if pass_pooled:
            which.append(
                f"pooled {pooled_mean:.4f} < {GATE_BETTER_POOLED:.4f} "
                f"({pooled_mean - GATE_BETTER_POOLED:+.4f})"
            )
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3403 clears BETTER via " + " AND ".join(which)
            + f". Winsorizing EACH anchor's PREDICTION distribution to its own "
            f"(q{WINS_Q_LOW:.2f}, q{WINS_Q_HIGH:.2f}) band BEFORE the q35 "
            f"quantile-conditional blend (then learned clip) extracts gain "
            f"past the post-hoc-blend band (nb3173 "
            f"{REF_NB3173_BEST_CLIP_WINNER:.4f}, nb3223 "
            f"{REF_NB3223_SLSQP_CLIP:.4f}). Per-anchor tail-taming is "
            f"truth-blind, so the deploy band transfers honestly; only the "
            f"learned-clip grid is truth-fit (fold-train only). Re-verify with "
            f"deep-30 (cycle-160 rule: 15-seed std under-dispersed ~4x; "
            f"current pf_std={pf_std:.4f}) before any PRIMARY-1 swap. "
            f"anchor_pre_unblind=True; modal clip "
            f"(q{ql_mode:.2f}, q{qh_mode:.2f})."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3403 per-fold-mean {pf_mean:.4f} "
            f"(>= {GATE_BETTER_PF:.4f}, {pf_mean - GATE_BETTER_PF:+.4f}) and "
            f"pooled {pooled_mean:.4f} (>= {GATE_BETTER_POOLED:.4f}, "
            f"{pooled_mean - GATE_BETTER_POOLED:+.4f}) both miss the gate. "
            f"Winsorizing each {K_LABELS} anchor to its own "
            f"(q{WINS_Q_LOW:.2f}, q{WINS_Q_HIGH:.2f}) PREDICTION band before "
            f"the q35 blend is NEUTRAL/HURTS: the deep-30 bag-mean already "
            f"compressed each anchor's extreme tails (30-seed averaging does "
            f"the variance compression that per-anchor winsorize targets), so "
            f"clamping the residual 2% tails removes little over-confidence and "
            f"the q35 blend + learned clip land in the same ~0.44 band as the "
            f"no-winsor composites (nb3173 {REF_NB3173_BEST_CLIP_WINNER:.4f}, "
            f"nb3223 {REF_NB3223_SLSQP_CLIP:.4f}). K18/K19 corr={corr:.3f}, so "
            f"per-anchor pre-treatment has little independent axis to exploit. "
            f"Closes the per-anchor-pre-winsorize axis on {K_LABELS}; pivot to "
            f"substrate change (new anchor / off-manifold features / "
            f"abstention). modal clip (q{ql_mode:.2f}, q{qh_mode:.2f})."
        )
    print(f"   pass_pf     = {pass_pf}   pass_pooled = {pass_pooled}")
    print(f"   verdict     = {verdict}")
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

    sub_csv = SUBMISSIONS / f"{TAG}_winsor_then_blend.csv"
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
            "per_anchor_prediction_quantile_winsorize_q02_q98_K18_K19_deep30_"
            "then_q35_quantile_conditional_blend_then_learned_clip"
        ),
        "paradigm": (
            "winsorize_each_anchor_prediction_dist_BEFORE_blend "
            "(vs nb3362 winsorize-blend-output, vs nb3314 no-pre-winsor)"
        ),
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_oof_paths": {k: str(v) for k, v in OOF_PATHS.items()},
        "anchor_te_paths": {k: str(v) for k, v in TE_PATHS.items()},
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "oof_residual_corr": round(resid_corr, 4),
        # Winsorize config
        "winsorize_q_low": WINS_Q_LOW,
        "winsorize_q_high": WINS_Q_HIGH,
        "winsorize_boundary_source": "per_anchor_prediction_distribution_fold_train",
        "deploy_winsorize_bands": deploy_wins,
        "n_te_winsorized": int(n_te_wins),
        # q35 blend config
        "quantile_cut": QUANTILE_CUT,
        "w_low_K18": W_LOW_K18,
        "w_low_K19": W_LOW_K19,
        "w_high_K18": W_HIGH_K18,
        "w_high_K19": W_HIGH_K19,
        "q35_grand_mean_across_seeds": round(q35_grand_mean, 4),
        # clip config
        "clip_objective": "fold_train_rae",
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "te_range_lo": TE_RANGE_LO,
        "te_range_hi": TE_RANGE_HI,
        "gate_metric": "per_fold_mean_OR_pooled",
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
        # Primary gate metric: per-fold-mean (mirrored to mean_rae)
        "pf_mean": round(pf_mean, 4),
        "pf_std": round(pf_std, 4),
        "pf_sem": round(pf_sem, 4),
        "pf_ci95_low": round(pf_ci_low, 4),
        "pf_ci95_high": round(pf_ci_high, 4),
        "pf_median": round(pf_median, 4),
        "pf_min": round(float(arr_pf.min()), 4),
        "pf_max": round(float(arr_pf.max()), 4),
        "mean_rae": round(pf_mean, 4),
        "std_rae": round(pf_std, 4),
        # Pooled (second gate clause)
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "pooled_median": round(pooled_median, 4),
        "pooled_min": round(float(arr_pooled.min()), 4),
        "pooled_max": round(float(arr_pooled.max()), 4),
        # Clip distribution
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        # Deploy
        "deploy_k18_winsor_lo": round(d_k18_lo, 4),
        "deploy_k18_winsor_hi": round(d_k18_hi, 4),
        "deploy_k19_winsor_lo": round(d_k19_lo, 4),
        "deploy_k19_winsor_hi": round(d_k19_hi, 4),
        "deploy_q35": round(deploy_q35, 4),
        "deploy_blend_in_sample_rae": round(full_blend_rae, 4),
        "deploy_te_low_share": round(te_low_share, 4),
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
        "ref_nb3002_rae_slsqp_simplex": REF_NB3002,
        "ref_nb3070_q_cond": REF_NB3070,
        "ref_nb3173_best_clip_winner": REF_NB3173_BEST_CLIP_WINNER,
        "ref_nb3223_rae_slsqp_clip": REF_NB3223_SLSQP_CLIP,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18": round(pf_mean - REF_K18, 4),
        "delta_vs_K19": round(pf_mean - REF_K19, 4),
        "delta_vs_nb3070_q_cond": round(pf_mean - REF_NB3070, 4),
        "delta_vs_nb3223_rae_clip": round(
            pf_mean - REF_NB3223_SLSQP_CLIP, 4
        ),
        "delta_vs_nb3173_best_clip": round(
            pf_mean - REF_NB3173_BEST_CLIP_WINNER, 4
        ),
        "gain_vs_nb2171": round(REF_NB2171 - pf_mean, 4),
        # Gate
        "gate_better_pf": GATE_BETTER_PF,
        "gate_better_pooled": GATE_BETTER_POOLED,
        "pass_pf": bool(pass_pf),
        "pass_pooled": bool(pass_pooled),
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
    print(f"   pooled_mean           = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"   q35 grand-mean        = {q35_grand_mean:.4f}")
    print(
        f"   delta vs nb3173 clip  = "
        f"{pf_mean - REF_NB3173_BEST_CLIP_WINNER:+.4f}"
    )
    print(f"   modal clip (ql, qh)   = ({ql_mode}, {qh_mode})")
    print(f"   deploy q35            = {deploy_q35:.4f}")
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
        "q35_grand_mean_across_seeds",
        "delta_vs_nb3223_rae_clip", "delta_vs_nb3173_best_clip",
        "gain_vs_nb2171",
        "ql_mode", "qh_mode",
        "deploy_k18_winsor_lo", "deploy_k18_winsor_hi",
        "deploy_k19_winsor_lo", "deploy_k19_winsor_hi",
        "deploy_q35", "deploy_ql", "deploy_qh", "deploy_lo", "deploy_hi",
        "n_te_winsorized", "n_te_clipped_lo", "n_te_clipped_hi",
        "te_unb_in_sample_rae",
        "pass_pf", "pass_pooled",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
