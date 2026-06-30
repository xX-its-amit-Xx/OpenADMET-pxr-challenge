"""nb3373 -- Huber-loss-optimal blend weight on {K18, K19} deep-30 THEN
            per-fold learned clip.

NEW PARADIGM:
    Pick the {K18, K19} convex blend weight by minimizing HUBER loss
    (delta=0.5, robust to outliers) instead of RAE/MSE, then apply the
    per-fold learned tail clip. The hypothesis: the L1 (RAE) and L2 (MSE)
    blend objectives both sit in the ~0.45 simplex band (nb3002 RAE-SLSQP
    0.4501, equal-weight 0.4500); a Huber objective sits between L1 and L2
    (quadratic for |resid|<=delta, linear beyond) and may pick a blend
    weight that down-weights the heavy-tail novel-scaffold rows that
    dominate this analog-expansion test set -- a different w than either
    pure L1 or pure L2 -- on top of which the established clip primitive
    extracts its usual variance-decompression gain.

    Composition (per outer fold, fold-TRAIN only for all fitting):
        STEP 1: golden-section search for scalar w in [0,1] minimizing
                Huber_delta=0.5( y_tr, w*K18_tr + (1-w)*K19_tr ).
                blend = w*K18 + (1-w)*K19.
        STEP 2: on the fold-TRAIN blend, inner grid (q_low, q_high) over
                truth quantiles minimizing fold-TRAIN RAE (clip operator,
                identical grid/protocol to nb3223 / nb3201 / nb3173).
        STEP 3: apply BOTH the fold w AND the fold (lo, hi) clip to
                fold-VAL rows. Stitch -> blended_clipped_oof.
        Pooled + per-fold-mean RAE across 5 outer folds; 15 FRESH kf_seeds
        {1216..1230}, per-fold-mean aggregation.

    Deploy:
        - Golden-section w on FULL 253 (Huber).
        - Re-pick (q_low, q_high) on FULL 253 blend by inner RAE grid.
        - Apply w + clip to the (513, 2) anchor te arrays -> te_nb3373.

GATE (on per-fold-mean across 15 seeds):
    mean < 0.4423 -> "BETTER"   (task-supplied gate)
    else          -> "FAIL"

NOTE ON OBJECTIVE: the BLEND WEIGHT is chosen by Huber; the CLIP knots are
    chosen by RAE (the leaderboard metric), and the GATE is RAE. Huber is a
    *robust surrogate for the blend-weight search only* -- it does not
    replace the RAE evaluation/decision metric anywhere.

References (all PRE-unblind anchor chain, deep-30 {K18,K19}):
    nb2960 K18 deep-30 OOF        = 0.4536
    nb3000 K19 deep-30 OOF        = 0.4607
    equal-weight {K18,K19} OOF    = 0.4500  (in-sample)
    global RAE-opt w_K18=0.71     = 0.4488  (in-sample)
    global Huber-opt w_K18=0.62   = 0.4491  (in-sample; same band)
    nb3002 RAE-SLSQP simplex      = 0.4501  (15-seed pooled, no clip)
    nb3070 quantile-conditional   = 0.4509  (other base)
    nb3201 learned clip on K18    = ~0.4437
    nb3173 best clip-winner       = 0.4422
    nb3223 RAE-SLSQP simplex+clip = 0.4424  (RAE-objective sibling of THIS)
    nb2171 prior post-hoc PRIMARY-1 = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3373_summary.json
    data/processed/nb3373_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3373.npy         (513,) float32 -- deploy te
    submissions/nb3373_huber_blend_K18_K19.csv  (only on BETTER verdict)
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

TAG = "nb3373"
PARENT_TAG = "nb3223_huber_variant"

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

# -- Huber blend-weight search -------------------------------------------------
HUBER_DELTA = 0.5
GS_TOL = 1e-6        # golden-section convergence tol on w
GS_MAX_ITER = 200

# -- Per-fold clip grid (matches nb3223 / nb3201 / nb3173 family) --------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Gates (on PER-FOLD-MEAN over 15 seeds) ------------------------------------
GATE_BETTER = 0.4423  # mean < this -> BETTER (task-supplied gate)

# -- References ----------------------------------------------------------------
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_EQUAL_WEIGHT = 0.4500          # in-sample equal blend {K18,K19}
REF_NB3002 = 0.4501                # RAE-SLSQP simplex {K18,K19}, no clip
REF_NB3070 = 0.4509                # quantile-conditional verify
REF_NB3201_LEARNED_CLIP_K18 = 0.4437
REF_NB3173_BEST_CLIP_WINNER = 0.4422
REF_NB3223_SLSQP_CLIP = 0.4424     # RAE-objective sibling of THIS notebook
REF_NB2171 = 0.4682


def _huber_loss(resid: np.ndarray, delta: float = HUBER_DELTA) -> float:
    """Mean Huber loss of a residual vector.

    quadratic for |r| <= delta, linear (delta*(|r| - 0.5*delta)) beyond.
    """
    a = np.abs(resid)
    quad = 0.5 * a * a
    lin = delta * (a - 0.5 * delta)
    return float(np.where(a <= delta, quad, lin).mean())


def _golden_section_w(
    k18: np.ndarray,
    k19: np.ndarray,
    y: np.ndarray,
    delta: float = HUBER_DELTA,
    tol: float = GS_TOL,
    max_iter: int = GS_MAX_ITER,
) -> tuple[float, float]:
    """Golden-section search for w in [0,1] minimizing Huber loss of
    (y, w*k18 + (1-w)*k19). Returns (w_star, huber_at_w_star).

    The 1-D Huber-of-convex-blend objective is convex in w (Huber is convex,
    the blend is affine in w), so golden-section converges to the global min.
    """
    invphi = (np.sqrt(5.0) - 1.0) / 2.0        # 1/phi  ~ 0.618
    invphi2 = (3.0 - np.sqrt(5.0)) / 2.0       # 1/phi^2

    def f(w: float) -> float:
        return _huber_loss((w * k18 + (1.0 - w) * k19) - y, delta)

    a, b = 0.0, 1.0
    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if h <= tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            h = b - a
            c = a + invphi2 * h
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            h = b - a
            d = a + invphi * h
            fd = f(d)
    w_star = float((a + b) / 2.0)
    # Guard the interior optimum against the two boundary blends (pure K18 /
    # pure K19): golden-section can stall one bracket short of an endpoint
    # optimum on a near-monotone objective.
    cands = {0.0: f(0.0), 1.0: f(1.0), w_star: f(w_star)}
    w_best = min(cands, key=cands.get)
    return float(w_best), float(cands[w_best])


def _pick_best_clip(
    y_tr: np.ndarray,
    pred_tr: np.ndarray,
) -> tuple[float, float, float, float]:
    """Inner grid: pick (q_low*, q_high*) on truth quantiles minimizing
    fold-train RAE. Identical to nb3223 clip operator."""
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
    """Per-fold Huber-blend-weight + learned clip at one kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_final = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes: list[float] = []
    fold_train_raes: list[float] = []
    fold_w: list[float] = []
    fold_huber: list[float] = []
    fold_blend_train_raes: list[float] = []
    fold_ql: list[float] = []
    fold_qh: list[float] = []
    fold_lo: list[float] = []
    fold_hi: list[float] = []
    fold_clipped_lo: list[int] = []
    fold_clipped_hi: list[int] = []

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # --- Step 1: Huber-optimal blend weight on fold-train ----------------
        w, huber_tr = _golden_section_w(
            k18_unb[tr_loc], k19_unb[tr_loc], y_unb[tr_loc],
        )
        fold_w.append(w)
        fold_huber.append(huber_tr)

        tr_blend = w * k18_unb[tr_loc] + (1.0 - w) * k19_unb[tr_loc]
        va_blend = w * k18_unb[va_loc] + (1.0 - w) * k19_unb[va_loc]
        fold_blend_train_raes.append(float(rae(y_unb[tr_loc], tr_blend)))

        # --- Step 2: learned clip on fold-train blend ------------------------
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], tr_blend)
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)

        n_lo = int(np.sum(va_blend < lo))
        n_hi = int(np.sum(va_blend > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)

        # --- Step 3: apply clip to fold-val blend ----------------------------
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
        "fold_w": fold_w,
        "fold_w_mean": float(np.mean(fold_w)),
        "fold_huber_mean": float(np.mean(fold_huber)),
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
        f"{TAG} -- Huber(delta={HUBER_DELTA})-optimal blend weight on "
        f"{K_LABELS} deep-30 THEN per-fold learned clip"
    )
    print(
        f"          paradigm: blend w by golden-section min Huber "
        f"(robust); clip + GATE still RAE"
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

    # Reference baselines (in-sample) for sanity / interpretation -------------
    eq_blend = 0.5 * k18_unb + 0.5 * k19_unb
    eq_rae = float(rae(y_unb, eq_blend))
    w_glob, huber_glob = _golden_section_w(k18_unb, k19_unb, y_unb)
    glob_blend = w_glob * k18_unb + (1.0 - w_glob) * k19_unb
    glob_rae = float(rae(y_unb, glob_blend))
    print(
        f"\n   in-sample equal-weight blend RAE   = {eq_rae:.4f}"
    )
    print(
        f"   in-sample Huber-opt w_K18={w_glob:.3f} -> blend RAE={glob_rae:.4f}"
        f"  (huber={huber_glob:.4f})"
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
    all_fold_w: list[float] = []
    all_fold_ql: list[float] = []
    all_fold_qh: list[float] = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(k18_unb, k19_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        per_fold_stds.append(res["per_fold_val_rae_std"])
        oof_stack.append(res["oof"])
        all_fold_w.extend(res["fold_w"])
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
            "fold_w": [round(v, 4) for v in res["fold_w"]],
            "fold_w_mean": round(res["fold_w_mean"], 4),
            "fold_huber_mean": round(res["fold_huber_mean"], 4),
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
            f"w_K18~{res['fold_w_mean']:.3f}  "
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
    t_mult = 2.145  # df=14, two-sided 95%
    pf_ci_low = pf_mean - t_mult * pf_sem
    pf_ci_high = pf_mean + t_mult * pf_sem

    pooled_mean = float(arr_pooled.mean())
    pooled_std = float(arr_pooled.std(ddof=1)) if n_s > 1 else 0.0

    w_arr = np.asarray(all_fold_w, dtype=np.float64)  # (75,)
    w_mean = float(w_arr.mean())
    w_std = float(w_arr.std(ddof=1))

    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   POOLED (with fitted Huber-w + clip):")
    print(f"     mean = {pooled_mean:.4f}   std = {pooled_std:.4f}")
    print(f"     min/max = [{arr_pooled.min():.4f}, {arr_pooled.max():.4f}]")
    print(f"\n   PER-FOLD-MEAN (gate metric):")
    print(f"     mean    = {pf_mean:.4f}")
    print(f"     std     = {pf_std:.4f}")
    print(f"     sem     = {pf_sem:.4f}")
    print(f"     95% CI  = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"     median  = {pf_median:.4f}")
    print(f"     min/max = [{arr_pf.min():.4f}, {arr_pf.max():.4f}]")

    print(
        f"\n   Huber-w (75 folds): mean={w_mean:.4f}  std={w_std:.4f}  "
        f"min={w_arr.min():.4f}  max={w_arr.max():.4f}"
    )
    print(f"   ql_distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh_distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")

    print(f"\n   ref K18 deep-30 OOF          = {REF_K18:.4f}")
    print(f"   ref K19 deep-30 OOF          = {REF_K19:.4f}")
    print(
        f"   ref equal-weight (in-sample) = {REF_EQUAL_WEIGHT:.4f}  "
        f"(actual here {eq_rae:.4f})"
    )
    print(
        f"   ref nb3002 RAE-SLSQP simplex = {REF_NB3002:.4f}  "
        f"<- L1 blend, no clip"
    )
    print(
        f"   ref nb3223 RAE-SLSQP+clip    = {REF_NB3223_SLSQP_CLIP:.4f}  "
        f"<- L1 sibling of THIS"
    )
    print(f"   delta vs nb3223 (pf_mean)    = {pf_mean - REF_NB3223_SLSQP_CLIP:+.4f}")
    print(
        f"   ref nb3173 best clip-winner  = {REF_NB3173_BEST_CLIP_WINNER:.4f}"
    )
    print(f"   delta vs nb3173 (pf_mean)    = {pf_mean - REF_NB3173_BEST_CLIP_WINNER:+.4f}")
    print(f"   ref nb2171 prior PRIMARY-1   = {REF_NB2171:.4f}")
    print(f"   gain vs nb2171 (pf_mean)     = {REF_NB2171 - pf_mean:+.4f}")

    # -- Deploy: full-253 Huber-w + full-253 grid-picked clip -----------------
    print("\n" + "-" * 78)
    print("DEPLOY: refit Huber-w on FULL 253, then pick clip on FULL 253")
    print("-" * 78)
    deploy_w, deploy_huber = _golden_section_w(k18_unb, k19_unb, y_unb)
    full_blend = deploy_w * k18_unb + (1.0 - deploy_w) * k19_unb
    te_blend = deploy_w * te_k18 + (1.0 - deploy_w) * te_k19
    full_blend_rae = float(rae(y_unb, full_blend))
    print(
        f"   deploy w_K18 = {deploy_w:.4f}  (w_K19 = {1.0 - deploy_w:.4f})  "
        f"huber={deploy_huber:.4f}  in-sample blend RAE={full_blend_rae:.4f}"
    )

    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(
        y_unb, full_blend,
    )
    print(
        f"   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f}) from FULL 253 y"
    )

    te_pred = np.clip(te_blend, deploy_lo, deploy_hi).astype(np.float32)
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

    # -- Gate (on per-fold-mean) ---------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (per-fold-mean over 15 seeds)")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3373 15-seed per-fold-mean {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f} "
            f"({pf_mean - GATE_BETTER:+.4f}). Choosing the {K_LABELS} blend "
            f"weight by HUBER (delta={HUBER_DELTA}, robust) rather than RAE "
            f"selects w_K18~{w_mean:.2f} and, with the learned clip "
            f"(modal q{ql_mode:.2f}/q{qh_mode:.2f}), beats the RAE-objective "
            f"sibling nb3223 ({REF_NB3223_SLSQP_CLIP:.4f}, "
            f"delta {pf_mean - REF_NB3223_SLSQP_CLIP:+.4f}) and the best "
            f"clip-winner nb3173 ({REF_NB3173_BEST_CLIP_WINNER:.4f}, "
            f"delta {pf_mean - REF_NB3173_BEST_CLIP_WINNER:+.4f}). The robust "
            f"blend objective down-weights heavy-tail novel-scaffold residuals "
            f"during weight selection, yielding a different-and-better convex "
            f"blend substrate for the clip. Re-verify with deep-30 before any "
            f"PRIMARY-1 swap; watch Huber-w fold dispersion "
            f"(std={w_std:.3f}) for selection variance. anchor_pre_unblind=True."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3373 15-seed per-fold-mean {pf_mean:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"Selecting the {K_LABELS} blend weight by Huber "
            f"(delta={HUBER_DELTA}) instead of RAE/MSE picks w_K18~{w_mean:.2f} "
            f"but lands in the same ~0.45 simplex band as the L1/L2 blends "
            f"(nb3002 {REF_NB3002:.4f}, equal-weight {eq_rae:.4f}); the clip "
            f"then reaches {pf_mean:.4f}, delta vs RAE-objective sibling "
            f"nb3223 ({REF_NB3223_SLSQP_CLIP:.4f}) = "
            f"{pf_mean - REF_NB3223_SLSQP_CLIP:+.4f} and vs best clip-winner "
            f"nb3173 ({REF_NB3173_BEST_CLIP_WINNER:.4f}) = "
            f"{pf_mean - REF_NB3173_BEST_CLIP_WINNER:+.4f}. K18/K19 are "
            f"corr={corr:.3f}, so the convex blend has almost no free axis "
            f"and the choice of blend-loss (L1 vs L2 vs Huber) is "
            f"second-order at n=253; the post-hoc-blend ceiling holds. Closes "
            f"the Huber-blend-weight axis on {K_LABELS}; pivot to substrate "
            f"change (new anchor / off-manifold features / abstention)."
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

    sub_csv = SUBMISSIONS / f"{TAG}_huber_blend_K18_K19.csv"
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
            "huber_delta0.5_golden_section_blend_weight_K18_K19_deep30_"
            "then_per_fold_learned_clip"
        ),
        "paradigm": "robust_huber_blend_weight_vs_rae_slsqp_sibling_nb3223",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_oof_paths": {k: str(v) for k, v in OOF_PATHS.items()},
        "anchor_te_paths": {k: str(v) for k, v in TE_PATHS.items()},
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "oof_residual_corr": round(resid_corr, 4),
        "huber_delta": HUBER_DELTA,
        "blend_weight_objective": "huber_delta_0.5",
        "blend_weight_search": "golden_section_1d_w_in_[0,1]",
        "clip_objective": "fold_train_rae",
        "gate_metric": "rae_per_fold_mean",
        "in_sample_equal_weight_rae": round(eq_rae, 4),
        "in_sample_huber_w_K18": round(w_glob, 4),
        "in_sample_huber_blend_rae": round(glob_rae, 4),
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
        # Pooled (reference)
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "pooled_min": round(float(arr_pooled.min()), 4),
        "pooled_max": round(float(arr_pooled.max()), 4),
        # Huber-w distribution
        "huber_w_mean": round(w_mean, 4),
        "huber_w_std": round(w_std, 4),
        "huber_w_min": round(float(w_arr.min()), 4),
        "huber_w_max": round(float(w_arr.max()), 4),
        # Clip distribution
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        # Deploy
        "deploy_w_K18": round(float(deploy_w), 4),
        "deploy_w_K19": round(float(1.0 - deploy_w), 4),
        "deploy_huber": round(float(deploy_huber), 4),
        "deploy_blend_in_sample_rae": round(full_blend_rae, 4),
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
        "ref_equal_weight": REF_EQUAL_WEIGHT,
        "ref_nb3002_rae_slsqp_simplex": REF_NB3002,
        "ref_nb3070_quantile_cond": REF_NB3070,
        "ref_nb3201_learned_clip_K18": REF_NB3201_LEARNED_CLIP_K18,
        "ref_nb3173_best_clip_winner": REF_NB3173_BEST_CLIP_WINNER,
        "ref_nb3223_rae_slsqp_clip": REF_NB3223_SLSQP_CLIP,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18": round(pf_mean - REF_K18, 4),
        "delta_vs_K19": round(pf_mean - REF_K19, 4),
        "delta_vs_nb3002_rae_simplex": round(pf_mean - REF_NB3002, 4),
        "delta_vs_nb3223_rae_clip_sibling": round(
            pf_mean - REF_NB3223_SLSQP_CLIP, 4
        ),
        "delta_vs_nb3173_best_clip": round(
            pf_mean - REF_NB3173_BEST_CLIP_WINNER, 4
        ),
        "gain_vs_nb2171": round(REF_NB2171 - pf_mean, 4),
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
    print(f"   Huber-w mean          = {w_mean:.4f} +/- {w_std:.4f}")
    print(f"   delta vs nb3223 clip  = {pf_mean - REF_NB3223_SLSQP_CLIP:+.4f}")
    print(f"   delta vs nb3173 clip  = {pf_mean - REF_NB3173_BEST_CLIP_WINNER:+.4f}")
    print(f"   modal clip (ql, qh)   = ({ql_mode}, {qh_mode})")
    print(f"   deploy w_K18          = {deploy_w:.4f}")
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
        "huber_w_mean", "huber_w_std",
        "delta_vs_nb3223_rae_clip_sibling", "delta_vs_nb3173_best_clip",
        "gain_vs_nb2171",
        "ql_mode", "qh_mode",
        "deploy_w_K18", "deploy_w_K19", "deploy_ql", "deploy_qh",
        "deploy_lo", "deploy_hi", "n_te_clipped_lo", "n_te_clipped_hi",
        "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
