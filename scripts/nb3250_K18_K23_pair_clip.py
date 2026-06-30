"""nb3250 -- Per-fold SLSQP simplex on {K18, K23} deep-30 THEN per-fold
            learned clip on the simplex output.

NEW PARADIGM:
    Try K23 as partner for K18 (vs K19/K20 pairs tested previously) plus the
    per-fold learned-clip primitive. The hypothesis is that K23 contributes
    different residual error structure than K19/K20 (different feature subset
    / different SHAP cut), and an SLSQP simplex over the {K18, K23} pair
    followed by a tail clip might extract additional decompression gain.

    Composition:
        STEP 1 (per fold): SLSQP simplex w on fold-TRAIN over (K18, K23)
                           minimizing RAE; produces simplex blend per fold.
        STEP 2 (per fold): on the simplex fold-TRAIN predictions, do an
                           inner grid (q_low, q_high) search to pick the
                           tail-clip percentiles minimizing fold-TRAIN RAE.
        STEP 3 (per fold): apply both the per-fold simplex weights AND the
                           per-fold (lo, hi) clip to fold-VAL rows.
        Stitch -> blended_clipped_oof; pooled + per-fold-mean RAE across
        5 outer folds. 15 fresh kf_seeds {1216..1230}.

    Deploy:
        - Refit SLSQP simplex on FULL 253 -> single global (w_K18, w_K23).
        - Re-pick (q_low, q_high) on FULL 253 simplex output by inner grid.
        - Apply both to (513, 2) anchor te arrays -> te_nb3250.

GATE (on per-fold-mean across 15 seeds):
    mean < 0.4423 -> "BETTER"
    else          -> "FAIL"

References (all PRE-unblind anchor chain):
    nb2960 K18 deep-30 OOF        = 0.4536
    nb3020 K23 deep-30 bag mean   = 0.4750
    nb3002 SLSQP simplex {K18,K19} 15-seed pooled = 0.4501
    nb3070 quantile-conditional 15-seed pooled    = 0.4509
    nb3201 learned clip on K18 alone              = ~0.4437
    nb3173 learned clip on nb3080 wide-bag        = 0.4422 (best clip-winner)
    nb3214 SLSQP on 3 clip winners                = ~0.4418
    nb3223 SLSQP simplex {K18,K19} + clip         = (parent template)

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3020_K23_30seed_oof.npy
    data/processed/te_nb3020_K23.npy

Outputs:
    data/processed/nb3250_summary.json
    data/processed/nb3250_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3250.npy         (513,) float32 -- deploy te
    submissions/nb3250_K18_K23_pair_clip.csv  (only on BETTER verdict)
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
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3250"
PARENT_TAG = "nb3223_simplex+clip"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K23"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K23": DATA_PROCESSED / "nb3020_K23_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K23": DATA_PROCESSED / "te_nb3020_K23.npy",
}
K_DEPTH = {"K18": "deep30", "K23": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- SLSQP options -------------------------------------------------------------
N_SLSQP_STARTS_FOLD = 8
N_SLSQP_STARTS_FULL = 12
DEGEN_MAX_W = 0.95  # 2-anchor case can legitimately go heavy on K18

# -- Per-fold clip grid (matches nb3201 / nb3173 / nb3223 family) --------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Gates (on PER-FOLD-MEAN over 15 seeds) ------------------------------------
GATE_BETTER = 0.4423  # mean < this -> BETTER (user-supplied gate)

# -- References ----------------------------------------------------------------
REF_K18 = 0.4536
REF_K23 = 0.4750
REF_NB3002 = 0.4501            # SLSQP simplex {K18,K19} 15-seed pooled
REF_NB3070 = 0.4509            # quantile-conditional verify
REF_NB3201_LEARNED_CLIP_K18 = 0.4437
REF_NB3173_BEST_CLIP_WINNER = 0.4422
REF_NB3214_SLSQP_CLIP_3 = 0.4418
REF_NB3223 = 0.4424            # template parent: {K18,K19} + clip
REF_K19 = 0.4607               # K19 deep-30 (reference for FAIL ladder text)
REF_NB2171 = 0.4682


def _simplex_slsqp(
    P: np.ndarray,
    y: np.ndarray,
    n_starts: int = N_SLSQP_STARTS_FOLD,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Minimize RAE(y, P @ w) on the simplex (w>=0, sum(w)=1) with multi-start."""
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w: np.ndarray) -> float:
        return float(rae(y, P @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K

    starts = [np.full(K, 1.0 / K)]
    for _ in range(max(0, n_starts - 1)):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w: np.ndarray | None = None
    best_r = np.inf
    for x0 in starts:
        try:
            res = minimize(
                loss, x0, method="SLSQP",
                bounds=bnds, constraints=cons,
                options={"maxiter": 300, "ftol": 1e-9},
            )
            w = np.clip(res.x, 0.0, 1.0)
            s = float(w.sum())
            if s <= 0.0:
                continue
            w = w / s
            r = float(rae(y, P @ w))
            if r < best_r:
                best_r = r
                best_w = w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_r = float(rae(y, P @ best_w))
    return best_w, best_r


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
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Per-fold SLSQP simplex + learned clip at one kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    K = P_unb.shape[1]
    oof_final = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes: list[float] = []
    fold_train_raes: list[float] = []
    fold_weights: list[np.ndarray] = []
    fold_ql: list[float] = []
    fold_qh: list[float] = []
    fold_lo: list[float] = []
    fold_hi: list[float] = []
    fold_clipped_lo: list[int] = []
    fold_clipped_hi: list[int] = []
    fold_simplex_train_raes: list[float] = []

    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # --- Step 1: SLSQP simplex on fold-train ----------------------------
        w, r_tr_simplex = _simplex_slsqp(
            P_unb[tr_loc], y_unb[tr_loc],
            n_starts=N_SLSQP_STARTS_FOLD,
            seed=kf_seed * 100 + fold_i,
        )
        fold_weights.append(w)
        fold_simplex_train_raes.append(float(r_tr_simplex))

        # Apply weights to fold-train AND fold-val
        tr_simplex = P_unb[tr_loc] @ w
        va_simplex = P_unb[va_loc] @ w

        # --- Step 2: learned clip on fold-train simplex output --------------
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], tr_simplex)
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)

        n_lo = int(np.sum(va_simplex < lo))
        n_hi = int(np.sum(va_simplex > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)

        # --- Step 3: apply clip to fold-val simplex output ------------------
        val_pred = np.clip(va_simplex, lo, hi)
        oof_final[va_loc] = val_pred
        r_tr = float(rae(y_unb[tr_loc], np.clip(tr_simplex, lo, hi)))
        fold_train_raes.append(r_tr)
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))

    if np.isnan(oof_final).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_final))
    W = np.stack(fold_weights, axis=0)  # (n_folds, K)
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "per_fold_train_rae_mean": float(np.mean(fold_train_raes)),
        "per_fold_simplex_train_rae_mean": float(np.mean(fold_simplex_train_raes)),
        "fold_weights": W,
        "any_fold_degenerate": bool(W.max(axis=1).max() > DEGEN_MAX_W),
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
        f"{TAG} -- per-fold SLSQP simplex on {K_LABELS} deep-30 "
        f"THEN per-fold learned clip"
    )
    print(f"          paradigm: K23 as partner for K18 (vs K19/K20)")
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
    print("STEP 1: load K18, K23 deep-30 OOF + te arrays")
    print("-" * 78)
    oof_cols: list[np.ndarray] = []
    te_cols: list[np.ndarray] = []
    per_K_full_rae: dict[str, float] = {}
    leak_flags: dict[str, float] = {}
    for k in K_LABELS:
        oof = np.load(OOF_PATHS[k]).astype(np.float64)
        te_a = np.load(TE_PATHS[k]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(
                f"{k} OOF shape {oof.shape} != ({n_unb},)"
            )
        if te_a.shape != (n_test,):
            raise ValueError(
                f"{k} te shape {te_a.shape} != ({n_test},)"
            )
        oof_cols.append(oof)
        te_cols.append(te_a)
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
    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)

    corr = float(np.corrcoef(P_unb.T)[0, 1])
    print(f"\n   pairwise corr({K_LABELS[0]}, {K_LABELS[1]}) = {corr:.4f}")

    # Truth stats
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
    all_fold_weights: list[np.ndarray] = []
    all_fold_ql: list[float] = []
    all_fold_qh: list[float] = []
    any_degen = False
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        per_fold_stds.append(res["per_fold_val_rae_std"])
        oof_stack.append(res["oof"])
        all_fold_weights.append(res["fold_weights"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        any_degen = any_degen or res["any_fold_degenerate"]
        seed_mean_w = res["fold_weights"].mean(axis=0)
        seed_mean_w = seed_mean_w / max(seed_mean_w.sum(), 1e-12)
        per_fold_w_round = [
            {K_LABELS[k]: round(float(res["fold_weights"][f, k]), 3)
             for k in range(len(K_LABELS))}
            for f in range(res["fold_weights"].shape[0])
        ]
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_train_rae_mean": round(res["per_fold_train_rae_mean"], 4),
            "per_fold_simplex_train_rae_mean": round(
                res["per_fold_simplex_train_rae_mean"], 4
            ),
            "per_fold_weights": per_fold_w_round,
            "any_fold_degenerate": res["any_fold_degenerate"],
            "fold_ql": [round(v, 3) for v in res["fold_ql"]],
            "fold_qh": [round(v, 3) for v in res["fold_qh"]],
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        wmean_str = "  ".join(
            f"{K_LABELS[k]}={seed_mean_w[k]:.2f}"
            for k in range(len(K_LABELS))
        )
        print(
            f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"w_mean({wmean_str})  "
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

    # Mean-of-fold weights across all 75 folds
    all_W = np.concatenate(all_fold_weights, axis=0)  # (75, 2)
    w_mean_of_folds = all_W.mean(axis=0)
    w_mean_of_folds = w_mean_of_folds / max(w_mean_of_folds.sum(), 1e-12)

    # Most-picked q values
    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   POOLED (split-variant, with fitted SLSQP+clip):")
    print(f"     mean = {pooled_mean:.4f}   std = {pooled_std:.4f}")
    print(f"     min/max = [{arr_pooled.min():.4f}, {arr_pooled.max():.4f}]")
    print(f"\n   PER-FOLD-MEAN (gate metric):")
    print(f"     mean    = {pf_mean:.4f}")
    print(f"     std     = {pf_std:.4f}")
    print(f"     sem     = {pf_sem:.4f}")
    print(f"     95% CI  = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"     median  = {pf_median:.4f}")
    print(f"     min/max = [{arr_pf.min():.4f}, {arr_pf.max():.4f}]")

    print("\n   mean-of-folds weights (across 75 folds):")
    for k in range(len(K_LABELS)):
        print(f"     w[{K_LABELS[k]:>4s}] = {w_mean_of_folds[k]:+.4f}")

    print(f"\n   ql_distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh_distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")

    print(
        f"\n   ref K18 deep-30 OOF         = {REF_K18:.4f}"
    )
    print(
        f"   ref K23 deep-30 bag         = {REF_K23:.4f}"
    )
    print(
        f"   ref nb3002 SLSQP {{K18,K19}} = {REF_NB3002:.4f}  "
        f"<- prior pair simplex"
    )
    print(
        f"   ref nb3223 {{K18,K19}}+clip  = {REF_NB3223:.4f}  "
        f"<- template parent"
    )
    print(
        f"   delta vs nb3223 (pair+clip) = "
        f"{pf_mean - REF_NB3223:+.4f}"
    )
    print(
        f"   ref nb3201 clip K18 alone   = {REF_NB3201_LEARNED_CLIP_K18:.4f}"
    )
    print(
        f"   ref nb3173 best clip-winner = {REF_NB3173_BEST_CLIP_WINNER:.4f}"
    )
    print(
        f"   ref nb3214 SLSQP-clip-3     = {REF_NB3214_SLSQP_CLIP_3:.4f}"
    )

    # -- Deploy: full-253 SLSQP simplex + full-253 grid-picked clip -----------
    print("\n" + "-" * 78)
    print("DEPLOY: refit SLSQP simplex on FULL 253, then pick clip on FULL 253")
    print("-" * 78)
    w_full, r_full_simplex = _simplex_slsqp(
        P_unb, y_unb, n_starts=N_SLSQP_STARTS_FULL, seed=0,
    )
    full_pool_weights = {
        K_LABELS[k]: round(float(w_full[k]), 4) for k in range(len(K_LABELS))
    }
    full_pool_degen = bool(w_full.max() > DEGEN_MAX_W)
    print(f"   simplex in-sample RAE = {r_full_simplex:.4f}  "
          f"max_w={w_full.max():.4f}  degen={full_pool_degen}")
    for k in range(len(K_LABELS)):
        flag = " (zeroed)" if w_full[k] < 1e-6 else ""
        print(f"     w[{K_LABELS[k]:>4s}] = {w_full[k]:+.4f}{flag}")

    # Apply simplex to full-253 (for clip picking) and to te
    full_simplex = P_unb @ w_full
    te_simplex = P_te @ w_full

    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(
        y_unb, full_simplex,
    )
    print(
        f"\n   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
        f"({deploy_lo:.3f}, {deploy_hi:.3f}) from FULL 253 y"
    )

    te_pred = np.clip(te_simplex, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_simplex < deploy_lo))
    n_te_hi = int(np.sum(te_simplex > deploy_hi))
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
            f"PROMOTE-CANDIDATE. nb3250 15-seed per-fold-mean {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f} "
            f"({pf_mean - GATE_BETTER:+.4f}). Swapping K19 -> K23 as the K18 "
            f"partner in the pair-simplex+clip composition produces a new "
            f"ceiling candidate. Mean-of-folds weights = "
            f"{{{K_LABELS[0]}={w_mean_of_folds[0]:.2f}, "
            f"{K_LABELS[1]}={w_mean_of_folds[1]:.2f}}}, "
            f"modal clip = (q{ql_mode:.2f}, q{qh_mode:.2f}). "
            f"K23 brings different residual-error structure than K19/K20 "
            f"(different SHAP feature subset). Re-verify with deep-30 before "
            f"any PRIMARY-1 swap; cycle-160 deep-30 rule mandatory for "
            f"gate-grade decisions."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3250 15-seed per-fold-mean {pf_mean:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"Substituting K23 for K19 as the K18 partner does NOT beat the "
            f"{K_LABELS} simplex+clip ceiling. K23 (deep-30 OOF "
            f"{REF_K23:.4f}) is weaker than K19 ({REF_K19:.4f}) and the "
            f"residual decorrelation versus K18 is insufficient to "
            f"compensate. Keep "
            f"current clip-winner ladder (nb3173 best single 0.4422). "
            f"Mean-of-folds weights = "
            f"{{{K_LABELS[0]}={w_mean_of_folds[0]:.2f}, "
            f"{K_LABELS[1]}={w_mean_of_folds[1]:.2f}}}, "
            f"modal clip = (q{ql_mode:.2f}, q{qh_mode:.2f})."
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

    sub_csv = SUBMISSIONS / f"{TAG}_K18_K23_pair_clip.csv"
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
        "method": "per_fold_slsqp_simplex_K18_K23_deep30_then_per_fold_learned_clip",
        "paradigm": "K23_as_K18_partner_vs_K19_K20_pair_clip",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "n_folds": N_FOLDS,
        "n_slsqp_starts_fold": N_SLSQP_STARTS_FOLD,
        "n_slsqp_starts_full": N_SLSQP_STARTS_FULL,
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
        # Weights / clips
        "any_fold_degenerate_across_seeds": bool(any_degen),
        "mean_of_fold_weights": {
            K_LABELS[k]: round(float(w_mean_of_folds[k]), 4)
            for k in range(len(K_LABELS))
        },
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        # Deploy
        "deploy_simplex_weights": full_pool_weights,
        "deploy_simplex_in_sample_rae": round(float(r_full_simplex), 4),
        "deploy_simplex_max_w": round(float(w_full.max()), 4),
        "deploy_simplex_degenerate": full_pool_degen,
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
        "ref_K23_deep30": REF_K23,
        "ref_nb3002_simplex_K18_K19": REF_NB3002,
        "ref_nb3070_quantile_cond": REF_NB3070,
        "ref_nb3201_learned_clip_K18": REF_NB3201_LEARNED_CLIP_K18,
        "ref_nb3173_best_clip_winner": REF_NB3173_BEST_CLIP_WINNER,
        "ref_nb3214_slsqp_clip_3": REF_NB3214_SLSQP_CLIP_3,
        "ref_nb3223_pair_clip": REF_NB3223,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18": round(pf_mean - REF_K18, 4),
        "delta_vs_K23": round(pf_mean - REF_K23, 4),
        "delta_vs_nb3002_raw_simplex": round(pf_mean - REF_NB3002, 4),
        "delta_vs_nb3070_quantile_cond": round(pf_mean - REF_NB3070, 4),
        "delta_vs_nb3173_best_clip": round(pf_mean - REF_NB3173_BEST_CLIP_WINNER, 4),
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
    print(f"   delta vs nb3223 pair  = {pf_mean - REF_NB3223:+.4f}")
    print(f"   delta vs nb3173 clip  = {pf_mean - REF_NB3173_BEST_CLIP_WINNER:+.4f}")
    print(f"   mean-fold weights     = "
          f"{ {K_LABELS[k]: round(float(w_mean_of_folds[k]),3) for k in range(len(K_LABELS))} }")
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
        "delta_vs_nb3223_pair_clip", "delta_vs_nb3173_best_clip",
        "mean_of_fold_weights", "ql_mode", "qh_mode",
        "deploy_simplex_weights", "deploy_ql", "deploy_qh",
        "deploy_lo", "deploy_hi", "n_te_clipped_lo", "n_te_clipped_hi",
        "te_unb_in_sample_rae",
        "any_fold_degenerate_across_seeds",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
