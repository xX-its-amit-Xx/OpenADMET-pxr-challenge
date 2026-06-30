"""nb3421 -- SLSQP weights minimizing POOLED RAE on clip winners (vs per-fold).

NEW PARADIGM: a GLOBAL SLSQP that minimizes the POOLED (LB-faithful) RAE directly.

Motivation (from nb3402 meta-analysis): the public LB scores all 513 rows with
ONE RAE denominator -- it is a POOLED computation by construction. The honest
253-unblind analog is therefore the pooled RAE over all 253 cross-fit
predictions (a single rae() call on the concatenated OOF vector), NOT the mean
of 5 per-fold ratios. This script makes that objective explicit at BOTH levels:

  (a) INNER fit: per outer fold, SLSQP minimizes the POOLED RAE on the
      fold-TRAIN portion -- i.e. one rae(y_tr, P_tr @ w) call, a single
      denominator over all fold-train rows (NOT a mean of inner per-fold
      ratios). This is the "pooled objective" the paradigm asks for.
  (b) OUTER aggregate: fold-val predictions (params fit on fold-train, applied
      to held-out fold-val) are concatenated across all 5 folds and scored with
      ONE rae() call over all 253 -> the POOLED cross-fit RAE. This is the
      LB-faithful estimand we gate on, repeated over 15 FRESH kf_seeds and
      reported as the across-seed mean.

ANCHORS (all clip-winner operators on the frozen PRE-unblind chemprop_aux chain,
all inside the ~0.4416-0.4430 PRE-unblind ceiling band, all leak-free):
    nb3173 -- learned per-fold clip on nb3080 anchor   (oof pooled 0.4423)
    nb3174 -- fixed (q05,q95) clip on nb3090 anchor     (oof pooled 0.4430)
    nb3190 -- learned per-fold clip on nb3090 anchor    (oof pooled 0.4422)
    nb3200 -- deep-verify learned clip on nb3090        (oof pooled 0.4416)

The 4 anchors are >0.998 cross-correlated, so SLSQP has very little orthogonal
signal to exploit; the open question is whether minimizing the pooled objective
(rather than the per-fold-train-mean objective nb3214 implicitly used) buys the
last sliver below the equal-weight pooled baseline (0.4415).

PROTOCOL (per kf_seed, outer 5-fold scaffold split on 253 unblind):
    P  = stack([nb3173_oof, nb3174_oof, nb3190_oof, nb3200_oof], axis=1)  (253,4)
    For each outer fold f:
        w_f = argmin_w  rae(y[tr_f], P[tr_f] @ w)   s.t. w>=0, sum(w)=1
              (POOLED objective on fold-train; multi-start SLSQP)
        oof[val_f] = P[val_f] @ w_f                 (no leak)
    pooled = rae(y, oof)                            (ONE denominator over all 253)
    Repeat for 15 FRESH kf_seeds {1416..1430}; report across-seed POOLED mean.

Deploy te uses the mean-of-fold weights (averaged across all 75 folds) applied
to the anchor te matrix (513,4).

GATE (on 15-seed pooled mean):
    pooled < 0.4414 -> "BETTER"
    pooled < 0.4424 -> "MARGINAL"
    else            -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy           (253,)
    data/processed/_audit_unblind_y.npy             (253,)
    data/processed/nb3173_pred_oof.npy   data/processed/te_nb3173.npy
    data/processed/nb3174_pred_oof.npy   data/processed/te_nb3174.npy
    data/processed/nb3190_pred_oof.npy   data/processed/te_nb3190.npy
    data/processed/nb3200_pred_oof.npy   data/processed/te_nb3200.npy

Outputs:
    data/processed/nb3421_summary.json
    data/processed/nb3421_pred_oof.npy   (253,) float32 -- median-seed pooled OOF
    data/processed/te_nb3421.npy         (513,) float32 -- deploy te
    submissions/nb3421_pooled_slsqp.csv  (only on BETTER or MARGINAL)
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
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3421"

# -- Anchors (4 clip winners) --------------------------------------------------
ANCHOR_NAMES = ["nb3173", "nb3174", "nb3190", "nb3200"]
ANCHOR_OOF_PATHS = {n: DATA_PROCESSED / f"{n}_pred_oof.npy" for n in ANCHOR_NAMES}
ANCHOR_TE_PATHS = {n: DATA_PROCESSED / f"te_{n}.npy" for n in ANCHOR_NAMES}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1416, 1431))  # 15 FRESH seeds {1416..1430}

# -- SLSQP options -------------------------------------------------------------
N_SLSQP_STARTS = 12
DEGEN_MAX_W = 0.85

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4414
GATE_MARGINAL = 0.4424

# -- References (PRE-unblind ceiling cluster) ----------------------------------
REF = {
    "nb3200_oof_pooled": 0.4416,   # best single clip winner (oof pooled)
    "nb3190_oof_pooled": 0.4422,
    "nb3173_oof_pooled": 0.4423,
    "nb3174_oof_pooled": 0.4430,
    "nb3204_equal_weight_pooled_15": 0.4418,  # equal-weight 3-clip ens (cycle 267)
    "nb2171_primary1": 0.4682,     # incumbent post-hoc-blend PRIMARY-1
}


def _pooled_slsqp(
    P: np.ndarray,
    y: np.ndarray,
    n_starts: int = N_SLSQP_STARTS,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """argmin_w POOLED rae(y, P @ w) on the simplex (w>=0, sum(w)=1).

    The objective is the POOLED RAE over ALL rows of (y, P): a single L1
    numerator over |y - P@w| divided by a single denominator |y - mean(y)|.
    This is the LB-faithful single-denominator functional, NOT a mean of
    per-fold ratios. Multi-start (1 uniform + dirichlet randoms) to escape the
    non-smooth-ratio local optima of RAE.
    """
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w: np.ndarray) -> float:
        return float(rae(y, P @ w))  # POOLED: one denominator over all rows

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K

    starts = [np.full(K, 1.0 / K)]
    for _ in range(n_starts - 1):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w: np.ndarray | None = None
    best_r = np.inf
    for x0 in starts:
        try:
            res = minimize(
                loss, x0, method="SLSQP",
                bounds=bnds, constraints=cons,
                options={"maxiter": 400, "ftol": 1e-10},
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


def _run_one_seed(
    P: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Outer 5-fold scaffold CV; per fold fit POOLED-objective SLSQP on
    fold-train, apply to fold-val; aggregate as POOLED over all 253."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    K = P.shape[1]
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes: list[float] = []
    fold_weights: list[np.ndarray] = []
    fold_train_raes: list[float] = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        w, r_tr = _pooled_slsqp(
            P[tr_loc], y_unb[tr_loc],
            n_starts=N_SLSQP_STARTS, seed=kf_seed * 100 + fold_i,
        )
        fold_weights.append(w)
        fold_train_raes.append(float(r_tr))
        val_pred = P[va_loc] @ w
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
    if np.isnan(oof_blend).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_blend))  # OUTER pooled: one denom over 253
    pf_mean = float(np.mean(fold_val_raes))  # sidecar (per-fold-mean)
    W = np.stack(fold_weights, axis=0)  # (n_folds, K)
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": pf_mean,
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "per_fold_train_rae_mean": float(np.mean(fold_train_raes)),
        "fold_weights": W,
        "any_fold_degenerate": bool(W.max(axis=1).max() > DEGEN_MAX_W),
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- GLOBAL SLSQP minimizing POOLED RAE on 4 clip winners")
    print(f"          anchors  = {ANCHOR_NAMES}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate: pooled < {GATE_BETTER:.4f} BETTER, "
        f"< {GATE_MARGINAL:.4f} MARGINAL, else FAIL"
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

    # -- Load anchors ---------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load 4 clip-winner anchors (pred_oof + te)")
    print("-" * 78)
    P_list = []
    te_list = []
    indiv_rae = {}
    leak_eq = {}
    for name in ANCHOR_NAMES:
        oof = np.load(ANCHOR_OOF_PATHS[name]).astype(np.float64)
        te_a = np.load(ANCHOR_TE_PATHS[name]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{name} pred_oof shape {oof.shape} != ({n_unb},)")
        if te_a.shape != (n_test,):
            raise ValueError(f"{name} te shape {te_a.shape} != ({n_test},)")
        r = float(rae(y_unb, oof))
        leak = float(np.mean(np.isclose(oof, y_unb, atol=1e-6)))
        indiv_rae[name] = round(r, 4)
        leak_eq[name] = round(leak, 4)
        P_list.append(oof)
        te_list.append(te_a)
        print(
            f"   {name}: oof_RAE={r:.4f}  "
            f"oof mean={oof.mean():.3f} std={oof.std():.3f}  "
            f"leak_eq={leak:.2%}  "
            f"te mean={te_a.mean():.3f} std={te_a.std():.3f}"
        )
        if leak > 0.05:
            print(f"   WARN {name}: {leak:.1%} rows == truth -- possible leak")
    P = np.stack(P_list, axis=1)   # (253, 4)
    TE = np.stack(te_list, axis=1)  # (513, 4)

    # Cross-anchor correlation
    corr_mat = np.corrcoef(P.T)
    print("\n   cross-anchor Pearson corr (OOF):")
    for i, ni in enumerate(ANCHOR_NAMES):
        cells = "  ".join(
            f"{corr_mat[i, j]:.4f}" for j in range(len(ANCHOR_NAMES))
        )
        print(f"     {ni:>8}: {cells}")

    # Equal-weight pooled baseline (split-invariant; LB analog for the ens)
    w_eq = np.full(P.shape[1], 1.0 / P.shape[1])
    eq_oof = P @ w_eq
    eq_oof_rae = float(rae(y_unb, eq_oof))
    print(
        f"\n   equal-weight POOLED baseline = {eq_oof_rae:.4f} "
        f"(split-invariant; SLSQP must beat this to add value)"
    )

    # In-sample global-pooled SLSQP (no CV) -- diagnostic ceiling/optimism gauge
    w_global, r_global = _pooled_slsqp(P, y_unb, n_starts=24, seed=999)
    print(
        f"   in-sample global pooled SLSQP = {r_global:.4f}  "
        f"w={ {ANCHOR_NAMES[k]: round(float(w_global[k]),3) for k in range(len(ANCHOR_NAMES))} } "
        f"(optimistic; df>0, NOT the gate)"
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
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}  (POOLED objective + POOLED aggregate)"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes: list[float] = []
    pf_means: list[float] = []
    oof_stack: list[np.ndarray] = []
    all_fold_weights: list[np.ndarray] = []
    any_degen = False
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        pf_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_fold_weights.append(res["fold_weights"])  # (n_folds, K)
        any_degen = any_degen or res["any_fold_degenerate"]
        per_fold_w_round = [
            {ANCHOR_NAMES[k]: round(float(res["fold_weights"][f, k]), 3)
             for k in range(len(ANCHOR_NAMES))}
            for f in range(res["fold_weights"].shape[0])
        ]
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_train_rae_mean": round(res["per_fold_train_rae_mean"], 4),
            "per_fold_weights": per_fold_w_round,
            "any_fold_degenerate": res["any_fold_degenerate"],
        })
        seed_mean_w = res["fold_weights"].mean(axis=0)
        seed_mean_w = seed_mean_w / max(seed_mean_w.sum(), 1e-12)
        wmean_str = "  ".join(
            f"{ANCHOR_NAMES[k]}={seed_mean_w[k]:.2f}"
            for k in range(len(ANCHOR_NAMES))
        )
        print(
            f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"w_mean({wmean_str})  "
            f"degen={res['any_fold_degenerate']}  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    pf_arr = np.asarray(pf_means, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    # Mean-of-fold weights (across all n_seeds * n_folds = 75 folds)
    all_W = np.concatenate(all_fold_weights, axis=0)  # (75, 4)
    w_mean_of_folds = all_W.mean(axis=0)
    w_mean_of_folds = w_mean_of_folds / max(w_mean_of_folds.sum(), 1e-12)

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds, POOLED metric)")
    print("-" * 78)
    print(f"   pooled mean   = {mean_rae:.4f}")
    print(f"   pooled std    = {std_rae:.4f}")
    print(f"   pooled sem    = {sem:.4f}")
    print(f"   pooled 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled median = {median_rae:.4f}")
    print(f"   pooled min/max= [{arr.min():.4f}, {arr.max():.4f}]")
    print(
        f"   per-fold-mean = {pf_arr.mean():.4f} +/- "
        f"{(pf_arr.std(ddof=1) if n_s>1 else 0.0):.4f}  (sidecar only)"
    )
    print(f"\n   equal-weight POOLED baseline = {eq_oof_rae:.4f}")
    print(f"   delta vs equal-weight (pooled mean) = {mean_rae - eq_oof_rae:+.4f}")
    print(
        f"   delta vs best single anchor nb3200 ({REF['nb3200_oof_pooled']:.4f}) = "
        f"{mean_rae - REF['nb3200_oof_pooled']:+.4f}"
    )
    print(
        f"   delta vs incumbent nb2171 ({REF['nb2171_primary1']:.4f}) = "
        f"{mean_rae - REF['nb2171_primary1']:+.4f}"
    )

    print("\n   mean-of-folds weight (across 75 folds):")
    for k, n in enumerate(ANCHOR_NAMES):
        print(f"     {n:>8}: {w_mean_of_folds[k]:.4f}")

    # -- Deploy te ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("DEPLOY te (mean-of-folds weights applied to anchor te(513,4))")
    print("-" * 78)
    te_pred = (TE @ w_mean_of_folds).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(
        f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}  "
        f"(deploy-refit optimism, NOT LB-faithful)"
    )

    # Median-seed OOF for storage (pooled-sorted median)
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (pooled={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (on pooled mean)")
    print("-" * 78)
    w_dep = {n: round(float(w_mean_of_folds[k]), 3)
             for k, n in enumerate(ANCHOR_NAMES)}
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3421 {n_s}-seed POOLED mean {mean_rae:.4f} "
            f"beats BETTER gate {GATE_BETTER:.4f} ({mean_rae - GATE_BETTER:+.4f}) "
            f"and the equal-weight pooled baseline {eq_oof_rae:.4f} "
            f"({mean_rae - eq_oof_rae:+.4f}). Minimizing the POOLED (single-"
            f"denominator, LB-faithful) objective extracted a learnable blend "
            f"across the 4 clip winners despite >0.998 cross-correlation. "
            f"Mean-of-fold deploy weights = {w_dep}. Re-verify with deep-30 "
            f"before any PRIMARY-1 swap; +0.10 POST-unblind safety shift still "
            f"applies to any te-based LB projection."
        )
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL"
        ladder_action = (
            f"ALTERNATE/MARGINAL. nb3421 {n_s}-seed POOLED mean {mean_rae:.4f} "
            f"clears MARGINAL gate {GATE_MARGINAL:.4f} "
            f"({mean_rae - GATE_MARGINAL:+.4f}) but not BETTER {GATE_BETTER:.4f} "
            f"({mean_rae - GATE_BETTER:+.4f}). It sits in the ~0.442 PRE-unblind "
            f"ceiling band with the 4 anchors and the equal-weight pooled "
            f"baseline {eq_oof_rae:.4f} (delta {mean_rae - eq_oof_rae:+.4f}); the "
            f"pooled objective neither helps nor hurts beyond noise given >0.998 "
            f"anchor correlation. Hold as alternate; do not displace incumbent "
            f"nb2171 {REF['nb2171_primary1']:.4f}. Deploy weights = {w_dep}."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3421 {n_s}-seed POOLED mean {mean_rae:.4f} fails the "
            f"MARGINAL gate {GATE_MARGINAL:.4f} ({mean_rae - GATE_MARGINAL:+.4f}). "
            f"The 4 clip winners are too correlated (>0.998) for a pooled-"
            f"objective SLSQP to extract residual orthogonality; the cross-fit "
            f"blend collapses to/above the equal-weight pooled baseline "
            f"{eq_oof_rae:.4f} (delta {mean_rae - eq_oof_rae:+.4f}). Switching "
            f"the objective from per-fold-train-mean to pooled does NOT break "
            f"the substrate ceiling. Keep nb3200 {REF['nb3200_oof_pooled']:.4f} "
            f"on the ladder; close the SLSQP-on-clip-winners axis. Substrate "
            f"change (orthogonal anchor) remains the open lever."
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

    sub_csv = SUBMISSIONS / f"{TAG}_pooled_slsqp.csv"
    if verdict in ("BETTER", "MARGINAL"):
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
        "method": "global_slsqp_minimizing_pooled_rae_on_4_clip_winners",
        "paradigm": (
            "INNER fit minimizes POOLED rae(y_tr, P_tr@w) (single denom over "
            "fold-train); OUTER aggregate is POOLED rae over all 253 cross-fit "
            "preds (single rae() call) -- the LB-faithful estimand"
        ),
        "anchors": ANCHOR_NAMES,
        "anchor_oof_paths": {k: str(v) for k, v in ANCHOR_OOF_PATHS.items()},
        "anchor_te_paths": {k: str(v) for k, v in ANCHOR_TE_PATHS.items()},
        "anchor_pre_unblind": True,
        "indiv_oof_rae": indiv_rae,
        "leak_eq_truth_frac": leak_eq,
        "cross_anchor_corr_oof": [
            [round(float(corr_mat[i, j]), 4) for j in range(len(ANCHOR_NAMES))]
            for i in range(len(ANCHOR_NAMES))
        ],
        "equal_weight_pooled_rae": round(eq_oof_rae, 4),
        "in_sample_global_pooled_slsqp_rae": round(r_global, 4),
        "in_sample_global_slsqp_weights": {
            ANCHOR_NAMES[k]: round(float(w_global[k]), 4)
            for k in range(len(ANCHOR_NAMES))
        },
        "n_folds": N_FOLDS,
        "n_slsqp_starts": N_SLSQP_STARTS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_array": [round(float(v), 4) for v in pf_means],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "per_fold_mean_mean": round(float(pf_arr.mean()), 4),
        "per_fold_mean_std": round(
            float(pf_arr.std(ddof=1)) if n_s > 1 else 0.0, 4
        ),
        "any_fold_degenerate_across_seeds": bool(any_degen),
        "mean_of_fold_weights": {
            ANCHOR_NAMES[k]: round(float(w_mean_of_folds[k]), 4)
            for k in range(len(ANCHOR_NAMES))
        },
        "delta_vs_equal_weight_pooled": round(mean_rae - eq_oof_rae, 4),
        "delta_vs_best_single_nb3200": round(
            mean_rae - REF["nb3200_oof_pooled"], 4
        ),
        "delta_vs_nb2171_primary1": round(mean_rae - REF["nb2171_primary1"], 4),
        "ref": REF,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (
            str(sub_csv) if verdict in ("BETTER", "MARGINAL") else None
        ),
        "gate_better": GATE_BETTER,
        "gate_marginal": GATE_MARGINAL,
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
    print(f"   pooled mean ({n_s} seeds) = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                   = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs equal-weight    = {mean_rae - eq_oof_rae:+.4f}")
    print(f"   delta vs nb3200          = {mean_rae - REF['nb3200_oof_pooled']:+.4f}")
    print(f"   mean-of-fold weights     = {w_dep}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_equal_weight_pooled", "delta_vs_best_single_nb3200",
        "equal_weight_pooled_rae", "in_sample_global_pooled_slsqp_rae",
        "indiv_oof_rae", "mean_of_fold_weights",
        "any_fold_degenerate_across_seeds", "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
