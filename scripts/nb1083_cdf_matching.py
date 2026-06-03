"""nb1083 -- Empirical CDF matching (rank-preserving quantile transport) on
te_nb1014.

Motivation
----------
nb562's rank-stretch (`mu + s*(p - mu)`) is a *scalar-multiplier* variance
decompression: it stretches the predicted distribution around its mean,
preserving rank but forcing a single-parameter linear shape. nb1082's Beta
calibration is a 3-parameter smooth monotone map.

CDF matching is the *non-parametric limit* of both: it replaces the predicted
empirical CDF with the truth empirical CDF, point-by-point, via

        p  ->  rank_in_train_pred  ->  CDF_train_pred(p)  ->
                                        CDF_train_truth^{-1}(.)  ->  y_hat

This is exactly the rank-preserving transform the rank-stretch family
approximates with a single slope. If train-pred and train-truth are both
unimodal with the same support, CDF matching reduces to a slope very close
to nb562's s. Where they differ in *shape* (skew, tails, central plateau),
CDF matching captures that for free.

Why this might work on nb1014
-----------------------------
The 253-unblind failure tail is dominated by novel-scaffold variance
compression (pred_std ~ 0.75 vs truth_std ~ 1.03). nb562 corrects the std
ratio. CDF matching additionally corrects skew/kurtosis differences if
they exist -- the cost is zero degrees of freedom (purely empirical), so
the only risk is fold-to-fold support mismatch.

Protocol
--------
Honest 5-fold cross-fit on 253 unblind:
    For each fold k:
        1. Sort train preds -> empirical CDF over train-fold preds.
        2. Sort train truths -> empirical CDF over train-fold truths.
        3. For each held-out p:
              - u = CDF_train_pred(p)        (via np.interp on sorted preds)
              - y_hat = quantile_train_truth(u)  (via np.interp on sorted ys)
        4. Held-out p outside train-pred support is clamped to the
           [min,max] truth quantile (no extrapolation -- forces honest
           failure on out-of-support points).
    Pooled OOF RAE = headline.

Seeds: 5 KFold seeds -> per-seed pooled RAE + median-bagged OOF (mirrors
nb1070 / nb1082 protocol so the comparison is apples-to-apples).

Hypothesis
----------
Explicit rank-preserving transform should replace nb562's scalar variance
decompression directly. If CDF matching beats nb562's 0.5065 / nb1070's
0.5771 / nb1082's beta cal, the win is from shape (skew/tail) correction
on top of scale. If it ties or loses, scalar stretch was capturing the
full available signal at n=253.

Outputs
-------
    data/processed/te_nb1083.npy           (513 deploy)
    data/processed/nb1083_summary.json
    submissions/nb1083_cdf_matching.csv
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
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1083"
ANCHOR_STEM = "nb1014"
SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

# Reference numbers for verdicts.
NB1070_BAG_MEDIAN_REF = 0.5771   # nb1070 per-quantile median bag
NB562_REF = 0.5065               # nb562 rank-stretch honest cross-fit
NB1082_REF = None                # filled if nb1082_summary.json is on disk


# =====================================================================
# CDF matching primitives
# =====================================================================
def fit_cdf_map(p_train: np.ndarray, y_train: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort train preds and truths; return the two sorted arrays plus
    matching rank-fractions for np.interp.

    Returns
    -------
    p_sorted : (n,) ascending sorted train preds
    u_p      : (n,) rank fractions for p_sorted: (i + 0.5)/n  (mid-rank
               convention -- avoids 0 and 1 endpoints).
    y_sorted : (n,) ascending sorted train truths
    u_y      : (n,) rank fractions for y_sorted: (i + 0.5)/n
    """
    n = len(p_train)
    p_sorted = np.sort(p_train.astype(np.float64))
    y_sorted = np.sort(y_train.astype(np.float64))
    u = (np.arange(n) + 0.5) / n
    return p_sorted, u, y_sorted, u


def apply_cdf_map(p_query: np.ndarray,
                  p_sorted: np.ndarray, u_p: np.ndarray,
                  y_sorted: np.ndarray, u_y: np.ndarray
                  ) -> np.ndarray:
    """Map p_query through the empirical CDF chain.

    Step 1: u = CDF_train_pred(p_query)  via np.interp on (p_sorted, u_p).
            np.interp clamps to [u_p[0], u_p[-1]] for OOB inputs, so query
            below p_sorted[0] -> u_p[0]  (smallest train-truth quantile),
            and query above p_sorted[-1] -> u_p[-1] (largest train-truth
            quantile). This is the honest no-extrapolation behavior.

    Step 2: y_hat = quantile_train_truth(u)  via np.interp on (u_y, y_sorted).
    """
    u = np.interp(p_query.astype(np.float64), p_sorted, u_p)
    y_hat = np.interp(u, u_y, y_sorted)
    return y_hat


# =====================================================================
# Cross-fit driver
# =====================================================================
def one_seed(p_unb: np.ndarray, y_unb: np.ndarray, seed: int
             ) -> tuple[float, np.ndarray, list[dict]]:
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        p_sorted, u_p, y_sorted, u_y = fit_cdf_map(
            p_unb[tr_loc], y_unb[tr_loc]
        )
        oof[va_loc] = apply_cdf_map(p_unb[va_loc], p_sorted, u_p,
                                    y_sorted, u_y)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        rae_tr = float(rae(
            y_unb[tr_loc],
            apply_cdf_map(p_unb[tr_loc], p_sorted, u_p, y_sorted, u_y),
        ))
        # How many val points fell OUTSIDE train-pred support
        # (clamped -> identity-of-extreme).
        n_below = int(np.sum(p_unb[va_loc] < p_sorted[0]))
        n_above = int(np.sum(p_unb[va_loc] > p_sorted[-1]))
        fold_records.append({
            "fold": k,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "p_tr_min": float(p_sorted[0]),
            "p_tr_max": float(p_sorted[-1]),
            "y_tr_min": float(y_sorted[0]),
            "y_tr_max": float(y_sorted[-1]),
            "n_va_below_support": n_below,
            "n_va_above_support": n_above,
            "train_rae": rae_tr,
            "val_rae": rae_va,
        })
    pooled = float(rae(y_unb, oof))
    return pooled, oof, fold_records


# =====================================================================
# Main
# =====================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Empirical CDF matching on te_{ANCHOR_STEM}")
    print(f"      seeds = {SEEDS}  N_FOLDS = {N_FOLDS}")
    print("=" * 78)

    # ------ Optionally pull nb1082 ref number -----------------------------
    global NB1082_REF
    nb1082_path = DATA_PROCESSED / "nb1082_summary.json"
    if nb1082_path.exists():
        try:
            with open(nb1082_path) as f:
                NB1082_REF = float(json.load(f).get("bag_median_rae"))
        except Exception:
            NB1082_REF = None

    te = load_test()
    te_names = te["name"].values
    preds_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR_STEM}.npy").astype(
        np.float64)
    print(f"[load] te_{ANCHOR_STEM}.npy shape = {preds_513.shape}  "
          f"mean={preds_513.mean():.3f} std={preds_513.std():.3f}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513[unb_idx]
    print(f"[load] p_unb shape = {p_unb.shape}  y shape = {y_unb.shape}")
    print(f"[load] p_unb std = {p_unb.std():.3f}  "
          f"y_unb std = {y_unb.std():.3f}  "
          f"std ratio (y/p) = {(y_unb.std() / max(p_unb.std(), 1e-9)):.3f}")

    raw_in_rae = float(rae(y_unb, p_unb))
    print(f"\n[anchor] raw te_{ANCHOR_STEM} in_RAE(253) = {raw_in_rae:.4f}  "
          "(pre-calibration baseline)")

    # -------- Cross-fit per seed --------
    print("\n" + "-" * 78)
    print(f"Honest {N_FOLDS}-fold cross-fit over {len(SEEDS)} seeds")
    print("-" * 78)
    oof_stack = np.zeros((len(SEEDS), len(y_unb)), dtype=np.float64)
    per_seed_rae: list[float] = []
    seed_records: list[dict] = []
    total_oob_below = 0
    total_oob_above = 0
    for i, seed in enumerate(SEEDS):
        pooled, oof, folds = one_seed(p_unb, y_unb, seed)
        oof_stack[i] = oof
        per_seed_rae.append(pooled)
        seed_records.append({"seed": seed, "pooled_rae": pooled,
                             "folds": folds})
        below = sum(f["n_va_below_support"] for f in folds)
        above = sum(f["n_va_above_support"] for f in folds)
        total_oob_below += below
        total_oob_above += above
        print(f"   seed {seed:>3d}: pooled_RAE = {pooled:.4f}   "
              f"OOB below={below:3d}  above={above:3d}")

    per_seed_arr = np.array(per_seed_rae)
    seed_mean = float(per_seed_arr.mean())
    seed_std = float(per_seed_arr.std())
    seed_min = float(per_seed_arr.min())
    seed_max = float(per_seed_arr.max())
    print(f"\n[CV] per-seed RAE  mean={seed_mean:.4f}  std={seed_std:.4f}  "
          f"min={seed_min:.4f}  max={seed_max:.4f}")
    print(f"[CV] total OOB clamped: below={total_oob_below}  "
          f"above={total_oob_above}  "
          f"(across {len(SEEDS) * len(y_unb)} val rows)")

    bagged_median_oof = np.median(oof_stack, axis=0)
    bagged_mean_oof = oof_stack.mean(axis=0)
    bag_median_rae = float(rae(y_unb, bagged_median_oof))
    bag_mean_rae = float(rae(y_unb, bagged_mean_oof))
    print(f"[CV] MEDIAN-bag OOF RAE = {bag_median_rae:.4f}  (HEADLINE)")
    print(f"[CV] MEAN-bag OOF RAE   = {bag_mean_rae:.4f}")

    # -------- Deploy: fit ONCE on all 253; apply to 513 -----------------
    print("\n" + "-" * 78)
    print("DEPLOY  (fit on all 253 -- CDF matching is deterministic,")
    print("         one fit serves all seeds)")
    print("-" * 78)
    p_sorted, u_p, y_sorted, u_y = fit_cdf_map(p_unb, y_unb)
    deploy_253 = apply_cdf_map(p_unb, p_sorted, u_p, y_sorted, u_y)
    in_rae_deploy = float(rae(y_unb, deploy_253))
    deploy_513 = apply_cdf_map(preds_513, p_sorted, u_p, y_sorted, u_y
                               ).astype(np.float32)
    n_below_513 = int(np.sum(preds_513 < p_sorted[0]))
    n_above_513 = int(np.sum(preds_513 > p_sorted[-1]))
    print(f"   anchor train support: p in [{p_sorted[0]:.3f}, "
          f"{p_sorted[-1]:.3f}]   y in [{y_sorted[0]:.3f}, "
          f"{y_sorted[-1]:.3f}]")
    print(f"   in-sample RAE(253) = {in_rae_deploy:.4f}  (lower bound)")
    print(f"   513 OOB clamped: below={n_below_513}  above={n_above_513}")
    print(f"   deploy_513 mean = {deploy_513.mean():.3f}  "
          f"std = {deploy_513.std():.3f}  "
          f"min = {deploy_513.min():.3f}  max = {deploy_513.max():.3f}")

    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_cdf_matching.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    # -------- Verdicts ---------------------------------------------------
    delta_vs_nb1070 = bag_median_rae - NB1070_BAG_MEDIAN_REF
    delta_vs_raw = bag_median_rae - raw_in_rae
    delta_vs_nb562 = bag_median_rae - NB562_REF
    beats_nb1070 = bool(bag_median_rae < NB1070_BAG_MEDIAN_REF)
    beats_nb562 = bool(bag_median_rae < NB562_REF)
    if delta_vs_nb1070 < -0.003:
        verdict = "BEATS_NB1070"
    elif abs(delta_vs_nb1070) <= 0.003:
        verdict = "TIES_NB1070"
    else:
        verdict = "WORSE_THAN_NB1070"

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   raw te_{ANCHOR_STEM} in_RAE(253)   = {raw_in_rae:.4f}")
    print(f"   per-seed mean cross-fit RAE   = {seed_mean:.4f}  "
          f"std {seed_std:.4f}")
    print(f"   MEDIAN-bagged OOF RAE         = {bag_median_rae:.4f}  "
          "(HEADLINE)")
    print(f"   nb1070 per-q median bag ref   = {NB1070_BAG_MEDIAN_REF:.4f}  "
          f"delta = {delta_vs_nb1070:+.4f}")
    if NB1082_REF is not None:
        print(f"   nb1082 beta cal ref           = {NB1082_REF:.4f}  "
              f"delta = {bag_median_rae - NB1082_REF:+.4f}")
    print(f"   nb562 rank-stretch ref        = {NB562_REF:.4f}  "
          f"delta = {delta_vs_nb562:+.4f}")
    print(f"   vs raw anchor                 = {delta_vs_raw:+.4f}")
    print(f"   beats_nb1070                  = {beats_nb1070}")
    print(f"   beats_nb562                   = {beats_nb562}")
    print(f"   verdict                       = {verdict}")

    summary = {
        "tag": TAG,
        "method": "empirical_cdf_matching",
        "anchor": ANCHOR_STEM,
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "raw_anchor_in_rae_253": raw_in_rae,
        "raw_anchor_p_std": float(p_unb.std()),
        "y_unb_std": float(y_unb.std()),
        "std_ratio_y_over_p": float(y_unb.std() / max(p_unb.std(), 1e-9)),
        "per_seed_pooled_rae": per_seed_rae,
        "per_seed_rae_mean": seed_mean,
        "per_seed_rae_std": seed_std,
        "per_seed_rae_min": seed_min,
        "per_seed_rae_max": seed_max,
        "bag_median_rae": bag_median_rae,
        "bag_mean_rae": bag_mean_rae,
        "seed_records": seed_records,
        "total_oob_below_cv": total_oob_below,
        "total_oob_above_cv": total_oob_above,
        "deploy_p_tr_min": float(p_sorted[0]),
        "deploy_p_tr_max": float(p_sorted[-1]),
        "deploy_y_tr_min": float(y_sorted[0]),
        "deploy_y_tr_max": float(y_sorted[-1]),
        "deploy_in_rae_253": in_rae_deploy,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "deploy_te_min": float(deploy_513.min()),
        "deploy_te_max": float(deploy_513.max()),
        "deploy_513_oob_below": n_below_513,
        "deploy_513_oob_above": n_above_513,
        "nb1070_reference": NB1070_BAG_MEDIAN_REF,
        "nb1082_reference": NB1082_REF,
        "nb562_reference": NB562_REF,
        "delta_vs_nb1070": delta_vs_nb1070,
        "delta_vs_nb562": delta_vs_nb562,
        "delta_vs_raw_anchor": delta_vs_raw,
        "beats_nb1070": beats_nb1070,
        "beats_nb562": beats_nb562,
        "verdict": verdict,
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("raw_anchor_in_rae_253",
              "per_seed_pooled_rae", "per_seed_rae_mean",
              "bag_median_rae", "bag_mean_rae",
              "deploy_in_rae_253",
              "delta_vs_nb1070", "delta_vs_nb562",
              "beats_nb1070", "beats_nb562", "verdict",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
