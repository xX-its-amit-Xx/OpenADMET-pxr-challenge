"""nb3391 -- Asymmetric clip: learn q_low and q_high on INDEPENDENT finer grids.

NEW PARADIGM (vs nb3200):
    nb3200 ran a JOINT inner grid search -- it scored every (q_low, q_high)
    cell of a 3x4 grid against fold-train RAE and picked the best *pair*. The
    two clip edges were therefore coupled through a single joint objective.

    nb3391 decouples them. The low edge and the high edge are optimized on
    SEPARATE, finer 1-D grids, each against its OWN one-sided objective on
    fold-train:
      * q_low  in {0.005, 0.01, 0.02, 0.05}    -> pick lo* that minimizes
        fold-train RAE of clip(pred, lo, +inf)        (high edge open)
      * q_high in {0.95, 0.97, 0.98, 0.99, 0.995} -> pick hi* that minimizes
        fold-train RAE of clip(pred, -inf, hi)        (low edge open)
    The deployed clip is clip(pred, lo*, hi*) with the two edges chosen
    independently. Different lo/hi aggressiveness, learned independently.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    For each outer fold:
        a) lo* = argmin_{ql in Q_LOW_GRID}  RAE(y_tr, clip(pred_tr, q(y_tr,ql), inf))
        b) hi* = argmin_{qh in Q_HIGH_GRID} RAE(y_tr, clip(pred_tr, -inf, q(y_tr,qh)))
        c) apply clip(pred_val, lo*, hi*)
    Pool 5 folds -> pooled_rae (also track per-fold val RAE mean).
    Repeat over 15 fresh kf_seeds {1216..1230}; aggregate per-fold-mean.

GATE (per task, on the per-fold-mean across seeds):
    per_fold_mean < 0.4423 -> "BETTER"
    else                   -> "FAIL"

References:
    nb3090 parent anchor 15-seed   = 0.4472
    nb3190 learned-clip 15-seed    = 0.4426
    nb3200 deep-30 verify (joint)  = see nb3200_summary.json (parent of gate)
    nb2171 prior post-hoc ceiling  = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy
    data/processed/te_nb3090.npy

Outputs:
    data/processed/nb3391_summary.json
    data/processed/nb3391_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3391.npy         (513,) float32 -- deploy te
    submissions/nb3391_asymmetric_clip.csv   (only on BETTER verdict)
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

TAG = "nb3391"
PARENT_TAG = "nb3090"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3090.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 fresh seeds {1216..1230}

# -- Independent finer grids (asymmetric, decoupled) ---------------------------
Q_LOW_GRID = [0.005, 0.01, 0.02, 0.05]
Q_HIGH_GRID = [0.95, 0.97, 0.98, 0.99, 0.995]

# -- Gate (per task) -----------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ----------------------------------------------------------------
REF_PARENT_NB3090 = 0.4472
REF_NB3190 = 0.4426
REF_NB2171 = 0.4682


def _pick_lo_independent(y_tr: np.ndarray, pred_tr: np.ndarray) -> tuple[float, float]:
    """Pick low clip edge on its OWN 1-D grid (high edge left OPEN = +inf).

    Returns (best_ql, best_lo).
    """
    best_rae = np.inf
    best_ql = Q_LOW_GRID[0]
    best_lo = float(np.quantile(y_tr, Q_LOW_GRID[0]))
    for ql in Q_LOW_GRID:
        lo = float(np.quantile(y_tr, ql))
        clipped = np.clip(pred_tr, lo, None)  # high edge open
        r = float(rae(y_tr, clipped))
        if r < best_rae:
            best_rae = r
            best_ql = ql
            best_lo = lo
    return best_ql, best_lo


def _pick_hi_independent(y_tr: np.ndarray, pred_tr: np.ndarray) -> tuple[float, float]:
    """Pick high clip edge on its OWN 1-D grid (low edge left OPEN = -inf).

    Returns (best_qh, best_hi).
    """
    best_rae = np.inf
    best_qh = Q_HIGH_GRID[-1]
    best_hi = float(np.quantile(y_tr, Q_HIGH_GRID[-1]))
    for qh in Q_HIGH_GRID:
        hi = float(np.quantile(y_tr, qh))
        clipped = np.clip(pred_tr, None, hi)  # low edge open
        r = float(rae(y_tr, clipped))
        if r < best_rae:
            best_rae = r
            best_qh = qh
            best_hi = hi
    return best_qh, best_hi


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run independent-edge asymmetric-clip pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_ql, fold_qh = [], []
    fold_lo, fold_hi = [], []
    fold_clipped_lo, fold_clipped_hi = [], []
    for tr_loc, va_loc in splits:
        y_tr = y_unb[tr_loc]
        pred_tr = pred_base[tr_loc]
        # INDEPENDENT axes: each edge picked on its own 1-D grid + objective
        ql, lo = _pick_lo_independent(y_tr, pred_tr)
        qh, hi = _pick_hi_independent(y_tr, pred_tr)
        if hi <= lo:
            # degenerate guard: fall back to full data quantiles, keep order
            lo = float(np.quantile(y_tr, ql))
            hi = float(np.quantile(y_tr, qh))
            if hi <= lo:
                hi = lo + 1e-6
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)
        val_pred = pred_base[va_loc]
        fold_clipped_lo.append(int(np.sum(val_pred < lo)))
        fold_clipped_hi.append(int(np.sum(val_pred > hi)))
        clipped = np.clip(val_pred, lo, hi)
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
    print(f"{TAG} -- asymmetric clip: INDEPENDENT finer grids on {PARENT_TAG}")
    print(f"          Q_LOW_GRID  = {Q_LOW_GRID}")
    print(f"          Q_HIGH_GRID = {Q_HIGH_GRID}")
    print(f"          (low edge picked vs clip(.,lo,inf); "
          f"high edge picked vs clip(.,-inf,hi); decoupled)")
    print(f"          kf_seeds = {len(KF_SEEDS)} fresh "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          gate: per_fold_mean < {GATE_BETTER:.4f} -> BETTER, else FAIL")
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

    # -- Load nb3090 anchor pred_oof + te -------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {PARENT_TAG} pred_oof + te")
    print("-" * 78)
    pred_base = np.load(PRED_OOF_PATH).astype(np.float64)
    te_base = np.load(TE_PATH).astype(np.float64)
    if pred_base.shape != (n_unb,):
        raise ValueError(
            f"{PARENT_TAG} pred_oof shape {pred_base.shape} != ({n_unb},)"
        )
    if te_base.shape != (n_test,):
        raise ValueError(
            f"{PARENT_TAG} te shape {te_base.shape} != ({n_test},)"
        )
    full_oof_rae = float(rae(y_unb, pred_base))
    print(
        f"   pred_base: oof_RAE={full_oof_rae:.4f}  "
        f"mean={pred_base.mean():.3f}  std={pred_base.std():.3f}  "
        f"min={pred_base.min():.3f}  max={pred_base.max():.3f}"
    )
    print(
        f"   te_base:   mean={te_base.mean():.3f}  std={te_base.std():.3f}  "
        f"min={te_base.min():.3f}  max={te_base.max():.3f}"
    )

    # Leak sanity on parent
    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN parent: {leak_eq:.1%} rows == truth -- possible leak")

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
    print(f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    oof_stack = []
    all_fold_ql, all_fold_qh = [], []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
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
            "fold_ql": [round(v, 3) for v in res["fold_ql"]],
            "fold_qh": [round(v, 3) for v in res["fold_qh"]],
            "fold_lo_mean": round(res["fold_lo_mean"], 4),
            "fold_hi_mean": round(res["fold_hi_mean"], 4),
            "n_clipped_lo": res["n_clipped_lo"],
            "n_clipped_hi": res["n_clipped_hi"],
        })
        print(
            f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
            f"foldmean={res['per_fold_val_rae_mean']:.4f}  "
            f"ql={res['fold_ql']}  qh={res['fold_qh']}  "
            f"clip(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    foldmean_arr = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(pooled_arr)

    # GATE metric per task = per-fold-mean across seeds
    foldmean_mean = float(foldmean_arr.mean())
    foldmean_std = float(foldmean_arr.std(ddof=1)) if n_s > 1 else 0.0

    # pooled for reference
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = foldmean_std / np.sqrt(n_s) if n_s > 1 else 0.0
    # df=14, two-sided 95% t_mult = 2.1448
    t_mult = 2.1448
    ci_low = foldmean_mean - t_mult * sem
    ci_high = foldmean_mean + t_mult * sem
    median_foldmean = float(np.median(foldmean_arr))

    # Most-picked q values across all 5*15=75 folds
    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   per_fold_mean (GATE metric) = {foldmean_mean:.4f} +/- {foldmean_std:.4f}")
    print(f"   95% CI (df=14)              = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   per_fold_mean median        = {median_foldmean:.4f}")
    print(f"   per_fold_mean min/max       = [{foldmean_arr.min():.4f}, {foldmean_arr.max():.4f}]")
    print(f"   pooled_rae (reference)      = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"\n   ref nb3090 parent 15-seed   = {REF_PARENT_NB3090:.4f}")
    print(f"   delta vs nb3090             = {foldmean_mean - REF_PARENT_NB3090:+.4f}")
    print(f"   ref nb3190 learned-clip     = {REF_NB3190:.4f}")
    print(f"   delta vs nb3190             = {foldmean_mean - REF_NB3190:+.4f}")
    print(f"\n   ql_distribution (75 folds)  = {dict(ql_counter)}")
    print(f"   qh_distribution (75 folds)  = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")

    # -- Deploy: pick edges independently on FULL 253 -------------------------
    deploy_ql, deploy_lo = _pick_lo_independent(y_unb, pred_base)
    deploy_qh, deploy_hi = _pick_hi_independent(y_unb, pred_base)
    if deploy_hi <= deploy_lo:
        deploy_hi = deploy_lo + 1e-6
    te_pred = np.clip(te_base, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_base < deploy_lo))
    n_te_hi = int(np.sum(te_base > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy independent pick: lo=(q{deploy_ql:.3f}->{deploy_lo:.3f})  "
        f"hi=(q{deploy_qh:.3f}->{deploy_hi:.3f})"
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

    # Median-seed OOF for storage (ranked by per-fold-mean, the gate metric)
    med_seed_idx = int(np.argsort(foldmean_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (foldmean={foldmean_arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if foldmean_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE candidate. nb3391 asymmetric-clip (INDEPENDENT finer grids, "
            f"low edge q{ql_mode} / high edge q{qh_mode} modal) per-fold-mean "
            f"{foldmean_mean:.4f} +/- {foldmean_std:.4f} (15 seeds) beats gate "
            f"{GATE_BETTER:.4f} by {foldmean_mean - GATE_BETTER:+.4f}. Decoupling "
            f"the two clip edges onto separate 1-D objectives improves over the "
            f"joint-grid clip (nb3200 lineage) and the nb3090 anchor "
            f"({REF_PARENT_NB3090:.4f}). 15-seed dispersion only -- run deep-30 "
            f"re-verify before PRIMARY-1 promotion per cycle-160 rule."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3391 asymmetric-clip (INDEPENDENT finer grids) "
            f"per-fold-mean {foldmean_mean:.4f} +/- {foldmean_std:.4f} (15 seeds) "
            f"does not beat gate {GATE_BETTER:.4f} ({foldmean_mean - GATE_BETTER:+.4f}). "
            f"Learning q_low and q_high on independent finer grids does not break "
            f"the clip ceiling on the nb3090 anchor. Modal pick "
            f"(low q{ql_mode}, high q{qh_mode}). Keep prior PRIMARY-1; clip-edge "
            f"decoupling adds no honest gain over joint-grid clip at n=253."
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

    sub_csv = SUBMISSIONS / f"{TAG}_asymmetric_clip.csv"
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
        "method": "asymmetric_clip_independent_finer_grids_on_nb3090",
        "paradigm_note": (
            "low edge and high edge optimized on SEPARATE 1-D grids with "
            "one-sided objectives (clip(.,lo,inf) for low; clip(.,-inf,hi) for "
            "high); decoupled vs nb3200 joint grid"
        ),
        "anchor_pred_oof_path": str(PRED_OOF_PATH),
        "anchor_te_path": str(TE_PATH),
        "anchor_pre_unblind": True,
        "parent_full_oof_rae": round(full_oof_rae, 4),
        "parent_leak_eq_truth_frac": round(leak_eq, 4),
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
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "gate_metric": "per_fold_mean",
        "per_fold_mean": round(foldmean_mean, 4),
        "per_fold_mean_std": round(foldmean_std, 4),
        "per_fold_mean_median": round(median_foldmean, 4),
        "per_fold_mean_min": round(float(foldmean_arr.min()), 4),
        "per_fold_mean_max": round(float(foldmean_arr.max()), 4),
        "sem": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "pooled_mean_rae": round(pooled_mean, 4),
        "pooled_std_rae": round(pooled_std, 4),
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "ref_parent_nb3090": REF_PARENT_NB3090,
        "delta_vs_parent": round(foldmean_mean - REF_PARENT_NB3090, 4),
        "ref_nb3190": REF_NB3190,
        "delta_vs_nb3190": round(foldmean_mean - REF_NB3190, 4),
        "ref_nb2171": REF_NB2171,
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
    print(f"   per_fold_mean ({n_s} seeds) = {foldmean_mean:.4f} +/- {foldmean_std:.4f}")
    print(f"   95% CI                    = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled (reference)        = {pooled_mean:.4f}")
    print(f"   delta vs nb3090           = {foldmean_mean - REF_PARENT_NB3090:+.4f}")
    print(f"   modal pick (lo,hi)        = (q{ql_mode}, q{qh_mode})")
    print(f"   verdict                   = {verdict}")
    print(f"   wall                      = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean", "per_fold_mean_std", "ci95_low", "ci95_high",
        "pooled_mean_rae",
        "delta_vs_parent", "delta_vs_nb3190",
        "ql_mode", "qh_mode",
        "deploy_ql", "deploy_qh", "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
