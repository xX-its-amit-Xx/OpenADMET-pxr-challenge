"""nb3222 -- Learned per-fold clip percentiles ON TOP OF nb2983.

NEW PARADIGM: clip on a stretch-calibrated anchor.

CONTEXT:
    nb2983 is per-fold SLSQP simplex on K18/K20/K24/K28 composed with per-fold
    golden-section rank-stretch. Its full-OOF RAE = 0.4552, per-fold val
    mean = 0.4593. nb2983 is PRE-clean (4-K LGBM anchors, no nb730 / no nb562
    POST-unblind contamination chain). Stretch is already applied per fold,
    so the OOF predictions live on a variance-decompressed scale. A learned
    outer clip on top can re-tighten tail mass at the stretched envelope
    without re-touching the inner simplex+stretch operator.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    pred_base = nb2983_pred_oof  (253,) -- simplex+stretch OOF
    Per outer fold:
        a) Inner search on fold-train ONLY:
            for q_low in {0.01, 0.05, 0.10}:
              for q_high in {0.95, 0.98, 0.99}:
                lo = quantile(y[fold_train], q_low)
                hi = quantile(y[fold_train], q_high)
                pred_clipped = np.clip(pred_base[fold_train], lo, hi)
                rae_tr = rae(y[fold_train], pred_clipped)
            Pick (q_low*, q_high*) minimizing fold-train RAE.
        b) Apply (lo*, hi*) to fold-val: val_pred = np.clip(...)
        c) Stitch into oof_learned; pooled + per-fold-mean RAE across 5 folds.
    Repeat for 15 FRESH kf_seeds {1216..1230}.

GATE (on per-fold-mean across 15 seeds):
    pf_mean < 0.4424 -> "BETTER"
    else            -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2983_pred_oof.npy   (253,) float32 -- simplex+stretch OOF
    data/processed/te_nb2983.npy         (513,) float32 -- simplex+stretch te

Outputs:
    data/processed/nb3222_summary.json
    data/processed/nb3222_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3222.npy         (513,) float32 -- deploy te (full-data clip)
    submissions/nb3222_clip_on_nb2983.csv  (only on BETTER)
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

TAG = "nb3222"
PARENT_TAG = "nb2983"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / f"{PARENT_TAG}_pred_oof.npy"
TE_PATH = DATA_PROCESSED / f"te_{PARENT_TAG}.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Per-fold grid -------------------------------------------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.95, 0.98, 0.99]

# -- Gate (per task: per-fold-mean) --------------------------------------------
GATE_BETTER = 0.4424

# -- References ----------------------------------------------------------------
REF_PARENT_NB2983_FULL_OOF = 0.4552
REF_PARENT_NB2983_PF_MEAN = 0.4593
REF_NB2171 = 0.4682


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
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_clip = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_ql = []
    fold_qh = []
    fold_lo = []
    fold_hi = []
    fold_clipped_lo = []
    fold_clipped_hi = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], pred_base[tr_loc])
        fold_ql.append(ql)
        fold_qh.append(qh)
        fold_lo.append(lo)
        fold_hi.append(hi)
        val_pred = pred_base[va_loc]
        n_lo = int(np.sum(val_pred < lo))
        n_hi = int(np.sum(val_pred > hi))
        fold_clipped_lo.append(n_lo)
        fold_clipped_hi.append(n_hi)
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
    print(
        f"{TAG} -- LEARNED per-fold clip percentiles ON TOP OF {PARENT_TAG} "
        f"(per-fold simplex + stretch anchor)"
    )
    print(f"          Q_LOW_GRID  = {Q_LOW_GRID}")
    print(f"          Q_HIGH_GRID = {Q_HIGH_GRID}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"          gate (pf_mean): < {GATE_BETTER:.4f} -> BETTER, else FAIL")
    print("=" * 78)

    # -- Verify anchor files exist (SKIP if missing) --------------------------
    if not PRED_OOF_PATH.exists():
        skip = {
            "tag": TAG,
            "parent_tag": PARENT_TAG,
            "verdict": "SKIP",
            "reason": f"missing anchor pred_oof: {PRED_OOF_PATH}",
        }
        out_path = DATA_PROCESSED / f"{TAG}_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(skip, f, indent=2)
        print(f"[skip] {skip['reason']}")
        return skip
    if not TE_PATH.exists():
        skip = {
            "tag": TAG,
            "parent_tag": PARENT_TAG,
            "verdict": "SKIP",
            "reason": f"missing anchor te: {TE_PATH}",
        }
        out_path = DATA_PROCESSED / f"{TAG}_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(skip, f, indent=2)
        print(f"[skip] {skip['reason']}")
        return skip

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

    # -- Load nb2983 anchor pred_oof + te -------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {PARENT_TAG} pred_oof + te (per-fold simplex+stretch)")
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

    leak_eq = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN parent: {leak_eq:.1%} rows == truth -- possible leak")

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
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"ql={res['fold_ql']}  qh={res['fold_qh']}  "
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

    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled mean    = {mean_rae:.4f}  std={std_rae:.4f}  sem={sem:.4f}")
    print(f"   pooled 95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled median  = {median_rae:.4f}  min/max=[{arr.min():.4f}, {arr.max():.4f}]")
    print(
        f"   per_fold_mean RAE = {pf_mean:.4f} +/- {pf_std:.4f} "
        f"(sem={pf_sem:.4f})  [PRIMARY GATE METRIC]"
    )
    print(
        f"\n   ref {PARENT_TAG} full_oof_rae = {REF_PARENT_NB2983_FULL_OOF:.4f}"
    )
    print(
        f"   ref {PARENT_TAG} pf_mean      = {REF_PARENT_NB2983_PF_MEAN:.4f}"
    )
    print(f"   delta pf_mean vs parent      = {pf_mean - REF_PARENT_NB2983_PF_MEAN:+.4f}")
    print(f"   ref nb2171 PRIMARY-1         = {REF_NB2171:.4f}")
    print(f"   gain vs nb2171 (pf_mean)     = {REF_NB2171 - pf_mean:+.4f}")
    print(f"\n   ql_distribution (75 folds) = {dict(ql_counter)}")
    print(f"   qh_distribution (75 folds) = {dict(qh_counter)}")
    print(f"   ql_mode = {ql_mode}  qh_mode = {qh_mode}")

    # -- Deploy: pick (q_low, q_high) on FULL 253 by same inner search --------
    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(y_unb, pred_base)
    te_pred = np.clip(te_base, deploy_lo, deploy_hi).astype(np.float32)
    n_te_lo = int(np.sum(te_base < deploy_lo))
    n_te_hi = int(np.sum(te_base > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
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

    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate (per-fold-mean) -------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE  (metric = per-fold-mean across 15 seeds)")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3222 15-seed pf_mean {pf_mean:.4f} clears "
            f"BETTER gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"Learned per-fold clip on nb2983 (per-fold simplex+stretch) "
            f"compounds beyond parent pf_mean {REF_PARENT_NB2983_PF_MEAN:.4f} "
            f"-> {pf_mean:.4f}. Clip-on-stretch is a new operator stack: "
            f"stretch already decompresses variance, outer clip then re-tightens "
            f"only at the tail percentiles where stretch over-extends. Modal "
            f"pick (q{ql_mode}, q{qh_mode}). anchor_pre_unblind=True (parent "
            f"nb2983 is K18/K20/K24/K28 LGBM, no nb730 / no nb562 chain). "
            f"Re-verify with deep-30 before PRIMARY-1 swap."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3222 15-seed pf_mean {pf_mean:.4f} fails BETTER gate "
            f"{GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). Delta vs parent "
            f"nb2983 pf_mean = {pf_mean - REF_PARENT_NB2983_PF_MEAN:+.4f}. "
            f"Outer learned clip on a per-fold simplex+stretch anchor does not "
            f"compound; per-fold stretch already absorbed the available tail "
            f"calibration capacity, leaving no headroom for an outer clip pass. "
            f"Closes the clip-on-stretch-calibrated-anchor axis; keep nb2983 "
            f"as-is on the ladder."
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

    sub_csv = SUBMISSIONS / f"{TAG}_clip_on_nb2983.csv"
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
            "per_fold_learned_clip_grid_search_on_nb2983_per_fold_simplex_stretch_anchor"
        ),
        "paradigm": "clip_on_stretch_calibrated_anchor",
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
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "ref_parent_full_oof_rae": REF_PARENT_NB2983_FULL_OOF,
        "ref_parent_pf_mean": REF_PARENT_NB2983_PF_MEAN,
        "delta_pf_vs_parent": round(pf_mean - REF_PARENT_NB2983_PF_MEAN, 4),
        "ref_nb2171": REF_NB2171,
        "gain_pf_vs_nb2171": round(REF_NB2171 - pf_mean, 4),
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
        "gate_metric": "per_fold_mean_rae_mean",
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
    print(f"   pf_mean ({n_s} seeds)   = {pf_mean:.4f} +/- {pf_std:.4f}  [GATE]")
    print(f"   pooled mean             = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI pooled           = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta pf vs nb2983      = {pf_mean - REF_PARENT_NB2983_PF_MEAN:+.4f}")
    print(f"   gain pf vs nb2171       = {REF_NB2171 - pf_mean:+.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   wall                    = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean_rae_mean", "per_fold_mean_rae_std",
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_pf_vs_parent",
        "ql_mode", "qh_mode",
        "deploy_ql", "deploy_qh", "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
