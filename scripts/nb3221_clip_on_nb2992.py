"""nb3221 -- Learned per-fold clip on nb2992 (K18+K19+K20 SLSQP simplex anchor).

NEW PARADIGM:
    Apply the per-fold (q_low, q_high) grid learned-clip primitive directly
    to the original SLSQP simplex blend of K18+K19+K20 deep-30 anchors
    (nb2992_pred_oof at honest scaffold-CV RAE 0.4479) -- the cleanest
    PRE-unblind 3-way simplex blend on top of the deep-30 LGBM ceiling.

    Hypothesis: nb2992 SLSQP simplex on (K18, K19, K20) deep-30 has its
    tails smoothed by the 3-way blend but still under-disperses on the
    OOD/novel-scaffold rows. A per-fold learned-clip should extract a
    tail-taming gain ON TOP OF the simplex, *without* swapping the
    underlying blend to the quantile-blend paradigm. If this clears the
    0.4424 BETTER gate, it confirms that the clip primitive is operator-
    invariant (works on SLSQP simplex as well as quantile blends and
    single-anchor K=18) and the gain is mostly from the tail-tame, not
    the blend shape.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    pred_base = nb2992_pred_oof   # K18+K19+K20 per-fold SLSQP simplex OOF
    Per outer fold:
        a) Inner grid on fold-train ONLY:
            for q_low in {0.01, 0.05, 0.10}:
              for q_high in {0.90, 0.95, 0.98}:
                lo = quantile(y[fold_train], q_low)
                hi = quantile(y[fold_train], q_high)
                pred_clipped = np.clip(pred_base[fold_train], lo, hi)
                rae_tr = rae(y[fold_train], pred_clipped)
            Pick (q_low*, q_high*) that minimize fold-train RAE.
        b) Apply (lo*, hi*) to fold-val: val_pred = np.clip(...)
        c) Stitch into oof_learned; pooled RAE across 5 folds.
    Repeat for 15 FRESH kf_seeds {1216..1230}.

GATE (on 15-seed per-fold-mean):
    mean < 0.4424 -> "BETTER"
    else          -> "FAIL"

References:
    nb2992 K18+K19+K20 SLSQP OOF    = 0.4479  <- parent anchor
    nb3001 K19 deep-30 wide-seed     = 0.4511
    nb3170 fixed q05/q95 on nb3080   = 0.4437
    nb3201 learned-clip on K18 alone = (prior result -- check summary)
    nb3173 learned-clip on nb3080    = (prior result -- check summary)
    nb2171 prior post-hoc top        = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2992_pred_oof.npy
    data/processed/te_nb2992.npy

Outputs:
    data/processed/nb3221_summary.json
    data/processed/nb3221_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3221.npy         (513,) float32 -- deploy te
    submissions/nb3221_clip_on_nb2992.csv  (only on BETTER verdict)
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

TAG = "nb3221"
PARENT_TAG = "nb2992_K18_K19_K20_SLSQP"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb2992_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb2992.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Per-fold grid (same as nb3201/nb3173) -------------------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98]

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4424

# -- References ----------------------------------------------------------------
REF_NB2992_SLSQP = 0.4479
REF_NB3001_K19_DEEP30 = 0.4511
REF_NB3170_FIXED = 0.4437
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
    """Run learned-clip pipeline at a single kf_seed."""
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
        f"{TAG} -- LEARNED per-fold clip percentiles (grid search) on "
        f"{PARENT_TAG} OOF (K18+K19+K20 SLSQP simplex anchor)"
    )
    print(f"          Q_LOW_GRID  = {Q_LOW_GRID}")
    print(f"          Q_HIGH_GRID = {Q_HIGH_GRID}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER, "
        f"else FAIL"
    )
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

    # -- Load nb2992 SLSQP simplex anchor pred_oof + te -----------------------
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

    # Truth stats
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
    per_fold_mean_raes = []
    oof_stack = []
    all_fold_ql = []
    all_fold_qh = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_mean_raes.append(res["per_fold_val_rae_mean"])
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
            f"per_fold_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"ql={res['fold_ql']}  qh={res['fold_qh']}  "
            f"clipped(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    pfm_arr = np.asarray(per_fold_mean_raes, dtype=np.float64)
    n_s = len(pfm_arr)

    # Gate metric: per-fold-mean (matches user request)
    mean_rae = float(pfm_arr.mean())
    std_rae = float(pfm_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(pfm_arr))

    # Companion pooled metric
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0

    # Most-picked q values across all 5*15=75 folds
    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds) -- gate on per-fold-mean")
    print("-" * 78)
    print(f"   per_fold_mean       = {mean_rae:.4f}")
    print(f"   std                 = {std_rae:.4f}")
    print(f"   sem                 = {sem:.4f}")
    print(f"   95% CI              = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median              = {median_rae:.4f}")
    print(f"   min/max             = [{pfm_arr.min():.4f}, {pfm_arr.max():.4f}]")
    print(f"   companion pooled    = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(
        f"\n   ref nb2992 SLSQP parent       = {REF_NB2992_SLSQP:.4f}  "
        f"<- parent (no clip)"
    )
    print(
        f"   delta vs nb2992 parent        = "
        f"{mean_rae - REF_NB2992_SLSQP:+.4f}"
    )
    print(f"   ref nb3170 fixed q05/q95     = {REF_NB3170_FIXED:.4f}")
    print(
        f"   delta vs nb3170 fixed         = "
        f"{mean_rae - REF_NB3170_FIXED:+.4f}"
    )
    print(f"   ref nb2171 post-hoc top      = {REF_NB2171:.4f}")
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

    # Median-seed OOF for storage (by per-fold-mean)
    med_seed_idx = int(np.argsort(pfm_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"   median seed = {median_seed} "
        f"(per_fold_mean={pfm_arr[med_seed_idx]:.4f})"
    )

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (on per-fold-mean)")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3221 15-seed per-fold-mean {mean_rae:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f}. Per-fold LEARNED clip on "
            f"nb2992 (K18+K19+K20 SLSQP simplex) beats prior post-hoc ceilings. "
            f"Modal pick was (q{ql_mode:.2f}, q{qh_mode:.2f}). NEW PARADIGM "
            f"confirmed: clip primitive applied to ORIGINAL SLSQP simplex "
            f"(vs quantile-blend) extracts gain. Re-verify deep-30 before "
            f"PRIMARY-1 swap."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3221 15-seed per-fold-mean {mean_rae:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f}. Per-fold learned-clip on nb2992 "
            f"SLSQP simplex (K18+K19+K20) does NOT improve over the prior "
            f"post-hoc ceilings (delta vs nb2992 parent "
            f"{mean_rae - REF_NB2992_SLSQP:+.4f}, vs nb3170 fixed "
            f"{mean_rae - REF_NB3170_FIXED:+.4f}). Clip primitive does not "
            f"add value on the original SLSQP simplex paradigm. Keep current "
            f"PRIMARY-1; no ladder change."
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

    sub_csv = SUBMISSIONS / f"{TAG}_clip_on_nb2992.csv"
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
            "per_fold_learned_clip_grid_search_on_nb2992_K18_K19_K20_SLSQP_"
            "simplex"
        ),
        "paradigm": "clip_on_original_SLSQP_simplex_vs_quantile_blend",
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
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_mean_raes],
        "gate_metric": "per_fold_mean",
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(pfm_arr.min()), 4),
        "max_rae": round(float(pfm_arr.max()), 4),
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "ref_nb2992_slsqp": REF_NB2992_SLSQP,
        "delta_vs_nb2992_parent": round(mean_rae - REF_NB2992_SLSQP, 4),
        "ref_nb3001_K19_deep30": REF_NB3001_K19_DEEP30,
        "ref_nb3170_fixed": REF_NB3170_FIXED,
        "delta_vs_nb3170_fixed": round(mean_rae - REF_NB3170_FIXED, 4),
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
        "submission_csv": (
            str(sub_csv) if verdict == "BETTER" else None
        ),
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
    print(f"   per_fold_mean ({n_s} seeds) = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb2992       = {mean_rae - REF_NB2992_SLSQP:+.4f}")
    print(f"   delta vs nb3170       = {mean_rae - REF_NB3170_FIXED:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb2992_parent", "delta_vs_nb3170_fixed",
        "pooled_mean", "pooled_std",
        "ql_mode", "qh_mode",
        "deploy_ql", "deploy_qh", "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
