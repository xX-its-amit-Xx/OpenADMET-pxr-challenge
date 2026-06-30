"""nb3200 -- DEEP-30 verification of nb3190 (learned per-fold clip on nb3090).

CONTEXT:
    nb3190 reported 15-seed mean 0.4426 +/- 0.0022 on fresh seeds {1171..1185},
    landing in MARGINAL band (gate_better=0.4422, gate_marginal=0.4437).

    Per cycle-160 deep-30 rule (feedback_cycle160_deep_verify_dispersion.md):
    5/15-seed std is severely under-dispersed vs 30-seed std (4-5x ratio),
    so PRIMARY decisions must use deep-30. nb3190's std=0.0022 at 15 seeds
    is suspect -- if true deep-30 std is ~0.0090 (4x), the 95% CI widens
    enough to question MARGINAL placement.

PROTOCOL (mirror of nb3190 exactly, only kf_seeds change):
    For each kf_seed in {1186..1215} (30 FRESH seeds, disjoint from nb3190):
        scaffold_kfold_indices(n_splits=5, seed=kf_seed)
        For each fold:
            a) Inner grid search on fold-train:
                ql in {0.01, 0.05, 0.10}, qh in {0.90, 0.95, 0.98, 0.99}
                Pick (ql*, qh*) minimizing fold-train pooled RAE
            b) Apply clip(pred_base[fold_val], lo*, hi*)
        Pool 5 folds -> pooled_rae
    Aggregate mean +/- std over 30 seeds.

GATE (per task):
    mean < 0.4426 -> "VERIFIED_PROMOTE_PRIMARY1"  (matches/beats 15-seed)
    mean < 0.4437 -> "VERIFIED_MARGINAL"          (still under MARGINAL band)
    shift > +0.005 -> "LUCKY_SEED_TRAP"
    else            -> "REJECT"

References:
    nb3190 15-seed mean = 0.4426 +/- 0.0022 (target to verify)
    nb3090 15-seed mean = 0.4472          (parent anchor)
    nb3173 ceiling      = 0.4437          (prior learned-clip ceiling)
    nb2171 prior PRIMARY = 0.4682         (post-hoc-blend ceiling on nb730 anchor)

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy
    data/processed/te_nb3090.npy

Outputs:
    data/processed/nb3200_summary.json
    data/processed/nb3200_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3200.npy         (513,) float32 -- deploy te
    submissions/nb3200_deep_verify_nb3190.csv  (only on VERIFIED_*)
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

TAG = "nb3200"
PARENT_TAG = "nb3090"
VERIFY_TAG = "nb3190"

# -- Inputs --------------------------------------------------------------------
PRED_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3090.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1186, 1216))  # 30 FRESH seeds {1186..1215}, disjoint from nb3190

# -- Per-fold grid (must MATCH nb3190 exactly) ---------------------------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98, 0.99]

# -- Gates (per task) ----------------------------------------------------------
GATE_PROMOTE_PRIMARY1 = 0.4426    # mean < this -> VERIFIED_PROMOTE_PRIMARY1
GATE_MARGINAL = 0.4437            # mean < this -> VERIFIED_MARGINAL
GATE_LUCKY_SHIFT = 0.005          # shift > this from nb3190 -> LUCKY_SEED_TRAP

# -- References ----------------------------------------------------------------
REF_NB3190 = 0.4426
REF_NB3190_STD = 0.0022
REF_PARENT_NB3090 = 0.4472
REF_NB3173 = 0.4437
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
    """Run learned-clip pipeline at a single kf_seed (mirror of nb3190)."""
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
        f"{TAG} -- DEEP-30 VERIFICATION of {VERIFY_TAG} "
        f"(learned clip on {PARENT_TAG})"
    )
    print(
        f"          Q_LOW_GRID  = {Q_LOW_GRID}"
    )
    print(
        f"          Q_HIGH_GRID = {Q_HIGH_GRID}"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}} (disjoint from nb3190 1171..1185)"
    )
    print(
        f"          gate: mean < {GATE_PROMOTE_PRIMARY1:.4f} -> PROMOTE_PRIMARY1, "
        f"< {GATE_MARGINAL:.4f} -> MARGINAL, shift > +{GATE_LUCKY_SHIFT:.3f} -> LUCKY_SEED_TRAP"
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
    print(
        f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    oof_stack = []
    all_fold_ql = []
    all_fold_qh = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
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
            f"ql={res['fold_ql']}  qh={res['fold_qh']}  "
            f"clipped(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    # df=29, two-sided 95% t_mult = 2.0452
    t_mult = 2.0452
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    # Dispersion ratio vs 15-seed
    dispersion_ratio = std_rae / REF_NB3190_STD if REF_NB3190_STD > 0 else np.nan
    shift = mean_rae - REF_NB3190

    # Welch t-test approximation vs nb3190 mean (treat nb3190 as fixed reference)
    # one-sample t-test: t = shift / (std_rae / sqrt(n_s))
    t_stat = shift / (std_rae / np.sqrt(n_s)) if std_rae > 0 else np.nan

    # Most-picked q values across all 5*30=150 folds
    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   mean              = {mean_rae:.4f}")
    print(f"   std               = {std_rae:.4f}")
    print(f"   sem               = {sem:.4f}")
    print(f"   95% CI (df=29)    = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median            = {median_rae:.4f}")
    print(f"   min/max           = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"\n   ref nb3190 15-seed mean = {REF_NB3190:.4f} +/- {REF_NB3190_STD:.4f}")
    print(f"   shift vs nb3190         = {shift:+.4f}")
    print(f"   dispersion ratio        = {dispersion_ratio:.2f}x (deep-30 std / nb3190 std)")
    print(f"   one-sample t            = {t_stat:.3f}")
    print(f"\n   ref nb3090 parent       = {REF_PARENT_NB3090:.4f}")
    print(f"   delta vs nb3090         = {mean_rae - REF_PARENT_NB3090:+.4f}")
    print(f"   ref nb3173 ceiling      = {REF_NB3173:.4f}")
    print(f"   delta vs nb3173         = {mean_rae - REF_NB3173:+.4f}")
    print(f"\n   ql_distribution (150 folds) = {dict(ql_counter)}")
    print(f"   qh_distribution (150 folds) = {dict(qh_counter)}")
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

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if shift > GATE_LUCKY_SHIFT:
        verdict = "LUCKY_SEED_TRAP"
        ladder_action = (
            f"REJECT - LUCKY-SEED TRAP. nb3200 deep-30 mean {mean_rae:.4f} "
            f"shifted +{shift:.4f} vs nb3190 15-seed {REF_NB3190:.4f} "
            f"(> +{GATE_LUCKY_SHIFT:.3f} threshold). nb3190's 0.4426 result "
            f"was a lucky-seed batch; true deep-30 mean is materially worse. "
            f"Dispersion ratio {dispersion_ratio:.2f}x confirms 15-seed std "
            f"was under-estimated. Same root cause as nb1086 / nb2060. "
            f"DROP nb3190 from MARGINAL band; do not promote nb3200; keep "
            f"current ladder (nb2171 0.4682 PRIMARY-1, chemprop_aux SAFETY)."
        )
    elif mean_rae < GATE_PROMOTE_PRIMARY1:
        verdict = "VERIFIED_PROMOTE_PRIMARY1"
        ladder_action = (
            f"PROMOTE-PRIMARY1. nb3200 deep-30 mean {mean_rae:.4f} matches/beats "
            f"nb3190 15-seed {REF_NB3190:.4f} (shift {shift:+.4f}, within "
            f"+{GATE_LUCKY_SHIFT:.3f} noise band) AND clears strict gate "
            f"{GATE_PROMOTE_PRIMARY1:.4f}. nb3190 verified as honest learned-clip "
            f"gain on stronger nb3090 anchor. Dispersion ratio {dispersion_ratio:.2f}x. "
            f"Modal pick (q{ql_mode}, q{qh_mode}). PROMOTE nb3190/nb3200 to "
            f"PRIMARY-1 candidate, displaces nb2171 (0.4682 -> {mean_rae:.4f}, "
            f"-{REF_NB2171 - mean_rae:.4f} RAE) on the ladder."
        )
    elif mean_rae < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
        ladder_action = (
            f"VERIFIED-MARGINAL. nb3200 deep-30 mean {mean_rae:.4f} > strict "
            f"gate {GATE_PROMOTE_PRIMARY1:.4f} but < MARGINAL gate "
            f"{GATE_MARGINAL:.4f}. Shift vs nb3190 = {shift:+.4f} (within lucky "
            f"band). Dispersion ratio {dispersion_ratio:.2f}x. Operator carries "
            f"to deep-30 but doesn't quite reach the BETTER threshold. Keep as "
            f"ALTERNATE, prefer nb2171 PRIMARY-1 until LB re-grade confirms."
        )
    else:
        verdict = "REJECT"
        ladder_action = (
            f"REJECT. nb3200 deep-30 mean {mean_rae:.4f} fails both gates "
            f"(PROMOTE {GATE_PROMOTE_PRIMARY1:.4f}, MARGINAL {GATE_MARGINAL:.4f}). "
            f"Shift vs nb3190 = {shift:+.4f}. Even though not flagged LUCKY-SEED-TRAP "
            f"(< +{GATE_LUCKY_SHIFT:.3f}), the deep-30 mean lands above nb3173 "
            f"ceiling -- learned-clip on nb3090 has been over-credited. Keep nb2171 "
            f"PRIMARY-1; drop nb3190/nb3200 from band-band candidates."
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

    sub_csv = SUBMISSIONS / f"{TAG}_deep_verify_nb3190.csv"
    if verdict.startswith("VERIFIED"):
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
        "verify_tag": VERIFY_TAG,
        "method": (
            "deep_30_verify_of_nb3190_learned_clip_on_nb3090"
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
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "ref_nb3190": REF_NB3190,
        "ref_nb3190_std": REF_NB3190_STD,
        "shift_vs_nb3190": round(shift, 4),
        "dispersion_ratio_vs_nb3190": round(dispersion_ratio, 3),
        "one_sample_t_vs_nb3190": round(float(t_stat), 3) if not np.isnan(t_stat) else None,
        "ref_parent_nb3090": REF_PARENT_NB3090,
        "delta_vs_parent": round(mean_rae - REF_PARENT_NB3090, 4),
        "ref_nb3173": REF_NB3173,
        "delta_vs_nb3173": round(mean_rae - REF_NB3173, 4),
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
            str(sub_csv) if verdict.startswith("VERIFIED") else None
        ),
        "gate_promote_primary1": GATE_PROMOTE_PRIMARY1,
        "gate_marginal": GATE_MARGINAL,
        "gate_lucky_shift": GATE_LUCKY_SHIFT,
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
    print(f"   mean_rae ({n_s} seeds)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   shift vs nb3190 mean  = {shift:+.4f}")
    print(f"   dispersion ratio      = {dispersion_ratio:.2f}x")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "shift_vs_nb3190", "dispersion_ratio_vs_nb3190",
        "delta_vs_parent", "delta_vs_nb3173",
        "ql_mode", "qh_mode",
        "deploy_ql", "deploy_qh", "deploy_lo", "deploy_hi",
        "n_te_clipped_lo", "n_te_clipped_hi",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
