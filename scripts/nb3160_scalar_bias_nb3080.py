"""nb3160 -- PER-FOLD SCALAR BIAS SHIFT on nb3080 (no scale/stretch, just shift).

NEW PARADIGM: just one parameter (shift), simpler than stretch.

Cycle-160 / cycle-149 lineage: prior post-hoc gains came from 1-param
rank-stretch (s ~ 1.05-1.10) on variance-compressed predictors. nb3080
(K18/K19 quantile-conditional hard-split blend) achieved deep-15 mean
0.4475. Stretch axis was closed at deep-ensemble level (nb2200) because
averaging already does the variance decompression. This script tests the
ORTHOGONAL axis: scalar BIAS shift (just add a constant b), which is
conceptually inert at the global-mean level (mean already calibrated per
feedback_rank_stretch_universal) but may extract per-fold residual bias
if the OOF mean is locally off.

PROTOCOL (per kf_seed, 5-fold scaffold split):
    1. Load nb3080_pred_oof.npy (253,) base.
    2. Per fold k:
         golden-section search b in [-0.10, +0.10] minimizing
             RAE(y[tr_fold], pred[tr_fold] + b)
         apply b to held-out:  oof[va] = pred[va] + b_k
    3. Stitch held-out -> oof_shifted (253,), pooled RAE.
    4. Repeat over 15 FRESH kf_seeds {1141..1155}.
    5. Report wide-seed mean +/- std + 95% CI.

GATE (on 15-seed mean):
    mean < 0.4475 -> "BETTER"   (beats nb3080 deep-15)
    else          -> "FAIL"

References:
    nb3080 15-seed wide-mean = 0.4475 +/- 0.0006
    nb3063 5-seed mean       = 0.4477
    nb3030 wide-seed         = 0.4509
    nb2171 prior post-hoc    = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3080_pred_oof.npy   (253,) median-seed OOF base
    data/processed/te_nb3080.npy         (513,) deploy te base

Outputs:
    data/processed/nb3160_summary.json
    data/processed/nb3160_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3160.npy         (513,) float32 -- deploy te
    submissions/nb3160_scalar_bias_nb3080.csv  (only on BETTER verdict)
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3160"
PARENT_TAG = "nb3080"

# -- Inputs --------------------------------------------------------------------
BASE_OOF_PATH = DATA_PROCESSED / "nb3080_pred_oof.npy"
BASE_TE_PATH = DATA_PROCESSED / "te_nb3080.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1141, 1156))  # 15 FRESH seeds {1141..1155}

# -- Golden-section search params ----------------------------------------------
SHIFT_LOW = -0.10
SHIFT_HIGH = +0.10
GS_TOL = 1e-5
GS_MAX_ITER = 60

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4475  # mean < this -> BETTER (beats nb3080 deep-15)

# -- References ----------------------------------------------------------------
REF_NB3080_DEEP15 = 0.4475
REF_NB3080_DEEP15_STD = 0.0006
REF_NB3063_5SEED = 0.4477
REF_NB3030 = 0.4509
REF_NB2171 = 0.4682


def _golden_section_min(
    f,
    a: float,
    b: float,
    tol: float = GS_TOL,
    max_iter: int = GS_MAX_ITER,
) -> tuple[float, float]:
    """Golden-section search for univariate minimum of f on [a, b].

    Returns (x_star, f_star).
    """
    invphi = (np.sqrt(5.0) - 1.0) / 2.0       # 1/phi
    invphi2 = (3.0 - np.sqrt(5.0)) / 2.0      # 1/phi^2
    h = b - a
    if h <= tol:
        x_mid = 0.5 * (a + b)
        return x_mid, f(x_mid)
    c = a + invphi2 * h
    d = a + invphi * h
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
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
        if (b - a) < tol:
            break
    if fc < fd:
        return 0.5 * (a + d), fc
    return 0.5 * (c + b), fd


def _run_one_seed(
    pred_base: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Run per-fold scalar-bias-shift pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_shift = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes: list[float] = []
    fold_bs: list[float] = []
    fold_b_rae: list[float] = []
    for tr_loc, va_loc in splits:
        p_tr = pred_base[tr_loc]
        y_tr = y_unb[tr_loc]

        def objective(b: float, _p=p_tr, _y=y_tr) -> float:
            return float(rae(_y, _p + b))

        b_k, r_tr = _golden_section_min(objective, SHIFT_LOW, SHIFT_HIGH)
        fold_bs.append(float(b_k))
        fold_b_rae.append(float(r_tr))
        val_pred = pred_base[va_loc] + b_k
        oof_shift[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))

    if np.isnan(oof_shift).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_shift))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_b_mean": float(np.mean(fold_bs)),
        "fold_b_std": float(np.std(fold_bs, ddof=1)),
        "fold_b_min": float(np.min(fold_bs)),
        "fold_b_max": float(np.max(fold_bs)),
        "fold_b_train_rae_mean": float(np.mean(fold_b_rae)),
        "oof": oof_shift,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- PER-FOLD SCALAR BIAS SHIFT on {PARENT_TAG} "
        "(just shift, no scale/stretch)"
    )
    print(
        f"          golden-section b in [{SHIFT_LOW:+.2f}, {SHIFT_HIGH:+.2f}] "
        f"per fold; minimize fold-train RAE"
    )
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          ref {PARENT_TAG} deep-15 = "
        f"{REF_NB3080_DEEP15:.4f} +/- {REF_NB3080_DEEP15_STD:.4f}"
    )
    print(f"          gate: mean < {GATE_BETTER:.4f} -> BETTER")
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

    # -- Load base OOF + te ---------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {PARENT_TAG} base OOF + te")
    print("-" * 78)
    pred_base = np.load(BASE_OOF_PATH).astype(np.float64)
    te_base = np.load(BASE_TE_PATH).astype(np.float64)
    if pred_base.shape != (n_unb,):
        raise ValueError(
            f"{PARENT_TAG} OOF shape {pred_base.shape} != ({n_unb},)"
        )
    if te_base.shape != (n_test,):
        raise ValueError(
            f"{PARENT_TAG} te shape {te_base.shape} != ({n_test},)"
        )
    base_rae = float(rae(y_unb, pred_base))
    print(
        f"   base OOF: rae={base_rae:.4f}  "
        f"mean={pred_base.mean():.3f}  std={pred_base.std():.3f}"
    )
    print(
        f"   base te:  mean={te_base.mean():.3f}  std={te_base.std():.3f}"
    )

    leak_frac = float(np.mean(np.isclose(pred_base, y_unb, atol=1e-6)))
    if leak_frac > 0.05:
        print(f"   WARN base: {leak_frac:.1%} rows == truth -- possible leak")
    print(f"   leak_eq_truth_frac = {leak_frac:.4f}")

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
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(pred_base, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_b_mean": round(res["fold_b_mean"], 5),
            "fold_b_std": round(res["fold_b_std"], 5),
            "fold_b_min": round(res["fold_b_min"], 5),
            "fold_b_max": round(res["fold_b_max"], 5),
            "fold_b_train_rae_mean": round(res["fold_b_train_rae_mean"], 4),
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"b_mean={res['fold_b_mean']:+.4f}  "
            f"b_std={res['fold_b_std']:.4f}  "
            f"b_range=[{res['fold_b_min']:+.4f}, {res['fold_b_max']:+.4f}]  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    # 95% CI via t-multiplier (n=15, df=14, t ~ 2.145)
    t_mult = 2.145
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    delta_vs_base = mean_rae - REF_NB3080_DEEP15

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   sem     = {sem:.4f}")
    print(f"   95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median  = {median_rae:.4f}")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(
        f"\n   ref {PARENT_TAG} deep-15 mean = "
        f"{REF_NB3080_DEEP15:.4f} +/- {REF_NB3080_DEEP15_STD:.4f}"
    )
    print(f"   delta vs {PARENT_TAG}        = {delta_vs_base:+.4f}")
    print(f"   ref nb3063 5-seed         = {REF_NB3063_5SEED:.4f}")
    print(f"   ref nb3030 wide-seed      = {REF_NB3030:.4f}")
    print(f"   ref nb2171 prior posthoc  = {REF_NB2171:.4f}")

    # -- Deploy: refit b on FULL 253, apply to te ----------------------------
    def obj_full(b: float, _p=pred_base, _y=y_unb) -> float:
        return float(rae(_y, _p + b))

    deploy_b, deploy_train_rae = _golden_section_min(
        obj_full, SHIFT_LOW, SHIFT_HIGH,
    )
    te_pred = (te_base + deploy_b).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"\n   deploy b (full 253 refit) = {deploy_b:+.5f}  "
        f"train_RAE={deploy_train_rae:.4f}"
    )
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE     = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} (rae={arr[med_seed_idx]:.4f})")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE. nb3160 wide-seed 15-mean {mean_rae:.4f} beats "
            f"nb3080 deep-15 {GATE_BETTER:.4f} ({delta_vs_base:+.4f}). "
            f"Per-fold scalar bias shift (b range "
            f"[{min(sr['fold_b_min'] for sr in seed_records):+.4f}, "
            f"{max(sr['fold_b_max'] for sr in seed_records):+.4f}]) "
            "extracts residual bias. Candidate for ladder promotion."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3160 wide-seed 15-mean {mean_rae:.4f} does NOT beat "
            f"nb3080 deep-15 {GATE_BETTER:.4f} ({delta_vs_base:+.4f}). "
            "Scalar-bias-shift axis closed at deep-15 anchor: mean already "
            "calibrated, residual fold-train bias does not transfer to "
            "fold-val. Keep nb3080 PRIMARY-1."
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

    sub_csv = SUBMISSIONS / f"{TAG}_scalar_bias_nb3080.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    # Aggregate per-fold b distribution across all seeds (75 folds total)
    all_bs = []
    for sr in seed_records:
        # Use mean as a one-number summary per seed (golden-section is deterministic per fold);
        # for the full distribution we re-derive from min/max per seed; cheap.
        pass
    fold_b_grand_mean = float(np.mean([sr["fold_b_mean"] for sr in seed_records]))
    fold_b_grand_std = float(
        np.std([sr["fold_b_mean"] for sr in seed_records], ddof=1)
    )

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "per_fold_scalar_bias_shift_golden_section_no_stretch",
        "base_oof_path": str(BASE_OOF_PATH),
        "base_te_path": str(BASE_TE_PATH),
        "base_oof_rae": round(base_rae, 4),
        "base_oof_leak_eq_truth_frac": round(leak_frac, 4),
        "anchor_pre_unblind": True,  # nb3080 anchors {K18, K19} are PRE
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "shift_low": SHIFT_LOW,
        "shift_high": SHIFT_HIGH,
        "gs_tol": GS_TOL,
        "gs_max_iter": GS_MAX_ITER,
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
        "ref_nb3080_deep15": REF_NB3080_DEEP15,
        "ref_nb3080_deep15_std": REF_NB3080_DEEP15_STD,
        "delta_vs_nb3080": round(delta_vs_base, 4),
        "ref_nb3063_5seed": REF_NB3063_5SEED,
        "ref_nb3030": REF_NB3030,
        "ref_nb2171": REF_NB2171,
        "fold_b_grand_mean": round(fold_b_grand_mean, 5),
        "fold_b_grand_std": round(fold_b_grand_std, 5),
        "deploy_b": round(float(deploy_b), 5),
        "deploy_train_rae": round(float(deploy_train_rae), 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv) if verdict == "BETTER" else None),
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
    print(f"   mean_rae ({n_s} seeds)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3080       = {delta_vs_base:+.4f}")
    print(f"   deploy b              = {deploy_b:+.5f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3080", "fold_b_grand_mean", "fold_b_grand_std",
        "deploy_b",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
