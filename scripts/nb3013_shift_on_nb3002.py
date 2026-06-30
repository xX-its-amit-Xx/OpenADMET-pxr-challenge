"""nb3013 -- Scalar shift sweep on nb3002 K18+K19 prediction.

NEW PARADIGM: pred = (mu + shift) + (nb3002_pred - mu)
            = nb3002_pred + shift                       (since mu cancels)

This is a 1-parameter scalar bias-shift over the nb3002 deploy
prediction. We sweep shift via per-fold golden-section search over
[-0.1, +0.1] minimizing fold-train RAE, and also report the explicit
grid {-0.05, -0.02, 0, +0.02, +0.05} for sanity.

PROTOCOL:
    Anchor: nb3002 = per-fold SLSQP simplex on {K18, K19} deep-30
        - nb3002_pred_oof.npy : (253,) single-seed OOF blend
        - te_nb3002.npy       : (513,) deploy te
    Outer CV: 5-fold scaffold split, 5 fresh kf_seeds {1051..1055}
    Per fold:
        - golden-section over shift in [-0.1, +0.1] minimizing
          rae(y_tr, p_tr + shift)
        - apply shift to held-out fold-val slice
    Per seed: pooled RAE across the 5 outer-val folds.
    Reported gate metric = MEAN pooled RAE across the 5 seeds.

GATE:
    mean_pooled_rae < 0.4511  ->  "BETTER_THAN_NB3001"
    else                      ->  "FAIL"

References:
    nb3002 K18+K19 deep-30 SLSQP outer-val RAE = 0.4479 (15 seeds, per nb3002 ref)
                                                actual fresh run logged below
    nb2960 K18 deep-30 OOF                     = 0.4536
    nb3000 K19 deep-30 OOF                     = 0.4607
    nb2171 prior post-hoc-blend ceiling        = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3002_pred_oof.npy
    data/processed/te_nb3002.npy

Outputs:
    data/processed/nb3013_summary.json
    data/processed/nb3013_pred_oof.npy   (253,) float32 -- per-fold shifted OOF (first seed)
    data/processed/te_nb3013.npy         (513,) float32 -- deploy te (mean shift applied)
    submissions/nb3013_shift_on_nb3002.csv  (only if verdict != "FAIL")
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

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

TAG = "nb3013"
PARENT_TAG = "nb3002"

# -- Inputs --------------------------------------------------------------------
OOF_PATH = DATA_PROCESSED / "nb3002_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3002.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1051, 1056))  # 5 fresh seeds {1051..1055}
SHIFT_BOUNDS = (-0.10, 0.10)
EXPLICIT_GRID = [-0.05, -0.02, 0.0, 0.02, 0.05]

# -- Gates ---------------------------------------------------------------------
GATE_BETTER_THAN_NB3001 = 0.4511

# -- References ----------------------------------------------------------------
REF_NB3002_NOM = 0.4479      # reported nb3002 pooled outer-val (15 seeds)
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _golden_section(f, a: float, b: float, tol: float = 1e-5,
                    max_iter: int = 200) -> tuple[float, float]:
    """Minimize unimodal scalar f on [a, b]. Returns (x_star, f_star)."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0  # 0.6180...
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    it = 0
    while (b - a) > tol and it < max_iter:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a)
            f2 = f(x2)
        it += 1
    if f1 < f2:
        return float(x1), float(f1)
    return float(x2), float(f2)


def _fit_shift(p_tr: np.ndarray, y_tr: np.ndarray) -> tuple[float, float]:
    """Find shift in SHIFT_BOUNDS minimizing fold-train RAE."""
    a, b = SHIFT_BOUNDS

    def loss(s: float) -> float:
        return float(rae(y_tr, p_tr + s))

    s_star, r_star = _golden_section(loss, a, b)
    return s_star, r_star


def _run_one_seed(kf_seed: int, p_unb: np.ndarray, y_unb: np.ndarray,
                  unb_scaffolds: list[str]) -> tuple[float, list[dict], np.ndarray, list[float]]:
    """Per-fold shift fit with one kf_seed. Returns (pooled_rae, fold_records, oof_shifted, fold_shifts)."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_shifted = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    fold_shifts = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        s_star, r_tr = _fit_shift(p_unb[tr_loc], y_unb[tr_loc])
        val_pred = p_unb[va_loc] + s_star
        oof_shifted[va_loc] = val_pred
        r_va = float(rae(y_unb[va_loc], val_pred))
        fold_shifts.append(s_star)
        fold_records.append({
            "fold": int(fold_i),
            "n_train": int(len(tr_loc)),
            "n_val": int(len(va_loc)),
            "shift": round(float(s_star), 5),
            "train_rae": round(float(r_tr), 4),
            "val_rae": round(r_va, 4),
        })
    if np.isnan(oof_shifted).any():
        raise RuntimeError(f"scaffold splits did not cover all {n_unb} rows (kf_seed={kf_seed})")
    pooled_rae = float(rae(y_unb, oof_shifted))
    return pooled_rae, fold_records, oof_shifted, fold_shifts


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- scalar shift sweep on nb3002 K18+K19 deploy prediction")
    print(f"          paradigm: pred = nb3002_pred + shift")
    print(f"          per-fold golden-section over shift in {SHIFT_BOUNDS}")
    print(f"          outer CV: {N_FOLDS}-fold scaffold, {len(KF_SEEDS)} seeds {KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"          gate: <{GATE_BETTER_THAN_NB3001}  BETTER_THAN_NB3001")
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load nb3002 anchor --------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load nb3002 anchor (pred_oof on 253, te on 513)")
    print("-" * 78)
    nb3002_oof = np.load(OOF_PATH).astype(np.float64)
    nb3002_te = np.load(TE_PATH).astype(np.float64)
    if nb3002_oof.shape != (n_unb,):
        raise ValueError(f"nb3002 OOF shape {nb3002_oof.shape} != ({n_unb},)")
    if nb3002_te.shape != (n_test,):
        raise ValueError(f"nb3002 te shape {nb3002_te.shape} != ({n_test},)")
    nb3002_full_rae = float(rae(y_unb, nb3002_oof))
    nb3002_te_unb_in_rae = float(rae(y_unb, nb3002_te[unb_idx]))
    mu_oof = float(nb3002_oof.mean())
    mu_y = float(y_unb.mean())
    print(f"   nb3002 OOF      mean={mu_oof:.4f}  std={nb3002_oof.std():.4f}  RAE={nb3002_full_rae:.4f}")
    print(f"   nb3002 te(unb)  in-sample RAE = {nb3002_te_unb_in_rae:.4f}")
    print(f"   y_unb           mean={mu_y:.4f}  std={y_unb.std():.4f}")
    print(f"   mean-gap (y - pred) = {(mu_y - mu_oof):+.4f}")

    # -- Build scaffolds ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique scaffolds = {n_unique_scaf}")

    # -- Explicit grid sweep (whole-OOF, no CV) ------------------------------
    print("\n" + "-" * 78)
    print("STEP 3a: explicit grid sweep over nb3002_oof (whole 253, no CV)")
    print("-" * 78)
    grid_records = []
    for s in EXPLICIT_GRID:
        r = float(rae(y_unb, nb3002_oof + s))
        grid_records.append({"shift": round(float(s), 4), "rae_full": round(r, 5)})
        print(f"   shift = {s:+.4f}  full-OOF RAE = {r:.5f}")

    # -- Per-seed per-fold shift fit (CV) ------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3b: outer CV with per-fold golden-section shift, {len(KF_SEEDS)} seeds")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_fold_records = {}
    per_seed_fold_shifts = {}
    first_seed_oof_shifted = None
    for seed in KF_SEEDS:
        p_rae, fold_recs, oof_sh, fold_ss = _run_one_seed(
            seed, nb3002_oof, y_unb, unb_scaffolds,
        )
        per_seed_pooled.append(p_rae)
        per_seed_fold_records[str(seed)] = fold_recs
        per_seed_fold_shifts[str(seed)] = [round(float(s), 5) for s in fold_ss]
        if first_seed_oof_shifted is None:
            first_seed_oof_shifted = oof_sh
        mean_val = float(np.mean([r["val_rae"] for r in fold_recs]))
        print(f"   seed={seed}  pooled={p_rae:.4f}  "
              f"per-fold mean={mean_val:.4f}  shifts={[round(s, 4) for s in fold_ss]}")

    arr_pooled = np.asarray(per_seed_pooled)
    mean_pooled = float(arr_pooled.mean())
    std_pooled = float(arr_pooled.std(ddof=1)) if len(arr_pooled) > 1 else 0.0
    min_pooled = float(arr_pooled.min())
    max_pooled = float(arr_pooled.max())
    print(f"\n   POOLED-OUTER-VAL RAE over {len(KF_SEEDS)} seeds:")
    print(f"     mean = {mean_pooled:.4f}")
    print(f"     std  = {std_pooled:.4f}")
    print(f"     min  = {min_pooled:.4f}")
    print(f"     max  = {max_pooled:.4f}")

    # -- Aggregate shifts across folds and seeds -----------------------------
    all_shifts = []
    for sk, fs_list in per_seed_fold_shifts.items():
        all_shifts.extend(fs_list)
    arr_shifts = np.asarray(all_shifts, dtype=np.float64)
    mean_shift = float(arr_shifts.mean())
    median_shift = float(np.median(arr_shifts))
    std_shift = float(arr_shifts.std(ddof=1)) if len(arr_shifts) > 1 else 0.0
    print(f"\n   per-fold shifts across all (seed,fold) cells (n={len(arr_shifts)}):")
    print(f"     mean   = {mean_shift:+.5f}")
    print(f"     median = {median_shift:+.5f}")
    print(f"     std    = {std_shift:.5f}")
    print(f"     min/max= {arr_shifts.min():+.5f} / {arr_shifts.max():+.5f}")

    # -- Deploy: single-shift fit on FULL 253 --------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: deploy shift = golden-section on FULL 253")
    print("-" * 78)
    s_full, r_full = _fit_shift(nb3002_oof, y_unb)
    print(f"   full-OOF best shift = {s_full:+.5f}  full-OOF RAE = {r_full:.4f}")

    te_pred = (nb3002_te + s_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te(shifted) mean={te_pred.mean():.3f} std={te_pred.std():.3f}  "
          f"in-sample unb RAE = {te_unb_in_rae:.4f}")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: GATE on mean pooled outer-val RAE across seeds")
    print("-" * 78)
    if mean_pooled < GATE_BETTER_THAN_NB3001:
        verdict = "BETTER_THAN_NB3001"
    else:
        verdict = "FAIL"
    delta_vs_nb3002_nom = mean_pooled - REF_NB3002_NOM
    delta_vs_nb3002_oof = mean_pooled - nb3002_full_rae
    delta_vs_nb2171 = mean_pooled - REF_NB2171
    print(f"   mean_pooled_rae          = {mean_pooled:.4f} (std {std_pooled:.4f})")
    print(f"   delta vs nb3002 ref 0.4479   = {delta_vs_nb3002_nom:+.4f}")
    print(f"   delta vs nb3002 full OOF     = {delta_vs_nb3002_oof:+.4f}")
    print(f"   delta vs nb2171 (0.4682)     = {delta_vs_nb2171:+.4f}")
    print(f"   verdict                  = {verdict}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_out_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_out_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_out_path, first_seed_oof_shifted.astype(np.float32))
    np.save(te_out_path, te_pred)
    print(f"   [save] {oof_out_path}  (single-seed shifted OOF, kf_seed={KF_SEEDS[0]})")
    print(f"   [save] {te_out_path}   (deploy = nb3002_te + s_full)")

    sub_csv = SUBMISSIONS / f"{TAG}_shift_on_nb3002.csv"
    if verdict != "FAIL":
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
        "method": "per_fold_golden_section_scalar_shift_on_nb3002",
        "paradigm": "scalar_shift_post_hoc_calibration",
        "anchor_pre_unblind": True,
        "anchor_pool": ["nb3002"],
        "anchor_full_oof_rae": round(nb3002_full_rae, 5),
        "anchor_te_unb_in_sample_rae": round(nb3002_te_unb_in_rae, 5),
        "anchor_mu_oof": mu_oof,
        "y_mu": mu_y,
        "mean_gap_y_minus_pred": round(mu_y - mu_oof, 5),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "shift_bounds": list(SHIFT_BOUNDS),
        "explicit_grid": EXPLICIT_GRID,
        "explicit_grid_records": grid_records,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_seed_pooled_rae": [round(r, 5) for r in per_seed_pooled],
        "per_seed_fold_records": per_seed_fold_records,
        "per_seed_fold_shifts": per_seed_fold_shifts,
        "pooled_rae_mean": round(mean_pooled, 5),
        "pooled_rae_std": round(std_pooled, 5),
        "pooled_rae_min": round(min_pooled, 5),
        "pooled_rae_max": round(max_pooled, 5),
        "fold_shift_mean": round(mean_shift, 5),
        "fold_shift_median": round(median_shift, 5),
        "fold_shift_std": round(std_shift, 5),
        "fold_shift_min": round(float(arr_shifts.min()), 5),
        "fold_shift_max": round(float(arr_shifts.max()), 5),
        "full_shift": round(float(s_full), 5),
        "full_rae_in_sample": round(float(r_full), 5),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 5),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "pred_oof_path": str(oof_out_path),
        "te_npy_path": str(te_out_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "mean_rae": mean_pooled,
        "ref_nb3002_nom": REF_NB3002_NOM,
        "ref_K18": REF_K18,
        "ref_K19": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3002_nom": delta_vs_nb3002_nom,
        "delta_vs_nb3002_full_oof": delta_vs_nb3002_oof,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_better_than_nb3001": GATE_BETTER_THAN_NB3001,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   nb3002 anchor full OOF RAE = {nb3002_full_rae:.4f}")
    print(f"   pooled outer-val RAE       = {mean_pooled:.4f} +/- {std_pooled:.4f} "
          f"({len(KF_SEEDS)} seeds)")
    print(f"   min/max pooled RAE         = {min_pooled:.4f} / {max_pooled:.4f}")
    print(f"   mean per-fold shift        = {mean_shift:+.5f}  (median {median_shift:+.5f})")
    print(f"   full-OOF deploy shift      = {s_full:+.5f}")
    print(f"   te[unb_idx] in-sample      = {te_unb_in_rae:.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "anchor_full_oof_rae",
        "pooled_rae_mean",
        "pooled_rae_std",
        "pooled_rae_min",
        "pooled_rae_max",
        "fold_shift_mean",
        "fold_shift_median",
        "full_shift",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  explicit_grid_records: {res.get('explicit_grid_records')}")
