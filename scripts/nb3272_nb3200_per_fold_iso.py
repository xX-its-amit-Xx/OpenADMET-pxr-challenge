"""nb3272 -- Per-fold isotonic calibration on nb3200 ultra-verified PRIMARY-1.

NEW PARADIGM: iso applied to the ultra-verified PRIMARY-1 (nb3200) rather than
to the prior PRIMARY-1 nb3080 (cf. nb3144).

CONTEXT:
    nb3200 is the deep-30 verify (kf_seeds {1186..1215}) of the nb3090 -> learned
    -clip chain. nb3232 60-seed extra-deep verify locked in mean 0.4424 -> nb3200
    is the ultra-verified PRIMARY-1 ceiling for the chemprop_aux + learned-clip
    substrate.

    nb3144 applied per-fold IsotonicRegression on the prior PRIMARY-1 (nb3080,
    pooled mean 0.4475). This script repeats that exact operator stack but on
    the strictly better nb3200 anchor (0.4424).

PROTOCOL:
    Anchor: nb3200 = deep-30 verify of nb3190 learned-clip on nb3090
        - nb3200_pred_oof.npy : (253,) median-seed OOF (kf_seed=1203)
        - te_nb3200.npy       : (513,) deploy te
    Outer CV: 5-fold scaffold split, 15 fresh kf_seeds {1216..1230}
        (disjoint from nb3200 {1186..1215} AND from nb3232 {1246..1305})
    Per fold:
        - IsotonicRegression(y_min=3.0, y_max=8.0, increasing=True, out_of_bounds="clip")
          fit on (fold-train anchor, fold-train y)
        - apply on fold-val anchor predictions
    Per seed: pooled RAE across the 5 outer-val folds.
    Reported gate metric = MEAN pooled RAE across the 15 seeds.

GATE:
    mean_pooled_rae < 0.4423  ->  "BETTER"
    else                      ->  "FAIL"

References:
    nb3200 ultra-verified PRIMARY-1 (deep-30 mean)   = 0.4424 (gate is one below)
    nb3232 extra-deep 60-seed verify of nb3200       = 0.4424 (confirmed)
    nb3190 15-seed verify of learned-clip on nb3090  = 0.4426
    nb3090 anchor (learned clip on chemprop_aux)     = 0.4472
    nb3173 prior ceiling                             = 0.4437
    nb3144 iso on nb3080 (prior paradigm result)     = depends on per-seed
    nb2171 prior post-hoc-blend ceiling              = 0.4682
    nb3080 prior PRIMARY-1 (q-cond hard-split blend) = 0.4475

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy
    data/processed/te_nb3200.npy

Outputs:
    data/processed/nb3272_summary.json
    data/processed/nb3272_pred_oof.npy   (253,) float32 -- per-fold iso OOF (first seed)
    data/processed/te_nb3272.npy         (513,) float32 -- deploy te (full-fit iso)
    submissions/nb3272_nb3200_per_fold_iso.csv  (only if verdict != "FAIL")
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
from sklearn.isotonic import IsotonicRegression

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3272"
PARENT_TAG = "nb3200"

# -- Inputs --------------------------------------------------------------------
OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3200.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 fresh seeds {1216..1230}
ISO_Y_MIN = 3.0
ISO_Y_MAX = 8.0

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4423   # strictly under nb3200's verified 0.4424

# -- References ----------------------------------------------------------------
REF_NB3200_NOM = 0.4424     # nb3200 deep-30 mean (ultra-verified PRIMARY-1)
REF_NB3232_NOM = 0.4424     # nb3232 60-seed extra-deep verify (locked PRIMARY-1)
REF_NB3190_NOM = 0.4426     # nb3190 15-seed verify
REF_NB3090_NOM = 0.4472     # parent of nb3200 (learned clip on chemprop_aux)
REF_NB3173_NOM = 0.4437     # prior ceiling
REF_NB3080_NOM = 0.4475     # prior PRIMARY-1 (nb3144 anchor)
REF_NB2171 = 0.4682


def _fit_iso(p_tr: np.ndarray, y_tr: np.ndarray) -> IsotonicRegression:
    """Fit increasing IsotonicRegression with y bounds clipped to [3.0, 8.0]."""
    iso = IsotonicRegression(
        y_min=ISO_Y_MIN,
        y_max=ISO_Y_MAX,
        increasing=True,
        out_of_bounds="clip",
    )
    iso.fit(p_tr, y_tr)
    return iso


def _run_one_seed(
    kf_seed: int,
    p_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
) -> tuple[float, list[dict], np.ndarray]:
    """Per-fold iso fit at one kf_seed. Returns (pooled_rae, fold_records, oof_iso)."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_iso = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        iso = _fit_iso(p_unb[tr_loc], y_unb[tr_loc])
        train_pred = iso.transform(p_unb[tr_loc])
        val_pred = iso.transform(p_unb[va_loc])
        oof_iso[va_loc] = val_pred
        r_tr = float(rae(y_unb[tr_loc], train_pred))
        r_va = float(rae(y_unb[va_loc], val_pred))
        fold_records.append({
            "fold": int(fold_i),
            "n_train": int(len(tr_loc)),
            "n_val": int(len(va_loc)),
            "train_rae": round(float(r_tr), 4),
            "val_rae": round(r_va, 4),
        })
    if np.isnan(oof_iso).any():
        raise RuntimeError(
            f"scaffold splits did not cover all {n_unb} rows (kf_seed={kf_seed})"
        )
    pooled_rae = float(rae(y_unb, oof_iso))
    return pooled_rae, fold_records, oof_iso


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold isotonic calibration on {PARENT_TAG} ultra-verified PRIMARY-1")
    print(f"          paradigm: per-fold IsotonicRegression on (anchor, y)")
    print(f"          y bounds: [{ISO_Y_MIN}, {ISO_Y_MAX}]")
    print(
        f"          outer CV: {N_FOLDS}-fold scaffold, "
        f"{len(KF_SEEDS)} seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"          gate: mean_pooled_rae < {GATE_BETTER}  -> BETTER, else FAIL")
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

    # -- Load nb3200 anchor --------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {PARENT_TAG} anchor (pred_oof on 253, te on 513)")
    print("-" * 78)
    p_oof = np.load(OOF_PATH).astype(np.float64)
    p_te = np.load(TE_PATH).astype(np.float64)
    if p_oof.shape != (n_unb,):
        raise ValueError(f"{PARENT_TAG} OOF shape {p_oof.shape} != ({n_unb},)")
    if p_te.shape != (n_test,):
        raise ValueError(f"{PARENT_TAG} te shape {p_te.shape} != ({n_test},)")
    anchor_full_rae = float(rae(y_unb, p_oof))
    anchor_te_unb_in_rae = float(rae(y_unb, p_te[unb_idx]))
    mu_oof = float(p_oof.mean())
    mu_y = float(y_unb.mean())
    print(
        f"   {PARENT_TAG} OOF      mean={mu_oof:.4f}  std={p_oof.std():.4f}  "
        f"RAE={anchor_full_rae:.4f}"
    )
    print(f"   {PARENT_TAG} te(unb)  in-sample RAE = {anchor_te_unb_in_rae:.4f}")
    print(f"   y_unb           mean={mu_y:.4f}  std={y_unb.std():.4f}")

    # Leak sanity on anchor
    leak_eq = float(np.mean(np.isclose(p_oof, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN anchor: {leak_eq:.1%} rows == truth -- possible leak")

    # -- Scaffolds -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique scaffolds = {n_unique_scaf}")

    # -- Per-seed per-fold iso fit (CV) --------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 3: outer CV with per-fold isotonic, {len(KF_SEEDS)} seeds")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_fold_records = {}
    first_seed_oof_iso = None
    for seed in KF_SEEDS:
        p_rae, fold_recs, oof_sh = _run_one_seed(
            seed, p_oof, y_unb, unb_scaffolds,
        )
        per_seed_pooled.append(p_rae)
        per_seed_fold_records[str(seed)] = fold_recs
        if first_seed_oof_iso is None:
            first_seed_oof_iso = oof_sh
        mean_val = float(np.mean([r["val_rae"] for r in fold_recs]))
        print(f"   seed={seed}  pooled={p_rae:.4f}  per-fold mean={mean_val:.4f}")

    arr_pooled = np.asarray(per_seed_pooled)
    n_s = len(arr_pooled)
    mean_pooled = float(arr_pooled.mean())
    std_pooled = float(arr_pooled.std(ddof=1)) if n_s > 1 else 0.0
    sem_pooled = std_pooled / np.sqrt(n_s) if n_s > 1 else 0.0
    # df=14, two-sided 95% t_mult = 2.1448
    t_mult = 2.1448
    ci_low = mean_pooled - t_mult * sem_pooled
    ci_high = mean_pooled + t_mult * sem_pooled
    median_pooled = float(np.median(arr_pooled))
    min_pooled = float(arr_pooled.min())
    max_pooled = float(arr_pooled.max())
    print(f"\n   POOLED-OUTER-VAL RAE over {n_s} seeds:")
    print(f"     mean   = {mean_pooled:.4f}")
    print(f"     std    = {std_pooled:.4f}")
    print(f"     sem    = {sem_pooled:.4f}")
    print(f"     95% CI = [{ci_low:.4f}, {ci_high:.4f}] (df=14)")
    print(f"     median = {median_pooled:.4f}")
    print(f"     min    = {min_pooled:.4f}")
    print(f"     max    = {max_pooled:.4f}")

    # -- Deploy: fit iso on FULL 253 -----------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: deploy iso = fit on FULL 253")
    print("-" * 78)
    iso_full = _fit_iso(p_oof, y_unb)
    full_train_pred = iso_full.transform(p_oof)
    r_full = float(rae(y_unb, full_train_pred))
    print(f"   full-OOF in-sample iso RAE = {r_full:.4f}")

    te_pred = iso_full.transform(p_te).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"   te(iso) mean={te_pred.mean():.3f} std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}  "
        f"in-sample unb RAE = {te_unb_in_rae:.4f}"
    )

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 5: GATE on mean pooled outer-val RAE across {n_s} seeds")
    print("-" * 78)
    if mean_pooled < GATE_BETTER:
        verdict = "BETTER"
    else:
        verdict = "FAIL"
    delta_vs_nb3200_nom = mean_pooled - REF_NB3200_NOM
    delta_vs_nb3200_oof = mean_pooled - anchor_full_rae
    delta_vs_nb3232 = mean_pooled - REF_NB3232_NOM
    delta_vs_nb3173 = mean_pooled - REF_NB3173_NOM
    delta_vs_nb3080 = mean_pooled - REF_NB3080_NOM
    delta_vs_nb2171 = mean_pooled - REF_NB2171
    print(f"   mean_pooled_rae              = {mean_pooled:.4f} (std {std_pooled:.4f})")
    print(f"   delta vs nb3200 nom 0.4424   = {delta_vs_nb3200_nom:+.4f}")
    print(f"   delta vs nb3200 full OOF     = {delta_vs_nb3200_oof:+.4f}")
    print(f"   delta vs nb3232 (0.4424)     = {delta_vs_nb3232:+.4f}")
    print(f"   delta vs nb3173 (0.4437)     = {delta_vs_nb3173:+.4f}")
    print(f"   delta vs nb3080 (0.4475)     = {delta_vs_nb3080:+.4f}")
    print(f"   delta vs nb2171 (0.4682)     = {delta_vs_nb2171:+.4f}")
    print(f"   verdict                      = {verdict}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: save artifacts")
    print("-" * 78)
    oof_out_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_out_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_out_path, first_seed_oof_iso.astype(np.float32))
    np.save(te_out_path, te_pred)
    print(f"   [save] {oof_out_path}  (single-seed iso OOF, kf_seed={KF_SEEDS[0]})")
    print(f"   [save] {te_out_path}   (deploy = iso_full({PARENT_TAG}_te))")

    sub_csv = SUBMISSIONS / f"{TAG}_nb3200_per_fold_iso.csv"
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
        "method": "per_fold_isotonic_calibration_on_nb3200_ultra_verified_primary1",
        "paradigm": "isotonic_post_hoc_calibration",
        "anchor_pre_unblind": True,
        "anchor_pool": [PARENT_TAG],
        "anchor_full_oof_rae": round(anchor_full_rae, 5),
        "anchor_te_unb_in_sample_rae": round(anchor_te_unb_in_rae, 5),
        "anchor_mu_oof": mu_oof,
        "anchor_leak_eq_truth_frac": round(leak_eq, 4),
        "y_mu": mu_y,
        "iso_y_min": ISO_Y_MIN,
        "iso_y_max": ISO_Y_MAX,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_seed_pooled_rae": [round(r, 5) for r in per_seed_pooled],
        "per_seed_fold_records": per_seed_fold_records,
        "pooled_rae_mean": round(mean_pooled, 5),
        "pooled_rae_std": round(std_pooled, 5),
        "pooled_rae_sem": round(sem_pooled, 5),
        "pooled_rae_ci95_low": round(ci_low, 5),
        "pooled_rae_ci95_high": round(ci_high, 5),
        "pooled_rae_median": round(median_pooled, 5),
        "pooled_rae_min": round(min_pooled, 5),
        "pooled_rae_max": round(max_pooled, 5),
        "full_rae_in_sample": round(float(r_full), 5),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 5),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "pred_oof_path": str(oof_out_path),
        "te_npy_path": str(te_out_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "mean_rae": mean_pooled,
        "ref_nb3200_nom": REF_NB3200_NOM,
        "ref_nb3232_nom": REF_NB3232_NOM,
        "ref_nb3190_nom": REF_NB3190_NOM,
        "ref_nb3090_nom": REF_NB3090_NOM,
        "ref_nb3173_nom": REF_NB3173_NOM,
        "ref_nb3080_nom": REF_NB3080_NOM,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3200_nom": round(delta_vs_nb3200_nom, 5),
        "delta_vs_nb3200_full_oof": round(delta_vs_nb3200_oof, 5),
        "delta_vs_nb3232": round(delta_vs_nb3232, 5),
        "delta_vs_nb3173": round(delta_vs_nb3173, 5),
        "delta_vs_nb3080": round(delta_vs_nb3080, 5),
        "delta_vs_nb2171": round(delta_vs_nb2171, 5),
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   {PARENT_TAG} anchor full OOF RAE = {anchor_full_rae:.4f}")
    print(
        f"   pooled outer-val RAE       = {mean_pooled:.4f} +/- {std_pooled:.4f} "
        f"({n_s} seeds)"
    )
    print(f"   95% CI (df=14)             = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   min/max pooled RAE         = {min_pooled:.4f} / {max_pooled:.4f}")
    print(f"   full-OOF in-sample iso RAE = {r_full:.4f}")
    print(f"   te[unb_idx] in-sample      = {te_unb_in_rae:.4f}")
    print(f"   delta vs nb3200 nom        = {delta_vs_nb3200_nom:+.4f}")
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
        "full_rae_in_sample",
        "te_unb_in_sample_rae",
        "delta_vs_nb3200_nom",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
