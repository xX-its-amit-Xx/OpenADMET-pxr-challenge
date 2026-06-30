"""nb3423 -- Per-fold isotonic on nb3200, scored under the POOLED metric.

NEW PARADIGM (the question this script answers)
-----------------------------------------------
Cycles 261 / 144 (nb3272, nb3144) rejected per-fold IsotonicRegression because
the candidate was scored on the PER-FOLD-MEAN of the 5 outer-val RAE ratios
(nb3272 mean ~0.46-0.47, FAIL vs 0.4423). But nb3402 established that the public
LB scores all rows with ONE RAE denominator -- a POOLED computation:

    RAE(S) = sum_{i in S} |y_i - p_i|  /  sum_{i in S} |y_i - mean_S(y)|

The 253-unblind analog of the LB number is therefore the POOLED RAE over the
full cross-fit OOF vector (a single rae() call), NOT the mean of 5 per-fold
ratios. nb3402 proved PER-FOLD-MEAN overstates by a Jensen / denominator-
reweighting gap (~+0.01) driven by fold truth-dispersion GEOMETRY, not by
prediction quality. So per-fold isotonic -- dismissed on per-fold-mean -- might
genuinely tie or beat nb3200 under the LB-faithful pooled metric.

WHY POOLED IS *NOT* SEED-INVARIANT HERE (unlike nb3410's frozen vectors)
------------------------------------------------------------------------
nb3410 scored DEPLOY-FROZEN pred_oof vectors: for a fixed (y, p), rae() has no
seed dependence (std == 0). IsotonicRegression is a PER-FOLD-FITTED operator:
each kf_seed produces a DIFFERENT scaffold split -> a DIFFERENT cross-fit OOF
vector -> a DIFFERENT pooled rae(). So the honest pooled estimand is the MEAN of
the per-seed pooled RAE across the 15 fresh seeds (with a real across-seed std).
This mirrors nb3200's own deep-30 *re-fit* mean (each seed re-derives the OOF).

PROTOCOL
--------
    Anchor:  nb3200 = deep-30 verify of nb3190 learned-clip on nb3090
        nb3200_pred_oof.npy : (253,) median-seed OOF
        te_nb3200.npy       : (513,) deploy te
    Outer CV: 5-fold scaffold split, 15 FRESH kf_seeds {1216..1230}
        (disjoint from nb3200 {1186..1215} and nb3232 {1246..1305})
    Per fold:
        IsotonicRegression(y_min=3.0, y_max=8.0, increasing=True,
                           out_of_bounds="clip") fit on (fold-train anchor,
        fold-train y); applied to fold-val anchor predictions.
    Per seed: POOLED RAE = single rae() over the 5 outer-val folds (LB-faithful);
              PER-FOLD-MEAN RAE = mean of the 5 fold ratios (sidecar diagnostic).
    Gate metric = MEAN POOLED RAE across the 15 seeds.

GATE (vs nb3200 pooled reference 0.4424)
----------------------------------------
    mean_pooled < 0.4414  ->  "BETTER"    (nb3200 - 0.001, meaningful win)
    mean_pooled < 0.4424  ->  "MARGINAL"  (tie band, inside 0.001 noise floor)
    else                  ->  "FAIL"

References (pooled-metric ladder)
---------------------------------
    nb3200 ultra-verified PRIMARY-1 (deep-30 pooled)   = 0.4424  (gate reference)
    nb3232 60-seed extra-deep verify of nb3200          = 0.4424
    nb3190 15-seed verify learned-clip on nb3090        = 0.4426
    nb3173 prior ceiling                                = 0.4437
    nb3090 anchor (learned clip on chemprop_aux)        = 0.4472
    nb3080 prior PRIMARY-1 (q-cond hard-split blend)    = 0.4475
    nb3272 iso-on-nb3200 PER-FOLD-MEAN reject           = ~0.46-0.47 (old metric)
    nb2171 prior post-hoc-blend ceiling                 = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy
    data/processed/te_nb3200.npy

Outputs:
    data/processed/nb3423_summary.json
    data/processed/nb3423_pred_oof.npy   (253,) float32 -- per-fold iso OOF (first seed)
    data/processed/te_nb3423.npy         (513,) float32 -- deploy te (full-fit iso)
    submissions/nb3423_pooled_isotonic.csv  (only if verdict != "FAIL")
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

TAG = "nb3423"
PARENT_TAG = "nb3200"

# -- Inputs --------------------------------------------------------------------
OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"
TE_PATH = DATA_PROCESSED / "te_nb3200.npy"

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 fresh seeds {1216..1230}
ISO_Y_MIN = 3.0
ISO_Y_MAX = 8.0
TE_CLIP_LO = 3.0
TE_CLIP_HI = 9.0

# -- Gates (POOLED metric, vs nb3200 pooled reference 0.4424) ------------------
NB3200_POOLED_REF = 0.4424
GATE_BETTER = round(NB3200_POOLED_REF - 0.001, 4)  # 0.4414  (meaningful win)
GATE_MARGINAL_HI = NB3200_POOLED_REF               # 0.4424  (tie band upper)

# -- References (reconciliation only) ------------------------------------------
REF_NB3200_NOM = 0.4424     # nb3200 deep-30 pooled mean (PRIMARY-1, gate ref)
REF_NB3232_NOM = 0.4424     # nb3232 60-seed extra-deep verify
REF_NB3190_NOM = 0.4426     # nb3190 15-seed verify
REF_NB3090_NOM = 0.4472     # parent of nb3200 (learned clip on chemprop_aux)
REF_NB3173_NOM = 0.4437     # prior ceiling
REF_NB3080_NOM = 0.4475     # prior PRIMARY-1 (nb3144 / nb3272 anchor)
REF_NB2171 = 0.4682         # prior post-hoc-blend ceiling


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
) -> dict:
    """Per-fold iso fit at one kf_seed.

    Returns a dict with BOTH:
      - pooled : single rae() over the full cross-fit OOF vector (LB-faithful)
      - pf_mean: mean of the 5 per-fold val RAE ratios (sidecar diagnostic)
    plus the OOF vector (for first-seed deploy artifact) and fold records.
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_iso = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_rae = []
    fold_records = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        iso = _fit_iso(p_unb[tr_loc], y_unb[tr_loc])
        train_pred = iso.transform(p_unb[tr_loc])
        val_pred = iso.transform(p_unb[va_loc])
        oof_iso[va_loc] = val_pred
        r_tr = float(rae(y_unb[tr_loc], train_pred))
        r_va = float(rae(y_unb[va_loc], val_pred))
        fold_val_rae.append(r_va)
        fold_records.append({
            "fold": int(fold_i),
            "n_train": int(len(tr_loc)),
            "n_val": int(len(va_loc)),
            "train_rae": round(r_tr, 4),
            "val_rae": round(r_va, 4),
        })
    if np.isnan(oof_iso).any():
        raise RuntimeError(
            f"scaffold splits did not cover all {n_unb} rows (kf_seed={kf_seed})"
        )
    pooled_rae = float(rae(y_unb, oof_iso))          # LB-faithful (single denom)
    pf_mean_rae = float(np.mean(fold_val_rae))        # sidecar (Jensen-inflated)
    return {
        "kf_seed": int(kf_seed),
        "pooled": pooled_rae,
        "pf_mean": pf_mean_rae,
        "oof_iso": oof_iso,
        "fold_records": fold_records,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-fold isotonic on {PARENT_TAG}, scored under POOLED metric")
    print(f"          paradigm: per-fold IsotonicRegression on (anchor, y), "
          f"y bounds [{ISO_Y_MIN}, {ISO_Y_MAX}]")
    print(f"          outer CV: {N_FOLDS}-fold scaffold, "
          f"{len(KF_SEEDS)} fresh seeds {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          LB-faithful metric = POOLED (single rae() over 253; nb3402)")
    print(f"          gate vs nb3200 pooled {NB3200_POOLED_REF}: "
          f"BETTER<{GATE_BETTER}, MARGINAL[{GATE_BETTER},{GATE_MARGINAL_HI}], else FAIL")
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
    anchor_pooled_rae = float(rae(y_unb, p_oof))          # anchor's own pooled
    anchor_te_unb_in_rae = float(rae(y_unb, p_te[unb_idx]))
    mu_oof = float(p_oof.mean())
    mu_y = float(y_unb.mean())
    D_full_253 = float(np.sum(np.abs(y_unb - mu_y)))      # the one pooled denom
    print(f"   {PARENT_TAG} OOF  mean={mu_oof:.4f} std={p_oof.std():.4f}  "
          f"POOLED RAE={anchor_pooled_rae:.4f}")
    print(f"   {PARENT_TAG} te(unb) in-sample RAE = {anchor_te_unb_in_rae:.4f}")
    print(f"   y_unb  mean={mu_y:.4f} std={y_unb.std():.4f}  "
          f"L1 dispersion D(U)={D_full_253:.2f} (the pooled/LB denominator)")

    # Leak sanity on anchor (rows whose OOF == truth -> anchor leak).
    leak_eq = float(np.mean(np.isclose(p_oof, y_unb, atol=1e-6)))
    if leak_eq > 0.02:
        print(f"   WARN anchor: {leak_eq:.1%} rows == truth -- possible leak")

    # -- Scaffolds -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique scaffolds = {n_unique_scaf}")

    # -- Per-seed per-fold iso fit (CV), reporting BOTH pooled and pf_mean ----
    print("\n" + "-" * 78)
    print(f"STEP 3: outer CV per-fold isotonic, {len(KF_SEEDS)} seeds "
          f"(report POOLED + per-fold-mean)")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_pf_mean = []
    per_seed_fold_records = {}
    seed_rows = []
    first_seed_oof_iso = None
    for seed in KF_SEEDS:
        r = _run_one_seed(seed, p_oof, y_unb, unb_scaffolds)
        per_seed_pooled.append(r["pooled"])
        per_seed_pf_mean.append(r["pf_mean"])
        per_seed_fold_records[str(seed)] = r["fold_records"]
        if first_seed_oof_iso is None:
            first_seed_oof_iso = r["oof_iso"]
        seed_rows.append({
            "kf_seed": r["kf_seed"],
            "pooled": round(r["pooled"], 5),
            "pf_mean": round(r["pf_mean"], 5),
            "gap_pf_minus_pooled": round(r["pf_mean"] - r["pooled"], 5),
        })
        print(f"   seed={seed}  POOLED={r['pooled']:.5f}  "
              f"pf_mean={r['pf_mean']:.5f}  "
              f"(gap +{r['pf_mean'] - r['pooled']:.4f})")

    arr_pooled = np.asarray(per_seed_pooled)
    arr_pf = np.asarray(per_seed_pf_mean)
    n_s = len(arr_pooled)

    # -- POOLED aggregate (the gate metric) ----------------------------------
    mean_pooled = float(arr_pooled.mean())
    std_pooled = float(arr_pooled.std(ddof=1)) if n_s > 1 else 0.0
    sem_pooled = std_pooled / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.1448  # df=14 two-sided 95%
    ci_low = mean_pooled - t_mult * sem_pooled
    ci_high = mean_pooled + t_mult * sem_pooled
    median_pooled = float(np.median(arr_pooled))
    min_pooled = float(arr_pooled.min())
    max_pooled = float(arr_pooled.max())

    # -- PER-FOLD-MEAN aggregate (sidecar; the OLD, rejected metric) ----------
    mean_pf = float(arr_pf.mean())
    std_pf = float(arr_pf.std(ddof=1)) if n_s > 1 else 0.0
    median_pf = float(np.median(arr_pf))
    mean_gap = mean_pf - mean_pooled

    print(f"\n   POOLED (LB-faithful gate metric) over {n_s} seeds:")
    print(f"     mean   = {mean_pooled:.5f}")
    print(f"     std    = {std_pooled:.5f}")
    print(f"     sem    = {sem_pooled:.5f}")
    print(f"     95% CI = [{ci_low:.5f}, {ci_high:.5f}] (df={n_s-1})")
    print(f"     median = {median_pooled:.5f}   min={min_pooled:.5f}  max={max_pooled:.5f}")
    print(f"\n   PER-FOLD-MEAN sidecar (OLD metric, cycle-261/144 rejected on this):")
    print(f"     mean   = {mean_pf:.5f}   std={std_pf:.5f}   median={median_pf:.5f}")
    print(f"     mean Jensen gap (pf_mean - pooled) = +{mean_gap:.5f}")

    # -- Deploy: fit iso on FULL 253 -----------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: deploy iso = fit on FULL 253")
    print("-" * 78)
    iso_full = _fit_iso(p_oof, y_unb)
    full_train_pred = iso_full.transform(p_oof)
    r_full = float(rae(y_unb, full_train_pred))
    print(f"   full-OOF in-sample iso RAE = {r_full:.4f}")

    te_pred = iso_full.transform(p_te).astype(np.float32)
    te_pred = np.clip(te_pred, TE_CLIP_LO, TE_CLIP_HI)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te(iso) mean={te_pred.mean():.3f} std={te_pred.std():.3f}  "
          f"min={te_pred.min():.3f} max={te_pred.max():.3f}  "
          f"in-sample unb RAE = {te_unb_in_rae:.4f}")

    # -- Gate on MEAN POOLED RAE ---------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 5: GATE on MEAN POOLED RAE across {n_s} seeds (LB-faithful)")
    print("-" * 78)
    if mean_pooled < GATE_BETTER:
        verdict = "BETTER"
    elif mean_pooled < GATE_MARGINAL_HI:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    delta_vs_nb3200_nom = mean_pooled - REF_NB3200_NOM
    delta_vs_anchor_pooled = mean_pooled - anchor_pooled_rae
    delta_vs_nb3232 = mean_pooled - REF_NB3232_NOM
    delta_vs_nb3173 = mean_pooled - REF_NB3173_NOM
    delta_vs_nb3080 = mean_pooled - REF_NB3080_NOM
    delta_vs_nb2171 = mean_pooled - REF_NB2171
    print(f"   mean_pooled_rae                 = {mean_pooled:.5f} (std {std_pooled:.5f})")
    print(f"   delta vs nb3200 nom 0.4424      = {delta_vs_nb3200_nom:+.5f}")
    print(f"   delta vs nb3200 anchor pooled   = {delta_vs_anchor_pooled:+.5f}")
    print(f"   delta vs nb3232 (0.4424)        = {delta_vs_nb3232:+.5f}")
    print(f"   delta vs nb3173 (0.4437)        = {delta_vs_nb3173:+.5f}")
    print(f"   delta vs nb3080 (0.4475)        = {delta_vs_nb3080:+.5f}")
    print(f"   delta vs nb2171 (0.4682)        = {delta_vs_nb2171:+.5f}")
    print(f"   GATE: BETTER<{GATE_BETTER}  MARGINAL[{GATE_BETTER},{GATE_MARGINAL_HI})  "
          f"-> verdict = {verdict}")

    # Honest framing: did the pooled re-eval rescue iso vs its old pf_mean reject?
    pooled_vs_pf_rescue = bool(mean_pooled < mean_pf - 1e-9)
    print(f"   pooled rescues iso vs its own pf_mean? "
          f"{mean_pooled:.5f} < {mean_pf:.5f} = {pooled_vs_pf_rescue} "
          f"(but gate is vs nb3200, not vs its own sidecar)")

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

    sub_csv = SUBMISSIONS / f"{TAG}_pooled_isotonic.csv"
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
        "method": "per_fold_isotonic_on_nb3200_scored_under_pooled_metric",
        "paradigm": "isotonic_post_hoc_calibration_pooled_reeval",
        "lb_faithful_metric": "POOLED (single rae() over all 253; nb3402)",
        "gate_metric": "mean_pooled_rae",
        "anchor_pre_unblind": True,
        "anchor_pool": [PARENT_TAG],
        "anchor_pooled_rae": round(anchor_pooled_rae, 5),
        "anchor_te_unb_in_sample_rae": round(anchor_te_unb_in_rae, 5),
        "anchor_mu_oof": mu_oof,
        "anchor_leak_eq_truth_frac": round(leak_eq, 4),
        "y_mu": mu_y,
        "D_full_253": round(D_full_253, 4),
        "iso_y_min": ISO_Y_MIN,
        "iso_y_max": ISO_Y_MAX,
        "te_clip_lo": TE_CLIP_LO,
        "te_clip_hi": TE_CLIP_HI,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_rows": seed_rows,
        "per_seed_pooled_rae": [round(r, 5) for r in per_seed_pooled],
        "per_seed_pf_mean_rae": [round(r, 5) for r in per_seed_pf_mean],
        "per_seed_fold_records": per_seed_fold_records,
        # POOLED aggregate (gate metric) --------------------------------------
        "pooled_rae_mean": round(mean_pooled, 5),
        "pooled_rae_std": round(std_pooled, 5),
        "pooled_rae_sem": round(sem_pooled, 5),
        "pooled_rae_ci95_low": round(ci_low, 5),
        "pooled_rae_ci95_high": round(ci_high, 5),
        "pooled_rae_median": round(median_pooled, 5),
        "pooled_rae_min": round(min_pooled, 5),
        "pooled_rae_max": round(max_pooled, 5),
        # PER-FOLD-MEAN aggregate (sidecar, OLD metric) -----------------------
        "pf_mean_rae_mean": round(mean_pf, 5),
        "pf_mean_rae_std": round(std_pf, 5),
        "pf_mean_rae_median": round(median_pf, 5),
        "mean_jensen_gap_pf_minus_pooled": round(mean_gap, 5),
        "pooled_rescues_iso_vs_pf": pooled_vs_pf_rescue,
        # deploy --------------------------------------------------------------
        "full_rae_in_sample": round(r_full, 5),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 5),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "pred_oof_path": str(oof_out_path),
        "te_npy_path": str(te_out_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        # canonical ranking key (POOLED is the gate metric here) --------------
        "mean_rae": round(mean_pooled, 5),
        # references ----------------------------------------------------------
        "nb3200_pooled_ref": NB3200_POOLED_REF,
        "ref_nb3200_nom": REF_NB3200_NOM,
        "ref_nb3232_nom": REF_NB3232_NOM,
        "ref_nb3190_nom": REF_NB3190_NOM,
        "ref_nb3090_nom": REF_NB3090_NOM,
        "ref_nb3173_nom": REF_NB3173_NOM,
        "ref_nb3080_nom": REF_NB3080_NOM,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3200_nom": round(delta_vs_nb3200_nom, 5),
        "delta_vs_anchor_pooled": round(delta_vs_anchor_pooled, 5),
        "delta_vs_nb3232": round(delta_vs_nb3232, 5),
        "delta_vs_nb3173": round(delta_vs_nb3173, 5),
        "delta_vs_nb3080": round(delta_vs_nb3080, 5),
        "delta_vs_nb2171": round(delta_vs_nb2171, 5),
        "gate_better": GATE_BETTER,
        "gate_marginal_hi": GATE_MARGINAL_HI,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   {PARENT_TAG} anchor POOLED RAE   = {anchor_pooled_rae:.5f}")
    print(f"   POOLED outer-val RAE (GATE)   = {mean_pooled:.5f} +/- {std_pooled:.5f} "
          f"({n_s} seeds)")
    print(f"   95% CI (df={n_s-1})              = [{ci_low:.5f}, {ci_high:.5f}]")
    print(f"   POOLED min/max                = {min_pooled:.5f} / {max_pooled:.5f}")
    print(f"   PER-FOLD-MEAN sidecar         = {mean_pf:.5f} +/- {std_pf:.5f} "
          f"(OLD metric, gap +{mean_gap:.5f})")
    print(f"   full-OOF in-sample iso RAE    = {r_full:.5f}")
    print(f"   te[unb_idx] in-sample         = {te_unb_in_rae:.5f}")
    print(f"   delta vs nb3200 nom 0.4424    = {delta_vs_nb3200_nom:+.5f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "lb_faithful_metric",
        "anchor_pooled_rae",
        "pooled_rae_mean",
        "pooled_rae_std",
        "pooled_rae_min",
        "pooled_rae_max",
        "pf_mean_rae_mean",
        "mean_jensen_gap_pf_minus_pooled",
        "full_rae_in_sample",
        "te_unb_in_sample_rae",
        "delta_vs_nb3200_nom",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
