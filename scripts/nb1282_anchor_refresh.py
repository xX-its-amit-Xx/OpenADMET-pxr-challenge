"""nb1282 -- Anchor refresh: rebuild anchor as median-bag of strong predictions.

Hypothesis
----------
All ChEMBL/MACCS residual learning (nb1183, nb1211, nb1242 = 0.5431) has
been anchored on nb1070 alone (a per-quantile-stretched bag).  This single-
anchor framing might cap the residual learner's ceiling because all residual
correlation we see is conditioned on nb1070's specific failure structure.

Try a FRESH anchor that's the median-bag of two diverse strong-anchor
candidates:
    new_anchor = median([nb1070_pred_oof, nb1014_pred_oof])
nb1014 is a 5-seed bag (median across nb988 seeds) -- different upstream
chemistry than nb1070's per-quantile-stretched bag.  If the median-bag
RAE is similar or better than nb1070 standalone, run a MACCS+ChEMBL
residual learner on the new anchor.

NOTE on PRE-unblind anchors
---------------------------
The task spec says "use only PRE-unblind anchors."  The 253-row honest
cross-fit OOFs we have are:
    nb1070_pred_oof.npy            (253,)  RAE 0.5771  -- canonical
    nb1133_nb1014_pred_oof.npy     (253,)  RAE 0.5798  -- nb1014-anchored
                                                          honest cross-fit
                                                          residual-corrected
We use these as the fresh anchor inputs.  chemprop_aux has no honest 253
cross-fit OOF (only te_chemprop_aux on 513) -- skipped per spec.

Pipeline
--------
1. Load nb1070_pred_oof (253) and nb1133_nb1014_pred_oof (253).
2. new_anchor_oof = median across the two.
3. Verify RAE(new_anchor) vs nb1070; if WORSE by > 0.003, abandon.
4. Residual learner features: MACCS-167 + pred_chembl_pec50 + sim_chembl
   (169 cols total) on the 253 unblind rows.
5. residual = y_unb - new_anchor; 5-seed bag shallow LGBM Huber, identical
   hyperparams as nb1242, 5-fold cross-fit per seed.
6. mean_bag = mean(per_seed_corrected), median_bag = median.
7. Verdict at 0.003 margin vs nb1242 (0.5431).

Outputs
-------
    scripts/nb1282_anchor_refresh.py                  (this file)
    data/processed/nb1282_summary.json
    data/processed/nb1282_new_anchor_oof.npy          (253,) float32
    data/processed/nb1282_mean_bag_oof.npy            (253,) float32
    data/processed/nb1282_median_bag_oof.npy          (253,) float32
    data/processed/nb1282_per_seed_corrected_oof.npy  (5, 253) float32
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1282"

# --- Anchor pool (honest 253 cross-fit OOFs only) ---
ANCHOR_FILES = [
    ("nb1070",        "nb1070_pred_oof.npy"),
    ("nb1014_xfit",   "nb1133_nb1014_pred_oof.npy"),
]
# chemprop_aux skipped (no honest 253 cross-fit OOF; te_chemprop_aux is
# deploy-only, would leak post-unblind label noise into the anchor).

NB1070_REF = 0.5771
NB1242_REF = 0.5431          # mean-bag pooled RAE, ChEMBL kNN feature residual on nb1070
NB1014_XFIT_REF = 0.5798     # nb1133_nb1014_pred_oof pooled RAE
DECISION_MARGIN = 0.003
ABANDON_MARGIN = 0.003       # if new anchor is worse than nb1070 by > this, stop

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]


def _lgbm_params(seed: int) -> dict:
    """Identical capacity to nb1242 residual learner."""
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Anchor refresh: median-bag of strong anchors + "
          f"MACCS+ChEMBL residual")
    print(f"          anchors  = {[a for a, _ in ANCHOR_FILES]}")
    print(f"          seeds    = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + pred_chembl_pec50 + sim  (169)")
    print(f"          target ref = nb1242 mean_bag {NB1242_REF:.4f}  "
          f"(margin {DECISION_MARGIN})")
    print("=" * 78)

    # ---- Truth + indices ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}  y_unb mean={y_unb.mean():.3f} "
          f"std={y_unb.std():.3f}")

    # ---- Load anchor OOFs ----
    print("\n" + "-" * 78)
    print("ANCHOR LOAD (honest 253 cross-fit OOFs)")
    print("-" * 78)
    anchor_stack = []
    anchor_records = []
    for label, fname in ANCHOR_FILES:
        p = DATA_PROCESSED / fname
        if not p.exists():
            raise FileNotFoundError(f"anchor not found: {p}")
        a = np.load(p).astype(np.float64)
        if a.shape != (n_unb,):
            raise ValueError(
                f"{label} shape mismatch: {a.shape} vs ({n_unb},)"
            )
        rae_a = float(rae(y_unb, a))
        anchor_stack.append(a)
        anchor_records.append({
            "label": label,
            "file": fname,
            "rae": rae_a,
            "mean": float(a.mean()),
            "std": float(a.std()),
        })
        print(f"   {label:14s}: rae={rae_a:.4f}  "
              f"mean={a.mean():.3f}  std={a.std():.3f}")

    anchor_mat = np.stack(anchor_stack, axis=0)   # (n_anchors, 253)

    # ---- Build new anchor: median across anchors ----
    new_anchor = np.median(anchor_mat, axis=0)
    mean_anchor = np.mean(anchor_mat, axis=0)
    rae_new_anchor = float(rae(y_unb, new_anchor))
    rae_mean_anchor = float(rae(y_unb, mean_anchor))
    print("\n" + "-" * 78)
    print("FRESH ANCHOR (median-bag)")
    print("-" * 78)
    print(f"   median-bag  RAE = {rae_new_anchor:.4f}  "
          f"(d vs nb1070 = {rae_new_anchor - NB1070_REF:+.4f})")
    print(f"   mean-bag    RAE = {rae_mean_anchor:.4f}  "
          f"(d vs nb1070 = {rae_mean_anchor - NB1070_REF:+.4f})")
    print(f"   new_anchor  mean={new_anchor.mean():.3f}  "
          f"std={new_anchor.std():.3f}")

    abandon = rae_new_anchor > NB1070_REF + ABANDON_MARGIN
    if abandon:
        print(f"   [ABANDON] new anchor worse than nb1070 by > {ABANDON_MARGIN}; "
              f"emitting summary and exiting before residual stage.")

    residual = y_unb - new_anchor
    print(f"   residual mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL feature pool ----
    print("\n" + "-" * 78)
    print("CHEMBL kNN FEATURES (reuse cached arrays from nb1242 ChEMBL probe)")
    print("-" * 78)
    pred_chembl_513 = np.load(DATA_PROCESSED / "pred_chembl_pec50_513.npy")
    sim_chembl_513 = np.load(DATA_PROCESSED / "sim_chembl_513.npy")
    if pred_chembl_513.shape[0] != 513 or sim_chembl_513.shape[0] != 513:
        raise ValueError(
            f"chembl feature shapes off: pred={pred_chembl_513.shape} "
            f"sim={sim_chembl_513.shape}"
        )
    pred_chembl_unb = pred_chembl_513[unb_idx].astype(np.float32)
    sim_chembl_unb = sim_chembl_513[unb_idx].astype(np.float32)
    print(f"   pred_chembl  unb: mean={pred_chembl_unb.mean():.3f}  "
          f"std={pred_chembl_unb.std():.3f}")
    print(f"   sim_chembl   unb: p10={np.percentile(sim_chembl_unb,10):.3f}  "
          f"p50={np.percentile(sim_chembl_unb,50):.3f}  "
          f"p90={np.percentile(sim_chembl_unb,90):.3f}")

    # ---- MACCS-167 (unblind slice) ----
    X_maccs_te = np.load(DATA_PROCESSED / "te_maccs.npy")
    if X_maccs_te.shape[0] != 513:
        raise ValueError(f"te_maccs shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"   MACCS unb shape = {X_maccs_unb.shape}")

    # ---- Build residual feature matrix ----
    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_chembl_unb.reshape(-1, 1),
            sim_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + pred_chembl + sim)")

    # ---- Save new anchor regardless of abandon flag ----
    np.save(DATA_PROCESSED / f"{TAG}_new_anchor_oof.npy",
            new_anchor.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_new_anchor_oof.npy'}")

    if abandon:
        summary = {
            "tag": TAG,
            "abandoned": True,
            "abandon_reason": (
                f"new_anchor RAE {rae_new_anchor:.4f} > nb1070 ref "
                f"{NB1070_REF:.4f} + margin {ABANDON_MARGIN}"
            ),
            "anchor_records": anchor_records,
            "rae_new_anchor_median": rae_new_anchor,
            "rae_new_anchor_mean": rae_mean_anchor,
            "nb1070_ref": NB1070_REF,
            "nb1242_ref": NB1242_REF,
            "verdict": "NEW_ANCHOR_WORSE_THAN_NB1070_ABANDON",
            "wall_sec": round(time.time() - t0, 2),
        }
        out_path = DATA_PROCESSED / f"{TAG}_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[save] {out_path}")
        print("=" * 78)
        return summary

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow LGBM Huber, "
          f"dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = new_anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_new_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_new_anchor": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_new_anchor = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_new_anchor = {rae_mean_bag - rae_new_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_new_anchor = {rae_median_bag - rae_new_anchor:+.4f})")
    print(f"   nb1070 ref             = {NB1070_REF:.4f}")
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (target to beat)")

    beats_nb1070 = rae_mean_bag < NB1070_REF - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN
    new_anchor_helps = rae_new_anchor < NB1070_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "ANCHOR_REFRESH_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif rae_mean_bag < NB1242_REF + DECISION_MARGIN:
        verdict = "ANCHOR_REFRESH_TIES_NB1242_WITHIN_MARGIN"
    elif beats_nb1070:
        verdict = "ANCHOR_REFRESH_BEATS_NB1070_BUT_NOT_NB1242"
    else:
        verdict = "ANCHOR_REFRESH_NO_GAIN_OVER_NB1242"
    print(f"   verdict                = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "abandoned": False,
        "anchor_records": anchor_records,
        "anchor_fusion": "median across honest 253 cross-fit OOFs",
        "rae_new_anchor_median": rae_new_anchor,
        "rae_new_anchor_mean": rae_mean_anchor,
        "delta_new_anchor_vs_nb1070": rae_new_anchor - NB1070_REF,
        "new_anchor_helps_vs_nb1070": bool(new_anchor_helps),
        "feature_dim": feat_dim,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_new_anchor": rae_mean_bag - rae_new_anchor,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - NB1070_REF,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1242": bool(beats_nb1242),
        "verdict": verdict,
        "nb1070_ref": NB1070_REF,
        "nb1242_ref": NB1242_REF,
        "nb1014_xfit_ref": NB1014_XFIT_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_new_anchor_median", "rae_new_anchor_mean",
        "delta_new_anchor_vs_nb1070",
        "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070", "delta_mean_bag_vs_nb1242",
        "beats_nb1070", "beats_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
