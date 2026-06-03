"""nb1141 -- Bag of shallow LGBM Huber residual on chemprop_aux anchor.

Hypothesis
----------
nb1130 bagged a shallow LGBM Huber residual on the nb1070 anchor (a heavily
post-processed stretch median-bag). nb1133 demonstrated that the chemprop_aux
residual carries ~4x more extractable signal than the nb1070 residual at the
single-seed level (delta -0.0337 single-seed). Since chemprop_aux is a raw
multi-head MPNN with NO per-quantile stretch or post-hoc calibration applied
on top, its residual still contains chemistry-feature structure that a
shallow LGBM can recover -- and bagging across KFold seeds should isolate
the structural signal from partition luck and may extract more lift than
the single-seed point estimate.

Anchor
------
  chemprop_aux  -- raw multi-head MPNN PRE-unblind, in_RAE 0.6216 on the
                   253 unblind (predicted LB ~0.6246, current LB best).

Procedure (per seed s in {0, 1, 7, 42, 137})
--------------------------------------------
  1. anchor_oof on 253 = te_chemprop_aux.npy[unb_idx]  (constant across
     seeds; PRE-unblind, NOT contaminated by 253 labels).
  2. residual = y_unb - anchor_oof  (signed; constant across seeds).
  3. KFold(n=5, shuffle=True, random_state=s) split of the 253 unblind.
  4. Shallow LGBM Huber (max_depth=3, n_est=80, lr=0.05, alpha=1.0,
     min_child_samples=20) on combined Morgan+RDKit (2265 features);
     honest cross-fit -> residual_oof_s.
  5. pred_corrected_s = anchor_oof + residual_oof_s
  6. Record pooled RAE(y_unb, pred_corrected_s).

Bag aggregation
---------------
  - mean_bag    = mean over seeds of pred_corrected_s; pooled RAE
  - median_bag  = median over seeds of pred_corrected_s; pooled RAE

Expected: mean_bag pooled RAE ~0.59 (chemprop_aux 0.6216 floor minus the
residual delta, ideally extracting beyond the nb1133 single-seed -0.0337
via seed averaging). A mean_bag RAE clearly below 0.6216 (chemprop_aux
floor) AND below nb1133's single-seed chemprop_aux corrected RAE confirms
that bagging the residual-LGBM on a less-calibrated anchor yields real
lift; a tie with the single-seed says the residual ceiling is already
saturated at one partition.

Outputs
-------
  data/processed/nb1141_per_seed_corrected_oof.npy   (5, 253) float32
  data/processed/nb1141_mean_bag_oof.npy             (253,) float32
  data/processed/nb1141_median_bag_oof.npy           (253,) float32
  data/processed/nb1141_summary.json
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1141"
ANCHOR_LABEL = "chemprop_aux"
ANCHOR_TE_FILE = "te_chemprop_aux.npy"

# Residual model: shallow LGBM Huber (verbatim nb1123/nb1130/nb1133 spec).
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Reference floors
CHEMPROP_AUX_REF_RAE = 0.6216       # in_RAE on 253 unblind
NB1133_CHEMPROP_AUX_SINGLE_SEED_DELTA = -0.0337  # single-seed delta from nb1133


def _lgbm_params(seed: int) -> dict:
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


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    """Honest 5-fold cross-fit shallow LGBM Huber on residual; return OOF."""
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
    print(f"{TAG} -- BAG of shallow LGBM Huber residual on "
          f"{ANCHOR_LABEL} anchor")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - {ANCHOR_LABEL}_oof")
    print(f"          features = combined Morgan+RDKit (2265)")
    print(f"          LGBM: max_depth=3, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print("=" * 78)

    # ---- Load 513 test, unblind index + truth ----
    te = load_test()
    te_smiles = te["smiles"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape = {y_unb.shape}")

    # ---- Load chemprop_aux anchor (te file on 513; subset to unblind 253) ----
    anchor_te_path = DATA_PROCESSED / ANCHOR_TE_FILE
    if not anchor_te_path.exists():
        raise FileNotFoundError(
            f"{anchor_te_path} not found; "
            f"chemprop_aux te file required."
        )
    anchor_te = np.load(anchor_te_path).astype(np.float64)
    if anchor_te.shape[0] != 513:
        raise ValueError(
            f"{anchor_te_path} shape {anchor_te.shape} != 513"
        )
    anchor_oof = anchor_te[unb_idx].astype(np.float64)
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR_LABEL} (from {ANCHOR_TE_FILE}) "
          f"in_RAE on 253 = {rae_anchor:.4f}  "
          f"(ref ~{CHEMPROP_AUX_REF_RAE:.4f})")

    # ---- Signed residual (constant across residual seeds) ----
    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ---- Features (compute once; reused across seeds) ----
    smi_unb = te_smiles[unb_idx].tolist()
    print(f"[feat] computing combined(Morgan+RDKit) on n={len(smi_unb)} "
          f"unblind SMILES")
    X_unb = impute(combined(smi_unb))
    print(f"[feat] X_unb shape = {X_unb.shape}")

    # ---- Cross-fit shallow LGBM Huber on residual, per seed ----
    print("\n" + "-" * 78)
    print("PER-SEED RESIDUAL CROSS-FIT")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_anchor": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    # ---- Bag aggregations ----
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list   = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean       = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median     = {rae_per_seed_median:.4f}")
    print(f"   per-seed std        = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max    = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_anchor = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1133 single-seed delta ref = "
          f"{NB1133_CHEMPROP_AUX_SINGLE_SEED_DELTA:+.4f}")

    nb1133_implied_single_seed_rae = (
        rae_anchor + NB1133_CHEMPROP_AUX_SINGLE_SEED_DELTA
    )
    beats_anchor = (
        rae_mean_bag < rae_anchor - 0.003
        or rae_median_bag < rae_anchor - 0.003
    )
    beats_nb1130 = False  # compare to nb1130 mean_bag if summary is present
    nb1130_mean_bag = None
    nb1130_path = DATA_PROCESSED / "nb1130_summary.json"
    if nb1130_path.exists():
        try:
            with open(nb1130_path) as f:
                nb1130_summary = json.load(f)
            nb1130_mean_bag = float(nb1130_summary.get("rae_mean_bag"))
            beats_nb1130 = rae_mean_bag < nb1130_mean_bag - 0.003
        except Exception:
            nb1130_mean_bag = None

    bag_beats_single_seed = rae_mean_bag < nb1133_implied_single_seed_rae - 0.003

    if beats_anchor and bag_beats_single_seed:
        verdict = "BAG_EXTRACTS_MORE_THAN_SINGLE_SEED"
    elif beats_anchor and not bag_beats_single_seed:
        verdict = "BAG_CONFIRMS_NB1133_SINGLE_SEED_NO_EXTRA_LIFT"
    elif not beats_anchor:
        verdict = "BAG_DOES_NOT_BEAT_CHEMPROP_AUX_ANCHOR"
    else:
        verdict = "BAG_UNCLEAR"
    print(f"   verdict             = {verdict}")
    if nb1130_mean_bag is not None:
        print(f"   nb1130 mean_bag ref = {nb1130_mean_bag:.4f}  "
              f"(beats_nb1130 = {beats_nb1130})")

    # ---- Save artefacts ----
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
        "anchor": ANCHOR_LABEL,
        "anchor_te_file": ANCHOR_TE_FILE,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "rae_anchor_chemprop_aux": rae_anchor,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_median_bag_vs_anchor": rae_median_bag - rae_anchor,
        "nb1133_chemprop_aux_single_seed_delta_ref":
            NB1133_CHEMPROP_AUX_SINGLE_SEED_DELTA,
        "nb1133_implied_single_seed_rae": nb1133_implied_single_seed_rae,
        "bag_beats_single_seed": bool(bag_beats_single_seed),
        "beats_anchor": bool(beats_anchor),
        "nb1130_mean_bag_ref": nb1130_mean_bag,
        "beats_nb1130": bool(beats_nb1130),
        "verdict": verdict,
        "chemprop_aux_ref_rae": CHEMPROP_AUX_REF_RAE,
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
    for k in ("rae_anchor_chemprop_aux", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_anchor", "delta_median_bag_vs_anchor",
              "bag_beats_single_seed",
              "beats_anchor", "beats_nb1130", "verdict"):
        print(f"  {k}: {res.get(k)}")
