"""nb1142 -- Bag of shallow LGBM Huber residual on COMBINED anchor.

Hypothesis
----------
nb1130 bagged the residual on the nb1070 anchor (mean_bag RAE 0.5673).
nb1141 bagged the residual on the chemprop_aux anchor. The two anchors
have orthogonal failure modes -- chemprop_aux is a raw multi-head MPNN
(PRE-unblind in_RAE 0.6216), nb1070 is a heavily post-processed stretch
median-bag (in_RAE 0.5771). A 50/50 convex combination of the two
anchors should yield a SMOOTHER residual target -- the per-anchor noise
partially cancels, leaving the structural signal the shallow LGBM can
recover. If so, the residual model fit on the combined residual should
generalise BETTER across the 5 folds than either of nb1130 / nb1141,
because it sees a more stable per-row residual.

Anchor
------
  combined = 0.5 * chemprop_aux_oof + 0.5 * nb1070_oof   (on 253 unblind)
    chemprop_aux_oof  = te_chemprop_aux.npy[unb_idx]     (PRE-unblind)
    nb1070_oof        = nb1070_pred_oof.npy              (cross-fit OOF)

Procedure (per seed s in {0, 1, 7, 42, 137})
--------------------------------------------
  1. anchor_oof on 253 = 0.5 * chemprop_aux_oof + 0.5 * nb1070_oof
     (constant across seeds).
  2. residual = y_unb - anchor_oof   (signed; constant across seeds).
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

Reference floors
----------------
  chemprop_aux in_RAE     ~ 0.6216
  nb1070 in_RAE           ~ 0.5771
  combined anchor in_RAE  (computed live below; expected ~0.55-0.58)
  nb1130 mean_bag         = 0.5673   (residual on nb1070)
  nb1141 mean_bag         = (loaded if present)

Win criteria
------------
  - mean_bag RAE < combined-anchor RAE by >= 0.003   (residual extracts lift)
  - mean_bag RAE < nb1130 mean_bag 0.5673 by >= 0.003  (combo anchor helps)

Outputs
-------
  data/processed/nb1142_per_seed_corrected_oof.npy   (5, 253) float32
  data/processed/nb1142_mean_bag_oof.npy             (253,) float32
  data/processed/nb1142_median_bag_oof.npy           (253,) float32
  data/processed/nb1142_summary.json
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

TAG = "nb1142"
ANCHOR_LABEL = "combined_chemprop_aux_nb1070"
CHEMPROP_AUX_TE_FILE = "te_chemprop_aux.npy"
NB1070_OOF_FILE = "nb1070_pred_oof.npy"
ANCHOR_W_CHEMPROP = 0.5
ANCHOR_W_NB1070 = 0.5

# Residual model: shallow LGBM Huber (verbatim nb1130/nb1141 spec).
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Reference floors
CHEMPROP_AUX_REF_RAE = 0.6216
NB1070_REF_RAE = 0.5771
NB1130_MEAN_BAG_RAE = 0.5673


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
    print(f"{TAG} -- BAG of shallow LGBM Huber residual on COMBINED anchor")
    print(f"          anchor = {ANCHOR_W_CHEMPROP:.2f} * chemprop_aux "
          f"+ {ANCHOR_W_NB1070:.2f} * nb1070_oof")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - combined_anchor_oof")
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
    chemprop_te_path = DATA_PROCESSED / CHEMPROP_AUX_TE_FILE
    if not chemprop_te_path.exists():
        raise FileNotFoundError(
            f"{chemprop_te_path} not found; chemprop_aux te file required."
        )
    chemprop_te = np.load(chemprop_te_path).astype(np.float64)
    if chemprop_te.shape[0] != 513:
        raise ValueError(
            f"{chemprop_te_path} shape {chemprop_te.shape} != 513"
        )
    chemprop_oof = chemprop_te[unb_idx].astype(np.float64)
    rae_chemprop = float(rae(y_unb, chemprop_oof))
    print(f"[load] chemprop_aux (from {CHEMPROP_AUX_TE_FILE}) "
          f"in_RAE on 253 = {rae_chemprop:.4f}  "
          f"(ref ~{CHEMPROP_AUX_REF_RAE:.4f})")

    # ---- Load nb1070 cross-fit OOF on 253 (constant across seeds) ----
    nb1070_oof_path = DATA_PROCESSED / NB1070_OOF_FILE
    if not nb1070_oof_path.exists():
        raise FileNotFoundError(
            f"{nb1070_oof_path} not found; run nb1070 first to regenerate it."
        )
    nb1070_oof = np.load(nb1070_oof_path).astype(np.float64)
    if nb1070_oof.shape[0] != n_unb:
        raise ValueError(
            f"{nb1070_oof_path} shape mismatch: "
            f"{nb1070_oof.shape} vs n_unb={n_unb}"
        )
    rae_nb1070 = float(rae(y_unb, nb1070_oof))
    print(f"[load] nb1070_pred_oof.npy shape={nb1070_oof.shape}  "
          f"in_RAE = {rae_nb1070:.4f}  (ref ~{NB1070_REF_RAE:.4f})")

    # ---- Build combined anchor (constant across seeds) ----
    anchor_oof = (
        ANCHOR_W_CHEMPROP * chemprop_oof + ANCHOR_W_NB1070 * nb1070_oof
    )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[anchor] combined ({ANCHOR_W_CHEMPROP:.2f}*chemprop_aux + "
          f"{ANCHOR_W_NB1070:.2f}*nb1070) in_RAE = {rae_anchor:.4f}")

    # ---- Signed residual (constant across residual seeds) ----
    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # Compare residual std to the per-anchor residual stds for diagnostic
    resid_chemprop_std = float((y_unb - chemprop_oof).std())
    resid_nb1070_std = float((y_unb - nb1070_oof).std())
    print(f"[resid-cmp] |y - chemprop_aux|.std = {resid_chemprop_std:.4f}  "
          f"|y - nb1070|.std = {resid_nb1070_std:.4f}  "
          f"|y - combined|.std = {residual.std():.4f}")

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

    # ---- Reference comparisons ----
    beats_anchor = (
        rae_mean_bag < rae_anchor - 0.003
        or rae_median_bag < rae_anchor - 0.003
    )

    # nb1130 mean_bag (residual on nb1070 alone)
    nb1130_mean_bag = None
    nb1130_path = DATA_PROCESSED / "nb1130_summary.json"
    if nb1130_path.exists():
        try:
            with open(nb1130_path) as f:
                nb1130_summary = json.load(f)
            nb1130_mean_bag = float(nb1130_summary.get("rae_mean_bag"))
        except Exception:
            nb1130_mean_bag = None
    beats_nb1130 = (
        nb1130_mean_bag is not None
        and rae_mean_bag < nb1130_mean_bag - 0.003
    )
    if nb1130_mean_bag is not None:
        print(f"   nb1130 mean_bag ref = {nb1130_mean_bag:.4f}  "
              f"(beats_nb1130 = {beats_nb1130})")

    # nb1141 mean_bag (residual on chemprop_aux alone), if available
    nb1141_mean_bag = None
    nb1141_path = DATA_PROCESSED / "nb1141_summary.json"
    if nb1141_path.exists():
        try:
            with open(nb1141_path) as f:
                nb1141_summary = json.load(f)
            nb1141_mean_bag = float(nb1141_summary.get("rae_mean_bag"))
        except Exception:
            nb1141_mean_bag = None
    beats_nb1141 = (
        nb1141_mean_bag is not None
        and rae_mean_bag < nb1141_mean_bag - 0.003
    )
    if nb1141_mean_bag is not None:
        print(f"   nb1141 mean_bag ref = {nb1141_mean_bag:.4f}  "
              f"(beats_nb1141 = {beats_nb1141})")

    if beats_anchor and beats_nb1130 and beats_nb1141:
        verdict = "COMBINED_ANCHOR_RESIDUAL_BEATS_BOTH_SINGLE_ANCHOR_BAGS"
    elif beats_anchor and beats_nb1130:
        verdict = "COMBINED_BEATS_NB1130_TIES_OR_LOSES_NB1141"
    elif beats_anchor and beats_nb1141:
        verdict = "COMBINED_BEATS_NB1141_TIES_OR_LOSES_NB1130"
    elif beats_anchor:
        verdict = "COMBINED_RESIDUAL_ADDS_LIFT_OVER_OWN_ANCHOR_ONLY"
    else:
        verdict = "COMBINED_RESIDUAL_FAILS"
    print(f"   verdict             = {verdict}")

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
        "anchor_weights": {
            "chemprop_aux": ANCHOR_W_CHEMPROP,
            "nb1070": ANCHOR_W_NB1070,
        },
        "chemprop_te_file": CHEMPROP_AUX_TE_FILE,
        "nb1070_oof_file": NB1070_OOF_FILE,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "rae_chemprop_aux_alone": rae_chemprop,
        "rae_nb1070_alone": rae_nb1070,
        "rae_combined_anchor": rae_anchor,
        "residual_std_chemprop_aux": resid_chemprop_std,
        "residual_std_nb1070": resid_nb1070_std,
        "residual_std_combined": float(residual.std()),
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
        "beats_anchor": bool(beats_anchor),
        "nb1130_mean_bag_ref": nb1130_mean_bag,
        "beats_nb1130": bool(beats_nb1130),
        "nb1141_mean_bag_ref": nb1141_mean_bag,
        "beats_nb1141": bool(beats_nb1141),
        "verdict": verdict,
        "chemprop_aux_ref_rae": CHEMPROP_AUX_REF_RAE,
        "nb1070_ref_rae": NB1070_REF_RAE,
        "nb1130_ref_mean_bag": NB1130_MEAN_BAG_RAE,
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
    for k in ("rae_chemprop_aux_alone", "rae_nb1070_alone",
              "rae_combined_anchor",
              "per_seed_rae", "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_anchor", "delta_median_bag_vs_anchor",
              "beats_anchor", "beats_nb1130", "beats_nb1141", "verdict"):
        print(f"  {k}: {res.get(k)}")
