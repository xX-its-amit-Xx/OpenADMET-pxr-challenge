"""nb1201 -- DEPLOY artifacts for nb1183 (MACCS residual bag), nb1192 (4-way
                mean), and nb1190 (bag-of-bags) on the 513-row test set.

Three deploys in one script so the shared anchors (te_nb1070, te_nb1140,
te_nb1162, te_nb1172) are loaded exactly once:

  (1) nb1183 deploy  -> te_nb1183 = te_nb1070 + mean-bag (5 LGBM Huber seeds)
                                    of residual on MACCS-167, each seed fit on
                                    all 253 unblind rows, predict on 513.
      Output:  data/processed/te_nb1183.npy
               submissions/nb1201_deploy_nb1183.csv

  (2) nb1192 deploy  -> te_nb1192 = (te_nb1140 + te_nb1162 + te_nb1172
                                     + te_nb1183) / 4
      Output:  data/processed/te_nb1192.npy
               submissions/nb1202_deploy_nb1192.csv

  (3) nb1190 deploy  -> bag-of-bags across 5 outer seeds {0,1,7,42,137}.
                        Per outer o: inner seeds = [o*1000 + b for b in BASES],
                        rebuild three component deploys (nb1130_o, nb1153_o,
                        nb1172_o) by fitting 5 LGBM seeds on 253 unblind,
                        predicting on 513, mean-bagging the 5 deploy residuals,
                        adding to te_nb1070 -- producing component_o on 513.
                        Triple naive mean per outer = (Ao + Bo + Co) / 3.
                        Final te_nb1190 = mean over the 5 per-outer triples.
      Output:  data/processed/te_nb1190.npy
               submissions/nb1203_deploy_nb1190.csv

Per feedback_lb_two_regime_calibration: each deploy is POST-unblind refit, so
in_RAE on te[unb_idx] is in-sample and optimistic. LB-faithful anchors are the
honest cross-fit RAEs reported by the matching diagnostic notebooks:
  nb1183 honest cross-fit RAE = 0.5513   (MACCS residual mean-bag)
  nb1192 honest cross-fit RAE = 0.5514   (4-way naive mean)
  nb1190 honest cross-fit RAE = 0.5499   (bag-of-bags triple)
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
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

TAG = "nb1201"
ANCHOR = "nb1070"

# Single-seed grid (nb1183 deploy uses BASE_SEEDS verbatim).
BASE_SEEDS = [0, 1, 7, 42, 137]

# Outer/inner grid for nb1190 bag-of-bags.
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASES = [0, 1, 7, 42, 137]   # inner = outer * 1000 + base

# Cached feature caches.
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"      # (513, 167) uint8
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"  # (513, 2048) uint8
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")        # X_mordred_test.npy

# Honest cross-fit reference anchors (from diagnostic summaries on 253 unblind).
NB1183_CROSSFIT_LB_ANCHOR = 0.5513
NB1192_CROSSFIT_LB_ANCHOR = 0.5514
NB1190_CROSSFIT_LB_ANCHOR = 0.5499

SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Shared LGBM Huber config (identical capacity to nb1130 / nb1153 / nb1172 / nb1183).
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Feature loaders.
# -----------------------------------------------------------------------------
def _load_maccs_test(n_test_expected: int) -> np.ndarray:
    if not MACCS_TE_PATH.exists():
        raise FileNotFoundError(f"MACCS test cache missing: {MACCS_TE_PATH}")
    X = np.load(MACCS_TE_PATH)
    if X.shape[0] != n_test_expected:
        raise ValueError(
            f"MACCS test cache shape mismatch: {X.shape} vs n_test={n_test_expected}"
        )
    if X.shape[1] not in (166, 167):
        raise ValueError(
            f"MACCS test cache unexpected width: {X.shape[1]}"
        )
    return X.astype(np.float32)


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    p = MORDRED_DIR / "X_mordred_test.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({p})"
        )
    X = np.load(p).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X.shape} vs n_test={n_test_expected}"
        )
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_atompair_test(n_test_expected: int) -> np.ndarray:
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}")
    X = np.load(ATOMPAIR_TE_PATH)
    if X.shape[0] != n_test_expected or X.shape[1] != 2048:
        raise ValueError(
            f"AtomPair test cache shape mismatch: {X.shape} vs (n_test={n_test_expected}, 2048)"
        )
    return X.astype(np.float32)


# -----------------------------------------------------------------------------
# Generic deploy primitive: fit one LGBM seed on (X_unb, residual), predict 513.
# -----------------------------------------------------------------------------
def _fit_one_predict_513(
    X_unb: np.ndarray, residual: np.ndarray,
    X_test: np.ndarray, seed: int
) -> np.ndarray:
    mdl = LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_test).astype(np.float64)


def _mean_bag_residual_513(
    X_unb: np.ndarray, residual: np.ndarray,
    X_test: np.ndarray, seeds: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_bag_residual_513, per_seed_residual_513 [K x 513])."""
    n_test = X_test.shape[0]
    per_seed = np.zeros((len(seeds), n_test), dtype=np.float64)
    for j, s in enumerate(seeds):
        per_seed[j] = _fit_one_predict_513(X_unb, residual, X_test, s)
    return per_seed.mean(axis=0), per_seed


# -----------------------------------------------------------------------------
# CSV save helper -- enforces 3-column convention + validation.
# -----------------------------------------------------------------------------
def _save_submission_csv(
    te_pred: np.ndarray, te_smiles, te_names, csv_path: str, label: str
) -> dict:
    assert te_pred.shape[0] == 513, (
        f"{label}: te_pred shape {te_pred.shape}, expected (513,)"
    )
    assert np.all(np.isfinite(te_pred)), f"{label}: te_pred has NaN/Inf"
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred.astype(np.float64),
    })
    assert len(sub) == 513, f"{label}: row count {len(sub)} != 513"
    assert list(sub.columns) == ["SMILES", "Molecule Name", "pEC50"], (
        f"{label}: column order wrong: {list(sub.columns)}"
    )
    assert sub.isna().sum().sum() == 0, f"{label}: CSV has NaN"
    sub.to_csv(csv_path, index=False)
    return {
        "csv_path": csv_path,
        "n_rows": int(len(sub)),
        "columns": list(sub.columns),
    }


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1183 (MACCS residual) + nb1192 (4-way mean) "
          f"+ nb1190 (bag-of-bags)")
    print(f"          anchor       = {ANCHOR} (te_{ANCHOR}.npy + {ANCHOR}_pred_oof.npy)")
    print(f"          base seeds   = {BASE_SEEDS}")
    print(f"          outer seeds  = {OUTER_SEEDS}  (nb1190 bag-of-bags)")
    print(f"          inner bases  = {INNER_BASES}  (inner = outer*1000 + base)")
    print(f"          LGBM:  depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child=20, obj=huber(alpha=1.0)")
    print(f"          honest cross-fit anchors:  nb1183={NB1183_CROSSFIT_LB_ANCHOR:.4f}  "
          f"nb1192={NB1192_CROSSFIT_LB_ANCHOR:.4f}  "
          f"nb1190={NB1190_CROSSFIT_LB_ANCHOR:.4f}")
    print("=" * 78)

    # ---- Load 513 test, unblind index + truth, anchors ----
    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_test = len(te_smiles)

    te_nb1070 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    nb1070_oof = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert te_nb1070.shape[0] == n_test
    assert nb1070_oof.shape[0] == n_unb

    rae_anchor_oof = float(rae(y_unb, nb1070_oof))
    rae_anchor_te_in = float(rae(y_unb, te_nb1070[unb_idx]))
    print(f"[load] te_{ANCHOR}.npy shape={te_nb1070.shape}  "
          f"in_RAE(unb_idx) = {rae_anchor_te_in:.4f}")
    print(f"[load] {ANCHOR}_pred_oof.npy shape={nb1070_oof.shape}  "
          f"pooled RAE = {rae_anchor_oof:.4f}")

    # Existing 513 deploy anchors needed for nb1192.
    te_nb1140 = np.load(DATA_PROCESSED / "te_nb1140.npy").astype(np.float64)
    te_nb1162 = np.load(DATA_PROCESSED / "te_nb1162.npy").astype(np.float64)
    te_nb1172 = np.load(DATA_PROCESSED / "te_nb1172.npy").astype(np.float64)
    assert te_nb1140.shape == (n_test,)
    assert te_nb1162.shape == (n_test,)
    assert te_nb1172.shape == (n_test,)
    in_rae_1140 = float(rae(y_unb, te_nb1140[unb_idx]))
    in_rae_1162 = float(rae(y_unb, te_nb1162[unb_idx]))
    in_rae_1172 = float(rae(y_unb, te_nb1172[unb_idx]))
    print(f"[load] te_nb1140  in_RAE(unb)={in_rae_1140:.4f}")
    print(f"[load] te_nb1162  in_RAE(unb)={in_rae_1162:.4f}")
    print(f"[load] te_nb1172  in_RAE(unb)={in_rae_1172:.4f}")

    # Residual target (constant across seeds for all deploys).
    residual_target = y_unb - nb1070_oof
    print(f"[resid] target mean={residual_target.mean():+.4f}  "
          f"std={residual_target.std():.4f}  "
          f"min={residual_target.min():+.4f}  max={residual_target.max():+.4f}")

    # =========================================================================
    # (1) nb1183 DEPLOY -- MACCS residual mean-bag.
    # =========================================================================
    print("\n" + "=" * 78)
    print("(1) nb1183 DEPLOY -- MACCS-167 residual mean-bag")
    print("=" * 78)
    X_maccs_test = _load_maccs_test(n_test)
    X_maccs_unb = X_maccs_test[unb_idx]
    print(f"[feat] X_maccs_test shape = {X_maccs_test.shape}")
    print(f"[feat] X_maccs_unb  shape = {X_maccs_unb.shape}  "
          f"bit_density={X_maccs_unb.mean():.4f}")

    print(f"\n[fit] 5 LGBM Huber seeds on n={n_unb} unblind, residual = "
          f"y_unb - nb1070_oof, predict 513 ...")
    mean_residual_513_1183, per_seed_residual_513_1183 = _mean_bag_residual_513(
        X_maccs_unb, residual_target, X_maccs_test, BASE_SEEDS
    )
    for j, s in enumerate(BASE_SEEDS):
        r513 = per_seed_residual_513_1183[j]
        te_seed = te_nb1070 + r513
        in_rae_s = float(rae(y_unb, te_seed[unb_idx]))
        print(f"   seed {s:3d}:  resid_513 mean={r513.mean():+.4f} "
              f"std={r513.std():.4f}  in_RAE(te)={in_rae_s:.4f}")
    print(f"[bag] mean_residual_513 mean={mean_residual_513_1183.mean():+.4f}  "
          f"std={mean_residual_513_1183.std():.4f}  "
          f"min={mean_residual_513_1183.min():+.4f}  "
          f"max={mean_residual_513_1183.max():+.4f}")

    te_nb1183 = te_nb1070 + mean_residual_513_1183
    in_rae_1183 = float(rae(y_unb, te_nb1183[unb_idx]))
    te_nb1183_path = DATA_PROCESSED / "te_nb1183.npy"
    np.save(te_nb1183_path, te_nb1183.astype(np.float32))
    csv_1183_path = os.path.join(SUBMISSIONS_DIR, "nb1201_deploy_nb1183.csv")
    csv_1183_info = _save_submission_csv(
        te_nb1183, te_smiles, te_names, csv_1183_path, "nb1183"
    )
    print(f"[deploy] te_nb1183 mean={te_nb1183.mean():.3f}  std={te_nb1183.std():.3f}  "
          f"min={te_nb1183.min():.3f}  max={te_nb1183.max():.3f}")
    print(f"[deploy] in_RAE(te_nb1183[unb_idx]) = {in_rae_1183:.4f}  "
          f"(honest cross-fit LB anchor = {NB1183_CROSSFIT_LB_ANCHOR:.4f})")
    print(f"[save] {te_nb1183_path}")
    print(f"[save] {csv_1183_path}  rows={csv_1183_info['n_rows']}")

    # =========================================================================
    # (2) nb1192 DEPLOY -- 4-way naive mean.
    # =========================================================================
    print("\n" + "=" * 78)
    print("(2) nb1192 DEPLOY -- (te_nb1140 + te_nb1162 + te_nb1172 + te_nb1183) / 4")
    print("=" * 78)
    te_nb1192 = (te_nb1140 + te_nb1162 + te_nb1172 + te_nb1183) / 4.0
    in_rae_1192 = float(rae(y_unb, te_nb1192[unb_idx]))
    te_nb1192_path = DATA_PROCESSED / "te_nb1192.npy"
    np.save(te_nb1192_path, te_nb1192.astype(np.float32))
    csv_1192_path = os.path.join(SUBMISSIONS_DIR, "nb1202_deploy_nb1192.csv")
    csv_1192_info = _save_submission_csv(
        te_nb1192, te_smiles, te_names, csv_1192_path, "nb1192"
    )
    print(f"[deploy] te_nb1192 mean={te_nb1192.mean():.3f}  std={te_nb1192.std():.3f}  "
          f"min={te_nb1192.min():.3f}  max={te_nb1192.max():.3f}")
    print(f"[deploy] in_RAE(te_nb1192[unb_idx]) = {in_rae_1192:.4f}  "
          f"(honest cross-fit LB anchor = {NB1192_CROSSFIT_LB_ANCHOR:.4f})")
    print(f"[save] {te_nb1192_path}")
    print(f"[save] {csv_1192_path}  rows={csv_1192_info['n_rows']}")

    # =========================================================================
    # (3) nb1190 DEPLOY -- bag-of-bags across 5 outer seeds, three components per outer.
    # =========================================================================
    print("\n" + "=" * 78)
    print("(3) nb1190 DEPLOY -- bag-of-bags (5 outer x 3 components x 5 inner seeds)")
    print("=" * 78)
    # Load remaining feature caches once (Morgan+RDKit on 513 needs SMILES too).
    print(f"[feat] computing combined(Morgan+RDKit) on n={n_test} test SMILES ...")
    X_mr_test = impute(combined(te_smiles.tolist()))
    X_mr_unb = X_mr_test[unb_idx]
    print(f"[feat] X_mr_test shape={X_mr_test.shape}  X_mr_unb shape={X_mr_unb.shape}")

    print(f"[feat] loading cached Mordred (1533) ...")
    X_mor_test = _load_mordred_test(n_test)
    X_mor_unb = X_mor_test[unb_idx]
    print(f"[feat] X_mor_test shape={X_mor_test.shape}  X_mor_unb shape={X_mor_unb.shape}")

    print(f"[feat] loading cached AtomPair (2048) ...")
    X_ap_test = _load_atompair_test(n_test)
    X_ap_unb = X_ap_test[unb_idx]
    print(f"[feat] X_ap_test shape={X_ap_test.shape}  X_ap_unb shape={X_ap_unb.shape}")

    per_outer_triple_513 = np.zeros((len(OUTER_SEEDS), n_test), dtype=np.float64)
    per_outer_records: list[dict] = []
    print("\n" + "-" * 78)
    print("PER-OUTER DEPLOY REBUILD")
    print("-" * 78)
    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 1000 + int(b) for b in INNER_BASES]

        # Component A: nb1130 deploy under inner_seeds (Morgan+RDKit residual).
        meanA_resid_513, _ = _mean_bag_residual_513(
            X_mr_unb, residual_target, X_mr_test, inner_seeds
        )
        compA_513 = te_nb1070 + meanA_resid_513

        # Component B: nb1153 deploy under inner_seeds (Mordred residual).
        meanB_resid_513, _ = _mean_bag_residual_513(
            X_mor_unb, residual_target, X_mor_test, inner_seeds
        )
        compB_513 = te_nb1070 + meanB_resid_513

        # Component C: nb1172 deploy under inner_seeds (AtomPair residual).
        meanC_resid_513, _ = _mean_bag_residual_513(
            X_ap_unb, residual_target, X_ap_test, inner_seeds
        )
        compC_513 = te_nb1070 + meanC_resid_513

        # Triple naive mean (per-outer).
        triple_513 = (compA_513 + compB_513 + compC_513) / 3.0
        per_outer_triple_513[oi] = triple_513

        in_rae_A = float(rae(y_unb, compA_513[unb_idx]))
        in_rae_B = float(rae(y_unb, compB_513[unb_idx]))
        in_rae_C = float(rae(y_unb, compC_513[unb_idx]))
        in_rae_triple = float(rae(y_unb, triple_513[unb_idx]))
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "in_rae_compA_nb1130": in_rae_A,
            "in_rae_compB_nb1153": in_rae_B,
            "in_rae_compC_nb1172": in_rae_C,
            "in_rae_triple_mean": in_rae_triple,
            "elapsed_sec": round(time.time() - t_outer, 1),
        })
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     in_RAE(compA={in_rae_A:.4f}, compB={in_rae_B:.4f}, "
              f"compC={in_rae_C:.4f}, triple={in_rae_triple:.4f})  "
              f"elapsed={time.time() - t_outer:.1f}s")

    # Final bag-of-bags mean across outer seeds.
    te_nb1190 = per_outer_triple_513.mean(axis=0)
    in_rae_1190 = float(rae(y_unb, te_nb1190[unb_idx]))
    te_nb1190_path = DATA_PROCESSED / "te_nb1190.npy"
    np.save(te_nb1190_path, te_nb1190.astype(np.float32))
    csv_1190_path = os.path.join(SUBMISSIONS_DIR, "nb1203_deploy_nb1190.csv")
    csv_1190_info = _save_submission_csv(
        te_nb1190, te_smiles, te_names, csv_1190_path, "nb1190"
    )
    print(f"\n[bag-of-bags] te_nb1190 mean={te_nb1190.mean():.3f}  "
          f"std={te_nb1190.std():.3f}  min={te_nb1190.min():.3f}  "
          f"max={te_nb1190.max():.3f}")
    print(f"[bag-of-bags] in_RAE(te_nb1190[unb_idx]) = {in_rae_1190:.4f}  "
          f"(honest cross-fit LB anchor = {NB1190_CROSSFIT_LB_ANCHOR:.4f})")
    print(f"[save] {te_nb1190_path}")
    print(f"[save] {csv_1190_path}  rows={csv_1190_info['n_rows']}")

    # =========================================================================
    # Summary.
    # =========================================================================
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  nb1183 deploy:  te mean/std/min/max = "
          f"{te_nb1183.mean():.3f}/{te_nb1183.std():.3f}/"
          f"{te_nb1183.min():.3f}/{te_nb1183.max():.3f}  "
          f"in_RAE={in_rae_1183:.4f}  (cross-fit anchor {NB1183_CROSSFIT_LB_ANCHOR:.4f})")
    print(f"  nb1192 deploy:  te mean/std/min/max = "
          f"{te_nb1192.mean():.3f}/{te_nb1192.std():.3f}/"
          f"{te_nb1192.min():.3f}/{te_nb1192.max():.3f}  "
          f"in_RAE={in_rae_1192:.4f}  (cross-fit anchor {NB1192_CROSSFIT_LB_ANCHOR:.4f})")
    print(f"  nb1190 deploy:  te mean/std/min/max = "
          f"{te_nb1190.mean():.3f}/{te_nb1190.std():.3f}/"
          f"{te_nb1190.min():.3f}/{te_nb1190.max():.3f}  "
          f"in_RAE={in_rae_1190:.4f}  (cross-fit anchor {NB1190_CROSSFIT_LB_ANCHOR:.4f})")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "rae_anchor_oof_253": rae_anchor_oof,
        "rae_anchor_te_in_sample_253": rae_anchor_te_in,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "deploys": {
            "nb1183": {
                "te_path": str(te_nb1183_path),
                "csv_path": csv_1183_path,
                "te_mean": float(te_nb1183.mean()),
                "te_std": float(te_nb1183.std()),
                "te_min": float(te_nb1183.min()),
                "te_max": float(te_nb1183.max()),
                "in_rae_253": in_rae_1183,
                "crossfit_lb_anchor": NB1183_CROSSFIT_LB_ANCHOR,
                "feature_source": "maccs_cached_167",
                "seeds": BASE_SEEDS,
            },
            "nb1192": {
                "te_path": str(te_nb1192_path),
                "csv_path": csv_1192_path,
                "te_mean": float(te_nb1192.mean()),
                "te_std": float(te_nb1192.std()),
                "te_min": float(te_nb1192.min()),
                "te_max": float(te_nb1192.max()),
                "in_rae_253": in_rae_1192,
                "crossfit_lb_anchor": NB1192_CROSSFIT_LB_ANCHOR,
                "components": [
                    "te_nb1140", "te_nb1162", "te_nb1172", "te_nb1183"
                ],
                "in_rae_components": {
                    "te_nb1140": in_rae_1140,
                    "te_nb1162": in_rae_1162,
                    "te_nb1172": in_rae_1172,
                    "te_nb1183": in_rae_1183,
                },
            },
            "nb1190": {
                "te_path": str(te_nb1190_path),
                "csv_path": csv_1190_path,
                "te_mean": float(te_nb1190.mean()),
                "te_std": float(te_nb1190.std()),
                "te_min": float(te_nb1190.min()),
                "te_max": float(te_nb1190.max()),
                "in_rae_253": in_rae_1190,
                "crossfit_lb_anchor": NB1190_CROSSFIT_LB_ANCHOR,
                "outer_seeds": OUTER_SEEDS,
                "inner_bases": INNER_BASES,
                "per_outer_records": per_outer_records,
            },
        },
        "lgbm_params_template": _lgbm_params(0),
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "POST-unblind deploy artifacts: each fit uses ALL 253 unblind rows, "
            "so in_RAE on te[unb_idx] is in-sample and optimistic. LB-faithful "
            "numbers are the matching honest cross-fit anchors."
        ),
    }
    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {summary_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== STRUCTURED SUMMARY ====")
    for tag in ("nb1183", "nb1192", "nb1190"):
        d = res["deploys"][tag]
        print(f"  {tag}:")
        for k in ("te_mean", "te_std", "te_min", "te_max",
                  "in_rae_253", "crossfit_lb_anchor",
                  "te_path", "csv_path"):
            print(f"    {k}: {d.get(k)}")
