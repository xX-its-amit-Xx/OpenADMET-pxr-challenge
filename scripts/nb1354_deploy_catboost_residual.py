"""nb1354 -- DEPLOY artifact for nb1341 (CatBoost residual on MACCS-167 +
ChEMBL-kNN features) on the 513-row test set.

PRECEDENT
---------
nb1341 (diagnostic) 5-seed mean-bag pooled RAE on 253 unblind = 0.5420
(median-bag = 0.5395), honest cross-fit; LB-faithful anchor.

PROTOCOL
--------
1. Anchor = nb1070 (te_nb1070.npy on 513, nb1070_pred_oof.npy on 253).
2. residual_target = y_unb - nb1070_pred_oof (253,)
3. Feature matrix:
     X_test  (513, 169) = MACCS-167[513] + pred_chembl_pec50[513] + sim_chembl[513]
     X_unb   (253, 169) = X_test[unb_idx]
4. 5-seed CatBoost bag (seeds [0, 1, 7, 42, 137]):
     CatBoostRegressor(loss_function='MAE', depth=4, iterations=200,
                       learning_rate=0.05, l2_leaf_reg=5.0, verbose=False)
     Each seed fit on ALL 253 unblind rows (no CV), predicting residual
     on 513 test rows.
5. mean_bag_residual_513 = mean across 5 seeds (513,)
6. te_nb1354 = te_nb1070 + mean_bag_residual_513
7. Save submission CSV (SMILES + Molecule Name + pEC50, 513 rows).

NOTE
----
Per feedback_lb_two_regime_calibration / feedback_te_vs_pred_oof_protocol:
each inner CatBoost is fit on ALL 253 unblind rows, so in_RAE on te[unb_idx]
is in-sample optimistic. The LB-faithful anchor is the honest 0.5420 mean
(0.5395 median) cross-fit RAE from nb1341.

Outputs:
  data/processed/te_nb1354.npy                  (513,) float32
  submissions/nb1354_deploy_catboost.csv        (513 rows)
  data/processed/nb1354_summary.json
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

try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
    _CATBOOST_VERSION = __import__("catboost").__version__
except Exception as e:  # noqa: BLE001
    _CATBOOST_AVAILABLE = False
    _CATBOOST_VERSION = None
    _CATBOOST_IMPORT_ERR = repr(e)

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1354"
ANCHOR = "nb1070"
NB1341_TAG = "nb1341"

RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"

NB1341_HONEST_LB_ANCHOR_MEAN = 0.5420  # mean-bag cross-fit RAE on 253 unblind
NB1341_HONEST_LB_ANCHOR_MEDIAN = 0.5395  # median-bag cross-fit RAE on 253 unblind

SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def _catboost_params(seed: int) -> dict:
    return dict(
        loss_function="MAE",
        depth=4,
        iterations=200,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )


def _save_submission_csv(te_pred, te_smiles, te_names, csv_path: str,
                         label: str) -> dict:
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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1341 CatBoost residual bag on 513 test set")
    print(f"          anchor      = {ANCHOR}")
    print(f"          resid seeds = {RESID_SEEDS}")
    print(f"          features    = MACCS-167 + pred_chembl_pec50 + sim  (169)")
    print(f"          learner     = CatBoost MAE depth=4 iter=200 lr=0.05 l2=5")
    print(f"          honest cross-fit LB anchors:  mean={NB1341_HONEST_LB_ANCHOR_MEAN:.4f}  "
          f"median={NB1341_HONEST_LB_ANCHOR_MEDIAN:.4f}")
    print("=" * 78)

    if not _CATBOOST_AVAILABLE:
        raise RuntimeError(f"catboost not importable: {_CATBOOST_IMPORT_ERR}")
    print(f"[env] catboost {_CATBOOST_VERSION}")

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

    residual_target = y_unb - nb1070_oof
    print(f"[resid] target mean={residual_target.mean():+.4f}  "
          f"std={residual_target.std():.4f}")

    # ---- Load 513-row feature caches ----
    pred_chembl_path = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
    sim_chembl_path = DATA_PROCESSED / "sim_chembl_513.npy"
    if not (pred_chembl_path.exists() and sim_chembl_path.exists()):
        raise FileNotFoundError(
            f"Missing ChEMBL kNN caches: {pred_chembl_path}, {sim_chembl_path}. "
            f"Run nb1250 first to build them."
        )
    pred_chembl = np.load(pred_chembl_path).astype(np.float32)
    sim_chembl = np.load(sim_chembl_path).astype(np.float32)
    if pred_chembl.shape[0] != n_test or sim_chembl.shape[0] != n_test:
        raise ValueError(
            f"Cached ChEMBL feats shape mismatch: "
            f"pred={pred_chembl.shape}, sim={sim_chembl.shape}, expected ({n_test},)"
        )
    print(f"[feat] CACHED pred_chembl_pec50_513.npy + sim_chembl_513.npy loaded")
    print(f"   pred_chembl_pec50  mean={pred_chembl.mean():.3f}  "
          f"std={pred_chembl.std():.3f}  "
          f"min={pred_chembl.min():.3f}  max={pred_chembl.max():.3f}")
    print(f"   sim_chembl         mean={sim_chembl.mean():.3f}  "
          f"std={sim_chembl.std():.3f}  "
          f"min={sim_chembl.min():.3f}  max={sim_chembl.max():.3f}")

    # ---- Build (513, 169) deploy feature matrix + (253, 169) train slice ----
    X_maccs_te = np.load(MACCS_TE_PATH).astype(np.float32)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")

    X_test = np.concatenate(
        [X_maccs_te,
         pred_chembl.reshape(-1, 1).astype(np.float32),
         sim_chembl.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    X_unb = X_test[unb_idx]
    feat_dim = X_test.shape[1]
    print(f"[feat] X_test shape = {X_test.shape}  X_unb shape = {X_unb.shape}  "
          f"(MACCS-167 + pred_chembl + sim, dim={feat_dim})")

    # ---- 5-seed deploy bag (fit on ALL 253, predict 513 residual) ----
    print("\n" + "-" * 78)
    print(f"PER-SEED DEPLOY (CatBoost fit on ALL {n_unb} unblind, predict 513) -- 5-seed bag")
    print("-" * 78)
    per_seed_resid_513 = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_records = []
    for j, s in enumerate(RESID_SEEDS):
        mdl = CatBoostRegressor(**_catboost_params(s))
        mdl.fit(X_unb, residual_target)
        resid_pred_513 = mdl.predict(X_test).astype(np.float64)
        per_seed_resid_513[j] = resid_pred_513
        te_seed = te_nb1070 + resid_pred_513
        in_rae_s = float(rae(y_unb, te_seed[unb_idx]))
        per_seed_records.append({
            "seed": int(s),
            "in_rae_te_seed": in_rae_s,
            "resid_513_mean": float(resid_pred_513.mean()),
            "resid_513_std": float(resid_pred_513.std()),
            "resid_513_min": float(resid_pred_513.min()),
            "resid_513_max": float(resid_pred_513.max()),
        })
        print(f"   seed {s:3d}:  in_RAE(te_seed[unb]) = {in_rae_s:.4f}  "
              f"resid_513 mean={resid_pred_513.mean():+.4f} "
              f"std={resid_pred_513.std():.4f}")

    # ---- Mean-bag residual + final deploy ----
    te_residual_513 = per_seed_resid_513.mean(axis=0)
    te_nb1354 = te_nb1070 + te_residual_513
    in_rae_mean = float(rae(y_unb, te_nb1354[unb_idx]))

    # also compute median-bag deploy as a diagnostic (not saved as primary)
    te_residual_513_med = np.median(per_seed_resid_513, axis=0)
    te_nb1354_med = te_nb1070 + te_residual_513_med
    in_rae_median = float(rae(y_unb, te_nb1354_med[unb_idx]))

    print("\n" + "=" * 78)
    print("MEAN-BAG DEPLOY")
    print("=" * 78)
    print(f"   te_residual_513   mean={te_residual_513.mean():+.4f}  "
          f"std={te_residual_513.std():.4f}  "
          f"min={te_residual_513.min():+.4f}  max={te_residual_513.max():+.4f}")
    print(f"   te_nb1354         mean={te_nb1354.mean():.3f}  "
          f"std={te_nb1354.std():.3f}  "
          f"min={te_nb1354.min():.3f}  max={te_nb1354.max():.3f}")
    print(f"   in_RAE(unb)       = {in_rae_mean:.4f}  "
          f"(honest cross-fit LB anchor mean = {NB1341_HONEST_LB_ANCHOR_MEAN:.4f})")
    print(f"   in_RAE(median-bag, diag) = {in_rae_median:.4f}  "
          f"(honest cross-fit median = {NB1341_HONEST_LB_ANCHOR_MEDIAN:.4f})")

    # ---- Save artifacts ----
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb1354.astype(np.float32))
    print(f"[save] {te_path}")

    csv_path = os.path.join(SUBMISSIONS_DIR, f"{TAG}_deploy_catboost.csv")
    csv_info = _save_submission_csv(
        te_nb1354, te_smiles, te_names, csv_path, TAG
    )
    print(f"[save] {csv_path}  rows={csv_info['n_rows']}  "
          f"cols={csv_info['columns']}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": "maccs_cached_167+chembl_knn_2",
        "maccs_cache_test": str(MACCS_TE_PATH),
        "pred_chembl_path": str(pred_chembl_path),
        "sim_chembl_path": str(sim_chembl_path),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "resid_seeds": RESID_SEEDS,
        "feature_dim": int(feat_dim),
        "catboost_version": _CATBOOST_VERSION,
        "catboost_params_template": _catboost_params(0),
        "rae_anchor_oof_253": rae_anchor_oof,
        "rae_anchor_te_in_sample_253": rae_anchor_te_in,
        "residual_target_mean": float(residual_target.mean()),
        "residual_target_std": float(residual_target.std()),
        "pred_chembl_stats": {
            "mean": float(pred_chembl.mean()),
            "std": float(pred_chembl.std()),
            "min": float(pred_chembl.min()),
            "max": float(pred_chembl.max()),
        },
        "sim_chembl_stats": {
            "mean": float(sim_chembl.mean()),
            "std": float(sim_chembl.std()),
            "min": float(sim_chembl.min()),
            "max": float(sim_chembl.max()),
        },
        "per_seed_records": per_seed_records,
        "te_stats": {
            "mean": float(te_nb1354.mean()),
            "std": float(te_nb1354.std()),
            "min": float(te_nb1354.min()),
            "max": float(te_nb1354.max()),
        },
        "in_rae_mean_bag_253": in_rae_mean,
        "in_rae_median_bag_253_diag": in_rae_median,
        "crossfit_lb_anchor_nb1341_mean": NB1341_HONEST_LB_ANCHOR_MEAN,
        "crossfit_lb_anchor_nb1341_median": NB1341_HONEST_LB_ANCHOR_MEDIAN,
        "te_path": str(te_path),
        "csv_path": csv_path,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "POST-unblind deploy: each CatBoost is fit on ALL 253 unblind rows, "
            "so in_RAE on te[unb_idx] is in-sample optimistic. The LB-faithful "
            "anchor is the honest 0.5420 mean (0.5395 median) cross-fit RAE "
            "from nb1341."
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
    print(f"  te_mean: {res['te_stats']['mean']:.4f}")
    print(f"  te_std:  {res['te_stats']['std']:.4f}")
    print(f"  te_min:  {res['te_stats']['min']:.4f}")
    print(f"  te_max:  {res['te_stats']['max']:.4f}")
    print(f"  in_rae_253: {res['in_rae_mean_bag_253']:.4f}")
    print(f"  crossfit_lb_anchor_nb1341_mean: {res['crossfit_lb_anchor_nb1341_mean']:.4f}")
    print(f"  crossfit_lb_anchor_nb1341_median: {res['crossfit_lb_anchor_nb1341_median']:.4f}")
    print(f"  te_path: {res['te_path']}")
    print(f"  csv_path: {res['csv_path']}")
