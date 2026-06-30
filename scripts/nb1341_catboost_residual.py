"""nb1341 -- CatBoost residual on MACCS-167 + ChEMBL-kNN features.

Hypothesis:
    CatBoost's Ordered Boosting + Symmetric (oblivious) Trees use a different
    gradient-estimation scheme than LightGBM's leaf-wise standard boosting.
    On the (253, 169) residual fit anchored at nb1070, this should extract a
    slightly different (decorrelated) residual signal vs nb1242's LGBM bag at
    pooled RAE 0.5431.  This is a *learner-axis* probe -- features and anchor
    are identical to nb1242, only the residual learner changes.

Pipeline:
    Anchor:    nb1070_pred_oof  (253,)
    Residual:  y_unb - nb1070_pred_oof
    Features:  MACCS-167[unb_idx]   (253, 167)
            ++ pred_chembl_pec50[unb_idx]   (253, 1)   from nb1242 cache
            ++ sim_chembl[unb_idx]          (253, 1)   from nb1242 cache
               -> X_unb shape (253, 169)
    Learner:   CatBoostRegressor(loss_function='MAE', depth=4,
                                  n_estimators=200, learning_rate=0.05,
                                  l2_leaf_reg=5, verbose=False)
    Bag:       5 seeds [0, 1, 7, 42, 137], 5-fold cross-fit per seed
    Pool:      mean across seeds -> RAE; verdict at 0.003 margin vs nb1242
               (0.5431).

Outputs:
    scripts/nb1341_catboost_residual.py            (this file)
    data/processed/nb1341_summary.json
    data/processed/nb1341_mean_bag_oof.npy         (253,) float32
    data/processed/nb1341_per_seed_corrected_oof.npy (5, 253) float32
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
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

# CatBoost availability gate -- if missing, the orchestrator should
# pip-install it before invoking this script.
try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
    _CATBOOST_VERSION = __import__("catboost").__version__
except Exception as e:  # noqa: BLE001
    _CATBOOST_AVAILABLE = False
    _CATBOOST_VERSION = None
    _CATBOOST_IMPORT_ERR = repr(e)

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1341"
ANCHOR = "nb1070"
NB1242_TAG = "nb1242"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

NB1070_REF = 0.5771
NB1242_REF = 0.5431  # ChEMBL kNN feature shallow LGBM bag pooled RAE
DECISION_MARGIN = 0.003


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


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = CatBoostRegressor(**_catboost_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _write_unavailable_summary(t0: float) -> dict:
    summary = {
        "tag": TAG,
        "status": "catboost_unavailable",
        "catboost_version": None,
        "import_error": _CATBOOST_IMPORT_ERR,
        "anchor": ANCHOR,
        "nb1242_ref": NB1242_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    return summary


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CatBoost residual on MACCS-167 + ChEMBL kNN features")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + pred_chembl + sim  (169)")
    print(f"          learner  = CatBoost MAE depth=4 iter=200 lr=0.05 l2=5")
    print("=" * 78)

    if not _CATBOOST_AVAILABLE:
        print(f"[err] catboost not importable: {_CATBOOST_IMPORT_ERR}")
        return _write_unavailable_summary(t0)
    print(f"[env] catboost {_CATBOOST_VERSION}")

    # ---- Load anchor + truth ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Feature cache -- reuse nb1242 ChEMBL kNN outputs ----
    maccs_te = np.load(DATA_PROCESSED / "te_maccs.npy")
    if maccs_te.shape[0] != 513:
        raise ValueError(f"te_maccs shape mismatch: {maccs_te.shape}")
    X_maccs_unb = maccs_te[unb_idx].astype(np.float32)
    print(f"[feat] MACCS unb shape = {X_maccs_unb.shape}")

    pred_chembl_513 = np.load(
        DATA_PROCESSED / "pred_chembl_pec50_513.npy"
    ).astype(np.float32)
    sim_chembl_513 = np.load(DATA_PROCESSED / "sim_chembl_513.npy").astype(np.float32)
    if pred_chembl_513.shape != (513,) or sim_chembl_513.shape != (513,):
        raise ValueError(
            f"ChEMBL kNN cache shape mismatch: pred={pred_chembl_513.shape} "
            f"sim={sim_chembl_513.shape}"
        )
    pred_chembl_unb = pred_chembl_513[unb_idx]
    sim_chembl_unb = sim_chembl_513[unb_idx]
    print(f"[feat] pred_chembl_pec50 unb mean={pred_chembl_unb.mean():.3f} "
          f"std={pred_chembl_unb.std():.3f}")
    print(f"[feat] sim_chembl       unb mean={sim_chembl_unb.mean():.3f} "
          f"std={sim_chembl_unb.std():.3f}")

    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_chembl_unb.reshape(-1, 1),
            sim_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"[feat] residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + pred_chembl + sim)")

    # ---- Per-seed CatBoost cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (CatBoost MAE depth=4 iter=200, "
          f"dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_anchor = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_anchor,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_anchor:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())

    # ---- Pearson vs nb1242 mean-bag OOF (learner-axis decorrelation probe) ----
    pearson_vs_nb1242 = None
    nb1242_oof = None
    nb1242_path = DATA_PROCESSED / f"{NB1242_TAG}_mean_bag_oof.npy"
    if nb1242_path.exists():
        nb1242_oof = np.load(nb1242_path).astype(np.float64)
        if nb1242_oof.shape == (n_unb,):
            r, _ = pearsonr(mean_bag_oof, nb1242_oof)
            pearson_vs_nb1242 = float(r)
        else:
            print(f"[warn] {NB1242_TAG} shape mismatch: {nb1242_oof.shape}")
    else:
        print(f"[warn] {nb1242_path} missing -- cannot compute Pearson")

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})  "
          f"(d_vs_nb1242 = {rae_mean_bag - NB1242_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}")
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  "
          f"(MACCS+ChEMBL LGBM bag)")
    if pearson_vs_nb1242 is not None:
        print(f"   Pearson(mean_bag, nb1242_mean_bag) = {pearson_vs_nb1242:.4f}")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "CATBOOST_RESIDUAL_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN:
        verdict = "CATBOOST_RESIDUAL_TIES_NB1242_FLAT"
    elif beats_nb1070:
        verdict = "CATBOOST_RESIDUAL_HELPS_NB1070_BUT_WORSE_THAN_NB1242"
    else:
        verdict = "CATBOOST_RESIDUAL_HURTS_VS_NB1070"
    print(f"   verdict                = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "status": "ok",
        "catboost_version": _CATBOOST_VERSION,
        "anchor": ANCHOR,
        "features": "MACCS-167 + pred_chembl_pec50 + sim_chembl",
        "feature_dim": feat_dim,
        "feature_cache_source": NB1242_TAG,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "catboost_loss_function": "MAE",
        "catboost_depth": 4,
        "catboost_iterations": 200,
        "catboost_learning_rate": 0.05,
        "catboost_l2_leaf_reg": 5.0,
        "n_unb": n_unb,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "pearson_vs_nb1242_mean_bag": pearson_vs_nb1242,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1242": bool(beats_nb1242),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1242_ref": NB1242_REF,
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
        "status", "catboost_version",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1242",
        "pearson_vs_nb1242_mean_bag",
        "beats_nb1070", "beats_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
