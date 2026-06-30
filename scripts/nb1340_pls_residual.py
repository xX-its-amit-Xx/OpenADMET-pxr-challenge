"""nb1340 -- Partial Least Squares (PLS) regression residual on MACCS + ChEMBL features.

Hypothesis:
    PLS is the classical chemometrics tool for high-dim regression. It builds
    latent components that maximize covariance between X and y, may extract
    correlation structure tree splits miss.

Protocol:
    1. Anchor nb1070_pred_oof.npy. Residual target = y_unb - nb1070_pred_oof.
    2. Features: MACCS-167 + pred_chembl_pec50 + sim_chembl = 169 cols.
       Standardize per fold.
    3. 5-seed bag: for each seed, 5-fold cross-fit PLSRegression(n_components
       in {3,5,7,10,15}, picked via inner 5-fold CV on training fold).
    4. Pool mean_bag corrected RAE.
    5. Verdict at 0.003 margin vs nb1242 (0.5431, LGBM same features).
    6. Compute Pearson of nb1340 OOF vs nb1242 OOF.

Outputs:
    scripts/nb1340_pls_residual.py
    data/processed/nb1340_summary.json
    data/processed/nb1340_mean_bag_oof.npy
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1340"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
PLS_COMPONENT_GRID = [3, 5, 7, 10, 15]
INNER_FOLDS = 5

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
PRED_CHEMBL_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

NB1070_REF = 0.5771
NB1242_REF = 0.5431      # LGBM residual bag on same MACCS+ChEMBL features
DECISION_MARGIN = 0.003


def _pick_n_components_inner_cv(X_tr: np.ndarray, y_tr: np.ndarray,
                                grid: list[int], seed: int) -> tuple[int, dict]:
    """Inner 5-fold CV across n_components grid -> pick lowest mean MSE.
    Returns (best_n, per_n_records)."""
    n = len(y_tr)
    kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed + 1000)
    per_n_mse: dict[int, list[float]] = {nc: [] for nc in grid}
    for tr_loc, va_loc in kf.split(np.arange(n)):
        Xi_tr = X_tr[tr_loc]
        Xi_va = X_tr[va_loc]
        yi_tr = y_tr[tr_loc]
        yi_va = y_tr[va_loc]
        sc = StandardScaler(with_mean=True, with_std=True)
        Xi_tr_s = sc.fit_transform(Xi_tr)
        Xi_va_s = sc.transform(Xi_va)
        for nc in grid:
            try:
                pls = PLSRegression(n_components=nc, scale=False, max_iter=500)
                pls.fit(Xi_tr_s, yi_tr)
                yhat = pls.predict(Xi_va_s).ravel()
                mse = float(np.mean((yi_va - yhat) ** 2))
            except Exception:
                mse = np.inf
            per_n_mse[nc].append(mse)
    per_n_mean = {nc: float(np.mean(per_n_mse[nc])) for nc in grid}
    # Pick min mean MSE
    best_n = min(per_n_mean.items(), key=lambda kv: kv[1])[0]
    return int(best_n), per_n_mean


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int):
    """Outer 5-fold cross-fit. Inner 5-fold CV picks n_components per fold.
    Returns (oof, picked_per_fold list of ints, inner_mse_records)."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    picked = []
    inner_records = []
    for fold_i, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        X_tr = X[tr_loc]
        X_va = X[va_loc]
        y_tr = residual[tr_loc]
        best_n, per_n_mean = _pick_n_components_inner_cv(
            X_tr, y_tr, PLS_COMPONENT_GRID, seed
        )
        picked.append(best_n)
        inner_records.append({
            "fold": fold_i,
            "best_n_components": best_n,
            "inner_mse_per_n": per_n_mean,
        })
        sc = StandardScaler(with_mean=True, with_std=True)
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)
        pls = PLSRegression(n_components=best_n, scale=False, max_iter=500)
        pls.fit(X_tr_s, y_tr)
        oof[va_loc] = pls.predict(X_va_s).ravel()
    return oof, picked, inner_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PLS regression residual on MACCS+ChEMBL features (169 dim)")
    print(f"          seeds = {RESID_SEEDS}  outer_folds = {RESID_FOLDS}  "
          f"inner_folds = {INNER_FOLDS}")
    print(f"          n_components grid = {PLS_COMPONENT_GRID}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] n_unb={n_unb}  {ANCHOR} RAE = {rae_anchor:.4f}  "
          f"residual mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Feature matrix: MACCS-167 + pred_chembl + sim_chembl ----
    X_maccs_te = np.load(MACCS_TE_PATH).astype(np.float32)
    pred_chembl = np.load(PRED_CHEMBL_PATH).astype(np.float32)
    sim_chembl = np.load(SIM_CHEMBL_PATH).astype(np.float32)
    n_test = X_maccs_te.shape[0]
    if pred_chembl.shape[0] != n_test or sim_chembl.shape[0] != n_test:
        raise ValueError(
            f"Cache shape mismatch: maccs={X_maccs_te.shape}, "
            f"pred_chembl={pred_chembl.shape}, sim={sim_chembl.shape}"
        )

    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    pred_chembl_unb = pred_chembl[unb_idx].astype(np.float32)
    sim_chembl_unb = sim_chembl[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_chembl_unb.reshape(-1, 1),
            sim_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float64)
    feat_dim = X_unb.shape[1]
    print(f"[feat] residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + pred_chembl + sim)")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED PLS RESIDUAL CROSS-FIT (dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s, picked_s, inner_recs = _residual_cross_fit_one_seed(
            X_unb, residual, s
        )
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "best_n_components_per_fold": [int(x) for x in picked_s],
            "inner_cv_records": inner_recs,
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"picked_nc = {picked_s}  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())

    # ---- Pearson vs nb1242 ----
    nb1242_oof = np.load(DATA_PROCESSED / "nb1242_mean_bag_oof.npy").astype(np.float64)
    if nb1242_oof.shape[0] != n_unb:
        raise ValueError(f"nb1242 shape mismatch: {nb1242_oof.shape}")
    pearson_vs_nb1242, p_vs_nb1242 = pearsonr(mean_bag_oof, nb1242_oof)
    pearson_vs_nb1242 = float(pearson_vs_nb1242)
    p_vs_nb1242 = float(p_vs_nb1242)

    # Also Pearson of residuals (more meaningful for diversity)
    resid_nb1340 = mean_bag_oof - anchor
    resid_nb1242 = nb1242_oof - anchor
    pearson_resid_vs_nb1242, _ = pearsonr(resid_nb1340, resid_nb1242)
    pearson_resid_vs_nb1242 = float(pearson_resid_vs_nb1242)

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}")
    print(f"   nb1242 ref (LGBM)      = {NB1242_REF:.4f}")
    print(f"   Pearson(corrected_oof, nb1242_oof) = {pearson_vs_nb1242:.4f}")
    print(f"   Pearson(resid_only, nb1242_resid)   = "
          f"{pearson_resid_vs_nb1242:.4f}")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "PLS_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN:
        if beats_nb1070:
            verdict = "PLS_TIES_NB1242_DIVERSITY_CANDIDATE"
        else:
            verdict = "PLS_TIES_NB1242_BUT_DOES_NOT_BEAT_NB1070"
    elif beats_nb1070:
        verdict = "PLS_HELPS_NB1070_BUT_WORSE_THAN_NB1242"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "PLS_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "PLS_HURTS_NB1070"
    print(f"   verdict                = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    # Aggregate picked_n histogram across all (seed, fold) pairs
    all_picks = []
    for rec in per_seed_records:
        all_picks.extend(rec["best_n_components_per_fold"])
    pick_hist = {int(nc): int(all_picks.count(nc)) for nc in PLS_COMPONENT_GRID}

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "method": "PLSRegression with per-fold inner-CV n_components selection",
        "features": "MACCS-167 + pred_chembl_pec50 + sim_chembl",
        "feature_dim": feat_dim,
        "n_components_grid": PLS_COMPONENT_GRID,
        "inner_folds": INNER_FOLDS,
        "outer_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
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
        "pearson_vs_nb1242_corrected": pearson_vs_nb1242,
        "pearson_vs_nb1242_p_value": p_vs_nb1242,
        "pearson_vs_nb1242_residual_only": pearson_resid_vs_nb1242,
        "picked_n_components_histogram": pick_hist,
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
        "rae_anchor_nb1070",
        "per_seed_rae",
        "rae_per_seed_mean",
        "rae_per_seed_std",
        "rae_mean_bag",
        "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1242",
        "pearson_vs_nb1242_corrected",
        "pearson_vs_nb1242_residual_only",
        "picked_n_components_histogram",
        "beats_nb1070",
        "beats_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
