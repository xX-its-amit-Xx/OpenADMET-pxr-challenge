"""nb1280 -- Ridge regression residual learner on MACCS+ChEMBL features.

HYPOTHESIS:
    LGBM tree splits on bit-features are good for substructure rules but
    miss continuous-correlation structure.  Ridge linear regression
    captures linear-feature additivity differently -- may extract
    complementary signal from kNN-derived ChEMBL features (which are
    continuous), serving as an orthogonal residual learner to nb1242.

PROTOCOL:
    1. Anchor = nb1070_pred_oof.npy on 253 unblind.
       Residual target = y_unb - nb1070_pred_oof.
    2. Features (169 cols):
         MACCS-167 (unblind slice of te_maccs.npy)
       + pred_chembl_pec50 (kNN-mean from data/processed/pred_chembl_pec50_513.npy)
       + sim_chembl (mean k=5 Tanimoto sim from data/processed/sim_chembl_513.npy)
       Standardize each column to mean 0 / std 1 (training-fold statistics only).
    3. 5-seed bag: [0, 1, 7, 42, 137]
       Ridge alpha CV per fold over [0.1, 1.0, 10.0, 100.0]
       (inner CV picks alpha minimizing nested-RAE).
    4. 5-fold cross-fit per seed, pool mean_bag RAE.
    5. Verdict at 0.003 margin vs nb1242 (0.5431, LGBM on same features).
    6. Pearson(nb1280_mean_bag_oof, nb1242_mean_bag_oof) for orthogonality.

Outputs:
    scripts/nb1280_ridge_residual.py
    data/processed/nb1280_summary.json
    data/processed/nb1280_mean_bag_oof.npy
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1280"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
INNER_FOLDS = 5

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
PRED_CHEMBL_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

NB1070_REF = 0.5771
NB1242_REF = 0.5431       # LGBM on same MACCS + ChEMBL features
DECISION_MARGIN = 0.003


def _pick_alpha_inner(X_tr: np.ndarray, y_tr: np.ndarray,
                      seed: int) -> tuple[float, dict]:
    """Pick alpha by inner 5-fold CV minimizing pooled RAE on residual target."""
    n = len(y_tr)
    kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed + 991)
    alpha_scores: dict[float, float] = {}
    for alpha in ALPHAS:
        oof = np.full(n, np.nan, dtype=np.float64)
        for in_tr, in_va in kf.split(np.arange(n)):
            sc = StandardScaler()
            X_in_tr = sc.fit_transform(X_tr[in_tr])
            X_in_va = sc.transform(X_tr[in_va])
            mdl = Ridge(alpha=alpha, random_state=seed, fit_intercept=True)
            mdl.fit(X_in_tr, y_tr[in_tr])
            oof[in_va] = mdl.predict(X_in_va)
        # Use MSE on residual target as the alpha-selection score
        # (RAE on raw residual is ill-defined when residual mean ~0;
        # MSE is a faithful proxy and stable across folds).
        mse = float(np.mean((y_tr - oof) ** 2))
        alpha_scores[alpha] = mse
    best_alpha = min(alpha_scores, key=alpha_scores.get)
    return best_alpha, alpha_scores


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int):
    """One-seed nested cross-fit. Inner CV picks alpha per outer fold."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    for fold_i, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        X_tr_raw = X[tr_loc]
        X_va_raw = X[va_loc]
        y_tr = residual[tr_loc]

        best_alpha, alpha_scores = _pick_alpha_inner(X_tr_raw, y_tr, seed)

        # Refit on full outer-train fold with picked alpha
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_raw)
        X_va_s = sc.transform(X_va_raw)
        mdl = Ridge(alpha=best_alpha, random_state=seed, fit_intercept=True)
        mdl.fit(X_tr_s, y_tr)
        oof[va_loc] = mdl.predict(X_va_s)

        fold_records.append({
            "fold": int(fold_i),
            "alpha_picked": float(best_alpha),
            "alpha_scores": {str(a): float(s) for a, s in alpha_scores.items()},
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
        })
    return oof, fold_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Ridge residual learner on MACCS + ChEMBL features")
    print(f"          anchor   = {ANCHOR}")
    print(f"          seeds    = {RESID_SEEDS}  outer_folds = {RESID_FOLDS}")
    print(f"          alphas   = {ALPHAS}  inner_folds = {INNER_FOLDS}")
    print(f"          features = MACCS-167 + pred_chembl_pec50 + sim  (169)")
    print(f"          standardize: per-column StandardScaler (train-only stats)")
    print("=" * 78)

    # ---- Load anchor + truth ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb = {n_unb}")

    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")

    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Build feature matrix on 253 unblind ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] < n_unb:
        raise ValueError(f"MACCS shape: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"[feat] MACCS-167 unb: {X_maccs_unb.shape}  "
          f"density={X_maccs_unb.mean():.3f}")

    pred_chembl_513 = np.load(PRED_CHEMBL_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_PATH).astype(np.float32)
    if pred_chembl_513.shape[0] != 513 or sim_chembl_513.shape[0] != 513:
        raise ValueError(
            f"ChEMBL feats shape: pred={pred_chembl_513.shape}  "
            f"sim={sim_chembl_513.shape}")
    pred_chembl_unb = pred_chembl_513[unb_idx]
    sim_chembl_unb = sim_chembl_513[unb_idx]
    print(f"[feat] pred_chembl_pec50 unb: mean={pred_chembl_unb.mean():.3f}  "
          f"std={pred_chembl_unb.std():.3f}")
    print(f"[feat] sim_chembl unb       : mean={sim_chembl_unb.mean():.3f}  "
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
          f"(169 = 167 MACCS + 1 pred_chembl + 1 sim_chembl)")

    # ---- Per-seed nested Ridge cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RIDGE CROSS-FIT (alpha picked by inner 5-fold MSE)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    alpha_per_fold_table = []

    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s, fold_records = _residual_cross_fit_one_seed(
            X_unb, residual, s
        )
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        alphas_picked = [fr["alpha_picked"] for fr in fold_records]
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "alphas_picked": alphas_picked,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "fold_records": fold_records,
        })
        alpha_per_fold_table.append({
            "seed": int(s),
            "alphas_picked": alphas_picked,
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"alphas/fold = {alphas_picked}")

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
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1242 ref (LGBM, same feats) = {NB1242_REF:.4f}")
    print(f"   delta vs nb1242         = {rae_mean_bag - NB1242_REF:+.4f}")

    # ---- Pearson vs nb1242 (orthogonality probe) ----
    nb1242_path = DATA_PROCESSED / "nb1242_mean_bag_oof.npy"
    pearson_vs_nb1242 = None
    residual_pearson_vs_nb1242 = None
    if nb1242_path.exists():
        nb1242_oof = np.load(nb1242_path).astype(np.float64)
        if nb1242_oof.shape[0] == n_unb:
            # Pearson on the corrected predictions themselves
            x1 = mean_bag_oof
            x2 = nb1242_oof
            pearson_vs_nb1242 = float(
                np.corrcoef(x1, x2)[0, 1]
            )
            # Pearson on the residual-corrections (corrected - anchor)
            r1 = mean_bag_oof - anchor
            r2 = nb1242_oof - anchor
            if r1.std() > 1e-9 and r2.std() > 1e-9:
                residual_pearson_vs_nb1242 = float(
                    np.corrcoef(r1, r2)[0, 1]
                )
            rae_nb1242_local = float(rae(y_unb, nb1242_oof))
            print("\n" + "-" * 78)
            print("ORTHOGONALITY PROBE vs nb1242 (LGBM on same MACCS+ChEMBL)")
            print("-" * 78)
            print(f"   nb1242 mean_bag pooled RAE (local) = {rae_nb1242_local:.4f}")
            print(f"   Pearson(corrected_pred)            = "
                  f"{pearson_vs_nb1242:+.4f}")
            print(f"   Pearson(residual_correction)       = "
                  f"{'NA' if residual_pearson_vs_nb1242 is None else f'{residual_pearson_vs_nb1242:+.4f}'}")
        else:
            print(f"[warn] nb1242 shape mismatch: {nb1242_oof.shape}")
    else:
        print(f"[warn] {nb1242_path} not found; skipping Pearson probe")

    # ---- Verdict ----
    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN
    matches_nb1242 = abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN

    if beats_nb1242:
        verdict = "RIDGE_BEATS_NB1242_LGBM_NEW_CANDIDATE"
    elif matches_nb1242:
        verdict = "RIDGE_TIES_NB1242_LGBM_CHECK_ORTHOGONALITY"
    elif beats_nb1070:
        verdict = "RIDGE_HELPS_NB1070_BUT_NOT_NB1242"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "RIDGE_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "RIDGE_HURTS_NB1070"

    print(f"\n   verdict                = {verdict}")
    print(f"   beats_nb1070           = {beats_nb1070}")
    print(f"   beats_nb1242           = {beats_nb1242}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "method": "Ridge_regression_residual",
        "features": "MACCS-167 + pred_chembl_pec50 + sim_chembl",
        "feature_dim": int(feat_dim),
        "standardize": "StandardScaler per fold (train-only stats)",
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "alphas_grid": ALPHAS,
        "inner_folds": INNER_FOLDS,
        "n_unb": int(n_unb),
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "alpha_per_fold_table": alpha_per_fold_table,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1242": bool(beats_nb1242),
        "matches_nb1242": bool(matches_nb1242),
        "verdict": verdict,
        "pearson_vs_nb1242_corrected": pearson_vs_nb1242,
        "pearson_vs_nb1242_residual_correction": residual_pearson_vs_nb1242,
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
        "alpha_per_fold_table",
        "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1242",
        "pearson_vs_nb1242_corrected",
        "pearson_vs_nb1242_residual_correction",
        "beats_nb1070", "beats_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
