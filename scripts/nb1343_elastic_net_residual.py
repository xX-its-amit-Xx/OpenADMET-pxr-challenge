"""nb1343 -- ElasticNet residual on MACCS + ChEMBL features.

Hypothesis:
    ElasticNet (L1+L2) gives sparse + ridge regularization.  May find a
    parsimonious linear residual that captures the additive part of the signal
    that the shallow LGBM (nb1242 mean-bag = 0.5431) over-fits.  L1 zeroes
    redundant MACCS bits while L2 stabilizes the kept coefficients; the
    expectation is that pred_chembl and sim_chembl survive with large weights
    and a handful of MACCS bits add a sparse correction.

Protocol:
    1. Anchor = nb1070_pred_oof on 253 unblind rows.
    2. Residual = y_unb - nb1070_pred_oof.
    3. Features: MACCS-167 + pred_chembl_pec50 + sim_chembl  (169 cols).
       Standardized per fold inside the pipeline.
    4. 5-seed bag of ElasticNet with per-fold CV over
         alpha_grid = {0.01, 0.1, 1.0}
         l1_ratio_grid = {0.1, 0.5, 0.9}
       max_iter = 10000.
    5. 5-fold cross-fit per seed; per-seed RAE + mean-bag pooled RAE.
    6. Verdict at 0.003 margin vs nb1242 (0.5431).
    7. Report # non-zero coefficients in the best fold's model (sparsity).

Outputs:
    scripts/nb1343_elastic_net_residual.py        (this file)
    data/processed/nb1343_summary.json
    data/processed/nb1343_mean_bag_oof.npy        (253,) float32
    data/processed/nb1343_per_seed_corrected_oof.npy (5, 253) float32
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
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1343"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

ALPHA_GRID = [0.01, 0.1, 1.0]
L1_RATIO_GRID = [0.1, 0.5, 0.9]
MAX_ITER = 10_000

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
PRED_CHEMBL_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

NB1070_REF = 0.5771
NB1242_REF = 0.5431
DECISION_MARGIN = 0.003


def _inner_cv_pick(X_tr: np.ndarray, y_tr: np.ndarray, seed: int):
    """3-fold inner CV over (alpha, l1_ratio); returns best params + fitted
    scaler + fitted ElasticNet on FULL train fold."""
    inner = KFold(n_splits=3, shuffle=True, random_state=seed + 9999)
    best = None  # (mse, alpha, l1_ratio)
    for alpha in ALPHA_GRID:
        for l1_ratio in L1_RATIO_GRID:
            mses = []
            for itr, iva in inner.split(np.arange(len(y_tr))):
                sc = StandardScaler().fit(X_tr[itr])
                Xi = sc.transform(X_tr[itr])
                Xv = sc.transform(X_tr[iva])
                mdl = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=MAX_ITER,
                    random_state=seed,
                    selection="cyclic",
                )
                mdl.fit(Xi, y_tr[itr])
                p = mdl.predict(Xv)
                mses.append(float(np.mean((p - y_tr[iva]) ** 2)))
            mean_mse = float(np.mean(mses))
            if best is None or mean_mse < best[0]:
                best = (mean_mse, alpha, l1_ratio)
    _, alpha_b, l1_b = best
    # Final refit on full train fold with picked params
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    mdl = ElasticNet(
        alpha=alpha_b,
        l1_ratio=l1_b,
        max_iter=MAX_ITER,
        random_state=seed,
        selection="cyclic",
    )
    mdl.fit(X_tr_s, y_tr)
    return alpha_b, l1_b, scaler, mdl


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray, seed: int):
    """5-fold cross-fit; per-fold (alpha, l1_ratio, n_nonzero)."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    for fi, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        alpha_b, l1_b, scaler, mdl = _inner_cv_pick(X[tr_loc], residual[tr_loc], seed)
        Xv = scaler.transform(X[va_loc])
        oof[va_loc] = mdl.predict(Xv)
        n_nonzero = int(np.sum(np.abs(mdl.coef_) > 1e-12))
        fold_records.append({
            "fold": int(fi),
            "alpha": float(alpha_b),
            "l1_ratio": float(l1_b),
            "n_nonzero": n_nonzero,
            "intercept": float(mdl.intercept_),
            "coef_abs_max": float(np.max(np.abs(mdl.coef_))),
        })
    return oof, fold_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ElasticNet residual on MACCS + ChEMBL features")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          alpha_grid = {ALPHA_GRID}  l1_ratio_grid = {L1_RATIO_GRID}")
    print(f"          features = MACCS-167 + pred_chembl_pec50 + sim_chembl (169)")
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
    print(f"[load] n_unb={n_unb}")
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Features on 253 ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != 513:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    pred_chembl_te = np.load(PRED_CHEMBL_PATH).astype(np.float32)
    sim_chembl_te = np.load(SIM_CHEMBL_PATH).astype(np.float32)
    if pred_chembl_te.shape[0] != 513 or sim_chembl_te.shape[0] != 513:
        raise ValueError("ChEMBL feature shape mismatch (expected 513)")

    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    pred_chembl_unb = pred_chembl_te[unb_idx].astype(np.float32)
    sim_chembl_unb = sim_chembl_te[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [X_maccs_unb,
         pred_chembl_unb.reshape(-1, 1),
         sim_chembl_unb.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"[feat] residual feature matrix: {X_unb.shape}  (= 167 + 1 + 1)")

    # ---- Per-seed cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (ElasticNet inner-3-fold CV)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    all_fold_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s, fold_records = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        n_nz_per_fold = [r["n_nonzero"] for r in fold_records]
        alpha_per_fold = [r["alpha"] for r in fold_records]
        l1_per_fold = [r["l1_ratio"] for r in fold_records]
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "fold_records": fold_records,
            "fold_alpha": alpha_per_fold,
            "fold_l1_ratio": l1_per_fold,
            "fold_n_nonzero": n_nz_per_fold,
            "mean_n_nonzero": float(np.mean(n_nz_per_fold)),
        })
        for fr in fold_records:
            fr2 = dict(fr)
            fr2["seed"] = int(s)
            all_fold_records.append(fr2)
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"nnz/fold={n_nz_per_fold}  "
              f"alpha/fold={alpha_per_fold}  "
              f"l1/fold={l1_per_fold}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())

    # ---- Best fold model (smallest fold MSE proxy: smallest |resid_oof_var-resid_var|
    #      not stored; instead report config of the best-RAE seed's lowest-nnz fold)
    # We pick best-fold-overall as: min n_nonzero among (alpha, l1_ratio) that
    # appear in the BEST seed's records.
    best_seed_idx = int(np.argmin(per_seed_rae_arr))
    best_seed = RESID_SEEDS[best_seed_idx]
    best_seed_folds = per_seed_records[best_seed_idx]["fold_records"]
    best_fold = min(best_seed_folds, key=lambda r: r["n_nonzero"])
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
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (ChEMBL kNN feat LGBM bag)")
    print(f"   best seed              = {best_seed}  rae={per_seed_rae[best_seed_idx]:.4f}")
    print(f"   best fold (in best seed, min-nnz):")
    print(f"      fold={best_fold['fold']}  alpha={best_fold['alpha']}  "
          f"l1_ratio={best_fold['l1_ratio']}  "
          f"n_nonzero={best_fold['n_nonzero']}/{feat_dim}  "
          f"intercept={best_fold['intercept']:+.4f}  "
          f"coef_abs_max={best_fold['coef_abs_max']:.4f}")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "ELASTICNET_RESIDUAL_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif beats_nb1070:
        verdict = "ELASTICNET_RESIDUAL_HELPS_NB1070_BUT_NOT_NB1242"
    elif abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN:
        verdict = "ELASTICNET_RESIDUAL_TIES_NB1242"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "ELASTICNET_RESIDUAL_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "ELASTICNET_RESIDUAL_HURTS"
    print(f"   verdict                = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "model": "ElasticNet",
        "feature_set": "MACCS-167 + pred_chembl_pec50 + sim_chembl",
        "feature_dim": feat_dim,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "alpha_grid": ALPHA_GRID,
        "l1_ratio_grid": L1_RATIO_GRID,
        "max_iter": MAX_ITER,
        "standardize": "StandardScaler per outer fold (fit on train fold)",
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "all_fold_records": all_fold_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1242": bool(beats_nb1242),
        "best_seed": int(best_seed),
        "best_fold_config": best_fold,
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
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1242",
        "beats_nb1070", "beats_nb1242",
        "best_seed", "best_fold_config",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
