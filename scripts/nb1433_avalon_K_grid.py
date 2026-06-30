"""nb1433 -- SHAP-pruned Avalon-512 top-K grid search over residual learner.

Hypothesis:
    nb1392 fixed K=30 for SHAP-pruned Avalon-512 and produced mean-bag pooled
    RAE = 0.5391 (beats nb1163 full Avalon 0.5788, but worse than nb1373
    AtomPair-pruned 0.5095).  K=30 was an analogy pick from nb1352/nb1373; it
    was never grid-searched.  This notebook sweeps K in {5, 10, 15, 20, 40,
    50, 75, 100} to find the residual-capacity sweet spot for Avalon.

Protocol:
    1.  Reuse the SHAP importance ranking from nb1392 verbatim (top_avalon_bit
        _indices_ranked).  These are the same 512 bits sorted by SHAP |mean|
        over a seed-0 LGBM Huber trained on the full 514-col residual matrix
        (Avalon-512 + pred_chembl + sim) on the 253 unblind rows.
    2.  For K in {5, 10, 15, 20, 40, 50, 75, 100}:
            features = top-K Avalon bits + pred_chembl + sim   (dim = K + 2)
    3.  Per K, run a 5-seed bag (seeds [0, 1, 7, 42, 137]) of shallow LGBM
        Huber under 5-fold cross-fit on residual = y_unb - nb1070.  Aggregate
        with mean-bag and report pooled RAE on the corrected anchor.
    4.  Verdict at 0.003 margin vs nb1392 K=30 = 0.5391.

Outputs:
    scripts/nb1433_avalon_K_grid.py                  (this file)
    data/processed/nb1433_summary.json
    data/processed/nb1433_best_K_oof.npy             (253,) float32
        -- mean_bag corrected anchor at best K
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

TAG = "nb1433"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

K_GRID = [5, 10, 15, 20, 40, 50, 75, 100]

AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
PRED_CHEMBL_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_PATH = DATA_PROCESSED / "sim_chembl_513.npy"
NB1392_SUMMARY_PATH = DATA_PROCESSED / "nb1392_summary.json"

NB1070_REF = 0.5771
NB1163_REF = 0.5788
NB1373_REF = 0.5095
NB1392_K30_REF = 0.5391       # mean-bag pooled RAE at K=30 (target to beat)
DECISION_MARGIN = 0.003


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


def _bag_one_K(X_unb_K: np.ndarray, residual: np.ndarray,
               anchor: np.ndarray, y_unb: np.ndarray):
    per_seed_corr = np.zeros((len(RESID_SEEDS), len(y_unb)), dtype=np.float64)
    per_seed_rae_list: list[float] = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb_K, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corr[i] = pred_corr_s
        per_seed_rae_list.append(float(rae(y_unb, pred_corr_s)))
    mean_bag_oof = per_seed_corr.mean(axis=0)
    median_bag_oof = np.median(per_seed_corr, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    return {
        "mean_bag_oof": mean_bag_oof,
        "median_bag_oof": median_bag_oof,
        "per_seed_corr": per_seed_corr,
        "per_seed_rae": per_seed_rae_list,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SHAP-pruned Avalon-512 K-grid vs nb1392 K=30 ({NB1392_K30_REF:.4f})")
    print(f"          K_grid = {K_GRID}")
    print(f"          seeds  = {RESID_SEEDS}  folds = {RESID_FOLDS}  margin = {DECISION_MARGIN}")
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

    # ---- Cached ChEMBL kNN columns ----
    if not PRED_CHEMBL_PATH.exists():
        raise FileNotFoundError(f"Missing cached pred_chembl: {PRED_CHEMBL_PATH}")
    if not SIM_CHEMBL_PATH.exists():
        raise FileNotFoundError(f"Missing cached sim_chembl: {SIM_CHEMBL_PATH}")
    pred_chembl_513 = np.load(PRED_CHEMBL_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_PATH).astype(np.float32)
    pred_chembl_unb = pred_chembl_513[unb_idx]
    mean_sim_unb = sim_chembl_513[unb_idx]
    print(f"[load] pred_chembl (unb): mean={pred_chembl_unb.mean():.3f}  "
          f"std={pred_chembl_unb.std():.3f}")
    print(f"[load] mean_sim    (unb): mean={mean_sim_unb.mean():.3f}")

    # ---- Avalon-512 (unblind slice) ----
    if not AVALON_TE_PATH.exists():
        raise FileNotFoundError(f"Avalon test cache missing: {AVALON_TE_PATH}")
    X_av_te = np.load(AVALON_TE_PATH)
    if X_av_te.shape[0] != 513:
        raise ValueError(f"Avalon cache shape mismatch: {X_av_te.shape}")
    n_av = int(X_av_te.shape[1])
    X_av_unb = X_av_te[unb_idx].astype(np.float32)
    print(f"[load] Avalon cache shape = {X_av_te.shape}  (n_bits={n_av})")

    # ---- Reuse nb1392 SHAP importance ranking ----
    if not NB1392_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing nb1392 summary: {NB1392_SUMMARY_PATH}")
    with open(NB1392_SUMMARY_PATH, "r") as f:
        nb1392 = json.load(f)
    ranked_bits_full_top30 = nb1392["top_avalon_bit_indices_ranked"]
    print(f"[load] nb1392 SHAP top-30 ranked bits  (head): "
          f"{ranked_bits_full_top30[:8]}...")

    # nb1392 only stored the top-30; we need a longer ranking for K up to 100.
    # Recompute the full SHAP importance vector deterministically (seed=0,
    # same params, same feature matrix) so the ranking extends past rank 30.
    X_unb_full = np.concatenate(
        [
            X_av_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)

    print("\n" + "-" * 78)
    print("RECOMPUTE FULL SHAP IMPORTANCE (seed=0; nb1392-identical recipe)")
    print("-" * 78)
    mdl = LGBMRegressor(**_lgbm_params(0))
    mdl.fit(X_unb_full, residual)
    imp_src = "lgbm_gain_fallback"
    try:
        import shap
        explainer = shap.TreeExplainer(mdl)
        sv = explainer.shap_values(X_unb_full)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 0]
        imp_full = np.abs(sv).mean(axis=0).astype(np.float64)
        imp_src = "shap_tree_explainer"
    except Exception as e:
        print(f"   [shap] WARN: shap failed ({e}); falling back to LGBM gain")
        imp_full = mdl.booster_.feature_importance(importance_type="gain").astype(np.float64)
    av_imp = imp_full[:n_av]
    ranked_bits_full = np.argsort(-av_imp).astype(int).tolist()
    print(f"   importance source = {imp_src}")
    print(f"   nonzero importance bits: {(av_imp > 0).sum()}/{n_av}")
    # consistency check against nb1392 top-30
    head30 = ranked_bits_full[:30]
    head_match = (head30 == ranked_bits_full_top30)
    print(f"   head[:30] matches nb1392? {head_match}")
    if not head_match:
        # report overlap rather than fail; importance ties can permute order
        ov = len(set(head30) & set(ranked_bits_full_top30))
        print(f"   set overlap on top-30 = {ov}/30")

    # ---- K-grid sweep ----
    print("\n" + "-" * 78)
    print("K-GRID SWEEP")
    print("-" * 78)
    per_K_records = []
    per_K_mean_bag_rae: dict[int, float] = {}
    per_K_mean_bag_oof: dict[int, np.ndarray] = {}

    for K in K_GRID:
        if K > n_av:
            print(f"   K={K} > n_av={n_av}; skipping")
            continue
        top_K_idx = np.array(ranked_bits_full[:K], dtype=int)
        X_av_K = X_av_unb[:, top_K_idx]
        X_unb_K = np.concatenate(
            [
                X_av_K,
                pred_chembl_unb.reshape(-1, 1),
                mean_sim_unb.reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)
        feat_dim_K = X_unb_K.shape[1]

        bag = _bag_one_K(X_unb_K, residual, anchor, y_unb)
        rae_mb = bag["rae_mean_bag"]
        rae_med = bag["rae_median_bag"]
        per_K_mean_bag_rae[K] = rae_mb
        per_K_mean_bag_oof[K] = bag["mean_bag_oof"].astype(np.float32)

        record = {
            "K": int(K),
            "feat_dim": int(feat_dim_K),
            "per_seed_rae": bag["per_seed_rae"],
            "per_seed_mean": float(np.mean(bag["per_seed_rae"])),
            "per_seed_std": float(np.std(bag["per_seed_rae"])),
            "rae_mean_bag": rae_mb,
            "rae_median_bag": rae_med,
            "delta_vs_nb1392_K30": rae_mb - NB1392_K30_REF,
            "delta_vs_nb1070": rae_mb - rae_anchor,
        }
        per_K_records.append(record)
        print(f"   K={K:3d}  dim={feat_dim_K:3d}  "
              f"mean_bag={rae_mb:.4f}  median_bag={rae_med:.4f}  "
              f"d_vs_nb1392(K30)={rae_mb - NB1392_K30_REF:+.4f}  "
              f"d_vs_nb1070={rae_mb - rae_anchor:+.4f}  "
              f"per_seed=[{', '.join(f'{r:.4f}' for r in bag['per_seed_rae'])}]")

    # ---- Pick best K ----
    if not per_K_mean_bag_rae:
        raise RuntimeError("K-grid produced no results")
    best_K = min(per_K_mean_bag_rae, key=lambda k: per_K_mean_bag_rae[k])
    best_rae = per_K_mean_bag_rae[best_K]
    best_oof = per_K_mean_bag_oof[best_K]

    beats_nb1392_K30 = best_rae < NB1392_K30_REF - DECISION_MARGIN
    flat_vs_nb1392_K30 = abs(best_rae - NB1392_K30_REF) <= DECISION_MARGIN
    beats_nb1373 = best_rae < NB1373_REF - DECISION_MARGIN
    beats_nb1070 = best_rae < rae_anchor - DECISION_MARGIN

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   best K = {best_K}  -> mean_bag RAE = {best_rae:.4f}")
    print(f"   nb1392 K=30 ref      = {NB1392_K30_REF:.4f}  "
          f"(delta = {best_rae - NB1392_K30_REF:+.4f})")
    print(f"   nb1373 AtomPair ref  = {NB1373_REF:.4f}  "
          f"(delta = {best_rae - NB1373_REF:+.4f})")
    print(f"   nb1070 anchor        = {rae_anchor:.4f}  "
          f"(delta = {best_rae - rae_anchor:+.4f})")

    if beats_nb1373:
        verdict = "AVALON_K_GRID_BEST_BEATS_NB1373_NEW_PRIMARY_CANDIDATE"
    elif beats_nb1392_K30:
        verdict = "AVALON_K_GRID_BEATS_NB1392_K30_BUT_WORSE_THAN_NB1373"
    elif flat_vs_nb1392_K30:
        verdict = "AVALON_K_GRID_FLAT_VS_NB1392_K30"
    elif beats_nb1070:
        verdict = "AVALON_K_GRID_HELPS_NB1070_BUT_WORSE_THAN_NB1392_K30"
    else:
        verdict = "AVALON_K_GRID_HURTS_NB1070"
    print(f"   verdict              = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_K_oof.npy", best_oof)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_K_oof.npy'}  (K={best_K})")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": "avalon512_cached_te_avalon512_npy",
        "chembl_source": "pred_chembl_pec50_513_npy + sim_chembl_513_npy (cached)",
        "shap_ranking_source": "recomputed_seed0_nb1392_recipe",
        "shap_importance_source": imp_src,
        "shap_head30_matches_nb1392": bool(head_match),
        "n_unb": n_unb,
        "n_avalon_bits": n_av,
        "K_grid": K_GRID,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_K_records": per_K_records,
        "per_K_mean_bag_rae": {str(k): v for k, v in per_K_mean_bag_rae.items()},
        "best_K": int(best_K),
        "best_rae_mean_bag": float(best_rae),
        "delta_best_vs_nb1392_K30": float(best_rae - NB1392_K30_REF),
        "delta_best_vs_nb1373": float(best_rae - NB1373_REF),
        "delta_best_vs_nb1070": float(best_rae - rae_anchor),
        "beats_nb1392_K30": bool(beats_nb1392_K30),
        "flat_vs_nb1392_K30": bool(flat_vs_nb1392_K30),
        "beats_nb1373": bool(beats_nb1373),
        "beats_nb1070": bool(beats_nb1070),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1163_ref": NB1163_REF,
        "nb1373_ref": NB1373_REF,
        "nb1392_K30_ref": NB1392_K30_REF,
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
    print(f"  best_K                : {res['best_K']}")
    print(f"  best_rae_mean_bag     : {res['best_rae_mean_bag']:.4f}")
    print(f"  nb1392_K30_ref        : {res['nb1392_K30_ref']:.4f}")
    print(f"  delta_vs_nb1392_K30   : {res['delta_best_vs_nb1392_K30']:+.4f}")
    print(f"  beats_nb1392_K30      : {res['beats_nb1392_K30']}")
    print(f"  beats_nb1373          : {res['beats_nb1373']}")
    print(f"  verdict               : {res['verdict']}")
    print(f"  per_K_mean_bag_rae    : {res['per_K_mean_bag_rae']}")
