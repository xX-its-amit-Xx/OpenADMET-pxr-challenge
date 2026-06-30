"""nb1392 -- SHAP-pruned Avalon-512 top-30 + ChEMBL residual learner.

Hypothesis:
    nb1163 used the full Avalon-512 fingerprint as residual features over the
    nb1070 anchor and DID NOT help (pooled mean-bag RAE ~0.5788, marginally
    worse than nb1070 ~0.5771).  nb1352/nb1373 demonstrated that
    SHAP-pruning a high-dim fingerprint to the top-K bits (plus pred_chembl
    + sim) shrinks residual capacity at n=253 and can recover a real win
    (nb1352 MACCS top-20 -> 0.5323; nb1373 AtomPair top-30 -> 0.5095).
    By analogy, SHAP-pruning Avalon from 512 -> 30 bits (plus pred_chembl +
    sim) may extract orthogonal substructural signal not captured by
    AtomPair / MACCS, possibly competitive with nb1373.

Protocol:
    1.  Anchor = nb1070_pred_oof on 253 unblind rows.
        residual = y_unb - anchor.
    2.  Reuse cached ChEMBL features:
            pred_chembl_pec50_513.npy  (513,) -- already produced by nb1373
                pipeline via kNN-5 Tanimoto on Morgan-2048 over ChEMBL pool
            sim_chembl_513.npy         (513,) -- mean_sim from same kNN
        Slice both to unb_idx (253,).
    3.  Build FULL 514-col feature matrix on 253:
            Avalon-512 (cached te_avalon512.npy sliced to unb) + pred_chembl
            + mean_sim
        Train ONE seed-0 shallow LGBM Huber on residual, compute SHAP
        importance via shap.TreeExplainer (LGBM gain fallback).  Slice
        Avalon-only importance, pick top-30 bit indices.
    4.  Build PRUNED 32-col feature matrix = top-30 Avalon + pred_chembl +
        mean_sim.
    5.  5-seed bag (seeds [0, 1, 7, 42, 137]), KFold(n=5) cross-fit per seed
        on shallow LGBM Huber (depth=3, num_leaves=7, n_est=80, lr=0.05,
        huber_alpha=1.0, min_child_samples=20).  Mean-bag pooled RAE.
    6.  Verdict at 0.003 margin vs:
            nb1373  (SHAP-pruned AtomPair top-30 baseline, 0.5095 mean-bag)
            nb1163  (full Avalon-512 standalone, 0.5788 mean-bag)
            nb1070  (anchor, ~0.5771 pooled)

Outputs:
    scripts/nb1392_shap_pruned_avalon.py        (this file)
    data/processed/nb1392_summary.json
    data/processed/nb1392_mean_bag_oof.npy        (253,) float32
    data/processed/nb1392_median_bag_oof.npy      (253,) float32
    data/processed/nb1392_per_seed_corrected_oof.npy (5, 253) float32
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

TAG = "nb1392"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"       # (513, 512) uint8
PRED_CHEMBL_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"  # (513,) f32
SIM_CHEMBL_PATH = DATA_PROCESSED / "sim_chembl_513.npy"    # (513,) f32

NB1070_REF = 0.5771
NB1163_REF = 0.5788      # full Avalon-512 standalone residual bag
NB1373_REF = 0.5095      # SHAP-pruned AtomPair top-30 + ChEMBL residual bag
DECISION_MARGIN = 0.003

TOP_K_AVALON = 30


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


def _compute_shap_importance(X: np.ndarray, residual: np.ndarray, seed: int = 0):
    """Train one global LGBM on residual; return (importance vector, source_tag)."""
    mdl = LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X, residual)
    try:
        import shap
        explainer = shap.TreeExplainer(mdl)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 0]
        imp = np.abs(sv).mean(axis=0)
        return imp.astype(np.float64), "shap_tree_explainer"
    except Exception as e:
        print(f"   [shap] WARN: shap failed ({e}); falling back to LGBM gain")
        imp = mdl.booster_.feature_importance(importance_type="gain")
        return imp.astype(np.float64), "lgbm_gain_fallback"


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SHAP-pruned Avalon-512 top-{TOP_K_AVALON} + ChEMBL residual; "
          f"anchor={ANCHOR}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          baselines: nb1373 ({NB1373_REF:.4f}), "
          f"nb1163 ({NB1163_REF:.4f})  margin = {DECISION_MARGIN}")
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
    if pred_chembl_513.shape[0] != 513 or sim_chembl_513.shape[0] != 513:
        raise ValueError("cached ChEMBL kNN shapes != 513")
    pred_chembl_unb = pred_chembl_513[unb_idx]
    mean_sim_unb = sim_chembl_513[unb_idx]
    print(f"[load] pred_chembl (unb): mean={pred_chembl_unb.mean():.3f}  "
          f"std={pred_chembl_unb.std():.3f}")
    print(f"[load] mean_sim    (unb): mean={mean_sim_unb.mean():.3f}  "
          f"p50={np.percentile(mean_sim_unb, 50):.3f}")

    # ---- Avalon-512 (unblind slice) ----
    if not AVALON_TE_PATH.exists():
        raise FileNotFoundError(f"Avalon test cache missing: {AVALON_TE_PATH}")
    X_av_te = np.load(AVALON_TE_PATH)
    if X_av_te.shape[0] != 513:
        raise ValueError(f"Avalon cache shape mismatch: {X_av_te.shape}")
    n_av = int(X_av_te.shape[1])
    print(f"[load] Avalon cache shape = {X_av_te.shape}  (n_bits={n_av})")
    X_av_unb = X_av_te[unb_idx].astype(np.float32)
    const_cols = int((X_av_unb.var(axis=0) == 0).sum())
    print(f"   bit density (unb) = {X_av_unb.mean():.4f}  "
          f"const cols = {const_cols}/{n_av}")

    # ---- Build FULL feature matrix (Avalon + pred_chembl + sim) ----
    X_unb_full = np.concatenate(
        [
            X_av_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_full = X_unb_full.shape[1]
    print(f"   FULL feature matrix: {X_unb_full.shape}  "
          f"(Avalon-{n_av} + pred_chembl + sim)")

    # ---- SHAP importance frame ----
    print("\n" + "-" * 78)
    print("SHAP IMPORTANCE FRAME (1 global LGBM, seed=0, full feature matrix)")
    print("-" * 78)
    imp_full, imp_src = _compute_shap_importance(X_unb_full, residual, seed=0)
    print(f"   importance source = {imp_src}")
    print(f"   importance vector shape = {imp_full.shape}")
    print(f"   pred_chembl importance = {imp_full[n_av]:.4f}")
    print(f"   sim importance         = {imp_full[n_av + 1]:.4f}")

    # Avalon-only importance slice
    av_imp = imp_full[:n_av]
    top_k = min(TOP_K_AVALON, n_av)
    top_bit_order = np.argsort(-av_imp)
    top_bit_idx = top_bit_order[:top_k].astype(int)
    top_bit_idx_sorted = np.sort(top_bit_idx)
    top_bit_imp = av_imp[top_bit_idx]
    print(f"   top-{top_k} Avalon bit indices (ranked by importance):")
    for rank, (bit, val) in enumerate(zip(top_bit_idx.tolist(),
                                          top_bit_imp.tolist())):
        print(f"      rank {rank+1:2d}:  bit {bit:5d}   imp = {val:.5f}")
    print(f"   top-{top_k} bit indices (sorted asc): {top_bit_idx_sorted.tolist()}")

    n_nonzero_imp = int((av_imp > 0).sum())
    print(f"   Avalon bits with nonzero importance: {n_nonzero_imp}/{n_av}")

    # ---- Build PRUNED 32-col feature matrix ----
    X_av_unb_pruned = X_av_unb[:, top_bit_idx]
    X_unb_pruned = np.concatenate(
        [
            X_av_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_pruned = X_unb_pruned.shape[1]
    print(f"\n   PRUNED feature matrix: {X_unb_pruned.shape}  "
          f"(top-{top_k} Avalon + pred_chembl + sim)")

    # ---- Per-seed residual cross-fit on PRUNED features ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (PRUNED, dim={feat_dim_pruned})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb_pruned, residual, s)
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
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

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
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f}"
          f"  d_vs_nb1163 = {rae_mean_bag - NB1163_REF:+.4f}"
          f"  d_vs_nb1373 = {rae_mean_bag - NB1373_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f}"
          f"  d_vs_nb1163 = {rae_median_bag - NB1163_REF:+.4f}"
          f"  d_vs_nb1373 = {rae_median_bag - NB1373_REF:+.4f})")
    print(f"   nb1070 ref            = {NB1070_REF:.4f}")
    print(f"   nb1163 ref            = {NB1163_REF:.4f}  (full Avalon-512)")
    print(f"   nb1373 ref            = {NB1373_REF:.4f}  (SHAP-pruned AtomPair top-30)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1163 = rae_mean_bag < NB1163_REF - DECISION_MARGIN
    beats_nb1373 = rae_mean_bag < NB1373_REF - DECISION_MARGIN

    if beats_nb1373:
        verdict = "SHAP_PRUNED_AVALON_BEATS_NB1373_NEW_PRIMARY_CANDIDATE"
    elif abs(rae_mean_bag - NB1373_REF) < DECISION_MARGIN:
        verdict = "SHAP_PRUNED_AVALON_FLAT_VS_NB1373"
    elif beats_nb1163:
        verdict = "SHAP_PRUNED_AVALON_BEATS_NB1163_BUT_WORSE_THAN_NB1373"
    elif beats_nb1070:
        verdict = "SHAP_PRUNED_AVALON_HELPS_NB1070_BUT_WORSE_THAN_NB1163"
    else:
        verdict = "SHAP_PRUNED_AVALON_HURTS_NB1070"
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
        "anchor": ANCHOR,
        "feature_source": "avalon512_cached_te_avalon512_npy",
        "chembl_source": "pred_chembl_pec50_513_npy + sim_chembl_513_npy (cached)",
        "n_unb": n_unb,
        "n_avalon_bits": n_av,
        "avalon_bit_density_unb": float(X_av_unb.mean()),
        "avalon_const_cols": const_cols,
        "avalon_nonzero_imp_bits": n_nonzero_imp,
        "shap_importance_source": imp_src,
        "top_k_avalon": int(top_k),
        "top_avalon_bit_indices_ranked": [int(b) for b in top_bit_idx.tolist()],
        "top_avalon_bit_importance_ranked": [float(v) for v in top_bit_imp.tolist()],
        "top_avalon_bit_indices_sorted_asc": [int(b) for b in top_bit_idx_sorted.tolist()],
        "pred_chembl_importance": float(imp_full[n_av]),
        "sim_importance": float(imp_full[n_av + 1]),
        "feat_dim_full": int(feat_dim_full),
        "feat_dim_pruned": int(feat_dim_pruned),
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
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1163": rae_mean_bag - NB1163_REF,
        "delta_mean_bag_vs_nb1373": rae_mean_bag - NB1373_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1163": bool(beats_nb1163),
        "beats_nb1373": bool(beats_nb1373),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1163_ref": NB1163_REF,
        "nb1373_ref": NB1373_REF,
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
        "n_avalon_bits", "shap_importance_source",
        "top_k_avalon",
        "top_avalon_bit_indices_ranked",
        "top_avalon_bit_importance_ranked",
        "pred_chembl_importance", "sim_importance",
        "feat_dim_full", "feat_dim_pruned",
        "avalon_nonzero_imp_bits",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1163",
        "delta_mean_bag_vs_nb1373",
        "beats_nb1070", "beats_nb1163", "beats_nb1373",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
