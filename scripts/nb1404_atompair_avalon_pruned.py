"""nb1404 -- Dual SHAP-pruned: top-30 AtomPair + top-30 Avalon + ChEMBL residual.

Hypothesis:
    nb1373 (SHAP-pruned AtomPair top-30 + ChEMBL, 32 cols) -> mean_bag 0.5095.
    nb1392 (SHAP-pruned Avalon  top-30 + ChEMBL, 32 cols) -> mean_bag 0.5391.
    AtomPair encodes pair-distance topology; Avalon encodes substructural
    fragments.  These are two different bit-pattern axes -- pair-distance
    fingerprints are designed to capture inter-atom distance graphs while
    Avalon path-based bits capture small-fragment substructural identity.
    Concatenating the SHAP-pruned subset of each (30 + 30 = 60 fp bits)
    with the ChEMBL kNN columns (pred_chembl_pec50 + sim) gives a 62-col
    residual learner that may extract complementary signal not captured by
    either fingerprint alone.

Protocol:
    1.  Anchor = nb1070_pred_oof on 253 unblind rows.
        residual = y_unb - anchor.
    2.  Reuse top-30 AtomPair bit indices from nb1373_summary.json
        ("top_atompair_bit_indices_ranked").
    3.  Reuse top-30 Avalon   bit indices from nb1392_summary.json
        ("top_avalon_bit_indices_ranked").
    4.  Reuse cached ChEMBL kNN columns:
            pred_chembl_pec50_513.npy  (513,)
            sim_chembl_513.npy         (513,)
        Slice both to unb_idx (253,).
    5.  Build 62-col feature matrix on 253:
            30 AtomPair bits (from te_atompair.npy[unb, top_ap])
          + 30 Avalon  bits (from te_avalon512.npy[unb, top_av])
          + pred_chembl_pec50
          + mean_sim
    6.  5-seed bag (seeds [0, 1, 7, 42, 137]), KFold(n=5) cross-fit per seed
        on shallow LGBM Huber (depth=3, num_leaves=7, n_est=80, lr=0.05,
        huber_alpha=1.0, min_child_samples=20).  Mean-bag pooled RAE.
    7.  Verdict at 0.003 margin vs:
            nb1373  (SHAP-pruned AtomPair top-30, 0.5095 mean-bag)
            nb1392  (SHAP-pruned Avalon  top-30, 0.5391 mean-bag)
    8.  Report Pearson(mean_bag_oof, nb1373_mean_bag_oof) for redundancy
        check vs the stronger anchor.

Outputs:
    scripts/nb1404_atompair_avalon_pruned.py         (this file)
    data/processed/nb1404_summary.json
    data/processed/nb1404_mean_bag_oof.npy           (253,) float32
    data/processed/nb1404_median_bag_oof.npy         (253,) float32
    data/processed/nb1404_per_seed_corrected_oof.npy (5, 253) float32
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

TAG = "nb1404"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"      # (513, 2048) uint8
AVALON_TE_PATH   = DATA_PROCESSED / "te_avalon512.npy"     # (513, 512)  uint8
PRED_CHEMBL_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"  # (513,) f32
SIM_CHEMBL_PATH  = DATA_PROCESSED / "sim_chembl_513.npy"   # (513,) f32

NB1373_SUMMARY = DATA_PROCESSED / "nb1373_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"

NB1373_MEAN_BAG = DATA_PROCESSED / "nb1373_mean_bag_oof.npy"

NB1373_REF = 0.5095   # SHAP-pruned AtomPair top-30 + ChEMBL residual mean_bag
NB1392_REF = 0.5391   # SHAP-pruned Avalon   top-30 + ChEMBL residual mean_bag
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


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Dual SHAP-pruned: top-30 AtomPair + top-30 Avalon + ChEMBL; "
          f"anchor={ANCHOR}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          baselines: nb1373 ({NB1373_REF:.4f}), "
          f"nb1392 ({NB1392_REF:.4f})  margin = {DECISION_MARGIN}")
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
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load top-30 AtomPair bit indices from nb1373 ----
    if not NB1373_SUMMARY.exists():
        raise FileNotFoundError(f"Missing nb1373 summary: {NB1373_SUMMARY}")
    with open(NB1373_SUMMARY) as f:
        s1373 = json.load(f)
    top_ap_idx = np.array(
        s1373["top_atompair_bit_indices_ranked"], dtype=int
    )
    n_top_ap = len(top_ap_idx)
    print(f"[load] nb1373 top-{n_top_ap} AtomPair bit indices: "
          f"{top_ap_idx.tolist()}")

    # ---- Load top-30 Avalon bit indices from nb1392 ----
    if not NB1392_SUMMARY.exists():
        raise FileNotFoundError(f"Missing nb1392 summary: {NB1392_SUMMARY}")
    with open(NB1392_SUMMARY) as f:
        s1392 = json.load(f)
    top_av_idx = np.array(
        s1392["top_avalon_bit_indices_ranked"], dtype=int
    )
    n_top_av = len(top_av_idx)
    print(f"[load] nb1392 top-{n_top_av} Avalon   bit indices: "
          f"{top_av_idx.tolist()}")

    # ---- Cached ChEMBL kNN columns ----
    if not PRED_CHEMBL_PATH.exists():
        raise FileNotFoundError(f"Missing cached pred_chembl: {PRED_CHEMBL_PATH}")
    if not SIM_CHEMBL_PATH.exists():
        raise FileNotFoundError(f"Missing cached sim_chembl: {SIM_CHEMBL_PATH}")
    pred_chembl_513 = np.load(PRED_CHEMBL_PATH).astype(np.float32)
    sim_chembl_513  = np.load(SIM_CHEMBL_PATH).astype(np.float32)
    if pred_chembl_513.shape[0] != 513 or sim_chembl_513.shape[0] != 513:
        raise ValueError("cached ChEMBL kNN shapes != 513")
    pred_chembl_unb = pred_chembl_513[unb_idx]
    mean_sim_unb   = sim_chembl_513[unb_idx]
    print(f"[load] pred_chembl (unb): mean={pred_chembl_unb.mean():.3f}  "
          f"std={pred_chembl_unb.std():.3f}")
    print(f"[load] mean_sim    (unb): mean={mean_sim_unb.mean():.3f}  "
          f"p50={np.percentile(mean_sim_unb, 50):.3f}")

    # ---- AtomPair-2048 (unblind slice + bit slice) ----
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}")
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    if X_ap_te.shape[0] != 513:
        raise ValueError(f"AtomPair cache shape mismatch: {X_ap_te.shape}")
    n_ap_full = int(X_ap_te.shape[1])
    print(f"[load] AtomPair cache shape = {X_ap_te.shape}  (n_bits={n_ap_full})")
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_pruned = X_ap_unb[:, top_ap_idx]
    print(f"   AtomPair pruned shape = {X_ap_pruned.shape}  "
          f"density = {X_ap_pruned.mean():.4f}")

    # ---- Avalon-512 (unblind slice + bit slice) ----
    if not AVALON_TE_PATH.exists():
        raise FileNotFoundError(f"Avalon test cache missing: {AVALON_TE_PATH}")
    X_av_te = np.load(AVALON_TE_PATH)
    if X_av_te.shape[0] != 513:
        raise ValueError(f"Avalon cache shape mismatch: {X_av_te.shape}")
    n_av_full = int(X_av_te.shape[1])
    print(f"[load] Avalon cache shape   = {X_av_te.shape}  (n_bits={n_av_full})")
    X_av_unb = X_av_te[unb_idx].astype(np.float32)
    X_av_pruned = X_av_unb[:, top_av_idx]
    print(f"   Avalon   pruned shape = {X_av_pruned.shape}  "
          f"density = {X_av_pruned.mean():.4f}")

    # ---- Build 62-col feature matrix ----
    X_unb = np.concatenate(
        [
            X_ap_pruned,
            X_av_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"\n   FEATURE matrix: {X_unb.shape}  "
          f"(top-{n_top_ap} AP + top-{n_top_av} AV + pred_chembl + sim)")
    expected_dim = n_top_ap + n_top_av + 2
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim mismatch: {feat_dim} vs expected {expected_dim}")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (DUAL-PRUNED, dim={feat_dim})")
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

    # ---- Pearson vs nb1373 mean_bag ----
    if NB1373_MEAN_BAG.exists():
        nb1373_mb = np.load(NB1373_MEAN_BAG).astype(np.float64)
        if nb1373_mb.shape[0] == n_unb:
            pearson_vs_nb1373 = _pearson(mean_bag_oof, nb1373_mb)
        else:
            pearson_vs_nb1373 = float("nan")
            print(f"   [warn] nb1373_mean_bag shape mismatch: "
                  f"{nb1373_mb.shape}")
    else:
        pearson_vs_nb1373 = float("nan")
        print(f"   [warn] nb1373_mean_bag missing: {NB1373_MEAN_BAG}")

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
          f"  d_vs_nb1373 = {rae_mean_bag - NB1373_REF:+.4f}"
          f"  d_vs_nb1392 = {rae_mean_bag - NB1392_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f}"
          f"  d_vs_nb1373 = {rae_median_bag - NB1373_REF:+.4f}"
          f"  d_vs_nb1392 = {rae_median_bag - NB1392_REF:+.4f})")
    print(f"   Pearson(mean_bag, nb1373_mean_bag) = {pearson_vs_nb1373:.4f}")
    print(f"   nb1373 ref            = {NB1373_REF:.4f}  (SHAP-pruned AtomPair top-30)")
    print(f"   nb1392 ref            = {NB1392_REF:.4f}  (SHAP-pruned Avalon  top-30)")

    beats_nb1373 = rae_mean_bag < NB1373_REF - DECISION_MARGIN
    beats_nb1392 = rae_mean_bag < NB1392_REF - DECISION_MARGIN
    flat_vs_nb1373 = abs(rae_mean_bag - NB1373_REF) < DECISION_MARGIN

    if beats_nb1373:
        verdict = "DUAL_PRUNED_BEATS_NB1373_NEW_PRIMARY_CANDIDATE"
    elif flat_vs_nb1373:
        verdict = "DUAL_PRUNED_FLAT_VS_NB1373"
    elif beats_nb1392:
        verdict = "DUAL_PRUNED_BEATS_NB1392_BUT_WORSE_THAN_NB1373"
    else:
        verdict = "DUAL_PRUNED_HURTS_VS_NB1373"
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
        "atompair_source": "te_atompair_npy (sliced by nb1373 top-30 SHAP)",
        "avalon_source":   "te_avalon512_npy (sliced by nb1392 top-30 SHAP)",
        "chembl_source":   "pred_chembl_pec50_513_npy + sim_chembl_513_npy (cached)",
        "n_unb": n_unb,
        "n_atompair_bits_full": n_ap_full,
        "n_avalon_bits_full":   n_av_full,
        "top_k_atompair": int(n_top_ap),
        "top_k_avalon":   int(n_top_av),
        "top_atompair_bit_indices_ranked": [int(b) for b in top_ap_idx.tolist()],
        "top_avalon_bit_indices_ranked":   [int(b) for b in top_av_idx.tolist()],
        "atompair_pruned_density_unb": float(X_ap_pruned.mean()),
        "avalon_pruned_density_unb":   float(X_av_pruned.mean()),
        "feat_dim": int(feat_dim),
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
        "residual_std":  float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag":   rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1373": rae_mean_bag - NB1373_REF,
        "delta_mean_bag_vs_nb1392": rae_mean_bag - NB1392_REF,
        "pearson_mean_bag_vs_nb1373": pearson_vs_nb1373,
        "beats_nb1373": bool(beats_nb1373),
        "beats_nb1392": bool(beats_nb1392),
        "flat_vs_nb1373": bool(flat_vs_nb1373),
        "verdict": verdict,
        "nb1373_ref": NB1373_REF,
        "nb1392_ref": NB1392_REF,
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
        "n_atompair_bits_full", "n_avalon_bits_full",
        "top_k_atompair", "top_k_avalon",
        "atompair_pruned_density_unb", "avalon_pruned_density_unb",
        "feat_dim",
        "rae_anchor_nb1070",
        "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1373",
        "delta_mean_bag_vs_nb1392",
        "pearson_mean_bag_vs_nb1373",
        "beats_nb1373", "beats_nb1392", "flat_vs_nb1373",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
