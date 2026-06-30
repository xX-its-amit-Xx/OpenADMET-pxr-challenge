"""nb1371 -- CatBoost residual on SHAP-pruned 22-col features (top-20 MACCS + ChEMBL).

Hypothesis:
    CatBoost in nb1341 (mean 0.5420 / median 0.5395) used the full 169-col
    matrix (MACCS-167 + pred_chembl + sim).  With nb1352's SHAP-pruned
    top-20 MACCS bits + pred_chembl + sim (22 cols), CatBoost should avoid
    overfit at n=253 and add a genuinely orthogonal residual axis to LGBM
    nb1352 (mean 0.5323 / median 0.5315).  This is a learner-axis probe on
    the *pruned* feature support -- isolates whether the residual gap
    between LGBM and CatBoost was caused by feature noise.

Pipeline:
    Anchor:    nb1070_pred_oof  (253,)
    Residual:  y_unb - nb1070_pred_oof
    Features:  MACCS-top20[unb_idx]                (253, 20)  from nb1352
            ++ pred_chembl_pec50[unb_idx]          (253, 1)
            ++ sim_chembl[unb_idx]                 (253, 1)
               -> X_unb_pruned shape (253, 22)
    Learner:   CatBoostRegressor(loss=MAE, depth=4, iter=200, lr=0.05,
                                  l2_leaf_reg=5, random_seed=seed)
    Bag:       5 seeds [0, 1, 7, 42, 137], 5-fold cross-fit per seed
    Pool:      mean (and median) across seeds -> RAE
    Decorr:    Pearson vs nb1352 mean_bag AND median_bag
    Blend:     50/50 mean(nb1371_mean_bag, nb1352_median_bag) pooled RAE
    Verdict:   beats nb1352 (0.5323 mean / 0.5315 median) by 0.003 margin?

Outputs:
    scripts/nb1371_catboost_pruned.py              (this file)
    data/processed/nb1371_summary.json
    data/processed/nb1371_mean_bag_oof.npy         (253,) float32
    data/processed/nb1371_per_seed_corrected_oof.npy (5, 253) float32
    data/processed/nb1371_median_bag_oof.npy       (253,) float32
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

TAG = "nb1371"
ANCHOR = "nb1070"
NB1352_TAG = "nb1352"
NB1341_TAG = "nb1341"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

NB1070_REF = 0.5771
NB1352_MEAN_REF = 0.5323
NB1352_MEDIAN_REF = 0.5315
NB1341_MEAN_REF = 0.5420
NB1341_MEDIAN_REF = 0.5395
DECISION_MARGIN = 0.003

# SHAP-derived top-20 MACCS bits from nb1352_summary.json (ranked order).
TOP_MACCS_BITS_RANKED = [
    131, 80, 126, 106, 115, 90, 116, 155, 74, 38,
    82, 100, 138, 59, 153, 77, 147, 86, 118, 145,
]


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
    print(f"{TAG} -- CatBoost residual on SHAP-pruned (top-20 MACCS + ChEMBL)")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = top-20 MACCS + pred_chembl + sim  (22)")
    print(f"          learner  = CatBoost MAE depth=4 iter=200 lr=0.05 l2=5")
    print(f"          nb1341 ref (full 169) = mean {NB1341_MEAN_REF} / med {NB1341_MEDIAN_REF}")
    print(f"          nb1352 ref (pruned 22 LGBM) = mean {NB1352_MEAN_REF} / med {NB1352_MEDIAN_REF}")
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

    # ---- Pruned MACCS feature build ----
    maccs_te = np.load(DATA_PROCESSED / "te_maccs.npy")
    if maccs_te.shape[0] != 513:
        raise ValueError(f"te_maccs shape mismatch: {maccs_te.shape}")
    X_maccs_unb = maccs_te[unb_idx].astype(np.float32)
    n_maccs = int(X_maccs_unb.shape[1])
    print(f"[feat] MACCS unb shape = {X_maccs_unb.shape}  (n_bits={n_maccs})")

    top_bit_idx = np.array(TOP_MACCS_BITS_RANKED, dtype=int)
    X_maccs_unb_pruned = X_maccs_unb[:, top_bit_idx]
    print(f"[feat] PRUNED MACCS unb shape = {X_maccs_unb_pruned.shape}")
    print(f"[feat] top-20 MACCS bit indices (ranked): {top_bit_idx.tolist()}")

    # ---- ChEMBL kNN cached features ----
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
            X_maccs_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            sim_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"[feat] PRUNED residual feature matrix: {X_unb.shape}  "
          f"(top-20 MACCS + pred_chembl + sim)")

    # ---- Per-seed CatBoost cross-fit on pruned features ----
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

    # ---- Pearson vs nb1352 (LGBM pruned, same feature support) ----
    pearson_vs_nb1352_mean = None
    pearson_vs_nb1352_median = None
    blend_50_50_rae = None
    nb1352_median_oof = None
    nb1352_mean_path = DATA_PROCESSED / f"{NB1352_TAG}_mean_bag_oof.npy"
    nb1352_median_path = DATA_PROCESSED / f"{NB1352_TAG}_median_bag_oof.npy"
    if nb1352_mean_path.exists():
        nb1352_mean_oof = np.load(nb1352_mean_path).astype(np.float64)
        if nb1352_mean_oof.shape == (n_unb,):
            r, _ = pearsonr(mean_bag_oof, nb1352_mean_oof)
            pearson_vs_nb1352_mean = float(r)
        else:
            print(f"[warn] {NB1352_TAG}_mean shape mismatch: "
                  f"{nb1352_mean_oof.shape}")
    else:
        print(f"[warn] {nb1352_mean_path} missing")
    if nb1352_median_path.exists():
        nb1352_median_oof = np.load(nb1352_median_path).astype(np.float64)
        if nb1352_median_oof.shape == (n_unb,):
            r, _ = pearsonr(mean_bag_oof, nb1352_median_oof)
            pearson_vs_nb1352_median = float(r)
            blend = 0.5 * mean_bag_oof + 0.5 * nb1352_median_oof
            blend_50_50_rae = float(rae(y_unb, blend))
        else:
            print(f"[warn] {NB1352_TAG}_median shape mismatch: "
                  f"{nb1352_median_oof.shape}")
    else:
        print(f"[warn] {nb1352_median_path} missing")

    # ---- Pearson vs nb1341 (CatBoost full 169) -- ablation ----
    pearson_vs_nb1341 = None
    nb1341_path = DATA_PROCESSED / f"{NB1341_TAG}_mean_bag_oof.npy"
    if nb1341_path.exists():
        nb1341_oof = np.load(nb1341_path).astype(np.float64)
        if nb1341_oof.shape == (n_unb,):
            r, _ = pearsonr(mean_bag_oof, nb1341_oof)
            pearson_vs_nb1341 = float(r)

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1352_mean = {rae_mean_bag - NB1352_MEAN_REF:+.4f})  "
          f"(d_vs_nb1341_mean = {rae_mean_bag - NB1341_MEAN_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1352_med  = {rae_median_bag - NB1352_MEDIAN_REF:+.4f})")
    if pearson_vs_nb1352_mean is not None:
        print(f"   Pearson(nb1371_mean,  nb1352_mean)   = "
              f"{pearson_vs_nb1352_mean:.4f}")
    if pearson_vs_nb1352_median is not None:
        print(f"   Pearson(nb1371_mean,  nb1352_median) = "
              f"{pearson_vs_nb1352_median:.4f}")
    if pearson_vs_nb1341 is not None:
        print(f"   Pearson(nb1371_mean,  nb1341_mean)   = "
              f"{pearson_vs_nb1341:.4f}")
    if blend_50_50_rae is not None:
        d_blend_vs_1352 = blend_50_50_rae - NB1352_MEDIAN_REF
        d_blend_vs_1371 = blend_50_50_rae - rae_mean_bag
        print(f"   50/50 blend RAE(nb1371_mean, nb1352_median) = "
              f"{blend_50_50_rae:.4f}  "
              f"(d_vs_nb1352_med = {d_blend_vs_1352:+.4f}; "
              f"d_vs_nb1371_mean = {d_blend_vs_1371:+.4f})")

    # ---- Verdict (vs nb1352 mean is the dominant baseline) ----
    beats_nb1352_mean = rae_mean_bag < NB1352_MEAN_REF - DECISION_MARGIN
    beats_nb1352_median = rae_mean_bag < NB1352_MEDIAN_REF - DECISION_MARGIN
    blend_beats_nb1352 = (
        blend_50_50_rae is not None
        and blend_50_50_rae < NB1352_MEDIAN_REF - DECISION_MARGIN
    )

    if beats_nb1352_median:
        verdict = "CATBOOST_PRUNED_STANDALONE_BEATS_NB1352_NEW_PRIMARY_CANDIDATE"
    elif blend_beats_nb1352:
        verdict = "CATBOOST_PRUNED_BLEND_BEATS_NB1352_NEW_BLEND_CANDIDATE"
    elif beats_nb1352_mean:
        verdict = "CATBOOST_PRUNED_BEATS_NB1352_MEAN_BUT_NOT_MEDIAN"
    elif abs(rae_mean_bag - NB1352_MEDIAN_REF) < DECISION_MARGIN:
        verdict = "CATBOOST_PRUNED_TIES_NB1352_FLAT"
    elif rae_mean_bag < NB1341_MEAN_REF - DECISION_MARGIN:
        verdict = "CATBOOST_PRUNED_BEATS_NB1341_BUT_WORSE_THAN_NB1352"
    else:
        verdict = "CATBOOST_PRUNED_NO_GAIN_OVER_NB1341_OR_NB1352"
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
        "status": "ok",
        "catboost_version": _CATBOOST_VERSION,
        "anchor": ANCHOR,
        "features": "top-20 MACCS (SHAP-pruned) + pred_chembl_pec50 + sim_chembl",
        "feature_dim": feat_dim,
        "feature_cache_source": NB1352_TAG,
        "top_maccs_bit_indices_ranked": [int(b) for b in top_bit_idx.tolist()],
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
        "delta_mean_bag_vs_nb1352_mean": rae_mean_bag - NB1352_MEAN_REF,
        "delta_mean_bag_vs_nb1352_median": rae_mean_bag - NB1352_MEDIAN_REF,
        "delta_mean_bag_vs_nb1341_mean": rae_mean_bag - NB1341_MEAN_REF,
        "pearson_vs_nb1352_mean": pearson_vs_nb1352_mean,
        "pearson_vs_nb1352_median": pearson_vs_nb1352_median,
        "pearson_vs_nb1341_mean": pearson_vs_nb1341,
        "blend_50_50_nb1352_median_rae": blend_50_50_rae,
        "delta_blend_vs_nb1352_median": (
            blend_50_50_rae - NB1352_MEDIAN_REF
            if blend_50_50_rae is not None else None
        ),
        "delta_blend_vs_nb1371_mean": (
            blend_50_50_rae - rae_mean_bag
            if blend_50_50_rae is not None else None
        ),
        "beats_nb1352_mean": bool(beats_nb1352_mean),
        "beats_nb1352_median": bool(beats_nb1352_median),
        "blend_beats_nb1352_median": bool(blend_beats_nb1352),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1352_mean_ref": NB1352_MEAN_REF,
        "nb1352_median_ref": NB1352_MEDIAN_REF,
        "nb1341_mean_ref": NB1341_MEAN_REF,
        "nb1341_median_ref": NB1341_MEDIAN_REF,
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
        "feature_dim",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1352_mean",
        "delta_mean_bag_vs_nb1352_median",
        "delta_mean_bag_vs_nb1341_mean",
        "pearson_vs_nb1352_mean",
        "pearson_vs_nb1352_median",
        "pearson_vs_nb1341_mean",
        "blend_50_50_nb1352_median_rae",
        "delta_blend_vs_nb1352_median",
        "beats_nb1352_mean",
        "beats_nb1352_median",
        "blend_beats_nb1352_median",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
