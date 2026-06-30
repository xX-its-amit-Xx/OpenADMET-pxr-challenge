"""nb1351 -- Outer-bag CatBoost residual (5 outer seeds rebuild of nb1341).

PROTOCOL
--------
For each OUTER seed o in {0, 1, 7, 42, 137}:
    Inner seeds = [o*1000 + s for s in {0, 1, 7, 42, 137}]
    Per inner seed: 5-fold KFold cross-fit of CatBoostRegressor(
        loss_function='MAE', depth=4, iterations=200, learning_rate=0.05,
        l2_leaf_reg=5, random_seed=inner_seed, verbose=False)
      on residual = y_unb - nb1070_pred_oof using features
      X_unb = MACCS-167(unb) ++ pred_chembl_pec50(unb) ++ sim_chembl(unb)  (169)
    Per-outer corrected = anchor + residual_oof
    Per-outer mean_bag = mean over 5 inner seeds

Aggregate:
    per_outer_rae  -- pooled RAE(nb1341_o vs y_unb) for each o
    BoB MEAN       -- row-level mean across the 5 outer-seed nb1341_o vectors
    BoB MEDIAN     -- row-level median across the 5 outer-seed nb1341_o vectors

Reference: nb1341 mean_bag (outer=0 family) pooled RAE = 0.5420
           nb1342 mean_bag                            = 0.5486
           nb1242 mean_bag                            = 0.5431
Verdict NB1341_REPRODUCES if abs(per_outer_rae[0] - 0.5420) <= 0.003.

Outputs:
    scripts/nb1351_bag_catboost.py                (this file)
    data/processed/nb1351_summary.json
    data/processed/nb1351_per_outer_oof.npy       (5, 253) float32
    data/processed/nb1351_bob_mean_oof.npy        (253,)   float32
    data/processed/nb1351_bob_median_oof.npy     (253,)   float32
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

try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
    _CATBOOST_VERSION = __import__("catboost").__version__
    _CATBOOST_IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    _CATBOOST_AVAILABLE = False
    _CATBOOST_VERSION = None
    _CATBOOST_IMPORT_ERR = repr(e)

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1351"
ANCHOR = "nb1070"
NB1341_TAG = "nb1341"
NB1242_TAG = "nb1242"
NB1342_TAG = "nb1342"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASES = [0, 1, 7, 42, 137]   # inner_seed = outer * 1000 + base
RESID_FOLDS = 5

NB1070_REF = 0.5771
NB1242_REF = 0.5431
NB1341_REF = 0.5420    # nb1341 mean_bag pooled RAE (CatBoost residual, seeds 0..137)
NB1342_REF = 0.5486
REPRO_MARGIN = 0.003


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
        "nb1341_ref": NB1341_REF,
        "nb1242_ref": NB1242_REF,
        "nb1342_ref": NB1342_REF,
        "repro_margin": REPRO_MARGIN,
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
    print(f"{TAG} -- OUTER-BAG of nb1341 CatBoost residual (5 outer rebuild)")
    print(f"          outer seeds = {OUTER_SEEDS}")
    print(f"          inner bases = {INNER_BASES}   inner = outer*1000 + base")
    print(f"          learner     = CatBoost MAE depth=4 iter=200 lr=0.05 l2=5")
    print(f"          features    = MACCS-167 + pred_chembl + sim_chembl (169)")
    print(f"          nb1341 ref  = {NB1341_REF:.4f}  margin = {REPRO_MARGIN:.3f}")
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

    # ---- Feature cache (reuse nb1242 ChEMBL kNN outputs) ----
    maccs_te = np.load(DATA_PROCESSED / "te_maccs.npy")
    if maccs_te.shape[0] != 513:
        raise ValueError(f"te_maccs shape mismatch: {maccs_te.shape}")
    X_maccs_unb = maccs_te[unb_idx].astype(np.float32)

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
          f"(MACCS-167 + pred_chembl + sim_chembl)")

    # ---- Per-outer 5-seed inner bag ----
    print("\n" + "-" * 78)
    print(f"PER-OUTER REBUILD: 5 inner CatBoost bags x {RESID_FOLDS}-fold cross-fit")
    print("-" * 78)

    per_outer_corrected = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_rae: list[float] = []
    per_outer_records: list[dict] = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_o = time.time()
        inner_seeds = [int(o) * 1000 + int(b) for b in INNER_BASES]
        bag = np.zeros((len(inner_seeds), n_unb), dtype=np.float64)
        per_inner_rae = []
        for j, s in enumerate(inner_seeds):
            resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
            bag[j] = anchor + resid_oof_s
            per_inner_rae.append(float(rae(y_unb, bag[j])))
        nb1341_o = bag.mean(axis=0)
        per_outer_corrected[oi] = nb1341_o
        rae_o = float(rae(y_unb, nb1341_o))
        per_outer_rae.append(rae_o)

        repro_note = ""
        if o == 0:
            d = rae_o - NB1341_REF
            repro_note = (f"  [REPRO outer=0 vs nb1341 {NB1341_REF:.4f}: "
                          f"d={d:+.4f}, |d|<{REPRO_MARGIN}? "
                          f"{abs(d) < REPRO_MARGIN}]")

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "per_inner_rae": per_inner_rae,
            "rae_outer_mean_bag": rae_o,
            "delta_vs_nb1070": rae_o - rae_anchor,
            "elapsed_sec": round(time.time() - t_o, 1),
        })
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     per-inner RAE = [{', '.join(f'{r:.4f}' for r in per_inner_rae)}]")
        print(f"     mean_bag RAE  = {rae_o:.4f}  "
              f"(elapsed {time.time()-t_o:.1f}s){repro_note}")

    # ---- Aggregate per-outer ----
    arr = np.array(per_outer_rae)
    outer_mean = float(arr.mean())
    outer_std = float(arr.std())
    outer_min = float(arr.min())
    outer_max = float(arr.max())

    # ---- BoB mean / median across outer-seed mean_bag vectors ----
    bob_mean_oof = per_outer_corrected.mean(axis=0)
    bob_median_oof = np.median(per_outer_corrected, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    # ---- Compare to nb1342 / nb1242 BoB if available ----
    bob_pearson_vs_nb1242 = None
    nb1242_path = DATA_PROCESSED / f"{NB1242_TAG}_mean_bag_oof.npy"
    if nb1242_path.exists():
        nb1242_oof = np.load(nb1242_path).astype(np.float64)
        if nb1242_oof.shape == (n_unb,):
            from scipy.stats import pearsonr
            r, _ = pearsonr(bob_mean_oof, nb1242_oof)
            bob_pearson_vs_nb1242 = float(r)

    bob_pearson_vs_nb1342 = None
    nb1342_path = DATA_PROCESSED / f"{NB1342_TAG}_mean_bag_oof.npy"
    if nb1342_path.exists():
        nb1342_oof = np.load(nb1342_path).astype(np.float64)
        if nb1342_oof.shape == (n_unb,):
            from scipy.stats import pearsonr
            r, _ = pearsonr(bob_mean_oof, nb1342_oof)
            bob_pearson_vs_nb1342 = float(r)

    bob_pearson_vs_nb1341 = None
    nb1341_path = DATA_PROCESSED / f"{NB1341_TAG}_mean_bag_oof.npy"
    if nb1341_path.exists():
        nb1341_oof = np.load(nb1341_path).astype(np.float64)
        if nb1341_oof.shape == (n_unb,):
            from scipy.stats import pearsonr
            r, _ = pearsonr(bob_mean_oof, nb1341_oof)
            bob_pearson_vs_nb1341 = float(r)

    # ---- Verdict ----
    delta_outer0 = per_outer_rae[0] - NB1341_REF
    outer0_reproduces = abs(delta_outer0) <= REPRO_MARGIN
    delta_outer_mean_vs_1341 = outer_mean - NB1341_REF
    reproduces = abs(delta_outer_mean_vs_1341) <= REPRO_MARGIN

    if outer0_reproduces:
        verdict_repro = "NB1341_REPRODUCES"
    elif per_outer_rae[0] < NB1341_REF - REPRO_MARGIN:
        verdict_repro = "NB1341_PESSIMISTIC_OUTER0_BETTER"
    else:
        verdict_repro = "NB1341_OPTIMISTIC_OUTER0_WORSE"

    best_bob_rae = min(rae_bob_mean, rae_bob_median)
    best_bob_tag = "bob_mean" if rae_bob_mean <= rae_bob_median else "bob_median"

    if best_bob_rae < outer_mean - REPRO_MARGIN:
        verdict_var = "BOB_VARIANCE_REDUCTION_WIN"
    elif abs(best_bob_rae - outer_mean) < REPRO_MARGIN:
        verdict_var = "BOB_FLAT_VS_PER_OUTER_MEAN"
    else:
        verdict_var = "BOB_HURTS_VS_PER_OUTER_MEAN"

    if best_bob_rae < NB1242_REF - REPRO_MARGIN:
        verdict_vs_nb1242 = "BOB_BEATS_NB1242_NEW_CANDIDATE"
    elif abs(best_bob_rae - NB1242_REF) < REPRO_MARGIN:
        verdict_vs_nb1242 = "BOB_FLAT_VS_NB1242"
    else:
        verdict_vs_nb1242 = "BOB_HURTS_VS_NB1242"

    if best_bob_rae < NB1342_REF - REPRO_MARGIN:
        verdict_vs_nb1342 = "BOB_BEATS_NB1342"
    elif abs(best_bob_rae - NB1342_REF) < REPRO_MARGIN:
        verdict_vs_nb1342 = "BOB_FLAT_VS_NB1342"
    else:
        verdict_vs_nb1342 = "BOB_HURTS_VS_NB1342"

    print("\n" + "=" * 78)
    print("AGGREGATION + VERDICT")
    print("=" * 78)
    print(f"   per-outer RAE list           = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae)}]")
    print(f"   per-outer mean               = {outer_mean:.4f}")
    print(f"   per-outer std                = {outer_std:.4f}")
    print(f"   per-outer min                = {outer_min:.4f}")
    print(f"   per-outer max                = {outer_max:.4f}")
    print(f"   BoB MEAN   row-level RAE     = {rae_bob_mean:.4f}")
    print(f"   BoB MEDIAN row-level RAE     = {rae_bob_median:.4f}")
    print(f"   nb1341 mean_bag ref          = {NB1341_REF:.4f}")
    print(f"   nb1242 mean_bag ref          = {NB1242_REF:.4f}")
    print(f"   nb1342 mean_bag ref          = {NB1342_REF:.4f}")
    print(f"   delta(outer0 - nb1341 ref)   = {delta_outer0:+.4f}  "
          f"(margin {REPRO_MARGIN:.3f})")
    print(f"   delta(outer_mean - nb1341)   = {delta_outer_mean_vs_1341:+.4f}")
    print(f"   outer0 reproduces nb1341?    = {outer0_reproduces}")
    print(f"   verdict (repro)              = {verdict_repro}")
    print(f"   verdict (variance reduction) = {verdict_var}")
    print(f"   verdict (vs nb1242)          = {verdict_vs_nb1242}")
    print(f"   verdict (vs nb1342)          = {verdict_vs_nb1342}")
    if bob_pearson_vs_nb1341 is not None:
        print(f"   Pearson(BoB mean, nb1341)    = {bob_pearson_vs_nb1341:.4f}")
    if bob_pearson_vs_nb1242 is not None:
        print(f"   Pearson(BoB mean, nb1242)    = {bob_pearson_vs_nb1242:.4f}")
    if bob_pearson_vs_nb1342 is not None:
        print(f"   Pearson(BoB mean, nb1342)    = {bob_pearson_vs_nb1342:.4f}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_oof.npy",
            per_outer_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_outer_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "status": "ok",
        "catboost_version": _CATBOOST_VERSION,
        "anchor": ANCHOR,
        "features": "MACCS-167 + pred_chembl_pec50 + sim_chembl",
        "feature_dim": feat_dim,
        "feature_cache_source": NB1242_TAG,
        "outer_seeds": OUTER_SEEDS,
        "inner_bases": INNER_BASES,
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
        "per_outer_records": per_outer_records,
        "per_outer_rae": [float(x) for x in per_outer_rae],
        "outer_mean": outer_mean,
        "outer_std": outer_std,
        "outer_min": outer_min,
        "outer_max": outer_max,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "best_bob_tag": best_bob_tag,
        "best_bob_rae": best_bob_rae,
        "delta_outer0_vs_nb1341_ref": delta_outer0,
        "delta_outer_mean_vs_nb1341_ref": delta_outer_mean_vs_1341,
        "delta_best_bob_vs_outer_mean": best_bob_rae - outer_mean,
        "delta_best_bob_vs_nb1242_ref": best_bob_rae - NB1242_REF,
        "delta_best_bob_vs_nb1342_ref": best_bob_rae - NB1342_REF,
        "outer0_reproduces_nb1341": bool(outer0_reproduces),
        "outer_mean_reproduces_nb1341": bool(reproduces),
        "pearson_bob_mean_vs_nb1341": bob_pearson_vs_nb1341,
        "pearson_bob_mean_vs_nb1242": bob_pearson_vs_nb1242,
        "pearson_bob_mean_vs_nb1342": bob_pearson_vs_nb1342,
        "verdict_repro": verdict_repro,
        "verdict_var": verdict_var,
        "verdict_vs_nb1242": verdict_vs_nb1242,
        "verdict_vs_nb1342": verdict_vs_nb1342,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1341_ref": NB1341_REF,
        "nb1242_ref": NB1242_REF,
        "nb1342_ref": NB1342_REF,
        "repro_margin": REPRO_MARGIN,
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
        "rae_anchor_nb1070",
        "per_outer_rae",
        "outer_mean", "outer_std", "outer_min", "outer_max",
        "rae_bob_mean", "rae_bob_median",
        "best_bob_tag", "best_bob_rae",
        "delta_outer0_vs_nb1341_ref",
        "delta_outer_mean_vs_nb1341_ref",
        "delta_best_bob_vs_outer_mean",
        "delta_best_bob_vs_nb1242_ref",
        "delta_best_bob_vs_nb1342_ref",
        "outer0_reproduces_nb1341",
        "pearson_bob_mean_vs_nb1341",
        "pearson_bob_mean_vs_nb1242",
        "pearson_bob_mean_vs_nb1342",
        "verdict_repro", "verdict_var",
        "verdict_vs_nb1242", "verdict_vs_nb1342",
    ):
        print(f"  {k}: {res.get(k)}")
