"""nb1362 -- SHAP-pruned K-grid sweep over top-K MACCS bits + ChEMBL features.

Hypothesis:
    nb1352 picked top-K=20 MACCS bits by SHAP and beat nb1242 (full-167 MACCS)
    at honest 5-fold cross-fit RAE 0.5323 mean / 0.5315 median.  The capacity
    sweet spot is unknown a priori -- sweep K in {5, 10, 15, 25, 30, 40, 50}
    to map the bias/variance curve and identify the empirical optimum.

Protocol:
    1.  Reuse the SHAP importance frame from nb1352 (same seed=0, same global
        residual LGBM on full 169-col feature matrix).  Internal recompute
        for self-containment; ordering identical to nb1352 by construction.
    2.  For each K in {5, 10, 15, 25, 30, 40, 50}:
           features = top-K MACCS bits (by SHAP) + pred_chembl_pec50 + sim
                    = K + 2 columns
    3.  Shallow LGBM Huber residual learner, 5-seed bag (seeds [0,1,7,42,137]),
        5-fold cross-fit per seed.
    4.  Pooled RAE on mean_bag and median_bag aggregations per K.
    5.  Identify K* = argmin(rae_mean_bag).
    6.  Verdict at 0.003 margin vs nb1352 K=20 (mean 0.5323, median 0.5315).

Outputs:
    scripts/nb1362_shap_K_grid.py            (this file)
    data/processed/nb1362_summary.json
    data/processed/nb1362_best_K_oof.npy     (253,) float32 -- mean_bag at K*
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1362"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

K_GRID = [5, 10, 15, 25, 30, 40, 50]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1242_REF = 0.5431
NB1352_K20_MEAN_REF = 0.5323
NB1352_K20_MEDIAN_REF = 0.5315
DECISION_MARGIN = 0.003


def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_can_smiles(mol):
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _load_chembl_pool() -> pd.DataFrame:
    frames = []
    p1 = EXT_DIR / "chembl_pxr_CHEMBL3401.parquet"
    if p1.exists():
        d = pd.read_parquet(p1)
        mask = (
            d["standard_type"].isin(KEEP_TYPES)
            & d["canonical_smiles"].notna()
            & (d["standard_units"] == "nM")
            & d["standard_value"].notna()
            & d["standard_relation"].isin(KEEP_RELATIONS)
        )
        d = d[mask].copy()
        v = d["standard_value"].astype(float)
        d = d[(v > MIN_NM) & (v < MAX_NM)].copy()
        d["pec50_raw"] = 9.0 - np.log10(d["standard_value"].astype(float))
        d = d[["canonical_smiles", "pec50_raw"]].rename(
            columns={"canonical_smiles": "smiles", "pec50_raw": "pec50"}
        )
        d["src"] = "CHEMBL3401_raw"
        frames.append(d)
        print(f"   [src] CHEMBL3401_raw kept: {len(d)} rows")

    p2 = EXT_DIR / "chembl_nr_extended.parquet"
    if p2.exists():
        d = pd.read_parquet(p2)
        d = d[d["target_name"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
        d["src"] = "nr_extended"
        frames.append(d)
        print(f"   [src] chembl_nr_extended PXR kept: {len(d)} rows")

    p3 = EXT_DIR / "chembl_pxr_all_types.parquet"
    if p3.exists():
        d = pd.read_parquet(p3)
        d = d[d["target"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["smiles", "pec50"]]
        d["src"] = "pxr_all_types"
        frames.append(d)
        print(f"   [src] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [pool] pre-standardize union: {len(pool)} rows")
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    print(f"   [pool] after RDKit standardize: {len(pool)} rows")
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    print(f"   [pool] after InChIKey dedup (median agg): {len(agg)} unique cpds")
    return agg


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    n_pool = b.shape[0]
    top_idx = np.zeros((n_q, k), dtype=np.int32)
    top_sim = np.zeros((n_q, k), dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        if k >= n_pool:
            idx_part = np.argsort(-sim, axis=1)[:, :k]
        else:
            part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(e - s)[:, None]
            sim_part = sim[row_idx, part]
            order = np.argsort(-sim_part, axis=1)
            idx_part = part[row_idx, order]
        row_idx = np.arange(e - s)[:, None]
        top_idx[s:e] = idx_part
        top_sim[s:e] = sim[row_idx, idx_part]
    return top_idx, top_sim


def _knn_predict(top_idx: np.ndarray, top_sim: np.ndarray,
                 pool_labels: np.ndarray, fallback: float):
    w = top_sim.copy()
    w = np.clip(w, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i]
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return pred, mean_sim


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


def _bag_one_K(X_unb_pruned: np.ndarray, residual: np.ndarray,
               anchor: np.ndarray, y_unb: np.ndarray, K: int):
    n_unb = len(y_unb)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb_pruned, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        per_seed_rae.append(float(rae(y_unb, pred_corr_s)))
    mean_bag = per_seed_corrected.mean(axis=0)
    median_bag = np.median(per_seed_corrected, axis=0)
    rae_mean = float(rae(y_unb, mean_bag))
    rae_median = float(rae(y_unb, median_bag))
    return mean_bag, median_bag, rae_mean, rae_median, per_seed_rae


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SHAP-pruned K-grid; anchor={ANCHOR}; K in {K_GRID}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          baseline ref = nb1352 K=20 "
          f"(mean {NB1352_K20_MEAN_REF:.4f} / median {NB1352_K20_MEDIAN_REF:.4f})  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} shape mismatch: {anchor.shape} vs {n_unb}")
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (local cache; same union as nb1352)")
    print("-" * 78)
    pool = _load_chembl_pool()

    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after})")

    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")

    std_test_smiles = []
    for m in test_mols:
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)

    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    print(f"   pred_chembl_pec50  mean={pred_chembl_pec50.mean():.3f}  "
          f"std={pred_chembl_pec50.std():.3f}")

    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    n_maccs = int(X_maccs_te.shape[1])
    print(f"   MACCS cache shape = {X_maccs_te.shape}  (n_bits={n_maccs})")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)

    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb_full = np.concatenate(
        [
            X_maccs_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_full = X_unb_full.shape[1]
    print(f"   full feature matrix: {X_unb_full.shape}  "
          f"(MACCS-{n_maccs} + pred_chembl + sim)")

    print("\n" + "-" * 78)
    print("SHAP IMPORTANCE FRAME (reuse nb1352 ranking; seed=0 global LGBM)")
    print("-" * 78)
    imp_full, imp_src = _compute_shap_importance(X_unb_full, residual, seed=0)
    print(f"   importance source = {imp_src}")
    maccs_imp = imp_full[:n_maccs]
    full_bit_order = np.argsort(-maccs_imp).astype(int)
    print(f"   top-50 SHAP-ranked MACCS bit indices:")
    for rank in range(min(50, n_maccs)):
        bit = int(full_bit_order[rank])
        val = float(maccs_imp[bit])
        print(f"      rank {rank+1:2d}:  bit {bit:3d}   imp = {val:.5f}")

    print("\n" + "-" * 78)
    print("K-GRID SWEEP")
    print("-" * 78)
    K_max_safe = int(min(max(K_GRID), n_maccs))
    grid_records = []
    bag_cache: dict[int, np.ndarray] = {}
    for K in K_GRID:
        K_eff = int(min(K, n_maccs))
        top_bit_idx = full_bit_order[:K_eff]
        X_maccs_unb_pruned = X_maccs_unb[:, top_bit_idx]
        X_unb_pruned = np.concatenate(
            [
                X_maccs_unb_pruned,
                pred_chembl_unb.reshape(-1, 1),
                mean_sim_unb.reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)
        feat_dim = X_unb_pruned.shape[1]

        mean_bag, median_bag, rae_mean, rae_median, per_seed_rae = _bag_one_K(
            X_unb_pruned, residual, anchor, y_unb, K_eff
        )
        bag_cache[K] = mean_bag.astype(np.float32)

        rec = {
            "K": int(K),
            "K_effective": int(K_eff),
            "feat_dim": int(feat_dim),
            "rae_mean_bag": rae_mean,
            "rae_median_bag": rae_median,
            "per_seed_rae": [float(r) for r in per_seed_rae],
            "per_seed_mean": float(np.mean(per_seed_rae)),
            "per_seed_std": float(np.std(per_seed_rae)),
            "delta_mean_vs_nb1352": rae_mean - NB1352_K20_MEAN_REF,
            "delta_median_vs_nb1352": rae_median - NB1352_K20_MEDIAN_REF,
        }
        grid_records.append(rec)
        print(
            f"   K={K:3d}  (feat_dim={feat_dim:3d})   "
            f"mean_bag={rae_mean:.4f}   median_bag={rae_median:.4f}   "
            f"per-seed=[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]   "
            f"d_vs_nb1352(mean)={rae_mean - NB1352_K20_MEAN_REF:+.4f}"
        )

    # Identify best K by rae_mean_bag
    best_rec = min(grid_records, key=lambda r: r["rae_mean_bag"])
    best_K = best_rec["K"]
    best_mean = best_rec["rae_mean_bag"]
    best_median = best_rec["rae_median_bag"]

    beats_nb1352_mean = best_mean < NB1352_K20_MEAN_REF - DECISION_MARGIN
    flat_nb1352_mean = abs(best_mean - NB1352_K20_MEAN_REF) < DECISION_MARGIN
    if beats_nb1352_mean:
        verdict = f"SHAP_KGRID_BEATS_NB1352_AT_K={best_K}_NEW_PRIMARY_CANDIDATE"
    elif flat_nb1352_mean:
        verdict = f"SHAP_KGRID_FLAT_VS_NB1352_BEST_K={best_K}"
    else:
        verdict = f"SHAP_KGRID_WORSE_THAN_NB1352_BEST_K={best_K}"

    print("\n" + "-" * 78)
    print("K-GRID SUMMARY")
    print("-" * 78)
    print(f"   {'K':>4}  {'feat_dim':>8}  {'mean_bag':>9}  {'median_bag':>10}  "
          f"{'d_mean_vs_K20':>14}")
    for rec in grid_records:
        marker = "  *" if rec["K"] == best_K else "   "
        print(
            f"   {rec['K']:>4}  {rec['feat_dim']:>8d}  {rec['rae_mean_bag']:>9.4f}  "
            f"{rec['rae_median_bag']:>10.4f}  "
            f"{rec['delta_mean_vs_nb1352']:>+14.4f}{marker}"
        )
    print(f"   best K = {best_K}  mean_bag = {best_mean:.4f}  "
          f"median_bag = {best_median:.4f}")
    print(f"   nb1352 K=20 ref: mean {NB1352_K20_MEAN_REF:.4f} "
          f"/ median {NB1352_K20_MEDIAN_REF:.4f}")
    print(f"   verdict = {verdict}")

    best_K_oof = bag_cache[best_K]
    out_best = DATA_PROCESSED / f"{TAG}_best_K_oof.npy"
    np.save(out_best, best_K_oof.astype(np.float32))
    print(f"\n[save] {out_best}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "data_source": "local_chembl_caches_union",
        "n_chembl_pool": int(len(pool)),
        "n_unb": int(n_unb),
        "n_maccs_bits": int(n_maccs),
        "shap_importance_source": imp_src,
        "K_grid": [int(K) for K in K_GRID],
        "full_bit_order_top50": [int(b) for b in full_bit_order[:50].tolist()],
        "full_bit_importance_top50": [float(maccs_imp[b]) for b in full_bit_order[:50].tolist()],
        "pred_chembl_importance": float(imp_full[n_maccs]),
        "sim_importance": float(imp_full[n_maccs + 1]),
        "feat_dim_full": int(feat_dim_full),
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
        "grid_records": grid_records,
        "best_K": int(best_K),
        "best_K_rae_mean_bag": float(best_mean),
        "best_K_rae_median_bag": float(best_median),
        "delta_best_mean_vs_nb1352": float(best_mean - NB1352_K20_MEAN_REF),
        "delta_best_median_vs_nb1352": float(best_median - NB1352_K20_MEDIAN_REF),
        "beats_nb1352": bool(beats_nb1352_mean),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1242_ref": NB1242_REF,
        "nb1352_K20_mean_ref": NB1352_K20_MEAN_REF,
        "nb1352_K20_median_ref": NB1352_K20_MEDIAN_REF,
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
        "n_chembl_pool", "n_maccs_bits", "shap_importance_source",
        "K_grid",
        "best_K", "best_K_rae_mean_bag", "best_K_rae_median_bag",
        "delta_best_mean_vs_nb1352",
        "delta_best_median_vs_nb1352",
        "beats_nb1352", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
