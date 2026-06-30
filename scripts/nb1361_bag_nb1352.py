"""nb1361 -- Outer-bag VALIDATION of nb1352 (SHAP-pruned residual learner).

Protocol:
    For each outer seed in {0, 1, 7, 42, 137}:
        Rebuild the nb1352 inner 5-seed bag of shallow LGBM Huber on
            top-20 MACCS bits (SHAP-pruned, seed=0 SHAP frame)
              + pred_chembl_pec50 + sim
        over residual y_unb - nb1070_pred_oof, but with inner seeds remapped
        as inner_seeds = [outer*1000 + s for s in {0,1,7,42,137}].
        5-fold cross-fit per inner seed.
        Aggregate the 5-seed inner bag -> pooled corrected OOF; record per-outer
        pooled RAE.

    Outer=0 (inner seeds [0,1,7,42,137]) MUST reproduce nb1352 exactly
        (rae_mean_bag = 0.5323, rae_median_bag = 0.5315).

    Then aggregate the 5 outer mean-bag vectors row-wise:
        bob_mean_oof   = mean   across 5 outer vectors
        bob_median_oof = median across 5 outer vectors

Verdict NB1352_REPRODUCES if per-outer-mean within 0.003 of 0.5323.
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

TAG = "nb1361"
ANCHOR = "nb1070"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_SEEDS_BASE = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

TOP_K_MACCS = 20

NB1352_MEAN_REF = 0.5323
NB1352_MEDIAN_REF = 0.5315
REPRODUCE_MARGIN = 0.003


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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- outer-bag VALIDATION of nb1352 (SHAP-pruned residual)")
    print(f"          outer seeds = {OUTER_SEEDS}")
    print(f"          inner seeds = [outer*1000 + s for s in {INNER_SEEDS_BASE}]")
    print(f"          folds = {RESID_FOLDS}  top-{TOP_K_MACCS} MACCS + 2 ChEMBL feats")
    print(f"          nb1352 ref: mean={NB1352_MEAN_REF}  median={NB1352_MEDIAN_REF}")
    print("=" * 78)

    # ---- Load anchor + truth ----
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

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (same union as nb1352)")
    print("-" * 78)
    pool = _load_chembl_pool()

    # ---- Test InChIKey leak guard ----
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

    # ---- Morgan + kNN ChEMBL features ----
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

    # ---- MACCS-167 (unblind slice) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    n_maccs = int(X_maccs_te.shape[1])
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

    # ---- SHAP importance (seed=0, identical to nb1352) ----
    print("\n" + "-" * 78)
    print("SHAP IMPORTANCE FRAME (seed=0, identical to nb1352)")
    print("-" * 78)
    imp_full, imp_src = _compute_shap_importance(X_unb_full, residual, seed=0)
    print(f"   importance source = {imp_src}")
    maccs_imp = imp_full[:n_maccs]
    top_k = min(TOP_K_MACCS, n_maccs)
    top_bit_order = np.argsort(-maccs_imp)
    top_bit_idx = top_bit_order[:top_k].astype(int)
    print(f"   top-{top_k} MACCS bit indices (ranked): {top_bit_idx.tolist()}")

    X_maccs_unb_pruned = X_maccs_unb[:, top_bit_idx]
    X_unb_pruned = np.concatenate(
        [
            X_maccs_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim_pruned = X_unb_pruned.shape[1]
    print(f"   PRUNED feature matrix: {X_unb_pruned.shape}")

    # ---- Outer x Inner cross-fit ----
    print("\n" + "-" * 78)
    print(f"OUTER-BAG VALIDATION ({len(OUTER_SEEDS)} outer x "
          f"{len(INNER_SEEDS_BASE)} inner = {len(OUTER_SEEDS)*len(INNER_SEEDS_BASE)} configs)")
    print("-" * 78)

    n_outer = len(OUTER_SEEDS)
    outer_mean_bag = np.zeros((n_outer, n_unb), dtype=np.float64)
    outer_median_bag = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_rae_mean: list[float] = []
    per_outer_rae_median: list[float] = []
    per_outer_records: list[dict] = []

    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [o * 1000 + s for s in INNER_SEEDS_BASE]
        n_inner = len(inner_seeds)
        inner_corrected = np.zeros((n_inner, n_unb), dtype=np.float64)
        inner_per_seed_rae: list[float] = []
        for ii, isd in enumerate(inner_seeds):
            resid_oof_s = _residual_cross_fit_one_seed(X_unb_pruned, residual, isd)
            pred_corr_s = anchor + resid_oof_s
            inner_corrected[ii] = pred_corr_s
            r_s = float(rae(y_unb, pred_corr_s))
            inner_per_seed_rae.append(r_s)
        mean_bag_o = inner_corrected.mean(axis=0)
        median_bag_o = np.median(inner_corrected, axis=0)
        rae_mean_o = float(rae(y_unb, mean_bag_o))
        rae_median_o = float(rae(y_unb, median_bag_o))
        outer_mean_bag[oi] = mean_bag_o
        outer_median_bag[oi] = median_bag_o
        per_outer_rae_mean.append(rae_mean_o)
        per_outer_rae_median.append(rae_median_o)
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": [int(s) for s in inner_seeds],
            "inner_per_seed_rae": inner_per_seed_rae,
            "rae_mean_bag": rae_mean_o,
            "rae_median_bag": rae_median_o,
        })
        print(f"   outer {o:3d}  inner={inner_seeds}")
        print(f"             inner per-seed RAE = "
              f"[{', '.join(f'{r:.4f}' for r in inner_per_seed_rae)}]")
        print(f"             pooled mean_bag RAE   = {rae_mean_o:.4f}")
        print(f"             pooled median_bag RAE = {rae_median_o:.4f}")

    per_outer_rae_mean_arr = np.array(per_outer_rae_mean)
    per_outer_rae_median_arr = np.array(per_outer_rae_median)

    # ---- Row-level BoB aggregations across outer-seed mean_bag vectors ----
    bob_mean_oof = outer_mean_bag.mean(axis=0)
    bob_median_oof = np.median(outer_mean_bag, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    print("\n" + "-" * 78)
    print("AGGREGATIONS")
    print("-" * 78)
    print(f"   per-outer pooled mean_bag RAE   = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_mean)}]")
    print(f"   per-outer pooled median_bag RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_median)}]")
    print(f"   per-outer mean   (of mean_bag)  = {per_outer_rae_mean_arr.mean():.4f}")
    print(f"   per-outer std    (of mean_bag)  = {per_outer_rae_mean_arr.std():.4f}")
    print(f"   per-outer median (of mean_bag)  = {np.median(per_outer_rae_mean_arr):.4f}")
    print(f"   BoB mean   RAE (row-mean across 5 outers) = {rae_bob_mean:.4f}")
    print(f"   BoB median RAE (row-med  across 5 outers) = {rae_bob_median:.4f}")
    print(f"   nb1352 reference  mean={NB1352_MEAN_REF:.4f}  "
          f"median={NB1352_MEDIAN_REF:.4f}")

    # ---- Reproduction check for outer=0 ----
    rae_outer0 = per_outer_rae_mean[0]
    rae_outer0_median = per_outer_rae_median[0]
    delta_outer0_mean = rae_outer0 - NB1352_MEAN_REF
    delta_outer0_median = rae_outer0_median - NB1352_MEDIAN_REF
    print(f"\n   outer=0 reproduces nb1352?  "
          f"d_mean={delta_outer0_mean:+.5f}  d_median={delta_outer0_median:+.5f}")
    outer0_reproduces = (
        abs(delta_outer0_mean) < REPRODUCE_MARGIN
        and abs(delta_outer0_median) < REPRODUCE_MARGIN
    )
    per_outer_mean_in_band = bool(
        abs(per_outer_rae_mean_arr.mean() - NB1352_MEAN_REF) < REPRODUCE_MARGIN
    )

    if per_outer_mean_in_band:
        verdict = "NB1352_REPRODUCES"
    else:
        verdict = "NB1352_DRIFTS"
    print(f"   verdict = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_outer_mean_bag.npy",
            outer_mean_bag.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_outer_median_bag.npy",
            outer_median_bag.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_unb": n_unb,
        "n_maccs_bits": n_maccs,
        "feat_dim_full": int(feat_dim_full),
        "feat_dim_pruned": int(feat_dim_pruned),
        "shap_importance_source": imp_src,
        "top_k_maccs": int(top_k),
        "top_maccs_bit_indices_ranked": [int(b) for b in top_bit_idx.tolist()],
        "outer_seeds": OUTER_SEEDS,
        "inner_seeds_base": INNER_SEEDS_BASE,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_nb1070": rae_anchor,
        "per_outer_records": per_outer_records,
        "per_outer_rae_mean_bag": per_outer_rae_mean,
        "per_outer_rae_median_bag": per_outer_rae_median,
        "per_outer_mean_of_meanbag": float(per_outer_rae_mean_arr.mean()),
        "per_outer_std_of_meanbag": float(per_outer_rae_mean_arr.std()),
        "per_outer_median_of_meanbag": float(np.median(per_outer_rae_mean_arr)),
        "per_outer_mean_of_medianbag": float(per_outer_rae_median_arr.mean()),
        "per_outer_std_of_medianbag": float(per_outer_rae_median_arr.std()),
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "nb1352_mean_ref": NB1352_MEAN_REF,
        "nb1352_median_ref": NB1352_MEDIAN_REF,
        "reproduce_margin": REPRODUCE_MARGIN,
        "outer0_rae_mean_bag": rae_outer0,
        "outer0_rae_median_bag": rae_outer0_median,
        "outer0_delta_mean_vs_nb1352": delta_outer0_mean,
        "outer0_delta_median_vs_nb1352": delta_outer0_median,
        "outer0_reproduces": bool(outer0_reproduces),
        "per_outer_mean_in_band": per_outer_mean_in_band,
        "verdict": verdict,
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
        "outer_seeds",
        "per_outer_rae_mean_bag",
        "per_outer_rae_median_bag",
        "per_outer_mean_of_meanbag",
        "per_outer_std_of_meanbag",
        "per_outer_median_of_meanbag",
        "rae_bob_mean",
        "rae_bob_median",
        "outer0_rae_mean_bag",
        "outer0_rae_median_bag",
        "outer0_delta_mean_vs_nb1352",
        "outer0_reproduces",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
