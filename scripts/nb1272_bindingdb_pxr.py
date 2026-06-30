"""nb1272 -- BindingDB PXR external bioactivity kNN residual feature.

Hypothesis:
    BindingDB is a public database of measured binding affinities (Kd, Ki,
    IC50) for protein-ligand pairs.  The locally cached `bindingdb_nr_data`
    parquet contains 346 PXR rows (UniProt O75469, human PXR) drawn directly
    from BindingDB literature curation, with pEC50 mean 6.21 (HIGHER than the
    ChEMBL CHEMBL3401 pool mean 5.60).  Even where InChIKeys overlap ChEMBL
    (208/346), the source curation differs (BindingDB literature-extracted
    IC50/Ki vs ChEMBL ChEMBL3401 EC50/AC50).  We test whether a separate
    BindingDB-only kNN residual feature improves on nb1242 (ChEMBL kNN, RAE
    0.5431) by exposing a complementary potency context.

Protocol:
    1. Try BindingDB REST.  No public anonymous REST endpoint for sustained
       cross-target ligand pull exists; the UniProt-keyed Marvin endpoint is
       interactive-only.  Fall back to the local cached pull:
           data/external/bindingdb_nr_data.parquet  (5690 NR rows, 346 PXR)
       Filter target_name == 'PXR'  AND  uniprot == 'O75469'.
    2. Standardize SMILES via src.pxr.chem.standardize, dedupe by InChIKey
       (median pEC50 per InChIKey).  Drop test-set InChIKey overlaps
       (leak guard).
    3. Compute Morgan-2048 for BindingDB pool and 513 test rows.
    4. Per 513-test row: k=5 Tanimoto neighbors over BindingDB pool,
       similarity-weighted mean pEC50 -> pred_bdb_pec50_i; also save
       mean_sim_i.  Rows with no neighbor (top1 sim < eps) fall back to pool
       median.
    5. Residual learner anchored on nb1070_pred_oof.  Features: MACCS-167
       on 253 unblind  ++  pred_bdb_pec50[unb_idx]  ++  mean_sim[unb_idx]
       -> (253, 169).
    6. 5-seed shallow LGBM Huber bag (identical capacity to nb1242), 5-fold
       cross-fit per seed, mean-bag pooled RAE.
    7. Verdict at 0.003 margin vs nb1242 (0.5431).

Outputs:
    scripts/nb1272_bindingdb_pxr.py        (this file)
    data/processed/nb1272_summary.json
    data/processed/nb1272_mean_bag_oof.npy           (253,) float32 (if achieved)
    data/processed/nb1272_per_seed_corrected_oof.npy (5, 253) float32
    data/processed/nb1272_median_bag_oof.npy         (253,) float32
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

TAG = "nb1272"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

BDB_NR_PARQUET = EXT_DIR / "bindingdb_nr_data.parquet"
BDB_PXR_DIRECT_PARQUET = EXT_DIR / "bindingdb_pxr_direct.parquet"

PXR_UNIPROT = "O75469"

KNN_K = 5
SIM_FLOOR = 1e-6
PEC50_LOW = 3.0
PEC50_HIGH = 11.0

NB1070_REF = 0.5771
NB1242_REF = 0.5431
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


def _probe_bindingdb_rest() -> dict:
    """Attempt to confirm reachability of BindingDB REST endpoints.

    BindingDB hosts a few JSP endpoints but no documented anonymous REST API
    for bulk per-target ligand pull (UniProt-keyed endpoints require a Marvin
    sketch payload).  We do NOT attempt to scrape from a background script;
    we just verify the host is up and note the fallback path.
    """
    import urllib.request, urllib.error, socket
    out = {"probed": True, "reachable": False, "note": ""}
    url = "https://www.bindingdb.org/rwd/bind/index.jsp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pxr-challenge-nb1272/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            code = r.getcode()
            out["reachable"] = (code == 200)
            out["http_status"] = code
        out["note"] = (
            "BindingDB host is reachable but exposes no documented anonymous "
            "REST endpoint for UniProt-keyed bulk per-target ligand pull. "
            "We use the local cached pull (data/external/bindingdb_nr_data.parquet)."
        )
    except (urllib.error.URLError, socket.timeout, Exception) as e:
        out["reachable"] = False
        out["note"] = f"BindingDB REST probe failed: {type(e).__name__}: {e}"
    return out


def _load_bindingdb_pxr_pool() -> tuple[pd.DataFrame, dict]:
    """Load BindingDB PXR cache (filter target_name=PXR & uniprot=O75469).
    Standardize SMILES, dedupe InChIKey (median pEC50).

    Returns (frame[inchikey, std_smiles, pec50, n_meas, src], stats dict).
    """
    stats = {"used_path": None, "raw_rows": 0, "pxr_rows": 0}
    if not BDB_NR_PARQUET.exists():
        raise FileNotFoundError(
            f"BindingDB NR cache not found: {BDB_NR_PARQUET}. "
            "Cannot proceed without REST access."
        )
    stats["used_path"] = str(BDB_NR_PARQUET)
    d = pd.read_parquet(BDB_NR_PARQUET)
    stats["raw_rows"] = int(len(d))
    print(f"   [bdb] raw rows in cache: {len(d)}")
    print(f"   [bdb] target_name counts: "
          f"{d['target_name'].value_counts().head(10).to_dict()}")

    pxr = d[(d["target_name"] == "PXR") & (d["uniprot"] == PXR_UNIPROT)].copy()
    stats["pxr_rows"] = int(len(pxr))
    print(f"   [bdb] PXR rows (uniprot {PXR_UNIPROT}): {len(pxr)}")
    if len(pxr) == 0:
        raise RuntimeError("BindingDB cache contained zero PXR rows.")

    pxr = pxr[pxr["pec50"].notna() & pxr["std_smiles"].notna()].copy()
    pxr["pec50"] = pxr["pec50"].astype(float)
    pxr = pxr[(pxr["pec50"] >= PEC50_LOW) & (pxr["pec50"] <= PEC50_HIGH)].copy()
    print(f"   [bdb] after pEC50 range filter: {len(pxr)} rows")

    # Re-standardize defensively (don't trust cached std_smiles fully)
    mols = pxr["std_smiles"].apply(standardize)
    pxr["inchikey_local"] = mols.apply(_safe_inchikey)
    pxr["std_smiles_local"] = mols.apply(_safe_can_smiles)
    pxr = pxr[pxr["inchikey_local"].notna() & pxr["std_smiles_local"].notna()].copy()
    print(f"   [bdb] after RDKit standardize: {len(pxr)} rows")

    if "affinity_type" in pxr.columns:
        print(f"   [bdb] affinity_type counts: "
              f"{pxr['affinity_type'].value_counts().head(8).to_dict()}")
        stats["affinity_type_counts"] = (
            pxr["affinity_type"].value_counts().to_dict()
        )

    agg = (
        pxr.groupby("inchikey_local", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles_local", "first"),
             n_meas=("pec50", "count"))
        .rename(columns={"inchikey_local": "inchikey"})
    )
    agg["src"] = "bindingdb_nr_O75469"
    print(f"   [bdb] after InChIKey dedup (median agg): {len(agg)} unique cpds")
    print(f"   [bdb] n_meas: min={agg['n_meas'].min()} "
          f"median={agg['n_meas'].median():.1f} max={agg['n_meas'].max()}")
    print(f"   [bdb] pEC50: mean={agg['pec50'].mean():.3f} "
          f"std={agg['pec50'].std():.3f} "
          f"min={agg['pec50'].min():.3f} max={agg['pec50'].max():.3f}")
    return agg, stats


def _morgan_uint8(smiles_list) -> np.ndarray:
    return morgan_fp_batch(list(smiles_list))


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


def _knn_predict(top_idx, top_sim, pool_labels, fallback):
    w = np.clip(top_sim, 0.0, 1.0)
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


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- BindingDB PXR external kNN residual feature; "
          f"shallow residual-LGBM bag on nb1070 anchor")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + pred_bdb_pec50 + sim  (169)")
    print(f"          verdict_ref = nb1242 (0.5431) at margin {DECISION_MARGIN}")
    print("=" * 78)

    # ---- BindingDB REST probe (informational) ----
    print("\n" + "-" * 78)
    print("BINDINGDB REST PROBE")
    print("-" * 78)
    rest_probe = _probe_bindingdb_rest()
    print(f"   probe: reachable={rest_probe['reachable']}")
    print(f"   note: {rest_probe['note']}")

    # ---- Load anchor + truth ----
    print("\n" + "-" * 78)
    print("LOAD ANCHOR + UNBLIND TRUTH")
    print("-" * 78)
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
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- BindingDB pool ----
    print("\n" + "-" * 78)
    print("BINDINGDB PXR POOL (local cache; REST probe done above)")
    print("-" * 78)
    pool, pool_load_stats = _load_bindingdb_pxr_pool()

    # ---- Test InChIKey leak guard ----
    print("\n" + "-" * 78)
    print("TEST-SET LEAK GUARD (drop any BindingDB cpd whose InChIKey is in 513)")
    print("-" * 78)
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after} "
          f"test-overlapping cpds)")

    # ---- Morgan FPs ----
    print("\n" + "-" * 78)
    print("MORGAN-2048 FINGERPRINTS")
    print("-" * 78)
    fp_pool = _morgan_uint8(pool["std_smiles"].tolist())
    print(f"   pool FP: {fp_pool.shape}  density={fp_pool.mean():.4f}")
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        n_drop = int((~keep_pool).sum())
        print(f"   dropped {n_drop} pool rows with zero FP")
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
    print(f"   test FP: {fp_test.shape}  density={fp_test.mean():.4f}")

    # ---- Tanimoto kNN ----
    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN (k={KNN_K}) -- test (513) vs BindingDB pool ({len(pool)})")
    print("-" * 78)
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_bdb_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    top1_sim = top_sim[:, 0]
    print(f"   pred_bdb_pec50  mean={pred_bdb_pec50.mean():.3f}  "
          f"std={pred_bdb_pec50.std():.3f}  "
          f"min={pred_bdb_pec50.min():.3f}  max={pred_bdb_pec50.max():.3f}")
    print(f"   top1 sim   p10={np.percentile(top1_sim, 10):.3f}  "
          f"p50={np.percentile(top1_sim, 50):.3f}  "
          f"p90={np.percentile(top1_sim, 90):.3f}  "
          f"max={top1_sim.max():.3f}")
    print(f"   mean5 sim  p10={np.percentile(mean_sim, 10):.3f}  "
          f"p50={np.percentile(mean_sim, 50):.3f}  "
          f"p90={np.percentile(mean_sim, 90):.3f}")
    n_zero_neighbor = int((top1_sim < SIM_FLOOR).sum())
    print(f"   {n_zero_neighbor}/513 test rows had no neighbor "
          f"(fell back to pool median {pool_median:.3f})")

    # ---- MACCS slice ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"   MACCS unb shape = {X_maccs_unb.shape}")

    pred_bdb_unb = pred_bdb_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_bdb_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + pred_bdb + sim)")

    # ---- Per-seed cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow LGBM Huber, dim={feat_dim})")
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
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (ChEMBL kNN residual bag)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "BINDINGDB_KNN_FEAT_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif beats_nb1070:
        verdict = "BINDINGDB_KNN_FEAT_HELPS_NB1070_BUT_NOT_NB1242"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "BINDINGDB_KNN_FEAT_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "BINDINGDB_KNN_FEAT_HURTS_NB1070"
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
        "data_source": "local_bindingdb_nr_data_cache",
        "data_source_path": str(BDB_NR_PARQUET),
        "bindingdb_rest_probe": rest_probe,
        "bindingdb_status": (
            "bindingdb_local_cache_used"
            if pool_load_stats["pxr_rows"] > 0
            else "bindingdb_not_accessible"
        ),
        "pxr_uniprot": PXR_UNIPROT,
        "raw_cache_rows": pool_load_stats["raw_rows"],
        "pxr_rows_after_target_filter": pool_load_stats["pxr_rows"],
        "n_bindingdb_pool_pre_leak_guard": int(n_before),
        "n_bindingdb_pool_post_leak_guard": int(n_after),
        "test_inchikeys_in_pool_dropped": int(n_before - n_after),
        "pool_pec50_mean": float(pool_labels.mean()),
        "pool_pec50_std": float(pool_labels.std()),
        "pool_pec50_median": pool_median,
        "knn_k": KNN_K,
        "top1_sim_p10": float(np.percentile(top1_sim, 10)),
        "top1_sim_p50": float(np.percentile(top1_sim, 50)),
        "top1_sim_p90": float(np.percentile(top1_sim, 90)),
        "top1_sim_max": float(top1_sim.max()),
        "mean5_sim_p10": float(np.percentile(mean_sim, 10)),
        "mean5_sim_p50": float(np.percentile(mean_sim, 50)),
        "mean5_sim_p90": float(np.percentile(mean_sim, 90)),
        "n_zero_neighbor_rows": n_zero_neighbor,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": feat_dim,
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
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
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
        "bindingdb_status",
        "n_bindingdb_pool_post_leak_guard",
        "pool_pec50_mean", "pool_pec50_std",
        "top1_sim_p10", "top1_sim_p50", "top1_sim_p90",
        "n_zero_neighbor_rows",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1242",
        "beats_nb1070", "beats_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
