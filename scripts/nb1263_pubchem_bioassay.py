"""nb1263 -- PubChem BioAssay PXR external bioactivity kNN residual feature.

Hypothesis:
    PubChem hosts large dose-response qHTS PXR assays (Tox21) that are
    independent of ChEMBL.  Combined human + rat PXR transactivation panels
    provide a SCAFFOLD-DIVERSE potency anchor for kNN, complementing the
    ChEMBL pool used by nb1242.

Source AIDs (PubChem REST API, live fetch):
    AID 720659  -- qHTS for small molecule activators of the human PXR
                   (Tox21, dose response, AC50 in microMolar)  ~2327 cpds
    AID 651751  -- qHTS for small molecule activators of the rat PXR
                   (Tox21, dose response, AC50 in microMolar)  ~2327 cpds
    AID 1259395 -- larger PXR-tagged screen                    ~7671 cpds
                   (single concentration; outcome only)

Pipeline:
    1. Pull PubChem REST CSV table for each AID; parse CID, SMILES,
       outcome, potency (uM).  Compute pEC50 = 6 - log10(potency_uM) where
       potency is finite and > 0 (i.e. dose-response active rows).  Drop
       inactive / inconclusive rows.
    2. Standardize SMILES via src/pxr/chem.standardize, InChIKey dedup
       (median pEC50 across AIDs and replicates).
    3. Leak guard: drop any PubChem InChIKey present in the 513-test set.
    4. Build Morgan-2048 fingerprints for PubChem pool and 513 test rows.
       For each test row: top-k=5 Tanimoto neighbors, similarity-weighted
       mean of pool pEC50 -> pred_pubchem_pec50.  Mean similarity -> sim.
       Rows with no neighbor fall back to pool median pEC50.
    5. Residual learner:  anchor = nb1070_pred_oof on 253 unblind rows;
       residual = y_unb - nb1070_oof;
       features = concat[MACCS-167(unb), pred_pubchem_pec50[unb_idx],
                         sim[unb_idx]]  -> (253, 169).
    6. 5-seed shallow LGBM Huber residual bag, 5-fold cross-fit per seed,
       mean-bag pooled RAE.
    7. Verdict at 0.003 margin vs nb1070 (0.5771), nb1183 (0.5513),
       nb1211 (0.5451), nb1242 (0.5431).

If PubChem REST fails entirely (network down, throttled, no usable rows),
report status='failed' and an explanation; this is acceptable as a NULL
trial.  The hypothesis is then "no external PubChem PXR signal available
in this run".

Outputs:
    scripts/nb1263_pubchem_bioassay.py             (this file)
    data/external/pubchem_pxr_pool.parquet         (pool cache)
    data/processed/nb1263_summary.json
    data/processed/nb1263_mean_bag_oof.npy         (253,) float32
    data/processed/nb1263_per_seed_corrected_oof.npy (5, 253) float32
    data/processed/nb1263_median_bag_oof.npy       (253,) float32
"""
from __future__ import annotations

import io
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
import requests
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1263"
ANCHOR = "nb1070"

# AIDs to attempt.  Each entry: (aid, label, is_dose_response).
# Dose-response yields AC50 in microMolar -> convertible to pEC50.
PUBCHEM_AIDS = [
    (720659, "Tox21_human_PXR", True),
    (651751, "Tox21_rat_PXR",   True),
]

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
POOL_CACHE = EXT_DIR / "pubchem_pxr_pool.parquet"

KNN_K = 5
SIM_FLOOR = 1e-6

NB1070_REF = 0.5771
NB1183_REF = 0.5513
NB1211_REF = 0.5451
NB1242_REF = 0.5431
DECISION_MARGIN = 0.003

REQ_TIMEOUT = 90
REQ_SLEEP = 0.5


def _safe_inchikey(mol) -> str | None:
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_can_smiles(mol) -> str | None:
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _fetch_aid_csv(aid: int) -> pd.DataFrame | None:
    """Fetch /pug/assay/aid/<AID>/CSV.  Returns parsed dataframe or None on fail.

    PubChem CSV header structure (Tox21-style dose-response):
        Row 0: column names (PUBCHEM_RESULT_TAG, PUBCHEM_SID, PUBCHEM_CID,
               PUBCHEM_EXT_DATASOURCE_SMILES, PUBCHEM_ACTIVITY_OUTCOME,
               PUBCHEM_ACTIVITY_SCORE, ... assay-specific cols ...,
               Potency-Replicate_1, ...)
        Rows 1-5: metadata (RESULT_TYPE, RESULT_DESCR, RESULT_UNIT, ...)
        Rows 6+: data
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
    try:
        r = requests.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            print(f"   [aid {aid}] HTTP {r.status_code}; skip")
            return None
        # Read with 5 metadata rows after header
        df = pd.read_csv(io.StringIO(r.text), low_memory=False, skiprows=[1, 2, 3, 4, 5])
        return df
    except Exception as e:
        print(f"   [aid {aid}] fetch error: {e}")
        return None


def _extract_pec50_rows(df: pd.DataFrame, aid: int, label: str) -> pd.DataFrame:
    """Parse one PubChem assay df -> rows with (cid, smiles, pec50, outcome).

    Keep only dose-response actives that have a finite Potency-Replicate_1 (uM).
    pEC50 = 6 - log10(potency_uM).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["cid", "smiles", "pec50", "outcome", "src"])
    cols = {c.strip(): c for c in df.columns}
    cid_col = cols.get("PUBCHEM_CID")
    smi_col = cols.get("PUBCHEM_EXT_DATASOURCE_SMILES")
    out_col = cols.get("PUBCHEM_ACTIVITY_OUTCOME")
    # Find any column whose name starts with 'Potency' (microMolar)
    pot_cols = [c for c in df.columns if c.startswith("Potency")]
    if cid_col is None or smi_col is None or out_col is None or not pot_cols:
        print(f"   [aid {aid}] missing expected columns; cols={list(df.columns)[:8]}")
        return pd.DataFrame(columns=["cid", "smiles", "pec50", "outcome", "src"])

    pot_col = pot_cols[0]
    sub = df[[cid_col, smi_col, out_col, pot_col]].copy()
    sub.columns = ["cid", "smiles", "outcome", "potency_uM"]
    sub["outcome"] = sub["outcome"].astype(str).str.strip().str.lower()
    # Keep only rows with real potency (active dose-response).  Drop inactives
    # / inconclusives.
    sub["potency_uM"] = pd.to_numeric(sub["potency_uM"], errors="coerce")
    sub = sub[sub["potency_uM"].notna() & (sub["potency_uM"] > 0)].copy()
    sub = sub[sub["smiles"].notna() & (sub["smiles"].astype(str).str.len() > 0)].copy()
    # Reasonable potency bounds: 1 pM .. 100 mM
    sub = sub[(sub["potency_uM"] >= 1e-6) & (sub["potency_uM"] <= 1e5)].copy()
    # pEC50 = -log10([M]) = -log10(potency_uM * 1e-6) = 6 - log10(potency_uM)
    sub["pec50"] = 6.0 - np.log10(sub["potency_uM"].astype(float))
    sub["src"] = f"pubchem_AID{aid}_{label}"
    return sub[["cid", "smiles", "pec50", "outcome", "src"]]


def _load_pubchem_pool(force_refresh: bool = False) -> pd.DataFrame:
    """Union all configured AIDs into one (smi, pec50) pool, standardized.

    Columns returned: ['inchikey', 'std_smiles', 'pec50', 'n_meas', 'src'].
    Uses cached parquet if present and not force_refresh.
    """
    if POOL_CACHE.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(POOL_CACHE)
            print(f"   [pool] cache HIT -> {POOL_CACHE}  rows={len(cached)}")
            return cached
        except Exception as e:
            print(f"   [pool] cache read failed ({e}); refetching")

    frames = []
    aids_attempted = []
    aids_succeeded = []
    for aid, label, is_dr in PUBCHEM_AIDS:
        aids_attempted.append(aid)
        print(f"   [aid {aid}] fetching CSV ({label}) ...")
        t0 = time.time()
        df = _fetch_aid_csv(aid)
        if df is None:
            continue
        rows = _extract_pec50_rows(df, aid, label)
        if rows.empty:
            print(f"   [aid {aid}] no usable rows after filter")
            continue
        print(f"   [aid {aid}] kept {len(rows)} active rows  "
              f"({time.time() - t0:.1f}s)")
        frames.append(rows)
        aids_succeeded.append(aid)
        time.sleep(REQ_SLEEP)

    if not frames:
        raise RuntimeError(
            f"PubChem REST fetch returned 0 usable rows from AIDs={aids_attempted}"
        )

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [pool] pre-standardize union: {len(pool)} rows  "
          f"(AIDs={aids_succeeded})")

    # Standardize + InChIKey dedup
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    print(f"   [pool] after RDKit standardize: {len(pool)} rows")

    # Aggregate: median pEC50 per InChIKey
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(
            pec50=("pec50", "median"),
            std_smiles=("std_smiles", "first"),
            src=("src", "first"),
            n_meas=("pec50", "count"),
        )
    )
    print(f"   [pool] after InChIKey dedup (median): {len(agg)} unique cpds")
    print(f"   [pool] n_meas: min={agg['n_meas'].min()}  "
          f"median={agg['n_meas'].median():.1f}  max={agg['n_meas'].max()}")
    print(f"   [pool] pec50: mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}  "
          f"min={agg['pec50'].min():.3f}  max={agg['pec50'].max():.3f}")

    EXT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(POOL_CACHE, index=False)
    print(f"   [pool] cached -> {POOL_CACHE}")
    return agg


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
    w = np.clip(top_sim.copy(), 0.0, 1.0)
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


def _write_failure_summary(reason: str, partial: dict | None = None) -> dict:
    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "status": "failed",
        "failure_reason": reason,
        "pubchem_aids_attempted": [a for a, _, _ in PUBCHEM_AIDS],
        "data_source": "PubChem BioAssay REST",
    }
    if partial:
        summary.update(partial)
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}  (FAILURE summary)")
    return summary


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PubChem BioAssay PXR external kNN residual feature; "
          f"shallow residual-LGBM bag on nb1070 anchor")
    print(f"          AIDs = {[a for a, _, _ in PUBCHEM_AIDS]}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + pred_pubchem_pec50 + sim  (169)")
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
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- PubChem pool ----
    print("\n" + "-" * 78)
    print("PUBCHEM PXR POOL (PubChem REST API, live fetch + parquet cache)")
    print("-" * 78)
    try:
        pool = _load_pubchem_pool(force_refresh=False)
    except Exception as e:
        print(f"[FATAL] PubChem pool load failed: {e}")
        return _write_failure_summary(
            f"PubChem pool load failed: {e}",
            partial={"wall_sec": round(time.time() - t0, 2)},
        )

    if len(pool) < 10:
        return _write_failure_summary(
            f"PubChem pool too small after dedup: {len(pool)} rows",
            partial={"n_pool_pre_leakguard": int(len(pool)),
                     "wall_sec": round(time.time() - t0, 2)},
        )

    # ---- Test InChIKey leak guard ----
    print("\n" + "-" * 78)
    print("TEST-SET LEAK GUARD (drop any PubChem cpd whose InChIKey is in the 513)")
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
    print(f"   pool: {n_before} -> {n_after}  "
          f"(dropped {n_before - n_after} test-overlapping cpds)")

    # ---- Morgan fingerprints ----
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

    # ---- kNN k=5 Tanimoto ----
    print("\n" + "-" * 78)
    print(f"TANIMOTO kNN (k={KNN_K}) -- test (513) vs PubChem pool ({len(pool)})")
    print("-" * 78)
    top_idx, top_sim = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_pubchem_pec50, mean_sim = _knn_predict(
        top_idx, top_sim, pool_labels, fallback=pool_median
    )
    top1_sim = top_sim[:, 0]
    print(f"   pred_pubchem_pec50  mean={pred_pubchem_pec50.mean():.3f}  "
          f"std={pred_pubchem_pec50.std():.3f}  "
          f"min={pred_pubchem_pec50.min():.3f}  max={pred_pubchem_pec50.max():.3f}")
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

    # ---- MACCS-167 (unblind slice) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"   MACCS unb shape = {X_maccs_unb.shape}")

    # ---- Build residual feature matrix on 253 ----
    pred_pubchem_unb = pred_pubchem_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_pubchem_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}  "
          f"(MACCS-167 + pred_pubchem + sim)")

    # ---- Per-seed residual cross-fit ----
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
    print(f"   nb1183 ref             = {NB1183_REF:.4f}  (MACCS residual bag)")
    print(f"   nb1211 ref             = {NB1211_REF:.4f}  (SLSQP variants bag)")
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (ChEMBL kNN residual bag)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_REF - DECISION_MARGIN
    beats_nb1211 = rae_mean_bag < NB1211_REF - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "PUBCHEM_BIOASSAY_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif beats_nb1211:
        verdict = "PUBCHEM_BIOASSAY_BEATS_NB1211_BUT_NOT_NB1242"
    elif beats_nb1183:
        verdict = "PUBCHEM_BIOASSAY_BEATS_NB1183_BUT_NOT_NB1211"
    elif beats_nb1070:
        verdict = "PUBCHEM_BIOASSAY_HELPS_NB1070_BUT_NOT_NB1183"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "PUBCHEM_BIOASSAY_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "PUBCHEM_BIOASSAY_HURTS_NB1070"
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
        "status": "ok",
        "data_source": "PubChem BioAssay REST",
        "pubchem_aids_attempted": [a for a, _, _ in PUBCHEM_AIDS],
        "pool_cache_path": str(POOL_CACHE),
        "n_pubchem_pool_pre_leakguard": int(n_before),
        "n_pubchem_pool_post_leakguard": int(len(pool)),
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
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_REF,
        "delta_mean_bag_vs_nb1211": rae_mean_bag - NB1211_REF,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1211": bool(beats_nb1211),
        "beats_nb1242": bool(beats_nb1242),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1183_ref": NB1183_REF,
        "nb1211_ref": NB1211_REF,
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
        "status", "n_pubchem_pool_post_leakguard",
        "top1_sim_p10", "top1_sim_p50", "top1_sim_p90",
        "n_zero_neighbor_rows",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1183",
        "delta_mean_bag_vs_nb1211",
        "delta_mean_bag_vs_nb1242",
        "beats_nb1070", "beats_nb1183", "beats_nb1211", "beats_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
