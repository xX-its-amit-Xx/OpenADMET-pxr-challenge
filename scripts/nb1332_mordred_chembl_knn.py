"""nb1332 -- ChEMBL PXR external bioactivity kNN residual feature (Mordred edition).

Hypothesis (vs nb1242):
    nb1242 used Morgan-Tanimoto distance for the ChEMBL kNN step. That selects
    neighbors by SUBSTRUCTURE overlap. Mordred-Euclidean distance over normalized
    descriptor space selects neighbors by PHYSICOCHEMICAL similarity instead --
    different axis. If the two neighbor sets differ for the same test compound,
    the resulting pred_chembl_pec50 carries orthogonal context, and the residual
    learner may extract additional gain over the nb1242 0.5431 baseline.

Protocol:
    1. Reuse the same 945-cpd ChEMBL PXR pool as nb1242 (same _load_chembl_pool
       union + test-InChIKey leak guard).
    2. Compute Mordred (~1613 descriptors, ignore_3D=True) on the 945 pool + 513
       standardized test rows; coerce object/inf -> NaN.
    3. Drop columns that are NaN or constant on the pool, take top-100 by VARIANCE
       across the pool, z-score each feature (mean/std from POOL only -- the test
       set is downstream and must not leak into the standardization).
    4. For each of 513 test rows: Euclidean distance to all 945 ChEMBL rows in
       the (z-scored, top-100-variance) Mordred space. Top-5 NN ->
         pred_chembl_pec50_mordred = inverse-distance weighted mean of pool pec50
         mean_5_dist_mordred       = mean of 5 NN distances
       Test rows with all-zero weights fall back to pool median.
    5. Anchor = nb1070_pred_oof; residual = y_unb - nb1070_pred_oof.
    6. Features = MACCS-167 + pred_chembl_pec50_mordred + mean_5_dist_mordred
       (169 cols) on 253 unblind rows.
    7. 5-seed shallow LGBM Huber bag (same capacity as nb1242: depth=3, leaves=7,
       n_est=80, lr=0.05, min_child=20, lambda=1, huber alpha=1), 5-fold
       cross-fit per seed, mean-bag pooled RAE.
    8. Verdict at 0.003 margin vs nb1242 (0.5431). Pearson(nb1332, nb1242)
       reports orthogonality probe -- different neighbors?

Outputs:
    scripts/nb1332_mordred_chembl_knn.py            (this file)
    data/processed/nb1332_summary.json
    data/processed/nb1332_mean_bag_oof.npy          (253,) float32
    data/processed/nb1332_per_seed_corrected_oof.npy (5, 253) float32
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

# numpy 2.x compat patch for mordred 0.6
if not hasattr(np, "product"):
    np.product = np.prod  # type: ignore[attr-defined]

from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch  # morgan_fp_batch unused but keeps import parity
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1332"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
DIST_FLOOR = 1e-6           # avoid div-by-zero in inverse-distance weighting
TOPN_VAR_FEATURES = 100     # keep top-N variance features after standardization

NB1070_REF = 0.5771
NB1242_REF = 0.5431          # Morgan-Tanimoto kNN variant, mean-bag pooled RAE
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
    """Same union as nb1242: CHEMBL3401 raw + nr_extended PXR + pxr_all_types.

    Standardize -> InChIKey dedupe (median pEC50).  Returns columns
    ['inchikey', 'std_smiles', 'pec50', 'src', 'n_meas'].
    """
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
    print(f"   [pool] pec50: mean={agg['pec50'].mean():.3f}  "
          f"std={agg['pec50'].std():.3f}  "
          f"min={agg['pec50'].min():.3f}  max={agg['pec50'].max():.3f}")
    return agg


def _compute_mordred(smiles_list, n_proc=4):
    """Compute mordred descriptors (~1613, ignore_3D=True).  Returns
    (X (n, d) float64 with NaN, names list, n_desc int)."""
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    n_desc = len(calc.descriptors)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_mask = np.array([m is not None for m in mols])
    for i, m in enumerate(mols):
        if m is None:
            mols[i] = Chem.MolFromSmiles("C")  # placeholder; row NaN'd
    df = calc.pandas(mols, nproc=n_proc, quiet=True)
    names = list(df.columns)
    X = np.full(df.shape, np.nan, dtype=np.float64)
    for j, col in enumerate(names):
        col_vals = df[col].values
        arr = np.empty(len(col_vals), dtype=np.float64)
        for i, v in enumerate(col_vals):
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    fv = np.nan
                arr[i] = fv
            except Exception:
                arr[i] = np.nan
        X[:, j] = arr
    if not valid_mask.all():
        X[~valid_mask, :] = np.nan
    return X, names, n_desc


def _select_and_zscore(X_pool: np.ndarray, X_test: np.ndarray,
                        names: list[str]) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray]:
    """Drop NaN/constant pool cols, z-score by pool mean/std (no test leak),
    pick top TOPN_VAR_FEATURES by VARIANCE across pool (post-zscore variance is
    1 by construction -- so 'top variance' refers to the RAW variance, which is
    the natural source of cross-compound separation).  Returns
    (Xz_pool (n_pool, K), Xz_test (n_test, K), kept_col_idx, mu, sigma).
    """
    n_pool, d = X_pool.shape

    # Median-impute pool col-wise to handle stray NaNs; if entire pool col is
    # NaN, drop.
    col_nan_mask = np.isnan(X_pool).all(axis=0)
    keep0 = ~col_nan_mask
    X_pool_ = X_pool[:, keep0].copy()
    X_test_ = X_test[:, keep0].copy()
    cols_after_nan = np.where(keep0)[0]

    # column medians for impute
    medians = np.nanmedian(X_pool_, axis=0)
    inds = np.where(np.isnan(X_pool_))
    if len(inds[0]) > 0:
        X_pool_[inds] = np.take(medians, inds[1])
    # impute test with pool medians
    inds_t = np.where(np.isnan(X_test_))
    if len(inds_t[0]) > 0:
        X_test_[inds_t] = np.take(medians, inds_t[1])

    # Raw variance across pool -> top-N selection BEFORE z-score (variance
    # ranking would otherwise be destroyed by /std).
    raw_var = X_pool_.var(axis=0)
    # drop near-constant columns
    nonconst = raw_var > 1e-12
    X_pool_ = X_pool_[:, nonconst]
    X_test_ = X_test_[:, nonconst]
    cols_after_const = cols_after_nan[nonconst]
    raw_var = raw_var[nonconst]
    print(f"   [feat] after NaN+constant drop: {X_pool_.shape[1]} cols")

    # top-N by raw variance
    k = min(TOPN_VAR_FEATURES, X_pool_.shape[1])
    top = np.argsort(-raw_var)[:k]
    top.sort()  # keep deterministic column order
    X_pool_ = X_pool_[:, top]
    X_test_ = X_test_[:, top]
    kept_cols = cols_after_const[top].tolist()
    print(f"   [feat] top-{k} variance features kept")

    # z-score using POOL mean/std only
    mu = X_pool_.mean(axis=0)
    sigma = X_pool_.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    Xz_pool = (X_pool_ - mu) / sigma
    Xz_test = (X_test_ - mu) / sigma

    # Clip extreme outliers in test (caused by sigma being pool-only) to
    # bound Euclidean distance contributions
    Xz_test = np.clip(Xz_test, -10.0, 10.0)
    return (Xz_pool.astype(np.float32),
            Xz_test.astype(np.float32),
            kept_cols, mu.astype(np.float64), sigma.astype(np.float64))


def _euclidean_topk(Q: np.ndarray, P: np.ndarray, k: int):
    """Euclidean top-k. Returns (top_idx (n_q, k), top_dist (n_q, k))."""
    n_q = Q.shape[0]
    n_p = P.shape[0]
    top_idx = np.zeros((n_q, k), dtype=np.int32)
    top_dist = np.zeros((n_q, k), dtype=np.float32)

    p_sq = (P * P).sum(axis=1)            # (n_p,)
    q_sq = (Q * Q).sum(axis=1)            # (n_q,)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        cross = Q[s:e] @ P.T              # (b, n_p)
        d2 = q_sq[s:e, None] + p_sq[None, :] - 2.0 * cross
        d2 = np.maximum(d2, 0.0)
        d = np.sqrt(d2, dtype=np.float32)
        if k >= n_p:
            order = np.argsort(d, axis=1)[:, :k]
            row_idx = np.arange(e - s)[:, None]
            top_idx[s:e] = order
            top_dist[s:e] = d[row_idx, order]
        else:
            part = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(e - s)[:, None]
            d_part = d[row_idx, part]
            order = np.argsort(d_part, axis=1)
            top_idx[s:e] = part[row_idx, order]
            top_dist[s:e] = d_part[row_idx, order]
    return top_idx, top_dist


def _knn_predict_inv_dist(top_idx: np.ndarray, top_dist: np.ndarray,
                          pool_labels: np.ndarray, fallback: float):
    """Inverse-distance weighted mean of pool_labels at top_idx; pool labels
    are float32.  Weight = 1 / (dist + DIST_FLOOR).  Returns (pred, mean_dist)."""
    w = 1.0 / (top_dist + DIST_FLOOR)     # (n_q, k)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if not np.isfinite(w_sum[i]) or w_sum[i] <= 0:
            pred[i] = fallback
        else:
            pred[i] = np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i]
    mean_dist = top_dist.mean(axis=1).astype(np.float32)
    return pred, mean_dist


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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChEMBL PXR Mordred-Euclidean kNN residual feature;")
    print(f"          orthogonality probe vs nb1242 Morgan-Tanimoto kNN")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          features = MACCS-167 + pred_chembl_pec50_mordred + mean_5_dist_mordred")
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
    print(f"[load] {ANCHOR} pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- ChEMBL pool ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (same union as nb1242)")
    print("-" * 78)
    pool = _load_chembl_pool()

    # ---- Test InChIKey leak guard ----
    print("\n" + "-" * 78)
    print("TEST-SET LEAK GUARD")
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

    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    pool_smiles = pool["std_smiles"].tolist()

    std_test_smiles = []
    for m in test_mols:
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))

    # ---- Mordred on pool + test ----
    print("\n" + "-" * 78)
    print("MORDRED DESCRIPTORS (ignore_3D=True)")
    print("-" * 78)
    n_proc = min(4, max(1, (os.cpu_count() or 4) // 2))
    print(f"   computing pool ({len(pool_smiles)} cpds), nproc={n_proc} ...")
    X_pool_raw, names, n_desc_raw = _compute_mordred(pool_smiles, n_proc=n_proc)
    print(f"   pool raw shape: {X_pool_raw.shape}  (mordred n_desc={n_desc_raw})")
    print(f"   computing test ({len(std_test_smiles)} cpds), nproc={n_proc} ...")
    X_test_raw, _, _ = _compute_mordred(std_test_smiles, n_proc=n_proc)
    print(f"   test raw shape: {X_test_raw.shape}")

    # ---- Select + z-score ----
    print("\n" + "-" * 78)
    print("FEATURE SELECTION + Z-SCORE (top-{} variance, pool-only mean/std)"
          .format(TOPN_VAR_FEATURES))
    print("-" * 78)
    Xz_pool, Xz_test, kept_cols, mu, sigma = _select_and_zscore(
        X_pool_raw, X_test_raw, names
    )
    print(f"   z-scored pool shape: {Xz_pool.shape}")
    print(f"   z-scored test shape: {Xz_test.shape}")

    # ---- kNN k=5 Euclidean ----
    print("\n" + "-" * 78)
    print(f"EUCLIDEAN kNN (k={KNN_K}) -- test (513) vs pool ({Xz_pool.shape[0]})")
    print("-" * 78)
    top_idx, top_dist = _euclidean_topk(Xz_test, Xz_pool, k=KNN_K)
    pred_chembl_pec50, mean_dist = _knn_predict_inv_dist(
        top_idx, top_dist, pool_labels, fallback=pool_median
    )
    top1_dist = top_dist[:, 0]
    print(f"   pred_chembl_pec50_mordred  mean={pred_chembl_pec50.mean():.3f}  "
          f"std={pred_chembl_pec50.std():.3f}  "
          f"min={pred_chembl_pec50.min():.3f}  max={pred_chembl_pec50.max():.3f}")
    print(f"   top1 dist  p10={np.percentile(top1_dist, 10):.3f}  "
          f"p50={np.percentile(top1_dist, 50):.3f}  "
          f"p90={np.percentile(top1_dist, 90):.3f}  "
          f"max={top1_dist.max():.3f}")
    print(f"   mean5 dist  p10={np.percentile(mean_dist, 10):.3f}  "
          f"p50={np.percentile(mean_dist, 50):.3f}  "
          f"p90={np.percentile(mean_dist, 90):.3f}")

    # ---- MACCS-167 (unblind slice) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"   MACCS unb shape = {X_maccs_unb.shape}")

    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_dist_unb = mean_dist[unb_idx].astype(np.float32)
    X_unb = np.concatenate(
        [
            X_maccs_unb,
            pred_chembl_unb.reshape(-1, 1),
            mean_dist_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   residual feature matrix: {X_unb.shape}")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow LGBM Huber)")
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
    print(f"   nb1242 ref             = {NB1242_REF:.4f}  (Morgan-Tanimoto variant)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    # ---- Orthogonality probe vs nb1242 ----
    nb1242_path = DATA_PROCESSED / "nb1242_mean_bag_oof.npy"
    pearson_vs_nb1242 = None
    if nb1242_path.exists():
        nb1242_oof = np.load(nb1242_path).astype(np.float64)
        if nb1242_oof.shape[0] == n_unb:
            a = mean_bag_oof - mean_bag_oof.mean()
            b = nb1242_oof - nb1242_oof.mean()
            denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
            if denom > 0:
                pearson_vs_nb1242 = float((a * b).sum() / denom)
            print(f"   Pearson(nb1332 vs nb1242) = {pearson_vs_nb1242:.4f}")
        else:
            print(f"   nb1242 oof shape mismatch: {nb1242_oof.shape}")
    else:
        print(f"   nb1242 oof not found at {nb1242_path}")

    if beats_nb1242:
        verdict = "MORDRED_CHEMBL_KNN_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif beats_nb1070 and abs(rae_mean_bag - NB1242_REF) < DECISION_MARGIN:
        verdict = "MORDRED_CHEMBL_KNN_TIES_NB1242"
    elif beats_nb1070:
        verdict = "MORDRED_CHEMBL_KNN_HELPS_NB1070_BUT_WORSE_THAN_NB1242"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "MORDRED_CHEMBL_KNN_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "MORDRED_CHEMBL_KNN_HURTS_NB1070"
    print(f"   verdict                = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "distance_metric": "euclidean_zscored_mordred_top100_variance",
        "compared_to": "nb1242 (Morgan-Tanimoto kNN, same pool + anchor + LGBM capacity)",
        "n_chembl_pool": int(len(pool)),
        "test_inchikeys_in_pool_dropped": int(n_before - n_after),
        "pool_pec50_mean": float(pool_labels.mean()),
        "pool_pec50_std": float(pool_labels.std()),
        "pool_pec50_median": pool_median,
        "mordred_n_desc_raw": int(n_desc_raw),
        "n_mordred_features_kept": int(Xz_pool.shape[1]),
        "topn_var_features_target": TOPN_VAR_FEATURES,
        "knn_k": KNN_K,
        "top1_dist_p10": float(np.percentile(top1_dist, 10)),
        "top1_dist_p50": float(np.percentile(top1_dist, 50)),
        "top1_dist_p90": float(np.percentile(top1_dist, 90)),
        "top1_dist_max": float(top1_dist.max()),
        "mean5_dist_p10": float(np.percentile(mean_dist, 10)),
        "mean5_dist_p50": float(np.percentile(mean_dist, 50)),
        "mean5_dist_p90": float(np.percentile(mean_dist, 90)),
        "pred_chembl_pec50_mean": float(pred_chembl_pec50.mean()),
        "pred_chembl_pec50_std": float(pred_chembl_pec50.std()),
        "pred_chembl_pec50_min": float(pred_chembl_pec50.min()),
        "pred_chembl_pec50_max": float(pred_chembl_pec50.max()),
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
        "pearson_vs_nb1242": pearson_vs_nb1242,
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
        "n_chembl_pool", "n_mordred_features_kept",
        "top1_dist_p10", "top1_dist_p50", "top1_dist_p90",
        "mean5_dist_p50",
        "pred_chembl_pec50_mean", "pred_chembl_pec50_std",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1242",
        "beats_nb1070", "beats_nb1242",
        "pearson_vs_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
