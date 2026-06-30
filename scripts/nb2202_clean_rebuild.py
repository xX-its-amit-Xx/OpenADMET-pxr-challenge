"""nb2202 -- CLEAN rebuild: same-anchor train+deploy using te_chemprop_aux.

PROTOCOL:
    1. Load te_chemprop_aux.npy (513,) -- anchor on BOTH train (via
       te_chemprop_aux[unb_idx]) AND deploy (full 513).
    2. Verify te_chemprop_aux[unb_idx] is honest PRE-unblind:
       sha256(te_chemprop_aux[unb_idx]) != sha256(y_unb_253);
       RAE on 253 should equal 0.6216 (CHEMPROP_AUX_REF).
    3. Compute residual r = y_unb - te_chemprop_aux[unb_idx].
    4. Refresh SHAP via LGBM(MSE) full-fit on the 117-col 5-way K-tuned matrix
       vs this residual (fresh SHAP top-K ranking).
    5. For K in {15, 20, 28, 40}: 5-seed bag (seeds 0,1,7,42,137),
       5-fold cross-fit LGBM(MSE) L=15 lr=0.03 mc=5 lambda=2 n_est=300.
    6. Final = te_chemprop_aux[unb_idx] + cross-fit-residual; report mean-bag
       and median-bag RAE.
    7. This SHOULD reproduce nb2103's 0.4737/0.4698 (same anchor, possibly
       slightly different SHAP from refreshed ranking).
    8. If any K beats 0.4698 (median_bag), build clean deploy:
       residual LGBM refit on all 253 (5 outer x 5 inner = 25 fits) with
       K=best, predict on 513, row-MEDIAN, te_final = te_chemprop_aux +
       residual_513.
    9. Save submissions/nb2202_deploy_clean_chemprop.csv and te_nb2202.npy.

Outputs:
    scripts/nb2202_clean_rebuild.py
    data/processed/nb2202_summary.json
    data/processed/nb2202_shap_importance_refresh.npy   (117,) float32
    data/processed/nb2202_mean_bag_oof_K{K}.npy         (253,) float32 per K
    data/processed/nb2202_median_bag_oof_K{K}.npy       (253,) float32 per K
    [if deploy] data/processed/te_nb2202.npy            (513,) float32
    [if deploy] submissions/nb2202_deploy_clean_chemprop.csv
"""
from __future__ import annotations

import hashlib
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
import lightgbm as lgb
import shap
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2202"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
K_GRID = [15, 20, 28, 40]

# Deploy seeds: 5 outer x 5 inner = 25 fits (matches nb2112)
DEPLOY_OUTER_SEEDS = [0, 1, 7, 42, 137]
DEPLOY_INNER_OFFSETS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# Verification anchors
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.0001  # deploy if any K beats 0.4698


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


def _sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _load_chembl_pool() -> pd.DataFrame:
    """Same union as nb1852/nb1861/nb2063/nb2081/nb2091/nb2103."""
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
    """LGBM(MSE) -- identical hyperparams to nb2063/nb2081/nb2091/nb2103."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CLEAN rebuild: same-anchor train+deploy using "
          f"te_chemprop_aux")
    print(f"          anchor={ANCHOR}  K_grid={K_GRID}")
    print(f"          seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean={NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median={NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print("=" * 78)

    # ---- Step 1: Load anchor ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    print(f"[load] te_chemprop_aux.npy shape = {te_anchor_513.shape}")

    # ---- Load test set + truth ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "name" in te.columns:
        mol_names = te["name"].astype(str).tolist()
    else:
        raise KeyError("no name column on test set")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"te_chemprop_aux shape {te_anchor_513.shape} vs n_test={n_test}"
        )
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Step 2: Verify anchor is honest PRE-unblind ----
    anchor_unb = te_anchor_513[unb_idx]
    sha_anchor = _sha256_arr(anchor_unb.astype(np.float32))
    sha_truth = _sha256_arr(y_unb.astype(np.float32))
    rae_anchor = float(rae(y_unb, anchor_unb))
    print("\n" + "-" * 78)
    print("VERIFY: anchor is honest PRE-unblind")
    print("-" * 78)
    print(f"   sha256(te[unb_idx])  = {sha_anchor[:32]}...")
    print(f"   sha256(y_unb_253)    = {sha_truth[:32]}...")
    anchor_is_distinct = sha_anchor != sha_truth
    print(f"   distinct from truth: {anchor_is_distinct}")
    print(f"   in_RAE(te[unb_idx], y_unb) = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    rae_matches_ref = abs(rae_anchor - CHEMPROP_AUX_REF) < 0.005
    print(f"   matches ref (|d|<0.005): {rae_matches_ref}")
    if not anchor_is_distinct:
        raise ValueError(
            "CONTAMINATION: te[unb_idx] sha256 == y_unb_253 sha256"
        )
    if not rae_matches_ref:
        print(f"   [warn] RAE drifts from ref by "
              f"{abs(rae_anchor - CHEMPROP_AUX_REF):.4f}")

    # ---- Step 3: Compute residual ----
    residual = y_unb - anchor_unb
    print(f"\n[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")
    print(f"        min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ---- Load K-grid winners (same 117-col stack as nb2063/nb2103) ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f:
        sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f:
        sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f:
        sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f:
        sum_1541 = json.load(f)

    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                      best_K_key="best_K")
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(len(top_avalon_bit_idx))
    print(f"[reuse] top-{n_top_ap}     AtomPair bits (nb1524 K={K_AP_best})")
    print(f"[reuse] top-{n_top_maccs}     MACCS bits  (nb1352)")
    print(f"[reuse] top-{n_top_mord}     Mordred cols (nb1523 K={K_Mord_best})")
    print(f"[reuse] top-{n_top_embed}     ChempropEmbed dims (nb1541 K={K_Embed_best})")
    print(f"[reuse] top-{n_top_avalon}    Avalon bits (nb1392 SHAP K=30)")

    # ---- Feature matrices on FULL 513 + unb slice ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_ap_unb_top = X_ap_te_top[unb_idx]

    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_te_top[unb_idx]

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_mord_unb_top = X_mord_te_top[unb_idx]

    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_emb_unb_top = X_emb_te_top[unb_idx]

    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)
    X_av_unb_top = X_av_te_top[unb_idx]
    print(f"[feat] X_ap_te_top={X_ap_te_top.shape}  "
          f"X_maccs_te_top={X_maccs_te_top.shape}  "
          f"X_mord_te_top={X_mord_te_top.shape}")
    print(f"       X_emb_te_top={X_emb_te_top.shape}  "
          f"X_av_te_top={X_av_te_top.shape}")

    # ---- ChEMBL kNN ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
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

    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_te = pred_chembl_pec50.astype(np.float32)
    mean_sim_te = mean_sim.astype(np.float32)
    pred_chembl_unb = pred_chembl_te[unb_idx]
    mean_sim_unb = mean_sim_te[unb_idx]

    # ---- Stack full 513 + 253 unb 117-col matrices ----
    X_te_117 = np.concatenate(
        [
            X_ap_te_top, X_maccs_te_top, X_mord_te_top,
            X_emb_te_top, X_av_te_top,
            pred_chembl_te.reshape(-1, 1),
            mean_sim_te.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    X_unb_117 = np.concatenate(
        [
            X_ap_unb_top, X_maccs_unb_top, X_mord_unb_top,
            X_emb_unb_top, X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb_117.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   COMBINED 5-WAY K-TUNED matrix: X_te_117={X_te_117.shape}  "
          f"X_unb_117={X_unb_117.shape}")

    # ---- Feature names ----
    feat_names: list[str] = []
    feat_family: list[str] = []
    for b in top_ap_bit_idx:
        feat_names.append(f"AtomPair_bit_{int(b)}")
        feat_family.append("AtomPair")
    for b in top_maccs_bit_idx:
        feat_names.append(f"MACCS_bit_{int(b)}")
        feat_family.append("MACCS")
    for c in top_mord_col_idx:
        feat_names.append(f"Mordred_col_{int(c)}")
        feat_family.append("Mordred")
    for d in top_embed_col_idx:
        feat_names.append(f"ChempropEmbed_dim_{int(d)}")
        feat_family.append("ChempropEmbed")
    for b in top_avalon_bit_idx:
        feat_names.append(f"Avalon_bit_{int(b)}")
        feat_family.append("Avalon")
    feat_names.append("pred_chembl_pec50")
    feat_family.append("ChEMBL_kNN")
    feat_names.append("mean_sim")
    feat_family.append("ChEMBL_kNN")
    assert len(feat_names) == feat_dim

    # ---- Step 4: Refresh SHAP ----
    print("\n" + "-" * 78)
    print(f"STEP 4: refresh SHAP via LGBM(MSE) full-fit on 117-col residual")
    print("-" * 78)
    t_shap = time.time()
    mdl_full = lgb.LGBMRegressor(**_lgbm_params(seed=0))
    mdl_full.fit(X_unb_117, residual)
    explainer = shap.TreeExplainer(mdl_full)
    shap_vals = explainer.shap_values(X_unb_117)
    shap_imp = np.abs(shap_vals).mean(axis=0).astype(np.float32)
    if shap_imp.shape[0] != feat_dim:
        raise ValueError(
            f"SHAP importance shape {shap_imp.shape} != feat_dim {feat_dim}"
        )
    np.save(DATA_PROCESSED / f"{TAG}_shap_importance_refresh.npy", shap_imp)
    print(f"   refreshed SHAP done   wall = {time.time() - t_shap:.1f}s")
    full_rank_order = np.argsort(-shap_imp).astype(np.int32)
    print(f"   top-15 SHAP cols (head): {full_rank_order[:15].tolist()}")
    top10_fam = [feat_family[i] for i in full_rank_order[:10]]
    print(f"   top-10 families: {top10_fam}")

    # ---- Step 5: K-grid sweep, 5-seed bag x 5-fold cross-fit ----
    print("\n" + "-" * 78)
    print(f"STEP 5: K-grid sweep {K_GRID} -- 5-seed bag x 5-fold cross-fit")
    print("-" * 78)
    per_K_results: list[dict] = []
    for K in K_GRID:
        print(f"\n--- K={K} ---")
        topK_idx = full_rank_order[:K].astype(np.int32)
        topK_family = [feat_family[i] for i in topK_idx]
        fam_counts: dict[str, int] = {}
        for fam in topK_family:
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
        print(f"   top-{K} family breakdown: {fam_counts}")

        X_topK = X_unb_117[:, topK_idx].astype(np.float32)
        per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb),
                                       dtype=np.float64)
        per_seed_rae: list[float] = []
        per_seed_records = []
        for i, s in enumerate(RESID_SEEDS):
            ts = time.time()
            resid_oof_s = _residual_cross_fit_one_seed(X_topK, residual, s)
            pred_corr_s = anchor_unb + resid_oof_s
            per_seed_corrected[i] = pred_corr_s
            rae_s = float(rae(y_unb, pred_corr_s))
            per_seed_rae.append(rae_s)
            delta_s = rae_s - rae_anchor
            per_seed_records.append({
                "seed": int(s),
                "rae_corrected": rae_s,
                "delta_vs_anchor": delta_s,
                "resid_oof_std": float(resid_oof_s.std()),
                "resid_oof_mean": float(resid_oof_s.mean()),
                "wall_sec": round(time.time() - ts, 2),
            })
            print(f"   K={K} seed={s:3d}:  rae_corr = {rae_s:.4f}  "
                  f"(d_vs_anchor = {delta_s:+.4f})  "
                  f"wall = {time.time() - ts:.1f}s")

        mean_bag_oof = per_seed_corrected.mean(axis=0)
        median_bag_oof = np.median(per_seed_corrected, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        rae_median_bag = float(rae(y_unb, median_bag_oof))

        per_seed_rae_arr = np.array(per_seed_rae)
        rae_per_seed_mean = float(per_seed_rae_arr.mean())
        rae_per_seed_std = float(per_seed_rae_arr.std())

        beats_2103_mean = rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
        beats_2103_median = rae_median_bag < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN

        print(f"   K={K} per-seed RAE   = "
              f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
        print(f"   K={K} per-seed mean  = {rae_per_seed_mean:.4f}  "
              f"std = {rae_per_seed_std:.4f}")
        print(f"   K={K} pooled mean    = {rae_mean_bag:.4f}  "
              f"(vs nb2103 K=28 mean {NB2103_K28_MEAN_BAG_REF:.4f}: "
              f"{rae_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f})")
        print(f"   K={K} pooled median  = {rae_median_bag:.4f}  "
              f"(vs nb2103 K=28 median {NB2103_K28_MEDIAN_BAG_REF:.4f}: "
              f"{rae_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")
        if beats_2103_median:
            print(f"   K={K} BEATS nb2103 K=28 median_bag")

        # Save per-K OOF
        np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K}.npy",
                mean_bag_oof.astype(np.float32))
        np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof_K{K}.npy",
                median_bag_oof.astype(np.float32))

        per_K_results.append({
            "K": int(K),
            "family_counts": fam_counts,
            "top_K_idx_in_117": topK_idx.tolist(),
            "per_seed_rae": per_seed_rae,
            "per_seed_records": per_seed_records,
            "rae_per_seed_mean": rae_per_seed_mean,
            "rae_per_seed_std": rae_per_seed_std,
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_mean_bag_vs_nb2103_K28": (
                rae_mean_bag - NB2103_K28_MEAN_BAG_REF
            ),
            "delta_median_bag_vs_nb2103_K28": (
                rae_median_bag - NB2103_K28_MEDIAN_BAG_REF
            ),
            "beats_nb2103_K28_mean": bool(beats_2103_mean),
            "beats_nb2103_K28_median": bool(beats_2103_median),
        })

    # ---- Summary table ----
    print("\n" + "=" * 78)
    print("K-GRID SUMMARY TABLE")
    print("=" * 78)
    print(f"   {'K':>4s}  {'mean_bag':>10s}  {'median_bag':>10s}  "
          f"{'per_seed_mean':>13s}  {'per_seed_std':>12s}  "
          f"{'d_mean_vs2103':>13s}  {'d_med_vs2103':>13s}")
    for r in per_K_results:
        print(f"   {r['K']:>4d}  {r['rae_mean_bag']:>10.4f}  "
              f"{r['rae_median_bag']:>10.4f}  {r['rae_per_seed_mean']:>13.4f}  "
              f"{r['rae_per_seed_std']:>12.4f}  "
              f"{r['delta_mean_bag_vs_nb2103_K28']:>+13.4f}  "
              f"{r['delta_median_bag_vs_nb2103_K28']:>+13.4f}")

    # ---- Best K by MEDIAN bag (matches nb2103 best metric) ----
    median_bag_rae = [r["rae_median_bag"] for r in per_K_results]
    best_i = int(np.argmin(median_bag_rae))
    best_K = int(per_K_results[best_i]["K"])
    best_median_bag = float(per_K_results[best_i]["rae_median_bag"])
    best_mean_bag = float(per_K_results[best_i]["rae_mean_bag"])
    print(f"\n   best K (by median_bag) = {best_K}  "
          f"median_bag={best_median_bag:.4f}  mean_bag={best_mean_bag:.4f}")

    # Should reproduce nb2103 K=28 0.4737/0.4698 (same anchor, refreshed SHAP)
    nb2103_match_note = ""
    for r in per_K_results:
        if r["K"] == 28:
            nb2103_match_note = (
                f"K=28 mean_bag={r['rae_mean_bag']:.4f} "
                f"vs nb2103 ref {NB2103_K28_MEAN_BAG_REF:.4f} "
                f"(d={r['rae_mean_bag'] - NB2103_K28_MEAN_BAG_REF:+.4f}); "
                f"median_bag={r['rae_median_bag']:.4f} "
                f"vs nb2103 ref {NB2103_K28_MEDIAN_BAG_REF:.4f} "
                f"(d={r['rae_median_bag'] - NB2103_K28_MEDIAN_BAG_REF:+.4f})"
            )
            break
    print(f"   reproduce-check vs nb2103: {nb2103_match_note}")

    # ---- Step 8: Deploy if best beats 0.4698 ----
    deploy_done = False
    deploy_summary = {}
    if best_median_bag < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN:
        print("\n" + "=" * 78)
        print(f"STEP 8: DEPLOY -- best K={best_K} median_bag={best_median_bag:.4f} "
              f"beats nb2103 K=28 ({NB2103_K28_MEDIAN_BAG_REF:.4f})")
        print("=" * 78)
        topK_best_idx = full_rank_order[:best_K].astype(np.int32)
        X_te_best = X_te_117[:, topK_best_idx].astype(np.float32)
        X_unb_best = X_unb_117[:, topK_best_idx].astype(np.float32)
        print(f"   feat_dim deploy = {best_K}  "
              f"X_te={X_te_best.shape}  X_unb={X_unb_best.shape}")

        n_total = len(DEPLOY_OUTER_SEEDS) * len(DEPLOY_INNER_OFFSETS)
        print(f"   deploy: {n_total} LGBM fits on all {n_unb} unblind, "
              f"predict {n_test}")
        all_resid_513 = np.zeros((n_total, n_test), dtype=np.float64)
        k_global = 0
        for o in DEPLOY_OUTER_SEEDS:
            t_o = time.time()
            inner_seeds = [o * 1000 + s for s in DEPLOY_INNER_OFFSETS]
            for s in inner_seeds:
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_unb_best, residual)
                resid_513 = mdl.predict(X_te_best)
                all_resid_513[k_global] = resid_513
                k_global += 1
            print(f"   outer {o:3d}:  wall={time.time() - t_o:.1f}s")

        median_resid_513 = np.median(all_resid_513, axis=0)
        te_final = te_anchor_513 + median_resid_513
        in_pred_unb = te_final[unb_idx]
        rae_in_unb = float(rae(y_unb, in_pred_unb))
        print(f"\n   in-sample RAE on unb_idx = {rae_in_unb:.4f}  "
              f"(deploy is fit on all 253, so in_RAE will be optimistic)")
        print(f"   te_final stats: mean={te_final.mean():.4f}  "
              f"std={te_final.std():.4f}  "
              f"min={te_final.min():.4f}  max={te_final.max():.4f}")
        print(f"   median_resid_513 stats: mean={median_resid_513.mean():+.4f}  "
              f"std={median_resid_513.std():.4f}")

        # Save te artifact
        te_path = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_path, te_final.astype(np.float32))
        print(f"   [save] {te_path}")

        # Save submission CSV
        df_sub = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_final.astype(np.float32),
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)} != 513")
        sub_path = SUBMISSIONS_DIR / f"{TAG}_deploy_clean_chemprop.csv"
        df_sub.to_csv(sub_path, index=False)
        print(f"   [save] {sub_path}  ({len(df_sub)} rows)")

        deploy_done = True
        deploy_summary = {
            "deploy_K": int(best_K),
            "deploy_n_total_fits": int(n_total),
            "deploy_outer_seeds": DEPLOY_OUTER_SEEDS,
            "deploy_inner_offsets": DEPLOY_INNER_OFFSETS,
            "in_RAE_unb_idx": rae_in_unb,
            "te_final_mean": float(te_final.mean()),
            "te_final_std": float(te_final.std()),
            "te_final_min": float(te_final.min()),
            "te_final_max": float(te_final.max()),
            "median_resid_513_mean": float(median_resid_513.mean()),
            "median_resid_513_std": float(median_resid_513.std()),
            "te_artifact": str(te_path),
            "submission_csv": str(sub_path),
        }
    else:
        print("\n" + "=" * 78)
        print(f"NO DEPLOY: best K={best_K} median_bag={best_median_bag:.4f} "
              f"does NOT beat nb2103 K=28 ({NB2103_K28_MEDIAN_BAG_REF:.4f})")
        print("=" * 78)

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("clean_rebuild_same_anchor_train_deploy_te_chemprop_aux"
                   "_refreshed_SHAP_K_grid_15_20_28_40"),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "K_grid": K_GRID,
        "feat_dim_full": int(feat_dim),
        "feat_breakdown_full": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "n_chembl_pool": int(len(pool)),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "sha256_anchor_unb": sha_anchor,
        "sha256_y_unb": sha_truth,
        "anchor_is_distinct_from_truth": bool(anchor_is_distinct),
        "rae_anchor_unb": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "rae_matches_chemprop_aux_ref": bool(rae_matches_ref),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "shap_top15_idx_in_117": full_rank_order[:15].tolist(),
        "shap_top10_families": [feat_family[i] for i in full_rank_order[:10]],
        "per_K_records": per_K_results,
        "best_K_by_median_bag": best_K,
        "best_K_mean_bag": best_mean_bag,
        "best_K_median_bag": best_median_bag,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "deploy_triggered": bool(deploy_done),
        "deploy": deploy_summary,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K_grid", "feat_dim_full", "n_chembl_pool",
        "rae_anchor_unb", "chemprop_aux_ref",
        "rae_matches_chemprop_aux_ref",
        "anchor_is_distinct_from_truth",
        "best_K_by_median_bag", "best_K_mean_bag", "best_K_median_bag",
        "nb2103_K28_mean_bag_ref", "nb2103_K28_median_bag_ref",
        "deploy_triggered",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-K TABLE ====")
    for r in res["per_K_records"]:
        print(f"  K={r['K']:>3d}  mean_bag={r['rae_mean_bag']:.4f}  "
              f"median_bag={r['rae_median_bag']:.4f}  "
              f"d_mean_vs2103={r['delta_mean_bag_vs_nb2103_K28']:+.4f}  "
              f"d_med_vs2103={r['delta_median_bag_vs_nb2103_K28']:+.4f}")
    if res.get("deploy_triggered"):
        print("\n==== DEPLOY ====")
        for k in ("deploy_K", "in_RAE_unb_idx", "te_final_mean",
                  "te_final_std", "te_artifact", "submission_csv"):
            print(f"  {k}: {res['deploy'].get(k)}")
