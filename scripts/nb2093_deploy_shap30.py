"""nb2093 -- Deploy SHAP top-30 (the new floor 0.4788) to 513-row CSV.

PROTOCOL:
    1. Reuse top-30 SHAP indices by re-ranking the cached nb2063 SHAP
       importance vector (data/processed/nb2063_shap_importance_full117.npy)
       and taking the head 30 indices of argsort(-shap_imp).  This matches
       the K=30 winner identified by nb2081 (mean-bag honest cross-fit RAE
       0.4788, beating the K=50 reference 0.4933 by -0.0145).
    2. Build the same 117-col 5-way K-tuned feature stack as nb2063 / nb2072
       (AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN).
    3. 5-seed bag (seeds 0, 1, 7, 42, 137) of LGBM(MSE) fit on ALL 253
       unblind compounds with residual = y_unb - chemprop_aux te[unb_idx];
       predict residual on full 513 -> residual_513_k.
    4. Mean across 5 seeds.  te_nb2093 = te_chemprop_aux + mean_residual_513.
    5. Save submissions/nb2093_deploy_shap30.csv.

ANCHORS:
    nb2081 K=30 honest mean-bag cross-fit RAE = 0.4788 (the new floor;
    predicted LB band 0.4777-0.4788 given the PRE-unblind anchor and
    same residual-on-anchor recipe as nb2063/nb2072).
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
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2093"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# 5-seed bag (matches nb2063 / nb2072 / nb2081 inner seeds)
BAG_SEEDS = [0, 1, 7, 42, 137]
TOP_K_SHAP = 30

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
NB2063_SUMMARY = DATA_PROCESSED / "nb2063_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB2081_SUMMARY = DATA_PROCESSED / "nb2081_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# Reference (honest cross-fit) -- the new floor 0.4788 from nb2081 K=30
NB2081_K30_HONEST_REF = 0.4788
PREDICTED_LB_LO = 0.4777
PREDICTED_LB_HI = 0.4788


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
    """Same union as nb1852/nb1861/nb2063/nb2072."""
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
    """LGBM(MSE) -- identical to nb2063/nb2072/nb2081."""
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
    print(f"{TAG} -- DEPLOY SHAP top-{TOP_K_SHAP} LGBM-MSE to 513-row CSV")
    print(f"          bag seeds = {BAG_SEEDS}  total fits = {len(BAG_SEEDS)}")
    print(f"          anchor    = {ANCHOR}  (PRE-unblind te slice)")
    print(f"          ref       = nb2081 K=30 honest cross-fit RAE = "
          f"{NB2081_K30_HONEST_REF:.4f}  predicted LB "
          f"{PREDICTED_LB_LO:.4f}-{PREDICTED_LB_HI:.4f}")
    print("=" * 78)

    # ---- Load anchor + truth ----
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
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual_unb = y_unb - anchor_unb
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Load cached nb2063 SHAP importance over 117 cols; rank top-30 ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(
            f"missing nb2063 shap importance: {NB2063_SHAP_IMP}"
        )
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    top30_idx = full_rank_order[:TOP_K_SHAP].astype(np.int32)
    print(f"[shap] full SHAP importance shape = {shap_imp_full117.shape}")
    print(f"[shap] top-{TOP_K_SHAP} indices (head 10) = "
          f"{top30_idx[:10].tolist()}")

    # ---- Sanity: pull K=30 ranking that nb2081 itself used (if cached) ----
    if NB2081_SUMMARY.exists():
        with open(NB2081_SUMMARY) as f:
            sum_2081 = json.load(f)
        for r in sum_2081["per_K_records"]:
            if int(r["K"]) == TOP_K_SHAP:
                nb2081_top30 = np.array(r["top_K_idx_in_117"], dtype=int)
                same = np.array_equal(nb2081_top30, top30_idx)
                print(f"[check] nb2081.K=30 top_K_idx match = {same}  "
                      f"(nb2081 head 10 = {nb2081_top30[:10].tolist()})")
                if not same:
                    print("   [warn] re-rank differs from nb2081 cached idx; "
                          "using nb2081 cached for exact reproducibility")
                    top30_idx = nb2081_top30.astype(np.int32)
                break

    # ---- Load all K-grid winners (same 117-col stack as nb2063 / nb2072) ----
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
    assert K_Mord_best == int(sum_1523["best_K"])

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

    # ---- Feature matrices ON FULL 513 + unb slice ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_ap_unb_top = X_ap_te_top[unb_idx]
    print(f"[feat] X_ap_te_top       = {X_ap_te_top.shape}")

    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_te_top[unb_idx]
    print(f"[feat] X_maccs_te_top    = {X_maccs_te_top.shape}")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_mord_unb_top = X_mord_te_top[unb_idx]
    print(f"[feat] X_mord_te_top     = {X_mord_te_top.shape}")

    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_emb_unb_top = X_emb_te_top[unb_idx]
    print(f"[feat] X_emb_te_top      = {X_emb_te_top.shape}")

    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)
    X_av_unb_top = X_av_te_top[unb_idx]
    print(f"[feat] X_av_te_top       = {X_av_te_top.shape}")

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
    pred_chembl_te = pred_chembl_pec50.astype(np.float32)  # 513
    mean_sim_te = mean_sim.astype(np.float32)              # 513
    pred_chembl_unb = pred_chembl_te[unb_idx]
    mean_sim_unb = mean_sim_te[unb_idx]

    # ---- Stack full 513 117-col feature matrix ----
    X_te_117 = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_te.reshape(-1, 1),
            mean_sim_te.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)

    # ---- Stack 253 unb 117-col feature matrix ----
    X_unb_117 = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)

    feat_dim_full = X_te_117.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim_full != expected_dim:
        raise ValueError(
            f"feat_dim_full {feat_dim_full} != expected {expected_dim}"
        )
    if X_unb_117.shape[1] != feat_dim_full:
        raise ValueError(
            f"X_unb_117 feat dim {X_unb_117.shape[1]} != "
            f"X_te_117 feat dim {feat_dim_full}"
        )
    if feat_dim_full != shap_imp_full117.shape[0]:
        raise ValueError(
            f"feat_dim_full {feat_dim_full} != SHAP importance length "
            f"{shap_imp_full117.shape[0]}"
        )
    print(f"\n   COMBINED 5-WAY K-TUNED matrix: X_te_117={X_te_117.shape}  "
          f"X_unb_117={X_unb_117.shape}")

    # ---- Restrict to top-30 SHAP cols ----
    if int(top30_idx.max()) >= feat_dim_full:
        raise ValueError(
            f"top30_idx max {int(top30_idx.max())} >= feat_dim_full "
            f"{feat_dim_full}"
        )
    X_te_30 = X_te_117[:, top30_idx].astype(np.float32)
    X_unb_30 = X_unb_117[:, top30_idx].astype(np.float32)
    print(f"\n   SHAP TOP-{TOP_K_SHAP}: X_te_30={X_te_30.shape}  "
          f"X_unb_30={X_unb_30.shape}")

    # ---- DEPLOY: 5-seed bag on full 253 unblind, predict 513 ----
    print("\n" + "-" * 78)
    print(f"DEPLOY: {len(BAG_SEEDS)} LGBM(MSE) fits on all {n_unb} unblind, "
          f"predict {n_test}")
    print("-" * 78)
    all_resid_513 = np.zeros((len(BAG_SEEDS), n_test), dtype=np.float64)
    fit_records = []
    for k, s in enumerate(BAG_SEEDS):
        t_in = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb_30, residual_unb)
        resid_513 = mdl.predict(X_te_30)
        all_resid_513[k] = resid_513
        fit_records.append({
            "k": int(k),
            "seed": int(s),
            "resid_513_mean": float(resid_513.mean()),
            "resid_513_std": float(resid_513.std()),
            "wall_sec": round(time.time() - t_in, 2),
        })
        print(f"   fit {k+1:1d}/{len(BAG_SEEDS)}  seed={s:4d}  "
              f"resid_mean={resid_513.mean():+.4f}  "
              f"resid_std={resid_513.std():.4f}  "
              f"wall={time.time() - t_in:.1f}s")

    mean_resid_513 = all_resid_513.mean(axis=0)
    te_nb2093 = te_anchor_513 + mean_resid_513

    # ---- In-sample check on unb_idx ----
    in_pred_unb = te_nb2093[unb_idx]
    rae_in_unb = float(rae(y_unb, in_pred_unb))
    print("\n" + "-" * 78)
    print(f"in-sample RAE on unb_idx = {rae_in_unb:.4f}")
    print(f"anchor in_RAE             = {rae_anchor:.4f}")
    print(f"nb2081 K=30 honest cross-fit = {NB2081_K30_HONEST_REF:.4f}  "
          f"(predicted LB {PREDICTED_LB_LO:.4f}-{PREDICTED_LB_HI:.4f})")
    print("-" * 78)

    # ---- Stats ----
    te_mean = float(te_nb2093.mean())
    te_std = float(te_nb2093.std())
    te_min = float(te_nb2093.min())
    te_max = float(te_nb2093.max())
    print(f"\nte_nb2093 stats: mean={te_mean:.4f}  std={te_std:.4f}  "
          f"min={te_min:.4f}  max={te_max:.4f}")

    # ---- Save submission CSV ----
    df_sub = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb2093.astype(np.float32),
    })
    if len(df_sub) != 513:
        raise ValueError(f"submission rows {len(df_sub)} != 513")
    sub_path = SUBMISSIONS_DIR / f"{TAG}_deploy_shap30.csv"
    df_sub.to_csv(sub_path, index=False)
    print(f"\n[save] submission CSV: {sub_path}  ({len(df_sub)} rows)")

    # ---- Save te artifact ----
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb2093.astype(np.float32))
    print(f"[save] te artifact:  {te_path}")
    np.save(DATA_PROCESSED / f"{TAG}_all_resid_513.npy",
            all_resid_513.astype(np.float32))
    print(f"[save] resid stack:  {DATA_PROCESSED / f'{TAG}_all_resid_513.npy'}")

    summary = {
        "tag": TAG,
        "method": "deploy_shap_top30_5seed_bag_lgbm_mse",
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "bag_seeds": BAG_SEEDS,
        "n_fits": int(len(BAG_SEEDS)),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "top_k_shap": TOP_K_SHAP,
        "top30_idx_in_117": top30_idx.tolist(),
        "feat_dim_full": int(feat_dim_full),
        "feat_dim_top30": int(X_te_30.shape[1]),
        "feat_breakdown_full": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim_full),
        },
        "rae_anchor_chemprop_aux_unb": rae_anchor,
        "residual_mean_unb": float(residual_unb.mean()),
        "residual_std_unb": float(residual_unb.std()),
        "te_nb2093_mean": te_mean,
        "te_nb2093_std": te_std,
        "te_nb2093_min": te_min,
        "te_nb2093_max": te_max,
        "in_RAE_unb_idx": rae_in_unb,
        "nb2081_K30_honest_ref": NB2081_K30_HONEST_REF,
        "predicted_LB_lo": PREDICTED_LB_LO,
        "predicted_LB_hi": PREDICTED_LB_HI,
        "submission_csv": str(sub_path),
        "te_artifact": str(te_path),
        "fit_records": fit_records,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] summary:      {out_path}")
    print(f"\n[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_test", "n_unb", "feat_dim_full", "feat_dim_top30", "n_fits",
        "rae_anchor_chemprop_aux_unb",
        "te_nb2093_mean", "te_nb2093_std",
        "te_nb2093_min", "te_nb2093_max",
        "in_RAE_unb_idx",
        "nb2081_K30_honest_ref", "predicted_LB_lo", "predicted_LB_hi",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
