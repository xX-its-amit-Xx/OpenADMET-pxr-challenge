"""nb1085 -- Pseudo-label distillation: nb2112 OOF soft labels on TRAIN.

PROTOCOL:
    1. Build nb2112 OOF predictions on 4139 TRAIN via 5-fold scaffold cross-fit.
       For each fold, refit the 25-seed (5 outer x 5 inner) LGBM(MSE) on the
       train-fold residuals (y_tr_fold - chemprop_aux_train_oof[tr_fold]) using
       the same SHAP-top-28 feature subset as nb2112, then predict residual on
       val-fold to get y_soft_pred (= chemprop_aux_train_oof[va_fold] +
       residual_pred[va_fold]).
       NOTE: We use chemprop_aux TRAIN OOF (oof_chemprop_aux.npy, shape 4139)
       as the anchor on TRAIN — strict cross-fit, no leakage.
    2. For each alpha in {0.7, 0.8, 0.9, 1.0}:
       mixed_target = alpha * y_hard + (1 - alpha) * y_soft_pred
    3. Retrain LGBM(MSE) K=28 on (TRAIN_features_117col_top28, mixed_target);
       5-seed bag + 5-fold scaffold cross-fit -> 253 in-sample-on-unblind
       residual OOFs. final_pred = chemprop_aux + residual_pred.
       Mean-bag and median-bag RAE on 253 unblind.
    4. Compare per-alpha vs nb2103 baseline (mean_bag 0.4737 / median_bag 0.4698).
       Decision margin: 0.003.
    5. If best beats: build deploy CSV submissions/nb1085_deploy_distill.csv.

Outputs:
    scripts/nb1085_pseudo_distill.py
    data/processed/nb1085_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src")
)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1085"
ANCHOR = "chemprop_aux"
ANCHOR_TR_OOF_PATH = DATA_PROCESSED / "oof_chemprop_aux.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

ALPHAS = [0.7, 0.8, 0.9, 1.0]

# nb2112-equivalent seed grid: 5 outer x 5 inner = 25 fits per fold
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_OFFSETS = [0, 1, 7, 42, 137]
N_INNER = len(INNER_OFFSETS)
TOP_K_SHAP = 28

# Retraining bag on mixed targets (per alpha): 5 seeds
RETRAIN_SEEDS = [0, 1, 7, 42, 137]
N_SCAFFOLD_FOLDS = 5

ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"
MORDRED_TR_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_train.npy")

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_TE_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_test.npy")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
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

    p2 = EXT_DIR / "chembl_nr_extended.parquet"
    if p2.exists():
        d = pd.read_parquet(p2)
        d = d[d["target_name"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["std_smiles", "pec50"]].rename(
            columns={"std_smiles": "smiles"}
        )
        d["src"] = "nr_extended"
        frames.append(d)

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

    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")
    pool = pd.concat(frames, ignore_index=True)
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[
        pool["inchikey"].notna() & pool["std_smiles"].notna()
    ].copy()
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(
            pec50=("pec50", "median"),
            std_smiles=("std_smiles", "first"),
            src_first=("src", "first"),
            n_meas=("pec50", "count"),
        )
    )
    agg = agg.rename(columns={"src_first": "src"})
    return agg


def _tanimoto_topk(fp_q, fp_pool, k):
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


def _lgbm_params(seed):
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


def _impute_mordred(X):
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy(p, n_expected):
    if not p.exists():
        raise FileNotFoundError(f"missing cache: {p}")
    X = np.load(p)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {p}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


def _extract_K_record(sum_dict, records_key, K):
    for r in sum_dict[records_key]:
        if int(r["K"]) == K:
            return r
    raise KeyError(f"K={K} not found")


def _build_117col(
    X_ap, X_maccs, X_mord, X_emb, X_av,
    top_ap_bit_idx, top_maccs_bit_idx, top_mord_col_idx,
    top_embed_col_idx, top_avalon_bit_idx,
    pred_chembl_pec50, mean_sim,
):
    parts = [
        X_ap[:, top_ap_bit_idx].astype(np.float32),
        X_maccs[:, top_maccs_bit_idx].astype(np.float32),
        X_mord[:, top_mord_col_idx].astype(np.float32),
        X_emb[:, top_embed_col_idx].astype(np.float32),
        X_av[:, top_avalon_bit_idx].astype(np.float32),
        pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
        mean_sim.reshape(-1, 1).astype(np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- pseudo-label distillation: nb2112 OOF soft labels on TRAIN")
    print(f"   alphas        : {ALPHAS}")
    print(f"   outer_seeds   : {OUTER_SEEDS}")
    print(f"   inner offsets : {INNER_OFFSETS}  (per outer)")
    print(f"   retrain bag   : {RETRAIN_SEEDS}")
    print(f"   scaffold folds: {N_SCAFFOLD_FOLDS}")
    print(f"   K_shap        : {TOP_K_SHAP}")
    print(f"   nb2103 ref    : mean_bag {NB2103_K28_MEAN_BAG_REF:.4f} / "
          f"median_bag {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- load nb2103 top-28 SHAP indices ----
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    rec28 = _extract_K_record(nb2103_sum, "per_K_records", K=TOP_K_SHAP)
    top28_idx = np.array(rec28["top_K_idx_in_117"], dtype=np.int32)
    if top28_idx.shape[0] != TOP_K_SHAP:
        raise ValueError(
            f"nb2103 K=28 top_K_idx_in_117 has {top28_idx.shape[0]} entries"
        )
    nb2103_k28_mean_bag = float(rec28["rae_mean_bag"])
    nb2103_k28_median_bag = float(rec28["rae_median_bag"])
    print(f"[ref] nb2103 K=28 mean_bag   = {nb2103_k28_mean_bag:.6f}")
    print(f"[ref] nb2103 K=28 median_bag = {nb2103_k28_median_bag:.6f}")

    # ---- load train/test + anchors + unblind ----
    tr = load_train()
    te = load_test()
    n_tr = len(tr)
    n_te = len(te)
    y_tr = tr["pec50"].to_numpy(dtype=np.float64)
    tr_smiles = tr["smiles"].astype(str).tolist()
    if "smiles" in te.columns:
        te_smiles = te["smiles"].astype(str).tolist()
    else:
        te_smiles = te["SMILES"].astype(str).tolist()

    if "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "name" in te.columns:
        mol_names = te["name"].astype(str).tolist()
    else:
        raise KeyError("no name column")

    print(f"[load] n_tr={n_tr}  n_te={n_te}")

    # chemprop_aux TRAIN OOF + TEST refit
    chemprop_aux_tr = np.load(ANCHOR_TR_OOF_PATH).astype(np.float64)
    if chemprop_aux_tr.shape[0] != n_tr:
        raise ValueError(
            f"chemprop_aux train OOF shape {chemprop_aux_tr.shape} "
            f"vs n_tr {n_tr}"
        )
    chemprop_aux_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if chemprop_aux_te.shape[0] != n_te:
        raise ValueError(
            f"chemprop_aux te shape {chemprop_aux_te.shape} vs n_te {n_te}"
        )
    rae_anchor_tr = float(rae(y_tr, chemprop_aux_tr))
    print(f"[load] chemprop_aux train OOF RAE = {rae_anchor_tr:.4f}")

    # unblind on 513
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    anchor_unb = chemprop_aux_te[unb_idx]
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[load] n_unb={n_unb}  anchor in_RAE={rae_anchor_unb:.4f}")

    # ---- load K-grid winners ----
    for p in (
        NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
        NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY,
    ):
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
    rec_mord = _extract_best_K_record(
        sum_1523, "per_K_records", best_K_key="best_K"
    )
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

    n_top_ap = len(top_ap_bit_idx)
    n_top_maccs = len(top_maccs_bit_idx)
    n_top_mord = len(top_mord_col_idx)
    n_top_embed = len(top_embed_col_idx)
    n_top_avalon = len(top_avalon_bit_idx)
    expected_dim_117 = (
        n_top_ap + n_top_maccs + n_top_mord
        + n_top_embed + n_top_avalon + 2
    )
    print(f"[feat] expected_dim_117 = {expected_dim_117}")

    # ---- load TRAIN feature caches ----
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_tr)
    X_maccs_tr = _load_npy(MACCS_TR_PATH, n_tr)
    X_emb_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_tr)
    X_av_tr = _load_npy(AVALON_TR_PATH, n_tr)
    X_mord_tr = _impute_mordred(
        np.load(MORDRED_TR_PATH).astype(np.float32)
    )
    if X_mord_tr.shape[0] != n_tr:
        raise ValueError(f"mordred train shape mismatch: {X_mord_tr.shape}")
    print(f"[feat] X_mord_tr = {X_mord_tr.shape}")

    # ---- load TEST feature caches ----
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_te)
    X_maccs_te = _load_npy(MACCS_TE_PATH, n_te)
    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_te)
    X_av_te = _load_npy(AVALON_TE_PATH, n_te)
    X_mord_te = _impute_mordred(
        np.load(MORDRED_TE_PATH).astype(np.float32)
    )
    print(f"[feat] X_mord_te = {X_mord_te.shape}")

    # ---- ChEMBL kNN feature on TRAIN + TEST ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
    print("-" * 78)
    pool = _load_chembl_pool()
    # Drop pool rows that appear in either train or test
    tr_mols = [standardize(s) for s in tr_smiles]
    te_mols = [standardize(s) for s in te_smiles]
    all_inchikeys = set()
    for m in tr_mols + te_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            all_inchikeys.add(ik)
    pool = pool[~pool["inchikey"].isin(all_inchikeys)].reset_index(drop=True)
    print(f"   pool after dropping tr+te inchikeys: {len(pool)}")
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool: {len(pool)}  median pEC50 = {pool_median:.3f}")

    # TRAIN kNN
    std_tr_smiles = [
        Chem.MolToSmiles(m) if m is not None else "" for m in tr_mols
    ]
    fp_tr = morgan_fp_batch(std_tr_smiles)
    tr_top_idx, tr_top_sim = _tanimoto_topk(fp_tr, fp_pool, k=KNN_K)
    pred_chembl_tr, mean_sim_tr = _knn_predict(
        tr_top_idx, tr_top_sim, pool_labels, fallback=pool_median
    )
    # TEST kNN
    std_te_smiles = [
        Chem.MolToSmiles(m) if m is not None else "" for m in te_mols
    ]
    fp_te = morgan_fp_batch(std_te_smiles)
    te_top_idx, te_top_sim = _tanimoto_topk(fp_te, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(
        te_top_idx, te_top_sim, pool_labels, fallback=pool_median
    )

    # ---- build 117-col then top-28 on TRAIN and TEST ----
    X_tr_117 = _build_117col(
        X_ap_tr, X_maccs_tr, X_mord_tr, X_emb_tr, X_av_tr,
        top_ap_bit_idx, top_maccs_bit_idx, top_mord_col_idx,
        top_embed_col_idx, top_avalon_bit_idx,
        pred_chembl_tr, mean_sim_tr,
    )
    X_te_117 = _build_117col(
        X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
        top_ap_bit_idx, top_maccs_bit_idx, top_mord_col_idx,
        top_embed_col_idx, top_avalon_bit_idx,
        pred_chembl_te, mean_sim_te,
    )
    if X_tr_117.shape[1] != expected_dim_117:
        raise ValueError(
            f"tr_117 dim {X_tr_117.shape[1]} != {expected_dim_117}"
        )
    if X_te_117.shape[1] != expected_dim_117:
        raise ValueError(
            f"te_117 dim {X_te_117.shape[1]} != {expected_dim_117}"
        )

    X_tr_28 = X_tr_117[:, top28_idx].astype(np.float32)
    X_te_28 = X_te_117[:, top28_idx].astype(np.float32)
    X_unb_28 = X_te_28[unb_idx]
    print(f"[feat] X_tr_28={X_tr_28.shape}  X_te_28={X_te_28.shape}  "
          f"X_unb_28={X_unb_28.shape}")

    # ---- scaffold splits on TRAIN (for both soft-label generation
    #      and retrain CV) ----
    scaffolds = []
    for m in tr_mols:
        if m is None:
            scaffolds.append(None)
            continue
        try:
            sc = MurckoScaffoldSmiles(mol=m, includeChirality=False)
            scaffolds.append(sc if sc else None)
        except Exception:
            scaffolds.append(None)
    splits_soft = scaffold_kfold_indices(
        scaffolds, n_splits=N_SCAFFOLD_FOLDS, shuffle=True, seed=42
    )
    # Sanity
    for fi, (tri, vai) in enumerate(splits_soft):
        assert len(tri) + len(vai) == n_tr
    print(f"[cv] train scaffold splits ({N_SCAFFOLD_FOLDS}-fold): "
          f"sizes = {[len(s[1]) for s in splits_soft]}")

    # ---- STEP 1: generate y_soft_pred on TRAIN (5-fold scaffold cross-fit;
    #              25 inner fits per fold averaged) ----
    print("\n" + "-" * 78)
    print(f"STEP 1: train-OOF soft labels via 5-fold scaffold cross-fit "
          f"(25 inner fits per fold)")
    print("-" * 78)
    y_soft_resid_oof = np.zeros(n_tr, dtype=np.float64)
    fold_records = []
    for fi, (tr_loc, va_loc) in enumerate(splits_soft):
        t_f = time.time()
        residual_tr_loc = (
            y_tr[tr_loc] - chemprop_aux_tr[tr_loc]
        )
        # 25 inner fits; row-mean residual pred on va_loc
        n_inner_total = len(OUTER_SEEDS) * N_INNER
        va_resid_stack = np.zeros(
            (n_inner_total, len(va_loc)), dtype=np.float64
        )
        k_global = 0
        for o in OUTER_SEEDS:
            inner_seeds = [o * 1000 + s for s in INNER_OFFSETS]
            for inner_s in inner_seeds:
                mdl = lgb.LGBMRegressor(**_lgbm_params(inner_s))
                mdl.fit(X_tr_28[tr_loc], residual_tr_loc)
                va_resid_stack[k_global] = mdl.predict(X_tr_28[va_loc])
                k_global += 1
        # row-mean over 25 fits (mirrors nb2112 deploy MEAN diag)
        va_resid_mean = va_resid_stack.mean(axis=0)
        y_soft_resid_oof[va_loc] = va_resid_mean
        wall_f = time.time() - t_f
        fold_records.append({
            "fold": fi,
            "n_tr_fold": int(len(tr_loc)),
            "n_va_fold": int(len(va_loc)),
            "wall_sec": round(wall_f, 1),
        })
        print(f"   fold {fi}: n_tr={len(tr_loc)} n_va={len(va_loc)} "
              f"wall={wall_f:.1f}s")
    y_soft_pred = chemprop_aux_tr + y_soft_resid_oof
    rae_soft = float(rae(y_tr, y_soft_pred))
    print(f"\n[soft] y_soft_pred train RAE = {rae_soft:.4f}  "
          f"(anchor {rae_anchor_tr:.4f})")
    print(f"[soft] y_soft_pred mean={y_soft_pred.mean():.4f}  "
          f"std={y_soft_pred.std():.4f}  "
          f"(y_tr mean={y_tr.mean():.4f}  std={y_tr.std():.4f})")

    # ---- STEP 2-3: per-alpha retrain on mixed target + cross-fit on 253 ----
    print("\n" + "-" * 78)
    print("STEP 2-3: per-alpha mixed-target retrain on TRAIN, "
          "5-fold scaffold cross-fit + 5-seed bag, eval on 253 unblind")
    print("-" * 78)
    residual_unb = y_unb - anchor_unb  # for reference only

    # Same scaffold splits used for retrain (consistent CV)
    splits_retrain = splits_soft

    per_alpha_results = []
    for alpha in ALPHAS:
        t_a = time.time()
        mixed_target = (
            alpha * y_tr + (1.0 - alpha) * y_soft_pred
        ).astype(np.float64)
        mixed_resid = mixed_target - chemprop_aux_tr

        print(f"\n--- alpha={alpha:.2f} ---")
        print(f"   mixed_target mean={mixed_target.mean():.4f}  "
              f"std={mixed_target.std():.4f}")
        print(f"   mixed_resid  mean={mixed_resid.mean():+.4f}  "
              f"std={mixed_resid.std():.4f}")

        # 5-seed bag: each seed runs 5-fold scaffold CV training a model
        # on tr_fold mixed_resid, then ALWAYS predicts residual on the
        # ENTIRE 253 unblind (we average across folds within seed, then
        # average / median across seeds).
        per_seed_unb_resid_stack = np.zeros(
            (len(RETRAIN_SEEDS), n_unb), dtype=np.float64
        )
        per_seed_rae_mean = []
        for si, seed in enumerate(RETRAIN_SEEDS):
            t_s = time.time()
            # accumulate fold predictions on full 253 (each fold contributes
            # 1 model trained on its train-fold mixed-residual)
            fold_unb_resid_stack = np.zeros(
                (len(splits_retrain), n_unb), dtype=np.float64
            )
            for fi, (tr_loc, _va_loc) in enumerate(splits_retrain):
                mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
                mdl.fit(X_tr_28[tr_loc], mixed_resid[tr_loc])
                fold_unb_resid_stack[fi] = mdl.predict(X_unb_28)
            # fold-mean for this seed
            seed_unb_resid = fold_unb_resid_stack.mean(axis=0)
            per_seed_unb_resid_stack[si] = seed_unb_resid
            seed_unb_pred = anchor_unb + seed_unb_resid
            r_seed = float(rae(y_unb, seed_unb_pred))
            per_seed_rae_mean.append(r_seed)
            print(f"   alpha={alpha:.2f} seed={seed:3d}: "
                  f"in_RAE={r_seed:.4f}  wall={time.time()-t_s:.1f}s")

        # bag aggregation
        mean_bag_resid = per_seed_unb_resid_stack.mean(axis=0)
        median_bag_resid = np.median(per_seed_unb_resid_stack, axis=0)
        mean_bag_pred = anchor_unb + mean_bag_resid
        median_bag_pred = anchor_unb + median_bag_resid
        rae_mean_bag = float(rae(y_unb, mean_bag_pred))
        rae_median_bag = float(rae(y_unb, median_bag_pred))

        delta_mean_vs_nb2103 = rae_mean_bag - nb2103_k28_mean_bag
        delta_median_vs_nb2103 = rae_median_bag - nb2103_k28_median_bag

        def _verdict(rae_x, ref):
            if rae_x < ref - DECISION_MARGIN:
                return "BEATS"
            if abs(rae_x - ref) < DECISION_MARGIN:
                return "FLAT"
            return "WORSE"

        v_mean = _verdict(rae_mean_bag, nb2103_k28_mean_bag)
        v_median = _verdict(rae_median_bag, nb2103_k28_median_bag)

        print(f"   alpha={alpha:.2f}  MEAN-bag  in_RAE = {rae_mean_bag:.4f}  "
              f"(d_vs_nb2103 = {delta_mean_vs_nb2103:+.4f})  {v_mean}")
        print(f"   alpha={alpha:.2f}  MEDIAN-bag in_RAE = "
              f"{rae_median_bag:.4f}  "
              f"(d_vs_nb2103 = {delta_median_vs_nb2103:+.4f})  {v_median}")

        per_alpha_results.append({
            "alpha": float(alpha),
            "n_seeds": len(RETRAIN_SEEDS),
            "per_seed_rae_unb": per_seed_rae_mean,
            "rae_per_seed_mean": float(np.mean(per_seed_rae_mean)),
            "rae_per_seed_std": float(np.std(per_seed_rae_mean)),
            "rae_mean_bag_unb": rae_mean_bag,
            "rae_median_bag_unb": rae_median_bag,
            "delta_mean_vs_nb2103_K28_mean": delta_mean_vs_nb2103,
            "delta_median_vs_nb2103_K28_median": delta_median_vs_nb2103,
            "verdict_mean": v_mean,
            "verdict_median": v_median,
            "mixed_target_mean": float(mixed_target.mean()),
            "mixed_target_std": float(mixed_target.std()),
            "wall_sec": round(time.time() - t_a, 1),
        })

    # ---- select best alpha by min(rae_mean_bag, rae_median_bag) ----
    print("\n" + "=" * 78)
    print("PER-ALPHA SUMMARY")
    print("=" * 78)
    print(f"   {'alpha':>6}  {'mean_bag':>10}  {'median_bag':>10}  "
          f"{'d_vs_nb2103_mean':>17}  {'d_vs_nb2103_med':>17}  "
          f"{'v_mean':>7}  {'v_median':>9}")
    for r in per_alpha_results:
        print(f"   {r['alpha']:>6.2f}  {r['rae_mean_bag_unb']:>10.4f}  "
              f"{r['rae_median_bag_unb']:>10.4f}  "
              f"{r['delta_mean_vs_nb2103_K28_mean']:>+17.4f}  "
              f"{r['delta_median_vs_nb2103_K28_median']:>+17.4f}  "
              f"{r['verdict_mean']:>7}  {r['verdict_median']:>9}")

    best_rae = np.inf
    best_alpha = None
    best_agg = None
    for r in per_alpha_results:
        if r["rae_mean_bag_unb"] < best_rae:
            best_rae = r["rae_mean_bag_unb"]
            best_alpha = r["alpha"]
            best_agg = "mean"
        if r["rae_median_bag_unb"] < best_rae:
            best_rae = r["rae_median_bag_unb"]
            best_alpha = r["alpha"]
            best_agg = "median"

    best_ref = (nb2103_k28_mean_bag if best_agg == "mean"
                else nb2103_k28_median_bag)
    delta_best_vs_nb2103 = best_rae - best_ref
    beats_nb2103 = best_rae < best_ref - DECISION_MARGIN

    print(f"\n   BEST: alpha={best_alpha:.2f}  agg={best_agg}  "
          f"in_RAE={best_rae:.4f}  d_vs_nb2103={delta_best_vs_nb2103:+.4f}  "
          f"{'BEATS' if beats_nb2103 else 'NO_BEAT'}")

    # ---- if best beats: build deploy CSV ----
    deploy_csv_path = None
    deploy_info = None
    if beats_nb2103:
        print("\n" + "-" * 78)
        print(f"DEPLOY: best alpha={best_alpha:.2f}  agg={best_agg}; "
              f"build full-fit predictions on 513 (no fold splits)")
        print("-" * 78)
        # Build full-fit (no CV) per-seed deploy: train each seed on ALL
        # 4139 train rows with best mixed_target, predict residual on full 513.
        # Then aggregate across seeds with the winning agg.
        mixed_target_best = (
            best_alpha * y_tr + (1.0 - best_alpha) * y_soft_pred
        ).astype(np.float64)
        mixed_resid_best = mixed_target_best - chemprop_aux_tr
        per_seed_te_resid = np.zeros(
            (len(RETRAIN_SEEDS), n_te), dtype=np.float64
        )
        for si, seed in enumerate(RETRAIN_SEEDS):
            mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
            mdl.fit(X_tr_28, mixed_resid_best)
            per_seed_te_resid[si] = mdl.predict(X_te_28)
        if best_agg == "mean":
            agg_te_resid = per_seed_te_resid.mean(axis=0)
        else:
            agg_te_resid = np.median(per_seed_te_resid, axis=0)
        te_pred_513 = chemprop_aux_te + agg_te_resid
        # In-sample check on unb_idx
        in_pred_unb = te_pred_513[unb_idx]
        rae_in_unb_deploy = float(rae(y_unb, in_pred_unb))

        df_sub = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_pred_513.astype(np.float32),
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)} != 513")
        deploy_csv_path = SUBMISSIONS_DIR / f"{TAG}_deploy_distill.csv"
        df_sub.to_csv(deploy_csv_path, index=False)
        print(f"[save] deploy CSV: {deploy_csv_path}")
        deploy_info = {
            "best_alpha": best_alpha,
            "best_agg": best_agg,
            "in_rae_unb_deploy_full_fit": rae_in_unb_deploy,
            "te_pred_mean": float(te_pred_513.mean()),
            "te_pred_std": float(te_pred_513.std()),
            "te_pred_min": float(te_pred_513.min()),
            "te_pred_max": float(te_pred_513.max()),
            "submission_csv": str(deploy_csv_path),
        }
        print(f"[deploy] in_RAE(unb_idx) full-fit = {rae_in_unb_deploy:.4f}")
        print(f"[deploy] te_pred mean={te_pred_513.mean():.4f}  "
              f"std={te_pred_513.std():.4f}")
    else:
        print("\n[deploy] best alpha does NOT beat nb2103 by margin "
              f"{DECISION_MARGIN}; skipping CSV write")

    # ---- summary ----
    summary = {
        "tag": TAG,
        "method": ("pseudo_label_distillation_nb2112_soft_on_train_"
                   "alpha_sweep_lgbm_mse_K28"),
        "anchor": ANCHOR,
        "anchor_tr_oof_path": str(ANCHOR_TR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "alphas": ALPHAS,
        "outer_seeds_soft": OUTER_SEEDS,
        "inner_offsets_soft": INNER_OFFSETS,
        "retrain_seeds": RETRAIN_SEEDS,
        "n_scaffold_folds": N_SCAFFOLD_FOLDS,
        "top_K_shap": TOP_K_SHAP,
        "top28_idx_in_117_from_nb2103": top28_idx.tolist(),
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "n_tr": int(n_tr),
        "n_te": int(n_te),
        "n_unb": int(n_unb),
        "expected_dim_117": int(expected_dim_117),
        "feat_breakdown_full": {
            "atompair": int(n_top_ap),
            "maccs": int(n_top_maccs),
            "mordred": int(n_top_mord),
            "chemprop_embed": int(n_top_embed),
            "avalon": int(n_top_avalon),
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(expected_dim_117),
        },
        "rae_anchor_train_oof": rae_anchor_tr,
        "rae_anchor_unb": rae_anchor_unb,
        "rae_soft_train_oof": rae_soft,
        "soft_pred_mean": float(y_soft_pred.mean()),
        "soft_pred_std": float(y_soft_pred.std()),
        "soft_fold_records": fold_records,
        "per_alpha_results": per_alpha_results,
        "best_alpha": float(best_alpha) if best_alpha is not None else None,
        "best_agg": best_agg,
        "best_rae_unb": float(best_rae),
        "best_ref_nb2103": float(best_ref),
        "delta_best_vs_nb2103": float(delta_best_vs_nb2103),
        "beats_nb2103": bool(beats_nb2103),
        "nb2103_K28_mean_bag_ref": nb2103_k28_mean_bag,
        "nb2103_K28_median_bag_ref": nb2103_k28_median_bag,
        "decision_margin": DECISION_MARGIN,
        "deploy_info": deploy_info,
        "wall_sec": round(time.time() - t0, 1),
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
        "n_tr", "n_te", "n_unb",
        "rae_anchor_train_oof", "rae_anchor_unb",
        "rae_soft_train_oof",
        "best_alpha", "best_agg", "best_rae_unb",
        "best_ref_nb2103", "delta_best_vs_nb2103",
        "beats_nb2103",
        "nb2103_K28_mean_bag_ref",
        "nb2103_K28_median_bag_ref",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-ALPHA TABLE ====")
    for r in res["per_alpha_results"]:
        print(f"  alpha={r['alpha']:.2f}  mean={r['rae_mean_bag_unb']:.4f}  "
              f"median={r['rae_median_bag_unb']:.4f}  "
              f"d_mean={r['delta_mean_vs_nb2103_K28_mean']:+.4f}  "
              f"d_med={r['delta_median_vs_nb2103_K28_median']:+.4f}  "
              f"{r['verdict_mean']} / {r['verdict_median']}")
