"""nb1158 -- Deploy CSV at the true scaffold-CV optimum K=32 (per nb1157).

OBJECTIVE:
    nb1157 fine-grid SCAFFOLD-CV verification of nb1151 found that K=32
    (not K=35 as nb1151 claimed) is the actual mean-bag RAE optimum on the
    253 unblind, with 0.4902 mean-bag (vs K=28 0.4927, K=30 0.4907, K=35
    0.4937).  This notebook builds the deploy CSV at K=32 using the SAME
    deploy recipe but refit on ALL 4392 labels (4139 train + 253 unblind)
    instead of just the 253 unblind that nb1151 used.

PROTOCOL:
    1. Reload the 117-col 5-way K-tuned feature matrix on:
         - 253 unblind (for honest scaffold-CV evaluation + residual fit)
         - 513 test (for deploy prediction)
         - 4139 train (for the extended deploy refit)
    2. Slice top-32 columns via the nb2063 SHAP importance ordering.
    3. Honest scaffold-CV evaluation: LGBM(MSE) K=32 with 5-seed bag
       {0, 1, 7, 42, 137} and 5-fold Murcko scaffold CV on residual =
       y_unb - chemprop_aux[unb_idx].  Report mean-bag + median-bag RAE.
    4. Deploy refit on ALL 4392 labels:
         residual_train_4139 = y_train - oof_chemprop_aux
         residual_unb_253    = y_unb   - te_chemprop_aux[unb_idx]
         combined residual on 4392 rows, X_combined = stack(X_train_32, X_unb_32)
         5-seed LGBM bag, mean-aggregated prediction on 513 test.
    5. Final = te_chemprop_aux + LGBM_deploy_residual_prediction.
    6. In-sample sanity: build a deploy fit on the 253 unblind only, slice
       te[unb_idx], confirm in-sample RAE far below honest scaffold-CV RAE
       (expected gap because deploy fit sees all 253 labels).
    7. Write submissions/nb1158_deploy_K32.csv (Molecule Name, SMILES, pEC50).

References (PRE-unblind path):
    chemprop_aux anchor       0.6216
    nb2103 K=28 random-KF     0.4737    (sanity)
    nb2103 K=28 scaffold-CV   0.5057    (task spec reference)
    nb1151 K=35 (claimed)     0.5024    (refuted by nb1157 -- K=32 is real opt)
    nb1157 K=32 fresh-seed    0.4902    (true scaffold-CV optimum)
    nb1157 K=30 fresh-seed    0.4907
    nb1157 K=28 fresh-seed    0.4927
    nb1157 K=35 fresh-seed    0.4937

Outputs:
    scripts/nb1158_k32_deploy.py
    data/processed/nb1158_summary.json
    submissions/nb1158_deploy_K32.csv
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
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1158"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ANCHOR_OOF_PATH = DATA_PROCESSED / "oof_chemprop_aux.npy"

# Protocol (identical to nb1151 K=32 slice + nb1157 verification)
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]          # original nb1151 seeds (per task spec)
K_DEPLOY = 32                              # winning K per nb1157

# Train-side feature caches (4139 rows)
ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"

# Test-side feature caches (513 rows)
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"

# Mordred cached out-of-repo
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
MORDRED_TR_PATH = MORDRED_DIR / "X_mordred_train.npy"
MORDRED_TE_PATH = MORDRED_DIR / "X_mordred_test.npy"

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

# Feature-family K-tuning summaries (same recipe as nb1151)
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB1151_SUMMARY = DATA_PROCESSED / "nb1151_summary.json"
NB1157_SUMMARY = DATA_PROCESSED / "nb1157_summary.json"

# ChEMBL kNN feature config (identical to nb1151)
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# References (from nb1151/nb1157)
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_SCAFFOLD_REF = 0.5057
NB1157_K32_REF = 0.4902
NB1157_K35_REF = 0.4937


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


def _murcko_scaffold(mol):
    try:
        if mol is None:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc is not None else None
    except Exception:
        return None


def _load_chembl_pool() -> pd.DataFrame:
    """Identical to nb1151 _load_chembl_pool."""
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
        d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
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
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
        .rename(columns={"src_first": "src"})
    )
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
    """LGBM(MSE) -- identical to nb1151/nb1157/nb2063."""
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


def _residual_scaffold_cross_fit_one_seed(X, residual, scaffolds, seed):
    n = len(residual)
    splits = scaffold_kfold_indices(scaffolds, n_splits=RESID_FOLDS,
                                    shuffle=True, seed=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred(path: Path, n_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mordred cache missing: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"Mordred shape {X.shape} vs expected {n_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy(path: Path, n_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair not found")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY at scaffold-CV optimum K={K_DEPLOY} (per nb1157)")
    print(f"          honest scaffold-CV ref @ K=32: {NB1157_K32_REF:.4f}")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          deploy refit on ALL 4392 = 4139 train + 253 unblind")
    print("=" * 78)

    # ---- nb2063 SHAP importance ----
    shap_imp = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank = np.argsort(-shap_imp).astype(np.int32)
    print(f"[ref] nb2063 SHAP importance shape = {shap_imp.shape}")

    # ---- Load train + test + chemprop_aux anchors ----
    tr_df = load_train()
    te_df = load_test()
    n_train = len(tr_df)
    n_test = len(te_df)
    train_smiles = tr_df["smiles"].astype(str).tolist()
    test_smiles = (te_df["smiles"].astype(str).tolist()
                   if "smiles" in te_df.columns
                   else te_df["SMILES"].astype(str).tolist())
    y_train = tr_df["pec50"].to_numpy(dtype=np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    oof_anchor_4139 = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"te_chemprop_aux shape {te_anchor_513.shape} vs {n_test}")
    if oof_anchor_4139.shape[0] != n_train:
        raise ValueError(f"oof_chemprop_aux shape {oof_anchor_4139.shape} vs {n_train}")

    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual_unb = y_unb - anchor_unb
    residual_train = y_train - oof_anchor_4139
    rae_anchor_train = float(rae(y_train, oof_anchor_4139))
    print(f"[load] n_train={n_train}  n_test={n_test}  n_unb={n_unb}")
    print(f"[load] chemprop_aux te[unb_idx] RAE  = {rae_anchor:.4f}  (ref {CHEMPROP_AUX_REF:.4f})")
    print(f"[load] chemprop_aux oof_4139 RAE     = {rae_anchor_train:.4f}")
    print(f"[resid_unb] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")
    print(f"[resid_tr]  mean={residual_train.mean():+.4f}  std={residual_train.std():.4f}")

    # ---- Scaffolds for 253 unblind (honest scaffold-CV) ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_mols = [standardize(s) for s in unb_smiles]
    scaffolds_unb = [_murcko_scaffold(m) for m in unb_mols]
    print(f"[scaf] n_unique_scaf_unb = "
          f"{len({s for s in scaffolds_unb if s})}")

    # ---- Load K-tuning summaries (same recipe as nb1151) ----
    with open(NB1352_SUMMARY) as f: sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f: sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f: sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f: sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f: sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f: sum_1541 = json.load(f)

    top_maccs = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
    top_mord = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap = _extract_atompair_top_idx(sum_1484)
    K_AP = int(sum_1524["best_K"])
    top_ap = full_ap[:K_AP]
    K_emb = int(sum_1541["best_K"])
    top_emb = np.array(sum_1541["top_dim_order_top100"], dtype=int)[:K_emb]
    top_av = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)
    print(f"[reuse] AP={len(top_ap)}  MACCS={len(top_maccs)}  Mord={len(top_mord)}  "
          f"Emb={len(top_emb)}  Avalon={len(top_av)}")

    # ---- Test-side feature matrices (513) ----
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test)
    X_maccs_te = _load_npy(MACCS_TE_PATH, n_test)
    X_mord_te = _load_mordred(MORDRED_TE_PATH, n_test)
    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test)
    X_av_te = _load_npy(AVALON_TE_PATH, n_test)

    # ---- Train-side feature matrices (4139) ----
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_train)
    X_maccs_tr = _load_npy(MACCS_TR_PATH, n_train)
    X_mord_tr = _load_mordred(MORDRED_TR_PATH, n_train)
    X_emb_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_train)
    X_av_tr = _load_npy(AVALON_TR_PATH, n_train)

    # ---- ChEMBL kNN pool (build once, apply to train + test) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
    print("-" * 78)
    pool = _load_chembl_pool()
    # Filter out any test InChIKeys (same convention as nb1151)
    test_mols = [standardize(s) for s in test_smiles]
    test_iks = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_iks.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_iks)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   pool: {n_before} -> {len(pool)}  median pEC50 = {pool_median:.3f}")

    # ChEMBL kNN feature on 513 test
    std_test_smiles = ["" if m is None else Chem.MolToSmiles(m) for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_te, top_sim_te = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(top_idx_te, top_sim_te,
                                                pool_labels, pool_median)

    # ChEMBL kNN feature on 4139 train (held-out pool excludes test InChIKeys
    # already; train InChIKey leakage is acceptable here since the train labels
    # are the ones we're learning).  Same recipe applied symmetrically.
    train_mols = [standardize(s) for s in train_smiles]
    std_train_smiles = ["" if m is None else Chem.MolToSmiles(m) for m in train_mols]
    fp_train = morgan_fp_batch(std_train_smiles)
    top_idx_tr, top_sim_tr = _tanimoto_topk(fp_train, fp_pool, k=KNN_K)
    pred_chembl_tr, mean_sim_tr = _knn_predict(top_idx_tr, top_sim_tr,
                                                pool_labels, pool_median)

    # ---- Build full 117-col matrices on train + test ----
    X_te_full = np.concatenate(
        [X_ap_te[:, top_ap], X_maccs_te[:, top_maccs], X_mord_te[:, top_mord],
         X_emb_te[:, top_emb], X_av_te[:, top_av],
         pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1)],
        axis=1).astype(np.float32)
    X_tr_full = np.concatenate(
        [X_ap_tr[:, top_ap], X_maccs_tr[:, top_maccs], X_mord_tr[:, top_mord],
         X_emb_tr[:, top_emb], X_av_tr[:, top_av],
         pred_chembl_tr.reshape(-1, 1), mean_sim_tr.reshape(-1, 1)],
        axis=1).astype(np.float32)
    feat_dim = X_te_full.shape[1]
    if feat_dim != shap_imp.shape[0]:
        raise ValueError(f"feat_dim {feat_dim} != SHAP dim {shap_imp.shape[0]}")
    X_unb_full = X_te_full[unb_idx]
    print(f"\n[feat] X_te (513, {feat_dim})  X_tr (4139, {feat_dim})  "
          f"X_unb (253, {feat_dim})")

    # ---- Slice top-K=32 ----
    topK_idx = full_rank[:K_DEPLOY].astype(np.int32)
    X_te_K = X_te_full[:, topK_idx].astype(np.float32)
    X_tr_K = X_tr_full[:, topK_idx].astype(np.float32)
    X_unb_K = X_unb_full[:, topK_idx].astype(np.float32)
    print(f"[K=32] X_te_K {X_te_K.shape}  X_tr_K {X_tr_K.shape}  X_unb_K {X_unb_K.shape}")

    # ---- Step 3/4: Honest scaffold-CV evaluation on 253 unblind ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD-CV EVAL @ K={K_DEPLOY}  seeds={RESID_SEEDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_scaffold_cross_fit_one_seed(
            X_unb_K, residual_unb, scaffolds_unb, s
        )
        pred_corr_s = anchor_unb + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        print(f"   seed={s:>3d}  rae={rae_s:.4f}  d_anchor={rae_s - rae_anchor:+.4f}  "
              f"wall={time.time() - ts:.1f}s")
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    arr = np.array(per_seed_rae)
    print(f"\n   per_seed RAE = [{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per_seed mean = {arr.mean():.4f}  std = {arr.std():.4f}  "
          f"range = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   MEAN-BAG   = {rae_mean_bag:.4f}  (nb1157 K=32 ref {NB1157_K32_REF:.4f}, "
          f"d={rae_mean_bag - NB1157_K32_REF:+.4f})")
    print(f"   MEDIAN-BAG = {rae_median_bag:.4f}")

    # ---- Step 5/6: Deploy refit on ALL 4392 = 4139 train + 253 unblind ----
    print("\n" + "-" * 78)
    print(f"DEPLOY REFIT on 4392 = 4139 train + 253 unblind  seeds={RESID_SEEDS}")
    print("-" * 78)
    X_combined = np.concatenate([X_tr_K, X_unb_K], axis=0).astype(np.float32)
    residual_combined = np.concatenate([residual_train, residual_unb], axis=0)
    print(f"[combine] X_combined {X_combined.shape}  residual {residual_combined.shape}")
    print(f"[combine] residual mean={residual_combined.mean():+.4f}  "
          f"std={residual_combined.std():.4f}")

    seed_preds_513 = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_combined, residual_combined)
        seed_preds_513[i] = mdl.predict(X_te_K)
        print(f"   deploy seed={s:>3d}  resid_pred mean={seed_preds_513[i].mean():+.4f}  "
              f"std={seed_preds_513[i].std():.4f}  wall={time.time() - ts:.1f}s")
    resid_pred_513 = seed_preds_513.mean(axis=0)
    final_513 = te_anchor_513 + resid_pred_513
    print(f"\n[deploy] mean_bag residual on 513: mean={resid_pred_513.mean():+.4f}  "
          f"std={resid_pred_513.std():.4f}")
    print(f"[deploy] final pred 513: mean={final_513.mean():.4f}  "
          f"std={final_513.std():.4f}  min={final_513.min():.3f}  "
          f"max={final_513.max():.3f}")

    # ---- In-sample sanity (te[unb_idx] vs y_unb under DEPLOY fit) ----
    deploy_in_sample_unb = final_513[unb_idx]
    rae_in_sample = float(rae(y_unb, deploy_in_sample_unb))
    print(f"\n[in-sample] te[unb_idx] RAE (deploy fit, contains 253 labels) "
          f"= {rae_in_sample:.4f}")
    print(f"[in-sample] honest scaffold-CV mean-bag RAE = {rae_mean_bag:.4f}")
    print(f"[in-sample] gap (deploy_fit - honest) = "
          f"{rae_in_sample - rae_mean_bag:+.4f}  "
          f"(expected NEGATIVE: deploy fit sees 253 labels in-sample)")

    # ---- Write deploy CSV ----
    name_col = "name" if "name" in te_df.columns else "Molecule Name"
    names = te_df[name_col].astype(str).tolist()
    smis_full = test_smiles
    sub_df = pd.DataFrame({
        "Molecule Name": names,
        "SMILES": smis_full,
        "pEC50": final_513.astype(np.float64),
    })
    SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    deploy_path = SUB_DIR / f"{TAG}_deploy_K{K_DEPLOY}.csv"
    sub_df.to_csv(deploy_path, index=False)
    print(f"\n[deploy] wrote {deploy_path}  ({len(sub_df)} rows, "
          f"{len(sub_df.columns)} cols)")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("lgbm_mse_K32_5seed_bag_deploy_on_4392_with_chemprop_aux_anchor_"
                   "scaffold_cv_eval_on_253_unblind"),
        "anchor": ANCHOR,
        "K_deploy": int(K_DEPLOY),
        "K_deploy_source": "nb1157 fine-grid scaffold-CV optimum",
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "cv_protocol": ("scaffold_kfold_indices(Murcko, n=5, shuffle=True, seed=s) "
                        "for s in RESID_SEEDS"),
        "lgbm_params_seed0": _lgbm_params(0),
        "feat_dim_full": int(feat_dim),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_deploy_combined": int(n_train + n_unb),
        "n_chembl_pool": int(len(pool)),
        "rae_anchor_unb": rae_anchor,
        "rae_anchor_train_oof": rae_anchor_train,
        "residual_unb_mean": float(residual_unb.mean()),
        "residual_unb_std": float(residual_unb.std()),
        "residual_train_mean": float(residual_train.mean()),
        "residual_train_std": float(residual_train.std()),
        "residual_combined_mean": float(residual_combined.mean()),
        "residual_combined_std": float(residual_combined.std()),
        "per_seed_rae_unb": per_seed_rae,
        "rae_per_seed_mean": float(arr.mean()),
        "rae_per_seed_median": float(np.median(arr)),
        "rae_per_seed_std": float(arr.std()),
        "rae_per_seed_min": float(arr.min()),
        "rae_per_seed_max": float(arr.max()),
        "rae_mean_bag_unb_scaffold_cv": rae_mean_bag,
        "rae_median_bag_unb_scaffold_cv": rae_median_bag,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1157_K32_ref": rae_mean_bag - NB1157_K32_REF,
        "delta_mean_bag_vs_nb2103_K28_scaffold": rae_mean_bag - NB2103_K28_SCAFFOLD_REF,
        "deploy_resid_pred_513_mean": float(resid_pred_513.mean()),
        "deploy_resid_pred_513_std": float(resid_pred_513.std()),
        "deploy_final_513_mean": float(final_513.mean()),
        "deploy_final_513_std": float(final_513.std()),
        "deploy_final_513_min": float(final_513.min()),
        "deploy_final_513_max": float(final_513.max()),
        "in_sample_rae_te_at_unb_idx": rae_in_sample,
        "in_sample_minus_scaffold_cv_gap": rae_in_sample - rae_mean_bag,
        "deploy_csv_path": str(deploy_path),
        "deploy_csv_rows": int(len(sub_df)),
        "deploy_csv_cols": list(sub_df.columns),
        "references": {
            "chemprop_aux_anchor": CHEMPROP_AUX_REF,
            "nb2103_K28_scaffold_cv": NB2103_K28_SCAFFOLD_REF,
            "nb1157_K32_fresh_seed": NB1157_K32_REF,
            "nb1157_K35_fresh_seed": NB1157_K35_REF,
        },
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
    print("\n==== nb1158 SUMMARY ====")
    for k in (
        "K_deploy", "rae_anchor_unb", "rae_mean_bag_unb_scaffold_cv",
        "rae_median_bag_unb_scaffold_cv", "delta_mean_bag_vs_nb1157_K32_ref",
        "in_sample_rae_te_at_unb_idx", "in_sample_minus_scaffold_cv_gap",
        "deploy_final_513_mean", "deploy_final_513_std",
        "deploy_csv_path", "deploy_csv_rows",
    ):
        print(f"  {k}: {res.get(k)}")
