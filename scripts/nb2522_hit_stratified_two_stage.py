"""nb2522 -- Hit-stratified two-stage model on 117-col K-tuned features.

NEW PARADIGM (different from residual-on-anchor pyramids):
    Stage 1: LGBMClassifier trained on FULL 4139 train labels with binary
             y_bin = (y >= 5.5) (hit/non-hit at threshold 5.5; ~9.2% hits).
             Fits per LGBM fold inside 5-fold scaffold CV on 4139 train.
             At inference, refit on all 4139 -> outputs P(hit) for each
             of 513 test rows.

    Stage 2a: LGBMRegressor trained ONLY on hits (y >= 5.5 subset of 4139)
              with same 117-col features.  Predicts pEC50 magnitude conditioned
              on "this row is a hit".

    Stage 2b: LGBMRegressor trained ONLY on non-hits (y < 5.5 subset of 4139)
              with same 117-col features.  Predicts pEC50 magnitude conditioned
              on "this row is not a hit".

    Inference (per test row):
        pred = P(hit) * stage2a(row) + (1 - P(hit)) * stage2b(row)

    Evaluated on 253 unblind via 5-fold scaffold CV on the UNBLIND scaffolds
    (the gate decision metric).  5 kf_seeds {1001..1005}.

    The cross-fit protocol for the gate:
        For each kf_seed and each of 5 scaffold-CV folds OVER 253 UNBLIND:
            - "train" rows on the unblind side are unused for refit; the
              4139-trained classifier+regressors run on the held-out unblind
              fold (i.e. the 4139 -> 253 pipeline is fixed; the kf-folds
              just chunk the 253 unblind for fair pooled scoring).
        Per-seed pooled RAE = RAE(y_unb, oof_blend).

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4601  -> MARGINAL_BEAT
    else FAIL

Outputs:
    scripts/nb2522_hit_stratified_two_stage.py
    data/processed/nb2522_summary.json
    data/processed/nb2522_pred_oof.npy       (253,) float32  pooled across kf_seeds
    data/processed/te_nb2522.npy             (513,) float32  deploy
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
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2522"

# Threshold for hit class label (per task: 5.5 for class balance ~9%)
HIT_THRESHOLD = 5.5

SEEDS = [0, 1, 7, 42, 137]          # bagging seeds inside each stage
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# -------- feature build paths (same as nb1983 -> 117 cols) -------------------
ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# -------- helpers (same as nb1983) -------------------------------------------

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


def _load_mordred(path, n_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"Mordred shape mismatch: {X.shape} vs {n_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy(path, n_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} vs {n_expected}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _extract_atompair_top_idx(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484")


def _clf_params(seed):
    """LGBMClassifier (binary) -- mild capacity to avoid overfit on 9% hits."""
    return dict(
        objective="binary",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=20,
        is_unbalance=True,           # auto-balance 9% positives
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _reg_params(seed):
    """LGBMRegressor for stage 2a/2b -- same shape as nb1983 / nb2240."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=10,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _bag_fit_predict_clf(X_tr, y_bin_tr, X_eval):
    """Fit |SEEDS| LGBMClassifier on (X_tr, y_bin_tr), return mean P(hit) on X_eval."""
    probs = np.zeros(X_eval.shape[0], dtype=np.float64)
    for s in SEEDS:
        m = lgb.LGBMClassifier(**_clf_params(s))
        m.fit(X_tr, y_bin_tr)
        p = m.predict_proba(X_eval)
        # column 1 is the "hit" probability (positive class)
        idx_hit = list(m.classes_).index(1)
        probs += p[:, idx_hit]
    return probs / len(SEEDS)


def _bag_fit_predict_reg(X_tr, y_tr, X_eval):
    preds = np.zeros(X_eval.shape[0], dtype=np.float64)
    for s in SEEDS:
        m = lgb.LGBMRegressor(**_reg_params(s))
        m.fit(X_tr, y_tr)
        preds += m.predict(X_eval)
    return preds / len(SEEDS)


# =============================================================================
# Build 117-col features for 4139 train + 513 test (same recipe as nb1983)
# =============================================================================

def build_117_matrices(tr_smiles_raw, te_smiles_raw, n_train, n_test):
    print("\n" + "-" * 78)
    print("BUILDING 117-COL FEATURE MATRIX (train + test)")
    print("-" * 78)

    # Standardize SMILES
    tr_mols = [standardize(s) for s in tr_smiles_raw]
    te_mols = [standardize(s) for s in te_smiles_raw]
    tr_std_smi = [Chem.MolToSmiles(m) if m is not None else "" for m in tr_mols]
    te_std_smi = [Chem.MolToSmiles(m) if m is not None else "" for m in te_mols]

    # K indices
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

    top_maccs = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    top_avalon = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)
    K_AP = int(sum_1524["best_K"])
    full_ap = _extract_atompair_top_idx(sum_1484)
    top_ap = full_ap[:K_AP]
    K_Mord = int(sum_1523["best_K"])
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
    top_mord = np.array(rec_mord["top_col_idx"], dtype=int)
    K_Emb = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed = top_embed_full[:K_Emb]

    print(f"   AtomPair top-{len(top_ap)}  MACCS top-{len(top_maccs)}  "
          f"Mordred top-{len(top_mord)}  ChempropEmbed top-{len(top_embed)}  "
          f"Avalon top-{len(top_avalon)}")

    # Load each family + slice
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_train)[:, top_ap].astype(np.float32)
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test)[:, top_ap].astype(np.float32)
    X_mac_tr = _load_npy(MACCS_TR_PATH, n_train)[:, top_maccs].astype(np.float32)
    X_mac_te = _load_npy(MACCS_TE_PATH, n_test)[:, top_maccs].astype(np.float32)
    X_mord_tr_full = _load_mordred(MORDRED_DIR / "X_mordred_train.npy", n_train)
    X_mord_te_full = _load_mordred(MORDRED_DIR / "X_mordred_test.npy", n_test)
    X_mord_tr = X_mord_tr_full[:, top_mord].astype(np.float32)
    X_mord_te = X_mord_te_full[:, top_mord].astype(np.float32)
    X_emb_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_train)[:, top_embed].astype(np.float32)
    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed].astype(np.float32)
    X_av_tr = _load_npy(AVALON_TR_PATH, n_train)[:, top_avalon].astype(np.float32)
    X_av_te = _load_npy(AVALON_TE_PATH, n_test)[:, top_avalon].astype(np.float32)

    # ChEMBL kNN features
    pool = _load_chembl_pool()
    tr_te_ik = set()
    for m in tr_mols + te_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            tr_te_ik.add(ik)
    pool = pool[~pool["inchikey"].isin(tr_te_ik)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    fp_tr = morgan_fp_batch(tr_std_smi)
    fp_te = morgan_fp_batch(te_std_smi)
    top_idx_tr, top_sim_tr = _tanimoto_topk(fp_tr, fp_pool, k=KNN_K)
    pred_chembl_tr, mean_sim_tr = _knn_predict(
        top_idx_tr, top_sim_tr, pool_labels, fallback=pool_median
    )
    top_idx_te, top_sim_te = _tanimoto_topk(fp_te, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_te, top_sim_te, pool_labels, fallback=pool_median
    )

    X_tr = np.concatenate(
        [X_ap_tr, X_mac_tr, X_mord_tr, X_emb_tr, X_av_tr,
         pred_chembl_tr.reshape(-1, 1), mean_sim_tr.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    X_te = np.concatenate(
        [X_ap_te, X_mac_te, X_mord_te, X_emb_te, X_av_te,
         pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    assert X_tr.shape[1] == X_te.shape[1] == 117, (
        f"feat dim mismatch: tr={X_tr.shape[1]} te={X_te.shape[1]} expected 117"
    )
    print(f"   X_tr = {X_tr.shape}  X_te = {X_te.shape}  (117-col K-tuned)")
    return X_tr, X_te, tr_std_smi, te_std_smi


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Hit-stratified two-stage (clf P(hit) + 2 reg heads)")
    print(f"   HIT_THRESHOLD = {HIT_THRESHOLD}   SEEDS = {SEEDS}")
    print(f"   KF_SEEDS      = {KF_SEEDS}        N_FOLDS = {N_FOLDS}")
    print("=" * 78)

    # ---- Load truth ----
    tr = load_train()
    te = load_test()
    n_train = len(tr)
    n_test = len(te)
    tr_smiles_raw = tr["smiles"].astype(str).tolist()
    te_smiles_raw = te["smiles"].astype(str).tolist()
    y_train = tr["pec50"].astype(np.float64).to_numpy()
    y_bin_train = (y_train >= HIT_THRESHOLD).astype(int)
    print(f"[load] n_train={n_train}  n_test={n_test}  "
          f"hits frac = {y_bin_train.mean():.4f}  "
          f"({y_bin_train.sum()} hits / {n_train})")
    if y_bin_train.sum() < 50:
        raise ValueError(f"too few hits {y_bin_train.sum()}; relax threshold")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}  y_unb range "
          f"[{y_unb.min():.2f},{y_unb.max():.2f}]  "
          f"hits frac unblind = {(y_unb >= HIT_THRESHOLD).mean():.4f}")

    if n_train != 4139:
        raise ValueError(f"expected 4139 train, got {n_train}")
    if n_test != 513:
        raise ValueError(f"expected 513 test, got {n_test}")
    if n_unb != 253:
        raise ValueError(f"expected 253 unblind, got {n_unb}")

    # ---- Build 117-col features for train + test ----
    X_tr, X_te, _, te_std_smi = build_117_matrices(
        tr_smiles_raw, te_smiles_raw, n_train, n_test
    )
    X_unb = X_te[unb_idx]

    # ---- STAGE 1: classifier scaffold-CV sanity on 4139 ----
    print("\n" + "-" * 78)
    print("STAGE 1 SANITY -- classifier scaffold-CV on 4139 train (5 folds)")
    print("-" * 78)
    tr_scaf = [bemis_murcko(s) for s in tr_smiles_raw]
    sanity_splits = scaffold_kfold_indices(
        tr_scaf, n_splits=5, shuffle=True, seed=42
    )
    clf_oof = np.zeros(n_train, dtype=np.float64)
    for f, (tr_idx, va_idx) in enumerate(sanity_splits):
        probs_va = _bag_fit_predict_clf(
            X_tr[tr_idx], y_bin_train[tr_idx], X_tr[va_idx]
        )
        clf_oof[va_idx] = probs_va
    # AUC + Brier (cheap sanity)
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auc = float(roc_auc_score(y_bin_train, clf_oof))
        brier = float(brier_score_loss(y_bin_train, clf_oof))
    except Exception:
        auc = float("nan")
        brier = float("nan")
    print(f"   classifier OOF AUC = {auc:.4f}  Brier = {brier:.4f}  "
          f"P_hit mean = {clf_oof.mean():.4f}")

    # ---- STAGE 1 DEPLOY: refit classifier on all 4139, predict on 513 ----
    print("\n[stage1 deploy] refit on full 4139 -> P(hit) for 513 test")
    p_hit_te = _bag_fit_predict_clf(X_tr, y_bin_train, X_te)
    p_hit_unb = p_hit_te[unb_idx]
    print(f"   p_hit_te mean = {p_hit_te.mean():.4f}  "
          f"std = {p_hit_te.std():.4f}  range "
          f"[{p_hit_te.min():.4f},{p_hit_te.max():.4f}]")
    print(f"   p_hit_unb mean = {p_hit_unb.mean():.4f}  "
          f"std = {p_hit_unb.std():.4f}")
    print(f"   (target unblind hit frac empirical = "
          f"{(y_unb >= HIT_THRESHOLD).mean():.4f})")

    # ---- STAGE 2a/2b DEPLOY: regressors on hits/non-hits subsets of 4139 ----
    print("\n[stage2 deploy] hits-only regressor on (y >= 5.5) subset of 4139")
    hits_mask = y_bin_train == 1
    n_hits = int(hits_mask.sum())
    n_nh = n_train - n_hits
    print(f"   stage2a training rows (hits)     = {n_hits}")
    print(f"   stage2b training rows (non-hits) = {n_nh}")
    if n_hits < 30:
        raise ValueError(f"too few hits {n_hits} -- can't train stage2a")

    pred2a_te = _bag_fit_predict_reg(
        X_tr[hits_mask], y_train[hits_mask], X_te
    )
    pred2b_te = _bag_fit_predict_reg(
        X_tr[~hits_mask], y_train[~hits_mask], X_te
    )
    print(f"   pred2a_te mean = {pred2a_te.mean():.3f} "
          f"std = {pred2a_te.std():.3f}  range "
          f"[{pred2a_te.min():.3f},{pred2a_te.max():.3f}]")
    print(f"   pred2b_te mean = {pred2b_te.mean():.3f} "
          f"std = {pred2b_te.std():.3f}  range "
          f"[{pred2b_te.min():.3f},{pred2b_te.max():.3f}]")

    # ---- Soft-blend on 513 ----
    deploy_te = (p_hit_te * pred2a_te + (1.0 - p_hit_te) * pred2b_te).astype(np.float32)
    pred_unb_deploy = deploy_te[unb_idx]
    in_sample_unb_rae = float(rae(y_unb, pred_unb_deploy))
    print(f"\n[deploy 513] blended pred mean = {deploy_te.mean():.3f}  "
          f"std = {deploy_te.std():.3f}  "
          f"range [{deploy_te.min():.3f},{deploy_te.max():.3f}]")
    print(f"[deploy 513] te[unb_idx] in-sample RAE = {in_sample_unb_rae:.4f}  "
          f"(diagnostic only -- gate uses cross-fit below)")

    # ---- Cross-fit on 253 unblind scaffolds ----
    # NB: stages 1/2a/2b are all 4139-trained; the per-kf-seed scaffold CV
    # over 253 just chunks the unblind for pooled scoring.  Therefore the OOF
    # pred is the SAME across all kf_seeds (since we never refit on unblind),
    # but we report per-seed pooled RAE to verify the metric is invariant to
    # fold partition (sanity).
    print("\n" + "-" * 78)
    print("CROSS-FIT ON 253 UNBLIND  (gate decision)")
    print("-" * 78)
    unb_scaffolds = [bemis_murcko(te_std_smi[i]) for i in unb_idx]

    per_seed_results = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            # Predictions for held-out unblind rows are the DEPLOY pipeline
            # outputs (4139-trained classifier + regressors); fold split only
            # determines which rows enter the held-out pooled-RAE bin.
            oof[va_loc] = pred_unb_deploy[va_loc]
        assert not np.isnan(oof).any(), "oof has NaN -- fold cover incomplete"
        pooled_rae_s = float(rae(y_unb, oof))
        per_seed_results.append({"kf_seed": int(kf_seed),
                                 "pooled_rae": pooled_rae_s})
        all_oofs.append(oof)
        print(f"   kf_seed={kf_seed}  pooled RAE = {pooled_rae_s:.4f}")
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    mean_rae = float(np.mean([r["pooled_rae"] for r in per_seed_results]))
    std_rae = float(np.std([r["pooled_rae"] for r in per_seed_results]))
    rae_mean_of_oofs = float(rae(y_unb, mean_oof))
    print(f"\n   mean pooled RAE across {len(KF_SEEDS)} kf_seeds = "
          f"{mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   RAE of mean-of-seed OOFs                    = {rae_mean_of_oofs:.4f}")

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[GATE] mean_rae={mean_rae:.4f}  vs PROMOTE<{GATE_PROMOTE}  "
          f"MARGINAL<{GATE_MARGINAL}  -> {verdict}")

    # ---- Save ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "hit-stratified two-stage (LGBM clf + 2 LGBM reg heads, soft-blend)",
        "hit_threshold": HIT_THRESHOLD,
        "seeds_bag": SEEDS,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_train": n_train,
        "n_test": n_test,
        "n_unb": n_unb,
        "n_hits_train": n_hits,
        "n_nonhits_train": n_nh,
        "hits_frac_train": float(y_bin_train.mean()),
        "hits_frac_unblind": float((y_unb >= HIT_THRESHOLD).mean()),
        "feat_dim": int(X_tr.shape[1]),
        "stage1_clf_auc_train_oof": auc,
        "stage1_clf_brier_train_oof": brier,
        "p_hit_te_mean": float(p_hit_te.mean()),
        "p_hit_te_std": float(p_hit_te.std()),
        "p_hit_unb_mean": float(p_hit_unb.mean()),
        "stage2a_te_mean": float(pred2a_te.mean()),
        "stage2a_te_std": float(pred2a_te.std()),
        "stage2b_te_mean": float(pred2b_te.mean()),
        "stage2b_te_std": float(pred2b_te.std()),
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "in_sample_unb_rae": in_sample_unb_rae,
        "per_seed_results": per_seed_results,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_of_seed_oofs": rae_mean_of_oofs,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(oof_path),
        "te_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    sumpath = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(sumpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {sumpath}")
    print(f"\n[done] {TAG}  verdict={verdict}  mean_rae={mean_rae:.4f}  "
          f"wall={time.time()-t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "hit_threshold", "n_hits_train", "hits_frac_train", "hits_frac_unblind",
        "stage1_clf_auc_train_oof", "stage1_clf_brier_train_oof",
        "p_hit_te_mean", "p_hit_unb_mean",
        "stage2a_te_mean", "stage2b_te_mean", "deploy_te_mean",
        "in_sample_unb_rae", "mean_rae", "std_rae", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
