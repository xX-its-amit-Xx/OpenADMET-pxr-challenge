"""nb3283 -- Deep kernel learning (NN feature extractor + GP-style RBF head)
on the nb3090 residual.

NEW PARADIGM (deep kernel / metric learning):
    An NN encoder maps the K=20 SHAP feature vector into an 8-dim latent space.
    The encoder is trained with a CONTRASTIVE-STYLE metric loss so that pairs of
    compounds with CLOSE residuals land CLOSE in latent space (and far residuals
    land far). At inference we do a GP-style RBF-weighted kNN-5 regression on the
    residual in that learned latent space -- i.e. an approximate deep kernel
    (NN encoder feeding an RBF distance kernel) implemented in pure torch.

    This is distinct from prior post-hoc-blend / quantile-conditional moves on
    this anchor: the residual is predicted by a *learned-metric* neighborhood
    rather than a tree split or convex weight. The anchor whose residual we model
    is nb3090 (the finer-q_cut quantile-conditional K18/K19 deep-30 blend,
    pred_oof RAE 0.4472-band), so this is a substrate-change attempt layered on
    the current best post-hoc-blend ceiling.

ARCHITECTURE:
    Encoder:  Linear(20, 32) -> ReLU -> Linear(32, 8)  (latent z)
    Loss:     pairwise metric loss aligning squared latent distance to squared
              residual difference (close residuals -> close latent), plus a small
              variance/spread regularizer to prevent latent collapse.
    Optim:    Adam lr=1e-3, 200 epochs (full-batch pairwise on fold-train).
    Head:     latent kNN-5 regression on residual, RBF-weighted by latent
              distance (median-heuristic bandwidth per query).
    Final:    pred = nb3090 + latent_kNN_residual_pred.

PROTOCOL (per kf_seed, 5-fold scaffold split on the 253 unblind):
    For each outer fold:
      1. StandardScaler fit on fold-train K=20 features (transform val).
      2. Train encoder (200 epochs Adam) on fold-train (features + residual).
      3. Encode fold-train and fold-val into 8-dim latent.
      4. For each fold-val row, find 5 nearest fold-train rows in latent L2;
         RBF weight w = exp(-d^2 / (2*h^2)) with h = median of the 5 dists;
         residual_hat = sum(w * r_train) / sum(w).
      5. oof_pred[val] = nb3090_oof[val] + residual_hat.
    pooled_rae across the 5 folds. Repeat for 15 kf_seeds {1216..1230};
    report per-fold-mean (mean of the 15 pooled RAEs).

GATE (on 15-seed per-fold-mean):
    mean < 0.4423 -> "BETTER"
    else          -> "FAIL"

DEPLOY te (513):
    Train encoder on ALL 253 (full-batch pairwise), build latent bank from all
    253, RBF-weighted kNN-5 residual for each of the 513 test rows, add to
    nb3090 te. (In-sample 253 slice reported as a diagnostic, NOT the gate.)

References:
    nb3090 pred_oof (finer q_cut blend)  = 0.4472 (15-seed mean)
    nb3080 wide-seed quantile-conditional = 0.4475
    nb2960 K18 deep-30 OOF                = 0.4536
    nb3000 K19 deep-30 OOF                = 0.4607
    nb2171 prior post-hoc ceiling         = 0.4682
    GATE target                           = 0.4423

Feature source (K=20 SHAP top of the 117-col 5-way K-tuned matrix), rebuilt
identically to nb2103 / nb2112 (AtomPair + MACCS + Mordred + ChempropEmbed +
Avalon + ChEMBL-kNN(pred,sim)), then sliced to the top-20 by nb2063 SHAP
importance. Built on FULL 513 then sliced to the 253 unblind rows so deploy te
uses the same columns.

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy
    data/processed/te_nb3090.npy
    data/processed/nb2063_shap_importance_full117.npy
    data/processed/te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy,
        te_avalon512.npy
    C:/pxr_artifacts/nb1030/X_mordred_test.npy
    data/processed/nb1352|nb1392|nb1484|nb1523|nb1524|nb1541_summary.json
    data/external/chembl_*.parquet

Outputs:
    data/processed/nb3283_summary.json
    data/processed/nb3283_pred_oof.npy  (253,) float32 -- median-seed OOF
    data/processed/te_nb3283.npy        (513,) float32 -- deploy te
    submissions/nb3283_deep_kernel.csv  (only on BETTER verdict)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3283"
PARENT_TAG = "nb3090"

# -- Anchor (residual base) ----------------------------------------------------
NB3090_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"   # (253,)
NB3090_TE_PATH = DATA_PROCESSED / "te_nb3090.npy"          # (513,)

# -- Feature build (identical 117-col stack to nb2103 / nb2112) ----------------
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

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
KNN_CHEMBL_K = 5
SIM_FLOOR = 1e-6

# -- Deep-kernel feature dim ---------------------------------------------------
TOP_K_FEAT = 20          # Linear(20, 32) per prescription
LATENT_DIM = 8
HIDDEN_DIM = 32
N_EPOCHS = 200
LR = 1e-3
KNN_K = 5                # latent kNN-5 residual regression
SPREAD_REG = 1e-3        # latent variance regularizer weight (anti-collapse)

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 fresh seeds {1216..1230}
ENC_SEED_BASE = 7000                # encoder init seed = ENC_SEED_BASE + kf_seed

# -- Gate ----------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ----------------------------------------------------------------
REF_NB3090 = 0.4472
REF_NB3080 = 0.4475
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


# =============================================================================
# Feature-build helpers (copied faithfully from nb2112_deploy_shap28.py)
# =============================================================================
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


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs {n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


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


def build_feature_matrices(test_smiles, unb_idx, n_test):
    """Rebuild the 117-col 5-way K-tuned matrix on FULL 513, slice top-20 SHAP.

    Returns (X_te_K20 (513,20), X_unb_K20 (253,20), top20_idx (20,)).
    """
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
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", "best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    n_top_ap = int(len(top_ap_bit_idx))
    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(len(top_avalon_bit_idx))
    print(f"[reuse] AtomPair={n_top_ap} MACCS={n_top_maccs} "
          f"Mordred={n_top_mord} ChempropEmbed={n_top_embed} "
          f"Avalon={n_top_avalon} + ChEMBL-kNN=2")

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)

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
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
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
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_CHEMBL_K)
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median)
    pred_chembl_te = pred_chembl_te.astype(np.float32)
    mean_sim_te = mean_sim_te.astype(np.float32)

    X_te_117 = np.concatenate(
        [X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top,
         X_av_te_top, pred_chembl_te.reshape(-1, 1),
         mean_sim_te.reshape(-1, 1)], axis=1).astype(np.float32)
    feat_dim_full = X_te_117.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim_full != expected_dim:
        raise ValueError(f"feat_dim_full {feat_dim_full} != {expected_dim}")

    shap_imp = np.load(NB2063_SHAP_IMP).astype(np.float32)
    if shap_imp.shape[0] != feat_dim_full:
        raise ValueError(
            f"SHAP importance len {shap_imp.shape[0]} != feat_dim {feat_dim_full}")
    full_rank_order = np.argsort(-shap_imp).astype(np.int32)
    top20_idx = full_rank_order[:TOP_K_FEAT].astype(np.int32)
    print(f"\n   X_te_117={X_te_117.shape}; SHAP top-{TOP_K_FEAT} "
          f"idx head: {top20_idx[:8].tolist()}")

    X_te_K20 = X_te_117[:, top20_idx].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx].astype(np.float32)
    return X_te_K20, X_unb_K20, top20_idx


# =============================================================================
# Deep kernel: NN encoder + RBF-kNN latent residual head
# =============================================================================
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Encoder(nn.Module):
    """Linear(20,32) -> ReLU -> Linear(32,8) latent."""

    def __init__(self, in_dim=TOP_K_FEAT, hidden=HIDDEN_DIM, latent=LATENT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, latent)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def _pdist2(z: torch.Tensor) -> torch.Tensor:
    """Squared pairwise euclidean distances (n, n), numerically floored >=0."""
    sq = (z * z).sum(dim=1, keepdim=True)
    d2 = sq + sq.t() - 2.0 * (z @ z.t())
    return torch.clamp(d2, min=0.0)


def train_encoder(X_tr: np.ndarray, r_tr: np.ndarray, seed: int) -> Encoder:
    """Train the encoder with a contrastive metric loss.

    Loss: align squared LATENT distance to squared RESIDUAL difference over all
    fold-train pairs (close residuals -> close latent). Targets are normalized
    by the residual-diff scale so the loss is dimensionless and stable; a small
    negative spread term discourages latent collapse.
    """
    _set_seed(seed)
    device = torch.device("cpu")
    enc = Encoder().to(device)
    opt = Adam(enc.parameters(), lr=LR)

    Xt = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    rt = torch.from_numpy(r_tr.astype(np.float32)).to(device).view(-1, 1)

    # Target pairwise squared residual differences (n, n), scale-normalized.
    with torch.no_grad():
        rd = rt - rt.t()                       # (n, n)
        rd2 = rd * rd                          # squared residual diff
        scale = rd2.mean().clamp(min=1e-6)     # mean off+on diagonal
        rd2_norm = rd2 / scale
    n = Xt.shape[0]
    eye = torch.eye(n, device=device)
    offdiag = 1.0 - eye

    enc.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        z = enc(Xt)                            # (n, latent)
        d2 = _pdist2(z)                        # (n, n) squared latent dist
        d2_norm = d2 / d2.mean().clamp(min=1e-6)
        # Metric alignment over off-diagonal pairs.
        align = ((d2_norm - rd2_norm) ** 2 * offdiag).sum() / offdiag.sum()
        # Anti-collapse: encourage non-trivial latent spread (variance per dim).
        spread = z.var(dim=0, unbiased=False).mean()
        loss = align - SPREAD_REG * spread
        loss.backward()
        opt.step()
    enc.eval()
    return enc


def _encode(enc: Encoder, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        z = enc(torch.from_numpy(X.astype(np.float32))).numpy()
    return z.astype(np.float64)


def rbf_knn_residual(
    z_bank: np.ndarray, r_bank: np.ndarray, z_query: np.ndarray, k: int = KNN_K,
) -> np.ndarray:
    """GP-style RBF-weighted kNN-5 residual regression in latent space.

    For each query: pick k nearest bank rows by latent L2, weight each by
    exp(-d^2 / (2 h^2)) with h = median of the k neighbor distances (median
    heuristic; floored). residual_hat = sum(w*r) / sum(w).
    """
    nq = z_query.shape[0]
    nb = z_bank.shape[0]
    kk = min(k, nb)
    out = np.zeros(nq, dtype=np.float64)
    # bank norms for squared-distance expansion
    bank_sq = (z_bank * z_bank).sum(axis=1)            # (nb,)
    for i in range(nq):
        q = z_query[i]
        d2 = bank_sq + (q * q).sum() - 2.0 * (z_bank @ q)
        d2 = np.maximum(d2, 0.0)
        if kk >= nb:
            nn_idx = np.argsort(d2)[:kk]
        else:
            part = np.argpartition(d2, kk - 1)[:kk]
            nn_idx = part[np.argsort(d2[part])]
        d2_nn = d2[nn_idx]
        d_nn = np.sqrt(d2_nn)
        h = np.median(d_nn)
        if not np.isfinite(h) or h < 1e-6:
            h = max(float(d_nn.max()), 1e-6)
        w = np.exp(-d2_nn / (2.0 * h * h))
        wsum = w.sum()
        if wsum < SIM_FLOOR:
            out[i] = float(r_bank[nn_idx].mean())
        else:
            out[i] = float(np.sum(w * r_bank[nn_idx]) / wsum)
    return out


def run_one_seed(
    X_unb: np.ndarray, residual: np.ndarray, anchor_oof: np.ndarray,
    y_unb: np.ndarray, unb_scaffolds, kf_seed: int,
) -> dict:
    """One kf_seed: 5-fold scaffold cross-fit deep-kernel residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    n = len(y_unb)
    oof_pred = np.full(n, np.nan, dtype=np.float64)
    resid_oof = np.full(n, np.nan, dtype=np.float64)
    for fi, (tr_loc, va_loc) in enumerate(splits):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_unb[tr_loc]).astype(np.float32)
        X_va = sc.transform(X_unb[va_loc]).astype(np.float32)
        r_tr = residual[tr_loc]
        enc = train_encoder(X_tr, r_tr, seed=ENC_SEED_BASE + kf_seed + fi)
        z_tr = _encode(enc, X_tr)
        z_va = _encode(enc, X_va)
        r_hat = rbf_knn_residual(z_tr, r_tr, z_va, k=KNN_K)
        resid_oof[va_loc] = r_hat
        oof_pred[va_loc] = anchor_oof[va_loc] + r_hat
    if np.isnan(oof_pred).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits left NaNs")
    pooled = float(rae(y_unb, oof_pred))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "oof": oof_pred,
        "resid_oof": resid_oof,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Deep kernel (NN encoder + RBF-kNN latent head) on "
          f"{PARENT_TAG} residual")
    print(f"          arch: Linear({TOP_K_FEAT},{HIDDEN_DIM}) -> ReLU -> "
          f"Linear({HIDDEN_DIM},{LATENT_DIM}) latent")
    print(f"          head: latent kNN-{KNN_K} RBF-weighted residual regression")
    print(f"          train: contrastive metric loss, {N_EPOCHS} epochs "
          f"Adam lr={LR}")
    print(f"          kf_seeds = {KF_SEEDS}  ({len(KF_SEEDS)} seeds)")
    print(f"          gate: per-fold-mean < {GATE_BETTER} -> BETTER")
    print("=" * 78)

    # ---- Load test, truth, anchor ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        te_smiles = te["smiles"].astype(str).tolist()
    else:
        te_smiles = te["SMILES"].astype(str).tolist()
    if "Molecule Name" in te.columns:
        te_names = te["Molecule Name"].astype(str).tolist()
    elif "molecule_name" in te.columns:
        te_names = te["molecule_name"].astype(str).tolist()
    elif "name" in te.columns:
        te_names = te["name"].astype(str).tolist()
    else:
        raise KeyError("no name column on test set")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_oof = np.load(NB3090_OOF_PATH).astype(np.float64)   # (253,)
    anchor_te = np.load(NB3090_TE_PATH).astype(np.float64)     # (513,)
    if anchor_oof.shape != (n_unb,):
        raise ValueError(f"nb3090 oof shape {anchor_oof.shape} != ({n_unb},)")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"nb3090 te shape {anchor_te.shape} != ({n_test},)")
    rae_anchor_oof = float(rae(y_unb, anchor_oof))
    rae_anchor_te_unb = float(rae(y_unb, anchor_te[unb_idx]))
    residual = y_unb - anchor_oof
    print(f"[anchor] nb3090 oof RAE      = {rae_anchor_oof:.4f}  "
          f"(ref {REF_NB3090:.4f})")
    print(f"[anchor] nb3090 te[unb] RAE  = {rae_anchor_te_unb:.4f}")
    print(f"[resid]  mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Build K=20 SHAP features on 513 + unb slice ----
    print("\n" + "-" * 78)
    print(f"STEP 1: rebuild 117-col matrix, slice SHAP top-{TOP_K_FEAT}")
    print("-" * 78)
    X_te_K20, X_unb_K20, top20_idx = build_feature_matrices(
        te_smiles, unb_idx, n_test)
    print(f"   X_unb_K20={X_unb_K20.shape}  X_te_K20={X_te_K20.shape}")

    # ---- Scaffolds for outer CV ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # ---- 15-seed cross-fit ----
    print("\n" + "-" * 78)
    print(f"CROSS-FIT: {len(KF_SEEDS)} seeds x {N_FOLDS}-fold scaffold")
    print("-" * 78)
    pooled_raes = []
    oof_stack = []
    resid_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = run_one_seed(
            X_unb_K20, residual, anchor_oof, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        resid_stack.append(res["resid_oof"])
        print(f"   kf_seed={s}:  pooled_rae={res['pooled_rae']:.4f}  "
              f"resid_std={res['resid_oof'].std():.4f}  "
              f"wall={time.time()-ts:.1f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    per_fold_mean = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    min_rae, max_rae = float(arr.min()), float(arr.max())
    print(f"\n   per-fold-mean (15 seeds) = {per_fold_mean:.4f}")
    print(f"   std={std_rae:.4f}  min={min_rae:.4f}  max={max_rae:.4f}")
    print(f"   ref nb3090 = {REF_NB3090:.4f}  gate = {GATE_BETTER}")
    print(f"   delta vs nb3090 = {per_fold_mean - REF_NB3090:+.4f}")
    print(f"   delta vs gate   = {per_fold_mean - GATE_BETTER:+.4f}")

    # median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[len(arr) // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)

    # ---- Deploy te: encoder on ALL 253, RBF-kNN over 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY te (encoder on ALL 253; RBF-kNN-5 residual over 513)")
    print("-" * 78)
    sc_full = StandardScaler()
    X_unb_std = sc_full.fit_transform(X_unb_K20).astype(np.float32)
    X_te_std = sc_full.transform(X_te_K20).astype(np.float32)
    enc_full = train_encoder(X_unb_std, residual, seed=ENC_SEED_BASE)
    z_bank = _encode(enc_full, X_unb_std)
    z_te = _encode(enc_full, X_te_std)
    resid_te = rbf_knn_residual(z_bank, residual, z_te, k=KNN_K)
    te_pred = (anchor_te + resid_te).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   resid_te mean={resid_te.mean():+.4f}  std={resid_te.std():.4f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}  "
          f"(in-sample optimism vs cross-fit {per_fold_mean:.4f})")

    # ---- Gate ----
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if per_fold_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE candidate. nb3283 deep-kernel (NN encoder + RBF-kNN-5 "
            f"latent residual on {PARENT_TAG}) 15-seed per-fold-mean "
            f"{per_fold_mean:.4f} beats gate {GATE_BETTER} "
            f"({per_fold_mean - GATE_BETTER:+.4f}) and the nb3090 anchor "
            f"{REF_NB3090:.4f} ({per_fold_mean - REF_NB3090:+.4f}). "
            "Learned-metric neighborhood extracted residual signal the "
            "post-hoc-blend ceiling could not. Recommend deep-30 re-verify "
            "before deploy (cycle-160 rule)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3283 deep-kernel 15-seed per-fold-mean "
            f"{per_fold_mean:.4f} does not beat gate {GATE_BETTER} "
            f"({per_fold_mean - GATE_BETTER:+.4f}); delta vs nb3090 anchor "
            f"{per_fold_mean - REF_NB3090:+.4f}. A learned RBF-kNN metric on "
            f"the {PARENT_TAG} residual does not break the post-hoc-blend "
            "ceiling -- residual neighborhood in K=20 SHAP latent carries no "
            "exploitable structure at n=253. Keep nb3090 / prior PRIMARY-1."
        )
    print(f"   verdict       = {verdict}")
    print(f"   per_fold_mean = {per_fold_mean:.4f}")
    print(f"   ladder action = {ladder_action}")

    # ---- Save ----
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_deep_kernel.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": ("deep_kernel_nn_encoder_rbf_knn5_latent_residual_on_"
                   f"{PARENT_TAG}"),
        "paradigm": "deep_kernel_learning_metric_latent_rbf_knn",
        "anchor": PARENT_TAG,
        "anchor_pre_unblind": True,
        "anchor_oof_rae": round(rae_anchor_oof, 4),
        "anchor_te_unb_in_rae": round(rae_anchor_te_unb, 4),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "feature_K": TOP_K_FEAT,
        "top20_idx_in_117": top20_idx.tolist(),
        "encoder_arch": (f"Linear({TOP_K_FEAT},{HIDDEN_DIM})->ReLU->"
                         f"Linear({HIDDEN_DIM},{LATENT_DIM})"),
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "optimizer": "Adam",
        "loss": "pairwise_metric_align_resid_diff_plus_spread_reg",
        "spread_reg": SPREAD_REG,
        "knn_k": KNN_K,
        "rbf_bandwidth": "median_heuristic_per_query",
        "standardize_within_fold": True,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "enc_seed_base": ENC_SEED_BASE,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "pooled_raes": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean": round(per_fold_mean, 4),
        "std_rae": round(std_rae, 4),
        "min_rae": round(min_rae, 4),
        "max_rae": round(max_rae, 4),
        "median_seed": int(median_seed),
        "delta_vs_nb3090": round(per_fold_mean - REF_NB3090, 4),
        "delta_vs_gate": round(per_fold_mean - GATE_BETTER, 4),
        "gate_better": GATE_BETTER,
        "ref_nb3090": REF_NB3090,
        "ref_nb3080": REF_NB3080,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "resid_te_mean": float(resid_te.mean()),
        "resid_te_std": float(resid_te.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per_fold_mean (15 seeds) = {per_fold_mean:.4f}")
    print(f"   delta vs nb3090          = {per_fold_mean - REF_NB3090:+.4f}")
    print(f"   delta vs gate {GATE_BETTER}   = {per_fold_mean - GATE_BETTER:+.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "anchor_oof_rae", "per_fold_mean", "std_rae", "min_rae", "max_rae",
        "delta_vs_nb3090", "delta_vs_gate", "te_unb_in_sample_rae",
        "median_seed", "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
