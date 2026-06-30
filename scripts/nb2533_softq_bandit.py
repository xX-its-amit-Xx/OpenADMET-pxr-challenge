"""nb2533 -- Soft Q-learning bandit for per-row anchor selection.

NEW PARADIGM (substrate change, not operator change):
A small Q-network ingests row features and learns soft Q-values over 2 actions:
  action 0 -> use nb2240 prediction
  action 1 -> use chemprop_aux prediction

Soft Q-learning with entropy regularization:
  Q(s,a) = mlp(s, a)
  pi(a|s) = softmax(Q(s,a)/tau) with tau=1.0
  reward r(s,a) = -|y_truth - pred_a|
  target Q* = r(s,a) (bandit -- one step, no bootstrap)
  loss = MSE(Q(s,a), target) + entropy_bonus * -H[pi]
    -> we want HIGH entropy, so loss subtracts entropy_bonus*H

Inference:
  pred = sum_a pi(a|s) * pred_a   (expected prediction under soft policy)

Outer eval:
  5-fold scaffold CV on the 253 unblind, 5 kf_seeds {1001..1005}.
  Per fold-train: train Q-net 200 epochs, Adam lr=1e-3.
  Per fold-val: compute pred from policy weights -> OOF.
  Pool 5 seeds (mean over per-seed pooled OOFs).

State features: X_117 (rebuilt 5-way K-tuned feature matrix used by nb2240/nb2523).

GATES (mean over 5 kf_seeds of pooled scaffold-CV RAE):
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4601  -> MARGINAL_BEAT
    else                -> FAIL

OUTPUTS:
    data/processed/nb2533_summary.json
    data/processed/nb2533_pred_oof.npy   (253,) float32  mean-bag OOF
    data/processed/te_nb2533.npy          (513,) float32  deploy policy expected pred
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2533"

# ---- anchor sources ----
NB2240_TE_PATH = DATA_PROCESSED / "te_nb2240.npy"
CHEMPROP_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
CHEMPROP_AUX_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"

# ---- 117-col feature substrate ----
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
KNN_K = 5
SIM_FLOOR = 1e-6

# ---- bandit / Q-net hyperparams ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_ACTIONS = 2
HIDDEN = 64
EPOCHS = 200
LR = 1e-3
ENTROPY_BONUS = 0.1
WEIGHT_DECAY = 1e-4
TAU = 1.0

# ---- gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

DEVICE = torch.device("cpu")


# ============================================================================
# helpers (mirror nb2240 / nb2523 substrate build)
# ============================================================================
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
    return agg.rename(columns={"src_first": "src"})


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


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X_te_m.shape}")
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
    return np.where(np.isfinite(X), X, 0.0).astype(np.float32)


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


def build_X_117(test_smiles: list, n_test: int) -> np.ndarray:
    """Rebuild full 117-col 5-way K-tuned feature matrix on the 513 test compounds.
    Identical recipe to nb2240/nb2523."""
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    sum_1352 = json.loads(NB1352_SUMMARY.read_text())
    sum_1392 = json.loads(NB1392_SUMMARY.read_text())
    sum_1484 = json.loads(NB1484_SUMMARY.read_text())
    sum_1523 = json.loads(NB1523_SUMMARY.read_text())
    sum_1524 = json.loads(NB1524_SUMMARY.read_text())
    sum_1541 = json.loads(NB1541_SUMMARY.read_text())

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", best_K_key="best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN aux features
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = {ik for m in test_mols if (ik := _safe_inchikey(m)) is not None}
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te,
            X_maccs_te,
            X_mord_te,
            X_emb_te,
            X_av_te,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full


# ============================================================================
# Q-network
# ============================================================================
class QNet(nn.Module):
    """Q(s,a) head. Takes state features and outputs one Q-value per action."""

    def __init__(self, in_dim: int, hidden: int = HIDDEN, n_actions: int = N_ACTIONS):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)  # (N, n_actions)


def train_qnet(X_tr: np.ndarray, y_tr: np.ndarray, preds_tr: np.ndarray,
               n_epochs: int = EPOCHS, lr: float = LR, seed: int = 0) -> QNet:
    """Train soft-Q bandit.
    X_tr: (N, D) state features (standardized)
    y_tr: (N,) truth
    preds_tr: (N, n_actions) predictions per action
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = X_tr.shape[1]
    net = QNet(in_dim).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    x = torch.from_numpy(X_tr.astype(np.float32)).to(DEVICE)
    y = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    preds = torch.from_numpy(preds_tr.astype(np.float32)).to(DEVICE)  # (N, A)
    # bandit reward: r(s,a) = -|y - pred_a|
    rewards = -(y.unsqueeze(1) - preds).abs()  # (N, A)

    for ep in range(n_epochs):
        opt.zero_grad()
        q = net(x)  # (N, A)
        # Q-fit loss: regress Q toward bandit reward (one-step, no bootstrap)
        q_loss = F.mse_loss(q, rewards)
        # Entropy of soft policy pi = softmax(Q/tau)
        pi = F.softmax(q / TAU, dim=-1)
        log_pi = F.log_softmax(q / TAU, dim=-1)
        entropy = -(pi * log_pi).sum(dim=-1).mean()
        # Total loss: Q-fit MINUS entropy_bonus * entropy (we want HIGH entropy)
        loss = q_loss - ENTROPY_BONUS * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    return net


def predict_expected(net: QNet, X: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Expected prediction under soft policy: sum_a pi(a|s) * pred_a."""
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        p = torch.from_numpy(preds.astype(np.float32)).to(DEVICE)
        q = net(x)
        pi = F.softmax(q / TAU, dim=-1)  # (N, A)
        out = (pi * p).sum(dim=-1).cpu().numpy()
    return out


def predict_policy(net: QNet, X: np.ndarray) -> np.ndarray:
    """Return policy probabilities pi(a|s) for diagnostics."""
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        q = net(x)
        pi = F.softmax(q / TAU, dim=-1).cpu().numpy()
    return pi


# ============================================================================
# CV runner
# ============================================================================
def cv_one_seed(X_unb: np.ndarray, y_unb: np.ndarray, preds_unb: np.ndarray,
                unb_scaffolds: list, kf_seed: int) -> tuple:
    """5-fold scaffold CV with per-fold scaler-fit + Q-net train.
    Returns (pooled_rae, oof_pred, mean_pi_a0)."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(y_unb)
    oof = np.full(n, np.nan, dtype=np.float64)
    pi_a0_va = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mu = X_unb[tr_loc].mean(axis=0, keepdims=True)
        sd = X_unb[tr_loc].std(axis=0, keepdims=True) + 1e-6
        Xtr = (X_unb[tr_loc] - mu) / sd
        Xva = (X_unb[va_loc] - mu) / sd
        net = train_qnet(Xtr, y_unb[tr_loc], preds_unb[tr_loc], seed=kf_seed)
        oof[va_loc] = predict_expected(net, Xva, preds_unb[va_loc])
        pi_va = predict_policy(net, Xva)
        pi_a0_va[va_loc] = pi_va[:, 0]
    return float(rae(y_unb, oof)), oof, float(np.nanmean(pi_a0_va))


def deploy_predict(X_unb: np.ndarray, y_unb: np.ndarray, preds_unb: np.ndarray,
                   X_te: np.ndarray, preds_te: np.ndarray, seed: int = 1001) -> np.ndarray:
    """Refit Q-net on full 253 -> expected pred on 513."""
    mu = X_unb.mean(axis=0, keepdims=True)
    sd = X_unb.std(axis=0, keepdims=True) + 1e-6
    Xtr = (X_unb - mu) / sd
    Xte = (X_te - mu) / sd
    net = train_qnet(Xtr, y_unb, preds_unb, seed=seed)
    return predict_expected(net, Xte, preds_te)


# ============================================================================
# MAIN
# ============================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Soft Q-learning bandit for per-row anchor selection")
    print(f"        actions = (0: nb2240, 1: chemprop_aux)")
    print(f"        state   = X_117 (5-way K-tuned)")
    print(f"        epochs={EPOCHS}  lr={LR}  entropy_bonus={ENTROPY_BONUS}  tau={TAU}")
    print(f"        kf_seeds = {KF_SEEDS}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE; <{GATE_MARGINAL} MARGINAL_BEAT")
    print("=" * 78)

    # ---- load truth + anchors ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in unb 253 = {n_unique_scaf}")

    # action 0: nb2240
    te_nb2240 = np.load(NB2240_TE_PATH).astype(np.float64)
    # action 1: chemprop_aux
    te_chemprop = np.load(CHEMPROP_AUX_TE_PATH).astype(np.float64)
    assert te_nb2240.shape == (n_test,) and te_chemprop.shape == (n_test,)

    # On the 253, prefer honest OOFs for fold-train signal:
    if NB2240_OOF_PATH.exists():
        nb2240_oof = np.load(NB2240_OOF_PATH).astype(np.float64)
        if nb2240_oof.shape != (n_unb,):
            print(f"[warn] nb2240 OOF shape mismatch {nb2240_oof.shape}; falling back to te[unb_idx]")
            nb2240_oof = te_nb2240[unb_idx]
    else:
        nb2240_oof = te_nb2240[unb_idx]
    if CHEMPROP_AUX_OOF_PATH.exists():
        chemprop_oof = np.load(CHEMPROP_AUX_OOF_PATH).astype(np.float64)
        if chemprop_oof.shape != (n_unb,):
            print(f"[warn] chemprop_aux OOF shape mismatch {chemprop_oof.shape}; falling back")
            chemprop_oof = te_chemprop[unb_idx]
    else:
        chemprop_oof = te_chemprop[unb_idx]

    preds_unb = np.column_stack([nb2240_oof, chemprop_oof])  # (253, 2)
    preds_te = np.column_stack([te_nb2240, te_chemprop])      # (513, 2)
    rae_nb2240 = float(rae(y_unb, nb2240_oof))
    rae_chemprop = float(rae(y_unb, chemprop_oof))
    print(f"[anchors] nb2240        oof_RAE = {rae_nb2240:.4f}")
    print(f"[anchors] chemprop_aux  oof_RAE = {rae_chemprop:.4f}")

    # ---- build X_117 state ----
    print("\n[build] X_117 (5-way K-tuned) on 513 test compounds...")
    ts = time.time()
    X_te_117 = build_X_117(test_smiles, n_test)
    X_unb_117 = X_te_117[unb_idx]
    print(f"[build] X_te_117 = {X_te_117.shape}  X_unb_117 = {X_unb_117.shape}  "
          f"wall={time.time() - ts:.1f}s")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"SOFT-Q BANDIT 5-FOLD SCAFFOLD CV  seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed = []
    per_seed_oof = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    for i, kf_seed in enumerate(KF_SEEDS):
        ts = time.time()
        pooled, oof_s, mean_pi_a0 = cv_one_seed(
            X_unb_117, y_unb, preds_unb, unb_scaffolds, kf_seed
        )
        per_seed_oof[i] = oof_s
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "mean_pi_a0_nb2240": float(mean_pi_a0),
        })
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"mean_pi(nb2240)={mean_pi_a0:.3f}  wall={time.time() - ts:.1f}s")

    per_seed_rae = np.asarray([r["pooled_rae"] for r in per_seed])
    per_seed_mean = float(per_seed_rae.mean())
    per_seed_std = float(per_seed_rae.std())
    mean_bag_oof = per_seed_oof.mean(axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    print(f"\n[cv] per_seed_mean RAE = {per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean-bag-of-seed-OOFs RAE = {rae_mean_bag:.4f}")

    # ---- deploy: refit on full 253 with first seed -> 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit Q-net on all 253, predict 513)")
    print("-" * 78)
    deploy_seeds = KF_SEEDS  # use same seed bag for deploy ensemble
    deploy_preds = np.zeros((len(deploy_seeds), n_test), dtype=np.float64)
    for i, s in enumerate(deploy_seeds):
        deploy_preds[i] = deploy_predict(
            X_unb_117, y_unb, preds_unb, X_te_117, preds_te, seed=s
        )
    te_deploy = deploy_preds.mean(axis=0).astype(np.float32)
    te_unb_in_sample = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"   te(513) mean/std = {te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in_sample:.4f}  "
          f"(deploy refit; in-sample optimism expected)")

    # ---- save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_bag_oof.astype(np.float32))
    np.save(te_path, te_deploy)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    # ---- gate ----
    mean_rae = per_seed_mean  # gate on per-seed mean (matches spec)
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae (5-seed mean of pooled OOFs) = {mean_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f} (PROMOTE)        = {mean_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f} (MARGINAL_BEAT) = {mean_rae < GATE_MARGINAL}")
    print(f"   VERDICT                               = {verdict}")

    summary = {
        "tag": TAG,
        "method": "soft_Q_bandit_per_row_anchor_selection",
        "actions": ["nb2240", "chemprop_aux"],
        "state": "X_117_5way_K_tuned",
        "n_actions": N_ACTIONS,
        "hidden": HIDDEN,
        "epochs": EPOCHS,
        "lr": LR,
        "entropy_bonus": ENTROPY_BONUS,
        "weight_decay": WEIGHT_DECAY,
        "tau": TAU,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "anchor_oof_rae": {
            "nb2240": rae_nb2240,
            "chemprop_aux": rae_chemprop,
        },
        "per_seed_results": per_seed,
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_rae": mean_rae,
        "mean_bag_rae": rae_mean_bag,
        "te_unb_in_sample_rae": te_unb_in_sample,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"\nwall = {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "mean_rae",
        "anchor_oof_rae",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
