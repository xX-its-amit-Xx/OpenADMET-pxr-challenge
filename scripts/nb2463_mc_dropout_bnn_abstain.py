"""nb2463 -- MC-Dropout BNN with epistemic abstention on chemprop_aux fallback.

HYPOTHESIS:
    The cycle-160+ "ceiling cluster" (nb2171/nb2095/nb2240 at scaffold-CV
    RAE 0.4682-0.4720) is bounded by the chemprop_aux anchor's failure on
    novel-scaffold rows. A Bayesian neural net trained on the chemprop_aux
    residual gives an epistemic uncertainty estimate sigma (via 50 MC dropout
    forward passes). On rows where sigma is large (the model has no idea),
    we abstain from the residual correction and FALL BACK to the
    chemprop_aux raw prediction; on confident rows we KEEP the
    nb2240 + mu prediction. Sweep tau_epi over five sigma quantiles to find
    the operating point.

PROTOCOL:
    1. Load anchors: te_nb2240_K20.npy (513) and nb2240_mean_bag_oof_K20.npy
       (253), y_unb.npy (253). chemprop_aux fallback from te_chemprop_aux.npy.
    2. Build X_117 feature matrix (5-way SHAP-top + ChEMBL kNN), cache to
       data/processed/pyramid/{X_117_unb,X_117_te}.npy for reuse.
    3. Target = residual = y_unb - nb2240_mean_bag_oof_K20.
    4. 3-layer MLP with Dropout(0.3): 117 -> 256 -> 128 -> 1
       Train Adam lr=1e-3, MSE, 200 epochs, batch_size=32, scaffold-5-fold CV.
    5. INFERENCE with dropout ON (train mode), 50 MC forward passes
       per held-out row. mu = mean of passes, sigma = std of passes.
    6. Sweep tau_epi in {p10, p25, p50, p75, p90} of sigma distribution.
       For each tau:
            keep_mask = (sigma < tau)
            pred = nb2240[i] + mu[i]            if keep_mask[i]
                 = chemprop_aux[i]              otherwise (abstain)
       Compute RAE on 253 unblind.
    7. Build deploy te_nb2463.npy on the 513 (best-tau application).

GATE:
    best tau_epi mean_rae < 0.4570 -> PROMOTE. Else FAIL.
    (Beats nb2240 reference 0.4598 by 0.0028.)

OUTPUTS:
    scripts/nb2463_mc_dropout_bnn_abstain.py
    data/processed/nb2463_summary.json
    data/processed/nb2463_pred_oof.npy   (253,) float32 best-tau pred on unb
    data/processed/te_nb2463.npy         (513,) float32 best-tau deploy
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
from torch.utils.data import DataLoader, TensorDataset

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2463"

# ------------------------------ paths ---------------------------------------
ANCHOR_TE_CHEMPROP = DATA_PROCESSED / "te_chemprop_aux.npy"
TE_NB2240_K20      = DATA_PROCESSED / "te_nb2240_K20.npy"
OOF_NB2240_K20     = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
UNB_IDX_PATH       = DATA_PROCESSED / "_audit_unblind_idx.npy"
Y_UNB_PATH         = DATA_PROCESSED / "_audit_unblind_y.npy"

PYRAMID_DIR        = DATA_PROCESSED / "pyramid"
X_117_UNB_PATH     = PYRAMID_DIR / "X_117_unb.npy"
X_117_TE_PATH      = PYRAMID_DIR / "X_117_te.npy"

# Family caches (used only to rebuild X_117 if not cached)
ATOMPAIR_TE_PATH        = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH           = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH  = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH          = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR             = Path("C:/pxr_artifacts/nb1030")

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

# ------------------------------ MC-Dropout BNN hyperparams ------------------
DROPOUT_P     = 0.3
HIDDEN_1      = 256
HIDDEN_2      = 128
N_EPOCHS      = 200
BATCH_SIZE    = 32
LR            = 1e-3
N_FOLDS       = 5
N_MC_PASSES   = 50
TRAIN_SEED    = 42      # torch seed for reproducibility
CV_SEED       = 1001    # scaffold-CV seed
TAU_QUANTILES = [10, 25, 50, 75, 90]
GATE_THRESH   = 0.4570
NB2240_REF    = 0.4598


# ============================================================================
# helpers for X_117 rebuild (only used if cache missing)
# ============================================================================

def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
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
    pool["std_smiles"] = mols.apply(lambda m: Chem.MolToSmiles(m) if m is not None else None)
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


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    return np.where(np.isfinite(X), X, 0.0).astype(np.float32)


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
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


def _extract_atompair_top_idx_from_nb1484(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def build_X_117_te(n_test, te_smiles):
    """Rebuild the 117-col 5-way + ChEMBL kNN feature matrix on the 513 test."""
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

    X_ap_te    = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te  = _load_mordred_test(n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te   = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te    = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
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
    std_test_smiles = []
    for m in test_mols:
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
         pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
         mean_sim.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"feat_dim {X_te_full.shape[1]} != 117"
    return X_te_full


# ============================================================================
# MC-Dropout BNN
# ============================================================================

class MCDropoutBNN(nn.Module):
    def __init__(self, in_dim: int = 117, h1: int = HIDDEN_1, h2: int = HIDDEN_2, p: float = DROPOUT_P):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.drop1 = nn.Dropout(p)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(p)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(h2, 1)

    def forward(self, x):
        # Spec order: Linear -> Dropout -> ReLU
        x = self.relu1(self.drop1(self.fc1(x)))
        x = self.relu2(self.drop2(self.fc2(x)))
        return self.out(x).squeeze(-1)


def train_one_fold(X_tr, y_tr, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MCDropoutBNN(in_dim=X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    X_t = torch.from_numpy(X_tr).float()
    y_t = torch.from_numpy(y_tr).float()
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=False)
    model.train()
    for epoch in range(N_EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
    return model


def mc_predict(model, X, n_passes: int = N_MC_PASSES):
    """Run n_passes forward passes with dropout ACTIVE. Return (mu, sigma)."""
    model.train()  # keep dropout on
    X_t = torch.from_numpy(X).float()
    preds = np.zeros((n_passes, len(X)), dtype=np.float32)
    with torch.no_grad():
        for p in range(n_passes):
            preds[p] = model(X_t).cpu().numpy()
    return preds.mean(axis=0), preds.std(axis=0)


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- MC-Dropout BNN with epistemic abstention")
    print("=" * 78)

    # ---- load anchors / truth ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(Y_UNB_PATH).astype(np.float64)
    n_unb = len(y_unb)

    te_nb2240 = np.load(TE_NB2240_K20).astype(np.float64)
    oof_nb2240 = np.load(OOF_NB2240_K20).astype(np.float64)
    te_chemprop = np.load(ANCHOR_TE_CHEMPROP).astype(np.float64)
    assert te_nb2240.shape == (n_test,) and te_chemprop.shape == (n_test,)
    assert oof_nb2240.shape == (n_unb,)

    rae_chemprop_unb = float(rae(y_unb, te_chemprop[unb_idx]))
    rae_nb2240_unb_oof = float(rae(y_unb, oof_nb2240))
    rae_nb2240_unb_te = float(rae(y_unb, te_nb2240[unb_idx]))
    print(f"[anchors] chemprop_aux[unb] RAE = {rae_chemprop_unb:.4f}")
    print(f"[anchors] nb2240 OOF RAE        = {rae_nb2240_unb_oof:.4f}")
    print(f"[anchors] nb2240 te[unb] RAE    = {rae_nb2240_unb_te:.4f} (in-sample deploy)")

    # residual target on unb
    residual = (y_unb - oof_nb2240).astype(np.float32)
    print(f"[target] residual mean={residual.mean():+.4f} std={residual.std():.4f}")

    # ---- X_117 (cache or rebuild) ----
    PYRAMID_DIR.mkdir(parents=True, exist_ok=True)
    if X_117_UNB_PATH.exists() and X_117_TE_PATH.exists():
        X_117_te = np.load(X_117_TE_PATH).astype(np.float32)
        X_117_unb = np.load(X_117_UNB_PATH).astype(np.float32)
        assert X_117_te.shape == (n_test, 117), f"cached X_117_te {X_117_te.shape}"
        assert X_117_unb.shape == (n_unb, 117), f"cached X_117_unb {X_117_unb.shape}"
        print(f"[feat] loaded cached X_117_te {X_117_te.shape}, X_117_unb {X_117_unb.shape}")
    else:
        print(f"[feat] cache miss -- rebuilding 117-col matrix (one-shot)")
        X_117_te = build_X_117_te(n_test, te_smiles)
        X_117_unb = X_117_te[unb_idx].astype(np.float32)
        np.save(X_117_TE_PATH, X_117_te)
        np.save(X_117_UNB_PATH, X_117_unb)
        print(f"[feat] saved {X_117_TE_PATH}  shape={X_117_te.shape}")
        print(f"[feat] saved {X_117_UNB_PATH} shape={X_117_unb.shape}")

    # ---- StandardScaler-style normalize per column (fit on unb only for CV honesty) ----
    # We fit scaler INSIDE each CV fold below; outer pass uses unb-level scaler for deploy.

    # ---- scaffold-CV ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                    shuffle=True, seed=CV_SEED)
    print(f"[cv] scaffold {N_FOLDS}-fold splits built  (seed={CV_SEED})")

    print("\n" + "-" * 78)
    print(f"TRAIN BNN ({HIDDEN_1}->{HIDDEN_2}, dropout={DROPOUT_P}) "
          f"{N_EPOCHS} epochs, {N_MC_PASSES} MC passes")
    print("-" * 78)

    mu_oof = np.full(n_unb, np.nan, dtype=np.float64)
    sigma_oof = np.full(n_unb, np.nan, dtype=np.float64)

    for fold, (tr_loc, va_loc) in enumerate(splits):
        ts = time.time()
        X_tr_raw = X_117_unb[tr_loc].astype(np.float32)
        X_va_raw = X_117_unb[va_loc].astype(np.float32)
        y_tr = residual[tr_loc].astype(np.float32)

        # per-fold StandardScaler (fit on train ONLY)
        mu_f  = X_tr_raw.mean(axis=0)
        std_f = X_tr_raw.std(axis=0) + 1e-6
        X_tr  = (X_tr_raw - mu_f) / std_f
        X_va  = (X_va_raw - mu_f) / std_f

        model = train_one_fold(X_tr, y_tr, seed=TRAIN_SEED + fold)
        mu_va, sigma_va = mc_predict(model, X_va, n_passes=N_MC_PASSES)
        mu_oof[va_loc] = mu_va.astype(np.float64)
        sigma_oof[va_loc] = sigma_va.astype(np.float64)
        print(f"   fold {fold+1}/{N_FOLDS}  n_tr={len(tr_loc):3d}  n_va={len(va_loc):3d}  "
              f"mu_mean={mu_va.mean():+.4f}  sigma_mean={sigma_va.mean():.4f}  "
              f"wall={time.time()-ts:.1f}s")

    # ---- sweep tau_epi (quantiles of OOF sigma distribution) ----
    print("\n" + "-" * 78)
    print(f"TAU SWEEP  quantiles={TAU_QUANTILES}")
    print("-" * 78)
    tau_results = []
    pred_keep_no_abstain = (oof_nb2240 + mu_oof).astype(np.float64)
    rae_no_abstain = float(rae(y_unb, pred_keep_no_abstain))
    print(f"   baseline  (no abstain)  RAE = {rae_no_abstain:.4f}")

    chemprop_unb_fallback = te_chemprop[unb_idx]
    sigmas_for_quantile = sigma_oof  # all 253 rows

    best_tau_q = None
    best_rae = float("inf")
    best_pred = None
    best_keep = None
    best_tau_val = None

    for q in TAU_QUANTILES:
        tau = float(np.percentile(sigmas_for_quantile, q))
        keep_mask = sigmas_for_quantile < tau   # tau-strict abstain on >=
        pred = np.where(keep_mask, pred_keep_no_abstain, chemprop_unb_fallback)
        r = float(rae(y_unb, pred))
        n_keep = int(keep_mask.sum())
        n_abst = int((~keep_mask).sum())
        tau_results.append({
            "q": q,
            "tau": tau,
            "n_keep": n_keep,
            "n_abstain": n_abst,
            "rae": r,
        })
        print(f"   q={q:3d}  tau={tau:.4f}  keep={n_keep:3d}  abstain={n_abst:3d}  RAE={r:.4f}")
        if r < best_rae:
            best_rae = r
            best_tau_q = q
            best_tau_val = tau
            best_pred = pred
            best_keep = keep_mask

    # ---- always evaluate q=100 (no abstain edge case) for diagnostics ----
    print(f"\n[best] tau_quantile={best_tau_q}  tau_val={best_tau_val:.4f}  "
          f"keep={int(best_keep.sum())}  RAE={best_rae:.4f}")

    # ---- gate ----
    promote = best_rae < GATE_THRESH
    verdict = "PROMOTE" if promote else "FAIL"
    print("\n" + "-" * 78)
    print(f"GATE  threshold={GATE_THRESH:.4f}  nb2240_ref={NB2240_REF:.4f}")
    print("-" * 78)
    print(f"   best_rae = {best_rae:.4f}  -> {verdict}")

    # ---- deploy te: same abstain rule applied on the 513 ----
    # Use a model trained on ALL 253 unblind rows (deploy refit) to get MC sigma
    # on the 513 test set.
    print("\n" + "-" * 78)
    print("DEPLOY: refit on all 253 unb -> MC predict on 513 test")
    print("-" * 78)
    mu_unb_all = X_117_unb.mean(axis=0)
    std_unb_all = X_117_unb.std(axis=0) + 1e-6
    X_unb_norm = (X_117_unb - mu_unb_all) / std_unb_all
    X_te_norm = (X_117_te - mu_unb_all) / std_unb_all
    deploy_model = train_one_fold(X_unb_norm.astype(np.float32),
                                  residual.astype(np.float32),
                                  seed=TRAIN_SEED)
    mu_te, sigma_te = mc_predict(deploy_model, X_te_norm.astype(np.float32),
                                 n_passes=N_MC_PASSES)
    pred_keep_te = (te_nb2240 + mu_te).astype(np.float64)

    # ABSTAIN: apply tau threshold from BEST OOF quantile but RECOMPUTE
    # against the 513-row sigma distribution for honesty.
    tau_te = float(np.percentile(sigma_te, best_tau_q))
    keep_mask_te = sigma_te < tau_te
    deploy_te = np.where(keep_mask_te, pred_keep_te, te_chemprop).astype(np.float32)
    print(f"   te sigma range  [{sigma_te.min():.4f}, {sigma_te.max():.4f}]  "
          f"mean={sigma_te.mean():.4f}")
    print(f"   tau_te (q={best_tau_q})  = {tau_te:.4f}")
    print(f"   keep={int(keep_mask_te.sum())}/513  abstain={int((~keep_mask_te).sum())}/513")
    print(f"   deploy_te mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   te[unb_idx] RAE (in-sample) = {te_unb_rae:.4f}")

    # ---- save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, best_pred.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {pred_oof_path}  shape={best_pred.shape}")
    print(f"[save] {te_path}  shape={deploy_te.shape}")

    summary = {
        "tag": TAG,
        "method": "mc_dropout_bnn_epistemic_abstain",
        "arch": {
            "in_dim": 117, "h1": HIDDEN_1, "h2": HIDDEN_2,
            "dropout_p": DROPOUT_P, "order": "Linear->Dropout->ReLU",
        },
        "training": {
            "optimizer": "Adam", "lr": LR, "loss": "MSE",
            "epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "n_folds": N_FOLDS, "cv_seed": CV_SEED, "train_seed": TRAIN_SEED,
        },
        "inference": {
            "n_mc_passes": N_MC_PASSES,
            "dropout_active_at_inference": True,
        },
        "anchors": {
            "te_nb2240_K20": str(TE_NB2240_K20),
            "oof_nb2240_K20": str(OOF_NB2240_K20),
            "te_chemprop_aux": str(ANCHOR_TE_CHEMPROP),
        },
        "anchor_raes": {
            "chemprop_aux_unb": rae_chemprop_unb,
            "nb2240_oof_unb": rae_nb2240_unb_oof,
            "nb2240_te_unb_insample": rae_nb2240_unb_te,
        },
        "features": {
            "x_117_unb_path": str(X_117_UNB_PATH),
            "x_117_te_path": str(X_117_TE_PATH),
            "feat_dim": 117,
        },
        "oof_diagnostics": {
            "mu_mean": float(np.nanmean(mu_oof)),
            "mu_std": float(np.nanstd(mu_oof)),
            "sigma_mean": float(np.nanmean(sigma_oof)),
            "sigma_std": float(np.nanstd(sigma_oof)),
            "sigma_min": float(np.nanmin(sigma_oof)),
            "sigma_max": float(np.nanmax(sigma_oof)),
            "rae_no_abstain": rae_no_abstain,
        },
        "tau_sweep": tau_results,
        "best_tau_quantile": best_tau_q,
        "best_tau_value": best_tau_val,
        "best_n_keep": int(best_keep.sum()),
        "best_n_abstain": int((~best_keep).sum()),
        "best_rae": best_rae,
        "gate_threshold": GATE_THRESH,
        "nb2240_ref_rae": NB2240_REF,
        "delta_vs_nb2240": best_rae - NB2240_REF,
        "verdict": verdict,
        "promote": bool(promote),
        "deploy": {
            "tau_te": tau_te,
            "n_keep_te": int(keep_mask_te.sum()),
            "n_abstain_te": int((~keep_mask_te).sum()),
            "te_mean": float(deploy_te.mean()),
            "te_std": float(deploy_te.std()),
            "te_unb_rae_insample": te_unb_rae,
            "sigma_te_mean": float(sigma_te.mean()),
            "sigma_te_std": float(sigma_te.std()),
        },
        "outputs": {
            "pred_oof_npy": str(pred_oof_path),
            "te_npy": str(te_path),
        },
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   best tau quantile  = {best_tau_q}  (tau={best_tau_val:.4f})")
    print(f"   best RAE           = {best_rae:.4f}")
    print(f"   delta vs nb2240    = {best_rae - NB2240_REF:+.4f}")
    print(f"   gate threshold     = {GATE_THRESH:.4f}")
    print(f"   verdict            = {verdict}")
    print(f"   wall               = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_tau_quantile", "best_tau_value", "best_n_keep", "best_n_abstain",
        "best_rae", "delta_vs_nb2240", "verdict", "promote",
    ):
        print(f"  {k}: {res.get(k)}")
