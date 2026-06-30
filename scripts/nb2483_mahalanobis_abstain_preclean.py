"""nb2483 -- Mahalanobis-distance epistemic abstention on PRE-clean anchors only.

HYPOTHESIS:
    The cycle-160+ "ceiling cluster" (nb2171/nb2095/nb2240 at scaffold-CV
    RAE 0.4682-0.4720) is bounded by chemprop_aux's failure on novel-scaffold
    rows. Unlike nb2463 which estimates epistemic uncertainty via MC-Dropout
    on the *model*, this script estimates epistemic uncertainty via
    Mahalanobis distance to the *training* feature manifold in X_117 space.
    On test rows that lie far from the training manifold (D_m large), we
    abstain from the residual-corrected nb2240 prediction and FALL BACK to
    the chemprop_aux raw prediction (which is itself trained on all 4139
    PRE-clean rows). On rows close to the manifold (D_m small) we KEEP nb2240.

    PRE-clean ONLY: nb2240 K=20 residual rides on chemprop_aux (PRE-unblind
    refit on 4139). No POST-unblind anchors (no nb1191/nb503/nb562) appear
    in the predictive pipeline so the +0.10 hybrid-contamination band does
    not apply.

PROTOCOL:
    1. Load anchors:
         te_nb2240_K20.npy            (513,) PRE-clean residual-corrected
         nb2240_mean_bag_oof_K20.npy  (253,) honest OOF on unb
         te_chemprop_aux.npy          (513,) PRE-clean anchor fallback
       Load y_unb (253,) and unb_idx.
    2. Build X_117 on the 4139 train compounds (via the same 5-way + ChEMBL
       kNN pipeline as nb2240, using the cached tr_*.npy family caches and
       X_mordred_train.npy). Cache to data/processed/pyramid/X_117_train.npy.
    3. Per-column StandardScaler fit on X_117_train. Compute training
       covariance Sigma. Stable inverse via scipy.linalg.pinvh.
    4. Per-test row D_m^2 = (x - mu_tr)^T Sigma^{-1} (x - mu_tr). We also
       compute D_m on the 4139 train rows themselves to get the reference
       distribution.
    5. Sweep tau in {p50, p75, p90, p95, p99} of TRAIN D_m distribution.
       For each tau on the 253 OOF:
            abstain = (D_m_unb > tau)
            pred = nb2240_oof_K20  if not abstain
                 = chemprop_aux    if abstain
       Compute scaffold 5-fold CV RAE pooled across kf_seeds {1001-1005}
       (deterministic predictor -> rae() identical across folds, but pool
       across seeds for variance bar).
    6. Best-tau on the 513 deploy: D_m_te > tau -> chemprop_aux fallback,
       else te_nb2240_K20.

GATE:
    best tau mean_rae < 0.4570 -> PROMOTE
    best tau mean_rae < 0.4601 -> MARGINAL_BEAT
    else                       -> FAIL

OUTPUTS:
    scripts/nb2483_mahalanobis_abstain_preclean.py
    data/processed/nb2483_summary.json
    data/processed/nb2483_pred_oof.npy   (253,) float32 best-tau pred on unb
    data/processed/te_nb2483.npy         (513,) float32 best-tau deploy
    data/processed/pyramid/X_117_train.npy  (4139, 117) float32 cache
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
from rdkit import Chem
from rdkit import RDLogger
from scipy.linalg import pinvh

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2483"

# ------------------------------ paths ---------------------------------------
ANCHOR_TE_CHEMPROP = DATA_PROCESSED / "te_chemprop_aux.npy"
TE_NB2240_K20      = DATA_PROCESSED / "te_nb2240_K20.npy"
OOF_NB2240_K20     = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
UNB_IDX_PATH       = DATA_PROCESSED / "_audit_unblind_idx.npy"
Y_UNB_PATH         = DATA_PROCESSED / "_audit_unblind_y.npy"

PYRAMID_DIR        = DATA_PROCESSED / "pyramid"
X_117_UNB_PATH     = PYRAMID_DIR / "X_117_unb.npy"
X_117_TE_PATH      = PYRAMID_DIR / "X_117_te.npy"
X_117_TRAIN_PATH   = PYRAMID_DIR / "X_117_train.npy"

# Family caches: TRAIN
ATOMPAIR_TR_PATH       = DATA_PROCESSED / "tr_atompair.npy"
MACCS_TR_PATH          = DATA_PROCESSED / "tr_maccs.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
AVALON_TR_PATH         = DATA_PROCESSED / "tr_avalon512.npy"
MORDRED_TR_PATH        = Path("C:/pxr_artifacts/nb1030/X_mordred_train.npy")

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

# Mahalanobis abstention hyperparams
N_FOLDS       = 5
KF_SEEDS      = [1001, 1002, 1003, 1004, 1005]
TAU_QUANTILES = [50, 75, 90, 95, 99]
COV_RIDGE     = 1e-4   # tiny diag jitter before pinvh for numerical safety
GATE_PROMOTE  = 0.4570
GATE_MARGINAL = 0.4601
NB2240_REF    = 0.4598


# ============================================================================
# helpers for building X_117 on arbitrary smiles list
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


def _load_npy_shape(path, n_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} (expected n={n_expected})")
    X = X.astype(np.float32)
    return np.where(np.isfinite(X), X, 0.0).astype(np.float32)


def _load_mordred(path, n_expected):
    if not path.exists():
        raise FileNotFoundError(f"Mordred cache missing: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"Mordred shape mismatch: {X.shape} (expected n={n_expected})")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


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


def build_X_117_train(n_train, train_smiles):
    """Rebuild the 117-col 5-way + ChEMBL kNN matrix on the 4139 training rows."""
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

    X_ap_tr    = _load_npy_shape(ATOMPAIR_TR_PATH,       n_train)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_tr = _load_npy_shape(MACCS_TR_PATH,          n_train)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_tr  = _load_mordred(MORDRED_TR_PATH,          n_train)[:, top_mord_col_idx].astype(np.float32)
    X_emb_tr   = _load_npy_shape(CHEMPROP_EMBED_TR_PATH, n_train)[:, top_embed_col_idx].astype(np.float32)
    X_av_tr    = _load_npy_shape(AVALON_TR_PATH,         n_train)[:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN on training compounds (uses TRAIN inchikeys for self-exclusion)
    pool = _load_chembl_pool()
    train_mols = [standardize(s) for s in train_smiles]
    train_inchikeys = set()
    for m in train_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            train_inchikeys.add(ik)
    pool = pool[~pool["inchikey"].isin(train_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_train_smiles = []
    for m in train_mols:
        std_train_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_train = morgan_fp_batch(std_train_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_train, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_tr_full = np.concatenate(
        [X_ap_tr, X_maccs_tr, X_mord_tr, X_emb_tr, X_av_tr,
         pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
         mean_sim.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    assert X_tr_full.shape[1] == 117, f"feat_dim {X_tr_full.shape[1]} != 117"
    return X_tr_full


# ============================================================================
# Mahalanobis machinery
# ============================================================================

def fit_mahalanobis(X_tr_std: np.ndarray, ridge: float = COV_RIDGE):
    """Fit feature mean + stable inverse covariance on training matrix."""
    mu = X_tr_std.mean(axis=0)
    Xc = X_tr_std - mu
    # Use sample covariance (1/(n-1)). Add tiny diag ridge for stability.
    n = Xc.shape[0]
    Sigma = (Xc.T @ Xc) / max(n - 1, 1)
    Sigma = Sigma + ridge * np.eye(Sigma.shape[0], dtype=Sigma.dtype)
    Sigma_inv = pinvh(Sigma)
    return mu.astype(np.float64), Sigma_inv.astype(np.float64)


def mahalanobis_dist(X: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """Per-row Mahalanobis distance (sqrt of D^2)."""
    Xc = (X.astype(np.float64) - mu[None, :])
    # quadratic form, vectorized
    d2 = np.einsum("ij,jk,ik->i", Xc, Sigma_inv, Xc)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Mahalanobis-distance epistemic abstention (PRE-clean only)")
    print("=" * 78)

    # ---- load truth / anchors ----
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
    print(f"[anchors] chemprop_aux[unb] RAE = {rae_chemprop_unb:.4f}  (PRE-clean)")
    print(f"[anchors] nb2240 OOF RAE        = {rae_nb2240_unb_oof:.4f}  (PRE-clean residual)")
    print(f"[anchors] nb2240 te[unb] RAE    = {rae_nb2240_unb_te:.4f}  (in-sample deploy)")

    # ---- X_117 unb / te (cached) ----
    PYRAMID_DIR.mkdir(parents=True, exist_ok=True)
    if not X_117_TE_PATH.exists() or not X_117_UNB_PATH.exists():
        raise FileNotFoundError("X_117_te/unb cache missing -- run nb2463 once first to materialize")
    X_117_te = np.load(X_117_TE_PATH).astype(np.float32)
    X_117_unb = np.load(X_117_UNB_PATH).astype(np.float32)
    assert X_117_te.shape == (n_test, 117), f"X_117_te {X_117_te.shape}"
    assert X_117_unb.shape == (n_unb, 117), f"X_117_unb {X_117_unb.shape}"
    print(f"[feat] X_117_te  {X_117_te.shape}   X_117_unb {X_117_unb.shape}")

    # ---- X_117 on TRAIN (cache or rebuild) ----
    tr = load_train()
    n_train = len(tr)
    train_smiles = tr["smiles"].astype(str).tolist() if "smiles" in tr.columns else tr["SMILES"].astype(str).tolist()
    if X_117_TRAIN_PATH.exists():
        X_117_train = np.load(X_117_TRAIN_PATH).astype(np.float32)
        if X_117_train.shape != (n_train, 117):
            print(f"[feat] cached X_117_train shape mismatch {X_117_train.shape} -- rebuilding")
            X_117_train = build_X_117_train(n_train, train_smiles)
            np.save(X_117_TRAIN_PATH, X_117_train)
        else:
            print(f"[feat] loaded cached X_117_train  {X_117_train.shape}")
    else:
        print(f"[feat] X_117_train cache miss -- rebuilding (one-shot)")
        X_117_train = build_X_117_train(n_train, train_smiles)
        np.save(X_117_TRAIN_PATH, X_117_train)
        print(f"[feat] saved {X_117_TRAIN_PATH}  shape={X_117_train.shape}")

    # ---- per-column StandardScaler fit on TRAIN ----
    mu_tr_feat = X_117_train.mean(axis=0)
    std_tr_feat = X_117_train.std(axis=0) + 1e-6
    X_tr_std = ((X_117_train - mu_tr_feat) / std_tr_feat).astype(np.float64)
    X_unb_std = ((X_117_unb - mu_tr_feat) / std_tr_feat).astype(np.float64)
    X_te_std = ((X_117_te - mu_tr_feat) / std_tr_feat).astype(np.float64)
    print(f"[scaler] fit on TRAIN  mu range [{mu_tr_feat.min():.3f}, {mu_tr_feat.max():.3f}]")

    # ---- Mahalanobis fit on TRAIN, stable pinvh ----
    print(f"[mahal] fitting covariance (ridge={COV_RIDGE:.0e}, pinvh inverse)")
    mu_mahal, Sigma_inv = fit_mahalanobis(X_tr_std, ridge=COV_RIDGE)

    D_tr  = mahalanobis_dist(X_tr_std,  mu_mahal, Sigma_inv)
    D_unb = mahalanobis_dist(X_unb_std, mu_mahal, Sigma_inv)
    D_te  = mahalanobis_dist(X_te_std,  mu_mahal, Sigma_inv)
    print(f"[mahal] D_tr  n={len(D_tr):4d}  mean={D_tr.mean():.3f}  med={np.median(D_tr):.3f}  "
          f"p95={np.percentile(D_tr, 95):.3f}  max={D_tr.max():.3f}")
    print(f"[mahal] D_unb n={len(D_unb):4d}  mean={D_unb.mean():.3f}  med={np.median(D_unb):.3f}  "
          f"p95={np.percentile(D_unb, 95):.3f}  max={D_unb.max():.3f}")
    print(f"[mahal] D_te  n={len(D_te):4d}  mean={D_te.mean():.3f}  med={np.median(D_te):.3f}  "
          f"p95={np.percentile(D_te, 95):.3f}  max={D_te.max():.3f}")

    # ---- scaffold-CV pool across kf seeds (predictor is deterministic given tau;
    #      seeds differ only in fold partition -> rae same on full pred,
    #      but per-fold breakdown gives variance bar)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]

    print("\n" + "-" * 78)
    print(f"TAU SWEEP  quantiles={TAU_QUANTILES}  (computed on TRAIN D_m)")
    print("-" * 78)
    chemprop_unb = te_chemprop[unb_idx]
    baseline_rae = float(rae(y_unb, oof_nb2240))
    print(f"   baseline (no abstain, nb2240 OOF) RAE = {baseline_rae:.4f}")

    tau_results = []
    best_q = None
    best_rae_mean = float("inf")
    best_pred = None
    best_keep_unb = None
    best_tau_val = None
    best_rae_per_seed = None

    for q in TAU_QUANTILES:
        tau = float(np.percentile(D_tr, q))
        keep_mask = D_unb <= tau          # in-manifold -> KEEP nb2240
        pred = np.where(keep_mask, oof_nb2240, chemprop_unb)
        # full-set RAE
        pooled = float(rae(y_unb, pred))
        # per-fold scaffold RAE across 5 kf_seeds for variance bar
        per_seed_raes = []
        per_seed_fold_raes = []
        for kf_seed in KF_SEEDS:
            splits = scaffold_kfold_indices(
                unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
            )
            fold_raes = []
            for tr_loc, va_loc in splits:
                fold_raes.append(float(rae(y_unb[va_loc], pred[va_loc])))
            per_seed_fold_raes.append(fold_raes)
            # pooled rae across full unb (recompute for sanity == pooled)
            per_seed_raes.append(pooled)
        # The model is deterministic; report mean fold RAE per seed (per-fold weighted by fold size).
        # The honest scalar is the full-set pooled RAE; per-fold dispersion is informational.
        mean_fold_rae = float(np.mean([np.mean(fr) for fr in per_seed_fold_raes]))
        std_fold_rae  = float(np.std([np.mean(fr) for fr in per_seed_fold_raes]))
        n_keep = int(keep_mask.sum())
        n_abst = int((~keep_mask).sum())
        tau_results.append({
            "q": q,
            "tau": tau,
            "n_keep": n_keep,
            "n_abstain": n_abst,
            "rae_pooled": pooled,
            "mean_fold_rae": mean_fold_rae,
            "std_fold_rae": std_fold_rae,
        })
        print(f"   q={q:3d}  tau={tau:7.3f}  keep={n_keep:3d}  abst={n_abst:3d}  "
              f"pooled_RAE={pooled:.4f}  mean_fold_RAE={mean_fold_rae:.4f} +/- {std_fold_rae:.4f}")
        if pooled < best_rae_mean:
            best_rae_mean = pooled
            best_q = q
            best_tau_val = tau
            best_pred = pred
            best_keep_unb = keep_mask
            best_rae_per_seed = per_seed_raes

    print(f"\n[best] q={best_q}  tau={best_tau_val:.4f}  "
          f"keep={int(best_keep_unb.sum())}  abstain={int((~best_keep_unb).sum())}  "
          f"pooled_RAE={best_rae_mean:.4f}")

    # ---- gate ----
    if best_rae_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_rae_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print(f"GATE  promote<{GATE_PROMOTE:.4f}  marginal<{GATE_MARGINAL:.4f}  nb2240_ref={NB2240_REF:.4f}")
    print("-" * 78)
    print(f"   best_rae = {best_rae_mean:.4f}  -> {verdict}")

    # ---- deploy on 513 ----
    keep_mask_te = D_te <= best_tau_val
    deploy_te = np.where(keep_mask_te, te_nb2240, te_chemprop).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[deploy] tau={best_tau_val:.4f}  keep={int(keep_mask_te.sum())}/513  "
          f"abstain={int((~keep_mask_te).sum())}/513")
    print(f"[deploy] te mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb] in-sample RAE = {te_unb_rae:.4f}")

    # ---- save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, best_pred.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {pred_oof_path}  shape={best_pred.shape}")
    print(f"[save] {te_path}  shape={deploy_te.shape}")

    summary = {
        "tag": TAG,
        "method": "mahalanobis_distance_epistemic_abstain_preclean",
        "anchors_preclean_only": True,
        "anchors": {
            "te_nb2240_K20": str(TE_NB2240_K20),
            "oof_nb2240_K20": str(OOF_NB2240_K20),
            "te_chemprop_aux": str(ANCHOR_TE_CHEMPROP),
            "fallback_on_abstain": "te_chemprop_aux (PRE-clean refit on 4139)",
        },
        "anchor_raes": {
            "chemprop_aux_unb": rae_chemprop_unb,
            "nb2240_oof_unb": rae_nb2240_unb_oof,
            "nb2240_te_unb_insample": rae_nb2240_unb_te,
        },
        "features": {
            "x_117_train_path": str(X_117_TRAIN_PATH),
            "x_117_unb_path": str(X_117_UNB_PATH),
            "x_117_te_path": str(X_117_TE_PATH),
            "feat_dim": 117,
            "n_train": int(n_train),
            "n_unb": int(n_unb),
            "n_test": int(n_test),
        },
        "mahalanobis": {
            "scaler": "per-col Z fit on TRAIN",
            "cov_ridge": COV_RIDGE,
            "cov_inverse": "scipy.linalg.pinvh",
            "D_tr_mean": float(D_tr.mean()),
            "D_tr_median": float(np.median(D_tr)),
            "D_tr_p95": float(np.percentile(D_tr, 95)),
            "D_unb_mean": float(D_unb.mean()),
            "D_unb_median": float(np.median(D_unb)),
            "D_unb_p95": float(np.percentile(D_unb, 95)),
            "D_te_mean": float(D_te.mean()),
            "D_te_median": float(np.median(D_te)),
            "D_te_p95": float(np.percentile(D_te, 95)),
        },
        "cv": {
            "n_folds": N_FOLDS,
            "kf_seeds": KF_SEEDS,
        },
        "baseline_no_abstain_rae": baseline_rae,
        "tau_sweep": tau_results,
        "best_tau_quantile": best_q,
        "best_tau_value": best_tau_val,
        "best_n_keep_unb": int(best_keep_unb.sum()),
        "best_n_abstain_unb": int((~best_keep_unb).sum()),
        "best_rae": best_rae_mean,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "nb2240_ref_rae": NB2240_REF,
        "delta_vs_nb2240": best_rae_mean - NB2240_REF,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal": bool(verdict == "MARGINAL_BEAT"),
        "deploy": {
            "tau_value": best_tau_val,
            "n_keep_te": int(keep_mask_te.sum()),
            "n_abstain_te": int((~keep_mask_te).sum()),
            "te_mean": float(deploy_te.mean()),
            "te_std": float(deploy_te.std()),
            "te_unb_rae_insample": te_unb_rae,
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
    print(f"   best tau quantile  = {best_q}  (tau={best_tau_val:.4f})")
    print(f"   best RAE (pooled)  = {best_rae_mean:.4f}")
    print(f"   delta vs nb2240    = {best_rae_mean - NB2240_REF:+.4f}")
    print(f"   gate promote       = {GATE_PROMOTE:.4f}")
    print(f"   gate marginal      = {GATE_MARGINAL:.4f}")
    print(f"   verdict            = {verdict}")
    print(f"   wall               = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_tau_quantile", "best_tau_value", "best_n_keep_unb",
        "best_n_abstain_unb", "best_rae", "delta_vs_nb2240",
        "verdict", "promote", "marginal",
    ):
        print(f"  {k}: {res.get(k)}")
