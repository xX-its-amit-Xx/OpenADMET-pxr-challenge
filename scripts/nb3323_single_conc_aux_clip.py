"""nb3323 -- Single-concentration screen log2FC as a K=18+1 aux feature + clip.

NEW PARADIGM (substrate change, cycle 169+):
    Cycles 134/136/139/169 closed every operator-axis on the chemprop_aux
    residual K=18 RFE pyramid; the only open lever is SUBSTRATE change. This
    script tries an ORTHOGONAL EXPERIMENTAL feature: the single-concentration
    primary screen log2 fold-change (pxr-challenge_single_concentration_TRAIN,
    21,003 rows / 10,870 unique compounds) is a cheap, noisy readout of PXR
    transcriptional activation. It overlaps a large fraction of the 4,139 CRC
    train compounds but has ZERO test overlap, so the raw value cannot be used
    on the 513 directly. We therefore:

        1. Aggregate per-compound MEAN log2FC over the SP replicates.
        2. Train a chemistry -> mean_log2FC LGBM on the CRC-train subset that
           HAS SP data; predict mean_log2FC on ALL 513 test compounds
           (purely from structure -- no SP label needed at inference).
        3. Append that 1 predicted column to the canonical K=18 SHAP/RFE
           feature set (nb2604 k18_idx_in_117col on the 117-col 5-way matrix:
           AtomPair/MACCS/Mordred/ChempropEmbed/Avalon + ChEMBL-kNN + mean_sim)
           -> K = 18 + 1 = 19 columns.
        4. Honest residual cross-fit on the 253 unblind: residual learner
           (shallow LGBM) on (y_unb - chemprop_aux_oof), 5-fold, with the K=19
           feature matrix; predict-corrected = anchor_oof + resid_oof; CLIP
           corrected prediction to the train pEC50 range +/- 0.5.
        5. 15 fresh kf_seeds {1216..1230}, PER-FOLD-mean across all 75 folds.

    Distinct from the K=18 deep-30 deploy bag (nb2960/nb3114): this is an
    HONEST 253-only residual cross-fit (LB-faithful, like nb1241), and it adds
    a genuinely orthogonal EXPERIMENTAL axis (SP transcriptional fold-change)
    rather than chemistry-only features. If SP carries even ~0.1 RAE of
    pEC50-relevant signal, the residual learner picks it up.

GATE:
    per-fold-mean RAE < 0.4423  ->  "BETTER"
    else                        ->  "FAIL"

References (honest cross-fit / deep-30, chemprop_aux anchor):
    chemprop_aux anchor (nb1133 OOF on 253) ~ 0.6216
    nb2960 K18 deep-30 (no aug)             = 0.4536
    nb2171 PRIMARY-1 post-hoc-blend ceiling = 0.4682

Outputs:
    data/processed/nb3323_summary.json
    data/processed/nb3323_pred_oof.npy           (253,) float32  per-fold mean-bag corrected OOF
    data/processed/te_nb3323.npy                 (513,) float32  deploy-refit corrected test preds
    data/processed/nb3323_pred_mean_log2fc_513.npy (513,) float32
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
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, standardize_smiles, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test, load_single_conc
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3323"

# ---- cached feature paths (117-col 5-way layout, identical to nb3114) ----
ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
MORDRED_TRAIN_PATH = MORDRED_DIR / "X_mordred_train.npy"
MORDRED_TEST_PATH = MORDRED_DIR / "X_mordred_test.npy"

# ---- family-slice summaries ----
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"   # K=18 idx in 117-col

# ---- anchor (chemprop_aux) ----
ANCHOR_OOF_253 = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"  # (253,)
ANCHOR_TE_513 = DATA_PROCESSED / "te_chemprop_aux.npy"               # (513,)

# ---- ChEMBL kNN external pool (col 115 of the 117-layout is in K18) ----
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# ---- residual cross-fit protocol ----
KF_SEEDS = list(range(1216, 1231))   # 15 fresh seeds {1216..1230}
N_FOLDS = 5
SP_PRED_SEED = 42
SP_PRED_FOLDS = 5

# ---- gate ----
GATE_BETTER = 0.4423

# ---- references ----
CHEMPROP_AUX_REF = 0.6216
NB2960_K18_REF = 0.4536
NB2171_REF = 0.4682


# ============================================================================
# helpers
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


def _load_npy(path, n_expected, name):
    if not path.exists():
        raise FileNotFoundError(f"missing cache {name}: {path}")
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {name} {path}: {X.shape}, expected n={n_expected}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred(path, n_expected, name):
    if not path.exists():
        raise FileNotFoundError(f"Mordred cache missing {name}: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"Mordred shape mismatch {name}: {X.shape}, expected n={n_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
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


def _sp_predictor_params(seed: int) -> dict:
    """SP mean_log2FC predictor: regular LGBM (deeper)."""
    return dict(
        objective="regression_l1",
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _resid_params(seed: int) -> dict:
    """Shallow LGBM Huber residual learner (nb1183/nb1241 family capacity)."""
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


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- single-conc log2FC aux feature (K=18+1) on chemprop_aux residual + clip")
    print(f"       seeds {KF_SEEDS[0]}..{KF_SEEDS[-1]} ({len(KF_SEEDS)})  x {N_FOLDS}-fold  per-fold-mean")
    print(f"       gate: per-fold-mean < {GATE_BETTER} BETTER / else FAIL")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Truth / anchor / K=18 idx
    # ------------------------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist()
    te_names = te["name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_scaffolds = [bemis_murcko(te_smiles[i]) for i in unb_idx]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] unique_scaffolds(unb)={n_unique_scaf}")

    anchor_oof = np.load(ANCHOR_OOF_253).astype(np.float64)   # (253,)
    assert anchor_oof.shape == (n_unb,), f"anchor_oof {anchor_oof.shape}"
    rae_anchor = float(rae(y_unb, anchor_oof))
    anchor_te_513 = np.load(ANCHOR_TE_513).astype(np.float64)  # (513,)
    print(f"[load] chemprop_aux anchor OOF(253) RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")

    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    K18_idx = np.array(nb2604["k18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"[load] K=18 idx (n={len(K18_idx)}): {sorted(K18_idx.tolist())}")

    # ------------------------------------------------------------------
    # 2. Build 117-col feature matrix on train (4139) + test (513)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: build 117-col 5-way feature matrix (train + test)")
    print("-" * 78)

    tr = load_train()
    tr = tr.dropna(subset=["smiles", "pec50"]).copy()
    tr_smiles = tr["smiles"].astype(str).tolist()
    n_tr = len(tr)
    print(f"[train] n={n_tr}")

    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f: sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f: sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f: sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f: sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f: sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f: sum_1541 = json.load(f)

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", best_K_key="best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    # train + test cached families
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_tr, "ap_tr")
    X_maccs_tr = _load_npy(MACCS_TR_PATH, n_tr, "maccs_tr")
    X_av_tr = _load_npy(AVALON_TR_PATH, n_tr, "av_tr")
    X_emb_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_tr, "embed_tr")
    X_mord_tr = _load_mordred(MORDRED_TRAIN_PATH, n_tr, "mordred_tr")
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test, "ap_te")
    X_maccs_te = _load_npy(MACCS_TE_PATH, n_test, "maccs_te")
    X_av_te = _load_npy(AVALON_TE_PATH, n_test, "av_te")
    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test, "embed_te")
    X_mord_te = _load_mordred(MORDRED_TEST_PATH, n_test, "mordred_te")

    X_ap_tr_top = X_ap_tr[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_tr_top = X_maccs_tr[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_tr_top = X_mord_tr[:, top_mord_col_idx].astype(np.float32)
    X_emb_tr_top = X_emb_tr[:, top_embed_col_idx].astype(np.float32)
    X_av_tr_top = X_av_tr[:, top_avalon_bit_idx].astype(np.float32)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN (col 115 of the 117 layout is in K18 -> required)
    print("[chembl] loading external PXR pool for kNN feature (col 115)...")
    test_mols = [standardize(s) for s in te_smiles]
    test_iks = set(_safe_inchikey(m) for m in test_mols if m is not None)
    test_iks.discard(None)
    train_mols = [standardize(s) for s in tr_smiles]

    pool = _load_chembl_pool()
    pool = pool[~pool["inchikey"].isin(test_iks)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"[chembl] pool n={len(pool)}")

    fp_test = morgan_fp_batch([_safe_can_smiles(m) or "" for m in test_mols])
    top_idx_te, top_sim_te = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(top_idx_te, top_sim_te, pool_labels, pool_median)
    fp_train = morgan_fp_batch([_safe_can_smiles(m) or "" for m in train_mols])
    top_idx_tr, top_sim_tr = _tanimoto_topk(fp_train, fp_pool, k=KNN_K)
    pred_chembl_tr, mean_sim_tr = _knn_predict(top_idx_tr, top_sim_tr, pool_labels, pool_median)

    X_te_full = np.concatenate(
        [X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
         pred_chembl_te.reshape(-1, 1).astype(np.float32),
         mean_sim_te.reshape(-1, 1).astype(np.float32)], axis=1).astype(np.float32)
    X_tr_full = np.concatenate(
        [X_ap_tr_top, X_maccs_tr_top, X_mord_tr_top, X_emb_tr_top, X_av_tr_top,
         pred_chembl_tr.reshape(-1, 1).astype(np.float32),
         mean_sim_tr.reshape(-1, 1).astype(np.float32)], axis=1).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"X_te_full {X_te_full.shape}"
    assert X_tr_full.shape[1] == 117, f"X_tr_full {X_tr_full.shape}"
    print(f"[feat] X_tr_full{X_tr_full.shape}  X_te_full{X_te_full.shape}")

    X_tr_K18 = X_tr_full[:, K18_idx].astype(np.float32)
    X_te_K18 = X_te_full[:, K18_idx].astype(np.float32)
    print(f"[K18] X_tr_K18{X_tr_K18.shape}  X_te_K18{X_te_K18.shape}")

    # ------------------------------------------------------------------
    # 3. Single-conc aux feature: per-compound MEAN log2FC, predicted on 513
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: single-conc MEAN log2FC -> chemistry predictor -> 513 column")
    print("-" * 78)
    sc = load_single_conc()
    sc_uni = sc["smiles"].drop_duplicates().to_frame()
    sc_uni["std"] = sc_uni["smiles"].map(standardize_smiles)
    sc = sc.merge(sc_uni, on="smiles", how="left")
    sc = sc[sc["std"].notna()].copy()
    sp_agg = (
        sc.groupby("std")
        .agg(mean_log2fc=("log2_fc_estimate", "mean"))
        .reset_index()
    )
    print(f"[sp ] unique SP compounds aggregated: {len(sp_agg)}")
    print(f"[sp ] mean_log2fc: mean={sp_agg['mean_log2fc'].mean():+.3f}  "
          f"std={sp_agg['mean_log2fc'].std():.3f}  "
          f"range=[{sp_agg['mean_log2fc'].min():+.3f}, {sp_agg['mean_log2fc'].max():+.3f}]")

    tr["std"] = tr["smiles"].map(standardize_smiles)
    tr_pos = tr.reset_index(drop=True)
    tr_with_sp = tr_pos.merge(sp_agg, on="std", how="inner")
    sp_train_idx = tr_with_sp.index.to_numpy()  # positional into tr_pos (== cache order)
    # NOTE: merge preserves left positional index only if left index is RangeIndex.
    # tr_pos is reset_index(drop=True) so tr_with_sp.index aligns to merge output,
    # not tr_pos rows. Recover true positional index via std lookup instead.
    std_to_pos = {}
    for i, s in enumerate(tr_pos["std"].tolist()):
        if s is not None and s not in std_to_pos:
            std_to_pos[s] = i
    sp_train_pos = np.array([std_to_pos[s] for s in tr_with_sp["std"].tolist()], dtype=int)
    n_sp_train = len(tr_with_sp)
    print(f"[join] CRC train compounds with SP data: {n_sp_train}/{n_tr} "
          f"({100*n_sp_train/n_tr:.1f}%)")

    # use the K18 features themselves as the SP-predictor inputs (same substrate)
    X_sp_train = X_tr_K18[sp_train_pos]
    y_sp = tr_with_sp["mean_log2fc"].to_numpy(dtype=np.float64)

    kf_sp = KFold(n_splits=SP_PRED_FOLDS, shuffle=True, random_state=SP_PRED_SEED)
    oof_sp = np.full(n_sp_train, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf_sp.split(np.arange(n_sp_train)):
        mdl = lgb.LGBMRegressor(**_sp_predictor_params(SP_PRED_SEED))
        mdl.fit(X_sp_train[tr_loc], y_sp[tr_loc])
        oof_sp[va_loc] = mdl.predict(X_sp_train[va_loc])
    rae_sp_cv = float(rae(y_sp, oof_sp))
    mdl_dep = lgb.LGBMRegressor(**_sp_predictor_params(SP_PRED_SEED))
    mdl_dep.fit(X_sp_train, y_sp)
    pred_mean_log2fc_513 = mdl_dep.predict(X_te_K18).astype(np.float32)
    np.save(DATA_PROCESSED / f"{TAG}_pred_mean_log2fc_513.npy", pred_mean_log2fc_513)
    print(f"[sp ] SP predictor 5-fold CV RAE = {rae_sp_cv:.4f}  (train_std={y_sp.std():.3f})")
    print(f"[sp ] pred_mean_log2fc_513: mean={pred_mean_log2fc_513.mean():+.3f}  "
          f"std={pred_mean_log2fc_513.std():.3f}  "
          f"range=[{pred_mean_log2fc_513.min():+.3f}, {pred_mean_log2fc_513.max():+.3f}]")

    # ------------------------------------------------------------------
    # 4. Assemble K=18+1 matrix (append SP column), residual cross-fit (253)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: K=18+1 residual cross-fit on 253 unblind (15 seeds x 5 folds)")
    print("-" * 78)
    X_te_K19 = np.concatenate(
        [X_te_K18, pred_mean_log2fc_513.reshape(-1, 1)], axis=1).astype(np.float32)
    X_unb_K19 = X_te_K19[unb_idx]
    print(f"[K19] X_unb_K19{X_unb_K19.shape}  (18 SHAP + 1 SP)")

    # clip bounds from train pEC50 range +/- 0.5
    y_tr_all = tr["pec50"].astype(float).to_numpy()
    clip_lo = float(y_tr_all.min() - 0.5)
    clip_hi = float(y_tr_all.max() + 0.5)
    print(f"[clip] bounds = [{clip_lo:.3f}, {clip_hi:.3f}]")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    all_fold_rae = []          # per-fold RAE across all seeds x folds (75 values)
    per_seed_pooled_rae = []   # pooled (across folds) RAE per seed
    sum_corr_oof = np.zeros(n_unb, dtype=np.float64)  # mean-bag corrected OOF over seeds

    for si, seed in enumerate(KF_SEEDS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        resid_oof_s = np.full(n_unb, np.nan, dtype=np.float64)
        seed_fold_rae = []
        for tr_loc, va_loc in kf.split(np.arange(n_unb)):
            mdl = lgb.LGBMRegressor(**_resid_params(seed))
            mdl.fit(X_unb_K19[tr_loc], residual[tr_loc])
            r_hat = mdl.predict(X_unb_K19[va_loc])
            corr = np.clip(anchor_oof[va_loc] + r_hat, clip_lo, clip_hi)
            resid_oof_s[va_loc] = corr
            fr = float(rae(y_unb[va_loc], corr))
            seed_fold_rae.append(fr)
            all_fold_rae.append(fr)
        pooled = float(rae(y_unb, resid_oof_s))
        per_seed_pooled_rae.append(pooled)
        sum_corr_oof += resid_oof_s
        if (si % 5) == 0 or si == len(KF_SEEDS) - 1:
            print(f"   seed={seed}  pooled_RAE={pooled:.4f}  "
                  f"fold_mean={np.mean(seed_fold_rae):.4f}  ({si+1}/{len(KF_SEEDS)})")

    mean_bag_corr_oof = sum_corr_oof / len(KF_SEEDS)
    per_fold_mean = float(np.mean(all_fold_rae))     # PRIMARY gate metric
    per_fold_std = float(np.std(all_fold_rae, ddof=1))
    per_seed_pooled_mean = float(np.mean(per_seed_pooled_rae))
    per_seed_pooled_std = float(np.std(per_seed_pooled_rae, ddof=1))
    rae_mean_bag = float(rae(y_unb, mean_bag_corr_oof))

    print(f"\n[xfit] PER-FOLD-mean RAE = {per_fold_mean:.4f} +/- {per_fold_std:.4f}  "
          f"(n_folds={len(all_fold_rae)})")
    print(f"[xfit] per-seed pooled RAE = {per_seed_pooled_mean:.4f} +/- {per_seed_pooled_std:.4f}")
    print(f"[xfit] mean-bag-OOF pooled RAE = {rae_mean_bag:.4f}")
    print(f"[xfit] vs anchor ({rae_anchor:.4f}) delta = {per_fold_mean - rae_anchor:+.4f}")

    # ------------------------------------------------------------------
    # 5. Deploy refit: residual learner on ALL 253 -> corrected 513 test
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: deploy refit (residual on all 253) -> corrected 513 test")
    print("-" * 78)
    sum_te = np.zeros(n_test, dtype=np.float64)
    for seed in KF_SEEDS:
        mdl = lgb.LGBMRegressor(**_resid_params(seed))
        mdl.fit(X_unb_K19, residual)
        r_hat_te = mdl.predict(X_te_K19)
        sum_te += np.clip(anchor_te_513 + r_hat_te, clip_lo, clip_hi)
    te_corr = (sum_te / len(KF_SEEDS)).astype(np.float64)
    te_unb_in_rae = float(rae(y_unb, te_corr[unb_idx]))
    print(f"[deploy] te_corr mean={te_corr.mean():.3f}  std={te_corr.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_rae:.4f} "
          f"(in-sample optimism vs honest {per_fold_mean:.4f} EXPECTED)")

    # ------------------------------------------------------------------
    # 6. Gate
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 6: GATE")
    print("-" * 78)
    verdict = "BETTER" if per_fold_mean < GATE_BETTER else "FAIL"
    print(f"   per_fold_mean              = {per_fold_mean:.4f}")
    print(f"   gate (<{GATE_BETTER})            = {verdict}")
    print(f"   delta vs nb2960 K18 ({NB2960_K18_REF})  = {per_fold_mean - NB2960_K18_REF:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_REF})    = {per_fold_mean - NB2171_REF:+.4f}")

    # ------------------------------------------------------------------
    # 7. Save artifacts
    # ------------------------------------------------------------------
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_bag_corr_oof.astype(np.float32))
    np.save(te_path, te_corr.astype(np.float32))
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_single_conc_aux_clip.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_corr.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] verdict=FAIL; no submission CSV")

    summary = {
        "tag": TAG,
        "status": "OK",
        "method": "single_conc_mean_log2FC_aux_feature_K18+1_on_chemprop_aux_residual_clip",
        "paradigm": "substrate_change_orthogonal_experimental_feature",
        "anchor": "chemprop_aux",
        "anchor_pre_unblind": True,
        "anchor_oof_path": str(ANCHOR_OOF_253),
        "rae_anchor": rae_anchor,
        # SP feature
        "sp_unique_compounds": int(len(sp_agg)),
        "n_train_with_sp": int(n_sp_train),
        "sp_pred_cv_rae_mean_log2fc": rae_sp_cv,
        "pred_mean_log2fc_513_mean": float(pred_mean_log2fc_513.mean()),
        "pred_mean_log2fc_513_std": float(pred_mean_log2fc_513.std()),
        "pred_mean_log2fc_513_min": float(pred_mean_log2fc_513.min()),
        "pred_mean_log2fc_513_max": float(pred_mean_log2fc_513.max()),
        # features / model
        "K": 18,
        "K_plus_aux": 19,
        "K18_idx_in_117col": sorted(K18_idx.tolist()),
        "resid_params": _resid_params(0),
        "sp_predictor_params": _sp_predictor_params(0),
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "clip_lo": clip_lo,
        "clip_hi": clip_hi,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        # results
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_unique_unb_scaffolds": int(n_unique_scaf),
        "per_fold_rae_all": [float(x) for x in all_fold_rae],
        "per_fold_mean": per_fold_mean,
        "per_fold_std": per_fold_std,
        "mean_rae": per_fold_mean,   # alias for ladder consistency
        "per_seed_pooled_rae": per_seed_pooled_rae,
        "per_seed_pooled_mean": per_seed_pooled_mean,
        "per_seed_pooled_std": per_seed_pooled_std,
        "rae_mean_bag_oof": rae_mean_bag,
        "te_mean": float(te_corr.mean()),
        "te_std": float(te_corr.std()),
        "te_unb_in_sample_rae": te_unb_in_rae,
        # gate
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "delta_vs_anchor": per_fold_mean - rae_anchor,
        "delta_vs_nb2960_K18": per_fold_mean - NB2960_K18_REF,
        "delta_vs_nb2171": per_fold_mean - NB2171_REF,
        # refs
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2960_K18_ref": NB2960_K18_REF,
        "nb2171_ref": NB2171_REF,
        # paths
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   SP unique compounds        = {len(sp_agg)}")
    print(f"   n_train_with_sp            = {n_sp_train}")
    print(f"   SP predictor CV RAE        = {rae_sp_cv:.4f}")
    print(f"   anchor (chemprop_aux) RAE  = {rae_anchor:.4f}")
    print(f"   per-fold-mean RAE          = {per_fold_mean:.4f} +/- {per_fold_std:.4f}")
    print(f"   per-seed pooled RAE        = {per_seed_pooled_mean:.4f}")
    print(f"   mean-bag OOF RAE           = {rae_mean_bag:.4f}")
    print(f"   te[unb_idx] in-RAE         = {te_unb_in_rae:.4f}")
    print(f"   delta vs nb2960 K18        = {per_fold_mean - NB2960_K18_REF:+.4f}")
    print(f"   verdict (<{GATE_BETTER})         = {verdict}")
    print(f"   wall                       = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "sp_unique_compounds", "n_train_with_sp", "sp_pred_cv_rae_mean_log2fc",
        "rae_anchor", "per_fold_mean", "per_fold_std", "per_seed_pooled_mean",
        "rae_mean_bag_oof", "te_unb_in_sample_rae",
        "delta_vs_nb2960_K18", "delta_vs_nb2171", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
