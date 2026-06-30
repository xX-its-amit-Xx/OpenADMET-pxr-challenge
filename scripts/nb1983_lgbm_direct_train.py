"""nb1983 -- LGBM directly on FULL 4139 train labels with 117-col K-tuned matrix.

ORTHOGONAL APPROACH HYPOTHESIS:
    All prior nb18xx/nb19xx 5-way K-tuned variants train an LGBM **residual**
    on top of the chemprop_aux PRE-unblind anchor (in_RAE 0.6216), with only
    the 253 unblind compounds as labels.  This caps the achievable signal at
    chemprop_aux's failure modes.

    nb1983 flips the protocol: train LGBM **directly** on the FULL 4139-row
    train dose-response labels (pEC50) with the same 117-col 5-way K-tuned
    matrix, no anchor, no residual.  Predict on the 253 unblind compounds and
    report RAE.  Compare to chemprop_aux 0.6216.

PROTOCOL:
    1. Load 4139 train labels via src/pxr/data.load_train.  Standardize SMILES.
    2. Build the 117-col 5-way K-tuned feature matrix for
       FULL 4139 train + 513 test (which contains the 253 unblind via unb_idx).
         AtomPair top-25 from nb1484 (K=25 from nb1524)
         MACCS    top-20 from nb1352
         Mordred  top-20 from nb1523 best_K record top_col_idx
         ChempropEmbed top-20 from nb1541 top_dim_order_top100[:20]
         Avalon   top-30 from nb1392
         pred_chembl_pec50    (ChEMBL PXR union, kNN-5 Tanimoto on Morgan-2048)
         mean_sim             (k=5 mean Tanimoto)
       Sum = 25+20+20+20+30 + 2 = 117 cols.
    3. LGBM(MSE, max_depth=4, num_leaves=15, n_est=300, lr=0.03,
       min_child_samples=20).  5-seed bag (seeds 0,1,7,42,137).
    4. Scaffold 5-fold CV on the 4139 train (pxr.eval.scaffold_kfold_indices)
       for sanity OOF report.
    5. Train each seed-bagged LGBM on ALL 4139 train; predict on 513 test;
       slice te[unb_idx] for the 253 unblind eval.
    6. Per-seed RAE on 253 + mean-bag (avg over 5 seeds) RAE on 253.
    7. Verdict at 0.003 margin vs chemprop_aux 0.6216.

Outputs:
    scripts/nb1983_lgbm_direct_train.py    (this file)
    data/processed/nb1983_summary.json
    data/processed/nb1983_te_per_seed.npy   (5, 513) float32
    data/processed/nb1983_te_mean_bag.npy   (513,)   float32
    data/processed/nb1983_oof_per_seed_train.npy  (5, 4139) float32 (scaffold CV)
    data/processed/tr_chemprop_embed_300.npy  (4139, 300) float32  (CACHE)
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

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1983"

SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

CHEMPROP_FOLD_DIRS = [
    DATA_PROCESSED / f"chemprop_fold_{i}" / "model" / "model_0" / "checkpoints"
    for i in range(5)
]

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

CHEMPROP_AUX_REF = 0.6216
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
    """Same union as nb1484/nb1553/nb1861 (CHEMBL3401 + nr_extended + all_types)."""
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


def _load_mordred(path: Path, n_expected: int) -> np.ndarray:
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


def _load_npy(path: Path, n_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} vs {n_expected}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _find_best_ckpt(ckpt_dir: Path):
    if not ckpt_dir.is_dir():
        return None
    cands = sorted(ckpt_dir.glob("best-*.ckpt"))
    if cands:
        return cands[0]
    last = ckpt_dir / "last.ckpt"
    return last if last.exists() else None


def _extract_chemprop_embed(smiles_list, cache_path: Path,
                            label: str) -> np.ndarray:
    """Forward SMILES through 5 chemprop_aux fold MPN encoders, mean-pool.

    Returns (n, 300) float32.  Caches to cache_path.
    """
    if cache_path.exists():
        X = np.load(cache_path).astype(np.float32)
        if X.shape[0] == len(smiles_list) and X.shape[1] == 300:
            print(f"[embed-{label}] HIT cache  {cache_path}  shape={X.shape}")
            return X
        else:
            print(f"[embed-{label}] cache shape mismatch ({X.shape} vs "
                  f"({len(smiles_list)},300)); re-extracting")

    import torch
    from chemprop.models import MPNN
    from chemprop.data import MoleculeDatapoint, MoleculeDataset
    from chemprop.data.collate import collate_batch

    fold_ckpts = []
    for d in CHEMPROP_FOLD_DIRS:
        ck = _find_best_ckpt(d)
        if ck is not None:
            fold_ckpts.append(ck)
    if not fold_ckpts:
        raise FileNotFoundError(
            "No chemprop_aux fold checkpoints found under "
            f"{CHEMPROP_FOLD_DIRS[0].parent.parent}"
        )
    print(f"[embed-{label}] {len(fold_ckpts)} fold checkpoints located")

    print(f"[embed-{label}] building MoleculeDataset for "
          f"{len(smiles_list)} SMILES")
    dps = [MoleculeDatapoint.from_smi(s) for s in smiles_list]
    ds = MoleculeDataset(dps)

    chunk = 64
    n = len(ds)
    accum = np.zeros((n, 300), dtype=np.float64)
    for fi, ckpt_path in enumerate(fold_ckpts):
        t0 = time.time()
        mdl = MPNN.load_from_checkpoint(str(ckpt_path), map_location="cpu")
        mdl.eval()
        d_h = int(mdl.message_passing.output_dim)
        if d_h != 300:
            raise RuntimeError(
                f"fold {fi}: expected mp.output_dim=300, got {d_h}"
            )
        with torch.no_grad():
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                batch = collate_batch([ds[j] for j in range(start, end)])
                bmg = batch.bmg
                H = mdl.message_passing(bmg, V_d=None)
                h_mol = mdl.agg(H, bmg.batch)
                accum[start:end] += h_mol.cpu().numpy().astype(np.float64)
        del mdl
        print(f"[embed-{label}] fold {fi}: forwarded {n} mols in "
              f"{time.time()-t0:.1f}s")

    embed = (accum / float(len(fold_ckpts))).astype(np.float32)
    np.save(cache_path, embed)
    print(f"[embed-{label}] SAVED cache {cache_path}  shape={embed.shape}")
    return embed


def _lgbm_params(seed: int) -> dict:
    """User-specified knobs."""
    return dict(
        objective="regression",  # MSE
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=20,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM(MSE, d=4, leaves=15, n=300, lr=0.03, mcs=20) "
          f"DIRECT on full 4139 train labels")
    print(f"          5-seed bag, seeds={SEEDS}")
    print(f"          ref: chemprop_aux PRE-unblind in_RAE {CHEMPROP_AUX_REF:.4f}")
    print(f"          117-col 5-way K-tuned matrix "
          f"(AP-25 + MACCS-20 + Mord-20 + ChempropEmbed-20 + Avalon-30 + 2)")
    print("=" * 78)

    # ---- Truth + indices ----
    tr = load_train()
    te = load_test()
    n_train = len(tr)
    n_test = len(te)
    tr_smiles_raw = tr["smiles"].astype(str).tolist()
    te_smiles_raw = te["smiles"].astype(str).tolist()
    y_train = tr["pec50"].astype(np.float64).to_numpy()
    print(f"[load] n_train={n_train}  n_test={n_test}  "
          f"y_train mean={y_train.mean():.3f} std={y_train.std():.3f}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}  y_unb mean={y_unb.mean():.3f} "
          f"std={y_unb.std():.3f}")

    if n_train != 4139:
        raise ValueError(f"expected 4139 train, got {n_train}")
    if n_test != 513:
        raise ValueError(f"expected 513 test, got {n_test}")
    if n_unb != 253:
        raise ValueError(f"expected 253 unblind, got {n_unb}")

    # ---- Standardize SMILES ----
    print("[std] standardizing SMILES")
    tr_mols = [standardize(s) for s in tr_smiles_raw]
    te_mols = [standardize(s) for s in te_smiles_raw]
    tr_std_smi = [Chem.MolToSmiles(m) if m is not None else ""
                  for m in tr_mols]
    te_std_smi = [Chem.MolToSmiles(m) if m is not None else ""
                  for m in te_mols]
    print(f"[std] train: {sum(1 for s in tr_std_smi if s)}/{n_train} parsed")
    print(f"[std] test:  {sum(1 for s in te_std_smi if s)}/{n_test} parsed")

    # ---- Load K-tuned summaries ----
    print("\n" + "-" * 78)
    print("LOADING K-TUNED INDICES")
    print("-" * 78)
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing summary: {p}")
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
    full_ap_ranked = None
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            full_ap_ranked = np.array(f["top_idx_ranked"], dtype=int)
            break
    if full_ap_ranked is None:
        raise KeyError("AtomPair entry missing in nb1484")
    top_ap = full_ap_ranked[:K_AP]

    K_Mord = int(sum_1523["best_K"])
    rec_mord = None
    for r in sum_1523["per_K_records"]:
        if int(r["K"]) == K_Mord:
            rec_mord = r
            break
    if rec_mord is None:
        raise KeyError(f"Mordred best_K {K_Mord} record not found")
    top_mord = np.array(rec_mord["top_col_idx"], dtype=int)

    K_Emb = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed = top_embed_full[:K_Emb]

    print(f"   AtomPair  top-{len(top_ap)}  (nb1524 K={K_AP})")
    print(f"   MACCS     top-{len(top_maccs)}  (nb1352)")
    print(f"   Mordred   top-{len(top_mord)}  (nb1523 K={K_Mord})")
    print(f"   ChempropEmbed top-{len(top_embed)}  (nb1541 K={K_Emb})")
    print(f"   Avalon    top-{len(top_avalon)}  (nb1392)")

    # ---- Build feature matrices (train + test) ----
    print("\n" + "-" * 78)
    print("BUILDING 117-COL FEATURE MATRIX (train + test)")
    print("-" * 78)
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_train)[:, top_ap].astype(np.float32)
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test)[:, top_ap].astype(np.float32)
    print(f"   AtomPair tr={X_ap_tr.shape}  te={X_ap_te.shape}")

    X_mac_tr = _load_npy(MACCS_TR_PATH, n_train)[:, top_maccs].astype(np.float32)
    X_mac_te = _load_npy(MACCS_TE_PATH, n_test)[:, top_maccs].astype(np.float32)
    print(f"   MACCS    tr={X_mac_tr.shape}  te={X_mac_te.shape}")

    X_mord_tr_full = _load_mordred(MORDRED_DIR / "X_mordred_train.npy", n_train)
    X_mord_te_full = _load_mordred(MORDRED_DIR / "X_mordred_test.npy", n_test)
    X_mord_tr = X_mord_tr_full[:, top_mord].astype(np.float32)
    X_mord_te = X_mord_te_full[:, top_mord].astype(np.float32)
    print(f"   Mordred  tr={X_mord_tr.shape}  te={X_mord_te.shape}")

    # Chemprop embeddings (cache train + load test)
    X_emb_tr_full = _extract_chemprop_embed(
        tr_std_smi, CHEMPROP_EMBED_TR_PATH, label="train"
    )
    X_emb_te_full = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_tr = X_emb_tr_full[:, top_embed].astype(np.float32)
    X_emb_te = X_emb_te_full[:, top_embed].astype(np.float32)
    print(f"   ChempropEmbed tr={X_emb_tr.shape}  te={X_emb_te.shape}")

    X_av_tr = _load_npy(AVALON_TR_PATH, n_train)[:, top_avalon].astype(np.float32)
    X_av_te = _load_npy(AVALON_TE_PATH, n_test)[:, top_avalon].astype(np.float32)
    print(f"   Avalon   tr={X_av_tr.shape}  te={X_av_te.shape}")

    # ---- ChEMBL kNN pred_chembl + mean_sim for train + test ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL + kNN feature build (train + test)")
    print("-" * 78)
    pool = _load_chembl_pool()

    # Drop both train and test InChIKeys from pool (to avoid leak when
    # constructing pred_chembl for train labels).
    tr_te_inchikeys = set()
    for m in tr_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            tr_te_inchikeys.add(ik)
    for m in te_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            tr_te_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(tr_te_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after} "
          f"that match train+test InChIKey)")

    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")

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
    print(f"   pred_chembl_tr mean={pred_chembl_tr.mean():.3f} "
          f"std={pred_chembl_tr.std():.3f}")
    print(f"   pred_chembl_te mean={pred_chembl_te.mean():.3f} "
          f"std={pred_chembl_te.std():.3f}")
    print(f"   mean_sim_tr    mean={mean_sim_tr.mean():.3f}")
    print(f"   mean_sim_te    mean={mean_sim_te.mean():.3f}")

    # ---- Concatenate to 117-col matrix ----
    X_tr = np.concatenate(
        [
            X_ap_tr, X_mac_tr, X_mord_tr, X_emb_tr, X_av_tr,
            pred_chembl_tr.reshape(-1, 1),
            mean_sim_tr.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    X_te = np.concatenate(
        [
            X_ap_te, X_mac_te, X_mord_te, X_emb_te, X_av_te,
            pred_chembl_te.reshape(-1, 1),
            mean_sim_te.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    expected_dim = (len(top_ap) + len(top_maccs) + len(top_mord)
                    + len(top_embed) + len(top_avalon) + 2)
    if X_tr.shape[1] != expected_dim or X_te.shape[1] != expected_dim:
        raise ValueError(
            f"feat dim mismatch: tr={X_tr.shape[1]} te={X_te.shape[1]} "
            f"expected={expected_dim}"
        )
    print(f"\n   X_tr = {X_tr.shape}  X_te = {X_te.shape}  (117-col K-tuned)")

    # ---- Scaffold 5-fold CV (sanity OOF) ----
    print("\n" + "-" * 78)
    print("SCAFFOLD 5-FOLD CV (sanity OOF on 4139 train)")
    print("-" * 78)
    scaffolds_tr = [bemis_murcko(s) for s in tr_std_smi]
    cv_splits = scaffold_kfold_indices(scaffolds_tr, n_splits=5, seed=42)
    print(f"   scaffold CV: {len(cv_splits)} folds")
    for f, (tr_idx, va_idx) in enumerate(cv_splits):
        print(f"     fold {f}: train={len(tr_idx)}  val={len(va_idx)}")

    oof_per_seed = np.zeros((len(SEEDS), n_train), dtype=np.float32)
    cv_rae_per_seed = []
    for si, seed in enumerate(SEEDS):
        t_s = time.time()
        oof_s = np.zeros(n_train, dtype=np.float64)
        for f, (tr_idx, va_idx) in enumerate(cv_splits):
            mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
            mdl.fit(X_tr[tr_idx], y_train[tr_idx])
            oof_s[va_idx] = mdl.predict(X_tr[va_idx])
        oof_per_seed[si] = oof_s.astype(np.float32)
        rae_s = float(rae(y_train, oof_s))
        cv_rae_per_seed.append(rae_s)
        print(f"   seed={seed:3d}  scaffold-CV RAE (train) = {rae_s:.4f}  "
              f"wall={time.time()-t_s:.1f}s")
    mean_bag_oof = oof_per_seed.mean(axis=0)
    cv_rae_mean_bag = float(rae(y_train, mean_bag_oof))
    print(f"   mean-bag scaffold-CV RAE (train) = {cv_rae_mean_bag:.4f}")

    # ---- Train on full 4139, predict on 513 test, eval on 253 unblind ----
    print("\n" + "-" * 78)
    print("FULL TRAIN -> TEST PRED -> EVAL on 253 unblind")
    print("-" * 78)
    te_per_seed = np.zeros((len(SEEDS), n_test), dtype=np.float32)
    unb_rae_per_seed = []
    for si, seed in enumerate(SEEDS):
        t_s = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X_tr, y_train)
        te_pred = mdl.predict(X_te)
        te_per_seed[si] = te_pred.astype(np.float32)
        unb_pred = te_pred[unb_idx]
        rae_unb = float(rae(y_unb, unb_pred))
        unb_rae_per_seed.append(rae_unb)
        print(f"   seed={seed:3d}  UNBLIND-253 RAE = {rae_unb:.4f}  "
              f"wall={time.time()-t_s:.1f}s")
    te_mean_bag = te_per_seed.mean(axis=0)
    unb_pred_mean_bag = te_mean_bag[unb_idx]
    rae_mean_bag = float(rae(y_unb, unb_pred_mean_bag))
    rae_per_seed_mean = float(np.mean(unb_rae_per_seed))
    print(f"   per-seed mean RAE     = {rae_per_seed_mean:.4f}")
    print(f"   mean-bag pooled RAE   = {rae_mean_bag:.4f}")

    # ---- Verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    delta_vs_chemprop = rae_mean_bag - CHEMPROP_AUX_REF
    if rae_mean_bag < CHEMPROP_AUX_REF - DECISION_MARGIN:
        verdict = "BEATS_CHEMPROP_AUX_BY_>0.003"
    elif abs(delta_vs_chemprop) <= DECISION_MARGIN:
        verdict = "FLAT_VS_CHEMPROP_AUX"
    else:
        verdict = "UNDERPERFORMS_CHEMPROP_AUX"
    print(f"   chemprop_aux ref         = {CHEMPROP_AUX_REF:.4f}")
    print(f"   mean-bag UNBLIND RAE     = {rae_mean_bag:.4f}")
    print(f"   d(mean_bag, chemprop)    = {delta_vs_chemprop:+.4f}")
    print(f"   per-seed mean UNBLIND    = {rae_per_seed_mean:.4f}")
    print(f"   mean-bag scaffold-CV RAE = {cv_rae_mean_bag:.4f}")
    print(f"   verdict                  = {verdict}")

    # ---- Save ----
    np.save(DATA_PROCESSED / f"{TAG}_te_per_seed.npy", te_per_seed)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_te_per_seed.npy'}")
    np.save(DATA_PROCESSED / f"{TAG}_te_mean_bag.npy",
            te_mean_bag.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / f'{TAG}_te_mean_bag.npy'}")
    np.save(DATA_PROCESSED / f"{TAG}_oof_per_seed_train.npy", oof_per_seed)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_oof_per_seed_train.npy'}")

    summary = {
        "tag": TAG,
        "approach": "LGBM_direct_on_full_4139_train_labels_no_anchor_no_residual",
        "lgbm_params": {
            "objective": "regression",
            "max_depth": 4,
            "num_leaves": 15,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "min_child_samples": 20,
        },
        "seeds": SEEDS,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_chembl_pool": int(len(pool)),
        "feat_dim": int(X_tr.shape[1]),
        "K_tuned_config": {
            "AtomPair": int(len(top_ap)),
            "MACCS": int(len(top_maccs)),
            "Mordred": int(len(top_mord)),
            "ChempropEmbed": int(len(top_embed)),
            "Avalon": int(len(top_avalon)),
            "extras": ["pred_chembl_pec50", "mean_sim"],
        },
        "cv_rae_per_seed_train": [float(x) for x in cv_rae_per_seed],
        "cv_rae_mean_bag_train": float(cv_rae_mean_bag),
        "unb_rae_per_seed": [float(x) for x in unb_rae_per_seed],
        "unb_rae_per_seed_mean": float(rae_per_seed_mean),
        "unb_rae_mean_bag": float(rae_mean_bag),
        "chemprop_aux_ref": float(CHEMPROP_AUX_REF),
        "delta_mean_bag_vs_chemprop_aux": float(delta_vs_chemprop),
        "decision_margin": float(DECISION_MARGIN),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] total wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_train", "n_test", "n_unb", "n_chembl_pool", "feat_dim",
        "K_tuned_config",
        "cv_rae_per_seed_train", "cv_rae_mean_bag_train",
        "unb_rae_per_seed", "unb_rae_per_seed_mean", "unb_rae_mean_bag",
        "chemprop_aux_ref", "delta_mean_bag_vs_chemprop_aux", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
