"""nb1501 -- CatBoost residual on PRE-unblind 4-way pruned features (112 cols).

HYPOTHESIS:
    nb1483 CatBoost(MAE, d4, n200, lr0.05, l2=5) on the 3-way 82-col pruned
    matrix (top-30 AtomPair + top-20 MACCS + top-30 Mordred + pred_chembl +
    sim, chemprop_aux anchor) is the best PRE-unblind CatBoost number to
    date.  nb1484 extended the LGBM-Huber residual learner to 4-way by
    splicing in a top-30 chemprop-embed family, which broke nb1472's
    0.5330 LGBM ceiling and produced best SLSQP cross-fit 0.5231 on 253.

    Swap nb1484's LGBM-Huber residual learner for nb1483's CatBoost(MAE,
    d4, n200, lr0.05, l2=5), trained on the *concatenated* 4-way pruned
    matrix in a single model (top-30 AP + top-20 MACCS + top-30 Mordred +
    top-30 chemprop-embed + pred_chembl + sim = 112 cols).  Goal is a
    CatBoost candidate that combines nb1483's loss-shape advantage with
    nb1484's chemprop-embed channel.

PROTOCOL:
    1.  Anchor = chemprop_aux te[unb_idx]  (PRE-unblind).
        residual = y_unb - anchor.
    2.  Reuse 4-way SHAP picks:
            top-30 AtomPair bits      (from nb1373 / nb1484)
            top-20 MACCS bits         (from nb1352 / nb1484)
            top-30 Mordred cols       (from nb1364 / nb1484)
            top-30 chemprop embed     (from nb1484 family entry)
        Concat with pred_chembl (ChEMBL PXR kNN-5 Tanimoto over Morgan
        union pool) and mean_sim --> 30 + 20 + 30 + 30 + 2 = 112 cols.
    3.  5-seed bag CatBoost(loss=MAE, depth=4, n_est=200, lr=0.05, l2=5).
        KFold(n=5, shuffle=True) cross-fit per seed.
    4.  Pool mean_bag and median_bag.
    5.  Verdict at 0.003 margin vs:
            chemprop_aux standalone           0.6216
            nb1472 chemprop_aux+3way blend    0.5330  (PRE-unblind ref)
            nb1483 CatBoost 3-way 82-col      (lookup)
            nb1484 4-way LGBM-Huber best      0.5231  (KEY REF -- 0.003)

Outputs:
    scripts/nb1501_catboost_preunblind_4way.py
    data/processed/nb1501_summary.json
    data/processed/nb1501_mean_bag_oof.npy            (253,) float32
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
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1501"
ANCHOR = "chemprop_aux"           # PRE-unblind te file on 513
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1364_SUMMARY = DATA_PROCESSED / "nb1364_summary.json"
NB1373_SUMMARY = DATA_PROCESSED / "nb1373_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# References (PRE-unblind path)
CHEMPROP_AUX_REF = 0.6216         # chemprop_aux te[unb_idx]
NB1472_REF = 0.5330               # nb1472 chemprop_aux + 3-way blend mean
NB1483_REF = None                 # loaded from nb1483_summary.json
NB1484_REF = 0.5231               # nb1484 4-way LGBM-Huber best
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
    """Same union as nb1441 / nb1460 / nb1483 / nb1484."""
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


def _cat_params(seed: int) -> dict:
    return dict(
        loss_function="MAE",
        depth=4,
        iterations=200,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = CatBoostRegressor(**_cat_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


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


def _load_chemprop_embed_test(n_test_expected: int) -> np.ndarray:
    if not CHEMPROP_EMBED_TE_PATH.exists():
        raise FileNotFoundError(
            f"Chemprop embed cache missing: {CHEMPROP_EMBED_TE_PATH}"
        )
    X = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"Chemprop embed shape mismatch: {X.shape}")
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_embed_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "ChempropEmbed":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("ChempropEmbed entry not found in nb1484_summary.json")


def main() -> dict:
    global NB1483_REF
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CatBoost(MAE, d4, n200, lr0.05, l2=5) on 4-way pruned "
          f"112-col features; PRE-unblind anchor={ANCHOR}")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
    print(f"          refs: chemprop_aux ({CHEMPROP_AUX_REF:.4f}), "
          f"nb1472 ({NB1472_REF:.4f}), nb1484 ({NB1484_REF:.4f})  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # PRE-unblind anchor: chemprop_aux te file on 513, slice to unb_idx
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Reuse SHAP picks from nb1352 + nb1364 + nb1373 + nb1484 ----
    for p in (NB1352_SUMMARY, NB1364_SUMMARY, NB1373_SUMMARY, NB1484_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p} -- run prerequisite first")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1364_SUMMARY) as f:
        sum_1364 = json.load(f)
    with open(NB1373_SUMMARY) as f:
        sum_1373 = json.load(f)
    with open(NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    top_mord_col_idx = np.array(
        sum_1364["top_mordred_col_indices_ranked"], dtype=int
    )
    top_ap_bit_idx = np.array(
        sum_1373["top_atompair_bit_indices_ranked"], dtype=int
    )
    top_embed_col_idx = _extract_embed_top_idx_from_nb1484(sum_1484)
    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    print(f"[reuse] top-{n_top_maccs} MACCS bits         (from nb1352)")
    print(f"[reuse] top-{n_top_mord} Mordred cols       (from nb1364)")
    print(f"[reuse] top-{n_top_ap} AtomPair bits      (from nb1373)")
    print(f"[reuse] top-{n_top_embed} Chemprop embed col (from nb1484)")

    # Try to load nb1483 ref number from its summary
    nb1483_path = DATA_PROCESSED / "nb1483_summary.json"
    if nb1483_path.exists():
        try:
            with open(nb1483_path) as f:
                NB1483_REF = float(json.load(f).get("rae_mean_bag", 0.0))
            print(f"[ref ] nb1483 rae_mean_bag = {NB1483_REF:.4f}")
        except Exception as e:
            print(f"[ref ] nb1483 summary unreadable ({e}); NB1483_REF=None")

    # ---- AtomPair (top-30 cols) ----
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(
            f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}"
        )
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)
    print(f"[feat] X_ap_unb_top shape      = {X_ap_unb_top.shape}")

    # ---- MACCS (top-20 cols) ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb_top shape   = {X_maccs_unb_top.shape}")

    # ---- Mordred (top-30 cols) ----
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top shape    = {X_mord_unb_top.shape}")

    # ---- Chemprop embed (top-30 cols) ----
    X_emb_te = _load_chemprop_embed_test(n_test_expected=n_test)
    X_emb_unb = X_emb_te[unb_idx].astype(np.float32)
    X_emb_unb_top = X_emb_unb[:, top_embed_col_idx].astype(np.float32)
    print(f"[feat] X_emb_unb_top shape     = {X_emb_unb_top.shape}")

    # ---- ChEMBL pool + kNN feature build (513-level) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (same union as nb1441 / nb1483 / nb1484)")
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
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- Build QUAD-PRUNED 112-col feature matrix on 253 ----
    X_unb = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected_dim = n_top_ap + n_top_maccs + n_top_mord + n_top_embed + 2
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   4-WAY PRUNED feature matrix: {X_unb.shape}  "
          f"(top-{n_top_ap} AP + top-{n_top_maccs} MACCS + "
          f"top-{n_top_mord} Mordred + top-{n_top_embed} Embed "
          f"+ pred_chembl + sim)")

    # ---- Per-seed residual cross-fit on QUAD-PRUNED features ----
    print("\n" + "-" * 78)
    print(f"PER-SEED CATBOOST RESIDUAL CROSS-FIT (dim={feat_dim})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    # ---- Pearson vs prior PRE-unblind candidates (orthogonality) ----
    def _pearson_vs(path: Path):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(mean_bag_oof, oof)[0, 1])

    pearson_vs_nb1483 = _pearson_vs(DATA_PROCESSED / "nb1483_mean_bag_oof.npy")
    pearson_vs_nb1484 = _pearson_vs(DATA_PROCESSED / "nb1484_best_oof.npy")
    pearson_vs_nb1472 = _pearson_vs(DATA_PROCESSED / "nb1472_mean_oof.npy")
    pearson_vs_anchor = float(np.corrcoef(mean_bag_oof, anchor)[0, 1])

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f}"
          f"  d_vs_nb1472 = {rae_mean_bag - NB1472_REF:+.4f}"
          f"  d_vs_nb1484 = {rae_mean_bag - NB1484_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}")
    print(f"   Pearson(mean_bag, anchor)         = {pearson_vs_anchor:.4f}")
    if pearson_vs_nb1483 is not None:
        print(f"   Pearson(mean_bag, nb1483_mean)    = {pearson_vs_nb1483:.4f}")
    if pearson_vs_nb1484 is not None:
        print(f"   Pearson(mean_bag, nb1484_best)    = {pearson_vs_nb1484:.4f}")
    if pearson_vs_nb1472 is not None:
        print(f"   Pearson(mean_bag, nb1472_mean)    = {pearson_vs_nb1472:.4f}")

    beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1472 = rae_mean_bag < NB1472_REF - DECISION_MARGIN
    beats_nb1484 = rae_mean_bag < NB1484_REF - DECISION_MARGIN
    beats_nb1483 = (NB1483_REF is not None) and (
        rae_mean_bag < NB1483_REF - DECISION_MARGIN
    )
    flat_vs_nb1484 = abs(rae_mean_bag - NB1484_REF) < DECISION_MARGIN

    if beats_nb1484:
        verdict = "CATBOOST_4WAY_PRE_UNBLIND_BEATS_NB1484_NEW_PRE_UNBLIND_PRIMARY"
    elif flat_vs_nb1484:
        verdict = "CATBOOST_4WAY_PRE_UNBLIND_FLAT_VS_NB1484"
    elif beats_nb1472:
        verdict = "CATBOOST_4WAY_PRE_UNBLIND_BEATS_NB1472_BUT_WORSE_THAN_NB1484"
    elif beats_anchor:
        verdict = "CATBOOST_4WAY_PRE_UNBLIND_BEATS_ANCHOR_BUT_WORSE_THAN_NB1472"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "CATBOOST_4WAY_PRE_UNBLIND_FLAT_VS_ANCHOR"
    else:
        verdict = "CATBOOST_4WAY_PRE_UNBLIND_HURTS_ANCHOR"

    pre_unblind_clean = True   # anchor is chemprop_aux, no 253-label leak
    print(f"   verdict                = {verdict}")
    print(f"   PRE-unblind clean      = {pre_unblind_clean}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": "AtomPair-cache + MACCS-cache + Mordred-cached_nb1030 + "
                       "ChempropEmbed-cache + local_chembl_caches_union",
        "model_family": "CatBoost",
        "catboost_loss": "MAE",
        "catboost_depth": 4,
        "catboost_iterations": 200,
        "catboost_learning_rate": 0.05,
        "catboost_l2_leaf_reg": 5.0,
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "n_top_atompair": n_top_ap,
        "n_top_maccs": n_top_maccs,
        "n_top_mordred": n_top_mord,
        "n_top_chemprop_embed": n_top_embed,
        "feat_dim": int(feat_dim),
        "feat_breakdown": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_chemprop_aux": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1472": rae_mean_bag - NB1472_REF,
        "delta_mean_bag_vs_nb1484": rae_mean_bag - NB1484_REF,
        "delta_mean_bag_vs_nb1483": (
            None if NB1483_REF is None else rae_mean_bag - NB1483_REF
        ),
        "beats_chemprop_aux": bool(beats_anchor),
        "beats_nb1472": bool(beats_nb1472),
        "beats_nb1484": bool(beats_nb1484),
        "beats_nb1483": bool(beats_nb1483),
        "flat_vs_nb1484": bool(flat_vs_nb1484),
        "pearson_vs_anchor": pearson_vs_anchor,
        "pearson_vs_nb1483_mean_bag": pearson_vs_nb1483,
        "pearson_vs_nb1484_best": pearson_vs_nb1484,
        "pearson_vs_nb1472_mean": pearson_vs_nb1472,
        "verdict": verdict,
        "pre_unblind_clean": pre_unblind_clean,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1472_ref": NB1472_REF,
        "nb1483_ref": NB1483_REF,
        "nb1484_ref": NB1484_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_chembl_pool", "feat_dim", "feat_breakdown",
        "rae_anchor_chemprop_aux", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_chemprop_aux",
        "delta_mean_bag_vs_nb1472",
        "delta_mean_bag_vs_nb1483",
        "delta_mean_bag_vs_nb1484",
        "beats_chemprop_aux", "beats_nb1472",
        "beats_nb1483", "beats_nb1484", "flat_vs_nb1484",
        "pearson_vs_anchor",
        "pearson_vs_nb1483_mean_bag",
        "pearson_vs_nb1484_best",
        "pearson_vs_nb1472_mean",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
