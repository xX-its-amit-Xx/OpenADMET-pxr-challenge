"""nb2031 -- Outer-bag of nb2023 best-lambda (reg_lambda=3.0) for honest validation.

HYPOTHESIS:
    nb2023's lambda grid identified reg_lambda=3.0 as the best single-bag setting
    (rae_mean_bag = 0.5079, beating lambda=2.0 reference 0.5100 by -0.0021).
    nb1861 ran the outer-bag-of-bags (BoB) honest protocol at lambda=2.0 and
    reported BoB MEAN = 0.5078, pooled-25 = 0.5013.

    Now run the IDENTICAL outer-bag protocol at lambda=3.0 to see whether the
    single-bag advantage of lambda=3 transfers to the BoB-honest aggregate, or
    whether it disappears under outer-seed averaging.

    Configuration (mirrors nb1861 exactly except reg_lambda):
        objective='regression'        (MSE)
        max_depth=4, num_leaves=15
        n_estimators=300, lr=0.03
        min_child_samples=5
        reg_lambda=3.0
        outer_seeds = [0, 1, 7, 42, 137]
        inner_per_outer = 5  (inner = outer*1000 + j for j in 0..4)
        resid_folds = 5

PROTOCOL:
    1. Anchor = chemprop_aux te[unb_idx]   (PRE-unblind, in_RAE 0.6216).
       residual = y_unb - anchor.
    2. Build 117-col feature stack (identical to nb1852/nb1861/nb2023).
    3. For each outer seed s in {0,1,7,42,137}:
         inner_seeds = [s*1000 + j for j in 0..4]
         per-inner: 5-fold cross-fit LGBM(regression/MSE, lambda=3.0, ...) on residual
         outer_mean_oof = mean of 5 inner corrected OOFs (anchor + resid)
         rae_outer = rae(y_unb, outer_mean_oof)
       Also pool all 5 outer mean OOFs --> pooled-25-bag RAE.
    4. Report per-outer RAE list, BoB MEAN, BoB MEDIAN, pooled-25.
    5. Verdict vs nb1861 (lambda=2.0) reference.

Outputs:
    scripts/nb2031_bag_lambda3.py
    data/processed/nb2031_summary.json
    data/processed/nb2031_per_outer_oof.npy   (5, 253) float32
    data/processed/nb2031_pooled_25bag_oof.npy (253,) float32
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
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2031"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_PER_OUTER = 5
RESID_FOLDS = 5
REG_LAMBDA = 3.0   # nb2023 grid winner

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
NB1861_SUMMARY = DATA_PROCESSED / "nb1861_summary.json"
NB2023_SUMMARY = DATA_PROCESSED / "nb2023_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB1554_REF = 0.5163
NB1771_REF = 0.5100
NB1852_REF = 0.5100              # single-bag MSE lambda=2.0
NB1861_BOB_MEAN_REF = 0.5078     # 5-outer BoB MEAN at lambda=2.0
NB1861_POOLED25_REF = 0.5013     # pooled-25 at lambda=2.0
NB2023_LAMBDA3_REF = 0.5079      # nb2023 single-bag at lambda=3.0
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


def _lgbm_params(seed: int) -> dict:
    """LGBM(regression/MSE) -- mirrors nb1861 with reg_lambda swapped to 3.0."""
    return dict(
        objective="regression",     # standard MSE / L2
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=REG_LAMBDA,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
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


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG of nb2023 lambda={REG_LAMBDA} (LGBM regression/MSE) "
          f"on 5-way K-tuned 117-col matrix; PRE-unblind anchor={ANCHOR}")
    print(f"          outer seeds = {OUTER_SEEDS}  inner/outer = "
          f"{INNER_PER_OUTER}  folds = {RESID_FOLDS}")
    print(f"          refs: chemprop_aux ({CHEMPROP_AUX_REF:.4f}), "
          f"nb1852 lambda=2 single-bag ({NB1852_REF:.4f})")
    print(f"          refs: nb1861 BoB MEAN ({NB1861_BOB_MEAN_REF:.4f}), "
          f"pooled-25 ({NB1861_POOLED25_REF:.4f})")
    print(f"          refs: nb2023 lambda=3 single-bag ({NB2023_LAMBDA3_REF:.4f})")
    print(f"          decision_margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- nb1861 reference (live) ----
    if NB1861_SUMMARY.exists():
        with open(NB1861_SUMMARY) as f:
            nb1861_sum = json.load(f)
        nb1861_bob_mean = float(nb1861_sum.get("bob_mean_rae", NB1861_BOB_MEAN_REF))
        nb1861_pooled25 = float(nb1861_sum.get("rae_pooled_25bag", NB1861_POOLED25_REF))
        print(f"[ref] nb1861_summary.bob_mean_rae      = {nb1861_bob_mean:.4f}")
        print(f"[ref] nb1861_summary.rae_pooled_25bag  = {nb1861_pooled25:.4f}")
    else:
        nb1861_bob_mean = NB1861_BOB_MEAN_REF
        nb1861_pooled25 = NB1861_POOLED25_REF
        print(f"[ref] nb1861_summary.json missing -- using fallback "
              f"BoB={nb1861_bob_mean:.4f}  pooled25={nb1861_pooled25:.4f}")

    # ---- nb2023 reference (live) ----
    if NB2023_SUMMARY.exists():
        with open(NB2023_SUMMARY) as f:
            nb2023_sum = json.load(f)
        nb2023_lambda3_rae = None
        for r in nb2023_sum.get("lambda_records", []):
            if abs(float(r["reg_lambda"]) - REG_LAMBDA) < 1e-9:
                nb2023_lambda3_rae = float(r["rae_mean_bag"])
                break
        if nb2023_lambda3_rae is None:
            nb2023_lambda3_rae = NB2023_LAMBDA3_REF
        print(f"[ref] nb2023_summary.lambda={REG_LAMBDA}.rae_mean_bag = "
              f"{nb2023_lambda3_rae:.4f}")
    else:
        nb2023_lambda3_rae = NB2023_LAMBDA3_REF
        print(f"[ref] nb2023_summary.json missing -- using fallback "
              f"{nb2023_lambda3_rae:.4f}")

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

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

    # ---- Load all K-grid winners + SHAP rankings ----
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
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )

    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                       best_K_key="best_K")
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    assert K_Mord_best == int(sum_1523["best_K"])

    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]

    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]

    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )
    K_Avalon_used = int(len(top_avalon_bit_idx))

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(K_Avalon_used)
    print(f"[reuse] top-{n_top_ap}     AtomPair bits (nb1524 K={K_AP_best})")
    print(f"[reuse] top-{n_top_maccs}     MACCS bits  (nb1352)")
    print(f"[reuse] top-{n_top_mord}     Mordred cols (nb1523 K={K_Mord_best})")
    print(f"[reuse] top-{n_top_embed}     ChempropEmbed dims (nb1541 K={K_Embed_best})")
    print(f"[reuse] top-{n_top_avalon}     Avalon bits (nb1392 SHAP K=30)")

    # ---- Feature matrices ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)
    print(f"[feat] X_ap_unb_top      = {X_ap_unb_top.shape}")

    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb_top   = {X_maccs_unb_top.shape}")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top    = {X_mord_unb_top.shape}")

    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb = X_emb_te[unb_idx].astype(np.float32)
    X_emb_unb_top = X_emb_unb[:, top_embed_col_idx].astype(np.float32)
    print(f"[feat] X_emb_unb_top     = {X_emb_unb_top.shape}")

    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb = X_av_te[unb_idx].astype(np.float32)
    X_av_unb_top = X_av_unb[:, top_avalon_bit_idx].astype(np.float32)
    print(f"[feat] X_av_unb_top      = {X_av_unb_top.shape}")

    # ---- ChEMBL kNN ----
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

    X_unb = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   COMBINED 5-WAY K-TUNED matrix: {X_unb.shape}")

    # ---- OUTER BAG LOOP ----
    print("\n" + "-" * 78)
    print(f"OUTER-BAG: {len(OUTER_SEEDS)} outer seeds x {INNER_PER_OUTER} inner "
          f"seeds x {RESID_FOLDS}-fold cross-fit  (dim={feat_dim})  "
          f"reg_lambda={REG_LAMBDA}")
    print("-" * 78)

    per_outer_oof = np.zeros((len(OUTER_SEEDS), n_unb), dtype=np.float64)
    per_outer_rae: list[float] = []
    per_outer_records = []

    for o_i, o_seed in enumerate(OUTER_SEEDS):
        t_out = time.time()
        inner_seeds = [o_seed * 1000 + j for j in range(INNER_PER_OUTER)]
        inner_corrected = np.zeros((INNER_PER_OUTER, n_unb), dtype=np.float64)
        inner_rae = []
        for i_j, i_seed in enumerate(inner_seeds):
            t_in = time.time()
            resid_oof_i = _residual_cross_fit_one_seed(X_unb, residual, i_seed)
            pred_corr_i = anchor + resid_oof_i
            inner_corrected[i_j] = pred_corr_i
            r_i = float(rae(y_unb, pred_corr_i))
            inner_rae.append(r_i)
            print(f"   outer={o_seed:4d}  inner={i_seed:6d}  "
                  f"rae={r_i:.4f}  wall={time.time() - t_in:.1f}s")

        outer_mean_oof = inner_corrected.mean(axis=0)
        per_outer_oof[o_i] = outer_mean_oof
        rae_o = float(rae(y_unb, outer_mean_oof))
        per_outer_rae.append(rae_o)
        per_outer_records.append({
            "outer_seed": int(o_seed),
            "inner_seeds": [int(s) for s in inner_seeds],
            "inner_rae": [float(r) for r in inner_rae],
            "inner_rae_mean": float(np.mean(inner_rae)),
            "rae_outer_mean_bag": rae_o,
            "delta_outer_vs_anchor": rae_o - rae_anchor,
            "wall_sec": round(time.time() - t_out, 2),
        })
        print(f"   ==> outer={o_seed:4d}  rae_outer_mean_bag={rae_o:.4f}  "
              f"(d_vs_anchor={rae_o - rae_anchor:+.4f})  "
              f"wall={time.time() - t_out:.1f}s")

    # ---- BoB aggregates ----
    arr = np.array(per_outer_rae)
    bob_mean = float(arr.mean())
    bob_median = float(np.median(arr))
    bob_std = float(arr.std())
    bob_min = float(arr.min())
    bob_max = float(arr.max())

    # Pooled corrected oof across all 5 outer x 5 inner = 25 cross-fits
    pooled_oof = per_outer_oof.mean(axis=0)
    rae_pooled = float(rae(y_unb, pooled_oof))

    print("\n" + "-" * 78)
    print("BAG-OF-BAGS AGGREGATIONS")
    print("-" * 78)
    print(f"   per-outer RAE list  = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae)}]")
    print(f"   BoB MEAN   RAE      = {bob_mean:.4f}")
    print(f"   BoB MEDIAN RAE      = {bob_median:.4f}")
    print(f"   BoB std/min/max     = {bob_std:.4f} / {bob_min:.4f} / "
          f"{bob_max:.4f}")
    print(f"   pooled 25-bag RAE   = {rae_pooled:.4f}")
    print()
    print(f"   nb1861 BoB MEAN ref     = {nb1861_bob_mean:.4f}  (lambda=2.0)")
    print(f"   nb1861 pooled-25 ref    = {nb1861_pooled25:.4f}  (lambda=2.0)")
    print(f"   nb2023 lambda=3 ref     = {nb2023_lambda3_rae:.4f}  "
          f"(single-bag)")
    print(f"   d(BoB_MEAN, nb1861)     = {bob_mean - nb1861_bob_mean:+.4f}")
    print(f"   d(pooled25, nb1861p25)  = {rae_pooled - nb1861_pooled25:+.4f}")
    print(f"   d(BoB_MEAN, nb2023_l3)  = {bob_mean - nb2023_lambda3_rae:+.4f}")

    # ---- Verdict (BoB-MEAN vs nb1861 BoB-MEAN, both honest) ----
    delta_bob_vs_nb1861 = bob_mean - nb1861_bob_mean
    delta_pooled_vs_nb1861 = rae_pooled - nb1861_pooled25
    if delta_bob_vs_nb1861 < -DECISION_MARGIN:
        verdict = "LAMBDA3_BOB_IMPROVES_OVER_LAMBDA2"
    elif delta_bob_vs_nb1861 > DECISION_MARGIN:
        verdict = "LAMBDA3_BOB_HURTS_VS_LAMBDA2"
    else:
        verdict = "LAMBDA3_BOB_NEUTRAL_VS_LAMBDA2"

    print(f"   verdict             = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_outer_oof.npy",
            per_outer_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_outer_oof.npy'}")
    np.save(DATA_PROCESSED / f"{TAG}_pooled_25bag_oof.npy",
            pooled_oof.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / f'{TAG}_pooled_25bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("AtomPair-cache + MACCS-cache + "
                        "Mordred-cached_nb1030 + ChempropEmbed-cache + "
                        "Avalon-cache + local_chembl_caches_union"),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": REG_LAMBDA,
        "outer_seeds": OUTER_SEEDS,
        "inner_per_outer": INNER_PER_OUTER,
        "resid_folds": RESID_FOLDS,
        "K_AP_best": K_AP_best,
        "K_Mord_best": K_Mord_best,
        "K_Embed_best": K_Embed_best,
        "K_Avalon_used": K_Avalon_used,
        "K_MACCS_fixed": n_top_maccs,
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "feat_dim": int(feat_dim),
        "feat_breakdown": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_outer_rae": per_outer_rae,
        "per_outer_records": per_outer_records,
        "bob_mean_rae": bob_mean,
        "bob_median_rae": bob_median,
        "bob_std_rae": bob_std,
        "bob_min_rae": bob_min,
        "bob_max_rae": bob_max,
        "rae_pooled_25bag": rae_pooled,
        "delta_bob_mean_vs_anchor": bob_mean - rae_anchor,
        "delta_bob_mean_vs_nb1554": bob_mean - NB1554_REF,
        "delta_bob_mean_vs_nb1771": bob_mean - NB1771_REF,
        "delta_bob_mean_vs_nb1852": bob_mean - NB1852_REF,
        "delta_bob_mean_vs_nb1861": delta_bob_vs_nb1861,
        "delta_pooled25_vs_nb1861": delta_pooled_vs_nb1861,
        "delta_bob_mean_vs_nb2023_lambda3": bob_mean - nb2023_lambda3_rae,
        "verdict": verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1554_ref": NB1554_REF,
        "nb1771_ref": NB1771_REF,
        "nb1852_ref": NB1852_REF,
        "nb1861_bob_mean_ref": nb1861_bob_mean,
        "nb1861_pooled25_ref": nb1861_pooled25,
        "nb2023_lambda3_ref": nb2023_lambda3_rae,
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
        "lgbm_reg_lambda",
        "outer_seeds", "inner_per_outer", "resid_folds",
        "K_AP_best", "K_Mord_best", "K_Embed_best", "K_Avalon_used",
        "n_chembl_pool", "feat_dim",
        "rae_anchor_chemprop_aux", "per_outer_rae",
        "bob_mean_rae", "bob_median_rae",
        "bob_std_rae", "bob_min_rae", "bob_max_rae",
        "rae_pooled_25bag",
        "delta_bob_mean_vs_anchor",
        "delta_bob_mean_vs_nb1852",
        "delta_bob_mean_vs_nb1861",
        "delta_pooled25_vs_nb1861",
        "delta_bob_mean_vs_nb2023_lambda3",
        "nb1861_bob_mean_ref",
        "nb1861_pooled25_ref",
        "nb2023_lambda3_ref",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
