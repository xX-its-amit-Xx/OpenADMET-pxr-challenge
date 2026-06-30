"""nb1091 -- Stratified scaffold-frequency LGBM (K=28, 5-seed bag).

HYPOTHESIS:
    nb2103 K=28 mean-bag RAE = 0.4737 / median-bag = 0.4698 is the best single-K
    LGBM residual-corrector on top of chemprop_aux (PRE-unblind anchor in_RAE
    0.6216).  MEMORY says 197/253 unblind compounds have unseen scaffolds
    (scaf_train_freq=0 in the 4139 train) -- the dominant failure tail.
    A single global LGBM cannot match the residual structure of {singletons,
    small-families, large-families, unseen} simultaneously.  This notebook
    trains one LGBM per stratum (4 strata) and routes each test compound to
    its scaffold-frequency stratum.

PROTOCOL:
    1. Compute Bemis-Murcko scaffold for all 4139 TRAIN compounds; build a
       scaffold->train-frequency map.
    2. Map each 513 TEST compound's scaffold to its TRAIN frequency
       (0 = unseen, 1 = singleton-in-train, etc.).
    3. Define 4 strata: {0 (unseen, fallback=global), 1, 2-5, 6+}.
    4. Reuse the nb2103 117-col 5-way K-tuned feature matrix and the
       SHAP-pruned K=28 top column indices (full_rank_order[:28] from
       nb2063_shap_importance_full117.npy).
    5. Train per-stratum LGBM(MSE) residual-correctors on TRAIN at K=28:
         - stratum-0 model trained on ALL 4139 train rows (fallback for
           unseen scaffolds)
         - stratum-1 trained on train rows with scaf_train_freq==1 (~3532)
         - stratum-2-5 trained on train rows with 2<=freq<=5 (~315)
         - stratum-6+ trained on train rows with freq>=6 (~292)
       Residual target = pec50_train - chemprop_aux_train_oof.
    6. Honest 5-fold scaffold cross-fit ON 253 UNB:
         For each fold (KFold scaffold-aware on 253), refit per-stratum LGBMs
         on the TRAIN-fold of 253 (still routed by per-row stratum), predict
         the held-out fold.  Pooled OOF = honest cross-fit RAE.
       NOTE: We instead use 5-fold KFold(seed=k) on the 253 (random, same as
       nb2103) because the 253-only scaffold groups are too sparse for
       stratified scaffold-CV with 4 strata.
    7. 5-seed bag (seeds 0, 1, 7, 42, 137).  Final per-test pred =
       chemprop_aux + mean-bag-routed-residual.
    8. Compare vs nb2103 K=28 (mean-bag 0.4737 / median-bag 0.4698) at
       decision_margin = 0.003.
    9. Report stratum-0 (rare/unseen, n_unb=197) subset RAE.
   10. If beats overall AND beats rare subset, build deploy CSV.

Outputs:
    scripts/nb1091_scaf_strat.py
    data/processed/nb1091_summary.json
    data/processed/nb1091_mean_bag_oof.npy  (253,) float32
    data/processed/te_nb1091.npy            (513,) float32   if winner
    submissions/nb1091_scaf_strat.csv                         if winner
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
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1091"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ANCHOR_OOF_PATH = DATA_PROCESSED / "oof_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
K_USE = 28
DECISION_MARGIN = 0.003

# References from memory
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698

# Feature cache paths (same as nb2103)
ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6


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


def _safe_scaffold(smi: str) -> str | None:
    m = standardize(smi)
    if m is None:
        return None
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc)
    except Exception:
        return None


def _load_chembl_pool() -> pd.DataFrame:
    """Identical to nb2103."""
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


def _load_npy(path: Path, n_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} vs n={n_expected}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred(path: Path, n_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} vs n={n_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _extract_atompair_top_idx(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found")


def _extract_best_K_record(sum_dict: dict, records_key: str) -> dict:
    best_K = int(sum_dict["best_K"])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


def assign_stratum(freq: int) -> int:
    """0=unseen, 1=singleton, 2=2-5, 3=6+ family."""
    if freq <= 0:
        return 0
    if freq == 1:
        return 1
    if 2 <= freq <= 5:
        return 2
    return 3


STRATUM_NAMES = {0: "unseen(0)", 1: "singleton(1)", 2: "small(2-5)", 3: "large(6+)"}


def _fit_stratum_models(X_tr: np.ndarray, resid_tr: np.ndarray,
                        strata_tr: np.ndarray, seed: int):
    """Fit one LGBM per stratum on train.
    Stratum-0 (unseen) uses ALL rows (fallback / global).
    Stratum 1/2/3 train only on rows of that stratum.
    """
    models: dict[int, lgb.LGBMRegressor] = {}
    n_per: dict[int, int] = {}
    # Stratum-0 = global fallback (use all rows)
    mdl0 = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl0.fit(X_tr, resid_tr)
    models[0] = mdl0
    n_per[0] = int(len(X_tr))
    for k in (1, 2, 3):
        mask = strata_tr == k
        n_per[k] = int(mask.sum())
        if mask.sum() < 20:  # too few rows: fall back to global
            models[k] = mdl0
        else:
            mk = lgb.LGBMRegressor(**_lgbm_params(seed))
            mk.fit(X_tr[mask], resid_tr[mask])
            models[k] = mk
    return models, n_per


def _predict_stratum(models, X_va: np.ndarray, strata_va: np.ndarray) -> np.ndarray:
    out = np.empty(len(X_va), dtype=np.float64)
    for i in range(len(X_va)):
        out[i] = models[int(strata_va[i])].predict(X_va[i:i+1])[0]
    return out


def _cross_fit_one_seed(X_unb: np.ndarray, residual_unb: np.ndarray,
                        strata_unb: np.ndarray, seed: int) -> np.ndarray:
    n = len(residual_unb)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        models, _ = _fit_stratum_models(
            X_unb[tr_loc], residual_unb[tr_loc], strata_unb[tr_loc], seed
        )
        oof[va_loc] = _predict_stratum(models, X_unb[va_loc], strata_unb[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 4-stratum scaffold-freq LGBM (K={K_USE}, 5-seed bag)")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean_bag {NB2103_K28_MEAN_BAG_REF:.4f} "
          f"/ median_bag {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load data ----
    tr = load_train()
    te = load_test()
    n_train = len(tr)
    n_test = len(te)
    print(f"[load] n_train={n_train}  n_test={n_test}")

    y_train = tr["pec50"].astype(float).to_numpy()
    train_smiles = tr["smiles"].astype(str).tolist()
    test_smiles = te["smiles"].astype(str).tolist()

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Compute Bemis-Murcko scaffolds and freq ----
    print("[scaf] computing Bemis-Murcko scaffolds ...")
    train_sc = [_safe_scaffold(s) for s in train_smiles]
    test_sc = [_safe_scaffold(s) for s in test_smiles]
    freq_map: dict[str, int] = {}
    for s in train_sc:
        if s is None:
            continue
        freq_map[s] = freq_map.get(s, 0) + 1

    train_freq = np.array(
        [freq_map.get(s, 0) for s in train_sc], dtype=int
    )
    test_freq = np.array(
        [freq_map.get(s, 0) for s in test_sc], dtype=int
    )
    strata_train = np.array([assign_stratum(f) for f in train_freq], dtype=int)
    strata_test = np.array([assign_stratum(f) for f in test_freq], dtype=int)
    strata_unb = strata_test[unb_idx]
    print(f"[scaf] TRAIN strata: 0(unseen)={int((strata_train==0).sum())}  "
          f"1(single)={int((strata_train==1).sum())}  "
          f"2-5={int((strata_train==2).sum())}  "
          f"6+={int((strata_train==3).sum())}")
    print(f"[scaf] TEST  strata: 0={int((strata_test==0).sum())}  "
          f"1={int((strata_test==1).sum())}  "
          f"2-5={int((strata_test==2).sum())}  "
          f"6+={int((strata_test==3).sum())}")
    print(f"[scaf] UNB   strata: 0={int((strata_unb==0).sum())}  "
          f"1={int((strata_unb==1).sum())}  "
          f"2-5={int((strata_unb==2).sum())}  "
          f"6+={int((strata_unb==3).sum())}")

    # ---- Load anchor ----
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape} vs {n_test}")
    oof_anchor_train = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    if oof_anchor_train.shape[0] != n_train:
        raise ValueError(
            f"anchor oof shape {oof_anchor_train.shape} vs {n_train}"
        )
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual_unb = y_unb - anchor_unb
    residual_train = y_train - oof_anchor_train
    print(f"[resid] unb   mean={residual_unb.mean():+.4f} "
          f"std={residual_unb.std():.4f}")
    print(f"[resid] train mean={residual_train.mean():+.4f} "
          f"std={residual_train.std():.4f}")

    # ---- Load K-tuned feature index sets (same as nb2103) ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY,
              NB2063_SHAP_IMP):
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

    top_maccs = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
    top_mord = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap = _extract_atompair_top_idx(sum_1484)
    top_ap = full_ap[: int(sum_1524["best_K"])]
    top_embed = np.array(sum_1541["top_dim_order_top100"], dtype=int)[
        : int(sum_1541["best_K"])
    ]
    top_avalon = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

    shap_imp = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp).astype(np.int32)
    topK_idx = full_rank_order[:K_USE]
    print(f"[reuse] K={K_USE} cols (SHAP top); 117-col base feat dim")

    # ---- Build feature matrices on TRAIN and TEST ----
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_train)
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test)
    X_mc_tr = _load_npy(MACCS_TR_PATH, n_train)
    X_mc_te = _load_npy(MACCS_TE_PATH, n_test)
    X_em_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_train)
    X_em_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test)
    X_av_tr = _load_npy(AVALON_TR_PATH, n_train)
    X_av_te = _load_npy(AVALON_TE_PATH, n_test)
    X_md_tr = _load_mordred(MORDRED_DIR / "X_mordred_train.npy", n_train)
    X_md_te = _load_mordred(MORDRED_DIR / "X_mordred_test.npy", n_test)

    # Slice by top-K per family (matches the 117-col base)
    X_ap_tr_top = X_ap_tr[:, top_ap]
    X_ap_te_top = X_ap_te[:, top_ap]
    X_mc_tr_top = X_mc_tr[:, top_maccs]
    X_mc_te_top = X_mc_te[:, top_maccs]
    X_md_tr_top = X_md_tr[:, top_mord]
    X_md_te_top = X_md_te[:, top_mord]
    X_em_tr_top = X_em_tr[:, top_embed]
    X_em_te_top = X_em_te[:, top_embed]
    X_av_tr_top = X_av_tr[:, top_avalon]
    X_av_te_top = X_av_te[:, top_avalon]

    # ---- ChEMBL kNN feature ----
    print("[chembl] building ChEMBL kNN feature ...")
    pool = _load_chembl_pool()
    train_mols = [standardize(s) for s in train_smiles]
    test_mols = [standardize(s) for s in test_smiles]
    train_inchikeys = set()
    for m in train_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            train_inchikeys.add(ik)
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    # Drop pool rows overlapping with train OR test (no leakage)
    pool = pool[~pool["inchikey"].isin(train_inchikeys | test_inchikeys)] \
        .reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep = fp_pool.sum(axis=1) > 0
    if not keep.all():
        pool = pool[keep].reset_index(drop=True)
        fp_pool = fp_pool[keep]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   pool size (after dedup): {len(pool)}  median pEC50 = "
          f"{pool_median:.3f}")

    std_train_smiles = [
        Chem.MolToSmiles(m) if m is not None else "" for m in train_mols
    ]
    std_test_smiles = [
        Chem.MolToSmiles(m) if m is not None else "" for m in test_mols
    ]
    fp_train = morgan_fp_batch(std_train_smiles)
    fp_test = morgan_fp_batch(std_test_smiles)
    ti_tr, ts_tr = _tanimoto_topk(fp_train, fp_pool, k=KNN_K)
    ti_te, ts_te = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_tr, sim_tr = _knn_predict(ti_tr, ts_tr, pool_labels, pool_median)
    pred_chembl_te, sim_te = _knn_predict(ti_te, ts_te, pool_labels, pool_median)

    # ---- Concatenate the 117-col base, then slice top-K=28 ----
    X_tr_full = np.concatenate(
        [X_ap_tr_top, X_mc_tr_top, X_md_tr_top, X_em_tr_top, X_av_tr_top,
         pred_chembl_tr.reshape(-1, 1).astype(np.float32),
         sim_tr.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    X_te_full = np.concatenate(
        [X_ap_te_top, X_mc_te_top, X_md_te_top, X_em_te_top, X_av_te_top,
         pred_chembl_te.reshape(-1, 1).astype(np.float32),
         sim_te.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)

    if X_tr_full.shape[1] != shap_imp.shape[0]:
        raise ValueError(
            f"feat dim mismatch: train {X_tr_full.shape} vs SHAP "
            f"{shap_imp.shape}"
        )

    X_tr_K = X_tr_full[:, topK_idx]
    X_te_K = X_te_full[:, topK_idx]
    X_unb_K = X_te_K[unb_idx]
    print(f"[feat] X_tr_K={X_tr_K.shape}  X_te_K={X_te_K.shape}  "
          f"X_unb_K={X_unb_K.shape}")

    # =============================================================
    # 1) Honest 5-fold cross-fit on 253 UNB, 5-seed bag
    # =============================================================
    print("\n" + "-" * 78)
    print("CROSS-FIT on 253 UNB (5-fold KFold per seed, 5-seed bag)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae = []
    per_seed_records = []
    per_stratum_rae_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _cross_fit_one_seed(X_unb_K, residual_unb, strata_unb, s)
        pred_corr_s = anchor_unb + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        # Per-stratum RAE this seed
        sub = {}
        for k in (0, 1, 2, 3):
            mask = strata_unb == k
            if mask.sum() >= 5:
                sub[STRATUM_NAMES[k]] = {
                    "n": int(mask.sum()),
                    "rae": float(rae(y_unb[mask], pred_corr_s[mask])),
                    "anchor_rae": float(rae(y_unb[mask], anchor_unb[mask])),
                }
            else:
                sub[STRATUM_NAMES[k]] = {
                    "n": int(mask.sum()),
                    "rae": None,
                    "anchor_rae": float(rae(y_unb[mask], anchor_unb[mask]))
                    if mask.sum() else None,
                }
        per_stratum_rae_records.append(sub)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_anchor": rae_s - rae_anchor,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:3d}: rae={rae_s:.4f}  "
              f"(d_vs_anchor={rae_s - rae_anchor:+.4f})  "
              f"wall={time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_rae_arr = np.array(per_seed_rae)
    print(f"\n   pooled mean_bag  = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor={rae_mean_bag - rae_anchor:+.4f}  "
          f"d_vs_nb2103_K28={rae_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"   pooled median_bag = {rae_median_bag:.4f}  "
          f"(d_vs_nb2103_K28_median={rae_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")

    # Per-stratum RAE on mean_bag
    stratum_summary = {}
    for k in (0, 1, 2, 3):
        mask = strata_unb == k
        if mask.sum() == 0:
            continue
        rae_k = float(rae(y_unb[mask], mean_bag_oof[mask])) \
            if mask.sum() >= 2 else None
        anc_k = float(rae(y_unb[mask], anchor_unb[mask])) \
            if mask.sum() >= 2 else None
        nb2103_k28_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy")
        nb2103_k = float(rae(y_unb[mask], nb2103_k28_oof[mask])) \
            if mask.sum() >= 2 else None
        stratum_summary[STRATUM_NAMES[k]] = {
            "n": int(mask.sum()),
            "rae_nb1091": rae_k,
            "rae_chemprop_aux": anc_k,
            "rae_nb2103_K28": nb2103_k,
        }
        print(f"   stratum {STRATUM_NAMES[k]:<14s} n={int(mask.sum()):3d}  "
              f"nb1091={rae_k if rae_k is None else f'{rae_k:.4f}':<8} "
              f"chemprop={anc_k if anc_k is None else f'{anc_k:.4f}':<8} "
              f"nb2103_K28={nb2103_k if nb2103_k is None else f'{nb2103_k:.4f}'}")

    # ---- Compare vs nb2103 K=28 ----
    beats_nb2103_mean = rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    beats_nb2103_median = rae_median_bag < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN
    flat_mean = abs(rae_mean_bag - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN
    flat_median = abs(rae_median_bag - NB2103_K28_MEDIAN_BAG_REF) < DECISION_MARGIN

    # Rare subset RAE check: stratum 0 (n_unb=197) is the dominant failure cluster
    rare_mask = strata_unb == 0
    rae_nb1091_rare = float(rae(y_unb[rare_mask], mean_bag_oof[rare_mask]))
    nb2103_k28_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy")
    rae_nb2103_rare = float(rae(y_unb[rare_mask], nb2103_k28_oof[rare_mask]))
    rae_anchor_rare = float(rae(y_unb[rare_mask], anchor_unb[rare_mask]))
    beats_rare = rae_nb1091_rare < rae_nb2103_rare - DECISION_MARGIN

    print(f"\n   RARE subset (stratum=0 unseen, n={int(rare_mask.sum())}):")
    print(f"      chemprop_aux   RAE = {rae_anchor_rare:.4f}")
    print(f"      nb2103 K=28    RAE = {rae_nb2103_rare:.4f}")
    print(f"      nb1091 (4-str) RAE = {rae_nb1091_rare:.4f}  "
          f"(d={rae_nb1091_rare - rae_nb2103_rare:+.4f}, "
          f"beats={beats_rare})")

    if beats_nb2103_mean:
        verdict_overall = "BEATS_NB2103_K28_MEAN_BAG"
    elif beats_nb2103_median:
        verdict_overall = "BEATS_NB2103_K28_MEDIAN_BAG_ONLY"
    elif flat_mean or flat_median:
        verdict_overall = "FLAT_VS_NB2103_K28"
    else:
        verdict_overall = "WORSE_THAN_NB2103_K28"
    print(f"   verdict (overall)    = {verdict_overall}")

    # ---- Deploy if winner ----
    deploy_built = False
    deploy_path = None
    if (beats_nb2103_mean or beats_nb2103_median) and beats_rare:
        print("\n   [WINNER] beats nb2103 K=28 (overall + rare subset) -- "
              "building deploy.")
        # Refit per-stratum on FULL train residual (4139), 5-seed bag, predict
        # 513 test, route by per-test stratum.
        te_preds_seeds = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            models, _ = _fit_stratum_models(
                X_tr_K, residual_train.astype(np.float64),
                strata_train, s
            )
            te_preds_seeds[i] = _predict_stratum(models, X_te_K, strata_test)
        te_resid_mean = te_preds_seeds.mean(axis=0)
        te_pred_513 = te_anchor_513 + te_resid_mean
        te_out = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_out, te_pred_513.astype(np.float32))
        # CSV
        sub_df = pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": te_pred_513,
        })
        sub_path = SUBMISSIONS / f"{TAG}_scaf_strat.csv"
        sub_df.to_csv(sub_path, index=False)
        deploy_built = True
        deploy_path = str(sub_path)
        print(f"   [save] {te_out}")
        print(f"   [save] {sub_path}")
    else:
        print("\n   [no deploy] does not beat nb2103 K=28 on overall AND rare.")

    # ---- Save mean-bag OOF ----
    out_oof = DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy"
    np.save(out_oof, mean_bag_oof.astype(np.float32))
    print(f"[save] {out_oof}")

    summary = {
        "tag": TAG,
        "method": "lgbm_mse_4stratum_scaf_freq_K28_5seed_bag",
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "K_use": int(K_USE),
        "stratum_definition": {
            "0_unseen": "scaf_train_freq == 0 (test only); model = global "
                        "fallback trained on ALL rows",
            "1_singleton": "scaf_train_freq == 1",
            "2_small": "2 <= scaf_train_freq <= 5",
            "3_large": "scaf_train_freq >= 6",
        },
        "train_stratum_counts": {
            "0_unseen": int((strata_train == 0).sum()),
            "1_singleton": int((strata_train == 1).sum()),
            "2_small": int((strata_train == 2).sum()),
            "3_large": int((strata_train == 3).sum()),
        },
        "test_stratum_counts": {
            "0_unseen": int((strata_test == 0).sum()),
            "1_singleton": int((strata_test == 1).sum()),
            "2_small": int((strata_test == 2).sum()),
            "3_large": int((strata_test == 3).sum()),
        },
        "unb_stratum_counts": {
            "0_unseen": int((strata_unb == 0).sum()),
            "1_singleton": int((strata_unb == 1).sum()),
            "2_small": int((strata_unb == 2).sum()),
            "3_large": int((strata_unb == 3).sum()),
        },
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_unb_mean": float(residual_unb.mean()),
        "residual_unb_std": float(residual_unb.std()),
        "residual_train_mean": float(residual_train.mean()),
        "residual_train_std": float(residual_train.std()),
        "per_seed_rae": [float(x) for x in per_seed_rae],
        "per_seed_records": per_seed_records,
        "per_seed_per_stratum_rae": per_stratum_rae_records,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "rae_per_seed_mean": float(per_seed_rae_arr.mean()),
        "rae_per_seed_std": float(per_seed_rae_arr.std()),
        "rae_per_seed_min": float(per_seed_rae_arr.min()),
        "rae_per_seed_max": float(per_seed_rae_arr.max()),
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb2103_K28": rae_mean_bag - NB2103_K28_MEAN_BAG_REF,
        "delta_median_bag_vs_nb2103_K28": (
            rae_median_bag - NB2103_K28_MEDIAN_BAG_REF
        ),
        "beats_nb2103_K28_mean": bool(beats_nb2103_mean),
        "beats_nb2103_K28_median": bool(beats_nb2103_median),
        "flat_vs_nb2103_K28_mean": bool(flat_mean),
        "stratum_rae_mean_bag": stratum_summary,
        "rare_subset_rae": {
            "n": int(rare_mask.sum()),
            "rae_chemprop_aux": rae_anchor_rare,
            "rae_nb2103_K28": rae_nb2103_rare,
            "rae_nb1091": rae_nb1091_rare,
            "delta_vs_nb2103_K28": rae_nb1091_rare - rae_nb2103_rare,
            "beats_nb2103_K28_on_rare": bool(beats_rare),
        },
        "verdict": verdict_overall,
        "deploy_built": deploy_built,
        "deploy_path": deploy_path,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall={time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K_use", "rae_anchor_chemprop_aux",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb2103_K28",
        "delta_median_bag_vs_nb2103_K28",
        "beats_nb2103_K28_mean", "beats_nb2103_K28_median",
        "verdict", "deploy_built", "deploy_path",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== RARE SUBSET (stratum=0 unseen) ====")
    rs = res["rare_subset_rae"]
    print(f"  n={rs['n']}  chemprop={rs['rae_chemprop_aux']:.4f}  "
          f"nb2103={rs['rae_nb2103_K28']:.4f}  nb1091={rs['rae_nb1091']:.4f}  "
          f"delta_vs_nb2103={rs['delta_vs_nb2103_K28']:+.4f}  "
          f"beats={rs['beats_nb2103_K28_on_rare']}")
    print("\n==== PER-STRATUM RAE (mean_bag) ====")
    for k, v in res["stratum_rae_mean_bag"].items():
        rae_n = v["rae_nb1091"]
        rae_a = v["rae_chemprop_aux"]
        rae_2 = v["rae_nb2103_K28"]
        rae_n_s = "N/A" if rae_n is None else f"{rae_n:.4f}"
        rae_a_s = "N/A" if rae_a is None else f"{rae_a:.4f}"
        rae_2_s = "N/A" if rae_2 is None else f"{rae_2:.4f}"
        print(f"  {k:<14s} n={v['n']:>3d}  "
              f"nb1091={rae_n_s:<8}  chemprop={rae_a_s:<8}  "
              f"nb2103_K28={rae_2_s}")
