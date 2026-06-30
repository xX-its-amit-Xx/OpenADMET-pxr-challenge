"""nb3331 -- K=18 residual-LGBM with SCAFFOLD-BALANCED sample weights.

NEW PARADIGM (training-time reweighting, NOT post-hoc operator):
    Every prior K-pyramid candidate fits the residual-LGBM on the 253 unblind
    rows with UNIFORM sample weights.  Because the train manifold is
    scaffold-imbalanced (a handful of large analog families dominate, most
    scaffolds are singletons), the unweighted fit is implicitly biased toward
    the frequent scaffolds -- exactly the OPPOSITE of the test-513 distribution,
    which is 90.5% novel-scaffold tail.

    nb3331 reweights each training row so that each scaffold contributes
    EQUALLY to the loss:

        weight(row) = 1 / scaffold_count(row's scaffold)

    Rare/singleton scaffolds are upweighted to 1.0; a row inside a size-N
    analog family is downweighted to 1/N.  The total mass of every scaffold
    family becomes 1.0, so the LGBM residual fit "sees" a scaffold-uniform
    training distribution that better matches the novel-scaffold test tail.

    Anchor = chemprop_aux te[unb_idx] (the only verified-clean PRE-unblind
    anchor); the K=18 feature slice and LGBM hyperparams are byte-identical to
    nb2604 / nb2960 / nb3110.  The ONLY change vs nb2960-K18 is the per-row
    sample_weight passed to LGBMRegressor.fit().

    Weights are computed PER FOLD on the fold's TRAIN rows only (honest --
    no validation-row scaffold counts leak into the fold's weighting).

PROTOCOL:
    1. Build the 117-col 5-way K-tuned feature matrix (AtomPair / MACCS /
       Mordred / ChempropEmbed / Avalon + ChEMBL kNN), slice to K=18 via
       nb2604's k18_idx_in_117col.
    2. residual = y_unb - chemprop_aux[unb_idx].
    3. For each of 15 fresh kf_seeds {1216..1230}:
         splits = scaffold_kfold_indices(unb_scaffolds, n_splits=5, seed=kf)
         per fold (tr, va):
           w_tr = 1 / scaffold_count_within_tr(scaffold(row))   # rare upweight
           bag = mean over 30 resid-seeds {3001..3030} of
                 LGBM(resid).fit(X[tr], residual[tr], sample_weight=w_tr)
                            .predict(X[va])
           pred_va = clip(anchor[va] + bag, 3.0, 9.0)
           fold_rae = rae(y[va], pred_va)
         per-fold-MEAN RAE = mean(fold_rae over 5 folds)
         pooled OOF = concat of per-fold pred_va  (for pred_oof storage)
    4. Aggregate per-fold-mean RAE across the 15 kf_seeds: mean +/- std + CI.
    5. Deploy te(513): 30-seed bag of LGBM fit on ALL 253 with GLOBAL
       scaffold weights; te = clip(chemprop_aux_te + bag, 3, 9).

GATE (on 15-seed mean of the per-fold-mean RAE):
    mean < 0.4423 -> "BETTER"
    else          -> "FAIL"

References:
    nb2960 K18 deep-30 (random-KFold cross-fit) = 0.4536
    nb3080 5-way K18-K19 quantile-conditional   = 0.4475  (current PRIMARY-1)
    nb2171 5-anchor pyramid post-hoc ceiling     = 0.4682
    chemprop_aux PRE-unblind anchor              = 0.6216

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2604_summary.json            (k18_idx_in_117col)
    data/processed/nb1352/1392/1484/1523/1524/1541_summary.json
    data/processed/te_atompair.npy / te_maccs.npy / te_chemprop_embed_300.npy /
        te_avalon512.npy
    C:/pxr_artifacts/nb1030/X_mordred_test.npy
    data/external/chembl_pxr_*.parquet  (ChEMBL kNN pool)

Outputs:
    data/processed/nb3331_summary.json
    data/processed/nb3331_pred_oof.npy   (253,) float32 -- kf-seed-avg pooled OOF
    data/processed/te_nb3331.npy         (513,) float32 -- deploy te
    submissions/nb3331_scaffold_split_train.csv  (only on BETTER)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter
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
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3331"
PARENT_TAG = "nb2960_K18"

# -- Anchor + residual params (IDENTICAL recipe to nb2604 / nb2960) -----------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

RESID_FOLDS = 5
RESID_SEEDS_DEEP = list(range(3001, 3031))   # 30 fresh resid-LGBM seeds

# -- Feature cache paths ------------------------------------------------------
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
NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"

# -- ChEMBL kNN params (identical to nb2604 / nb2960) -------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- CV protocol --------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))   # 15 fresh seeds {1216..1230}

# -- Prediction clip ----------------------------------------------------------
CLIP_LO = 3.0
CLIP_HI = 9.0

# -- Gate ---------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ---------------------------------------------------------------
REF_K18_DEEP30 = 0.4536        # nb2960 K18 random-KFold cross-fit
REF_NB3080 = 0.4475            # current PRIMARY-1
REF_NB2171 = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (feature build lifted verbatim from nb2960)
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


def _load_chembl_pool():
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


def _lgbm_params(seed):
    """LGBM(MSE) -- byte-identical to nb2604 / nb2960 / nb3110."""
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


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing (run nb1030): {mte_p}")
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


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
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


def build_117col_feature_matrix(te_smiles, n_test):
    """117-col matrix identical to nb2604 / nb2960."""
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
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full, int(len(pool))


# ============================================================================
# scaffold-balanced sample weights
# ============================================================================

def scaffold_balance_weights(scaffolds_subset: list[str]) -> np.ndarray:
    """weight(row) = 1 / count(scaffold of row) within this subset.

    Rare / singleton scaffolds -> weight 1.0; a row inside a size-N family ->
    1/N.  Each distinct scaffold thus contributes total mass 1.0 to the loss,
    so the LGBM fit sees a scaffold-uniform training distribution that better
    matches the novel-scaffold test tail.

    Empty-string scaffolds (ring-free / failed parse) are treated as distinct
    singletons keyed by position so they are NOT collapsed into one giant
    bucket (which would crush every ring-free molecule to weight 1/M).
    """
    keyed = [
        sc if sc else f"__singleton_{i}__"
        for i, sc in enumerate(scaffolds_subset)
    ]
    counts = Counter(keyed)
    w = np.array([1.0 / counts[k] for k in keyed], dtype=np.float64)
    return w


def _bag_predict_weighted(
    X_tr: np.ndarray,
    resid_tr: np.ndarray,
    w_tr: np.ndarray,
    X_pred: np.ndarray,
    seeds: list[int],
) -> np.ndarray:
    """Mean over `seeds` of LGBM(resid).fit(X_tr, resid_tr, sample_weight=w_tr)
    .predict(X_pred).  Returns (len(X_pred),) float64 residual prediction."""
    acc = np.zeros(X_pred.shape[0], dtype=np.float64)
    for s in seeds:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_tr, resid_tr, sample_weight=w_tr)
        acc += mdl.predict(X_pred)
    return acc / len(seeds)


def _run_one_kf_seed(
    X_unb_K: np.ndarray,
    residual: np.ndarray,
    anchor: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """One scaffold-CV pass with scaffold-balanced weighted 30-seed bag.

    Returns pooled OOF + per-fold RAE list + per-fold-mean RAE + pooled RAE.
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_rae = []
    per_fold = []
    for fold_i, (tr, va) in enumerate(splits):
        scaf_tr = [unb_scaffolds[i] for i in tr]
        w_tr = scaffold_balance_weights(scaf_tr)
        resid_bag_va = _bag_predict_weighted(
            X_unb_K[tr], residual[tr], w_tr, X_unb_K[va], RESID_SEEDS_DEEP,
        )
        pred_va = np.clip(anchor[va] + resid_bag_va, CLIP_LO, CLIP_HI)
        oof[va] = pred_va
        r = float(rae(y_unb[va], pred_va))
        fold_rae.append(r)
        per_fold.append({
            "fold": int(fold_i),
            "n_tr": int(len(tr)),
            "n_va": int(len(va)),
            "n_scaf_tr": int(len(set(
                s for s in scaf_tr if s
            ))),
            "w_tr_mean": round(float(w_tr.mean()), 4),
            "w_tr_min": round(float(w_tr.min()), 4),
            "w_tr_max": round(float(w_tr.max()), 4),
            "fold_val_rae": round(r, 4),
        })
    if np.isnan(oof).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    per_fold_mean = float(np.mean(fold_rae))
    pooled = float(rae(y_unb, oof))
    return {
        "kf_seed": int(kf_seed),
        "per_fold_mean_rae": per_fold_mean,
        "per_fold_std_rae": float(np.std(fold_rae, ddof=1)),
        "pooled_rae": pooled,
        "fold_rae": [round(x, 4) for x in fold_rae],
        "per_fold": per_fold,
        "oof": oof,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K=18 residual-LGBM with SCAFFOLD-BALANCED sample weights")
    print(f"          weight(row) = 1 / scaffold_count(row)  (rare upweight)")
    print(f"          resid seeds = {RESID_SEEDS_DEEP[0]}..{RESID_SEEDS_DEEP[-1]} "
          f"(n={len(RESID_SEEDS_DEEP)})  + clip[{CLIP_LO},{CLIP_HI}]")
    print(f"          kf_seeds    = {KF_SEEDS[0]}..{KF_SEEDS[-1]} "
          f"(n={len(KF_SEEDS)}), per-fold-mean")
    print(f"          gate: per-fold-mean < {GATE_BETTER} -> BETTER, else FAIL")
    print("=" * 78)

    # -- Load truth, anchor, scaffolds, test --------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"chemprop_aux te shape {te_anchor_513.shape} != {n_test}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # -- Scaffolds (kf_seed-independent) ------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    n_ringfree = int(sum(1 for s in unb_scaffolds if not s))
    # global scaffold-balance weight diagnostics (full 253)
    w_full = scaffold_balance_weights(unb_scaffolds)
    scaf_sizes = Counter(s for s in unb_scaffolds if s)
    print(f"[scaf] n_unique_scaffolds={n_unique_scaf}  ring_free={n_ringfree}")
    print(f"[scaf] largest families: "
          f"{[(k[:18], v) for k, v in scaf_sizes.most_common(5)]}")
    print(f"[weight] full-253 w: mean={w_full.mean():.4f}  "
          f"median={np.median(w_full):.4f}  min={w_full.min():.4f}  "
          f"max={w_full.max():.4f}  "
          f"eff_n={ (w_full.sum()**2)/(w_full**2).sum():.1f}")

    # -- Build 117-col matrix, slice to K18 ---------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: build 117-col 5-way matrix + slice K18")
    print("-" * 78)
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    K18_idx = np.array(nb2604["k18_idx_in_117col"], dtype=int)
    assert len(K18_idx) == 18, f"K18 len {len(K18_idx)} != 18"
    print(f"   K18 idx (n={len(K18_idx)}): {K18_idx.tolist()}")

    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K = X_te_full[:, K18_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx].astype(np.float32)
    print(f"   X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")

    # -- Wide-seed scaffold-CV sweep ----------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 2: scaffold-CV sweep over {len(KF_SEEDS)} fresh kf_seeds "
          f"({KF_SEEDS[0]}..{KF_SEEDS[-1]})")
    print(f"        each fold: weighted 30-seed bag (sample_weight=1/scaf_count)")
    print("-" * 78)
    seed_records = []
    per_fold_means = []
    pooled_raes = []
    oof_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    for i, s in enumerate(KF_SEEDS):
        ts = time.time()
        res = _run_one_kf_seed(
            X_unb_K, residual, anchor, y_unb, unb_scaffolds, s,
        )
        per_fold_means.append(res["per_fold_mean_rae"])
        pooled_raes.append(res["pooled_rae"])
        oof_stack[i] = res["oof"]
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "per_fold_mean_rae": round(res["per_fold_mean_rae"], 4),
            "per_fold_std_rae": round(res["per_fold_std_rae"], 4),
            "pooled_rae": round(res["pooled_rae"], 4),
            "fold_rae": res["fold_rae"],
            "per_fold": res["per_fold"],
        })
        print(f"   kf={s}: per_fold_mean={res['per_fold_mean_rae']:.4f}  "
              f"pooled={res['pooled_rae']:.4f}  "
              f"folds={res['fold_rae']}  wall={time.time()-ts:.1f}s")

    pfm = np.asarray(per_fold_means, dtype=np.float64)
    pld = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(pfm)
    mean_rae = float(pfm.mean())            # GATE metric: mean of per-fold-mean
    std_rae = float(pfm.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145   # n=15, df=14
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(pfm))
    pooled_mean = float(pld.mean())
    pooled_std = float(pld.std(ddof=1)) if n_s > 1 else 0.0

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} fresh kf_seeds)")
    print("-" * 78)
    print(f"   per-fold-mean : mean={mean_rae:.4f}  std={std_rae:.4f}  "
          f"sem={sem:.4f}")
    print(f"   95% CI        : [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   per-fold-mean : median={median_rae:.4f}  "
          f"min={pfm.min():.4f}  max={pfm.max():.4f}")
    print(f"   pooled (ref)  : mean={pooled_mean:.4f}  std={pooled_std:.4f}")

    delta_vs_K18 = mean_rae - REF_K18_DEEP30
    delta_vs_nb3080 = mean_rae - REF_NB3080
    delta_vs_nb2171 = mean_rae - REF_NB2171
    print(f"\n   K18 deep-30 (nb2960) = {REF_K18_DEEP30:.4f}  "
          f"delta = {delta_vs_K18:+.4f}")
    print(f"   nb3080 PRIMARY-1     = {REF_NB3080:.4f}  "
          f"delta = {delta_vs_nb3080:+.4f}")
    print(f"   nb2171 ceiling       = {REF_NB2171:.4f}  "
          f"delta = {delta_vs_nb2171:+.4f}")

    # -- kf-seed-avg pooled OOF (for storage) -------------------------------
    oof_for_save = oof_stack.mean(axis=0).astype(np.float32)
    oof_avg_rae = float(rae(y_unb, oof_for_save.astype(np.float64)))
    print(f"\n   kf-seed-avg pooled OOF RAE = {oof_avg_rae:.4f}  "
          f"(saved as {TAG}_pred_oof.npy)")

    # -- Deploy: 30-seed bag on ALL 253 with GLOBAL scaffold weights --------
    print("\n" + "-" * 78)
    print("STEP 3: deploy te(513) -- 30-seed weighted bag fit on full 253")
    print("-" * 78)
    te_resid_bag = _bag_predict_weighted(
        X_unb_K, residual, w_full, X_te_K, RESID_SEEDS_DEEP,
    )
    te_pred = np.clip(te_anchor_513 + te_resid_bag, CLIP_LO, CLIP_HI).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx].astype(np.float64)))
    # leak sanity
    leak_frac = float(np.mean(np.isclose(te_pred[unb_idx], y_unb, atol=1e-6)))
    print(f"   te mean/std = {te_pred.mean():.3f}/{te_pred.std():.3f}  "
          f"(anchor_te {te_anchor_513.mean():.3f}/{te_anchor_513.std():.3f})")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}  "
          f"(in-sample optimism vs OOF expected)")
    print(f"   te[unb] leak(==truth) = {leak_frac:.4%}")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE nb3331 (K18 + scaffold-balanced sample weights). "
            f"15-seed per-fold-mean {mean_rae:.4f} beats gate {GATE_BETTER:.4f} "
            f"and K18-uniform deep-30 {REF_K18_DEEP30:.4f} by {-delta_vs_K18:.4f}"
            f"; delta vs nb3080 PRIMARY-1 {delta_vs_nb3080:+.4f}. Training-time "
            f"scaffold reweighting (rare-scaffold upweight to match novel-tail "
            f"test) is a substrate-level move that breaks the post-hoc ceiling. "
            f"Recommend deep-30 wide-seed re-verify before locking PRIMARY."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT nb3331. 15-seed per-fold-mean {mean_rae:.4f} above gate "
            f"{GATE_BETTER:.4f} (delta vs K18 {delta_vs_K18:+.4f}, vs nb3080 "
            f"{delta_vs_nb3080:+.4f}). Scaffold-balanced reweighting of the 253 "
            f"residual fit does NOT improve scaffold-CV RAE -- upweighting rare "
            f"scaffolds shifts the fit toward high-variance singletons without "
            f"net OOD gain. Keep nb3080 PRIMARY-1."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_scaffold_split_train.csv"
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
        "method": ("K18_residual_lgbm_scaffold_balanced_sample_weights_"
                   "deep30_clip"),
        "paradigm": ("training_time_scaffold_reweight_1_over_count_not_posthoc"),
        "anchor_base": "chemprop_aux",
        "anchor_pre_unblind": True,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_in_rae": round(rae_anchor, 4),
        "K_label": "K18",
        "K18_idx_in_117col": K18_idx.tolist(),
        "feat_dim": 18,
        "sample_weight_rule": "1 / scaffold_count_within_train_subset",
        "weight_computed_per_fold_on_train_only": True,
        "ringfree_treated_as_singletons": True,
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        "resid_seeds_deep": RESID_SEEDS_DEEP,
        "resid_seeds_deep_n": len(RESID_SEEDS_DEEP),
        "resid_folds": RESID_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_seeds": int(n_s),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_chembl_pool": int(chembl_pool_size),
        "n_unique_scaffolds": int(n_unique_scaf),
        "n_ringfree": int(n_ringfree),
        "scaf_largest_families_top5": [
            [k, int(v)] for k, v in scaf_sizes.most_common(5)
        ],
        "weight_full253_stats": {
            "mean": float(w_full.mean()),
            "median": float(np.median(w_full)),
            "min": float(w_full.min()),
            "max": float(w_full.max()),
            "eff_n": float((w_full.sum() ** 2) / (w_full ** 2).sum()),
        },
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "seed_records": seed_records,
        "per_fold_mean_rae_array": [round(float(v), 4) for v in per_fold_means],
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(pfm.min()), 4),
        "max_rae": round(float(pfm.max()), 4),
        "pooled_rae_mean": round(pooled_mean, 4),
        "pooled_rae_std": round(pooled_std, 4),
        "oof_avg_rae": round(oof_avg_rae, 4),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_leak_eq_truth_frac": round(leak_frac, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "anchor_te_mean": float(te_anchor_513.mean()),
        "anchor_te_std": float(te_anchor_513.std()),
        "ref_K18_deep30": REF_K18_DEEP30,
        "ref_nb3080": REF_NB3080,
        "ref_nb2171": REF_NB2171,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_K18_deep30": round(delta_vs_K18, 4),
        "delta_vs_nb3080": round(delta_vs_nb3080, 4),
        "delta_vs_nb2171": round(delta_vs_nb2171, 4),
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-fold-mean (15 seeds) = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                   = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled (ref)             = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"   delta vs K18 deep-30     = {delta_vs_K18:+.4f}")
    print(f"   delta vs nb3080          = {delta_vs_nb3080:+.4f}")
    print(f"   te[unb] in-sample RAE    = {te_unb_in_rae:.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "pooled_rae_mean",
        "delta_vs_K18_deep30", "delta_vs_nb3080",
        "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
