"""nb1981 -- LGBM with binary-bit cols marked as categorical_feature.

HYPOTHESIS:
    nb1861 (bag of nb1852, LGBM regression/MSE on 117-col 5-way K-tuned matrix
    with chemprop_aux anchor) lands at BoB MEAN RAE 0.5078 / pooled-25bag 0.5013.
    The 117 columns split: 75 BINARY-bit cols (AtomPair=25, MACCS=20, Avalon=30)
    and 42 CONTINUOUS cols (Mordred=20, ChempropEmbed=20, pred_chembl_pec50=1,
    mean_sim=1). LightGBM treats binary 0/1 columns as continuous numeric by
    default. Different from CatBoost cat_features, LightGBM's native
    categorical_feature handling uses one-vs-rest ordered partitioning that may
    give cleaner splits on sparse Bernoulli bits.

    Test: mark the 75 binary-bit indices via `categorical_feature=<list>` at
    .fit() time and see whether 5-seed bag / 5-fold cross-fit beats nb1861's
    0.5013 baseline (pooled) or 0.5078 (single-outer mean-bag) reference.

PROTOCOL:
    1. Build 117-col 5-way K-tuned matrix (identical to nb1861/nb1852/nb1771).
    2. Anchor = chemprop_aux te[unb_idx]   (PRE-unblind, in_RAE 0.6216).
       residual = y_unb - anchor.
    3. Compute binary_indices_list = [0..n_AP) U
                                     [n_AP, n_AP+n_MACCS) U
                                     [n_AP+n_MACCS+n_Mord+n_Embed,
                                      n_AP+n_MACCS+n_Mord+n_Embed+n_Av)
       (positions of AP + MACCS + Avalon bits inside the 117-col stack).
    4. LGBM(regression/MSE, max_depth=4, num_leaves=15, n_est=300, lr=0.03,
       min_child_samples=5, reg_lambda=2, random_state=seed) fit with
       categorical_feature=binary_indices_list at .fit() time.
    5. 5 inner seeds {0,1,2,3,4}, each running 5-fold KFold cross-fit
       (shuffle=True, random_state=seed); each seed yields one OOF vector.
    6. mean_bag_oof = mean(5 inner OOFs).
    7. Verdict vs nb1861 (0.5013).

Outputs:
    scripts/nb1981_lgbm_categorical.py
    data/processed/nb1981_summary.json
    data/processed/nb1981_per_seed_oof.npy   (5, 253) float32
    data/processed/nb1981_mean_bag_oof.npy   (253,) float32
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

TAG = "nb1981"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

SEEDS = [0, 1, 2, 3, 4]
RESID_FOLDS = 5

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

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB1861_REF = 0.5013  # nb1861 pooled-25bag baseline
NB1861_BOB_MEAN_REF = 0.5078
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
    """LGBM(regression/MSE) -- same hyperparams as nb1852/nb1861."""
    return dict(
        objective="regression",     # standard MSE / L2
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


def _residual_cross_fit_one_seed_with_categorical(
    X: np.ndarray, residual: np.ndarray, seed: int,
    cat_indices: list[int],
) -> np.ndarray:
    """5-fold cross-fit LGBM with categorical_feature=cat_indices.

    Uses Pandas DataFrame so LightGBM's categorical_feature index list is
    respected (sklearn API accepts a list of ints when X is array-like, but
    DataFrame with explicit dtype 'category' is the cleanest and most robust
    way to mark Bernoulli bits as categorical).
    """
    n, p = X.shape
    feat_names = [f"f{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=feat_names)
    # Cast each binary column to pandas categorical
    for ci in cat_indices:
        df.iloc[:, ci] = df.iloc[:, ci].astype("int8").astype("category")

    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    cat_feat_names = [feat_names[ci] for ci in cat_indices]
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(
            df.iloc[tr_loc], residual[tr_loc],
            categorical_feature=cat_feat_names,
        )
        oof[va_loc] = mdl.predict(df.iloc[va_loc])
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


def _extract_embed_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "ChempropEmbed":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("ChempropEmbed entry not found in nb1484_summary.json")


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
    print(f"{TAG} -- LGBM with binary-bit cols marked as categorical_feature; "
          f"5-seed bag, 5-fold cross-fit; PRE-unblind anchor={ANCHOR}")
    print(f"          seeds = {SEEDS}  folds = {RESID_FOLDS}")
    print(f"          refs: chemprop_aux ({CHEMPROP_AUX_REF:.4f}), "
          f"nb1861 pooled ({NB1861_REF:.4f}), "
          f"nb1861 BoB mean ({NB1861_BOB_MEAN_REF:.4f})")
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

    # Build the 117-col stack in the SAME order as nb1861/nb1852.
    # Order: [AP=25][MACCS=20][Mord=20][Embed=20][Av=30][chembl=1][sim=1]
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

    # ---- Build the binary indices list ----
    # AP bits occupy [0, n_top_ap)
    # MACCS bits occupy [n_top_ap, n_top_ap + n_top_maccs)
    # Mordred occupies [n_top_ap+n_top_maccs, n_top_ap+n_top_maccs+n_top_mord)
    # Embed occupies [+n_top_mord, +n_top_embed)
    # Avalon bits occupy [..., n_top_ap+n_top_maccs+n_top_mord+n_top_embed,
    #                     ...+n_top_avalon)
    ap_start = 0
    maccs_start = n_top_ap
    mord_start = maccs_start + n_top_maccs
    embed_start = mord_start + n_top_mord
    avalon_start = embed_start + n_top_embed
    extras_start = avalon_start + n_top_avalon

    binary_indices_list: list[int] = []
    binary_indices_list.extend(range(ap_start, ap_start + n_top_ap))
    binary_indices_list.extend(range(maccs_start, maccs_start + n_top_maccs))
    binary_indices_list.extend(range(avalon_start, avalon_start + n_top_avalon))
    n_binary = len(binary_indices_list)
    n_continuous = feat_dim - n_binary
    print(f"\n   binary indices (AP + MACCS + Avalon) : n={n_binary}  "
          f"first10={binary_indices_list[:10]}  last5="
          f"{binary_indices_list[-5:]}")
    print(f"   continuous indices                   : n={n_continuous}  "
          f"(Mordred + ChempropEmbed + chembl + sim)")
    if n_binary != 75:
        raise ValueError(f"expected 75 binary cols but got {n_binary}")
    if n_continuous != 42:
        raise ValueError(f"expected 42 continuous cols but got {n_continuous}")

    # Sanity: AP / MACCS / Avalon should all be {0, 1}-valued
    for label, sl_start, sl_n in [("AP", ap_start, n_top_ap),
                                  ("MACCS", maccs_start, n_top_maccs),
                                  ("Avalon", avalon_start, n_top_avalon)]:
        block = X_unb[:, sl_start:sl_start + sl_n]
        u = np.unique(block)
        print(f"   sanity[{label}]  block shape={block.shape}  "
              f"n_unique_vals={len(u)}  example_first5_unique={u[:5]}")

    # ---- 5-SEED BAG / 5-FOLD CROSS-FIT ----
    print("\n" + "-" * 78)
    print(f"5-SEED BAG x {RESID_FOLDS}-fold cross-fit  (dim={feat_dim})  "
          f"categorical_feature -> {n_binary} bits")
    print("-" * 78)

    per_seed_oof = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    for i, seed in enumerate(SEEDS):
        t_s = time.time()
        resid_oof_i = _residual_cross_fit_one_seed_with_categorical(
            X_unb, residual, seed, cat_indices=binary_indices_list,
        )
        pred_corr_i = anchor + resid_oof_i
        per_seed_oof[i] = pred_corr_i
        r_i = float(rae(y_unb, pred_corr_i))
        per_seed_rae.append(r_i)
        print(f"   seed={seed:4d}  rae={r_i:.4f}  "
              f"wall={time.time() - t_s:.1f}s")

    mean_bag_oof = per_seed_oof.mean(axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))

    arr = np.array(per_seed_rae)
    seeds_mean = float(arr.mean())
    seeds_median = float(np.median(arr))
    seeds_std = float(arr.std())
    seeds_min = float(arr.min())
    seeds_max = float(arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list   = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed MEAN RAE   = {seeds_mean:.4f}")
    print(f"   per-seed MEDIAN RAE = {seeds_median:.4f}")
    print(f"   per-seed std/min/max= {seeds_std:.4f} / {seeds_min:.4f} / "
          f"{seeds_max:.4f}")
    print(f"   MEAN-BAG OOF  RAE   = {rae_mean_bag:.4f}    "
          f"<-- primary verdict number")
    print(f"   nb1861 pooled ref   = {NB1861_REF:.4f}")
    print(f"   nb1861 BoB mean ref = {NB1861_BOB_MEAN_REF:.4f}")
    print(f"   d(mean_bag, nb1861_pooled)  = "
          f"{rae_mean_bag - NB1861_REF:+.4f}")
    print(f"   d(mean_bag, nb1861_bob_mean)= "
          f"{rae_mean_bag - NB1861_BOB_MEAN_REF:+.4f}")

    # ---- Verdict ----
    delta_vs_nb1861 = rae_mean_bag - NB1861_REF
    if rae_mean_bag < NB1861_REF - DECISION_MARGIN:
        verdict = "BEATS_NB1861_POOLED"
    elif rae_mean_bag > NB1861_REF + DECISION_MARGIN:
        verdict = "WORSE_THAN_NB1861_POOLED"
    else:
        verdict = "TIES_NB1861_POOLED"

    print(f"   verdict             = {verdict}  "
          f"(decision_margin = {DECISION_MARGIN})")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_oof.npy",
            per_seed_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_oof.npy'}")
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

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
        "lgbm_reg_lambda": 2.0,
        "seeds": SEEDS,
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
        "n_binary_cols_marked_categorical": n_binary,
        "n_continuous_cols": n_continuous,
        "binary_indices_list": binary_indices_list,
        "binary_block_starts": {
            "ap_start": ap_start,
            "maccs_start": maccs_start,
            "avalon_start": avalon_start,
        },
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "rae_mean_bag": rae_mean_bag,
        "seeds_mean_rae": seeds_mean,
        "seeds_median_rae": seeds_median,
        "seeds_std_rae": seeds_std,
        "seeds_min_rae": seeds_min,
        "seeds_max_rae": seeds_max,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1861_pooled": delta_vs_nb1861,
        "delta_mean_bag_vs_nb1861_bob_mean":
            rae_mean_bag - NB1861_BOB_MEAN_REF,
        "verdict": verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1861_pooled_ref": NB1861_REF,
        "nb1861_bob_mean_ref": NB1861_BOB_MEAN_REF,
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
        "seeds", "resid_folds",
        "K_AP_best", "K_Mord_best", "K_Embed_best", "K_Avalon_used",
        "n_chembl_pool", "feat_dim",
        "n_binary_cols_marked_categorical", "n_continuous_cols",
        "rae_anchor_chemprop_aux",
        "per_seed_rae",
        "rae_mean_bag",
        "seeds_mean_rae", "seeds_median_rae",
        "seeds_std_rae", "seeds_min_rae", "seeds_max_rae",
        "delta_mean_bag_vs_anchor",
        "delta_mean_bag_vs_nb1861_pooled",
        "delta_mean_bag_vs_nb1861_bob_mean",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
