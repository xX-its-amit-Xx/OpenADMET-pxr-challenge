"""nb1551 -- Outer-bag VALIDATION of nb1550 upgraded 4-way residual blend.

HYPOTHESIS:
    nb1550 is the "upgraded" 4-way nb1484 with the ChempropEmbed slot swapped
    from K=30 -> K=20 (nb1541 chemprop K-grid winner; mean_bag 0.5352 vs the
    original K=30 slot at 0.5401). The other three families stay pinned at the
    nb1484 K config:
        AtomPair      K=30  (top idx from nb1373_summary)
        MACCS         K=20  (top idx from nb1352_summary)
        Mordred       K=30  (top idx from nb1364_summary)
        ChempropEmbed K=20  (first 20 of nb1484/nb1541 ranking -- identical heads)
    Anchor = chemprop_aux te[unb_idx] (PRE-unblind, in_RAE 0.6216 on 253).
    Per-family LGBM Huber (depth=3, leaves=7, n_est=80, lr=0.05, alpha=1.0) on
    [top-K feats + pred_chembl + sim], KFold(n=5, shuffle=True, random_state=o)
    cross-fit residual at random_state=o.
    nb1550_o = naive 1/4 mean over the 4 family-corrected OOFs.

PROTOCOL
    1. For each outer seed o in {0, 1, 7, 42, 137}:
         a. Build 4 family pruned matrices (cols pinned).
         b. Train 4 LGBM Huber residual learners with KFold(o) + random_state=o.
         c. Per-family pred_corr_f = anchor + resid_oof_f.
         d. nb1550_o = mean over 4 family-corrected OOFs.
         e. Per-outer pooled RAE = rae(y_unb, nb1550_o).
    2. Row-level BoB MEAN + MEDIAN across the 5 nb1550_o vectors.
    3. Verdict NB1550_REPRODUCES iff
           |per_outer_mean - nb1550_ref| < 0.003.

    nb1550_ref is the per-outer pooled RAE we would expect from the upgraded
    4-way naive 1/4 mean. Since nb1550 itself was not separately persisted as
    a summary JSON, the reference is taken as nb1484_rae_best (0.5231) bumped
    by the chemprop K=20-vs-K=30 slot delta (-0.0049) -> 0.5182 expectation
    (informational only -- the verdict prints the gap).

Outputs:
    scripts/nb1551_bag_nb1550.py                  (this file)
    data/processed/nb1551_summary.json
    data/processed/nb1551_bob_mean_oof.npy        (253,) float32
    data/processed/nb1551_bob_median_oof.npy      (253,) float32
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
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1551"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

NB1484_REF_BEST = 0.5231              # nb1484 best (naive 1/4 mean, top-30 embed)
NB1484_CHEMPROP_SLOT_K30 = 0.5401     # nb1484 ChempropEmbed slot mean-bag (K=30)
NB1541_CHEMPROP_SLOT_K20 = 0.5352     # nb1541 ChempropEmbed slot mean-bag (K=20)
NB1550_REF_EXPECTED = round(
    NB1484_REF_BEST + (NB1541_CHEMPROP_SLOT_K20 - NB1484_CHEMPROP_SLOT_K30) / 4.0,
    4,
)                                     # informational: expected per-outer pooled RAE
REPRODUCE_MARGIN = 0.003

OUTER_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1364_SUMMARY = DATA_PROCESSED / "nb1364_summary.json"
NB1373_SUMMARY = DATA_PROCESSED / "nb1373_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

FAMILIES = ["AtomPair", "MACCS", "Mordred", "ChempropEmbed"]

# Upgraded K config -- "top-30 AP, top-20 MACCS, top-30 Mord, top-20 chemprop embed"
TOP_K_CONFIG = {
    "AtomPair": 30,
    "MACCS": 20,
    "Mordred": 30,
    "ChempropEmbed": 20,
}


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
    """Same union as nb1501 / nb1484 / nb1521."""
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


# ---- LGBM Huber (nb1484-style) --------------------------------------------
def _lgbm_params(seed: int) -> dict:
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


def _lgbm_residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                      seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
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
            f"Mordred test shape mismatch: {X_te_m.shape} vs n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _extract_embed_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "ChempropEmbed":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("ChempropEmbed entry not found in nb1484_summary.json")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OUTER-BAG VALIDATION of nb1550 upgraded 4-way")
    print(f"         outer seeds        = {OUTER_SEEDS}")
    print(f"         family seed(o)     = o  (KFold split + LGBM random_state)")
    print(f"         top-K config       = {TOP_K_CONFIG}")
    print(f"         nb1550_ref (info)  = {NB1550_REF_EXPECTED}  "
          f"(nb1484 best + chemprop K20 vs K30 slot delta / 4)")
    print(f"         reproduce margin   = {REPRODUCE_MARGIN}")
    print("=" * 78)

    # ---- Pull pinned SHAP top-idx per family ----
    for p in (NB1352_SUMMARY, NB1364_SUMMARY, NB1373_SUMMARY, NB1484_SUMMARY,
              NB1541_SUMMARY):
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
    with open(NB1541_SUMMARY) as f:
        sum_1541 = json.load(f)

    top_maccs_bit_idx_full = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    top_mord_col_idx_full = np.array(
        sum_1364["top_mordred_col_indices_ranked"], dtype=int
    )
    top_ap_bit_idx_full = np.array(
        sum_1373["top_atompair_bit_indices_ranked"], dtype=int
    )
    top_embed_col_idx_full = _extract_embed_top_idx_from_nb1484(sum_1484)

    # Slice to the upgraded K config.
    top_ap_bit_idx = top_ap_bit_idx_full[:TOP_K_CONFIG["AtomPair"]]
    top_maccs_bit_idx = top_maccs_bit_idx_full[:TOP_K_CONFIG["MACCS"]]
    top_mord_col_idx = top_mord_col_idx_full[:TOP_K_CONFIG["Mordred"]]
    top_embed_col_idx = top_embed_col_idx_full[:TOP_K_CONFIG["ChempropEmbed"]]

    # Confirm chemprop top-20 matches nb1541's K=20 SHAP-ranked head.
    nb1541_top20 = np.array(
        sum_1541["top_dim_order_top100"], dtype=int
    )[:TOP_K_CONFIG["ChempropEmbed"]]
    chemprop_match_nb1541 = bool(
        np.array_equal(top_embed_col_idx, nb1541_top20)
    )

    n_top_ap = int(len(top_ap_bit_idx))
    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_embed = int(len(top_embed_col_idx))
    print(f"\n[pin]  top-{n_top_ap}    AtomPair      bits  (from nb1373)")
    print(f"[pin]  top-{n_top_maccs}    MACCS         bits  (from nb1352)")
    print(f"[pin]  top-{n_top_mord}    Mordred       cols  (from nb1364)")
    print(f"[pin]  top-{n_top_embed}    ChempropEmbed cols  (from nb1484 head, "
          f"matches nb1541: {chemprop_match_nb1541})")

    # ---- Load truth + anchor (PRE-unblind chemprop_aux) ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch: {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Per-family pruned slices on 253 ----
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    X_ap_unb = X_ap_te[unb_idx].astype(np.float32)
    X_ap_unb_top = X_ap_unb[:, top_ap_bit_idx].astype(np.float32)
    print(f"\n[feat] X_ap_unb_top    shape = {X_ap_unb_top.shape}")

    X_maccs_te = np.load(MACCS_TE_PATH)
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_maccs_unb_top = X_maccs_unb[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb_top shape = {X_maccs_unb_top.shape}")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb = X_mord_te[unb_idx].astype(np.float32)
    X_mord_unb_top = X_mord_unb[:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top  shape = {X_mord_unb_top.shape}")

    if not CHEMPROP_EMBED_TE_PATH.exists():
        raise FileNotFoundError(
            f"Chemprop embed cache missing: {CHEMPROP_EMBED_TE_PATH}"
        )
    X_emb_te = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
    X_emb_te = np.where(np.isfinite(X_emb_te), X_emb_te, 0.0).astype(np.float32)
    X_emb_unb = X_emb_te[unb_idx].astype(np.float32)
    X_emb_unb_top = X_emb_unb[:, top_embed_col_idx].astype(np.float32)
    print(f"[feat] X_emb_unb_top   shape = {X_emb_unb_top.shape}")

    # ---- ChEMBL pool + kNN feature build ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL + kNN feature build")
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
    print(f"   pred_chembl_pec50 (unb) mean={pred_chembl_unb.mean():.3f}  "
          f"std={pred_chembl_unb.std():.3f}")
    print(f"   mean_sim (unb)         mean={mean_sim_unb.mean():.3f}")

    # ---- Build 4 per-family pruned matrices (upgraded K config) ----
    family_matrices = {
        "AtomPair": np.concatenate(
            [X_ap_unb_top,
             pred_chembl_unb.reshape(-1, 1),
             mean_sim_unb.reshape(-1, 1)],
            axis=1,
        ).astype(np.float32),
        "MACCS": np.concatenate(
            [X_maccs_unb_top,
             pred_chembl_unb.reshape(-1, 1),
             mean_sim_unb.reshape(-1, 1)],
            axis=1,
        ).astype(np.float32),
        "Mordred": np.concatenate(
            [X_mord_unb_top,
             pred_chembl_unb.reshape(-1, 1),
             mean_sim_unb.reshape(-1, 1)],
            axis=1,
        ).astype(np.float32),
        "ChempropEmbed": np.concatenate(
            [X_emb_unb_top,
             pred_chembl_unb.reshape(-1, 1),
             mean_sim_unb.reshape(-1, 1)],
            axis=1,
        ).astype(np.float32),
    }
    for fam in FAMILIES:
        print(f"   nb1550 {fam:<14s} PRUNED matrix: "
              f"{family_matrices[fam].shape}")

    # ---- Outer-bag rebuild of nb1550_o ------------------------------------
    print("\n" + "=" * 78)
    print("OUTER-BAG x [4 family LGBM Huber residual learners, naive 1/4 mean]")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)

    outer_nb1550 = np.zeros((n_outer, n_unb), dtype=np.float64)
    per_outer_records = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        print(f"\n   --- outer seed {o}  family seed = {o} ---")

        family_corrected = np.zeros((len(FAMILIES), n_unb), dtype=np.float64)
        family_rae: dict[str, float] = {}
        for fi, fam in enumerate(FAMILIES):
            ts = time.time()
            X_fam_pruned = family_matrices[fam]
            resid_oof_fam = _lgbm_residual_cross_fit_one_seed(
                X_fam_pruned, residual, seed=int(o)
            )
            pred_corr_fam = anchor + resid_oof_fam
            family_corrected[fi] = pred_corr_fam
            r_f = float(rae(y_unb, pred_corr_fam))
            family_rae[fam] = r_f
            print(f"     [nb1550] outer {o:3d}  family {fam:<14s} "
                  f"seed = {o:3d}: rae = {r_f:.4f}  "
                  f"|resid|.std = {resid_oof_fam.std():.3f}  "
                  f"wall = {time.time() - ts:.1f}s")
        nb1550_o = family_corrected.mean(axis=0)
        outer_nb1550[oi] = nb1550_o
        rae_nb1550_o = float(rae(y_unb, nb1550_o))
        print(f"     [nb1550_o = naive 1/4 mean over families] pooled RAE = "
              f"{rae_nb1550_o:.4f}")

        per_outer_records.append({
            "outer_seed": int(o),
            "family_seed": int(o),
            "rae_nb1550_per_family": family_rae,
            "rae_nb1550_o": rae_nb1550_o,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"   outer {o:3d}  (outer wall = "
              f"{time.time() - t_outer:.1f}s)")

    # ---- Per-outer summary ----
    per_outer_rae_nb1550: list[float] = [
        rec["rae_nb1550_o"] for rec in per_outer_records
    ]
    per_outer_arr = np.array(per_outer_rae_nb1550)
    per_outer_mean = float(per_outer_arr.mean())
    per_outer_std = float(per_outer_arr.std())
    per_outer_min = float(per_outer_arr.min())
    per_outer_max = float(per_outer_arr.max())
    per_outer_median = float(np.median(per_outer_arr))

    # ---- BoB row-level aggregations across 5 nb1550_o vectors ----
    bob_mean_oof = outer_nb1550.mean(axis=0)
    bob_median_oof = np.median(outer_nb1550, axis=0)
    rae_bob_mean = float(rae(y_unb, bob_mean_oof))
    rae_bob_median = float(rae(y_unb, bob_median_oof))

    print("\n" + "=" * 78)
    print("OUTER-BAG SUMMARY")
    print("=" * 78)
    print(f"   per-outer nb1550_o RAE list = "
          f"[{', '.join(f'{r:.4f}' for r in per_outer_rae_nb1550)}]")
    print(f"   per-outer nb1550_o mean    = {per_outer_mean:.4f}")
    print(f"   per-outer nb1550_o std     = {per_outer_std:.4f}")
    print(f"   per-outer nb1550_o min     = {per_outer_min:.4f}")
    print(f"   per-outer nb1550_o max     = {per_outer_max:.4f}")
    print(f"   per-outer nb1550_o median  = {per_outer_median:.4f}")
    print(f"   BoB MEAN   pooled RAE = {rae_bob_mean:.4f}  "
          f"(d vs nb1550_ref = {rae_bob_mean - NB1550_REF_EXPECTED:+.4f})")
    print(f"   BoB MEDIAN pooled RAE = {rae_bob_median:.4f}  "
          f"(d vs nb1550_ref = {rae_bob_median - NB1550_REF_EXPECTED:+.4f})")

    # ---- Pearson sanity vs nb1484 / nb1541 -----------------------
    def _pearson_vs(path: Path, vec: np.ndarray):
        if not path.exists():
            return None
        oof = np.load(path).astype(np.float64)
        if oof.shape[0] != n_unb:
            return None
        return float(np.corrcoef(vec, oof)[0, 1])

    pearson_bobmean_vs_nb1484 = _pearson_vs(
        DATA_PROCESSED / "nb1484_best_oof.npy", bob_mean_oof
    )
    pearson_bobmean_vs_nb1541_K = _pearson_vs(
        DATA_PROCESSED / "nb1541_best_K_oof.npy", bob_mean_oof
    )
    pearson_bobmean_vs_anchor = float(np.corrcoef(bob_mean_oof, anchor)[0, 1])
    if pearson_bobmean_vs_nb1484 is not None:
        print(f"   Pearson(bob_mean, nb1484_best)     = "
              f"{pearson_bobmean_vs_nb1484:.4f}")
    if pearson_bobmean_vs_nb1541_K is not None:
        print(f"   Pearson(bob_mean, nb1541_best_K)   = "
              f"{pearson_bobmean_vs_nb1541_K:.4f}")
    print(f"   Pearson(bob_mean, anchor)          = "
          f"{pearson_bobmean_vs_anchor:.4f}")

    # ---- Verdict (per-outer-mean vs nb1550_ref within REPRODUCE_MARGIN) ----
    delta_per_outer = per_outer_mean - NB1550_REF_EXPECTED
    reproduces = abs(delta_per_outer) < REPRODUCE_MARGIN
    if reproduces:
        verdict = "NB1550_REPRODUCES"
    elif per_outer_mean < NB1550_REF_EXPECTED - REPRODUCE_MARGIN:
        verdict = "NB1550_OPTIMISTIC_OUTER_BAG_BETTER"
    else:
        verdict = "NB1550_LUCKY_SEED_OUTER_BAG_WORSE"

    print("\n" + "-" * 78)
    print(f"VERDICT (per-outer nb1550_o mean within {REPRODUCE_MARGIN} of "
          f"nb1550_ref = {NB1550_REF_EXPECTED:.4f}):")
    print(f"   per-outer-mean = {per_outer_mean:.4f}   "
          f"(d vs ref = {delta_per_outer:+.4f})")
    print(f"   verdict        = {verdict}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_bob_mean_oof.npy",
            bob_mean_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_bob_median_oof.npy",
            bob_median_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_bob_mean_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_bob_median_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": "AtomPair-cache + MACCS-cache + Mordred-cached_nb1030 + "
                       "ChempropEmbed-cache + local_chembl_caches_union",
        "top_k_config": TOP_K_CONFIG,
        "chemprop_top_idx_matches_nb1541": chemprop_match_nb1541,
        "nb1550_model": "LGBM(huber alpha=1.0, d3, leaves=7, n_est=80, lr=0.05)",
        "nb1550_family_aggregation": "naive_1_4_mean",
        "n_unb": n_unb,
        "n_test": int(n_test),
        "n_chembl_pool": int(len(pool)),
        "outer_seeds": OUTER_SEEDS,
        "resid_folds": RESID_FOLDS,
        "n_top_atompair": n_top_ap,
        "n_top_maccs": n_top_maccs,
        "n_top_mordred": n_top_mord,
        "n_top_chemprop_embed": n_top_embed,
        "feat_dim_per_family": {
            fam: int(family_matrices[fam].shape[1]) for fam in FAMILIES
        },
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb1484_rae_best_ref": NB1484_REF_BEST,
        "nb1484_chemprop_slot_k30_ref": NB1484_CHEMPROP_SLOT_K30,
        "nb1541_chemprop_slot_k20_ref": NB1541_CHEMPROP_SLOT_K20,
        "nb1550_ref_expected": NB1550_REF_EXPECTED,
        "reproduce_margin": REPRODUCE_MARGIN,
        "per_outer_records": per_outer_records,
        "per_outer_rae_nb1550": per_outer_rae_nb1550,
        "per_outer_mean": per_outer_mean,
        "per_outer_std": per_outer_std,
        "per_outer_min": per_outer_min,
        "per_outer_max": per_outer_max,
        "per_outer_median": per_outer_median,
        "rae_bob_mean": rae_bob_mean,
        "rae_bob_median": rae_bob_median,
        "delta_per_outer_mean_vs_nb1550_ref": delta_per_outer,
        "delta_bob_mean_vs_nb1550_ref": rae_bob_mean - NB1550_REF_EXPECTED,
        "delta_bob_median_vs_nb1550_ref": rae_bob_median - NB1550_REF_EXPECTED,
        "reproduces": bool(reproduces),
        "pearson_bobmean_vs_nb1484_best": pearson_bobmean_vs_nb1484,
        "pearson_bobmean_vs_nb1541_best_K": pearson_bobmean_vs_nb1541_K,
        "pearson_bobmean_vs_anchor": pearson_bobmean_vs_anchor,
        "verdict": verdict,
        "pre_unblind_clean": True,
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
        "n_unb", "n_test", "n_chembl_pool",
        "feat_dim_per_family",
        "top_k_config",
        "chemprop_top_idx_matches_nb1541",
        "outer_seeds",
        "rae_anchor_chemprop_aux",
        "nb1484_rae_best_ref",
        "nb1484_chemprop_slot_k30_ref",
        "nb1541_chemprop_slot_k20_ref",
        "nb1550_ref_expected",
        "per_outer_rae_nb1550",
        "per_outer_mean", "per_outer_std",
        "per_outer_min", "per_outer_max", "per_outer_median",
        "rae_bob_mean", "rae_bob_median",
        "delta_per_outer_mean_vs_nb1550_ref",
        "delta_bob_mean_vs_nb1550_ref",
        "delta_bob_median_vs_nb1550_ref",
        "reproduces",
        "pearson_bobmean_vs_nb1484_best",
        "pearson_bobmean_vs_nb1541_best_K",
        "pearson_bobmean_vs_anchor",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
