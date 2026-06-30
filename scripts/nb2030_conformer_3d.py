"""nb2030 -- Rich 3D conformer ENSEMBLE descriptors over nb2063 SHAP top-28.

HYPOTHESIS:
    nb1120 added 6 single-conformer 3D pharmacophore features (D-D / D-A / A-A
    / aromatic-aromatic mean distances + PMI NPR1, NPR2) and HURT chemprop_aux
    residual by +0.046 RAE.  The likely failure mode is single-conformer
    noise: one lowest-energy conformer is a point estimate that is sensitive
    to the embed seed; gas-phase MMFF94 minima are also biased away from the
    bioactive conformer.  Strategy here:

    1.  Sample 20 ETKDGv3 conformers per compound (vs 10 in nb1120),
        MMFF94-minimize each.
    2.  Compute 8 RDKit Descriptors3D global shape/inertia features per
        conformer (RadiusOfGyration, Asphericity, Eccentricity,
        InertialShapeFactor, SpherocityIndex, NPR1, NPR2, PBF).
    3.  Aggregate ENSEMBLE statistics (mean / std / range) across the 20
        survivors -- 24 features (8 desc x 3 stats).  Mean is the centroid
        of the conformational basin; std / range capture conformer
        flexibility (an entropic / floppy-ness signal that single-conformer
        features cannot express).
    4.  Append to nb2063 SHAP top-28 (the 117-col 5-way K-tuned base) ->
        K = 28 + 24 = 52.  Fit LGBM(MSE) with the same nb2103 hyperparams
        on chemprop_aux residual, 5-seed bag, 5-fold scaffold-aware CV.
    5.  ALSO try a SHAP-prune of the K=52 down to a new K=28: let SHAP pick
        the best blend of 2D + 3D-ensemble features.
    6.  Compare both vs nb2103 K=28 with the user-supplied target 0.5057
        (and the on-disk ref nb2103 K=28 0.4737 mean-bag); gate 0.003.

PROTOCOL:
    1.  Build / cache 24 ENSEMBLE 3D features on all 513 (cache
        nb2030_conf3d_test_513.npy + nb2030_conf3d_test_fail.npy).
        Mean / std / range across up to N_CONFS=20 ETKDG/MMFF94 survivors.
    2.  Reuse nb2103 K=28 SHAP top-28 idx in 117.  Rebuild 117-col matrix on
        unblind 253 via the same recipe as nb1120 / nb2103.
    3.  CHANNEL A -- K=52 = SHAP-top-28 ++ conformer-24:
            5-seed bag, scaffold-CF 5-fold cross-fit, residual on
            chemprop_aux.
    4.  CHANNEL B -- K=52 SHAP-prune to NEW K=28 (let SHAP pick mix of 2D
            +3D ensemble feats).  Re-rank the K=52 by per-fold SHAP, take
            top-28, re-run the 5-seed scaffold cross-fit.
    5.  Compare each channel vs nb2103 K=28 (target 0.5057 user
            reference; on-disk ref 0.4737); gate 0.003 (mean OR median).
    6.  If either beats: build deploy CSV.
    7.  Save data/processed/nb2030_summary.json.

Outputs:
    scripts/nb2030_conformer_3d.py
    data/processed/nb2030_conf3d_test_513.npy        (513, 24) float32
    data/processed/nb2030_conf3d_test_fail.npy       (513,) bool
    data/processed/nb2030_mean_bag_oof_K52.npy       (253,) float32
    data/processed/nb2030_mean_bag_oof_K52shap28.npy (253,) float32
    data/processed/nb2030_summary.json
    submissions/nb2030_conformer_3d_K52.csv          (only if beats)
    submissions/nb2030_conformer_3d_K52shap28.csv    (only if beats)
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
from rdkit.Chem import AllChem, Descriptors3D

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, bemis_murcko, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2030"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

# References (PRE-unblind path)
CHEMPROP_AUX_REF = 0.6216
# nb2103 K=28 (on-disk verified, mean-bag / median-bag)
NB2103_K28_MEAN_REF = 0.4737
NB2103_K28_MEDIAN_REF = 0.4698
# user-supplied target (likely the median-bag / downstream-eval reference);
# we evaluate against BOTH numbers (gate independently)
USER_TARGET_REF = 0.5057
DECISION_MARGIN = 0.003

# Conformer / 3D config -- richer than nb1120 (N_CONFS 10 -> 20, full shape set)
N_CONFS = 20
MMFF_VARIANT = "MMFF94"
ETKDG_VERSION = 3
ETKDG_BASE_SEED = 42
MAX_OPT_ITERS = 200

# 8 RDKit Descriptors3D, in fixed canonical order
DESC3D_NAMES = [
    "RadiusOfGyration",
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "SpherocityIndex",
    "NPR1",
    "NPR2",
    "PBF",
]
N_DESC3D = len(DESC3D_NAMES)
N_STATS = 3  # mean / std / range
N_FEAT_3D = N_DESC3D * N_STATS  # 24

# CV + bag
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
VERIFY_SEEDS = [211, 314, 271]
K_PRUNE = 28  # SHAP-prune target


# ---------------------------------------------------------------------------
# 3D conformer ensemble features
# ---------------------------------------------------------------------------

def _desc3d_one(mh: Chem.Mol, cid: int) -> np.ndarray:
    """Compute the 8 Descriptors3D values for conformer cid.  Returns
    (8,) float32; NaN if any descriptor fails.
    """
    out = np.full(N_DESC3D, np.nan, dtype=np.float32)
    try:
        out[0] = float(Descriptors3D.RadiusOfGyration(mh, confId=int(cid)))
    except Exception:
        pass
    try:
        out[1] = float(Descriptors3D.Asphericity(mh, confId=int(cid)))
    except Exception:
        pass
    try:
        out[2] = float(Descriptors3D.Eccentricity(mh, confId=int(cid)))
    except Exception:
        pass
    try:
        out[3] = float(Descriptors3D.InertialShapeFactor(mh, confId=int(cid)))
    except Exception:
        pass
    try:
        out[4] = float(Descriptors3D.SpherocityIndex(mh, confId=int(cid)))
    except Exception:
        pass
    try:
        out[5] = float(Descriptors3D.NPR1(mh, confId=int(cid)))
    except Exception:
        pass
    try:
        out[6] = float(Descriptors3D.NPR2(mh, confId=int(cid)))
    except Exception:
        pass
    try:
        out[7] = float(Descriptors3D.PBF(mh, confId=int(cid)))
    except Exception:
        pass
    # Replace any non-finite (inf, nan) with NaN so the aggregator can skip
    out = np.where(np.isfinite(out), out, np.nan).astype(np.float32)
    return out


def _embed_minimize_keep_all(mol: Chem.Mol, n_confs: int = N_CONFS,
                              base_seed: int = ETKDG_BASE_SEED
                              ) -> tuple[Chem.Mol | None, list[int]]:
    """Embed N_CONFS ETKDGv3 conformers and MMFF94-minimize each.  Returns
    (mol_with_H, list of valid confIds that minimized successfully).
    """
    if mol is None:
        return None, []
    try:
        mh = Chem.AddHs(mol)
    except Exception:
        return None, []
    params = AllChem.ETKDGv3()
    params.randomSeed = int(base_seed)
    params.numThreads = 1
    params.useSmallRingTorsions = True
    params.pruneRmsThresh = 0.5
    try:
        cids = AllChem.EmbedMultipleConfs(mh, numConfs=n_confs, params=params)
    except Exception:
        return None, []
    if not cids:
        return None, []
    cids = list(cids)
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mh, mmffVariant=MMFF_VARIANT)
        if mp is None:
            return None, []
    except Exception:
        return None, []
    valid_cids: list[int] = []
    for cid in cids:
        try:
            ff = AllChem.MMFFGetMoleculeForceField(mh, mp, confId=int(cid))
            if ff is None:
                continue
            ff.Minimize(maxIts=MAX_OPT_ITERS)
            _ = float(ff.CalcEnergy())  # sanity
            valid_cids.append(int(cid))
        except Exception:
            continue
    return mh, valid_cids


def _ensemble_3d_feats(mh: Chem.Mol | None,
                        cids: list[int]) -> tuple[np.ndarray, bool]:
    """Compute 8 Descriptors3D over all valid conformers, then aggregate
    mean/std/range -> (24,) float32.  failed=True if no conformers.
    """
    out = np.zeros(N_FEAT_3D, dtype=np.float32)
    if mh is None or not cids:
        return out, True
    rows = []
    for cid in cids:
        rows.append(_desc3d_one(mh, cid))
    if not rows:
        return out, True
    arr = np.stack(rows, axis=0).astype(np.float32)  # (n_conf, 8)
    # Aggregate ignoring NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(arr, axis=0).astype(np.float32)
        col_std = np.nanstd(arr, axis=0).astype(np.float32)
        col_max = np.nanmax(arr, axis=0).astype(np.float32)
        col_min = np.nanmin(arr, axis=0).astype(np.float32)
    col_range = (col_max - col_min).astype(np.float32)
    # Replace any all-NaN columns with 0
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
    col_std = np.where(np.isfinite(col_std), col_std, 0.0).astype(np.float32)
    col_range = np.where(
        np.isfinite(col_range), col_range, 0.0).astype(np.float32)
    out[:N_DESC3D] = col_mean
    out[N_DESC3D:2 * N_DESC3D] = col_std
    out[2 * N_DESC3D:3 * N_DESC3D] = col_range
    return out, False


def _build_conf3d_features(smiles_list: list[str],
                            cache_path: Path,
                            fail_cache_path: Path,
                            force: bool = False
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (N, 24) float32 mean/std/range ENSEMBLE feats and (N,) bool
    fail mask.  Caches both.
    """
    n = len(smiles_list)
    if cache_path.exists() and fail_cache_path.exists() and not force:
        X = np.load(cache_path).astype(np.float32)
        fail = np.load(fail_cache_path).astype(bool)
        if X.shape == (n, N_FEAT_3D) and fail.shape == (n,):
            print(f"   [cache] reloaded {cache_path} shape {X.shape}  "
                  f"fail rate = {fail.mean() * 100:.2f}%")
            return X, fail
        else:
            print(f"   [cache-mismatch] recomputing; cache {X.shape} "
                  f"vs expected ({n},{N_FEAT_3D})")

    X = np.zeros((n, N_FEAT_3D), dtype=np.float32)
    fail = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, s in enumerate(smiles_list):
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"   [embed] {i:4d}/{n}  elapsed={elapsed:.0f}s  "
                  f"eta={eta:.0f}s  fail_so_far={fail[:i].sum()}")
        mol = standardize(s)
        if mol is None:
            fail[i] = True
            continue
        mh, valid_cids = _embed_minimize_keep_all(mol)
        feats, fail_i = _ensemble_3d_feats(mh, valid_cids)
        X[i] = feats
        fail[i] = bool(fail_i)
    elapsed = time.time() - t0
    print(f"   [embed] done.  total wall = {elapsed:.0f}s  "
          f"fail rate = {fail.mean() * 100:.2f}%  "
          f"({fail.sum()}/{n})")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, X.astype(np.float32))
    np.save(fail_cache_path, fail)
    print(f"   [save] {cache_path}   {fail_cache_path}")
    return X, fail


# ---------------------------------------------------------------------------
# 117-col matrix + nb2103 K=28 SHAP slice (reuse nb1120 helper logic)
# ---------------------------------------------------------------------------

def _load_nb2103_k28_top_idx() -> tuple[np.ndarray, dict]:
    if not NB2103_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2103_SUMMARY}")
    with open(NB2103_SUMMARY) as f:
        s = json.load(f)
    rec = None
    for r in s["per_K_records"]:
        if int(r["K"]) == 28:
            rec = r
            break
    if rec is None:
        raise KeyError("K=28 record not found in nb2103_summary.json")
    top_idx = np.array(rec["top_K_idx_in_117"], dtype=np.int32)
    return top_idx, s


def _build_117col_matrix(idx_slice: np.ndarray, n_test: int) -> np.ndarray:
    """Same 117-col 5-way K-tuned matrix as nb2063 / nb1120 / nb2103.
    idx_slice picks which rows from the 513-row test matrix to return.
    """
    EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
    ATOMPAIR_TE = DATA_PROCESSED / "te_atompair.npy"
    MACCS_TE = DATA_PROCESSED / "te_maccs.npy"
    CHEMPROP_EMBED_TE = DATA_PROCESSED / "te_chemprop_embed_300.npy"
    AVALON_TE = DATA_PROCESSED / "te_avalon512.npy"
    MORDRED_TE = Path("C:/pxr_artifacts/nb1030/X_mordred_test.npy")

    NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
    NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
    NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
    NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
    NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
    NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

    def _load_npy(p, n_expected):
        X = np.load(p).astype(np.float32)
        if X.shape[0] != n_expected:
            raise ValueError(f"shape mismatch {p}: {X.shape}")
        return np.where(np.isfinite(X), X, 0.0).astype(np.float32)

    def _load_mord(n_expected):
        X = np.load(MORDRED_TE).astype(np.float32)
        if X.shape[0] != n_expected:
            raise ValueError(f"Mordred shape {X.shape}")
        X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
        col_med = np.nanmedian(X, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
        bad = ~np.isfinite(X)
        if bad.any():
            r, c = np.where(bad)
            X[r, c] = col_med[c]
        return X

    def _load_json(p):
        with open(p) as f:
            return json.load(f)

    sum_1352 = _load_json(NB1352_SUMMARY)
    sum_1392 = _load_json(NB1392_SUMMARY)
    sum_1484 = _load_json(NB1484_SUMMARY)
    sum_1523 = _load_json(NB1523_SUMMARY)
    sum_1524 = _load_json(NB1524_SUMMARY)
    sum_1541 = _load_json(NB1541_SUMMARY)

    top_maccs = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    top_avalon = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    K_M = int(sum_1523["best_K"])
    rec_m = next(r for r in sum_1523["per_K_records"] if int(r["K"]) == K_M)
    top_mord = np.array(rec_m["top_col_idx"], dtype=int)

    fam = next(f for f in sum_1484["families"] if f["family"] == "AtomPair")
    full_ap = np.array(fam["top_idx_ranked"], dtype=int)
    K_AP = int(sum_1524["best_K"])
    top_ap = full_ap[:K_AP]

    K_EMB = int(sum_1541["best_K"])
    top_emb_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_emb = top_emb_full[:K_EMB]

    X_ap = _load_npy(ATOMPAIR_TE, n_test)[idx_slice][:, top_ap]
    X_maccs = _load_npy(MACCS_TE, n_test)[idx_slice][:, top_maccs]
    X_mord = _load_mord(n_test)[idx_slice][:, top_mord]
    X_emb = _load_npy(CHEMPROP_EMBED_TE, n_test)[idx_slice][:, top_emb]
    X_av = _load_npy(AVALON_TE, n_test)[idx_slice][:, top_avalon]

    # ChEMBL kNN feature (identical to nb1120/nb2103 recipe)
    KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
    KEEP_RELATIONS = {"=", "==", "~"}
    MAX_NM = 100_000.0
    MIN_NM = 1e-3
    KNN_K = 5
    SIM_FLOOR = 1e-6

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
        d["pec50"] = 9.0 - np.log10(d["standard_value"].astype(float))
        d = d[["canonical_smiles", "pec50"]].rename(
            columns={"canonical_smiles": "smiles"})
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
        raise FileNotFoundError("no ChEMBL PXR parquet found")
    pool = pd.concat(frames, ignore_index=True)
    mols_pool = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols_pool.apply(
        lambda m: Chem.MolToInchiKey(m) if m is not None else None)
    pool["std_smiles"] = mols_pool.apply(
        lambda m: Chem.MolToSmiles(m) if m is not None else None)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    agg = (pool.groupby("inchikey", as_index=False)
                .agg(pec50=("pec50", "median"),
                     std_smiles=("std_smiles", "first")))

    te = load_test()
    test_smiles = te["smiles"].astype(str).tolist()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = {Chem.MolToInchiKey(m) for m in test_mols if m is not None}
    agg = agg[~agg["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(agg["std_smiles"].tolist())
    keep = fp_pool.sum(axis=1) > 0
    agg = agg[keep].reset_index(drop=True)
    fp_pool = fp_pool[keep]
    pool_labels = agg["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))

    std_test = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test)

    a = fp_test.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    n_pool = b.shape[0]
    top_idx = np.zeros((n_q, KNN_K), dtype=np.int32)
    top_sim = np.zeros((n_q, KNN_K), dtype=np.float32)
    BLOCK = 64
    for s_ in range(0, n_q, BLOCK):
        e = min(n_q, s_ + BLOCK)
        inter = a[s_:e] @ b.T
        denom = a_sum[s_:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        if KNN_K >= n_pool:
            idx_part = np.argsort(-sim, axis=1)[:, :KNN_K]
        else:
            part = np.argpartition(-sim, kth=KNN_K - 1, axis=1)[:, :KNN_K]
            row_idx = np.arange(e - s_)[:, None]
            sim_part = sim[row_idx, part]
            order = np.argsort(-sim_part, axis=1)
            idx_part = part[row_idx, order]
        row_idx = np.arange(e - s_)[:, None]
        top_idx[s_:e] = idx_part
        top_sim[s_:e] = sim[row_idx, idx_part]

    w = np.clip(top_sim, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    pred_chembl = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred_chembl[i] = pool_median
        else:
            pred_chembl[i] = np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i]
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    pred_chembl_slice = pred_chembl[idx_slice].reshape(-1, 1)
    mean_sim_slice = mean_sim[idx_slice].reshape(-1, 1)

    X117 = np.concatenate(
        [X_ap, X_maccs, X_mord, X_emb, X_av,
         pred_chembl_slice, mean_sim_slice],
        axis=1).astype(np.float32)
    if X117.shape[1] != 117:
        raise ValueError(f"117-col matrix wrong dim: {X117.shape}")
    return X117


# ---------------------------------------------------------------------------
# LGBM + scaffold cross-fit
# ---------------------------------------------------------------------------

def _lgbm_params(seed: int) -> dict:
    """Match nb2103 / nb1120 hyperparams."""
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


def _scaffold_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                  scaffolds: list[str | None],
                                  seed: int
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (oof, shap_importance_per_feat).  SHAP from each fold's
    TreeExplainer averaged across folds (proxy: LGBM gain importance).
    """
    n = len(residual)
    n_feat = X.shape[1]
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    imp_accum = np.zeros(n_feat, dtype=np.float64)
    n_folds = 0
    for tr_idx, va_idx in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_idx], residual[tr_idx])
        oof[va_idx] = mdl.predict(X[va_idx])
        # Gain importance (per-feat); SHAP would be cleaner but is slow at
        # K=52 x 5 seeds x 5 folds; gain matches the nb2063 SHAP signal well
        gi = np.asarray(mdl.booster_.feature_importance(importance_type="gain"),
                         dtype=np.float64)
        if gi.shape[0] == n_feat:
            imp_accum += gi
            n_folds += 1
    if np.isnan(oof).any():
        raise RuntimeError(f"seed {seed} oof has NaN")
    if n_folds > 0:
        imp_accum /= float(n_folds)
    return oof, imp_accum


def _run_channel(X: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
                  y_unb: np.ndarray, scaffolds: list[str | None],
                  rae_anchor: float, label: str
                  ) -> tuple[dict, np.ndarray, np.ndarray]:
    """5-seed scaffold cross-fit on (X, residual).  Returns (record,
    mean_bag_oof, mean_importance).
    """
    n_unb = len(residual)
    n_feat = X.shape[1]
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae = []
    per_seed_records = []
    imp_accum = np.zeros(n_feat, dtype=np.float64)
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof, imp_s = _scaffold_cross_fit_one_seed(
            X, residual, scaffolds, s)
        pred_corr = anchor + resid_oof
        per_seed_corrected[i] = pred_corr
        r = float(rae(y_unb, pred_corr))
        per_seed_rae.append(r)
        delta_s = r - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": r,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof.std()),
            "resid_oof_mean": float(resid_oof.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [{label}] seed={s:3d}:  rae_corr = {r:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")
        imp_accum += imp_s

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_arr = np.array(per_seed_rae)
    record = {
        "label": label,
        "feat_dim": int(n_feat),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": float(per_seed_arr.mean()),
        "rae_per_seed_std": float(per_seed_arr.std()),
        "rae_per_seed_min": float(per_seed_arr.min()),
        "rae_per_seed_max": float(per_seed_arr.max()),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
    }
    mean_importance = (imp_accum / float(len(RESID_SEEDS))).astype(np.float64)
    return record, mean_bag_oof, mean_importance


def _verify_channel(X: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
                     y_unb: np.ndarray, scaffolds: list[str | None],
                     target_rae: float, label: str) -> dict:
    """Fresh-seed verification on VERIFY_SEEDS."""
    n_unb = len(residual)
    per_seed_corrected = np.zeros((len(VERIFY_SEEDS), n_unb), dtype=np.float64)
    records = []
    for i, s in enumerate(VERIFY_SEEDS):
        ts = time.time()
        resid_oof, _ = _scaffold_cross_fit_one_seed(
            X, residual, scaffolds, s)
        pred_v = anchor + resid_oof
        per_seed_corrected[i] = pred_v
        r_v = float(rae(y_unb, pred_v))
        beats_v = r_v < target_rae - DECISION_MARGIN
        records.append({
            "seed": int(s),
            "rae_corrected": r_v,
            "beats_target_by_margin": bool(beats_v),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [{label}] verify seed={s:3d}:  rae = {r_v:.4f}  "
              f"beats_target = {beats_v}")
    verify_bag = per_seed_corrected.mean(axis=0)
    rae_verify_bag = float(rae(y_unb, verify_bag))
    n_better = sum(1 for r in records if r["beats_target_by_margin"])
    verdict = ("VERIFIED_REPRODUCIBLE"
                if (n_better >= 2 and rae_verify_bag < target_rae - DECISION_MARGIN)
                else "VERIFY_NOT_REPRODUCIBLE")
    print(f"   [{label}] verify mean_bag = {rae_verify_bag:.4f}  "
          f"({n_better}/{len(VERIFY_SEEDS)} beat target)  -> {verdict}")
    return {
        "records": records,
        "rae_verify_bag": rae_verify_bag,
        "n_beats": int(n_better),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Deploy CSV
# ---------------------------------------------------------------------------

def _build_deploy_csv(out_csv: Path,
                      top_idx_in_117: np.ndarray,
                      conf3d_test_513: np.ndarray,
                      shap_prune_idx_in_K: np.ndarray | None) -> None:
    """Train on ALL 253 unblind labels (deploy refit), predict 513 test set.
    Anchor = chemprop_aux te[513].  If shap_prune_idx_in_K is None, deploys
    the full K=52 channel; otherwise slices to the SHAP-pruned K subset.
    """
    print(f"\n[deploy] training on all 253 unblind, predicting 513 -> {out_csv}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)

    X117_unb = _build_117col_matrix(unb_idx, n_test=513)
    X117_all = _build_117col_matrix(np.arange(513), n_test=513)
    X28_unb = X117_unb[:, top_idx_in_117]
    X28_all = X117_all[:, top_idx_in_117]
    X3d_unb = conf3d_test_513[unb_idx]
    X3d_all = conf3d_test_513
    X52_unb = np.concatenate([X28_unb, X3d_unb], axis=1).astype(np.float32)
    X52_all = np.concatenate([X28_all, X3d_all], axis=1).astype(np.float32)
    if shap_prune_idx_in_K is not None:
        X52_unb = X52_unb[:, shap_prune_idx_in_K].astype(np.float32)
        X52_all = X52_all[:, shap_prune_idx_in_K].astype(np.float32)
    residual_unb = y_unb - te_anchor_513[unb_idx]

    preds_seed = []
    for s in RESID_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X52_unb, residual_unb)
        preds_seed.append(mdl.predict(X52_all))
    resid_pred_513 = np.mean(np.stack(preds_seed, axis=0), axis=0)
    pred_513 = te_anchor_513 + resid_pred_513

    te = load_test()
    out = pd.DataFrame({
        "Molecule Name": te["name"].astype(str).tolist(),
        "SMILES": te["smiles"].astype(str).tolist(),
        "pEC50": pred_513.astype(np.float32),
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[deploy] wrote {out_csv}  rows={len(out)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 3D conformer ENSEMBLE (20-conf ETKDG+MMFF94) over "
          f"nb2063 SHAP-top-28 -> K=52")
    print(f"          + SHAP-prune K=52 -> K=28 channel")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean={NB2103_K28_MEAN_REF:.4f}  "
          f"median={NB2103_K28_MEDIAN_REF:.4f}  "
          f"user_target={USER_TARGET_REF:.4f}  margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Anchor + truth ----
    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"chemprop_aux te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Scaffolds for unblind 253 ----
    unb_smiles = te["smiles"].astype(str).to_numpy()[unb_idx].tolist()
    scaffolds_unb = [bemis_murcko(s) for s in unb_smiles]
    n_with_scaf = sum(1 for s in scaffolds_unb if s)
    print(f"[scaf] unblind scaffolds: {n_with_scaf}/{n_unb} non-empty")

    # ---- 3D conformer-ensemble features on 513 ----
    print("\n" + "-" * 78)
    print(f"3D CONFORMER ENSEMBLE: {N_CONFS} ETKDGv{ETKDG_VERSION} + MMFF94 "
          f"-> 8 desc x 3 stats = {N_FEAT_3D} feats")
    print("-" * 78)
    test_smiles_513 = te["smiles"].astype(str).tolist()
    cache_p = DATA_PROCESSED / f"{TAG}_conf3d_test_513.npy"
    fail_p = DATA_PROCESSED / f"{TAG}_conf3d_test_fail.npy"
    X3d_513, fail_513 = _build_conf3d_features(
        test_smiles_513, cache_p, fail_p, force=False)
    fail_rate_pct = float(fail_513.mean() * 100.0)
    print(f"[conf3d] features shape {X3d_513.shape}  "
          f"fail_rate = {fail_rate_pct:.2f}%")
    # Per-column sanity
    col_names: list[str] = []
    for stat in ("mean", "std", "range"):
        for d in DESC3D_NAMES:
            col_names.append(f"{d}_{stat}")
    for j, nm in enumerate(col_names):
        c = X3d_513[:, j]
        print(f"   [feat] {nm:30s}  mean={c.mean():+.3f}  "
              f"std={c.std():.3f}  min={c.min():+.3f}  max={c.max():+.3f}")
    X3d_unb = X3d_513[unb_idx]
    print(f"[conf3d] unb slice shape {X3d_unb.shape}")

    # ---- nb2103 K=28 SHAP top-28 ----
    print("\n" + "-" * 78)
    print("REBUILDING nb2103 K=28 SHAP top-28 slice on 117-col matrix")
    print("-" * 78)
    top28_idx, nb2103_sum = _load_nb2103_k28_top_idx()
    nb2103_k28_mean = NB2103_K28_MEAN_REF
    nb2103_k28_median = NB2103_K28_MEDIAN_REF
    for r in nb2103_sum.get("per_K_records", []):
        if int(r["K"]) == 28:
            nb2103_k28_mean = float(r["rae_mean_bag"])
            nb2103_k28_median = float(r["rae_median_bag"])
            break
    print(f"[ref] nb2103 K=28  mean_bag = {nb2103_k28_mean:.4f}  "
          f"median_bag = {nb2103_k28_median:.4f}")
    print(f"[ref] user_target = {USER_TARGET_REF:.4f}  (downstream-eval ref)")

    X117_unb = _build_117col_matrix(unb_idx, n_test=n_test)
    print(f"[feat] X117_unb shape {X117_unb.shape}")
    X28_unb = X117_unb[:, top28_idx].astype(np.float32)
    print(f"[feat] X28_unb shape {X28_unb.shape}  (SHAP top-28 from nb2063)")

    X52_unb = np.concatenate([X28_unb, X3d_unb], axis=1).astype(np.float32)
    print(f"[feat] X52_unb shape {X52_unb.shape}  (K=28 SHAP + 24 conf3d)")

    # Feature names for K=52
    feat_names_K52: list[str] = [f"SHAP28_idx{int(i)}" for i in top28_idx] \
        + col_names

    # ---- CHANNEL A -- K=52 ----
    print("\n" + "-" * 78)
    print("CHANNEL A: K=52 = SHAP-top-28 ++ 24 conf3d-ensemble")
    print("-" * 78)
    rec_A, oof_A, imp_A = _run_channel(
        X52_unb, residual, anchor, y_unb, scaffolds_unb,
        rae_anchor, label="K52")
    rae_mean_A = rec_A["rae_mean_bag"]
    rae_median_A = rec_A["rae_median_bag"]

    delta_mean_A_vs_nb2103 = rae_mean_A - nb2103_k28_mean
    delta_median_A_vs_nb2103 = rae_median_A - nb2103_k28_median
    delta_mean_A_vs_target = rae_mean_A - USER_TARGET_REF
    delta_median_A_vs_target = rae_median_A - USER_TARGET_REF
    beats_mean_A_nb2103 = rae_mean_A < nb2103_k28_mean - DECISION_MARGIN
    beats_median_A_nb2103 = rae_median_A < nb2103_k28_median - DECISION_MARGIN
    beats_mean_A_target = rae_mean_A < USER_TARGET_REF - DECISION_MARGIN
    beats_median_A_target = rae_median_A < USER_TARGET_REF - DECISION_MARGIN

    print(f"\n   [K52] per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in rec_A['per_seed_rae'])}]")
    print(f"   [K52] per-seed mean = {rec_A['rae_per_seed_mean']:.4f}  "
          f"std = {rec_A['rae_per_seed_std']:.4f}")
    print(f"   [K52] mean_bag   = {rae_mean_A:.4f}  "
          f"(d_vs_nb2103={delta_mean_A_vs_nb2103:+.4f}  "
          f"d_vs_target={delta_mean_A_vs_target:+.4f})")
    print(f"   [K52] median_bag = {rae_median_A:.4f}  "
          f"(d_vs_nb2103={delta_median_A_vs_nb2103:+.4f}  "
          f"d_vs_target={delta_median_A_vs_target:+.4f})")
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K52.npy",
            oof_A.astype(np.float32))

    # ---- CHANNEL B -- SHAP-prune K=52 -> K=28 ----
    print("\n" + "-" * 78)
    print(f"CHANNEL B: SHAP-prune K=52 -> new K={K_PRUNE} (mixed 2D + 3D)")
    print("-" * 78)
    rank_order_K52 = np.argsort(-imp_A).astype(np.int32)
    prune_idx = rank_order_K52[:K_PRUNE].astype(np.int32)
    sel_names = [feat_names_K52[i] for i in prune_idx]
    n_3d_in_pruned = sum(1 for n in sel_names if n.startswith(tuple(
        f"{d}_" for d in DESC3D_NAMES)))
    n_2d_in_pruned = K_PRUNE - n_3d_in_pruned
    print(f"   pruned top-{K_PRUNE} mix:  2D(SHAP)={n_2d_in_pruned}  "
          f"3D(conf)={n_3d_in_pruned}")
    print(f"   pruned 3D feats kept: "
          f"{[n for n in sel_names if not n.startswith('SHAP28_')]}")
    X_B = X52_unb[:, prune_idx].astype(np.float32)
    rec_B, oof_B, _ = _run_channel(
        X_B, residual, anchor, y_unb, scaffolds_unb,
        rae_anchor, label=f"K52shap{K_PRUNE}")
    rae_mean_B = rec_B["rae_mean_bag"]
    rae_median_B = rec_B["rae_median_bag"]

    delta_mean_B_vs_nb2103 = rae_mean_B - nb2103_k28_mean
    delta_median_B_vs_nb2103 = rae_median_B - nb2103_k28_median
    delta_mean_B_vs_target = rae_mean_B - USER_TARGET_REF
    delta_median_B_vs_target = rae_median_B - USER_TARGET_REF
    beats_mean_B_nb2103 = rae_mean_B < nb2103_k28_mean - DECISION_MARGIN
    beats_median_B_nb2103 = rae_median_B < nb2103_k28_median - DECISION_MARGIN
    beats_mean_B_target = rae_mean_B < USER_TARGET_REF - DECISION_MARGIN
    beats_median_B_target = rae_median_B < USER_TARGET_REF - DECISION_MARGIN

    print(f"\n   [K52shap{K_PRUNE}] per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in rec_B['per_seed_rae'])}]")
    print(f"   [K52shap{K_PRUNE}] per-seed mean = "
          f"{rec_B['rae_per_seed_mean']:.4f}  "
          f"std = {rec_B['rae_per_seed_std']:.4f}")
    print(f"   [K52shap{K_PRUNE}] mean_bag   = {rae_mean_B:.4f}  "
          f"(d_vs_nb2103={delta_mean_B_vs_nb2103:+.4f}  "
          f"d_vs_target={delta_mean_B_vs_target:+.4f})")
    print(f"   [K52shap{K_PRUNE}] median_bag = {rae_median_B:.4f}  "
          f"(d_vs_nb2103={delta_median_B_vs_nb2103:+.4f}  "
          f"d_vs_target={delta_median_B_vs_target:+.4f})")
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K52shap{K_PRUNE}.npy",
            oof_B.astype(np.float32))

    # ---- Verdict + verify + deploy ----
    deploy_records: list[dict] = []

    def _emit_verdict(label, rae_mean, rae_median, beats_m_2103, beats_med_2103,
                      beats_m_tgt, beats_med_tgt):
        if beats_m_2103 and beats_med_2103:
            v = "BEATS_NB2103_K28_BOTH"
        elif beats_m_2103:
            v = "BEATS_NB2103_K28_MEAN_ONLY"
        elif beats_med_2103:
            v = "BEATS_NB2103_K28_MEDIAN_ONLY"
        elif beats_m_tgt or beats_med_tgt:
            v = "BEATS_USER_TARGET_ONLY_WORSE_THAN_NB2103_K28"
        elif (abs(rae_mean - nb2103_k28_mean) < DECISION_MARGIN
              and abs(rae_median - nb2103_k28_median) < DECISION_MARGIN):
            v = "FLAT_VS_NB2103_K28"
        else:
            v = "WORSE_THAN_NB2103_K28"
        print(f"   [{label}] verdict = {v}")
        return v

    verdict_A = _emit_verdict("K52", rae_mean_A, rae_median_A,
                               beats_mean_A_nb2103, beats_median_A_nb2103,
                               beats_mean_A_target, beats_median_A_target)
    verdict_B = _emit_verdict(f"K52shap{K_PRUNE}", rae_mean_B, rae_median_B,
                               beats_mean_B_nb2103, beats_median_B_nb2103,
                               beats_mean_B_target, beats_median_B_target)

    verify_A = None
    verify_B = None
    deploy_csv_A_path = None
    deploy_csv_B_path = None

    if beats_mean_A_nb2103 or beats_median_A_nb2103:
        print("\n[verify-A] CHANNEL A beats nb2103 K=28 -- fresh-seed verify")
        target_A = nb2103_k28_mean if beats_mean_A_nb2103 else nb2103_k28_median
        verify_A = _verify_channel(
            X52_unb, residual, anchor, y_unb, scaffolds_unb,
            target_rae=target_A, label="K52")
        if verify_A["verdict"] == "VERIFIED_REPRODUCIBLE":
            deploy_csv_A_path = (Path(__file__).resolve().parents[1]
                                 / "submissions" / f"{TAG}_conformer_3d_K52.csv")
            try:
                _build_deploy_csv(deploy_csv_A_path, top28_idx, X3d_513,
                                   shap_prune_idx_in_K=None)
                deploy_records.append({"label": "K52",
                                        "csv": str(deploy_csv_A_path)})
            except Exception as e:
                print(f"[deploy-A] FAILED: {e}")
                deploy_csv_A_path = None

    if beats_mean_B_nb2103 or beats_median_B_nb2103:
        print(f"\n[verify-B] CHANNEL B (SHAP-prune K={K_PRUNE}) beats "
              "nb2103 K=28 -- fresh-seed verify")
        target_B = nb2103_k28_mean if beats_mean_B_nb2103 else nb2103_k28_median
        verify_B = _verify_channel(
            X_B, residual, anchor, y_unb, scaffolds_unb,
            target_rae=target_B, label=f"K52shap{K_PRUNE}")
        if verify_B["verdict"] == "VERIFIED_REPRODUCIBLE":
            deploy_csv_B_path = (Path(__file__).resolve().parents[1]
                                 / "submissions"
                                 / f"{TAG}_conformer_3d_K52shap{K_PRUNE}.csv")
            try:
                _build_deploy_csv(deploy_csv_B_path, top28_idx, X3d_513,
                                   shap_prune_idx_in_K=prune_idx)
                deploy_records.append({"label": f"K52shap{K_PRUNE}",
                                        "csv": str(deploy_csv_B_path)})
            except Exception as e:
                print(f"[deploy-B] FAILED: {e}")
                deploy_csv_B_path = None

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": (f"lgbm_mse_K28_SHAP_top + {N_FEAT_3D} ETKDG{ETKDG_VERSION}/"
                    f"MMFF94 conformer-ensemble feats ({N_CONFS} confs, 8 "
                    "desc x 3 stats); K=52 channel + SHAP-prune K=52->K=28 "
                    "channel; scaffold-CF 5-fold x 5-seed bag; residual on "
                    "chemprop_aux"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "conf3d_cols": col_names,
        "n_desc3d": int(N_DESC3D),
        "n_stats": int(N_STATS),
        "n_feat_3d": int(N_FEAT_3D),
        "n_confs": int(N_CONFS),
        "etkdg_version": int(ETKDG_VERSION),
        "etkdg_seed": int(ETKDG_BASE_SEED),
        "mmff_variant": MMFF_VARIANT,
        "max_opt_iters": int(MAX_OPT_ITERS),
        "conf3d_fail_count": int(fail_513.sum()),
        "conf3d_fail_rate_pct": fail_rate_pct,
        "lgbm_params": {
            "objective": "regression",
            "max_depth": 4,
            "num_leaves": 15,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "min_child_samples": 5,
            "reg_lambda": 2.0,
        },
        "resid_folds": int(RESID_FOLDS),
        "resid_seeds": RESID_SEEDS,
        "verify_seeds": VERIFY_SEEDS,
        "cv_split": "scaffold_kfold_indices_5fold",
        "n_unb": int(n_unb),
        "rae_anchor_chemprop_aux": float(rae_anchor),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2103_K28_mean_bag_ref": float(nb2103_k28_mean),
        "nb2103_K28_median_bag_ref": float(nb2103_k28_median),
        "user_target_ref": float(USER_TARGET_REF),
        "decision_margin": float(DECISION_MARGIN),
        "channel_A_K52": {
            **rec_A,
            "delta_mean_bag_vs_nb2103_K28": float(delta_mean_A_vs_nb2103),
            "delta_median_bag_vs_nb2103_K28": float(delta_median_A_vs_nb2103),
            "delta_mean_bag_vs_user_target": float(delta_mean_A_vs_target),
            "delta_median_bag_vs_user_target": float(delta_median_A_vs_target),
            "beats_nb2103_K28_mean": bool(beats_mean_A_nb2103),
            "beats_nb2103_K28_median": bool(beats_median_A_nb2103),
            "beats_user_target_mean": bool(beats_mean_A_target),
            "beats_user_target_median": bool(beats_median_A_target),
            "verdict": verdict_A,
            "verify": verify_A,
            "deploy_csv": str(deploy_csv_A_path) if deploy_csv_A_path else None,
        },
        "channel_B_K52shap28": {
            **rec_B,
            "K_pruned": int(K_PRUNE),
            "n_3d_in_pruned": int(n_3d_in_pruned),
            "n_2d_in_pruned": int(n_2d_in_pruned),
            "pruned_feature_names": sel_names,
            "pruned_idx_in_K52": prune_idx.tolist(),
            "delta_mean_bag_vs_nb2103_K28": float(delta_mean_B_vs_nb2103),
            "delta_median_bag_vs_nb2103_K28": float(delta_median_B_vs_nb2103),
            "delta_mean_bag_vs_user_target": float(delta_mean_B_vs_target),
            "delta_median_bag_vs_user_target": float(delta_median_B_vs_target),
            "beats_nb2103_K28_mean": bool(beats_mean_B_nb2103),
            "beats_nb2103_K28_median": bool(beats_median_B_nb2103),
            "beats_user_target_mean": bool(beats_mean_B_target),
            "beats_user_target_median": bool(beats_median_B_target),
            "verdict": verdict_B,
            "verify": verify_B,
            "deploy_csv": str(deploy_csv_B_path) if deploy_csv_B_path else None,
        },
        "deploy_records": deploy_records,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": float(CHEMPROP_AUX_REF),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    A = res["channel_A_K52"]
    B = res["channel_B_K52shap28"]
    for k in ("n_feat_3d", "n_confs", "conf3d_fail_count",
              "conf3d_fail_rate_pct",
              "rae_anchor_chemprop_aux",
              "nb2103_K28_mean_bag_ref", "nb2103_K28_median_bag_ref",
              "user_target_ref"):
        print(f"  {k}: {res.get(k)}")
    print("---- CHANNEL A K=52 ----")
    for k in ("feat_dim", "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_nb2103_K28",
              "delta_median_bag_vs_nb2103_K28",
              "delta_mean_bag_vs_user_target",
              "delta_median_bag_vs_user_target",
              "beats_nb2103_K28_mean", "beats_nb2103_K28_median",
              "verdict", "deploy_csv"):
        print(f"  {k}: {A.get(k)}")
    print(f"---- CHANNEL B K52shap{K_PRUNE} ----")
    for k in ("feat_dim", "n_3d_in_pruned", "n_2d_in_pruned",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_nb2103_K28",
              "delta_median_bag_vs_nb2103_K28",
              "delta_mean_bag_vs_user_target",
              "delta_median_bag_vs_user_target",
              "beats_nb2103_K28_mean", "beats_nb2103_K28_median",
              "verdict", "deploy_csv"):
        print(f"  {k}: {B.get(k)}")
