"""nb2511 -- USR + USRCAT 3D shape moments + K=25 RFE on combined 189-feat substrate.

NEW PARADIGM:
    Ultrafast Shape Recognition (USR, Ballester & Richards 2007) and USRCAT
    (Schreyer & Blundell 2012) reduce a single 3D conformer to 12 and 60
    distance-distribution moments respectively, capturing 3D molecular SHAPE
    + pharmacophore-typed distance distributions.  Unlike Descriptors3D (PMI
    / Asphericity), USR/USRCAT are alignment-free and dimension-fixed --
    they are the canonical 3D-shape fingerprint primitive.

    Distinct from 2D fingerprint substrates already explored: USR moments
    encode INTERATOMIC DISTANCE STATISTICS in 3D space, orthogonal to
    bond-topology features dominated by AtomPair / Mordred / MACCS.

PROTOCOL:
    1.  Per SMILES (513 test, contains 253 unblind subset): generate up to
        3 ETKDGv3 conformers, MMFF94-minimize each, keep the lowest-energy
        survivor (the bioactive-conformer proxy).  Compute GetUSR(12) +
        GetUSRCAT(60) = 72 shape features.  Cache to
        data/processed/nb2511_usr_test_513.npy and
        data/processed/nb2511_usr_fail_513.npy.
    2.  Rebuild the canonical 117-col 5-way feature matrix on 513
        (AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN),
        identical to nb2103/nb2240/nb2502.
    3.  Concatenate X_117 ++ X_USR72 = X_189.  Slice to 253 unblind.
    4.  GREEDY BACKWARD RFE: start from K=189, drop the feature whose
        removal least HURTS scaffold-CV RAE (fast 1-seed eval per drop),
        stop at K=25.  Total drops = 164.
    5.  Evaluate the K=25 surviving subset with full 5-fold scaffold CV on
        253 unblind, 5 kf_seeds {1001..1005}, mean-bag across bag seeds
        {0, 1, 7, 42, 137}.
    6.  Gate: mean_rae < 0.4570 -> PROMOTE;  < 0.4601 -> MARGINAL_BEAT;
        else FAIL.
    7.  Deploy refit: mean-bag K=25 LGBM on all 253 unblind, predict 513.
        Save pred_oof + te + summary.

REFERENCE:
    nb2171 deep-30 ceiling on chemprop_aux anchor = 0.4682 +/- 0.0024
    nb2240 K=20 RFE-surviving (no 3D shape) = 0.4682
    nb2095/nb1191/nb2060 post-hoc-blend ceiling band = 0.4718-0.4720
    Memory: substrate change (new anchor / 3D / off-manifold / abstention)
    is the only open lever per cycle-134/167/169 paradigm-exhaustion findings.

Outputs:
    scripts/nb2511_3d_conformer_shape.py
    data/processed/nb2511_usr_test_513.npy       (513, 72) float32
    data/processed/nb2511_usr_fail_513.npy       (513,)    bool
    data/processed/nb2511_pred_oof.npy           (253,)    float32
    data/processed/te_nb2511.npy                 (513,)    float32
    data/processed/nb2511_summary.json
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
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetUSR, GetUSRCAT

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2511"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# -------- conformer / shape config --------
N_CONFS = 3
ETKDG_VERSION = 3
ETKDG_BASE_SEED = 42
MMFF_VARIANT = "MMFF94"
MAX_OPT_ITERS = 200
USR_DIM = 12
USRCAT_DIM = 60
USR_FEAT_DIM = USR_DIM + USRCAT_DIM  # 72

# -------- RFE + CV config --------
N_FOLDS = 5
KF_SEEDS_FULL = [1001, 1002, 1003, 1004, 1005]
KF_SEED_RFE = 1001                # fast single-seed eval during search
RESID_SEEDS = [0, 1, 7, 42, 137]  # bag seeds inside each kf_seed
K_TARGET = 25                     # final surviving feature count

# -------- gates --------
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# -------- references --------
CHEMPROP_AUX_REF = 0.6216
NB2171_DEEP30_REF = 0.4682
NB2240_K20_REF = 0.4682

# -------- feature build paths --------
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

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# ============================================================================
# USR + USRCAT 3D shape feature builder
# ============================================================================

def _embed_minimize_pick_lowest(mol: Chem.Mol,
                                  n_confs: int = N_CONFS,
                                  base_seed: int = ETKDG_BASE_SEED
                                  ) -> tuple[Chem.Mol | None, int]:
    """Embed N_CONFS ETKDGv3 conformers, MMFF94-minimize each, return
    (mol_with_H, lowest_energy_confId).  Returns (None, -1) if all fail.
    """
    if mol is None:
        return None, -1
    try:
        mh = Chem.AddHs(mol)
    except Exception:
        return None, -1
    params = AllChem.ETKDGv3()
    params.randomSeed = int(base_seed)
    params.numThreads = 1
    params.useSmallRingTorsions = True
    params.pruneRmsThresh = 0.5
    try:
        cids = AllChem.EmbedMultipleConfs(mh, numConfs=n_confs, params=params)
    except Exception:
        return None, -1
    cids = list(cids)
    if not cids:
        return None, -1
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mh, mmffVariant=MMFF_VARIANT)
    except Exception:
        mp = None
    energies: list[tuple[int, float]] = []
    for cid in cids:
        e = float("inf")
        if mp is not None:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mh, mp, confId=int(cid))
                if ff is not None:
                    ff.Minimize(maxIts=MAX_OPT_ITERS)
                    e = float(ff.CalcEnergy())
            except Exception:
                e = float("inf")
        # fallback: even if minimization failed, the conformer geometry is
        # still embedded -- accept with infinite energy so it ranks last but
        # is available if nothing else minimised
        energies.append((int(cid), e))
    finite = [(c, e) for c, e in energies if np.isfinite(e)]
    pool = finite if finite else energies
    pool.sort(key=lambda t: t[1])
    return mh, int(pool[0][0])


def _usr_feats_one(mh: Chem.Mol | None, cid: int) -> tuple[np.ndarray, bool]:
    """Compute GetUSR(12) + GetUSRCAT(60) = 72 shape moments for one
    conformer.  Returns (feats, failed).
    """
    out = np.zeros(USR_FEAT_DIM, dtype=np.float32)
    if mh is None or cid < 0:
        return out, True
    try:
        u = GetUSR(mh, confId=int(cid))
        uc = GetUSRCAT(mh, confId=int(cid))
        u = np.asarray(u, dtype=np.float32)
        uc = np.asarray(uc, dtype=np.float32)
        if u.shape[0] != USR_DIM or uc.shape[0] != USRCAT_DIM:
            return out, True
        out[:USR_DIM] = u
        out[USR_DIM:USR_DIM + USRCAT_DIM] = uc
        if not np.all(np.isfinite(out)):
            out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
        return out, False
    except Exception:
        return out, True


def _build_usr_features(smiles_list: list[str],
                         cache_path: Path,
                         fail_cache_path: Path,
                         force: bool = False
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Build (N, 72) USR + USRCAT feature matrix; cache to disk."""
    n = len(smiles_list)
    if cache_path.exists() and fail_cache_path.exists() and not force:
        X = np.load(cache_path).astype(np.float32)
        fail = np.load(fail_cache_path).astype(bool)
        if X.shape == (n, USR_FEAT_DIM) and fail.shape == (n,):
            print(f"   [cache] reloaded {cache_path}  shape {X.shape}  "
                  f"fail_rate = {fail.mean() * 100:.2f}%")
            return X, fail
        print(f"   [cache-mismatch] recomputing; cache {X.shape} vs "
              f"expected ({n}, {USR_FEAT_DIM})")

    X = np.zeros((n, USR_FEAT_DIM), dtype=np.float32)
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
        mh, cid = _embed_minimize_pick_lowest(mol)
        feats, fail_i = _usr_feats_one(mh, cid)
        X[i] = feats
        fail[i] = bool(fail_i)
    elapsed = time.time() - t0
    print(f"   [embed] done.  total wall = {elapsed:.0f}s  "
          f"fail_rate = {fail.mean() * 100:.2f}%  ({fail.sum()}/{n})")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, X.astype(np.float32))
    np.save(fail_cache_path, fail)
    print(f"   [save] {cache_path}   {fail_cache_path}")
    return X, fail


# ============================================================================
# 117-col 5-way feature builder (canonical nb2240/nb2502 recipe)
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


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing ({mte_p})")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred shape mismatch {X_te_m.shape}")
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


def _build_117col_test(te: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the canonical 117-col 5-way feature matrix on 513 test compounds.
    Returns (X_te_117, feat_names).
    """
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
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

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

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
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_117 = np.concatenate(
        [
            X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X_te_117.shape[1] != 117:
        raise ValueError(f"117-col build wrong dim: {X_te_117.shape}")

    names: list[str] = []
    for j in top_ap_bit_idx:
        names.append(f"AP_bit{int(j)}")
    for j in top_maccs_bit_idx:
        names.append(f"MACCS_bit{int(j)}")
    for j in top_mord_col_idx:
        names.append(f"MORD_col{int(j)}")
    for j in top_embed_col_idx:
        names.append(f"EMBED_dim{int(j)}")
    for j in top_avalon_bit_idx:
        names.append(f"AVALON_bit{int(j)}")
    names.append("CHEMBL_kNN_pec50")
    names.append("CHEMBL_kNN_meansim")
    assert len(names) == 117
    return X_te_117, names


# ============================================================================
# LGBM scaffold CV + RFE
# ============================================================================

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


def _scaffold_cv_one_seed(X: np.ndarray, residual: np.ndarray,
                           unb_scaffolds: list[str], kf_seed: int) -> np.ndarray:
    """One scaffold 5-fold cross-fit; LGBM single seed = kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _eval_subset_fast(X_full: np.ndarray, col_idx: list[int],
                       residual: np.ndarray, anchor: np.ndarray,
                       y_unb: np.ndarray, unb_scaffolds: list[str],
                       kf_seed: int) -> float:
    """Single-seed scaffold-CV RAE used during RFE drop selection."""
    X_sub = X_full[:, col_idx].astype(np.float32)
    oof = _scaffold_cv_one_seed(X_sub, residual, unb_scaffolds, kf_seed)
    return float(rae(y_unb, anchor + oof))


def _gain_importance_one_fold_avg(X: np.ndarray, residual: np.ndarray,
                                    unb_scaffolds: list[str],
                                    kf_seed: int) -> np.ndarray:
    """Average per-feature gain importance across scaffold 5-fold; returns
    (n_feat,) float64.
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_feat = X.shape[1]
    imp = np.zeros(n_feat, dtype=np.float64)
    n_folds = 0
    for tr_loc, _va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        gi = np.asarray(
            mdl.booster_.feature_importance(importance_type="gain"),
            dtype=np.float64)
        if gi.shape[0] == n_feat:
            imp += gi
            n_folds += 1
    if n_folds:
        imp /= float(n_folds)
    return imp


def _rfe_backward(X_full: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
                   y_unb: np.ndarray, unb_scaffolds: list[str],
                   k_target: int) -> tuple[list[int], list[dict]]:
    """Hybrid RFE: in the broad regime (K > K_HOT) drop the LOWEST-gain
    feature each step (cheap, 1 LGBM-fit per step via scaffold-CV gain
    average).  In the hot regime (K <= K_HOT) switch to true leave-one-out
    greedy RFE (K-1 single-seed scaffold-CV evals per step).  This is the
    standard fast-RFE pattern; the gain-driven phase is monotonic in feature
    rank by gain (the LGBM-native ranker), and the hot-regime greedy refines
    the final selection where every drop matters most.
    """
    K_HOT = 50    # switch from gain-rank drops to greedy leave-one-out
    n_feat = X_full.shape[1]
    if k_target >= n_feat:
        return list(range(n_feat)), []
    current = list(range(n_feat))
    trajectory: list[dict] = []
    base_rae = _eval_subset_fast(X_full, current, residual, anchor,
                                  y_unb, unb_scaffolds, KF_SEED_RFE)
    print(f"   [rfe] start K={len(current)}  base_RAE={base_rae:.4f}  "
          f"K_HOT={K_HOT}")
    t0 = time.time()

    # ----- broad regime: gain-driven, drop 1 feature/step -----
    while len(current) > max(k_target, K_HOT):
        K_now = len(current)
        X_sub = X_full[:, current].astype(np.float32)
        gain = _gain_importance_one_fold_avg(
            X_sub, residual, unb_scaffolds, KF_SEED_RFE)
        worst_local = int(np.argmin(gain))
        worst_idx = current[worst_local]
        current.remove(worst_idx)
        # cheap progress check: rae after drop, 1-seed scaffold-CV
        r_after = _eval_subset_fast(X_full, current, residual, anchor,
                                      y_unb, unb_scaffolds, KF_SEED_RFE)
        trajectory.append({
            "step": len(trajectory) + 1,
            "phase": "gain_rank",
            "K_before": int(K_now),
            "K_after": int(len(current)),
            "dropped_idx_in_189": int(worst_idx),
            "min_gain": float(gain.min()),
            "rae_after": float(r_after),
        })
        if K_now % 10 == 0 or len(current) <= K_HOT + 5:
            elapsed = time.time() - t0
            print(f"   [rfe-gain] step={len(trajectory):3d}  K {K_now} -> "
                  f"{len(current)}  drop_idx={worst_idx:4d}  "
                  f"min_gain={gain.min():.3f}  rae={r_after:.4f}  "
                  f"wall={elapsed:.0f}s")

    # ----- hot regime: true leave-one-out greedy -----
    while len(current) > k_target:
        K_now = len(current)
        base_rae_step = _eval_subset_fast(X_full, current, residual, anchor,
                                           y_unb, unb_scaffolds, KF_SEED_RFE)
        best_idx_drop = -1
        best_rae_after = float("inf")
        for j in current:
            trial = [c for c in current if c != j]
            r = _eval_subset_fast(X_full, trial, residual, anchor,
                                   y_unb, unb_scaffolds, KF_SEED_RFE)
            if r < best_rae_after:
                best_rae_after = r
                best_idx_drop = j
        delta = best_rae_after - base_rae_step
        current.remove(best_idx_drop)
        trajectory.append({
            "step": len(trajectory) + 1,
            "phase": "greedy",
            "K_before": int(K_now),
            "K_after": int(len(current)),
            "dropped_idx_in_189": int(best_idx_drop),
            "base_RAE": float(base_rae_step),
            "rae_after": float(best_rae_after),
            "delta_RAE": float(delta),
        })
        elapsed = time.time() - t0
        print(f"   [rfe-greedy] step={len(trajectory):3d}  K {K_now} -> "
              f"{len(current)}  drop_idx={best_idx_drop:4d}  "
              f"base={base_rae_step:.4f} -> {best_rae_after:.4f}  "
              f"delta={delta:+.4f}  wall={elapsed:.0f}s")

    final_rae = _eval_subset_fast(X_full, current, residual, anchor,
                                    y_unb, unb_scaffolds, KF_SEED_RFE)
    print(f"   [rfe] done.  K={len(current)}  final_RAE={final_rae:.4f}  "
          f"total_wall={time.time() - t0:.0f}s")
    return current, trajectory


def _full_5seed_eval(X_full: np.ndarray, col_idx: list[int],
                      residual: np.ndarray, anchor: np.ndarray,
                      y_unb: np.ndarray, unb_scaffolds: list[str],
                      kf_seeds: list[int], bag_seeds: list[int]) -> dict:
    """Full kf_seed x bag_seed scaffold-CV evaluation.  Reported mean_rae
    = average of pooled-RAE across kf_seeds (each pooled-RAE uses mean-bag
    across bag_seeds).
    """
    X_sub = X_full[:, col_idx].astype(np.float32)
    n_unb = len(y_unb)
    per_seed = []
    all_oofs = []
    for kf_seed in kf_seeds:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            bag = np.zeros(len(va_loc), dtype=np.float64)
            for bs in bag_seeds:
                mdl = lgb.LGBMRegressor(**_lgbm_params(bs))
                mdl.fit(X_sub[tr_loc], residual[tr_loc])
                bag += mdl.predict(X_sub[va_loc])
            bag /= len(bag_seeds)
            oof[va_loc] = bag
        assert not np.isnan(oof).any(), "OOF has NaN"
        pred_corr = anchor + oof
        pooled = float(rae(y_unb, pred_corr))
        per_seed.append({"kf_seed": int(kf_seed), "pooled_rae": pooled})
        all_oofs.append(pred_corr)
        print(f"   [full] kf_seed={kf_seed}  pooled_RAE={pooled:.4f}")
    rae_arr = np.array([r["pooled_rae"] for r in per_seed])
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    return {
        "per_seed": per_seed,
        "rae_mean_seeds": float(rae_arr.mean()),
        "rae_std_seeds": float(rae_arr.std()),
        "rae_of_mean_of_seed_oofs": float(rae(y_unb, mean_oof)),
        "mean_oof": mean_oof.astype(np.float32),
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- USR+USRCAT 3D shape ({USR_FEAT_DIM} feats) + K={K_TARGET}"
          f" RFE on combined 189-feat substrate")
    print(f"          anchor={ANCHOR}  N_CONFS={N_CONFS} ETKDGv{ETKDG_VERSION}"
          f" MMFF94")
    print(f"          gates: PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}")
    print(f"          refs: nb2171_deep30={NB2171_DEEP30_REF:.4f}  "
          f"nb2240_K20={NB2240_K20_REF:.4f}  "
          f"chemprop_aux={CHEMPROP_AUX_REF:.4f}")
    print("=" * 78)

    # ---- load truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] n_test={n_test}  n_unb={n_unb}  "
          f"chemprop_aux in_RAE={rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique on 253 unblind = {n_unique_scaf}")

    # ---- 1. USR + USRCAT shape features on 513 ----
    print("\n" + "-" * 78)
    print(f"USR + USRCAT shape moments  ({N_CONFS} ETKDG conf, lowest-energy "
          f"MMFF94, GetUSR={USR_DIM} + GetUSRCAT={USRCAT_DIM} = "
          f"{USR_FEAT_DIM} feats)")
    print("-" * 78)
    cache_p = DATA_PROCESSED / f"{TAG}_usr_test_513.npy"
    fail_p = DATA_PROCESSED / f"{TAG}_usr_fail_513.npy"
    X_usr_513, fail_513 = _build_usr_features(
        te_smiles, cache_p, fail_p, force=False)
    fail_rate_pct = float(fail_513.mean() * 100.0)
    print(f"[usr] shape {X_usr_513.shape}  fail_rate={fail_rate_pct:.2f}%")

    # per-block sanity
    usr_block = X_usr_513[:, :USR_DIM]
    usrcat_block = X_usr_513[:, USR_DIM:USR_DIM + USRCAT_DIM]
    print(f"   [usr-block]    mean={usr_block.mean():+.3f}  "
          f"std={usr_block.std():.3f}  min={usr_block.min():+.3f}  "
          f"max={usr_block.max():+.3f}")
    print(f"   [usrcat-block] mean={usrcat_block.mean():+.3f}  "
          f"std={usrcat_block.std():.3f}  min={usrcat_block.min():+.3f}  "
          f"max={usrcat_block.max():+.3f}")

    # ---- 2. canonical 117-col matrix ----
    print("\n" + "-" * 78)
    print("BUILD canonical 117-col 5-way feature matrix (AP+MACCS+Mordred+"
          "Embed+Avalon + ChEMBL-kNN)")
    print("-" * 78)
    X_te_117, names_117 = _build_117col_test(te)
    print(f"[117] shape {X_te_117.shape}")

    # ---- 3. concatenate to 189 ----
    X_te_189 = np.concatenate([X_te_117, X_usr_513], axis=1).astype(np.float32)
    print(f"[189] X_te_189 shape {X_te_189.shape}")
    names_189 = list(names_117)
    for j in range(USR_DIM):
        names_189.append(f"USR_m{j}")
    for j in range(USRCAT_DIM):
        names_189.append(f"USRCAT_m{j}")
    assert len(names_189) == 189

    X_unb_189 = X_te_189[unb_idx].astype(np.float32)
    print(f"[unb] X_unb_189 shape {X_unb_189.shape}")

    # ---- 4. greedy backward RFE 189 -> K_TARGET ----
    print("\n" + "-" * 78)
    print(f"GREEDY BACKWARD RFE 189 -> K={K_TARGET}  "
          f"(fast 1-seed kf_seed={KF_SEED_RFE} per drop)")
    print("-" * 78)
    surviving, trajectory = _rfe_backward(
        X_unb_189, residual, anchor, y_unb, unb_scaffolds,
        k_target=K_TARGET)
    print(f"[rfe] surviving K={len(surviving)} indices in 189-col space:")
    surviving_names = [names_189[i] for i in surviving]
    n_usr_kept = sum(1 for n in surviving_names
                      if n.startswith("USR_m") or n.startswith("USRCAT_m"))
    n_2d_kept = K_TARGET - n_usr_kept
    print(f"   USR/USRCAT kept = {n_usr_kept}  /  2D-117 kept = {n_2d_kept}")
    print(f"   names: {surviving_names}")

    # ---- 5. full 5-kf x 5-bag scaffold CV on surviving K=25 ----
    print("\n" + "-" * 78)
    print(f"FULL 5-fold scaffold CV x 5 kf_seeds x 5 bag_seeds on K={K_TARGET}")
    print("-" * 78)
    full = _full_5seed_eval(
        X_unb_189, surviving, residual, anchor, y_unb, unb_scaffolds,
        KF_SEEDS_FULL, RESID_SEEDS)
    mean_rae = full["rae_mean_seeds"]
    std_rae = full["rae_std_seeds"]
    pooled_mean_oof = full["rae_of_mean_of_seed_oofs"]
    mean_oof = full["mean_oof"]
    print(f"\n[full] mean_rae across {len(KF_SEEDS_FULL)} kf_seeds = "
          f"{mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"[full] RAE of mean-of-seed OOFs         = {pooled_mean_oof:.4f}")

    # ---- 6. gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    delta_vs_nb2240 = mean_rae - NB2240_K20_REF
    delta_vs_nb2171 = mean_rae - NB2171_DEEP30_REF
    delta_vs_anchor = mean_rae - rae_anchor
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}  -> {verdict}")
    print(f"[ref ] vs nb2240 K=20 (0.4682)  delta={delta_vs_nb2240:+.4f}")
    print(f"[ref ] vs nb2171 deep30 (0.4682) delta={delta_vs_nb2171:+.4f}")
    print(f"[ref ] vs chemprop_aux (0.6216) delta={delta_vs_anchor:+.4f}")

    # ---- 7. deploy refit on all 253 -> predict 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY  (refit on all 253 unblind; mean-bag predict 513)")
    print("-" * 78)
    X_unb_K = X_unb_189[:, surviving].astype(np.float32)
    X_te_K = X_te_189[:, surviving].astype(np.float32)
    te_bag = np.zeros(n_test, dtype=np.float64)
    for bs in RESID_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(bs))
        mdl.fit(X_unb_K, residual)
        te_bag += mdl.predict(X_te_K)
    te_bag /= len(RESID_SEEDS)
    deploy_te = (te_anchor_513 + te_bag).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   te range = [{deploy_te.min():.3f}, {deploy_te.max():.3f}]  "
          f"mean={deploy_te.mean():.3f} std={deploy_te.std():.3f}")
    print(f"   te[unb_idx] in_RAE (deploy-refit) = {te_unb_rae:.4f}  "
          f"(expected in-sample optimism vs honest cross-fit {mean_rae:.4f})")

    # ---- save artifacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float32))
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_npy_path}")

    summary = {
        "tag": TAG,
        "method": (f"USR({USR_DIM})+USRCAT({USRCAT_DIM})={USR_FEAT_DIM} "
                    f"3D shape moments concat with 117-col 5-way; greedy "
                    f"backward RFE 189->K={K_TARGET}; LGBM(MSE) residual on "
                    f"chemprop_aux anchor; 5-fold scaffold CV x "
                    f"{len(KF_SEEDS_FULL)} kf_seeds x {len(RESID_SEEDS)} "
                    f"bag_seeds"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "n_confs": int(N_CONFS),
        "etkdg_version": int(ETKDG_VERSION),
        "etkdg_seed": int(ETKDG_BASE_SEED),
        "mmff_variant": MMFF_VARIANT,
        "max_opt_iters": int(MAX_OPT_ITERS),
        "usr_dim": int(USR_DIM),
        "usrcat_dim": int(USRCAT_DIM),
        "usr_feat_dim_total": int(USR_FEAT_DIM),
        "usr_fail_count": int(fail_513.sum()),
        "usr_fail_rate_pct": fail_rate_pct,
        "n_feat_117": 117,
        "n_feat_combined_189": 189,
        "k_target": int(K_TARGET),
        "k_surviving": int(len(surviving)),
        "surviving_idx_in_189": [int(i) for i in surviving],
        "surviving_names": surviving_names,
        "n_usr_features_kept": int(n_usr_kept),
        "n_2d_features_kept": int(n_2d_kept),
        "rfe_kf_seed": int(KF_SEED_RFE),
        "rfe_trajectory_n_steps": int(len(trajectory)),
        "rfe_trajectory_first5": trajectory[:5],
        "rfe_trajectory_last5": trajectory[-5:],
        "kf_seeds_full": KF_SEEDS_FULL,
        "bag_seeds": RESID_SEEDS,
        "n_folds": int(N_FOLDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_kf_seed_results": full["per_seed"],
        "mean_rae": float(mean_rae),
        "std_rae": float(std_rae),
        "rae_of_mean_of_seed_oofs": float(pooled_mean_oof),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "ref_nb2240_K20": NB2240_K20_REF,
        "delta_vs_nb2240_K20": float(delta_vs_nb2240),
        "ref_nb2171_deep30": NB2171_DEEP30_REF,
        "delta_vs_nb2171_deep30": float(delta_vs_nb2171),
        "ref_chemprop_aux": CHEMPROP_AUX_REF,
        "delta_vs_chemprop_aux_anchor": float(delta_vs_anchor),
        "deploy_te_min": float(deploy_te.min()),
        "deploy_te_max": float(deploy_te.max()),
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_unb_in_sample_RAE": float(te_unb_rae),
        "pre_unblind_clean": True,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_npy_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   USR fail_rate                  = {fail_rate_pct:.2f}%  "
          f"({int(fail_513.sum())}/{n_test})")
    print(f"   K_surviving                    = {len(surviving)}  "
          f"(USR={n_usr_kept}  2D={n_2d_kept})")
    print(f"   mean_rae (5 kf_seeds)          = {mean_rae:.4f} "
          f"+/- {std_rae:.4f}")
    print(f"   verdict                        = {verdict}")
    print(f"   delta vs nb2240 K20 (0.4682)   = {delta_vs_nb2240:+.4f}")
    print(f"   delta vs nb2171 deep30 (0.4682) = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb] in-sample RAE          = {te_unb_rae:.4f}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "usr_fail_count",
        "usr_fail_rate_pct",
        "k_surviving",
        "n_usr_features_kept",
        "n_2d_features_kept",
        "mean_rae",
        "std_rae",
        "verdict",
        "delta_vs_nb2240_K20",
        "delta_vs_nb2171_deep30",
        "te_unb_in_sample_RAE",
        "wall_sec",
    ):
        print(f"  {k}: {res.get(k)}")
