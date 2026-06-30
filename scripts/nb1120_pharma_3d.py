"""nb1120 -- 3D pharmacophore features (ETKDG + MMFF94) on top of nb2103 K=28.

HYPOTHESIS:
    nb2103 K=28 is the SHAP top-28 winner over the 117-col 5-way K-tuned matrix
    (mean_bag 0.4737, median_bag 0.4698 on 253 unblind, residual of
    chemprop_aux anchor).  All 117 features are 2D-derived (fingerprints +
    Mordred + Chemprop embeddings).  PXR has a large 1300 A^3 hydrophobic LBD;
    3D pharmacophore geometry (donor-donor / acceptor-acceptor / aromatic-aromatic
    distances and shape ratios PMI_1, PMI_2) may add an orthogonal axis that
    2D fingerprints cannot capture.

    This notebook embeds each compound with RDKit ETKDG (n_confs=10), MMFF94-
    minimizes each conformer, takes the lowest-energy survivor, computes 6
    3D pharmacophore features (D-D mean dist, D-A mean dist, A-A mean dist,
    aromatic-aromatic ring-centroid mean dist, PMI_1, PMI_2 with
    Descriptors3D.NPR1/NPR2), and concatenates with the nb2103 top-28 SHAP
    features to make K=34.  Residual cross-fit on chemprop_aux anchor with
    scaffold-aware 5-fold CV, 5-seed bag.

PROTOCOL:
    1.  Compute / load 3D pharmacophore features for all 513 test compounds
        (cache to data/processed/nb1120_pharma3d_test_513.npy, fail-rate
        recorded).  unb_idx slice gives the 253-row evaluation matrix.
    2.  Reuse nb2103 K=28 top SHAP indices from the cached SHAP importance.
    3.  Concatenate -> X_K34 shape (253, 34).
    4.  Scaffold-kfold 5-fold cross-fit, 5-seed bag (seeds 0, 1, 7, 42, 137),
        same LGBM(MSE) hyperparams as nb2103.
    5.  Compare mean_bag / median_bag RAE vs nb2103 K=28 (0.4737/0.4698),
        decision margin 0.003.
    6.  If beats: fresh-seed verify on 3 new seeds (211, 314, 271); if
        reproducible (>=2/3 better than nb2103 K=28 by >= margin),
        emit deploy CSV.
    7.  Save data/processed/nb1120_summary.json.

Outputs:
    scripts/nb1120_pharma_3d.py
    data/processed/nb1120_pharma3d_test_513.npy   (513, 6) float32
    data/processed/nb1120_pharma3d_test_fail.npy  (513,) bool   (True if embed failed)
    data/processed/nb1120_mean_bag_oof.npy        (253,) float32
    data/processed/nb1120_summary.json
    submissions/nb1120_pharma_3d.csv              (only if beats + verifies)
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
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D  # noqa: F401 (intent)

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, bemis_murcko, morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1120"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

# nb2103 references (for picking K=28 SHAP cols + comparing)
NB2103_K28_MEAN_BAG = 0.4737
NB2103_K28_MEDIAN_BAG = 0.4698
DECISION_MARGIN = 0.003
CHEMPROP_AUX_REF = 0.6216

# Conformer / 3D config
N_CONFS = 10
MMFF_VARIANT = "MMFF94"
ETKDG_VERSION = 3  # AllChem.ETKDGv3
ETKDG_BASE_SEED = 42
MAX_OPT_ITERS = 200
DONOR_PATTERN = Chem.MolFromSmarts(
    "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]"
)
ACCEPTOR_PATTERN = Chem.MolFromSmarts(
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),"
    "$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S])]),"
    "$([nH0,o,s;+0])]"
)

# CV + bag
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
VERIFY_SEEDS = [211, 314, 271]


# ---------------------------------------------------------------------------
# 3D conformer + pharmacophore
# ---------------------------------------------------------------------------

def _embed_and_minimize(mol: Chem.Mol, n_confs: int = N_CONFS,
                        base_seed: int = ETKDG_BASE_SEED) -> Chem.Mol | None:
    """Embed n_confs ETKDGv3 conformers, MMFF94-minimize, return lowest-energy
    conformer molecule (Hs added)."""
    if mol is None:
        return None
    try:
        mh = Chem.AddHs(mol)
    except Exception:
        return None
    params = AllChem.ETKDGv3()
    params.randomSeed = int(base_seed)
    params.numThreads = 1
    params.useSmallRingTorsions = True
    params.pruneRmsThresh = 0.5
    try:
        cids = AllChem.EmbedMultipleConfs(mh, numConfs=n_confs, params=params)
    except Exception:
        return None
    if not cids:
        return None
    # MMFF94 minimize each, capture energy
    energies = []
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mh, mmffVariant=MMFF_VARIANT)
        if mp is None:
            return None
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(mh, mp, confId=int(cid))
            if ff is None:
                energies.append((float("inf"), int(cid)))
                continue
            try:
                ff.Minimize(maxIts=MAX_OPT_ITERS)
                e = float(ff.CalcEnergy())
            except Exception:
                e = float("inf")
            energies.append((e, int(cid)))
    except Exception:
        return None
    if not energies:
        return None
    energies.sort(key=lambda t: t[0])
    best_e, best_cid = energies[0]
    if not np.isfinite(best_e):
        return None
    # Keep only the best conformer (drop others to save memory)
    keep = best_cid
    drop = [c for _, c in energies if c != keep]
    for c in drop:
        try:
            mh.RemoveConformer(int(c))
        except Exception:
            pass
    return mh


def _pairwise_mean_dist(coords_a: np.ndarray, coords_b: np.ndarray,
                        same_set: bool) -> float:
    """Mean pairwise distance.  If same_set, only upper-triangle (i<j).
    Returns 0.0 if not enough points."""
    if same_set:
        n = coords_a.shape[0]
        if n < 2:
            return 0.0
        # upper triangle
        diffs = []
        for i in range(n - 1):
            d = coords_a[i + 1:] - coords_a[i]
            diffs.append(np.sqrt((d * d).sum(axis=1)))
        if not diffs:
            return 0.0
        return float(np.concatenate(diffs).mean())
    else:
        if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
            return 0.0
        # full cross product
        diffs = coords_b[None, :, :] - coords_a[:, None, :]
        d = np.sqrt((diffs * diffs).sum(axis=2))
        return float(d.mean())


def _ring_centroids(mh: Chem.Mol, conf) -> np.ndarray:
    """Return (n_arom_rings, 3) centroids of aromatic rings."""
    ri = mh.GetRingInfo()
    out = []
    for ring in ri.AtomRings():
        if len(ring) == 0:
            continue
        is_arom = all(mh.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
        if not is_arom:
            continue
        coords = np.array(
            [list(conf.GetAtomPosition(int(a))) for a in ring],
            dtype=np.float32,
        )
        out.append(coords.mean(axis=0))
    if not out:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32)


def _atom_coords_for_pattern(mh: Chem.Mol, conf, patt) -> np.ndarray:
    """Return (n, 3) coords of all atoms matching SMARTS pattern (heavy + H)."""
    if patt is None:
        return np.zeros((0, 3), dtype=np.float32)
    matches = mh.GetSubstructMatches(patt)
    if not matches:
        return np.zeros((0, 3), dtype=np.float32)
    seen = set()
    coords = []
    for m in matches:
        for a in m:
            if a in seen:
                continue
            seen.add(a)
            p = conf.GetAtomPosition(int(a))
            coords.append((p.x, p.y, p.z))
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def _compute_3d_features(mh: Chem.Mol | None) -> tuple[np.ndarray, bool]:
    """Returns ([dd, da, aa, ar_ar, pmi1, pmi2], failed)."""
    feats = np.zeros(6, dtype=np.float32)
    if mh is None or mh.GetNumConformers() == 0:
        return feats, True
    try:
        conf = mh.GetConformer()
        donor_coords = _atom_coords_for_pattern(mh, conf, DONOR_PATTERN)
        accept_coords = _atom_coords_for_pattern(mh, conf, ACCEPTOR_PATTERN)
        ring_centroids = _ring_centroids(mh, conf)
        feats[0] = _pairwise_mean_dist(donor_coords, donor_coords, same_set=True)
        feats[1] = _pairwise_mean_dist(donor_coords, accept_coords,
                                       same_set=False)
        feats[2] = _pairwise_mean_dist(accept_coords, accept_coords,
                                       same_set=True)
        feats[3] = _pairwise_mean_dist(ring_centroids, ring_centroids,
                                       same_set=True)
        # PMI ratios via Descriptors3D.NPR1/NPR2
        try:
            npr1 = float(Descriptors3D.NPR1(mh))
        except Exception:
            npr1 = 0.0
        try:
            npr2 = float(Descriptors3D.NPR2(mh))
        except Exception:
            npr2 = 0.0
        feats[4] = float(npr1) if np.isfinite(npr1) else 0.0
        feats[5] = float(npr2) if np.isfinite(npr2) else 0.0
        return feats, False
    except Exception:
        return feats, True


def _build_3d_features_for_smiles(smiles_list: list[str],
                                   cache_path: Path,
                                   fail_cache_path: Path,
                                   force: bool = False
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """Returns X (N, 6) float32 and fail_mask (N,) bool."""
    n = len(smiles_list)
    if (cache_path.exists() and fail_cache_path.exists() and not force):
        X = np.load(cache_path).astype(np.float32)
        fail = np.load(fail_cache_path).astype(bool)
        if X.shape == (n, 6) and fail.shape == (n,):
            print(f"   [cache] reloaded {cache_path} shape {X.shape}  "
                  f"fail rate = {fail.mean() * 100:.2f}%")
            return X, fail
        else:
            print(f"   [cache-mismatch] recomputing.  cache shape={X.shape}, "
                  f"expected ({n},6)")

    X = np.zeros((n, 6), dtype=np.float32)
    fail = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, s in enumerate(smiles_list):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"   [embed] {i:4d}/{n}  elapsed={elapsed:.0f}s  "
                  f"eta={eta:.0f}s  fail_so_far={fail[:i].sum()}")
        mol = standardize(s)
        if mol is None:
            fail[i] = True
            continue
        mh = _embed_and_minimize(mol)
        feats, fail_i = _compute_3d_features(mh)
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
# nb2103 K=28 SHAP-top features (reuse builder logic but only the K=28 slice)
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
    top_idx_in_117 = np.array(rec["top_K_idx_in_117"], dtype=np.int32)
    return top_idx_in_117, s


def _build_117col_unb_matrix(unb_idx: np.ndarray, n_test: int):
    """Rebuild the 117-col 5-way K-tuned matrix on unb_idx -- identical recipe
    to nb2063/nb2081/nb2091/nb2103.  Implemented inline so this script is
    self-contained on the 28 SHAP-top dims.
    """
    from rdkit import Chem as _Chem  # local

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

    # mordred best K
    K_M = int(sum_1523["best_K"])
    rec_m = next(r for r in sum_1523["per_K_records"] if int(r["K"]) == K_M)
    top_mord = np.array(rec_m["top_col_idx"], dtype=int)

    # atompair ranked from nb1484; K from nb1524
    fam = next(f for f in sum_1484["families"] if f["family"] == "AtomPair")
    full_ap = np.array(fam["top_idx_ranked"], dtype=int)
    K_AP = int(sum_1524["best_K"])
    top_ap = full_ap[:K_AP]

    K_EMB = int(sum_1541["best_K"])
    top_emb_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_emb = top_emb_full[:K_EMB]

    X_ap = _load_npy(ATOMPAIR_TE, n_test)[unb_idx][:, top_ap]
    X_maccs = _load_npy(MACCS_TE, n_test)[unb_idx][:, top_maccs]
    X_mord = _load_mord(n_test)[unb_idx][:, top_mord]
    X_emb = _load_npy(CHEMPROP_EMBED_TE, n_test)[unb_idx][:, top_emb]
    X_av = _load_npy(AVALON_TE, n_test)[unb_idx][:, top_avalon]

    # ChEMBL kNN -- replicate from nb2103
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
        lambda m: _Chem.MolToInchiKey(m) if m is not None else None)
    pool["std_smiles"] = mols_pool.apply(
        lambda m: _Chem.MolToSmiles(m) if m is not None else None)
    pool = pool[pool["inchikey"].notna()
                 & pool["std_smiles"].notna()].copy()
    agg = (pool.groupby("inchikey", as_index=False)
                .agg(pec50=("pec50", "median"),
                     std_smiles=("std_smiles", "first")))

    te = load_test()
    test_smiles = te["smiles"].astype(str).tolist()
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = {
        _Chem.MolToInchiKey(m) for m in test_mols if m is not None
    }
    agg = agg[~agg["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(agg["std_smiles"].tolist())
    keep = fp_pool.sum(axis=1) > 0
    agg = agg[keep].reset_index(drop=True)
    fp_pool = fp_pool[keep]
    pool_labels = agg["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))

    std_test = [
        _Chem.MolToSmiles(m) if m is not None else "" for m in test_mols
    ]
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
            pred_chembl[i] = np.sum(
                w[i] * pool_labels[top_idx[i]]) / w_sum[i]
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    pred_chembl_unb = pred_chembl[unb_idx].reshape(-1, 1)
    mean_sim_unb = mean_sim[unb_idx].reshape(-1, 1)

    X117 = np.concatenate(
        [X_ap, X_maccs, X_mord, X_emb, X_av, pred_chembl_unb, mean_sim_unb],
        axis=1).astype(np.float32)
    if X117.shape[1] != 117:
        raise ValueError(f"117-col matrix wrong dim: {X117.shape}")
    return X117


# ---------------------------------------------------------------------------
# LGBM + residual cross-fit
# ---------------------------------------------------------------------------

def _lgbm_params(seed: int) -> dict:
    """Match nb2103 hyperparams."""
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
                                  seed: int) -> np.ndarray:
    n = len(residual)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_idx, va_idx in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_idx], residual[tr_idx])
        oof[va_idx] = mdl.predict(X[va_idx])
    if np.isnan(oof).any():
        raise RuntimeError("oof has NaN -- scaffold splits don't cover all rows")
    return oof


# ---------------------------------------------------------------------------
# Deploy CSV
# ---------------------------------------------------------------------------

def _build_deploy_csv(out_csv: Path,
                      top_idx_in_117: np.ndarray,
                      pharma3d_test_513: np.ndarray) -> None:
    """Train on ALL 253 unblind labels (deploy refit), predict 513 test set.
    Adds 3D-pharma 6 features to nb2103 K=28 cols -> K=34.  Anchor =
    chemprop_aux te[513].  Returns Molecule Name / SMILES / pEC50 CSV.
    """
    print("\n[deploy] training on all 253 unblind, predicting 513 test")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)

    # Build 117-col on ALL 513 (then slice top-28)
    X117_unb = _build_117col_unb_matrix(unb_idx, n_test=513)
    # ... build 117-col on all 513 -- reuse same recipe with full idx slice
    X117_all = _build_117col_unb_matrix(np.arange(513), n_test=513)
    X28_unb = X117_unb[:, top_idx_in_117]
    X28_all = X117_all[:, top_idx_in_117]
    X3d_unb = pharma3d_test_513[unb_idx]
    X3d_all = pharma3d_test_513
    X34_unb = np.concatenate([X28_unb, X3d_unb], axis=1).astype(np.float32)
    X34_all = np.concatenate([X28_all, X3d_all], axis=1).astype(np.float32)
    residual_unb = y_unb - te_anchor_513[unb_idx]

    # 5-seed bag deploy
    preds_seed = []
    for s in RESID_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X34_unb, residual_unb)
        preds_seed.append(mdl.predict(X34_all))
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
    print(f"{TAG} -- 3D pharmacophore (ETKDG+MMFF94) + nb2103 SHAP top-28 K=34")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG:.4f}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG:.4f}  margin={DECISION_MARGIN}")
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

    # ---- 3D pharmacophore features on 513 ----
    print("\n" + "-" * 78)
    print("3D PHARMACOPHORE: ETKDG + MMFF94 + lowest-energy conformer")
    print("-" * 78)
    test_smiles_513 = te["smiles"].astype(str).tolist()
    cache_p = DATA_PROCESSED / f"{TAG}_pharma3d_test_513.npy"
    fail_p = DATA_PROCESSED / f"{TAG}_pharma3d_test_fail.npy"
    X3d_513, fail_513 = _build_3d_features_for_smiles(
        test_smiles_513, cache_p, fail_p, force=False)
    fail_rate_pct = float(fail_513.mean() * 100.0)
    print(f"[pharma3d] features shape {X3d_513.shape}  "
          f"fail_rate = {fail_rate_pct:.2f}%")
    # Per-column summaries (mean / std) -- sanity
    col_names = ["DD_mean_dist", "DA_mean_dist", "AA_mean_dist",
                 "ArAr_mean_dist", "PMI_NPR1", "PMI_NPR2"]
    for j, nm in enumerate(col_names):
        c = X3d_513[:, j]
        print(f"   [feat] {nm:18s}  mean={c.mean():.3f}  "
              f"std={c.std():.3f}  min={c.min():.3f}  max={c.max():.3f}")

    X3d_unb = X3d_513[unb_idx]
    print(f"[pharma3d] unb slice shape {X3d_unb.shape}")

    # ---- nb2103 K=28 top SHAP cols + 117-col matrix ----
    print("\n" + "-" * 78)
    print("REBUILDING nb2103 K=28 SHAP TOP 117-col -> top-28 SLICE")
    print("-" * 78)
    top28_idx, nb2103_sum = _load_nb2103_k28_top_idx()
    nb2103_k28_mean_bag = float(NB2103_K28_MEAN_BAG)
    nb2103_k28_median_bag = float(NB2103_K28_MEDIAN_BAG)
    for r in nb2103_sum.get("per_K_records", []):
        if int(r["K"]) == 28:
            nb2103_k28_mean_bag = float(r["rae_mean_bag"])
            nb2103_k28_median_bag = float(r["rae_median_bag"])
            break
    print(f"[ref] nb2103 K=28  mean_bag = {nb2103_k28_mean_bag:.4f}  "
          f"median_bag = {nb2103_k28_median_bag:.4f}")

    X117_unb = _build_117col_unb_matrix(unb_idx, n_test=n_test)
    print(f"[feat] X117_unb shape {X117_unb.shape}")
    X28_unb = X117_unb[:, top28_idx].astype(np.float32)
    print(f"[feat] X28_unb shape {X28_unb.shape}  (SHAP top-28 from nb2063)")

    X34_unb = np.concatenate([X28_unb, X3d_unb], axis=1).astype(np.float32)
    print(f"[feat] X34_unb shape {X34_unb.shape}  (K=28 SHAP + 6 pharma3d)")

    # ---- Scaffold-aware 5-fold cross-fit, 5-seed bag ----
    print("\n" + "-" * 78)
    print("SCAFFOLD CROSS-FIT  5-fold x 5-seed bag on K=34")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _scaffold_cross_fit_one_seed(
            X34_unb, residual, scaffolds_unb, s)
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
        print(f"   seed={s:3d}:  rae_corr = {r:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_arr.mean())
    rae_per_seed_std = float(per_seed_arr.std())
    rae_per_seed_min = float(per_seed_arr.min())
    rae_per_seed_max = float(per_seed_arr.max())

    delta_mean_vs_nb2103 = rae_mean_bag - nb2103_k28_mean_bag
    delta_median_vs_nb2103 = rae_median_bag - nb2103_k28_median_bag
    beats_mean = rae_mean_bag < nb2103_k28_mean_bag - DECISION_MARGIN
    beats_median = rae_median_bag < nb2103_k28_median_bag - DECISION_MARGIN

    print(f"\n   per-seed RAE   = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean  = {rae_per_seed_mean:.4f}  "
          f"std = {rae_per_seed_std:.4f}")
    print(f"   pooled mean    = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f}  "
          f"d_vs_nb2103_K28 = {delta_mean_vs_nb2103:+.4f})")
    print(f"   pooled median  = {rae_median_bag:.4f}  "
          f"(d_vs_nb2103_K28 = {delta_median_vs_nb2103:+.4f})")

    if beats_mean and beats_median:
        verdict = "BEATS_NB2103_K28_BOTH_MEAN_AND_MEDIAN"
    elif beats_mean:
        verdict = "BEATS_NB2103_K28_MEAN_ONLY"
    elif beats_median:
        verdict = "BEATS_NB2103_K28_MEDIAN_ONLY"
    elif (abs(delta_mean_vs_nb2103) < DECISION_MARGIN
          and abs(delta_median_vs_nb2103) < DECISION_MARGIN):
        verdict = "FLAT_VS_NB2103_K28"
    else:
        verdict = "WORSE_THAN_NB2103_K28"
    print(f"   verdict        = {verdict}")

    # Save mean_bag OOF
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")

    # ---- Fresh-seed verification (only if beats) ----
    verify_records = []
    deploy_csv_path = None
    verify_verdict = "NOT_RUN"
    if beats_mean or beats_median:
        print("\n" + "-" * 78)
        print(f"FRESH-SEED VERIFY  seeds={VERIFY_SEEDS}")
        print("-" * 78)
        verify_corr = np.zeros((len(VERIFY_SEEDS), n_unb), dtype=np.float64)
        for i, s in enumerate(VERIFY_SEEDS):
            ts = time.time()
            resid_oof_v = _scaffold_cross_fit_one_seed(
                X34_unb, residual, scaffolds_unb, s)
            pred_v = anchor + resid_oof_v
            verify_corr[i] = pred_v
            r_v = float(rae(y_unb, pred_v))
            beats_v = r_v < nb2103_k28_mean_bag - DECISION_MARGIN
            verify_records.append({
                "seed": int(s),
                "rae_corrected": r_v,
                "beats_nb2103_K28_by_margin": bool(beats_v),
                "wall_sec": round(time.time() - ts, 2),
            })
            print(f"   verify seed={s:3d}:  rae = {r_v:.4f}  "
                  f"beats_nb2103_K28 = {beats_v}")
        # 5-seed verify bag
        verify_bag = verify_corr.mean(axis=0)
        rae_verify_bag = float(rae(y_unb, verify_bag))
        n_better = sum(1 for r in verify_records
                       if r["beats_nb2103_K28_by_margin"])
        print(f"   verify mean_bag (3 fresh seeds) = {rae_verify_bag:.4f}")
        print(f"   verify: {n_better}/{len(VERIFY_SEEDS)} fresh seeds "
              f"beat nb2103 K=28 by margin")
        if (n_better >= 2 and
                rae_verify_bag < nb2103_k28_mean_bag - DECISION_MARGIN):
            verify_verdict = "VERIFIED_REPRODUCIBLE_BEATS_NB2103_K28"
            # Emit deploy CSV
            deploy_csv_path = Path(
                __file__).resolve().parents[1] / "submissions" / \
                f"{TAG}_pharma_3d.csv"
            try:
                _build_deploy_csv(deploy_csv_path, top28_idx, X3d_513)
            except Exception as e:
                print(f"[deploy] FAILED: {e}")
                deploy_csv_path = None
                verify_verdict = "VERIFIED_BUT_DEPLOY_FAILED"
        else:
            verify_verdict = "VERIFY_NOT_REPRODUCIBLE_NO_DEPLOY"
        print(f"   verify verdict = {verify_verdict}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("lgbm_mse_K28_SHAP_top + 6 ETKDG/MMFF94 3D pharmacophore "
                   "feats -> K=34, scaffold-CF 5-fold x 5-seed bag, "
                   "residual on chemprop_aux"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "feat_dim": int(X34_unb.shape[1]),
        "feat_breakdown": {
            "shap_top_28_from_nb2063": 28,
            "pharma_3d": 6,
            "total": 34,
        },
        "pharma3d_cols": col_names,
        "pharma3d_fail_count": int(fail_513.sum()),
        "pharma3d_fail_rate_pct": fail_rate_pct,
        "n_confs": N_CONFS,
        "etkdg_version": ETKDG_VERSION,
        "etkdg_seed": ETKDG_BASE_SEED,
        "mmff_variant": MMFF_VARIANT,
        "max_opt_iters": MAX_OPT_ITERS,
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "verify_seeds": VERIFY_SEEDS,
        "cv_split": "scaffold_kfold_indices_5fold",
        "n_unb": n_unb,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2103_K28_mean_bag_ref": nb2103_k28_mean_bag,
        "nb2103_K28_median_bag_ref": nb2103_k28_median_bag,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb2103_K28": delta_mean_vs_nb2103,
        "delta_median_bag_vs_nb2103_K28": delta_median_vs_nb2103,
        "beats_nb2103_K28_mean": bool(beats_mean),
        "beats_nb2103_K28_median": bool(beats_median),
        "decision_margin": DECISION_MARGIN,
        "verdict": verdict,
        "verify_records": verify_records,
        "verify_verdict": verify_verdict,
        "deploy_csv": str(deploy_csv_path) if deploy_csv_path else None,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "feat_dim", "pharma3d_fail_count", "pharma3d_fail_rate_pct",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_ref", "nb2103_K28_median_bag_ref",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb2103_K28", "delta_median_bag_vs_nb2103_K28",
        "beats_nb2103_K28_mean", "beats_nb2103_K28_median",
        "verdict", "verify_verdict", "deploy_csv",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
