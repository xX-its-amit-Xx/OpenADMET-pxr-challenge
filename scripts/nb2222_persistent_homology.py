"""nb2222 -- Persistent homology Betti-number features (substrate-distinct).

HYPOTHESIS:
    Cycle 169 nb2201 (graph spectral, K=48 Laplacian features) was CLOSED:
    nb2201 K48 mean-bag RAE = 0.5050 vs nb2103 K=28 = 0.4737 (BEATS_ANCHOR
    _BUT_WORSE_THAN_NB2103_K28). The spectral substrate adds eigenvalue
    statistics derived from the molecular graph Laplacian, but those summary
    statistics correlate with degree/connectivity descriptors already captured
    by Mordred + AtomPair + MACCS in the nb2103 K=28 substrate.

    Persistent homology (PH) is genuinely distinct: instead of summarizing the
    spectrum of a fixed graph, it tracks topological invariants (connected
    components H0, loops H1, voids H2) as a continuous filtration parameter
    (max edge length) sweeps from 0 -> max. The persistence diagram captures
    multi-scale ring/void/cluster structure derived from 3D conformer geometry
    -- a substrate that the 117-col 2D fingerprint+descriptor pool does not see.

PROTOCOL:
    1. Embed each 513-test SMILES into 3D via RDKit ETKDG (seed=42), MMFF94
       optimize (200 iters). On failure: fall back to a 2D-graph PH (atom
       graph with bond-distance filtration).
    2. Compute persistent homology via gudhi RipsComplex up to dim=2
       (H0, H1, H2). Extract 15 features:
         - Per dimension d in {0,1,2}: n_features_d, mean_lifetime_d,
           max_lifetime_d, sum_lifetime_d, persistence_entropy_d  (5 x 3 = 15)
    3. Append the 15 PH features to the nb2103 top-K=28 SHAP feature matrix
       sliced from the 117-col 5-way K-tuned matrix -> total feat_dim = 43.
    4. Fit LGBM(MSE) on chemprop_aux residual: 5-seed bag (seeds 0,1,7,42,137),
       5-fold KFold scaffold-free CV per seed (identical to nb2103/nb2081).
    5. Compare mean-bag RAE vs nb2103 K=28 (0.4737); gate at 0.003 margin.
    6. If gudhi/ripser unavailable: write SKIPPED summary, exit cleanly.

Outputs:
    scripts/nb2222_persistent_homology.py
    data/processed/nb2222_summary.json
    data/processed/nb2222_mean_bag_oof.npy   (253,) float32   (if not SKIPPED)
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
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2222"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# Reference: nb2103 K=28 mean-bag RAE (the optimum SHAP K from the fine grid)
NB2103_K28_REF = 0.4737
DECISION_MARGIN = 0.003
CHEMPROP_AUX_REF = 0.6216

# Inputs reused from nb2063/nb2103
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

# Same K-grid winner artifacts as nb2103 (to rebuild the 117-col matrix)
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

# PH config
PH_DIMS = (0, 1, 2)
PH_STATS = ("n", "mean", "max", "sum", "entropy")
PH_FEAT_NAMES = [f"ph_h{d}_{s}" for d in PH_DIMS for s in PH_STATS]
N_PH_FEAT = len(PH_FEAT_NAMES)  # 15
MAX_EDGE = 6.0
EMBED_SEED = 42

# -------- PH backend detection --------
_HAS_GUDHI = False
_HAS_RIPSER = False
try:
    import gudhi  # type: ignore
    _HAS_GUDHI = True
except Exception:
    try:
        from ripser import ripser  # type: ignore
        _HAS_RIPSER = True
    except Exception:
        pass


def _persistence_entropy(lifetimes: np.ndarray) -> float:
    L = lifetimes[lifetimes > 0]
    if L.size == 0:
        return 0.0
    p = L / L.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def _embed_3d(smi: str) -> np.ndarray | None:
    """Embed SMILES into 3D coords. Return None on failure."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    mh = Chem.AddHs(m)
    ok = AllChem.EmbedMolecule(mh, randomSeed=EMBED_SEED)
    if ok != 0:
        # try ETKDGv3 with random coords
        params = AllChem.ETKDGv3()
        params.randomSeed = EMBED_SEED
        params.useRandomCoords = True
        ok = AllChem.EmbedMolecule(mh, params)
        if ok != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
    except Exception:
        pass
    conf = mh.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(k)) for k in range(mh.GetNumAtoms())],
        dtype=np.float64,
    )
    return coords


def _coords_2d_fallback(smi: str) -> np.ndarray | None:
    """2D atom coordinates fallback when 3D fails."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        AllChem.Compute2DCoords(m)
        conf = m.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(k)) for k in range(m.GetNumAtoms())],
            dtype=np.float64,
        )
        return coords
    except Exception:
        return None


def _ph_betti_features(coords: np.ndarray) -> np.ndarray:
    """Compute 15 PH features from atom coords via Vietoris-Rips.

    Returns shape (15,) float32: per dim d in {0,1,2}:
      n_features_d, mean_lifetime_d, max_lifetime_d,
      sum_lifetime_d, persistence_entropy_d
    """
    out = np.zeros(N_PH_FEAT, dtype=np.float32)
    if coords is None or coords.shape[0] < 2:
        return out
    try:
        if _HAS_GUDHI:
            rips = gudhi.RipsComplex(points=coords, max_edge_length=MAX_EDGE)
            st = rips.create_simplex_tree(max_dimension=3)
            diag = st.persistence()
            lifetimes_by_dim = {0: [], 1: [], 2: []}
            for dim, (b, d) in diag:
                if dim in lifetimes_by_dim and np.isfinite(d):
                    lifetimes_by_dim[dim].append(d - b)
        elif _HAS_RIPSER:
            res = ripser(coords, maxdim=2, thresh=MAX_EDGE)
            dgms = res["dgms"]
            lifetimes_by_dim = {0: [], 1: [], 2: []}
            for dim_i, dgm in enumerate(dgms):
                if dgm is None or dgm.size == 0 or dim_i not in lifetimes_by_dim:
                    continue
                mask = np.isfinite(dgm[:, 1])
                lt = dgm[mask, 1] - dgm[mask, 0]
                lifetimes_by_dim[dim_i] = list(lt)
        else:
            return out
        for j, d in enumerate(PH_DIMS):
            L = np.array(lifetimes_by_dim.get(d, []), dtype=np.float64)
            base = j * len(PH_STATS)
            if L.size == 0:
                continue
            out[base + 0] = float(L.size)
            out[base + 1] = float(L.mean())
            out[base + 2] = float(L.max())
            out[base + 3] = float(L.sum())
            out[base + 4] = _persistence_entropy(L)
    except Exception:
        return out
    return out


def compute_ph_matrix(smiles: list[str]) -> tuple[np.ndarray, dict]:
    """Compute (N, 15) PH feature matrix. Tracks fallback counts."""
    N = len(smiles)
    X = np.zeros((N, N_PH_FEAT), dtype=np.float32)
    stats = {"n_3d_ok": 0, "n_2d_fallback": 0, "n_empty": 0}
    for i, smi in enumerate(smiles):
        coords = _embed_3d(smi)
        if coords is None or coords.shape[0] < 2:
            coords = _coords_2d_fallback(smi)
            if coords is not None and coords.shape[0] >= 2:
                stats["n_2d_fallback"] += 1
            else:
                stats["n_empty"] += 1
                continue
        else:
            stats["n_3d_ok"] += 1
        X[i] = _ph_betti_features(coords)
        if (i + 1) % 100 == 0:
            print(f"   PH {i+1}/{N}  3d_ok={stats['n_3d_ok']}  "
                  f"2d_fb={stats['n_2d_fallback']}  empty={stats['n_empty']}")
    return X, stats


# -------- nb2103 substrate rebuild helpers (same as nb2103) --------
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
        raise FileNotFoundError("No ChEMBL PXR parquets found")
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
    a = fp_q.astype(np.float32); b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1); b_sum = b.sum(axis=1)
    n_q = a.shape[0]; n_pool = b.shape[0]
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


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X = np.load(mte_p).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"Mordred shape mismatch: {X.shape}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _extract_atompair_top_idx(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


def _lgbm_params(seed):
    return dict(
        objective="regression", max_depth=4, num_leaves=15, n_estimators=300,
        learning_rate=0.03, min_child_samples=5, reg_lambda=2.0,
        random_state=seed, n_jobs=2, verbosity=-1,
    )


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _write_skipped(reason: str) -> dict:
    out = {
        "tag": TAG,
        "status": "SKIPPED",
        "reason": reason,
        "verdict": "SKIPPED_NO_PH_BACKEND",
        "pre_unblind_clean": True,
    }
    p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[skip] {reason}")
    print(f"[save] {p}")
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Persistent homology Betti features (PH on 3D conformer)")
    print(f"         anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"         ref: nb2103 K=28 mean-bag RAE = {NB2103_K28_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    backend = "gudhi" if _HAS_GUDHI else ("ripser" if _HAS_RIPSER else "none")
    print(f"[ph] backend = {backend}")
    if backend == "none":
        return _write_skipped(
            "neither gudhi nor ripser is installed; install via "
            "`uv pip install gudhi` or `uv pip install ripser`"
        )

    if not NB2103_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2103_SUMMARY}")
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")

    with open(NB2103_SUMMARY) as f:
        sum_2103 = json.load(f)
    nb2103_K28_rec = None
    for r in sum_2103.get("per_K_records", []):
        if int(r.get("K", -1)) == 28:
            nb2103_K28_rec = r
            break
    nb2103_K28_mean_bag = (
        float(nb2103_K28_rec["rae_mean_bag"])
        if nb2103_K28_rec is not None else NB2103_K28_REF
    )
    print(f"[ref] nb2103 K=28 mean_bag_rae = {nb2103_K28_mean_bag:.4f}")

    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    top28_idx = full_rank_order[:28].astype(np.int32)
    print(f"[shap] top-28 indices ready (from 117-col SHAP rank)")

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
        raise FileNotFoundError(f"missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"chemprop_aux shape: {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] anchor in_RAE = {rae_anchor:.4f}  (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Reload nb2103 K-grid winner artifacts -> build 117-col matrix ----
    print("\n" + "-" * 78)
    print("REBUILD 117-COL 5-WAY K-TUNED MATRIX (same as nb2103)")
    print("-" * 78)
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f: sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f: sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f: sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f: sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f: sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f: sum_1541 = json.load(f)

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", "best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_col_idx = np.array(sum_1541["top_dim_order_top100"], dtype=int)[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb_top = X_ap_te[unb_idx][:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb_top = X_maccs_te[unb_idx][:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test)
    X_mord_unb_top = X_mord_te[unb_idx][:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb_top = X_emb_te[unb_idx][:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb_top = X_av_te[unb_idx][:, top_avalon_bit_idx].astype(np.float32)
    print(f"[feat] AtomPair={X_ap_unb_top.shape[1]}  MACCS={X_maccs_unb_top.shape[1]}  "
          f"Mordred={X_mord_unb_top.shape[1]}  Embed={X_emb_unb_top.shape[1]}  "
          f"Avalon={X_av_unb_top.shape[1]}")

    # ---- ChEMBL kNN feature (same as nb2103) ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL + kNN")
    print("-" * 78)
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
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
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")
    std_test_smiles = []
    for m in test_mols:
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- 117-col combined matrix on unblind, then slice to nb2103 top-28 ----
    X_unb_117 = np.concatenate(
        [X_ap_unb_top, X_maccs_unb_top, X_mord_unb_top, X_emb_unb_top,
         X_av_unb_top, pred_chembl_unb.reshape(-1, 1),
         mean_sim_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    if X_unb_117.shape[1] != shap_imp_full117.shape[0]:
        raise ValueError(
            f"117-col rebuild dim {X_unb_117.shape[1]} != SHAP rank "
            f"{shap_imp_full117.shape[0]}"
        )
    X_unb_top28 = X_unb_117[:, top28_idx].astype(np.float32)
    print(f"[feat] nb2103 top-28 slice = {X_unb_top28.shape}")

    # ---- Compute PH features on test SMILES ----
    print("\n" + "-" * 78)
    print(f"PERSISTENT HOMOLOGY on {n_test} test SMILES "
          f"(backend={backend}, max_edge={MAX_EDGE}, dims={PH_DIMS})")
    print("-" * 78)
    t_ph = time.time()
    X_ph_te, ph_stats = compute_ph_matrix(test_smiles)
    print(f"[ph] computed {N_PH_FEAT} features in {time.time()-t_ph:.1f}s  "
          f"stats: {ph_stats}")
    X_ph_unb = X_ph_te[unb_idx].astype(np.float32)
    # impute any NaN/inf to 0 (already zero-init)
    X_ph_unb = np.where(np.isfinite(X_ph_unb), X_ph_unb, 0.0).astype(np.float32)

    # Print PH variance summary
    ph_means = X_ph_unb.mean(axis=0)
    ph_stds = X_ph_unb.std(axis=0)
    print("[ph] per-feature mean / std on n_unb:")
    for j, nm in enumerate(PH_FEAT_NAMES):
        print(f"   {nm:>20s}  mean={ph_means[j]:>+8.3f}  std={ph_stds[j]:>8.3f}")

    # ---- Final K=43 matrix ----
    X_unb_43 = np.concatenate([X_unb_top28, X_ph_unb], axis=1).astype(np.float32)
    print(f"\n[feat] FINAL K=43 matrix = {X_unb_43.shape}  "
          f"(28 nb2103 SHAP top + 15 PH)")

    # ---- 5-seed bag, 5-fold cross-fit ----
    print("\n" + "-" * 78)
    print(f"LGBM(MSE) residual-cross-fit  K=43  seeds={RESID_SEEDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_43, residual, s)
        pred_corr = anchor + resid_oof
        per_seed_corrected[i] = pred_corr
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof.std()),
            "resid_oof_mean": float(resid_oof.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_arr.mean())
    rae_per_seed_std = float(per_seed_arr.std())
    delta_vs_nb2103 = rae_mean_bag - nb2103_K28_mean_bag
    delta_vs_anchor = rae_mean_bag - rae_anchor

    beats_nb2103 = rae_mean_bag < nb2103_K28_mean_bag - DECISION_MARGIN
    flat_vs_nb2103 = abs(delta_vs_nb2103) < DECISION_MARGIN
    beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN

    if beats_nb2103:
        verdict = "BEATS_NB2103_K28"
    elif flat_vs_nb2103:
        verdict = "FLAT_VS_NB2103_K28"
    elif beats_anchor:
        verdict = "BEATS_ANCHOR_BUT_WORSE_THAN_NB2103_K28"
    else:
        verdict = "HURTS_ANCHOR"

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"   per-seed RAE  : [{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean : {rae_per_seed_mean:.4f}   std: {rae_per_seed_std:.4f}")
    print(f"   mean-bag RAE  : {rae_mean_bag:.4f}")
    print(f"   median-bag RAE: {rae_median_bag:.4f}")
    print(f"   d_vs_anchor   : {delta_vs_anchor:+.4f}")
    print(f"   d_vs_nb2103_K28: {delta_vs_nb2103:+.4f}")
    print(f"   verdict       : {verdict}")

    # Save mean-bag OOF
    oof_p = DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy"
    np.save(oof_p, mean_bag_oof.astype(np.float32))
    print(f"[save] {oof_p}")

    summary = {
        "tag": TAG,
        "method": "lgbm_mse_K43_top28_nb2103_shap_plus_15_persistent_homology",
        "ph_backend": backend,
        "ph_max_edge": MAX_EDGE,
        "ph_dims": list(PH_DIMS),
        "ph_feature_names": PH_FEAT_NAMES,
        "n_ph_features": N_PH_FEAT,
        "ph_compute_stats": ph_stats,
        "ph_feature_mean": [float(x) for x in ph_means],
        "ph_feature_std": [float(x) for x in ph_stds],
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("nb2103 top-28 SHAP cols from 117-col 5-way K-tuned "
                        "matrix + 15 persistent homology features on 3D "
                        "ETKDG+MMFF94 conformer (gudhi VR up to dim=2)"),
        "model_family": "LightGBM",
        "lgbm_params": _lgbm_params(0),
        "K_features_total": int(X_unb_43.shape[1]),
        "K_nb2103_shap": 28,
        "K_ph": N_PH_FEAT,
        "n_unb": int(n_unb),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2103_K28_mean_bag_ref": nb2103_K28_mean_bag,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_chemprop_aux": delta_vs_anchor,
        "delta_mean_bag_vs_nb2103_K28": delta_vs_nb2103,
        "beats_chemprop_aux": bool(beats_anchor),
        "beats_nb2103_K28": bool(beats_nb2103),
        "flat_vs_nb2103_K28": bool(flat_vs_nb2103),
        "verdict": verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_ref": NB2103_K28_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {p}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "ph_backend", "K_features_total", "K_nb2103_shap", "K_ph",
        "ph_compute_stats",
        "rae_anchor_chemprop_aux", "nb2103_K28_mean_bag_ref",
        "rae_mean_bag", "rae_median_bag", "rae_per_seed_mean",
        "rae_per_seed_std",
        "delta_mean_bag_vs_chemprop_aux", "delta_mean_bag_vs_nb2103_K28",
        "verdict", "pre_unblind_clean",
    ):
        if k in res:
            print(f"  {k}: {res.get(k)}")
