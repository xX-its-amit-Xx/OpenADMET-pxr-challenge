"""nb2013 -- 2-stage cascade on nb1191:
    Stage 1: nb1191 OOF on 253 (RAE 0.4703).
    Stage 2: LGBM K=28 (SHAP top-28 indices from nb2103) on residual
             r = y - nb1191_pred, with 3 extra meta features:
                - scaffold_id (hash mod 8 one-hot, 8 cols)
                - max_train_sim (Tanimoto to 4139 train cpds)
                - scaf_train_freq (count of train cpds with same scaffold)

Final blend: pred_final = nb1191_oof + LGBM_residual_oof.
5-fold scaffold cross-fit on 253 unblind.  Decision margin = 0.003.

Outputs:
    data/processed/nb2013_summary.json
    data/processed/te_nb2013.npy                  (513, float32)
    submissions/nb2013_two_stage.csv              (only if beats nb1191)
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
from scipy.optimize import minimize
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2013"

# ----- nb1191 reconstruction inputs --------------------------------
NB1191_SUMMARY = DATA_PROCESSED / "nb1191_summary.json"
NB1191_TE_PATH = DATA_PROCESSED / "te_nb1191.npy"

# nb1191 anchor OOFs (reconstruct nb1150 + load the other 3)
NB1191_ANCHOR_OOFS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy"),
]
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.0, 0.2942, 0.0, 0.7058]
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]

# ----- 117-col stack inputs ----------------------------------------
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
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6
TOP_K_SHAP = 28
N_SCAFFOLD_BUCKETS = 8

# Residual-LGBM bag (smaller than nb2103 to keep wall < 5 min)
RESID_SEEDS = [0, 1, 7]
RESID_FOLDS = 5

# Reference
NB1191_OOF_REF = 0.4703
DECISION_MARGIN = 0.003


# ===================================================================
# Reconstruct nb1191 OOF
# ===================================================================
def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS_FULL_POOL, dtype=np.float64)
    return P @ w


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s, best_r


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(
            blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID,
        )
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
    return oof_blend


def get_nb1191_oof(P_unb, y_unb, unb_scaffolds):
    """Average per-seed scaffold-CV OOFs across 5 seeds (matches nb1191)."""
    all_oofs = []
    for kf_seed in KF_SEEDS:
        oof = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        all_oofs.append(oof)
    return np.mean(np.column_stack(all_oofs), axis=1)


# ===================================================================
# 117-col stack rebuild
# ===================================================================
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
             n_meas=("pec50", "count"))
    )
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
    w = np.clip(top_sim.copy(), 0.0, 1.0)
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
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"mordred shape {X_te_m.shape} vs {n_test_expected}")
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = np.where(np.isfinite(X.astype(np.float32)), X, 0.0).astype(np.float32)
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


def _extract_K_record(sum_dict: dict, records_key: str, K: int) -> dict:
    for r in sum_dict[records_key]:
        if int(r["K"]) == K:
            return r
    raise KeyError(f"K={K} not found in {records_key}")


def build_117col_stack(test_smiles, n_test, unb_idx):
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
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

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
    std_test_smiles = [
        Chem.MolToSmiles(m) if m is not None else "" for m in test_mols
    ]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_117 = np.concatenate(
        [
            X_ap_te_top, X_maccs_te_top, X_mord_te_top,
            X_emb_te_top, X_av_te_top,
            pred_chembl_te.reshape(-1, 1).astype(np.float32),
            mean_sim_te.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    X_unb_117 = X_te_117[unb_idx]
    return X_te_117, X_unb_117, fp_test


# ===================================================================
# Meta features
# ===================================================================
def build_meta_features(test_smiles, n_test, unb_idx, fp_test):
    """Returns X_meta_te (n_test, K_meta) and X_meta_unb (n_unb, K_meta)."""
    # Load train smiles + scaffolds
    tr = load_train()
    tr_smi = tr["smiles"].astype(str).tolist()
    print(f"   train n = {len(tr_smi)}")
    fp_tr = morgan_fp_batch(tr_smi)
    keep_tr = fp_tr.sum(axis=1) > 0
    fp_tr = fp_tr[keep_tr]
    tr_smi_kept = [s for s, k in zip(tr_smi, keep_tr) if k]
    print(f"   train fps kept = {len(tr_smi_kept)}")

    # max train sim (top-1 Tanimoto)
    top_idx_1, top_sim_1 = _tanimoto_topk(fp_test, fp_tr, k=1)
    max_train_sim_te = top_sim_1[:, 0].astype(np.float32)
    print(f"   max_train_sim_te: mean={max_train_sim_te.mean():.3f} "
          f"min={max_train_sim_te.min():.3f}")

    # scaffolds
    tr_scaf = [bemis_murcko(s) or "" for s in tr_smi_kept]
    te_scaf = [bemis_murcko(s) or "" for s in test_smiles]
    scaf_count = {}
    for s in tr_scaf:
        if s:
            scaf_count[s] = scaf_count.get(s, 0) + 1
    scaf_train_freq_te = np.array(
        [float(scaf_count.get(s, 0)) for s in te_scaf], dtype=np.float32,
    )
    print(f"   scaf_train_freq_te: mean={scaf_train_freq_te.mean():.2f} "
          f"max={int(scaf_train_freq_te.max())}")

    # scaffold-id hash buckets (one-hot, K=N_SCAFFOLD_BUCKETS)
    scaf_bucket_te = np.zeros(
        (n_test, N_SCAFFOLD_BUCKETS), dtype=np.float32,
    )
    for i, s in enumerate(te_scaf):
        if s:
            b = (hash(s) % N_SCAFFOLD_BUCKETS + N_SCAFFOLD_BUCKETS) \
                % N_SCAFFOLD_BUCKETS
            scaf_bucket_te[i, b] = 1.0
    X_meta_te = np.concatenate(
        [
            max_train_sim_te.reshape(-1, 1),
            scaf_train_freq_te.reshape(-1, 1),
            scaf_bucket_te,
        ],
        axis=1,
    ).astype(np.float32)
    X_meta_unb = X_meta_te[unb_idx]
    print(f"   X_meta_te shape = {X_meta_te.shape}")
    return X_meta_te, X_meta_unb


# ===================================================================
# Residual cross-fit + deploy
# ===================================================================
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


def residual_scaffold_cv(X_unb, residual, unb_scaffolds, seed):
    """Honest 5-fold scaffold cross-fit OOF for one seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X_unb[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X_unb[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 2-stage cascade on nb1191 (residual LGBM K=28 + meta)")
    print("=" * 78)

    # ---- truth + scaffolds
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist()
    if "name" in te.columns:
        mol_names = te["name"].astype(str).tolist()
    elif "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    else:
        raise KeyError("no name column")
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]

    # ---- Stage 1: nb1191 OOF on 253 ----
    print("\n[stage1] reconstruct nb1191 OOF on 253")
    oof_cols = []
    for disp, oof_rel in NB1191_ANCHOR_OOFS:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            oof = reconstruct_nb1150_oof(n_unb)
        else:
            p = DATA_PROCESSED / oof_rel
            oof = np.load(p).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} shape {oof.shape}"
        oof_cols.append(oof)
        print(f"   anchor {disp:14s} oof_RAE = {rae(y_unb, oof):.4f}")
    P_unb_anchors = np.column_stack(oof_cols)

    nb1191_oof = get_nb1191_oof(P_unb_anchors, y_unb, unb_scaffolds)
    nb1191_oof_rae = float(rae(y_unb, nb1191_oof))
    print(f"\n[stage1] nb1191 OOF RAE on 253 = {nb1191_oof_rae:.4f}  "
          f"(ref {NB1191_OOF_REF:.4f})")

    # nb1191 deploy te (513) for final blend
    te_nb1191 = np.load(NB1191_TE_PATH).astype(np.float64)
    assert te_nb1191.shape == (n_test,), f"te_nb1191 {te_nb1191.shape}"

    residual = y_unb - nb1191_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Stage 2: LGBM K=28 SHAP + meta features ----
    print("\n[stage2] build 117-col stack + slice top-28 SHAP")
    X_te_117, X_unb_117, fp_test = build_117col_stack(
        test_smiles, n_test, unb_idx,
    )
    print(f"   X_te_117={X_te_117.shape}  X_unb_117={X_unb_117.shape}")

    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    rec28 = _extract_K_record(nb2103_sum, "per_K_records", K=TOP_K_SHAP)
    top28_idx = np.array(rec28["top_K_idx_in_117"], dtype=np.int32)
    X_te_28 = X_te_117[:, top28_idx].astype(np.float32)
    X_unb_28 = X_unb_117[:, top28_idx].astype(np.float32)
    print(f"   SHAP top-{TOP_K_SHAP} sliced: X_unb_28={X_unb_28.shape}")

    print("\n[stage2] build meta features (max_train_sim, scaf_train_freq, "
          "scaf_id_buckets)")
    X_meta_te, X_meta_unb = build_meta_features(
        test_smiles, n_test, unb_idx, fp_test,
    )

    X_te_full = np.concatenate([X_te_28, X_meta_te], axis=1).astype(np.float32)
    X_unb_full = np.concatenate(
        [X_unb_28, X_meta_unb], axis=1,
    ).astype(np.float32)
    K_total = X_unb_full.shape[1]
    print(f"\n[stage2] final feature dim = {K_total}  "
          f"(28 SHAP + {K_total - 28} meta)")

    # ---- 5-fold scaffold cross-fit for residual, 3 seeds ----
    print(f"\n[stage2] residual cross-fit (5-fold scaffold, seeds={RESID_SEEDS})")
    per_seed_oof_resid = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        t1 = time.time()
        resid_oof_s = residual_scaffold_cv(
            X_unb_full, residual, unb_scaffolds, seed=s,
        )
        per_seed_oof_resid[i] = resid_oof_s
        pred_corr_s = nb1191_oof + resid_oof_s
        r_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(r_s)
        print(f"   seed={s:3d}:  final OOF RAE = {r_s:.4f}  "
              f"(delta vs nb1191 = {r_s - nb1191_oof_rae:+.4f})  "
              f"wall={time.time() - t1:.1f}s")

    mean_resid_oof = per_seed_oof_resid.mean(axis=0)
    pred_corr_mean = nb1191_oof + mean_resid_oof
    final_oof_rae_mean = float(rae(y_unb, pred_corr_mean))
    delta_mean = final_oof_rae_mean - nb1191_oof_rae
    beats = delta_mean < -DECISION_MARGIN
    flat = abs(delta_mean) <= DECISION_MARGIN
    if beats:
        verdict = "BEATS_NB1191"
    elif flat:
        verdict = "FLAT_VS_NB1191"
    else:
        verdict = "HURTS_NB1191"
    print("\n" + "-" * 78)
    print(f"FINAL mean-bag OOF RAE = {final_oof_rae_mean:.4f}  "
          f"(delta {delta_mean:+.4f}  {verdict})")
    print("-" * 78)

    # ---- Deploy: fit residual on all 253, predict 513 (bag mean over seeds) ----
    deploy_te_resid_sum = np.zeros(n_test, dtype=np.float64)
    for s in RESID_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb_full, residual)
        deploy_te_resid_sum += mdl.predict(X_te_full)
    deploy_te_resid_mean = deploy_te_resid_sum / len(RESID_SEEDS)
    deploy_te = (te_nb1191 + deploy_te_resid_mean).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[deploy] te(513)  mean={deploy_te.mean():.3f}  "
          f"std={deploy_te.std():.3f}  "
          f"min={deploy_te.min():.3f}  max={deploy_te.max():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE = {te_unb_rae:.4f}  "
          f"(in-sample; not a deploy guarantee)")

    # ---- Save te artifact regardless ----
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, deploy_te)
    print(f"\n[save] {te_path}")

    sub_path = SUBMISSIONS / f"{TAG}_two_stage.csv"
    if beats:
        pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": deploy_te,
        }).to_csv(sub_path, index=False)
        print(f"[save] {sub_path}  (BEATS nb1191 by {-delta_mean:.4f})")
    else:
        print(f"[skip] verdict={verdict} -- no CSV (would be {sub_path})")

    summary = {
        "tag": TAG,
        "method": "two_stage_nb1191_plus_residual_LGBM_K28_with_meta",
        "stage1_anchor": "nb1191_reconstructed",
        "nb1191_oof_rae_local": nb1191_oof_rae,
        "nb1191_oof_rae_ref": NB1191_OOF_REF,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "stage2_top_K_shap": TOP_K_SHAP,
        "stage2_meta_features": [
            "max_train_sim",
            "scaf_train_freq",
            f"scaf_id_hash_buckets_{N_SCAFFOLD_BUCKETS}",
        ],
        "feat_dim_total": int(K_total),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "per_seed_final_oof_rae": per_seed_rae,
        "final_oof_rae_mean_bag": final_oof_rae_mean,
        "delta_vs_nb1191": delta_mean,
        "decision_margin": DECISION_MARGIN,
        "beats_nb1191": bool(beats),
        "verdict": verdict,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_unb_rae_in_sample": te_unb_rae,
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_path) if beats else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")
    print(f"\n[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "nb1191_oof_rae_local",
        "residual_std",
        "feat_dim_total",
        "per_seed_final_oof_rae",
        "final_oof_rae_mean_bag",
        "delta_vs_nb1191",
        "beats_nb1191",
        "verdict",
        "te_unb_rae_in_sample",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
