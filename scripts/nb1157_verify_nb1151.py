"""nb1157 -- STRICT verification of nb1151 K=35 scaffold-CV refit claim.

CLAIM TO VERIFY:
    nb1151_summary.json reports K=35 mean_bag RAE = 0.5024 on the 253 unblind
    using scaffold-CV with seeds {0, 1, 7, 42, 137} (5 seeds).  Beats the
    nb2103 K=28 scaffold-CV reference (0.5057) and clears the 0.5027 deploy
    gate. Was this a real optimum or a lucky-seed artefact?

PROTOCOL:
    1. Rebuild the SAME 117-col 5-way K-tuned matrix on the 253 unblind
       (same recipe as nb1151_scaffold_k_sweep.py; reuse cached test-side
       feature .npy + the deterministic ChEMBL kNN feature).
    2. Slice top-K columns by the nb2063 SHAP importance ranking.
    3. For each K in fine grid {30, 32, 35, 38, 40} (focused around K=35)
       AND original {15, 20, 28, 35, 50} sanity points:
         - run LGBM(MSE) residual stack with FRESH seeds {1001..1010}
           (10 seeds, none overlap with original {0,1,7,42,137})
         - scaffold_kfold_indices(Murcko, n=5, shuffle=True, seed=kf_seed)
         - per-seed RAE, then mean/median/std/min/max + mean-bag RAE
    4. Compare to nb1151 claimed values.  Compute gap_mean_bag and assess
       whether K=35 reproducibly beats K=28/K=30/K=32/K=38/K=40.
    5. Family composition table: which feature family enters at K=29..35
       (i.e. the 7 features added on top of K=28)?
    6. Verdict:
         PROMOTE        if K=35 mean-bag <= 0.5027 reproducibly
         FLAT           if 0.5027 < K=35 mean-bag <= 0.5057 (matches nb2103)
         REJECT         if K=35 mean-bag > 0.5057 (lucky-seed artefact)
         REJECT_NOT_OPT if some other K beats K=35 by >0.003 reproducibly

Outputs:
    scripts/nb1157_verify_nb1151.py
    data/processed/nb1157_summary.json
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
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1157"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Verification protocol
RESID_FOLDS = 5
FRESH_SEEDS = list(range(1001, 1011))   # 10 fresh seeds, disjoint from nb1151
K_FINE_GRID = [30, 32, 35, 38, 40]
K_SANITY_GRID = [15, 20, 28, 35, 50]
K_GRID = sorted(set(K_FINE_GRID + K_SANITY_GRID))

# Reuse same artefacts as nb1151
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
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"
NB1151_SUMMARY = DATA_PROCESSED / "nb1151_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

NB1151_K35_CLAIM = 0.5024              # claimed mean-bag at K=35
NB1151_K35_FULL = 0.5023992153179189   # exact value from summary
NB2103_K28_SCAFFOLD_REF = 0.5057
DEPLOY_GATE = 0.5027
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


def _murcko_scaffold(mol):
    try:
        if mol is None:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc is not None else None
    except Exception:
        return None


def _load_chembl_pool() -> pd.DataFrame:
    """Identical to nb1151 _load_chembl_pool."""
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
        .rename(columns={"src_first": "src"})
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
    """LGBM(MSE) -- identical to nb1151/nb2063/nb2103."""
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


def _residual_scaffold_cross_fit_one_seed(X, residual, scaffolds, seed):
    n = len(residual)
    splits = scaffold_kfold_indices(scaffolds, n_splits=RESID_FOLDS,
                                    shuffle=True, seed=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred_test(n_test_expected):
    X = np.load(MORDRED_DIR / "X_mordred_test.npy").astype(np.float32)
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy_test(path, n_test_expected):
    X = np.load(path).astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair not found")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- STRICT verification of nb1151 K=35 (claimed mean_bag={NB1151_K35_CLAIM})")
    print(f"          fresh seeds = {FRESH_SEEDS}  (disjoint from nb1151 {{0,1,7,42,137}})")
    print(f"          K-grid (fine+sanity) = {K_GRID}")
    print(f"          deploy gate = {DEPLOY_GATE}")
    print("=" * 78)

    shap_imp = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank = np.argsort(-shap_imp).astype(np.int32)
    print(f"[ref] nb2063 SHAP importance: {shap_imp.shape}")

    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] n_test={n_test}  n_unb={n_unb}  anchor RAE={rae_anchor:.4f}")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_mols = [standardize(s) for s in unb_smiles]
    scaffolds_unb = [_murcko_scaffold(m) for m in unb_mols]
    print(f"[scaf] n_unb={n_unb}  n_unique_scaf={len({s for s in scaffolds_unb if s})}")

    # --- Build 117-col feature matrix (same recipe as nb1151) ---
    with open(NB1352_SUMMARY) as f: sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f: sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f: sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f: sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f: sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f: sum_1541 = json.load(f)

    top_maccs = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
    top_mord = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap = _extract_atompair_top_idx(sum_1484)
    K_AP = int(sum_1524["best_K"])
    top_ap = full_ap[:K_AP]
    K_emb = int(sum_1541["best_K"])
    top_emb = np.array(sum_1541["top_dim_order_top100"], dtype=int)[:K_emb]
    top_av = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb = X_ap_te[unb_idx][:, top_ap].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb = X_maccs_te[unb_idx][:, top_maccs].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test)
    X_mord_unb = X_mord_te[unb_idx][:, top_mord].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb = X_emb_te[unb_idx][:, top_emb].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb = X_av_te[unb_idx][:, top_av].astype(np.float32)

    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_iks = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_iks.add(ik)
    pool = pool[~pool["inchikey"].isin(test_iks)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_test_smiles = []
    for m in test_mols:
        std_test_smiles.append("" if m is None else Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl, mean_sim = _knn_predict(top_idx_knn, top_sim_knn, pool_labels, pool_median)
    pred_chembl_unb = pred_chembl[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)
    print(f"[chembl] pool={len(pool)}  median={pool_median:.3f}")

    X_unb = np.concatenate([
        X_ap_unb, X_maccs_unb, X_mord_unb, X_emb_unb, X_av_unb,
        pred_chembl_unb.reshape(-1, 1), mean_sim_unb.reshape(-1, 1),
    ], axis=1).astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"[feat] X_unb (117-col) = {X_unb.shape}")
    if feat_dim != shap_imp.shape[0]:
        raise ValueError(f"feat_dim {feat_dim} != SHAP {shap_imp.shape[0]}")

    # --- Feature family names ---
    feat_family: list[str] = []
    feat_names: list[str] = []
    for b in top_ap:
        feat_family.append("AtomPair"); feat_names.append(f"AtomPair_bit_{int(b)}")
    for b in top_maccs:
        feat_family.append("MACCS"); feat_names.append(f"MACCS_bit_{int(b)}")
    for c in top_mord:
        feat_family.append("Mordred"); feat_names.append(f"Mordred_col_{int(c)}")
    for d in top_emb:
        feat_family.append("ChempropEmbed"); feat_names.append(f"ChempropEmbed_dim_{int(d)}")
    for b in top_av:
        feat_family.append("Avalon"); feat_names.append(f"Avalon_bit_{int(b)}")
    feat_family.append("ChEMBL_kNN"); feat_names.append("pred_chembl_pec50")
    feat_family.append("ChEMBL_kNN"); feat_names.append("mean_sim")

    # --- Feature delta K=28 -> K=35: what's added? ---
    set_K28 = set(int(i) for i in full_rank[:28])
    set_K35 = set(int(i) for i in full_rank[:35])
    added = [i for i in full_rank[28:35].tolist()]
    delta_features = []
    for rk, fi in enumerate(added, start=29):
        delta_features.append({
            "rank_added": rk,
            "feat_idx": int(fi),
            "feat_name": feat_names[fi],
            "family": feat_family[fi],
            "shap_importance": float(shap_imp[fi]),
        })
    print("\n[delta K=28->K=35] features added at ranks 29..35:")
    for d in delta_features:
        print(f"   r{d['rank_added']:>2d}  idx={d['feat_idx']:>3d}  "
              f"{d['family']:<14s} {d['feat_name']:<32s} shap={d['shap_importance']:.4f}")

    # --- K-sweep with FRESH seeds ---
    print("\n" + "-" * 78)
    print(f"FRESH-SEED K-SWEEP  seeds={FRESH_SEEDS}  scaffold-CV folds={RESID_FOLDS}")
    print("-" * 78)
    per_K: list[dict] = []
    for K in K_GRID:
        print(f"\n--- K={K} ---")
        topK_idx = full_rank[:K].astype(np.int32)
        topK_family = [feat_family[i] for i in topK_idx]
        fam_counts: dict[str, int] = {}
        for fam in topK_family:
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
        print(f"   family breakdown: {fam_counts}")
        X_topK = X_unb[:, topK_idx].astype(np.float32)

        per_seed_corrected = np.zeros((len(FRESH_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae: list[float] = []
        for i, s in enumerate(FRESH_SEEDS):
            resid_oof = _residual_scaffold_cross_fit_one_seed(
                X_topK, residual, scaffolds_unb, s
            )
            pred_corr = anchor + resid_oof
            per_seed_corrected[i] = pred_corr
            rae_s = float(rae(y_unb, pred_corr))
            per_seed_rae.append(rae_s)
            print(f"   K={K} seed={s}  rae={rae_s:.4f}  d_anchor={rae_s - rae_anchor:+.4f}")

        mean_bag = per_seed_corrected.mean(axis=0)
        median_bag = np.median(per_seed_corrected, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag))
        rae_median_bag = float(rae(y_unb, median_bag))
        arr = np.array(per_seed_rae)
        rec = {
            "K": int(K),
            "family_counts": fam_counts,
            "per_seed_rae": per_seed_rae,
            "rae_per_seed_mean": float(arr.mean()),
            "rae_per_seed_median": float(np.median(arr)),
            "rae_per_seed_std": float(arr.std()),
            "rae_per_seed_min": float(arr.min()),
            "rae_per_seed_max": float(arr.max()),
            "rae_per_seed_range": float(arr.max() - arr.min()),
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
            "delta_mean_bag_vs_nb1151_K35": rae_mean_bag - NB1151_K35_FULL,
            "delta_mean_bag_vs_nb2103_K28_ref": rae_mean_bag - NB2103_K28_SCAFFOLD_REF,
            "delta_mean_bag_vs_deploy_gate": rae_mean_bag - DEPLOY_GATE,
            "beats_deploy_gate": bool(rae_mean_bag < DEPLOY_GATE),
            "beats_nb1151_K35": bool(rae_mean_bag < NB1151_K35_FULL),
            "flat_vs_nb1151_K35": bool(abs(rae_mean_bag - NB1151_K35_FULL) < DECISION_MARGIN),
        }
        per_K.append(rec)
        print(f"   K={K} MEAN-BAG = {rae_mean_bag:.4f}  "
              f"per_seed_mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")
        print(f"   K={K} vs nb1151_K35 ({NB1151_K35_FULL:.4f}): "
              f"d={rae_mean_bag - NB1151_K35_FULL:+.4f}  "
              f"vs deploy gate ({DEPLOY_GATE:.4f}): d={rae_mean_bag - DEPLOY_GATE:+.4f}")

    # --- Compare nb1151's K=35 to fresh-seed K=35 ---
    K35_rec = next(r for r in per_K if r["K"] == 35)
    K35_mean_bag_fresh = K35_rec["rae_mean_bag"]
    K35_per_seed_mean_fresh = K35_rec["rae_per_seed_mean"]
    K35_per_seed_std_fresh = K35_rec["rae_per_seed_std"]
    gap_K35 = K35_mean_bag_fresh - NB1151_K35_FULL

    # nb1151 per-seed mean at K=35 (from summary)
    nb1151_K35_per_seed_mean = 0.5287103918579461
    nb1151_K35_per_seed_std = 0.01041336390720942
    nb1151_K35_per_seed_min = 0.5152862876998328
    nb1151_K35_per_seed_max = 0.5474257365748165

    # Is fresh per-seed mean within 1 SD of original?
    delta_per_seed_mean = K35_per_seed_mean_fresh - nb1151_K35_per_seed_mean
    n_orig = 5; n_fresh = 10
    se_pooled = float(np.sqrt(
        nb1151_K35_per_seed_std ** 2 / n_orig
        + K35_per_seed_std_fresh ** 2 / n_fresh
    ))
    z_per_seed = delta_per_seed_mean / se_pooled if se_pooled > 0 else 0.0

    # --- Identify optimum K under fresh seeds ---
    K_vals = [r["K"] for r in per_K]
    rae_vals = [r["rae_mean_bag"] for r in per_K]
    best_i = int(np.argmin(rae_vals))
    best_K_fresh = int(per_K[best_i]["K"])
    best_K_rae_fresh = float(per_K[best_i]["rae_mean_bag"])

    # --- Is K=35 robustly the optimum or beaten by neighbors? ---
    K35_idx = K_vals.index(35)
    neighbor_K = [K for K in [30, 32, 38, 40] if K in K_vals]
    K35_vs_neighbors = []
    K35_beats_all_neighbors = True
    for nK in neighbor_K:
        nrec = next(r for r in per_K if r["K"] == nK)
        d = K35_mean_bag_fresh - nrec["rae_mean_bag"]
        K35_vs_neighbors.append({
            "K_neighbor": nK,
            "neighbor_rae_mean_bag": nrec["rae_mean_bag"],
            "delta_K35_minus_neighbor": d,
            "K35_beats_neighbor": bool(d < -DECISION_MARGIN),
            "neighbor_beats_K35": bool(d > DECISION_MARGIN),
        })
        if d > -DECISION_MARGIN:
            K35_beats_all_neighbors = False

    # --- VERDICT ---
    # PROMOTE: K=35 fresh mean_bag <= 0.5027 (deploy gate)
    # FLAT:    0.5027 < K=35 mean_bag <= 0.5057 (matches nb2103 ref but not gate)
    # REJECT_LUCKY_SEED:  K=35 mean_bag > 0.5057 (worse than nb2103 ref)
    # REJECT_NOT_OPT: some K in {30,32,38,40} beats K=35 by > 0.003
    K35_reproduces_claim = bool(abs(gap_K35) < DECISION_MARGIN)
    if K35_mean_bag_fresh <= DEPLOY_GATE:
        verdict = "PROMOTE"
    elif K35_mean_bag_fresh <= NB2103_K28_SCAFFOLD_REF:
        verdict = "FLAT_VS_NB2103_REF"
    else:
        verdict = "REJECT_LUCKY_SEED"
    if (verdict == "PROMOTE" or verdict == "FLAT_VS_NB2103_REF") \
            and not K35_beats_all_neighbors:
        beat_by = [n for n in K35_vs_neighbors if n["neighbor_beats_K35"]]
        if beat_by:
            verdict = f"REJECT_NOT_OPT (beaten by K={beat_by[0]['K_neighbor']})"

    print("\n" + "=" * 78)
    print("VERIFICATION SUMMARY")
    print("=" * 78)
    print(f"  {'K':>4s}  {'mean_bag':>10s}  {'per_seed_mean':>14s}  "
          f"{'per_seed_std':>12s}  {'range':>14s}  beats_gate")
    for r in per_K:
        rng = f"[{r['rae_per_seed_min']:.4f},{r['rae_per_seed_max']:.4f}]"
        print(f"  {r['K']:>4d}  {r['rae_mean_bag']:>10.4f}  "
              f"{r['rae_per_seed_mean']:>14.4f}  {r['rae_per_seed_std']:>12.4f}  "
              f"{rng:>14s}  {r['beats_deploy_gate']}")
    print(f"\n  nb1151 claim K=35 mean_bag = {NB1151_K35_FULL:.4f}")
    print(f"  fresh   K=35 mean_bag      = {K35_mean_bag_fresh:.4f}  "
          f"(gap = {gap_K35:+.4f})")
    print(f"  nb1151 K=35 per_seed_mean  = {nb1151_K35_per_seed_mean:.4f} "
          f"(std={nb1151_K35_per_seed_std:.4f}, n=5)")
    print(f"  fresh   K=35 per_seed_mean = {K35_per_seed_mean_fresh:.4f} "
          f"(std={K35_per_seed_std_fresh:.4f}, n=10)  "
          f"d={delta_per_seed_mean:+.4f}  z={z_per_seed:+.2f}")
    print(f"  fresh best K = {best_K_fresh}  rae={best_K_rae_fresh:.4f}")
    print(f"  K=35 vs neighbors:")
    for n in K35_vs_neighbors:
        print(f"    K={n['K_neighbor']:>2d}: rae={n['neighbor_rae_mean_bag']:.4f}  "
              f"d(K35-K{n['K_neighbor']})={n['delta_K35_minus_neighbor']:+.4f}")
    print(f"\n  VERDICT: {verdict}")

    summary = {
        "tag": TAG,
        "verifies": "nb1151",
        "claim": {
            "K35_mean_bag": NB1151_K35_FULL,
            "K35_per_seed_mean": nb1151_K35_per_seed_mean,
            "K35_per_seed_std": nb1151_K35_per_seed_std,
            "K35_per_seed_min": nb1151_K35_per_seed_min,
            "K35_per_seed_max": nb1151_K35_per_seed_max,
            "nb1151_seeds": [0, 1, 7, 42, 137],
        },
        "protocol": {
            "fresh_seeds": FRESH_SEEDS,
            "K_grid": K_GRID,
            "K_fine_grid": K_FINE_GRID,
            "K_sanity_grid": K_SANITY_GRID,
            "resid_folds": RESID_FOLDS,
            "cv": "scaffold_kfold_indices(Murcko, n=5, shuffle=True, seed=kf_seed)",
            "lgbm": _lgbm_params(0),
            "anchor": "chemprop_aux te[unb_idx]",
        },
        "rae_anchor": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "feat_dim": int(feat_dim),
        "n_unb": int(n_unb),
        "n_chembl_pool": int(len(pool)),
        "deploy_gate": DEPLOY_GATE,
        "nb2103_K28_scaffold_ref": NB2103_K28_SCAFFOLD_REF,
        "decision_margin": DECISION_MARGIN,
        "delta_features_K28_to_K35": delta_features,
        "per_K_records": per_K,
        "K35_fresh_mean_bag": K35_mean_bag_fresh,
        "K35_fresh_per_seed_mean": K35_per_seed_mean_fresh,
        "K35_fresh_per_seed_std": K35_per_seed_std_fresh,
        "K35_fresh_per_seed_min": float(per_K[K35_idx]["rae_per_seed_min"]),
        "K35_fresh_per_seed_max": float(per_K[K35_idx]["rae_per_seed_max"]),
        "gap_K35_fresh_minus_claim": gap_K35,
        "K35_reproduces_claim_within_margin": K35_reproduces_claim,
        "per_seed_mean_delta_fresh_minus_orig": delta_per_seed_mean,
        "per_seed_mean_z_score": z_per_seed,
        "best_K_fresh": best_K_fresh,
        "best_K_rae_fresh": best_K_rae_fresh,
        "K35_vs_neighbors": K35_vs_neighbors,
        "K35_beats_all_neighbors": K35_beats_all_neighbors,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
