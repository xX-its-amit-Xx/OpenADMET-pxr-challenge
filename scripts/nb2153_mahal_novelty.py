"""nb2153 -- Mahalanobis novelty scaffold weighting on nb2095.

HYPOTHESIS:
    nb2095 (PRE-pyramid SLSQP blend) achieves scaffold-CV RAE 0.4720 on the
    253 unblind. The F2 (greasy-novel-inactive) tail dominates residual error;
    on novel rows the predictor is over-confident toward training-supported
    activity. Per-row novelty weighting toward the train-mean prior is a
    1-parameter shrinkage that, like nb562 rank-stretch, may extract a small
    margin without overfit-at-n=253.

PROTOCOL:
    1. Rebuild the same 117-col 5-way K-tuned feature matrix as nb2103/nb2095
       for BOTH train (4139) and test (513).
    2. Compute per-row Mahalanobis distance from the TRAIN centroid in the
       117-col space (regularized covariance, diagonal floor).
    3. Per-row weight  w = sigmoid(-alpha * (dist - median_train_dist)).
       w in [0,1]; high novelty -> small w -> blend leans on train_mean.
    4. Per-row blend = w * nb2095_pred + (1 - w) * train_mean_pec50.
    5. Sweep alpha in {0.5, 1.0, 2.0, 5.0}.
    6. Scaffold 5-fold CV RAE on the 253 unblind, K_seeds = {1001..1005}.
    7. Compare each alpha vs nb2095 0.4720 at decision_margin = 0.003.

Outputs:
    scripts/nb2153_mahal_novelty.py
    data/processed/nb2153_summary.json
    data/processed/nb2153_mahal_dist_te.npy
    data/processed/nb2153_w_per_alpha.npy        (513, 4)
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
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, morgan_fp_batch, standardize
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2153"
NB2095_REF_RAE = 0.4720
DECISION_MARGIN = 0.003

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
ALPHA_GRID = [0.5, 1.0, 2.0, 5.0]
COV_REG = 1e-3            # diagonal floor for cov regularization
SIM_FLOOR = 1e-6

# ---- feature artifacts (mirror nb2103 / nb2095) ----
ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TR_PATH    = DATA_PROCESSED / "tr_maccs.npy"
MACCS_TE_PATH    = DATA_PROCESSED / "te_maccs.npy"
AVALON_TR_PATH   = DATA_PROCESSED / "tr_avalon512.npy"
AVALON_TE_PATH   = DATA_PROCESSED / "te_avalon512.npy"
EMBED_TR_PATH    = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
EMBED_TE_PATH    = DATA_PROCESSED / "te_chemprop_embed_300.npy"
MORDRED_TR_PATH  = Path("C:/pxr_artifacts/nb1030/X_mordred_train.npy")
MORDRED_TE_PATH  = Path("C:/pxr_artifacts/nb1030/X_mordred_test.npy")

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

NB2095_TE_PATH = DATA_PROCESSED / "te_nb2095.npy"

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5


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


def _extract_atompair_top_idx_from_nb1484(sum_dict: dict) -> np.ndarray:
    for fam in sum_dict.get("families", []):
        if fam.get("family") == "AtomPair":
            return np.array(fam["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _impute_mordred(X: np.ndarray) -> np.ndarray:
    """Mirror nb2103: replace non-finite Mordred cells with col-median."""
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _load_chembl_pool() -> pd.DataFrame:
    """Mirror nb2063 / nb2103 union (used as ChEMBL kNN feature)."""
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
        .agg(pec50=("pec50", "median"), std_smiles=("std_smiles", "first"))
    )
    return agg.reset_index(drop=True)


def _tanimoto_topk(fp_query, fp_pool, k=5):
    fp_q = fp_query.astype(np.float32)
    fp_p = fp_pool.astype(np.float32)
    pop_q = fp_q.sum(axis=1)
    pop_p = fp_p.sum(axis=1)
    inter = fp_q @ fp_p.T
    union = pop_q[:, None] + pop_p[None, :] - inter
    sim = np.where(union > 0, inter / union, 0.0)
    top_idx = np.argsort(-sim, axis=1)[:, :k]
    top_sim = np.take_along_axis(sim, top_idx, axis=1)
    return top_idx.astype(np.int32), top_sim.astype(np.float32)


def _knn_predict(top_idx, top_sim, pool_labels, fallback):
    n = top_idx.shape[0]
    pred = np.full(n, fallback, dtype=np.float32)
    mean_sim = np.zeros(n, dtype=np.float32)
    for i in range(n):
        sims = top_sim[i]
        w = np.maximum(sims, SIM_FLOOR)
        labs = pool_labels[top_idx[i]]
        s = float(w.sum())
        if s > 0:
            pred[i] = float((w * labs).sum() / s)
        mean_sim[i] = float(sims.mean())
    return pred, mean_sim


def build_117_matrix(use_train: bool, te_smiles: list[str],
                     tr_smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_tr, X_te) as 117-col matrices using same top idx as nb2103."""
    # Load top-K indices per family (cached from nb1352/1392/1484/1523/1524/1541)
    with open(NB1352_SUMMARY) as f:
        s_maccs = json.load(f)
    with open(NB1392_SUMMARY) as f:
        s_av = json.load(f)
    with open(NB1484_SUMMARY) as f:
        s_ap_full = json.load(f)
    with open(NB1523_SUMMARY) as f:
        s_mord = json.load(f)
    with open(NB1524_SUMMARY) as f:
        s_ap_K = json.load(f)
    with open(NB1541_SUMMARY) as f:
        s_emb = json.load(f)

    top_maccs = np.array(s_maccs["top_maccs_bit_indices_ranked"], dtype=int)
    top_av = np.array(s_av["top_avalon_bit_indices_ranked"], dtype=int)
    full_ap = _extract_atompair_top_idx_from_nb1484(s_ap_full)
    K_ap = int(s_ap_K["best_K"])
    top_ap = full_ap[:K_ap]
    rec_mord = _extract_best_K_record(s_mord, "per_K_records")
    top_mord = np.array(rec_mord["top_col_idx"], dtype=int)
    K_emb = int(s_emb["best_K"])
    top_emb = np.array(s_emb["top_dim_order_top100"], dtype=int)[:K_emb]

    # Train + test feature matrices (raw, sliced)
    ap_tr = np.load(ATOMPAIR_TR_PATH, mmap_mode="r")[:, top_ap].astype(np.float32)
    ap_te = np.load(ATOMPAIR_TE_PATH, mmap_mode="r")[:, top_ap].astype(np.float32)
    mc_tr = np.load(MACCS_TR_PATH, mmap_mode="r")[:, top_maccs].astype(np.float32)
    mc_te = np.load(MACCS_TE_PATH, mmap_mode="r")[:, top_maccs].astype(np.float32)
    mo_tr = _impute_mordred(
        np.load(MORDRED_TR_PATH, mmap_mode="r")[:, top_mord].astype(np.float32)
    )
    mo_te = _impute_mordred(
        np.load(MORDRED_TE_PATH, mmap_mode="r")[:, top_mord].astype(np.float32)
    )
    em_tr = np.load(EMBED_TR_PATH, mmap_mode="r")[:, top_emb].astype(np.float32)
    em_te = np.load(EMBED_TE_PATH, mmap_mode="r")[:, top_emb].astype(np.float32)
    av_tr = np.load(AVALON_TR_PATH, mmap_mode="r")[:, top_av].astype(np.float32)
    av_te = np.load(AVALON_TE_PATH, mmap_mode="r")[:, top_av].astype(np.float32)

    # ChEMBL kNN feature: pred_chembl_pec50 + mean_sim (2 cols)
    pool = _load_chembl_pool()
    test_inchikeys = set()
    for s in te_smiles:
        m = standardize(s)
        ik = _safe_inchikey(m)
        if ik:
            test_inchikeys.add(ik)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))

    std_te = []
    for s in te_smiles:
        m = standardize(s)
        std_te.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_te = morgan_fp_batch(std_te)
    top_idx_te, top_sim_te = _tanimoto_topk(fp_te, fp_pool, k=KNN_K)
    pred_te, sim_te = _knn_predict(top_idx_te, top_sim_te, pool_labels, pool_median)

    std_tr = []
    for s in tr_smiles:
        m = standardize(s)
        std_tr.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_tr = morgan_fp_batch(std_tr)
    top_idx_tr, top_sim_tr = _tanimoto_topk(fp_tr, fp_pool, k=KNN_K)
    pred_tr, sim_tr = _knn_predict(top_idx_tr, top_sim_tr, pool_labels, pool_median)

    X_tr = np.concatenate(
        [ap_tr, mc_tr, mo_tr, em_tr, av_tr,
         pred_tr.reshape(-1, 1), sim_tr.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    X_te = np.concatenate(
        [ap_te, mc_te, mo_te, em_te, av_te,
         pred_te.reshape(-1, 1), sim_te.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    print(f"   [build] X_tr={X_tr.shape}  X_te={X_te.shape}")
    return X_tr, X_te


def mahalanobis_distances(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Compute Mahalanobis distance from train centroid for tr (self) and te."""
    mu = X_tr.mean(axis=0)
    Xc_tr = X_tr - mu
    Xc_te = X_te - mu
    # Regularized covariance (Ledoit-Wolf style diagonal shrinkage)
    n = X_tr.shape[0]
    cov = (Xc_tr.T @ Xc_tr) / max(n - 1, 1)
    # Diagonal floor: add COV_REG * mean(diag) to diagonal
    diag_mean = float(np.diag(cov).mean())
    reg = COV_REG * diag_mean if diag_mean > 0 else COV_REG
    cov.flat[:: cov.shape[0] + 1] += reg
    # Solve once: inv_cov @ Xc.T
    inv_cov = np.linalg.pinv(cov.astype(np.float64))
    d2_tr = np.einsum("ij,jk,ik->i", Xc_tr.astype(np.float64),
                      inv_cov, Xc_tr.astype(np.float64))
    d2_te = np.einsum("ij,jk,ik->i", Xc_te.astype(np.float64),
                      inv_cov, Xc_te.astype(np.float64))
    d2_tr = np.maximum(d2_tr, 0.0)
    d2_te = np.maximum(d2_te, 0.0)
    return np.sqrt(d2_tr), np.sqrt(d2_te), mu


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Mahalanobis novelty scaffold weighting on nb2095")
    print(f"          alpha_grid = {ALPHA_GRID}")
    print(f"          ref nb2095 RAE = {NB2095_REF_RAE:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load core ----
    tr = load_train()
    te = load_test()
    # find smiles + name columns
    tr_smi_col = "smiles" if "smiles" in tr.columns else (
        "SMILES" if "SMILES" in tr.columns else "std_smiles"
    )
    te_smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    tr_smiles = tr[tr_smi_col].astype(str).tolist()
    te_smiles = te[te_smi_col].astype(str).tolist()
    te_names = te["name"].astype(str).tolist() if "name" in te.columns else \
        te.iloc[:, 0].astype(str).tolist()
    print(f"[load] n_tr={len(tr_smiles)}  n_te={len(te_smiles)}")

    tr_pec50 = tr["pec50"].astype(float).to_numpy()
    train_mean_pec50 = float(np.nanmean(tr_pec50))
    print(f"[load] train mean pEC50 = {train_mean_pec50:.4f}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- nb2095 base predictions ----
    if not NB2095_TE_PATH.exists():
        raise FileNotFoundError(f"missing {NB2095_TE_PATH} -- run nb2095 first")
    te_nb2095 = np.load(NB2095_TE_PATH).astype(np.float64)
    assert te_nb2095.shape == (len(te_smiles),), (
        f"nb2095 te shape {te_nb2095.shape} vs {len(te_smiles)}"
    )
    base_unb = te_nb2095[unb_idx]
    base_rae = float(rae(y_unb, base_unb))
    print(f"[nb2095] te_unb_rae (in-sample) = {base_rae:.4f}  "
          f"vs reference scaffold-CV = {NB2095_REF_RAE:.4f}")

    # ---- 117-col features ----
    print("\n" + "-" * 78)
    print("BUILD 117-COL FEATURE MATRICES (train + test)")
    print("-" * 78)
    X_tr, X_te = build_117_matrix(True, te_smiles, tr_smiles)
    assert X_tr.shape[1] == 117 and X_te.shape[1] == 117, (
        f"feature dim mismatch: tr={X_tr.shape}, te={X_te.shape}"
    )

    # ---- Mahalanobis distance ----
    print("\n" + "-" * 78)
    print("MAHALANOBIS DISTANCE (train centroid, diag-floored cov)")
    print("-" * 78)
    d_tr, d_te, mu = mahalanobis_distances(X_tr, X_te)
    median_tr = float(np.median(d_tr))
    median_te = float(np.median(d_te))
    print(f"   d_tr  median={median_tr:.3f}  mean={d_tr.mean():.3f}  "
          f"std={d_tr.std():.3f}  max={d_tr.max():.3f}")
    print(f"   d_te  median={median_te:.3f}  mean={d_te.mean():.3f}  "
          f"std={d_te.std():.3f}  max={d_te.max():.3f}")
    print(f"   d_te[unb] median={float(np.median(d_te[unb_idx])):.3f}")

    np.save(DATA_PROCESSED / f"{TAG}_mahal_dist_te.npy", d_te.astype(np.float32))

    # ---- Sweep alpha ----
    print("\n" + "-" * 78)
    print(f"ALPHA SWEEP {ALPHA_GRID} -- scaffold 5-fold CV, "
          f"{len(KF_SEEDS)} seeds")
    print("-" * 78)

    # Per-row weights against TRAIN median (centered novelty)
    w_per_alpha = np.zeros((len(te_smiles), len(ALPHA_GRID)),
                           dtype=np.float32)
    per_alpha_results = []
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})

    for ai, alpha in enumerate(ALPHA_GRID):
        # Weight: high novelty -> small w -> blend leans on train mean
        w_te_all = 1.0 / (1.0 + np.exp(alpha * (d_te - median_tr)))
        w_per_alpha[:, ai] = w_te_all.astype(np.float32)
        w_unb = w_te_all[unb_idx]

        blend_unb = w_unb * base_unb + (1.0 - w_unb) * train_mean_pec50
        in_rae = float(rae(y_unb, blend_unb))

        # Scaffold-CV RAE: blend is deterministic per row (no fit), so
        # CV "RAE" is the pooled RAE on the held-out folds; for a deterministic
        # transform pooled OOF RAE == in-sample RAE. We still report per-seed
        # per-fold to mirror nb2095 protocol and give a stability check.
        per_seed_rae = []
        per_fold_rae_all = []
        for kf_seed in KF_SEEDS:
            splits = scaffold_kfold_indices(
                unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
            )
            oof = np.full(n_unb, np.nan)
            fold_raes = []
            for tr_loc, va_loc in splits:
                oof[va_loc] = blend_unb[va_loc]
                fold_raes.append(float(rae(y_unb[va_loc], blend_unb[va_loc])))
            per_seed_rae.append(float(rae(y_unb, oof)))
            per_fold_rae_all.append(fold_raes)
        mean_seed_rae = float(np.mean(per_seed_rae))
        std_seed_rae = float(np.std(per_seed_rae))
        mean_fold_rae = float(np.mean([r for fr in per_fold_rae_all for r in fr]))
        std_fold_rae = float(np.std([r for fr in per_fold_rae_all for r in fr]))

        # F2-style decompression check: shrinkage moves blend mean toward train
        shrink_share = float((1.0 - w_unb).mean())
        d_vs_ref = mean_seed_rae - NB2095_REF_RAE
        verdict = (
            "BEAT" if d_vs_ref < -DECISION_MARGIN else
            ("LOSE" if d_vs_ref > DECISION_MARGIN else "TIE")
        )

        rec = {
            "alpha": float(alpha),
            "w_te_mean": float(w_te_all.mean()),
            "w_te_std": float(w_te_all.std()),
            "w_te_min": float(w_te_all.min()),
            "w_te_max": float(w_te_all.max()),
            "w_unb_mean": float(w_unb.mean()),
            "shrink_share_unb": shrink_share,
            "blend_unb_mean": float(blend_unb.mean()),
            "blend_unb_std": float(blend_unb.std()),
            "in_sample_rae_unb": in_rae,
            "cv_rae_per_seed": [float(x) for x in per_seed_rae],
            "cv_rae_mean_seeds": mean_seed_rae,
            "cv_rae_std_seeds": std_seed_rae,
            "cv_rae_mean_fold": mean_fold_rae,
            "cv_rae_std_fold": std_fold_rae,
            "delta_vs_nb2095_ref": d_vs_ref,
            "verdict_vs_nb2095_ref": verdict,
        }
        per_alpha_results.append(rec)
        print(f"   alpha={alpha:>4.1f}  w_mean={w_te_all.mean():.3f}  "
              f"shrink_unb={shrink_share:.3f}  in_RAE={in_rae:.4f}  "
              f"CV_mean={mean_seed_rae:.4f} (+/-{std_seed_rae:.4f})  "
              f"d_vs_ref={d_vs_ref:+.4f}  {verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_w_per_alpha.npy", w_per_alpha)

    # ---- Best alpha ----
    best = min(per_alpha_results, key=lambda r: r["cv_rae_mean_seeds"])
    print("\n" + "-" * 78)
    print(f"BEST alpha = {best['alpha']}  CV_mean = {best['cv_rae_mean_seeds']:.4f}"
          f"  delta_vs_nb2095_ref = {best['delta_vs_nb2095_ref']:+.4f}  "
          f"({best['verdict_vs_nb2095_ref']})")
    print("-" * 78)

    summary = {
        "tag": TAG,
        "method": "mahalanobis_novelty_sigmoid_blend_on_nb2095",
        "n_tr": int(len(tr_smiles)),
        "n_te": int(len(te_smiles)),
        "n_unb": int(n_unb),
        "n_unique_unb_scaffolds": int(n_unique_scaf),
        "feature_dim": 117,
        "cov_reg": COV_REG,
        "alpha_grid": ALPHA_GRID,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "train_mean_pec50": train_mean_pec50,
        "median_d_tr": median_tr,
        "median_d_te": median_te,
        "median_d_te_unb": float(np.median(d_te[unb_idx])),
        "nb2095_te_unb_in_rae": base_rae,
        "nb2095_ref_scaffold_cv_rae": NB2095_REF_RAE,
        "decision_margin": DECISION_MARGIN,
        "per_alpha_results": per_alpha_results,
        "best_alpha": best["alpha"],
        "best_cv_rae_mean_seeds": best["cv_rae_mean_seeds"],
        "best_delta_vs_nb2095_ref": best["delta_vs_nb2095_ref"],
        "best_verdict_vs_nb2095_ref": best["verdict_vs_nb2095_ref"],
        "outputs": {
            "mahal_dist_te":  str(DATA_PROCESSED / f"{TAG}_mahal_dist_te.npy"),
            "w_per_alpha":    str(DATA_PROCESSED / f"{TAG}_w_per_alpha.npy"),
        },
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_json}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   nb2095 ref scaffold-CV RAE  = {NB2095_REF_RAE:.4f}")
    for r in per_alpha_results:
        print(f"   alpha={r['alpha']:>4.1f}  CV_mean_seeds={r['cv_rae_mean_seeds']:.4f}  "
              f"(d {r['delta_vs_nb2095_ref']:+.4f}  {r['verdict_vs_nb2095_ref']})")
    print(f"   best alpha = {best['alpha']}  CV = {best['cv_rae_mean_seeds']:.4f}")
    print(f"   wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
