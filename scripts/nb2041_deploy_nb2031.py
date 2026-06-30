"""nb2041 -- DEPLOY artifact for nb2031 (5x5=25 BoB lambda=3 pooled-25 residual stack).

nb2031 produced the honest cross-fit pooled-25 RAE = 0.5007 on the 253 unblind
using the lambda=3 outer-bag-of-bags protocol. This script deploys that exact
LGBM(reg_lambda=3.0) configuration to predict residuals on the full 513-row
test set.

PROTOCOL:
    1. Anchor = te_chemprop_aux (PRE-unblind, full 513). residual_target = y_unb - anchor[unb_idx].
    2. Build identical 117-col feature stack for BOTH 253 unblind AND 513 test
       (top-K MACCS + Mordred + AtomPair + ChempropEmbed + Avalon + ChEMBL kNN).
    3. For each outer s in {0, 1, 7, 42, 137}:
         inner_seeds = [s*1000 + j for j in {0, 1, 7, 42, 137}]
         For each inner i_seed:
           Fit LGBM(MSE, max_depth=4, num_leaves=15, n_est=300, lr=0.03,
                   min_child_samples=5, reg_lambda=3.0, random_state=i_seed)
             on ALL 253 unblind residuals.
           residual_pred_513_si = mdl.predict(X_test_117col)
       Mean across 25 fits -> mean_residual_513.
    4. te_nb2041 = te_chemprop_aux + mean_residual_513.
    5. Save submissions/nb2041_deploy_nb2031.csv and data/processed/te_nb2041.npy.

Caveat (per feedback_lb_two_regime_calibration): POST-unblind deploy artifact.
in_RAE(te_nb2041[unb_idx]) is in-sample and OPTIMISTIC. The LB-faithful number
is the nb2031 honest cross-fit pooled-25 RAE = 0.5007 (predicted LB ~0.504).
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2041"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_PER_OUTER_SEEDS = [0, 1, 7, 42, 137]
REG_LAMBDA = 3.0

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
NB2031_SUMMARY = DATA_PROCESSED / "nb2031_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

NB2031_POOLED25_REF = 0.5007  # honest cross-fit LB anchor

SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


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
    """Identical to nb2031: LGBM(regression/MSE) with reg_lambda=3.0."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=REG_LAMBDA,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _load_mordred_matrix(path: Path, n_rows_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({path})")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_rows_expected:
        raise ValueError(
            f"Mordred shape mismatch: {X.shape} vs n_expected={n_rows_expected}"
        )
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy(path: Path, n_rows_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_rows_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY of nb2031 pooled-25 BoB lambda={REG_LAMBDA}")
    print(f"          outer seeds = {OUTER_SEEDS}")
    print(f"          inner_per_outer_seeds = {INNER_PER_OUTER_SEEDS}")
    print(f"          25 fits each on ALL 253 unblind residuals; "
          f"deploy on 513 test")
    print(f"          honest cross-fit LB anchor (nb2031) = "
          f"{NB2031_POOLED25_REF:.4f}")
    print("=" * 78)

    # ---- Live nb2031 reference ----
    if NB2031_SUMMARY.exists():
        with open(NB2031_SUMMARY) as f:
            nb2031_sum = json.load(f)
        nb2031_pooled25 = float(nb2031_sum.get("rae_pooled_25bag",
                                                NB2031_POOLED25_REF))
        print(f"[ref] nb2031_summary.rae_pooled_25bag = "
              f"{nb2031_pooled25:.4f}")
    else:
        nb2031_pooled25 = NB2031_POOLED25_REF
        print(f"[ref] nb2031_summary missing -- using fallback "
              f"{nb2031_pooled25:.4f}")

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns
                 else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].astype(str).tolist()
                if "name" in te.columns
                else te["Molecule Name"].astype(str).tolist())

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: "
            f"{te_anchor_513.shape} vs {n_test}"
        )
    anchor_253 = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_253))
    print(f"[load] te_{ANCHOR}[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual_target = y_unb - anchor_253
    print(f"[resid] mean={residual_target.mean():+.4f}  "
          f"std={residual_target.std():.4f}  "
          f"min={residual_target.min():+.4f}  "
          f"max={residual_target.max():+.4f}")

    # ---- Load K-grid winners + SHAP rankings ----
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

    # ---- Feature matrices: 513 test FULL, then slice unblind ----
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)

    X_maccs_te = _load_npy(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)

    X_mord_te = _load_mordred_matrix(MORDRED_DIR / "X_mordred_test.npy",
                                      n_rows_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)

    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)

    X_av_te = _load_npy(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)
    print(f"[feat] per-family TEST shapes: "
          f"AP={X_ap_te_top.shape}, MACCS={X_maccs_te_top.shape}, "
          f"Mord={X_mord_te_top.shape}, Emb={X_emb_te_top.shape}, "
          f"Av={X_av_te_top.shape}")

    # ---- ChEMBL kNN on full 513 test ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
    print("-" * 78)
    pool = _load_chembl_pool()

    test_mols = [standardize(s) for s in te_smiles]
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
    pred_chembl_pec50_513, mean_sim_513 = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    # ---- Assemble 513-row 117-col matrix ----
    X_test_117 = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_pec50_513.reshape(-1, 1).astype(np.float32),
            mean_sim_513.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_test_117.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"\n   COMBINED 5-WAY K-TUNED matrix (TEST 513): "
          f"{X_test_117.shape}")

    # 253 unblind = slice of 513
    X_unb_117 = X_test_117[unb_idx].astype(np.float32)
    print(f"   COMBINED 5-WAY K-TUNED matrix (UNB 253):  "
          f"{X_unb_117.shape}")

    # ---- DEPLOY LOOP: 25 fits ----
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_PER_OUTER_SEEDS)
    n_fits = n_outer * n_inner
    print("\n" + "-" * 78)
    print(f"DEPLOY: {n_outer} outer x {n_inner} inner = {n_fits} fits "
          f"each on ALL n={n_unb} unblind residuals  "
          f"(reg_lambda={REG_LAMBDA})")
    print("-" * 78)

    per_fit_residual_513 = np.zeros((n_fits, n_test), dtype=np.float64)
    per_fit_records = []
    fit_i = 0
    for o_i, o_seed in enumerate(OUTER_SEEDS):
        for i_j, _i_offset in enumerate(INNER_PER_OUTER_SEEDS):
            i_seed = o_seed * 1000 + _i_offset
            t_in = time.time()
            mdl = lgb.LGBMRegressor(**_lgbm_params(i_seed))
            mdl.fit(X_unb_117, residual_target)
            resid_513_s = mdl.predict(X_test_117).astype(np.float64)
            per_fit_residual_513[fit_i] = resid_513_s

            # in-sample diagnostic
            in_resid_253_s = mdl.predict(X_unb_117).astype(np.float64)
            in_corr_253_s = anchor_253 + in_resid_253_s
            in_rae_s = float(rae(y_unb, in_corr_253_s))
            per_fit_records.append({
                "fit_i": int(fit_i),
                "outer_seed": int(o_seed),
                "inner_seed": int(i_seed),
                "resid_513_mean": float(resid_513_s.mean()),
                "resid_513_std": float(resid_513_s.std()),
                "in_rae_253": in_rae_s,
            })
            print(f"   fit {fit_i:2d}/{n_fits}: outer={o_seed:4d} "
                  f"inner={i_seed:6d}  resid_513 mean={resid_513_s.mean():+.4f} "
                  f"std={resid_513_s.std():.4f}  "
                  f"in_RAE(253)={in_rae_s:.4f}  "
                  f"wall={time.time() - t_in:.1f}s")
            fit_i += 1

    # ---- Mean across 25 fits ----
    mean_residual_513 = per_fit_residual_513.mean(axis=0)
    print(f"\n[bag] mean_residual_513: shape={mean_residual_513.shape}  "
          f"mean={mean_residual_513.mean():+.4f}  "
          f"std={mean_residual_513.std():.4f}  "
          f"min={mean_residual_513.min():+.4f}  "
          f"max={mean_residual_513.max():+.4f}")

    # ---- te_nb2041 = te_chemprop_aux + mean_residual_513 ----
    te_nb2041 = te_anchor_513 + mean_residual_513
    te_mean = float(te_nb2041.mean())
    te_std = float(te_nb2041.std())
    te_min = float(te_nb2041.min())
    te_max = float(te_nb2041.max())
    print(f"[deploy] te_nb2041 shape={te_nb2041.shape}  "
          f"mean={te_mean:.4f}  std={te_std:.4f}  "
          f"min={te_min:.4f}  max={te_max:.4f}")

    # ---- In-sample RAE (OPTIMISTIC; LB anchor is nb2031 pooled-25 = 0.5007) ----
    in_rae_253 = float(rae(y_unb, te_nb2041[unb_idx]))
    delta_vs_anchor = in_rae_253 - rae_anchor
    print("\n" + "-" * 78)
    print("IN-SAMPLE DIAGNOSTIC (optimistic; LB-faithful is nb2031 0.5007)")
    print("-" * 78)
    print(f"   in_RAE(te_nb2041[unb_idx])      = {in_rae_253:.4f}")
    print(f"   in_RAE(te_{ANCHOR}[unb_idx])    = {rae_anchor:.4f}")
    print(f"   delta vs anchor (in-sample)      = {delta_vs_anchor:+.4f}")
    print(f"   nb2031 honest cross-fit pooled-25 = "
          f"{nb2031_pooled25:.4f}  (LB anchor)")
    print(f"   predicted LB (regime-2 calibration) ~ 0.504")

    # ---- Save te artefact ----
    te_out = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_out, te_nb2041.astype(np.float32))
    print(f"\n[save] {te_out}")

    # ---- Save submission CSV ----
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_nb2041.astype(np.float64),
    })
    sub_path = SUBMISSIONS_DIR / f"{TAG}_deploy_nb2031.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[save] {sub_path}  rows={len(sub)}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "deploy_companion_of": "nb2031",
        "outer_seeds": OUTER_SEEDS,
        "inner_per_outer_seeds": INNER_PER_OUTER_SEEDS,
        "n_fits": int(n_fits),
        "lgbm_params_template": _lgbm_params(0),
        "reg_lambda": REG_LAMBDA,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
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
        "K_AP_best": K_AP_best,
        "K_Mord_best": K_Mord_best,
        "K_Embed_best": K_Embed_best,
        "K_Avalon_used": K_Avalon_used,
        "K_MACCS_fixed": n_top_maccs,
        "n_chembl_pool": int(len(pool)),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_target_mean": float(residual_target.mean()),
        "residual_target_std": float(residual_target.std()),
        "mean_residual_513_mean": float(mean_residual_513.mean()),
        "mean_residual_513_std": float(mean_residual_513.std()),
        "te_mean": te_mean,
        "te_std": te_std,
        "te_min": te_min,
        "te_max": te_max,
        "in_rae_253": in_rae_253,
        "delta_in_sample_vs_anchor": delta_vs_anchor,
        "nb2031_pooled25_lb_anchor": nb2031_pooled25,
        "predicted_lb": 0.504,
        "per_fit_records": per_fit_records,
        "te_path": str(te_out),
        "submission_path": str(sub_path),
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "DEPLOY artifact: 25 LGBM(MSE, reg_lambda=3.0) fits each on ALL "
            "253 unblind residuals; deploy residuals mean-averaged on 513. "
            "in_RAE is in-sample and optimistic; LB-faithful number is "
            "nb2031 honest cross-fit pooled-25 RAE = 0.5007 "
            "(predicted LB ~0.504)."
        ),
    }
    sum_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {sum_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "reg_lambda", "outer_seeds", "inner_per_outer_seeds", "n_fits",
        "feat_dim", "n_chembl_pool",
        "rae_anchor_chemprop_aux",
        "mean_residual_513_mean", "mean_residual_513_std",
        "te_mean", "te_std", "te_min", "te_max",
        "in_rae_253", "delta_in_sample_vs_anchor",
        "nb2031_pooled25_lb_anchor", "predicted_lb",
        "te_path", "submission_path",
    ):
        print(f"  {k}: {res.get(k)}")
