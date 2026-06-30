"""nb2034 -- LightGBM iteration-trajectory averaging on 5-way K-tuned 117-col features.

HYPOTHESIS:
    Instead of using only the final iteration's prediction from each LightGBM
    fit (n_estimators=300 -> predict @ 300 rounds), tap into the BOOSTING
    TRAJECTORY: collect predictions at multiple intermediate snapshots
    (rounds 200, 225, 250, 275, 300) and average them WITHIN each seed-fold
    fit before bagging across seeds.  This is a form of stochastic weight
    averaging (SWA) in iteration space -- it sees a tighter posterior over the
    region of the loss surface near the convergence basin, smoothing
    high-frequency noise in the late boosting rounds.

    Concretely, for each seed * fold * iteration-snapshot point we get a
    predict_oof slice; averaging the 5 snapshots inside one fit gives one
    smoothed fold output, then KFold reassembles 253 OOFs, then 5 seeds get
    bagged.  Total predict count = 5 seeds * 5 folds * 5 snapshots = 125
    predict calls (cheap; predict on intermediate rounds is via
    num_iteration=N kwarg, no retraining).

    Anchor / feature stack identical to nb1861 / nb2013 / nb2023.

PROTOCOL:
    1. Anchor = chemprop_aux te[unb_idx] (PRE-unblind, in_RAE 0.6216).
       residual = y_unb - anchor.
    2. Build 117-col 5-way K-tuned feature matrix.
    3. For each seed in {0,1,2,3,4}:
         KFold(5, shuffle=True, random_state=seed) cross-fit:
           train LGBMRegressor(MSE, depth=4, leaves=15, 300 rounds, lr=0.03,
             min_child=5, lambda=2.0)
           predict at iteration {200, 225, 250, 275, 300}
           snapshot_pred = mean of 5 snapshots  (within-fit SWA)
         assemble per-seed OOF residual_oof
         per-seed corrected anchor = anchor + residual_oof
    4. mean_bag = mean across 5 seeds; median_bag = median.
    5. Compare also vs LAST-ONLY baseline (rounds=300 only, no SWA) using the
       SAME random_state draws -- pure ablation, the only difference is the
       trajectory-average step.

Outputs:
    scripts/nb2034_lgbm_trajectory.py
    data/processed/nb2034_summary.json
    data/processed/nb2034_mean_bag_oof.npy           (253,) float32 -- trajectory
    data/processed/nb2034_mean_bag_oof_last_only.npy (253,) float32 -- ablation
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

TAG = "nb2034"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

SEEDS = [0, 1, 2, 3, 4]
RESID_FOLDS = 5
N_ESTIMATORS = 300
TRAJ_SNAPSHOTS = [200, 225, 250, 275, 300]   # 5-point SWA window

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

CHEMPROP_AUX_REF = 0.6216
NB1852_REF = 0.5100   # MSE single-seed
NB1861_REF = 0.5013   # 5-outer-seed BoB MSE
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
    """Identical to nb1852/nb1861/nb2013: MSE / depth=4 / 300 rounds / lr=0.03."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=N_ESTIMATORS,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_one_seed_trajectory(X: np.ndarray, residual: np.ndarray,
                                            seed: int):
    """Return (oof_traj, oof_last):
       oof_traj is per-fold mean across TRAJ_SNAPSHOTS predictions,
       oof_last is per-fold prediction at n_estimators only.

    Both are 253-length 1-D arrays.  Same fits, same random_state, same KFold;
    the only difference is which iteration(s) we read out.
    """
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof_traj = np.full(n, np.nan, dtype=np.float64)
    oof_last = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        # iteration trajectory snapshots
        snap_preds = []
        for n_iter in TRAJ_SNAPSHOTS:
            p = mdl.predict(X[va_loc], num_iteration=int(n_iter))
            snap_preds.append(p)
        snap_arr = np.vstack(snap_preds)        # (5, n_val)
        oof_traj[va_loc] = snap_arr.mean(axis=0)
        oof_last[va_loc] = snap_arr[-1]         # n_estimators == 300 row
    return oof_traj, oof_last


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
    print(f"{TAG} -- LightGBM iteration-trajectory averaging "
          f"(snapshots {TRAJ_SNAPSHOTS})")
    print(f"          anchor={ANCHOR}  seeds={SEEDS}  folds={RESID_FOLDS}")
    print(f"          refs: chemprop_aux ({CHEMPROP_AUX_REF:.4f}), "
          f"nb1852 ({NB1852_REF:.4f}), nb1861 ({NB1861_REF:.4f})")
    print(f"          decision_margin = {DECISION_MARGIN}")
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

    # ---- Load all K-grid winners ----
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
    X_ap_unb_top = X_ap_te[unb_idx][:, top_ap_bit_idx].astype(np.float32)
    print(f"[feat] X_ap_unb_top      = {X_ap_unb_top.shape}")

    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb_top = X_maccs_te[unb_idx][:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_unb_top   = {X_maccs_unb_top.shape}")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb_top = X_mord_te[unb_idx][:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_unb_top    = {X_mord_unb_top.shape}")

    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb_top = X_emb_te[unb_idx][:, top_embed_col_idx].astype(np.float32)
    print(f"[feat] X_emb_unb_top     = {X_emb_unb_top.shape}")

    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb_top = X_av_te[unb_idx][:, top_avalon_bit_idx].astype(np.float32)
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

    # ---- 5-seed trajectory bag ----
    print("\n" + "=" * 78)
    print(f"5-SEED TRAJECTORY BAG  ({len(TRAJ_SNAPSHOTS)} snapshots per fold)")
    print(f"   snapshots = {TRAJ_SNAPSHOTS}")
    print(f"   dim = {feat_dim}")
    print("=" * 78)

    per_seed_traj = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_last = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_records = []

    for s_i, seed in enumerate(SEEDS):
        t_s = time.time()
        oof_traj, oof_last = _residual_cross_fit_one_seed_trajectory(
            X_unb, residual, seed
        )
        pred_traj = anchor + oof_traj
        pred_last = anchor + oof_last
        per_seed_traj[s_i] = pred_traj
        per_seed_last[s_i] = pred_last
        rae_traj_s = float(rae(y_unb, pred_traj))
        rae_last_s = float(rae(y_unb, pred_last))
        delta = rae_traj_s - rae_last_s
        per_seed_records.append({
            "seed": int(seed),
            "rae_traj": rae_traj_s,
            "rae_last": rae_last_s,
            "delta_traj_minus_last": delta,
            "wall_sec": round(time.time() - t_s, 2),
        })
        print(f"   seed={seed:3d}  traj={rae_traj_s:.4f}  "
              f"last={rae_last_s:.4f}  d_traj_minus_last={delta:+.4f}  "
              f"wall={time.time() - t_s:.1f}s")

    mean_bag_traj = per_seed_traj.mean(axis=0)
    median_bag_traj = np.median(per_seed_traj, axis=0)
    mean_bag_last = per_seed_last.mean(axis=0)
    median_bag_last = np.median(per_seed_last, axis=0)

    rae_mean_bag_traj = float(rae(y_unb, mean_bag_traj))
    rae_median_bag_traj = float(rae(y_unb, median_bag_traj))
    rae_mean_bag_last = float(rae(y_unb, mean_bag_last))
    rae_median_bag_last = float(rae(y_unb, median_bag_last))

    rae_traj_arr = np.array([r["rae_traj"] for r in per_seed_records])
    rae_last_arr = np.array([r["rae_last"] for r in per_seed_records])

    print()
    print(f"   per-seed RAE traj  = "
          f"[{', '.join(f'{r:.4f}' for r in rae_traj_arr)}]")
    print(f"   per-seed RAE last  = "
          f"[{', '.join(f'{r:.4f}' for r in rae_last_arr)}]")
    print(f"   seed_mean traj/last= {rae_traj_arr.mean():.4f} / "
          f"{rae_last_arr.mean():.4f}")
    print(f"   seed_std  traj/last= {rae_traj_arr.std():.4f} / "
          f"{rae_last_arr.std():.4f}")
    print(f"   mean_bag   traj    = {rae_mean_bag_traj:.4f}")
    print(f"   mean_bag   last    = {rae_mean_bag_last:.4f}")
    print(f"   median_bag traj    = {rae_median_bag_traj:.4f}")
    print(f"   median_bag last    = {rae_median_bag_last:.4f}")
    print(f"   d(mean_bag) traj-last = "
          f"{rae_mean_bag_traj - rae_mean_bag_last:+.4f}")
    print(f"   d(mean_bag traj) vs nb1861 = "
          f"{rae_mean_bag_traj - NB1861_REF:+.4f}")
    print(f"   d(mean_bag traj) vs anchor = "
          f"{rae_mean_bag_traj - rae_anchor:+.4f}")

    # ---- Save outputs ----
    out_traj = DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy"
    out_last = DATA_PROCESSED / f"{TAG}_mean_bag_oof_last_only.npy"
    np.save(out_traj, mean_bag_traj.astype(np.float32))
    np.save(out_last, mean_bag_last.astype(np.float32))
    print(f"\n[save] {out_traj}")
    print(f"[save] {out_last}")

    # ---- Verdict ----
    delta_vs_nb1861 = rae_mean_bag_traj - NB1861_REF
    delta_vs_last = rae_mean_bag_traj - rae_mean_bag_last
    if delta_vs_nb1861 < -DECISION_MARGIN:
        verdict = "TRAJECTORY_IMPROVES_OVER_NB1861"
    elif delta_vs_nb1861 > DECISION_MARGIN:
        verdict = "TRAJECTORY_HURTS_VS_NB1861"
    else:
        verdict = "TRAJECTORY_NEUTRAL_VS_NB1861"
    if delta_vs_last < -0.0005:
        ablation = "TRAJECTORY_BEATS_LAST_ONLY"
    elif delta_vs_last > 0.0005:
        ablation = "LAST_ONLY_BEATS_TRAJECTORY"
    else:
        ablation = "TRAJECTORY_EQUALS_LAST_ONLY"
    print(f"\n   verdict   = {verdict}")
    print(f"   ablation  = {ablation}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("AtomPair-cache + MACCS-cache + "
                        "Mordred-cached_nb1030 + ChempropEmbed-cache + "
                        "Avalon-cache + local_chembl_caches_union"),
        "model_family": "LightGBM",
        "trajectory_method": "within-fit-iteration-snapshot-average",
        "snapshots": TRAJ_SNAPSHOTS,
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": N_ESTIMATORS,
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
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_records": per_seed_records,
        "seed_mean_rae_traj": float(rae_traj_arr.mean()),
        "seed_std_rae_traj": float(rae_traj_arr.std()),
        "seed_mean_rae_last": float(rae_last_arr.mean()),
        "seed_std_rae_last": float(rae_last_arr.std()),
        "rae_mean_bag_traj": rae_mean_bag_traj,
        "rae_median_bag_traj": rae_median_bag_traj,
        "rae_mean_bag_last": rae_mean_bag_last,
        "rae_median_bag_last": rae_median_bag_last,
        "delta_mean_bag_traj_vs_anchor": rae_mean_bag_traj - rae_anchor,
        "delta_mean_bag_traj_vs_nb1852": rae_mean_bag_traj - NB1852_REF,
        "delta_mean_bag_traj_vs_nb1861": delta_vs_nb1861,
        "delta_mean_bag_traj_vs_last_only": delta_vs_last,
        "mean_bag_traj_oof_path": str(out_traj),
        "mean_bag_last_oof_path": str(out_last),
        "verdict": verdict,
        "ablation_vs_last_only": ablation,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb1852_ref": NB1852_REF,
        "nb1861_ref": NB1861_REF,
        "decision_margin": DECISION_MARGIN,
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
    print(f"  anchor RAE             : {res['rae_anchor_chemprop_aux']:.4f}")
    print(f"  nb1861 ref             : {res['nb1861_ref']:.4f}")
    print(f"  mean_bag traj          : {res['rae_mean_bag_traj']:.4f}")
    print(f"  mean_bag last_only     : {res['rae_mean_bag_last']:.4f}")
    print(f"  median_bag traj        : {res['rae_median_bag_traj']:.4f}")
    print(f"  median_bag last_only   : {res['rae_median_bag_last']:.4f}")
    print(f"  d(traj vs nb1861)      : {res['delta_mean_bag_traj_vs_nb1861']:+.4f}")
    print(f"  d(traj vs last_only)   : {res['delta_mean_bag_traj_vs_last_only']:+.4f}")
    print(f"  verdict                : {res['verdict']}")
    print(f"  ablation               : {res['ablation_vs_last_only']}")
