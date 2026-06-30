"""nb2531 -- AdaBoost on shallow LGBM weak learners (X_117 substrate).

NEW PARADIGM: sequential boosting with sample reweighting based on prior
weak learner error. 5 weak K=20-family LGBMs (shallow: max_depth=2,
num_leaves=5, n_est=50, lr=0.1) chained via sklearn AdaBoostRegressor.

Each weak learner is reweighted toward the residuals where prior learners
performed poorly -- a fundamentally different orthogonality axis than bag
averaging (nb2103/nb2241 LGBM) or stacking (nb1150/nb1158). The shallow
depth-2 base is intentional: AdaBoost wants weak (not strong) base
learners so the reweighting signal stays informative across rounds.

PROTOCOL:
    1. Rebuild the 117-col 5-way K-tuned feature matrix exactly as
       nb2240/nb2241/nb2521/nb2523 (AtomPair top-K / MACCS top-K / Mordred
       top-K / ChempropEmbed top-K / Avalon top-K + ChEMBL kNN pred +
       mean_sim). Use full X_117 (no K=20 slice).
    2. Build chemprop_aux residual on the 253 unblind compounds (anchor =
       te_chemprop_aux, the only verified-clean PRE-unblind anchor).
    3. For each kf_seed in {1001..1005}, scaffold 5-fold CV on the 253
       unblind:
         per fold-train: AdaBoostRegressor(estimator=LGBM(depth=2,
         num_leaves=5, n_est=50, lr=0.1), n_estimators=5, learning_rate=0.5,
         loss='linear') fit on residual; predict fold-val residual.
       Aggregate per-seed OOF; report per_seed RAE and mean-bag RAE.
    4. Deploy: refit AdaBoost on all 253 residual per seed, predict 513 test
       residual; mean-bag across the 5 seeds.

GATE (on the 5-seed mean-bag RAE):
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4601  -> MARGINAL_BEAT
    else               -> FAIL

OUTPUTS:
    scripts/nb2531_adaboost_lgbm.py
    data/processed/nb2531_summary.json
    data/processed/nb2531_pred_oof.npy   (253,) float32  mean-bag corrected
    data/processed/te_nb2531.npy         (513,) float32  deploy refit
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
from sklearn.ensemble import AdaBoostRegressor

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2531"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# AdaBoost hyperparams (per task spec)
ADA_N_ESTIMATORS = 5
ADA_LEARNING_RATE = 0.5
ADA_LOSS = "linear"

# Base LGBM weak learner (per task spec)
BASE_MAX_DEPTH = 2
BASE_NUM_LEAVES = 5
BASE_N_EST = 50
BASE_LR = 0.1

# Substrate sources (mirror nb2241 / nb2523)
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
NB2241_K20_MEAN_BAG_REF = 0.4763


# -------------------------- helpers (mirror nb2523) --------------------------
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


def _base_lgbm(seed: int) -> lgb.LGBMRegressor:
    """Shallow LGBM weak learner (per task spec)."""
    return lgb.LGBMRegressor(
        objective="regression",
        max_depth=BASE_MAX_DEPTH,
        num_leaves=BASE_NUM_LEAVES,
        n_estimators=BASE_N_EST,
        learning_rate=BASE_LR,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _adaboost_lgbm(seed: int) -> AdaBoostRegressor:
    """AdaBoostRegressor wrapping shallow LGBM (per task spec)."""
    return AdaBoostRegressor(
        estimator=_base_lgbm(seed),
        n_estimators=ADA_N_ESTIMATORS,
        learning_rate=ADA_LEARNING_RATE,
        loss=ADA_LOSS,
        random_state=seed,
    )


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X_te_m.shape}")
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


# ----------------------------- AdaBoost cross-fit ----------------------------
def _adaboost_cv_one_seed(X: np.ndarray, residual: np.ndarray,
                          unb_scaffolds: list, kf_seed: int) -> np.ndarray:
    """Scaffold 5-fold CV; per fold-train fit AdaBoost(LGBM) on residual,
    predict fold-val residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = _adaboost_lgbm(seed=kf_seed)
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _adaboost_deploy_te(X_unb: np.ndarray, residual: np.ndarray,
                        X_te: np.ndarray, seed: int) -> np.ndarray:
    """Refit AdaBoost on all 253 unb features, predict 513 te residual."""
    mdl = _adaboost_lgbm(seed=seed)
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


# ============================================================================
# MAIN
# ============================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- AdaBoost on shallow LGBM weak learners (X_117 substrate)")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold")
    print(f"        kf_seeds = {KF_SEEDS}")
    print(f"        AdaBoost(n_est={ADA_N_ESTIMATORS}, lr={ADA_LEARNING_RATE}, "
          f"loss={ADA_LOSS})")
    print(f"        BaseLGBM(depth={BASE_MAX_DEPTH}, leaves={BASE_NUM_LEAVES}, "
          f"n_est={BASE_N_EST}, lr={BASE_LR})")
    print(f"        GATE: mean_rae < {GATE_PROMOTE} PROMOTE; "
          f"< {GATE_MARGINAL} MARGINAL_BEAT; else FAIL")
    print(f"        ref: nb2241 K=20 (raw) mean_bag RAE = "
          f"{NB2241_K20_MEAN_BAG_REF:.4f}")
    print("=" * 78)

    # ---- load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in unb 253 = {n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- rebuild 117-col matrix on 513 test (then slice unb_idx) ----
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

    # ChEMBL kNN feature
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

    X_te_full = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_te_full.shape[1]
    if feat_dim != 117:
        raise ValueError(f"feat_dim {feat_dim} != 117")
    X_unb_117 = X_te_full[unb_idx]
    print(f"[feat] X_unb_117 = {X_unb_117.shape}  X_te_117 = {X_te_full.shape}")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"ADABOOST 5-FOLD SCAFFOLD CV  seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed_oof = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(KF_SEEDS):
        ts = time.time()
        oof_s = _adaboost_cv_one_seed(X_unb_117, residual, unb_scaffolds, s)
        per_seed_oof[i] = oof_s
        per_seed_te_resid[i] = _adaboost_deploy_te(
            X_unb_117, residual, X_te_full, s
        )
        r_s = float(rae(y_unb, anchor + oof_s))
        per_seed_rae.append(r_s)
        print(f"   seed={s}  rae_corr={r_s:.4f}  wall={time.time()-ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_oof = per_seed_oof.mean(axis=0)
    median_bag_oof = np.median(per_seed_oof, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor + mean_bag_oof))
    rae_median_bag = float(rae(y_unb, anchor + median_bag_oof))
    print(f"\n[cv] per_seed_mean RAE = {per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag RAE      = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE    = {rae_median_bag:.4f}")
    delta_vs_raw = rae_mean_bag - NB2241_K20_MEAN_BAG_REF
    delta_vs_anchor = rae_mean_bag - rae_anchor
    print(f"[cv] delta vs nb2241 (raw K=20 ref) = {delta_vs_raw:+.4f}")
    print(f"[cv] delta vs chemprop_aux anchor   = {delta_vs_anchor:+.4f}")

    # ---- Deploy te ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = {te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- Save artefacts (mean-bag corrected on 253, deploy te on 513) ----
    pred_oof_corrected = (anchor + mean_bag_oof).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    # ---- Gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae (mean_bag) = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f} (PROMOTE)        = {rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f} (MARGINAL_BEAT) = {rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "adaboost_shallow_lgbm_X117_residual_on_chemprop_aux",
        "anchor": ANCHOR,
        "rae_anchor_chemprop_aux": rae_anchor,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_features": int(X_te_full.shape[1]),
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "adaboost_params": {
            "n_estimators": ADA_N_ESTIMATORS,
            "learning_rate": ADA_LEARNING_RATE,
            "loss": ADA_LOSS,
        },
        "base_lgbm_params": {
            "max_depth": BASE_MAX_DEPTH,
            "num_leaves": BASE_NUM_LEAVES,
            "n_estimators": BASE_N_EST,
            "learning_rate": BASE_LR,
        },
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,
        "delta_vs_nb2241_raw_K20_mean_bag": delta_vs_raw,
        "delta_vs_anchor": delta_vs_anchor,
        "compare_nb2241_raw_K20_mean_bag": NB2241_K20_MEAN_BAG_REF,
        "te_unb_in_sample_rae": te_unb_in_sample,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"\nwall = {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_nb2241_raw_K20_mean_bag",
        "delta_vs_anchor",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
