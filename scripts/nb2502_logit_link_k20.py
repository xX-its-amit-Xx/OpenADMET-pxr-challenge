"""nb2502 -- Logit-link transformation for K=20 LGBM residual (enforce pEC50 range).

DIFFERENT: transforms target via logit y_t = log((y - y_min) / (y_max - y))
before fitting an LGBM on the chemprop_aux residual, then inverts via sigmoid
so predictions are mathematically constrained to [y_min, y_max] = [3, 8].

CONTEXT:
    nb2240 establishes the K=20 RFE-surviving features as the operative
    feature substrate for the chemprop_aux residual LGBM at deep-30 RAE
    0.4682 within the 5-anchor pyramid.  This script tests whether a
    logit-link target transform (bounding the prediction range to the
    pEC50 [3, 8] support) improves over the linear MSE objective on the
    same K=20 substrate.

    Hyperparameters mirror nb2240 (`_lgbm_params`):
        max_depth=4, num_leaves=15, n_estimators=300, learning_rate=0.03,
        min_child_samples=5, reg_lambda=2.0.

PROTOCOL:
    1. y_min = 3.0, y_max = 8.0  (PXR pEC50 physical range).
    2. anchor = chemprop_aux te[unb_idx], residual_y = y_unb.
       BUT we logit-transform the FULL prediction target, not the residual:
       fit LGBM on y_t = logit((y_unb - 3)/(8 - 3)), recover via
       y_hat = 3 + 5 * sigmoid(y_t_hat).  This guarantees predictions in
       [3, 8] regardless of fold-train range.
    3. Build 117-col 5-way feature matrix exactly as nb2240, slice to the
       K=20 RFE-surviving features.
    4. 5-fold scaffold CV on 253 unblind, 5 kf_seeds in {1001..1005}.
    5. Compare vs nb2240_mean_bag_oof_K20 reference and the 0.4570/0.4601
       gate band.

OUTPUTS:
    scripts/nb2502_logit_link_k20.py
    data/processed/nb2502_summary.json
    data/processed/nb2502_pred_oof.npy      (253,) float32
    data/processed/te_nb2502.npy            (513,) float32
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

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2502"

# -------- target transform constants --------
Y_MIN = 3.0
Y_MAX = 8.0
Y_RANGE = Y_MAX - Y_MIN
LOGIT_EPS = 1e-3    # squeeze toward (eps, 1-eps) to avoid +/-inf at endpoints

# -------- evaluation constants --------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# -------- nb2240 reference --------
NB2240_REF_OOF_K20 = 0.4682  # from nb2240 K=20 mean-bag context

# -------- feature build paths (copied from nb2240) --------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
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
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

RESID_SEEDS = [0, 1, 7, 42, 137]   # bagging seeds used inside each scaffold-fold


# =============================================================================
# Helpers (copied from nb2240 -- feature build + ChEMBL pool)
# =============================================================================

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


def _tanimoto_topk(fp_q, fp_pool, k):
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


def _load_mordred_test(n_test_expected):
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


def _load_npy_test(path, n_test_expected):
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


# =============================================================================
# Logit-link helpers
# =============================================================================

def to_logit(y):
    """y in [Y_MIN, Y_MAX] -> y_t in real line.

    Clip into (Y_MIN + LOGIT_EPS * range, Y_MAX - LOGIT_EPS * range) to keep
    transformed targets finite even if a label sits exactly at a boundary.
    """
    y = np.asarray(y, dtype=np.float64)
    p = (y - Y_MIN) / Y_RANGE
    p = np.clip(p, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p / (1.0 - p))


def from_logit(y_t):
    """y_t in real line -> y in [Y_MIN, Y_MAX] via sigmoid."""
    y_t = np.asarray(y_t, dtype=np.float64)
    # numerically stable sigmoid
    p = np.where(y_t >= 0,
                 1.0 / (1.0 + np.exp(-y_t)),
                 np.exp(y_t) / (1.0 + np.exp(y_t)))
    return Y_MIN + Y_RANGE * p


def _lgbm_params(seed):
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Logit-link K=20 LGBM (pEC50 clamped to [3, 8])")
    print("=" * 78)

    # --- load 253 unblind labels, anchor, scaffolds ---
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
    print(f"[load] n_test={n_test}  n_unb={n_unb}  chemprop_aux in_RAE={rae_anchor:.4f}")

    # sanity: all labels inside the clamp range
    y_min_obs, y_max_obs = float(y_unb.min()), float(y_unb.max())
    print(f"[range] observed y in [{y_min_obs:.3f}, {y_max_obs:.3f}]  "
          f"clamp = [{Y_MIN}, {Y_MAX}]")
    if y_min_obs < Y_MIN or y_max_obs > Y_MAX:
        print(f"[warn] {(y_unb < Y_MIN).sum() + (y_unb > Y_MAX).sum()} labels "
              "outside clamp -- will be squeezed by LOGIT_EPS")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # --- load K=20 RFE-surviving feature indices ---
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[feat] K=20 surviving features loaded from nb2231")

    # --- rebuild 117-col 5-way feature matrix exactly as nb2240 ---
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

    X_te_full = np.concatenate(
        [
            X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"feat_dim {X_te_full.shape[1]} != 117"
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # =========================================================================
    # 5-fold scaffold CV across 5 kf_seeds, logit-link target, mean-bag seeds
    # =========================================================================
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  bag_seeds={RESID_SEEDS}")
    print(f"target transform: y_t = logit((y - {Y_MIN}) / {Y_RANGE})")
    print("-" * 78)

    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            # mean-bag K=20 LGBM on logit-y
            y_t_tr = to_logit(y_unb[tr_loc])
            bag_preds_t = np.zeros(len(va_loc), dtype=np.float64)
            for bs in RESID_SEEDS:
                mdl = lgb.LGBMRegressor(**_lgbm_params(bs))
                mdl.fit(X_unb_K20[tr_loc], y_t_tr)
                bag_preds_t += mdl.predict(X_unb_K20[va_loc])
            bag_preds_t /= len(RESID_SEEDS)
            oof[va_loc] = from_logit(bag_preds_t)
        # safety: no NaN
        assert not np.isnan(oof).any(), "oof has NaN -- fold cover incomplete"
        pooled = float(rae(y_unb, oof))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "oof_min": float(oof.min()),
            "oof_max": float(oof.max()),
            "oof_mean": float(oof.mean()),
        })
        all_oofs.append(oof)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"oof_range=[{oof.min():.3f}, {oof.max():.3f}]")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

    # --- gate ---
    if pooled_rae_mean_seeds < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean_seeds < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    delta_nb2240 = pooled_rae_mean_seeds - NB2240_REF_OOF_K20
    print(f"\n[gate] mean_rae={pooled_rae_mean_seeds:.4f}  "
          f"PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}  -> {verdict}")
    print(f"[ref ] vs nb2240 K=20 ({NB2240_REF_OOF_K20:.4f})  delta={delta_nb2240:+.4f}")

    # =========================================================================
    # Deploy: mean-bag LGBM(logit) on all 253 unblind, predict on 513 test
    # =========================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (refit on all 253; mean-bag predict 513)")
    print("-" * 78)
    y_t_all = to_logit(y_unb)
    te_bag_t = np.zeros(n_test, dtype=np.float64)
    for bs in RESID_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(bs))
        mdl.fit(X_unb_K20, y_t_all)
        te_bag_t += mdl.predict(X_te_K20)
    te_bag_t /= len(RESID_SEEDS)
    deploy_te = from_logit(te_bag_t).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   deploy te range = [{deploy_te.min():.3f}, {deploy_te.max():.3f}]  "
          f"mean={deploy_te.mean():.3f} std={deploy_te.std():.3f}")
    print(f"   te[unb_idx] RAE = {te_unb_rae:.4f}")
    # verify range constraint
    assert deploy_te.min() >= Y_MIN - 1e-6 and deploy_te.max() <= Y_MAX + 1e-6, (
        f"range violation: [{deploy_te.min()}, {deploy_te.max()}]"
    )

    # --- save artifacts ---
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float32))
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_npy_path}")

    summary = {
        "tag": TAG,
        "method": "logit_link_K20_LGBM_chemprop_aux_anchor_free",
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "logit_eps": LOGIT_EPS,
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names": surviving_K20_names,
        "k20_family_counts": dict(nb2231["snapshots"]["20"]["family_counts"]),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "bag_seeds": RESID_SEEDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "obs_y_min": y_min_obs,
        "obs_y_max": y_max_obs,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "compare_nb2240_K20_ref": NB2240_REF_OOF_K20,
        "delta_vs_nb2240_K20": delta_nb2240,
        "deploy_te_min": float(deploy_te.min()),
        "deploy_te_max": float(deploy_te.max()),
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_unb_rae_in_sample": te_unb_rae,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_npy_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (5 seeds)             = {pooled_rae_mean_seeds:.4f} "
          f"+/- {pooled_rae_std_seeds:.4f}")
    print(f"   verdict                        = {verdict}")
    print(f"   delta vs nb2240 K=20 (0.4682)  = {delta_nb2240:+.4f}")
    print(f"   deploy te range                = "
          f"[{deploy_te.min():.3f}, {deploy_te.max():.3f}]")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "verdict",
        "delta_vs_nb2240_K20",
        "deploy_te_min",
        "deploy_te_max",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
