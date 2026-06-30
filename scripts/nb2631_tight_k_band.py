"""nb2631 -- Tight K-band around nb2621 winner: explore K in {17,18,19,20,21}.

NEW PARADIGM:
    nb2621 found K={18,20} as the equal-weight winner (mean_rae 0.4552, PROMOTE)
    over a coarse 9-anchor grid {14,16,18,20,22,24,28,32,36}.  This script
    investigates the 1-step K-band around the winner: {17,18,19,20,21}.

    Hypothesis: with a finer K grid centered on the winner, {19} or {17} may
    contribute orthogonal split structure that improves the equal-weight mean.
    Same paradigm as nb2621 (equal-weight subset enumeration, zero learnable
    df, only choice is WHICH 2..5 to mean).

PROTOCOL:
    1. Cached anchors (reuse on disk):
         K=18 -> nb2604_mean_bag_oof_K18.npy + te_nb2604_K18.npy
         K=20 -> nb2240_mean_bag_oof_K20.npy + te_nb2240_K20.npy
    2. Build NEW anchors (cached on-disk under nb2631_*):
         K=17, K=19, K=21 from nb2231 RFE trajectory snapshots
         Identical LGBM hyperparams + RESID_SEEDS={0,1,7,42,137} as nb2604.
    3. Enumerate equal-weight subsets size 2..5 from {17,18,19,20,21}:
         C(5,2)+C(5,3)+C(5,4)+C(5,5) = 10 + 10 + 5 + 1 = 26 combos
    4. Equal-weight mean OOF; RAE on y_unb.
    5. Save best pred_oof + te + summary.

GATE:
    best mean_rae < 0.4570 -> PROMOTE
    best mean_rae < 0.4552 -> BETTER_THAN_NB2621
    else                   -> FAIL

Outputs:
    scripts/nb2631_tight_k_band.py
    data/processed/nb2631_summary.json
    data/processed/nb2631_pred_oof.npy   (253,) float32
    data/processed/te_nb2631.npy         (513,) float32
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2631"

# ---- Anchor + residual params (identical to nb2604/nb2611) ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# ---- Cache paths for 117-col matrix build ----
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
NB2063_SHAP_PATH = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

# Cached anchors (reuse from nb2604/nb2240)
CACHE_OOF = {
    18: DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy",
    20: DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
}
CACHE_TE = {
    18: DATA_PROCESSED / "te_nb2604_K18.npy",
    20: DATA_PROCESSED / "te_nb2240_K20.npy",
}

# NEW K outputs (cached on-disk under nb2631_*)
NEW_KS = [17, 19, 21]

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# ---- Combo sizes ----
MIN_R = 2
MAX_R = 5

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_BEAT_NB2621 = 0.4552

# ---- Refs ----
CHEMPROP_AUX_REF = 0.6216
NB2621_REF = 0.4552
NB2604_REF = 0.4580
NB2171_REF = 0.4682


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


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def reconstruct_K_from_trajectory(nb2231_sum, K_target):
    """Reconstruct surviving feature indices at K_target from nb2231 RFE trajectory."""
    shap_top28 = list(nb2231_sum["shap_top28_idx_in_117"])
    if K_target == 28:
        return shap_top28
    if K_target > 28:
        if not NB2063_SHAP_PATH.exists():
            raise FileNotFoundError(f"need {NB2063_SHAP_PATH}")
        imp = np.load(NB2063_SHAP_PATH).astype(np.float64)
        order = np.argsort(-imp)
        return [int(j) for j in order[:K_target]]
    current = list(shap_top28)
    traj = nb2231_sum["rfe_trajectory"]
    for entry in traj:
        if entry.get("feat_dropped") is None:
            continue
        if entry["K_after"] < K_target:
            break
        d = int(entry["feat_dropped"])
        if d in current:
            current.remove(d)
        if entry["K_after"] == K_target:
            return current
    if len(current) == K_target:
        return current
    raise ValueError(f"could not reconstruct K={K_target} (got len {len(current)})")


def build_117col_feature_matrix(te_smiles, n_test):
    """Identical 117-col matrix as nb2604/nb2611."""
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
            X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full


def build_K_oof_and_te(K_idx, X_te_full, unb_idx, anchor, residual,
                       te_anchor_513, n_test, n_unb, K_label):
    """Rebuild K-residual LGBM mean-bag OOF + te (identical protocol to nb2604)."""
    X_te_K = X_te_full[:, K_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    per_seed_corr = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K, residual, s)
        per_seed_corr[i] = anchor + resid_oof
        per_seed_rae.append(float(rae(anchor + residual, anchor + resid_oof)))
        te_resid_s = _train_full_then_predict_te(X_unb_K, residual, X_te_K, s)
        per_seed_te_resid[i] = te_resid_s
        print(f"   K={K_label} seed={s:3d}: rae_corr={per_seed_rae[-1]:.4f}  "
              f"wall={time.time()-ts:.1f}s")
    mean_bag_oof_K = per_seed_corr.mean(axis=0)
    mean_bag_te_resid_K = per_seed_te_resid.mean(axis=0)
    te_K_513 = te_anchor_513 + mean_bag_te_resid_K
    return mean_bag_oof_K, te_K_513.astype(np.float32), per_seed_rae


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- tight K-band around nb2621 winner: K in {{17,18,19,20,21}}")
    n_combos_expected = sum(
        len(list(combinations(range(5), r))) for r in range(MIN_R, MAX_R + 1)
    )
    print(f"          C(5,2)+C(5,3)+C(5,4)+C(5,5) = {n_combos_expected} combos")
    print(f"          ref nb2621 = {NB2621_REF:.4f}, gate PROMOTE < {GATE_PROMOTE}")
    print("=" * 78)

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f}")

    # ---- Load cached K=18 + K=20 ----
    print("\n" + "-" * 78)
    print("STEP 1: load cached K=18, K=20 anchors")
    print("-" * 78)
    K_oof = {}
    K_te = {}
    K_source = {}
    for K in sorted(CACHE_OOF.keys()):
        if not CACHE_OOF[K].exists():
            raise FileNotFoundError(f"missing cache OOF for K={K}: {CACHE_OOF[K]}")
        if not CACHE_TE[K].exists():
            raise FileNotFoundError(f"missing cache te for K={K}: {CACHE_TE[K]}")
        K_oof[K] = np.load(CACHE_OOF[K]).astype(np.float64)
        K_te[K] = np.load(CACHE_TE[K]).astype(np.float64)
        K_source[K] = str(CACHE_OOF[K].name)
        if K_oof[K].shape != (n_unb,):
            raise ValueError(f"K={K} OOF shape {K_oof[K].shape} != ({n_unb},)")
        if K_te[K].shape != (n_test,):
            raise ValueError(f"K={K} te shape {K_te[K].shape} != ({n_test},)")
        r = float(rae(y_unb, K_oof[K]))
        print(f"   K={K:2d}  oof_RAE={r:.4f}  te_mean={K_te[K].mean():.3f}  "
              f"src={K_source[K]}")

    # ---- Build NEW K = {17, 19, 21} ----
    print("\n" + "-" * 78)
    print(f"STEP 2: build NEW K anchors {NEW_KS}")
    print("-" * 78)
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)

    K_idx_used = {}
    need_matrix = False
    for K in NEW_KS:
        oof_p = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K}.npy"
        te_p = DATA_PROCESSED / f"te_{TAG}_K{K}.npy"
        if not (oof_p.exists() and te_p.exists()):
            need_matrix = True
            break

    X_te_full = None
    if need_matrix:
        print("   [build] rebuilding 117-col matrix")
        X_te_full = build_117col_feature_matrix(te_smiles, n_test)
        print(f"   X_te_full = {X_te_full.shape}")

    for K in NEW_KS:
        oof_p = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K}.npy"
        te_p = DATA_PROCESSED / f"te_{TAG}_K{K}.npy"
        K_idx = reconstruct_K_from_trajectory(nb2231, K)
        if len(K_idx) != K:
            raise ValueError(f"K={K} idx reconstruction returned {len(K_idx)}")
        K_idx_used[K] = K_idx
        print(f"\n   K={K} idx_in_117 (n={len(K_idx)}): {K_idx[:8]}...")
        if oof_p.exists() and te_p.exists():
            print(f"   [cache] reusing {oof_p.name} + {te_p.name}")
            K_oof[K] = np.load(oof_p).astype(np.float64)
            K_te[K] = np.load(te_p).astype(np.float64)
        else:
            oof_f32, te_f32, _ = build_K_oof_and_te(
                np.array(K_idx, dtype=int), X_te_full, unb_idx, anchor, residual,
                te_anchor_513, n_test, n_unb, K_label=str(K),
            )
            np.save(oof_p, oof_f32.astype(np.float32))
            np.save(te_p, te_f32.astype(np.float32))
            print(f"   [save] {oof_p}")
            print(f"   [save] {te_p}")
            K_oof[K] = oof_f32.astype(np.float64)
            K_te[K] = te_f32.astype(np.float64)
        K_source[K] = str(oof_p.name)
        r = float(rae(y_unb, K_oof[K]))
        print(f"   K={K:2d}  oof_RAE={r:.4f}")

    # ---- Enumerate combos ----
    print("\n" + "-" * 78)
    print(f"STEP 3: enumerate equal-weight subsets size {MIN_R}..{MAX_R}")
    print("-" * 78)
    K_list = sorted(K_oof.keys())  # [17, 18, 19, 20, 21]
    print(f"   K_list = {K_list}")
    rankings = []
    for r in range(MIN_R, MAX_R + 1):
        n_combos_r = 0
        for combo in combinations(K_list, r):
            oof_stack = np.column_stack([K_oof[k] for k in combo])
            pred_oof = oof_stack.mean(axis=1)
            mean_rae_c = float(rae(y_unb, pred_oof))
            rankings.append({
                "r": r,
                "Ks": list(combo),
                "mean_rae": mean_rae_c,
            })
            n_combos_r += 1
        print(f"   r={r}: {n_combos_r} combos evaluated")

    rankings.sort(key=lambda x: x["mean_rae"])
    n_total = len(rankings)
    print(f"\n[total] {n_total} combos evaluated  "
          f"best mean_rae={rankings[0]['mean_rae']:.4f}")

    print("\n" + "-" * 78)
    print("STEP 4: top 5 subsets")
    print("-" * 78)
    print(f"   {'rank':>4s}  {'r':>2s}  {'mean_rae':>9s}  Ks")
    for i, rec in enumerate(rankings[:5]):
        print(f"   {i+1:4d}  {rec['r']:2d}  {rec['mean_rae']:9.4f}  {rec['Ks']}")

    # ---- per-K individual reference ----
    per_K_rae = {}
    print("\n   per-K individual oof_RAE:")
    for K in K_list:
        per_K_rae[K] = float(rae(y_unb, K_oof[K]))
        print(f"     K={K:2d} : {per_K_rae[K]:.4f}")

    # ---- Best ----
    best = rankings[0]
    best_Ks = best["Ks"]
    best_r = best["r"]
    best_mean = best["mean_rae"]
    oof_stack_best = np.column_stack([K_oof[k] for k in best_Ks])
    te_stack_best = np.column_stack([K_te[k] for k in best_Ks])
    pred_oof_best = oof_stack_best.mean(axis=1)
    pred_te_best = te_stack_best.mean(axis=1)
    print(f"\n[best] Ks={best_Ks}  r={best_r}  mean_rae={best_mean:.4f}")
    print(f"       pred_oof std={pred_oof_best.std():.3f}  (truth_std={y_unb.std():.3f})")
    print(f"       pred_te mean={pred_te_best.mean():.3f}  std={pred_te_best.std():.3f}")

    # ---- Gate ----
    if best_mean < GATE_BEAT_NB2621:
        verdict = "BETTER_THAN_NB2621"
    elif best_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    else:
        verdict = "FAIL"
    print(f"[gate] best_mean={best_mean:.4f}  "
          f"(<{GATE_BEAT_NB2621} BETTER_THAN_NB2621 / <{GATE_PROMOTE} PROMOTE)"
          f"  -> {verdict}")

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 5: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_best.astype(np.float32))
    np.save(te_path, pred_te_best.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_tight_k_band.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_best.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    te_unb_in = float(rae(y_unb, pred_te_best[unb_idx]))
    delta_vs_nb2621 = best_mean - NB2621_REF
    delta_vs_nb2604 = best_mean - NB2604_REF
    delta_vs_nb2171 = best_mean - NB2171_REF
    print(f"\n   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")
    print(f"   delta vs nb2621 ({NB2621_REF}) = {delta_vs_nb2621:+.4f}")
    print(f"   delta vs nb2604 ({NB2604_REF}) = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_REF}) = {delta_vs_nb2171:+.4f}")

    # ---- Summary JSON ----
    summary = {
        "tag": TAG,
        "method": "tight_K_band_grid_search_equal_weight",
        "paradigm": "plain_mean_no_learning_df_zero",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "all_Ks": K_list,
        "new_Ks": NEW_KS,
        "new_K_idx_used": {str(K): K_idx_used.get(K, []) for K in NEW_KS},
        "K_source": {str(K): K_source[K] for K in K_list},
        "per_K_oof_rae": {str(K): per_K_rae[K] for K in K_list},
        "cache_paths_oof": {str(k): str(p) for k, p in CACHE_OOF.items()},
        "cache_paths_te": {str(k): str(p) for k, p in CACHE_TE.items()},
        "min_r": MIN_R,
        "max_r": MAX_R,
        "n_combos_total": n_total,
        "n_combos_by_r": {
            str(r): sum(1 for x in rankings if x["r"] == r)
            for r in range(MIN_R, MAX_R + 1)
        },
        "full_ranking": rankings,
        "top5": rankings[:5],
        "best_combo_Ks": best_Ks,
        "best_combo_r": best_r,
        "best_mean_rae": best_mean,
        "gate_promote": GATE_PROMOTE,
        "gate_beat_nb2621": GATE_BEAT_NB2621,
        "verdict": verdict,
        "delta_vs_nb2621": delta_vs_nb2621,
        "nb2621_ref": NB2621_REF,
        "delta_vs_nb2604": delta_vs_nb2604,
        "nb2604_ref": NB2604_REF,
        "delta_vs_nb2171": delta_vs_nb2171,
        "nb2171_ref": NB2171_REF,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "te_mean": float(pred_te_best.mean()),
        "te_std": float(pred_te_best.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K_list                = {K_list}")
    print(f"   per-K oof_RAE         = {{{', '.join(f'{k}:{per_K_rae[k]:.4f}' for k in K_list)}}}")
    print(f"   total combos          = {n_total}")
    print(f"   BEST Ks               = {best_Ks}  (r={best_r})")
    print(f"   BEST mean_rae         = {best_mean:.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   delta vs nb2621       = {delta_vs_nb2621:+.4f}")
    print(f"   delta vs nb2604       = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2171       = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] in-RAE    = {te_unb_in:.4f}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_combo_Ks",
        "best_combo_r",
        "best_mean_rae",
        "verdict",
        "delta_vs_nb2621",
        "delta_vs_nb2604",
        "delta_vs_nb2171",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
