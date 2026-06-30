"""nb2930 -- NEW PARADIGM: LGBM K=20 + nb1191 OOF as a 21st input feature.

CONTEXT:
    Cycles 167-169 closed the post-hoc-blend axis at the nb2171/nb2240 K=20
    deep-30 ceiling 0.4682. Substrate-change attempts since (alt anchors,
    spectral features, scaffold kNN, off-manifold neg-mine) have not
    cracked the 0.4570 floor on the chemprop_aux residual.

    NEW PARADIGM (this script):  instead of using nb1191 as a SLSQP
    blend partner POST-LGBM, fold it INTO the LGBM input as a 21st column
    next to the RFE K=20 features.  This lets the tree splits condition
    K=20 feature usage on the nb1191 prediction itself (per-row gating).
    The hypothesis is that nb1191 carries orthogonal signal the K=20 LGBM
    cannot recover on its own from raw features, and tree splits on it
    will produce a residual model the SLSQP convex blend cannot reach.

PROTOCOL:
    1. Rebuild the 117-col 5-way feature bank exactly as nb2240
       (AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN).
    2. Slice to the K=20 RFE-surviving indices from nb2231.
    3. Append nb1191 OOF (honest cross-fit on 253) as the 21st column on
       the 253, and te_nb1191 (in-sample prediction on 513) as the 21st
       column on the 513.
    4. Train LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03) on the
       chemprop_aux residual using the 21-feature matrix.
    5. 5-fold scaffold-CV on the 253 across 5 kf_seeds {1001..1005}.
    6. Gate: mean RAE < 0.4570 -> PROMOTE
            mean RAE < 0.4598 -> MARGINAL_BEAT
            else              -> FAIL.
    7. Save pred_oof + te + summary.json.

Outputs:
    scripts/nb2930_nb1191_as_feature.py
    data/processed/nb2930_summary.json
    data/processed/nb2930_pred_oof.npy   (253,) float32
    data/processed/te_nb2930.npy         (513,) float32
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

TAG = "nb2930"

# ---- Gate (vs cycle-169 deep-30 ceiling 0.4682 chemprop_aux residual) ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- LGBM hyperparameters (per task spec) ----
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_N_ESTIMATORS = 300
LGBM_LR = 0.03

# ---- CV protocol (5 kf_seeds x 5 folds) ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Anchor + feature paths ----
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

NB1191_OOF_PATH = DATA_PROCESSED / "nb1191_pred_oof.npy"
TE_NB1191_PATH = DATA_PROCESSED / "te_nb1191.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (copied/specialised from nb2240)
# ============================================================================

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
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LR,
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


def build_K20_feature_matrices(n_test, te_smiles):
    """Reproduce the K=20 RFE-surviving slice on (513, 20) test matrix."""
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY, NB2231_SUMMARY):
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
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)

    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20

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

    # ChEMBL kNN
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
            X_ap_te_top, X_maccs_te_top, X_mord_te_top,
            X_emb_te_top, X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape == (n_test, 117), f"feat_dim {X_te_full.shape}"

    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    return X_te_K20, surviving_K20, surviving_K20_names


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM K=20 + nb1191 OOF as 21st feature (NEW PARADIGM)")
    print("=" * 78)
    print(f"   gate PROMOTE  : mean RAE < {GATE_PROMOTE}")
    print(f"   gate MARGINAL : mean RAE < {GATE_MARGINAL}")
    print(f"   LGBM          : max_depth={LGBM_MAX_DEPTH}  num_leaves={LGBM_NUM_LEAVES}  "
          f"n_est={LGBM_N_ESTIMATORS}  lr={LGBM_LR}")
    print(f"   CV            : {N_FOLDS}-fold scaffold-CV across kf_seeds={KF_SEEDS}")

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}  unique_scaffolds={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor_unb

    # ---- Build K=20 feature matrix ----
    print("\n[feat] rebuilding 117-col 5-way feature bank + ChEMBL kNN ...")
    X_te_K20, surviving_K20, surviving_K20_names = build_K20_feature_matrices(
        n_test, te_smiles,
    )
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # ---- Load nb1191 as 21st feature ----
    assert NB1191_OOF_PATH.exists(), f"missing {NB1191_OOF_PATH}"
    assert TE_NB1191_PATH.exists(), f"missing {TE_NB1191_PATH}"
    nb1191_oof = np.load(NB1191_OOF_PATH).astype(np.float32)
    te_nb1191 = np.load(TE_NB1191_PATH).astype(np.float32)
    assert nb1191_oof.shape == (n_unb,), f"nb1191_oof shape {nb1191_oof.shape}"
    assert te_nb1191.shape == (n_test,), f"te_nb1191 shape {te_nb1191.shape}"
    rae_nb1191 = float(rae(y_unb, nb1191_oof.astype(np.float64)))
    print(f"[feat] nb1191 in_RAE on 253 = {rae_nb1191:.4f}")

    # ---- Concatenate: K=20 + nb1191 col -> 21 features ----
    X_unb_21 = np.concatenate(
        [X_unb_K20, nb1191_oof.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    X_te_21 = np.concatenate(
        [X_te_K20, te_nb1191.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    feature_names_21 = list(surviving_K20_names) + ["nb1191_oof"]
    print(f"[feat] X_unb_21 = {X_unb_21.shape}  X_te_21 = {X_te_21.shape}")
    print(f"[feat] feature_names (last 3): {feature_names_21[-3:]}")

    # ---- 5-fold scaffold-CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  n_folds={N_FOLDS}  kf_seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed = []
    all_oof = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
            mdl.fit(X_unb_21[tr_loc], residual[tr_loc])
            oof_resid[va_loc] = mdl.predict(X_unb_21[va_loc])
        oof_pred = anchor_unb + oof_resid
        pooled = float(rae(y_unb, oof_pred))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
        })
        all_oof.append(oof_pred)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  wall={time.time()-ts:.1f}s")

    pooled_arr = np.asarray([r["pooled_rae"] for r in per_seed])
    mean_rae = float(pooled_arr.mean())
    std_rae = float(pooled_arr.std())
    min_rae = float(pooled_arr.min())
    max_rae = float(pooled_arr.max())
    mean_oof = np.mean(np.column_stack(all_oof), axis=1)
    rae_of_mean_oof = float(rae(y_unb, mean_oof))
    print(f"\n[cv] pooled_RAE mean = {mean_rae:.4f} +/- {std_rae:.4f}  "
          f"[{min_rae:.4f}, {max_rae:.4f}]")
    print(f"[cv] RAE(mean-of-seed OOFs)  = {rae_of_mean_oof:.4f}")

    # ---- Deploy refit on all 253 -> predict on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY (mean-bag across 5 kf_seeds; refit on all 253)")
    print("-" * 78)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    for i, kf_seed in enumerate(KF_SEEDS):
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X_unb_21, residual)
        per_seed_te_resid[i] = mdl.predict(X_te_21)
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    deploy_te = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_rae_in_sample = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   deploy te[unb_idx] RAE (in-sample) = {te_unb_rae_in_sample:.4f}")
    print(f"   deploy te(513) mean/std            = "
          f"{float(deploy_te.mean()):.3f}/{float(deploy_te.std()):.3f}")

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae          = {mean_rae:.4f}")
    print(f"   GATE_PROMOTE      = < {GATE_PROMOTE}")
    print(f"   GATE_MARGINAL     = < {GATE_MARGINAL}")
    print(f"   verdict           = {verdict}")

    # ---- Save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "LGBM_K20_plus_nb1191_as_21st_feature",
        "lgbm_params": {
            "max_depth": LGBM_MAX_DEPTH,
            "num_leaves": LGBM_NUM_LEAVES,
            "n_estimators": LGBM_N_ESTIMATORS,
            "learning_rate": LGBM_LR,
        },
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "anchor": "chemprop_aux",
        "anchor_in_rae_253": rae_anchor,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "nb1191_in_rae_253": rae_nb1191,
        "nb1191_oof_path": str(NB1191_OOF_PATH),
        "te_nb1191_path": str(TE_NB1191_PATH),
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names": surviving_K20_names,
        "feature_names_21": feature_names_21,
        "per_seed_results": per_seed,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "rae_of_mean_of_seed_oofs": rae_of_mean_oof,
        "te_unb_rae_in_sample": te_unb_rae_in_sample,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (5-seed)      = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   rae(mean-of-OOFs)      = {rae_of_mean_oof:.4f}")
    print(f"   anchor (chemprop_aux)  = {rae_anchor:.4f}")
    print(f"   nb1191 (in_rae)        = {rae_nb1191:.4f}")
    print(f"   te[unb_idx] in-sample  = {te_unb_rae_in_sample:.4f}")
    print(f"   gate verdict           = {verdict}")
    print(f"   wall                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "std_rae",
        "min_rae",
        "max_rae",
        "rae_of_mean_of_seed_oofs",
        "anchor_in_rae_253",
        "nb1191_in_rae_253",
        "te_unb_rae_in_sample",
        "verdict",
        "pred_oof_path",
        "te_npy_path",
    ):
        print(f"  {k}: {res.get(k)}")
