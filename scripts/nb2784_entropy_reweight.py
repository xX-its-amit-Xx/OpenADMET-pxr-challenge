"""nb2784 -- Per-row entropy reweighting based on per-row anchor disagreement.

NEW PARADIGM (cycle 175 prescription):
    weight each row by exp(-H * lambda) where H = variance over normalized
    per-anchor predictions (low disagreement = high weight). Train a K=20
    LGBM on chemprop_aux residual with sample_weight=w. Sweep lambda in
    {0.5, 1.0, 2.0} and select the best by honest scaffold-5-fold CV across
    5 kf_seeds on the 253 unblind.

ANCHOR STACK (4-way, all PRE-clean or counter-clean):
    1. nb2240_K20      mean_bag_oof_K20.npy (K=20 RFE chemprop_aux residual)
    2. chemprop_aux    nb1133_chemprop_aux_pred_oof.npy (PRE-unblind anchor)
    3. counter_clean   nb2490_pred_oof.npy (clean counter-assay-axis anchor)
    4. nb1191          reconstructed from sub-anchors (PRE-unblind pyramid)

DISAGREEMENT METRIC:
    Per row i: zscore each anchor pred across rows (mu, sigma over 253),
    then H_i = variance of the 4 z-scored values at row i. Higher H means
    the anchors disagree more (in normalized units). Then
    w_i = exp(-H_i * lambda) / mean(exp(-H * lambda)) so weights have
    mean 1.0 (LGBM sample_weight convention; total weight preserved).

LGBM:
    K=20 RFE features sliced from the 117-col 5-way matrix
    (AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN).
    Hyperparams: max_depth=4, num_leaves=15, n_est=300, lr=0.03,
    min_child_samples=5, reg_lambda=2.0. Identical to nb2240 K=20.

GATE:
    best lambda mean_rae < 0.4570 -> "PROMOTE"
    best lambda mean_rae < 0.4598 -> "MARGINAL_BEAT"  (vs 2nd-best ceiling)
    else                          -> "FAIL"

Outputs:
    scripts/nb2784_entropy_reweight.py
    data/processed/nb2784_summary.json
    data/processed/nb2784_pred_oof.npy            (253,) float32 best lambda
    data/processed/te_nb2784.npy                  (513,) float32 best lambda
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
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2784"

# ---- Anchor paths (all PRE-clean) ----
NB2240_OOF = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
NB2240_TE = DATA_PROCESSED / "te_nb2240_K20.npy"
CHEMPROP_AUX_OOF = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
CHEMPROP_AUX_TE = DATA_PROCESSED / "te_chemprop_aux.npy"
COUNTER_CLEAN_OOF = DATA_PROCESSED / "nb2490_pred_oof.npy"
COUNTER_CLEAN_TE = DATA_PROCESSED / "te_nb2490.npy"

# nb1191 reconstructed from its 4 sub-anchor OOFs (PRE-unblind pyramid)
NB1191_SUB_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1191_SLSQP4_WEIGHTS = np.array(
    [0.0, 0.2942, 0.0, 0.7058], dtype=np.float64
)  # nb1150 sub-blend
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031
NB1158_OOF = DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
NB2112_OOF = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
NB1191_TE = DATA_PROCESSED / "te_nb1191.npy"

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

# ---- Feature paths (K=20 from nb2240 surviving subset) ----
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

# ---- Hyperparameters ----
LAMBDA_GRID = [0.5, 1.0, 2.0]
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
RESID_SEEDS = [0, 1, 7, 42, 137]  # mean-bag over 5 LGBM seeds within each fold

# ---- Gate thresholds (vs nb2171 0.4682 deep-30 ceiling band) ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


# ============================================================================
# helpers
# ============================================================================

def _murcko(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) or ""
    except Exception:
        return ""


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


def _load_chembl_pool() -> pd.DataFrame:
    from pxr.chem import standardize, morgan_fp_batch
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


def build_k20_features(te_smiles, n_test, unb_idx):
    """Build the 117-col 5-way feature matrix then slice to K=20."""
    from pxr.chem import standardize, morgan_fp_batch

    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    assert len(surviving_K20) == 20

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

    # ChEMBL kNN
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
    test_inchikeys = {ik for m in test_mols if (ik := _safe_inchikey(m)) is not None}
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_test_smiles = [_safe_can_smiles(m) or "" for m in test_mols]
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
    return X_unb_K20, X_te_K20, surviving_K20


# ---------------------------------------------------------------------------
# nb1191 OOF reconstruction (same recipe as nb2240)
# ---------------------------------------------------------------------------

def reconstruct_nb1191_oof(n_unb):
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    # nb1150 sub-blend on its 4 components
    nb1150_cols = []
    for rel in NB1191_SUB_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 sub-anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        nb1150_cols.append(v)
    nb1150_oof = np.column_stack(nb1150_cols) @ NB1191_SLSQP4_WEIGHTS
    nb1158_oof = np.load(NB1158_OOF).astype(np.float64)
    nb2112_oof = np.load(NB2112_OOF).astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


# ---------------------------------------------------------------------------
# entropy weights
# ---------------------------------------------------------------------------

def entropy_weights(P_unb, lam):
    """
    P_unb: (n, K) anchor predictions on 253 unblind.
    Returns w of shape (n,) normalized so mean(w) == 1.0.

    H_i = variance of z-scored anchor preds at row i (across K anchors).
    w_i_raw = exp(-H_i * lam); w_i = w_i_raw / mean(w_i_raw).
    """
    P = P_unb.astype(np.float64)
    mu = P.mean(axis=0, keepdims=True)
    sd = P.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (P - mu) / sd                            # (n, K)
    H = Z.var(axis=1, ddof=0)                    # (n,)
    w_raw = np.exp(-H * float(lam))
    w_raw = np.where(np.isfinite(w_raw), w_raw, 0.0)
    w_mean = float(w_raw.mean()) if w_raw.mean() > 0 else 1.0
    w = (w_raw / w_mean).astype(np.float64)
    return w, H


# ---------------------------------------------------------------------------
# cross-fit one weighted-LGBM bag of 5 seeds, per kf_seed
# ---------------------------------------------------------------------------

def cv_run_for_seed(X_unb, residual, weights, anchor, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = X_unb.shape[0]
    oof_resid = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        # mean-bag of 5 LGBM seeds per fold
        preds_va = np.zeros(len(va_loc), dtype=np.float64)
        for s in RESID_SEEDS:
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb[tr_loc], residual[tr_loc], sample_weight=weights[tr_loc])
            preds_va += mdl.predict(X_unb[va_loc]) / len(RESID_SEEDS)
        oof_resid[va_loc] = preds_va
    pred_oof = anchor + oof_resid
    return float(rae(y_unb, pred_oof)), pred_oof


def deploy_refit_te(X_unb, residual, weights, X_te, te_anchor):
    """Refit on all 253 with weights; predict residual on full 513."""
    te_resid_bag = np.zeros(X_te.shape[0], dtype=np.float64)
    for s in RESID_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb, residual, sample_weight=weights)
        te_resid_bag += mdl.predict(X_te) / len(RESID_SEEDS)
    return (te_anchor + te_resid_bag).astype(np.float32)


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-row entropy reweighting (lambda sweep)")
    print("=" * 78)

    # ---- Load truth ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Load 4 anchors (OOF on 253 + te on 513) ----
    anchor_specs = []
    nb2240_oof = np.load(NB2240_OOF).astype(np.float64)
    nb2240_te = np.load(NB2240_TE).astype(np.float64)
    assert nb2240_oof.shape == (n_unb,)
    assert nb2240_te.shape == (n_test,)
    anchor_specs.append(("nb2240_K20", nb2240_oof, nb2240_te))

    chemprop_oof = np.load(CHEMPROP_AUX_OOF).astype(np.float64)
    chemprop_te = np.load(CHEMPROP_AUX_TE).astype(np.float64)
    assert chemprop_oof.shape == (n_unb,)
    assert chemprop_te.shape == (n_test,)
    anchor_specs.append(("chemprop_aux", chemprop_oof, chemprop_te))

    counter_oof = np.load(COUNTER_CLEAN_OOF).astype(np.float64)
    counter_te = np.load(COUNTER_CLEAN_TE).astype(np.float64)
    assert counter_oof.shape == (n_unb,), f"counter_oof {counter_oof.shape}"
    assert counter_te.shape == (n_test,), f"counter_te {counter_te.shape}"
    anchor_specs.append(("counter_clean", counter_oof, counter_te))

    nb1191_added = False
    try:
        nb1191_oof = reconstruct_nb1191_oof(n_unb)
        nb1191_te = np.load(NB1191_TE).astype(np.float64)
        assert nb1191_oof.shape == (n_unb,)
        assert nb1191_te.shape == (n_test,)
        anchor_specs.append(("nb1191", nb1191_oof, nb1191_te))
        nb1191_added = True
    except Exception as e:
        print(f"[skip] nb1191 reconstruction failed: {e}")

    print(f"\n[anchors] using {len(anchor_specs)}:")
    indiv_rae = {}
    for name, oof, _te in anchor_specs:
        r = float(rae(y_unb, oof))
        indiv_rae[name] = r
        print(f"   {name:14s} oof_RAE={r:.4f}")

    P_unb = np.column_stack([a[1] for a in anchor_specs])
    P_te = np.column_stack([a[2] for a in anchor_specs])
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}")

    # ---- Anchor = chemprop_aux (canonical) ----
    anchor_unb = chemprop_oof
    anchor_te = chemprop_te
    residual = y_unb - anchor_unb
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux in_RAE = {rae_anchor:.4f}")

    # ---- Build K=20 features ----
    print("\n[feat] building K=20 5-way features ...")
    t1 = time.time()
    X_unb_K20, X_te_K20, surviving_K20 = build_k20_features(te_smiles, n_test, unb_idx)
    print(f"[feat] X_unb_K20={X_unb_K20.shape}  X_te_K20={X_te_K20.shape}  wall={time.time()-t1:.1f}s")

    # ---- Sweep lambda ----
    print("\n" + "-" * 78)
    print(f"LAMBDA SWEEP  grid={LAMBDA_GRID}  kf_seeds={KF_SEEDS}  resid_seeds={RESID_SEEDS}")
    print("-" * 78)

    per_lambda = []
    best_lam = None
    best_mean_rae = float("inf")
    best_pred_oof = None
    best_te = None
    best_diagnostics = None

    for lam in LAMBDA_GRID:
        t_lam = time.time()
        w, H = entropy_weights(P_unb, lam)
        diag = {
            "lambda": float(lam),
            "weight_mean": float(w.mean()),
            "weight_min": float(w.min()),
            "weight_max": float(w.max()),
            "weight_std": float(w.std()),
            "entropy_mean": float(H.mean()),
            "entropy_min": float(H.min()),
            "entropy_max": float(H.max()),
        }
        per_seed_results = []
        all_oofs = []
        for kf_seed in KF_SEEDS:
            r, oof = cv_run_for_seed(
                X_unb_K20, residual, w, anchor_unb, y_unb, unb_scaffolds, kf_seed,
            )
            per_seed_results.append({"kf_seed": int(kf_seed), "pooled_rae": r})
            all_oofs.append(oof)
        raes = np.array([s["pooled_rae"] for s in per_seed_results])
        mean_rae = float(raes.mean())
        std_rae = float(raes.std())
        mean_oof = np.mean(np.column_stack(all_oofs), axis=1)

        # deploy te refit
        te_deploy = deploy_refit_te(
            X_unb_K20, residual, w, X_te_K20, anchor_te,
        )

        rec = {
            "lambda": float(lam),
            "mean_rae": mean_rae,
            "std_rae": std_rae,
            "per_seed": per_seed_results,
            "weight_stats": diag,
            "wall_sec": round(time.time() - t_lam, 1),
        }
        per_lambda.append(rec)
        print(
            f"   lambda={lam:>4.2f}  mean_RAE={mean_rae:.4f} +/- {std_rae:.4f}  "
            f"w[min/mean/max]={diag['weight_min']:.3f}/{diag['weight_mean']:.3f}/"
            f"{diag['weight_max']:.3f}  wall={rec['wall_sec']:.1f}s"
        )

        if mean_rae < best_mean_rae:
            best_mean_rae = mean_rae
            best_lam = float(lam)
            best_pred_oof = mean_oof.astype(np.float32)
            best_te = te_deploy
            best_diagnostics = diag

    # ---- Gate ----
    if best_mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"

    te_unb_rae = float(rae(y_unb, best_te[unb_idx]))
    print("\n" + "-" * 78)
    print(f"BEST  lambda={best_lam}  mean_RAE={best_mean_rae:.4f}")
    print(f"  te[unb_idx] in-sample RAE = {te_unb_rae:.4f}")
    print(f"  verdict = {verdict}")
    print(
        f"  gates: PROMOTE<{GATE_PROMOTE}  MARGINAL_BEAT<{GATE_MARGINAL}"
    )
    print("-" * 78)

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", best_pred_oof)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", best_te)
    print(f"[save] {DATA_PROCESSED / (TAG + '_pred_oof.npy')}")
    print(f"[save] {DATA_PROCESSED / ('te_' + TAG + '.npy')}")

    summary = {
        "tag": TAG,
        "method": "per_row_entropy_reweight_lambda_sweep",
        "anchors_used": [a[0] for a in anchor_specs],
        "n_anchors": len(anchor_specs),
        "nb1191_included": nb1191_added,
        "indiv_oof_rae_unb": indiv_rae,
        "rae_anchor_chemprop_aux": rae_anchor,
        "lambda_grid": LAMBDA_GRID,
        "kf_seeds": KF_SEEDS,
        "resid_seeds": RESID_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "per_lambda_results": per_lambda,
        "best_lambda": best_lam,
        "best_mean_rae": best_mean_rae,
        "best_te_unb_rae_in_sample": te_unb_rae,
        "best_weight_stats": best_diagnostics,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "te_npy_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
        "pred_oof_path": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchors used                 = {[a[0] for a in anchor_specs]}")
    print(f"   best lambda                  = {best_lam}")
    print(f"   best mean_RAE (5 kf_seeds)   = {best_mean_rae:.4f}")
    print(f"   verdict                      = {verdict}")
    print(f"   wall                         = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "anchors_used",
        "indiv_oof_rae_unb",
        "best_lambda",
        "best_mean_rae",
        "best_te_unb_rae_in_sample",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
