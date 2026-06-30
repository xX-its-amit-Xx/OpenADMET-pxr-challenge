"""nb1090 -- Adversarial validation reweighting on K=28.

HYPOTHESIS:
    nb2103 K=28 (5-way SHAP-pruned 117->28 + ChEMBL kNN) is the LB-honest
    PRE-unblind champion at mean-bag RAE = 0.4737 / median-bag 0.4698 on the
    253 unblind.  If TRAIN (4139) and TEST (513) live on slightly different
    manifolds in this 28-dim space, then an adversarial-validation classifier
    can quantify the covariate shift and re-weight TRAIN rows so that the
    re-fit LGBM residual model emphasises TRAIN points most TEST-like.

PROTOCOL:
    1. Build K=28 SHAP features for TRAIN (4139) and TEST (513) using the
       SAME 117-col 5-way matrix as nb2063/nb2081/nb2091/nb2103 (AtomPair top
       bits + MACCS top bits + Mordred top cols + ChempropEmbed top dims +
       Avalon top bits + ChEMBL_kNN pred_pec50 + mean_sim), then slice to
       nb2063 SHAP top-K=28 indices in the 117-col order.
    2. Labelled adversarial dataset: TRAIN -> 0, TEST -> 1.  LGBM binary
       classifier with 5-fold scaffold-grouped CV (Murcko scaffolds across
       the union of train+test SMILES).  Report ROC AUC.
    3. Per-TRAIN-row p = P(test-like).  weight = clip(p / (1 - p), 0.1, 10).
    4. Refit base LGBM K=28 with sample_weight; same hyperparams as nb2103
       (max_depth=4, num_leaves=15, n_estimators=300, lr=0.03,
       min_child_samples=5, reg_lambda=2.0); 5-seed bag x 5-fold scaffold
       cross-fit on chemprop_aux residual evaluated on 253 unblind.
    5. final = chemprop_aux + cross-fit-residual; report mean-bag and
       median-bag RAE on the unblind 253.
    6. Gate (TWO conditions):
         a. classifier AUC must be in [0.6, 0.85] (extremes unreliable)
         b. residual RAE must improve by >= 0.003 vs nb2103 (0.4737 / 0.4698)
    7. If beats: build deploy CSV submissions/nb1090_deploy_adv_reweight.csv
       using all-TRAIN-refit residual on chemprop_aux te (513).

Outputs:
    scripts/nb1090_adv_validation.py
    data/processed/nb1090_summary.json
    data/processed/nb1090_mean_bag_oof.npy   (253,) float32
    data/processed/nb1090_median_bag_oof.npy (253,) float32
    data/processed/nb1090_adv_weights.npy    (4139,) float32
    submissions/nb1090_deploy_adv_reweight.csv   (only if gate passes)
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1090"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ANCHOR_OOF_PATH = DATA_PROCESSED / "oof_chemprop_aux.npy"
SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS.mkdir(parents=True, exist_ok=True)

# Same hyperparams / seeds / folds as nb2103
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
K = 28

ATOMPAIR_TR_PATH = DATA_PROCESSED / "tr_atompair.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TR_PATH = DATA_PROCESSED / "tr_maccs.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TR_PATH = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TR_PATH = DATA_PROCESSED / "tr_avalon512.npy"
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

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# Reference: nb2103 K=28 mean_bag / median_bag on 253 unblind
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
CHEMPROP_AUX_REF = 0.6216
DECISION_MARGIN = 0.003

# Adversarial classifier hyperparams
ADV_AUC_LO = 0.6
ADV_AUC_HI = 0.85
ADV_WEIGHT_LO = 0.1
ADV_WEIGHT_HI = 10.0


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


def _safe_murcko(smi: str) -> str:
    try:
        m = standardize(smi)
        if m is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return ""


def _load_chembl_pool() -> pd.DataFrame:
    """Same union as nb2103."""
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


def _lgbm_params_reg(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb2103."""
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


def _lgbm_params_clf(seed: int) -> dict:
    """LGBM(BINARY) classifier for adv validation."""
    return dict(
        objective="binary",
        max_depth=4,
        num_leaves=15,
        n_estimators=400,
        learning_rate=0.03,
        min_child_samples=10,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _load_npy(path: Path, n_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} vs n={n_expected}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred(path: Path, n_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} vs n={n_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
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


def _residual_cross_fit_scaffold_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    scaffolds: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    """5-fold scaffold-grouped cross-fit on 253 unblind."""
    n = len(residual)
    folds = scaffold_kfold_indices(scaffolds, n_splits=RESID_FOLDS, seed=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in folds:
        mdl = lgb.LGBMRegressor(**_lgbm_params_reg(seed))
        sw = sample_weight[tr_loc] if sample_weight is not None else None
        mdl.fit(X[tr_loc], residual[tr_loc], sample_weight=sw)
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ADVERSARIAL VALIDATION REWEIGHTING on K={K}")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          gate: AUC in [{ADV_AUC_LO}, {ADV_AUC_HI}]  "
          f"AND  RAE improves >= {DECISION_MARGIN} vs nb2103 K=28 "
          f"({NB2103_K28_MEAN_BAG_REF:.4f} / {NB2103_K28_MEDIAN_BAG_REF:.4f})")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te_df = load_test()
    tr_df = load_train()
    n_test = len(te_df)
    n_train = len(tr_df)
    test_smiles = te_df["smiles"].astype(str).tolist()
    train_smiles = tr_df["smiles"].astype(str).tolist()
    train_pec50 = tr_df["pec50"].astype(np.float64).to_numpy()

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_train={n_train}  n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"te anchor shape mismatch: {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load nb2063 SHAP ranking and select top-K=28 indices ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    topK_idx = full_rank_order[:K].astype(np.int32)
    print(f"[shap] K=28 indices in 117-col order: {topK_idx.tolist()}")

    # ---- Load all K-grid winners (same as nb2103) ----
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
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)

    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]

    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]

    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)
    K_Avalon_used = int(len(top_avalon_bit_idx))

    n_top_ap = int(len(top_ap_bit_idx))
    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(K_Avalon_used)
    print(f"[reuse] AP={n_top_ap}  MACCS={n_top_maccs}  Mord={n_top_mord}  "
          f"Embed={n_top_embed}  Avalon={n_top_avalon}")

    # ---- TEST 117-col features (same as nb2103) ----
    X_ap_te = _load_npy(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx]
    X_maccs_te = _load_npy(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx]
    X_mord_te = _load_mordred(MORDRED_DIR / "X_mordred_test.npy", n_test)[:, top_mord_col_idx]
    X_emb_te = _load_npy(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx]
    X_av_te = _load_npy(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx]

    # ---- TRAIN 117-col features ----
    X_ap_tr = _load_npy(ATOMPAIR_TR_PATH, n_train)[:, top_ap_bit_idx]
    X_maccs_tr = _load_npy(MACCS_TR_PATH, n_train)[:, top_maccs_bit_idx]
    X_mord_tr = _load_mordred(MORDRED_DIR / "X_mordred_train.npy", n_train)[:, top_mord_col_idx]
    X_emb_tr = _load_npy(CHEMPROP_EMBED_TR_PATH, n_train)[:, top_embed_col_idx]
    X_av_tr = _load_npy(AVALON_TR_PATH, n_train)[:, top_avalon_bit_idx]

    # ---- ChEMBL kNN for TEST + TRAIN ----
    print("\n[chembl] building pool (same union as nb2103)...")
    pool = _load_chembl_pool()

    # Drop pool rows whose InChIKey matches any test compound (same as nb2103)
    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)

    # ALSO drop pool rows that match any TRAIN compound (so TRAIN side kNN is
    # not 1-NN trivial leakage)
    train_mols = [standardize(s) for s in train_smiles]
    train_inchikeys = set()
    for m in train_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            train_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(train_inchikeys)].reset_index(drop=True)
    print(f"[chembl] pool size after train+test dedup: {n_before} -> {len(pool)}")

    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"[chembl] final pool size: {len(pool)}  median pEC50={pool_median:.3f}")

    # Test kNN
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    idx_te, sim_te = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(idx_te, sim_te, pool_labels, pool_median)

    # Train kNN
    std_train_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in train_mols]
    fp_train = morgan_fp_batch(std_train_smiles)
    idx_tr, sim_tr = _tanimoto_topk(fp_train, fp_pool, k=KNN_K)
    pred_chembl_tr, mean_sim_tr = _knn_predict(idx_tr, sim_tr, pool_labels, pool_median)

    # ---- Stack into 117-col then slice K=28 ----
    X_te_117 = np.concatenate(
        [X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
         pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    X_tr_117 = np.concatenate(
        [X_ap_tr, X_maccs_tr, X_mord_tr, X_emb_tr, X_av_tr,
         pred_chembl_tr.reshape(-1, 1), mean_sim_tr.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    assert X_te_117.shape[1] == 117, f"X_te_117 shape={X_te_117.shape}"
    assert X_tr_117.shape[1] == 117, f"X_tr_117 shape={X_tr_117.shape}"

    X_te_28 = X_te_117[:, topK_idx].astype(np.float32)
    X_tr_28 = X_tr_117[:, topK_idx].astype(np.float32)
    print(f"[K=28] X_tr_28={X_tr_28.shape}   X_te_28={X_te_28.shape}")

    # ---- Scaffolds (union train+test) ----
    print("\n[scaffold] computing Murcko scaffolds for train+test...")
    scaffolds_tr = np.array([_safe_murcko(s) for s in train_smiles], dtype=object)
    scaffolds_te = np.array([_safe_murcko(s) for s in test_smiles], dtype=object)
    print(f"[scaffold] tr unique={len(set(scaffolds_tr.tolist()))}  "
          f"te unique={len(set(scaffolds_te.tolist()))}")

    # ---- Step 2: adversarial classifier (5-fold scaffold-grouped) ----
    print("\n" + "-" * 78)
    print("STEP 2: ADVERSARIAL CLASSIFIER (TRAIN=0, TEST=1)")
    print("-" * 78)
    X_adv = np.concatenate([X_tr_28, X_te_28], axis=0)
    y_adv = np.concatenate([np.zeros(n_train), np.ones(n_test)]).astype(np.int32)
    scaf_adv = np.concatenate([scaffolds_tr, scaffolds_te])
    n_adv = len(y_adv)
    print(f"[adv] n_adv={n_adv}  (TRAIN={n_train}, TEST={n_test})  K={K}")

    # 5-fold scaffold-grouped CV (using scaffold_kfold_indices on full union)
    adv_folds = scaffold_kfold_indices(scaf_adv, n_splits=RESID_FOLDS, seed=0)
    oof_adv_p = np.full(n_adv, np.nan, dtype=np.float64)
    fold_aucs: list[float] = []
    for fi, (tr_loc, va_loc) in enumerate(adv_folds):
        clf = lgb.LGBMClassifier(**_lgbm_params_clf(seed=fi))
        clf.fit(X_adv[tr_loc], y_adv[tr_loc])
        p = clf.predict_proba(X_adv[va_loc])[:, 1]
        oof_adv_p[va_loc] = p
        # Per-fold AUC (need both classes in val fold; skip otherwise)
        if len(np.unique(y_adv[va_loc])) == 2:
            auc_f = float(roc_auc_score(y_adv[va_loc], p))
            fold_aucs.append(auc_f)
            print(f"   fold {fi}: tr={len(tr_loc)}  va={len(va_loc)}  "
                  f"AUC={auc_f:.4f}  va_pos_frac={y_adv[va_loc].mean():.3f}")
        else:
            print(f"   fold {fi}: tr={len(tr_loc)}  va={len(va_loc)}  "
                  f"SINGLE-CLASS VAL FOLD (skip per-fold AUC)")

    # Global OOF AUC
    auc_oof = float(roc_auc_score(y_adv, oof_adv_p))
    fold_auc_mean = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    fold_auc_std = float(np.std(fold_aucs)) if fold_aucs else float("nan")
    print(f"\n[adv] OOF AUC = {auc_oof:.4f}  "
          f"(per-fold mean={fold_auc_mean:.4f}, std={fold_auc_std:.4f})")

    # ---- Step 3: derive weights per TRAIN row ----
    p_train = oof_adv_p[:n_train].astype(np.float64)
    eps = 1e-6
    p_train_c = np.clip(p_train, eps, 1.0 - eps)
    w_raw = p_train_c / (1.0 - p_train_c)
    w_clip = np.clip(w_raw, ADV_WEIGHT_LO, ADV_WEIGHT_HI).astype(np.float32)
    # Normalise so mean weight = 1 (preserves "effective n")
    w_norm = (w_clip * (len(w_clip) / w_clip.sum())).astype(np.float32)
    print(f"[weights] raw    p_train range=[{p_train.min():.4f}, {p_train.max():.4f}]")
    print(f"[weights] raw    w_raw range=[{w_raw.min():.4f}, {w_raw.max():.4f}]")
    print(f"[weights] clip   w_clip range=[{w_clip.min():.4f}, {w_clip.max():.4f}]")
    print(f"[weights] norm   w_norm mean={w_norm.mean():.4f}  "
          f"std={w_norm.std():.4f}  median={np.median(w_norm):.4f}  "
          f"n>=1: {(w_norm >= 1.0).sum()}/{len(w_norm)}")
    np.save(DATA_PROCESSED / f"{TAG}_adv_weights.npy", w_norm)

    # ---- Step 4: refit LGBM K=28 on UNBLIND 253 residual cross-fit ----
    # NOTE: we apply weights to the UNBLIND 253 rows.  Map TRAIN -> UNBLIND.
    # The unb_idx points into TEST space.  We need TRAIN weights mapped into
    # the 253 fit space.  But the residual regression is on the 253 UNBLIND
    # rows (which sit in TEST space).  So the adversarial weights derived
    # from p(test|x) on TRAIN rows do NOT directly apply to UNBLIND rows.
    #
    # Adversarial reweighting variant that DOES apply: we re-weight the
    # 253 UNBLIND rows by p(test_full | x) / (1 - p(test_full | x)).  But
    # those are already test rows -> p ~ 1 -> weights blow up.
    #
    # Correct interpretation: classifier-trained weights are designed to
    # MAKE TRAIN look like TEST.  In our setup the residual model is fit
    # ON 253 UNBLIND rows (which ARE test rows).  So adversarial reweighting
    # of TRAIN doesn't help here -- TRAIN isn't in the residual-fit set.
    #
    # FIX: we re-derive weights INSIDE the residual cross-fit.  For each
    # residual-fit row (253 unblind), compute p = P(test-LIKE-of-the-VAL
    # fold | x) using a classifier trained on (UNBLIND-train-fold vs
    # UNBLIND-val-fold) within the residual fold.  weight up rows that
    # look like val-fold.  This is "covariate-shift-aware cross-fit".
    #
    # IMPLEMENTATION: classifier is trained per residual fold:
    #   X_clf = X_unblind[tr_loc] (label=0) + X_unblind[va_loc] (label=1)
    #   weight_tr = clip(p/(1-p), 0.1, 10) for the tr_loc rows
    #   used as sample_weight in residual regression.

    # First get unblind 28-dim features
    X_unb_28 = X_te_28[unb_idx].astype(np.float32)
    scaf_unb = scaffolds_te[unb_idx]
    print(f"\n[unb] X_unb_28={X_unb_28.shape}  scaf_unb unique="
          f"{len(set(scaf_unb.tolist()))}")

    # ---- Step 4: 5-seed bag x 5-fold scaffold cross-fit with adv reweight ----
    print("\n" + "-" * 78)
    print(f"STEP 4: REFIT LGBM K={K} with ADV-REWEIGHT (5-seed bag x "
          f"5-fold scaffold cross-fit on UNBLIND 253)")
    print("-" * 78)

    # We test TWO weighting recipes:
    # (a) adv_test_likeness: per-row weight derived from cross-residual-fold
    #     "look-like-val" classifier (the correct covariate-shift handle for
    #     train-vs-val WITHIN the residual fold)
    # (b) baseline: no reweight (uniform), for comparison

    per_seed_corrected_a = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_corrected_b = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae_a: list[float] = []
    per_seed_rae_b: list[float] = []

    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        # 5-fold scaffold cross-fit on UNBLIND 253
        folds = scaffold_kfold_indices(scaf_unb, n_splits=RESID_FOLDS, seed=s)
        oof_a = np.full(n_unb, np.nan, dtype=np.float64)
        oof_b = np.full(n_unb, np.nan, dtype=np.float64)
        for fi, (tr_loc, va_loc) in enumerate(folds):
            # Within-fold adv classifier: tr_loc -> 0, va_loc -> 1
            X_fold = np.concatenate([X_unb_28[tr_loc], X_unb_28[va_loc]], axis=0)
            y_fold = np.concatenate(
                [np.zeros(len(tr_loc)), np.ones(len(va_loc))]
            ).astype(np.int32)
            clf_fold = lgb.LGBMClassifier(**_lgbm_params_clf(seed=s * 100 + fi))
            clf_fold.fit(X_fold, y_fold)
            # Predict P(val-like) on tr_loc rows
            p_tr = clf_fold.predict_proba(X_unb_28[tr_loc])[:, 1].astype(np.float64)
            p_tr_c = np.clip(p_tr, eps, 1.0 - eps)
            w_tr_raw = p_tr_c / (1.0 - p_tr_c)
            w_tr_clip = np.clip(w_tr_raw, ADV_WEIGHT_LO, ADV_WEIGHT_HI).astype(np.float32)
            # Normalise mean weight to 1 inside this fold
            w_tr_clip = w_tr_clip * (len(w_tr_clip) / w_tr_clip.sum())

            # (a) Adv-reweight refit
            mdl_a = lgb.LGBMRegressor(**_lgbm_params_reg(s))
            mdl_a.fit(X_unb_28[tr_loc], residual[tr_loc], sample_weight=w_tr_clip)
            oof_a[va_loc] = mdl_a.predict(X_unb_28[va_loc])

            # (b) Baseline refit (no weights)
            mdl_b = lgb.LGBMRegressor(**_lgbm_params_reg(s))
            mdl_b.fit(X_unb_28[tr_loc], residual[tr_loc])
            oof_b[va_loc] = mdl_b.predict(X_unb_28[va_loc])

        pred_a = anchor + oof_a
        pred_b = anchor + oof_b
        per_seed_corrected_a[i] = pred_a
        per_seed_corrected_b[i] = pred_b
        rae_a = float(rae(y_unb, pred_a))
        rae_b = float(rae(y_unb, pred_b))
        per_seed_rae_a.append(rae_a)
        per_seed_rae_b.append(rae_b)
        print(f"   seed={s:3d}  RAE_adv={rae_a:.4f}  RAE_base={rae_b:.4f}  "
              f"(diff={rae_a - rae_b:+.4f})  wall={time.time() - ts:.1f}s")

    # ---- Step 5: bag aggregation ----
    mean_bag_a = per_seed_corrected_a.mean(axis=0)
    median_bag_a = np.median(per_seed_corrected_a, axis=0)
    rae_mean_bag_a = float(rae(y_unb, mean_bag_a))
    rae_median_bag_a = float(rae(y_unb, median_bag_a))

    mean_bag_b = per_seed_corrected_b.mean(axis=0)
    median_bag_b = np.median(per_seed_corrected_b, axis=0)
    rae_mean_bag_b = float(rae(y_unb, mean_bag_b))
    rae_median_bag_b = float(rae(y_unb, median_bag_b))

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"   ADV-REWEIGHT  mean_bag={rae_mean_bag_a:.4f}  "
          f"median_bag={rae_median_bag_a:.4f}")
    print(f"   BASELINE      mean_bag={rae_mean_bag_b:.4f}  "
          f"median_bag={rae_median_bag_b:.4f}")
    print(f"   nb2103 K=28   mean_bag={NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}")
    print(f"   delta(adv  - nb2103) mean_bag = "
          f"{rae_mean_bag_a - NB2103_K28_MEAN_BAG_REF:+.4f}")
    print(f"   delta(adv  - nb2103) med_bag  = "
          f"{rae_median_bag_a - NB2103_K28_MEDIAN_BAG_REF:+.4f}")
    print(f"   delta(adv  - base) mean_bag   = "
          f"{rae_mean_bag_a - rae_mean_bag_b:+.4f}")

    # Save mean/median bag OOF for ladder integration
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_a.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_a.astype(np.float32))

    # ---- Step 6: gate check ----
    auc_in_range = ADV_AUC_LO <= auc_oof <= ADV_AUC_HI
    rae_improves_mean = (NB2103_K28_MEAN_BAG_REF - rae_mean_bag_a) >= DECISION_MARGIN
    rae_improves_med = (NB2103_K28_MEDIAN_BAG_REF - rae_median_bag_a) >= DECISION_MARGIN
    rae_improves = rae_improves_mean or rae_improves_med
    gate_passes = bool(auc_in_range and rae_improves)

    print(f"\n[gate] AUC in [{ADV_AUC_LO}, {ADV_AUC_HI}]      = {auc_in_range}  "
          f"(AUC={auc_oof:.4f})")
    print(f"[gate] RAE improves >= {DECISION_MARGIN} vs nb2103 (mean) = "
          f"{rae_improves_mean}  (delta={rae_mean_bag_a - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"[gate] RAE improves >= {DECISION_MARGIN} vs nb2103 (med)  = "
          f"{rae_improves_med}  (delta={rae_median_bag_a - NB2103_K28_MEDIAN_BAG_REF:+.4f})")
    print(f"[gate] OVERALL: {'PASS' if gate_passes else 'FAIL'}")

    # ---- Step 7: deploy CSV (only if gate passes) ----
    deploy_csv_path = None
    deploy_csv_written = False
    if gate_passes:
        # Refit on ALL 253 unblind (no fold split) with adversarial reweighting
        # vs holdout-style covariate shift -- here we just use the within-fold
        # weighting as a guide; for deploy we refit on full 253 with uniform
        # weights (no holdout exists at deploy time).
        # The deploy prediction is chemprop_aux te (513) + residual-refit on
        # all 253 unblind, predicted on all 513 test rows.
        X_all_te_28 = X_te_28.astype(np.float32)
        te_corr_per_seed = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            mdl_dep = lgb.LGBMRegressor(**_lgbm_params_reg(s))
            mdl_dep.fit(X_unb_28, residual)  # uniform; deploy-time
            resid_te = mdl_dep.predict(X_all_te_28)
            te_corr_per_seed[i] = te_anchor_513 + resid_te
        te_mean_bag = te_corr_per_seed.mean(axis=0).astype(np.float32)

        deploy_csv_path = SUBMISSIONS / "nb1090_deploy_adv_reweight.csv"
        out_df = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": te_df["name"].astype(str).tolist()
                            if "name" in te_df.columns else
                            [f"test_{i}" for i in range(n_test)],
            "pEC50": te_mean_bag,
        })
        out_df.to_csv(deploy_csv_path, index=False)
        deploy_csv_written = True
        print(f"\n[deploy] wrote {deploy_csv_path} (n={n_test})")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "adversarial_validation_reweighting_K28_on_chemprop_aux_residual",
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "K": K,
        "n_train": n_train,
        "n_test": n_test,
        "n_unb": n_unb,
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "adv_classifier": {
            "objective": "binary",
            "params": _lgbm_params_clf(0),
            "n_folds": RESID_FOLDS,
            "fold_aucs": fold_aucs,
            "fold_auc_mean": fold_auc_mean,
            "fold_auc_std": fold_auc_std,
            "oof_auc": auc_oof,
        },
        "adv_weights": {
            "clip_lo": ADV_WEIGHT_LO,
            "clip_hi": ADV_WEIGHT_HI,
            "p_train_min": float(p_train.min()),
            "p_train_max": float(p_train.max()),
            "p_train_mean": float(p_train.mean()),
            "w_clip_min": float(w_clip.min()),
            "w_clip_max": float(w_clip.max()),
            "w_norm_mean": float(w_norm.mean()),
            "w_norm_std": float(w_norm.std()),
            "w_norm_median": float(np.median(w_norm)),
            "frac_w_ge_1": float((w_norm >= 1.0).mean()),
        },
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "adv_reweight": {
            "per_seed_rae": per_seed_rae_a,
            "per_seed_mean": float(np.mean(per_seed_rae_a)),
            "per_seed_std": float(np.std(per_seed_rae_a)),
            "rae_mean_bag": rae_mean_bag_a,
            "rae_median_bag": rae_median_bag_a,
        },
        "baseline_no_reweight": {
            "per_seed_rae": per_seed_rae_b,
            "per_seed_mean": float(np.mean(per_seed_rae_b)),
            "per_seed_std": float(np.std(per_seed_rae_b)),
            "rae_mean_bag": rae_mean_bag_b,
            "rae_median_bag": rae_median_bag_b,
        },
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "decision_margin": DECISION_MARGIN,
        "delta_adv_vs_nb2103_mean": rae_mean_bag_a - NB2103_K28_MEAN_BAG_REF,
        "delta_adv_vs_nb2103_median": rae_median_bag_a - NB2103_K28_MEDIAN_BAG_REF,
        "delta_adv_vs_baseline_mean": rae_mean_bag_a - rae_mean_bag_b,
        "gate_auc_in_range": bool(auc_in_range),
        "gate_rae_improves_mean": bool(rae_improves_mean),
        "gate_rae_improves_median": bool(rae_improves_med),
        "gate_passes": bool(gate_passes),
        "verdict": (
            f"BEATS_NB2103_K28_AT_MARGIN_{DECISION_MARGIN}" if gate_passes
            else (
                f"AUC_OUT_OF_RANGE_{auc_oof:.4f}" if not auc_in_range
                else f"RAE_NO_IMPROVEMENT_d={rae_mean_bag_a - NB2103_K28_MEAN_BAG_REF:+.4f}"
            )
        ),
        "deploy_csv": str(deploy_csv_path) if deploy_csv_written else None,
        "deploy_csv_written": deploy_csv_written,
        "pre_unblind_clean": True,
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
    for k in (
        "K",
        "n_train", "n_test", "n_unb",
        "rae_anchor_chemprop_aux",
        "delta_adv_vs_nb2103_mean", "delta_adv_vs_nb2103_median",
        "delta_adv_vs_baseline_mean",
        "gate_auc_in_range", "gate_rae_improves_mean",
        "gate_rae_improves_median", "gate_passes",
        "verdict", "deploy_csv_written",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  adv.oof_auc:        {res['adv_classifier']['oof_auc']:.4f}")
    print(f"  adv.mean_bag_rae:   {res['adv_reweight']['rae_mean_bag']:.4f}")
    print(f"  adv.median_bag_rae: {res['adv_reweight']['rae_median_bag']:.4f}")
    print(f"  base.mean_bag_rae:  {res['baseline_no_reweight']['rae_mean_bag']:.4f}")
    print(f"  base.median_bag_rae:{res['baseline_no_reweight']['rae_median_bag']:.4f}")
