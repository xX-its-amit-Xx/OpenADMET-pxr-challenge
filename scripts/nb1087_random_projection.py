"""nb1087 -- Random Gaussian Projection LGBM bag (K=28 SHAP -> K_proj).

HYPOTHESIS:
    Project K=28 SHAP top features -> K_proj via Gaussian random matrix,
    train diverse LGBM(MSE) base learners on each projection, mean-aggregate.
    Random projections approximately preserve pairwise distances (JL lemma)
    while injecting diversity through orthogonal subspace selection.
    Tests three modes:
        (a) K=28 -> K=14 (compression, JL-faithful for moderate distortion)
        (b) K=14 -> K=7  (further compression of K=14 grid)
        (c) K=14 -> K=28 (expansion / sparse-overcomplete-like)
    Reference: nb2103 K=28 mean_bag = 0.4737, median_bag = 0.4698.
    Decision margin = 0.003.

PROTOCOL:
    1. Reuse top-28 SHAP indices from nb2103_summary.json (K=28 record),
       rebuild the same 117-col 5-way K-tuned feature matrix (nb2063/nb2081/
       nb2091/nb2103 lineage), slice to top-28.
    2. Standardize features (z-score on the 253 unblind).
    3. For each (mode, K_in, K_out, seed_proj in {0,1,7,42,137}):
         a. Generate Gaussian projection matrix P in R^{K_in x K_out}
            with entries N(0, 1/K_out).
         b. Project X (253 x K_in) -> X_proj (253 x K_out).
         c. 5-seed inner bag (seeds {0,1,7,42,137}) of LGBM(MSE) with
            5-fold cross-fit per seed -> 5 cross-fit RAE values per proj.
         d. Mean-aggregate 5 cross-fit OOF vectors -> 1 OOF per projection.
    4. Mean-aggregate the 5 projection OOFs -> final OOF per mode.
    5. Compare each mode's mean-bag RAE vs nb2103 K=28 (0.4737/0.4698) at
       decision_margin = 0.003.
    6. If best mode beats nb2103 K=28: deploy 25-fit median CSV using same
       projection scheme on 513 test rows.

Outputs:
    scripts/nb1087_random_projection.py
    data/processed/nb1087_summary.json
    data/processed/nb1087_mean_bag_oof_mode_{a,b,c}.npy   (253,) float32 per mode
    submissions/nb1087_deploy_proj14.csv (if best mode beats threshold)
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

TAG = "nb1087"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
PROJ_SEEDS = [0, 1, 7, 42, 137]
TOP_K_SHAP = 28

# Deploy bag (only if winning mode beats threshold)
DEPLOY_OUTER = [0, 1, 7, 42, 137]
DEPLOY_INNER = [0, 1, 7, 42, 137]

# Three modes: (mode, K_in, K_out, label)
MODES = [
    ("a", 28, 14, "K28_to_K14"),
    ("b", 14, 7,  "K14_to_K7"),
    ("c", 14, 28, "K14_to_K28"),
]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

# nb2103 K=28 anchor (PRE-unblind honest cross-fit)
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.003
CHEMPROP_AUX_REF = 0.6216


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
    """Same union as nb2063/nb2081/nb2091/nb2103/nb2112."""
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
    """LGBM(MSE) per nb1087 spec: L=15 lr=0.03 mc=5 lambda=2 n_est=300."""
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


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred shape mismatch: {X_te_m.shape}")
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
    raise KeyError("AtomPair entry not found")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


def _extract_K_record(sum_dict: dict, records_key: str, K: int) -> dict:
    for r in sum_dict[records_key]:
        if int(r["K"]) == K:
            return r
    raise KeyError(f"K={K} not found")


def _standardize_zscore(X_unb: np.ndarray, X_te: np.ndarray):
    """Z-score on the 253 unblind, apply same shift/scale to 513 test."""
    mu = X_unb.mean(axis=0)
    sd = X_unb.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    X_unb_z = ((X_unb - mu) / sd).astype(np.float32)
    X_te_z = ((X_te - mu) / sd).astype(np.float32)
    return X_unb_z, X_te_z, mu.astype(np.float32), sd.astype(np.float32)


def _gen_gaussian_proj(K_in: int, K_out: int, seed: int) -> np.ndarray:
    """Gaussian random projection matrix P in R^{K_in x K_out}, entries N(0, 1/K_out)."""
    rng = np.random.default_rng(seed)
    P = rng.normal(loc=0.0, scale=1.0 / np.sqrt(K_out),
                   size=(K_in, K_out)).astype(np.float32)
    return P


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Random Gaussian Projection LGBM bag")
    print(f"          modes={MODES}  proj_seeds={PROJ_SEEDS}  "
          f"inner_seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG_REF}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF}  "
          f"margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load nb2103 top-28 SHAP indices ----
    if not NB2103_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2103_SUMMARY}")
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    rec28 = _extract_K_record(nb2103_sum, "per_K_records", K=TOP_K_SHAP)
    top28_idx = np.array(rec28["top_K_idx_in_117"], dtype=np.int32)
    if top28_idx.shape[0] != TOP_K_SHAP:
        raise ValueError(f"top28_idx len {top28_idx.shape[0]} != {TOP_K_SHAP}")
    nb2103_k28_mean_bag = float(rec28["rae_mean_bag"])
    nb2103_k28_median_bag = float(rec28["rae_median_bag"])
    print(f"[reuse] nb2103 K=28 mean_bag={nb2103_k28_mean_bag:.6f}  "
          f"median_bag={nb2103_k28_median_bag:.6f}")

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "name" in te.columns:
        mol_names = te["name"].astype(str).tolist()
    else:
        raise KeyError("no name column")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"{ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape {te_anchor_513.shape}")
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual_unb = y_unb - anchor_unb
    print(f"[resid] mean={residual_unb.mean():+.4f}  "
          f"std={residual_unb.std():.4f}")

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

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    # ---- Feature matrices (513 + unb slice) ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

    # ---- ChEMBL kNN ----
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
    std_test_smiles = [
        Chem.MolToSmiles(m) if m is not None else "" for m in test_mols
    ]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    print(f"   final ChEMBL pool size: {len(pool)}  "
          f"median pEC50 = {pool_median:.3f}")

    # ---- Stack full 117-col matrices ----
    X_te_117 = np.concatenate(
        [X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
         pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    X_unb_117 = X_te_117[unb_idx].astype(np.float32)
    feat_dim_full = X_te_117.shape[1]
    print(f"\n[stack] X_te_117={X_te_117.shape}  X_unb_117={X_unb_117.shape}")

    # ---- Slice to top-28 ----
    X_te_28 = X_te_117[:, top28_idx].astype(np.float32)
    X_unb_28 = X_unb_117[:, top28_idx].astype(np.float32)
    print(f"[slice top-28] X_te_28={X_te_28.shape}  X_unb_28={X_unb_28.shape}")

    # ---- Standardize on 253 unblind ----
    X_unb_28_z, X_te_28_z, mu28, sd28 = _standardize_zscore(X_unb_28, X_te_28)
    print(f"[std] X_unb_28_z mean={X_unb_28_z.mean():+.4f}  "
          f"std={X_unb_28_z.std():.4f}")

    # For mode (b) and (c), K_in=14: project K=28 -> K=14 via fixed seed=0
    # gaussian projection, then use that K=14 representation as the input.
    # This makes K=14 -> K=7 and K=14 -> K=28 meaningful.
    P_28_to_14_anchor = _gen_gaussian_proj(28, 14, seed=12345)
    X_unb_14_z = (X_unb_28_z @ P_28_to_14_anchor).astype(np.float32)
    X_te_14_z = (X_te_28_z @ P_28_to_14_anchor).astype(np.float32)
    # re-z-score the K=14 anchor (so its scale is unit again)
    mu14 = X_unb_14_z.mean(axis=0)
    sd14 = X_unb_14_z.std(axis=0)
    sd14 = np.where(sd14 < 1e-8, 1.0, sd14)
    X_unb_14_z = ((X_unb_14_z - mu14) / sd14).astype(np.float32)
    X_te_14_z = ((X_te_14_z - mu14) / sd14).astype(np.float32)
    print(f"[anchor K=14] X_unb_14_z={X_unb_14_z.shape}  "
          f"std={X_unb_14_z.std():.4f}")

    # ---- Per-mode sweep ----
    print("\n" + "-" * 78)
    print("MODE SWEEP")
    print("-" * 78)
    per_mode_results: list[dict] = []
    per_mode_oof: dict[str, np.ndarray] = {}

    for mode, K_in, K_out, label in MODES:
        print(f"\n=== mode={mode} ({label}: K_in={K_in} -> K_out={K_out}) ===")
        if K_in == 28:
            X_src_unb = X_unb_28_z
            X_src_te = X_te_28_z
        elif K_in == 14:
            X_src_unb = X_unb_14_z
            X_src_te = X_te_14_z
        else:
            raise ValueError(f"unsupported K_in={K_in}")

        per_proj_oof = np.zeros((len(PROJ_SEEDS), n_unb), dtype=np.float64)
        per_proj_rae_inner_mean: list[float] = []
        per_proj_records = []
        for p_i, p_seed in enumerate(PROJ_SEEDS):
            t_p = time.time()
            P = _gen_gaussian_proj(K_in, K_out, seed=p_seed)
            X_proj_unb = (X_src_unb @ P).astype(np.float32)
            per_inner_oof = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
            per_inner_rae = []
            for i_idx, s in enumerate(RESID_SEEDS):
                oof_s = _residual_cross_fit_one_seed(X_proj_unb, residual_unb, s)
                pred_corr_s = anchor_unb + oof_s
                rae_s = float(rae(y_unb, pred_corr_s))
                per_inner_oof[i_idx] = pred_corr_s
                per_inner_rae.append(rae_s)
            mean_inner_oof = per_inner_oof.mean(axis=0)
            rae_mean_inner = float(rae(y_unb, mean_inner_oof))
            per_proj_oof[p_i] = mean_inner_oof
            per_proj_rae_inner_mean.append(rae_mean_inner)
            per_proj_records.append({
                "proj_seed": int(p_seed),
                "per_inner_rae": [float(r) for r in per_inner_rae],
                "per_inner_rae_mean": float(np.mean(per_inner_rae)),
                "per_inner_rae_std": float(np.std(per_inner_rae)),
                "rae_inner_mean_bag": rae_mean_inner,
                "wall_sec": round(time.time() - t_p, 2),
            })
            print(f"   proj_seed={p_seed:3d}  inner_seeds RAE = "
                  f"[{', '.join(f'{r:.4f}' for r in per_inner_rae)}]  "
                  f"inner_mean_bag={rae_mean_inner:.4f}  "
                  f"wall={time.time() - t_p:.1f}s")

        # Mean-aggregate across 5 projections
        mean_bag_oof = per_proj_oof.mean(axis=0)
        median_bag_oof = np.median(per_proj_oof, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        rae_median_bag = float(rae(y_unb, median_bag_oof))

        delta_vs_nb2103_mean = rae_mean_bag - nb2103_k28_mean_bag
        delta_vs_nb2103_median = rae_mean_bag - nb2103_k28_median_bag
        beats_nb2103_mean = rae_mean_bag < nb2103_k28_mean_bag - DECISION_MARGIN
        beats_nb2103_median = rae_mean_bag < nb2103_k28_median_bag - DECISION_MARGIN
        flat_vs_nb2103_mean = abs(delta_vs_nb2103_mean) < DECISION_MARGIN

        if beats_nb2103_median:
            verdict_mode = "BEATS_NB2103_K28_MEDIAN"
        elif beats_nb2103_mean:
            verdict_mode = "BEATS_NB2103_K28_MEAN_ONLY"
        elif flat_vs_nb2103_mean:
            verdict_mode = "FLAT_VS_NB2103_K28"
        else:
            verdict_mode = "WORSE_THAN_NB2103_K28"

        print(f"   mode={mode} mean_bag (over 5 projs) = {rae_mean_bag:.4f}  "
              f"(d_vs_K28_mean={delta_vs_nb2103_mean:+.4f}  "
              f"d_vs_K28_median={delta_vs_nb2103_median:+.4f})")
        print(f"   mode={mode} median_bag (over 5 projs) = {rae_median_bag:.4f}")
        print(f"   mode={mode} verdict = {verdict_mode}")

        out_p = DATA_PROCESSED / f"{TAG}_mean_bag_oof_mode_{mode}.npy"
        np.save(out_p, mean_bag_oof.astype(np.float32))
        per_mode_oof[mode] = mean_bag_oof.astype(np.float32)

        per_mode_results.append({
            "mode": mode,
            "label": label,
            "K_in": int(K_in),
            "K_out": int(K_out),
            "per_proj_records": per_proj_records,
            "per_proj_rae_inner_mean_bag": per_proj_rae_inner_mean,
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_vs_nb2103_K28_mean": delta_vs_nb2103_mean,
            "delta_vs_nb2103_K28_median": delta_vs_nb2103_median,
            "beats_nb2103_K28_mean": bool(beats_nb2103_mean),
            "beats_nb2103_K28_median": bool(beats_nb2103_median),
            "flat_vs_nb2103_K28_mean": bool(flat_vs_nb2103_mean),
            "verdict": verdict_mode,
            "oof_path": str(out_p),
        })

    # ---- Choose best mode ----
    print("\n" + "=" * 78)
    print("MODE SUMMARY TABLE")
    print("=" * 78)
    print(f"   {'mode':>4s}  {'label':>14s}  {'K_in':>4s}  {'K_out':>5s}  "
          f"{'mean_bag':>9s}  {'median_bag':>10s}  "
          f"{'d_vs_K28_mean':>14s}  verdict")
    print(f"   {'K28':>4s}  {'nb2103 baseline':>14s}  {'-':>4s}  {'-':>5s}  "
          f"{nb2103_k28_mean_bag:>9.4f}  {nb2103_k28_median_bag:>10.4f}  "
          f"{0.0:>+14.4f}  BASELINE")
    for r in per_mode_results:
        print(f"   {r['mode']:>4s}  {r['label']:>14s}  {r['K_in']:>4d}  "
              f"{r['K_out']:>5d}  {r['rae_mean_bag']:>9.4f}  "
              f"{r['rae_median_bag']:>10.4f}  "
              f"{r['delta_vs_nb2103_K28_mean']:>+14.4f}  {r['verdict']}")

    sweep_rae = [r["rae_mean_bag"] for r in per_mode_results]
    best_i = int(np.argmin(sweep_rae))
    best_mode_rec = per_mode_results[best_i]
    best_mode = best_mode_rec["mode"]
    best_mode_rae = best_mode_rec["rae_mean_bag"]
    best_mode_rae_median = best_mode_rec["rae_median_bag"]

    beats_threshold = best_mode_rae < nb2103_k28_median_bag - DECISION_MARGIN
    print(f"\n   best mode in sweep      = {best_mode_rec['label']}")
    print(f"   best mode mean_bag RAE  = {best_mode_rae:.4f}")
    print(f"   best mode median_bag RAE = {best_mode_rae_median:.4f}")
    print(f"   beats nb2103 K=28 median = {beats_threshold}")

    # ---- Deploy 25-fit median CSV if best mode beats threshold ----
    deploy_path = None
    deploy_te_path = None
    deploy_rae_in_unb = None
    if beats_threshold:
        print("\n" + "-" * 78)
        print(f"DEPLOY: best mode = {best_mode_rec['label']} "
              f"(mean_bag {best_mode_rae:.4f} < threshold "
              f"{nb2103_k28_median_bag - DECISION_MARGIN:.4f})")
        print("-" * 78)
        K_in = best_mode_rec["K_in"]
        K_out = best_mode_rec["K_out"]
        if K_in == 28:
            X_dep_unb = X_unb_28_z
            X_dep_te = X_te_28_z
        else:
            X_dep_unb = X_unb_14_z
            X_dep_te = X_te_14_z

        all_resid_513 = np.zeros((len(PROJ_SEEDS) * len(DEPLOY_INNER), n_test),
                                 dtype=np.float64)
        k_global = 0
        for p_seed in PROJ_SEEDS:
            P = _gen_gaussian_proj(K_in, K_out, seed=p_seed)
            X_proj_unb_dep = (X_dep_unb @ P).astype(np.float32)
            X_proj_te_dep = (X_dep_te @ P).astype(np.float32)
            for s in DEPLOY_INNER:
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_proj_unb_dep, residual_unb)
                resid_513 = mdl.predict(X_proj_te_dep)
                all_resid_513[k_global] = resid_513
                k_global += 1
        median_resid_513 = np.median(all_resid_513, axis=0)
        te_deploy = te_anchor_513 + median_resid_513

        # In-sample diag
        deploy_rae_in_unb = float(rae(y_unb, te_deploy[unb_idx]))
        print(f"in-sample RAE on unb_idx (deploy median) = "
              f"{deploy_rae_in_unb:.4f}  (anchor {rae_anchor:.4f})")

        df_sub = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_deploy.astype(np.float32),
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)}")
        sub_path = SUBMISSIONS_DIR / f"{TAG}_deploy_proj14.csv"
        df_sub.to_csv(sub_path, index=False)
        deploy_path = str(sub_path)
        te_p = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_p, te_deploy.astype(np.float32))
        deploy_te_path = str(te_p)
        print(f"[save] submission CSV: {sub_path}")
        print(f"[save] te artifact:    {te_p}")

    # ---- Global verdict ----
    if beats_threshold:
        global_verdict = (
            f"RANDOM_PROJECTION_BEATS_NB2103_K28_MEDIAN_BEST_MODE={best_mode}"
        )
    elif best_mode_rae < nb2103_k28_mean_bag - DECISION_MARGIN:
        global_verdict = (
            f"RANDOM_PROJECTION_BEATS_NB2103_K28_MEAN_ONLY_BEST_MODE={best_mode}"
        )
    elif abs(best_mode_rae - nb2103_k28_mean_bag) < DECISION_MARGIN:
        global_verdict = (
            f"RANDOM_PROJECTION_FLAT_VS_NB2103_K28_BEST_MODE={best_mode}"
        )
    else:
        global_verdict = "RANDOM_PROJECTION_DOES_NOT_BEAT_NB2103_K28"
    print(f"\n   global verdict = {global_verdict}")

    summary = {
        "tag": TAG,
        "method": "random_gaussian_projection_lgbm_bag_K28_K14_K7_K28",
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("nb2103 top-28 SHAP indices on 117-col 5-way K-tuned "
                        "matrix (AtomPair/MACCS/Mordred/ChempropEmbed/Avalon "
                        "+ ChEMBL kNN); standardized z-score on 253 unblind"),
        "modes": [
            {"mode": m, "K_in": ki, "K_out": ko, "label": lbl}
            for (m, ki, ko, lbl) in MODES
        ],
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "proj_seeds": PROJ_SEEDS,
        "inner_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "top_k_shap": TOP_K_SHAP,
        "top28_idx_in_117_from_nb2103": top28_idx.tolist(),
        "feat_dim_full": int(feat_dim_full),
        "anchor_K28_to_K14_proj_seed": 12345,
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "n_chembl_pool": int(len(pool)),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual_unb.mean()),
        "residual_std": float(residual_unb.std()),
        "nb2103_K28_mean_bag_ref": nb2103_k28_mean_bag,
        "nb2103_K28_median_bag_ref": nb2103_k28_median_bag,
        "per_mode_records": per_mode_results,
        "best_mode": best_mode,
        "best_mode_label": best_mode_rec["label"],
        "best_mode_rae_mean_bag": best_mode_rae,
        "best_mode_rae_median_bag": best_mode_rae_median,
        "best_mode_delta_vs_nb2103_K28_mean":
            best_mode_rae - nb2103_k28_mean_bag,
        "best_mode_delta_vs_nb2103_K28_median":
            best_mode_rae - nb2103_k28_median_bag,
        "beats_nb2103_K28_median_at_margin": bool(beats_threshold),
        "deploy_csv": deploy_path,
        "deploy_te_path": deploy_te_path,
        "deploy_in_RAE_unb": deploy_rae_in_unb,
        "verdict": global_verdict,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_mean_bag_anchor": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_anchor": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
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
        "modes",
        "n_chembl_pool",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_ref",
        "nb2103_K28_median_bag_ref",
        "best_mode", "best_mode_label",
        "best_mode_rae_mean_bag",
        "best_mode_rae_median_bag",
        "best_mode_delta_vs_nb2103_K28_mean",
        "best_mode_delta_vs_nb2103_K28_median",
        "beats_nb2103_K28_median_at_margin",
        "deploy_csv",
        "deploy_in_RAE_unb",
        "verdict",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-MODE TABLE ====")
    for r in res["per_mode_records"]:
        print(f"  mode={r['mode']:>2s} ({r['label']:>14s}: "
              f"K_in={r['K_in']:>2d} -> K_out={r['K_out']:>2d})  "
              f"mean_bag={r['rae_mean_bag']:.4f}  "
              f"median_bag={r['rae_median_bag']:.4f}  "
              f"d_vs_K28_mean={r['delta_vs_nb2103_K28_mean']:+.4f}  "
              f"{r['verdict']}")
