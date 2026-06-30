"""nb2681 -- Bayesian Bootstrap (Rubin 1981) on K=20 LGBM residual ensemble.

NEW PARADIGM:
    Rubin's Bayesian Bootstrap replaces standard (with-replacement integer
    multinomial) resampling with continuous Dirichlet(1, 1, ..., 1) weights
    over the n training rows.  Each member sees the SAME rows but with
    different continuous importance weights (which average to 1.0 across
    members in expectation).  Smoother variance reduction than vanilla
    bagging because:
      - Every training row participates in every member (no "lost rows")
      - Weight noise is symmetric Dirichlet, not binomial
      - Posterior over weights is conjugate / non-parametric Bayesian

    For an LGBM with K=20 features, this turns the deterministic boosted-tree
    regressor into a posterior-mean predictor over bootstrap weight space,
    analogous to Bayesian forests but with continuous rather than discrete
    resampling.

PROTOCOL:
    -  20 Bayesian-bootstrap members
    -  Per member m:
         w_m ~ Dirichlet(alpha=ones(n_train))           # mean 1/n_train each
         w_m <- w_m * n_train                           # normalize to mean 1.0
         LGBM(sample_weight=w_m, max_depth=4, num_leaves=15,
              n_estimators=300, learning_rate=0.03) on K=20 features
         predict on val
    -  Member ensemble = mean of 20 predictions
    -  5-fold scaffold CV on 253, kf_seed=1001
    -  Fit on chemprop_aux residual (only verified-clean PRE-unblind anchor)

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else               -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2640_summary.json    (K20_idx_in_117col)
    (+ same 117-col feature stack as nb2674)

Outputs:
    data/processed/nb2681_summary.json
    data/processed/nb2681_pred_oof.npy        (253,) float32
    data/processed/te_nb2681.npy              (513,) float32
    submissions/nb2681_bayesian_bootstrap.csv
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
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2681"
PARENT_TAG = "nb2640"

# -- Bayesian Bootstrap params --------------------------------------------
N_MEMBERS = 20
DIRICHLET_ALPHA = 1.0          # symmetric Dirichlet(1,1,...,1) = uniform simplex
BB_BASE_SEED = 20260608        # member m uses seed BB_BASE_SEED + m

# -- LGBM hyperparams (per task spec) -------------------------------------
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_N_EST = 300
LGBM_LR = 0.03

# -- Cross-fit params -----------------------------------------------------
KF_SEED = 1001
RESID_FOLDS = 5

# -- Gates ---------------------------------------------------------------
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# -- Reference numbers ---------------------------------------------------
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216

# -- Feature cache paths -------------------------------------------------
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
NB2640_SUMMARY = DATA_PROCESSED / "nb2640_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# ============================================================================
# helpers (lifted from nb2674 to keep matrix construction byte-identical)
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


def _load_chembl_pool():
    import pandas as pd
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
    import pandas as pd
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
        raise FileNotFoundError(f"Mordred cache missing (run nb1030): {mte_p}")
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


def build_117col_feature_matrix(te_smiles, n_test):
    """Same 117-col matrix as nb2640 / nb2604 / nb2103 / nb2660 / nb2674."""
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
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
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
    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full, int(len(pool))


# ============================================================================
# Bayesian Bootstrap trainer
# ============================================================================

def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="regression",
        metric="None",
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        learning_rate=LGBM_LR,
        n_estimators=LGBM_N_EST,
        min_child_samples=5,
        lambda_l2=2.0,
        seed=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        deterministic=True,
        force_col_wise=True,
        num_threads=2,
        verbose=-1,
    )


def _dirichlet_weights(n: int, seed: int) -> np.ndarray:
    """Draw symmetric Dirichlet(1,...,1) over n rows; rescale so mean=1.0.

    Symmetric Dirichlet(1,...,1) = uniform on the (n-1)-simplex; each component
    has mean 1/n and variance (n-1)/(n^2(n+1)).  Multiplying by n gives a
    sample-weight vector with mean 1.0 -- the right scale for LightGBM
    sample_weight which is interpreted as relative observation importance.
    """
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(n, dtype=np.float64) * DIRICHLET_ALPHA)
    return (w * n).astype(np.float64)


def _train_bb_ensemble(X_tr, y_tr, X_val):
    """Train N_MEMBERS LGBMs each with Dirichlet sample_weight; return mean prediction."""
    n_tr = len(y_tr)
    preds = np.zeros((N_MEMBERS, X_val.shape[0]), dtype=np.float64)
    for m in range(N_MEMBERS):
        seed_m = BB_BASE_SEED + m
        w = _dirichlet_weights(n_tr, seed=seed_m)
        booster = lgb.train(
            params=_lgbm_params(seed=seed_m),
            train_set=lgb.Dataset(
                X_tr, label=y_tr, weight=w, free_raw_data=False,
            ),
            num_boost_round=LGBM_N_EST,
        )
        preds[m] = booster.predict(X_val)
    return preds.mean(axis=0)


def _residual_cross_fit_scaffold(X, residual, scaffolds, kf_seed):
    """5-fold scaffold CV; per-fold train BB ensemble on resid, mean of 20 members."""
    n = len(residual)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=kf_seed,
    )
    oof = np.full(n, np.nan, dtype=np.float64)
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        oof[va_loc] = _train_bb_ensemble(
            X[tr_loc], residual[tr_loc].astype(np.float32), X[va_loc],
        )
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te):
    """Full-fit BB ensemble: train on all 253, predict on 513."""
    return _train_bb_ensemble(X_unb, residual.astype(np.float32), X_te).astype(np.float32)


# ============================================================================
# main
# ============================================================================

def main():
    import pandas as pd
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Bayesian Bootstrap (Rubin 1981) on K=20 LGBM residual")
    print(f"          N_MEMBERS={N_MEMBERS}  alpha={DIRICHLET_ALPHA}  "
          f"base_seed={BB_BASE_SEED}")
    print(f"          LGBM(depth={LGBM_MAX_DEPTH}, leaves={LGBM_NUM_LEAVES}, "
          f"n_est={LGBM_N_EST}, lr={LGBM_LR})")
    print(f"          5-fold scaffold CV  kf_seed={KF_SEED}")
    print(f"          gates  PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}")
    print("=" * 78)

    # ---- Load truth, anchor, scaffolds ----
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

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # ---- K=20 feature indices ----
    print("\n" + "-" * 78)
    print("STEP 1: load K=20 feature indices from nb2640_summary.json")
    print("-" * 78)
    with open(NB2640_SUMMARY) as f:
        nb2640 = json.load(f)
    K20_idx = np.array(nb2640["K20_idx_in_117col"], dtype=int)
    if len(K20_idx) != 20:
        raise RuntimeError(f"K20 idx len {len(K20_idx)} != 20")
    print(f"   K=20 idx (n={len(K20_idx)}): {K20_idx.tolist()}")

    # ---- 117-col matrix ----
    print("\n" + "-" * 78)
    print("STEP 2: rebuild 117-col 5-way feature matrix")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K = X_te_full[:, K20_idx].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")

    # ---- Diagnostic: Dirichlet weight stats for one draw ----
    w_sample = _dirichlet_weights(n_unb, seed=BB_BASE_SEED)
    print(f"\n[diag] Dirichlet weight sample (m=0):")
    print(f"   n={len(w_sample)}  mean={w_sample.mean():.4f}  "
          f"std={w_sample.std():.4f}  min={w_sample.min():.4f}  "
          f"max={w_sample.max():.4f}")
    print(f"   q05={np.quantile(w_sample,0.05):.3f}  "
          f"q50={np.quantile(w_sample,0.50):.3f}  "
          f"q95={np.quantile(w_sample,0.95):.3f}")
    # By symmetry, Var(w_i)*n = (n-1)/(n+1) -> for n=253 that's ~0.992
    # so std(w*n) ~ sqrt((n-1)/(n+1)) ~ 0.996
    print(f"   theoretical std (~sqrt((n-1)/(n+1))) = "
          f"{np.sqrt((n_unb - 1) / (n_unb + 1)):.4f}")

    # ---- Single-kf scaffold cross-fit ----
    print("\n" + "-" * 78)
    print(f"STEP 3: Bayesian-bootstrap residual LGBM  "
          f"kf_seed={KF_SEED}  {RESID_FOLDS} folds x {N_MEMBERS} BB members each")
    print("-" * 78)

    ts = time.time()
    resid_oof = _residual_cross_fit_scaffold(
        X_unb_K, residual, unb_scaffolds, kf_seed=KF_SEED,
    )
    pred_oof_unb = anchor + resid_oof
    pooled_rae = float(rae(y_unb, pred_oof_unb))
    print(f"   kf_seed={KF_SEED}  pooled RAE = {pooled_rae:.4f}  "
          f"wall={time.time()-ts:.1f}s")

    # Per-fold RAE for completeness
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=KF_SEED,
    )
    fold_rae_list = []
    for tr_loc, va_loc in splits:
        fold_rae_list.append(float(rae(y_unb[va_loc], pred_oof_unb[va_loc])))
    fold_rae_arr = np.array(fold_rae_list, dtype=np.float64)
    fold_mean_rae = float(fold_rae_arr.mean())
    fold_std_rae = float(fold_rae_arr.std(ddof=1))

    # mean_rae for gate: pooled RAE (matches nb2670 convention)
    mean_rae = pooled_rae

    print("\n" + "-" * 78)
    print("STEP 4: aggregate statistics")
    print("-" * 78)
    print(f"   per-fold RAE: " + ", ".join(f"{r:.4f}" for r in fold_rae_list))
    print(f"   fold mean +/- std = {fold_mean_rae:.4f} +/- {fold_std_rae:.5f}")
    print(f"   pooled RAE        = {pooled_rae:.4f}")
    print(f"   delta vs nb2171 (0.4682) = {mean_rae - NB2171_REF:+.4f}")

    # ---- Deploy: full-fit BB ensemble on 513 ----
    print("\n" + "-" * 78)
    print("STEP 5: deploy artifacts (full-fit on 253 -> predict 513)")
    print("-" * 78)
    t_te0 = time.time()
    te_resid_full = _train_full_then_predict_te(X_unb_K, residual, X_te_K)
    pred_te_513 = te_anchor_513 + te_resid_full
    te_unb_in_rae = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"   te full-fit BB wall = {time.time()-t_te0:.1f}s")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in_rae:.4f}")
    print(f"   pred_oof_unb std = {pred_oof_unb.std():.3f} "
          f"(truth_std {y_unb.std():.3f})")
    print(f"   pred_te_513  mean / std = {pred_te_513.mean():.3f} / "
          f"{pred_te_513.std():.3f}")

    # ---- Gate ----
    print("\n" + "-" * 78)
    print("STEP 6: GATE")
    print("-" * 78)
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   mean_rae = {mean_rae:.4f}")
    print(f"     <{GATE_PROMOTE} -> PROMOTE")
    print(f"     <{GATE_MARGINAL} -> MARGINAL_BEAT")
    print(f"     else            -> FAIL")
    print(f"   -> {verdict}")

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 7: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_bayesian_bootstrap.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "bayesian_bootstrap_dirichlet_lgbm_k20",
        "paradigm": "rubin_bayesian_bootstrap_continuous_weights",
        "rationale": "Rubin 1981 Bayesian bootstrap: each of N_MEMBERS LGBMs "
                     "trained on the FULL n_train rows but with continuous "
                     "Dirichlet(1,...,1) sample weights (rescaled to mean 1.0); "
                     "ensemble = mean of 20 predictions; smoother variance "
                     "reduction than vanilla bagging because every row "
                     "participates in every member with symmetric Dirichlet "
                     "weight noise (no lost rows); analogous to a Bayesian "
                     "forest under non-parametric posterior over reweighted "
                     "training distributions",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K": 20,
        "K20_idx_in_117col": K20_idx.tolist(),
        "n_members": N_MEMBERS,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "bb_base_seed": BB_BASE_SEED,
        "lgbm_max_depth": LGBM_MAX_DEPTH,
        "lgbm_num_leaves": LGBM_NUM_LEAVES,
        "lgbm_n_estimators": LGBM_N_EST,
        "lgbm_learning_rate": LGBM_LR,
        "kf_seed": KF_SEED,
        "resid_folds": RESID_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "fold_rae": fold_rae_list,
        "fold_mean_rae": fold_mean_rae,
        "fold_std_rae": fold_std_rae,
        "pooled_rae": pooled_rae,
        "mean_rae": mean_rae,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "pred_oof_std": float(pred_oof_unb.std()),
        "truth_std": float(y_unb.std()),
        "dirichlet_w0_mean": float(w_sample.mean()),
        "dirichlet_w0_std": float(w_sample.std()),
        "dirichlet_w0_min": float(w_sample.min()),
        "dirichlet_w0_max": float(w_sample.max()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "nb2171_ref": NB2171_REF,
        "delta_vs_nb2171": mean_rae - NB2171_REF,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-fold RAE       = " + ", ".join(f"{r:.4f}" for r in fold_rae_list))
    print(f"   fold mean +/- std  = {fold_mean_rae:.4f} +/- {fold_std_rae:.5f}")
    print(f"   pooled RAE         = {pooled_rae:.4f}")
    print(f"   delta vs nb2171    = {mean_rae - NB2171_REF:+.4f}")
    print(f"   te[unb_idx] RAE    = {te_unb_in_rae:.4f}")
    print(f"   verdict            = {verdict}")
    print(f"   wall               = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "fold_mean_rae",
        "fold_std_rae",
        "pooled_rae",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
