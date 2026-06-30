"""nb3111 -- K=20 residual on COUNTER pEC50 anchor (NEW PARADIGM: substrate swap).

CONTEXT (per feedback_cycle169_axes_closed.md):
    nb2171 (chemprop_aux anchor + K=20 RFE residual + pyramid + stretch)
    converges at deep-30 RAE 0.4682, declared genuine ceiling for post-hoc
    blend on chemprop_aux anchor at n=253. Four orthogonal attacks (minimal
    anchor, trimmed deep-30, post-hoc stretch on deep-30, graph Laplacian
    features) all failed to break it.

    Cycle-134 thesis: cross-paradigm orthogonality only real when the model
    lives on a DIFFERENT ANCHOR axis. cf. nb730 null-ensemble on
    counter-assay axis (Phase-2 P3 winner -0.0325 vs nb703).

    The substrate change this script tests: REPLACE chemprop_aux anchor with
    counter_clean (nb2490 PRE-clean OOF) anchor. Counter_clean alone is
    poor (RAE 2.14 on 253, mean 3.11 vs y_unb mean 4.66 -- offset by -1.55)
    BUT the residual y - counter_clean has std 1.07 (vs 1.03 for y) and is
    biologically orthogonal to chemprop_aux. K=20 RFE LGBM should learn the
    additive correction (the +1.55 mean shift + the pEC50 shape).

PROTOCOL:
    1. Build the 117-col 5-way K-tuned feature matrix on 513 test (same as
       nb2240/nb2490), slice to the K=20 RFE surviving indices from
       nb2231_summary.json snapshots.20.surviving_idx_in_117.
    2. Counter anchor = te_nb2490_counter.npy (513,) PRE-clean.
    3. Build residual on 253 unblind: residual = y_unb - counter_clean[unb_idx].
    4. K=20 RFE LGBM residual: 5-fold scaffold-CV across 30 FRESH seeds
       {2001..2030}, mean-bag.
       Save nb3111_counter_K20_oof.npy (253,) = counter_clean[unb] + resid_oof
            te_nb3111_counter_K20.npy (513,) = counter_clean + resid_te
    5. Quantile-conditional blend with K18 deep-30 (nb2960):
         per fold-train K_counter q_cut=0.5 quantile threshold q
         low (K_counter<=q): w_counter=0.5, w_K18=0.5
         high(K_counter >q): w_counter=0.5, w_K18=0.5
         (start equal-blend; counter anchor is novel substrate, K18 is the
          verified ceiling -- 50/50 is the right default for novel paradigm
          orthogonality test)
       15 FRESH kf_seeds {1141..1155}, pooled_rae across 5 outer folds.
    6. Gate: blend mean < 0.4475 -> BETTER; else FAIL.

References:
    nb3091 K18+K20 quantile-conditional blend (PRIOR paradigm chemprop_aux)
    nb2490 counter_clean PRE-clean anchor (counter axis, RAE 2.14 alone)
    nb2960 K18 deep-30 OOF                  = 0.4536
    nb2171 prior post-hoc top                = 0.4682
    nb3030 wide-seed K18 ceiling             = 0.4509

Outputs:
    scripts/nb3111_counter_anchor_K.py
    data/processed/nb3111_counter_K20_oof.npy  (253,) float32
    data/processed/te_nb3111_counter_K20.npy   (513,) float32
    data/processed/nb3111_pred_oof.npy         (253,) float32 (blend median seed)
    data/processed/te_nb3111.npy               (513,) float32 (deploy te blend)
    data/processed/nb3111_summary.json
    submissions/nb3111_counter_anchor_K.csv    (only on BETTER verdict)
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3111"

# -----------------------------------------------------------------------------
# Step-1 (counter residual) configuration
# -----------------------------------------------------------------------------
COUNTER_TE_PATH = DATA_PROCESSED / "te_nb2490_counter.npy"  # PRE-clean 513
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

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

# 30 fresh seeds for counter-residual LGBM mean-bag
RESID_SEEDS = list(range(2001, 2031))  # {2001..2030}
RESID_FOLDS = 5

# -----------------------------------------------------------------------------
# Step-2 (quantile-conditional blend with K18) configuration
# -----------------------------------------------------------------------------
K18_OOF_PATH = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
K18_TE_PATH = DATA_PROCESSED / "nb2960_K18_30seed_te.npy"

N_FOLDS = 5
KF_SEEDS = list(range(1141, 1156))  # 15 FRESH seeds {1141..1155}

# Quantile-conditional weights (new paradigm, counter axis -- equal-blend default)
Q_CUT = 0.5
W_COUNTER_LOW = 0.5   # counter anchor in low half
W_K18_LOW = 1.0 - W_COUNTER_LOW
W_COUNTER_HIGH = 0.5  # counter anchor in high half (symmetric for novel paradigm)
W_K18_HIGH = 1.0 - W_COUNTER_HIGH

# -----------------------------------------------------------------------------
# Gates / references
# -----------------------------------------------------------------------------
GATE_BETTER = 0.4475  # task spec: blend mean < 0.4475 -> BETTER
REF_K18 = 0.4536       # nb2960 K18 deep-30 alone
REF_NB3030 = 0.4509    # nb3030 K18 wide-seed ceiling
REF_NB2171 = 0.4682    # prior chemprop_aux post-hoc ceiling


# ============================================================================
# helpers (copied / specialized from nb2240, nb2490)
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


def _build_X_te_117(te_smiles, n_test):
    """Build the 117-col 5-way K-tuned feature matrix on the 513 test."""
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
    assert feat_dim == 117, f"feat_dim {feat_dim} != 117"
    return X_te_full


def _residual_cross_fit_scaffold(X_unb, residual, unb_scaffolds, seed):
    """Scaffold-CV cross-fit on residual at a single seed."""
    folds = scaffold_kfold_indices(
        unb_scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in folds:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X_unb[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X_unb[va_loc])
    assert not np.any(np.isnan(oof)), f"seed={seed} residual OOF has NaNs"
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float64)


# ============================================================================
# Step 2: quantile-conditional blend helpers
# ============================================================================

def _blend_quantile_conditional(
    p_counter: np.ndarray,
    p_k18: np.ndarray,
    q_thr: float,
) -> np.ndarray:
    """Per-row hard-split blend.

    rows with p_counter <= q_thr -> (W_COUNTER_LOW, W_K18_LOW)
    rows with p_counter >  q_thr -> (W_COUNTER_HIGH, W_K18_HIGH)
    """
    low_mask = p_counter <= q_thr
    out = np.empty_like(p_counter, dtype=np.float64)
    out[low_mask] = (
        W_COUNTER_LOW * p_counter[low_mask] + W_K18_LOW * p_k18[low_mask]
    )
    out[~low_mask] = (
        W_COUNTER_HIGH * p_counter[~low_mask] + W_K18_HIGH * p_k18[~low_mask]
    )
    return out


def _run_one_blend_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> dict:
    """Quantile-conditional blend at one kf_seed (P_unb columns are [counter, K18])."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_q_thrs = []
    fold_high_share = []
    for tr_loc, va_loc in splits:
        q_thr = float(np.quantile(P_unb[tr_loc, 0], Q_CUT))
        fold_q_thrs.append(q_thr)
        val_p_counter = P_unb[va_loc, 0]
        val_p_k18 = P_unb[va_loc, 1]
        val_pred = _blend_quantile_conditional(val_p_counter, val_p_k18, q_thr)
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
        fold_high_share.append(float(np.mean(val_p_counter > q_thr)))
    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_q_thr_mean": float(np.mean(fold_q_thrs)),
        "fold_q_thr_std": float(np.std(fold_q_thrs, ddof=1)),
        "fold_high_share_mean": float(np.mean(fold_high_share)),
        "oof": oof_blend,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- COUNTER ANCHOR K=20 RFE residual + quantile blend with K18")
    print(f"         (NEW PARADIGM: substrate swap, counter axis not chemprop_aux)")
    print(f"         RESID_SEEDS={len(RESID_SEEDS)} fresh {{{RESID_SEEDS[0]}..{RESID_SEEDS[-1]}}}")
    print(f"         KF_SEEDS  ={len(KF_SEEDS)} fresh {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"         gate: blend mean < {GATE_BETTER:.4f} -> BETTER; else FAIL")
    print("=" * 78)

    # ---- Load data ----
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Load counter anchor (PRE-clean) ----
    counter_te_513 = np.load(COUNTER_TE_PATH).astype(np.float64)
    counter_unb = counter_te_513[unb_idx]
    rae_counter_unb = float(rae(y_unb, counter_unb))
    print(f"[anchor] counter_clean te mean={counter_te_513.mean():.3f}  std={counter_te_513.std():.3f}")
    print(f"[anchor] counter_clean unb RAE={rae_counter_unb:.4f}  (alone, poor by design)")

    residual = y_unb - counter_unb
    print(f"[anchor] residual mean={residual.mean():+.3f}  std={residual.std():.3f}")

    # ---- Load K=20 RFE surviving feature indices ----
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[feat] K=20 RFE features loaded (sample: {surviving_K20_names[:3]}...)")

    # ---- Build 117-col feature matrix and slice to K=20 ----
    X_te_117 = _build_X_te_117(te_smiles, n_test)
    X_te_K20 = X_te_117[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20={X_unb_K20.shape}  X_te_K20={X_te_K20.shape}")

    # ---- Unblind scaffolds for CV ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] n_unique={n_unique_scaf}")

    # ---- Step 1: K=20 LGBM residual mean-bag over 30 fresh seeds ----
    print("\n" + "-" * 78)
    print(f"STEP 1: K=20 LGBM residual (counter anchor)  seeds={len(RESID_SEEDS)}  folds={RESID_FOLDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_scaffold(X_unb_K20, residual, unb_scaffolds, s)
        per_seed_corrected[i] = counter_unb + resid_oof
        per_seed_rae.append(float(rae(y_unb, counter_unb + resid_oof)))
        te_resid_s = _train_full_then_predict_te(X_unb_K20, residual, X_te_K20, s)
        per_seed_te_resid[i] = te_resid_s
        if i < 3 or i >= len(RESID_SEEDS) - 2 or i % 10 == 0:
            print(f"   seed={s:4d}  rae_corr={per_seed_rae[-1]:.4f}  wall={time.time()-ts:.1f}s")

    counter_K20_oof = per_seed_corrected.mean(axis=0)
    counter_K20_te = (counter_te_513 + per_seed_te_resid.mean(axis=0))
    rae_counter_K20_meanbag = float(rae(y_unb, counter_K20_oof))
    rae_counter_K20_perseed = float(np.mean(per_seed_rae))
    std_counter_K20_perseed = float(np.std(per_seed_rae))
    print(f"\n[step1] per-seed mean RAE = {rae_counter_K20_perseed:.4f} (+/- {std_counter_K20_perseed:.4f})")
    print(f"[step1] mean-bag RAE      = {rae_counter_K20_meanbag:.4f}")

    # Save step-1 artifacts
    out_counter_K20_oof = DATA_PROCESSED / f"{TAG}_counter_K20_oof.npy"
    out_counter_K20_te = DATA_PROCESSED / f"te_{TAG}_counter_K20.npy"
    np.save(out_counter_K20_oof, counter_K20_oof.astype(np.float32))
    np.save(out_counter_K20_te, counter_K20_te.astype(np.float32))
    print(f"[save] {out_counter_K20_oof}")
    print(f"[save] {out_counter_K20_te}")

    # ---- Step 2: Load K18 deep-30 and quantile-conditional blend ----
    print("\n" + "-" * 78)
    print(f"STEP 2: quantile-conditional blend with K18 deep-30  kf_seeds={len(KF_SEEDS)}")
    print(f"        weights: q_cut={Q_CUT}, low ({W_COUNTER_LOW}/{W_K18_LOW}), high ({W_COUNTER_HIGH}/{W_K18_HIGH})")
    print("-" * 78)
    k18_oof = np.load(K18_OOF_PATH).astype(np.float64)
    k18_te = np.load(K18_TE_PATH).astype(np.float64)
    assert k18_oof.shape == (n_unb,), f"K18 oof shape {k18_oof.shape}"
    assert k18_te.shape == (n_test,), f"K18 te shape {k18_te.shape}"
    rae_k18 = float(rae(y_unb, k18_oof))
    print(f"   K18 deep-30 alone RAE     = {rae_k18:.4f}  (ref {REF_K18:.4f})")
    print(f"   counter_K20 alone RAE     = {rae_counter_K20_meanbag:.4f}")
    corr_counter_k18 = float(np.corrcoef(counter_K20_oof, k18_oof)[0, 1])
    print(f"   corr(counter_K20, K18)    = {corr_counter_k18:.4f}")

    # P_unb columns: [counter_K20_oof, k18_oof]
    P_unb = np.column_stack([counter_K20_oof, k18_oof])
    P_te = np.column_stack([counter_K20_te, k18_te])

    leak_flags = {}
    for i, lbl in enumerate(["counter_K20", "K18"]):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[lbl] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {lbl}: {frac:.1%} rows == truth -- possible leak")

    seed_records = []
    pooled_raes = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_blend_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_q_thr_mean": round(res["fold_q_thr_mean"], 4),
            "fold_q_thr_std": round(res["fold_q_thr_std"], 4),
            "fold_high_share_mean": round(res["fold_high_share_mean"], 4),
        })
        if KF_SEEDS.index(s) < 3 or KF_SEEDS.index(s) >= len(KF_SEEDS) - 2:
            print(
                f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
                f"q_thr={res['fold_q_thr_mean']:.3f}  "
                f"high_share={res['fold_high_share_mean']:.2f}  "
                f"wall={time.time()-ts:.2f}s"
            )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # 95% CI t-mult at df=14
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    print("\n" + "-" * 78)
    print(f"BLEND AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   mean    = {mean_rae:.4f}")
    print(f"   std     = {std_rae:.4f}")
    print(f"   sem     = {sem:.4f}")
    print(f"   95% CI  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median  = {median_rae:.4f}")
    print(f"   min/max = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   delta vs K18 alone        = {mean_rae - rae_k18:+.4f}")
    print(f"   delta vs gate ({GATE_BETTER}) = {mean_rae - GATE_BETTER:+.4f}")

    # ---- Deploy: q_thr from FULL 253 counter_K20 OOF, blend te ----
    deploy_q_thr = float(np.quantile(P_unb[:, 0], Q_CUT))
    te_pred = _blend_quantile_conditional(
        P_te[:, 0], P_te[:, 1], deploy_q_thr,
    ).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    te_low_share = float(np.mean(P_te[:, 0] <= deploy_q_thr))
    print(f"\n   deploy q_thr (full counter_K20 q{Q_CUT}) = {deploy_q_thr:.4f}")
    print(f"   te(513) low-half share = {te_low_share:.3f}")
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)

    # ---- Gate ----
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE. nb3111 counter-anchor 15-seed mean {mean_rae:.4f} "
            f"beats gate {GATE_BETTER:.4f} ({mean_rae - GATE_BETTER:+.4f}). "
            f"Substrate change (counter pEC50 anchor + K=20 RFE residual + "
            f"quantile blend with K18) opens a new paradigm axis. "
            f"Promote to PRIMARY candidate pending wide-seed cross-verify."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"FAIL. nb3111 counter-anchor 15-seed mean {mean_rae:.4f} "
            f"does not beat gate {GATE_BETTER:.4f} ({mean_rae - GATE_BETTER:+.4f}). "
            f"Counter pEC50 axis substrate change does not unlock new "
            f"orthogonality vs K18 chemprop_aux ceiling at n=253 with "
            f"equal-blend quantile pattern."
        )
    print("\n" + "-" * 78)
    print(f"GATE: blend mean={mean_rae:.4f}  ->  {verdict}")
    print(f"      BETTER if < {GATE_BETTER:.4f}")
    print("-" * 78)

    # ---- Save ----
    out_pred_oof = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    out_te = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(out_pred_oof, oof_for_save)
    np.save(out_te, te_pred)
    print(f"\n[save] {out_pred_oof}")
    print(f"[save] {out_te}")

    sub_csv = SUBMISSIONS / f"{TAG}_counter_anchor_K.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "method": (
            "counter_pEC50_anchor_K20_RFE_residual_quantile_blend_K18_deep30_"
            "NEW_PARADIGM_substrate_swap"
        ),
        "anchor_pre_unblind": True,
        "anchor_axis": "counter_pEC50",
        "purpose": "substrate change -- counter_clean anchor in place of chemprop_aux",
        # Step 1
        "step1_resid_seeds": RESID_SEEDS,
        "step1_resid_folds": RESID_FOLDS,
        "step1_rae_counter_unb": rae_counter_unb,
        "step1_residual_mean": float(residual.mean()),
        "step1_residual_std": float(residual.std()),
        "step1_rae_counter_K20_per_seed": [float(r) for r in per_seed_rae],
        "step1_rae_counter_K20_per_seed_mean": rae_counter_K20_perseed,
        "step1_rae_counter_K20_per_seed_std": std_counter_K20_perseed,
        "step1_rae_counter_K20_mean_bag": rae_counter_K20_meanbag,
        "step1_counter_K20_oof_path": str(out_counter_K20_oof),
        "step1_counter_K20_te_path": str(out_counter_K20_te),
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names_first5": surviving_K20_names[:5],
        # Step 2
        "blend_kf_seeds": KF_SEEDS,
        "blend_n_folds": N_FOLDS,
        "blend_weights": {
            "q_cut": Q_CUT,
            "w_counter_low": W_COUNTER_LOW,
            "w_k18_low": W_K18_LOW,
            "w_counter_high": W_COUNTER_HIGH,
            "w_k18_high": W_K18_HIGH,
        },
        "anchor_k18_alone_rae": rae_k18,
        "anchor_counter_K20_alone_rae": rae_counter_K20_meanbag,
        "anchor_oof_corr": round(corr_counter_k18, 4),
        "anchor_leak_eq_truth_frac": leak_flags,
        "blend_seed_records": seed_records,
        "blend_pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "blend_mean_rae": round(mean_rae, 4),
        "blend_std_rae": round(std_rae, 4),
        "blend_sem_rae": round(sem, 4),
        "blend_ci95_low": round(ci_low, 4),
        "blend_ci95_high": round(ci_high, 4),
        "blend_median_rae": round(median_rae, 4),
        "blend_min_rae": round(float(arr.min()), 4),
        "blend_max_rae": round(float(arr.max()), 4),
        # Deploy
        "deploy_q_thr": round(deploy_q_thr, 4),
        "deploy_te_low_share": round(te_low_share, 4),
        "deploy_te_mean": float(te_pred.mean()),
        "deploy_te_std": float(te_pred.std()),
        "deploy_te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(out_pred_oof),
        "te_npy_path": str(out_te),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        # References
        "ref_K18_deep30": REF_K18,
        "ref_nb3030": REF_NB3030,
        "ref_nb2171": REF_NB2171,
        "delta_vs_K18_alone": round(mean_rae - rae_k18, 4),
        "delta_vs_gate": round(mean_rae - GATE_BETTER, 4),
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds_unb": int(n_unique_scaf),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_summary = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {out_summary}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   counter_K20 mean-bag RAE  = {rae_counter_K20_meanbag:.4f}")
    print(f"   K18 deep-30 alone RAE     = {rae_k18:.4f}")
    print(f"   blend mean ({n_s} seeds)  = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                    = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs gate ({GATE_BETTER}) = {mean_rae - GATE_BETTER:+.4f}")
    print(f"   verdict                   = {verdict}")
    print(f"   wall                      = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "step1_rae_counter_K20_per_seed_mean",
        "step1_rae_counter_K20_mean_bag",
        "anchor_k18_alone_rae",
        "anchor_counter_K20_alone_rae",
        "anchor_oof_corr",
        "blend_mean_rae",
        "blend_std_rae",
        "blend_ci95_low",
        "blend_ci95_high",
        "delta_vs_K18_alone",
        "delta_vs_gate",
        "verdict",
        "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
