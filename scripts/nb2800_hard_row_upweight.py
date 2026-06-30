"""nb2800 -- Hard-row UP-weighting via EXPONENTIAL of absolute residual.

NEW PARADIGM (cycle 175+ prescription, fixing nb2791 inversion):
    nb2791 used w = 1/(|resid|+eps) which actually DOWN-weights hard rows
    (large |resid| -> small w). That was the WRONG direction.

    This script genuinely UP-weights hard rows via:
        w_i_raw = exp(0.5 * |y_i - chemprop_aux_pred_oof_i|)
        w_i     = w_i_raw / mean(w_i_raw)        (so mean(w) == 1.0)

    Intuition: residual-learner K=20 LGBM should focus on rows where
    chemprop_aux is wrong. Easy rows (small residual) are already covered
    by the anchor; hard rows carry the marginal signal.

    Scale factor 0.5 keeps dynamic range moderate. With residuals typically
    ~ U[0, 1.5], w_raw ranges [exp(0)=1.0, exp(0.75)~=2.12]; post-mean-
    normalize w ranges roughly [0.7x, 1.5x]. Wide enough to bias the learner
    toward hard rows without single-row dominance.

ANCHOR:
    chemprop_aux (nb1133_chemprop_aux_pred_oof.npy) -- the only verified-clean
    PRE-unblind anchor (cf. feedback_kaggle_chemprop_dead_end).
    Residual target = y_unb - chemprop_aux_oof.

LGBM:
    K=20 RFE features sliced from the 117-col 5-way matrix
    (AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN).
    Hyperparams identical to nb2240 / nb2791 K=20:
        max_depth=4, num_leaves=15, n_est=300, lr=0.03,
        min_child_samples=5, reg_lambda=2.0.
    K=20 LGBM bag seeds per fold (same as nb2791).

CV:
    5-fold scaffold CV on 253, 5 kf_seeds, mean over 20 LGBM seeds within
    each fold. Honest scaffold_kfold_indices from src/pxr/eval.py.

GATE:
    mean_rae < 0.4570 -> "PROMOTE"        (beats nb2171 deep-30 0.4682 by >0.011)
    mean_rae < 0.4598 -> "MARGINAL_BEAT"  (vs 2nd-best ceiling band)
    else              -> "FAIL"

OUTPUTS:
    data/processed/nb2800_summary.json
    data/processed/nb2800_pred_oof.npy   (253,) float32
    data/processed/te_nb2800.npy         (513,) float32
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
from pxr.paths import DATA_PROCESSED

TAG = "nb2800"

# ---- Anchor paths (chemprop_aux -- only verified-clean PRE-unblind anchor) ----
CHEMPROP_AUX_OOF = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
CHEMPROP_AUX_TE = DATA_PROCESSED / "te_chemprop_aux.npy"

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
ALPHA = 0.5                       # exponential scale factor (UP-weighting strength)
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
# K=20 LGBM seeds (same bag as nb2791)
RESID_SEEDS = [0, 1, 7, 13, 23, 42, 53, 67, 89, 101,
               137, 199, 211, 251, 313, 401, 503, 617, 727, 911]

# ---- Gate thresholds (vs nb2171 0.4682 deep-30 ceiling band) ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


# ============================================================================
# helpers (identical to nb2791 feature pipeline)
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
    from pxr.chem import standardize, morgan_fp_batch  # noqa: F401
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
# exponential UP-weighting of hard rows
# ---------------------------------------------------------------------------

def hard_row_upweight(y_train, anchor_train, alpha=ALPHA):
    """
    Per-row UP-weighting of hard rows via exponential of absolute residual.

    w_i_raw = exp(alpha * |y_i - anchor_i|)
    Normalize: w = w_raw / mean(w_raw) so mean(w) == 1.0.

    alpha=0.5 with residuals typically ~U[0, 1.5]: w_raw in [1.0, 2.12];
    post-mean-normalize w roughly in [0.7x, 1.5x]. Genuinely UP-weights
    high-|resid| rows since exp is monotone-increasing.
    """
    resid = np.asarray(y_train, dtype=np.float64) - np.asarray(anchor_train, dtype=np.float64)
    w_raw = np.exp(float(alpha) * np.abs(resid))
    w_raw = np.where(np.isfinite(w_raw), w_raw, 0.0)
    w_mean = float(w_raw.mean()) if w_raw.mean() > 0 else 1.0
    w = (w_raw / w_mean).astype(np.float64)
    return w


# ---------------------------------------------------------------------------
# cross-fit one weighted-LGBM bag of K=20 seeds, per kf_seed
# ---------------------------------------------------------------------------

def cv_run_for_seed(X_unb, y_unb, anchor, weights, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    residual = y_unb - anchor
    n = X_unb.shape[0]
    oof_resid = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        preds_va = np.zeros(len(va_loc), dtype=np.float64)
        for s in RESID_SEEDS:
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb[tr_loc], residual[tr_loc], sample_weight=weights[tr_loc])
            preds_va += mdl.predict(X_unb[va_loc]) / len(RESID_SEEDS)
        oof_resid[va_loc] = preds_va
    pred_oof = anchor + oof_resid
    return float(rae(y_unb, pred_oof)), pred_oof


def deploy_refit_te(X_unb, y_unb, anchor_unb, weights, X_te, te_anchor):
    """Refit on all 253 with weights; predict residual on full 513."""
    residual = y_unb - anchor_unb
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
    print(f"{TAG} -- exponential UP-weight of hard rows (fix nb2791 inversion)")
    print("=" * 78)

    # ---- Load truth ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Load anchor (chemprop_aux) ----
    anchor_oof = np.load(CHEMPROP_AUX_OOF).astype(np.float64)
    anchor_te = np.load(CHEMPROP_AUX_TE).astype(np.float64)
    assert anchor_oof.shape == (n_unb,), f"anchor_oof {anchor_oof.shape}"
    assert anchor_te.shape == (n_test,), f"anchor_te {anchor_te.shape}"
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[anchor] chemprop_aux in_RAE = {rae_anchor:.4f}")

    # ---- Build K=20 features ----
    print("\n[feat] building K=20 5-way features ...")
    t1 = time.time()
    X_unb_K20, X_te_K20, surviving_K20 = build_k20_features(te_smiles, n_test, unb_idx)
    print(f"[feat] X_unb_K20={X_unb_K20.shape}  X_te_K20={X_te_K20.shape}  wall={time.time()-t1:.1f}s")

    # ---- Compute exponential UP-weights for hard rows ----
    w = hard_row_upweight(y_unb, anchor_oof, alpha=ALPHA)
    abs_resid = np.abs(y_unb - anchor_oof)
    p90 = float(np.percentile(abs_resid, 90))
    p10 = float(np.percentile(abs_resid, 10))
    diag = {
        "alpha": ALPHA,
        "weight_mean": float(w.mean()),
        "weight_min": float(w.min()),
        "weight_max": float(w.max()),
        "weight_std": float(w.std()),
        "abs_resid_mean": float(abs_resid.mean()),
        "abs_resid_median": float(np.median(abs_resid)),
        "abs_resid_max": float(abs_resid.max()),
        "abs_resid_p90": p90,
        "abs_resid_p10": p10,
        "n_top_decile_resid": int((abs_resid >= p90).sum()),
        "weight_top_decile_resid_mean": float(w[abs_resid >= p90].mean()),
        "weight_bottom_decile_resid_mean": float(w[abs_resid <= p10].mean()),
    }
    print(
        f"[weights] alpha={ALPHA}  |resid|[mean/median/max]="
        f"{diag['abs_resid_mean']:.3f}/{diag['abs_resid_median']:.3f}/{diag['abs_resid_max']:.3f}"
    )
    print(
        f"[weights] w[min/mean/max]="
        f"{diag['weight_min']:.3f}/{diag['weight_mean']:.3f}/{diag['weight_max']:.3f}  "
        f"top-decile-resid w_mean={diag['weight_top_decile_resid_mean']:.3f}  "
        f"bot-decile-resid w_mean={diag['weight_bottom_decile_resid_mean']:.3f}"
    )
    # sanity: hard rows should have HIGHER weight than easy rows
    is_up_weighted = diag["weight_top_decile_resid_mean"] > diag["weight_bottom_decile_resid_mean"]
    print(
        f"[sanity] hard rows UP-weighted? "
        f"top_decile_resid_w_mean ({diag['weight_top_decile_resid_mean']:.3f}) "
        f"{'>' if is_up_weighted else '<='} "
        f"bot_decile_resid_w_mean ({diag['weight_bottom_decile_resid_mean']:.3f})  "
        f"[{'OK' if is_up_weighted else 'FAIL'}]"
    )
    assert is_up_weighted, (
        "weight formula did not UP-weight hard rows -- check exp(alpha*|resid|)"
    )

    # ---- Cross-fit across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"5-FOLD SCAFFOLD CV  kf_seeds={KF_SEEDS}  resid_seeds(K)={len(RESID_SEEDS)}")
    print("-" * 78)

    per_seed_results = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        t_seed = time.time()
        r, oof = cv_run_for_seed(
            X_unb_K20, y_unb, anchor_oof, w, unb_scaffolds, kf_seed,
        )
        per_seed_results.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": r,
            "wall_sec": round(time.time() - t_seed, 1),
        })
        all_oofs.append(oof)
        print(
            f"   kf_seed={kf_seed}  pooled_RAE={r:.4f}  "
            f"wall={per_seed_results[-1]['wall_sec']:.1f}s"
        )

    raes = np.array([s["pooled_rae"] for s in per_seed_results])
    mean_rae = float(raes.mean())
    std_rae = float(raes.std())
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1).astype(np.float32)

    # ---- Deploy refit to produce te ----
    print("\n[deploy] refitting on all 253 rows for full 513 test predictions ...")
    te_deploy = deploy_refit_te(
        X_unb_K20, y_unb, anchor_oof, w, X_te_K20, anchor_te,
    )
    te_unb_rae = float(rae(y_unb, te_deploy[unb_idx]))

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"

    print("\n" + "-" * 78)
    print(f"RESULTS  mean_RAE={mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"  te[unb_idx] in-sample RAE = {te_unb_rae:.4f}")
    print(f"  verdict = {verdict}")
    print(
        f"  gates: PROMOTE<{GATE_PROMOTE}  MARGINAL_BEAT<{GATE_MARGINAL}"
    )
    print("-" * 78)

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", mean_oof)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_deploy)
    print(f"[save] {DATA_PROCESSED / (TAG + '_pred_oof.npy')}")
    print(f"[save] {DATA_PROCESSED / ('te_' + TAG + '.npy')}")

    summary = {
        "tag": TAG,
        "method": "per_row_exponential_hard_row_upweight",
        "paradigm_note": (
            "UP-weight hard rows via w = exp(alpha * |y - anchor_pred|). "
            "Genuinely up-weights hard rows (vs nb2791 which used 1/(|resid|+eps) "
            "and inadvertently DOWN-weighted them)."
        ),
        "anchor": "chemprop_aux",
        "anchor_oof_path": str(CHEMPROP_AUX_OOF),
        "anchor_te_path": str(CHEMPROP_AUX_TE),
        "rae_anchor_chemprop_aux": rae_anchor,
        "alpha": ALPHA,
        "weight_recipe": "w_raw = exp(alpha * |y - anchor_pred|); w = w_raw / mean(w_raw)",
        "weight_stats": diag,
        "hard_rows_up_weighted": bool(is_up_weighted),
        "kf_seeds": KF_SEEDS,
        "resid_seeds": RESID_SEEDS,
        "n_resid_seeds": len(RESID_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "per_seed_results": per_seed_results,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "te_unb_rae_in_sample": te_unb_rae,
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
    print(f"   anchor                       = chemprop_aux")
    print(f"   alpha                        = {ALPHA}")
    print(f"   mean_RAE (5 kf_seeds, K=20)  = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   te[unb_idx] in-sample RAE    = {te_unb_rae:.4f}")
    print(f"   verdict                      = {verdict}")
    print(f"   wall                         = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "anchor",
        "rae_anchor_chemprop_aux",
        "alpha",
        "mean_rae",
        "std_rae",
        "te_unb_rae_in_sample",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
