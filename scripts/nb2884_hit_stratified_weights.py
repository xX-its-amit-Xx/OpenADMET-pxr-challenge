"""nb2884 -- Hit-class sample weights: UP-weight rare hit class on K=20
LGBM/chemprop_aux residual.

NEW PARADIGM (cycle 175+ hit-class sample-weight axis):
    Standard LGBM on chemprop_aux residual treats every training row equally
    in the MSE loss. The unblind 253 carries ~ same low (~2%) hit rate as
    the train (1.6%) and the K=20 residual carries the bulk of its
    geometric signal at high pEC50 (>= 5.5). With ~5 hit rows per fold
    of 50, the gradient is dominated by the ~45 non-hit rows; the model
    learns to track the inactive tail well but compresses the hit tail.

    nb2884 introduces a CLASS-based sample weight (NOT a per-row residual-
    based one):
        w_i = multiplier   if y_i >= 5.5
              1.0          else
    and sweeps multiplier in {1.5, 2.0, 3.0, 5.0}. The K=20 LGBM is
    refit per multiplier with `sample_weight=w`, evaluated in 5-fold
    scaffold CV across 5 kf_seeds. Best multiplier wins.

    Distinct from nb2800 (per-row exponential of |residual|, anchor-
    relative) and from nb2830 (hit-stratified scaffold folds, partition-
    side fix). nb2884 changes the LOSS surface by re-weighting the rare
    hit class so the LGBM gradient is no longer inactive-dominated.

ANCHOR:
    chemprop_aux (nb1133_chemprop_aux_pred_oof.npy + te_chemprop_aux.npy)
    -- the only verified-clean PRE-unblind anchor (cf.
    feedback_kaggle_chemprop_dead_end). Residual target = y_unb - anchor.

LGBM:
    K=20 RFE features sliced from the 117-col 5-way matrix (AtomPair /
    MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN).
    Hyperparams identical to nb2240 / nb2800 K=20:
        max_depth=4, num_leaves=15, n_est=300, lr=0.03,
        min_child_samples=5, reg_lambda=2.0.

CV:
    5-fold scaffold CV on 253, 5 kf_seeds, mean over 20 LGBM seeds
    within each fold. Honest scaffold_kfold_indices from src/pxr/eval.py.

GATE (best-multiplier mean RAE):
    mean_rae < 0.4570 -> "PROMOTE"        (beats nb2171 deep-30 0.4682 by >0.011)
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    data/processed/nb2884_summary.json
    data/processed/nb2884_pred_oof.npy   (253,) float32 best-multiplier
    data/processed/te_nb2884.npy         (513,) float32 best-multiplier deploy
    submissions/nb2884_hit_stratified_weights.csv
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

TAG = "nb2884"

# ---- Anchor paths (chemprop_aux -- only verified-clean PRE-unblind anchor) ----
CHEMPROP_AUX_OOF = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
CHEMPROP_AUX_TE = DATA_PROCESSED / "te_chemprop_aux.npy"

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

# ---- Feature paths (K=20 from nb2240 surviving subset, same as nb2800) ----
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

# ---- Hit-class sample weight sweep ----
HIT_THRESHOLD = 5.5
MULTIPLIERS = [1.5, 2.0, 3.0, 5.0]

# ---- CV protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
# K=20 LGBM bag seeds (same bag as nb2800)
RESID_SEEDS = [0, 1, 7, 13, 23, 42, 53, 67, 89, 101,
               137, 199, 211, 251, 313, 401, 503, 617, 727, 911]

# ---- Gate thresholds (vs nb2171 0.4682 deep-30 ceiling band) ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


# ============================================================================
# helpers (mirror nb2800 feature pipeline)
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
# hit-class sample weights
# ---------------------------------------------------------------------------

def hit_class_weights(y, multiplier, hit_threshold=HIT_THRESHOLD):
    """
    Per-row hit-class sample weight:
        w_i = multiplier   if y_i >= hit_threshold
              1.0          else
    Returns float64 weights (mean is NOT normalized to 1.0 -- we want the
    rare hit class to genuinely up-contribute to the gradient, and LGBM
    only uses these in relative terms anyway).
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.where(y >= float(hit_threshold), float(multiplier), 1.0)
    return w.astype(np.float64)


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
    """Refit on all 253 with hit-class weights; predict residual on full 513."""
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
    print(f"{TAG} -- hit-class sample weights (UP-weight y>={HIT_THRESHOLD} rows)")
    print(f"        multipliers sweep = {MULTIPLIERS}")
    print(f"        anchor = chemprop_aux  K=20 LGBM residual")
    print(f"        kf_seeds={KF_SEEDS}  resid_seeds(K)={len(RESID_SEEDS)}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load truth ----
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
    print(f"[feat] X_unb_K20={X_unb_K20.shape}  X_te_K20={X_te_K20.shape}  "
          f"wall={time.time()-t1:.1f}s")

    # ---- Diagnostic: hit-class distribution on unblind 253 ----
    is_hit = (y_unb >= HIT_THRESHOLD)
    n_hit = int(is_hit.sum())
    hit_rate = float(n_hit / n_unb)
    print(
        f"[hit] threshold={HIT_THRESHOLD}  n_hit={n_hit}/{n_unb}  "
        f"hit_rate={hit_rate:.4f}"
    )
    print(
        f"[hit] anchor in-RAE on hits  = "
        f"{float(rae(y_unb[is_hit], anchor_oof[is_hit])):.4f}  "
        f"(non-hits = {float(rae(y_unb[~is_hit], anchor_oof[~is_hit])):.4f})"
    )

    # ---- Sweep multipliers ----
    print("\n" + "-" * 78)
    print(f"HIT-CLASS WEIGHT SWEEP  multipliers={MULTIPLIERS}")
    print("-" * 78)

    per_mult_results = []
    per_mult_oofs = {}
    for mult in MULTIPLIERS:
        w = hit_class_weights(y_unb, mult, HIT_THRESHOLD)
        # Sanity log: weight stats
        w_hit_mean = float(w[is_hit].mean()) if n_hit > 0 else float("nan")
        w_nonhit_mean = float(w[~is_hit].mean())
        print(
            f"\n[mult={mult}]  w_hit_mean={w_hit_mean:.3f}  "
            f"w_nonhit_mean={w_nonhit_mean:.3f}  "
            f"(sum_w_hits={float(w[is_hit].sum()):.1f}  "
            f"sum_w_nonhits={float(w[~is_hit].sum()):.1f})"
        )

        per_seed_results = []
        all_oofs = []
        for kf_seed in KF_SEEDS:
            ts = time.time()
            r, oof = cv_run_for_seed(
                X_unb_K20, y_unb, anchor_oof, w, unb_scaffolds, kf_seed,
            )
            per_seed_results.append({
                "kf_seed": int(kf_seed),
                "pooled_rae": r,
                "wall_sec": round(time.time() - ts, 1),
            })
            all_oofs.append(oof)
            print(
                f"   mult={mult}  kf_seed={kf_seed}  pooled_RAE={r:.4f}  "
                f"wall={per_seed_results[-1]['wall_sec']:.1f}s"
            )
        raes = np.array([s["pooled_rae"] for s in per_seed_results])
        mean_rae = float(raes.mean())
        std_rae = float(raes.std())
        mean_oof = np.mean(np.column_stack(all_oofs), axis=1).astype(np.float32)
        per_mult_oofs[mult] = mean_oof
        per_mult_results.append({
            "multiplier": float(mult),
            "weight_hit_mean": w_hit_mean,
            "weight_nonhit_mean": w_nonhit_mean,
            "per_seed_results": per_seed_results,
            "mean_rae": mean_rae,
            "std_rae": std_rae,
        })
        print(
            f"   mult={mult}  MEAN_RAE = {mean_rae:.4f} +/- {std_rae:.4f}"
        )

    # ---- Best multiplier ----
    best = min(per_mult_results, key=lambda d: d["mean_rae"])
    best_mult = float(best["multiplier"])
    best_mean_rae = float(best["mean_rae"])
    best_std_rae = float(best["std_rae"])
    best_oof = per_mult_oofs[best_mult]

    # ---- Gate on best multiplier ----
    if best_mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"

    print("\n" + "-" * 78)
    print(f"BEST  multiplier={best_mult}  mean_RAE={best_mean_rae:.4f} "
          f"+/- {best_std_rae:.4f}")
    print(f"  delta_vs_anchor    = {best_mean_rae - rae_anchor:+.4f}")
    sweep_str = ", ".join(
        f"m={d['multiplier']}->{d['mean_rae']:.4f}" for d in per_mult_results
    )
    print(f"  vs ALL multipliers = [{sweep_str}]")
    print(f"  verdict = {verdict}")
    print(f"  gates: PROMOTE<{GATE_PROMOTE}  MARGINAL_BEAT<{GATE_MARGINAL}")
    print("-" * 78)

    # ---- Deploy refit (best multiplier only) ----
    print(f"\n[deploy] refitting on full 253 with best multiplier={best_mult} ...")
    best_w = hit_class_weights(y_unb, best_mult, HIT_THRESHOLD)
    te_deploy = deploy_refit_te(
        X_unb_K20, y_unb, anchor_oof, best_w, X_te_K20, anchor_te,
    )
    te_unb_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_rae:.4f}  "
          f"(in-sample optimism expected vs OOF {best_mean_rae:.4f})")

    # ---- Save artefacts (best multiplier as canonical pred_oof + te) ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, best_oof)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_hit_stratified_weights.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "hit_class_sample_weighted_K20_LGBM_chemprop_aux_residual",
        "rationale": (
            "Per-row hit-class sample weights: w = multiplier if y >= 5.5 "
            "else 1.0. Sweep multipliers in {1.5, 2.0, 3.0, 5.0} on K=20 "
            "LGBM residual (chemprop_aux anchor). UP-weights rare hit class "
            "to balance the gradient, which is otherwise dominated by the "
            "~98% non-hit majority. Distinct from nb2800 (per-row exp(|resid|) "
            "weight, anchor-relative) and nb2830 (hit-stratified scaffold "
            "folds, partition-side fix)."
        ),
        "anchor": "chemprop_aux",
        "anchor_oof_path": str(CHEMPROP_AUX_OOF),
        "anchor_te_path": str(CHEMPROP_AUX_TE),
        "anchor_pre_unblind": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "hit_threshold": float(HIT_THRESHOLD),
        "multipliers": MULTIPLIERS,
        "n_hit_unb": n_hit,
        "hit_rate_unb": hit_rate,
        "kf_seeds": KF_SEEDS,
        "resid_seeds": RESID_SEEDS,
        "n_resid_seeds": len(RESID_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "feat_dim": int(X_unb_K20.shape[1]),
        "model_class": "lightgbm.LGBMRegressor",
        "lgbm_params_sample": _lgbm_params(KF_SEEDS[0]),
        "per_multiplier_results": per_mult_results,
        "best_multiplier": best_mult,
        "best_mean_rae": best_mean_rae,
        "best_std_rae": best_std_rae,
        "best_delta_vs_anchor": best_mean_rae - rae_anchor,
        "te_unb_in_sample_rae": te_unb_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    # convenience: per-multiplier mean_rae as top-level dict
    summary["mean_rae_by_multiplier"] = {
        str(d["multiplier"]): d["mean_rae"] for d in per_mult_results
    }

    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                      = chemprop_aux ({rae_anchor:.4f})")
    print(f"   hit_threshold               = {HIT_THRESHOLD}  n_hit={n_hit}")
    for d in per_mult_results:
        print(
            f"   mult={d['multiplier']:>4}  mean_RAE = "
            f"{d['mean_rae']:.4f} +/- {d['std_rae']:.4f}"
        )
    print(f"   BEST_multiplier             = {best_mult}")
    print(f"   BEST_mean_RAE               = {best_mean_rae:.4f} +/- {best_std_rae:.4f}")
    print(f"   te[unb_idx] in-sample RAE   = {te_unb_rae:.4f}")
    print(f"   verdict                     = {verdict}")
    print(f"   wall                        = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "anchor",
        "rae_anchor_chemprop_aux",
        "hit_threshold",
        "n_hit_unb",
        "mean_rae_by_multiplier",
        "best_multiplier",
        "best_mean_rae",
        "best_std_rae",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
