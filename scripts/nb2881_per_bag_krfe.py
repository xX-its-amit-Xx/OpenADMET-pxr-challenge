"""nb2881 -- Per-bag K-RFE feature selection.

NEW PARADIGM (vs prior bagging on fixed K-feature slice):
    Standard bag-ensembles on the K=28 SHAP-pruned matrix (nb2103, nb1158,
    nb2240, etc.) feed every bag the SAME 20-or-28 feature subset.  All
    inductive diversity comes from row resampling + seed; the feature axis
    is shared.

    nb2881 ADDS feature-axis diversity by letting each of 10 bags pick its
    OWN top-20 features via greedy backward RFE *starting from the K=28
    SHAP-pruned set*.  Per bag:
        1. Subsample 80 percent of the 253 rows (with seed-controlled RNG).
        2. Run RFE: train LGBM on currently-surviving features, drop the
           feature with the smallest split-gain importance, repeat
           descending K=28 -> K=20.  Use the 80 percent in-bag rows for
           every RFE inner fit.
        3. Refit the LGBM on the 80 percent in-bag rows using the surviving
           20-feature subset.
        4. Predict on the FULL set of rows the outer fold leaves for
           validation.
    Bag predictions are mean-aggregated.

    Capacity sketch: 10 bags x C(28, 20) = 10 x 3_108_105 possible feature
    sets.  Each bag explores a different slice of that space; RFE on
    different 80 percent row samples picks different features because tree
    split gain depends on which rows are present.  Feature-axis disagreement
    across bags is the new diversity dimension; mean aggregation projects it
    back to a single prediction.  This is OOD-aware: bags that overfit to
    spurious features get partially cancelled by bags that drop those
    features.

PROTOCOL:
    - Build the 117-col 5-way K-tuned matrix exactly as nb2103/nb2240.
    - Slice to top-28 by nb2063 SHAP importance ranking (X_K28, 117 -> 28).
    - chemprop_aux anchor on residual y_unb - anchor.
    - Outer: 5-fold scaffold CV on 253 unblind, kf_seed=1001.
    - Per outer fold:
        for bag_seed in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}:
            rng = default_rng(bag_seed)
            row_perm = rng.permutation(len(tr_loc))
            bag_rows = tr_loc[row_perm[:int(0.8 * len(tr_loc))]]
            surviving_feat_idx = list(range(28))
            while len(surviving_feat_idx) > 20:
                mdl = LGBMRegressor(seed=bag_seed)
                mdl.fit(X_K28[bag_rows][:, surviving_feat_idx],
                        residual[bag_rows])
                imp = mdl.feature_importances_  # split-gain
                drop = surviving_feat_idx[int(np.argmin(imp))]
                surviving_feat_idx.remove(drop)
            mdl_final = LGBMRegressor(seed=bag_seed)
            mdl_final.fit(X_K28[bag_rows][:, surviving_feat_idx],
                          residual[bag_rows])
            pred_va_bag = mdl_final.predict(
                X_K28[va_loc][:, surviving_feat_idx]
            )
        pred_va = mean over 10 bags  (residual)
        oof[va_loc] = anchor[va_loc] + pred_va
    mean_rae = rae(y_unb, oof) under kf_seed=1001 scaffold-CV.

    Deploy: same loop on the FULL 253 set (no held-out), refit per bag on
    80 percent of all 253, predict residual on 513 test, mean over 10 bags,
    add to te_chemprop_aux 513 anchor.

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4598  -> MARGINAL_BEAT
    else                -> FAIL

Outputs:
    scripts/nb2881_per_bag_krfe.py
    data/processed/nb2881_summary.json
    data/processed/nb2881_pred_oof.npy   (253,) float32
    data/processed/te_nb2881.npy         (513,) float32
    submissions/nb2881_per_bag_krfe.csv  (on non-FAIL)
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

TAG = "nb2881"

# ---- experimental knobs ----
N_BAGS = 10
BAG_FRAC = 0.80
K_START = 28
K_FINAL = 20
N_FOLDS = 5
KF_SEED = 1001
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- anchor + 117-col matrix paths ----
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
NB2063_SHAP_IMP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers -- copied verbatim from nb2103 / nb2240 so the 117-col matrix is
# bit-identical
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
    """LGBM(MSE) identical to nb2103/nb2240."""
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


def _build_117_te(te_smiles, n_test) -> tuple[np.ndarray, list[str], list[str]]:
    """Rebuild the 117-col 5-way K-tuned feature matrix on full 513 test."""
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
    X_mord_te = _load_mordred_test(n_test_expected=n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

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
            X_ap_te,
            X_maccs_te,
            X_mord_te,
            X_emb_te,
            X_av_te,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X_te_full.shape[1] != 117:
        raise ValueError(f"117-col build failed: {X_te_full.shape}")

    # Build name + family arrays for top-28 reporting later
    feat_names: list[str] = []
    feat_family: list[str] = []
    for b in top_ap_bit_idx:
        feat_names.append(f"AtomPair_bit_{int(b)}")
        feat_family.append("AtomPair")
    for b in top_maccs_bit_idx:
        feat_names.append(f"MACCS_bit_{int(b)}")
        feat_family.append("MACCS")
    for c in top_mord_col_idx:
        feat_names.append(f"Mordred_col_{int(c)}")
        feat_family.append("Mordred")
    for d in top_embed_col_idx:
        feat_names.append(f"ChempropEmbed_dim_{int(d)}")
        feat_family.append("ChempropEmbed")
    for b in top_avalon_bit_idx:
        feat_names.append(f"Avalon_bit_{int(b)}")
        feat_family.append("Avalon")
    feat_names.append("pred_chembl_pec50")
    feat_family.append("ChEMBL_kNN")
    feat_names.append("mean_sim")
    feat_family.append("ChEMBL_kNN")
    assert len(feat_names) == 117
    return X_te_full, feat_names, feat_family


# ============================================================================
# core per-bag K-RFE
# ============================================================================

def per_bag_krfe_predict(
    X_tr_full28: np.ndarray,
    y_tr_resid: np.ndarray,
    X_va_full28: np.ndarray,
    n_bags: int,
    bag_frac: float,
    k_start: int,
    k_final: int,
) -> tuple[np.ndarray, list[dict]]:
    """Run N_BAGS x (subsample + RFE + refit) and return mean residual prediction.

    Each bag picks its OWN k_final features from k_start via greedy backward
    RFE.  Returns mean residual prediction on the validation rows and a
    per-bag log (selected feature indices + RAE on training fraction).
    """
    n_tr = X_tr_full28.shape[0]
    n_va = X_va_full28.shape[0]
    n_subsample = int(round(bag_frac * n_tr))
    if n_subsample < k_start + 5:
        raise ValueError(
            f"n_subsample={n_subsample} too small vs k_start={k_start}"
        )
    bag_preds = np.zeros((n_bags, n_va), dtype=np.float64)
    log: list[dict] = []
    for bag_id in range(n_bags):
        rng = np.random.default_rng(bag_id)
        row_perm = rng.permutation(n_tr)
        bag_rows = row_perm[:n_subsample]
        X_bag = X_tr_full28[bag_rows]   # (n_subsample, k_start)
        y_bag = y_tr_resid[bag_rows]

        # backward RFE descending k_start -> k_final on this bag's 80 percent
        surviving = list(range(k_start))
        while len(surviving) > k_final:
            mdl = lgb.LGBMRegressor(**_lgbm_params(bag_id))
            mdl.fit(X_bag[:, surviving], y_bag)
            imp = np.asarray(mdl.feature_importances_, dtype=np.float64)
            # smallest split-gain importance is the drop candidate
            drop_local = int(np.argmin(imp))
            drop_global = surviving[drop_local]
            surviving.remove(drop_global)

        # final fit on the surviving k_final features, same bag rows
        mdl_final = lgb.LGBMRegressor(**_lgbm_params(bag_id))
        mdl_final.fit(X_bag[:, surviving], y_bag)
        pred_va_bag = mdl_final.predict(X_va_full28[:, surviving])
        bag_preds[bag_id] = pred_va_bag

        # training-RAE on the IN-BAG slice for logging only (proxy)
        pred_bag_self = mdl_final.predict(X_bag[:, surviving])
        log.append({
            "bag_id": int(bag_id),
            "selected_idx_in_28": [int(i) for i in surviving],
            "n_subsample": int(n_subsample),
            "in_bag_resid_rae_proxy": float(rae(y_bag, pred_bag_self)),
        })
    return bag_preds.mean(axis=0), log


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-bag K-RFE  bags={N_BAGS}  k_start={K_START}  k_final={K_FINAL}")
    print(f"          outer {N_FOLDS}-fold scaffold-CV kf_seed={KF_SEED}")
    print("=" * 78)

    # ---- Load test + unblind ----
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

    # ---- Anchor: chemprop_aux ----
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux  in_RAE={rae_anchor:.4f}  (ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor_unb
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Build full 117-col on 513 ----
    X_te_117, feat_names_117, feat_family_117 = _build_117_te(te_smiles, n_test)
    print(f"[feat] X_te_117 = {X_te_117.shape}")

    # ---- SHAP top-28 slice (X_K28) ----
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing SHAP importance: {NB2063_SHAP_IMP}")
    shap_imp = np.load(NB2063_SHAP_IMP).astype(np.float32)
    if shap_imp.shape[0] != 117:
        raise ValueError(f"SHAP importance shape mismatch: {shap_imp.shape}")
    full_rank_order = np.argsort(-shap_imp).astype(np.int32)
    top28_idx_in_117 = full_rank_order[:K_START].tolist()
    top28_names = [feat_names_117[i] for i in top28_idx_in_117]
    top28_family = [feat_family_117[i] for i in top28_idx_in_117]
    fam_counts = {}
    for fam in top28_family:
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    print(f"[K28] top-28 family breakdown: {fam_counts}")

    X_te_K28 = X_te_117[:, top28_idx_in_117].astype(np.float32)   # 513 x 28
    X_unb_K28 = X_te_K28[unb_idx].astype(np.float32)              # 253 x 28
    print(f"[K28] X_unb_K28={X_unb_K28.shape}  X_te_K28={X_te_K28.shape}")

    # ---- Scaffold folds ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}  folds={N_FOLDS}  kf_seed={KF_SEED}")
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )

    # ---- Per-fold per-bag K-RFE CV ----
    print("\n" + "-" * 78)
    print(f"PER-BAG K-RFE 5-FOLD CV  bag_frac={BAG_FRAC}")
    print("-" * 78)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_rae = []
    per_fold_bag_log = []
    for f_idx, (tr_loc, va_loc) in enumerate(splits):
        ts = time.time()
        mean_resid_va, bag_log = per_bag_krfe_predict(
            X_tr_full28=X_unb_K28[tr_loc],
            y_tr_resid=residual[tr_loc],
            X_va_full28=X_unb_K28[va_loc],
            n_bags=N_BAGS,
            bag_frac=BAG_FRAC,
            k_start=K_START,
            k_final=K_FINAL,
        )
        pred_va = anchor_unb[va_loc] + mean_resid_va
        oof[va_loc] = pred_va
        r = float(rae(y_unb[va_loc], pred_va))
        fold_rae.append(r)
        per_fold_bag_log.append({
            "fold": int(f_idx),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "va_rae": r,
            "bags": bag_log,
        })
        # feature-selection diversity stats
        selected_counts = np.zeros(K_START, dtype=int)
        for bl in bag_log:
            for j in bl["selected_idx_in_28"]:
                selected_counts[j] += 1
        n_all_picks = int((selected_counts == N_BAGS).sum())
        n_no_picks = int((selected_counts == 0).sum())
        print(f"   fold {f_idx}  va_RAE={r:.4f}  n_tr={len(tr_loc)} n_va={len(va_loc)}  "
              f"feat_always_picked={n_all_picks}/28  feat_never_picked={n_no_picks}/28  "
              f"wall={time.time()-ts:.1f}s")

    pooled = float(rae(y_unb, oof))
    mean_fold = float(np.mean(fold_rae))
    mean_rae = pooled
    print(f"\n[cv] pooled RAE     = {pooled:.4f}")
    print(f"[cv] mean fold RAE  = {mean_fold:.4f}")
    print(f"[cv] delta vs anchor = {mean_rae - rae_anchor:+.4f}")

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] mean_rae {mean_rae:.4f}  "
          f"(< {GATE_PROMOTE} PROMOTE / < {GATE_MARGINAL} MARGINAL)  ->  {verdict}")

    # ---- Deploy: same loop on FULL 253; apply mean residual to 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: 10-bag K-RFE on FULL 253, mean residual on 513")
    print("-" * 78)
    ts = time.time()
    mean_resid_te, deploy_bag_log = per_bag_krfe_predict(
        X_tr_full28=X_unb_K28,
        y_tr_resid=residual,
        X_va_full28=X_te_K28,
        n_bags=N_BAGS,
        bag_frac=BAG_FRAC,
        k_start=K_START,
        k_final=K_FINAL,
    )
    deploy_te = (te_anchor_513 + mean_resid_te).astype(np.float32)
    deploy_te = np.clip(deploy_te, 3.0, 9.0)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   deploy mean residual std={mean_resid_te.std():.4f}  "
          f"mean={mean_resid_te.mean():+.4f}")
    print(f"   te mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_rae:.4f}  (expected << pooled)")
    selected_counts_deploy = np.zeros(K_START, dtype=int)
    for bl in deploy_bag_log:
        for j in bl["selected_idx_in_28"]:
            selected_counts_deploy[j] += 1
    n_all_deploy = int((selected_counts_deploy == N_BAGS).sum())
    n_never_deploy = int((selected_counts_deploy == 0).sum())
    print(f"   deploy feature pick freq: always={n_all_deploy}/28  "
          f"never={n_never_deploy}/28  wall={time.time()-ts:.1f}s")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_per_bag_krfe.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] no submission CSV (FAIL gate)")

    summary = {
        "tag": TAG,
        "method": ("per_bag_K_RFE_each_bag_picks_own_K20_from_K28"
                   "_chemprop_aux_residual_LGBM_MSE"),
        "anchor": "chemprop_aux",
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "rae_anchor": rae_anchor,
        "n_bags": N_BAGS,
        "bag_frac": BAG_FRAC,
        "k_start": K_START,
        "k_final": K_FINAL,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "shap_source": str(NB2063_SHAP_IMP),
        "top28_idx_in_117": top28_idx_in_117,
        "top28_names": top28_names,
        "top28_family_counts": fam_counts,
        "feature_117_family_total": {
            k: int(sum(1 for f in feat_family_117 if f == k))
            for k in sorted(set(feat_family_117))
        },
        "fold_rae": fold_rae,
        "pooled_rae": pooled,
        "mean_fold_rae": mean_fold,
        "mean_rae": mean_rae,
        "delta_vs_anchor": mean_rae - rae_anchor,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "per_fold_bag_log": per_fold_bag_log,
        "deploy_bag_log": deploy_bag_log,
        "deploy_feature_pick_always_count_28": n_all_deploy,
        "deploy_feature_pick_never_count_28": n_never_deploy,
        "te_unb_in_sample_rae": te_unb_rae,
        "te_mean": float(deploy_te.mean()),
        "te_std": float(deploy_te.std()),
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (pooled)        = {mean_rae:.4f}  ({verdict})")
    print(f"   mean fold RAE            = {mean_fold:.4f}")
    print(f"   anchor RAE               = {rae_anchor:.4f}  (delta {mean_rae - rae_anchor:+.4f})")
    print(f"   deploy te[unb] RAE       = {te_unb_rae:.4f}")
    print(f"   feature pick always/never= {n_all_deploy}/{n_never_deploy} of 28")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "pooled_rae", "mean_fold_rae",
        "delta_vs_anchor", "verdict",
        "deploy_feature_pick_always_count_28",
        "deploy_feature_pick_never_count_28",
        "te_unb_in_sample_rae",
        "te_mean", "te_std",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
