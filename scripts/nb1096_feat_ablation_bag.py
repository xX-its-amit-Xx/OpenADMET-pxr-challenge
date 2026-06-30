"""nb1096 -- Random feature ablation bagging on K=28 (SHAP-top-28).

HYPOTHESIS:
    nb2103 K=28 single-config mean-bag RAE = 0.4737, median-bag = 0.4698
    on the residual-LGBM(MSE) chemprop_aux anchor (5-seed bag, 5-fold KFold
    cross-fit per seed, 117-col 5-way K-tuned matrix, top-28 SHAP slice).

    Each LGBM is trained on the FULL 28 columns -- there is no feature
    subsampling. This notebook adds RANDOM-FEATURE-ABLATION BAGGING:

      * For each of 10 random seeds, randomly DROP 30% of the K=28 columns
        (keep 20 columns).
      * Per bag: 5-fold KFold(shuffle=True, seed=bag_seed) cross-fit of
        LGBM(MSE, L=15 lr=0.03 mc=5 lambda=2 n_est=300) on the surviving cols.
      * Aggregate residual_oof across the 10 bags via MEAN (and report
        MEDIAN as diagnostic).
      * Compute mean-bag RAE on 253 = rae(y_unb, anchor + mean_bag_oof).

    ALSO sweep drop-50% (keep 14 columns) as a stronger-ablation variant.

    Decision: BEAT nb2103 K=28 mean-bag (0.4737) by margin 0.003. If either
    variant clears, build a 513-row deploy CSV via the SAME bag protocol
    (10 random column drops, fit on all 253, predict 513, row-mean of
    513-pred across bags).

PROTOCOL:
    1. Load chemprop_aux residual on 253 (anchor) and cached top-28 feature
       matrix `data/processed/X_unb_28_nb2103.npy` (already SHAP-top-28
       indexed via nb2103, shape (253, 28)).
    2. For each (variant in {keep20, keep14}, bag_seed in 0..9):
         rng = np.random.default_rng(bag_seed)
         keep_cols = rng.choice(28, size=keep, replace=False) (sorted)
         X_bag = X_unb_28[:, keep_cols]
         kf = KFold(5, shuffle=True, random_state=bag_seed)
         resid_oof[bag, va] = LGBM(seed=bag_seed).fit(tr).predict(va)
    3. mean_bag_oof = resid_oof.mean(axis=0); pred = anchor + mean_bag_oof
       rae_mean_bag = rae(y_unb, pred)
    4. Decision vs nb2103 (mean_bag 0.4737, median_bag 0.4698) at margin
       0.003.
    5. If keep20 or keep14 beats 0.4737-0.003=0.4707: build deploy CSV via
       per-bag fit-on-all-253 + predict 513, row-mean across 10 bags.

OUTPUTS:
    scripts/nb1096_feat_ablation_bag.py
    data/processed/nb1096_summary.json
    data/processed/nb1096_mean_bag_oof_keep20.npy   (253,) float32
    data/processed/nb1096_mean_bag_oof_keep14.npy   (253,) float32
    [conditional] submissions/nb1096_feat_ablation_keep{K}.csv  (513 rows)
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

TAG = "nb1096"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Cached top-28 SHAP-pruned feature matrix on the 253 unblind rows
# (built by nb2103 / nb2112 from the 117-col 5-way K-tuned stack).
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"

# Reference: nb2103 K=28 single-config
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.003

# Bagging schedule: 10 random seeds (column-drop seeds), 5-fold cross-fit.
N_BAGS = 10
BAG_SEEDS = list(range(N_BAGS))
N_FOLDS = 5
K_TOTAL = 28

# Two ablation variants: drop 30% (keep 20) and drop 50% (keep 14)
VARIANTS = [
    {"name": "keep20", "keep": 20, "drop_frac": 0.30},
    {"name": "keep14", "keep": 14, "drop_frac": 0.50},
]

# Paths needed only if we promote to a 513-row deploy CSV. We rebuild the
# full 117-col stack the same way nb2112 does so the deploy slice on
# top28_idx is identical to the nb2103/nb2112 column convention.
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


def _lgbm_params(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb2103/nb2112 (max_depth=4, num_leaves=15,
    n_estimators=300, lr=0.03, min_child_samples=5, reg_lambda=2.0).
    """
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


def _pick_keep_cols(bag_seed: int, keep: int, k_total: int = K_TOTAL) -> np.ndarray:
    """Pick `keep` columns out of `k_total` uniformly at random (sorted)."""
    rng = np.random.default_rng(bag_seed)
    cols = rng.choice(k_total, size=keep, replace=False)
    return np.sort(cols).astype(np.int32)


def _residual_cross_fit_one_bag(X_bag: np.ndarray, residual: np.ndarray,
                                bag_seed: int) -> np.ndarray:
    """5-fold KFold(shuffle=True, seed=bag_seed) cross-fit on (X_bag, residual)."""
    n = len(residual)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=bag_seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(bag_seed))
        mdl.fit(X_bag[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X_bag[va_loc])
    return oof


# ---------- 513-row deploy reconstruction (only invoked if a variant wins) ----------
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
    """Identical union to nb2103/nb2112."""
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
             std_smiles=("std_smiles", "first"))
    )
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


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X = np.load(mte_p).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape {X.shape} vs n_test={n_test_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    return np.where(np.isfinite(X), X, 0.0).astype(np.float32)


def _build_X_te_28(top28_idx: np.ndarray, n_test: int,
                   test_smiles: list) -> np.ndarray:
    """Rebuild the same 117-col 5-way K-tuned stack as nb2112 and slice
    on the cached top-28 SHAP indices, returning a (n_test, 28) matrix.
    """
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
    rec_mord = next(r for r in sum_1523["per_K_records"]
                    if int(r["K"]) == int(sum_1523["best_K"]))
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = next(np.array(f["top_idx_ranked"], dtype=int)
                          for f in sum_1484["families"] if f["family"] == "AtomPair")
    top_ap_bit_idx = full_ap_ranked[:int(sum_1524["best_K"])]
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:int(sum_1541["best_K"])]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"],
                                  dtype=int)

    X_ap = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx]
    X_maccs = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx]
    X_mord = _load_mordred_test(n_test)[:, top_mord_col_idx]
    X_emb = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx]
    X_av = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx]

    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_iks = {ik for ik in (_safe_inchikey(m) for m in test_mols) if ik is not None}
    pool = pool[~pool["inchikey"].isin(test_iks)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    pool = pool[keep_pool].reset_index(drop=True)
    fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_117 = np.concatenate(
        [X_ap, X_maccs, X_mord, X_emb, X_av,
         pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    return X_te_117[:, top28_idx].astype(np.float32)


def _build_deploy_csv(variant: dict, X_te_28: np.ndarray,
                      X_unb_28: np.ndarray, residual_unb: np.ndarray,
                      anchor_te_513: np.ndarray, test_smiles, mol_names,
                      n_test: int) -> dict:
    """Per-bag fit-on-all-253 + predict 513, row-mean across bags."""
    keep = variant["keep"]
    name = variant["name"]
    all_resid_513 = np.zeros((N_BAGS, n_test), dtype=np.float64)
    fit_records = []
    for bi, bs in enumerate(BAG_SEEDS):
        keep_cols = _pick_keep_cols(bs, keep)
        Xtr_bag = X_unb_28[:, keep_cols]
        Xte_bag = X_te_28[:, keep_cols]
        t0 = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(bs))
        mdl.fit(Xtr_bag, residual_unb)
        resid_513 = mdl.predict(Xte_bag)
        all_resid_513[bi] = resid_513
        fit_records.append({
            "bag": int(bi),
            "bag_seed": int(bs),
            "keep_cols": keep_cols.tolist(),
            "resid_513_mean": float(resid_513.mean()),
            "resid_513_std": float(resid_513.std()),
            "wall_sec": round(time.time() - t0, 2),
        })
    mean_resid_513 = all_resid_513.mean(axis=0)
    median_resid_513 = np.median(all_resid_513, axis=0)
    te_pred = anchor_te_513 + mean_resid_513
    sub_path = SUBMISSIONS_DIR / f"{TAG}_feat_ablation_{name}.csv"
    pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_pred.astype(np.float32),
    }).to_csv(sub_path, index=False)
    te_path = DATA_PROCESSED / f"te_{TAG}_{name}.npy"
    np.save(te_path, te_pred.astype(np.float32))
    return {
        "submission_csv": str(sub_path),
        "te_artifact": str(te_path),
        "deploy_te_mean": float(te_pred.mean()),
        "deploy_te_std": float(te_pred.std()),
        "deploy_resid_mean": float(mean_resid_513.mean()),
        "deploy_resid_std": float(mean_resid_513.std()),
        "deploy_resid_median_mean": float(median_resid_513.mean()),
        "fit_records": fit_records,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RANDOM FEATURE ABLATION BAGGING on K=28 SHAP-pruned matrix")
    print(f"          variants = {[v['name'] for v in VARIANTS]}  "
          f"n_bags={N_BAGS}  folds={N_FOLDS}  K_total={K_TOTAL}")
    print(f"          anchor = {ANCHOR}")
    print(f"          ref = nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG_REF}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF}  margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load top-28 SHAP indices and cached feature matrix ----
    if not NB2103_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2103_SUMMARY}")
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    rec28 = next(r for r in nb2103_sum["per_K_records"] if int(r["K"]) == 28)
    top28_idx = np.array(rec28["top_K_idx_in_117"], dtype=np.int32)
    if top28_idx.shape[0] != K_TOTAL:
        raise ValueError(f"top28_idx has {top28_idx.shape[0]} entries, expected {K_TOTAL}")
    print(f"[reuse] nb2103 top-28 SHAP indices in 117 (head 10): "
          f"{top28_idx[:10].tolist()}")
    print(f"[reuse] nb2103 K=28 mean_bag (summary) = "
          f"{float(rec28['rae_mean_bag']):.6f}")
    print(f"[reuse] nb2103 K=28 median_bag (summary) = "
          f"{float(rec28['rae_median_bag']):.6f}")

    if not X_UNB_28_PATH.exists():
        raise FileNotFoundError(f"missing cached unb top-28 matrix: {X_UNB_28_PATH}")
    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float32)
    print(f"[load] X_unb_28 = {X_unb_28.shape}  dtype={X_unb_28.dtype}")
    if X_unb_28.shape[1] != K_TOTAL:
        raise ValueError(f"X_unb_28 has {X_unb_28.shape[1]} cols, expected {K_TOTAL}")

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = (te["smiles"] if "smiles" in te.columns else te["SMILES"]).astype(str).tolist()
    if "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "name" in te.columns:
        mol_names = te["name"].astype(str).tolist()
    else:
        raise KeyError("no name column on test set")
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    if X_unb_28.shape[0] != n_unb:
        raise ValueError(f"X_unb_28 rows {X_unb_28.shape[0]} != n_unb {n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape} vs n_test={n_test}")
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual_unb = y_unb - anchor_unb
    print(f"[load] n_test={n_test}  n_unb={n_unb}")
    print(f"[load] {ANCHOR} in_RAE = {rae_anchor:.4f}")
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Variant sweep ----
    variant_results = []
    for variant in VARIANTS:
        name = variant["name"]
        keep = variant["keep"]
        drop = K_TOTAL - keep
        print("\n" + "-" * 78)
        print(f"VARIANT {name}: keep={keep}  drop={drop}  "
              f"(drop_frac={variant['drop_frac']:.0%})")
        print("-" * 78)

        per_bag_oof = np.zeros((N_BAGS, n_unb), dtype=np.float64)
        per_bag_records = []
        for bi, bs in enumerate(BAG_SEEDS):
            t_b = time.time()
            keep_cols = _pick_keep_cols(bs, keep)
            X_bag = X_unb_28[:, keep_cols].astype(np.float32)
            resid_oof_b = _residual_cross_fit_one_bag(X_bag, residual_unb, bs)
            per_bag_oof[bi] = resid_oof_b
            pred_corr_b = anchor_unb + resid_oof_b
            rae_b = float(rae(y_unb, pred_corr_b))
            per_bag_records.append({
                "bag": int(bi),
                "bag_seed": int(bs),
                "keep_cols": keep_cols.tolist(),
                "rae_corrected": rae_b,
                "delta_vs_anchor": rae_b - rae_anchor,
                "resid_oof_mean": float(resid_oof_b.mean()),
                "resid_oof_std": float(resid_oof_b.std()),
                "wall_sec": round(time.time() - t_b, 2),
            })
            print(f"   {name} bag={bi:2d} seed={bs:2d}  "
                  f"keep_cols head={keep_cols[:6].tolist()}  "
                  f"rae={rae_b:.4f}  d_vs_anchor={rae_b - rae_anchor:+.4f}  "
                  f"wall={time.time() - t_b:.1f}s")

        mean_bag_oof = per_bag_oof.mean(axis=0)
        median_bag_oof = np.median(per_bag_oof, axis=0)
        pred_mean_bag = anchor_unb + mean_bag_oof
        pred_median_bag = anchor_unb + median_bag_oof
        rae_mean_bag = float(rae(y_unb, pred_mean_bag))
        rae_median_bag = float(rae(y_unb, pred_median_bag))

        per_bag_rae_arr = np.array([r["rae_corrected"] for r in per_bag_records])
        per_bag_mean = float(per_bag_rae_arr.mean())
        per_bag_median = float(np.median(per_bag_rae_arr))
        per_bag_std = float(per_bag_rae_arr.std())
        per_bag_min = float(per_bag_rae_arr.min())
        per_bag_max = float(per_bag_rae_arr.max())

        delta_vs_nb2103 = rae_mean_bag - NB2103_K28_MEAN_BAG_REF
        delta_vs_nb2103_median = rae_median_bag - NB2103_K28_MEDIAN_BAG_REF
        beats_nb2103_mean = rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
        beats_nb2103_median = rae_median_bag < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN
        flat_vs_nb2103 = abs(delta_vs_nb2103) < DECISION_MARGIN
        if beats_nb2103_mean:
            verdict = "BEATS_NB2103_K28_MEAN_BAG"
        elif beats_nb2103_median:
            verdict = "BEATS_NB2103_K28_MEDIAN_BAG_ONLY"
        elif flat_vs_nb2103:
            verdict = "FLAT_VS_NB2103_K28"
        else:
            verdict = "HURTS_VS_NB2103_K28"

        print(f"   {name} per-bag RAE  mean={per_bag_mean:.4f}  "
              f"median={per_bag_median:.4f}  std={per_bag_std:.4f}  "
              f"min={per_bag_min:.4f}  max={per_bag_max:.4f}")
        print(f"   {name} POOLED mean-bag RAE   = {rae_mean_bag:.4f}  "
              f"(d_vs_nb2103_mean={delta_vs_nb2103:+.4f})")
        print(f"   {name} POOLED median-bag RAE = {rae_median_bag:.4f}  "
              f"(d_vs_nb2103_median={delta_vs_nb2103_median:+.4f})")
        print(f"   {name} verdict = {verdict}")

        oof_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_{name}.npy"
        np.save(oof_path, mean_bag_oof.astype(np.float32))
        print(f"   [save] {oof_path}")

        variant_results.append({
            "variant": name,
            "keep": int(keep),
            "drop": int(drop),
            "drop_frac": float(variant["drop_frac"]),
            "n_bags": int(N_BAGS),
            "n_folds": int(N_FOLDS),
            "per_bag_records": per_bag_records,
            "per_bag_rae_mean": per_bag_mean,
            "per_bag_rae_median": per_bag_median,
            "per_bag_rae_std": per_bag_std,
            "per_bag_rae_min": per_bag_min,
            "per_bag_rae_max": per_bag_max,
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_mean_bag_vs_nb2103": delta_vs_nb2103,
            "delta_median_bag_vs_nb2103": delta_vs_nb2103_median,
            "beats_nb2103_mean": bool(beats_nb2103_mean),
            "beats_nb2103_median": bool(beats_nb2103_median),
            "flat_vs_nb2103": bool(flat_vs_nb2103),
            "verdict": verdict,
            "deploy": None,
        })

    # ---- Variant summary table ----
    print("\n" + "=" * 78)
    print("VARIANT SUMMARY TABLE")
    print("=" * 78)
    print(f"   {'variant':>10s}  {'keep':>4s}  {'mean_bag':>10s}  "
          f"{'median_bag':>10s}  {'per_bag_mean':>13s}  "
          f"{'per_bag_std':>11s}  {'d_vs_nb2103':>12s}  verdict")
    print(f"   {'nb2103_K28':>10s}  {28:>4d}  "
          f"{NB2103_K28_MEAN_BAG_REF:>10.4f}  "
          f"{NB2103_K28_MEDIAN_BAG_REF:>10.4f}  "
          f"{'N/A':>13s}  {'N/A':>11s}  {0.0:>+12.4f}  BASELINE")
    for r in variant_results:
        print(f"   {r['variant']:>10s}  {r['keep']:>4d}  "
              f"{r['rae_mean_bag']:>10.4f}  {r['rae_median_bag']:>10.4f}  "
              f"{r['per_bag_rae_mean']:>13.4f}  "
              f"{r['per_bag_rae_std']:>11.4f}  "
              f"{r['delta_mean_bag_vs_nb2103']:>+12.4f}  {r['verdict']}")

    # ---- Promote winners to deploy CSV ----
    winners = [r for r in variant_results if r["beats_nb2103_mean"]]
    print("\n" + "-" * 78)
    if winners:
        print(f"PROMOTE: {len(winners)} variant(s) beat nb2103 by margin "
              f"{DECISION_MARGIN}: {[w['variant'] for w in winners]}")
        print("-" * 78)
        # Rebuild full 117-col -> slice top-28 on the 513 test set ONCE
        X_te_28 = _build_X_te_28(top28_idx, n_test, test_smiles)
        print(f"   [feat] X_te_28 = {X_te_28.shape}")
        for r in variant_results:
            if not r["beats_nb2103_mean"]:
                continue
            v = next(v for v in VARIANTS if v["name"] == r["variant"])
            print(f"\n   DEPLOY variant {v['name']} (keep={v['keep']})  "
                  f"-- per-bag fit-on-all-253, predict 513, row-mean across "
                  f"{N_BAGS} bags")
            dep = _build_deploy_csv(
                v, X_te_28, X_unb_28, residual_unb,
                te_anchor_513, test_smiles, mol_names, n_test,
            )
            r["deploy"] = dep
            print(f"   [save] {dep['submission_csv']}")
            print(f"   [save] {dep['te_artifact']}")
            print(f"   te stats: mean={dep['deploy_te_mean']:.4f}  "
                  f"std={dep['deploy_te_std']:.4f}")
    else:
        print("NO DEPLOY: no variant beat nb2103 K=28 mean_bag by margin "
              f"{DECISION_MARGIN}.")
        print("-" * 78)

    # ---- Global verdict ----
    if winners:
        best_winner = min(winners, key=lambda r: r["rae_mean_bag"])
        global_verdict = (
            f"FEAT_ABLATION_BAG_BEATS_NB2103_K28_AT_{best_winner['variant']}_"
            f"mean_bag={best_winner['rae_mean_bag']:.4f}"
        )
    else:
        # Check flat
        any_flat = any(r["flat_vs_nb2103"] for r in variant_results)
        if any_flat:
            global_verdict = "FEAT_ABLATION_BAG_FLAT_VS_NB2103_K28"
        else:
            global_verdict = "FEAT_ABLATION_BAG_DOES_NOT_BEAT_NB2103_K28"
    print(f"\n   global verdict = {global_verdict}")

    summary = {
        "tag": TAG,
        "method": "random_feature_ablation_bagging_on_K28_shap_pruned",
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "feature_source": ("nb2103/nb2112 cached top-28 SHAP-pruned matrix "
                           "from the 117-col 5-way K-tuned stack "
                           "(AtomPair/MACCS/Mordred/ChempropEmbed/Avalon + ChEMBL kNN)"),
        "X_unb_28_path": str(X_UNB_28_PATH),
        "top28_idx_in_117": top28_idx.tolist(),
        "n_bags": int(N_BAGS),
        "bag_seeds": BAG_SEEDS,
        "n_folds": int(N_FOLDS),
        "k_total": int(K_TOTAL),
        "variants": VARIANTS,
        "lgbm_params": {
            "objective": "regression",
            "max_depth": 4,
            "num_leaves": 15,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "min_child_samples": 5,
            "reg_lambda": 2.0,
        },
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "rae_anchor": rae_anchor,
        "residual_mean": float(residual_unb.mean()),
        "residual_std": float(residual_unb.std()),
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "variant_results": variant_results,
        "global_verdict": global_verdict,
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
        "n_test", "n_unb", "k_total", "n_bags", "n_folds",
        "rae_anchor",
        "nb2103_K28_mean_bag_ref", "nb2103_K28_median_bag_ref",
        "decision_margin", "global_verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== VARIANT TABLE ====")
    for r in res["variant_results"]:
        deploy_str = (f"  DEPLOY={r['deploy']['submission_csv']}"
                      if r.get("deploy") else "")
        print(f"  {r['variant']:>8s}  keep={r['keep']:>3d}  "
              f"mean_bag={r['rae_mean_bag']:.4f}  "
              f"median_bag={r['rae_median_bag']:.4f}  "
              f"d_vs_nb2103={r['delta_mean_bag_vs_nb2103']:+.4f}  "
              f"{r['verdict']}{deploy_str}")
