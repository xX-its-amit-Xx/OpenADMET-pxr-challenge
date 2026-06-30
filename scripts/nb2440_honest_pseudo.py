"""nb2440 -- HONEST pseudo-labeling (per nb2430 leak diagnostic).

LEAK DIAGNOSIS RECAP (nb2430):
  -- bag_std collapsed to ~2.2e-16 across 5 seeds (max). LGBM K=20 params used
     no row/feature subsampling, so all 5 seeds learned identical trees =>
     "diversity" was floating-point noise => confidence gate fired on 513/513
     test rows.
  -- pseudo labels came from a model REFIT ON ALL 253 UNBLIND ROWS, then
     re-injected into each outer fold's training matrix. Since the pseudo-
     label model had already absorbed the labels for the validation rows of
     EVERY outer fold, the "OOF" predictions were in-sample (RAE 0.26 vs
     anchor 0.62 was a leak artifact, not honest skill).

FIXES IN nb2440:
  1. SEED DIVERSITY -- add feature_fraction=0.8, bagging_fraction=0.8,
     bagging_freq=1 to LGBM params and vary random_state across
     {0, 1, 7, 42, 137}. Now bag std on test residuals must be > 0.01 to
     pass the audit gate.
  2. HONEST PSEUDO LABELS -- pseudo labels are NOT deploy-refit. They are
     computed PER OUTER FOLD using only that fold's tr_loc rows. For outer
     fold k, the per-seed pseudo-label model is trained on
     X_unb[tr_loc_k] -> residual[tr_loc_k] (NOT the full 253), then predicts
     the test residual. Bag mean over seeds = pseudo label for that outer
     fold; bag std = epistemic gate.
  3. SCAFFOLD KFold (NOT random KFold) on the 253 unblind. Per CLAUDE.md
     "always use scaffold CV, not random." Plus seeds {0, 1, 7, 42, 137}.
  4. Audit: bag_std must be > 0.01 on test residuals; high-conf fire rate
     must fall in 10-50%. If outside that band, treat as IMPLAUSIBLE and
     refuse to declare a winner. Either suggests collapse (still no
     diversity) or runaway noise (no useful signal in pseudo labels).
  5. Compare vs nb2240 0.4601; gate 0.003.

Outputs:
  scripts/nb2440_honest_pseudo.py
  data/processed/nb2440_summary.json
  data/processed/nb2440_oof_w{0.1,0.3,0.5}.npy
  data/processed/te_nb2440.npy                   (gate + audit pass only)
  submissions/nb2440_honest_pseudo.csv           (gate + audit pass only)
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
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2440"

# ------------------------- references / constants ---------------------------
NB2240_REF_OOF = 0.4601
GATE_MARGIN = 0.003

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

STD_GATE = 0.05            # bag std on test residual must be < this to keep row
MIN_REAL_BAG_STD = 0.01    # audit floor: median bag_std must exceed this
HIGH_CONF_BAND = (0.10, 0.50)
PSEUDO_WEIGHTS = [0.1, 0.3, 0.5]

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


def _lgbm_params_diverse(seed):
    """Seed-diverse LGBM. feature_fraction + bagging_fraction give each
    seed a genuinely different sample of rows and columns; without these,
    LGBM at fixed (max_depth, num_leaves, n_estimators) is fully
    deterministic and all seeds learn identical trees (the nb2430 leak)."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


# ---------------- copy of nb2430 ChEMBL kNN + feature helpers ----------------

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


def build_X_te_K20(n_test, te_smiles):
    """Rebuild the K=20 RFE-surviving feature matrix on the 513 test compounds."""
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
    assert X_te_full.shape[1] == 117
    return X_te_full[:, surviving_K20].astype(np.float32)


# ----------------------- audit: bag-std on test ------------------------------

def audit_seed_diversity(X_unb, residual, X_te):
    """Train ALL 253 -> predict test. This audit is in-sample, but the only
    purpose is to check whether seed diversity (feature/row subsampling)
    actually moves the trees. If bag_std is still ~0 with the new params,
    something is broken in LGBM and we must abort."""
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), X_te.shape[0]), dtype=np.float64)
    for i, s in enumerate(RESID_SEEDS):
        mdl = lgb.LGBMRegressor(**_lgbm_params_diverse(s))
        mdl.fit(X_unb.astype(np.float32), residual)
        per_seed_te_resid[i] = mdl.predict(X_te.astype(np.float32))
    return per_seed_te_resid


# --------------- HONEST per-outer-fold pseudo labels + cross-fit -------------

def cross_fit_honest_pseudo(
    X_unb, residual, X_te, te_anchor_513, unb_idx, anchor, splits, pseudo_w,
):
    """For each outer fold k:
        1. Compute pseudo labels for test using ONLY tr_loc rows
           (per-seed K=5; bag mean = pseudo residual; bag std = gate).
        2. Filter test to high-conf rows (bag_std < STD_GATE).
        3. Append to tr_loc with sample_weight = pseudo_w. Refit per seed.
        4. Predict residual on va_loc. Average over seeds.

    Returns:
        oof_full          (n_unb,)            anchor + cross-fit residual OOF
        n_pseudo_per_fold (n_folds,)          high-conf count per outer fold
        bag_std_per_fold  (n_folds, n_test)   bag std over seeds per fold
        deploy_te         (n_test,)           anchor + final deploy residual
                                              from one full-data refit per seed
                                              using pseudo labels generated by
                                              another set of cross-fit models
                                              (this stays honest at test time:
                                              pseudo labels never see val rows).
    """
    n_unb = len(residual)
    n_test = X_te.shape[0]
    n_seeds = len(RESID_SEEDS)
    oof_full = np.full(n_unb, np.nan, dtype=np.float64)

    n_pseudo_per_fold = np.zeros(len(splits), dtype=int)
    bag_std_per_fold = np.zeros((len(splits), n_test), dtype=np.float64)
    per_seed_va_pred_resid = []  # diagnostic

    for fold_k, (tr_loc, va_loc) in enumerate(splits):
        # ---- step A: per-seed pseudo labels for test, using ONLY tr_loc ----
        per_seed_te_resid_fold = np.zeros((n_seeds, n_test), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            mdl_p = lgb.LGBMRegressor(**_lgbm_params_diverse(s))
            mdl_p.fit(X_unb[tr_loc].astype(np.float32), residual[tr_loc])
            per_seed_te_resid_fold[i] = mdl_p.predict(X_te.astype(np.float32))
        bag_mean_fold = per_seed_te_resid_fold.mean(axis=0)
        bag_std_fold = per_seed_te_resid_fold.std(axis=0)
        bag_std_per_fold[fold_k] = bag_std_fold

        # ---- step B: high-conf subset ----
        mask = bag_std_fold < STD_GATE
        n_pseudo_per_fold[fold_k] = int(mask.sum())
        X_te_pseudo_k = X_te[mask]
        resid_pseudo_k = bag_mean_fold[mask]

        # ---- step C: per-seed augmented fit, predict val ----
        per_seed_va = np.zeros((n_seeds, len(va_loc)), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            X_tr = X_unb[tr_loc]
            y_tr = residual[tr_loc]
            if X_te_pseudo_k.shape[0] > 0:
                X_fit = np.vstack([X_tr, X_te_pseudo_k]).astype(np.float32)
                y_fit = np.concatenate([y_tr, resid_pseudo_k])
                w_fit = np.concatenate([
                    np.ones(len(tr_loc), dtype=np.float64),
                    np.full(X_te_pseudo_k.shape[0], pseudo_w, dtype=np.float64),
                ])
            else:
                X_fit = X_tr.astype(np.float32)
                y_fit = y_tr
                w_fit = np.ones(len(tr_loc), dtype=np.float64)
            mdl = lgb.LGBMRegressor(**_lgbm_params_diverse(s))
            mdl.fit(X_fit, y_fit, sample_weight=w_fit)
            per_seed_va[i] = mdl.predict(X_unb[va_loc].astype(np.float32))
        oof_full[va_loc] = per_seed_va.mean(axis=0)
        per_seed_va_pred_resid.append(per_seed_va.mean(axis=0))

    # -------- deploy te prediction (honest: pseudo labels via LOO style) ----
    # For the deploy refit, we still need pseudo labels for test. To stay
    # honest at test time, recompute pseudo labels ONCE per seed using only
    # the K=5 outer fold models built on tr_loc (per fold). Then average
    # those pseudo-label estimates across outer folds (each test row sees
    # bag std across (5 folds x 5 seeds) = 25 votes). Pseudo labels used
    # for the deploy refit therefore do NOT come from a model trained on
    # the full 253.
    deploy_pseudo = np.zeros(n_test, dtype=np.float64)
    deploy_pseudo_std = np.zeros(n_test, dtype=np.float64)
    pool_preds = np.zeros((len(splits) * n_seeds, n_test), dtype=np.float64)
    j = 0
    for fold_k, (tr_loc, _va_loc) in enumerate(splits):
        for i, s in enumerate(RESID_SEEDS):
            mdl_p = lgb.LGBMRegressor(**_lgbm_params_diverse(s))
            mdl_p.fit(X_unb[tr_loc].astype(np.float32), residual[tr_loc])
            pool_preds[j] = mdl_p.predict(X_te.astype(np.float32))
            j += 1
    deploy_pseudo = pool_preds.mean(axis=0)
    deploy_pseudo_std = pool_preds.std(axis=0)
    dep_mask = deploy_pseudo_std < STD_GATE
    n_pseudo_deploy = int(dep_mask.sum())
    X_te_dep_pseudo = X_te[dep_mask]
    resid_dep_pseudo = deploy_pseudo[dep_mask]

    per_seed_te_dep = np.zeros((n_seeds, n_test), dtype=np.float64)
    for i, s in enumerate(RESID_SEEDS):
        if X_te_dep_pseudo.shape[0] > 0:
            X_dep = np.vstack([X_unb, X_te_dep_pseudo]).astype(np.float32)
            y_dep = np.concatenate([residual, resid_dep_pseudo])
            w_dep = np.concatenate([
                np.ones(n_unb, dtype=np.float64),
                np.full(X_te_dep_pseudo.shape[0], pseudo_w, dtype=np.float64),
            ])
        else:
            X_dep = X_unb.astype(np.float32)
            y_dep = residual
            w_dep = np.ones(n_unb, dtype=np.float64)
        mdl_dep = lgb.LGBMRegressor(**_lgbm_params_diverse(s))
        mdl_dep.fit(X_dep, y_dep, sample_weight=w_dep)
        per_seed_te_dep[i] = mdl_dep.predict(X_te.astype(np.float32))
    deploy_te_resid = per_seed_te_dep.mean(axis=0)
    deploy_te = te_anchor_513 + deploy_te_resid

    return (oof_full,
            n_pseudo_per_fold,
            bag_std_per_fold,
            deploy_te,
            n_pseudo_deploy,
            deploy_pseudo_std)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- HONEST pseudo-labeling (per-fold OOF, seed-diverse)")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[anchor] chemprop_aux in_RAE = {rae_anchor:.4f}")

    print("[feat] rebuilding K=20 feature matrix on 513 test rows...")
    X_te_K20 = build_X_te_K20(n_test, te_smiles)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20={X_unb_K20.shape}  X_te_K20={X_te_K20.shape}")

    # --- scaffold KFold on 253 unblind ---
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_mols = [standardize(s) for s in unb_smiles]
    unb_scaffolds = []
    for m in unb_mols:
        if m is None:
            unb_scaffolds.append(None)
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            unb_scaffolds.append(sc if sc else None)
        except Exception:
            unb_scaffolds.append(None)
    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=RESID_FOLDS,
                                    shuffle=True, seed=42)
    print(f"[splits] scaffold KFold n_splits={RESID_FOLDS}, "
          f"fold sizes={[len(s[1]) for s in splits]}")

    # ============ AUDIT: seed-diversity check ============
    print("\n[audit] seed-diversity check on FULL-253-trained K=5 bag...")
    per_seed_te_full = audit_seed_diversity(X_unb_K20, residual, X_te_K20)
    bag_std_full = per_seed_te_full.std(axis=0)
    bag_mean_full = per_seed_te_full.mean(axis=0)
    audit_med = float(np.median(bag_std_full))
    audit_max = float(bag_std_full.max())
    audit_min = float(bag_std_full.min())
    audit_mean = float(bag_std_full.mean())
    seed_diversity_ok = audit_med > MIN_REAL_BAG_STD
    print(f"   bag_std_full: min={audit_min:.4f}  median={audit_med:.4f}  "
          f"max={audit_max:.4f}  mean={audit_mean:.4f}")
    print(f"   gate MIN_REAL_BAG_STD={MIN_REAL_BAG_STD} -> "
          f"diversity_ok={seed_diversity_ok}")
    if not seed_diversity_ok:
        print("   CRITICAL: seeds still collapse despite feature/row subsampling -- "
              "audit FAIL, no winner can be declared")

    # ============ per-weight HONEST cross-fit ============
    print("\n[xfit] honest per-fold cross-fit (no leak) across pseudo-weights")
    per_weight_results = []
    per_weight_oof_paths = {}
    per_weight_te_arrays = {}

    for w in PSEUDO_WEIGHTS:
        print(f"\n--- pseudo_w = {w:.2f} ---")
        ts = time.time()
        (oof_full,
         n_pseudo_per_fold,
         bag_std_per_fold,
         deploy_te,
         n_pseudo_deploy,
         deploy_pseudo_std) = cross_fit_honest_pseudo(
            X_unb_K20, residual, X_te_K20, te_anchor_513, unb_idx,
            anchor, splits, pseudo_w=w,
        )
        # add anchor onto OOF residual
        oof_full = anchor + oof_full
        rae_xfit = float(rae(y_unb, oof_full))
        delta = rae_xfit - NB2240_REF_OOF
        beat = delta < -GATE_MARGIN
        n_pseudo_total = int(n_pseudo_per_fold.sum())
        n_pseudo_mean = float(n_pseudo_per_fold.mean())
        fire_rate_mean = float(n_pseudo_per_fold.mean() / n_test)
        bag_std_med_per_fold = [float(np.median(bag_std_per_fold[k]))
                                for k in range(len(splits))]
        bag_std_med_overall = float(np.median(bag_std_per_fold))
        print(f"   per-fold n_pseudo={n_pseudo_per_fold.tolist()}  "
              f"mean_fire_rate={fire_rate_mean*100:.1f}%")
        print(f"   per-fold bag_std median={[round(x, 4) for x in bag_std_med_per_fold]}")
        print(f"   deploy n_pseudo (25-vote bag)={n_pseudo_deploy}  "
              f"deploy_bag_std_med={float(np.median(deploy_pseudo_std)):.4f}")
        print(f"   cross-fit RAE = {rae_xfit:.4f}  "
              f"delta vs nb2240 = {delta:+.4f}  gate-beat? {beat}")

        oof_path = DATA_PROCESSED / f"{TAG}_oof_w{w}.npy"
        np.save(oof_path, oof_full.astype(np.float32))
        per_weight_oof_paths[str(w)] = str(oof_path)
        per_weight_te_arrays[str(w)] = deploy_te.astype(np.float32)

        per_weight_results.append({
            "pseudo_w": w,
            "cross_fit_rae": rae_xfit,
            "delta_vs_nb2240": delta,
            "gate_beat_nb2240": bool(beat),
            "n_pseudo_per_fold": n_pseudo_per_fold.tolist(),
            "n_pseudo_mean_per_fold": n_pseudo_mean,
            "fire_rate_mean_per_fold": fire_rate_mean,
            "fire_rate_in_band": (HIGH_CONF_BAND[0] <= fire_rate_mean <= HIGH_CONF_BAND[1]),
            "bag_std_med_per_fold": bag_std_med_per_fold,
            "bag_std_med_overall": bag_std_med_overall,
            "deploy_n_pseudo_25vote": n_pseudo_deploy,
            "deploy_bag_std_med": float(np.median(deploy_pseudo_std)),
            "oof_path": str(oof_path),
            "wall_sec": round(time.time() - ts, 2),
        })

    # ============ pick best weight subject to audit pass ============
    sorted_results = sorted(per_weight_results, key=lambda r: r["cross_fit_rae"])
    best = sorted_results[0]
    best_w = best["pseudo_w"]
    best_rae = best["cross_fit_rae"]
    best_delta = best["delta_vs_nb2240"]
    best_gate = best["gate_beat_nb2240"]

    audit_pass = (
        seed_diversity_ok
        and best["fire_rate_in_band"]
        and best["bag_std_med_overall"] > MIN_REAL_BAG_STD
    )

    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    for r in per_weight_results:
        flag = "*" if r["pseudo_w"] == best_w else " "
        print(f"  {flag} pseudo_w={r['pseudo_w']:.2f}  RAE={r['cross_fit_rae']:.4f}  "
              f"delta_vs_nb2240={r['delta_vs_nb2240']:+.4f}  "
              f"fire_rate={r['fire_rate_mean_per_fold']*100:.1f}%  "
              f"std_med={r['bag_std_med_overall']:.4f}")
    print(f"\n  BEST weight: {best_w}  RAE={best_rae:.4f}  "
          f"delta_vs_nb2240={best_delta:+.4f}  gate-beat? {best_gate}")
    print(f"  audit_pass (diversity_ok && fire_in_band && bag_std>0.01): {audit_pass}")
    if best_gate and audit_pass:
        verdict = "BEATS_NB2240_AUDIT_PASS"
    elif best_gate and not audit_pass:
        verdict = "BEATS_NB2240_AUDIT_FAIL_DO_NOT_PROMOTE"
    elif abs(best_delta) <= GATE_MARGIN:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "WORSE_THAN_NB2240"
    print(f"  verdict: {verdict}")

    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    sub_csv = SUBMISSIONS / f"{TAG}_honest_pseudo.csv"
    if best_gate and audit_pass:
        te_best = per_weight_te_arrays[str(best_w)]
        np.save(te_path, te_best)
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_best,
        }).to_csv(sub_csv, index=False)
        print(f"[save] {te_path}")
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] gate or audit not passed -- no te_*.npy / submission CSV")

    summary = {
        "tag": TAG,
        "method": "honest_pseudo_label_per_fold_seed_diverse",
        "anchor": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "n_unb": n_unb,
        "n_te": n_test,
        "k20_feature_dim": int(X_unb_K20.shape[1]),
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "splits": "scaffold_kfold",
        "lgbm_diversity_params": {
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
        },
        "std_gate": STD_GATE,
        "min_real_bag_std": MIN_REAL_BAG_STD,
        "high_conf_band": list(HIGH_CONF_BAND),
        "audit_full_bag_std_summary": {
            "min": audit_min,
            "median": audit_med,
            "max": audit_max,
            "mean": audit_mean,
        },
        "seed_diversity_ok": bool(seed_diversity_ok),
        "pseudo_weights": PSEUDO_WEIGHTS,
        "per_weight_results": per_weight_results,
        "best_pseudo_w": best_w,
        "best_cross_fit_rae": best_rae,
        "best_delta_vs_nb2240": best_delta,
        "best_gate_beat_nb2240": bool(best_gate),
        "audit_pass": bool(audit_pass),
        "nb2240_ref_oof": NB2240_REF_OOF,
        "gate_margin": GATE_MARGIN,
        "verdict_vs_nb2240": verdict,
        "oof_paths_by_weight": per_weight_oof_paths,
        "te_npy_path": str(te_path) if (best_gate and audit_pass) else None,
        "submission_csv": str(sub_csv) if (best_gate and audit_pass) else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"\n=== {TAG} DONE  wall={time.time()-t0:.1f}s ===")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "seed_diversity_ok",
        "best_pseudo_w",
        "best_cross_fit_rae",
        "best_delta_vs_nb2240",
        "best_gate_beat_nb2240",
        "audit_pass",
        "verdict_vs_nb2240",
    ):
        print(f"  {k}: {res.get(k)}")
