"""nb1163_f2_blacklist -- Hard F2 scaffold blacklist (drop greasy-novel-inactive cluster).

HYPOTHESIS:
    The dominant Phase-1 failure mode (pm06) is F2: novel-scaffold inactives
    over-predicted by +1.23 RAE. ~172 TRAIN rows carry the F2 fingerprint
    (scaf_train_freq=0 within train AND counter-assay mismatch). Removing
    them from supervision should NOT damage scaffold-CV (the model wasn't
    learning anything generalisable from them anyway) and may free capacity
    for the in-manifold subspace.

PROTOCOL:
    1. Load TRAIN (4139); compute Bemis-Murcko scaffold (Phase-1 train-only).
    2. Identify F2 rows in TRAIN: scaffold appears <=1 time in train (i.e.
       singleton, the "no-support" proxy for scaf_train_freq=0 on test) AND
       has a counter-assay mismatch (|pEC50 - pEC50_null| < 0.3, indicating
       non-PXR-specific binding -- the greasy-promiscuous signature).
    3. Cap n_dropped at 200 (gate).
    4. Build nb2103-style features: top-K=28 SHAP-selected columns from the
       117-col 5-way K-tuned feature matrix (AtomPair / MACCS / Mordred /
       ChempropEmbed / Avalon + ChEMBL kNN). Top-28 SHAP idx comes from
       nb2063_shap_importance_full117.npy.
    5. Train LGBM(MSE) with nb2103 hyperparams (depth=4, num_leaves=15,
       n_est=300, lr=0.03, min_child=5, reg_lambda=2.0) on TRAIN excluding F2.
       5-fold scaffold CV on remaining train; record OOF RAE.
    6. Predict 513 test (no exclusion at inference) for deploy.
    7. Gate: scaffold-CV RAE <= 0.5027 AND n_dropped <= 200 AND test_std
       drop <= 0.05 (compared to baseline LGBM-no-blacklist test_std).
    8. If gate passes -> write submissions/nb1163_f2_blacklist.csv.

Outputs:
    data/processed/nb1163_summary.json
    data/processed/nb1163_f2_mask_train.npy        (4139,) bool
    data/processed/nb1163_te_pred.npy              (513,) float32
    data/processed/nb1163_oof_pred.npy             (n_kept,) float32
    submissions/nb1163_f2_blacklist.csv            (if gate passes)
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

from pxr.chem import standardize, bemis_murcko, morgan_fp_batch
from pxr.data import load_train, load_test, load_counter
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1163"
K_FEATURES = 28
N_FOLDS = 5
SEED = 42

# F2 identification thresholds
F2_SCAF_SINGLETON_THRESH = 1     # scaffold appears <= this many times in train
F2_COUNTER_DELTA_MAX = 0.3       # |pEC50 - pEC50_null| < this -> mismatch

# Gates
GATE_RAE = 0.5027
GATE_N_DROPPED = 200
GATE_TEST_STD_DROP = 0.05

# nb2103 LGBM hyperparams (verbatim)
LGBM_PARAMS = dict(
    objective="regression",
    max_depth=4,
    num_leaves=15,
    n_estimators=300,
    learning_rate=0.03,
    min_child_samples=5,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=2,
    verbosity=-1,
)

# Feature paths
ATOMPAIR_TR = DATA_PROCESSED / "tr_atompair.npy"
MACCS_TR = DATA_PROCESSED / "tr_maccs.npy"
AVALON_TR = DATA_PROCESSED / "tr_avalon512.npy"
EMBED_TR = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
ATOMPAIR_TE = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE = DATA_PROCESSED / "te_maccs.npy"
AVALON_TE = DATA_PROCESSED / "te_avalon512.npy"
EMBED_TE = DATA_PROCESSED / "te_chemprop_embed_300.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

NB1352_S = DATA_PROCESSED / "nb1352_summary.json"
NB1392_S = DATA_PROCESSED / "nb1392_summary.json"
NB1484_S = DATA_PROCESSED / "nb1484_summary.json"
NB1523_S = DATA_PROCESSED / "nb1523_summary.json"
NB1524_S = DATA_PROCESSED / "nb1524_summary.json"
NB1541_S = DATA_PROCESSED / "nb1541_summary.json"
NB2063_SHAP = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

# ChEMBL pool (same as nb2103)
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5


def _safe_ik(m):
    try:
        return Chem.MolToInchiKey(m) if m is not None else None
    except Exception:
        return None


def _safe_smi(m):
    try:
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def _load_chembl_pool() -> pd.DataFrame:
    frames = []
    for p, src in [
        (EXT_DIR / "chembl_pxr_CHEMBL3401.parquet", "CHEMBL3401_raw"),
        (EXT_DIR / "chembl_nr_extended.parquet", "nr_extended"),
        (EXT_DIR / "chembl_pxr_all_types.parquet", "pxr_all_types"),
    ]:
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if src == "CHEMBL3401_raw":
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
            d["pec50"] = 9.0 - np.log10(d["standard_value"].astype(float))
            d = d[["canonical_smiles", "pec50"]].rename(
                columns={"canonical_smiles": "smiles"}
            )
        elif src == "nr_extended":
            d = d[d["target_name"] == "PXR"].copy()
            d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
            d["pec50"] = d["pec50"].astype(float)
            d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
            d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
        else:
            d = d[d["target"] == "PXR"].copy()
            d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
            d["pec50"] = d["pec50"].astype(float)
            d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
            d = d[["smiles", "pec50"]]
        d["src"] = src
        frames.append(d)
    pool = pd.concat(frames, ignore_index=True)
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_ik)
    pool["std_smiles"] = mols.apply(_safe_smi)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"))
    )
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
        if w_sum[i] < 1e-6:
            pred[i] = fallback
        else:
            pred[i] = np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i]
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return pred, mean_sim


def _load_mordred(path):
    if not path.exists():
        raise FileNotFoundError(path)
    X = np.load(path).astype(np.float32)
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_npy(path, n_expected):
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} vs {n_expected}")
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair not found")


def _extract_best_K(s, key):
    bk = int(s["best_K"])
    for r in s[key]:
        if int(r["K"]) == bk:
            return r, bk
    raise KeyError(bk)


def _build_117(X_ap, X_maccs, X_mord, X_emb, X_av, pred_pec50, mean_sim,
               top_ap, top_maccs, top_mord, top_emb, top_av):
    parts = [
        X_ap[:, top_ap].astype(np.float32),
        X_maccs[:, top_maccs].astype(np.float32),
        X_mord[:, top_mord].astype(np.float32),
        X_emb[:, top_emb].astype(np.float32),
        X_av[:, top_av].astype(np.float32),
        pred_pec50.reshape(-1, 1).astype(np.float32),
        mean_sim.reshape(-1, 1).astype(np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- F2 BLACKLIST  (drop greasy-novel-inactive train rows)")
    print(f"          K_feat={K_FEATURES}  folds={N_FOLDS}  seed={SEED}")
    print(f"          gates: RAE<={GATE_RAE}  n_dropped<={GATE_N_DROPPED}  "
          f"test_std_drop<={GATE_TEST_STD_DROP}")
    print("=" * 78)

    # ---- 1. Load TRAIN + counter ----
    tr = load_train()
    n_tr = len(tr)
    y = tr["pec50"].astype(np.float64).to_numpy()
    smiles_tr = tr["smiles"].astype(str).tolist()
    names_tr = tr["name"].astype(str).tolist()
    print(f"[load] TRAIN: {n_tr} rows  pEC50 range [{y.min():.2f}, {y.max():.2f}]")

    # Scaffold via Bemis-Murcko
    print("[scaf] computing Bemis-Murcko scaffolds ...")
    scaffolds = [bemis_murcko(s) or "" for s in smiles_tr]
    scaf_counts: dict[str, int] = {}
    for s in scaffolds:
        scaf_counts[s] = scaf_counts.get(s, 0) + 1
    scaf_freq_per_row = np.array([scaf_counts[s] for s in scaffolds], dtype=int)
    n_singleton = int((scaf_freq_per_row <= F2_SCAF_SINGLETON_THRESH).sum())
    print(f"[scaf] unique scaffolds  : {len(scaf_counts)}")
    print(f"[scaf] singleton (<= {F2_SCAF_SINGLETON_THRESH}) rows : {n_singleton}")

    # ---- Counter-assay join ----
    counter = load_counter()
    cmap = dict(zip(counter["name"].astype(str), counter["pec50"].astype(float)))
    pec50_null = np.array([cmap.get(n, np.nan) for n in names_tr], dtype=np.float64)
    has_null = ~np.isnan(pec50_null)
    print(f"[ctr ] counter-assay overlap: {int(has_null.sum())}/{n_tr}")

    # F2 mismatch: |pEC50 - pEC50_null| < threshold  (counter-active = greasy)
    delta_null = np.full(n_tr, np.nan)
    delta_null[has_null] = np.abs(y[has_null] - pec50_null[has_null])
    counter_mismatch = (delta_null < F2_COUNTER_DELTA_MAX) & has_null

    # ---- 2. F2 mask = singleton AND counter-mismatch ----
    f2_mask = (scaf_freq_per_row <= F2_SCAF_SINGLETON_THRESH) & counter_mismatch
    n_dropped = int(f2_mask.sum())
    n_kept = n_tr - n_dropped
    print(f"[F2  ] n_singleton                   = {n_singleton}")
    print(f"[F2  ] n_counter_mismatch            = {int(counter_mismatch.sum())}")
    print(f"[F2  ] n_F2 (singleton & mismatch)   = {n_dropped}")
    print(f"[F2  ] n_kept                        = {n_kept}")
    if n_dropped > 0:
        print(f"[F2  ] dropped mean pEC50            = {y[f2_mask].mean():.3f}")
        print(f"[F2  ] kept    mean pEC50            = {y[~f2_mask].mean():.3f}")

    gate_n = (n_dropped <= GATE_N_DROPPED)
    print(f"[gate] n_dropped <= {GATE_N_DROPPED} ? {gate_n}")

    np.save(DATA_PROCESSED / f"{TAG}_f2_mask_train.npy", f2_mask)

    # ---- 3. Load 117-col feature recipe ----
    print("\n[feat] loading 5-way K-tuned feature recipe ...")
    sum_1352 = json.load(open(NB1352_S))
    sum_1392 = json.load(open(NB1392_S))
    sum_1484 = json.load(open(NB1484_S))
    sum_1523 = json.load(open(NB1523_S))
    sum_1524 = json.load(open(NB1524_S))
    sum_1541 = json.load(open(NB1541_S))

    top_maccs = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord, K_M = _extract_best_K(sum_1523, "per_K_records")
    top_mord = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap = _extract_atompair_top(sum_1484)
    K_AP = int(sum_1524["best_K"])
    top_ap = full_ap[:K_AP]
    K_E = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed = top_embed_full[:K_E]
    top_avalon = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)
    print(f"[feat] AtomPair {len(top_ap)}  MACCS {len(top_maccs)}  "
          f"Mordred {len(top_mord)}  Embed {len(top_embed)}  "
          f"Avalon {len(top_avalon)}")

    # ---- 4. Load TRAIN + TEST feature matrices ----
    te = load_test()
    n_te = len(te)
    print(f"[load] TEST: {n_te} rows")

    X_ap_tr = _load_npy(ATOMPAIR_TR, n_tr)
    X_maccs_tr = _load_npy(MACCS_TR, n_tr)
    X_av_tr = _load_npy(AVALON_TR, n_tr)
    X_emb_tr = _load_npy(EMBED_TR, n_tr)
    X_mord_tr = _load_mordred(MORDRED_DIR / "X_mordred_train.npy")
    assert X_mord_tr.shape[0] == n_tr

    X_ap_te = _load_npy(ATOMPAIR_TE, n_te)
    X_maccs_te = _load_npy(MACCS_TE, n_te)
    X_av_te = _load_npy(AVALON_TE, n_te)
    X_emb_te = _load_npy(EMBED_TE, n_te)
    X_mord_te = _load_mordred(MORDRED_DIR / "X_mordred_test.npy")
    assert X_mord_te.shape[0] == n_te

    # ---- 5. ChEMBL kNN feature (compute for train AND test) ----
    print("[chembl] building external pool ...")
    pool = _load_chembl_pool()
    # drop test InChIKey leakage
    te_smiles = (te["smiles"] if "smiles" in te.columns
                 else te["SMILES"]).astype(str).tolist()
    te_mols = [standardize(s) for s in te_smiles]
    te_iks = {ik for ik in (_safe_ik(m) for m in te_mols) if ik is not None}
    pool = pool[~pool["inchikey"].isin(te_iks)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep = fp_pool.sum(axis=1) > 0
    if not keep.all():
        pool = pool[keep].reset_index(drop=True)
        fp_pool = fp_pool[keep]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"[chembl] pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")

    # train kNN feature
    tr_mols = [standardize(s) for s in smiles_tr]
    std_tr = [(_safe_smi(m) or "") for m in tr_mols]
    fp_tr_morgan = morgan_fp_batch(std_tr)
    ti, ts = _tanimoto_topk(fp_tr_morgan, fp_pool, k=KNN_K)
    pred_chembl_tr, mean_sim_tr = _knn_predict(ti, ts, pool_labels, pool_median)

    # test kNN feature
    std_te = [(_safe_smi(m) or "") for m in te_mols]
    fp_te_morgan = morgan_fp_batch(std_te)
    ti, ts = _tanimoto_topk(fp_te_morgan, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(ti, ts, pool_labels, pool_median)

    # ---- 6. Build 117-col matrices ----
    X_tr_117 = _build_117(X_ap_tr, X_maccs_tr, X_mord_tr, X_emb_tr, X_av_tr,
                          pred_chembl_tr, mean_sim_tr,
                          top_ap, top_maccs, top_mord, top_embed, top_avalon)
    X_te_117 = _build_117(X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
                          pred_chembl_te, mean_sim_te,
                          top_ap, top_maccs, top_mord, top_embed, top_avalon)
    print(f"[feat] X_tr_117 = {X_tr_117.shape}  X_te_117 = {X_te_117.shape}")

    # ---- 7. SHAP top-K=28 slice ----
    shap_imp = np.load(NB2063_SHAP).astype(np.float32)
    if shap_imp.shape[0] != X_tr_117.shape[1]:
        raise ValueError(f"SHAP {shap_imp.shape} vs feat {X_tr_117.shape[1]}")
    rank_order = np.argsort(-shap_imp).astype(np.int32)
    topK_idx = rank_order[:K_FEATURES]
    X_tr_K = X_tr_117[:, topK_idx].astype(np.float32)
    X_te_K = X_te_117[:, topK_idx].astype(np.float32)
    print(f"[feat] X_tr_K = {X_tr_K.shape}  X_te_K = {X_te_K.shape}")

    # ---- 8. Baseline (no blacklist) deploy test prediction for test_std anchor ----
    print("\n[base] baseline LGBM (no F2 blacklist) -> test_std anchor ...")
    mdl_base = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_base.fit(X_tr_K, y)
    te_pred_base = mdl_base.predict(X_te_K)
    test_std_base = float(np.std(te_pred_base))
    print(f"[base] test_std (no blacklist) = {test_std_base:.4f}")

    # ---- 9. Scaffold 5-fold CV on KEPT subset ----
    keep_idx = np.where(~f2_mask)[0]
    scaffolds_kept = [scaffolds[i] for i in keep_idx]
    y_kept = y[keep_idx]
    X_kept = X_tr_K[keep_idx]
    print(f"\n[cv  ] scaffold 5-fold on {len(keep_idx)} kept rows ...")
    folds = scaffold_kfold_indices(scaffolds_kept, n_splits=N_FOLDS,
                                    seed=SEED)
    oof = np.full(len(keep_idx), np.nan, dtype=np.float64)
    fold_raes = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_kept[tr_idx], y_kept[tr_idx])
        oof[va_idx] = mdl.predict(X_kept[va_idx])
        r_va = float(rae(y_kept[va_idx], oof[va_idx]))
        fold_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_idx):4d}  n_va={len(va_idx):4d}  "
              f"RAE={r_va:.4f}")
    rae_cv = float(rae(y_kept, oof))
    print(f"\n[cv  ] pooled scaffold CV RAE (kept) = {rae_cv:.4f}  "
          f"(per-fold {min(fold_raes):.4f}--{max(fold_raes):.4f})")

    np.save(DATA_PROCESSED / f"{TAG}_oof_pred.npy", oof.astype(np.float32))

    gate_rae = (rae_cv <= GATE_RAE)
    print(f"[gate] RAE <= {GATE_RAE} ? {gate_rae}  (margin {rae_cv - GATE_RAE:+.4f})")

    # ---- 10. Deploy: full refit on KEPT, predict 513 ----
    print("\n[depl] refit on kept-only train, predict 513 ...")
    mdl_full = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_full.fit(X_kept, y_kept)
    te_pred = mdl_full.predict(X_te_K).astype(np.float32)
    test_std_kept = float(np.std(te_pred))
    test_std_drop = test_std_base - test_std_kept
    print(f"[depl] test_std (kept-only) = {test_std_kept:.4f}  "
          f"(drop vs base = {test_std_drop:+.4f})")
    gate_std = (test_std_drop <= GATE_TEST_STD_DROP)
    print(f"[gate] test_std_drop <= {GATE_TEST_STD_DROP} ? {gate_std}")
    np.save(DATA_PROCESSED / f"{TAG}_te_pred.npy", te_pred)

    # ---- 11. Overall verdict ----
    all_pass = bool(gate_rae and gate_n and gate_std)
    verdict = "PASS_ALL_GATES" if all_pass else "FAIL_AT_LEAST_ONE_GATE"
    if not gate_rae:
        verdict += "_RAE"
    if not gate_n:
        verdict += "_NDROP"
    if not gate_std:
        verdict += "_TESTSTD"

    print(f"\n[VERDICT] {verdict}")

    # ---- 12. Write CSV only if pass ----
    csv_path = None
    if all_pass:
        sub_dir = Path(__file__).resolve().parents[1] / "submissions"
        sub_dir.mkdir(parents=True, exist_ok=True)
        csv_path = sub_dir / f"{TAG}_f2_blacklist.csv"
        out_df = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": (te["name"] if "name" in te.columns
                              else te["Molecule Name"]).astype(str).tolist(),
            "pEC50": te_pred.astype(float),
        })
        out_df.to_csv(csv_path, index=False)
        print(f"[write] {csv_path}")

    # ---- 13. Summary ----
    summary = {
        "tag": TAG,
        "method": "f2_blacklist_lgbm_K28_scaf5fold",
        "K_features": K_FEATURES,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "n_dropped_f2": int(n_dropped),
        "n_kept": int(n_kept),
        "n_singleton_scaf": int(n_singleton),
        "n_counter_mismatch": int(counter_mismatch.sum()),
        "scaf_singleton_thresh": int(F2_SCAF_SINGLETON_THRESH),
        "counter_delta_max": float(F2_COUNTER_DELTA_MAX),
        "scaffold_cv_rae_kept": float(rae_cv),
        "fold_raes": [float(r) for r in fold_raes],
        "test_std_baseline": float(test_std_base),
        "test_std_kept": float(test_std_kept),
        "test_std_drop": float(test_std_drop),
        "gate_rae_max": float(GATE_RAE),
        "gate_n_dropped_max": int(GATE_N_DROPPED),
        "gate_test_std_drop_max": float(GATE_TEST_STD_DROP),
        "pass_rae": bool(gate_rae),
        "pass_n_dropped": bool(gate_n),
        "pass_test_std": bool(gate_std),
        "pass_all_gates": bool(all_pass),
        "verdict": verdict,
        "csv_path": str(csv_path) if csv_path is not None else None,
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
        "n_dropped_f2", "n_kept", "n_singleton_scaf", "n_counter_mismatch",
        "scaffold_cv_rae_kept", "test_std_baseline", "test_std_kept",
        "test_std_drop", "pass_rae", "pass_n_dropped", "pass_test_std",
        "pass_all_gates", "verdict", "csv_path",
    ):
        print(f"  {k}: {res.get(k)}")
