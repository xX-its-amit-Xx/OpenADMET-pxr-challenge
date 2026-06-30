"""nb2420 -- Transductive LGBM: include 513 test SMILES as unlabeled
augmentation during fit. Per memory: substrate-distinct paradigm; may break
the 14-cycle plateau.

CONTEXT:
    nb2240 (K=20 pyramid) holds cycle-best honest cross-fit RAE 0.4601 on
    the 253-unblind. All recent moves are post-hoc rescales of the same
    253-row residual learner. Transductive learning is a substrate change:
    LGBM splits become aware of where the 513 test compounds live in the
    K=20 feature space, even though their labels do not contribute to the
    loss (sample_weight=0).

    Mechanism: LGBM histogram bin edges are computed from ALL rows passed
    to .fit(), regardless of weight. Including the 513 test rows in the
    construction matrix shifts the split candidates toward feature-density
    that actually matters for the test distribution. This is the
    cheapest non-leakage way to make the learner "test-aware" without
    using any test labels.

PROTOCOL:
    1. Re-use nb2240's K=20 feature matrix on 513 + slice to 253-unb.
    2. Anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    3. For each seed s in {0,1,7,42,137}: KFold(5) split the 253; for each
       fold, build training matrix = [X_tr ; X_te (513)] with weights
       [1...1 ; 0...0] and residual targets [resid_tr ; zeros(513)];
       predict residual on X_va (253) and on X_te (513).
    4. Mean-bag the 5 seeds. Compare honest cross-fit RAE vs nb2240 K=20
       0.4601. Also run a parallel control without test augmentation
       (identical seeds/folds, weights all 1, no zero rows) to isolate
       the transductive effect.
    5. Save data/processed/nb2420_summary.json with both arms +
       te_nb2420.npy if it beats nb2240 by GATE_MARGIN 0.003.

Outputs:
    scripts/nb2420_transductive.py
    data/processed/nb2420_summary.json
    data/processed/nb2420_oof_transductive.npy  (253,)
    data/processed/nb2420_oof_control.npy       (253,)
    data/processed/te_nb2420.npy                (513,) only on gate
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
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2420"

# ---------------------- references / constants -------------------------------
NB2240_REF_OOF = 0.4601  # K=20 mean-bag honest cross-fit RAE
GATE_MARGIN = 0.003

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

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


# ----------- copy of nb2240 ChEMBL kNN + feature-loading helpers --------------

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


# ---------------- transductive vs control cross-fit ---------------------------

def cross_fit_arm(X_unb, residual, X_te, seed, transductive: bool):
    """KFold-5 cross-fit; return (oof_resid_253, te_resid_513).
    transductive=True: append all 513 test rows with sample_weight=0 to each
    fold's training matrix; the labels are dummies (zeros) and contribute
    nothing to the gradient, but they DO contribute to bin edges and the
    histogram construction.
    """
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        X_tr = X_unb[tr_loc]
        y_tr = residual[tr_loc]
        if transductive:
            X_fit = np.vstack([X_tr, X_te]).astype(np.float32)
            y_fit = np.concatenate([y_tr, np.zeros(X_te.shape[0], dtype=np.float64)])
            w_fit = np.concatenate([
                np.ones(len(tr_loc), dtype=np.float64),
                np.zeros(X_te.shape[0], dtype=np.float64),
            ])
        else:
            X_fit = X_tr.astype(np.float32)
            y_fit = y_tr
            w_fit = np.ones(len(tr_loc), dtype=np.float64)
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X_fit, y_fit, sample_weight=w_fit)
        oof[va_loc] = mdl.predict(X_unb[va_loc])
    # full-data deploy refit
    if transductive:
        X_dep = np.vstack([X_unb, X_te]).astype(np.float32)
        y_dep = np.concatenate([residual, np.zeros(X_te.shape[0], dtype=np.float64)])
        w_dep = np.concatenate([
            np.ones(n, dtype=np.float64),
            np.zeros(X_te.shape[0], dtype=np.float64),
        ])
    else:
        X_dep = X_unb.astype(np.float32)
        y_dep = residual
        w_dep = np.ones(n, dtype=np.float64)
    mdl_dep = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl_dep.fit(X_dep, y_dep, sample_weight=w_dep)
    te_resid = mdl_dep.predict(X_te).astype(np.float32)
    return oof, te_resid


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- transductive LGBM (test as unlabeled augmentation)")
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
    X_te_K20 = build_X_te_K20(n_test, te_smiles)  # (513, 20)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    print("\n" + "-" * 78)
    print("TRANSDUCTIVE arm (X_tr || X_te with sample_weight=0 on test rows)")
    print("-" * 78)
    per_seed_trans_oof = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_trans_te = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_trans_rae = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        oof_r, te_r = cross_fit_arm(X_unb_K20, residual, X_te_K20, s, transductive=True)
        per_seed_trans_oof[i] = anchor + oof_r
        per_seed_trans_te[i] = te_anchor_513 + te_r
        r = float(rae(y_unb, anchor + oof_r))
        per_seed_trans_rae.append(r)
        print(f"   seed={s:3d}  trans_RAE={r:.4f}  wall={time.time()-ts:.1f}s")

    trans_oof = per_seed_trans_oof.mean(axis=0)
    trans_te = per_seed_trans_te.mean(axis=0).astype(np.float32)
    trans_rae_mean_bag = float(rae(y_unb, trans_oof))
    trans_rae_per_seed_mean = float(np.mean(per_seed_trans_rae))
    print(f"[trans] per-seed mean RAE = {trans_rae_per_seed_mean:.4f}")
    print(f"[trans] mean-bag RAE      = {trans_rae_mean_bag:.4f}")

    print("\n" + "-" * 78)
    print("CONTROL arm (same seeds/folds, NO test augmentation)")
    print("-" * 78)
    per_seed_ctrl_oof = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_ctrl_te = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_ctrl_rae = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        oof_r, te_r = cross_fit_arm(X_unb_K20, residual, X_te_K20, s, transductive=False)
        per_seed_ctrl_oof[i] = anchor + oof_r
        per_seed_ctrl_te[i] = te_anchor_513 + te_r
        r = float(rae(y_unb, anchor + oof_r))
        per_seed_ctrl_rae.append(r)
        print(f"   seed={s:3d}  ctrl_RAE={r:.4f}  wall={time.time()-ts:.1f}s")

    ctrl_oof = per_seed_ctrl_oof.mean(axis=0)
    ctrl_te = per_seed_ctrl_te.mean(axis=0).astype(np.float32)
    ctrl_rae_mean_bag = float(rae(y_unb, ctrl_oof))
    ctrl_rae_per_seed_mean = float(np.mean(per_seed_ctrl_rae))
    print(f"[ctrl] per-seed mean RAE = {ctrl_rae_per_seed_mean:.4f}")
    print(f"[ctrl] mean-bag RAE      = {ctrl_rae_mean_bag:.4f}")

    delta_trans_vs_ctrl = trans_rae_mean_bag - ctrl_rae_mean_bag
    delta_trans_vs_nb2240 = trans_rae_mean_bag - NB2240_REF_OOF
    gate_beat_nb2240 = delta_trans_vs_nb2240 < -GATE_MARGIN

    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print(f"   transductive mean-bag RAE  = {trans_rae_mean_bag:.4f}")
    print(f"   control      mean-bag RAE  = {ctrl_rae_mean_bag:.4f}")
    print(f"   delta trans - ctrl          = {delta_trans_vs_ctrl:+.4f}")
    print(f"   nb2240 K=20 ref             = {NB2240_REF_OOF:.4f}")
    print(f"   delta trans - nb2240        = {delta_trans_vs_nb2240:+.4f}")
    verdict = (
        "BEATS_NB2240" if gate_beat_nb2240
        else ("FLAT_VS_NB2240" if abs(delta_trans_vs_nb2240) <= GATE_MARGIN
              else "WORSE_THAN_NB2240")
    )
    print(f"   verdict (margin 0.003)      = {verdict}")

    oof_trans_path = DATA_PROCESSED / f"{TAG}_oof_transductive.npy"
    oof_ctrl_path = DATA_PROCESSED / f"{TAG}_oof_control.npy"
    np.save(oof_trans_path, trans_oof.astype(np.float32))
    np.save(oof_ctrl_path, ctrl_oof.astype(np.float32))
    print(f"\n[save] {oof_trans_path}")
    print(f"[save] {oof_ctrl_path}")

    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    sub_csv = SUBMISSIONS / f"{TAG}_transductive.csv"
    if gate_beat_nb2240:
        np.save(te_path, trans_te)
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": trans_te,
        }).to_csv(sub_csv, index=False)
        print(f"[save] {te_path}")
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] gate not beat -- no te_*.npy or submission CSV written")

    summary = {
        "tag": TAG,
        "method": "transductive_lgbm_test_unlabeled_aug",
        "anchor": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "n_unb": n_unb,
        "n_te": n_test,
        "k20_feature_dim": int(X_unb_K20.shape[1]),
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "trans_per_seed_rae": per_seed_trans_rae,
        "trans_per_seed_mean_rae": trans_rae_per_seed_mean,
        "trans_mean_bag_rae": trans_rae_mean_bag,
        "ctrl_per_seed_rae": per_seed_ctrl_rae,
        "ctrl_per_seed_mean_rae": ctrl_rae_per_seed_mean,
        "ctrl_mean_bag_rae": ctrl_rae_mean_bag,
        "delta_trans_vs_ctrl": delta_trans_vs_ctrl,
        "nb2240_ref_oof": NB2240_REF_OOF,
        "delta_trans_vs_nb2240": delta_trans_vs_nb2240,
        "gate_margin": GATE_MARGIN,
        "gate_beat_nb2240": bool(gate_beat_nb2240),
        "verdict_vs_nb2240": verdict,
        "oof_transductive_path": str(oof_trans_path),
        "oof_control_path": str(oof_ctrl_path),
        "te_npy_path": str(te_path) if gate_beat_nb2240 else None,
        "submission_csv": str(sub_csv) if gate_beat_nb2240 else None,
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
        "trans_mean_bag_rae",
        "ctrl_mean_bag_rae",
        "delta_trans_vs_ctrl",
        "delta_trans_vs_nb2240",
        "verdict_vs_nb2240",
        "gate_beat_nb2240",
    ):
        print(f"  {k}: {res.get(k)}")
