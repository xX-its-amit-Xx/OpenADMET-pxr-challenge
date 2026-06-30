"""nb2241 -- VERIFY nb2231 K=20 RFE snapshot with FRESH kf_seeds.

HYPOTHESIS:
    nb2231 K=20 RFE snapshot reports mean_bag RAE = 0.4763, per_seed_mean
    RAE = 0.5068 across 5 KFold seeds {1001..1005}. The 5-seed sample is
    susceptible to the lucky-seed pattern seen at cycle 137 (nb2154/nb2156)
    where in-sample mean across small seed pools collapsed under larger
    fresh seed pools. This script re-runs the K=20 substrate with 30 FRESH
    kf_seeds {1116..1145} (orthogonal pool) and reports per-seed mean_bag /
    median_bag RAE, the under-dispersion ratio of the 5-seed std vs the
    30-seed std, and a PASS/FAIL gate.

GATE:
    PASS if:
        30_seed_mean_bag_mean <= 0.4763   (matches claim)
        AND
        30_seed_mean_bag_std  <= 0.020    (looser than the K=28 0.005 since
                                           K=20 has fewer features hence
                                           larger seed variance)
    If PASS: nb2231 K=20 is verified, can serve as substrate for nb2171 K=20
             rebuild.
    If FAIL: K=20 snapshot is lucky-seed; reject as nb2171 substrate.

PROTOCOL:
    1. Reconstruct the 117-col 5-way K-tuned matrix (same as nb2103/nb2231).
    2. Slice to the K=20 surviving_idx_in_117 list from nb2231_summary.json
       (exactly the 20 features that survived the RFE).
    3. Use chemprop_aux PRE-unblind te slice as anchor (same as nb2231).
    4. For each fresh kf_seed in {1116..1145}:
         - Run residual LGBM scaffold 5-fold cross-fit (single seed per fold)
         - Compute mean_bag = OOF residual prediction
         - RAE(y_unb, anchor + mean_bag)
         - Same for median_bag (same as mean_bag for single-seed -- record
           both for parity with claim record).
    5. Compare:
         5-seed claim mean: 0.5068 (per_seed_mean) / 0.4763 (mean_bag)
         5-seed claim std : 0.0131
         30-seed result   : mean_bag mean/std, median_bag mean/std
         under_disp_ratio = 5seed_std / 30seed_std
                            (>1.5 = 5-seed under-dispersed = LUCKY)
    6. Save scripts/nb2241_verify_K20.py + data/processed/nb2241_summary.json

REFERENCES:
    nb2231 K=20 claim: per_seed_mean=0.5068 std=0.0131  mean_bag=0.4763
    nb2156 verify-pattern precedent (5-seed -> mean_bag rerun)
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
from pxr.paths import DATA_PROCESSED

TAG = "nb2241"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

N_FOLDS = 5
KF_SEEDS_FRESH = list(range(1116, 1146))   # 30 fresh seeds
KF_SEEDS_ORIGINAL = [1001, 1002, 1003, 1004, 1005]   # nb2231 reference seeds

NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

# nb2231 claims
NB2231_K20_PER_SEED_MEAN = 0.5068488273628804
NB2231_K20_PER_SEED_STD = 0.013141860219421262
NB2231_K20_MEAN_BAG = 0.47629055379732593
NB2231_K20_MEDIAN_BAG = 0.48053404809414346

# Gate thresholds (per task spec)
GATE_MEAN_BAG_THRESHOLD = NB2231_K20_MEAN_BAG    # 0.4763
GATE_STD_THRESHOLD = 0.020                       # looser since K=20

# Substrate sources (same as nb2231)
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

CHEMPROP_AUX_REF = 0.6216


# -------------------------- helpers (mirror nb2231) --------------------------
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
    """Identical to nb2231."""
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
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}"
        )
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
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _scaffold_cv_one_seed(X: np.ndarray, residual: np.ndarray,
                          unb_scaffolds: list[str], kf_seed: int) -> np.ndarray:
    """One scaffold 5-fold cross-fit of residual LGBM; returns OOF residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- VERIFY nb2231 K=20 RFE snapshot (fresh kf_seeds)")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold")
    print(f"        original 5 seeds  = {KF_SEEDS_ORIGINAL}")
    print(f"        fresh   30 seeds = {KF_SEEDS_FRESH[0]}..{KF_SEEDS_FRESH[-1]}")
    print(f"        nb2231 K=20 claim: per_seed_mean = {NB2231_K20_PER_SEED_MEAN:.4f}  "
          f"std={NB2231_K20_PER_SEED_STD:.4f}")
    print(f"                            mean_bag     = {NB2231_K20_MEAN_BAG:.4f}")
    print(f"                            median_bag   = {NB2231_K20_MEDIAN_BAG:.4f}")
    print(f"        GATE: 30-seed mean_bag mean <= {GATE_MEAN_BAG_THRESHOLD:.4f}")
    print(f"              AND 30-seed mean_bag std <= {GATE_STD_THRESHOLD:.4f}")
    print("=" * 78)

    # ---- load nb2231 K=20 surviving idx (the exact 20-feature subset) ----
    if not NB2231_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2231_SUMMARY}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    snap20 = nb2231["snapshots"]["20"]
    k20_idx = list(snap20["surviving_idx_in_117"])
    k20_names = list(snap20["surviving_names"])
    if len(k20_idx) != 20:
        raise ValueError(f"K=20 subset has {len(k20_idx)} indices, expected 20")
    print(f"[load] K=20 surviving idx (20 of 117): "
          f"{k20_idx[:6]}...{k20_idx[-3:]}")

    # ---- load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in unb 253 = {n_unique_scaf}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- rebuild 117-col matrix (same as nb2231) ----
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

    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                       best_K_key="best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)

    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]

    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]

    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_unb_top = X_ap_te[unb_idx][:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_unb_top = X_maccs_te[unb_idx][:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_unb_top = X_mord_te[unb_idx][:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_unb_top = X_emb_te[unb_idx][:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_unb_top = X_av_te[unb_idx][:, top_avalon_bit_idx].astype(np.float32)

    # ---- ChEMBL kNN feature ----
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
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    X_unb = np.concatenate(
        [
            X_ap_unb_top,
            X_maccs_unb_top,
            X_mord_unb_top,
            X_emb_unb_top,
            X_av_unb_top,
            pred_chembl_unb.reshape(-1, 1),
            mean_sim_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_unb.shape[1]
    if feat_dim != 117:
        raise ValueError(f"feat_dim {feat_dim} != 117")
    print(f"\n   COMBINED 5-way K-tuned matrix: {X_unb.shape}")

    # Slice to K=20 surviving subset
    X_k20 = X_unb[:, k20_idx].astype(np.float32)
    print(f"   K=20 slice                 : {X_k20.shape}")

    # ---- ORIGINAL 5-seed check (reproduce nb2231 K=20 claim) ----
    print("\n" + "-" * 78)
    print("ORIGINAL 5 kf_seeds (reproduce nb2231 K=20 claim)")
    print("-" * 78)
    t_o = time.time()
    orig_per_seed_oof = np.zeros((len(KF_SEEDS_ORIGINAL), n_unb), dtype=np.float64)
    orig_per_seed_rae = []
    for i, s in enumerate(KF_SEEDS_ORIGINAL):
        oof_s = _scaffold_cv_one_seed(X_k20, residual, unb_scaffolds, s)
        orig_per_seed_oof[i] = oof_s
        r_s = float(rae(y_unb, anchor + oof_s))
        orig_per_seed_rae.append(r_s)
        print(f"   kf_seed={s}  rae={r_s:.4f}")
    orig_per_seed_mean = float(np.mean(orig_per_seed_rae))
    orig_per_seed_std = float(np.std(orig_per_seed_rae))
    orig_mean_bag = float(rae(y_unb, anchor + orig_per_seed_oof.mean(axis=0)))
    orig_median_bag = float(rae(y_unb, anchor + np.median(orig_per_seed_oof, axis=0)))
    print(f"   ORIG per_seed_mean = {orig_per_seed_mean:.4f}  "
          f"std = {orig_per_seed_std:.4f}")
    print(f"   ORIG mean_bag      = {orig_mean_bag:.4f}")
    print(f"   ORIG median_bag    = {orig_median_bag:.4f}")
    print(f"   wall = {time.time() - t_o:.1f}s")
    # claim_reproduces if within 0.0010 RAE
    claim_repro_per_seed = abs(orig_per_seed_mean - NB2231_K20_PER_SEED_MEAN) < 0.001
    claim_repro_mean_bag = abs(orig_mean_bag - NB2231_K20_MEAN_BAG) < 0.001
    print(f"   claim_repro_per_seed (within 1e-3) = {claim_repro_per_seed}")
    print(f"   claim_repro_mean_bag (within 1e-3) = {claim_repro_mean_bag}")

    # ---- FRESH 30-seed pool ----
    print("\n" + "-" * 78)
    print(f"FRESH {len(KF_SEEDS_FRESH)} kf_seeds (orthogonal pool 1116..1145)")
    print("-" * 78)
    t_f = time.time()
    fresh_per_seed_oof = np.zeros((len(KF_SEEDS_FRESH), n_unb), dtype=np.float64)
    fresh_per_seed_rae = []
    for i, s in enumerate(KF_SEEDS_FRESH):
        oof_s = _scaffold_cv_one_seed(X_k20, residual, unb_scaffolds, s)
        fresh_per_seed_oof[i] = oof_s
        r_s = float(rae(y_unb, anchor + oof_s))
        fresh_per_seed_rae.append(r_s)
        if (i + 1) % 6 == 0 or i == 0:
            print(f"   [{i+1:2d}/{len(KF_SEEDS_FRESH)}] kf_seed={s}  rae={r_s:.4f}")
    fresh_per_seed_mean = float(np.mean(fresh_per_seed_rae))
    fresh_per_seed_std = float(np.std(fresh_per_seed_rae))
    fresh_per_seed_min = float(np.min(fresh_per_seed_rae))
    fresh_per_seed_max = float(np.max(fresh_per_seed_rae))
    fresh_per_seed_median = float(np.median(fresh_per_seed_rae))
    fresh_mean_bag = float(rae(y_unb, anchor + fresh_per_seed_oof.mean(axis=0)))
    fresh_median_bag = float(rae(y_unb, anchor + np.median(fresh_per_seed_oof, axis=0)))
    print(f"   FRESH per_seed_mean   = {fresh_per_seed_mean:.4f}  "
          f"std = {fresh_per_seed_std:.4f}")
    print(f"   FRESH per_seed_median = {fresh_per_seed_median:.4f}")
    print(f"   FRESH per_seed_min..max = {fresh_per_seed_min:.4f} .. "
          f"{fresh_per_seed_max:.4f}")
    print(f"   FRESH mean_bag        = {fresh_mean_bag:.4f}")
    print(f"   FRESH median_bag      = {fresh_median_bag:.4f}")
    print(f"   wall = {time.time() - t_f:.1f}s")

    # ---- under-dispersion ratio ----
    # Compares the 5-seed std to the 30-seed std; >1.5x = ORIG was under-dispersed.
    if fresh_per_seed_std > 1e-9:
        under_disp_ratio_orig_claim_vs_fresh = (
            NB2231_K20_PER_SEED_STD / fresh_per_seed_std
        )
        under_disp_ratio_orig_repro_vs_fresh = (
            orig_per_seed_std / fresh_per_seed_std
        )
    else:
        under_disp_ratio_orig_claim_vs_fresh = float("inf")
        under_disp_ratio_orig_repro_vs_fresh = float("inf")
    # rough 30-seed mean SE
    fresh_mean_se = fresh_per_seed_std / np.sqrt(len(KF_SEEDS_FRESH))
    print(f"\n   under-dispersion ratio "
          f"(claim_std {NB2231_K20_PER_SEED_STD:.4f} / fresh_std "
          f"{fresh_per_seed_std:.4f}) = "
          f"{under_disp_ratio_orig_claim_vs_fresh:.3f}")
    print(f"   under-dispersion ratio "
          f"(repro_std {orig_per_seed_std:.4f} / fresh_std "
          f"{fresh_per_seed_std:.4f}) = "
          f"{under_disp_ratio_orig_repro_vs_fresh:.3f}")
    print(f"   30-seed mean SE        = {fresh_mean_se:.4f}")

    # ---- GATE ----
    gate_pass_mean = bool(fresh_mean_bag <= GATE_MEAN_BAG_THRESHOLD)
    gate_pass_std = bool(fresh_per_seed_std <= GATE_STD_THRESHOLD)
    gate_pass = bool(gate_pass_mean and gate_pass_std)

    # Also test per_seed_mean variant: if fresh per_seed_mean is within
    # 1 SE of original, the original is within Monte-Carlo noise.
    per_seed_mean_delta = fresh_per_seed_mean - NB2231_K20_PER_SEED_MEAN
    per_seed_mean_delta_within_2se = bool(
        abs(per_seed_mean_delta) <= 2 * fresh_mean_se
    )
    mean_bag_delta = fresh_mean_bag - NB2231_K20_MEAN_BAG

    if gate_pass:
        verdict = "PASS_K20_VERIFIED"
    else:
        reasons = []
        if not gate_pass_mean:
            reasons.append(
                f"fresh_mean_bag {fresh_mean_bag:.4f} > "
                f"{GATE_MEAN_BAG_THRESHOLD:.4f}"
            )
        if not gate_pass_std:
            reasons.append(
                f"fresh_std {fresh_per_seed_std:.4f} > "
                f"{GATE_STD_THRESHOLD:.4f}"
            )
        verdict = "FAIL_K20_LUCKY_SEED_PATTERN: " + " AND ".join(reasons)

    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   GATE_MEAN_BAG_THRESHOLD = {GATE_MEAN_BAG_THRESHOLD:.4f}  "
          f"(nb2231 K=20 mean_bag claim)")
    print(f"   GATE_STD_THRESHOLD      = {GATE_STD_THRESHOLD:.4f}")
    print(f"   fresh mean_bag          = {fresh_mean_bag:.4f}  "
          f"PASS_MEAN={gate_pass_mean}")
    print(f"   fresh per_seed_std      = {fresh_per_seed_std:.4f}  "
          f"PASS_STD={gate_pass_std}")
    print(f"   per_seed_mean delta     = {per_seed_mean_delta:+.4f}  "
          f"within_2SE={per_seed_mean_delta_within_2se}")
    print(f"   mean_bag delta          = {mean_bag_delta:+.4f}")
    print(f"\n   VERDICT = {verdict}")

    if "PASS" in verdict:
        print("   -> nb2231 K=20 is VERIFIED; can serve as substrate for "
              "nb2171 K=20 rebuild")
    else:
        print("   -> nb2231 K=20 shows lucky-seed pattern; REJECT as "
              "nb2171 substrate")

    # ---- save summary ----
    summary = {
        "tag": TAG,
        "verifies": "nb2231_K20_snapshot",
        "method": ("scaffold_cv_residual_lgbm_K20_on_117col_matrix_"
                   "fresh_30_kf_seeds_pool_1116_1145"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "n_unb": n_unb,
        "unb_n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),

        "K20_surviving_idx_in_117": [int(j) for j in k20_idx],
        "K20_surviving_names": k20_names,

        "kf_seeds_original": KF_SEEDS_ORIGINAL,
        "kf_seeds_fresh": KF_SEEDS_FRESH,
        "n_folds": N_FOLDS,

        "nb2231_claim": {
            "per_seed_mean": NB2231_K20_PER_SEED_MEAN,
            "per_seed_std": NB2231_K20_PER_SEED_STD,
            "mean_bag": NB2231_K20_MEAN_BAG,
            "median_bag": NB2231_K20_MEDIAN_BAG,
        },

        "original_reproduction": {
            "kf_seeds": KF_SEEDS_ORIGINAL,
            "per_seed_rae": orig_per_seed_rae,
            "per_seed_mean": orig_per_seed_mean,
            "per_seed_std": orig_per_seed_std,
            "mean_bag": orig_mean_bag,
            "median_bag": orig_median_bag,
            "claim_repro_per_seed_within_1e-3": bool(claim_repro_per_seed),
            "claim_repro_mean_bag_within_1e-3": bool(claim_repro_mean_bag),
        },

        "fresh_30seed": {
            "kf_seeds": KF_SEEDS_FRESH,
            "per_seed_rae": fresh_per_seed_rae,
            "per_seed_mean": fresh_per_seed_mean,
            "per_seed_std": fresh_per_seed_std,
            "per_seed_min": fresh_per_seed_min,
            "per_seed_max": fresh_per_seed_max,
            "per_seed_median": fresh_per_seed_median,
            "per_seed_mean_se": float(fresh_mean_se),
            "mean_bag": fresh_mean_bag,
            "median_bag": fresh_median_bag,
        },

        "under_dispersion_ratio_claim_std_vs_fresh_std":
            float(under_disp_ratio_orig_claim_vs_fresh),
        "under_dispersion_ratio_repro_std_vs_fresh_std":
            float(under_disp_ratio_orig_repro_vs_fresh),

        "delta_fresh_mean_bag_vs_claim": float(mean_bag_delta),
        "delta_fresh_per_seed_mean_vs_claim": float(per_seed_mean_delta),
        "fresh_per_seed_mean_within_2SE_of_claim":
            bool(per_seed_mean_delta_within_2se),

        "gate": {
            "mean_bag_threshold": GATE_MEAN_BAG_THRESHOLD,
            "std_threshold": GATE_STD_THRESHOLD,
            "fresh_mean_bag": fresh_mean_bag,
            "fresh_per_seed_std": fresh_per_seed_std,
            "pass_mean_bag": gate_pass_mean,
            "pass_std": gate_pass_std,
            "pass_overall": gate_pass,
        },
        "verdict": verdict,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
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
        "n_unb",
        "rae_anchor_chemprop_aux",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n  nb2231 claim                : "
          f"per_seed_mean={NB2231_K20_PER_SEED_MEAN:.4f}  "
          f"std={NB2231_K20_PER_SEED_STD:.4f}  "
          f"mean_bag={NB2231_K20_MEAN_BAG:.4f}")
    rep = res["original_reproduction"]
    print(f"  original 5-seed reproduction: "
          f"per_seed_mean={rep['per_seed_mean']:.4f}  "
          f"std={rep['per_seed_std']:.4f}  "
          f"mean_bag={rep['mean_bag']:.4f}  "
          f"repro={rep['claim_repro_per_seed_within_1e-3']}")
    f30 = res["fresh_30seed"]
    print(f"  fresh 30-seed pool          : "
          f"per_seed_mean={f30['per_seed_mean']:.4f}  "
          f"std={f30['per_seed_std']:.4f}  "
          f"mean_bag={f30['mean_bag']:.4f}  "
          f"median_bag={f30['median_bag']:.4f}")
    print(f"  under-dispersion ratio      : "
          f"claim/fresh = {res['under_dispersion_ratio_claim_std_vs_fresh_std']:.3f}  "
          f"repro/fresh = {res['under_dispersion_ratio_repro_std_vs_fresh_std']:.3f}")
    g = res["gate"]
    print(f"  GATE mean_bag <= {g['mean_bag_threshold']:.4f} : "
          f"{g['pass_mean_bag']}  "
          f"(fresh = {g['fresh_mean_bag']:.4f})")
    print(f"  GATE std      <= {g['std_threshold']:.4f} : "
          f"{g['pass_std']}  (fresh = {g['fresh_per_seed_std']:.4f})")
    print(f"  GATE overall                = {g['pass_overall']}")
    print(f"  VERDICT                     = {res['verdict']}")
