"""nb1112 -- Tier-routed Mixture-of-Experts by chemprop_aux predicted pEC50 bucket.

HYPOTHESIS:
    The K=28 residual LGBM (nb2103) yields a global mean-bag RAE of 0.4737.
    A tier-routed MoE specializes each LGBM to one pEC50 bucket -- low (<4.5),
    mid (4.5-5.5), high (>=5.5) -- using soft-weighted training (in-tier w=1.0,
    out-of-tier w=0.3). At inference, hard-route by anchor (chemprop_aux) tier.

    Specialization may help LOW (inactive plateau, conservative shrink) and
    HIGH (rare hits, decompression). MID is the dense centre where the global
    model is already strong, so it must continue to perform.

    Decision margin = 0.003 vs nb2103 K=28 mean-bag (0.4737); median (0.4698).
    Gate: 2 of 3 tier subset-RAEs must beat the corresponding tier-subset RAE
    of the global nb2103 K=28 cross-fit; AND overall mean-bag RAE must beat
    nb2103 by >=0.003.

PROTOCOL:
    1. Stage 1 ROUTER: chemprop_aux te[unb_idx] gives per-row predicted pEC50;
       map -> tier {low:<4.5, mid:[4.5, 5.5), high:>=5.5}.
    2. Stage 2 EXPERTS: rebuild the same 117-col 5-way K-tuned feature matrix
       as nb2103, slice to nb2103 K=28 top_idx. For each tier t in {low,mid,high}
       train an LGBM regressor on residual y_unb - anchor with
       sample_weight = 1.0 if tier(anchor)==t else 0.3. 5 seeds.
    3. Stage 3 INFERENCE: hard-route per row -- residual_hat = expert_{tier}.
       (Also compute soft-route diagnostic via simple Gaussian-bucket softmax
       on anchor distance to bucket centres.)
    4. 5-seed bag, 5-fold scaffold-equiv cross-fit on 253 unblind (KFold shuffle
       per seed, matching nb2103).
    5. Pooled mean-bag + median-bag RAE. Per-tier subset RAE. Compare to
       nb2103 K=28 baseline. Compare per-tier vs nb2103 K=28 per-tier subset RAE.

OUTPUTS:
    scripts/nb1112_tier_route.py
    data/processed/nb1112_summary.json
    data/processed/nb1112_mean_bag_oof_K28_tier.npy   (253,) float32
    submissions/nb1112_tier_route.csv                  (if gate passes)
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1112_tier_route"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Match nb2103 protocol exactly
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
K_SLICE = 28

# Tier definitions
TIER_EDGES = (4.5, 5.5)
TIER_NAMES = ["low", "mid", "high"]
TIER_CENTERS = np.array([4.0, 5.0, 6.0])   # for soft-route diagnostic
TIER_SIGMA = 0.5

# MoE sample weights
W_IN = 1.0
W_OUT = 0.3

# Reference benchmarks (nb2103 K=28)
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
NB2103_K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
DECISION_MARGIN = 0.003

# Feature build paths (reused from nb2103 / family-K winners)
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
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


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


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
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


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _extract_atompair_top_idx_from_nb1484(sum_1484) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def assign_tier(values: np.ndarray) -> np.ndarray:
    """Return integer tier id (0=low, 1=mid, 2=high) per row."""
    tier = np.full(len(values), 1, dtype=np.int32)
    tier[values < TIER_EDGES[0]] = 0
    tier[values >= TIER_EDGES[1]] = 2
    return tier


def soft_route_weights(values: np.ndarray) -> np.ndarray:
    """Per-row softmax(-(d**2)/(2*sigma**2)) over tier centres -- diagnostic."""
    d2 = (values[:, None] - TIER_CENTERS[None, :]) ** 2
    logits = -d2 / (2 * TIER_SIGMA ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    return w


def tier_safe_rae(y, p, mask):
    """Compute RAE on subset; return NaN if <2 rows or 0 variance."""
    if int(mask.sum()) < 2:
        return float("nan")
    yy = y[mask]
    pp = p[mask]
    if yy.std() < 1e-9:
        return float("nan")
    return float(rae(yy, pp))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- tier-routed MoE on K={K_SLICE} (chemprop_aux router)")
    print(f"          tiers: low<{TIER_EDGES[0]}  mid[{TIER_EDGES[0]},{TIER_EDGES[1]})  "
          f"high>={TIER_EDGES[1]}")
    print(f"          weights: in-tier={W_IN}  out-tier={W_OUT}  "
          f"seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean-bag={NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median-bag={NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    test_names = te["name"].values if "name" in te.columns \
        else te["Molecule Name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[anchor] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f}")
    print(f"[resid]  mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Tier assignment on anchor (unb subset and full 513) ----
    tier_unb = assign_tier(anchor)
    tier_513 = assign_tier(te_anchor_513)
    counts_unb = {n: int((tier_unb == i).sum()) for i, n in enumerate(TIER_NAMES)}
    counts_513 = {n: int((tier_513 == i).sum()) for i, n in enumerate(TIER_NAMES)}
    print(f"[tier] 253 counts: {counts_unb}")
    print(f"[tier] 513 counts: {counts_513}")

    # ---- Load K=28 X_unb sliced matrix from nb2103 cache ----
    X_unb_p = DATA_PROCESSED / "X_unb_28_nb2103.npy"
    if not X_unb_p.exists():
        raise FileNotFoundError(f"missing {X_unb_p} -- run nb2103 first")
    X_unb = np.load(X_unb_p).astype(np.float32)
    if X_unb.shape != (n_unb, K_SLICE):
        raise ValueError(f"X_unb shape {X_unb.shape} != ({n_unb},{K_SLICE})")
    print(f"[feat] X_unb (cached K=28) = {X_unb.shape}")

    # ---- Recover nb2103 K=28 top_K_idx_in_117 (needed to slice te-513) ----
    if not NB2103_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2103_SUMMARY}")
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    top_K_idx = None
    for r in nb2103_sum["per_K_records"]:
        if int(r["K"]) == K_SLICE:
            top_K_idx = np.array(r["top_K_idx_in_117"], dtype=int)
            break
    if top_K_idx is None:
        raise KeyError(f"K={K_SLICE} record missing from nb2103_summary.json")
    print(f"[feat] nb2103 K={K_SLICE} top_K_idx_in_117 loaded "
          f"(len={len(top_K_idx)})")

    # ---- Build 117-col te (513) feature matrix (mirror nb2103) ----
    print("\n" + "-" * 78)
    print("BUILD 117-col te-513 feature matrix (mirror nb2103)")
    print("-" * 78)
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

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx]
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx]
    X_mord_te = _load_mordred_test(n_test)[:, top_mord_col_idx]
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx]
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx]

    # ChEMBL kNN (same as nb2103)
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
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else ""
                       for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_117 = np.concatenate([
        X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
        pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
        mean_sim.reshape(-1, 1).astype(np.float32),
    ], axis=1)
    print(f"[feat] X_te_117 = {X_te_117.shape}")

    X_te_K = X_te_117[:, top_K_idx].astype(np.float32)
    print(f"[feat] X_te_K{K_SLICE} = {X_te_K.shape}")

    # Sanity: X_te_K[unb_idx] should match cached X_unb
    delta_cache = float(np.abs(X_te_K[unb_idx] - X_unb).max())
    print(f"[check] |X_te_K[unb_idx] - X_unb_28_nb2103|_max = {delta_cache:.6e}")
    if delta_cache > 1e-3:
        print(f"   [warn] sliced te-mat differs from nb2103 cache by "
              f"{delta_cache:.4f} -- proceeding (numeric drift OK)")

    # ---- Cross-fit MoE on 253 ----
    print("\n" + "-" * 78)
    print(f"5-SEED x {RESID_FOLDS}-FOLD CROSS-FIT MoE (tier-routed, hard)")
    print("-" * 78)

    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []

    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=s)
        resid_hat = np.full(n_unb, np.nan, dtype=np.float64)
        for fold_id, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
            X_tr = X_unb[tr_loc]
            y_tr_resid = residual[tr_loc]
            tier_tr = tier_unb[tr_loc]
            tier_va = tier_unb[va_loc]
            X_va = X_unb[va_loc]

            # Train 3 specialized experts on the TRAIN slice
            experts: dict[int, lgb.LGBMRegressor] = {}
            for t_id in range(3):
                sw = np.where(tier_tr == t_id, W_IN, W_OUT).astype(np.float32)
                # need at least 2 in-tier rows for meaningful spec
                if int((tier_tr == t_id).sum()) < 2:
                    # degenerate -- fall back to uniform weights
                    sw = np.ones_like(sw)
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_tr, y_tr_resid, sample_weight=sw)
                experts[t_id] = mdl

            # Hard-route val rows to their expert
            for t_id in range(3):
                mask = tier_va == t_id
                if mask.any():
                    resid_hat[va_loc[mask]] = experts[t_id].predict(X_va[mask])

        pred_corr_s = anchor + resid_hat
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        d_anchor = rae_s - rae_anchor
        d_nb2103 = rae_s - NB2103_K28_MEAN_BAG_REF
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": d_anchor,
            "delta_vs_nb2103_K28": d_nb2103,
            "resid_hat_std": float(resid_hat.std()),
            "resid_hat_mean": float(resid_hat.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:>3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_anchor = {d_anchor:+.4f}  d_nb2103 = {d_nb2103:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_arr = np.array(per_seed_rae)
    print(f"\n[bag] per-seed RAE  = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"[bag] per-seed mean = {per_seed_arr.mean():.4f}  "
          f"std = {per_seed_arr.std():.4f}")
    print(f"[bag] MEAN  bag RAE  = {rae_mean_bag:.4f}  "
          f"(d_vs_nb2103 = {rae_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"[bag] MEDIAN bag RAE = {rae_median_bag:.4f}  "
          f"(d_vs_nb2103 = {rae_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")

    # ---- Per-tier subset RAE (MoE vs nb2103 global) ----
    nb2103_oof = None
    if NB2103_K28_OOF_PATH.exists():
        nb2103_oof = np.load(NB2103_K28_OOF_PATH).astype(np.float64)
    else:
        print(f"[warn] nb2103 K=28 OOF cache missing -> tier compare degraded")

    per_tier_table = []
    n_tiers_improve = 0
    print("\n[tier] PER-TIER SUBSET-RAE TABLE  (mean-bag)")
    print(f"   {'tier':>5s}  {'n_unb':>5s}  {'RAE_moe':>9s}  {'RAE_nb2103':>11s}  "
          f"{'delta':>8s}  {'better?'}")
    for t_id, t_name in enumerate(TIER_NAMES):
        mask = tier_unb == t_id
        n_t = int(mask.sum())
        r_moe = tier_safe_rae(y_unb, mean_bag_oof, mask)
        if nb2103_oof is not None:
            r_nb = tier_safe_rae(y_unb, nb2103_oof, mask)
        else:
            r_nb = float("nan")
        if not (np.isnan(r_moe) or np.isnan(r_nb)):
            delta_t = r_moe - r_nb
            better = bool(r_moe < r_nb - DECISION_MARGIN)
            if better:
                n_tiers_improve += 1
        else:
            delta_t = float("nan")
            better = False
        per_tier_table.append({
            "tier_id": t_id,
            "tier_name": t_name,
            "n_unb": n_t,
            "rae_moe_mean_bag": r_moe,
            "rae_nb2103_K28": r_nb,
            "delta_moe_minus_nb2103": delta_t,
            "moe_better": better,
        })
        print(f"   {t_name:>5s}  {n_t:>5d}  {r_moe:>9.4f}  {r_nb:>11.4f}  "
              f"{delta_t:>+8.4f}  {('YES' if better else 'no')}")

    # ---- Gate evaluation ----
    overall_beats = rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    tier_gate = n_tiers_improve >= 2
    gate_pass = overall_beats and tier_gate
    print(f"\n[gate] overall mean-bag better than nb2103 by >={DECISION_MARGIN}: "
          f"{overall_beats}  ({rae_mean_bag:.4f} vs "
          f"{NB2103_K28_MEAN_BAG_REF:.4f})")
    print(f"[gate] tiers-improving >= 2 of 3: {tier_gate}  "
          f"({n_tiers_improve}/3)")
    print(f"[gate] GATE_PASS = {gate_pass}")

    if rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
        overall_verdict = "MOE_BEATS_NB2103_K28"
    elif abs(rae_mean_bag - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN:
        overall_verdict = "MOE_FLAT_VS_NB2103_K28"
    else:
        overall_verdict = "MOE_HURTS_VS_NB2103_K28"
    if not tier_gate:
        overall_verdict += "_TIER_GATE_FAILED"
    print(f"[verdict] {overall_verdict}")

    # ---- Diagnostic: soft-route on cross-fit OOF (no retrain) ----
    soft_w = soft_route_weights(anchor)
    print(f"\n[soft-route diag] mean soft-weights per tier = "
          f"{soft_w.mean(axis=0).round(3).tolist()}")

    # ---- Save mean-bag OOF ----
    out_oof = DATA_PROCESSED / f"{TAG.split('_')[0]}_mean_bag_oof_K{K_SLICE}_tier.npy"
    np.save(out_oof, mean_bag_oof.astype(np.float32))
    print(f"[save] {out_oof}")

    # ---- Deploy on 513 (only if gate passes) ----
    deploy_summary = None
    if gate_pass:
        print("\n" + "-" * 78)
        print("GATE PASSED -- building deploy 513 with all-253 trained experts")
        print("-" * 78)
        deploy_513_seeds = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            experts: dict[int, lgb.LGBMRegressor] = {}
            for t_id in range(3):
                sw_full = np.where(tier_unb == t_id, W_IN, W_OUT).astype(np.float32)
                if int((tier_unb == t_id).sum()) < 2:
                    sw_full = np.ones_like(sw_full)
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_unb, residual, sample_weight=sw_full)
                experts[t_id] = mdl
            resid_513 = np.zeros(n_test, dtype=np.float64)
            for t_id in range(3):
                mask = tier_513 == t_id
                if mask.any():
                    resid_513[mask] = experts[t_id].predict(X_te_K[mask])
            deploy_513_seeds[i] = te_anchor_513 + resid_513
        deploy_513 = deploy_513_seeds.mean(axis=0).astype(np.float32)
        deploy_in_rae = float(rae(y_unb, deploy_513[unb_idx].astype(np.float64)))
        plain = SUBMISSIONS / f"{TAG}.csv"
        pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": test_names,
            "pEC50": deploy_513,
        }).to_csv(plain, index=False)
        np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
        print(f"[deploy] te shape={deploy_513.shape}  "
              f"mean={deploy_513.mean():.3f}  std={deploy_513.std():.3f}")
        print(f"[deploy] in-sample 253 RAE = {deploy_in_rae:.4f}")
        print(f"[save] {plain}")
        deploy_summary = {
            "submission_csv": str(plain),
            "te_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
            "deploy_in_sample_rae_253": deploy_in_rae,
            "deploy_mean": float(deploy_513.mean()),
            "deploy_std": float(deploy_513.std()),
        }
    else:
        print("\n[deploy] GATE FAILED -- skipping deploy CSV build.")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "tier_routed_moe_K28_chemprop_aux_router_soft_weighted_train",
        "anchor": ANCHOR,
        "k_slice": K_SLICE,
        "tier_edges": list(TIER_EDGES),
        "tier_names": TIER_NAMES,
        "w_in_tier": W_IN,
        "w_out_tier": W_OUT,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "n_unb": n_unb,
        "n_test": n_test,
        "rae_anchor_chemprop_aux": rae_anchor,
        "tier_counts_unb_253": counts_unb,
        "tier_counts_test_513": counts_513,
        "soft_route_mean_weights_unb": soft_w.mean(axis=0).tolist(),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "per_seed_mean": float(per_seed_arr.mean()),
        "per_seed_std": float(per_seed_arr.std()),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb2103_K28": rae_mean_bag - NB2103_K28_MEAN_BAG_REF,
        "delta_median_bag_vs_nb2103_K28": rae_median_bag - NB2103_K28_MEDIAN_BAG_REF,
        "per_tier_table": per_tier_table,
        "tiers_improving": int(n_tiers_improve),
        "tier_gate_pass": bool(tier_gate),
        "overall_beats_nb2103": bool(overall_beats),
        "gate_pass": bool(gate_pass),
        "verdict": overall_verdict,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "deploy": deploy_summary,
        "x_unb_cache_delta_max": delta_cache,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG.split('_')[0]}_summary.json"
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
        "tier_edges", "tier_counts_unb_253", "tier_counts_test_513",
        "rae_anchor_chemprop_aux",
        "per_seed_rae", "per_seed_mean", "per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb2103_K28",
        "tiers_improving", "tier_gate_pass",
        "overall_beats_nb2103", "gate_pass", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-TIER TABLE ====")
    for row in res["per_tier_table"]:
        print(f"  tier={row['tier_name']:>5s}  n={row['n_unb']:>3d}  "
              f"RAE_moe={row['rae_moe_mean_bag']:.4f}  "
              f"RAE_nb2103={row['rae_nb2103_K28']:.4f}  "
              f"delta={row['delta_moe_minus_nb2103']:+.4f}  "
              f"better={row['moe_better']}")
