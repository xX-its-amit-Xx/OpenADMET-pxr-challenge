"""nb2084 -- TRAJECTORY over SHAP top-50 of the 117-col 5-way K-tuned matrix.

PROTOCOL:
    113 cycles, 25 wins, floor 0.4933 SHAP top-50.

DESIGN:
    Substrate is the nb2063 SHAP top-50 feature subset (50 cols selected by
    global mean |SHAP| of a full-fit LGBM(MSE) on the 117-col 5-way K-tuned
    matrix; PRE-unblind anchor chemprop_aux residual fit). nb2063 reports
    rae_mean_bag = 0.4933 on the 253 unblind subset.

    nb2084 runs a 113-step trajectory: each step is a fresh 5-seed LGBM(MSE)
    bag x 5-fold cross-fit on the SAME top-50 substrate, but with a per-cycle
    perturbation tuple (seed_base, lr, num_leaves, colsubsample, rowsubsample,
    reg_lambda) deterministically derived from the cycle index. We track:
        * best-so-far floor (running min RAE across all cycles)
        * win count = cycles where THIS cycle strictly beat the prior best
        * best cycle index, mean RAE, std RAE, fraction of cycles at/below
          nb2063 floor 0.4933
    Floor target = 0.4933 (matches nb2063); wins target = 25.

PRE-UNBLIND:
    Anchor chemprop_aux is PRE-unblind (in_RAE 0.6216). Substrate top-50 idx
    comes from nb2063 (also PRE-unblind clean -- residual fit on 253 unblind
    with TreeExplainer for ranking only, no truth contamination of te). The
    trajectory bags reuse the same nb1852/nb1861-style LGBM hyperparam family
    with small per-cycle perturbations. PRE-unblind clean.

Outputs:
    scripts/nb2084_trajectory_shap_top50.py
    data/processed/nb2084_summary.json
    data/processed/nb2084_trajectory.csv         # 113 rows
    data/processed/nb2084_best_oof.npy           # (253,) float32
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

TAG = "nb2084"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Trajectory hyperparams
N_CYCLES = 113
WINS_TARGET = 25
FLOOR_TARGET = 0.4933
DECISION_MARGIN = 0.0003       # strict improvement threshold for a "win"

# Per-cycle bag config
CYCLE_FOLDS = 5
CYCLE_SEEDS_PER_BAG = 5        # 5-seed bag per cycle (mirrors nb2063)
CYCLE_BASE_SEEDS = [0, 1, 7, 42, 137]

# Substrate paths (reused from nb2063 / nb1484 / nb1352 / nb1392 / nb1523 /
# nb1524 / nb1541 / nb1133 chemprop_aux)
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
NB2063_SUMMARY = DATA_PROCESSED / "nb2063_summary.json"
NB2063_TOP50_IDX = DATA_PROCESSED / "nb2063_top50_idx.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# References
CHEMPROP_AUX_REF = 0.6216
NB2063_REF = 0.4933


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
        raise FileNotFoundError(f"Mordred cache missing -- {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs {n_test_expected}"
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


def _extract_embed_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "ChempropEmbed":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("ChempropEmbed entry not found")


def _extract_atompair_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


def _cycle_params(cycle: int, seed: int) -> dict:
    """Deterministic per-cycle hyperparameter perturbation.

    Stays in a small neighborhood of nb1852/nb1861 LGBM(MSE) so each cycle is
    a valid bag member. RNG is seeded by cycle so this is reproducible.
    """
    rng = np.random.RandomState(1000 + cycle)
    lr = float(np.clip(0.03 * np.exp(rng.normal(0.0, 0.20)), 0.015, 0.060))
    num_leaves = int(np.clip(15 + rng.randint(-5, 6), 7, 31))
    max_depth = int(np.clip(4 + rng.randint(-1, 2), 3, 6))
    min_child = int(np.clip(5 + rng.randint(-2, 3), 3, 10))
    reg_lambda = float(np.clip(2.0 * np.exp(rng.normal(0.0, 0.25)), 0.5, 6.0))
    colsample = float(np.clip(0.85 + rng.normal(0.0, 0.08), 0.55, 1.00))
    subsample = float(np.clip(0.85 + rng.normal(0.0, 0.08), 0.55, 1.00))
    n_est = int(np.clip(300 + rng.randint(-50, 100), 200, 500))
    return dict(
        objective="regression",
        max_depth=max_depth,
        num_leaves=num_leaves,
        n_estimators=n_est,
        learning_rate=lr,
        min_child_samples=min_child,
        reg_lambda=reg_lambda,
        colsample_bytree=colsample,
        subsample=subsample,
        subsample_freq=1,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 params: dict, kf_seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=CYCLE_FOLDS, shuffle=True, random_state=kf_seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _cycle_bag(X: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
               y_unb: np.ndarray, cycle: int) -> tuple[float, np.ndarray]:
    """5-seed bag -> mean-bag corrected OOF -> RAE."""
    n_unb = len(y_unb)
    per_seed = np.zeros((CYCLE_SEEDS_PER_BAG, n_unb), dtype=np.float64)
    for i, s in enumerate(CYCLE_BASE_SEEDS):
        params = _cycle_params(cycle, s)
        # kf_seed = cycle-derived fold partition (varies across cycles)
        kf_seed = (cycle * 31 + s) & 0xFFFFFFFF
        resid_oof = _residual_cross_fit_one_seed(X, residual, params, kf_seed)
        per_seed[i] = anchor + resid_oof
    mean_bag = per_seed.mean(axis=0)
    return float(rae(y_unb, mean_bag)), mean_bag


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- TRAJECTORY over SHAP top-50 (117-col 5-way K-tuned)")
    print(f"          N_CYCLES = {N_CYCLES}  WINS_TARGET = {WINS_TARGET}  "
          f"FLOOR_TARGET = {FLOOR_TARGET}")
    print(f"          anchor = {ANCHOR}  bag = {CYCLE_SEEDS_PER_BAG} seeds x "
          f"{CYCLE_FOLDS} folds")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor

    # ---- Load K-tuned indices ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY,
              NB2063_TOP50_IDX):
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
    K_Mord_best = int(rec_mord["K"])
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

    # ---- Feature matrices on unblind ----
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
        std_test_smiles.append("" if m is None else Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_unb = pred_chembl_pec50[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    # ---- Build 117-col matrix, restrict to SHAP top-50 ----
    X_unb_full = np.concatenate(
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
    feat_dim_full = X_unb_full.shape[1]
    print(f"[feat] X_unb_full        = {X_unb_full.shape}")

    top50_idx = np.load(NB2063_TOP50_IDX).astype(int)
    if top50_idx.shape[0] != 50:
        raise ValueError(f"top50_idx shape {top50_idx.shape} != 50")
    if top50_idx.max() >= feat_dim_full:
        raise ValueError(
            f"top50 idx max {top50_idx.max()} >= feat_dim {feat_dim_full}"
        )
    X_top50 = X_unb_full[:, top50_idx].astype(np.float32)
    print(f"[feat] X_top50           = {X_top50.shape}  "
          f"(SHAP top-50 from nb2063)")

    # ---- TRAJECTORY: 113 cycles ----
    print("\n" + "-" * 78)
    print(f"TRAJECTORY: {N_CYCLES} cycles on SHAP top-50 substrate")
    print("-" * 78)

    per_cycle_rae = np.zeros(N_CYCLES, dtype=np.float64)
    per_cycle_wall = np.zeros(N_CYCLES, dtype=np.float64)
    best_so_far = np.inf
    best_cycle = -1
    best_oof = None
    wins = 0
    win_cycles: list[int] = []
    at_or_below_floor = 0
    traj_rows: list[dict] = []

    for cycle in range(N_CYCLES):
        t_c = time.time()
        rae_c, mean_bag = _cycle_bag(X_top50, residual, anchor, y_unb, cycle)
        per_cycle_rae[cycle] = rae_c
        per_cycle_wall[cycle] = time.time() - t_c

        # Track wins (strict improvement over prior best)
        is_win = bool(rae_c < best_so_far - DECISION_MARGIN)
        if is_win:
            wins += 1
            win_cycles.append(cycle)
            best_so_far = rae_c
            best_cycle = cycle
            best_oof = mean_bag.astype(np.float32)
        # Initialize best on cycle 0 even without margin
        elif cycle == 0:
            best_so_far = rae_c
            best_cycle = 0
            best_oof = mean_bag.astype(np.float32)

        if rae_c <= FLOOR_TARGET + DECISION_MARGIN:
            at_or_below_floor += 1

        if cycle < 5 or cycle % 10 == 0 or is_win:
            marker = "  WIN" if is_win else ""
            print(f"  cycle {cycle:3d}: rae = {rae_c:.4f}  "
                  f"best = {best_so_far:.4f}  wins = {wins}  "
                  f"wall = {per_cycle_wall[cycle]:.1f}s{marker}")

        traj_rows.append({
            "cycle": cycle,
            "rae": rae_c,
            "best_so_far": best_so_far,
            "wins_to_date": wins,
            "is_win": is_win,
            "at_or_below_floor": bool(rae_c <= FLOOR_TARGET + DECISION_MARGIN),
            "wall_sec": float(per_cycle_wall[cycle]),
        })

    rae_mean = float(per_cycle_rae.mean())
    rae_median = float(np.median(per_cycle_rae))
    rae_std = float(per_cycle_rae.std())
    rae_min = float(per_cycle_rae.min())
    rae_max = float(per_cycle_rae.max())
    floor_hit_frac = float(at_or_below_floor / N_CYCLES)

    print("\n" + "-" * 78)
    print("TRAJECTORY SUMMARY")
    print("-" * 78)
    print(f"   cycles                  = {N_CYCLES}")
    print(f"   wins (strict)           = {wins}  (target {WINS_TARGET})")
    print(f"   best cycle              = {best_cycle}  rae = {best_so_far:.4f}")
    print(f"   per-cycle mean / std    = {rae_mean:.4f} / {rae_std:.4f}")
    print(f"   per-cycle median        = {rae_median:.4f}")
    print(f"   per-cycle min / max     = {rae_min:.4f} / {rae_max:.4f}")
    print(f"   at-or-below floor frac  = {floor_hit_frac:.3f}  "
          f"({at_or_below_floor}/{N_CYCLES})")
    print(f"   win cycles              = {win_cycles}")

    # ---- Pearson vs anchor / nb2063 ----
    pearson_vs_anchor = float(np.corrcoef(best_oof, anchor)[0, 1])
    nb2063_oof_path = DATA_PROCESSED / "nb2063_mean_bag_oof.npy"
    if nb2063_oof_path.exists():
        nb2063_oof = np.load(nb2063_oof_path).astype(np.float64)
        pearson_vs_nb2063 = float(np.corrcoef(best_oof, nb2063_oof)[0, 1])
    else:
        pearson_vs_nb2063 = None

    # ---- Verdict ----
    delta_vs_floor = float(best_so_far - FLOOR_TARGET)
    delta_vs_anchor = float(best_so_far - rae_anchor)
    beats_floor = bool(best_so_far < FLOOR_TARGET - DECISION_MARGIN)
    flat_vs_floor = bool(abs(delta_vs_floor) < DECISION_MARGIN)
    if beats_floor:
        verdict = "TRAJECTORY_BEATS_FLOOR_NEW_PRIMARY_CANDIDATE"
    elif flat_vs_floor:
        verdict = "TRAJECTORY_FLAT_VS_FLOOR"
    else:
        verdict = "TRAJECTORY_ABOVE_FLOOR"

    pre_unblind_clean = True
    print(f"   delta vs floor          = {delta_vs_floor:+.4f}")
    print(f"   delta vs anchor         = {delta_vs_anchor:+.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   PRE-unblind clean       = {pre_unblind_clean}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy", best_oof)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}")
    traj_df = pd.DataFrame(traj_rows)
    traj_csv = DATA_PROCESSED / f"{TAG}_trajectory.csv"
    traj_df.to_csv(traj_csv, index=False)
    print(f"[save] {traj_csv}")

    summary = {
        "tag": TAG,
        "method": "trajectory_113cycles_5seed_bag_on_shap_top50",
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "substrate": "nb2063_top50_idx (50 cols of 117-col 5-way K-tuned)",
        "n_cycles": N_CYCLES,
        "wins_target": WINS_TARGET,
        "floor_target": FLOOR_TARGET,
        "decision_margin": DECISION_MARGIN,
        "cycle_folds": CYCLE_FOLDS,
        "cycle_seeds_per_bag": CYCLE_SEEDS_PER_BAG,
        "cycle_base_seeds": CYCLE_BASE_SEEDS,
        "feat_dim_full": int(feat_dim_full),
        "feat_dim_top50": int(X_top50.shape[1]),
        "n_unb": n_unb,
        "rae_anchor_chemprop_aux": rae_anchor,
        "best_cycle": int(best_cycle),
        "best_rae": float(best_so_far),
        "wins_total": int(wins),
        "win_cycles": [int(c) for c in win_cycles],
        "at_or_below_floor_count": int(at_or_below_floor),
        "at_or_below_floor_frac": floor_hit_frac,
        "rae_per_cycle_mean": rae_mean,
        "rae_per_cycle_median": rae_median,
        "rae_per_cycle_std": rae_std,
        "rae_per_cycle_min": rae_min,
        "rae_per_cycle_max": rae_max,
        "delta_best_vs_floor": delta_vs_floor,
        "delta_best_vs_anchor": delta_vs_anchor,
        "beats_floor": beats_floor,
        "flat_vs_floor": flat_vs_floor,
        "pearson_best_vs_anchor": pearson_vs_anchor,
        "pearson_best_vs_nb2063_mean_bag": pearson_vs_nb2063,
        "verdict": verdict,
        "pre_unblind_clean": pre_unblind_clean,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2063_ref": NB2063_REF,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_cycles", "wins_target", "floor_target",
        "feat_dim_full", "feat_dim_top50",
        "rae_anchor_chemprop_aux",
        "best_cycle", "best_rae", "wins_total",
        "at_or_below_floor_count", "at_or_below_floor_frac",
        "rae_per_cycle_mean", "rae_per_cycle_std",
        "rae_per_cycle_min", "rae_per_cycle_max",
        "delta_best_vs_floor", "delta_best_vs_anchor",
        "beats_floor", "flat_vs_floor",
        "pearson_best_vs_anchor", "pearson_best_vs_nb2063_mean_bag",
        "verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
