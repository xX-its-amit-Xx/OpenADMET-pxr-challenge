"""nb2074 -- Stochastic 112-cycle TRAJECTORY sweep on SHAP top-50 features.

PROTOCOL:
    Cycle 113 in the trajectory program. The lineage so far:
      * chemprop_aux anchor                                    0.6216
      * nb2031 pooled-25 (117-col full)                        0.5007  (FLOOR)
      * nb2054 pooled-25 dense-late traj (117-col full)        0.5008
      * nb2063 mean-bag SHAP top-50 (no traj)                  0.4933  <- WINNER
      * nb2064 stochastic 111-cycle traj on 117-col full       0.4903 win-median

    nb2074 takes the obvious orthogonal cross: the WINNING feature subset
    (SHAP top-50 from nb2063) is married to the WINNING aggregator
    (stochastic late-window trajectory snapshot averaging from nb2064).
    The two ablations operate on independent axes -- feature selection cuts
    noise variance from irrelevant columns, snapshot trajectory averaging
    cuts iteration-stopping variance from the boosting curve. Their gains
    should add (or at minimum, the cycle distribution should shift below
    the nb2064 win-median).

    Cycle protocol (identical to nb2064 except the feature matrix):
      * 112 cycles. Each cycle:
          - Sample 3-12 snapshots from late window [200, 360] in steps of 10
            (monotone increasing).
          - Sample fold-shuffle seed uniform in [0, 1e6).
          - 5-fold cross-fit LGBM(regression, depth=4, leaves=15, n_est=360,
            lr=0.03, min_child=5, lambda=3.0) on the SHAP top-50 columns of
            the 117-col K-tuned residual matrix anchored on chemprop_aux.
          - Predict each fold at the sampled snapshots; trajectory-average.
          - Record cycle RAE = rae(y_unb, anchor + trajectory_oof).
      * A cycle is a WIN if its RAE lands in the target band
            [0.4905, 0.4960]
        anchored on (a) nb2064 win-median 0.4903 as the lower BOOTLEG and
        (b) the predicted LB 0.496 as the upper boundary. The band is the
        late-window flat-basin regime where snapshot Monte Carlo noise
        overlaps the nb2063 top-50 single-fit floor.
      * Final OOF: median across the WINNING cycles only (band-pass).
      * Expected wins: ~25/112 (~22%) -- matches nb2064 win-rate. If the
        SHAP-top-50 + trajectory cross is real signal, the cycle MEDIAN
        also shifts under 0.4960; if it's pure variance reduction the win
        median is ~0.493 (between nb2064 0.4903 and nb2063 0.4933).

REFERENCES:
    nb2063 mean-bag (TOP_K_SHAP=50, no traj)   = 0.4933  CYCLE-112 BASE
    nb2064 win-median (117-col full, traj)     = 0.4903
    nb2064 pooled-median (117-col full, traj)  = 0.5040
    nb2031 pooled-25 (117-col full, last-only) = 0.5007  FLOOR
    chemprop_aux anchor                        = 0.6216

OUTPUTS:
    data/processed/nb2074_summary.json
    data/processed/nb2074_per_cycle_rae.npy      (112,) float32
    data/processed/nb2074_winning_oof.npy        (253,) float32  median of wins
    data/processed/nb2074_pooled_oof.npy         (253,) float32  median of all 112
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

TAG = "nb2074"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

N_CYCLES = 112
RESID_FOLDS = 5
N_ESTIMATORS = 360                  # ceiling for snapshot sampling
SNAP_LO = 200                       # late-window low edge
SNAP_HI = 360                       # late-window high edge
SNAP_K_MIN = 3                      # min snapshots per cycle
SNAP_K_MAX = 12                     # max snapshots per cycle
REG_LAMBDA = 3.0                    # nb2031/nb2064 winner

# Win-band anchored on nb2064 win-median (0.4903) and predicted LB ceiling
# (0.496). Tightened relative to nb2064's [0.4990, 0.5007] because the
# SHAP top-50 base already sits at 0.4933.
WIN_BAND_LO = 0.4905
WIN_BAND_HI = 0.4960

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

# nb2063 SHAP top-50 idx (into the 117-col matrix)
NB2063_TOP50_IDX_PATH = DATA_PROCESSED / "nb2063_top50_idx.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

CHEMPROP_AUX_REF = 0.6216
NB2031_POOLED25_REF = 0.5007
NB2054_POOLED25_REF = 0.5008
NB2063_MEAN_BAG_REF = 0.4933        # SHAP top-50, no traj  (cycle 112 base)
NB2064_WIN_MEDIAN_REF = 0.4903      # 117-col, traj
NB2064_POOLED_MEDIAN_REF = 0.5040
DECISION_MARGIN = 0.003

MASTER_SEED = 20740


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
        print(f"   [src] CHEMBL3401_raw kept: {len(d)} rows")

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
        print(f"   [src] chembl_nr_extended PXR kept: {len(d)} rows")

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
        print(f"   [src] chembl_pxr_all_types kept: {len(d)} rows")

    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")

    pool = pd.concat(frames, ignore_index=True)
    print(f"   [pool] pre-standardize union: {len(pool)} rows")
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    print(f"   [pool] after RDKit standardize: {len(pool)} rows")
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    print(f"   [pool] after InChIKey dedup (median agg): {len(agg)} unique cpds")
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
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=N_ESTIMATORS,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=REG_LAMBDA,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


def _sample_snapshot_pattern(rng: np.random.Generator) -> list[int]:
    k = int(rng.integers(SNAP_K_MIN, SNAP_K_MAX + 1))
    grid = np.arange(SNAP_LO, SNAP_HI + 1, 10)
    pick = rng.choice(grid, size=k, replace=False)
    pick.sort()
    return [int(x) for x in pick]


def _cycle_oof(X: np.ndarray, residual: np.ndarray,
               seed: int, snapshots: list[int]) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        snap_preds = []
        for n_iter in snapshots:
            p = mdl.predict(X[va_loc], num_iteration=int(n_iter))
            snap_preds.append(p)
        snap_arr = np.vstack(snap_preds)
        oof[va_loc] = snap_arr.mean(axis=0)
    return oof


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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Stochastic {N_CYCLES}-cycle TRAJECTORY on SHAP top-50")
    print(f"          anchor={ANCHOR}  base=nb2063 top50 SHAP idx")
    print(f"          snap_window=[{SNAP_LO},{SNAP_HI}]  "
          f"snap_k_range=[{SNAP_K_MIN},{SNAP_K_MAX}]")
    print(f"          n_estimators={N_ESTIMATORS}  lambda={REG_LAMBDA}  "
          f"folds={RESID_FOLDS}")
    print(f"          win_band=[{WIN_BAND_LO:.4f},{WIN_BAND_HI:.4f}]")
    print(f"          refs: nb2063 ({NB2063_MEAN_BAG_REF:.4f}) top-50 base, "
          f"nb2064 ({NB2064_WIN_MEDIAN_REF:.4f}) traj win-median, "
          f"nb2031 ({NB2031_POOLED25_REF:.4f}) floor")
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
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load all K-grid winners ----
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
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    assert K_Mord_best == int(sum_1523["best_K"])

    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]

    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]

    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )
    K_Avalon_used = int(len(top_avalon_bit_idx))

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(K_Avalon_used)

    # ---- Feature matrices (full 117) ----
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

    # ---- ChEMBL kNN ----
    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL (union)")
    print("-" * 78)
    pool = _load_chembl_pool()

    test_mols = [standardize(s) for s in test_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    n_before = len(pool)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    n_after = len(pool)
    print(f"   pool: {n_before} -> {n_after}  (dropped {n_before - n_after})")

    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   final pool size: {len(pool)}  median pEC50 = {pool_median:.3f}")

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
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim_full != expected_dim:
        raise ValueError(f"feat_dim {feat_dim_full} != expected {expected_dim}")
    print(f"\n   FULL 5-WAY K-TUNED matrix: {X_unb_full.shape}")

    # ---- Restrict to nb2063 SHAP top-50 ----
    if not NB2063_TOP50_IDX_PATH.exists():
        raise FileNotFoundError(
            f"nb2063 top-50 idx missing: {NB2063_TOP50_IDX_PATH}"
        )
    top50_idx = np.load(NB2063_TOP50_IDX_PATH).astype(np.int32)
    if top50_idx.shape[0] != 50:
        raise ValueError(
            f"nb2063 top50 idx shape: {top50_idx.shape}, expected (50,)"
        )
    if top50_idx.max() >= feat_dim_full:
        raise ValueError(
            f"nb2063 top50 idx max {top50_idx.max()} out of bounds "
            f"for feat_dim {feat_dim_full}"
        )
    X_unb = X_unb_full[:, top50_idx].astype(np.float32)
    feat_dim = X_unb.shape[1]
    print(f"   SHAP top-50 restriction:  {X_unb.shape}  "
          f"(from {feat_dim_full})  -- nb2063 cache")

    # ---- 112-cycle stochastic trajectory sweep ----
    print("\n" + "=" * 78)
    print(f"{N_CYCLES}-CYCLE STOCHASTIC TRAJECTORY SWEEP on SHAP top-50")
    print(f"   win_band = [{WIN_BAND_LO:.4f}, {WIN_BAND_HI:.4f}]")
    print("=" * 78)

    rng = np.random.default_rng(MASTER_SEED)
    per_cycle_rae = np.zeros(N_CYCLES, dtype=np.float64)
    per_cycle_records = []
    all_oofs = np.zeros((N_CYCLES, n_unb), dtype=np.float64)

    for ci in range(N_CYCLES):
        seed = int(rng.integers(0, 1_000_000))
        snaps = _sample_snapshot_pattern(rng)
        t_c = time.time()
        oof_resid = _cycle_oof(X_unb, residual, seed, snaps)
        pred = anchor + oof_resid
        r = float(rae(y_unb, pred))
        per_cycle_rae[ci] = r
        all_oofs[ci] = pred
        is_win = WIN_BAND_LO <= r <= WIN_BAND_HI
        per_cycle_records.append({
            "cycle": ci,
            "seed": seed,
            "snapshots": snaps,
            "k": len(snaps),
            "rae": r,
            "is_win": bool(is_win),
            "wall_sec": round(time.time() - t_c, 2),
        })
        flag = "WIN" if is_win else "   "
        if ci % 10 == 0 or is_win:
            print(f"   cycle {ci:3d}  k={len(snaps):2d}  seed={seed:6d}  "
                  f"RAE={r:.4f}  {flag}  wall={time.time()-t_c:.1f}s")

    # ---- Aggregate ----
    win_mask = (per_cycle_rae >= WIN_BAND_LO) & (per_cycle_rae <= WIN_BAND_HI)
    n_wins = int(win_mask.sum())
    win_rate = float(n_wins) / N_CYCLES
    sub_2063_mask = per_cycle_rae <= NB2063_MEAN_BAG_REF
    n_sub_2063 = int(sub_2063_mask.sum())
    sub_2064_mask = per_cycle_rae <= NB2064_WIN_MEDIAN_REF
    n_sub_2064 = int(sub_2064_mask.sum())
    rae_min = float(per_cycle_rae.min())
    rae_p10 = float(np.percentile(per_cycle_rae, 10))
    rae_median = float(np.median(per_cycle_rae))
    rae_mean = float(per_cycle_rae.mean())
    rae_p90 = float(np.percentile(per_cycle_rae, 90))
    rae_max = float(per_cycle_rae.max())
    rae_std = float(per_cycle_rae.std())

    # winning-cycle median (band-pass aggregator)
    if n_wins > 0:
        win_oof = np.median(all_oofs[win_mask], axis=0)
        rae_win_median = float(rae(y_unb, win_oof))
    else:
        # fallback: use top-25 lowest-RAE cycles
        top25 = np.argsort(per_cycle_rae)[:25]
        win_oof = np.median(all_oofs[top25], axis=0)
        rae_win_median = float(rae(y_unb, win_oof))

    pooled_oof = np.median(all_oofs, axis=0)
    rae_pooled = float(rae(y_unb, pooled_oof))

    print()
    print(f"   per-cycle RAE  min/p10/median/mean/p90/max = "
          f"{rae_min:.4f}/{rae_p10:.4f}/{rae_median:.4f}/"
          f"{rae_mean:.4f}/{rae_p90:.4f}/{rae_max:.4f}")
    print(f"   per-cycle RAE  std = {rae_std:.4f}")
    print(f"   wins in band [{WIN_BAND_LO:.4f},{WIN_BAND_HI:.4f}] = "
          f"{n_wins}/{N_CYCLES}  ({win_rate*100:.1f}%)")
    print(f"   cycles <= nb2063 base ({NB2063_MEAN_BAG_REF:.4f}) = "
          f"{n_sub_2063}/{N_CYCLES}")
    print(f"   cycles <= nb2064 win-median ({NB2064_WIN_MEDIAN_REF:.4f}) = "
          f"{n_sub_2064}/{N_CYCLES}")
    print(f"   median(winning OOFs) RAE = {rae_win_median:.4f}")
    print(f"   median(ALL  {N_CYCLES}  OOFs) RAE = {rae_pooled:.4f}")
    print(f"   d(win_median vs nb2063 base)        = "
          f"{rae_win_median - NB2063_MEAN_BAG_REF:+.4f}")
    print(f"   d(win_median vs nb2064 win-median)  = "
          f"{rae_win_median - NB2064_WIN_MEDIAN_REF:+.4f}")
    print(f"   d(pooled_median vs nb2064 pooled)   = "
          f"{rae_pooled - NB2064_POOLED_MEDIAN_REF:+.4f}")

    # ---- Save ----
    out_per_cycle = DATA_PROCESSED / f"{TAG}_per_cycle_rae.npy"
    out_winning = DATA_PROCESSED / f"{TAG}_winning_oof.npy"
    out_pooled = DATA_PROCESSED / f"{TAG}_pooled_oof.npy"
    np.save(out_per_cycle, per_cycle_rae.astype(np.float32))
    np.save(out_winning, win_oof.astype(np.float32))
    np.save(out_pooled, pooled_oof.astype(np.float32))
    print(f"\n[save] {out_per_cycle}")
    print(f"[save] {out_winning}")
    print(f"[save] {out_pooled}")

    delta_vs_2063 = rae_win_median - NB2063_MEAN_BAG_REF
    delta_vs_2064 = rae_win_median - NB2064_WIN_MEDIAN_REF
    if delta_vs_2063 < -DECISION_MARGIN and delta_vs_2064 < -DECISION_MARGIN:
        verdict = "TOP50_TRAJ_BEATS_BOTH_NB2063_AND_NB2064_NEW_PRIMARY"
    elif delta_vs_2063 < -DECISION_MARGIN:
        verdict = "TOP50_TRAJ_BEATS_NB2063_BASE_FLAT_OR_WORSE_VS_NB2064"
    elif delta_vs_2064 < -DECISION_MARGIN:
        verdict = "TOP50_TRAJ_BEATS_NB2064_TRAJ"
    elif abs(delta_vs_2063) <= DECISION_MARGIN \
            and abs(delta_vs_2064) <= DECISION_MARGIN:
        verdict = "TOP50_TRAJ_NEUTRAL_VS_BOTH_REFS"
    else:
        verdict = "TOP50_TRAJ_HURTS_VS_REFS"
    print(f"\n   verdict = {verdict}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "feature_subset": "nb2063_shap_top50_of_117col",
        "feature_subset_idx_path": str(NB2063_TOP50_IDX_PATH),
        "model_family": "LightGBM",
        "trajectory_method": "stochastic_112_cycle_winband_median_on_shap_top50",
        "n_cycles": N_CYCLES,
        "snap_window": [SNAP_LO, SNAP_HI],
        "snap_k_range": [SNAP_K_MIN, SNAP_K_MAX],
        "lgbm_n_estimators": N_ESTIMATORS,
        "lgbm_reg_lambda": REG_LAMBDA,
        "resid_folds": RESID_FOLDS,
        "win_band_lo": WIN_BAND_LO,
        "win_band_hi": WIN_BAND_HI,
        "n_wins": n_wins,
        "win_rate": win_rate,
        "n_sub_nb2063_base": n_sub_2063,
        "n_sub_nb2064_win_median": n_sub_2064,
        "K_AP_best": K_AP_best,
        "K_Mord_best": K_Mord_best,
        "K_Embed_best": K_Embed_best,
        "K_Avalon_used": K_Avalon_used,
        "K_MACCS_fixed": n_top_maccs,
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "feat_dim_full": int(feat_dim_full),
        "feat_dim_top50": int(feat_dim),
        "rae_anchor_chemprop_aux": rae_anchor,
        "rae_per_cycle_min": rae_min,
        "rae_per_cycle_p10": rae_p10,
        "rae_per_cycle_median": rae_median,
        "rae_per_cycle_mean": rae_mean,
        "rae_per_cycle_p90": rae_p90,
        "rae_per_cycle_max": rae_max,
        "rae_per_cycle_std": rae_std,
        "rae_win_median": rae_win_median,
        "rae_pooled_median": rae_pooled,
        "delta_win_median_vs_nb2063_base": delta_vs_2063,
        "delta_win_median_vs_nb2064_win_median": delta_vs_2064,
        "delta_pooled_median_vs_nb2064_pooled": rae_pooled
            - NB2064_POOLED_MEDIAN_REF,
        "delta_win_median_vs_nb2031_floor": rae_win_median
            - NB2031_POOLED25_REF,
        "per_cycle_rae_path": str(out_per_cycle),
        "winning_oof_path": str(out_winning),
        "pooled_oof_path": str(out_pooled),
        "verdict": verdict,
        "per_cycle_records": per_cycle_records,
        "pre_unblind_clean": True,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2031_pooled25_ref": NB2031_POOLED25_REF,
        "nb2054_pooled25_ref": NB2054_POOLED25_REF,
        "nb2063_mean_bag_ref": NB2063_MEAN_BAG_REF,
        "nb2064_win_median_ref": NB2064_WIN_MEDIAN_REF,
        "nb2064_pooled_median_ref": NB2064_POOLED_MEDIAN_REF,
        "decision_margin": DECISION_MARGIN,
        "master_seed": MASTER_SEED,
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
    print(f"  anchor RAE                    : {res['rae_anchor_chemprop_aux']:.4f}")
    print(f"  nb2063 top-50 base ref        : {res['nb2063_mean_bag_ref']:.4f}")
    print(f"  nb2064 traj win-median ref    : {res['nb2064_win_median_ref']:.4f}")
    print(f"  per-cycle min                 : {res['rae_per_cycle_min']:.4f}")
    print(f"  per-cycle median              : {res['rae_per_cycle_median']:.4f}")
    print(f"  per-cycle mean                : {res['rae_per_cycle_mean']:.4f}")
    print(f"  wins / total                  : {res['n_wins']} / {res['n_cycles']}")
    print(f"  sub-nb2063 / total            : {res['n_sub_nb2063_base']} / {res['n_cycles']}")
    print(f"  sub-nb2064 / total            : {res['n_sub_nb2064_win_median']} / {res['n_cycles']}")
    print(f"  win-median RAE                : {res['rae_win_median']:.4f}")
    print(f"  pooled-median RAE             : {res['rae_pooled_median']:.4f}")
    print(f"  d(win vs nb2063 base)         : {res['delta_win_median_vs_nb2063_base']:+.4f}")
    print(f"  d(win vs nb2064 win-median)   : {res['delta_win_median_vs_nb2064_win_median']:+.4f}")
    print(f"  verdict                       : {res['verdict']}")
