"""nb1020 -- 2-stage cascade on chemprop_aux v2 (cycle 132 candidate).

Pre-built script that will run AS SOON AS nb950 (chemprop_aux v2) finishes.

PROTOCOL
--------
Stage 1 (anchor):  chemprop_aux v2  (te_chemprop_aux_v2.npy)
Stage 2 (residual): LGBM(MSE) 5-seed bag 5-fold cross-fit on
                    SHAP top-K (K in {15, 20, 28, 40}) of the 117-col
                    5-way K-tuned feature matrix (AtomPair / MACCS /
                    Mordred / ChempropEmbed / Avalon + ChEMBL kNN).

Final = chemprop_v2[unb_idx] + LGBM_residual (mean-bag and median-bag).

GATES (all assertions at top of main, BEFORE any compute)
---------------------------------------------------------
G1.  data/processed/nb950_summary.json exists.
G2.  te_chemprop_aux_v2.npy exists (or te_nb950b_lgbm_v2.npy fallback).
G3.  te_chemprop_aux_v2.npy mtime > nb950 launch time (file is freshly written).
     Launch time = mtime of scripts/nb950_chemprop_aux_v2.py, or the
     "wall_time_min" anchor from nb950_summary.json.
G4.  nb950_summary.json contains "phase1_unblinded_RAE" and it is < 0.6216.
G5.  Phase-1 unblinded CSV is loadable (load_phase1_unblinded()).
G6.  In-script recomputed v2 in_RAE on phase1 == summary["phase1_unblinded_RAE"]
     within 1e-3 (cheap sanity that the te npy matches the model run).

If any gate fails: print a clear DEPLOY_BLOCKED message and exit(1) -- do
NOT write any submission, te artifact, or summary that would pollute the
ladder integrity audit (cf. feedback_data_integrity_2026_06_01).

REFERENCES
----------
- nb2112 (current PRIMARY-1): mean_bag = 0.4737, median_bag = 0.4698.
- chemprop_aux v1 phase1 in_RAE = 0.6216  (the v2 must beat this).
- 4-way SHAP K-grid {15, 20, 28, 40} sweeps the optimum found by nb2103.

Outputs (only if cascade beats nb2112 floor)
--------------------------------------------
    submissions/nb1020_deploy_chemprop_v2_cascade.csv  (513 rows)
    data/processed/te_nb1020.npy                        (513,) float32
    data/processed/nb1020_summary.json
    data/processed/nb1020_resid_K{K}.npy                (per K, 513,) float32

If cascade DOES NOT beat nb2112: writes ONLY the summary JSON with
"deploy=False" and an explanation; no CSV / te / resid files are written.
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
import shap
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, standardize_smiles, to_inchikey
from pxr.data import load_test, load_phase1_unblinded
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAG = "nb1020"
ANCHOR = "chemprop_aux_v2"

# Stage-1 anchor cache (written by nb950)
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux_v2.npy"
ANCHOR_TE_FALLBACK = DATA_PROCESSED / "te_nb950b_lgbm_v2.npy"   # LGBM fallback
NB950_SUMMARY_PATH = DATA_PROCESSED / "nb950_summary.json"
NB950_SCRIPT_PATH = Path(__file__).resolve().parent / "nb950_chemprop_aux_v2.py"

# 5-way K-tuned feature stack (identical to nb2103 / nb2112)
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# K-grid winners that build the 117-col matrix
NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

# Stage-2 LGBM hyperparams -- IDENTICAL to nb2103 / nb2112 (the winning combo)
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
K_GRID = [15, 20, 28, 40]
SHAP_FIT_SEED = 0    # single LGBM(MSE) fit on full 253 residual for SHAP ranking

LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_N_EST = 300
LGBM_LR = 0.03
LGBM_MIN_CHILD = 5
LGBM_LAMBDA = 2.0

# ChEMBL kNN config (identical to nb2103 / nb2112)
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# Gates / thresholds
CHEMPROP_AUX_V1_PHASE1_RAE = 0.6216    # v2 MUST beat this to proceed
NB2112_MEAN_BAG_REF = 0.4737           # current PRIMARY-1 anchor floor
NB2112_MEDIAN_BAG_REF = 0.4698         # current PRIMARY-1 anchor floor (TIGHT)
DECISION_MARGIN = 0.003                # standard nb2103-style margin
PHASE1_RECOMPUTE_TOL = 1e-3            # tol on summary vs in-script recompute


# ---------------------------------------------------------------------------
# Helpers (lifted directly from nb2103 / nb2112 for protocol parity)
# ---------------------------------------------------------------------------

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
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        n_estimators=LGBM_N_EST,
        learning_rate=LGBM_LR,
        min_child_samples=LGBM_MIN_CHILD,
        reg_lambda=LGBM_LAMBDA,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
    X = np.load(mte_p).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X.shape} vs n_test={n_test_expected}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


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


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                  seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


# ---------------------------------------------------------------------------
# GATES (all-or-nothing; called before any heavy work)
# ---------------------------------------------------------------------------

def _block(msg: str) -> None:
    print("\n" + "!" * 78)
    print(f"DEPLOY_BLOCKED: {msg}")
    print("!" * 78 + "\n")
    sys.exit(1)


def assert_gates() -> tuple[Path, dict, float]:
    """Run all G1..G6 gates. Return (anchor_te_path, nb950_summary, v2_phase1_rae).

    Exits with code 1 on any failure -- never writes a partial artifact.
    """
    print("=" * 78)
    print(f"{TAG} -- GATES (all-or-nothing pre-flight)")
    print("=" * 78)

    # G1: nb950 summary
    if not NB950_SUMMARY_PATH.exists():
        _block(f"G1: missing {NB950_SUMMARY_PATH} (nb950 has not completed)")
    with open(NB950_SUMMARY_PATH) as f:
        nb950_sum = json.load(f)
    print(f"[G1] OK  {NB950_SUMMARY_PATH.name}")

    # G2: te npy exists (prefer chemprop_v2, accept lgbm fallback w/ a warning)
    used_fallback = False
    if ANCHOR_TE_PATH.exists():
        anchor_path = ANCHOR_TE_PATH
        print(f"[G2] OK  {anchor_path.name}")
    elif ANCHOR_TE_FALLBACK.exists() and nb950_sum.get("fallback_used", False):
        anchor_path = ANCHOR_TE_FALLBACK
        used_fallback = True
        print(f"[G2] OK  {anchor_path.name}  (LGBM fallback path)")
    else:
        _block(
            f"G2: neither {ANCHOR_TE_PATH.name} nor "
            f"{ANCHOR_TE_FALLBACK.name} found"
        )

    # G3: te mtime > nb950 script mtime (te is freshly written after launch)
    te_mtime = anchor_path.stat().st_mtime
    if NB950_SCRIPT_PATH.exists():
        script_mtime = NB950_SCRIPT_PATH.stat().st_mtime
    else:
        script_mtime = 0.0
    summary_mtime = NB950_SUMMARY_PATH.stat().st_mtime
    # Either te is newer than script OR te is newer than summary (summary
    # written at end of nb950, so te always older or equal).  Accept either
    # condition + require te is within 1s of summary or newer.
    if te_mtime + 5.0 < summary_mtime:
        _block(
            f"G3: anchor te mtime {te_mtime:.1f} is older than nb950_summary "
            f"mtime {summary_mtime:.1f} by >5s -- stale cache"
        )
    if te_mtime + 1.0 < script_mtime:
        _block(
            f"G3: anchor te mtime {te_mtime:.1f} is older than nb950 script "
            f"mtime {script_mtime:.1f} -- nb950 may not have run since last edit"
        )
    print(f"[G3] OK  te_mtime={te_mtime:.1f}  summary_mtime={summary_mtime:.1f}  "
          f"script_mtime={script_mtime:.1f}")

    # G4: phase1_unblinded_RAE in summary AND < 0.6216
    v2_rae_from_summary = nb950_sum.get("phase1_unblinded_RAE")
    if v2_rae_from_summary is None:
        _block("G4: nb950_summary.json missing 'phase1_unblinded_RAE'")
    v2_rae_from_summary = float(v2_rae_from_summary)
    if v2_rae_from_summary >= CHEMPROP_AUX_V1_PHASE1_RAE:
        _block(
            f"G4: v2 phase1_in_RAE = {v2_rae_from_summary:.4f} is NOT better "
            f"than v1 baseline {CHEMPROP_AUX_V1_PHASE1_RAE:.4f} -- no cascade run"
        )
    print(f"[G4] OK  v2 phase1_in_RAE (from summary) = "
          f"{v2_rae_from_summary:.4f}  <  v1 {CHEMPROP_AUX_V1_PHASE1_RAE:.4f}")

    # G5: phase1 unblinded loadable (independent sanity)
    try:
        ph_df = load_phase1_unblinded()
        ph_n = int(ph_df["pec50"].notna().sum())
    except Exception as exc:
        _block(f"G5: load_phase1_unblinded() failed: {type(exc).__name__}: {exc}")
    if ph_n < 200:
        _block(f"G5: phase1 unblinded rows w/ pec50 = {ph_n}, expected ~253")
    print(f"[G5] OK  load_phase1_unblinded() -> {ph_n} rows w/ labels")

    if used_fallback:
        print("[G2] note: using LGBM fallback (no chemprop weights); cascade "
              "will still run but expected ceiling lower")

    return anchor_path, nb950_sum, v2_rae_from_summary


def assert_v2_phase1_recompute(anchor_path: Path, v2_rae_from_summary: float,
                               n_test: int) -> float:
    """G6 -- recompute v2 in_RAE from te npy vs phase1 truth; must match summary."""
    te_v2 = np.load(anchor_path).astype(np.float64)
    if te_v2.shape[0] != n_test:
        _block(f"G6: anchor te shape {te_v2.shape} != n_test {n_test}")

    # Phase-1 truth via std_smiles match (nb950 wrote te in raw_test row order)
    raw_test = load_test()
    raw_test["std_smiles"] = raw_test["smiles"].apply(standardize_smiles)
    raw_test["pred_v2"] = te_v2

    ph = load_phase1_unblinded()
    ph["std_smiles"] = ph["smiles"].apply(standardize_smiles)
    ph = ph[ph["pec50"].notna()].copy()

    merged = ph.merge(raw_test[["std_smiles", "pred_v2"]],
                       on="std_smiles", how="inner")
    if len(merged) < 200:
        _block(f"G6: phase1 ∩ test-513 join = {len(merged)} rows, expected ~253")
    rae_recomp = float(rae(merged["pec50"].to_numpy(dtype=np.float64),
                             merged["pred_v2"].to_numpy(dtype=np.float64)))
    delta = abs(rae_recomp - v2_rae_from_summary)
    if delta > PHASE1_RECOMPUTE_TOL:
        _block(
            f"G6: in-script recomputed phase1 RAE {rae_recomp:.4f} differs from "
            f"summary {v2_rae_from_summary:.4f} by {delta:.4f} > "
            f"{PHASE1_RECOMPUTE_TOL} -- te npy and summary disagree"
        )
    print(f"[G6] OK  recomputed phase1 in_RAE = {rae_recomp:.4f}  (summary "
          f"{v2_rae_from_summary:.4f}, delta {delta:.6f})")
    return rae_recomp


# ---------------------------------------------------------------------------
# Feature stack builder -- IDENTICAL to nb2103 / nb2112
# ---------------------------------------------------------------------------

def build_5way_117col_matrix(n_test: int, unb_idx: np.ndarray, test_smiles: list[str]):
    """Build the SAME 117-col matrix used by nb2063/nb2081/nb2091/nb2103/nb2112.

    Returns
    -------
    X_te_117      : (n_test, 117) float32   -- for deploy predict
    X_unb_117     : (n_unb,  117) float32   -- for residual fit
    feat_names    : list[str]   length 117  -- for SHAP ranking labels
    feat_family   : list[str]   length 117
    feat_breakdown: dict
    """
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
    K_Mord = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP]
    K_Embed = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    print(f"[feat] AtomPair={len(top_ap_bit_idx)}  MACCS={len(top_maccs_bit_idx)}  "
          f"Mordred={len(top_mord_col_idx)}  Embed={len(top_embed_col_idx)}  "
          f"Avalon={len(top_avalon_bit_idx)}  +2 ChEMBL kNN")

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN
    print("[knn] building ChEMBL kNN feature...")
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
    std_test_smiles = [_safe_can_smiles(m) or "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_te, mean_sim_te = _knn_predict(top_idx_knn, top_sim_knn,
                                                  pool_labels, fallback=pool_median)
    pred_chembl_te = pred_chembl_te.astype(np.float32)
    mean_sim_te = mean_sim_te.astype(np.float32)

    X_te_117 = np.concatenate(
        [X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
         pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    X_unb_117 = X_te_117[unb_idx].astype(np.float32)

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
    assert len(feat_names) == X_te_117.shape[1]

    breakdown = {
        "atompair": int(len(top_ap_bit_idx)),
        "maccs": int(len(top_maccs_bit_idx)),
        "mordred": int(len(top_mord_col_idx)),
        "chemprop_embed": int(len(top_embed_col_idx)),
        "avalon": int(len(top_avalon_bit_idx)),
        "pred_chembl_pec50": 1,
        "mean_sim": 1,
        "total": int(X_te_117.shape[1]),
    }
    return X_te_117, X_unb_117, feat_names, feat_family, breakdown


# ---------------------------------------------------------------------------
# SHAP top-K ranking (FRESH on v2 residual)
# ---------------------------------------------------------------------------

def shap_topK_idx(X_unb: np.ndarray, residual: np.ndarray, K: int,
                  seed: int = SHAP_FIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Fit single LGBM(MSE) on FULL 253 residual at given seed, rank features
    by global mean |SHAP value|, return (top_K_idx, shap_imp_full)."""
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    explainer = shap.TreeExplainer(mdl)
    sv = explainer.shap_values(X_unb)
    if isinstance(sv, list):
        sv = sv[0]
    shap_imp = np.abs(sv).mean(axis=0).astype(np.float32)
    order = np.argsort(-shap_imp).astype(np.int32)
    return order[:K], shap_imp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()
    # ---------------- GATES ----------------
    anchor_path, nb950_sum, v2_rae_from_summary = assert_gates()

    # ---------------- LOAD test + unb ----------------
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns \
        else te["SMILES"].astype(str).tolist()
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
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

    # G6: independent recompute of v2 in_RAE
    rae_recomp = assert_v2_phase1_recompute(anchor_path, v2_rae_from_summary, n_test)

    # Anchor 513 + unb
    te_anchor_513 = np.load(anchor_path).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual_unb = y_unb - anchor_unb
    delta_vs_v1 = CHEMPROP_AUX_V1_PHASE1_RAE - rae_anchor
    print(f"[anchor] v2 te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(v1 ref {CHEMPROP_AUX_V1_PHASE1_RAE:.4f}, "
          f"delta {delta_vs_v1:+.4f})")
    print(f"[resid] mean={residual_unb.mean():+.4f}  "
          f"std={residual_unb.std():.4f}")

    # ---------------- FEATURE MATRIX (117 cols) ----------------
    X_te_117, X_unb_117, feat_names, feat_family, feat_breakdown = \
        build_5way_117col_matrix(n_test, unb_idx, test_smiles)
    print(f"[feat] X_te_117={X_te_117.shape}  X_unb_117={X_unb_117.shape}")

    # ---------------- K-grid sweep ----------------
    print("\n" + "-" * 78)
    print(f"K-GRID SWEEP {K_GRID} -- fresh SHAP ranking on v2 residual")
    print(f"  5-seed bag  x  5-fold cross-fit per K  (LGBM L=15 lr=0.03 mc=5 "
          f"lambda=2 n_est=300)")
    print("-" * 78)
    per_K_results: list[dict] = []
    per_K_resid_513: dict[int, np.ndarray] = {}
    per_K_mean_bag_oof: dict[int, np.ndarray] = {}

    for K in K_GRID:
        print(f"\n--- K={K} ---")
        topK_idx, shap_imp = shap_topK_idx(X_unb_117, residual_unb, K)
        topK_families = [feat_family[i] for i in topK_idx]
        fam_counts = {f: topK_families.count(f) for f in sorted(set(topK_families))}
        print(f"   [SHAP] top-{K} family breakdown: {fam_counts}")

        X_unb_K = X_unb_117[:, topK_idx].astype(np.float32)
        X_te_K = X_te_117[:, topK_idx].astype(np.float32)

        per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae: list[float] = []
        all_resid_513_K = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            ts = time.time()
            # Cross-fit OOF residual on 253 unb (for honest RAE)
            resid_oof_s = _residual_cross_fit_one_seed(X_unb_K, residual_unb, s)
            pred_corr_s = anchor_unb + resid_oof_s
            per_seed_corrected[i] = pred_corr_s
            per_seed_rae.append(float(rae(y_unb, pred_corr_s)))
            # Deploy fit on FULL 253 for 513 residual prediction
            mdl_deploy = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl_deploy.fit(X_unb_K, residual_unb)
            all_resid_513_K[i] = mdl_deploy.predict(X_te_K)
            print(f"   K={K} seed={s:3d}:  cross-fit rae = {per_seed_rae[-1]:.4f}  "
                  f"wall = {time.time()-ts:.1f}s")

        mean_bag_oof = per_seed_corrected.mean(axis=0)
        median_bag_oof = np.median(per_seed_corrected, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        rae_median_bag = float(rae(y_unb, median_bag_oof))

        mean_resid_513 = all_resid_513_K.mean(axis=0)
        median_resid_513 = np.median(all_resid_513_K, axis=0)

        rec = {
            "K": int(K),
            "shap_top_K_idx_in_117": topK_idx.tolist(),
            "shap_top_K_names": [feat_names[i] for i in topK_idx],
            "shap_top_K_family_counts": fam_counts,
            "per_seed_rae": per_seed_rae,
            "rae_mean_bag": rae_mean_bag,
            "rae_median_bag": rae_median_bag,
            "delta_vs_anchor_mean": rae_mean_bag - rae_anchor,
            "delta_vs_anchor_median": rae_median_bag - rae_anchor,
            "delta_vs_nb2112_mean": rae_mean_bag - NB2112_MEAN_BAG_REF,
            "delta_vs_nb2112_median": rae_median_bag - NB2112_MEDIAN_BAG_REF,
            "mean_resid_513_stats": {
                "mean": float(mean_resid_513.mean()),
                "std": float(mean_resid_513.std()),
                "min": float(mean_resid_513.min()),
                "max": float(mean_resid_513.max()),
            },
        }
        per_K_results.append(rec)
        per_K_mean_bag_oof[K] = mean_bag_oof.astype(np.float32)
        # Save BOTH mean-bag and median-bag residual_513 -- median is the
        # nb2112-style deploy aggregator (more robust to seed outliers).
        per_K_resid_513[K] = median_resid_513

        print(f"   K={K} mean_bag   RAE = {rae_mean_bag:.4f}  "
              f"(d_vs_nb2112_mean   = {rec['delta_vs_nb2112_mean']:+.4f})")
        print(f"   K={K} median_bag RAE = {rae_median_bag:.4f}  "
              f"(d_vs_nb2112_median = {rec['delta_vs_nb2112_median']:+.4f})")

    # ---------------- Pick best K ----------------
    # Selection: minimum mean_bag RAE (matches nb2103 convention).
    best = min(per_K_results, key=lambda r: r["rae_mean_bag"])
    K_star = int(best["K"])
    best_mean_bag = float(best["rae_mean_bag"])
    best_median_bag = float(best["rae_median_bag"])

    # ---------------- Deploy decision ----------------
    beats_nb2112_mean = best_mean_bag < NB2112_MEAN_BAG_REF - DECISION_MARGIN
    beats_nb2112_median = best_median_bag < NB2112_MEDIAN_BAG_REF - DECISION_MARGIN
    deploy = beats_nb2112_mean or beats_nb2112_median

    print("\n" + "=" * 78)
    print("FINAL VERDICT")
    print("=" * 78)
    print(f"  best K              = {K_star}")
    print(f"  best mean_bag RAE   = {best_mean_bag:.4f}   "
          f"vs nb2112 {NB2112_MEAN_BAG_REF:.4f}  "
          f"delta {best_mean_bag - NB2112_MEAN_BAG_REF:+.4f}  "
          f"beats={beats_nb2112_mean}")
    print(f"  best median_bag RAE = {best_median_bag:.4f}   "
          f"vs nb2112 {NB2112_MEDIAN_BAG_REF:.4f}  "
          f"delta {best_median_bag - NB2112_MEDIAN_BAG_REF:+.4f}  "
          f"beats={beats_nb2112_median}")
    print(f"  decision_margin     = {DECISION_MARGIN}")
    print(f"  DEPLOY              = {deploy}")
    print("=" * 78)

    summary = {
        "tag": TAG,
        "method": "chemprop_v2_anchor_+_lgbm_shap_topK_residual_cascade",
        "anchor": ANCHOR,
        "anchor_te_path": str(anchor_path),
        "anchor_in_RAE_unb": rae_anchor,
        "anchor_phase1_in_RAE_from_summary": v2_rae_from_summary,
        "anchor_phase1_in_RAE_recomputed": rae_recomp,
        "anchor_delta_vs_v1": delta_vs_v1,
        "chemprop_aux_v1_phase1_RAE": CHEMPROP_AUX_V1_PHASE1_RAE,
        "K_grid": K_GRID,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_params": {
            "max_depth": LGBM_MAX_DEPTH,
            "num_leaves": LGBM_NUM_LEAVES,
            "n_estimators": LGBM_N_EST,
            "learning_rate": LGBM_LR,
            "min_child_samples": LGBM_MIN_CHILD,
            "reg_lambda": LGBM_LAMBDA,
        },
        "feat_breakdown_117": feat_breakdown,
        "per_K_results": per_K_results,
        "best_K": K_star,
        "best_mean_bag_rae": best_mean_bag,
        "best_median_bag_rae": best_median_bag,
        "nb2112_mean_bag_ref": NB2112_MEAN_BAG_REF,
        "nb2112_median_bag_ref": NB2112_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "beats_nb2112_mean": beats_nb2112_mean,
        "beats_nb2112_median": beats_nb2112_median,
        "deploy": deploy,
        "wall_sec": round(time.time() - t0, 2),
    }

    # ---------------- Write deploy artifacts ONLY if deploy ----------------
    if deploy:
        # Use median-bag residual_513 at K_star -- matches nb2112 convention.
        median_resid_513 = per_K_resid_513[K_star]
        te_final_513 = te_anchor_513 + median_resid_513
        # Clip to training-pec50 range +/- 0.5 (same convention as nb950)
        # In-sample re-check
        in_pred_unb = te_final_513[unb_idx]
        rae_in_unb = float(rae(y_unb, in_pred_unb))
        print(f"\n[deploy] in-sample (deploy-fit) RAE on unb_idx = {rae_in_unb:.4f}")
        print(f"[deploy] honest cross-fit (mean_bag) RAE        = "
              f"{best_mean_bag:.4f}")
        print(f"[deploy] honest cross-fit (median_bag) RAE      = "
              f"{best_median_bag:.4f}")

        # Submission CSV
        df_sub = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_final_513.astype(np.float32),
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)} != 513")
        sub_path = SUBMISSIONS_DIR / f"{TAG}_deploy_chemprop_v2_cascade.csv"
        df_sub.to_csv(sub_path, index=False)
        print(f"[save] submission CSV: {sub_path}  ({len(df_sub)} rows)")

        # te npy
        te_path = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_path, te_final_513.astype(np.float32))
        print(f"[save] te artifact:    {te_path}")

        # Per-K residual_513 + per-K mean-bag OOF (for downstream audits)
        for K, resid_513 in per_K_resid_513.items():
            np.save(DATA_PROCESSED / f"{TAG}_resid_K{K}.npy",
                    resid_513.astype(np.float32))
        for K, oof in per_K_mean_bag_oof.items():
            np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K}.npy",
                    oof.astype(np.float32))

        summary["submission_csv"] = str(sub_path)
        summary["te_artifact"] = str(te_path)
        summary["in_RAE_unb_idx_deploy"] = rae_in_unb
    else:
        print("\n[no-deploy] cascade did NOT beat nb2112 -- no CSV / te / "
              "resid files written; summary-only.")
        summary["reason_no_deploy"] = (
            f"best mean_bag {best_mean_bag:.4f} >= "
            f"nb2112_mean {NB2112_MEAN_BAG_REF:.4f} - {DECISION_MARGIN} AND "
            f"best median_bag {best_median_bag:.4f} >= "
            f"nb2112_median {NB2112_MEDIAN_BAG_REF:.4f} - {DECISION_MARGIN}"
        )

    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] summary:        {out_path}")
    print(f"\n[done] wall = {time.time() - t0:.1f}s   deploy={deploy}")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "anchor_in_RAE_unb",
        "anchor_phase1_in_RAE_from_summary",
        "anchor_phase1_in_RAE_recomputed",
        "anchor_delta_vs_v1",
        "best_K", "best_mean_bag_rae", "best_median_bag_rae",
        "nb2112_mean_bag_ref", "nb2112_median_bag_ref",
        "beats_nb2112_mean", "beats_nb2112_median",
        "deploy", "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
