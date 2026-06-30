"""nb963 -- Heteroscedastic LGBM(MSE) on chemprop_aux v1 residual with SE^-2 weights.

M4 HYPOTHESIS:
    nb2103 K=28 unweighted LGBM(MSE) achieves mean_bag = 0.4737, median_bag =
    0.4698 on 253 honest cross-fit.  Per-row pEC50 SE in the unblind set
    ranges 0.04--0.61 (mean 0.17, median 0.13).  Low-SE rows are MORE
    informative -- the model should trust them more.  By weighting each
    residual fit by w_i = 1 / SE_i^2 (clipped to [0.16, 400]), the loss
    should calibrate predictions on low-SE rows even at the cost of
    high-SE rows.

PROTOCOL:
    1. Reuse the SHAP top-28 feature index from nb2103_summary.json (K=28
       record) and the SAME 117-col 5-way K-tuned matrix builder
       (AtomPair / MACCS / Mordred / ChempropEmbed / Avalon + ChEMBL kNN).
    2. Slice to top-28 cols (X_unb_28: 253 x 28).
    3. Load pEC50_std.error for each unblind row from
       data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv aligned by
       Molecule Name to test[unb_idx].
    4. Clip SE to [0.05, 2.5], compute w_i = 1 / SE_i^2, clip to [0.16, 400].
    5. Fit LGBM(MSE) on residual = y_unb - chemprop_aux_te[unb_idx] with
       sample_weight = w.  5-seed bag (seeds 0, 1, 7, 42, 137), KFold(n=5,
       shuffle=True) cross-fit per seed.  Identical LGBM hyperparams to
       nb2103 (max_depth=4, num_leaves=15, n_estimators=300, lr=0.03,
       min_child_samples=5, reg_lambda=2.0).
    6. Compute mean-bag and median-bag RAE on 253.  Decision margin 0.003
       vs nb2103 K=28 (mean 0.4737, median 0.4698).
    7. Subset analysis: stratify into SE bins (low/mid/high) and report
       per-bin RAE for hetero vs reuse unweighted nb2103 K=28 mean-bag OOF.
    8. If hetero beats unweighted at the decision margin on EITHER mean-bag
       or median-bag pooled RAE: deploy via 5 outer x 5 inner = 25-fit
       weighted refit on ALL 253 unblind, predict 513, row-MEDIAN, write
       submissions/nb963_deploy_hetero.csv.

OUTPUTS:
    scripts/nb963_hetero_lgbm.py
    data/processed/nb963_summary.json
    data/processed/nb963_mean_bag_oof.npy            (253,) float32
    data/processed/nb963_median_bag_oof.npy          (253,) float32
    submissions/nb963_deploy_hetero.csv              (if beats)
    data/processed/te_nb963.npy                      (if beats)
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

TAG = "nb963"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
TOP_K_SHAP = 28

# Deploy seeds (only used if beats)
OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_OFFSETS = [0, 1, 7, 42, 137]
N_INNER = len(INNER_OFFSETS)

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"
NB2103_K28_OOF = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"

UNBLIND_CSV = RAW_DIR / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3

KNN_K = 5
SIM_FLOOR = 1e-6

SE_CLIP_LO = 0.05
SE_CLIP_HI = 2.5
W_CLIP_LO = 0.16   # = 1 / 2.5^2
W_CLIP_HI = 400.0  # = 1 / 0.05^2

NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.003


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
    """Same union as nb2103 / nb2112."""
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


def _extract_K_record(sum_dict: dict, records_key: str, K: int) -> dict:
    for r in sum_dict[records_key]:
        if int(r["K"]) == K:
            return r
    raise KeyError(f"K={K} not found in {records_key}")


def _weighted_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 w: np.ndarray, seed: int) -> np.ndarray:
    """5-fold cross-fit with per-row sample_weight."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc],
                sample_weight=w[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _per_bin_rae(y: np.ndarray, p: np.ndarray, bins: np.ndarray, names):
    out = {}
    for bid, label in enumerate(names):
        m = bins == bid
        if m.sum() < 3:
            out[label] = {"n": int(m.sum()), "rae": None}
        else:
            out[label] = {
                "n": int(m.sum()),
                "rae": float(rae(y[m], p[m])),
                "y_mean": float(y[m].mean()),
                "y_std": float(y[m].std()),
                "p_mean": float(p[m].mean()),
            }
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- HETERO LGBM(MSE) w=1/SE^2 on SHAP top-{TOP_K_SHAP} "
          f"of nb2103 117-col matrix")
    print(f"          anchor = {ANCHOR}   seeds = {RESID_SEEDS}   "
          f"folds = {RESID_FOLDS}")
    print(f"          SE clip = [{SE_CLIP_LO}, {SE_CLIP_HI}]   "
          f"w clip = [{W_CLIP_LO}, {W_CLIP_HI}]")
    print(f"          ref nb2103 K=28 mean_bag = {NB2103_K28_MEAN_BAG_REF:.4f}, "
          f"median_bag = {NB2103_K28_MEDIAN_BAG_REF:.4f}   "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- nb2103 K=28 top-28 indices ----
    if not NB2103_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2103_SUMMARY} -- run nb2103 first")
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    rec28 = _extract_K_record(nb2103_sum, "per_K_records", K=TOP_K_SHAP)
    top28_idx = np.array(rec28["top_K_idx_in_117"], dtype=np.int32)
    nb2103_k28_mean_bag = float(rec28["rae_mean_bag"])
    nb2103_k28_median_bag = float(rec28["rae_median_bag"])
    print(f"[reuse] nb2103 K=28 top28 indices head 10: "
          f"{top28_idx[:10].tolist()}")
    print(f"[check] nb2103 K=28 mean_bag   = {nb2103_k28_mean_bag:.4f}")
    print(f"[check] nb2103 K=28 median_bag = {nb2103_k28_median_bag:.4f}")

    # ---- Anchor + truth + unb index ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
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
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load SE for 253 unblind rows, aligned by Molecule Name ----
    if not UNBLIND_CSV.exists():
        raise FileNotFoundError(f"missing {UNBLIND_CSV}")
    unb_df = pd.read_csv(UNBLIND_CSV)
    if "Molecule Name" not in unb_df.columns:
        raise KeyError("Molecule Name not in TEST_PHASE_1_UNBLINDED.csv")
    se_lookup = unb_df.set_index("Molecule Name")[
        "pEC50_std.error (-log10(molarity))"
    ].to_dict()
    te_names = np.array(mol_names)
    se_raw = np.array(
        [se_lookup.get(n, np.nan) for n in te_names[unb_idx]],
        dtype=float,
    )
    n_missing_se = int(np.isnan(se_raw).sum())
    if n_missing_se > 0:
        # fallback to median for any missing SE
        med = float(np.nanmedian(se_raw))
        se_raw = np.where(np.isnan(se_raw), med, se_raw)
        print(f"[se] WARN: {n_missing_se} missing SE -> imputed median {med:.4f}")
    se_clipped = np.clip(se_raw, SE_CLIP_LO, SE_CLIP_HI)
    w = 1.0 / (se_clipped ** 2)
    w = np.clip(w, W_CLIP_LO, W_CLIP_HI).astype(np.float64)
    print(f"[se] raw   p5/25/50/75/95 = "
          f"{np.percentile(se_raw, 5):.4f} / "
          f"{np.percentile(se_raw, 25):.4f} / "
          f"{np.percentile(se_raw, 50):.4f} / "
          f"{np.percentile(se_raw, 75):.4f} / "
          f"{np.percentile(se_raw, 95):.4f}")
    print(f"[w]   weight p5/25/50/75/95 = "
          f"{np.percentile(w, 5):.2f} / "
          f"{np.percentile(w, 25):.2f} / "
          f"{np.percentile(w, 50):.2f} / "
          f"{np.percentile(w, 75):.2f} / "
          f"{np.percentile(w, 95):.2f}")
    print(f"[w]   weight min/max/mean    = "
          f"{w.min():.4f} / {w.max():.4f} / {w.mean():.4f}")

    # ---- Define SE-tertile bins for subset analysis (terciles of raw SE) ----
    bin_edges = np.percentile(se_raw, [33.333, 66.667])
    bins = np.zeros(n_unb, dtype=np.int8)
    bins[se_raw > bin_edges[0]] = 1
    bins[se_raw > bin_edges[1]] = 2
    bin_names = ["low_SE", "mid_SE", "high_SE"]
    bin_counts = {bin_names[b]: int((bins == b).sum()) for b in range(3)}
    print(f"[bin] SE tertile edges = {bin_edges.tolist()}")
    print(f"[bin] tertile counts = {bin_counts}")

    # ---- Load nb2103 K=28 unweighted mean-bag OOF for unweighted bin baseline ----
    if not NB2103_K28_OOF.exists():
        raise FileNotFoundError(f"missing {NB2103_K28_OOF} -- run nb2103 first")
    pred_unweighted = np.load(NB2103_K28_OOF).astype(np.float64)
    if pred_unweighted.shape[0] != n_unb:
        raise ValueError(f"nb2103 K=28 OOF shape {pred_unweighted.shape} "
                         f"!= n_unb {n_unb}")
    rae_unweighted_pooled = float(rae(y_unb, pred_unweighted))
    print(f"[ref] nb2103 K=28 unweighted pooled RAE = "
          f"{rae_unweighted_pooled:.4f}  "
          f"(summary mean_bag {nb2103_k28_mean_bag:.4f})")

    # ---- Build 117-col feature matrix (same recipe as nb2103/nb2112) ----
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
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)

    print("\n" + "-" * 78)
    print("CHEMBL PXR POOL")
    print("-" * 78)
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
    pred_chembl_te, mean_sim_te = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_te = pred_chembl_te.astype(np.float32)
    mean_sim_te = mean_sim_te.astype(np.float32)

    X_te_117 = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_te.reshape(-1, 1),
            mean_sim_te.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    X_unb_117 = X_te_117[unb_idx]
    X_unb_28 = X_unb_117[:, top28_idx].astype(np.float32)
    X_te_28 = X_te_117[:, top28_idx].astype(np.float32)
    print(f"\n   feat: X_unb_28 = {X_unb_28.shape}   X_te_28 = {X_te_28.shape}")

    # ---- 5-seed bag x 5-fold cross-fit, weighted ----
    print("\n" + "-" * 78)
    print(f"WEIGHTED 5-SEED BAG x {RESID_FOLDS}-FOLD CROSS-FIT")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        oof_s = _weighted_cross_fit_one_seed(X_unb_28, residual, w, s)
        pred_corr_s = anchor + oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": rae_s - rae_anchor,
            "resid_oof_mean": float(oof_s.mean()),
            "resid_oof_std": float(oof_s.std()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {rae_s - rae_anchor:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_std = float(per_seed_rae_arr.std())

    print("\n" + "-" * 78)
    print(f"   per-seed RAE  = [{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean = {rae_per_seed_mean:.4f}  "
          f"std = {rae_per_seed_std:.4f}")
    print(f"   mean-bag RAE  = {rae_mean_bag:.4f}  "
          f"(d_vs_nb2103_K28_mean   = "
          f"{rae_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f})")
    print(f"   median-bag RAE= {rae_median_bag:.4f}  "
          f"(d_vs_nb2103_K28_median = "
          f"{rae_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")

    # ---- Verdict ----
    beats_mean = rae_mean_bag < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    beats_median = rae_median_bag < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN
    flat_mean = abs(rae_mean_bag - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN
    flat_median = abs(rae_median_bag - NB2103_K28_MEDIAN_BAG_REF) < DECISION_MARGIN
    hurts_mean = rae_mean_bag > NB2103_K28_MEAN_BAG_REF + DECISION_MARGIN
    hurts_median = rae_median_bag > NB2103_K28_MEDIAN_BAG_REF + DECISION_MARGIN

    if beats_mean or beats_median:
        if beats_mean and beats_median:
            verdict = "HETERO_BEATS_UNWEIGHTED_ON_BOTH"
        elif beats_mean:
            verdict = "HETERO_BEATS_UNWEIGHTED_ON_MEAN_BAG_ONLY"
        else:
            verdict = "HETERO_BEATS_UNWEIGHTED_ON_MEDIAN_BAG_ONLY"
    elif (flat_mean or flat_median) and not (hurts_mean and hurts_median):
        verdict = "HETERO_FLAT_VS_UNWEIGHTED"
    else:
        verdict = "HETERO_HURTS_VS_UNWEIGHTED"
    print(f"   verdict = {verdict}")

    # ---- Subset analysis: per-bin RAE for hetero vs unweighted ----
    bin_rae_hetero_mean = _per_bin_rae(y_unb, mean_bag_oof, bins, bin_names)
    bin_rae_hetero_median = _per_bin_rae(y_unb, median_bag_oof, bins, bin_names)
    bin_rae_unweighted = _per_bin_rae(y_unb, pred_unweighted, bins, bin_names)
    print("\n" + "-" * 78)
    print("PER-BIN RAE (SE tertiles)")
    print("-" * 78)
    print(f"   {'bin':>10s}  {'n':>4s}  "
          f"{'unweighted':>12s}  {'hetero_mean':>12s}  "
          f"{'hetero_med':>12s}  {'d_mean':>8s}  {'d_med':>8s}")
    for b in bin_names:
        unw = bin_rae_unweighted[b]["rae"]
        hm = bin_rae_hetero_mean[b]["rae"]
        hmed = bin_rae_hetero_median[b]["rae"]
        n_b = bin_rae_hetero_mean[b]["n"]
        d_mean = (hm - unw) if (unw is not None and hm is not None) else None
        d_med = (hmed - unw) if (unw is not None and hmed is not None) else None
        unw_s = f"{unw:.4f}" if unw is not None else "  N/A "
        hm_s = f"{hm:.4f}" if hm is not None else "  N/A "
        hmed_s = f"{hmed:.4f}" if hmed is not None else "  N/A "
        dm_s = f"{d_mean:+.4f}" if d_mean is not None else "  N/A "
        dmd_s = f"{d_med:+.4f}" if d_med is not None else "  N/A "
        print(f"   {b:>10s}  {n_b:>4d}  {unw_s:>12s}  {hm_s:>12s}  "
              f"{hmed_s:>12s}  {dm_s:>8s}  {dmd_s:>8s}")

    # Hypothesis check: does hetero IMPROVE the low_SE bin MORE than high_SE?
    low_d_mean = (bin_rae_hetero_mean["low_SE"]["rae"]
                  - bin_rae_unweighted["low_SE"]["rae"])
    high_d_mean = (bin_rae_hetero_mean["high_SE"]["rae"]
                   - bin_rae_unweighted["high_SE"]["rae"])
    hypothesis_supported = low_d_mean < high_d_mean - 0.005
    print(f"\n   low-SE delta_mean    = {low_d_mean:+.4f}")
    print(f"   high-SE delta_mean   = {high_d_mean:+.4f}")
    print(f"   hypothesis (low-SE improves more) supported? "
          f"{hypothesis_supported}")

    # ---- Save OOF artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] mean_bag_oof   -> "
          f"{DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] median_bag_oof -> "
          f"{DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    # ---- Optional deploy ----
    deploy_summary = None
    if beats_mean or beats_median:
        print("\n" + "-" * 78)
        print("DEPLOY: 25 weighted refits on full 253 -> 513 predictions")
        print("-" * 78)
        n_total = len(OUTER_SEEDS) * N_INNER
        all_resid_513 = np.zeros((n_total, n_test), dtype=np.float64)
        k_global = 0
        for o_i, o in enumerate(OUTER_SEEDS):
            t_o = time.time()
            inner_seeds = [o * 1000 + s for s in INNER_OFFSETS]
            for s in inner_seeds:
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_unb_28, residual, sample_weight=w)
                all_resid_513[k_global] = mdl.predict(X_te_28)
                k_global += 1
            print(f"   outer {o:3d}: inner_seeds={inner_seeds}  "
                  f"wall={time.time() - t_o:.1f}s")
        median_resid_513 = np.median(all_resid_513, axis=0)
        te_nb963 = te_anchor_513 + median_resid_513
        in_pred_unb = te_nb963[unb_idx]
        rae_in_unb = float(rae(y_unb, in_pred_unb))
        print(f"\n   in-sample RAE on unb_idx (deploy MEDIAN) = "
              f"{rae_in_unb:.4f}")

        df_sub = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_nb963.astype(np.float32),
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)} != 513")
        sub_path = SUBMISSIONS_DIR / f"{TAG}_deploy_hetero.csv"
        df_sub.to_csv(sub_path, index=False)
        print(f"[save] submission CSV -> {sub_path}  ({len(df_sub)} rows)")
        te_path = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(te_path, te_nb963.astype(np.float32))
        print(f"[save] te artifact    -> {te_path}")

        deploy_summary = {
            "deployed": True,
            "submission_csv": str(sub_path),
            "te_artifact": str(te_path),
            "n_total_fits": int(n_total),
            "in_RAE_unb_idx_median_deploy": rae_in_unb,
            "median_resid_513_mean": float(median_resid_513.mean()),
            "median_resid_513_std": float(median_resid_513.std()),
            "te_nb963_mean": float(te_nb963.mean()),
            "te_nb963_std": float(te_nb963.std()),
            "te_nb963_min": float(te_nb963.min()),
            "te_nb963_max": float(te_nb963.max()),
        }
    else:
        print("\n[deploy] SKIPPED -- hetero did not beat unweighted at "
              f"margin {DECISION_MARGIN}")
        deploy_summary = {"deployed": False}

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("heteroscedastic_lgbm_mse_w_inv_SE2_shap_top28_on_nb2103_117col"),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "top_k_shap": TOP_K_SHAP,
        "top28_idx_in_117_from_nb2103": top28_idx.tolist(),
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "se_clip_lo": SE_CLIP_LO,
        "se_clip_hi": SE_CLIP_HI,
        "w_clip_lo": W_CLIP_LO,
        "w_clip_hi": W_CLIP_HI,
        "n_unb": n_unb,
        "n_test": n_test,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "se_p5_25_50_75_95": [
            float(np.percentile(se_raw, q)) for q in (5, 25, 50, 75, 95)
        ],
        "w_p5_25_50_75_95": [
            float(np.percentile(w, q)) for q in (5, 25, 50, 75, 95)
        ],
        "w_min": float(w.min()),
        "w_max": float(w.max()),
        "w_mean": float(w.mean()),
        "n_se_missing_imputed": n_missing_se,
        "se_bin_edges_tertile": bin_edges.tolist(),
        "se_bin_counts": bin_counts,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb2103_K28_mean": (
            rae_mean_bag - NB2103_K28_MEAN_BAG_REF
        ),
        "delta_median_bag_vs_nb2103_K28_median": (
            rae_median_bag - NB2103_K28_MEDIAN_BAG_REF
        ),
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_median_bag_vs_anchor": rae_median_bag - rae_anchor,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "nb2103_K28_unweighted_pooled_rae_recomputed": rae_unweighted_pooled,
        "decision_margin": DECISION_MARGIN,
        "verdict": verdict,
        "beats_mean": bool(beats_mean),
        "beats_median": bool(beats_median),
        "flat_mean": bool(flat_mean),
        "flat_median": bool(flat_median),
        "hurts_mean": bool(hurts_mean),
        "hurts_median": bool(hurts_median),
        "per_bin_rae_hetero_mean": bin_rae_hetero_mean,
        "per_bin_rae_hetero_median": bin_rae_hetero_median,
        "per_bin_rae_unweighted": bin_rae_unweighted,
        "low_SE_delta_mean": float(low_d_mean),
        "high_SE_delta_mean": float(high_d_mean),
        "hypothesis_low_SE_improves_more": bool(hypothesis_supported),
        "deploy": deploy_summary,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] summary -> {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_unb", "n_test",
        "rae_anchor_chemprop_aux",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb2103_K28_mean",
        "delta_median_bag_vs_nb2103_K28_median",
        "low_SE_delta_mean", "high_SE_delta_mean",
        "hypothesis_low_SE_improves_more",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    if res.get("deploy", {}).get("deployed"):
        print(f"  deploy.submission_csv = "
              f"{res['deploy']['submission_csv']}")
        print(f"  deploy.in_RAE_unb_idx = "
              f"{res['deploy']['in_RAE_unb_idx_median_deploy']}")
