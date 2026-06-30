"""nb2172 -- Mono-family ablation K=28 per family on the 117-col 5-way matrix.

HYPOTHESIS:
    nb2165 found the XGB-SHAP K=28 family breakdown to be:
        Mordred 11/12, ChempropEmbed 8, AtomPair 4/5, Avalon 2,
        MACCS 1, ChEMBL_kNN 1
    -> the K=28 set is a MIXED tableau across all 6 families. Is the
    mixture genuinely synergistic, or does ONE family already carry the
    full signal? To test, we ablate each family in isolation: take K =
    min(family_size, 28) top features from that family alone (ranked by
    LGBM SHAP fit to the residual using only that family's features),
    eval the same nb2103 LGBM(MSE) config 5-seed bag 5-fold cross-fit on
    chemprop_aux residual. Rank families. Then test the two top-2
    mono-family combos at K=28+28 (or capped at family size).

PROTOCOL:
    1. Anchor = chemprop_aux te[unb_idx] (PRE-unblind, in_RAE 0.6216).
       residual = y_unb - anchor.
    2. Build same 117-col 5-way K-tuned feature matrix used by
       nb2063/nb2103/nb2159/nb2165.
       Family slice sizes (verified from nb2165 console):
         AtomPair 25 | MACCS 20 | Mordred 20 | ChempropEmbed 20 |
         Avalon 30 | ChEMBL_kNN 2
    3. PER-FAMILY MONO-ABLATION (6 arms):
         For each family F:
           Xf = X_unb[:, F_cols]; Kf = min(len(F_cols), 28)
           SHAP source: ONE LGBM(MSE) (seed=0) fit on Xf -> residual
             -> TreeExplainer -> mean |SHAP| per col -> top-Kf
           Eval: same LGBM(MSE) cfg, 5-seed bag (0,1,7,42,137),
             KFold(n=5, shuffle=True) cross-fit per seed on Xf[:, top-Kf]
           -> mean-bag and median-bag RAE
       Rank families by mean-bag.
    4. TWO-FAMILY COMBO (top-2 mono-winners): concat their per-family
       top-Kf features; eval LGBM(MSE) 5-seed bag 5-fold cross-fit;
       mean-bag and median-bag RAE.
       Also try top-1 + top-3 for diversity sanity.
    5. Compare every arm vs nb2103.K=28 (0.4737/0.4698) at
       decision_margin=0.003. Promote if any beats; else mark axis closed.

Outputs:
    scripts/nb2172_family_ablation.py
    data/processed/nb2172_summary.json
    data/processed/nb2172_<FAM>_top_idx_in_117.npy            (one per family)
    data/processed/nb2172_<FAM>_mean_bag_oof.npy              (one per family)
    data/processed/nb2172_top1_plus_top2_mean_bag_oof.npy
    data/processed/nb2172_top1_plus_top3_mean_bag_oof.npy
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

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2172"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
TARGET_K = 28
SHAP_SEED = 0

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
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.003

FAMILY_ORDER = [
    "AtomPair", "MACCS", "Mordred", "ChempropEmbed", "Avalon", "ChEMBL_kNN"
]


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
    """Same union as nb1852/nb1861/nb2063/nb2081/nb2091/nb2103/nb2159/nb2165."""
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
    """LGBM(MSE) -- identical to nb2103/nb2159/nb2165."""
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


def _residual_cross_fit_lgbm_one_seed(X: np.ndarray, residual: np.ndarray,
                                      seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
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


def _bag_eval(per_seed_corrected: np.ndarray, y_unb: np.ndarray,
              per_seed_rae: list[float], rae_anchor: float, label: str):
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    arr = np.array(per_seed_rae)
    info = {
        "label": label,
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": float(arr.mean()),
        "rae_per_seed_median": float(np.median(arr)),
        "rae_per_seed_std": float(arr.std()),
        "rae_per_seed_min": float(arr.min()),
        "rae_per_seed_max": float(arr.max()),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb2103_K28": rae_mean_bag - NB2103_K28_MEAN_BAG_REF,
        "delta_median_bag_vs_nb2103_K28": rae_median_bag - NB2103_K28_MEDIAN_BAG_REF,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
    }
    print(f"   [{label}] per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   [{label}] per-seed mean/median/std = "
          f"{info['rae_per_seed_mean']:.4f} / "
          f"{info['rae_per_seed_median']:.4f} / "
          f"{info['rae_per_seed_std']:.4f}")
    print(f"   [{label}] POOLED mean_bag   = {rae_mean_bag:.4f}  "
          f"(vs nb2103.K28 mean   = {NB2103_K28_MEAN_BAG_REF:.4f}, "
          f"delta = {info['delta_mean_bag_vs_nb2103_K28']:+.4f})")
    print(f"   [{label}] POOLED median_bag = {rae_median_bag:.4f}  "
          f"(vs nb2103.K28 median = {NB2103_K28_MEDIAN_BAG_REF:.4f}, "
          f"delta = {info['delta_median_bag_vs_nb2103_K28']:+.4f})")
    return mean_bag_oof.astype(np.float32), median_bag_oof.astype(np.float32), info


def _verdict_for(label: str, info: dict, rae_anchor: float) -> str:
    beats_mean = info["rae_mean_bag"] < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    flat_mean = abs(info["delta_mean_bag_vs_nb2103_K28"]) < DECISION_MARGIN
    beats_anchor = info["rae_mean_bag"] < rae_anchor - DECISION_MARGIN
    if beats_mean:
        return f"{label}_BEATS_NB2103_K28_NEW_CANDIDATE"
    if flat_mean:
        return f"{label}_FLAT_VS_NB2103_K28"
    if beats_anchor:
        return f"{label}_BEATS_ANCHOR_BUT_WORSE_THAN_NB2103_K28"
    if abs(info["rae_mean_bag"] - rae_anchor) < DECISION_MARGIN:
        return f"{label}_FLAT_VS_ANCHOR"
    return f"{label}_HURTS_ANCHOR"


def _family_shap_top_idx(X_fam: np.ndarray, residual: np.ndarray,
                         fam_label: str) -> np.ndarray:
    """Fit ONE LGBM seed=0 on this family's cols -> mean |SHAP| -> top-K."""
    n_cols = X_fam.shape[1]
    K_eff = min(TARGET_K, n_cols)
    mdl = lgb.LGBMRegressor(**_lgbm_params(SHAP_SEED))
    mdl.fit(X_fam, residual)
    explainer = shap.TreeExplainer(mdl)
    sv = explainer.shap_values(X_fam)
    imp = np.abs(sv).mean(axis=0).astype(np.float32)
    top_k_local = np.argsort(-imp)[:K_eff].astype(np.int32)
    print(f"   [{fam_label}] family_size={n_cols}  K_eff={K_eff}  "
          f"top-3 imp = "
          f"[{', '.join(f'{imp[i]:.4f}' for i in top_k_local[:3])}]")
    return top_k_local, imp


def _eval_arm(X_arm: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
              y_unb: np.ndarray, rae_anchor: float, label: str):
    """5-seed bag 5-fold LGBM(MSE) cross-fit; return (mean_bag, median_bag, info)."""
    n_unb = len(y_unb)
    per_seed_corr = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_lgbm_one_seed(X_arm, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corr[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [{label}] seed {s:3d}: rae = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  wall = {time.time() - ts:.1f}s")
    mean_bag, median_bag, info = _bag_eval(
        per_seed_corr, y_unb, per_seed_rae, rae_anchor, label
    )
    info["per_seed_records"] = per_seed_records
    return mean_bag, median_bag, info


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- mono-family ablation K=28 on 117-col 5-way K-tuned matrix")
    print(f"          anchor={ANCHOR}  eval-seeds={RESID_SEEDS}  "
          f"folds={RESID_FOLDS}")
    print(f"          ref: nb2103.K=28 mean_bag={NB2103_K28_MEAN_BAG_REF:.4f} "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin={DECISION_MARGIN}")
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

    n_top_ap = int(len(top_ap_bit_idx))
    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(K_Avalon_used)
    print(f"[reuse] AtomPair      slice = {n_top_ap}")
    print(f"[reuse] MACCS         slice = {n_top_maccs}")
    print(f"[reuse] Mordred       slice = {n_top_mord}")
    print(f"[reuse] ChempropEmbed slice = {n_top_embed}")
    print(f"[reuse] Avalon        slice = {n_top_avalon}")
    print(f"[reuse] ChEMBL_kNN    slice = 2")

    # ---- Feature matrices ----
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

    X_chembl_unb = np.concatenate(
        [pred_chembl_unb.reshape(-1, 1), mean_sim_unb.reshape(-1, 1)],
        axis=1
    ).astype(np.float32)

    # ---- Family slices in the 117-col matrix (for downstream traceability) ----
    family_slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    family_slices["AtomPair"] = (cursor, cursor + n_top_ap)
    cursor += n_top_ap
    family_slices["MACCS"] = (cursor, cursor + n_top_maccs)
    cursor += n_top_maccs
    family_slices["Mordred"] = (cursor, cursor + n_top_mord)
    cursor += n_top_mord
    family_slices["ChempropEmbed"] = (cursor, cursor + n_top_embed)
    cursor += n_top_embed
    family_slices["Avalon"] = (cursor, cursor + n_top_avalon)
    cursor += n_top_avalon
    family_slices["ChEMBL_kNN"] = (cursor, cursor + 2)
    cursor += 2
    feat_dim_total = cursor
    print(f"\n   COMBINED 5-way K-tuned 117-col matrix dim = {feat_dim_total}")

    family_to_X = {
        "AtomPair":     X_ap_unb_top,
        "MACCS":        X_maccs_unb_top,
        "Mordred":      X_mord_unb_top,
        "ChempropEmbed": X_emb_unb_top,
        "Avalon":       X_av_unb_top,
        "ChEMBL_kNN":   X_chembl_unb,
    }

    # ============================================================
    # STEP 1: PER-FAMILY MONO-ABLATION  (6 arms)
    # ============================================================
    print("\n" + "=" * 78)
    print(f"STEP 1: PER-FAMILY MONO-ABLATION K=min(family_size, {TARGET_K})")
    print("=" * 78)
    family_results: dict[str, dict] = {}
    family_top_idx_local: dict[str, np.ndarray] = {}
    family_mean_bag_oof: dict[str, np.ndarray] = {}

    for fam in FAMILY_ORDER:
        X_fam_full = family_to_X[fam]
        n_cols = X_fam_full.shape[1]
        print("\n" + "-" * 78)
        print(f"FAMILY: {fam}  (family_size={n_cols})")
        print("-" * 78)

        # Per-family SHAP source -> top local idx
        top_local, imp_full = _family_shap_top_idx(
            X_fam_full, residual, fam_label=fam
        )
        family_top_idx_local[fam] = top_local
        X_fam_sel = X_fam_full[:, top_local].astype(np.float32)

        # Eval
        mean_bag, median_bag, info = _eval_arm(
            X_fam_sel, residual, anchor, y_unb, rae_anchor,
            label=f"MONO_{fam}"
        )
        verdict = _verdict_for(f"MONO_{fam}", info, rae_anchor)
        info["verdict"] = verdict
        info["family"] = fam
        info["K_eff"] = int(X_fam_sel.shape[1])
        info["family_size"] = int(n_cols)
        # Map local -> 117-col global idx for traceability
        s0, _s1 = family_slices[fam]
        global_idx_117 = (top_local + s0).astype(np.int32)
        info["top_idx_in_117"] = [int(x) for x in global_idx_117.tolist()]
        info["top_idx_local"] = [int(x) for x in top_local.tolist()]
        family_results[fam] = info
        family_mean_bag_oof[fam] = mean_bag

        # Save artifacts
        np.save(DATA_PROCESSED / f"{TAG}_{fam}_top_idx_in_117.npy",
                global_idx_117)
        np.save(DATA_PROCESSED / f"{TAG}_{fam}_mean_bag_oof.npy", mean_bag)
        print(f"   verdict = {verdict}")

    # Rank families by mean_bag
    fam_rank = sorted(
        FAMILY_ORDER,
        key=lambda f: family_results[f]["rae_mean_bag"],
    )
    print("\n" + "=" * 78)
    print(f"FAMILY RANK by mean_bag (best -> worst):")
    print("=" * 78)
    for rank, fam in enumerate(fam_rank, 1):
        info = family_results[fam]
        print(f"   {rank}. {fam:14s} mean_bag={info['rae_mean_bag']:.4f}  "
              f"median_bag={info['rae_median_bag']:.4f}  "
              f"K_eff={info['K_eff']:3d}  "
              f"d_vs_nb2103.K28 = "
              f"{info['delta_mean_bag_vs_nb2103_K28']:+.4f}")

    top1_fam = fam_rank[0]
    top2_fam = fam_rank[1]
    top3_fam = fam_rank[2]
    print(f"\n   top-1 = {top1_fam}")
    print(f"   top-2 = {top2_fam}")
    print(f"   top-3 = {top3_fam}  (diversity sanity)")

    # ============================================================
    # STEP 2: TWO-FAMILY COMBO  (top1+top2  AND  top1+top3)
    # ============================================================
    print("\n" + "=" * 78)
    print("STEP 2: TWO-FAMILY COMBOS")
    print("=" * 78)

    combo_results: dict[str, dict] = {}
    combo_mean_bag: dict[str, np.ndarray] = {}

    for combo_label, (fa, fb) in [
        ("top1_plus_top2", (top1_fam, top2_fam)),
        ("top1_plus_top3", (top1_fam, top3_fam)),
    ]:
        print("\n" + "-" * 78)
        print(f"COMBO: {combo_label}  ({fa} + {fb})")
        print("-" * 78)
        X_a = family_to_X[fa][:, family_top_idx_local[fa]]
        X_b = family_to_X[fb][:, family_top_idx_local[fb]]
        X_combo = np.concatenate([X_a, X_b], axis=1).astype(np.float32)
        print(f"   X_combo shape = {X_combo.shape}  "
              f"({fa}:{X_a.shape[1]} + {fb}:{X_b.shape[1]})")
        mean_bag, median_bag, info = _eval_arm(
            X_combo, residual, anchor, y_unb, rae_anchor,
            label=f"COMBO_{combo_label}"
        )
        verdict = _verdict_for(f"COMBO_{combo_label}", info, rae_anchor)
        info["verdict"] = verdict
        info["combo_label"] = combo_label
        info["family_a"] = fa
        info["family_b"] = fb
        info["dim_a"] = int(X_a.shape[1])
        info["dim_b"] = int(X_b.shape[1])
        info["dim_total"] = int(X_combo.shape[1])
        combo_results[combo_label] = info
        combo_mean_bag[combo_label] = mean_bag

        np.save(DATA_PROCESSED / f"{TAG}_{combo_label}_mean_bag_oof.npy",
                mean_bag)
        print(f"   verdict = {verdict}")

    # ============================================================
    # GLOBAL VERDICT
    # ============================================================
    print("\n" + "=" * 78)
    print("GLOBAL VERDICT")
    print("=" * 78)
    best_mono_fam = top1_fam
    best_mono_rae = family_results[top1_fam]["rae_mean_bag"]
    best_combo_label = min(
        combo_results.keys(),
        key=lambda k: combo_results[k]["rae_mean_bag"],
    )
    best_combo_rae = combo_results[best_combo_label]["rae_mean_bag"]
    best_arm_label, best_arm_rae = min(
        [(f"MONO_{best_mono_fam}", best_mono_rae),
         (f"COMBO_{best_combo_label}", best_combo_rae)],
        key=lambda t: t[1],
    )
    print(f"   best mono   = {best_mono_fam}    mean_bag = {best_mono_rae:.4f}")
    print(f"   best combo  = {best_combo_label}  mean_bag = {best_combo_rae:.4f}")
    print(f"   best overall = {best_arm_label}  mean_bag = {best_arm_rae:.4f}")

    if best_arm_rae < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
        global_verdict = f"PROMOTE_{best_arm_label}_BEATS_NB2103_K28"
    elif abs(best_arm_rae - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN:
        global_verdict = f"FLAT_{best_arm_label}_VS_NB2103_K28_AXIS_FLAT"
    else:
        global_verdict = "AXIS_CLOSED_MIXED_K28_DOMINATES_MONO_FAMILY"

    print(f"   global_verdict = {global_verdict}")
    print(f"   PRE-unblind clean = True")

    # ============================================================
    # SAVE SUMMARY
    # ============================================================
    summary = {
        "tag": TAG,
        "method": (
            "mono_family_ablation_K28_per_family_then_top2_combos_on_117col"
        ),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": (
            "nb2063/nb2103/nb2159/nb2165 117-col 5-way K-tuned matrix: "
            "AtomPair/MACCS/Mordred/ChempropEmbed/Avalon + ChEMBL kNN"
        ),
        "shap_source_model": "LGBMRegressor_MSE_seed0",
        "shap_seed": SHAP_SEED,
        "lgbm_params": _lgbm_params(0),
        "target_K": TARGET_K,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "family_order": FAMILY_ORDER,
        "family_slices_in_117": {
            k: [int(v[0]), int(v[1])] for k, v in family_slices.items()
        },
        "feat_dim_total": int(feat_dim_total),
        "feat_breakdown_full": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "chembl_knn": 2,
            "total": int(feat_dim_total),
        },
        "n_chembl_pool": int(len(pool)),
        "n_unb": n_unb,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "family_results": {
            fam: {
                "family": family_results[fam]["family"],
                "family_size": family_results[fam]["family_size"],
                "K_eff": family_results[fam]["K_eff"],
                "per_seed_rae": family_results[fam]["per_seed_rae"],
                "per_seed_records": family_results[fam]["per_seed_records"],
                "rae_per_seed_mean": family_results[fam]["rae_per_seed_mean"],
                "rae_per_seed_median": family_results[fam]["rae_per_seed_median"],
                "rae_per_seed_std": family_results[fam]["rae_per_seed_std"],
                "rae_per_seed_min": family_results[fam]["rae_per_seed_min"],
                "rae_per_seed_max": family_results[fam]["rae_per_seed_max"],
                "rae_mean_bag": family_results[fam]["rae_mean_bag"],
                "rae_median_bag": family_results[fam]["rae_median_bag"],
                "delta_mean_bag_vs_nb2103_K28":
                    family_results[fam]["delta_mean_bag_vs_nb2103_K28"],
                "delta_median_bag_vs_nb2103_K28":
                    family_results[fam]["delta_median_bag_vs_nb2103_K28"],
                "delta_mean_bag_vs_anchor":
                    family_results[fam]["delta_mean_bag_vs_anchor"],
                "top_idx_in_117": family_results[fam]["top_idx_in_117"],
                "top_idx_local": family_results[fam]["top_idx_local"],
                "verdict": family_results[fam]["verdict"],
            }
            for fam in FAMILY_ORDER
        },
        "family_rank_by_mean_bag": fam_rank,
        "top1_family": top1_fam,
        "top2_family": top2_fam,
        "top3_family": top3_fam,
        "combo_results": {
            label: {
                "combo_label": combo_results[label]["combo_label"],
                "family_a": combo_results[label]["family_a"],
                "family_b": combo_results[label]["family_b"],
                "dim_a": combo_results[label]["dim_a"],
                "dim_b": combo_results[label]["dim_b"],
                "dim_total": combo_results[label]["dim_total"],
                "per_seed_rae": combo_results[label]["per_seed_rae"],
                "per_seed_records": combo_results[label]["per_seed_records"],
                "rae_per_seed_mean": combo_results[label]["rae_per_seed_mean"],
                "rae_per_seed_median": combo_results[label]["rae_per_seed_median"],
                "rae_per_seed_std": combo_results[label]["rae_per_seed_std"],
                "rae_per_seed_min": combo_results[label]["rae_per_seed_min"],
                "rae_per_seed_max": combo_results[label]["rae_per_seed_max"],
                "rae_mean_bag": combo_results[label]["rae_mean_bag"],
                "rae_median_bag": combo_results[label]["rae_median_bag"],
                "delta_mean_bag_vs_nb2103_K28":
                    combo_results[label]["delta_mean_bag_vs_nb2103_K28"],
                "delta_median_bag_vs_nb2103_K28":
                    combo_results[label]["delta_median_bag_vs_nb2103_K28"],
                "delta_mean_bag_vs_anchor":
                    combo_results[label]["delta_mean_bag_vs_anchor"],
                "verdict": combo_results[label]["verdict"],
            }
            for label in combo_results
        },
        "best_mono_family": best_mono_fam,
        "best_mono_rae_mean_bag": best_mono_rae,
        "best_combo_label": best_combo_label,
        "best_combo_rae_mean_bag": best_combo_rae,
        "best_arm_label": best_arm_label,
        "best_arm_rae_mean_bag": best_arm_rae,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "global_verdict": global_verdict,
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
        "target_K",
        "rae_anchor_chemprop_aux",
        "nb2103_K28_mean_bag_ref", "nb2103_K28_median_bag_ref",
        "best_mono_family", "best_mono_rae_mean_bag",
        "best_combo_label", "best_combo_rae_mean_bag",
        "best_arm_label", "best_arm_rae_mean_bag",
        "global_verdict", "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== PER-FAMILY MONO RAE ====")
    for fam in res["family_rank_by_mean_bag"]:
        r = res["family_results"][fam]
        print(f"  {fam:14s} mean_bag={r['rae_mean_bag']:.4f}  "
              f"median_bag={r['rae_median_bag']:.4f}  "
              f"K_eff={r['K_eff']:3d}  "
              f"d_vs_nb2103.K28 = {r['delta_mean_bag_vs_nb2103_K28']:+.4f}")
    print("\n==== COMBOS ====")
    for label, r in res["combo_results"].items():
        print(f"  {label:18s} ({r['family_a']}+{r['family_b']})  "
              f"mean_bag={r['rae_mean_bag']:.4f}  "
              f"median_bag={r['rae_median_bag']:.4f}  "
              f"d_vs_nb2103.K28 = {r['delta_mean_bag_vs_nb2103_K28']:+.4f}")
