"""nb2179 -- Anchor BLEND sweep + residual LGBM K=28.

HYPOTHESIS:
    nb2170 fixed anchor = 100% nb730 and achieved 0.3920 (mean-bag) /
    0.3936 (median-bag).  The pure chemprop_aux anchor (w=0.0) achieves
    a higher residual ceiling; somewhere in between, a CONVEX BLEND
    anchor may give the residual LGBM a better-conditioned target.
    Also test LGBM-stacked blend: feed anchor_blend as a feature so the
    booster can learn row-dependent weighting.

PROTOCOL:
    1.  Load te_nb730.npy (513,) and te_chemprop_aux.npy (513,).  Slice
        by unb_idx -> anchor_730[unb], anchor_aux[unb].
    2.  W_GRID = {0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0}.
    3.  For each w:
            anchor_w = w*nb730 + (1-w)*chemprop_aux
            rae_anchor_alone(w) = RAE(y_unb, anchor_w[unb])  (U-shape)
            residual_w = y_unb - anchor_w[unb]
            5-seed bag x 5-fold cross-fit LGBM(MSE) on X_unb_28
            mean_bag_w, median_bag_w RAE per w
    4.  Find w_opt (mean-bag minimum) and verdict vs nb2170 0.3920.
    5.  LGBM-stacked blend variant (per w):
            X_stack = concat[X_unb_28, anchor_w_col].astype(np.float32)
            residual = y_unb - anchor_w[unb] (same residual)
            cross-fit LGBM as above with the extra constant-per-row anchor
            feature, see if booster learns row-dependent weighting.
    6.  Conditional deploy CSV nb2179_deploy_blend_w<wopt>.csv if a w!=1
        beats 0.3920 by margin 0.003.

PRE-unblind features:  feature matrix X_unb_28 is rebuilt the same way
as nb2170 (117-col -> top-28 SHAP) -- no truth leakage.
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

TAG = "nb2179"
DEPLOY_TAG = "nb2179"

NB730_TE_PATH = DATA_PROCESSED / "te_nb730.npy"
CHEMPROP_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

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

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
TOP_K_SHAP = 28

# nb2170 reference (w=1.0 == 100% nb730 anchor)
NB2170_MEAN_BAG_REF = 0.3920
NB2170_MEDIAN_BAG_REF = 0.3936
TARGET_BEAT = NB2170_MEAN_BAG_REF  # 0.3920
DECISION_MARGIN = 0.003

W_GRID = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]


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
    w = np.clip(top_sim.copy(), 0.0, 1.0)
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


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X_te_m.shape} vs {n_test_expected}")
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


def _extract_K_record(sum_dict, records_key, K):
    for r in sum_dict[records_key]:
        if int(r["K"]) == K:
            return r
    raise KeyError(f"K={K} not found in {records_key}")


def _build_X_te_117(test_smiles, n_test):
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
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

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
    X_te_117 = np.concatenate(
        [
            X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
            pred_chembl_te.reshape(-1, 1).astype(np.float32),
            mean_sim_te.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return X_te_117


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Anchor BLEND sweep + residual LGBM K=28")
    print(f"          nb2170 ref (w=1.0): mean-bag={NB2170_MEAN_BAG_REF:.4f}  "
          f"median-bag={NB2170_MEDIAN_BAG_REF:.4f}")
    print(f"          TARGET = {TARGET_BEAT:.4f}, margin = {DECISION_MARGIN}")
    print(f"          W_GRID = {W_GRID}")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    test_smiles = (te["smiles"] if "smiles" in te.columns else te["SMILES"]).astype(str).tolist()
    if "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    elif "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    else:
        mol_names = te["name"].astype(str).tolist()

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not NB730_TE_PATH.exists() or not CHEMPROP_AUX_TE_PATH.exists():
        raise FileNotFoundError("missing nb730 or chemprop_aux te npy")
    te_nb730_513 = np.load(NB730_TE_PATH).astype(np.float64)
    te_aux_513 = np.load(CHEMPROP_AUX_TE_PATH).astype(np.float64)
    if te_nb730_513.shape[0] != n_test or te_aux_513.shape[0] != n_test:
        raise ValueError(
            f"anchor te shape mismatch: nb730={te_nb730_513.shape} aux={te_aux_513.shape} n_test={n_test}"
        )
    anchor_nb730_unb = te_nb730_513[unb_idx]
    anchor_aux_unb = te_aux_513[unb_idx]
    rae_nb730 = float(rae(y_unb, anchor_nb730_unb))
    rae_aux = float(rae(y_unb, anchor_aux_unb))
    print(f"[anchor] nb730 alone in_RAE = {rae_nb730:.4f}")
    print(f"[anchor] chemprop_aux alone in_RAE = {rae_aux:.4f}")

    # ---- Top-28 SHAP indices ----
    with open(NB2103_SUMMARY) as f:
        nb2103_sum = json.load(f)
    rec28 = _extract_K_record(nb2103_sum, "per_K_records", K=TOP_K_SHAP)
    top28_idx = np.array(rec28["top_K_idx_in_117"], dtype=np.int32)
    print(f"[reuse] nb2103 top-28 SHAP indices head 10: {top28_idx[:10].tolist()}")

    # ---- Feature matrix ----
    print("[feat] building 117-col X_te ... (this takes 60-120s for ChEMBL kNN)")
    X_te_117 = _build_X_te_117(test_smiles, n_test)
    X_te_28 = X_te_117[:, top28_idx].astype(np.float32)
    X_unb_28 = X_te_28[unb_idx]
    print(f"[feat] X_te_28 = {X_te_28.shape}  X_unb_28 = {X_unb_28.shape}")

    # ---- W-sweep ----
    print("\n" + "-" * 78)
    print("W-SWEEP: anchor_blend = w*nb730 + (1-w)*chemprop_aux")
    print("-" * 78)
    sweep_records = []
    best_mean_bag_oof_by_w = {}
    best_median_bag_oof_by_w = {}
    stacked_records = []

    for w in W_GRID:
        anchor_w_513 = w * te_nb730_513 + (1.0 - w) * te_aux_513
        anchor_w_unb = anchor_w_513[unb_idx]
        rae_anchor_w = float(rae(y_unb, anchor_w_unb))
        residual_w = y_unb - anchor_w_unb

        ts = time.time()
        per_seed = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae = []
        for i, s in enumerate(RESID_SEEDS):
            resid_oof_s = _residual_cross_fit_one_seed(X_unb_28, residual_w, s)
            corr_s = anchor_w_unb + resid_oof_s
            per_seed[i] = corr_s
            per_seed_rae.append(float(rae(y_unb, corr_s)))
        mean_bag = per_seed.mean(axis=0)
        median_bag = np.median(per_seed, axis=0)
        rae_mean = float(rae(y_unb, mean_bag))
        rae_median = float(rae(y_unb, median_bag))
        best_mean_bag_oof_by_w[w] = mean_bag.astype(np.float32)
        best_median_bag_oof_by_w[w] = median_bag.astype(np.float32)

        sweep_records.append({
            "w": float(w),
            "rae_anchor_alone": rae_anchor_w,
            "rae_mean_bag": rae_mean,
            "rae_median_bag": rae_median,
            "per_seed_rae": per_seed_rae,
            "resid_w_std": float(residual_w.std()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   w={w:.2f}  anchor_alone={rae_anchor_w:.4f}  "
              f"mean_bag={rae_mean:.4f}  median_bag={rae_median:.4f}  "
              f"wall={time.time() - ts:.1f}s")

        # ---- LGBM-stacked blend: add anchor_blend as extra feature ----
        # Use both anchors as features and residual to corpus-mean fit
        X_unb_29 = np.concatenate(
            [X_unb_28, anchor_w_unb.reshape(-1, 1).astype(np.float32)],
            axis=1,
        )
        per_seed_stk = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
        per_seed_stk_rae = []
        for i, s in enumerate(RESID_SEEDS):
            resid_oof_s = _residual_cross_fit_one_seed(X_unb_29, residual_w, s)
            corr_s = anchor_w_unb + resid_oof_s
            per_seed_stk[i] = corr_s
            per_seed_stk_rae.append(float(rae(y_unb, corr_s)))
        mean_bag_stk = per_seed_stk.mean(axis=0)
        median_bag_stk = np.median(per_seed_stk, axis=0)
        rae_mean_stk = float(rae(y_unb, mean_bag_stk))
        rae_median_stk = float(rae(y_unb, median_bag_stk))
        stacked_records.append({
            "w": float(w),
            "rae_mean_bag_stacked": rae_mean_stk,
            "rae_median_bag_stacked": rae_median_stk,
            "per_seed_rae_stacked": per_seed_stk_rae,
        })
        print(f"       stacked w={w:.2f}  mean_bag={rae_mean_stk:.4f}  "
              f"median_bag={rae_median_stk:.4f}")

    # ---- Pick best ----
    print("\n" + "=" * 78)
    print("W-SWEEP TABLE")
    print("=" * 78)
    print(f"   {'w':>4s}  {'anchor':>7s}  {'mean_bag':>8s}  {'median_bag':>10s}  "
          f"{'stk_mean':>8s}  {'stk_med':>8s}")
    for r, s in zip(sweep_records, stacked_records):
        print(f"   {r['w']:>4.2f}  {r['rae_anchor_alone']:>7.4f}  "
              f"{r['rae_mean_bag']:>8.4f}  {r['rae_median_bag']:>10.4f}  "
              f"{s['rae_mean_bag_stacked']:>8.4f}  {s['rae_median_bag_stacked']:>8.4f}")

    all_cands = []
    for r in sweep_records:
        all_cands.append((f"blend_w{r['w']:.2f}_mean", r["rae_mean_bag"], "mean", r["w"], False))
        all_cands.append((f"blend_w{r['w']:.2f}_median", r["rae_median_bag"], "median", r["w"], False))
    for s in stacked_records:
        all_cands.append((f"stack_w{s['w']:.2f}_mean", s["rae_mean_bag_stacked"], "mean", s["w"], True))
        all_cands.append((f"stack_w{s['w']:.2f}_median", s["rae_median_bag_stacked"], "median", s["w"], True))

    all_sorted = sorted(all_cands, key=lambda x: x[1])
    print("\n" + "=" * 78)
    print("TOP-10 CANDIDATES (sorted by RAE asc)")
    print("=" * 78)
    for name, r, kind, w_val, stacked in all_sorted[:10]:
        d_target = r - TARGET_BEAT
        flag = "  <-- BEATS TARGET" if d_target < -DECISION_MARGIN else (
            "  flat" if abs(d_target) < DECISION_MARGIN else "")
        print(f"   {name:30s}  RAE={r:.4f}  d_vs_{TARGET_BEAT:.4f}={d_target:+.4f}{flag}")

    best_name, best_rae, best_kind, best_w, best_stacked = all_sorted[0]
    beats_target = bool(best_rae < TARGET_BEAT - DECISION_MARGIN)
    flat_vs_target = bool(abs(best_rae - TARGET_BEAT) < DECISION_MARGIN)
    # Decide deploy: must be a w<1 (i.e. blend) AND beat target by margin
    is_blend = best_w != 1.0
    deploy_eligible = beats_target and is_blend

    if beats_target and is_blend:
        verdict = f"BLEND_w{best_w:.2f}_BEATS_NB2170_RAE_{best_rae:.4f}"
    elif beats_target and not is_blend:
        verdict = f"W1_STILL_BEST_NO_BLEND_GAIN_RAE_{best_rae:.4f}"
    elif flat_vs_target:
        verdict = f"FLAT_VS_NB2170_BEST_{best_name}_RAE_{best_rae:.4f}"
    else:
        verdict = f"NB2170_W1_REMAINS_BEST_{best_name}_RAE_{best_rae:.4f}"

    print(f"\n   global verdict = {verdict}")
    print(f"   best: name={best_name}  RAE={best_rae:.4f}  w={best_w}  "
          f"kind={best_kind}  stacked={best_stacked}")

    # ---- DEPLOY conditional ----
    deploy_built = False
    deploy_path = None
    te_deploy_stats = None
    if deploy_eligible:
        print("\n" + "-" * 78)
        print(f"DEPLOY: {best_name} beats {TARGET_BEAT:.4f} by "
              f"{TARGET_BEAT - best_rae:+.4f}, building deploy CSV")
        print("-" * 78)
        anchor_w_513 = best_w * te_nb730_513 + (1.0 - best_w) * te_aux_513
        anchor_w_unb = anchor_w_513[unb_idx]
        residual_w = y_unb - anchor_w_unb

        # 5 outer x 5 inner = 25 fits, then mean or median
        OUTER_SEEDS = [0, 1, 7, 42, 137]
        INNER_OFFSETS = [0, 1, 7, 42, 137]
        n_total = len(OUTER_SEEDS) * len(INNER_OFFSETS)
        all_resid_513 = np.zeros((n_total, n_test), dtype=np.float64)
        if best_stacked:
            X_train_full = np.concatenate(
                [X_unb_28, anchor_w_unb.reshape(-1, 1).astype(np.float32)], axis=1
            )
            X_test_full = np.concatenate(
                [X_te_28, anchor_w_513.reshape(-1, 1).astype(np.float32)], axis=1
            )
        else:
            X_train_full = X_unb_28
            X_test_full = X_te_28

        k_g = 0
        for o in OUTER_SEEDS:
            inner_seeds = [o * 1000 + s for s in INNER_OFFSETS]
            for s in inner_seeds:
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_train_full, residual_w)
                all_resid_513[k_g] = mdl.predict(X_test_full)
                k_g += 1
        if best_kind == "median":
            chosen_resid_513 = np.median(all_resid_513, axis=0)
        else:
            chosen_resid_513 = all_resid_513.mean(axis=0)

        te_deploy = anchor_w_513 + chosen_resid_513

        in_unb = te_deploy[unb_idx]
        rae_in = float(rae(y_unb, in_unb))
        print(f"   in-sample RAE on unb_idx = {rae_in:.4f}")
        print(f"   honest cross-fit RAE     = {best_rae:.4f}")

        df_sub = pd.DataFrame({
            "SMILES": test_smiles,
            "Molecule Name": mol_names,
            "pEC50": te_deploy.astype(np.float32),
        })
        if len(df_sub) != 513:
            raise ValueError(f"submission rows {len(df_sub)} != 513")
        w_tag = f"{best_w:.2f}".replace(".", "_")
        deploy_path = SUBMISSIONS_DIR / f"{DEPLOY_TAG}_deploy_blend_w{w_tag}.csv"
        df_sub.to_csv(deploy_path, index=False)
        te_path = DATA_PROCESSED / f"te_{DEPLOY_TAG}.npy"
        np.save(te_path, te_deploy.astype(np.float32))
        deploy_built = True
        te_deploy_stats = {
            "mean": float(te_deploy.mean()),
            "std": float(te_deploy.std()),
            "min": float(te_deploy.min()),
            "max": float(te_deploy.max()),
            "in_sample_rae_unb": rae_in,
            "honest_cross_fit_rae": best_rae,
            "winning_candidate": best_name,
            "winning_w": best_w,
            "winning_kind": best_kind,
            "winning_stacked": best_stacked,
        }
        print(f"   [save] {deploy_path}")
        print(f"   [save] {te_path}")
    else:
        print("\n   no deploy: best candidate is w=1.0 or does not beat target "
              f"by {DECISION_MARGIN}")

    # ---- Save w-keyed mean-bag OOFs for downstream ----
    np.savez(
        DATA_PROCESSED / f"{TAG}_mean_bag_by_w.npz",
        **{f"w_{w:.2f}": v for w, v in best_mean_bag_oof_by_w.items()},
    )

    summary = {
        "tag": TAG,
        "method": "anchor_blend_sweep_plus_residual_lgbm_K28_plus_stacked",
        "anchors": ["nb730", "chemprop_aux"],
        "anchor_te_paths": {
            "nb730": str(NB730_TE_PATH),
            "chemprop_aux": str(CHEMPROP_AUX_TE_PATH),
        },
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "feat_dim_full": int(X_te_117.shape[1]),
        "feat_dim_topK": int(TOP_K_SHAP),
        "feat_dim_stacked": int(TOP_K_SHAP + 1),
        "lgbm_params": _lgbm_params(0),
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "w_grid": W_GRID,
        "rae_nb730_anchor_alone": rae_nb730,
        "rae_chemprop_aux_anchor_alone": rae_aux,
        "sweep_records": sweep_records,
        "stacked_records": stacked_records,
        "nb2170_mean_bag_ref": NB2170_MEAN_BAG_REF,
        "nb2170_median_bag_ref": NB2170_MEDIAN_BAG_REF,
        "target_beat": TARGET_BEAT,
        "decision_margin": DECISION_MARGIN,
        "all_candidates_sorted_top20": [
            {"name": n, "rae": r, "kind": k, "w": w_val, "stacked": st}
            for n, r, k, w_val, st in all_sorted[:20]
        ],
        "best_name": best_name,
        "best_rae": best_rae,
        "best_kind": best_kind,
        "best_w": best_w,
        "best_stacked": best_stacked,
        "beats_target": beats_target,
        "is_blend": is_blend,
        "deploy_eligible": deploy_eligible,
        "flat_vs_target": flat_vs_target,
        "verdict": verdict,
        "deploy_built": deploy_built,
        "deploy_path": str(deploy_path) if deploy_path else None,
        "te_deploy_stats": te_deploy_stats,
        "pre_unblind_clean": True,
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
        "rae_nb730_anchor_alone",
        "rae_chemprop_aux_anchor_alone",
        "best_name",
        "best_rae",
        "best_w",
        "best_kind",
        "best_stacked",
        "beats_target",
        "is_blend",
        "deploy_eligible",
        "verdict",
        "deploy_built",
        "deploy_path",
    ):
        print(f"  {k}: {res.get(k)}")
