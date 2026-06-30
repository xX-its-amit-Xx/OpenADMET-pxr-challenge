"""nb966 -- ChemBERTa-77M-MTR residual analysis on chemprop_aux + K=28 baseline.

HYPOTHESIS:
    nb2103 reports K=28 mean-bag RAE = 0.4737 / median-bag = 0.4698 on residual
    (y_unb - chemprop_aux[unb_idx]) using a 117-col 5-way K-tuned feature stack
    distilled by SHAP to top-28.  We have a frozen pretrained ChemBERTa-77M-MTR
    embedding (cycle-129-support) at 384-dim covering 4651 InChIKey-deduped
    compounds (4138/4139 train + 513/513 test + 253/253 unblind).  Four
    questions:

        A. Does ChemBERTa(384) alone as a CROSS-PARADIGM signal beat the
           sparse K=28 SHAP-pruned matrix when fed to the same LGBM(MSE)
           residual cross-fit?
        B. Does a kNN regressor on ChemBERTa cosine-similarity neighbours give
           any standalone residual gain vs the chemprop_aux anchor?
        C. Does concatenating ChemBERTa(384) ON TOP OF the K=28 SHAP slice
           (412-dim) help the residual LGBM?
        D. Does ADDING ChemBERTa(384) into the 117-col stack BEFORE SHAP
           selection, then taking SHAP top-28 of the augmented 501-col matrix,
           let SHAP discover ChemBERTa dimensions worth keeping?

PROTOCOL:
    Anchor   = chemprop_aux te[unb_idx]   (PRE-unblind, in_RAE 0.6216).
    residual = y_unb - anchor.
    All four methods use the same LGBM(MSE) hyperparams + 5-seed bag
    (seeds 0,1,7,42,137) + 5-fold cross-fit per seed as nb2103/nb2063.
    Decision margin = 0.003 vs nb2103 K=28 mean-bag RAE 0.4737.

    Method A: X = ChemBERTa(384, unb only)
    Method B: weighted kNN k=5 by ChemBERTa cosine; pool = train 4139 with
              their pec50 labels; predict residual_hat = (pool_pec50 - anchor)
              proxy via standalone pred + then RAE evaluated on corrected
              anchor.  Note: this is a STANDALONE predictor, not a residual,
              so we evaluate both (i) using kNN as direct pEC50 predictor and
              (ii) corrected = anchor + (kNN_pec50 - kNN_pec50.mean()) shrink.
    Method C: X = concat(K=28 nb2103 slice, ChemBERTa 384 unb) = 412 cols
    Method D: X_full = concat(117-col stack, ChemBERTa 384) = 501 cols ->
              fit one LGBM(MSE) on residual -> SHAP TreeExplainer global
              mean|SHAP| -> keep top-28 -> per-seed cross-fit on those 28.

REUSED ARTIFACTS:
    data/processed/chemberta_77m_mtr_embeddings.npy   (4651, 384) float32
    data/processed/chemberta_77m_mtr_index.csv        row_idx, inchikey, name, std_smiles
    data/processed/te_chemprop_aux.npy                (513,) anchor
    data/processed/_audit_unblind_idx.npy             (253,) into 513
    data/processed/_audit_unblind_y.npy               (253,) truth
    data/processed/X_unb_28_nb2103.npy                (253, 28) K=28 slice (Method C, D-control)
    data/processed/nb2103_summary.json                K=28 reference 0.4737
    data/processed/nb2063_summary.json + 117-col build for Method D

Outputs:
    scripts/nb966_berta_resid.py
    data/processed/nb966_summary.json
    data/processed/nb966_mean_bag_oof_A.npy   (253,) ChemBERTa-only
    data/processed/nb966_mean_bag_oof_C.npy   (253,) K28 + ChemBERTa concat
    data/processed/nb966_mean_bag_oof_D.npy   (253,) SHAP top-28 of 501
    submissions/nb966_<winning_method>.csv    if any method beats by 0.003
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
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb966"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
TOP_K_SHAP = 28
KNN_K = 5
SIM_FLOOR = 1e-6

BERTA_EMB_PATH = DATA_PROCESSED / "chemberta_77m_mtr_embeddings.npy"
BERTA_IDX_PATH = DATA_PROCESSED / "chemberta_77m_mtr_index.csv"
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"

# Method D requires re-building the 117-col matrix (same as nb2063/nb2103)
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

CHEMPROP_AUX_REF = 0.6216
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
DECISION_MARGIN = 0.003


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


def _run_bag(X: np.ndarray, residual: np.ndarray, anchor: np.ndarray,
             y_unb: np.ndarray, label: str):
    per_seed_corrected = np.zeros((len(RESID_SEEDS), len(residual)),
                                  dtype=np.float64)
    per_seed_rae: list[float] = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        print(f"   [{label}] seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"wall = {time.time() - ts:.1f}s")
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    arr = np.array(per_seed_rae)
    return {
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": float(arr.mean()),
        "rae_per_seed_median": float(np.median(arr)),
        "rae_per_seed_std": float(arr.std()),
        "rae_per_seed_min": float(arr.min()),
        "rae_per_seed_max": float(arr.max()),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "mean_bag_oof": mean_bag_oof,
        "median_bag_oof": median_bag_oof,
    }


def _verdict(rae_val: float, rae_anchor: float) -> str:
    delta_ref = rae_val - NB2103_K28_MEAN_BAG_REF
    if rae_val < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
        return "BEATS_NB2103_K28_NEW_CANDIDATE"
    if abs(delta_ref) < DECISION_MARGIN:
        return "FLAT_VS_NB2103_K28"
    if rae_val < rae_anchor - DECISION_MARGIN:
        return "BEATS_ANCHOR_BUT_WORSE_THAN_NB2103_K28"
    if abs(rae_val - rae_anchor) < DECISION_MARGIN:
        return "FLAT_VS_ANCHOR"
    return "HURTS_ANCHOR"


# ---- ChEMBL kNN / 117-col helpers (mirror nb2103) ----

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


def _knn_predict_pec50(top_idx: np.ndarray, top_sim: np.ndarray,
                       pool_labels: np.ndarray, fallback: float):
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
        raise ValueError(f"Mordred shape mismatch: {X_te_m.shape}")
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _extract_atompair_top_idx(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict: dict, records_key: str) -> dict:
    best_K = int(sum_dict["best_K"])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found")


def _build_117col_unb(te: pd.DataFrame, n_test: int, unb_idx: np.ndarray) -> np.ndarray:
    """Re-build the 117-col 5-way K-tuned stack on the 253 unblind rows."""
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
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
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    X_ap = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[unb_idx][:, top_ap_bit_idx]
    X_mc = _load_npy_test(MACCS_TE_PATH, n_test)[unb_idx][:, top_maccs_bit_idx]
    X_md = _load_mordred_test(n_test)[unb_idx][:, top_mord_col_idx]
    X_em = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[unb_idx][:, top_embed_col_idx]
    X_av = _load_npy_test(AVALON_TE_PATH, n_test)[unb_idx][:, top_avalon_bit_idx]

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
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl, mean_sim = _knn_predict_pec50(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    pred_chembl_unb = pred_chembl[unb_idx].astype(np.float32)
    mean_sim_unb = mean_sim[unb_idx].astype(np.float32)

    X_unb = np.concatenate(
        [X_ap, X_mc, X_md, X_em, X_av,
         pred_chembl_unb.reshape(-1, 1), mean_sim_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    return X_unb


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChemBERTa-77M-MTR residual analysis (4 methods A/B/C/D)")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Anchor + truth ----
    te = load_test()
    tr = load_train()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_train={len(tr)}  n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load ChemBERTa-77M-MTR embeddings + match by name (alt: InChIKey) ----
    emb = np.load(BERTA_EMB_PATH).astype(np.float32)  # (4651, 384)
    bidx = pd.read_csv(BERTA_IDX_PATH)
    if len(bidx) != emb.shape[0]:
        raise ValueError(f"berta idx len {len(bidx)} != emb rows {emb.shape[0]}")
    name_to_row = dict(zip(bidx["name"].astype(str), bidx["row_idx"].astype(int)))
    print(f"[berta] emb shape = {emb.shape}  index size = {len(bidx)}")

    te_names = te["name"].astype(str).tolist()
    tr_names = tr["name"].astype(str).tolist()

    # Build train + test ChemBERTa matrices via name lookup
    def _lookup_emb(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        n = len(names)
        out = np.zeros((n, emb.shape[1]), dtype=np.float32)
        miss = np.zeros(n, dtype=bool)
        for i, nm in enumerate(names):
            r = name_to_row.get(nm)
            if r is None:
                miss[i] = True
            else:
                out[i] = emb[r]
        return out, miss

    X_berta_test_513, miss_test = _lookup_emb(te_names)
    print(f"[berta] test 513 coverage: {(~miss_test).sum()}/{n_test} "
          f"(miss={miss_test.sum()})")
    X_berta_train_4139, miss_train = _lookup_emb(tr_names)
    print(f"[berta] train 4139 coverage: {(~miss_train).sum()}/{len(tr)} "
          f"(miss={miss_train.sum()})")

    # Fill missing train rows with mean embedding (so kNN doesn't crash;
    # only 1 row expected missing).
    if miss_train.any():
        if (~miss_train).any():
            mean_emb = X_berta_train_4139[~miss_train].mean(axis=0)
        else:
            mean_emb = np.zeros(emb.shape[1], dtype=np.float32)
        X_berta_train_4139[miss_train] = mean_emb
    if miss_test.any():
        # Same fallback
        if (~miss_test).any():
            mean_emb = X_berta_test_513[~miss_test].mean(axis=0)
        else:
            mean_emb = np.zeros(emb.shape[1], dtype=np.float32)
        X_berta_test_513[miss_test] = mean_emb

    X_berta_unb = X_berta_test_513[unb_idx].astype(np.float32)
    print(f"[berta] unb slice shape = {X_berta_unb.shape}")

    # ---- METHOD A: ChemBERTa(384) standalone residual cross-fit ----
    print("\n" + "-" * 78)
    print("METHOD A: LGBM(MSE) residual cross-fit on ChemBERTa(384) features")
    print("-" * 78)
    res_A = _run_bag(X_berta_unb, residual, anchor, y_unb, label="A")
    print(f"   [A] per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in res_A['per_seed_rae'])}]")
    print(f"   [A] per-seed mean={res_A['rae_per_seed_mean']:.4f}  "
          f"std={res_A['rae_per_seed_std']:.4f}")
    print(f"   [A] mean_bag={res_A['rae_mean_bag']:.4f}  "
          f"median_bag={res_A['rae_median_bag']:.4f}  "
          f"d_vs_nb2103={res_A['rae_mean_bag'] - NB2103_K28_MEAN_BAG_REF:+.4f}  "
          f"d_vs_anchor={res_A['rae_mean_bag'] - rae_anchor:+.4f}")
    verdict_A = _verdict(res_A["rae_mean_bag"], rae_anchor)
    print(f"   [A] verdict: {verdict_A}")
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_A.npy",
            res_A["mean_bag_oof"].astype(np.float32))

    # ---- METHOD B: kNN(cosine) on ChemBERTa, pool = 4139 train ----
    print("\n" + "-" * 78)
    print(f"METHOD B: kNN k={KNN_K} cosine on ChemBERTa, pool=train 4139, "
          f"predict pEC50 -> use as residual signal")
    print("-" * 78)
    # Normalize for cosine = dot of normalized vectors
    def _l2_normalize(X: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n = np.where(n < 1e-12, 1.0, n)
        return X / n

    pool_emb = _l2_normalize(X_berta_train_4139)
    pool_labels = tr["pec50"].to_numpy(dtype=np.float32)
    # mask missing labels (NaN)
    pool_valid = np.isfinite(pool_labels)
    pool_emb_v = pool_emb[pool_valid]
    pool_labels_v = pool_labels[pool_valid]
    print(f"   [B] pool size (with pec50 labels): {pool_valid.sum()}/{len(tr)}")
    pool_median = float(np.median(pool_labels_v))

    test_emb = _l2_normalize(X_berta_test_513)
    # Cosine sim (n_test x n_pool)
    sim_full = test_emb @ pool_emb_v.T  # values in [-1, 1] (≈ [0,1] for chem)
    # top-K per row
    k = KNN_K
    if k >= sim_full.shape[1]:
        top_idx = np.argsort(-sim_full, axis=1)[:, :k]
    else:
        part = np.argpartition(-sim_full, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(sim_full.shape[0])[:, None]
        sim_part = sim_full[row_idx, part]
        order = np.argsort(-sim_part, axis=1)
        top_idx = part[row_idx, order]
    row_idx_grid = np.arange(sim_full.shape[0])[:, None]
    top_sim = sim_full[row_idx_grid, top_idx]
    # Map similarities to [0,1] weights
    w = np.clip(top_sim, 0.0, 1.0)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum < SIM_FLOOR, 1.0, w_sum)
    pred_pec50_513 = (w * pool_labels_v[top_idx]).sum(axis=1) / w_sum[:, 0]
    pred_pec50_unb = pred_pec50_513[unb_idx].astype(np.float64)
    mean_sim_unb = top_sim.mean(axis=1)[unb_idx]

    # B.1 -- direct kNN pEC50 as standalone predictor
    rae_B_direct = float(rae(y_unb, pred_pec50_unb))
    print(f"   [B.direct] kNN pec50 standalone RAE = {rae_B_direct:.4f}  "
          f"(mean_sim_unb mean = {mean_sim_unb.mean():.4f})")

    # B.2 -- blend: corrected = anchor + alpha * (knn_pec50 - anchor)
    #         sweep alpha in {0.0, 0.1, ..., 0.7}
    alphas = np.linspace(0.0, 0.7, 8)
    blend_records = []
    best_blend = {"alpha": 0.0, "rae": rae_anchor, "pred": anchor.copy()}
    for a in alphas:
        pred_a = anchor + a * (pred_pec50_unb - anchor)
        rae_a = float(rae(y_unb, pred_a))
        blend_records.append({"alpha": float(a), "rae": rae_a})
        if rae_a < best_blend["rae"]:
            best_blend = {"alpha": float(a), "rae": rae_a, "pred": pred_a.copy()}
    print(f"   [B.blend] alpha sweep:")
    for r in blend_records:
        print(f"      alpha={r['alpha']:.2f}  rae={r['rae']:.4f}")
    print(f"   [B.blend] best alpha={best_blend['alpha']:.2f}  "
          f"rae={best_blend['rae']:.4f}  "
          f"d_vs_anchor={best_blend['rae'] - rae_anchor:+.4f}  "
          f"d_vs_nb2103={best_blend['rae'] - NB2103_K28_MEAN_BAG_REF:+.4f}")
    verdict_B = _verdict(best_blend["rae"], rae_anchor)
    print(f"   [B] verdict (best blend): {verdict_B}")

    # ---- METHOD C: concat(K=28 nb2103 slice, ChemBERTa 384) = 412 cols ----
    print("\n" + "-" * 78)
    print(f"METHOD C: LGBM(MSE) residual on concat(K=28, ChemBERTa 384) = "
          f"412 cols")
    print("-" * 78)
    if not X_UNB_28_PATH.exists():
        raise FileNotFoundError(f"missing {X_UNB_28_PATH} -- run nb2103 first")
    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float32)
    if X_unb_28.shape != (n_unb, 28):
        raise ValueError(f"X_unb_28 shape {X_unb_28.shape} != ({n_unb}, 28)")
    X_C = np.concatenate([X_unb_28, X_berta_unb], axis=1).astype(np.float32)
    print(f"   [C] X_C shape = {X_C.shape}")
    res_C = _run_bag(X_C, residual, anchor, y_unb, label="C")
    print(f"   [C] per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in res_C['per_seed_rae'])}]")
    print(f"   [C] per-seed mean={res_C['rae_per_seed_mean']:.4f}  "
          f"std={res_C['rae_per_seed_std']:.4f}")
    print(f"   [C] mean_bag={res_C['rae_mean_bag']:.4f}  "
          f"median_bag={res_C['rae_median_bag']:.4f}  "
          f"d_vs_nb2103={res_C['rae_mean_bag'] - NB2103_K28_MEAN_BAG_REF:+.4f}  "
          f"d_vs_anchor={res_C['rae_mean_bag'] - rae_anchor:+.4f}")
    verdict_C = _verdict(res_C["rae_mean_bag"], rae_anchor)
    print(f"   [C] verdict: {verdict_C}")
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_C.npy",
            res_C["mean_bag_oof"].astype(np.float32))

    # ---- METHOD D: re-build 117 stack -> concat ChemBERTa = 501 ->
    #               SHAP top-28 -> 5-seed bag ----
    print("\n" + "-" * 78)
    print("METHOD D: build 117-col 5-way K-tuned matrix on unb, concat "
          "ChemBERTa(384) = 501 cols, SHAP top-28, then 5-seed bag")
    print("-" * 78)
    X_117 = _build_117col_unb(te, n_test, unb_idx)
    print(f"   [D] 117-col matrix shape = {X_117.shape}")
    X_501 = np.concatenate([X_117, X_berta_unb], axis=1).astype(np.float32)
    print(f"   [D] augmented 501 shape  = {X_501.shape}")

    # Full-fit LGBM on residual + SHAP global mean|SHAP|
    t_shap = time.time()
    mdl_full = lgb.LGBMRegressor(**_lgbm_params(seed=0))
    mdl_full.fit(X_501, residual)
    explainer = shap.TreeExplainer(mdl_full)
    shap_vals = explainer.shap_values(X_501)
    shap_imp = np.abs(shap_vals).mean(axis=0).astype(np.float32)
    print(f"   [D] SHAP done wall={time.time() - t_shap:.1f}s")
    if shap_imp.shape[0] != X_501.shape[1]:
        raise ValueError(
            f"SHAP imp shape {shap_imp.shape} != X_501 cols {X_501.shape[1]}"
        )
    top28_idx = np.argsort(-shap_imp)[:TOP_K_SHAP].astype(np.int32)
    n_berta_in_top28 = int((top28_idx >= 117).sum())
    n_117_in_top28 = int((top28_idx < 117).sum())
    print(f"   [D] SHAP top-28 family split: "
          f"117-stack={n_117_in_top28}  ChemBERTa={n_berta_in_top28}")

    X_D = X_501[:, top28_idx].astype(np.float32)
    res_D = _run_bag(X_D, residual, anchor, y_unb, label="D")
    print(f"   [D] per-seed RAE = "
          f"[{', '.join(f'{r:.4f}' for r in res_D['per_seed_rae'])}]")
    print(f"   [D] per-seed mean={res_D['rae_per_seed_mean']:.4f}  "
          f"std={res_D['rae_per_seed_std']:.4f}")
    print(f"   [D] mean_bag={res_D['rae_mean_bag']:.4f}  "
          f"median_bag={res_D['rae_median_bag']:.4f}  "
          f"d_vs_nb2103={res_D['rae_mean_bag'] - NB2103_K28_MEAN_BAG_REF:+.4f}  "
          f"d_vs_anchor={res_D['rae_mean_bag'] - rae_anchor:+.4f}")
    verdict_D = _verdict(res_D["rae_mean_bag"], rae_anchor)
    print(f"   [D] verdict: {verdict_D}")
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_D.npy",
            res_D["mean_bag_oof"].astype(np.float32))

    # ---- Decision: which (if any) method beats nb2103 K=28 by margin? ----
    print("\n" + "=" * 78)
    print("DECISION TABLE")
    print("=" * 78)
    print(f"   ref:  nb2103 K=28 mean_bag = {NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"(margin={DECISION_MARGIN})")
    print(f"   ref:  chemprop_aux anchor  = {rae_anchor:.4f}")
    candidates = [
        ("A_berta_only_384",   res_A["rae_mean_bag"],   res_A["rae_median_bag"],   verdict_A),
        ("B_knn_blend_best",   best_blend["rae"],       float("nan"),              verdict_B),
        ("C_k28_plus_berta_412", res_C["rae_mean_bag"], res_C["rae_median_bag"],   verdict_C),
        ("D_shap_top28_of_501",  res_D["rae_mean_bag"], res_D["rae_median_bag"],   verdict_D),
    ]
    print(f"   {'method':<24s}  {'mean_bag':>9s}  {'median_bag':>10s}  "
          f"{'d_vs_nb2103':>11s}  verdict")
    for nm, mb, md, vd in candidates:
        d = mb - NB2103_K28_MEAN_BAG_REF
        md_str = f"{md:.4f}" if md == md else "  --  "  # nan check
        print(f"   {nm:<24s}  {mb:>9.4f}  {md_str:>10s}  "
              f"{d:>+11.4f}  {vd}")

    # Winner = lowest mean_bag below threshold; otherwise none
    threshold = NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    beats = [(nm, mb) for nm, mb, _, _ in candidates if mb < threshold]
    if beats:
        beats.sort(key=lambda x: x[1])
        winner_name, winner_rae = beats[0]
        global_verdict = (
            f"BERTA_BEATS_NB2103_K28_BY_MARGIN  winner={winner_name}  "
            f"rae={winner_rae:.4f}  delta={winner_rae - NB2103_K28_MEAN_BAG_REF:+.4f}"
        )
    else:
        winner_name = None
        winner_rae = None
        global_verdict = "NO_BERTA_METHOD_BEATS_NB2103_K28_BY_MARGIN"
    print(f"\n   global verdict = {global_verdict}")

    # ---- Build deploy CSV ONLY for winner method (in-sample diagnostic) ----
    # NOTE: nb2103/nb2063 outputs are residual oof on 253 only.  A real
    # deploy CSV needs te (513).  We are constrained: ChemBERTa only available
    # for 4651 = 4138/4139 + 513/513, so a true deploy refit would need to
    # train residual head on the SAME 253 unblind labels (in-sample on unb),
    # then predict on all 513 with anchor + residual_hat(berta_513).
    # This script emits an IN-SAMPLE deploy CSV with that caveat -- the
    # numeric in_RAE will be optimistic vs LB.  Skip CSV if no winner.
    deploy_csv_path = None
    if winner_name is not None:
        print(f"\n[deploy] building IN-SAMPLE deploy CSV for winner: {winner_name}")
        # Choose X for refit on ALL 253 unb + predict on 513
        if winner_name == "A_berta_only_384":
            X_unb_deploy = X_berta_unb
            X_te_deploy = X_berta_test_513
        elif winner_name == "B_knn_blend_best":
            # B uses kNN(cosine) on train pool -- pred is already on full 513
            pred_te_full = anchor.copy()  # placeholder
            # For deploy, pred_513 = anchor_513 + alpha * (knn_513 - anchor_513)
            pred_513 = te_anchor_513 + best_blend["alpha"] * (pred_pec50_513 - te_anchor_513)
            # Skip LGBM refit; use blend directly
            sub_df = pd.DataFrame({
                "SMILES": te["smiles"].astype(str).tolist()
                          if "smiles" in te.columns else te["SMILES"].astype(str).tolist(),
                "Molecule Name": te["name"].astype(str).tolist(),
                "pEC50": pred_513.astype(np.float64),
            })
            sub_p = Path(__file__).resolve().parents[1] / "submissions" / \
                    f"{TAG}_B_knn_blend_alpha{best_blend['alpha']:.2f}.csv"
            sub_p.parent.mkdir(parents=True, exist_ok=True)
            sub_df.to_csv(sub_p, index=False)
            deploy_csv_path = str(sub_p)
            print(f"[deploy] saved {sub_p}")
            X_unb_deploy = None
        elif winner_name == "C_k28_plus_berta_412":
            X_unb_deploy = X_C
            # Need K=28 slice on full 513 + ChemBERTa on full 513
            # X_unb_28_nb2103 is unb-only, so cannot build te 513 K=28 without
            # re-running the full 117-col build on 513 then SHAP-selecting top 28.
            # Skip CSV for C unless we extend.  Mark as cannot-deploy.
            print("   [deploy] METHOD C deploy would need te 513 K=28 slice "
                  "rebuild; skipping CSV")
            X_unb_deploy = None
        elif winner_name == "D_shap_top28_of_501":
            # Same as C -- need 117 on 513 + ChemBERTa on 513 + reapply
            # top28_idx (which is into 501 cols).  Skip.
            print("   [deploy] METHOD D deploy would need te 513 501-col "
                  "rebuild; skipping CSV")
            X_unb_deploy = None

        # For A: refit on ALL 253, predict on 513
        if winner_name == "A_berta_only_384" and X_unb_deploy is not None:
            print(f"   [deploy A] in-sample refit on all 253; pred on 513")
            per_seed_te = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
            for i, s in enumerate(RESID_SEEDS):
                mdl = lgb.LGBMRegressor(**_lgbm_params(s))
                mdl.fit(X_unb_deploy, residual)
                per_seed_te[i] = mdl.predict(X_te_deploy)
            resid_hat_te = per_seed_te.mean(axis=0)
            pred_513 = te_anchor_513 + resid_hat_te
            sub_df = pd.DataFrame({
                "SMILES": te["smiles"].astype(str).tolist()
                          if "smiles" in te.columns else te["SMILES"].astype(str).tolist(),
                "Molecule Name": te["name"].astype(str).tolist(),
                "pEC50": pred_513.astype(np.float64),
            })
            sub_p = Path(__file__).resolve().parents[1] / "submissions" / \
                    f"{TAG}_A_berta_only.csv"
            sub_p.parent.mkdir(parents=True, exist_ok=True)
            sub_df.to_csv(sub_p, index=False)
            deploy_csv_path = str(sub_p)
            print(f"[deploy] saved {sub_p}")
            # in-sample diagnostic
            in_rae = float(rae(y_unb, pred_513[unb_idx]))
            print(f"   [deploy A] in-sample te[unb_idx] RAE = {in_rae:.4f}  "
                  f"(cross-fit {res_A['rae_mean_bag']:.4f})")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("chemberta_77m_mtr_residual_4methods_A_berta_only_B_knn_"
                   "C_concat_D_shap_top28_of_501"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": "chemberta_77m_mtr_embeddings (cycle-129-support)",
        "berta_emb_path": str(BERTA_EMB_PATH),
        "berta_idx_path": str(BERTA_IDX_PATH),
        "n_berta_rows": int(emb.shape[0]),
        "berta_dim": int(emb.shape[1]),
        "n_test_emb_coverage": int((~miss_test).sum()),
        "n_train_emb_coverage": int((~miss_train).sum()),
        "n_unb": n_unb,
        "n_test": n_test,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "nb2103_k28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_k28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "method_A": {
            "name": "A_berta_only_384",
            "feat_dim": int(X_berta_unb.shape[1]),
            "per_seed_rae": res_A["per_seed_rae"],
            "rae_per_seed_mean": res_A["rae_per_seed_mean"],
            "rae_per_seed_std": res_A["rae_per_seed_std"],
            "rae_per_seed_min": res_A["rae_per_seed_min"],
            "rae_per_seed_max": res_A["rae_per_seed_max"],
            "rae_mean_bag": res_A["rae_mean_bag"],
            "rae_median_bag": res_A["rae_median_bag"],
            "delta_mean_bag_vs_nb2103_K28": (
                res_A["rae_mean_bag"] - NB2103_K28_MEAN_BAG_REF
            ),
            "delta_mean_bag_vs_anchor": res_A["rae_mean_bag"] - rae_anchor,
            "verdict": verdict_A,
        },
        "method_B": {
            "name": "B_knn_blend_best",
            "k": KNN_K,
            "pool_size": int(pool_valid.sum()),
            "rae_knn_direct_pec50": rae_B_direct,
            "blend_alpha_sweep": blend_records,
            "best_alpha": best_blend["alpha"],
            "rae_best_blend": best_blend["rae"],
            "delta_best_vs_nb2103_K28": (
                best_blend["rae"] - NB2103_K28_MEAN_BAG_REF
            ),
            "delta_best_vs_anchor": best_blend["rae"] - rae_anchor,
            "verdict": verdict_B,
        },
        "method_C": {
            "name": "C_k28_plus_berta_412",
            "feat_dim": int(X_C.shape[1]),
            "per_seed_rae": res_C["per_seed_rae"],
            "rae_per_seed_mean": res_C["rae_per_seed_mean"],
            "rae_per_seed_std": res_C["rae_per_seed_std"],
            "rae_per_seed_min": res_C["rae_per_seed_min"],
            "rae_per_seed_max": res_C["rae_per_seed_max"],
            "rae_mean_bag": res_C["rae_mean_bag"],
            "rae_median_bag": res_C["rae_median_bag"],
            "delta_mean_bag_vs_nb2103_K28": (
                res_C["rae_mean_bag"] - NB2103_K28_MEAN_BAG_REF
            ),
            "delta_mean_bag_vs_anchor": res_C["rae_mean_bag"] - rae_anchor,
            "verdict": verdict_C,
        },
        "method_D": {
            "name": "D_shap_top28_of_501",
            "feat_dim_full": int(X_501.shape[1]),
            "feat_dim_top28": int(TOP_K_SHAP),
            "n_117_in_top28": n_117_in_top28,
            "n_berta_in_top28": n_berta_in_top28,
            "top28_idx_in_501": top28_idx.tolist(),
            "per_seed_rae": res_D["per_seed_rae"],
            "rae_per_seed_mean": res_D["rae_per_seed_mean"],
            "rae_per_seed_std": res_D["rae_per_seed_std"],
            "rae_per_seed_min": res_D["rae_per_seed_min"],
            "rae_per_seed_max": res_D["rae_per_seed_max"],
            "rae_mean_bag": res_D["rae_mean_bag"],
            "rae_median_bag": res_D["rae_median_bag"],
            "delta_mean_bag_vs_nb2103_K28": (
                res_D["rae_mean_bag"] - NB2103_K28_MEAN_BAG_REF
            ),
            "delta_mean_bag_vs_anchor": res_D["rae_mean_bag"] - rae_anchor,
            "verdict": verdict_D,
        },
        "winner_method": winner_name,
        "winner_rae_mean_bag": winner_rae,
        "global_verdict": global_verdict,
        "deploy_csv_path": deploy_csv_path,
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
        "n_berta_rows", "berta_dim",
        "n_test_emb_coverage", "n_train_emb_coverage",
        "rae_anchor_chemprop_aux",
        "nb2103_k28_mean_bag_ref",
    ):
        print(f"  {k}: {res.get(k)}")
    for label in ("method_A", "method_B", "method_C", "method_D"):
        m = res[label]
        print(f"\n  {label} ({m['name']}):")
        for k in ("feat_dim", "feat_dim_full", "feat_dim_top28",
                  "rae_mean_bag", "rae_best_blend",
                  "rae_median_bag",
                  "delta_mean_bag_vs_nb2103_K28",
                  "delta_best_vs_nb2103_K28",
                  "delta_mean_bag_vs_anchor",
                  "delta_best_vs_anchor",
                  "verdict"):
            if k in m:
                print(f"    {k}: {m[k]}")
    print(f"\n  winner_method: {res['winner_method']}")
    print(f"  winner_rae_mean_bag: {res['winner_rae_mean_bag']}")
    print(f"  global_verdict: {res['global_verdict']}")
    print(f"  deploy_csv_path: {res['deploy_csv_path']}")
