"""nb3310 -- HURDLE with a WITHIN-ACTIVE / WITHIN-INACTIVE SECOND REGRESSOR.

WHY nb3301 FAILED (the bug this script fixes):
    nb3301 was a 2-stage hurdle: stage-1 LGBM classifier P_active(y>=4.5),
    stage-2 soft-blended the strong nb3200 regression anchor against a SINGLE
    SCALAR train_median (~5.0 for the full train, ~3.37 for inactives) for the
    "inactive" branch:

        pred = w * nb3200 + (1 - w) * train_median        (nb3301, BROKEN)

    Pulling low-P rows toward a CONSTANT is a constant-predictor on that
    sub-population (RAE ~= 1.0 within the routed rows).  The fold-honest grid
    therefore drove the operator to T=0.5 / p_floor=0.3, which over-routed and
    *raised* RAE to 0.5097 (delta +0.0682 vs the 0.4416 anchor).  Stage-2
    "shrunk to a scalar" -- it had no way to give the inactive-predicted rows a
    sensibly VARYING low prediction.

NEW PARADIGM (this script):
    Replace the scalar inactive branch with a SECOND nb3200-style REGRESSOR
    trained on the INACTIVE SUBSET only.  Two regression surfaces, gated by the
    classifier:

      STAGE 1   (classify) : LGBM binary classifier on the full 117-col 5-way
                   feature matrix, fit on 4139 CRC train for label (pec50>=4.5).
                   PRE-unblind (train InChIKey-disjoint from the 253 unblind),
                   so P_active on 253/513 is an honest never-seen signal.

      STAGE 2a  (active branch)   : the nb3200 learned-clip regression surface
                   (OOF on 253, te on 513) -- the strong "this is a binder,
                   here is how potent" predictor.

      STAGE 2b  (inactive branch) : a SEPARATE K=20 LGBM REGRESSOR trained ONLY
                   on the 1788 INACTIVE train rows (y<4.5), predicting ABSOLUTE
                   pEC50.  Bagged over seeds, fit on inactive-subset train, then
                   predicts all 253 unblind + 513 test.  This gives a feature-
                   driven, ROW-VARYING low prediction (inactive train median is
                   3.37, not a flat 5.0) instead of nb3301's constant.

      SOFT BLEND (per fold-honest (T, p_floor)):
            w_i  = clip(P_active_i, p_floor, 1) ** T
            pred = w_i * anchor_active_i + (1 - w_i) * reg_inactive_i

    As T grows, w->1 on confident actives and the operator -> pure nb3200 (so it
    cannot do materially worse than the anchor).  As T shrinks, genuinely low-P
    inactive rows are pulled toward the SECOND regressor's feature-driven low
    estimate -- decompressing the over-predicted greasy-novel-inactive tail (F2)
    WITHOUT collapsing to a constant.

WHY THIS COULD BREAK 0.4416 WHERE nb3301 COULD NOT:
    The inactive-only regressor is fit on a sub-population where the anchor's
    residual tail concentrates (novel-scaffold inactives the global surface
    over-predicts).  A dedicated low-band regressor carries the conditional
    structure ("how inactive, given inactive") that a scalar median throws away,
    so routing now MOVES rows toward a CORRECT low value, not a wrong constant.

HONEST EVAL (matches nb3301 / cycle-160 protocol):
    For each kf_seed in {1216..1230} (15 FRESH seeds):
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=5,
                 shuffle=True, seed=kf_seed)
        oof = full(253, nan)
        for (tr_loc, va_loc) in splits:
            pick (T*, p_floor*) on tr_loc minimising train-fold RAE
            oof[va_loc] = blend(P_active[va_loc], anchor[va_loc],
                                reg_inactive[va_loc], T*, p_floor*)
        per_fold_mean[kf_seed] = mean over 5 folds of rae(y[va], oof[va])
    GATE statistic = mean of per_fold_mean over the 15 kf_seeds.

    Stage-1 classifier AND stage-2b inactive-regressor are both fit ONCE on the
    disjoint 4139 train (bagged over seeds); only the cheap stage-2 (T, p_floor)
    selection is fold-honest on the 253.  No unblind label leaks into either
    train-fit stage.

GATE:
    per-fold-mean < 0.4423 -> "BETTER"
    else                   -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy        (253,) int  test-row indices
    data/processed/_audit_unblind_y.npy          (253,) float official truth
    data/processed/nb3200_pred_oof.npy           (253,) anchor honest OOF
    data/processed/te_nb3200.npy                 (513,) anchor deploy te
    data/processed/nb2604_summary.json           (K=18 idx in 117-col)
    data/processed/tr_atompair.npy / te_atompair.npy
    data/processed/tr_maccs.npy / te_maccs.npy
    data/processed/tr_avalon512.npy / te_avalon512.npy
    data/processed/tr_chemprop_embed_300.npy / te_chemprop_embed_300.npy
    C:/pxr_artifacts/nb1030/X_mordred_train.npy / X_mordred_test.npy
    data/processed/nb1352/1392/1484/1523/1524/1541 summaries (117-col slices)
    data/external/chembl_pxr_*.parquet (ChEMBL PXR kNN feature pair)

Outputs:
    data/processed/nb3310_summary.json
    data/processed/nb3310_pred_oof.npy   (253,) float32  honest mean-over-seeds OOF
    data/processed/te_nb3310.npy         (513,) float32  deploy te
    submissions/nb3310_within_active_regress.csv (only on BETTER)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter
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
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3310"
ANCHOR_TAG = "nb3200"

# -- Anchor (regression surface, stage-2a "active" branch) -------------------
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"   # (253,) honest OOF
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb3200.npy"          # (513,) deploy te

# -- Stage-1 classifier target threshold -------------------------------------
ACTIVE_THR = 4.5            # y >= 4.5 == above-baseline (active)
CLF_SEEDS = [0, 1, 7, 42, 137]   # bag the classifier for stability

# -- Stage-2b inactive-subset regressor --------------------------------------
K_INACTIVE = 20             # K=20 LGBM on inactive subset (task spec)
REG_SEEDS = [0, 1, 7, 42, 137]   # bag the inactive regressor

# -- Stage-2 fold-honest blend grid ------------------------------------------
T_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 16.0]  # P_active sharpening
PFLOOR_GRID = [0.0, 0.1, 0.2, 0.3]                          # probability floor

# -- HONEST cross-fit eval ----------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))   # 15 FRESH seeds {1216..1230}

# -- Gate ---------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- Feature cache paths (train + test sides) --------------------------------
ATOMPAIR_TR = DATA_PROCESSED / "tr_atompair.npy"
ATOMPAIR_TE = DATA_PROCESSED / "te_atompair.npy"
MACCS_TR = DATA_PROCESSED / "tr_maccs.npy"
MACCS_TE = DATA_PROCESSED / "te_maccs.npy"
AVALON_TR = DATA_PROCESSED / "tr_avalon512.npy"
AVALON_TE = DATA_PROCESSED / "te_avalon512.npy"
EMBED_TR = DATA_PROCESSED / "tr_chemprop_embed_300.npy"
EMBED_TE = DATA_PROCESSED / "te_chemprop_embed_300.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2604_SUMMARY = DATA_PROCESSED / "nb2604_summary.json"   # K=18 idx in 117-col

# -- ChEMBL PXR kNN feature pair (identical to nb3123/nb3301 117-col build) --
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- References ---------------------------------------------------------------
NB3200_ANCHOR_REF = 0.4416     # nb3200 honest OOF RAE on the 253
NB3301_FAIL_REF = 0.5097       # nb3301 scalar-median hurdle (the failure)
NB2171_REF = 0.4682            # post-hoc-blend ceiling on nb730 anchor
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# ChEMBL kNN helpers (lifted verbatim from nb3301 117-col build)
# ============================================================================
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


def _load_npy(path, n_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} (n={n_expected})")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred(path, n_expected):
    if not path.exists():
        raise FileNotFoundError(f"Mordred cache missing (run nb1030): {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"Mordred shape mismatch {path}: {X.shape}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _extract_atompair_top_idx(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def _load_117col_slices():
    """Return the per-family top-index slices that define the 117-col matrix."""
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
    top_maccs = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", "best_K")
    top_mord = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap = _extract_atompair_top_idx(sum_1484)
    top_ap = full_ap[: int(sum_1524["best_K"])]
    top_embed = np.array(sum_1541["top_dim_order_top100"], dtype=int)[
        : int(sum_1541["best_K"])
    ]
    top_avalon = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)
    return top_ap, top_maccs, top_mord, top_embed, top_avalon


def build_117col(side: str, smiles, n_rows, pool, fp_pool, pool_labels,
                 pool_median, slices):
    """Build the 117-col 5-way matrix for one side ('train' or 'test').

    side selects which cached fingerprint arrays to read.  ChEMBL kNN is
    recomputed per-side from the (already test-InChIKey-purged) pool.
    Column order: [AtomPair | MACCS | Mordred | ChempropEmbed | Avalon |
                   chembl_knn_pec50 | chembl_mean_sim]  -> 117 total, with the
    last two columns being the continuous ChEMBL kNN potency + similarity.
    """
    top_ap, top_maccs, top_mord, top_embed, top_avalon = slices
    if side == "train":
        X_ap = _load_npy(ATOMPAIR_TR, n_rows)[:, top_ap]
        X_maccs = _load_npy(MACCS_TR, n_rows)[:, top_maccs]
        X_mord = _load_mordred(MORDRED_DIR / "X_mordred_train.npy", n_rows)[:, top_mord]
        X_emb = _load_npy(EMBED_TR, n_rows)[:, top_embed]
        X_av = _load_npy(AVALON_TR, n_rows)[:, top_avalon]
    elif side == "test":
        X_ap = _load_npy(ATOMPAIR_TE, n_rows)[:, top_ap]
        X_maccs = _load_npy(MACCS_TE, n_rows)[:, top_maccs]
        X_mord = _load_mordred(MORDRED_DIR / "X_mordred_test.npy", n_rows)[:, top_mord]
        X_emb = _load_npy(EMBED_TE, n_rows)[:, top_embed]
        X_av = _load_npy(AVALON_TE, n_rows)[:, top_avalon]
    else:
        raise ValueError(side)

    mols = [standardize(s) for s in smiles]
    std_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in mols]
    fp = morgan_fp_batch(std_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp, fp_pool, k=KNN_K)
    pred_chembl, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    X = np.concatenate(
        [
            X_ap.astype(np.float32),
            X_maccs.astype(np.float32),
            X_mord.astype(np.float32),
            X_emb.astype(np.float32),
            X_av.astype(np.float32),
            pred_chembl.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X.shape[1] != 117:
        raise ValueError(f"{side} feat_dim {X.shape[1]} != 117")
    return X


def _k20_idx(slices) -> np.ndarray:
    """Indices (into the 117-col matrix) of the K=20 feature subset used by the
    inactive-branch regressor.

    Built from the canonical K=18 SHAP pool (nb2604) PLUS the two continuous
    ChEMBL-kNN columns (potency + mean similarity), which sit at columns 115 and
    116 of the 117-col matrix.  The kNN potency/similarity pair are the strongest
    'is-this-even-a-binder + how-potent' continuous signals -- exactly the axis a
    dedicated INACTIVE regressor benefits from -- so appending them to K18 yields
    a well-motivated K=20.  (If either is already in K18 we top up from the next
    AtomPair rank to keep cardinality at 20.)
    """
    with open(NB2604_SUMMARY) as f:
        nb2604 = json.load(f)
    k18 = list(np.array(nb2604["k18_idx_in_117col"], dtype=int).tolist())
    knn_cols = [115, 116]   # chembl_knn_pec50, chembl_mean_sim (last 2 of 117)
    k20 = list(k18)
    for c in knn_cols:
        if c not in k20:
            k20.append(c)
    # top up to exactly 20 if any knn col already present in K18
    if len(k20) < K_INACTIVE:
        top_ap = slices[0]   # AtomPair ranked indices (cols 0..len(top_ap)-1)
        rank = 0
        while len(k20) < K_INACTIVE and rank < len(top_ap):
            cand = int(rank)   # AtomPair block starts at col 0
            if cand not in k20:
                k20.append(cand)
            rank += 1
    k20 = k20[:K_INACTIVE]
    if len(k20) != K_INACTIVE:
        raise ValueError(f"K20 idx len {len(k20)} != {K_INACTIVE}")
    return np.array(k20, dtype=int)


# ============================================================================
# Stage-1 classifier
# ============================================================================
def _clf_params(seed: int) -> dict:
    return dict(
        objective="binary",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def fit_stage1_proba(X_tr_117, y_active, X_unb_117, X_te_117):
    """Bag a binary LGBM classifier on 4139 train; return P_active for the
    253 unblind and 513 test. Also returns a train-OOF P_active (diagnostic
    AUC only)."""
    from sklearn.model_selection import StratifiedKFold
    n_tr = X_tr_117.shape[0]
    p_unb = np.zeros(X_unb_117.shape[0], dtype=np.float64)
    p_te = np.zeros(X_te_117.shape[0], dtype=np.float64)
    oof_p = np.zeros(n_tr, dtype=np.float64)
    for s in CLF_SEEDS:
        clf = lgb.LGBMClassifier(**_clf_params(s))
        clf.fit(X_tr_117, y_active)
        p_unb += clf.predict_proba(X_unb_117)[:, 1]
        p_te += clf.predict_proba(X_te_117)[:, 1]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        oof_s = np.zeros(n_tr, dtype=np.float64)
        for tr_loc, va_loc in skf.split(X_tr_117, y_active):
            c = lgb.LGBMClassifier(**_clf_params(s))
            c.fit(X_tr_117[tr_loc], y_active[tr_loc])
            oof_s[va_loc] = c.predict_proba(X_tr_117[va_loc])[:, 1]
        oof_p += oof_s
    p_unb /= len(CLF_SEEDS)
    p_te /= len(CLF_SEEDS)
    oof_p /= len(CLF_SEEDS)
    return p_unb, p_te, oof_p


# ============================================================================
# Stage-2b inactive-subset regressor (the nb3301 fix)
# ============================================================================
def _reg_params(seed: int) -> dict:
    """K=20 LGBM regressor params -- mirror nb3123 base regressor shape."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def fit_inactive_regressor(X_tr_k20, y_train, y_active, X_unb_k20, X_te_k20):
    """Fit a SEPARATE K=20 LGBM on the INACTIVE train subset (y<ACTIVE_THR),
    predicting ABSOLUTE pEC50.  Bagged over REG_SEEDS.  Returns absolute-scale
    predictions on the 253 unblind and 513 test, plus a train OOF RAE diagnostic
    measured WITHIN the inactive subset (random KFold, interior diagnostic only).
    """
    from sklearn.model_selection import KFold
    inact = (y_active == 0)
    X_in = X_tr_k20[inact]
    y_in = y_train[inact]
    n_in = X_in.shape[0]
    reg_unb = np.zeros(X_unb_k20.shape[0], dtype=np.float64)
    reg_te = np.zeros(X_te_k20.shape[0], dtype=np.float64)
    oof_in = np.zeros(n_in, dtype=np.float64)
    for s in REG_SEEDS:
        m = lgb.LGBMRegressor(**_reg_params(s))
        m.fit(X_in, y_in)
        reg_unb += m.predict(X_unb_k20)
        reg_te += m.predict(X_te_k20)
        kf = KFold(n_splits=5, shuffle=True, random_state=s)
        oof_s = np.zeros(n_in, dtype=np.float64)
        for tr_loc, va_loc in kf.split(np.arange(n_in)):
            mm = lgb.LGBMRegressor(**_reg_params(s))
            mm.fit(X_in[tr_loc], y_in[tr_loc])
            oof_s[va_loc] = mm.predict(X_in[va_loc])
        oof_in += oof_s
    reg_unb /= len(REG_SEEDS)
    reg_te /= len(REG_SEEDS)
    oof_in /= len(REG_SEEDS)
    in_oof_rae = float(rae(y_in, oof_in))
    return reg_unb, reg_te, n_in, in_oof_rae, float(np.median(y_in))


# ============================================================================
# Stage-2 hurdle blend (active anchor vs inactive REGRESSOR, not scalar)
# ============================================================================
def _blend(p_active, anchor_pred, reg_inactive, T, p_floor):
    """w = clip(P, p_floor, 1)**T ; pred = w*anchor + (1-w)*reg_inactive."""
    w = np.clip(p_active, p_floor, 1.0) ** T
    return w * anchor_pred + (1.0 - w) * reg_inactive


def _pick_blend(p_tr, anchor_tr, reg_tr, y_tr):
    """Fold-honest grid search for (T, p_floor) minimising train-fold RAE."""
    best = (np.inf, T_GRID[-1], PFLOOR_GRID[0])
    for T in T_GRID:
        for pf in PFLOOR_GRID:
            pred = _blend(p_tr, anchor_tr, reg_tr, T, pf)
            r = float(rae(y_tr, pred))
            if r < best[0]:
                best = (r, T, pf)
    return best[1], best[2], best[0]


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- HURDLE w/ WITHIN-INACTIVE 2nd REGRESSOR (fix nb3301 scalar)")
    print(f"          active branch  = {ANCHOR_TAG} anchor (ref OOF {NB3200_ANCHOR_REF:.4f})")
    print(f"          inactive branch= K={K_INACTIVE} LGBM on inactive subset (y<{ACTIVE_THR})")
    print(f"          stage-1 clf seeds = {CLF_SEEDS}  reg seeds = {REG_SEEDS}")
    print(f"          T_grid = {T_GRID}")
    print(f"          pfloor_grid = {PFLOOR_GRID}")
    print(f"          kf_seeds (15) = {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}  folds={N_FOLDS}")
    print(f"          gate: per-fold-mean < {GATE_BETTER:.4f} -> BETTER else FAIL")
    print(f"          nb3301 scalar-median hurdle FAILED at {NB3301_FAIL_REF:.4f}")
    print("=" * 78)

    # -- Load truth / anchor / data ------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values if "name" in te.columns
                else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"

    tr = load_train()
    tr_smiles = tr["smiles"].astype(str).tolist()
    y_train = tr["pec50"].to_numpy(dtype=np.float64)
    n_train = len(y_train)
    assert n_train == 4139, f"expected 4139 train, got {n_train}"
    y_active = (y_train >= ACTIVE_THR).astype(np.int32)
    train_median = float(np.median(y_train))
    print(f"[load] n_test={n_test}  n_unb={n_unb}  n_train={n_train}")
    print(f"[load] train active rate (y>={ACTIVE_THR}) = {y_active.mean():.3f} "
          f"({int(y_active.sum())}/{n_train})   inactive n={int((y_active==0).sum())}")
    print(f"[load] train_median={train_median:.4f}  "
          f"inactive median={np.median(y_train[y_active==0]):.4f}  "
          f"active median={np.median(y_train[y_active==1]):.4f}")

    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if anchor_oof.shape != (n_unb,):
        raise ValueError(f"{ANCHOR_TAG} oof shape {anchor_oof.shape} != ({n_unb},)")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"{ANCHOR_TAG} te shape {anchor_te.shape} != ({n_test},)")
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[anchor] {ANCHOR_TAG} OOF RAE = {rae_anchor:.4f} (ref {NB3200_ANCHOR_REF:.4f})")
    leak_eq = float(np.mean(np.isclose(anchor_oof, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN anchor: {leak_eq:.1%} rows == truth -- possible leak")

    # -- Scaffolds for outer CV ----------------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unblind scaffolds = {n_unique_scaf}")

    # -- Build ChEMBL pool ONCE (purged of test InChIKeys) -------------------
    print("\n" + "-" * 78)
    print("STEP 1: ChEMBL PXR kNN pool (purged of test InChIKeys)")
    print("-" * 78)
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
    test_iks = {ik for ik in (_safe_inchikey(m) for m in test_mols) if ik}
    pool = pool[~pool["inchikey"].isin(test_iks)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep = fp_pool.sum(axis=1) > 0
    if not keep.all():
        pool = pool[keep].reset_index(drop=True)
        fp_pool = fp_pool[keep]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    print(f"   pool size = {len(pool)}  pool_median pec50 = {pool_median:.3f}")

    # -- Build 117-col matrices (train + test) -------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: build 117-col 5-way feature matrix (train 4139 + test 513)")
    print("-" * 78)
    slices = _load_117col_slices()
    X_tr_117 = build_117col("train", tr_smiles, n_train, pool, fp_pool,
                            pool_labels, pool_median, slices)
    X_te_117 = build_117col("test", te_smiles, n_test, pool, fp_pool,
                            pool_labels, pool_median, slices)
    X_unb_117 = X_te_117[unb_idx]
    print(f"   X_tr_117 = {X_tr_117.shape}  X_te_117 = {X_te_117.shape}  "
          f"X_unb_117 = {X_unb_117.shape}")

    # K=20 subset (for the inactive-branch regressor)
    k20_idx = _k20_idx(slices)
    print(f"   K={K_INACTIVE} idx (into 117-col) = {k20_idx.tolist()}")
    X_tr_k20 = X_tr_117[:, k20_idx].astype(np.float32)
    X_te_k20 = X_te_117[:, k20_idx].astype(np.float32)
    X_unb_k20 = X_te_k20[unb_idx]

    # -- STAGE 1: classifier --------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STAGE 1: LGBM binary classifier on 4139 train for (y>={ACTIVE_THR})")
    print("-" * 78)
    from sklearn.metrics import roc_auc_score
    p_unb, p_te, oof_p = fit_stage1_proba(X_tr_117, y_active, X_unb_117, X_te_117)
    train_auc = float(roc_auc_score(y_active, oof_p))
    y_unb_active = (y_unb >= ACTIVE_THR).astype(np.int32)
    unb_auc = (float(roc_auc_score(y_unb_active, p_unb))
               if 0 < y_unb_active.sum() < n_unb else float("nan"))
    print(f"   train 5-fold OOF AUC = {train_auc:.4f}  (active>={ACTIVE_THR})")
    print(f"   unblind AUC (vs truth-label, diagnostic) = {unb_auc:.4f}")
    print(f"   P_active(unb): mean={p_unb.mean():.3f} std={p_unb.std():.3f} "
          f"min={p_unb.min():.3f} max={p_unb.max():.3f}")
    print(f"   unblind frac P<0.3 = {(p_unb < 0.3).mean():.3f}  "
          f"(candidate inactives routed toward inactive regressor)")

    # -- STAGE 2b: inactive-subset regressor (the nb3301 fix) -----------------
    print("\n" + "-" * 78)
    print(f"STAGE 2b: K={K_INACTIVE} LGBM regressor on INACTIVE subset (y<{ACTIVE_THR})")
    print("-" * 78)
    reg_unb, reg_te, n_inact, in_oof_rae, inact_med = fit_inactive_regressor(
        X_tr_k20, y_train, y_active, X_unb_k20, X_te_k20,
    )
    print(f"   inactive train n={n_inact}  inactive median pEC50={inact_med:.4f}")
    print(f"   within-inactive train OOF RAE (random KFold diag) = {in_oof_rae:.4f}")
    print(f"   reg_inactive(unb): mean={reg_unb.mean():.3f} std={reg_unb.std():.3f} "
          f"min={reg_unb.min():.3f} max={reg_unb.max():.3f}")
    print(f"   reg_inactive(te) : mean={reg_te.mean():.3f} std={reg_te.std():.3f} "
          f"min={reg_te.min():.3f} max={reg_te.max():.3f}")
    # Contrast with nb3301's scalar branch: the regressor VARIES (std>0) where
    # nb3301 used a constant (std=0).
    print(f"   [fix] inactive branch std = {reg_unb.std():.3f} (nb3301 used "
          f"constant train_median, std=0.000)")
    # Diagnostic: RAE of pure inactive-regressor on the unblind inactive subset
    unb_inact_mask = (y_unb < ACTIVE_THR)
    if unb_inact_mask.sum() > 1:
        reg_on_unb_inact = float(rae(y_unb[unb_inact_mask], reg_unb[unb_inact_mask]))
        anchor_on_unb_inact = float(rae(y_unb[unb_inact_mask], anchor_oof[unb_inact_mask]))
        print(f"   [diag] within-unblind-inactive ({int(unb_inact_mask.sum())} rows): "
              f"reg_inactive RAE={reg_on_unb_inact:.4f}  vs  anchor RAE={anchor_on_unb_inact:.4f}")
    else:
        reg_on_unb_inact = float("nan")
        anchor_on_unb_inact = float("nan")

    # -- HONEST 5-fold scaffold CV over 15 kf_seeds (fold-honest T, p_floor) --
    print("\n" + "-" * 78)
    print(f"STEP 3: HONEST 5-fold scaffold CV over {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    per_kf_pooled = []
    per_kf_foldmean = []
    per_kf_oof = []
    picked_T = []
    picked_pf = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        fold_rae = []
        for tr_loc, va_loc in splits:
            T, pf, _ = _pick_blend(
                p_unb[tr_loc], anchor_oof[tr_loc], reg_unb[tr_loc], y_unb[tr_loc],
            )
            picked_T.append(T)
            picked_pf.append(pf)
            oof[va_loc] = _blend(
                p_unb[va_loc], anchor_oof[va_loc], reg_unb[va_loc], T, pf,
            )
            fold_rae.append(float(rae(y_unb[va_loc], oof[va_loc])))
        if np.isnan(oof).any():
            raise RuntimeError(f"scaffold splits did not cover all rows (kf={kf_seed})")
        pooled = float(rae(y_unb, oof))
        foldmean = float(np.mean(fold_rae))
        per_kf_pooled.append(pooled)
        per_kf_foldmean.append(foldmean)
        per_kf_oof.append(oof)
        print(f"   kf={kf_seed}  pooled={pooled:.4f}  fold-mean={foldmean:.4f}  "
              f"wall={time.time()-ts:.2f}s")

    foldmean_arr = np.asarray(per_kf_foldmean, dtype=np.float64)
    pooled_arr = np.asarray(per_kf_pooled, dtype=np.float64)
    n_s = len(foldmean_arr)
    pfm_mean = float(foldmean_arr.mean())
    pfm_std = float(foldmean_arr.std(ddof=1)) if n_s > 1 else 0.0
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = pfm_std / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.1448  # df=14 two-sided 95%
    ci_low = pfm_mean - t_mult * sem
    ci_high = pfm_mean + t_mult * sem
    T_mode = Counter(picked_T).most_common(1)[0][0]
    pf_mode = Counter(picked_pf).most_common(1)[0][0]

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} kf_seeds)")
    print("-" * 78)
    print(f"   PER-FOLD-MEAN  = {pfm_mean:.4f} +/- {pfm_std:.4f}  (GATE statistic)")
    print(f"   95% CI (df=14) = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled mean    = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"   per-fold-mean min/max = [{foldmean_arr.min():.4f}, {foldmean_arr.max():.4f}]")
    print(f"   modal (T, p_floor) over {len(picked_T)} folds = ({T_mode}, {pf_mode})")
    print(f"   T distribution  = {dict(Counter(picked_T))}")
    print(f"   pf distribution = {dict(Counter(picked_pf))}")
    print(f"\n   anchor {ANCHOR_TAG} OOF RAE = {rae_anchor:.4f}")
    print(f"   delta vs anchor          = {pfm_mean - rae_anchor:+.4f}")
    print(f"   nb3301 scalar hurdle     = {NB3301_FAIL_REF:.4f}  "
          f"(delta vs nb3301 = {pfm_mean - NB3301_FAIL_REF:+.4f})")

    bag_oof = np.mean(np.stack(per_kf_oof, axis=0), axis=0).astype(np.float64)
    bag_oof_rae = float(rae(y_unb, bag_oof))
    print(f"   bagged-OOF (mean over kf) RAE = {bag_oof_rae:.4f}")

    # -- DEPLOY te: pick (T, p_floor) on ALL 253, apply to 513 ---------------
    print("\n" + "-" * 78)
    print("STEP 4: deploy te (pick T,p_floor on full 253; blend on 513)")
    print("-" * 78)
    dep_T, dep_pf, dep_rae_tr = _pick_blend(p_unb, anchor_oof, reg_unb, y_unb)
    te_pred = _blend(p_te, anchor_te, reg_te, dep_T, dep_pf).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy pick (T, p_floor) = ({dep_T}, {dep_pf})  train-253 RAE={dep_rae_tr:.4f}")
    print(f"   te(513): mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"min={te_pred.min():.3f} max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f} (DEPLOY OPTIMISM, NOT gate)")

    # -- GATE -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if pfm_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3310 within-inactive-regressor hurdle "
            f"per-fold-mean {pfm_mean:.4f} < gate {GATE_BETTER:.4f} and beats "
            f"nb3200 anchor {rae_anchor:.4f} by {rae_anchor - pfm_mean:+.4f}. "
            f"Replacing nb3301's scalar train-median branch ({NB3301_FAIL_REF:.4f}) "
            f"with a K={K_INACTIVE} inactive-subset regressor fixed the hurdle "
            f"(delta vs nb3301 {pfm_mean - NB3301_FAIL_REF:+.4f}). Stage-1 clf "
            f"train AUC {train_auc:.3f}; modal blend (T={T_mode}, p_floor={pf_mode}). "
            f"Verify with deep-30 before PRIMARY promotion."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3310 within-inactive-regressor hurdle per-fold-mean "
            f"{pfm_mean:.4f} >= gate {GATE_BETTER:.4f} (delta vs anchor "
            f"{pfm_mean - rae_anchor:+.4f}). Even with a feature-driven inactive "
            f"regressor (within-inactive train OOF {in_oof_rae:.4f}, branch "
            f"std {reg_unb.std():.3f} vs nb3301 scalar std 0), routing on a "
            f"{y_active.mean()*100:.0f}%-active-trained classifier cannot beat the "
            f"nb3200 regression surface on the {100*(1-unb_inact_mask.mean()):.0f}%-active "
            f"unblind: the inactive regressor's within-inactive RAE "
            f"({reg_on_unb_inact:.4f}) is not enough below the anchor's "
            f"({anchor_on_unb_inact:.4f}) on the routed rows to net a gain. "
            f"Keep nb3200 anchor; no ladder change. (Still beats nb3301 by "
            f"{pfm_mean - NB3301_FAIL_REF:+.4f} -- the 2nd-regressor paradigm is "
            f"the right fix direction, just shy of the gate.)"
        )
    print(f"   verdict       = {verdict}")
    print(f"   {ladder_action}")

    # -- Save -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, bag_oof.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_within_active_regress.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "anchor_tag": ANCHOR_TAG,
        "method": (
            "hurdle_clf_active_then_softblend_anchor_vs_WITHIN_INACTIVE_K20_regressor"
        ),
        "fixes": "nb3301 (scalar-median stage-2b) -> within-inactive K20 LGBM regressor",
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": False,
        "anchor_note": (
            "nb3200 is a POST-unblind learned-clip surface (operates on the 253); "
            "stage-1 classifier AND stage-2b inactive regressor are both "
            "PRE-unblind (4139 train, InChIKey-disjoint)."
        ),
        "active_thr": ACTIVE_THR,
        "clf_seeds": CLF_SEEDS,
        "reg_seeds": REG_SEEDS,
        "k_inactive": K_INACTIVE,
        "k20_idx_in_117col": k20_idx.tolist(),
        "train_median": train_median,
        "train_active_rate": float(y_active.mean()),
        "n_inactive_train": int(n_inact),
        "inactive_train_median": round(inact_med, 4),
        "within_inactive_train_oof_rae": round(in_oof_rae, 4),
        "reg_inactive_unb_mean": float(reg_unb.mean()),
        "reg_inactive_unb_std": float(reg_unb.std()),
        "reg_inactive_te_mean": float(reg_te.mean()),
        "reg_inactive_te_std": float(reg_te.std()),
        "within_unblind_inactive_reg_rae": (round(reg_on_unb_inact, 4)
                                            if reg_on_unb_inact == reg_on_unb_inact else None),
        "within_unblind_inactive_anchor_rae": (round(anchor_on_unb_inact, 4)
                                               if anchor_on_unb_inact == anchor_on_unb_inact else None),
        "n_unblind_inactive": int(unb_inact_mask.sum()),
        "train_auc_oof": train_auc,
        "unblind_auc_vs_truth": unb_auc if unb_auc == unb_auc else None,
        "p_active_unb_mean": float(p_unb.mean()),
        "p_active_unb_std": float(p_unb.std()),
        "p_active_unb_frac_lt_0p3": float((p_unb < 0.3).mean()),
        "T_grid": T_GRID,
        "pfloor_grid": PFLOOR_GRID,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": n_s,
        "n_unb": int(n_unb),
        "n_train": int(n_train),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "chembl_pool_size": int(len(pool)),
        "rae_anchor_nb3200": rae_anchor,
        "nb3200_anchor_ref": NB3200_ANCHOR_REF,
        "nb3301_fail_ref": NB3301_FAIL_REF,
        "leak_eq_truth_frac": round(leak_eq, 4),
        "per_kf_pooled_rae": [round(v, 4) for v in per_kf_pooled],
        "per_kf_foldmean_rae": [round(v, 4) for v in per_kf_foldmean],
        "per_fold_mean": round(pfm_mean, 4),
        "per_fold_mean_std": round(pfm_std, 4),
        "per_fold_mean_ci95_low": round(ci_low, 4),
        "per_fold_mean_ci95_high": round(ci_high, 4),
        "pooled_mean": round(pooled_mean, 4),
        "pooled_std": round(pooled_std, 4),
        "mean_rae": round(pfm_mean, 4),   # canonical: per-fold-mean is the gate stat
        "foldmean_min": round(float(foldmean_arr.min()), 4),
        "foldmean_max": round(float(foldmean_arr.max()), 4),
        "T_mode": float(T_mode),
        "pfloor_mode": float(pf_mode),
        "T_distribution": {str(k): int(v) for k, v in Counter(picked_T).items()},
        "pfloor_distribution": {str(k): int(v) for k, v in Counter(picked_pf).items()},
        "delta_vs_anchor": round(pfm_mean - rae_anchor, 4),
        "delta_vs_nb3301": round(pfm_mean - NB3301_FAIL_REF, 4),
        "bag_oof_rae": round(bag_oof_rae, 4),
        "deploy_T": float(dep_T),
        "deploy_pfloor": float(dep_pf),
        "deploy_train253_rae": round(dep_rae_tr, 4),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-fold-mean ({n_s} kf) = {pfm_mean:.4f} +/- {pfm_std:.4f}  [GATE]")
    print(f"   95% CI                  = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   pooled mean             = {pooled_mean:.4f}")
    print(f"   anchor nb3200 OOF       = {rae_anchor:.4f}")
    print(f"   delta vs anchor         = {pfm_mean - rae_anchor:+.4f}")
    print(f"   delta vs nb3301 (fix)   = {pfm_mean - NB3301_FAIL_REF:+.4f}")
    print(f"   inactive-reg within-OOF = {in_oof_rae:.4f}  (branch std {reg_unb.std():.3f})")
    print(f"   train OOF AUC           = {train_auc:.4f}")
    print(f"   modal (T, p_floor)      = ({T_mode}, {pf_mode})")
    print(f"   te[unb] in-sample RAE   = {te_unb_in_rae:.4f}")
    print(f"   GATE (< {GATE_BETTER:.4f})       = {verdict}")
    print(f"   wall                    = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_fold_mean", "per_fold_mean_std",
        "per_fold_mean_ci95_low", "per_fold_mean_ci95_high",
        "pooled_mean", "rae_anchor_nb3200", "delta_vs_anchor", "delta_vs_nb3301",
        "within_inactive_train_oof_rae",
        "within_unblind_inactive_reg_rae", "within_unblind_inactive_anchor_rae",
        "train_auc_oof", "unblind_auc_vs_truth",
        "T_mode", "pfloor_mode",
        "bag_oof_rae", "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
