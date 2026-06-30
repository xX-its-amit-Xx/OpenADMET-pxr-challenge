"""nb3351 -- blend FRESH chemprop v3 (nb3350) with nb3200 via per-fold
SLSQP simplex on {nb3200, fresh_K18_v3} + learned per-fold quantile clip.

NEW PARADIGM (substrate change, cf. cycle-167 nb2171 / cycle-169 axes-closed):
    The post-hoc-blend ceiling on the FROZEN chemprop_aux anchor chain has
    co-converged at ~0.4424 (nb3200 deep-30) and the off-the-frozen-anchor
    swap (nb2171) bottomed at 0.4682. Every operator-level move (rank-stretch,
    spectral feats, trimmed bags, residual-on-residual) is closed on the frozen
    substrate. The ONLY open lever is a genuinely NEW anchor axis.

    nb3350 trains a FRESH 5-fold multitask chemprop (chemprop_aux_v3) from
    scratch. If that fresh anchor is (a) usable (unblind RAE < 0.65) AND
    (b) decorrelated from the frozen chain (Pearson < 0.95 vs frozen
    chemprop_aux on the 253), then it carries orthogonal error structure the
    frozen-anchor pyramid cannot. This script tests whether folding it in
    breaks the 0.4423 wall.

PIPELINE (only if nb3350 anchor is usable + decorrelated):
    1. fresh_K18_v3 := learned residual lift on the FRESH chemprop v3 anchor.
       Build the SAME 117-col 5-way feature matrix used by the frozen-anchor
       residual chain (AtomPair/MACCS/Mordred/ChempropEmbed/Avalon + ChEMBL
       kNN), slice the top K=18 RFE survivors (first 18 of nb2280
       K20_rfe_surviving_idx_in_117 ranked list -- a K-sweep slice of the same
       RFE ranking that drove the K=20 residual). Per-fold LGBM(K=18) fits the
       (y - fresh_v3) residual and lifts the fresh anchor:
           fresh_K18_v3_oof[va] = fresh_v3_oof[va] + lgbm.predict(X_K18[va]).
       This puts the fresh anchor on the same residual-lift footing as nb3200's
       parent before blending, so SLSQP sees two comparably-strong members.
    2. Per-fold SLSQP simplex on P = stack([nb3200_oof, fresh_K18_v3_oof]):
       fit w (>=0, sum=1) minimizing RAE on fold-TRAIN, apply to fold-VAL.
    3. Learned per-fold quantile clip (same operator as nb3200): grid-search
       (ql, qh) on the fold-TRAIN blended residual; clip fold-VAL to the
       fold-train [Q_ql, Q_qh] band. Mode-vote the band across folds for deploy.

ANCHORS (both must be PRE-clean to keep the blend PRE-clean):
    1. nb3200      -- learned per-fold clip on nb3090 (frozen chain). deep-30
                      mean RAE 0.4424. PRE-unblind (anchor_pre_unblind=True).
    2. fresh_v3    -- nb3350 fresh multitask chemprop trained on 4139 train,
                      253+513 held out. PRE-unblind by construction (never
                      sees the 253 truth). Lifted to fresh_K18_v3 by an LGBM
                      whose features are PRE-unblind and whose residual target
                      is fold-local truth (honest, fit inside the CV only).

PROTOCOL:
    15 FRESH kf_seeds {1216..1230}, 5-fold scaffold split on 253 unblind,
    per-fold-mean reported (cycle-160 deep-30 rule: 15-seed is hypothesis-grade;
    a BREAKTHROUGH here MUST be re-verified at deep-30 before any PRIMARY swap).

GATE (on 15-seed per-fold-mean):
    per_fold_mean < 0.4423 -> "BREAKTHROUGH"
    else                   -> "FAIL"

DEPENDENCY GUARD (task protocol):
    * nb3350 outputs absent           -> "SKIP_UNUSABLE" (anchor not produced)
    * nb3350 unblind RAE > 0.70       -> "SKIP_UNUSABLE" (anchor unusable)
    * nb3350 usable but NOT decorrel. -> "SKIP_UNUSABLE" (no orthogonality to
                                          blend; Pearson >= 0.95 means the fresh
                                          anchor is a copy of the frozen chain)
    In every SKIP case a summary json is still written (verdict SKIP_UNUSABLE)
    and NO pred_oof / te / submission is emitted.

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy        data/processed/te_nb3200.npy
    data/processed/nb3350_chemprop_v3_oof.npy data/processed/te_nb3350_chemprop_v3.npy
    data/processed/nb3350_summary.json        (GATE flags)
    data/processed/te_chemprop_aux.npy        (frozen anchor, decorrel check)
    data/processed/nb2280_summary.json        (K20 RFE idx -> first 18)
    + nb1352/1392/1484/1523/1524/1541 summaries (117-col feature recipe)
    + te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy, te_avalon512.npy
    + C:/pxr_artifacts/nb1030/X_mordred_test.npy
    + data/external/chembl_pxr_CHEMBL3401.parquet (+ siblings)

Outputs (only on a NON-skip run):
    data/processed/nb3351_summary.json        (always written)
    data/processed/nb3351_pred_oof.npy        (253,) float32 -- median-seed OOF
    data/processed/te_nb3351.npy              (513,) float32 -- deploy te
    submissions/nb3351_blend_fresh_chemprop.csv  (only on BREAKTHROUGH)
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
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3351"

# -- Frozen-chain anchor (nb3200 = learned clip on nb3090) -------------------
NB3200_OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"
NB3200_TE_PATH = DATA_PROCESSED / "te_nb3200.npy"

# -- Fresh chemprop v3 anchor (nb3350) ---------------------------------------
FRESH_OOF_PATH = DATA_PROCESSED / "nb3350_chemprop_v3_oof.npy"
FRESH_TE_PATH = DATA_PROCESSED / "te_nb3350_chemprop_v3.npy"
NB3350_SUMMARY = DATA_PROCESSED / "nb3350_summary.json"

# -- Frozen chemprop_aux (for decorrelation cross-check) ---------------------
FROZEN_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# -- 117-col feature recipe (identical to nb3262 / nb3163 / nb2960) ----------
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
NB2280_SUMMARY = DATA_PROCESSED / "nb2280_summary.json"   # K20 RFE idx ranked

# -- ChEMBL kNN feature config (identical to upstream) -----------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- K=18 residual on fresh v3 (K-sweep slice of nb2280 RFE rank) -------------
K_RESID = 18

# -- CV protocol -------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- SLSQP options -----------------------------------------------------------
N_SLSQP_STARTS = 8
DEGEN_MAX_W = 0.90

# -- Learned per-fold quantile clip grid (same operator as nb3200) -----------
Q_LOW_GRID = [0.00, 0.01, 0.02, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98, 0.99, 1.00]

# -- Dependency gates (task protocol) ----------------------------------------
SKIP_RAE_THRESHOLD = 0.70      # nb3350 unblind RAE > this -> SKIP_UNUSABLE
DECORREL_PEARSON_MAX = 0.95    # Pearson >= this vs frozen -> not decorrelated

# -- Promotion gate ----------------------------------------------------------
GATE_BREAKTHROUGH = 0.4423     # per-fold-mean < this -> BREAKTHROUGH

# -- References --------------------------------------------------------------
REF_NB3200 = 0.4424      # cycle-160 deep-30 mean (PRIMARY-1 candidate)
REF_NB3200_STD = 0.0023
REF_NB3090 = 0.4472
REF_NB2171 = 0.4682
REF_NB1191 = 0.4718
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (117-col feature builder, lifted verbatim from nb3262)
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


def _load_chembl_pool():
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


def _lgbm_params(seed):
    """K=18 LGBM(MSE) residual lift -- same shape as nb3262 K=20 (depth=4)."""
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


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing (run nb1030): {mte_p}")
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


def build_117col_feature_matrix(te_smiles, n_test):
    """117-col matrix identical to nb3262 / nb3163 / nb2604 / nb2960."""
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
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

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

    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
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
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full, int(len(pool))


# ============================================================================
# SLSQP simplex (lifted from nb3254)
# ============================================================================

def _simplex_slsqp(P, y, n_starts=N_SLSQP_STARTS, seed=0):
    """Minimize RAE(y, P @ w) on the simplex (w>=0, sum(w)=1)."""
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w):
        return float(rae(y, P @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    starts = [np.full(K, 1.0 / K)]
    for _ in range(n_starts - 1):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w = None
    best_r = np.inf
    for x0 in starts:
        try:
            res = minimize(
                loss, x0, method="SLSQP",
                bounds=bnds, constraints=cons,
                options={"maxiter": 300, "ftol": 1e-9},
            )
            w = np.clip(res.x, 0.0, 1.0)
            s = float(w.sum())
            if s <= 0.0:
                continue
            w = w / s
            r = float(rae(y, P @ w))
            if r < best_r:
                best_r = r
                best_w = w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_r = float(rae(y, P @ best_w))
    return best_w, best_r


# ============================================================================
# learned per-fold quantile clip (same operator family as nb3200/nb3190)
# ============================================================================

def _best_clip_band(y_tr, pred_tr):
    """Grid-search (ql, qh) minimizing RAE on fold-train; return band+quantiles.

    The clip band is computed on the fold-TRAIN PREDICTION distribution (so the
    band is data-derived, not truth-derived) and chosen by the (ql, qh) that
    minimizes fold-train RAE of the clipped prediction. Returns the chosen
    (ql, qh, lo, hi). On the degenerate case the widest band (no clip) is kept.
    """
    best = (0.0, 1.0, -np.inf, np.inf)
    best_r = float(rae(y_tr, pred_tr))
    for ql in Q_LOW_GRID:
        lo = float(np.quantile(pred_tr, ql)) if ql > 0.0 else -np.inf
        for qh in Q_HIGH_GRID:
            hi = float(np.quantile(pred_tr, qh)) if qh < 1.0 else np.inf
            if lo >= hi:
                continue
            clipped = np.clip(pred_tr, lo, hi)
            r = float(rae(y_tr, clipped))
            if r < best_r - 1e-12:
                best_r = r
                best = (ql, qh, lo, hi)
    return best


# ============================================================================
# K=18 residual lift on the FRESH chemprop v3 anchor (honest, fold-local)
# ============================================================================

def _fresh_resid_lift_oof(
    X_unb_K18, fresh_oof, y_unb, splits, kf_seed,
):
    """fresh_K18_v3 OOF: per fold, LGBM(K=18) on (y - fresh) residual lifts
    the fresh anchor on the fold-VAL rows. Returns the lifted OOF (no leak)."""
    n_unb = len(y_unb)
    lifted = np.full(n_unb, np.nan, dtype=np.float64)
    resid = y_unb - fresh_oof
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X_unb_K18[tr_loc], resid[tr_loc])
        lifted[va_loc] = fresh_oof[va_loc] + mdl.predict(X_unb_K18[va_loc])
    if np.isnan(lifted).any():
        raise RuntimeError(f"fresh-lift OOF not fully covered (kf={kf_seed})")
    return lifted


# ============================================================================
# one kf_seed: fresh-lift -> per-fold SLSQP{nb3200, fresh_K18_v3} -> clip
# ============================================================================

def _run_one_seed(
    nb3200_oof, X_unb_K18, fresh_oof, y_unb, unb_scaffolds, kf_seed,
):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_train_raes = []
    fold_weights = []
    fold_bands = []   # (ql, qh)
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        # 1) fresh-lift inside this fold only (honest): fit residual LGBM on
        #    fold-TRAIN, lift BOTH fold-train and fold-val fresh predictions.
        resid_tr = y_unb[tr_loc] - fresh_oof[tr_loc]
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X_unb_K18[tr_loc], resid_tr)
        fresh_lift_tr = fresh_oof[tr_loc] + mdl.predict(X_unb_K18[tr_loc])
        fresh_lift_va = fresh_oof[va_loc] + mdl.predict(X_unb_K18[va_loc])

        # 2) per-fold SLSQP simplex on {nb3200, fresh_K18_v3} using fold-train.
        P_tr = np.stack([nb3200_oof[tr_loc], fresh_lift_tr], axis=1)
        P_va = np.stack([nb3200_oof[va_loc], fresh_lift_va], axis=1)
        w, _ = _simplex_slsqp(
            P_tr, y_unb[tr_loc],
            n_starts=N_SLSQP_STARTS, seed=kf_seed * 100 + fold_i,
        )
        fold_weights.append(w)
        blend_tr = P_tr @ w
        blend_va = P_va @ w

        # 3) learned per-fold quantile clip (band from fold-train blend).
        ql, qh, lo, hi = _best_clip_band(y_unb[tr_loc], blend_tr)
        fold_bands.append((ql, qh))
        clip_va = np.clip(blend_va, lo, hi)
        clip_tr = np.clip(blend_tr, lo, hi)

        oof_blend[va_loc] = clip_va
        fold_val_raes.append(float(rae(y_unb[va_loc], clip_va)))
        fold_train_raes.append(float(rae(y_unb[tr_loc], clip_tr)))

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf={kf_seed}: scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    W = np.stack(fold_weights, axis=0)  # (n_folds, 2)
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "per_fold_train_rae_mean": float(np.mean(fold_train_raes)),
        "fold_weights": W,
        "fold_bands": fold_bands,
        "any_fold_degenerate": bool(W.max(axis=1).max() > DEGEN_MAX_W),
        "oof": oof_blend,
    }


# ============================================================================
# SKIP path: write a SKIP summary, emit nothing else
# ============================================================================

def _write_skip(reason, extra=None):
    summary = {
        "tag": TAG,
        "method": ("blend_fresh_chemprop_v3_nb3350_with_nb3200_via_perfold_slsqp_"
                   "plus_learned_clip"),
        "depends_on": ["nb3350", "nb3200"],
        "verdict": "SKIP_UNUSABLE",
        "skip_reason": reason,
        "gate_breakthrough": GATE_BREAKTHROUGH,
        "skip_rae_threshold": SKIP_RAE_THRESHOLD,
        "decorrel_pearson_max": DECORREL_PEARSON_MAX,
        "ref_nb3200_deep30_mean": REF_NB3200,
        "pred_oof_path": None,
        "te_npy_path": None,
        "submission_csv": None,
    }
    if extra:
        summary.update(extra)
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SKIP_UNUSABLE] {reason}")
    print(f"   [save] {out_path}")
    return summary


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- blend FRESH chemprop v3 (nb3350) + nb3200 "
          f"(per-fold SLSQP + learned clip)")
    print(f"          fresh-lift K   : K={K_RESID} (first {K_RESID} of nb2280 "
          f"RFE rank)")
    print(f"          kf_seeds       : {len(KF_SEEDS)} FRESH "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          gate           : per_fold_mean < {GATE_BREAKTHROUGH:.4f}"
          f" -> BREAKTHROUGH else FAIL")
    print("=" * 78)

    # -- DEPENDENCY GUARD ----------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 0: nb3350 dependency guard")
    print("-" * 78)
    missing = [p.name for p in (FRESH_OOF_PATH, FRESH_TE_PATH)
               if not p.exists()]
    if missing:
        print(f"   nb3350 outputs missing: {missing}")
        return _write_skip(
            f"nb3350 anchor not produced (missing: {missing}); fresh "
            f"chemprop v3 5-fold run did not complete. Re-run "
            f"scripts/nb3350_fresh_chemprop_5fold.py to generate the fresh "
            f"anchor before this blend can be evaluated.",
            extra={"nb3350_outputs_missing": missing},
        )

    fresh_oof = np.load(FRESH_OOF_PATH).astype(np.float64)   # (253,)
    fresh_te = np.load(FRESH_TE_PATH).astype(np.float64)     # (513,)

    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns
                 else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values if "name" in te.columns
                else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    if fresh_oof.shape != (n_unb,):
        return _write_skip(
            f"nb3350 fresh oof shape {fresh_oof.shape} != ({n_unb},)",
            extra={"fresh_oof_shape": list(fresh_oof.shape)},
        )
    if fresh_te.shape != (n_test,):
        return _write_skip(
            f"nb3350 fresh te shape {fresh_te.shape} != ({n_test},)",
            extra={"fresh_te_shape": list(fresh_te.shape)},
        )

    fresh_rae = float(rae(y_unb, fresh_oof))
    fresh_leak = float(np.mean(np.isclose(fresh_oof, y_unb, atol=1e-6)))
    print(f"[load] n_test={n_test}  n_unb={n_unb}")
    print(f"[fresh] nb3350 oof RAE = {fresh_rae:.4f}  "
          f"leak_eq={fresh_leak:.2%}  "
          f"mean={fresh_oof.mean():.3f} std={fresh_oof.std():.3f}")

    # Decorrelation vs frozen chemprop_aux (on the 253) ----------------------
    pearson_vs_frozen_aux = None
    if FROZEN_AUX_TE_PATH.exists():
        frozen_aux_te = np.load(FROZEN_AUX_TE_PATH).astype(np.float64)
        if frozen_aux_te.shape == (n_test,):
            frozen_aux_253 = frozen_aux_te[unb_idx]
            pearson_vs_frozen_aux = float(
                np.corrcoef(fresh_oof, frozen_aux_253)[0, 1]
            )
    # Prefer nb3350's own recorded pearson_fresh_vs_frozen_253 when present.
    nb3350_pearson = None
    nb3350_usable_flag = None
    nb3350_decorrel_flag = None
    if NB3350_SUMMARY.exists():
        with open(NB3350_SUMMARY) as f:
            s3350 = json.load(f)
        nb3350_pearson = s3350.get("pearson_fresh_vs_frozen_253")
        nb3350_usable_flag = s3350.get("GATE_usable_anchor")
        nb3350_decorrel_flag = s3350.get("GATE_decorrelated")
    pearson_used = (
        nb3350_pearson if nb3350_pearson is not None
        else pearson_vs_frozen_aux
    )
    print(f"[fresh] Pearson vs frozen chemprop_aux(253) = "
          f"{pearson_vs_frozen_aux}  (nb3350-recorded={nb3350_pearson})")
    print(f"[fresh] nb3350 flags: usable={nb3350_usable_flag} "
          f"decorrelated={nb3350_decorrel_flag}")

    # Leak guard (a fresh anchor must NOT be bit-identical to truth) ----------
    if fresh_leak > 0.05:
        return _write_skip(
            f"nb3350 fresh anchor leaks: {fresh_leak:.1%} rows == truth "
            f"(atol 1e-6); cannot use a truth-contaminated anchor.",
            extra={"fresh_anchor_unblind_RAE": round(fresh_rae, 4),
                   "fresh_leak_eq_truth_frac": round(fresh_leak, 4)},
        )

    # RAE usability gate (task protocol: RAE > 0.70 -> SKIP) -----------------
    if fresh_rae > SKIP_RAE_THRESHOLD:
        return _write_skip(
            f"nb3350 fresh anchor unusable: unblind RAE {fresh_rae:.4f} "
            f"> {SKIP_RAE_THRESHOLD:.2f}. Fresh chemprop v3 is too weak to "
            f"contribute orthogonal signal to the nb3200 blend.",
            extra={"fresh_anchor_unblind_RAE": round(fresh_rae, 4)},
        )

    # Decorrelation gate -----------------------------------------------------
    if pearson_used is None:
        return _write_skip(
            "cannot assess decorrelation: neither nb3350 summary "
            "pearson_fresh_vs_frozen_253 nor te_chemprop_aux available.",
            extra={"fresh_anchor_unblind_RAE": round(fresh_rae, 4)},
        )
    if pearson_used >= DECORREL_PEARSON_MAX:
        return _write_skip(
            f"nb3350 fresh anchor NOT decorrelated: Pearson "
            f"{pearson_used:.4f} >= {DECORREL_PEARSON_MAX:.2f} vs frozen "
            f"chemprop_aux on the 253. A near-copy of the frozen chain carries "
            f"no orthogonal error structure for the blend to exploit.",
            extra={"fresh_anchor_unblind_RAE": round(fresh_rae, 4),
                   "pearson_fresh_vs_frozen_253": round(pearson_used, 4)},
        )

    print(f"   GUARD PASSED: fresh RAE {fresh_rae:.4f} < {SKIP_RAE_THRESHOLD} "
          f"AND Pearson {pearson_used:.4f} < {DECORREL_PEARSON_MAX} "
          f"(usable + decorrelated)")

    # -- Load nb3200 frozen-chain anchor -------------------------------------
    nb3200_oof = np.load(NB3200_OOF_PATH).astype(np.float64)
    nb3200_te = np.load(NB3200_TE_PATH).astype(np.float64)
    if nb3200_oof.shape != (n_unb,):
        raise ValueError(f"nb3200 oof shape {nb3200_oof.shape} != ({n_unb},)")
    if nb3200_te.shape != (n_test,):
        raise ValueError(f"nb3200 te shape {nb3200_te.shape} != ({n_test},)")
    rae_nb3200 = float(rae(y_unb, nb3200_oof))
    nb3200_leak = float(np.mean(np.isclose(nb3200_oof, y_unb, atol=1e-6)))
    corr_fresh_nb3200 = float(np.corrcoef(fresh_oof, nb3200_oof)[0, 1])
    print(f"[nb3200] oof RAE = {rae_nb3200:.4f}  (ref {REF_NB3200:.4f})  "
          f"leak_eq={nb3200_leak:.2%}")
    print(f"[corr] Pearson(fresh, nb3200) on 253 = {corr_fresh_nb3200:.4f}")

    # -- Scaffolds for honest CV ---------------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique = {n_unique_scaf}")

    # -- Build 117-col matrix, slice K=18 ------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: build 117-col 5-way matrix, slice K={K_RESID} (nb2280 RFE "
          f"rank[:{K_RESID}])")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    with open(NB2280_SUMMARY) as f:
        nb2280 = json.load(f)
    K20_idx_ranked = np.array(nb2280["K20_rfe_surviving_idx_in_117"], dtype=int)
    K18_idx = K20_idx_ranked[:K_RESID]
    assert len(K18_idx) == K_RESID, f"K18 len {len(K18_idx)} != {K_RESID}"
    print(f"   K={K_RESID} idx in 117col: {K18_idx.tolist()}")
    X_te_K18 = X_te_full[:, K18_idx].astype(np.float32)
    X_unb_K18 = X_te_K18[unb_idx]
    assert X_unb_K18.shape == (n_unb, K_RESID)
    print(f"   X_unb_K18 = {X_unb_K18.shape}  X_te_K18 = {X_te_K18.shape}")

    # -- Diagnostic: standalone fresh_K18_v3 OOF (no blend, full-CV lift) -----
    diag_splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEEDS[0],
    )
    fresh_lift_diag = _fresh_resid_lift_oof(
        X_unb_K18, fresh_oof, y_unb, diag_splits, kf_seed=KF_SEEDS[0],
    )
    rae_fresh_lift = float(rae(y_unb, fresh_lift_diag))
    print(f"   [diag] fresh_K18_v3 standalone OOF RAE "
          f"(seed {KF_SEEDS[0]}) = {rae_fresh_lift:.4f} "
          f"(fresh raw {fresh_rae:.4f}, lift "
          f"{rae_fresh_lift - fresh_rae:+.4f})")

    # -- Multi-seed honest cross-fit -----------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 2: HONEST {N_FOLDS}-fold scaffold CV over {len(KF_SEEDS)} "
          f"kf_seeds (fresh-lift -> SLSQP{{nb3200, fresh_K18_v3}} -> clip)")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    oof_stack = []
    all_fold_weights = []
    all_fold_bands = []
    any_degen = False
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(
            nb3200_oof, X_unb_K18, fresh_oof, y_unb, unb_scaffolds, kf_seed=s,
        )
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_fold_weights.append(res["fold_weights"])
        all_fold_bands.extend(res["fold_bands"])
        any_degen = any_degen or res["any_fold_degenerate"]
        seed_mean_w = res["fold_weights"].mean(axis=0)
        seed_mean_w = seed_mean_w / max(seed_mean_w.sum(), 1e-12)
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_train_rae_mean": round(res["per_fold_train_rae_mean"], 4),
            "mean_w_nb3200": round(float(seed_mean_w[0]), 3),
            "mean_w_fresh_K18_v3": round(float(seed_mean_w[1]), 3),
            "any_fold_degenerate": res["any_fold_degenerate"],
        })
        print(f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
              f"perfold_mean={res['per_fold_val_rae_mean']:.4f}  "
              f"perfold_std={res['per_fold_val_rae_std']:.4f}  "
              f"w(nb3200={seed_mean_w[0]:.2f},fresh={seed_mean_w[1]:.2f})  "
              f"degen={res['any_fold_degenerate']}  "
              f"wall={time.time()-ts:.2f}s")

    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    pf_arr = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(pooled_arr)

    mean_pooled = float(pooled_arr.mean())
    std_pooled = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem_pooled = std_pooled / np.sqrt(n_s) if n_s > 1 else 0.0

    mean_pf = float(pf_arr.mean())
    std_pf = float(pf_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem_pf = std_pf / np.sqrt(n_s) if n_s > 1 else 0.0

    t_mult = 2.145  # df=14, two-sided 95%
    ci_low_pf = mean_pf - t_mult * sem_pf
    ci_high_pf = mean_pf + t_mult * sem_pf
    median_pf = float(np.median(pf_arr))

    # Mean-of-fold SLSQP weights across all 75 folds
    all_W = np.concatenate(all_fold_weights, axis=0)  # (75, 2)
    w_mean_of_folds = all_W.mean(axis=0)
    w_mean_of_folds = w_mean_of_folds / max(w_mean_of_folds.sum(), 1e-12)

    # Mode-vote clip band across all 75 folds (for deploy)
    bands_arr = np.array(all_fold_bands, dtype=float)  # (75, 2) -> (ql, qh)
    ql_vals, ql_counts = np.unique(bands_arr[:, 0], return_counts=True)
    qh_vals, qh_counts = np.unique(bands_arr[:, 1], return_counts=True)
    ql_mode = float(ql_vals[int(np.argmax(ql_counts))])
    qh_mode = float(qh_vals[int(np.argmax(qh_counts))])

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled  mean    = {mean_pooled:.4f}  std = {std_pooled:.4f}")
    print(f"   perfold mean    = {mean_pf:.4f}  std = {std_pf:.4f}")
    print(f"   perfold sem     = {sem_pf:.4f}")
    print(f"   perfold 95% CI  = [{ci_low_pf:.4f}, {ci_high_pf:.4f}]")
    print(f"   perfold median  = {median_pf:.4f}")
    print(f"   perfold min/max = [{pf_arr.min():.4f}, {pf_arr.max():.4f}]")
    print(f"   mean-of-folds w : nb3200={w_mean_of_folds[0]:.4f}  "
          f"fresh_K18_v3={w_mean_of_folds[1]:.4f}")
    print(f"   clip mode band  : ql={ql_mode:.2f}  qh={qh_mode:.2f}")
    print(f"\n   ref nb3200 (deep-30 mean) = {REF_NB3200:.4f} +/- "
          f"{REF_NB3200_STD:.4f}")
    print(f"   delta vs nb3200           = {mean_pf - REF_NB3200:+.4f}")
    print(f"   delta vs gate             = {mean_pf - GATE_BREAKTHROUGH:+.4f}")

    # -- Deploy: refit fresh-lift on ALL 253, mean-bag te; SLSQP w; clip ------
    print("\n" + "-" * 78)
    print("STEP 3: deploy -- refit fresh-lift on all 253 (mean-bag te), "
          "apply mean-of-folds w, apply mode clip band")
    print("-" * 78)
    resid_all = y_unb - fresh_oof
    sum_te_resid = np.zeros(n_test, dtype=np.float64)
    sum_unb_resid = np.zeros(n_unb, dtype=np.float64)
    for s in KF_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb_K18, resid_all)
        sum_te_resid += mdl.predict(X_te_K18).astype(np.float64)
        sum_unb_resid += mdl.predict(X_unb_K18).astype(np.float64)
    fresh_lift_te = fresh_te + sum_te_resid / len(KF_SEEDS)
    fresh_lift_unb_deploy = fresh_oof + sum_unb_resid / len(KF_SEEDS)

    # Blend te with mean-of-folds weights
    TE_stack = np.stack([nb3200_te, fresh_lift_te], axis=1)  # (513, 2)
    blend_te = TE_stack @ w_mean_of_folds

    # Deploy clip band: derive lo/hi from the deploy blended-unblind dist using
    # the mode (ql, qh) -- same convention as nb3200 deploy quantiles.
    blend_unb_deploy = (
        np.stack([nb3200_oof, fresh_lift_unb_deploy], axis=1) @ w_mean_of_folds
    )
    deploy_lo = (float(np.quantile(blend_unb_deploy, ql_mode))
                 if ql_mode > 0.0 else -np.inf)
    deploy_hi = (float(np.quantile(blend_unb_deploy, qh_mode))
                 if qh_mode < 1.0 else np.inf)
    te_pred = np.clip(blend_te, deploy_lo, deploy_hi).astype(np.float32)
    n_clip_hi = int(np.sum(blend_te > deploy_hi))
    n_clip_lo = int(np.sum(blend_te < deploy_lo))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   fresh_lift_te mean={fresh_lift_te.mean():.3f} "
          f"std={fresh_lift_te.std():.3f}")
    print(f"   deploy clip band: lo={deploy_lo:.3f} hi={deploy_hi:.3f}  "
          f"(clipped hi={n_clip_hi} lo={n_clip_lo})")
    print(f"   te(513) final mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"min={te_pred.min():.3f} max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}  "
          f"(expected NEGATIVE gap vs honest mean -- deploy sees 253 labels)")
    print(f"   gap (in_sample - honest) = {te_unb_in_rae - mean_pf:+.4f}")

    # Median-seed OOF for storage
    med_seed_idx = int(np.argsort(pf_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} "
          f"(perfold_mean={pf_arr[med_seed_idx]:.4f})")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: GATE")
    print("-" * 78)
    if mean_pf < GATE_BREAKTHROUGH:
        verdict = "BREAKTHROUGH"
        ladder_action = (
            f"PROMOTE-CANDIDATE (deep-30 re-verify REQUIRED). nb3351 15-seed "
            f"per-fold-mean {mean_pf:.4f} beats the {GATE_BREAKTHROUGH:.4f} "
            f"wall ({mean_pf - GATE_BREAKTHROUGH:+.4f}) and nb3200 deep-30 "
            f"{REF_NB3200:.4f} ({mean_pf - REF_NB3200:+.4f}). The FRESH "
            f"chemprop v3 anchor (nb3350, RAE {fresh_rae:.4f}, Pearson "
            f"{pearson_used:.3f} vs frozen chemprop_aux) carries orthogonal "
            f"error structure the frozen-anchor pyramid could not -- this is "
            f"the substrate change cycle-134/167/169 flagged as the only open "
            f"lever. Mean-of-folds w = nb3200 {w_mean_of_folds[0]:.3f} / "
            f"fresh_K18_v3 {w_mean_of_folds[1]:.3f}. MUST re-verify at deep-30 "
            f"(>=30 kf_seeds) before any PRIMARY-1 swap: 15-seed is "
            f"hypothesis-grade and prior ceiling candidates showed 3-5x "
            f"under-dispersion at n=5/15 (cycle-160)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3351 15-seed per-fold-mean {mean_pf:.4f} fails the "
            f"{GATE_BREAKTHROUGH:.4f} wall ({mean_pf - GATE_BREAKTHROUGH:+.4f}; "
            f"delta vs nb3200 deep-30 {REF_NB3200:.4f} = "
            f"{mean_pf - REF_NB3200:+.4f}). Even though the fresh chemprop v3 "
            f"anchor passed usable+decorrelated (RAE {fresh_rae:.4f}, Pearson "
            f"{pearson_used:.3f}), SLSQP either over-shrinks to nb3200 "
            f"(mean w nb3200={w_mean_of_folds[0]:.2f}) or the fresh anchor's "
            f"orthogonal component is not aligned with the frozen-anchor error "
            f"residual. The fresh-anchor substrate-change route does not break "
            f"the 0.4423 ceiling at this strength; a stronger / more "
            f"decorrelated fresh anchor (deeper chemprop, different seed/"
            f"architecture) is required."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}  (median-seed honest OOF, 253,)")
    print(f"   [save] {te_path}   (deploy te, 513,)")

    sub_csv = SUBMISSIONS / f"{TAG}_blend_fresh_chemprop.csv"
    if verdict == "BREAKTHROUGH":
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
        "method": ("blend_fresh_chemprop_v3_nb3350_with_nb3200_via_perfold_"
                   "slsqp_simplex_plus_learned_quantile_clip_honest_5fold_"
                   "scaffold_cv_15_fresh_seeds"),
        "depends_on": ["nb3350", "nb3200"],
        "anchor_pre_unblind": True,
        # fresh anchor (nb3350)
        "fresh_oof_path": str(FRESH_OOF_PATH),
        "fresh_te_path": str(FRESH_TE_PATH),
        "fresh_anchor_unblind_RAE": round(fresh_rae, 4),
        "fresh_leak_eq_truth_frac": round(fresh_leak, 4),
        "fresh_K18_standalone_oof_rae": round(rae_fresh_lift, 4),
        "pearson_fresh_vs_frozen_aux_253": (
            round(pearson_vs_frozen_aux, 4)
            if pearson_vs_frozen_aux is not None else None
        ),
        "pearson_used_for_decorrel_gate": round(pearson_used, 4),
        "nb3350_recorded_pearson": nb3350_pearson,
        "nb3350_GATE_usable_anchor": nb3350_usable_flag,
        "nb3350_GATE_decorrelated": nb3350_decorrel_flag,
        "corr_fresh_vs_nb3200_253": round(corr_fresh_nb3200, 4),
        # nb3200 anchor
        "nb3200_oof_path": str(NB3200_OOF_PATH),
        "nb3200_te_path": str(NB3200_TE_PATH),
        "nb3200_oof_rae": round(rae_nb3200, 4),
        "nb3200_leak_eq_truth_frac": round(nb3200_leak, 4),
        # feature / lift config
        "feat_dim_full": 117,
        "K_resid_fresh_lift": K_RESID,
        "K18_idx_in_117col": K18_idx.tolist(),
        "chembl_pool_size": int(chembl_pool_size),
        "lgbm_params_seed0": _lgbm_params(0),
        # CV config
        "n_folds": N_FOLDS,
        "n_slsqp_starts": N_SLSQP_STARTS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        # results
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "mean_pooled_rae": round(mean_pooled, 4),
        "std_pooled_rae": round(std_pooled, 4),
        "sem_pooled_rae": round(sem_pooled, 4),
        "mean_per_fold_rae": round(mean_pf, 4),
        "std_per_fold_rae": round(std_pf, 4),
        "sem_per_fold_rae": round(sem_pf, 4),
        "ci95_per_fold_low": round(ci_low_pf, 4),
        "ci95_per_fold_high": round(ci_high_pf, 4),
        "median_per_fold_rae": round(median_pf, 4),
        "min_per_fold_rae": round(float(pf_arr.min()), 4),
        "max_per_fold_rae": round(float(pf_arr.max()), 4),
        "any_fold_degenerate_across_seeds": bool(any_degen),
        "mean_of_fold_weights": {
            "nb3200": round(float(w_mean_of_folds[0]), 4),
            "fresh_K18_v3": round(float(w_mean_of_folds[1]), 4),
        },
        "clip_ql_mode": ql_mode,
        "clip_qh_mode": qh_mode,
        "deploy_clip_lo": (None if deploy_lo == -np.inf else round(deploy_lo, 4)),
        "deploy_clip_hi": (None if deploy_hi == np.inf else round(deploy_hi, 4)),
        "n_te_clipped_hi": n_clip_hi,
        "n_te_clipped_lo": n_clip_lo,
        # refs
        "ref_nb3200_deep30_mean": REF_NB3200,
        "ref_nb3200_deep30_std": REF_NB3200_STD,
        "ref_nb3090": REF_NB3090,
        "ref_nb2171": REF_NB2171,
        "ref_nb1191": REF_NB1191,
        "ref_chemprop_aux": CHEMPROP_AUX_REF,
        "delta_vs_nb3200_perfold_mean": round(mean_pf - REF_NB3200, 4),
        "delta_vs_gate": round(mean_pf - GATE_BREAKTHROUGH, 4),
        # deploy te
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_unb_in_sample_minus_honest_gap": round(te_unb_in_rae - mean_pf, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv) if verdict == "BREAKTHROUGH" else None),
        "gate_breakthrough": GATE_BREAKTHROUGH,
        "skip_rae_threshold": SKIP_RAE_THRESHOLD,
        "decorrel_pearson_max": DECORREL_PEARSON_MAX,
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
    print(f"   per_fold_mean ({n_s} seeds) = {mean_pf:.4f} +/- {std_pf:.4f}")
    print(f"   95% CI                = [{ci_low_pf:.4f}, {ci_high_pf:.4f}]")
    print(f"   delta vs nb3200       = {mean_pf - REF_NB3200:+.4f}")
    print(f"   fresh anchor RAE      = {fresh_rae:.4f}  "
          f"Pearson(vs frozen aux) = {pearson_used:.4f}")
    print(f"   mean-of-fold weights  = nb3200 {w_mean_of_folds[0]:.3f} / "
          f"fresh_K18_v3 {w_mean_of_folds[1]:.3f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    if res.get("verdict") == "SKIP_UNUSABLE":
        for k in ("verdict", "skip_reason"):
            print(f"  {k}: {res.get(k)}")
    else:
        for k in (
            "mean_per_fold_rae", "std_per_fold_rae",
            "ci95_per_fold_low", "ci95_per_fold_high",
            "delta_vs_nb3200_perfold_mean",
            "fresh_anchor_unblind_RAE", "pearson_used_for_decorrel_gate",
            "mean_of_fold_weights",
            "te_unb_in_sample_rae", "te_unb_in_sample_minus_honest_gap",
            "verdict", "ladder_action",
        ):
            print(f"  {k}: {res.get(k)}")
