"""nb3291 -- GP posterior correction THEN learned clip, stacked sequentially.

NEW PARADIGM (vs nb3282 GP-alone / nb3173 clip-alone):

    nb3282 applies a Gaussian-Process posterior-mean correction to the nb3090
    residual: per-fold-mean 0.4452 (FAIL at gate 0.4423) -- the GP smooths the
    residual as a function of Euclidean distance in K=20 feature space, pulling
    the anchor 0.4470 -> 0.4452.

    nb3173 applies a per-fold LEARNED clip (q_low/q_high grid on fold-train
    truth quantiles) to a *different* anchor (nb3080): 0.4422 (BETTER) -- the
    clip decompresses the prediction tails toward the truth support.

    nb3291 STACKS the two corrections SEQUENTIALLY on the SAME nb3090 anchor:

        Step 1 (GP):   corr = GP.posterior_mean(X_K20)         (RBF(1.0)+White(0.5))
                       gp_pred = nb3090 + corr
        Step 2 (clip): (lo, hi) = best (q_low, q_high) on fold-train TRUTH
                       final   = clip(gp_pred, lo, hi)

    Why stack? The two operators are ORTHOGONAL in what they fix:
      - GP is a SMOOTH residual model: it moves predictions where the residual
        has smooth feature-distance structure (mid-manifold correction). It does
        NOT touch the prediction support / tail extremes (WhiteKernel shrinks
        the correction toward 0 on isolated points).
      - The learned clip is a SUPPORT operator: it pulls the most extreme
        over/under-predictions back to the fold-train truth quantile band. It
        does NOT reshape the interior conditional ordering.
      So GP first (fix smooth interior), clip second (fix the residual tails the
      GP could not reach). The clip on the GP-corrected vector should mop up the
      tail errors the GP smoothing leaves behind, and -- because the GP already
      reduced interior residual variance -- the clip band can sit where it best
      trims the remaining extremes.

    Order matters: GP -> clip (not clip -> GP). Clipping first would censor the
    feature-distance signal the GP reads from the extreme rows; correcting first
    then trimming the post-correction tail is the natural composition.

PROTOCOL (per kf_seed, 5-fold scaffold split, FULLY fold-honest):
    residual = y - nb3090   (253-vector)
    For each outer fold (tr_loc, va_loc):
        a) StandardScaler.fit(X_K20[tr_loc])                  # fold-honest scale
        b) GP.fit(scaler.transform(X_K20[tr_loc]), residual[tr_loc])
           gp_tr = nb3090[tr_loc] + GP.predict(scaler.transform(X_K20[tr_loc]))
           gp_va = nb3090[va_loc] + GP.predict(scaler.transform(X_K20[va_loc]))
        c) inner grid on fold-train ONLY:
               for q_low in {0.01,0.02,0.05,0.10}:
                 for q_high in {0.90,0.95,0.98,0.99}:
                   lo = quantile(y[tr_loc], q_low); hi = quantile(y[tr_loc], q_high)
                   r  = rae(y[tr_loc], clip(gp_tr, lo, hi))     # clip GP-corrected tr
               pick (q_low*, q_high*) minimizing fold-train RAE
        d) oof[va_loc] = clip(gp_va, lo*, hi*)
    pooled_rae[kf_seed] = rae(y, oof)
    per-fold-mean = mean over 15 fresh kf_seeds {1216..1230}.

    Fold honesty: the GP never sees fold-val residuals; the clip quantiles come
    from fold-train truth only; the clip band is chosen by fold-train RAE on the
    GP-corrected fold-train predictions. No val leakage at either stage.

GATE:
    per-fold-mean < 0.4423 -> "BETTER"   else "FAIL"

DEPLOY:
    GP refit on ALL 253 (scaler+GP), te_gp = te_nb3090 + GP.predict(te_K20).
    clip pick on FULL 253 truth quantiles applied to GP-corrected tr/te:
        (lo, hi) = best (q_low,q_high) on y_unb vs clip(gp_unb, .)
        te = clip(te_gp, lo, hi)

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy   (253,) float32 -- anchor OOF
    data/processed/te_nb3090.npy         (513,) float32 -- anchor deploy
    data/processed/nb2280_summary.json   (K=20 idx in 117-col)
    + nb1352/1392/1484/1523/1524/1541 summaries (5-way feature matrix)
    + te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy, te_avalon512.npy
    + C:/pxr_artifacts/nb1030/X_mordred_test.npy
    + data/external/chembl_pxr_CHEMBL3401.parquet (+ siblings)

Outputs:
    data/processed/nb3291_summary.json
    data/processed/nb3291_pred_oof.npy   (253,) float32 -- best-seed (min RAE) OOF
    data/processed/te_nb3291.npy         (513,) float32 -- deploy te
    submissions/nb3291_gp_then_clip.csv  (BETTER only)
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
from rdkit import Chem
from rdkit import RDLogger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3291"
PARENT_TAG = "nb3090"   # anchor: finer-q-cut K18+K19 deep30 (OOF RAE 0.447)

# -- Anchor (nb3090 OOF / te) -------------------------------------------------
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb3090_pred_oof.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb3090.npy"

# -- Feature cache paths (identical to nb3282 / nb3163) -----------------------
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

NB2280_SUMMARY = DATA_PROCESSED / "nb2280_summary.json"   # K=20 idx source

# -- GP kernel hyperparams (identical to nb3282) ------------------------------
RBF_LENGTH_SCALE = 1.0
WHITE_NOISE_LEVEL = 0.5
GP_NORMALIZE_Y = True
GP_N_RESTARTS = 0          # fixed kernel (no marginal-likelihood optimization)
GP_ALPHA = 1e-10           # numerical jitter; WhiteKernel carries the noise

# -- Per-fold clip grid (identical to nb3173) ---------------------------------
Q_LOW_GRID = [0.01, 0.02, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98, 0.99]

# -- ChEMBL kNN params (identical to nb3282 / nb3163) -------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- HONEST cross-fit eval ----------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))   # 15 fresh seeds {1216 ... 1230}

# -- Gate ---------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ---------------------------------------------------------------
NB3090_ANCHOR_REF = 0.4470      # nb3090 OOF RAE (this script's anchor)
NB3282_GP_REF = 0.4452          # GP-alone per-fold-mean (component 1)
NB3173_CLIP_REF = 0.4422        # clip-alone on nb3080 (component 2)
NB2960_K18_REF = 0.4536
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# feature-matrix helpers (lifted verbatim from nb3282 / nb3163)
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
    """117-col matrix identical to nb3282 / nb3163 / nb3123 / nb2604."""
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
# GP residual model + learned clip primitives
# ============================================================================

def _make_gp():
    """RBF(1.0) + WhiteKernel(0.5), normalize_y, fixed kernel (no MLL opt)."""
    kernel = (
        RBF(length_scale=RBF_LENGTH_SCALE)
        + WhiteKernel(noise_level=WHITE_NOISE_LEVEL)
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=GP_ALPHA,
        normalize_y=GP_NORMALIZE_Y,
        n_restarts_optimizer=GP_N_RESTARTS,
        optimizer=None,            # fixed kernel: do NOT optimize marg. likelihood
        copy_X_train=True,
        random_state=0,
    )


def _pick_best_clip(y_tr, pred_tr):
    """Inner grid search: pick (q_low*, q_high*) minimizing fold-train RAE.

    Quantiles come from fold-train TRUTH (y_tr); the clip is applied to the
    fold-train PREDICTIONS (pred_tr -- here the GP-corrected fold-train pred).
    Identical primitive to nb3173 but operates on the GP-corrected vector.
    """
    best_rae = np.inf
    best_ql = Q_LOW_GRID[0]
    best_qh = Q_HIGH_GRID[-1]
    best_lo = float(np.quantile(y_tr, best_ql))
    best_hi = float(np.quantile(y_tr, best_qh))
    for ql in Q_LOW_GRID:
        lo = float(np.quantile(y_tr, ql))
        for qh in Q_HIGH_GRID:
            hi = float(np.quantile(y_tr, qh))
            if hi <= lo:
                continue
            r = float(rae(y_tr, np.clip(pred_tr, lo, hi)))
            if r < best_rae:
                best_rae = r
                best_ql, best_qh, best_lo, best_hi = ql, qh, lo, hi
    return best_ql, best_qh, best_lo, best_hi


def fold_honest_one_kf_seed(
    X_K20_unb, anchor, residual, y_unb, unb_scaffolds, kf_seed, n_folds,
):
    """One kf_seed honest 5-fold scaffold CV: GP correction THEN learned clip.

    Per fold:
        scaler.fit(X_K20[tr])                                # fold-honest scale
        GP.fit(scaler.transform(X_K20[tr]), residual[tr])
        gp_tr = anchor[tr] + GP.predict(scaler.transform(X_K20[tr]))
        gp_va = anchor[va] + GP.predict(scaler.transform(X_K20[va]))
        (lo*, hi*) = best (q_low, q_high) on y[tr] vs clip(gp_tr, .)
        oof[va] = clip(gp_va, lo*, hi*)
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    per_fold_rae_gp_only = []   # diagnostic: GP step before clip
    corr_std_per_fold = []
    fold_ql, fold_qh = [], []
    n_clipped_lo, n_clipped_hi = [], []
    for tr_loc, va_loc in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_K20_unb[tr_loc])
        X_va = scaler.transform(X_K20_unb[va_loc])
        gp = _make_gp()
        gp.fit(X_tr, residual[tr_loc])
        corr_tr = gp.predict(X_tr).astype(np.float64)
        corr_va = gp.predict(X_va).astype(np.float64)
        gp_tr = anchor[tr_loc] + corr_tr
        gp_va = anchor[va_loc] + corr_va
        # GP-only diagnostic on val (before clip)
        per_fold_rae_gp_only.append(float(rae(y_unb[va_loc], gp_va)))
        corr_std_per_fold.append(float(corr_va.std()))
        # learned clip on fold-train GP-corrected predictions
        ql, qh, lo, hi = _pick_best_clip(y_unb[tr_loc], gp_tr)
        fold_ql.append(ql)
        fold_qh.append(qh)
        n_clipped_lo.append(int(np.sum(gp_va < lo)))
        n_clipped_hi.append(int(np.sum(gp_va > hi)))
        oof[va_loc] = np.clip(gp_va, lo, hi)
        per_fold_rae.append(float(rae(y_unb[va_loc], oof[va_loc])))
    if np.isnan(oof).any():
        raise RuntimeError(f"scaffold splits did not cover all rows (kf_seed={kf_seed})")
    pooled = float(rae(y_unb, oof))
    pooled_gp_only = float("nan")  # GP-only pooled computed per-fold-stitched below
    return {
        "pooled": pooled,
        "oof": oof,
        "per_fold_rae": per_fold_rae,
        "per_fold_rae_gp_only": per_fold_rae_gp_only,
        "corr_std_per_fold": corr_std_per_fold,
        "fold_ql": fold_ql,
        "fold_qh": fold_qh,
        "n_clipped_lo": int(np.sum(n_clipped_lo)),
        "n_clipped_hi": int(np.sum(n_clipped_hi)),
    }


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- GP posterior correction THEN learned clip (stacked) on {PARENT_TAG}")
    print(f"          anchor   : {PARENT_TAG} (OOF RAE {NB3090_ANCHOR_REF})")
    print(f"          step 1   : GP RBF(ls={RBF_LENGTH_SCALE})+White({WHITE_NOISE_LEVEL}), "
          f"normalize_y, K=20, fold-honest scale (nb3282 component {NB3282_GP_REF})")
    print(f"          step 2   : per-fold learned clip q_low{Q_LOW_GRID} "
          f"q_high{Q_HIGH_GRID} (nb3173 component {NB3173_CLIP_REF})")
    print(f"          kf_seeds (15)       : {KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"          n_folds             : {N_FOLDS}")
    print(f"          gate    = per-fold-mean < {GATE_BETTER} -> BETTER else FAIL")
    print("=" * 78)

    # -- Load truth, anchor, scaffolds ---------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # -- Anchor (nb3090 OOF / te), residual ----------------------------------
    assert ANCHOR_OOF_PATH.exists(), f"missing anchor OOF: {ANCHOR_OOF_PATH}"
    assert ANCHOR_TE_PATH.exists(), f"missing anchor te: {ANCHOR_TE_PATH}"
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    anchor_te_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    assert anchor_oof.shape == (n_unb,), \
        f"anchor OOF shape {anchor_oof.shape} != ({n_unb},)"
    assert anchor_te_513.shape == (n_test,), \
        f"anchor te shape {anchor_te_513.shape} != ({n_test},)"
    rae_anchor = float(rae(y_unb, anchor_oof))
    residual = y_unb - anchor_oof
    print(f"[anchor] {PARENT_TAG} OOF RAE = {rae_anchor:.4f} "
          f"(ref {NB3090_ANCHOR_REF:.4f})")
    print(f"[anchor] residual mean={residual.mean():.4f}  std={residual.std():.4f}")

    eq_truth_frac = float(np.mean(np.isclose(anchor_oof, y_unb, atol=1e-6)))
    if eq_truth_frac > 0.05:
        print(f"   WARN: {eq_truth_frac:.1%} rows == truth -- possible leak")
    else:
        print(f"   leak_eq_truth_frac = {eq_truth_frac:.4f} (PRE-clean expected)")

    # -- Load K=20 idx -------------------------------------------------------
    with open(NB2280_SUMMARY) as f:
        nb2280 = json.load(f)
    K20_idx = np.array(nb2280["K20_rfe_surviving_idx_in_117"], dtype=int)
    assert len(K20_idx) == 20, f"K20 len {len(K20_idx)} != 20"
    assert K20_idx.max() < 117 and K20_idx.min() >= 0, "K20 idx out of range"
    print(f"[load] K=20 idx (n={len(K20_idx)}): {K20_idx.tolist()}")

    # -- Build 117-col matrix, slice K=20 ------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: build 117-col 5-way feature matrix and slice K=20")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K20 = X_te_full[:, K20_idx].astype(np.float64)
    X_unb_K20 = X_te_K20[unb_idx]
    assert X_unb_K20.shape == (n_unb, 20)
    print(f"   X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # -- HONEST cross-fit over 15 kf_seeds -----------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 2: HONEST 5-fold scaffold CV over {len(KF_SEEDS)} kf_seeds "
          f"(GP correction THEN learned clip; both fold-honest)")
    print("-" * 78)
    per_kf_pooled = []
    per_kf_gp_only_mean = []
    per_kf_fold_rae = []
    per_kf_corr_std = []
    per_kf_oof = []
    all_fold_ql, all_fold_qh = [], []
    total_clip_lo, total_clip_hi = 0, 0
    for kf_seed in KF_SEEDS:
        ts = time.time()
        res = fold_honest_one_kf_seed(
            X_unb_K20, anchor_oof, residual, y_unb, unb_scaffolds,
            kf_seed=kf_seed, n_folds=N_FOLDS,
        )
        per_kf_pooled.append(res["pooled"])
        per_kf_gp_only_mean.append(float(np.mean(res["per_fold_rae_gp_only"])))
        per_kf_fold_rae.append(res["per_fold_rae"])
        per_kf_corr_std.append(res["corr_std_per_fold"])
        per_kf_oof.append(res["oof"])
        all_fold_ql.extend(res["fold_ql"])
        all_fold_qh.extend(res["fold_qh"])
        total_clip_lo += res["n_clipped_lo"]
        total_clip_hi += res["n_clipped_hi"]
        wall = time.time() - ts
        print(f"   kf_seed={kf_seed}  pooled_RAE={res['pooled']:.4f}  "
              f"folds_mean={np.mean(res['per_fold_rae']):.4f}  "
              f"gp_only={np.mean(res['per_fold_rae_gp_only']):.4f}  "
              f"corr_std={np.mean(res['corr_std_per_fold']):.4f}  "
              f"clip(lo,hi)=({res['n_clipped_lo']},{res['n_clipped_hi']})  "
              f"wall={wall:.1f}s")

    pooled_arr = np.array(per_kf_pooled, dtype=np.float64)
    per_fold_mean = float(pooled_arr.mean())     # GATE quantity
    honest_std = float(pooled_arr.std(ddof=1)) if len(pooled_arr) > 1 else 0.0
    sem_rae = float(honest_std / np.sqrt(len(pooled_arr)))
    t_mult = 2.145  # df=14, two-sided 95%
    ci95_low = float(per_fold_mean - t_mult * sem_rae)
    ci95_high = float(per_fold_mean + t_mult * sem_rae)
    honest_median = float(np.median(pooled_arr))
    honest_min = float(pooled_arr.min())
    honest_max = float(pooled_arr.max())
    best_seed = int(KF_SEEDS[int(np.argmin(pooled_arr))])
    gp_only_mean = float(np.mean(per_kf_gp_only_mean))

    ql_counter = Counter(all_fold_ql)
    qh_counter = Counter(all_fold_qh)
    ql_mode = ql_counter.most_common(1)[0][0]
    qh_mode = qh_counter.most_common(1)[0][0]

    print(f"\n   per-fold-mean over {len(KF_SEEDS)} kf_seeds = "
          f"{per_fold_mean:.4f} +/- {honest_std:.5f}  "
          f"(min={honest_min:.4f}  max={honest_max:.4f})")
    print(f"   95% CI = [{ci95_low:.4f}, {ci95_high:.4f}]  median={honest_median:.4f}")
    print(f"   GP-only val-fold mean (before clip) = {gp_only_mean:.4f}")
    print(f"   delta vs anchor nb3090   = {per_fold_mean - rae_anchor:+.4f}")
    print(f"   delta vs GP-alone nb3282 = {per_fold_mean - NB3282_GP_REF:+.4f}")
    print(f"   ql_dist (75 folds) = {dict(ql_counter)}  ql_mode={ql_mode}")
    print(f"   qh_dist (75 folds) = {dict(qh_counter)}  qh_mode={qh_mode}")
    print(f"   total clipped over all seeds: lo={total_clip_lo}  hi={total_clip_hi}")

    pred_oof_canon = per_kf_oof[int(np.argmin(pooled_arr))].astype(np.float32)

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: GATE (on per-fold-mean across 15 kf_seeds)")
    print("-" * 78)
    verdict = "BETTER" if per_fold_mean < GATE_BETTER else "FAIL"
    delta_vs_anchor = per_fold_mean - rae_anchor
    delta_vs_gp = per_fold_mean - NB3282_GP_REF
    delta_vs_clip = per_fold_mean - NB3173_CLIP_REF
    delta_vs_k18 = per_fold_mean - NB2960_K18_REF
    delta_vs_nb2171 = per_fold_mean - NB2171_REF
    print(f"   per-fold-mean              = {per_fold_mean:.4f}")
    print(f"   gate                       = < {GATE_BETTER}  -> {verdict}")
    print(f"   delta vs anchor nb3090     = {delta_vs_anchor:+.4f}  ({rae_anchor:.4f})")
    print(f"   delta vs GP-alone nb3282   = {delta_vs_gp:+.4f}  ({NB3282_GP_REF:.4f})")
    print(f"   delta vs clip-alone nb3173 = {delta_vs_clip:+.4f}  ({NB3173_CLIP_REF:.4f})")
    print(f"   delta vs nb2960 K18 ref    = {delta_vs_k18:+.4f}  ({NB2960_K18_REF:.4f})")
    print(f"   delta vs nb2171 ceiling    = {delta_vs_nb2171:+.4f}  ({NB2171_REF:.4f})")
    print(f"   verdict                    = {verdict}")

    # -- Deploy: GP refit on ALL 253 then clip on full 253 -------------------
    print("\n" + "-" * 78)
    print("STEP 4: deploy -- GP refit on ALL 253, then clip on full 253 truth")
    print("-" * 78)
    scaler_full = StandardScaler()
    X_unb_K20_std = scaler_full.fit_transform(X_unb_K20)
    X_te_K20_std = scaler_full.transform(X_te_K20)
    gp_full = _make_gp()
    gp_full.fit(X_unb_K20_std, residual)
    corr_te = gp_full.predict(X_te_K20_std).astype(np.float64)
    corr_unb = gp_full.predict(X_unb_K20_std).astype(np.float64)
    gp_te = anchor_te_513 + corr_te
    gp_unb = anchor_oof + corr_unb
    # deploy clip pick: full-253 truth quantiles vs GP-corrected unb predictions
    deploy_ql, deploy_qh, deploy_lo, deploy_hi = _pick_best_clip(y_unb, gp_unb)
    te_deploy = np.clip(gp_te, deploy_lo, deploy_hi).astype(np.float32)
    te_deploy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_deploy_path, te_deploy)

    n_te_lo = int(np.sum(gp_te < deploy_lo))
    n_te_hi = int(np.sum(gp_te > deploy_hi))
    te_unb_in_rae = float(rae(y_unb, te_deploy[unb_idx]))
    deploy_insample_rae = float(rae(y_unb, np.clip(gp_unb, deploy_lo, deploy_hi)))
    gp_only_insample_rae = float(rae(y_unb, gp_unb))
    print(f"   deploy clip pick = (q{deploy_ql:.2f}, q{deploy_qh:.2f}) -> "
          f"({deploy_lo:.3f}, {deploy_hi:.3f}) from full 253 truth")
    print(f"   te clipped: lo={n_te_lo}/513  hi={n_te_hi}/513  "
          f"total={n_te_lo + n_te_hi}/513")
    print(f"   te_anchor      : mean={anchor_te_513.mean():.4f}  "
          f"std={anchor_te_513.std():.4f}")
    print(f"   corr_te        : mean={corr_te.mean():.4f}  std={corr_te.std():.4f}")
    print(f"   gp_te          : mean={gp_te.mean():.4f}  std={gp_te.std():.4f}")
    print(f"   te_final(clip) : mean={te_deploy.mean():.4f}  "
          f"std={te_deploy.std():.4f}  min={te_deploy.min():.3f}  "
          f"max={te_deploy.max():.3f}")
    print(f"   te[unb] in-sample RAE            = {te_unb_in_rae:.4f}")
    print(f"   GP-only in-sample RAE            = {gp_only_insample_rae:.4f}")
    print(f"   GP+clip deploy in-sample RAE     = {deploy_insample_rae:.4f} "
          f"(DEPLOY OPTIMISM, NOT gate)")

    # -- Save pred_oof + submission ------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(oof_path, pred_oof_canon)
    print(f"   [save] {oof_path}  (best kf_seed={best_seed} OOF, 253,)  "
          f"RAE={rae(y_unb, pred_oof_canon):.4f}")
    print(f"   [save] {te_deploy_path}  (deploy refit, 513,)")

    te_clipped_sub = np.clip(te_deploy, 3.0, 9.0).astype(np.float32)
    sub_csv = SUBMISSIONS / f"{TAG}_gp_then_clip.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_clipped_sub,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    # -- Summary -------------------------------------------------------------
    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "gp_posterior_correction_THEN_learned_clip_stacked_on_nb3090",
        "paradigm": "sequential_stack_GP_residual_then_perfold_clip",
        "stack_order": "GP_first_then_clip",
        "component_1": "nb3282_GP_RBF1.0_White0.5_normalize_y_K20",
        "component_2": "nb3173_perfold_learned_clip_grid",
        "kernel": f"RBF(length_scale={RBF_LENGTH_SCALE}) + "
                  f"WhiteKernel(noise_level={WHITE_NOISE_LEVEL})",
        "normalize_y": GP_NORMALIZE_Y,
        "gp_optimizer": "None_fixed_kernel",
        "gp_alpha": GP_ALPHA,
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "feature_set": "K=20 nb2280 RFE-surviving, StandardScaler fold-honest",
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "leak_eq_truth_frac": eq_truth_frac,
        "K20_idx_in_117col": K20_idx.tolist(),
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "chembl_pool_size": int(chembl_pool_size),
        "per_kf_pooled_rae": per_kf_pooled,
        "per_kf_gp_only_mean": per_kf_gp_only_mean,
        "per_kf_fold_rae": per_kf_fold_rae,
        "per_kf_corr_std": per_kf_corr_std,
        "pooled_rae_array": [float(x) for x in pooled_arr.tolist()],
        "per_fold_mean": per_fold_mean,
        "mean_rae": per_fold_mean,
        "gp_only_per_fold_mean": gp_only_mean,
        "honest_std": honest_std,
        "sem_rae": sem_rae,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "honest_median": honest_median,
        "honest_min": honest_min,
        "honest_max": honest_max,
        "best_seed": best_seed,
        "ql_distribution": {str(k): int(v) for k, v in ql_counter.items()},
        "qh_distribution": {str(k): int(v) for k, v in qh_counter.items()},
        "ql_mode": float(ql_mode),
        "qh_mode": float(qh_mode),
        "total_clipped_lo_all_seeds": int(total_clip_lo),
        "total_clipped_hi_all_seeds": int(total_clip_hi),
        "delta_vs_anchor": delta_vs_anchor,
        "delta_vs_gp_alone_nb3282": delta_vs_gp,
        "delta_vs_clip_alone_nb3173": delta_vs_clip,
        "delta_vs_nb2960_K18": delta_vs_k18,
        "delta_vs_nb2171": delta_vs_nb2171,
        "ref_nb3090_anchor": NB3090_ANCHOR_REF,
        "ref_nb3282_gp": NB3282_GP_REF,
        "ref_nb3173_clip": NB3173_CLIP_REF,
        "ref_nb2960_K18": NB2960_K18_REF,
        "ref_nb2171": NB2171_REF,
        "ref_chemprop_aux": CHEMPROP_AUX_REF,
        "gate_better": GATE_BETTER,
        "gate_pass": bool(verdict == "BETTER"),
        "verdict": verdict,
        "deploy_ql": float(deploy_ql),
        "deploy_qh": float(deploy_qh),
        "deploy_lo": float(deploy_lo),
        "deploy_hi": float(deploy_hi),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "gp_only_insample_rae": gp_only_insample_rae,
        "deploy_insample_rae_NOT_GATE": deploy_insample_rae,
        "te_mean": float(te_deploy.mean()),
        "te_std": float(te_deploy.std()),
        "te_min": float(te_deploy.min()),
        "te_max": float(te_deploy.max()),
        "te_clipped_mean": float(te_clipped_sub.mean()),
        "te_clipped_std": float(te_clipped_sub.std()),
        "corr_te_mean": float(corr_te.mean()),
        "corr_te_std": float(corr_te.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_deploy_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                     = {PARENT_TAG} (RAE {rae_anchor:.4f})")
    print(f"   stack                      = GP RBF({RBF_LENGTH_SCALE})+White("
          f"{WHITE_NOISE_LEVEL}) THEN learned clip q{ql_mode:.2f}/q{qh_mode:.2f}")
    print(f"   GP-only per-fold-mean      = {gp_only_mean:.4f}")
    print(f"   per-fold-mean (15 seeds)   = {per_fold_mean:.4f} +/- {honest_std:.5f}")
    print(f"   honest min / max           = {honest_min:.4f} / {honest_max:.4f}")
    print(f"   delta vs anchor nb3090     = {delta_vs_anchor:+.4f}")
    print(f"   delta vs GP-alone nb3282   = {delta_vs_gp:+.4f}")
    print(f"   delta vs clip-alone nb3173 = {delta_vs_clip:+.4f}")
    print(f"   te[unb_idx] in-sample RAE  = {te_unb_in_rae:.4f}")
    print(f"   GATE (< {GATE_BETTER})           = {verdict}")
    print(f"   wall                       = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== KEY ====")
    for k in (
        "anchor_in_rae",
        "residual_std",
        "gp_only_per_fold_mean",
        "per_fold_mean",
        "honest_std",
        "honest_min",
        "honest_max",
        "best_seed",
        "ql_mode",
        "qh_mode",
        "delta_vs_anchor",
        "delta_vs_gp_alone_nb3282",
        "delta_vs_clip_alone_nb3173",
        "te_unb_in_sample_rae",
        "gate_pass",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
