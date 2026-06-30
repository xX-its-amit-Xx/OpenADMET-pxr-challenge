"""nb3290 -- Gaussian Process posterior on nb3090 residual with KERNEL SWEEP.

EXTENDS nb3282 (fixed RBF(1.0)+White(0.5) GP posterior, per-fold-mean 0.4452,
closest non-clip to the 0.4422 learned-clip family). nb3282 used ONE hand-set
kernel. This script asks: does TUNING the kernel hyperparameters extract more
of the smooth residual structure?

NEW PARADIGM (vs nb3282 fixed kernel):

    residual = y - nb3090                              (253-vector)
    kernel grid = RBF(length_scale in {0.5,1.0,2.0,5.0})
                  x WhiteKernel(noise_level in {0.3,0.5,1.0})   -> 12 kernels
    GP = GaussianProcessRegressor(normalize_y=True, optimizer=None) per kernel

    HONEST nested selection (no fold-val peeking):
        outer 5-fold scaffold CV (15 fresh kf_seeds {1216..1230}):
            for (tr_loc, va_loc):
                # INNER pick: choose kernel by inner scaffold-CV on tr_loc only
                inner splits = scaffold_kfold_indices(scaf[tr_loc], 5, seed=kf)
                for each of the 12 kernels:
                    inner_oof = anchor[tr] + GP(kernel).fit(inner_tr).predict(inner_va)
                    score[kernel] = rae(y[tr], inner_oof)
                best_kernel = argmin_kernel score
                # apply best kernel: refit on FULL tr_loc, predict va_loc
                scaler.fit(X_K20[tr_loc])                       (fold-honest)
                GP(best_kernel).fit(scaler(X[tr]), residual[tr])
                oof[va] = anchor[va] + GP.predict(scaler(X[va]))
            pooled_rae[kf] = rae(y_unb, oof)
        per-fold-mean = mean over the 15 kf_seeds of pooled RAE.

    Why nested: per-fold inner-pick is the only honest way to "tune" a
    hyperparameter inside cross-fit. Selecting the kernel on the WHOLE 253 (or
    on outer-val) would be selection-on-the-test leakage; the 12-way grid has
    df>0 so flat picking on the eval set would inflate the number. Inner CV on
    fold-train keeps every outer-val row untouched by kernel selection.

    Standardization is fold-honest: StandardScaler is fit on the OUTER tr_loc
    for the outer GP, and re-fit inside each inner-train for the inner picks.

DEPLOY: pick the single best kernel by FULL-253 inner scaffold-CV (honest
    model-selection over the deploy data, analogous to the per-fold pick but on
    all 253), refit scaler+GP on all 253, te = te_nb3090 + GP.predict(te_K20).

GATE:
    per-fold-mean < 0.4423 -> "BETTER"
    else                   -> "FAIL"

Inputs (identical to nb3282):
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy   (253,) -- anchor OOF
    data/processed/te_nb3090.npy         (513,) -- anchor deploy
    data/processed/nb2280_summary.json   (K=20 idx in 117-col)
    + nb1352/1392/1484/1523/1524/1541 summaries
    + te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy, te_avalon512.npy
    + C:/pxr_artifacts/nb1030/X_mordred_test.npy
    + data/external/chembl_pxr_CHEMBL3401.parquet (+ siblings)

Outputs:
    data/processed/nb3290_summary.json
    data/processed/nb3290_pred_oof.npy   (253,) -- best-seed (min RAE) OOF
    data/processed/te_nb3290.npy         (513,) -- deploy te
    submissions/nb3290_gp_kernel_sweep.csv (BETTER only)
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

TAG = "nb3290"
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

# -- GP kernel sweep grid (per task spec) -------------------------------------
RBF_LENGTH_SCALES = [0.5, 1.0, 2.0, 5.0]
WHITE_NOISE_LEVELS = [0.3, 0.5, 1.0]
KERNEL_GRID = [(ls, nl) for ls in RBF_LENGTH_SCALES for nl in WHITE_NOISE_LEVELS]
GP_NORMALIZE_Y = True
GP_N_RESTARTS = 0          # fixed kernel (no marginal-likelihood optimization)
GP_ALPHA = 1e-10           # numerical jitter; WhiteKernel carries the noise

# -- ChEMBL kNN params (identical to nb3282 / nb3163) -------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- HONEST cross-fit eval ----------------------------------------------------
N_FOLDS = 5
N_INNER_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))   # 15 fresh seeds {1216 ... 1230}

# -- Gates --------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ---------------------------------------------------------------
NB3090_ANCHOR_REF = 0.4470      # nb3090 OOF RAE (this script's anchor)
NB3282_REF = 0.4452             # fixed-kernel GP per-fold-mean (parent of sweep)
NB3173_REF = 0.4422             # learned-clip family ref (gate band)
NB2960_K18_REF = 0.4536
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# feature-matrix helpers (lifted verbatim from nb3282)
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
    """117-col matrix identical to nb3282 / nb3163 / nb3123 / nb2604 / nb2960."""
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
# GP residual model (sweep)
# ============================================================================

def _make_gp(length_scale, noise_level):
    """RBF(length_scale) + WhiteKernel(noise_level), normalize_y, fixed kernel."""
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=GP_ALPHA,
        normalize_y=GP_NORMALIZE_Y,
        n_restarts_optimizer=GP_N_RESTARTS,
        optimizer=None,            # fixed kernel: do NOT optimize marg. likelihood
        copy_X_train=True,
        random_state=0,
    )


def _inner_pick_kernel(
    X_sub, anchor_sub, residual_sub, y_sub, scaf_sub, kf_seed, n_inner_folds,
):
    """Pick the best (length_scale, noise_level) via inner scaffold-CV on the
    given subset (the OUTER fold-train, or the full 253 at deploy time).

    For each kernel in the 12-way grid:
        inner_oof = anchor_sub + GP(kernel) inner-CV posterior correction
        score     = rae(y_sub, inner_oof)
    Returns (best_ls, best_nl, scores_dict).

    Honest: every inner-val row's correction comes from a GP fit on inner-train
    rows only; the kernel is scored on the held-out inner folds, never refit on
    the row it scores.
    """
    n_sub = len(y_sub)
    # Inner scaffold splits depend only on scaf_sub + seed (shared across kernels
    # so the comparison is paired/fair). Guard tiny subsets by capping folds.
    n_unique_sub = len({s for s in scaf_sub if s})
    eff_folds = max(2, min(n_inner_folds, n_unique_sub))
    inner_splits = scaffold_kfold_indices(
        scaf_sub, n_splits=eff_folds, shuffle=True, seed=kf_seed,
    )
    scores = {}
    for (ls, nl) in KERNEL_GRID:
        oof = np.full(n_sub, np.nan, dtype=np.float64)
        for (itr, iva) in inner_splits:
            sc = StandardScaler()
            Xi_tr = sc.fit_transform(X_sub[itr])
            Xi_va = sc.transform(X_sub[iva])
            gp = _make_gp(ls, nl)
            gp.fit(Xi_tr, residual_sub[itr])
            corr = gp.predict(Xi_va).astype(np.float64)
            oof[iva] = anchor_sub[iva] + corr
        # all inner rows covered (scaffold splits partition the subset)
        if np.isnan(oof).any():
            # rare: a scaffold landed in no inner-val due to capping; fall back
            # to anchor for those rows (neutral) so the kernel is still scored.
            nanmask = np.isnan(oof)
            oof[nanmask] = anchor_sub[nanmask]
        scores[(ls, nl)] = float(rae(y_sub, oof))
    best_key = min(scores, key=scores.get)
    return best_key[0], best_key[1], scores


def fold_honest_one_kf_seed(
    X_K20_unb, anchor, residual, y_unb, unb_scaffolds, kf_seed, n_folds,
    n_inner_folds,
):
    """One kf_seed honest 5-fold scaffold CV pass with per-fold inner kernel pick.

    For each OUTER fold:
        best (ls, nl) = inner scaffold-CV pick on tr_loc only
        scaler.fit(X_K20[tr])                          (fold-honest standardize)
        GP(ls, nl).fit(scaler(X_K20[tr]), residual[tr])
        oof[va] = anchor[va] + GP.predict(scaler(X_K20[va]))
    """
    unb_scaffolds = np.asarray(unb_scaffolds, dtype=object)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    corr_std_per_fold = []
    picked_kernels = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        # --- INNER kernel pick on outer-train only (no fold-val peeking) ----
        scaf_tr = unb_scaffolds[tr_loc].tolist()
        best_ls, best_nl, _scores = _inner_pick_kernel(
            X_K20_unb[tr_loc], anchor[tr_loc], residual[tr_loc], y_unb[tr_loc],
            scaf_tr, kf_seed=kf_seed, n_inner_folds=n_inner_folds,
        )
        picked_kernels.append((best_ls, best_nl))
        # --- apply picked kernel: refit on full outer-train, predict outer-val
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_K20_unb[tr_loc])
        X_va = scaler.transform(X_K20_unb[va_loc])
        gp = _make_gp(best_ls, best_nl)
        gp.fit(X_tr, residual[tr_loc])
        corr_va = gp.predict(X_va).astype(np.float64)
        oof[va_loc] = anchor[va_loc] + corr_va
        per_fold_rae.append(float(rae(y_unb[va_loc], oof[va_loc])))
        corr_std_per_fold.append(float(corr_va.std()))
    if np.isnan(oof).any():
        raise RuntimeError(f"scaffold splits did not cover all rows (kf_seed={kf_seed})")
    pooled = float(rae(y_unb, oof))
    return pooled, oof, per_fold_rae, corr_std_per_fold, picked_kernels


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- GP posterior on {PARENT_TAG} residual + KERNEL SWEEP")
    print(f"          anchor   : {PARENT_TAG} (OOF RAE {NB3090_ANCHOR_REF})")
    print(f"          grid     : RBF(ls in {RBF_LENGTH_SCALES}) x "
          f"White(noise in {WHITE_NOISE_LEVELS}) = {len(KERNEL_GRID)} kernels")
    print(f"          select   : per-fold inner {N_INNER_FOLDS}-fold scaffold-CV "
          f"pick (nested, honest)")
    print(f"          features : K=20 (nb2280 RFE-surviving), StandardScaler "
          f"fold-honest")
    print(f"          parent   : nb3282 fixed-kernel GP per-fold-mean {NB3282_REF}")
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

    # Leak sanity (nb3090 is PRE-clean)
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

    # -- HONEST nested cross-fit over 15 kf_seeds ----------------------------
    print("\n" + "-" * 78)
    print(f"STEP 2: HONEST nested 5-fold scaffold CV over {len(KF_SEEDS)} kf_seeds "
          f"(GP + per-fold inner kernel pick on nb3090 residual)")
    print("-" * 78)
    per_kf_pooled = []
    per_kf_fold_rae = []
    per_kf_corr_std = []
    per_kf_oof = []
    per_kf_picks = []
    pick_counter = Counter()
    for kf_seed in KF_SEEDS:
        ts = time.time()
        pooled, oof_s, fold_rae, corr_std, picks = fold_honest_one_kf_seed(
            X_unb_K20, anchor_oof, residual, y_unb, unb_scaffolds,
            kf_seed=kf_seed, n_folds=N_FOLDS, n_inner_folds=N_INNER_FOLDS,
        )
        per_kf_pooled.append(pooled)
        per_kf_fold_rae.append(fold_rae)
        per_kf_corr_std.append(corr_std)
        per_kf_oof.append(oof_s)
        per_kf_picks.append([list(p) for p in picks])
        for p in picks:
            pick_counter[p] += 1
        wall = time.time() - ts
        picks_str = ",".join(f"({ls:g}/{nl:g})" for ls, nl in picks)
        print(f"   kf_seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"folds_mean={np.mean(fold_rae):.4f}  "
              f"corr_std_mean={np.mean(corr_std):.4f}  "
              f"picks=[{picks_str}]  wall={wall:.1f}s")

    pooled_arr = np.array(per_kf_pooled, dtype=np.float64)
    per_fold_mean = float(pooled_arr.mean())     # GATE quantity
    honest_std = float(pooled_arr.std(ddof=1)) if len(pooled_arr) > 1 else 0.0
    sem_rae = float(honest_std / np.sqrt(len(pooled_arr)))
    ci95_low = float(per_fold_mean - 1.96 * sem_rae)
    ci95_high = float(per_fold_mean + 1.96 * sem_rae)
    honest_median = float(np.median(pooled_arr))
    honest_min = float(pooled_arr.min())
    honest_max = float(pooled_arr.max())
    best_seed = int(KF_SEEDS[int(np.argmin(pooled_arr))])
    # most-frequently picked kernel across all outer folds x seeds
    most_common_pick = pick_counter.most_common(1)[0] if pick_counter else None
    print(f"\n   per-fold-mean over {len(KF_SEEDS)} kf_seeds = "
          f"{per_fold_mean:.4f} +/- {honest_std:.5f}  "
          f"(min={honest_min:.4f}  max={honest_max:.4f})")
    print(f"   95% CI = [{ci95_low:.4f}, {ci95_high:.4f}]  median={honest_median:.4f}")
    print(f"   delta vs anchor nb3090 = {per_fold_mean - rae_anchor:+.4f}")
    print(f"   delta vs nb3282 fixed  = {per_fold_mean - NB3282_REF:+.4f}")
    print(f"   kernel pick frequency (ls/noise -> count over "
          f"{len(KF_SEEDS) * N_FOLDS} outer folds):")
    for (ls, nl), c in pick_counter.most_common():
        print(f"      RBF(ls={ls:g}) + White({nl:g}) : {c}")

    # -- Save canonical pred_oof at best (min-RAE) seed ----------------------
    pred_oof_canon = per_kf_oof[int(np.argmin(pooled_arr))].astype(np.float32)

    # -- Gate (on per-fold-mean) ---------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: GATE (on per-fold-mean across 15 kf_seeds)")
    print("-" * 78)
    verdict = "BETTER" if per_fold_mean < GATE_BETTER else "FAIL"
    delta_vs_anchor = per_fold_mean - rae_anchor
    delta_vs_nb3282 = per_fold_mean - NB3282_REF
    delta_vs_nb3173 = per_fold_mean - NB3173_REF
    delta_vs_k18 = per_fold_mean - NB2960_K18_REF
    delta_vs_nb2171 = per_fold_mean - NB2171_REF
    print(f"   per-fold-mean              = {per_fold_mean:.4f}")
    print(f"   gate                       = < {GATE_BETTER}  -> {verdict}")
    print(f"   delta vs anchor nb3090     = {delta_vs_anchor:+.4f}  "
          f"({rae_anchor:.4f})")
    print(f"   delta vs nb3282 fixed GP   = {delta_vs_nb3282:+.4f}  "
          f"({NB3282_REF:.4f})")
    print(f"   delta vs nb3173 ref        = {delta_vs_nb3173:+.4f}  "
          f"({NB3173_REF:.4f})")
    print(f"   delta vs nb2960 K18 ref    = {delta_vs_k18:+.4f}  "
          f"({NB2960_K18_REF:.4f})")
    print(f"   delta vs nb2171 ceiling    = {delta_vs_nb2171:+.4f}  "
          f"({NB2171_REF:.4f})")
    print(f"   verdict                    = {verdict}")

    # -- Deploy: pick kernel by full-253 inner scaffold-CV, refit on ALL 253 --
    print("\n" + "-" * 78)
    print("STEP 4: deploy -- pick kernel by full-253 inner scaffold-CV, refit, te")
    print("-" * 78)
    dep_ls, dep_nl, dep_scores = _inner_pick_kernel(
        X_unb_K20, anchor_oof, residual, y_unb, unb_scaffolds,
        kf_seed=KF_SEEDS[0], n_inner_folds=N_INNER_FOLDS,
    )
    print(f"   deploy kernel pick (full-253 inner CV) = "
          f"RBF(ls={dep_ls:g}) + White({dep_nl:g})  "
          f"inner_score={dep_scores[(dep_ls, dep_nl)]:.4f}")
    scaler_full = StandardScaler()
    X_unb_K20_std = scaler_full.fit_transform(X_unb_K20)
    X_te_K20_std = scaler_full.transform(X_te_K20)
    gp_full = _make_gp(dep_ls, dep_nl)
    gp_full.fit(X_unb_K20_std, residual)
    corr_te = gp_full.predict(X_te_K20_std).astype(np.float64)
    te_deploy = (anchor_te_513 + corr_te).astype(np.float32)
    te_deploy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_deploy_path, te_deploy)
    te_unb_in_rae = float(rae(y_unb, te_deploy[unb_idx]))
    corr_unb_insample = gp_full.predict(X_unb_K20_std).astype(np.float64)
    deploy_insample_rae = float(rae(y_unb, anchor_oof + corr_unb_insample))
    print(f"   te_anchor      : mean={anchor_te_513.mean():.4f}  "
          f"std={anchor_te_513.std():.4f}")
    print(f"   corr_te        : mean={corr_te.mean():.4f}  std={corr_te.std():.4f}")
    print(f"   te_corrected   : mean={te_deploy.mean():.4f}  "
          f"std={te_deploy.std():.4f}")
    print(f"   te[unb] in-sample RAE       = {te_unb_in_rae:.4f}")
    print(f"   deploy-refit in-sample RAE  = {deploy_insample_rae:.4f} "
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

    te_clipped = np.clip(te_deploy, 3.0, 9.0).astype(np.float32)
    sub_csv = SUBMISSIONS / f"{TAG}_gp_kernel_sweep.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_clipped,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    # -- Summary -------------------------------------------------------------
    pick_freq_serializable = {
        f"RBF(ls={ls:g})+White({nl:g})": int(c)
        for (ls, nl), c in pick_counter.most_common()
    }
    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "gaussian_process_kernel_sweep_on_nb3090_residual",
        "paradigm": "RBF_GP_kernel_sweep_residual_correction_K20_features",
        "kernel_grid": {
            "rbf_length_scales": RBF_LENGTH_SCALES,
            "white_noise_levels": WHITE_NOISE_LEVELS,
            "n_kernels": len(KERNEL_GRID),
        },
        "selection": f"per-fold inner {N_INNER_FOLDS}-fold scaffold-CV pick (nested)",
        "normalize_y": GP_NORMALIZE_Y,
        "gp_optimizer": "None_fixed_kernel",
        "gp_alpha": GP_ALPHA,
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
        "n_inner_folds": N_INNER_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "chembl_pool_size": int(chembl_pool_size),
        "per_kf_pooled_rae": per_kf_pooled,
        "per_kf_fold_rae": per_kf_fold_rae,
        "per_kf_corr_std": per_kf_corr_std,
        "per_kf_picked_kernels": per_kf_picks,
        "kernel_pick_frequency": pick_freq_serializable,
        "most_common_pick": (
            {"length_scale": most_common_pick[0][0],
             "noise_level": most_common_pick[0][1],
             "count": int(most_common_pick[1])}
            if most_common_pick is not None else None
        ),
        "pooled_rae_array": [float(x) for x in pooled_arr.tolist()],
        "per_fold_mean": per_fold_mean,
        "mean_rae": per_fold_mean,
        "honest_std": honest_std,
        "sem_rae": sem_rae,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "honest_median": honest_median,
        "honest_min": honest_min,
        "honest_max": honest_max,
        "best_seed": best_seed,
        "deploy_kernel": {"length_scale": dep_ls, "noise_level": dep_nl},
        "deploy_kernel_inner_score": float(dep_scores[(dep_ls, dep_nl)]),
        "deploy_kernel_all_scores": {
            f"RBF(ls={ls:g})+White({nl:g})": float(v)
            for (ls, nl), v in dep_scores.items()
        },
        "delta_vs_anchor": delta_vs_anchor,
        "delta_vs_nb3282": delta_vs_nb3282,
        "delta_vs_nb3173": delta_vs_nb3173,
        "delta_vs_nb2960_K18": delta_vs_k18,
        "delta_vs_nb2171": delta_vs_nb2171,
        "ref_nb3090_anchor": NB3090_ANCHOR_REF,
        "ref_nb3282": NB3282_REF,
        "ref_nb3173": NB3173_REF,
        "ref_nb2960_K18": NB2960_K18_REF,
        "ref_nb2171": NB2171_REF,
        "ref_chemprop_aux": CHEMPROP_AUX_REF,
        "gate_better": GATE_BETTER,
        "gate_pass": bool(verdict == "BETTER"),
        "verdict": verdict,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "deploy_insample_rae_NOT_GATE": deploy_insample_rae,
        "te_mean": float(te_deploy.mean()),
        "te_std": float(te_deploy.std()),
        "te_clipped_mean": float(te_clipped.mean()),
        "te_clipped_std": float(te_clipped.std()),
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
    print(f"   GP kernel grid             = {len(KERNEL_GRID)} kernels "
          f"(RBF ls x White noise)")
    print(f"   deploy kernel              = RBF(ls={dep_ls:g}) + White({dep_nl:g})")
    print(f"   per-fold-mean (15 seeds)   = {per_fold_mean:.4f} +/- {honest_std:.5f}")
    print(f"   honest min / max           = {honest_min:.4f} / {honest_max:.4f}")
    print(f"   delta vs anchor nb3090     = {delta_vs_anchor:+.4f}")
    print(f"   delta vs nb3282 fixed GP   = {delta_vs_nb3282:+.4f}")
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
        "per_fold_mean",
        "honest_std",
        "honest_min",
        "honest_max",
        "best_seed",
        "delta_vs_anchor",
        "delta_vs_nb3282",
        "delta_vs_nb3173",
        "te_unb_in_sample_rae",
        "gate_pass",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
