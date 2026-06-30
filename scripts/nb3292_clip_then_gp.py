"""nb3292 -- Learned clip THEN GP posterior correction (reverse order of nb3291).

NEW PARADIGM: clip first, then GP residual correction.

    nb3291 (the script this reverses) applies the GP posterior correction to
    nb3090 FIRST and clips the GP-corrected output SECOND. nb3292 reverses the
    pipeline:

        Step 1: nb3200 = learned per-fold clip on nb3090   (already cached;
                deep-30 OOF RAE 0.4424). The clip operator re-scales the
                predictor's distribution tails toward the train quantile band.
        Step 2: GP posterior mean on the (y - nb3200) residual, add back:
                residual   = y - nb3200
                GP         = GaussianProcessRegressor(
                                 kernel = RBF(length_scale=1.0)
                                        + WhiteKernel(noise_level=0.5),
                                 normalize_y=True)
                fit GP on X_K20[tr] -> residual[tr]
                corr_va    = GP.posterior_mean(X_K20[va])
                final[va]  = nb3200[va] + corr_va

    Why clip-THEN-GP is a distinct hypothesis from GP-THEN-clip (nb3291) and
    from GP-on-raw-nb3090 (nb3282 -> 0.4452 FAIL):
      - nb3282 fit the GP on the RAW nb3090 residual. The raw anchor still
        carries the variance-compressed tail outliers (only 30 LGBM seeds,
        no tail-taming). Those tail rows dominate the residual variance, so
        the WhiteKernel(0.5) noise floor shrinks the smooth correction toward
        0 (corr_te_std=0.231) and the GP recovers ~nb3090 (0.4452 vs 0.4472).
      - Clipping FIRST (nb3200) removes the extreme tail outliers BEFORE the
        GP sees the residual. The post-clip residual has lower variance and a
        larger fraction of its remaining structure is smooth in K=20 feature
        distance, so the RBF posterior mean can extract more signal per unit
        of WhiteKernel shrinkage. If clip-then-GP < nb3200 0.4424, the GP is
        finding feature-smooth structure in the rows the clip mis-routed; if
        not, the clip already absorbed everything the smooth kernel can model
        and we recover nb3200 (safe -- GP collapses to prior mean 0).
      - Order matters because clip is a rank-preserving distributional
        rescale while GP is a feature-space additive correction; composing
        them in opposite orders yields different fixed points (cf. cycle-169
        rank-stretch axis closed at deep-ensemble level, but the clip+GP
        composition is a NEW operator pair not yet tested on this anchor).

PROTOCOL:
    1. 117-col 5-way feature matrix identical to nb3262 / nb3282 / nb3163
       (nb1352/1392/1484/1523/1524/1541 summaries + ChEMBL PXR kNN).
    2. K=20 idx from nb2280_summary ("K20_rfe_surviving_idx_in_117").
       Standardize columns (StandardScaler fit on TRAIN rows only, fold-honest).
    3. Anchor = nb3200_pred_oof (253) / te_nb3200 (513). residual = y - nb3200.
    4. HONEST 5-fold scaffold CV over 15 FRESH kf_seeds {1216..1230}:
         for each kf_seed:
             splits = scaffold_kfold_indices(unb_scaffolds, n_splits=5,
                      shuffle=True, seed=kf_seed)
             oof = full(n_unb, nan)
             for (tr_loc, va_loc) in splits:
                 scaler.fit(X_K20[tr_loc])                # fold-honest
                 GP.fit(scaler.transform(X_K20[tr_loc]), residual[tr_loc])
                 corr_va = GP.predict(scaler.transform(X_K20[va_loc]))
                 oof[va_loc] = anchor[va_loc] + corr_va
             per-fold-mean[kf_seed] = mean over the 5 folds of the val RAE.
         per-fold-mean = mean over the 15 kf_seeds of the per-fold val-RAE mean.
    5. DEPLOY: scaler+GP refit on ALL 253; te = te_nb3200 + GP.predict(te_K20).

GATE (on 15-seed per-fold-mean):
    per-fold-mean < 0.4423 -> "BETTER"   (must beat nb3200 anchor 0.4424)
    else                   -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy   (253,) float32 -- anchor OOF (clipped)
    data/processed/te_nb3200.npy         (513,) float32 -- anchor deploy (clipped)
    data/processed/nb2280_summary.json   (K=20 idx in 117-col)
    + nb1352/1392/1484/1523/1524/1541 summaries
    + te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy, te_avalon512.npy
    + C:/pxr_artifacts/nb1030/X_mordred_test.npy
    + data/external/chembl_pxr_CHEMBL3401.parquet (+ siblings)

Outputs:
    data/processed/nb3292_summary.json
    data/processed/nb3292_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3292.npy         (513,) float32 -- deploy te
    submissions/nb3292_clip_then_gp.csv  (BETTER only)
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

TAG = "nb3292"
PARENT_TAG = "nb3200"   # anchor: learned per-fold clip on nb3090 (deep-30 0.4424)

# -- Anchor (nb3200 OOF / te) -- the CLIPPED predictor (Step 1) --------------
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb3200.npy"

# -- Feature cache paths (identical to nb3262 / nb3282 / nb3163) -------------
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

# -- GP kernel hyperparams (identical to nb3282) -----------------------------
RBF_LENGTH_SCALE = 1.0
WHITE_NOISE_LEVEL = 0.5
GP_NORMALIZE_Y = True
GP_N_RESTARTS = 0          # fixed kernel (no marginal-likelihood optimization)
GP_ALPHA = 1e-10           # numerical jitter; WhiteKernel carries the noise

# -- ChEMBL kNN params (identical to nb3262 / nb3282) ------------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- HONEST cross-fit eval ---------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))   # 15 FRESH seeds {1216 ... 1230}

# -- Gates -------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References --------------------------------------------------------------
REF_NB3200 = 0.4424        # nb3200 deep-30 mean (this script's anchor, Step 1)
REF_NB3200_STD = 0.0023
REF_NB3090 = 0.4472        # parent of nb3200
REF_NB3282 = 0.4452        # GP on RAW nb3090 residual (FAIL -- contrast)
REF_NB3173 = 0.4422        # learned-clip family ref (gate band)
REF_NB2960_K18 = 0.4536
REF_NB2171 = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# feature-matrix helpers (lifted verbatim from nb3262 / nb3282)
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
    """117-col matrix identical to nb3262 / nb3282 / nb3163 / nb2960."""
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
# GP residual model (identical kernel config to nb3282)
# ============================================================================

def _make_gp():
    """RBF(1.0) + WhiteKernel(0.5), normalize_y, fixed kernel."""
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


def fold_honest_one_kf_seed(
    X_K20_unb, anchor, residual, y_unb, unb_scaffolds, kf_seed, n_folds,
):
    """One kf_seed honest 5-fold scaffold CV pass (clip-anchor then GP).

    For each fold:
        scaler.fit(X_K20[tr])                          (fold-honest standardize)
        GP.fit(scaler.transform(X_K20[tr]), residual[tr])
        corr_va = GP.posterior_mean(scaler.transform(X_K20[va]))
        oof[va] = anchor[va] + corr_va    (anchor == nb3200 clipped predictor)
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    corr_std_per_fold = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_K20_unb[tr_loc])
        X_va = scaler.transform(X_K20_unb[va_loc])
        gp = _make_gp()
        gp.fit(X_tr, residual[tr_loc])
        corr_va = gp.predict(X_va).astype(np.float64)
        oof[va_loc] = anchor[va_loc] + corr_va
        per_fold_rae.append(float(rae(y_unb[va_loc], oof[va_loc])))
        corr_std_per_fold.append(float(corr_va.std()))
    if np.isnan(oof).any():
        raise RuntimeError(f"scaffold splits did not cover all rows (kf_seed={kf_seed})")
    pooled = float(rae(y_unb, oof))
    per_fold_mean = float(np.mean(per_fold_rae))
    return pooled, per_fold_mean, oof, per_fold_rae, corr_std_per_fold


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CLIP (nb3200) THEN GP posterior on residual "
          f"(reverse of nb3291)")
    print(f"          Step 1 anchor : {PARENT_TAG} learned-clip "
          f"(deep-30 OOF RAE {REF_NB3200})")
    print(f"          Step 2 GP     : RBF(ls={RBF_LENGTH_SCALE}) + "
          f"WhiteKernel(noise={WHITE_NOISE_LEVEL}), normalize_y={GP_NORMALIZE_Y}")
    print(f"          features      : K=20 (nb2280 RFE-surviving), "
          f"StandardScaler fold-honest")
    print(f"          kf_seeds (15) : {KF_SEEDS[0]}..{KF_SEEDS[-1]}")
    print(f"          n_folds       : {N_FOLDS}")
    print(f"          gate = per-fold-mean < {GATE_BETTER} -> BETTER else FAIL")
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

    # -- Step 1 anchor (nb3200 CLIPPED OOF / te), residual -------------------
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
    print(f"[anchor] {PARENT_TAG} (Step1 clip) OOF RAE = {rae_anchor:.4f} "
          f"(ref {REF_NB3200:.4f}, d={rae_anchor - REF_NB3200:+.4f})")
    print(f"[anchor] residual mean={residual.mean():+.4f}  "
          f"std={residual.std():.4f}  min={residual.min():+.4f}  "
          f"max={residual.max():+.4f}")

    # Leak sanity (nb3200 inherits nb3090 PRE-clean chain)
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
    print("STEP A: build 117-col 5-way feature matrix and slice K=20")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K20 = X_te_full[:, K20_idx].astype(np.float64)
    X_unb_K20 = X_te_K20[unb_idx]
    assert X_unb_K20.shape == (n_unb, 20)
    print(f"   X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # -- HONEST cross-fit over 15 kf_seeds -----------------------------------
    print("\n" + "-" * 78)
    print(f"STEP B: HONEST 5-fold scaffold CV over {len(KF_SEEDS)} kf_seeds "
          f"(Step1 clip nb3200 anchor + Step2 GP posterior on residual)")
    print("-" * 78)
    per_kf_pooled = []
    per_kf_fold_mean = []
    per_kf_fold_rae = []
    per_kf_corr_std = []
    per_kf_oof = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        pooled, pf_mean, oof_s, fold_rae, corr_std = fold_honest_one_kf_seed(
            X_unb_K20, anchor_oof, residual, y_unb, unb_scaffolds,
            kf_seed=kf_seed, n_folds=N_FOLDS,
        )
        per_kf_pooled.append(pooled)
        per_kf_fold_mean.append(pf_mean)
        per_kf_fold_rae.append(fold_rae)
        per_kf_corr_std.append(corr_std)
        per_kf_oof.append(oof_s)
        wall = time.time() - ts
        print(f"   kf_seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"perfold_mean={pf_mean:.4f}  "
              f"corr_std_mean={np.mean(corr_std):.4f}  "
              f"wall={wall:.1f}s")

    pooled_arr = np.array(per_kf_pooled, dtype=np.float64)
    pf_arr = np.array(per_kf_fold_mean, dtype=np.float64)
    n_s = len(pf_arr)

    # GATE quantity = mean of per-fold val-RAE means across the 15 kf_seeds
    per_fold_mean = float(pf_arr.mean())
    pf_std = float(pf_arr.std(ddof=1)) if n_s > 1 else 0.0
    pf_sem = float(pf_std / np.sqrt(n_s)) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci95_low = float(per_fold_mean - t_mult * pf_sem)
    ci95_high = float(per_fold_mean + t_mult * pf_sem)
    pf_median = float(np.median(pf_arr))
    pf_min = float(pf_arr.min())
    pf_max = float(pf_arr.max())

    # pooled stats (diagnostic)
    mean_pooled = float(pooled_arr.mean())
    std_pooled = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0

    best_seed = int(KF_SEEDS[int(np.argmin(pf_arr))])
    print(f"\n   per-fold-mean over {n_s} kf_seeds = "
          f"{per_fold_mean:.4f} +/- {pf_std:.5f}  "
          f"(min={pf_min:.4f}  max={pf_max:.4f})")
    print(f"   95% CI = [{ci95_low:.4f}, {ci95_high:.4f}]  median={pf_median:.4f}")
    print(f"   pooled mean = {mean_pooled:.4f} +/- {std_pooled:.4f} (diagnostic)")
    print(f"   delta vs anchor nb3200 = {per_fold_mean - rae_anchor:+.4f}")
    print(f"   delta vs nb3282 (GP on raw nb3090) = "
          f"{per_fold_mean - REF_NB3282:+.4f}")

    # -- Save canonical pred_oof at median (by per-fold mean) seed -----------
    med_seed_idx = int(np.argsort(pf_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    pred_oof_canon = per_kf_oof[med_seed_idx].astype(np.float32)

    # -- Gate (on per-fold-mean) ---------------------------------------------
    print("\n" + "-" * 78)
    print("STEP C: GATE (on per-fold-mean across 15 kf_seeds)")
    print("-" * 78)
    verdict = "BETTER" if per_fold_mean < GATE_BETTER else "FAIL"
    delta_vs_anchor = per_fold_mean - rae_anchor
    delta_vs_nb3200 = per_fold_mean - REF_NB3200
    delta_vs_nb3282 = per_fold_mean - REF_NB3282
    delta_vs_nb3090 = per_fold_mean - REF_NB3090
    delta_vs_nb2171 = per_fold_mean - REF_NB2171
    print(f"   per-fold-mean              = {per_fold_mean:.4f}")
    print(f"   gate                       = < {GATE_BETTER}  -> {verdict}")
    print(f"   delta vs anchor nb3200     = {delta_vs_anchor:+.4f}  "
          f"({rae_anchor:.4f})")
    print(f"   delta vs nb3200 deep-30    = {delta_vs_nb3200:+.4f}  "
          f"({REF_NB3200:.4f})")
    print(f"   delta vs nb3282 (GP raw)   = {delta_vs_nb3282:+.4f}  "
          f"({REF_NB3282:.4f})")
    print(f"   delta vs nb3090 parent     = {delta_vs_nb3090:+.4f}  "
          f"({REF_NB3090:.4f})")
    print(f"   delta vs nb2171 ceiling    = {delta_vs_nb2171:+.4f}  "
          f"({REF_NB2171:.4f})")

    if verdict == "BETTER":
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3292 15-seed per-fold-mean "
            f"{per_fold_mean:.4f} clears BETTER gate {GATE_BETTER:.4f} "
            f"({per_fold_mean - GATE_BETTER:+.4f}) and beats nb3200 anchor "
            f"{REF_NB3200:.4f} ({delta_vs_nb3200:+.4f}). Clip-THEN-GP "
            f"(reverse of nb3291): clipping the tail outliers FIRST (nb3200) "
            f"lowers post-clip residual variance so the RBF posterior mean "
            f"extracts feature-smooth structure the raw-anchor GP (nb3282 "
            f"{REF_NB3282:.4f}) could not. PRE-clean chain (nb3090 -> nb3200). "
            f"Re-verify with deep-30 (kf_seeds 30+) before any PRIMARY-1 "
            f"swap; same under-dispersion root as cycle-160."
        )
    else:
        ladder_action = (
            f"REJECT. nb3292 15-seed per-fold-mean {per_fold_mean:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f} ({per_fold_mean - GATE_BETTER:+.4f}). "
            f"Delta vs nb3200 anchor {REF_NB3200:.4f} = {delta_vs_nb3200:+.4f}; "
            f"delta vs nb3282 GP-on-raw = {delta_vs_nb3282:+.4f}. Clipping "
            f"first does not give the RBF posterior mean enough additional "
            f"feature-smooth residual structure to beat the clip stage alone; "
            f"the learned-clip already absorbed what the K=20 smooth kernel "
            f"can model. GP-residual paradigm on the clipped anchor closed; "
            f"clip+GP operator-order does not change the fixed point materially. "
            f"Substrate change required (orthogonal anchor / off-manifold)."
        )
    print(f"   verdict                    = {verdict}")
    print(f"   ladder action              = {ladder_action}")

    # -- Deploy: GP refit on ALL 253 -> te -----------------------------------
    print("\n" + "-" * 78)
    print("STEP D: deploy GP refit on ALL 253 (clip anchor + GP) -> te_nb3292.npy")
    print("-" * 78)
    scaler_full = StandardScaler()
    X_unb_K20_std = scaler_full.fit_transform(X_unb_K20)
    X_te_K20_std = scaler_full.transform(X_te_K20)
    gp_full = _make_gp()
    gp_full.fit(X_unb_K20_std, residual)
    corr_te = gp_full.predict(X_te_K20_std).astype(np.float64)
    te_deploy = (anchor_te_513 + corr_te).astype(np.float32)
    te_deploy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_deploy_path, te_deploy)
    te_unb_in_rae = float(rae(y_unb, te_deploy[unb_idx]))
    # in-sample correction RAE on unblind (diagnostic)
    corr_unb_insample = gp_full.predict(X_unb_K20_std).astype(np.float64)
    deploy_insample_rae = float(rae(y_unb, anchor_oof + corr_unb_insample))
    print(f"   te_anchor(nb3200) : mean={anchor_te_513.mean():.4f}  "
          f"std={anchor_te_513.std():.4f}")
    print(f"   corr_te           : mean={corr_te.mean():+.4f}  "
          f"std={corr_te.std():.4f}")
    print(f"   te_corrected      : mean={te_deploy.mean():.4f}  "
          f"std={te_deploy.std():.4f}  min={te_deploy.min():.3f}  "
          f"max={te_deploy.max():.3f}")
    print(f"   te[unb] in-sample RAE       = {te_unb_in_rae:.4f}")
    print(f"   deploy-refit in-sample RAE  = {deploy_insample_rae:.4f} "
          f"(DEPLOY OPTIMISM, NOT gate)")
    print(f"   gap (in_sample - honest)    = {te_unb_in_rae - per_fold_mean:+.4f}")

    # -- Save pred_oof + submission ------------------------------------------
    print("\n" + "-" * 78)
    print("STEP E: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(oof_path, pred_oof_canon)
    print(f"   [save] {oof_path}  (median kf_seed={median_seed} OOF, 253,)  "
          f"RAE={rae(y_unb, pred_oof_canon):.4f}")
    print(f"   [save] {te_deploy_path}  (deploy refit, 513,)")

    te_clipped = np.clip(te_deploy, 3.0, 9.0).astype(np.float32)
    sub_csv = SUBMISSIONS / f"{TAG}_clip_then_gp.csv"
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
    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "clip_nb3200_then_gp_posterior_mean_on_residual_reverse_nb3291",
        "paradigm": "step1_learned_clip_step2_RBF_GP_kernel_residual_K20",
        "operator_order": "clip_first_then_gp",
        "kernel": f"RBF(length_scale={RBF_LENGTH_SCALE}) + "
                  f"WhiteKernel(noise_level={WHITE_NOISE_LEVEL})",
        "normalize_y": GP_NORMALIZE_Y,
        "gp_optimizer": "None_fixed_kernel",
        "gp_alpha": GP_ALPHA,
        "feature_set": "K=20 nb2280 RFE-surviving, StandardScaler fold-honest",
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_in_rae": round(rae_anchor, 4),
        "anchor_pre_unblind": True,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_min": float(residual.min()),
        "residual_max": float(residual.max()),
        "leak_eq_truth_frac": eq_truth_frac,
        "K20_idx_in_117col": K20_idx.tolist(),
        "kf_seeds": KF_SEEDS,
        "n_kf_seeds": n_s,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "chembl_pool_size": int(chembl_pool_size),
        "per_kf_pooled_rae": [round(float(x), 4) for x in per_kf_pooled],
        "per_kf_fold_mean": [round(float(x), 4) for x in per_kf_fold_mean],
        "per_kf_fold_rae": per_kf_fold_rae,
        "per_kf_corr_std": per_kf_corr_std,
        "pooled_rae_array": [round(float(x), 4) for x in pooled_arr.tolist()],
        "per_fold_mean_array": [round(float(x), 4) for x in pf_arr.tolist()],
        "per_fold_mean": per_fold_mean,
        "mean_rae": per_fold_mean,
        "per_fold_std": pf_std,
        "per_fold_sem": pf_sem,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "per_fold_median": pf_median,
        "per_fold_min": pf_min,
        "per_fold_max": pf_max,
        "mean_pooled_rae": mean_pooled,
        "std_pooled_rae": std_pooled,
        "best_seed": best_seed,
        "median_seed": int(median_seed),
        "delta_vs_anchor": round(delta_vs_anchor, 4),
        "delta_vs_nb3200": round(delta_vs_nb3200, 4),
        "delta_vs_nb3282": round(delta_vs_nb3282, 4),
        "delta_vs_nb3090": round(delta_vs_nb3090, 4),
        "delta_vs_nb2171": round(delta_vs_nb2171, 4),
        "ref_nb3200_deep30": REF_NB3200,
        "ref_nb3200_deep30_std": REF_NB3200_STD,
        "ref_nb3090": REF_NB3090,
        "ref_nb3282_gp_on_raw": REF_NB3282,
        "ref_nb3173": REF_NB3173,
        "ref_nb2960_K18": REF_NB2960_K18,
        "ref_nb2171": REF_NB2171,
        "ref_chemprop_aux": CHEMPROP_AUX_REF,
        "gate_better": GATE_BETTER,
        "gate_pass": bool(verdict == "BETTER"),
        "verdict": verdict,
        "ladder_action": ladder_action,
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_unb_in_sample_minus_honest_gap": round(te_unb_in_rae - per_fold_mean, 4),
        "deploy_insample_rae_NOT_GATE": round(deploy_insample_rae, 4),
        "te_mean": float(te_deploy.mean()),
        "te_std": float(te_deploy.std()),
        "te_min": float(te_deploy.min()),
        "te_max": float(te_deploy.max()),
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
    print(f"   Step1 anchor               = {PARENT_TAG} clip (RAE {rae_anchor:.4f})")
    print(f"   Step2 GP kernel            = RBF({RBF_LENGTH_SCALE}) + "
          f"White({WHITE_NOISE_LEVEL}), normalize_y")
    print(f"   per-fold-mean (15 seeds)   = {per_fold_mean:.4f} +/- {pf_std:.5f}")
    print(f"   honest min / max           = {pf_min:.4f} / {pf_max:.4f}")
    print(f"   delta vs anchor nb3200     = {delta_vs_anchor:+.4f}")
    print(f"   delta vs nb3282 (GP raw)   = {delta_vs_nb3282:+.4f}")
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
        "per_fold_std",
        "per_fold_min",
        "per_fold_max",
        "best_seed",
        "delta_vs_anchor",
        "delta_vs_nb3200",
        "delta_vs_nb3282",
        "te_unb_in_sample_rae",
        "gate_pass",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
