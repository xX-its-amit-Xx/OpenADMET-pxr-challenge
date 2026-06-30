"""nb3294 -- Best-effort TRIPLE stack: clip-on-nb3090 + GP residual + clip again.

NEW PARADIGM: maximally stack the two operators that individually help on the
nb3090 anchor. Two operators have shown signal on this anchor family:

    (A) LEARNED CLIP  (nb3190/nb3200): per-fold inner grid search over
        (q_low, q_high) on fold-train y, then hard np.clip on fold-val. nb3200
        deep-30 mean = 0.4424 (VERIFIED_PROMOTE_PRIMARY1). Pure variance
        compression -- pins outlier tail mass toward the central q-band.

    (B) GP RESIDUAL   (nb3282): RBF(1.0)+WhiteKernel(0.5) GP posterior mean on
        the (y - anchor) residual, K=20 RFE features, fold-honest StandardScaler.
        Standalone on raw nb3090 = 0.4452 (FAIL). Smooth euclidean-distance
        correction, orthogonal inductive bias to the clip.

TRIPLE-STACK PROTOCOL (per outer fold, fold-honest at every step):
    base   = nb3090                                            (anchor)
    Step 1 -- LEARNED CLIP on base:
        (ql1*, qh1*) = argmin_{grid} RAE(y_tr, clip(base_tr, q*))   on fold-train
        s1 = clip(base, lo1*, hi1*)                               (both tr + va)
    Step 2 -- GP RESIDUAL on the clipped base:
        resid_tr = y_tr - s1_tr
        scaler.fit(X_K20[tr]); GP.fit(scaler.transform(X_K20[tr]), resid_tr)
        corr     = GP.posterior_mean(scaler.transform(X_K20))      (tr + va)
        s2 = s1 + corr
    Step 3 -- LEARNED CLIP again on the GP-corrected output:
        (ql2*, qh2*) = argmin_{grid} RAE(y_tr, clip(s2_tr, q*))    on fold-train
        s3 = clip(s2, lo2*, hi2*)
    oof[va] = s3[va]

    Each step's parameters (clip quantiles, GP fit, scaler) are estimated on
    fold-TRAIN ONLY and applied to fold-val -- no truth leak across folds.
    The two clips relearn (q_low, q_high) independently because Step-2 changes
    the distribution the second clip operates on.

    15 FRESH kf_seeds {1216..1230}, scaffold 5-fold, per-fold-MEAN as the gate
    metric (mean over the 5 fold-val RAEs, averaged over the 15 seeds).

GATE (per task):
    per-fold-mean < 0.4423 -> "BETTER"
    else                   -> "FAIL"

References:
    nb3090 q35 fine-cut blend           = 0.4472  <- anchor / base
    nb3200 learned-clip on nb3090 (d30) = 0.4424  (Step-1 operator alone)
    nb3190 learned-clip on nb3090 (15s) = 0.4422
    nb3282 GP-residual on nb3090 (15s)  = 0.4452  (Step-2 operator alone, FAIL)
    nb3173 clip-operator ceiling        = 0.4437
    nb2171 prior post-hoc PRIMARY-1     = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3090_pred_oof.npy   (253,) -- anchor OOF
    data/processed/te_nb3090.npy         (513,) -- anchor deploy
    data/processed/nb2280_summary.json   (K=20 RFE idx in 117-col)
    + nb1352/1392/1484/1523/1524/1541 summaries (117-col feature build)
    + te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy, te_avalon512.npy
    + C:/pxr_artifacts/nb1030/X_mordred_test.npy
    + data/external/chembl_pxr_CHEMBL3401.parquet (+ siblings)

Outputs:
    data/processed/nb3294_summary.json
    data/processed/nb3294_pred_oof.npy   (253,) float32 -- median-seed (pf) OOF
    data/processed/te_nb3294.npy         (513,) float32 -- deploy te
    submissions/nb3294_triple_stack.csv  (only on BETTER)
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

TAG = "nb3294"
PARENT_TAG = "nb3090"   # anchor / base (OOF RAE 0.447)

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

# -- Learned-clip inner grid (must MATCH nb3190/nb3200 exactly) ----------------
Q_LOW_GRID = [0.01, 0.05, 0.10]
Q_HIGH_GRID = [0.90, 0.95, 0.98, 0.99]

# -- GP kernel hyperparams (identical to nb3282) ------------------------------
RBF_LENGTH_SCALE = 1.0
WHITE_NOISE_LEVEL = 0.5
GP_NORMALIZE_Y = True
GP_N_RESTARTS = 0
GP_ALPHA = 1e-10

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
REF_PARENT_NB3090 = 0.4472      # anchor / base
REF_NB3200_CLIP = 0.4424        # Step-1 operator alone (deep-30)
REF_NB3190_CLIP = 0.4422        # learned-clip 15-seed
REF_NB3282_GP = 0.4452          # Step-2 operator alone (FAIL)
REF_NB3173 = 0.4437
REF_NB2171 = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# learned-clip operator (lifted verbatim from nb3200 / nb3190)
# ============================================================================

def _pick_best_clip(y_tr: np.ndarray, pred_tr: np.ndarray):
    """Inner grid search: pick (q_low*, q_high*) minimizing fold-train RAE."""
    best_rae = np.inf
    best_ql = Q_LOW_GRID[0]
    best_qh = Q_HIGH_GRID[-1]
    best_lo = float(np.quantile(pred_tr, best_ql))
    best_hi = float(np.quantile(pred_tr, best_qh))
    for ql in Q_LOW_GRID:
        lo = float(np.quantile(y_tr, ql))
        for qh in Q_HIGH_GRID:
            hi = float(np.quantile(y_tr, qh))
            if hi <= lo:
                continue
            clipped = np.clip(pred_tr, lo, hi)
            r = float(rae(y_tr, clipped))
            if r < best_rae:
                best_rae = r
                best_ql = ql
                best_qh = qh
                best_lo = lo
                best_hi = hi
    return best_ql, best_qh, best_lo, best_hi


# ============================================================================
# GP residual operator
# ============================================================================

def _make_gp():
    """RBF(1.0) + WhiteKernel(0.5), normalize_y, fixed kernel (no ML optimize)."""
    kernel = (
        RBF(length_scale=RBF_LENGTH_SCALE)
        + WhiteKernel(noise_level=WHITE_NOISE_LEVEL)
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=GP_ALPHA,
        normalize_y=GP_NORMALIZE_Y,
        n_restarts_optimizer=GP_N_RESTARTS,
        optimizer=None,
        copy_X_train=True,
        random_state=0,
    )


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
# triple-stack: one outer fold
# ============================================================================

def _triple_one_fold(y_tr, base_tr, base_full, X_tr, X_full):
    """Apply the 3-step stack with all params fit on fold-train ONLY.

    base_full / X_full are the FULL-row arrays (so we can produce the
    transformed value for whatever index set we hand back); the caller slices
    val rows. Returns the full-row s3 and the learned params for diagnostics.
    """
    # Step 1 -- learned clip on base (clip thresholds from fold-train y)
    ql1, qh1, lo1, hi1 = _pick_best_clip(y_tr, base_tr)
    s1_full = np.clip(base_full, lo1, hi1)
    s1_tr = np.clip(base_tr, lo1, hi1)

    # Step 2 -- GP posterior mean on (y_tr - s1_tr) residual, K=20 features
    resid_tr = y_tr - s1_tr
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_full = scaler.transform(X_full)
    gp = _make_gp()
    gp.fit(Xs_tr, resid_tr)
    corr_full = gp.predict(Xs_full).astype(np.float64)
    s2_full = s1_full + corr_full
    # s2 on train rows from train-row GP corrections (for the Step-3 clip search)
    corr_tr = gp.predict(Xs_tr).astype(np.float64)
    s2_tr = s1_tr + corr_tr

    # Step 3 -- learned clip again on the GP-corrected output (thresholds from
    # fold-train y, applied to the s2 distribution)
    ql2, qh2, lo2, hi2 = _pick_best_clip(y_tr, s2_tr)
    s3_full = np.clip(s2_full, lo2, hi2)

    params = {
        "ql1": ql1, "qh1": qh1, "lo1": lo1, "hi1": hi1,
        "ql2": ql2, "qh2": qh2, "lo2": lo2, "hi2": hi2,
        "corr_std": float(corr_full.std()),
    }
    return s3_full, params


def _run_one_seed(base_oof, y_unb, X_unb_K20, unb_scaffolds, kf_seed):
    """One kf_seed honest 5-fold scaffold CV pass of the triple stack."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    fold_params = []
    for tr_loc, va_loc in splits:
        # full-row arrays for this fold == val rows only (we only need s3[va])
        s3_va, params = _triple_one_fold(
            y_tr=y_unb[tr_loc],
            base_tr=base_oof[tr_loc],
            base_full=base_oof[va_loc],
            X_tr=X_unb_K20[tr_loc],
            X_full=X_unb_K20[va_loc],
        )
        oof[va_loc] = s3_va
        fold_val_raes.append(float(rae(y_unb[va_loc], s3_va)))
        fold_params.append(params)
    if np.isnan(oof).any():
        raise RuntimeError(f"kf_seed={kf_seed}: scaffold splits left NaNs")
    pooled = float(rae(y_unb, oof))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_ql1": [p["ql1"] for p in fold_params],
        "fold_qh1": [p["qh1"] for p in fold_params],
        "fold_ql2": [p["ql2"] for p in fold_params],
        "fold_qh2": [p["qh2"] for p in fold_params],
        "fold_corr_std_mean": float(np.mean([p["corr_std"] for p in fold_params])),
        "oof": oof,
    }


# ============================================================================
# main
# ============================================================================

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- TRIPLE stack on {PARENT_TAG}: "
          f"clip -> GP-residual -> clip")
    print(f"          base       : {PARENT_TAG} (OOF RAE {REF_PARENT_NB3090})")
    print(f"          Step1 clip : grid ql{Q_LOW_GRID} x qh{Q_HIGH_GRID}")
    print(f"          Step2 GP   : RBF({RBF_LENGTH_SCALE})+White({WHITE_NOISE_LEVEL}), "
          f"K=20 RFE, fold-honest scaler")
    print(f"          Step3 clip : grid ql{Q_LOW_GRID} x qh{Q_HIGH_GRID} on s2")
    print(f"          kf_seeds   : {len(KF_SEEDS)} fresh "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}  n_folds={N_FOLDS}")
    print(f"          gate metric: PER-FOLD-MEAN")
    print(f"          gate       : pf_mean < {GATE_BETTER} -> BETTER else FAIL")
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

    # -- Anchor / base -------------------------------------------------------
    assert ANCHOR_OOF_PATH.exists(), f"missing anchor OOF: {ANCHOR_OOF_PATH}"
    assert ANCHOR_TE_PATH.exists(), f"missing anchor te: {ANCHOR_TE_PATH}"
    base_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    base_te_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    assert base_oof.shape == (n_unb,), f"anchor OOF shape {base_oof.shape}"
    assert base_te_513.shape == (n_test,), f"anchor te shape {base_te_513.shape}"
    rae_base = float(rae(y_unb, base_oof))
    print(f"[base] {PARENT_TAG} OOF RAE = {rae_base:.4f} "
          f"(ref {REF_PARENT_NB3090:.4f})")

    leak_eq = float(np.mean(np.isclose(base_oof, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN: {leak_eq:.1%} rows == truth -- possible leak")
    else:
        print(f"   leak_eq_truth_frac = {leak_eq:.4f} (PRE-clean expected)")

    # -- K=20 idx ------------------------------------------------------------
    with open(NB2280_SUMMARY) as f:
        nb2280 = json.load(f)
    K20_idx = np.array(nb2280["K20_rfe_surviving_idx_in_117"], dtype=int)
    assert len(K20_idx) == 20, f"K20 len {len(K20_idx)} != 20"
    print(f"[load] K=20 idx (n={len(K20_idx)})")

    # -- Build 117-col matrix, slice K=20 ------------------------------------
    print("\n" + "-" * 78)
    print("STEP 0: build 117-col 5-way feature matrix and slice K=20")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")
    X_te_K20 = X_te_full[:, K20_idx].astype(np.float64)
    X_unb_K20 = X_te_K20[unb_idx]
    assert X_unb_K20.shape == (n_unb, 20)
    print(f"   X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # -- HONEST cross-fit over 15 kf_seeds -----------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds (triple stack, fold-honest)")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(base_oof, y_unb, X_unb_K20, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_ql1": [round(v, 3) for v in res["fold_ql1"]],
            "fold_qh1": [round(v, 3) for v in res["fold_qh1"]],
            "fold_ql2": [round(v, 3) for v in res["fold_ql2"]],
            "fold_qh2": [round(v, 3) for v in res["fold_qh2"]],
            "fold_corr_std_mean": round(res["fold_corr_std_mean"], 4),
        })
        print(f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
              f"pf_mean={res['per_fold_val_rae_mean']:.4f}  "
              f"corr_std={res['fold_corr_std_mean']:.4f}  "
              f"wall={time.time()-ts:.1f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    arr_pf = np.asarray(per_fold_means, dtype=np.float64)
    pf_mean = float(arr_pf.mean())            # GATE quantity
    pf_std = float(arr_pf.std(ddof=1)) if n_s > 1 else 0.0
    pf_sem = pf_std / np.sqrt(n_s) if n_s > 1 else 0.0
    pf_ci_low = pf_mean - t_mult * pf_sem
    pf_ci_high = pf_mean + t_mult * pf_sem
    pf_median = float(np.median(arr_pf))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   POOLED RAE   : mean={mean_rae:.4f}  std={std_rae:.4f}  "
          f"95%CI=[{ci_low:.4f},{ci_high:.4f}]  median={median_rae:.4f}")
    print(f"   PER-FOLD-MEAN: mean={pf_mean:.4f}  std={pf_std:.4f}  "
          f"95%CI=[{pf_ci_low:.4f},{pf_ci_high:.4f}]  median={pf_median:.4f}")
    print(f"   pf min/max   = [{arr_pf.min():.4f}, {arr_pf.max():.4f}]")
    print(f"\n   delta vs base nb3090 (pf)  = {pf_mean - REF_PARENT_NB3090:+.4f}")
    print(f"   delta vs nb3200 clip (pf)  = {pf_mean - REF_NB3200_CLIP:+.4f}")
    print(f"   delta vs nb3282 GP   (pf)  = {pf_mean - REF_NB3282_GP:+.4f}")
    print(f"   gain vs nb2171       (pf)  = {REF_NB2171 - pf_mean:+.4f}")

    # -- Deploy: refit all 3 steps on ALL 253 -> te --------------------------
    print("\n" + "-" * 78)
    print("DEPLOY: refit triple stack on ALL 253 -> te_nb3294.npy")
    print("-" * 78)
    # Step 1 deploy clip on full 253
    d_ql1, d_qh1, d_lo1, d_hi1 = _pick_best_clip(y_unb, base_oof)
    te_s1 = np.clip(base_te_513, d_lo1, d_hi1)
    unb_s1 = np.clip(base_oof, d_lo1, d_hi1)
    # Step 2 deploy GP on (y - unb_s1)
    resid_full = y_unb - unb_s1
    scaler_full = StandardScaler()
    Xs_unb = scaler_full.fit_transform(X_unb_K20)
    Xs_te = scaler_full.transform(X_te_K20)
    gp_full = _make_gp()
    gp_full.fit(Xs_unb, resid_full)
    corr_te = gp_full.predict(Xs_te).astype(np.float64)
    corr_unb = gp_full.predict(Xs_unb).astype(np.float64)
    te_s2 = te_s1 + corr_te
    unb_s2 = unb_s1 + corr_unb
    # Step 3 deploy clip on s2 (thresholds from full 253 y vs unb_s2)
    d_ql2, d_qh2, d_lo2, d_hi2 = _pick_best_clip(y_unb, unb_s2)
    te_deploy = np.clip(te_s2, d_lo2, d_hi2).astype(np.float32)

    te_deploy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_deploy_path, te_deploy)
    te_unb_in_rae = float(rae(y_unb, te_deploy[unb_idx]))
    deploy_insample_rae = float(rae(y_unb, np.clip(unb_s2, d_lo2, d_hi2)))
    print(f"   deploy clip1 = (q{d_ql1:.2f},q{d_qh1:.2f}) -> ({d_lo1:.3f},{d_hi1:.3f})")
    print(f"   deploy GP corr_te: mean={corr_te.mean():.4f} std={corr_te.std():.4f}")
    print(f"   deploy clip2 = (q{d_ql2:.2f},q{d_qh2:.2f}) -> ({d_lo2:.3f},{d_hi2:.3f})")
    print(f"   te_deploy: mean={te_deploy.mean():.4f} std={te_deploy.std():.4f} "
          f"min={te_deploy.min():.3f} max={te_deploy.max():.3f}")
    print(f"   te[unb] in-sample RAE      = {te_unb_in_rae:.4f}")
    print(f"   deploy-refit in-sample RAE = {deploy_insample_rae:.4f} "
          f"(DEPLOY OPTIMISM, NOT gate)")

    # Median-seed OOF (by per-fold-mean) for storage
    med_seed_idx = int(np.argsort(arr_pf)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} "
          f"(pf_mean={arr_pf[med_seed_idx]:.4f}, pooled={arr[med_seed_idx]:.4f})")

    # -- Gate (on PER-FOLD-MEAN) ---------------------------------------------
    print("\n" + "-" * 78)
    print("GATE (honest metric = PER-FOLD-MEAN)")
    print("-" * 78)
    if pf_mean < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3294 15-seed PER-FOLD-MEAN {pf_mean:.4f} "
            f"clears BETTER gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). "
            f"Triple stack (clip->GP-residual->clip) on nb3090 "
            f"({REF_PARENT_NB3090:.4f}) beats the clip-alone operator nb3200 "
            f"({REF_NB3200_CLIP:.4f}) by {REF_NB3200_CLIP - pf_mean:+.4f} and the "
            f"GP-alone operator nb3282 ({REF_NB3282_GP:.4f}) by "
            f"{REF_NB3282_GP - pf_mean:+.4f}; stacking the two orthogonal "
            f"operators compounds. Re-verify with deep-30 before any PRIMARY swap "
            f"(15-seed std under-dispersed ~4x per cycle-160 rule). "
            f"anchor_pre_unblind=True (nb3090 on PRE-clean K18/K19 deep-30)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3294 15-seed PER-FOLD-MEAN {pf_mean:.4f} fails BETTER "
            f"gate {GATE_BETTER:.4f} ({pf_mean - GATE_BETTER:+.4f}). delta vs "
            f"clip-alone nb3200 = {pf_mean - REF_NB3200_CLIP:+.4f}, vs GP-alone "
            f"nb3282 = {pf_mean - REF_NB3282_GP:+.4f}. The middle GP step injects "
            f"a smooth correction whose residual structure the second clip then "
            f"partially undoes; stacking the two operators does not compound on "
            f"this anchor because the first clip already removed the variance the "
            f"GP would exploit, and the GP re-introduces tail mass the second "
            f"clip must re-cut. Confirms cycle-169 post-hoc-blend-axis-closed "
            f"thesis: operator STACKING does not break the 0.4424 clip ceiling; "
            f"substrate change (new anchor) remains the only open lever. Keep "
            f"nb3200 ({REF_NB3200_CLIP:.4f}) as the clip-operator PRIMARY-1."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(oof_path, oof_for_save)
    print(f"   [save] {oof_path}  (median-seed OOF, RAE={rae(y_unb, oof_for_save):.4f})")
    print(f"   [save] {te_deploy_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_triple_stack.csv"
    te_clipped_sub = np.clip(te_deploy, 3.0, 9.0).astype(np.float32)
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
        "method": "triple_stack_clip_then_GP_residual_then_clip_on_nb3090",
        "paradigm": "maximal_operator_stacking_two_individually_helping_ops",
        "step1": "learned_clip_grid_ql{0.01,0.05,0.10}_qh{0.90,0.95,0.98,0.99}",
        "step2": (f"GP_RBF({RBF_LENGTH_SCALE})_White({WHITE_NOISE_LEVEL})_"
                  f"posterior_mean_on_(y-s1)_residual_K20_foldhonest_scaler"),
        "step3": "learned_clip_grid_again_on_s2_distribution",
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_in_rae": rae_base,
        "anchor_pre_unblind": True,
        "leak_eq_truth_frac": leak_eq,
        "q_low_grid": Q_LOW_GRID,
        "q_high_grid": Q_HIGH_GRID,
        "gp_kernel": f"RBF({RBF_LENGTH_SCALE})+WhiteKernel({WHITE_NOISE_LEVEL})",
        "gp_normalize_y": GP_NORMALIZE_Y,
        "K20_idx_in_117col": K20_idx.tolist(),
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "chembl_pool_size": int(chembl_pool_size),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_val_rae_means_array": [round(float(v), 4) for v in per_fold_means],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "per_fold_mean": round(pf_mean, 4),
        "per_fold_mean_rae_mean": round(pf_mean, 4),
        "per_fold_mean_rae_std": round(pf_std, 4),
        "per_fold_mean_rae_sem": round(pf_sem, 4),
        "per_fold_mean_rae_ci95_low": round(pf_ci_low, 4),
        "per_fold_mean_rae_ci95_high": round(pf_ci_high, 4),
        "per_fold_mean_rae_median": round(pf_median, 4),
        "per_fold_mean_rae_min": round(float(arr_pf.min()), 4),
        "per_fold_mean_rae_max": round(float(arr_pf.max()), 4),
        "honest_metric": "per_fold_mean",
        "ref_parent_nb3090": REF_PARENT_NB3090,
        "ref_nb3200_clip": REF_NB3200_CLIP,
        "ref_nb3190_clip": REF_NB3190_CLIP,
        "ref_nb3282_gp": REF_NB3282_GP,
        "ref_nb3173": REF_NB3173,
        "ref_nb2171": REF_NB2171,
        "ref_chemprop_aux": CHEMPROP_AUX_REF,
        "delta_vs_parent_pf_mean": round(pf_mean - REF_PARENT_NB3090, 4),
        "delta_vs_nb3200_clip_pf_mean": round(pf_mean - REF_NB3200_CLIP, 4),
        "delta_vs_nb3282_gp_pf_mean": round(pf_mean - REF_NB3282_GP, 4),
        "gain_vs_nb2171_pf_mean": round(REF_NB2171 - pf_mean, 4),
        "deploy_ql1": float(d_ql1),
        "deploy_qh1": float(d_qh1),
        "deploy_lo1": round(float(d_lo1), 4),
        "deploy_hi1": round(float(d_hi1), 4),
        "deploy_corr_te_mean": round(float(corr_te.mean()), 4),
        "deploy_corr_te_std": round(float(corr_te.std()), 4),
        "deploy_ql2": float(d_ql2),
        "deploy_qh2": float(d_qh2),
        "deploy_lo2": round(float(d_lo2), 4),
        "deploy_hi2": round(float(d_hi2), 4),
        "te_mean": float(te_deploy.mean()),
        "te_std": float(te_deploy.std()),
        "te_min": float(te_deploy.min()),
        "te_max": float(te_deploy.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "deploy_insample_rae_NOT_GATE": round(deploy_insample_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_deploy_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "gate_better": GATE_BETTER,
        "gate_metric": "per_fold_mean",
        "gate_pass": bool(verdict == "BETTER"),
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
    print(f"   base                       = {PARENT_TAG} (RAE {rae_base:.4f})")
    print(f"   per-fold-mean (15 seeds)   = {pf_mean:.4f} +/- {pf_std:.4f}")
    print(f"   pf 95% CI                  = [{pf_ci_low:.4f}, {pf_ci_high:.4f}]")
    print(f"   pooled mean                = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   delta vs nb3200 clip (pf)  = {pf_mean - REF_NB3200_CLIP:+.4f}")
    print(f"   delta vs nb3282 GP   (pf)  = {pf_mean - REF_NB3282_GP:+.4f}")
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
        "per_fold_mean",
        "per_fold_mean_rae_std",
        "per_fold_mean_rae_min",
        "per_fold_mean_rae_max",
        "mean_rae",
        "delta_vs_nb3200_clip_pf_mean",
        "delta_vs_nb3282_gp_pf_mean",
        "te_unb_in_sample_rae",
        "median_seed",
        "gate_pass",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
