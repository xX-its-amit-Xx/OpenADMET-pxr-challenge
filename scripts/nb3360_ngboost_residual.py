"""nb3360 -- NGBoost (natural gradient boosting) on (y - nb3200) residual.

NEW PARADIGM (probabilistic boosting, distinct inductive bias):
    NGBoost fits a full PROBABILITY DISTRIBUTION per row (default Normal) by
    boosting on the NATURAL GRADIENT of the scoring rule (LogScore), rather
    than the ordinary gradient of a point-loss like LGBM (nb3262), ExtraTrees
    (nb3342), or the global linear BayesianRidge (nb3293) used on this same
    residual.  The natural gradient rescales the update by the Fisher
    information of the parameter manifold, so each boosting step moves the
    distribution by a fixed amount in KL-divergence rather than in raw
    parameter space.  We take the MEAN parameter (mu) of the predicted Normal
    as the residual correction:  final = nb3200 + NGB.predict()  (predict()
    returns mu == pred_dist().mean()).  The hypothesis: the natural-gradient
    geometry captures residual structure on the SAME K=20 5-way feature slice
    that ordinary-gradient trees (nb3262 LGBM, nb3342 ExtraTrees) and the
    linear nb3293 BayesianRidge all failed to extract -- probing whether the
    failure was the OPTIMIZATION GEOMETRY rather than the feature capacity.

ANCHOR (PRE-clean chain via nb3090 -> nb3200):
    nb3200 deep-30 OOF RAE  = 0.4424 (cycle-160 PRIMARY-1 candidate)
    target gate (BETTER)    < 0.4423 (per-fold-mean)

PROTOCOL (per kf_seed, 5-fold scaffold split on 253 unblind):
    residual = y_unb - nb3200_oof
    For each fold:
        mdl = NGBRegressor(n_estimators=200, learning_rate=0.01,
                           random_state=kf_seed, verbose=False)
        mdl.fit(X_unb_K20[tr_loc], residual[tr_loc])
        oof[va_loc] = clip(nb3200_oof[va_loc] + mdl.predict(X_unb_K20[va_loc]),
                           CLIP_LO, CLIP_HI)
    pooled            = rae(y_unb, oof)            (reported, not gated)
    per_fold_mean     = mean(per-fold val RAE)     <- GATED METRIC
    Repeat for 15 FRESH kf_seeds {1216..1230}.

DEPLOY:
    Refit NGBRegressor on (X_unb_K20, residual) once per kf_seed; mean-bag the
    predicted mu over the 15 seeds on X_te_K20; te_final = clip(te_nb3200 +
    mean-bag mu, CLIP_LO, CLIP_HI).

GATE (on 15-seed per-fold-mean):
    per_fold_mean < 0.4423 -> "BETTER" (beats nb3200 deep-30 mean)
    else                   -> "FAIL"

INSTALL GUARD:
    If `import ngboost` fails, write {"verdict": "INSTALL_FAILED", ...} to
    nb3360_summary.json and exit(0) cleanly -- no pred_oof / te / CSV written.

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy   data/processed/te_nb3200.npy
    data/processed/nb2280_summary.json   (K=20 idx in 117col)
    + nb1352/1392/1484/1523/1524/1541 summaries (117-col feature recipe)
    + te_atompair.npy, te_maccs.npy, te_chemprop_embed_300.npy, te_avalon512.npy
    + C:/pxr_artifacts/nb1030/X_mordred_test.npy
    + data/external/chembl_pxr_CHEMBL3401.parquet (+ siblings)

Outputs:
    data/processed/nb3360_summary.json
    data/processed/nb3360_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3360.npy         (513,) float32 -- deploy te
    submissions/nb3360_ngboost_residual.csv  (only on BETTER)
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

from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3360"
PARENT_TAG = "nb3200"


# ============================================================================
# INSTALL GUARD -- if ngboost is unavailable, save INSTALL_FAILED + exit clean
# ============================================================================
def _install_failed_exit(err_msg: str):
    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "ngboost_natural_gradient_boosting_on_y_minus_nb3200_residual",
        "verdict": "INSTALL_FAILED",
        "import_error": str(err_msg),
        "ladder_action": (
            "INSTALL_FAILED. `import ngboost` could not be satisfied in this "
            "environment; no NGBoost residual model was fit. No pred_oof/te/CSV "
            "written. nb3200 (deep-30 0.4424) remains the PRIMARY-1 candidate; "
            "ladder unchanged."
        ),
        "pred_oof_path": None,
        "te_npy_path": None,
        "submission_csv": None,
    }
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INSTALL_FAILED] {err_msg}")
    print(f"   [save] {out_path}")
    print("INSTALL_FAILED")
    sys.exit(0)


try:
    from ngboost import NGBRegressor  # noqa: F401
except Exception as _e:  # pragma: no cover - exercised only when import broken
    _install_failed_exit(_e)


# -- heavy imports AFTER the install guard passed ----------------------------
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices

# -- Anchor (nb3200) ---------------------------------------------------------
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb3200.npy"

# -- 117-col feature recipe (identical to nb3342 / nb3293 / nb3262 / nb3163) -
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
NB2280_SUMMARY = DATA_PROCESSED / "nb2280_summary.json"   # K=20 idx in 117col

# -- ChEMBL kNN feature config (identical to upstream) -----------------------
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

# -- NGBoost residual learner params (per task spec) -------------------------
NGB_N_ESTIMATORS = 200
NGB_LEARNING_RATE = 0.01

# -- Output range clip (matches nb3342 / nb3070 / nb3263 deploy stage) -------
CLIP_LO = 3.0
CLIP_HI = 9.0

# -- CV protocol -------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Gate --------------------------------------------------------------------
GATE_BETTER = 0.4423  # per-fold-mean < this -> BETTER (beats nb3200 0.4424)

# -- References --------------------------------------------------------------
REF_NB3200 = 0.4424     # cycle-160 deep-30 mean (PRIMARY-1 candidate)
REF_NB3200_STD = 0.0023
REF_NB3262 = 0.4424     # LGBM(K=20) on same 117col residual (FAIL)
REF_NB3313 = 0.4424     # LGBM(Avalon-512) on residual (FAIL)
REF_NB3342 = 0.4424     # ExtraTrees(K=20) on same residual (FAIL placeholder)
REF_NB3293 = 0.4472     # BayesianRidge(K=20) on nb3090 residual
REF_NB3090 = 0.4472     # parent of nb3200
REF_NB2171 = 0.4682     # prior PRIMARY-1 anchor swap
REF_NB1191 = 0.4718     # PRE-pyramid wide-seed mean
CHEMPROP_AUX_REF = 0.6216


def _ngb_params(seed):
    """NGBRegressor residual learner -- per task spec.

    n_estimators=200, learning_rate=0.01.  Default Dist=Normal, Score=LogScore,
    natural_gradient=True.  predict() returns the MEAN parameter (mu) of the
    predicted Normal (== pred_dist().mean()), which is the residual correction.
    The natural-gradient geometry is a DIFFERENT optimization bias than the
    ordinary-gradient trees (LGBM nb3262, ExtraTrees nb3342) on the SAME K=20
    features.
    """
    return dict(
        n_estimators=NGB_N_ESTIMATORS,
        learning_rate=NGB_LEARNING_RATE,
        random_state=seed,
        verbose=False,
    )


# ============================================================================
# helpers (lifted verbatim from nb3342 / nb3293 117-col feature builder)
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
    """117-col matrix identical to nb3342 / nb3293 / nb3262 / nb3163 / nb2960."""
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
# core honest CV: NGBoost(K=20) on (y - nb3200) residual per kf_seed
# ============================================================================

def _run_one_seed(
    X_unb_K20, anchor_oof, residual, y_unb, unb_scaffolds, kf_seed, n_folds,
):
    """One kf_seed honest n-fold scaffold CV; NGBRegressor(seed=kf_seed) resid.

    NGBoost is stochastic in the base-learner column subsampling and minibatch
    fraction, so random_state=kf_seed couples both the fold partition and the
    boosting randomness to the seed.  predict() returns the Normal mean mu.
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_val_rae = []
    per_fold_train_rae = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        mdl = NGBRegressor(**_ngb_params(kf_seed))
        mdl.fit(X_unb_K20[tr_loc], residual[tr_loc])
        resid_pred_va = np.asarray(mdl.predict(X_unb_K20[va_loc]), dtype=np.float64)
        oof[va_loc] = np.clip(
            anchor_oof[va_loc] + resid_pred_va, CLIP_LO, CLIP_HI
        )
        per_fold_val_rae.append(float(rae(y_unb[va_loc], oof[va_loc])))
        resid_pred_tr = np.asarray(mdl.predict(X_unb_K20[tr_loc]), dtype=np.float64)
        per_fold_train_rae.append(
            float(rae(
                y_unb[tr_loc],
                np.clip(anchor_oof[tr_loc] + resid_pred_tr, CLIP_LO, CLIP_HI),
            ))
        )
    if np.isnan(oof).any():
        raise RuntimeError(
            f"scaffold splits did not cover all rows (kf_seed={kf_seed})"
        )
    pooled = float(rae(y_unb, oof))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(per_fold_val_rae)),
        "per_fold_val_rae_std": (
            float(np.std(per_fold_val_rae, ddof=1))
            if len(per_fold_val_rae) > 1 else 0.0
        ),
        "per_fold_val_rae": per_fold_val_rae,
        "per_fold_train_rae_mean": float(np.mean(per_fold_train_rae)),
        "oof": oof,
    }


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- NGBoost (natural gradient) on (y - nb3200) residual, K=20")
    print(f"          parent     : {PARENT_TAG} (deep-30 mean {REF_NB3200:.4f})")
    print(f"          features   : 117-col 5-way -> K=20 (nb2280_RFE; same as nb3342)")
    print(f"          learner    : NGBRegressor(n_est={NGB_N_ESTIMATORS}, "
          f"lr={NGB_LEARNING_RATE}) -> mean param mu")
    print(f"          paradigm   : natural-gradient probabilistic boosting "
          f"(DIFFERENT geometry than nb3262/nb3342)")
    print(f"          clip       : [{CLIP_LO}, {CLIP_HI}]")
    print(f"          kf_seeds   : {len(KF_SEEDS)} FRESH "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          gate       : per_fold_mean < {GATE_BETTER:.4f}"
          f" -> BETTER else FAIL")
    print("=" * 78)

    # -- Load test + truth + unblind idx -------------------------------------
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
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Scaffolds for honest CV ---------------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique = {n_unique_scaf}")

    # -- Load nb3200 anchor (PRE-clean chain via nb3090) ---------------------
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)  # (253,)
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)    # (513,)
    if anchor_oof.shape != (n_unb,):
        raise ValueError(f"nb3200 oof shape {anchor_oof.shape} != ({n_unb},)")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"nb3200 te shape {anchor_te.shape} != ({n_test},)")
    rae_anchor = float(rae(y_unb, anchor_oof))
    leak_eq = float(np.mean(np.isclose(anchor_oof, y_unb, atol=1e-6)))
    residual = y_unb - anchor_oof
    print(f"[anchor] nb3200 oof RAE = {rae_anchor:.4f}  (ref {REF_NB3200:.4f}, "
          f"d={rae_anchor - REF_NB3200:+.4f})")
    print(f"[anchor] leak_eq_truth_frac = {leak_eq:.2%}")
    print(f"[residual] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # -- Build 117-col matrix, slice K=20 ------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: build 117-col 5-way feature matrix, slice K=20 (nb2280_RFE)")
    print("-" * 78)
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}  chembl_pool={chembl_pool_size}")

    with open(NB2280_SUMMARY) as f:
        nb2280 = json.load(f)
    K20_idx = np.array(nb2280["K20_rfe_surviving_idx_in_117"], dtype=int)
    assert len(K20_idx) == 20, f"K20 len {len(K20_idx)} != 20"
    print(f"   K=20 idx in 117col (n={len(K20_idx)}): {K20_idx.tolist()}")

    X_te_K20 = X_te_full[:, K20_idx].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    assert X_unb_K20.shape == (n_unb, 20)
    print(f"   X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # -- Multi-seed honest cross-fit -----------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 2: HONEST {N_FOLDS}-fold scaffold CV over {len(KF_SEEDS)}"
          f" kf_seeds (NGBoost residual)")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(
            X_unb_K20, anchor_oof, residual, y_unb, unb_scaffolds,
            kf_seed=s, n_folds=N_FOLDS,
        )
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_train_rae_mean": round(res["per_fold_train_rae_mean"], 4),
        })
        print(f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
              f"perfold_mean={res['per_fold_val_rae_mean']:.4f}  "
              f"perfold_std={res['per_fold_val_rae_std']:.4f}  "
              f"train_mean={res['per_fold_train_rae_mean']:.4f}  "
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

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled  mean    = {mean_pooled:.4f}  std = {std_pooled:.4f}")
    print(f"   perfold mean    = {mean_pf:.4f}  std = {std_pf:.4f}")
    print(f"   perfold sem     = {sem_pf:.4f}")
    print(f"   perfold 95% CI  = [{ci_low_pf:.4f}, {ci_high_pf:.4f}]")
    print(f"   perfold median  = {median_pf:.4f}")
    print(f"   perfold min/max = [{pf_arr.min():.4f}, {pf_arr.max():.4f}]")
    print(f"\n   ref nb3200 (deep-30 mean) = {REF_NB3200:.4f} +/- {REF_NB3200_STD:.4f}")
    print(f"   delta vs nb3200           = {mean_pf - REF_NB3200:+.4f}")
    print(f"   ref nb3262 (LGBM K=20)    = {REF_NB3262:.4f}")
    print(f"   ref nb3342 (ExtraTrees)   = {REF_NB3342:.4f}")
    print(f"   ref nb3090 (parent)       = {REF_NB3090:.4f}")
    print(f"   ref nb2171 (anchor-swap)  = {REF_NB2171:.4f}")

    # -- Deploy: refit per-kf_seed on ALL 253, mean-bag mu on te --------------
    print("\n" + "-" * 78)
    print(f"STEP 3: deploy refit on all 253 unblind, mean-bag {len(KF_SEEDS)}"
          f"-seed NGBoost mu, te = clip(nb3200_te + resid_pred_te)")
    print("-" * 78)
    sum_te_resid = np.zeros(n_test, dtype=np.float64)
    for s in KF_SEEDS:
        mdl = NGBRegressor(**_ngb_params(s))
        mdl.fit(X_unb_K20, residual)
        sum_te_resid += np.asarray(mdl.predict(X_te_K20), dtype=np.float64)
    mean_te_resid = sum_te_resid / len(KF_SEEDS)
    te_pred = np.clip(anchor_te + mean_te_resid, CLIP_LO, CLIP_HI).astype(np.float32)
    n_te_lo = int(np.sum(anchor_te + mean_te_resid < CLIP_LO))
    n_te_hi = int(np.sum(anchor_te + mean_te_resid > CLIP_HI))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te_resid mean={mean_te_resid.mean():+.4f}  "
          f"std={mean_te_resid.std():.4f}")
    print(f"   te clipped lo={n_te_lo}/513 hi={n_te_hi}/513")
    print(f"   te(513) final mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
          f"min={te_pred.min():.3f}  max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}  "
          f"(expected NEGATIVE gap vs honest mean -- deploy sees 253 labels)")
    print(f"   gap (in_sample - honest) = {te_unb_in_rae - mean_pf:+.4f}")

    # Median-seed OOF for storage (by perfold mean)
    med_seed_idx = int(np.argsort(pf_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} "
          f"(perfold_mean={pf_arr[med_seed_idx]:.4f})")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: GATE")
    print("-" * 78)
    if mean_pf < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3360 15-seed per-fold-mean "
            f"{mean_pf:.4f} beats BETTER gate {GATE_BETTER:.4f} "
            f"({mean_pf - GATE_BETTER:+.4f}) and nb3200 deep-30 "
            f"{REF_NB3200:.4f} ({mean_pf - REF_NB3200:+.4f}). "
            f"NGBoost natural-gradient probabilistic boosting (mean param mu) "
            f"on (y - nb3200) residual using the SAME 117-col K=20 slice that "
            f"ordinary-gradient LGBM (nb3262) and ExtraTrees (nb3342) failed on "
            f"extracts residual structure the KL-geometry update captures and "
            f"the point-loss gradient missed -- the prior residual-on-anchor "
            f"failures were the OPTIMIZATION GEOMETRY, not the feature capacity. "
            f"PRE-clean anchor chain (nb3090 -> nb3200). Re-verify with deep-30 "
            f"(kf_seeds 30+) before any PRIMARY-1 swap; same under-dispersion-"
            f"risk root as cycle-160."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3360 15-seed per-fold-mean {mean_pf:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f} ({mean_pf - GATE_BETTER:+.4f}). "
            f"Delta vs nb3200 (deep-30 mean {REF_NB3200:.4f}) = "
            f"{mean_pf - REF_NB3200:+.4f}. NGBoost natural-gradient boosting "
            f"(mean param mu) on (y - nb3200) residual using the 117-col K=20 "
            f"slice finds no new structure on top of the learned-clip "
            f"chemprop_aux stack -- swapping the OPTIMIZATION GEOMETRY "
            f"(ordinary-gradient LGBM nb3262 / ExtraTrees nb3342 -> "
            f"natural-gradient NGBoost) does NOT help, confirming the "
            f"bottleneck is the anchor's n=253 capacity on already-absorbed "
            f"K=20 columns, NOT the gradient geometry. Residual-on-residual on "
            f"the nb3200 anchor remains closed across LGBM (nb3262), "
            f"Avalon-LGBM (nb3313), ExtraTrees (nb3342), and NGBoost (nb3360). "
            f"Substrate change must come from a DIFFERENT ANCHOR axis."
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
    print(f"   [save] {te_path}   (deploy mean-bag te, 513,)")

    sub_csv = SUBMISSIONS / f"{TAG}_ngboost_residual.csv"
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
        "parent_tag": PARENT_TAG,
        "method": ("ngboost_natural_gradient_boosting_mean_param_on_y_minus_nb3200_"
                   "residual_with_chemprop_aux_117col_K20_features_plus_clip_"
                   "honest_5fold_scaffold_cv_15_fresh_seeds"),
        "paradigm": ("ngboost_natural_gradient_probabilistic_boosting_mean_param_"
                     "different_geometry_than_ordinary_gradient_lgbm_nb3262_"
                     "extratrees_nb3342"),
        "learner": "NGBRegressor",
        "ngb_dist": "Normal_default",
        "ngb_score": "LogScore_default",
        "ngb_natural_gradient": True,
        "ngb_correction": "mean_param_mu_predict_equals_pred_dist_mean",
        "ngb_n_estimators": NGB_N_ESTIMATORS,
        "ngb_learning_rate": NGB_LEARNING_RATE,
        "clip_lo": CLIP_LO,
        "clip_hi": CLIP_HI,
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
        "anchor_oof_rae": round(rae_anchor, 4),
        "anchor_leak_eq_truth_frac": round(leak_eq, 4),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_min": float(residual.min()),
        "residual_max": float(residual.max()),
        "feat_dim_full": 117,
        "K_residual": 20,
        "K20_idx_in_117col": K20_idx.tolist(),
        "chembl_pool_size": int(chembl_pool_size),
        "ngb_params_seed0": {k: (v if not isinstance(v, np.integer) else int(v))
                             for k, v in _ngb_params(0).items()},
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
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
        "ref_nb3200_deep30_mean": REF_NB3200,
        "ref_nb3200_deep30_std": REF_NB3200_STD,
        "ref_nb3262_lgbm_K20": REF_NB3262,
        "ref_nb3313_lgbm_avalon": REF_NB3313,
        "ref_nb3342_extratrees": REF_NB3342,
        "ref_nb3293_bayesridge": REF_NB3293,
        "ref_nb3090": REF_NB3090,
        "ref_nb2171": REF_NB2171,
        "ref_nb1191": REF_NB1191,
        "ref_chemprop_aux": CHEMPROP_AUX_REF,
        "delta_vs_nb3200_perfold_mean": round(mean_pf - REF_NB3200, 4),
        "delta_vs_nb3262": round(mean_pf - REF_NB3262, 4),
        "delta_vs_nb3342": round(mean_pf - REF_NB3342, 4),
        "delta_vs_anchor_in_sample": round(mean_pf - rae_anchor, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "n_te_clipped_lo": n_te_lo,
        "n_te_clipped_hi": n_te_hi,
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_unb_in_sample_minus_honest_gap": round(te_unb_in_rae - mean_pf, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv) if verdict == "BETTER" else None),
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
    print(f"   per_fold_mean ({n_s} seeds) = {mean_pf:.4f} +/- {std_pf:.4f}")
    print(f"   95% CI                = [{ci_low_pf:.4f}, {ci_high_pf:.4f}]")
    print(f"   delta vs nb3200       = {mean_pf - REF_NB3200:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_per_fold_rae", "std_per_fold_rae",
        "ci95_per_fold_low", "ci95_per_fold_high",
        "delta_vs_nb3200_perfold_mean", "delta_vs_nb3342",
        "anchor_oof_rae", "anchor_leak_eq_truth_frac",
        "te_unb_in_sample_rae", "te_unb_in_sample_minus_honest_gap",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
