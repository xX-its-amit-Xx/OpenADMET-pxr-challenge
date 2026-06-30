"""nb2541 -- Per-fold curriculum learning (easy-to-hard LGBM K=20 residual).

NEW PARADIGM: rank fold-train rows by anchor agreement (proxy for
difficulty), then train K=20-substrate LGBM in 3 warm-start stages.

  Per fold-train: difficulty = |chemprop_aux - nb2240_K20_anchor|
                  (low |delta| = easy, high |delta| = hard).

  Stage A : LGBM(n_est=100) on bottom 50% easiest.
  Stage B : warm-start (init_model = stage_A booster), train 200 MORE
            estimators on bottom 75%.
  Stage C : warm-start (init_model = stage_B booster), train 100 MORE
            estimators on 100% of fold-train.
  Predict : stage_C.predict(fold_val) -> residual; corrected = anchor+resid.

This is a fundamentally different orthogonality axis than bag-mean
(nb2103/nb2240/nb2241), AdaBoost reweighting (nb2531), or post-hoc
calibration (nb2534): it shapes the training distribution PROGRESSIVELY
during fit, with easy examples seeding the tree structure and hard ones
refining only after the basin has been established. Lit precedent:
Bengio 2009 curriculum learning, Hacohen-Weinshall 2019.

PROTOCOL:
    1. Rebuild the 117-col 5-way K-tuned feature matrix (AtomPair / MACCS
       / Mordred / ChempropEmbed / Avalon top-K + ChEMBL kNN pred +
       mean_sim) exactly as nb2531/nb2241/nb2523. Slice K=20 cols (first
       20 cols of full matrix) -- NO, use full 117 substrate; the K=20
       label refers to the anchor (nb2240) not feature slice.
    2. Anchor for residual: chemprop_aux (verified-clean PRE-unblind).
    3. Per fold-train: rank by |chemprop_aux - nb2240_anchor| as difficulty.
    4. 3-stage warm-start LGBM: 50% -> 75% -> 100% of fold-train.
    5. 5-fold scaffold CV on 253 unblind x 5 kf_seeds -> mean-bag oof.
    6. Deploy: refit 3-stage curriculum on ALL 253 per seed, predict 513.

GATE (mean-bag RAE):
    < 0.4570  PROMOTE
    < 0.4601  MARGINAL_BEAT
    else      FAIL

OUTPUTS:
    scripts/nb2541_curriculum_per_fold.py
    data/processed/nb2541_summary.json
    data/processed/nb2541_pred_oof.npy   (253,) float32  mean-bag corrected
    data/processed/te_nb2541.npy         (513,) float32  deploy refit
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2541"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ANCHOR_LGBM_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
ANCHOR_LGBM_TE_PATH = DATA_PROCESSED / "te_nb2240.npy"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# Curriculum stage parameters (per task spec)
STAGE_A_N_EST = 100
STAGE_A_FRAC = 0.50
STAGE_B_N_EST = 200
STAGE_B_FRAC = 0.75
STAGE_C_N_EST = 100
STAGE_C_FRAC = 1.00

# Common LGBM regressor params (K=20-family tuning from nb2240 / nb2241)
COMMON_LGBM_PARAMS = dict(
    objective="regression",
    metric="None",
    num_leaves=20,
    max_depth=6,
    learning_rate=0.05,
    min_child_samples=5,
    reg_lambda=1.0,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    verbosity=-1,
)

# Substrate sources (mirror nb2531 / nb2241 / nb2523)
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
NB2241_K20_MEAN_BAG_REF = 0.4763


# -------------------------- helpers (mirror nb2531) --------------------------
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


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
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


# --------------------------- curriculum 3-stage fit --------------------------
def _curriculum_fit(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    difficulty: np.ndarray,
    seed: int,
    tmpdir: Path,
) -> lgb.Booster:
    """Fit 3-stage warm-start LGBM curriculum.

    difficulty: per-row scalar; ascending order => easy -> hard.
    Stage A: bottom 50% easiest.
    Stage B: warm-start, bottom 75%.
    Stage C: warm-start, all 100%.
    Returns final booster (stage C).
    """
    n = len(y_tr)
    if n < 4:
        # tiny fold-train: skip curriculum, fit single stage on everything
        ds = lgb.Dataset(X_tr, label=y_tr)
        params = dict(COMMON_LGBM_PARAMS, seed=seed)
        return lgb.train(params, ds,
                         num_boost_round=STAGE_A_N_EST + STAGE_B_N_EST + STAGE_C_N_EST)

    order = np.argsort(difficulty, kind="stable")  # ascending: easy first
    # Stage A: bottom STAGE_A_FRAC easiest
    nA = max(1, int(np.floor(STAGE_A_FRAC * n)))
    idxA = order[:nA]
    dsA = lgb.Dataset(X_tr[idxA], label=y_tr[idxA])
    paramsA = dict(COMMON_LGBM_PARAMS, seed=seed)
    bstA = lgb.train(paramsA, dsA, num_boost_round=STAGE_A_N_EST)
    pathA = tmpdir / f"stage_A_seed{seed}.txt"
    bstA.save_model(str(pathA))

    # Stage B: warm-start, bottom STAGE_B_FRAC easiest, ADD STAGE_B_N_EST more
    nB = max(nA + 1, int(np.floor(STAGE_B_FRAC * n)))
    idxB = order[:nB]
    dsB = lgb.Dataset(X_tr[idxB], label=y_tr[idxB])
    paramsB = dict(COMMON_LGBM_PARAMS, seed=seed)
    bstB = lgb.train(
        paramsB, dsB,
        num_boost_round=STAGE_B_N_EST,
        init_model=str(pathA),
    )
    pathB = tmpdir / f"stage_B_seed{seed}.txt"
    bstB.save_model(str(pathB))

    # Stage C: warm-start, 100% of fold-train, ADD STAGE_C_N_EST more
    dsC = lgb.Dataset(X_tr, label=y_tr)
    paramsC = dict(COMMON_LGBM_PARAMS, seed=seed)
    bstC = lgb.train(
        paramsC, dsC,
        num_boost_round=STAGE_C_N_EST,
        init_model=str(pathB),
    )
    return bstC


def _curriculum_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    difficulty_unb: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
    tmpdir: Path,
) -> np.ndarray:
    """Scaffold 5-fold CV. Per fold, rank fold-train by difficulty,
    apply 3-stage curriculum, predict fold-val residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    for fi, (tr_loc, va_loc) in enumerate(splits):
        fold_tmpdir = tmpdir / f"seed{kf_seed}_fold{fi}"
        fold_tmpdir.mkdir(parents=True, exist_ok=True)
        bst = _curriculum_fit(
            X[tr_loc], residual[tr_loc],
            difficulty_unb[tr_loc], kf_seed, fold_tmpdir,
        )
        oof[va_loc] = bst.predict(X[va_loc])
    return oof


def _curriculum_deploy_te(
    X_unb: np.ndarray,
    residual_unb: np.ndarray,
    difficulty_unb: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    tmpdir: Path,
) -> np.ndarray:
    """Refit curriculum on all 253 unb features, predict 513 te residual."""
    deploy_tmpdir = tmpdir / f"deploy_seed{seed}"
    deploy_tmpdir.mkdir(parents=True, exist_ok=True)
    bst = _curriculum_fit(
        X_unb, residual_unb, difficulty_unb, seed, deploy_tmpdir,
    )
    return bst.predict(X_te).astype(np.float32)


# ============================================================================
# MAIN
# ============================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-fold curriculum learning (easy-to-hard K=20 LGBM)")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold")
    print(f"        kf_seeds = {KF_SEEDS}")
    print(f"        Stage A: {STAGE_A_N_EST} est on bottom {STAGE_A_FRAC*100:.0f}%")
    print(f"        Stage B: +{STAGE_B_N_EST} est warm-start on bottom "
          f"{STAGE_B_FRAC*100:.0f}%")
    print(f"        Stage C: +{STAGE_C_N_EST} est warm-start on 100%")
    print(f"        difficulty proxy = |chemprop_aux - nb2240_K20|")
    print(f"        GATE: mean_rae < {GATE_PROMOTE} PROMOTE; "
          f"< {GATE_MARGINAL} MARGINAL_BEAT; else FAIL")
    print(f"        ref: chemprop_aux RAE = {CHEMPROP_AUX_REF:.4f}")
    print(f"        ref: nb2241 K=20 mean_bag RAE = "
          f"{NB2241_K20_MEAN_BAG_REF:.4f}")
    print("=" * 78)

    # ---- load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in unb 253 = {n_unique_scaf}")

    # chemprop_aux anchor (verified clean)
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # nb2240 K=20 anchor (for difficulty proxy ONLY -- not used as label)
    if not ANCHOR_LGBM_OOF_PATH.exists():
        raise FileNotFoundError(f"missing K=20 anchor: {ANCHOR_LGBM_OOF_PATH}")
    nb2240_unb = np.load(ANCHOR_LGBM_OOF_PATH).astype(np.float64)
    if nb2240_unb.shape[0] != n_unb:
        raise ValueError(f"nb2240 oof shape {nb2240_unb.shape} != n_unb={n_unb}")
    difficulty_unb = np.abs(anchor - nb2240_unb)
    print(f"[diff] difficulty_unb mean={difficulty_unb.mean():.4f}  "
          f"std={difficulty_unb.std():.4f}  "
          f"min={difficulty_unb.min():.4f}  max={difficulty_unb.max():.4f}")

    # ---- rebuild 117-col matrix on 513 test (then slice unb_idx) ----
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

    # ChEMBL kNN feature
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
    feat_dim = X_te_full.shape[1]
    if feat_dim != 117:
        raise ValueError(f"feat_dim {feat_dim} != 117")
    X_unb_117 = X_te_full[unb_idx]
    print(f"[feat] X_unb_117 = {X_unb_117.shape}  X_te_117 = {X_te_full.shape}")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"CURRICULUM 5-FOLD SCAFFOLD CV  seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed_oof = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    with tempfile.TemporaryDirectory(prefix=f"{TAG}_") as raw_tmp:
        tmpdir = Path(raw_tmp)
        for i, s in enumerate(KF_SEEDS):
            ts = time.time()
            oof_s = _curriculum_cv_one_seed(
                X_unb_117, residual, difficulty_unb,
                unb_scaffolds, s, tmpdir,
            )
            per_seed_oof[i] = oof_s
            per_seed_te_resid[i] = _curriculum_deploy_te(
                X_unb_117, residual, difficulty_unb,
                X_te_full, s, tmpdir,
            )
            r_s = float(rae(y_unb, anchor + oof_s))
            per_seed_rae.append(r_s)
            print(f"   seed={s}  rae_corr={r_s:.4f}  wall={time.time()-ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_oof = per_seed_oof.mean(axis=0)
    median_bag_oof = np.median(per_seed_oof, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor + mean_bag_oof))
    rae_median_bag = float(rae(y_unb, anchor + median_bag_oof))
    print(f"\n[cv] per_seed_mean RAE = {per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag RAE      = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE    = {rae_median_bag:.4f}")
    delta_vs_raw = rae_mean_bag - NB2241_K20_MEAN_BAG_REF
    delta_vs_anchor = rae_mean_bag - rae_anchor
    print(f"[cv] delta vs nb2241 (raw K=20 ref) = {delta_vs_raw:+.4f}")
    print(f"[cv] delta vs chemprop_aux anchor   = {delta_vs_anchor:+.4f}")

    # ---- Deploy te ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = {te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_corrected = (anchor + mean_bag_oof).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    # ---- Gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_rae (mean_bag) = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f} (PROMOTE)        = {rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f} (MARGINAL_BEAT) = {rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "per_fold_curriculum_3stage_lgbm_X117_residual_on_chemprop_aux",
        "anchor": ANCHOR,
        "rae_anchor_chemprop_aux": rae_anchor,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_features": int(X_te_full.shape[1]),
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "difficulty_proxy": "abs(chemprop_aux - nb2240_K20_oof)",
        "difficulty_mean": float(difficulty_unb.mean()),
        "difficulty_std": float(difficulty_unb.std()),
        "stage_params": {
            "stage_A": {"n_est": STAGE_A_N_EST, "frac": STAGE_A_FRAC},
            "stage_B": {"n_est": STAGE_B_N_EST, "frac": STAGE_B_FRAC,
                         "warm_start_from": "stage_A"},
            "stage_C": {"n_est": STAGE_C_N_EST, "frac": STAGE_C_FRAC,
                         "warm_start_from": "stage_B"},
        },
        "lgbm_params": COMMON_LGBM_PARAMS,
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,
        "delta_vs_nb2241_raw_K20_mean_bag": delta_vs_raw,
        "delta_vs_anchor": delta_vs_anchor,
        "compare_nb2241_raw_K20_mean_bag": NB2241_K20_MEAN_BAG_REF,
        "te_unb_in_sample_rae": te_unb_in_sample,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"\nwall = {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_nb2241_raw_K20_mean_bag",
        "delta_vs_anchor",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
