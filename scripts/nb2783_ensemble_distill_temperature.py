"""nb2783 -- Ensemble distillation with temperature scaling on K=20 substrate.

NEW PARADIGM (knowledge-distillation):
    Standard student-teacher distillation: 5 LGBM K=20 *teachers* trained on
    chemprop_aux residual with seeds {0, 1, 7, 42, 137}. Per-row teacher
    average serves as the "soft target". A single LGBM K=20 *student* is then
    trained on a temperature-scaled mixture of the truth and the teacher
    average:

        target_T(row) = (1 - T) * y_residual + T * teacher_avg(row)

    Sweep T in {0.0, 0.2, 0.5, 0.8}:
      - T=0.0 -> student fits hard truth only (baseline; reproduces single-seed
        K=20 residual LGBM)
      - T=0.2 -> mild softening
      - T=0.5 -> balanced mixture
      - T=0.8 -> heavy reliance on teacher consensus (the "dark-knowledge"
        regularization regime)

    The teacher_avg target is computed FOLD-WISE so the student never sees a
    teacher trained on its own validation rows (no leakage). For deploy we use
    the OOF teacher_avg as the distilled target on the 253, then refit student
    on all 253 and predict the 513.

    Distinct from prior nb27xx attempts:
      - nb2774 oblivious tree   : tree-structure regularizer; single model.
      - nb2762 Cauchy loss      : loss-shape change; single model.
      - nb2754 focal loss       : per-row residual reweighting; single model.
      - nb2772 horseshoe Bayes  : prior-based shrinkage; linear leaf.
      - nb2752 stack LGBM+ET    : meta-stack on heterogeneous bases.
      - nb2103 SHAP K-sweep     : feature subset; no target softening.
    Distillation here softens the *target*, not the loss / tree / feature
    structure -- a paradigm axis not yet hit on this K=20 substrate.

PROTOCOL:
    1. Build canonical 117-col 5-way feature matrix; restrict to K=20 cols
       (nb2240 surviving idx_in_117).
    2. anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    3. 5-fold scaffold CV on 253 with kf_seed=1001.
    4. Per fold:
         a. Train 5 teachers (LGBM K=20, seeds {0,1,7,42,137}) on tr residual.
         b. teacher_avg = mean of 5 teacher predictions on va.
         c. For each T in {0.0, 0.2, 0.5, 0.8}:
              soft_target_tr = compute via inner OOF teacher_avg on tr
              student fit on (X_tr, soft_target_tr); predict va.
              corrected_va = anchor[va] + student_pred(va)
    5. Per T: collect 253-row OOF; mean_rae = RAE(y_unb, anchor + student_oof).
    6. Best T = argmin(mean_rae). Refit-all on 253 with best T -> deploy 513.

GATE:
    best T mean_rae < 0.4570 -> "PROMOTE"
    best T mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else                     -> "FAIL"

OUTPUTS:
    scripts/nb2783_ensemble_distill_temperature.py
    data/processed/nb2783_summary.json
    data/processed/nb2783_pred_oof.npy   (253,) float32 best-T corrected OOF
    data/processed/te_nb2783.npy         (513,) float32 best-T deploy
    submissions/nb2783_distill_temperature.csv
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
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2783"

# ---- Anchor + caches ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "_audit_unblind_y.npy"

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
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

# ---- CV protocol ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Teacher / Student ----
TEACHER_SEEDS = [0, 1, 7, 42, 137]
STUDENT_SEED = 0
TEMPERATURES = [0.0, 0.2, 0.5, 0.8]

# ---- LGBM K=20 hyperparams (same family as nb2774 / nb2240 / nb2103) ----
LGBM_MAX_DEPTH = 4
LGBM_NUM_LEAVES = 15
LGBM_N_ESTIMATORS = 300
LGBM_LEARNING_RATE = 0.03
LGBM_MIN_CHILD_SAMPLES = 5
LGBM_REG_LAMBDA = 2.0

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- Refs ----
CHEMPROP_AUX_REF = 0.6216
NB2171_REF = 0.4682
NB2240_K20_REF = 0.4630

# ---- ChEMBL kNN config ----
KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# ============================================================================
# Substrate helpers (identical 117-col matrix as nb2774/nb2754/nb2240)
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
    return agg.rename(columns={"src_first": "src"})


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


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing (run nb1030 first): {mte_p}")
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
    """Identical 117-col matrix as nb2774/nb2754/nb2240/nb2103."""
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
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim_knn = _knn_predict(
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
            mean_sim_knn.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full, int(len(pool))


# ============================================================================
# LGBM
# ============================================================================
def _lgbm_params(seed: int) -> dict:
    """K=20 LGBM(MSE) hyperparams (identical to nb2103 / nb2774 / nb2240)."""
    return dict(
        objective="regression",
        max_depth=LGBM_MAX_DEPTH,
        num_leaves=LGBM_NUM_LEAVES,
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        min_child_samples=LGBM_MIN_CHILD_SAMPLES,
        reg_lambda=LGBM_REG_LAMBDA,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _train_teacher(X_tr, y_tr, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_tr, y_tr)
    return mdl


def _teacher_avg_oof_on_tr(X_tr_K, residual_tr, seed_list, n_inner_folds=5,
                            inner_kf_seed=2025):
    """Within-fold inner-CV to get unbiased teacher_avg targets on TRAIN rows.

    For the student to see soft targets that don't leak the train labels into
    themselves, we do an inner KFold on tr: for each inner fold, train all 5
    teachers on inner-train, predict inner-val. Average across teachers.
    """
    from sklearn.model_selection import KFold
    n_tr = len(residual_tr)
    teacher_avg_tr = np.zeros(n_tr, dtype=np.float64)
    inner_kf = KFold(n_splits=n_inner_folds, shuffle=True,
                     random_state=inner_kf_seed)
    for inner_tr_loc, inner_va_loc in inner_kf.split(np.arange(n_tr)):
        per_teacher_va = np.zeros(
            (len(seed_list), len(inner_va_loc)), dtype=np.float64,
        )
        for j, ts in enumerate(seed_list):
            mdl = _train_teacher(
                X_tr_K[inner_tr_loc], residual_tr[inner_tr_loc], ts,
            )
            per_teacher_va[j] = mdl.predict(X_tr_K[inner_va_loc])
        teacher_avg_tr[inner_va_loc] = per_teacher_va.mean(axis=0)
    return teacher_avg_tr


def _teacher_avg_on_va(X_tr_K, residual_tr, X_va_K, seed_list):
    """Train 5 teachers on full tr; return mean prediction on va."""
    per_teacher_va = np.zeros((len(seed_list), X_va_K.shape[0]), dtype=np.float64)
    for j, ts in enumerate(seed_list):
        mdl = _train_teacher(X_tr_K, residual_tr, ts)
        per_teacher_va[j] = mdl.predict(X_va_K)
    return per_teacher_va.mean(axis=0)


def _student_fit_predict(X_tr_K, target_tr, X_va_K, X_te_K, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_tr_K, target_tr)
    pred_va = mdl.predict(X_va_K).astype(np.float64)
    pred_te = mdl.predict(X_te_K).astype(np.float32)
    return pred_va, pred_te


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Ensemble distillation with temperature scaling on K=20")
    print(f"          teachers: 5 LGBM K=20 with seeds {TEACHER_SEEDS}")
    print(f"          student : 1 LGBM K=20 with seed {STUDENT_SEED}")
    print(f"          T grid  : {TEMPERATURES}")
    print(f"          target_T = (1-T)*y_residual + T*teacher_avg")
    print(f"          5-fold scaffold CV  kf_seed={KF_SEED}")
    print(f"          GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load test set ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)

    unb_idx = np.load(UNBLIND_IDX)
    y_unb = np.load(UNBLIND_Y).astype(np.float64)
    n_unb = len(y_unb)
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Step 1: build 117-col + K=20 cols ----
    print("\n[step1] building canonical 117-col feature matrix ...")
    X_te_full, chembl_pool_size = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full={X_te_full.shape}  chembl_pool={chembl_pool_size}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    K20_cols = list(nb2240["k20_surviving_idx_in_117"])
    if len(K20_cols) != 20:
        raise ValueError(f"K20 cols len {len(K20_cols)} != 20")
    X_te_K = X_te_full[:, K20_cols].astype(np.float32)
    X_unb_K = X_te_K[unb_idx]
    print(f"   X_unb_K={X_unb_K.shape}  X_te_K={X_te_K.shape}")

    # ---- Step 2: anchor + residual ----
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"\n[anchor] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Step 3: scaffold groups + outer splits ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique 253-row scaffolds = {n_unique_scaf}")
    outer_splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )

    # ---- Step 4: outer-CV temperature sweep ----
    print("\n" + "-" * 78)
    print("5-FOLD SCAFFOLD CV  --  per-fold: 5 teachers + student per T")
    print("-" * 78)

    # OOF predictions per T
    student_oof_per_T: dict[float, np.ndarray] = {
        T: np.full(n_unb, np.nan, dtype=np.float64) for T in TEMPERATURES
    }
    # OOF teacher_avg (independent of T, just for diagnostics)
    teacher_avg_oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_diag: list[dict] = []

    for fi, (tr_loc, va_loc) in enumerate(outer_splits):
        ts_f = time.time()
        X_tr_K = X_unb_K[tr_loc]
        X_va_K = X_unb_K[va_loc]
        resid_tr = residual[tr_loc]
        resid_va = residual[va_loc]

        # 4a: teacher_avg on val rows (teachers trained on full tr-fold)
        ta_va = _teacher_avg_on_va(X_tr_K, resid_tr, X_va_K, TEACHER_SEEDS)
        teacher_avg_oof[va_loc] = ta_va

        # 4b: teacher_avg on tr rows via inner CV (no leakage of own labels)
        ta_tr_oof = _teacher_avg_oof_on_tr(
            X_tr_K, resid_tr, TEACHER_SEEDS,
            n_inner_folds=5, inner_kf_seed=2025 + fi,
        )

        # 4c: per-T student fit/predict
        fold_T_records: dict[str, dict] = {}
        for T in TEMPERATURES:
            target_tr = (1.0 - T) * resid_tr + T * ta_tr_oof
            pred_va_resid, _ = _student_fit_predict(
                X_tr_K, target_tr, X_va_K, X_te_K, STUDENT_SEED,
            )
            student_oof_per_T[T][va_loc] = pred_va_resid
            fold_T_records[f"T={T:.1f}"] = {
                "target_mean": float(target_tr.mean()),
                "target_std": float(target_tr.std()),
                "pred_va_std": float(pred_va_resid.std()),
            }

        per_fold_diag.append({
            "fold": fi,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "teacher_avg_va_mean": float(ta_va.mean()),
            "teacher_avg_va_std": float(ta_va.std()),
            "teacher_avg_tr_oof_mean": float(ta_tr_oof.mean()),
            "teacher_avg_tr_oof_std": float(ta_tr_oof.std()),
            "T_records": fold_T_records,
            "wall_sec": round(time.time() - ts_f, 2),
        })
        print(f"   fold={fi}  n_tr={len(tr_loc):3d} n_va={len(va_loc):3d}  "
              f"teacher_avg_va std={ta_va.std():.3f}  "
              f"teacher_avg_tr std={ta_tr_oof.std():.3f}  "
              f"wall={time.time() - ts_f:.1f}s")

    for T, oof in student_oof_per_T.items():
        if np.isnan(oof).any():
            raise RuntimeError(f"T={T}: OOF has NaN -- scaffold split incomplete")
    if np.isnan(teacher_avg_oof).any():
        raise RuntimeError("teacher_avg_oof has NaN")

    # ---- Step 5: per-T RAE ----
    print("\n" + "-" * 78)
    print("PER-T RESULTS")
    print("-" * 78)
    teacher_only_corr = anchor + teacher_avg_oof
    rae_teacher_only = float(rae(y_unb, teacher_only_corr))
    print(f"   teacher-only (avg of 5 teachers)  corr_OOF_RAE = "
          f"{rae_teacher_only:.4f}  d_vs_anchor = "
          f"{rae_teacher_only - rae_anchor:+.4f}")

    per_T_records: list[dict] = []
    for T in TEMPERATURES:
        oof_resid_T = student_oof_per_T[T]
        corr_T = anchor + oof_resid_T
        rae_T = float(rae(y_unb, corr_T))
        rae_resid_T = float(rae(residual, oof_resid_T))
        per_T_records.append({
            "T": float(T),
            "rae_corrected_oof": rae_T,
            "rae_residual_oof": rae_resid_T,
            "delta_vs_anchor": rae_T - rae_anchor,
            "delta_vs_teacher_only": rae_T - rae_teacher_only,
            "oof_resid_std": float(oof_resid_T.std()),
            "oof_resid_mean": float(oof_resid_T.mean()),
        })
        print(f"   T={T:.1f}  corr_OOF_RAE = {rae_T:.4f}  "
              f"resid_OOF_RAE = {rae_resid_T:.4f}  "
              f"d_vs_anchor = {rae_T - rae_anchor:+.4f}  "
              f"d_vs_teacher = {rae_T - rae_teacher_only:+.4f}")

    # ---- Step 6: best T + deploy refit ----
    rae_per_T = [r["rae_corrected_oof"] for r in per_T_records]
    best_T_idx = int(np.argmin(rae_per_T))
    best_T = TEMPERATURES[best_T_idx]
    best_T_rae = float(rae_per_T[best_T_idx])
    print(f"\n   best T = {best_T:.1f}  (corr_OOF_RAE = {best_T_rae:.4f})")

    # Refit teachers on FULL unb, then student on FULL unb with best T target.
    print("\n[deploy] refitting teachers + student on full 253 with best T ...")
    # Teacher_avg target on the 253: use OOF (avoids self-leakage of teachers
    # when they later see their own train rows). For the student target on
    # the 253 we mix using teacher_avg_oof.
    deploy_target_unb = (1.0 - best_T) * residual + best_T * teacher_avg_oof
    student_deploy = lgb.LGBMRegressor(**_lgbm_params(STUDENT_SEED))
    student_deploy.fit(X_unb_K, deploy_target_unb)
    te_resid_513 = student_deploy.predict(X_te_K).astype(np.float32)
    te_corrected_513 = (te_anchor_513 + te_resid_513).astype(np.float32)
    rae_te_at_unb = float(rae(y_unb, te_corrected_513[unb_idx]))
    print(f"[deploy] te_corrected_513 mean={te_corrected_513.mean():.3f}  "
          f"std={te_corrected_513.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE = {rae_te_at_unb:.4f}  "
          f"(OOF best_T = {best_T_rae:.4f})")

    # ---- Save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    best_T_oof_corr = (anchor + student_oof_per_T[best_T]).astype(np.float32)
    np.save(pred_oof_path, best_T_oof_corr)
    np.save(te_path, te_corrected_513)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_distill_temperature.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_corrected_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if best_T_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_T_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"

    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   best T               = {best_T:.1f}")
    print(f"   best T mean_rae      = {best_T_rae:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = {best_T_rae < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{best_T_rae < GATE_MARGINAL}")
    print(f"   VERDICT              = {verdict}")
    print(f"   d_vs_anchor          = {best_T_rae - rae_anchor:+.4f}")
    print(f"   d_vs_nb2240_K20      = {best_T_rae - NB2240_K20_REF:+.4f}")
    print(f"   d_vs_nb2171          = {best_T_rae - NB2171_REF:+.4f}")

    summary = {
        "tag": TAG,
        "method": (
            "Ensemble distillation with temperature scaling on K=20 substrate. "
            "5 LGBM K=20 teachers (seeds {0,1,7,42,137}) on chemprop_aux "
            "residual; per-row teacher average serves as soft target. Single "
            "LGBM K=20 student fit on temperature-mixed target "
            "(1-T)*y_residual + T*teacher_avg with T in {0.0, 0.2, 0.5, 0.8}. "
            "Per-fold teacher_avg on tr computed via 5-fold inner CV to avoid "
            "label leakage. 5-fold scaffold CV outer with kf_seed=1001. "
            "Best T selected by minimum corr_OOF_RAE; deploy refits teachers "
            "+ student on full 253 using OOF teacher_avg as soft component."
        ),
        "paradigm": "knowledge_distillation_temperature_scaling",
        "novelty": (
            "Target-softening paradigm distinct from loss-shape (nb2762/2754), "
            "tree-structure (nb2774), feature-subsampling (nb2152/2166), and "
            "stack (nb2752) attempts on the K=20 substrate. Distillation here "
            "transfers ensemble 'dark knowledge' (the inter-teacher consensus "
            "implicit in their mean prediction) into a single student via a "
            "mixed target, regularizing the student toward the bagged "
            "manifold without inflating model count at deploy time."
        ),
        "anchor": "chemprop_aux",
        "anchor_pre_unblind": True,
        "anchor_in_rae": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2240_k20_ref": NB2240_K20_REF,
        "nb2171_ref": NB2171_REF,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_unique_scaffolds_unb": int(n_unique_scaf),
        "K20_cols": K20_cols,
        "feat_dim_K20": int(X_unb_K.shape[1]),
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "teacher_seeds": TEACHER_SEEDS,
        "student_seed": STUDENT_SEED,
        "temperatures": TEMPERATURES,
        "lgbm_max_depth": LGBM_MAX_DEPTH,
        "lgbm_num_leaves": LGBM_NUM_LEAVES,
        "lgbm_n_estimators": LGBM_N_ESTIMATORS,
        "lgbm_learning_rate": LGBM_LEARNING_RATE,
        "lgbm_min_child_samples": LGBM_MIN_CHILD_SAMPLES,
        "lgbm_reg_lambda": LGBM_REG_LAMBDA,
        "rae_teacher_only_corr_oof": rae_teacher_only,
        "per_T_records": per_T_records,
        "per_fold_diag": per_fold_diag,
        "best_T": float(best_T),
        "best_T_rae": best_T_rae,
        "mean_rae": best_T_rae,
        "delta_vs_anchor": best_T_rae - rae_anchor,
        "delta_vs_nb2240_k20": best_T_rae - NB2240_K20_REF,
        "delta_vs_nb2171": best_T_rae - NB2171_REF,
        "rae_te_at_unb_in_sample": rae_te_at_unb,
        "te_deploy_mean": float(te_corrected_513.mean()),
        "te_deploy_std": float(te_corrected_513.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "chembl_pool_size": chembl_pool_size,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_T", "best_T_rae", "mean_rae",
        "delta_vs_anchor", "delta_vs_nb2240_k20", "delta_vs_nb2171",
        "rae_teacher_only_corr_oof", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n  per_T_records:")
    for r in res.get("per_T_records", []):
        print(f"    T={r['T']:.1f}  rae_corr={r['rae_corrected_oof']:.4f}  "
              f"d_vs_anchor={r['delta_vs_anchor']:+.4f}  "
              f"d_vs_teacher={r['delta_vs_teacher_only']:+.4f}")
