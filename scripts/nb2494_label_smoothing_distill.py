"""nb2494 -- Label-smoothing distillation through soft pEC50 targets.

DIFFERENT axis vs every prior cycle 134+ post-hoc attempt:
    - Prior cycles: chemprop_aux residual = (y_unb - anchor) trained on
      LGBM K=20.  Truth target is hard.
    - HERE: train K=20 student on a SOFT target that is a convex
      blend of truth and teacher prediction.
        y_soft = alpha * y_unb + (1 - alpha) * teacher_pred
      where teacher = nb2240_mean_bag_oof_K20.npy on 253.
      Sweep alpha in {0.5, 0.7, 0.8, 0.9, 1.0}.

    The idea: teacher mean-bag is variance-compressed (std 0.884 vs
    truth std 1.032 on the 253), and a smoother target may regularise
    the K=20 student vs the truth-noise floor (median SE 0.15) AND
    decouple the student from in-sample over-fit to the 253 labels.
    Distillation under MSE on a soft target is equivalent to a
    smoothing prior; the question is whether the optimal trade-off
    lies inside (0, 1) -- if alpha=1.0 wins, smoothing buys nothing.

    Student is the SAME K=20 LGBM-on-residual architecture as nb2240
    (mean-bag over 5 seeds {0,1,7,42,137}, KFold(5) cross-fit, lgbm
    {depth=4 leaves=15 ne=300 lr=0.03 minc=5 reg=2}).  Residual is
    re-defined per alpha: r = y_soft - anchor.  Cross-fit oof and
    deploy te computed for each alpha; ranked by mean_rae across 5
    kf_seeds {1001..1005} of 5-fold scaffold CV on the 253.

PROTOCOL (exact):
    1. Load teacher = nb2240_mean_bag_oof_K20.npy on 253.
    2. Build K=20 feature matrix on 513 (slice from 117-col 5-way)
       via same loader as nb2240 (AtomPair/MACCS/Mordred/Embed/Avalon
       + ChEMBL kNN), then slice via nb2231 K=20 surviving indices.
    3. For each alpha in {0.5, 0.7, 0.8, 0.9, 1.0}:
         a. y_soft = alpha * y_unb + (1-alpha) * teacher
         b. residual_soft = y_soft - anchor
         c. For each of 5 kf_seeds (NOT the LGBM seeds): run scaffold
            5-fold CV.  In each fold (train_idx, val_idx) of 253:
              - For each of 5 LGBM seeds:
                  fit LGBM on (X[train_idx], residual_soft[train_idx])
                  predict X[val_idx] -> per-seed val residual
              - Average across LGBM seeds -> mean-bag val residual
              - val_pred = anchor[val_idx] + mean-bag-val-resid
            Pool val_preds -> pooled scaffold-CV RAE for this kf_seed.
         d. Mean RAE across 5 kf_seeds = alpha mean_rae.
    4. Pick best alpha by min mean_rae across the 5-seed bag.
    5. Build deploy artefact for best alpha: refit on ALL 253 with
       (residual = y_soft - anchor), mean-bag 5 LGBM seeds, predict
       on 513 -> te_nb2494.npy.  pred_oof = best alpha's mean OOF
       over the 5 kf_seeds (averaged) on the 253.

GATE:
    best alpha mean_rae < 0.4570  -> "PROMOTE"
    best alpha mean_rae < 0.4601  -> "MARGINAL_BEAT"
    else                          -> "FAIL"

Outputs:
    data/processed/nb2494_summary.json
    data/processed/nb2494_pred_oof.npy     (253,) float32  best alpha
    data/processed/te_nb2494.npy           (513,) float32  best alpha
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2494"

# ------------------------------ paths --------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
TEACHER_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"

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
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

# ------------------------------ knobs --------------------------------
ALPHAS = [0.5, 0.7, 0.8, 0.9, 1.0]
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
LGBM_SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601


# ============================================================================
# feature build helpers (copied from nb2240, K=20 slice)
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


def _lgbm_params(seed):
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


def build_X_te_K20(n_test, te_smiles):
    """Rebuild the 117-col 5-way te matrix then slice to K=20 surviving."""
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY, NB2231_SUMMARY):
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
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)

    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20

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

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN
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
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te,
            X_maccs_te,
            X_mord_te,
            X_emb_te,
            X_av_te,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    return X_te_K20, surviving_K20, surviving_K20_names


# ============================================================================
# distillation evaluation
# ============================================================================

def scaffold_cv_one_alpha(X_unb, X_te, anchor_unb, anchor_te, y_unb, teacher,
                          unb_scaffolds, alpha, kf_seeds, lgbm_seeds, n_folds):
    """Returns per_kf_seed RAE list + mean OOF pred (anchor + mean-bag resid)."""
    n_unb = X_unb.shape[0]
    y_soft = alpha * y_unb + (1.0 - alpha) * teacher  # 253-long
    residual_soft = y_soft - anchor_unb              # 253-long

    per_seed_raes = []
    seed_oof_stack = []  # 5 oofs of length n_unb
    for kf_seed in kf_seeds:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
        )
        oof_pred = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            # mean-bag residual on val
            val_resid_bag = np.zeros(len(va_loc), dtype=np.float64)
            for ls in lgbm_seeds:
                mdl = lgb.LGBMRegressor(**_lgbm_params(ls))
                mdl.fit(X_unb[tr_loc], residual_soft[tr_loc])
                val_resid_bag += mdl.predict(X_unb[va_loc])
            val_resid_bag /= len(lgbm_seeds)
            oof_pred[va_loc] = anchor_unb[va_loc] + val_resid_bag
        per_seed_raes.append(float(rae(y_unb, oof_pred)))
        seed_oof_stack.append(oof_pred)
    seed_oof_stack = np.column_stack(seed_oof_stack)  # n_unb x 5
    mean_oof = seed_oof_stack.mean(axis=1)
    return per_seed_raes, mean_oof


def deploy_te_one_alpha(X_unb, X_te, anchor_unb, anchor_te, y_unb, teacher,
                        alpha, lgbm_seeds):
    """Refit on full 253 with soft target, mean-bag 5 lgbm seeds, predict 513."""
    y_soft = alpha * y_unb + (1.0 - alpha) * teacher
    residual_soft = y_soft - anchor_unb
    n_te = X_te.shape[0]
    te_resid_bag = np.zeros(n_te, dtype=np.float64)
    for ls in lgbm_seeds:
        mdl = lgb.LGBMRegressor(**_lgbm_params(ls))
        mdl.fit(X_unb, residual_soft)
        te_resid_bag += mdl.predict(X_te)
    te_resid_bag /= len(lgbm_seeds)
    return (anchor_te + te_resid_bag).astype(np.float32)


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- label-smoothing distillation, alpha sweep on K=20 student")
    print("=" * 78)

    # ---- truth + anchor + teacher ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f}")

    teacher = np.load(TEACHER_OOF_PATH).astype(np.float64)
    assert teacher.shape == (n_unb,)
    rae_teacher = float(rae(y_unb, teacher))
    pear_t = float(np.corrcoef(teacher, y_unb)[0, 1])
    print(f"[load] teacher nb2240_K20 in_RAE = {rae_teacher:.4f}  pearson_y = {pear_t:.4f}")
    print(f"[load] teacher mean={teacher.mean():.3f}  std={teacher.std():.3f}  "
          f"y_unb mean={y_unb.mean():.3f}  std={y_unb.std():.3f}")

    # ---- build K=20 feature matrix ----
    print("\n[feat] building 117-col 5-way te matrix + K=20 slice...")
    t_feat = time.time()
    X_te_K20, surviving_K20, surviving_names = build_X_te_K20(n_test, te_smiles)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}  "
          f"wall={time.time()-t_feat:.1f}s")

    anchor_te = te_anchor_513

    # ---- alpha sweep ----
    print("\n" + "-" * 78)
    print(f"ALPHA SWEEP  alphas={ALPHAS}  kf_seeds={KF_SEEDS}  lgbm_seeds={LGBM_SEEDS}")
    print("-" * 78)
    per_alpha_records = []
    per_alpha_mean_oof = {}
    for alpha in ALPHAS:
        t_a = time.time()
        per_seed_raes, mean_oof = scaffold_cv_one_alpha(
            X_unb_K20, X_te_K20, anchor_unb, anchor_te, y_unb, teacher,
            unb_scaffolds, alpha, KF_SEEDS, LGBM_SEEDS, N_FOLDS,
        )
        mean_rae = float(np.mean(per_seed_raes))
        std_rae = float(np.std(per_seed_raes))
        rae_mean_oof = float(rae(y_unb, mean_oof))
        per_alpha_records.append({
            "alpha": float(alpha),
            "per_kf_seed_rae": [float(x) for x in per_seed_raes],
            "mean_rae": mean_rae,
            "std_rae": std_rae,
            "rae_of_mean_oof": rae_mean_oof,
        })
        per_alpha_mean_oof[float(alpha)] = mean_oof.astype(np.float64)
        print(f"   alpha={alpha:.2f}  mean_rae={mean_rae:.4f} +/- {std_rae:.4f}  "
              f"rae_mean_oof={rae_mean_oof:.4f}  wall={time.time()-t_a:.1f}s")

    # ---- pick best alpha ----
    per_alpha_records.sort(key=lambda r: r["mean_rae"])
    best = per_alpha_records[0]
    best_alpha = float(best["alpha"])
    best_mean_rae = float(best["mean_rae"])
    print(f"\n[best] alpha={best_alpha:.2f}  mean_rae={best_mean_rae:.4f}")
    print(f"[ref]  teacher_in_RAE={rae_teacher:.4f}  "
          f"anchor_in_RAE={rae_anchor:.4f}  alpha=1.0_mean_rae="
          f"{next(r['mean_rae'] for r in per_alpha_records if r['alpha']==1.0):.4f}")

    # ---- gate ----
    if best_mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] best_mean_rae={best_mean_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE}/<{GATE_MARGINAL})  verdict={verdict}")

    # ---- deploy te for best alpha ----
    print("\n[deploy] refitting full-253 mean-bag student on best alpha...")
    te_best = deploy_te_one_alpha(
        X_unb_K20, X_te_K20, anchor_unb, anchor_te, y_unb, teacher,
        best_alpha, LGBM_SEEDS,
    )
    te_unb_rae = float(rae(y_unb, te_best[unb_idx]))
    print(f"[deploy] te_unb_rae(in-sample)={te_unb_rae:.4f}  "
          f"te(513) mean={te_best.mean():.3f}  std={te_best.std():.3f}")

    pred_oof_best = per_alpha_mean_oof[best_alpha].astype(np.float32)
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, pred_oof_best)
    np.save(te_path, te_best)
    print(f"[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "label_smoothing_distill_K20_student_chemprop_aux_anchor",
        "teacher_path": str(TEACHER_OOF_PATH),
        "teacher_in_rae_253": rae_teacher,
        "teacher_pearson_y_253": pear_t,
        "teacher_mean": float(teacher.mean()),
        "teacher_std": float(teacher.std()),
        "anchor_path": str(ANCHOR_TE_PATH),
        "anchor_in_rae_253": rae_anchor,
        "anchor_pre_unblind": True,
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names": surviving_names,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "alphas": ALPHAS,
        "kf_seeds": KF_SEEDS,
        "lgbm_seeds": LGBM_SEEDS,
        "n_folds": N_FOLDS,
        "per_alpha_results": per_alpha_records,
        "best_alpha": best_alpha,
        "best_mean_rae": best_mean_rae,
        "best_std_rae": float(best["std_rae"]),
        "best_rae_of_mean_oof": float(best["rae_of_mean_oof"]),
        "gate_promote_below": GATE_PROMOTE,
        "gate_marginal_below": GATE_MARGINAL,
        "verdict": verdict,
        "te_unb_rae_in_sample": te_unb_rae,
        "te_mean": float(te_best.mean()),
        "te_std": float(te_best.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   teacher in_RAE                 = {rae_teacher:.4f}")
    print(f"   anchor (chemprop_aux) in_RAE   = {rae_anchor:.4f}")
    for r in sorted(per_alpha_records, key=lambda x: x["alpha"]):
        print(f"   alpha={r['alpha']:.2f}  mean_rae={r['mean_rae']:.4f} +/- "
              f"{r['std_rae']:.4f}")
    print(f"   BEST alpha                     = {best_alpha:.2f}")
    print(f"   BEST mean_rae                  = {best_mean_rae:.4f}")
    print(f"   gate thresholds                = <{GATE_PROMOTE} PROMOTE | "
          f"<{GATE_MARGINAL} MARGINAL")
    print(f"   verdict                        = {verdict}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "teacher_in_rae_253",
        "anchor_in_rae_253",
        "best_alpha",
        "best_mean_rae",
        "best_std_rae",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
