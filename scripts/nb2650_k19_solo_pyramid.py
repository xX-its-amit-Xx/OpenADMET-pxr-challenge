"""nb2650 -- K=19 STANDALONE pyramid (cycle 212 nb2631 found K=19 contributed orthogonally).

NEW PARADIGM:
    Cycle 212 nb2631 enumerated equal-weight subsets of K in {17,18,19,20,21}.
    K=19 appeared in 5/5 top combos AND its standalone OOF RAE was 0.4625
    (just behind K=18 at 0.4619, K=21 at 0.4594).  All blends are post-hoc
    averages of K-anchors built with 5 residual seeds each; the dispersion
    structure of K=19 STANDALONE under DEEP-30 (30 residual seeds 0..29) has
    never been measured.

    Cycle 160 + 163 deep-verify rule: 5-seed under-disperses 4x vs 30-seed
    on this substrate; gate decisions MUST use deep-30.  Here we test the
    hypothesis that K=19 standalone with 30 residual seeds achieves a lower
    mean RAE than the 5-seed equal-weight K-blend (nb2604 = 0.4580) because
    bagging variance averages over a larger seed pool.

HYPOTHESIS:
    K=19 has the cleanest standalone OOF in the K-band (per nb2631 per-K
    table).  Going from 5 -> 30 residual seeds compresses the LGBM
    initialization-noise variance ~6x (Central Limit on per-row OOF).  If
    the K=19 feature subset is genuinely informative (not lucky-K), deep-30
    K=19 STANDALONE may break the equal-weight subset blend ceiling
    (nb2604 = 0.4580) without paying df cost.

PROTOCOL:
    1. Reconstruct K=19 surviving feature index set in the 117-col matrix
       (identical to nb2631; via nb2231_summary RFE trajectory):
         K=19 -> [45, 67, 66, 68, 65, 92, 27, 77, 81, 56, 1, 7, 115, 93,
                  80, 11, 70, 54, 8]
    2. Build the 117-col test feature matrix (reuses nb2604/nb2631 helper).
    3. Outer 5-fold SCAFFOLD CV on the 253 unblind rows, kf_seed=1001.
    4. INNER residual LGBM training: at each outer fold, fit on
       (anchor + residual) using RESID_SEEDS = list(range(30))
       (30 seeds 0..29).
       Inside each fold:
         - For each seed s in 0..29:
             mdl = LGBM(**_lgbm_params(s))
             mdl.fit(X_tr_K19, residual_tr)
             pred_va_s = anchor_va + mdl.predict(X_va_K19)
         - mean_bag_va = mean across 30 seeds
         - pred_oof_fold[va_idx] = mean_bag_va
       (each outer fold = 1 deep-30 mean-bag estimate per validation row)
    5. pooled_RAE = rae(y_unb, pred_oof)
    6. te-deploy: refit on ALL 253 with 30 seeds, mean-bag te-residual,
       deploy_te = te_anchor_513 + mean_bag_te_residual.

GATE:
    pooled_RAE < 0.4570 -> "PROMOTE"
    pooled_RAE < 0.4601 -> "MARGINAL_BEAT"
    else                -> "FAIL"

Outputs:
    scripts/nb2650_k19_solo_pyramid.py
    data/processed/nb2650_summary.json
    data/processed/nb2650_pred_oof.npy            (253,) float32
    data/processed/te_nb2650.npy                  (513,) float32
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2650"

# ---- Anchor + residual params (identical hyperparams to nb2604/nb2631) ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
RESID_SEEDS = list(range(30))         # 0..29 -> DEEP-30 mean-bag
N_FOLDS = 5
KF_SEED = 1001                        # single outer scaffold CV seed

# ---- Feature cache paths ----
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
NB2063_SHAP_PATH = DATA_PROCESSED / "nb2063_shap_importance_full117.npy"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

K_TARGET = 19

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# ---- Reference ----
NB2604_REF = 0.4580
NB2631_K19_STANDALONE_5SEED = 0.4625      # per nb2631 per-K table
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# helpers (lifted verbatim from nb2604 / nb2631)
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


def reconstruct_K_from_trajectory(nb2231_sum, K_target):
    """Reconstruct surviving feature indices at K_target from nb2231 RFE trajectory."""
    shap_top28 = list(nb2231_sum["shap_top28_idx_in_117"])
    if K_target == 28:
        return shap_top28
    if K_target > 28:
        if not NB2063_SHAP_PATH.exists():
            raise FileNotFoundError(f"need {NB2063_SHAP_PATH}")
        imp = np.load(NB2063_SHAP_PATH).astype(np.float64)
        order = np.argsort(-imp)
        return [int(j) for j in order[:K_target]]
    current = list(shap_top28)
    traj = nb2231_sum["rfe_trajectory"]
    for entry in traj:
        if entry.get("feat_dropped") is None:
            continue
        if entry["K_after"] < K_target:
            break
        d = int(entry["feat_dropped"])
        if d in current:
            current.remove(d)
        if entry["K_after"] == K_target:
            return current
    if len(current) == K_target:
        return current
    raise ValueError(f"could not reconstruct K={K_target} (got len {len(current)})")


def build_117col_feature_matrix(te_smiles, n_test):
    """Identical 117-col matrix as nb2604/nb2631."""
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
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te_top, X_maccs_te_top, X_mord_te_top, X_emb_te_top, X_av_te_top,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    if X_te_full.shape[1] != 117:
        raise ValueError(f"feat_dim {X_te_full.shape[1]} != 117")
    return X_te_full


def deep30_cv_pyramid(
    X_unb, residual, anchor, splits, seeds,
):
    """Compute deep-30 mean-bag OOF for a standalone K-anchor under outer scaffold CV.

    For each outer scaffold fold (tr, va):
        For each seed s in seeds (length 30):
            mdl = LGBM(seed=s).fit(X_unb[tr], residual[tr])
            pred_va_s = anchor[va] + mdl.predict(X_unb[va])
        mean_bag_va = mean across seeds
        pred_oof[va] = mean_bag_va

    Returns:
        pred_oof:  (n_unb,) float64
        per_fold_rae: list[float] length n_folds
        per_fold_seed_rae: list[list[float]] outer x inner
    """
    n_unb = X_unb.shape[0]
    pred_oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_rae = []
    per_fold_seed_rae = []
    truth = anchor + residual    # = y_unb
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        ts = time.time()
        per_seed_va = np.zeros((len(seeds), len(va_loc)), dtype=np.float64)
        per_seed_rae_fold = []
        for i, s in enumerate(seeds):
            mdl = lgb.LGBMRegressor(**_lgbm_params(int(s)))
            mdl.fit(X_unb[tr_loc], residual[tr_loc])
            pred_va = mdl.predict(X_unb[va_loc])
            per_seed_va[i] = anchor[va_loc] + pred_va
            per_seed_rae_fold.append(
                float(rae(truth[va_loc], anchor[va_loc] + pred_va))
            )
        mean_bag_va = per_seed_va.mean(axis=0)
        pred_oof[va_loc] = mean_bag_va
        rae_va = float(rae(truth[va_loc], mean_bag_va))
        per_fold_rae.append(rae_va)
        per_fold_seed_rae.append(per_seed_rae_fold)
        print(f"   fold {fold_i+1}/{N_FOLDS}: n_tr={len(tr_loc):3d} n_va={len(va_loc):3d}  "
              f"mean_bag_rae={rae_va:.4f}  "
              f"mean_seed_rae={np.mean(per_seed_rae_fold):.4f}  "
              f"std_seed_rae={np.std(per_seed_rae_fold):.4f}  "
              f"wall={time.time()-ts:.1f}s")
    return pred_oof, per_fold_rae, per_fold_seed_rae


def deep30_te_deploy(X_unb, residual, X_te, te_anchor_513, seeds):
    """Refit on ALL 253 with 30 seeds, mean-bag te-residual, deploy_te."""
    per_seed_te = np.zeros((len(seeds), X_te.shape[0]), dtype=np.float64)
    for i, s in enumerate(seeds):
        mdl = lgb.LGBMRegressor(**_lgbm_params(int(s)))
        mdl.fit(X_unb, residual)
        per_seed_te[i] = mdl.predict(X_te)
    mean_bag_te_resid = per_seed_te.mean(axis=0)
    return (te_anchor_513 + mean_bag_te_resid).astype(np.float32)


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K={K_TARGET} STANDALONE pyramid (DEEP-30, 30 residual seeds)")
    print(f"          outer scaffold {N_FOLDS}-fold CV, kf_seed={KF_SEED}")
    print(f"          n_seeds={len(RESID_SEEDS)}  (seeds 0..{RESID_SEEDS[-1]})")
    print(f"          ref nb2604 4-K equal-weight blend = {NB2604_REF:.4f}")
    print(f"          ref nb2631 K=19 5-seed standalone = "
          f"{NB2631_K19_STANDALONE_5SEED:.4f}")
    print(f"          gate: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load truth + anchor + scaffold ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] chemprop_aux te[unb_idx] in_RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")

    # ---- Reconstruct K=19 idx ----
    print("\n" + "-" * 78)
    print(f"STEP 1: reconstruct K={K_TARGET} idx via nb2231 RFE trajectory")
    print("-" * 78)
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    K19_idx_in_117 = reconstruct_K_from_trajectory(nb2231, K_TARGET)
    if len(K19_idx_in_117) != K_TARGET:
        raise ValueError(f"K=19 reconstruction returned len {len(K19_idx_in_117)}")
    print(f"   K={K_TARGET} idx_in_117 (n={len(K19_idx_in_117)}): {K19_idx_in_117}")

    # ---- Build 117-col matrix and slice K=19 ----
    print("\n" + "-" * 78)
    print(f"STEP 2: build 117-col matrix, slice K={K_TARGET} columns")
    print("-" * 78)
    X_te_full = build_117col_feature_matrix(te_smiles, n_test)
    print(f"   X_te_full = {X_te_full.shape}")
    X_te_K19 = X_te_full[:, K19_idx_in_117].astype(np.float32)
    X_unb_K19 = X_te_K19[unb_idx]
    print(f"   X_te_K19 = {X_te_K19.shape}  X_unb_K19 = {X_unb_K19.shape}")

    # ---- Outer scaffold CV ----
    print("\n" + "-" * 78)
    print(f"STEP 3: outer scaffold {N_FOLDS}-fold CV  kf_seed={KF_SEED}  "
          f"inner deep-30 mean-bag residual LGBM")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )
    splits_list = list(splits)
    pred_oof, per_fold_rae, per_fold_seed_rae = deep30_cv_pyramid(
        X_unb_K19, residual, anchor, splits_list, RESID_SEEDS,
    )

    if np.isnan(pred_oof).any():
        raise RuntimeError("pred_oof has NaN -- scaffold splits did not cover all rows")

    pooled_rae = float(rae(y_unb, pred_oof))
    per_fold_mean = float(np.mean(per_fold_rae))
    per_fold_std = float(np.std(per_fold_rae))
    print(f"\n[cv] pooled scaffold-CV RAE   = {pooled_rae:.4f}")
    print(f"[cv] per-fold mean+/-std      = "
          f"{per_fold_mean:.4f} +/- {per_fold_std:.4f}")
    print(f"[cv] per-fold RAE             = "
          f"[{', '.join(f'{r:.4f}' for r in per_fold_rae)}]")

    # ---- per-seed std diagnostic (within-fold dispersion) ----
    seed_rae_flat = [r for fold in per_fold_seed_rae for r in fold]
    print(f"\n[seed-disp] flat seed-RAE  mean = {np.mean(seed_rae_flat):.4f}  "
          f"std = {np.std(seed_rae_flat):.4f}  n = {len(seed_rae_flat)}")

    # ---- Gate ----
    if pooled_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] pooled_RAE={pooled_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL_BEAT)"
          f"  -> {verdict}")

    # ---- Deploy te (refit on ALL 253, 30 seeds mean-bag) ----
    print("\n" + "-" * 78)
    print(f"STEP 4: deploy te -- refit on ALL 253, {len(RESID_SEEDS)}-seed mean-bag")
    print("-" * 78)
    deploy_te = deep30_te_deploy(
        X_unb_K19, residual, X_te_K19, te_anchor_513, RESID_SEEDS,
    )
    te_unb_in = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   deploy_te mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 5: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof.astype(np.float32))
    np.save(te_path, deploy_te.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_k19_solo_pyramid.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_te.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    delta_vs_nb2604 = pooled_rae - NB2604_REF
    delta_vs_nb2631_5seed = pooled_rae - NB2631_K19_STANDALONE_5SEED
    delta_vs_nb2171 = pooled_rae - NB2171_REF

    summary = {
        "tag": TAG,
        "method": "K19_standalone_deep30_pyramid_outer_scaffold_CV",
        "paradigm": "single_K_anchor_no_blend_30_seed_mean_bag",
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "K_target": K_TARGET,
        "K_idx_in_117col": K19_idx_in_117,
        "n_resid_seeds": len(RESID_SEEDS),
        "resid_seeds": RESID_SEEDS,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "pooled_rae": pooled_rae,
        "per_fold_rae": per_fold_rae,
        "per_fold_mean": per_fold_mean,
        "per_fold_std": per_fold_std,
        "per_fold_seed_rae": per_fold_seed_rae,
        "seed_dispersion_mean": float(np.mean(seed_rae_flat)),
        "seed_dispersion_std": float(np.std(seed_rae_flat)),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "delta_vs_nb2604": delta_vs_nb2604,
        "nb2604_ref": NB2604_REF,
        "delta_vs_nb2631_K19_5seed": delta_vs_nb2631_5seed,
        "nb2631_K19_5seed_ref": NB2631_K19_STANDALONE_5SEED,
        "delta_vs_nb2171": delta_vs_nb2171,
        "nb2171_ref": NB2171_REF,
        "te_mean": float(deploy_te.mean()),
        "te_std": float(deploy_te.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K_target                = {K_TARGET}")
    print(f"   n_seeds                 = {len(RESID_SEEDS)}  (0..{RESID_SEEDS[-1]})")
    print(f"   outer kf_seed           = {KF_SEED}  n_folds={N_FOLDS}")
    print(f"   pooled scaffold-CV RAE  = {pooled_rae:.4f}")
    print(f"   per-fold mean+/-std     = {per_fold_mean:.4f} +/- {per_fold_std:.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   delta vs nb2604         = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2631_K19_5seed = {delta_vs_nb2631_5seed:+.4f}")
    print(f"   delta vs nb2171         = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] in-RAE      = {te_unb_in:.4f}")
    print(f"   wall                    = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "K_target",
        "n_resid_seeds",
        "pooled_rae",
        "per_fold_mean",
        "per_fold_std",
        "verdict",
        "delta_vs_nb2604",
        "delta_vs_nb2631_K19_5seed",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
