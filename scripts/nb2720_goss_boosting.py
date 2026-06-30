"""nb2720 -- LightGBM GOSS (Gradient-based One-Side Sampling) on K=20 chemprop_aux residual.

NEW PARADIGM:
    GOSS = Gradient-based One-Side Sampling.  At each boosting iteration:
      1. Sort training rows by |gradient| (loss curvature signal).
      2. KEEP the top fraction `top_rate` of high-gradient rows in full.
      3. RANDOMLY sample fraction `other_rate` of the remaining low-gradient
         rows; reweight them by (1 - top_rate) / other_rate to keep the
         gradient sum unbiased.
      4. Train each tree on (top + sampled-low) subset only.

    Different bias-variance regime than gbdt (all rows), dart (random tree
    dropout), or rf (no boosting).  GOSS focuses split capacity on the most
    informative rows -- the high-gradient ones the model is currently getting
    wrong -- without losing the corrective signal from the residual tail.

    Spec (per task):  top_rate=0.2, other_rate=0.1  ==> each tree sees
    ~30% of n_train (in expectation).  At n_train ~ 202 (4/5 of 253), that's
    ~60 rows/tree, of which 40 are highest-gradient and 20 are random.

PROTOCOL:
    1. Anchor chemprop_aux te[unb_idx], residual = y_unb - anchor.
    2. Reuse K=20 RFE-surviving features (nb2231 snapshot K=20) over the
       117-col 5-way feature matrix (AtomPair / MACCS / Mordred / ChempropEmbed
       / Avalon + ChEMBL kNN).
    3. Single configuration:  LGBM(boosting_type='goss', max_depth=4,
       num_leaves=15, n_estimators=300, learning_rate=0.03, top_rate=0.2,
       other_rate=0.1) on the residual.
    4. 5-fold scaffold CV on 253 unblind, 5 kf_seeds {1001..1005}.  Within each
       fold, mean-bag over 5 internal seeds {0,1,7,42,137} to denoise the GOSS
       random low-gradient draw.

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
             < 0.4598 -> "MARGINAL_BEAT"
             else     -> "FAIL"

OUTPUTS:
    scripts/nb2720_goss_boosting.py
    data/processed/nb2720_summary.json
    data/processed/nb2720_pred_oof.npy      (253,) float32
    data/processed/te_nb2720.npy            (513,) float32
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

TAG = "nb2720"

# -------- GOSS parameters (single config per task spec) --------
GOSS_TOP_RATE = 0.2     # fraction of high-gradient rows kept in full
GOSS_OTHER_RATE = 0.1   # fraction of low-gradient rows randomly sampled

# -------- evaluation --------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# -------- reference --------
NB2240_REF_OOF_K20 = 0.4682  # nb2240 K=20 LGBM(MSE) mean-bag, gbdt baseline

# -------- feature build paths (shared with nb2711/nb2702/nb2240/nb2571) --------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
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

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

RESID_SEEDS = [0, 1, 7, 42, 137]  # mean-bag seeds within each scaffold fold


# =============================================================================
# LGBM parameters (GOSS boosting)
# =============================================================================

def _lgbm_params(seed: int) -> dict:
    """K=20 baseline (max_depth=4, num_leaves=15, n_est=300, lr=0.03) with
    boosting_type='goss', top_rate=0.2, other_rate=0.1.

    Notes
    -----
    * `top_rate` and `other_rate` are NOT exposed as sklearn parameters in
      lgb.LGBMRegressor.get_params(), but they are forwarded to the underlying
      booster via **kwargs (verified: booster_.params['top_rate'] == 0.2).
    * GOSS internally requires `subsample`/`bagging_fraction` < 1 to be unset
      (GOSS performs its own row sampling), so we do NOT pass `subsample`.
    * Per-tree row sample = top_rate * n + other_rate * (1 - top_rate) * n
      = 0.2 * n + 0.1 * 0.8 * n = 0.28 * n.  At n_train ~ 202, ~57 rows/tree.
    """
    return dict(
        objective="regression",
        boosting_type="goss",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        top_rate=GOSS_TOP_RATE,
        other_rate=GOSS_OTHER_RATE,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


# =============================================================================
# Helpers (shared feature-build pipeline -- mirror of nb2711/nb2702/nb2240)
# =============================================================================

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


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LightGBM GOSS boosting on K=20 chemprop_aux residual")
    print(f"          boosting_type='goss', top_rate={GOSS_TOP_RATE}, "
          f"other_rate={GOSS_OTHER_RATE}")
    print(f"          gate: PROMOTE < {GATE_PROMOTE}  MARGINAL < {GATE_MARGINAL}")
    print("=" * 78)

    # --- load 253 unblind labels, anchor, scaffolds ---
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[load] n_test={n_test}  n_unb={n_unb}  "
          f"chemprop_aux in_RAE={rae_anchor:.4f}")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.3f}  max={residual.max():+.3f}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # --- load K=20 RFE-surviving feature indices ---
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20, f"expected 20 features, got {len(surviving_K20)}"
    print(f"[feat] K=20 surviving features loaded from nb2231")

    # --- rebuild 117-col 5-way feature matrix (shared with nb2240/nb2702/nb2711) ---
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

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

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
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"feat_dim {X_te_full.shape[1]} != 117"
    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    print(f"[feat] X_unb_K20 = {X_unb_K20.shape}  X_te_K20 = {X_te_K20.shape}")

    # =========================================================================
    # GOSS CV: 5-fold scaffold across 5 kf_seeds, mean-bag 5 internal seeds
    # =========================================================================
    print("\n" + "-" * 78)
    print(f"GOSS CV  kf_seeds={KF_SEEDS}  bag_seeds={RESID_SEEDS}")
    print(f"          top_rate={GOSS_TOP_RATE}  other_rate={GOSS_OTHER_RATE}")
    print(f"          per-tree row-fraction ~ {GOSS_TOP_RATE + GOSS_OTHER_RATE * (1 - GOSS_TOP_RATE):.3f}")
    print(f"target: residual = y_unb - chemprop_aux  (LGBM GOSS, K=20)")
    print("-" * 78)

    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            bag_resid = np.zeros(len(va_loc), dtype=np.float64)
            for bs in RESID_SEEDS:
                mdl = lgb.LGBMRegressor(**_lgbm_params(bs))
                mdl.fit(X_unb_K20[tr_loc], residual[tr_loc])
                bag_resid += mdl.predict(X_unb_K20[va_loc])
            bag_resid /= len(RESID_SEEDS)
            oof_resid[va_loc] = bag_resid
        assert not np.isnan(oof_resid).any(), "oof_resid has NaN -- fold cover incomplete"
        pred_corr = anchor + oof_resid
        pooled = float(rae(y_unb, pred_corr))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "resid_oof_std": float(oof_resid.std()),
            "resid_oof_mean": float(oof_resid.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        all_oofs.append(pred_corr)
        print(f"   kf_seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"resid_std={oof_resid.std():.3f}  "
              f"wall={time.time() - ts:.1f}s")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    rae_per_seed = np.array([r["pooled_rae"] for r in per_seed])
    mean_rae = float(rae_per_seed.mean())
    std_rae = float(rae_per_seed.std())
    rae_of_mean_oof = float(rae(y_unb, mean_oof))

    print("\n" + "=" * 78)
    print(f"GOSS  mean_rae = {mean_rae:.4f} (+/- {std_rae:.4f})")
    print(f"      rae_of_mean_of_seed_oofs = {rae_of_mean_oof:.4f}")
    print("=" * 78)

    # --- gate ---
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] mean_rae={mean_rae:.4f}  "
          f"PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}  -> {verdict}")

    # =========================================================================
    # Deploy: refit on all 253, mean-bag predict on 513 test
    # =========================================================================
    print("\n" + "-" * 78)
    print(f"DEPLOY  refit on 253, mean-bag predict 513 (GOSS, same params)")
    print("-" * 78)
    te_bag_resid = np.zeros(n_test, dtype=np.float64)
    for bs in RESID_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(bs))
        mdl.fit(X_unb_K20, residual)
        te_bag_resid += mdl.predict(X_te_K20)
    te_bag_resid /= len(RESID_SEEDS)
    deploy_te = (te_anchor_513 + te_bag_resid).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   deploy te range = [{deploy_te.min():.3f}, {deploy_te.max():.3f}]  "
          f"mean={deploy_te.mean():.3f} std={deploy_te.std():.3f}")
    print(f"   te[unb_idx] RAE = {te_unb_rae:.4f}  "
          f"(in-sample, expected lower than pred_oof)")

    # --- save artifacts ---
    pred_oof = mean_oof.astype(np.float32)
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, pred_oof)
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_npy_path}")

    summary = {
        "tag": TAG,
        "method": (
            "GOSS_boosting_LGBM_K20_chemprop_aux_residual_scaffold_CV_5seed_mean_bag"
        ),
        "paradigm": "gradient_based_one_side_sampling",
        "boosting_type": "goss",
        "goss_top_rate": GOSS_TOP_RATE,
        "goss_other_rate": GOSS_OTHER_RATE,
        "per_tree_row_fraction_expected":
            GOSS_TOP_RATE + GOSS_OTHER_RATE * (1 - GOSS_TOP_RATE),
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "anchor": "chemprop_aux",
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names": surviving_K20_names,
        "k20_family_counts": dict(nb2231["snapshots"]["20"]["family_counts"]),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "bag_seeds": RESID_SEEDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_results": per_seed,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_of_seed_oofs": rae_of_mean_oof,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "compare_nb2240_K20_ref": NB2240_REF_OOF_K20,
        "delta_vs_nb2240_K20": mean_rae - NB2240_REF_OOF_K20,
        "delta_vs_anchor": mean_rae - rae_anchor,
        "deploy_te_min": float(deploy_te.min()),
        "deploy_te_max": float(deploy_te.max()),
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_unb_rae_in_sample": te_unb_rae,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_npy_path),
        "pre_unblind_clean": True,  # chemprop_aux anchor only, no POST-unblind te
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   boosting_type        = goss")
    print(f"   top_rate / other_rate = {GOSS_TOP_RATE} / {GOSS_OTHER_RATE}")
    print(f"   mean_rae             = {mean_rae:.4f} (+/- {std_rae:.4f})")
    print(f"   verdict              = {verdict}")
    print(f"   delta vs nb2240 K=20 = {mean_rae - NB2240_REF_OOF_K20:+.4f}")
    print(f"   delta vs anchor      = {mean_rae - rae_anchor:+.4f}")
    print(f"   wall                 = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "boosting_type",
        "goss_top_rate",
        "goss_other_rate",
        "mean_rae",
        "std_rae",
        "verdict",
        "delta_vs_nb2240_K20",
        "delta_vs_anchor",
        "deploy_te_min",
        "deploy_te_max",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
