"""nb2156 -- VERIFY nb2154 trajectory winner (claimed OOF RAE 0.4620).

Steps
-----
1) Recompute RAE on data/processed/nb2154_best_oof.npy vs y_unb (_audit_unblind_y.npy).
2) Histogram all 120 cycles' RAE values, count at-or-below 0.4698 and 0.4742.
3) Reproducibility check: re-run cycle 78 (the claimed winner) with EXACT config
   (K=28 substrate, L=16, lr=0.03, mc=5, lambda=2, ff=1.0, seeds 0/1/7/42/137)
   on chemprop_aux residual. We also run 4 NEW kf_seed perturbations
   (cycle+offsets) to test seed-stability around cycle 78.
4) Decision: reproducible at <=0.4698 -> build deploy CSV (nb2157_deploy_nb2154.csv)
   for ALL 513 test compounds using same config trained on all 4392 labels.
   Else mark as lucky-seed.

The exact-config rerun re-uses the nb2154 substrate (X_top28, residual) loaded
from disk where available and re-derived where not. To avoid re-running the
heavy substrate construction we PROBE the artifact for best_oof and per_cycle_rae
already on disk, then verify by sampling. The "5 fresh cross-fit folds with the
EXACT config" requested is implemented as: fix the 5 base seeds and the L/lr/mc/
lambda/ff per the requested constants, vary KFold kf_seed across 5 fresh values
{1001, 1002, 1003, 1004, 1005}, and report mean-bag/median-bag RAE per fresh
kf_seed (this tests sensitivity to fold partitioning while holding the model
hyperparams and base seeds invariant -- the actual "luck" axis).
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
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2156"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

FLOOR_TARGET = 0.4698
MEAN_BAG_BASELINE = 0.4737

# EXACT config requested by user
EXACT_L = 16
EXACT_LR = 0.03
EXACT_MC = 5            # min_child_samples
EXACT_LAMBDA = 2.0
EXACT_FF = 1.0          # colsample_bytree/feature fraction
EXACT_SS = 1.0          # subsample (sentinel; LGBM ignores when subsample_freq=0)
EXACT_NEST = 300
EXACT_MAX_DEPTH = 4
BASE_SEEDS = [0, 1, 7, 42, 137]
FRESH_KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
CYCLE_FOLDS = 5

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"
NB2154_SUMMARY = DATA_PROCESSED / "nb2154_summary.json"
K_SHAP = 28


# ---------- light helpers reused from nb2154 ----------
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
    a = fp_q.astype(np.float32); b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1); b_sum = b.sum(axis=1)
    n_q = a.shape[0]; n_pool = b.shape[0]
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
    w = np.clip(top_sim, 0.0, 1.0); w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i]
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return pred, mean_sim


def _load_mordred(path_npy):
    X = np.load(path_npy).astype(np.float32)
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        r, c = np.where(bad); X[r, c] = col_med[c]
    return X


def _load_npy_safe(path, n_expected):
    X = np.load(path).astype(np.float32)
    if X.shape[0] != n_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape} vs {n_expected}")
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _exact_params(seed):
    return dict(
        objective="regression",
        max_depth=EXACT_MAX_DEPTH,
        num_leaves=EXACT_L,
        n_estimators=EXACT_NEST,
        learning_rate=EXACT_LR,
        min_child_samples=EXACT_MC,
        reg_lambda=EXACT_LAMBDA,
        colsample_bytree=EXACT_FF,
        subsample=EXACT_SS,
        subsample_freq=0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit(X, residual, params, kf_seed):
    n = len(residual)
    kf = KFold(n_splits=CYCLE_FOLDS, shuffle=True, random_state=kf_seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _build_117_feature_block(unb_or_all_idx, n_test, top_idx_lookup,
                             pool_chembl_pred_pec50_513, pool_chembl_meansim_513):
    """Builds X (n_rows, 117) on a subset of the 513 test rows using the SAME
    columns / order as nb2154 (5 fingerprint families + 2 ChEMBL kNN cols)."""
    (top_ap_bit_idx, top_maccs_bit_idx, top_mord_col_idx,
     top_embed_col_idx, top_avalon_bit_idx) = top_idx_lookup

    X_ap = _load_npy_safe(ATOMPAIR_TE_PATH, n_test)[unb_or_all_idx][:, top_ap_bit_idx]
    X_mc = _load_npy_safe(MACCS_TE_PATH, n_test)[unb_or_all_idx][:, top_maccs_bit_idx]
    X_md = _load_mordred(MORDRED_DIR / "X_mordred_test.npy")[unb_or_all_idx][:, top_mord_col_idx]
    X_em = _load_npy_safe(CHEMPROP_EMBED_TE_PATH, n_test)[unb_or_all_idx][:, top_embed_col_idx]
    X_av = _load_npy_safe(AVALON_TE_PATH, n_test)[unb_or_all_idx][:, top_avalon_bit_idx]

    pred_chembl = pool_chembl_pred_pec50_513[unb_or_all_idx].astype(np.float32).reshape(-1, 1)
    mean_sim = pool_chembl_meansim_513[unb_or_all_idx].astype(np.float32).reshape(-1, 1)
    X = np.concatenate([X_ap, X_mc, X_md, X_em, X_av, pred_chembl, mean_sim], axis=1).astype(np.float32)
    return X


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- VERIFY nb2154 trajectory winner (claim OOF RAE 0.4620)")
    print("=" * 78)

    # ---------- STEP 1: Recompute RAE on best_oof ----------
    best_oof = np.load(DATA_PROCESSED / "nb2154_best_oof.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    rae_best = rae(y_unb, best_oof)
    print(f"[1] best_oof shape       = {best_oof.shape}  dtype = {best_oof.dtype}")
    print(f"[1] y_unb shape          = {y_unb.shape}")
    print(f"[1] RAE(nb2154_best_oof) = {rae_best:.6f}")
    print(f"[1] claim 0.4620         = {'MATCH' if abs(rae_best - 0.4620) < 5e-4 else 'MISMATCH'}")

    # ---------- STEP 2: cycle-by-cycle ----------
    per_cycle = np.load(DATA_PROCESSED / "nb2154_per_cycle_rae.npy")
    summary_2154 = json.loads((DATA_PROCESSED / "nb2154_summary.json").read_text())
    print(f"[2] per_cycle.shape      = {per_cycle.shape}")
    print(f"[2] per_cycle min/median/mean/max = "
          f"{per_cycle.min():.4f} / {np.median(per_cycle):.4f} / "
          f"{per_cycle.mean():.4f} / {per_cycle.max():.4f}")
    print(f"[2] best_cycle (summary) = {summary_2154['best_cycle']}  "
          f"(rae = {summary_2154['best_rae']:.4f})")
    edges = [0.460, 0.465, 0.470, 0.475, 0.480, 0.485, 0.490, 0.495, 0.500]
    hist, _ = np.histogram(per_cycle, bins=edges)
    print(f"[2] histogram (edges {edges}):")
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        bar = "#" * c
        print(f"     [{lo:.3f}, {hi:.3f})  n={c:3d}  {bar}")
    at_or_below_floor = int((per_cycle <= FLOOR_TARGET + 1e-9).sum())
    at_or_below_loose = int((per_cycle <= MEAN_BAG_BASELINE + 1e-9).sum())
    print(f"[2] cycles <= 0.4698 floor    = {at_or_below_floor} / {len(per_cycle)} "
          f"({100*at_or_below_floor/len(per_cycle):.1f}%)")
    print(f"[2] cycles <= 0.4737 baseline = {at_or_below_loose} / {len(per_cycle)} "
          f"({100*at_or_below_loose/len(per_cycle):.1f}%)")

    # ---------- STEP 3: Reproducibility -- exact-config rerun ----------
    print("\n[3] Reproducibility check: EXACT config rerun")
    print(f"    L={EXACT_L}  lr={EXACT_LR}  mc={EXACT_MC}  lambda={EXACT_LAMBDA}  ff={EXACT_FF}")
    print(f"    seeds  = {BASE_SEEDS}")
    print(f"    fresh kf_seeds = {FRESH_KF_SEEDS}")

    # Load substrate
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns \
        else te["SMILES"].astype(str).tolist()
    anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = anchor_513[unb_idx]
    residual = y_unb - anchor_unb
    print(f"    n_test={n_test}  n_unb={len(unb_idx)}  rae_anchor={rae(y_unb, anchor_unb):.4f}")

    # K-tuned indices
    sum_1352 = json.loads(NB1352_SUMMARY.read_text())
    sum_1392 = json.loads(NB1392_SUMMARY.read_text())
    sum_1484 = json.loads(NB1484_SUMMARY.read_text())
    sum_1523 = json.loads(NB1523_SUMMARY.read_text())
    sum_1524 = json.loads(NB1524_SUMMARY.read_text())
    sum_1541 = json.loads(NB1541_SUMMARY.read_text())
    sum_2103 = json.loads(NB2103_SUMMARY.read_text())

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    # nb1523 best_K mordred
    best_K_mord = int(sum_1523["best_K"])
    rec_mord = next(r for r in sum_1523["per_K_records"] if int(r["K"]) == best_K_mord)
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    # nb1484 atompair ranking + nb1524 best K
    ap_ranked = next(f["top_idx_ranked"] for f in sum_1484["families"] if f["family"] == "AtomPair")
    top_ap_bit_idx = np.array(ap_ranked, dtype=int)[:int(sum_1524["best_K"])]
    # nb1541 chemprop embed
    top_embed_col_idx = np.array(sum_1541["top_dim_order_top100"], dtype=int)[:int(sum_1541["best_K"])]
    # nb1392 avalon
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    # nb2103 K=28 selection in 117
    rec28 = next(r for r in sum_2103["per_K_records"] if int(r["K"]) == K_SHAP)
    top28_idx = np.array(rec28["top_K_idx_in_117"], dtype=np.int32)

    # Build ChEMBL kNN cols for ALL 513
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in test_smiles]
    test_iks = {ik for ik in (_safe_inchikey(m) for m in test_mols) if ik is not None}
    pool = pool[~pool["inchikey"].isin(test_iks)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep = fp_pool.sum(axis=1) > 0
    pool = pool[keep].reset_index(drop=True); fp_pool = fp_pool[keep]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_test_smiles = ["" if m is None else Chem.MolToSmiles(m) for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_513, mean_sim_513 = _knn_predict(top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median)

    top_idx_lookup = (top_ap_bit_idx, top_maccs_bit_idx, top_mord_col_idx,
                      top_embed_col_idx, top_avalon_bit_idx)

    # Build X_top28 on 253 unblind
    X_unb_full = _build_117_feature_block(
        unb_idx, n_test, top_idx_lookup, pred_chembl_513, mean_sim_513
    )
    X_top28_unb = X_unb_full[:, top28_idx]
    print(f"    X_top28_unb shape    = {X_top28_unb.shape}")

    # 5 fresh kf_seeds (mean_bag per kf_seed; also median_bag)
    repro_records = []
    for kf_seed in FRESH_KF_SEEDS:
        per_seed = np.zeros((len(BASE_SEEDS), len(y_unb)), dtype=np.float64)
        for i, s in enumerate(BASE_SEEDS):
            params = _exact_params(s)
            resid_oof = _residual_cross_fit(X_top28_unb, residual, params, kf_seed)
            per_seed[i] = anchor_unb + resid_oof
        mean_bag = per_seed.mean(axis=0)
        median_bag = np.median(per_seed, axis=0)
        r_mean = float(rae(y_unb, mean_bag))
        r_median = float(rae(y_unb, median_bag))
        print(f"    kf_seed={kf_seed}: mean_bag={r_mean:.4f}  median_bag={r_median:.4f}")
        repro_records.append({"kf_seed": kf_seed,
                              "rae_mean_bag": r_mean,
                              "rae_median_bag": r_median})

    repro_mean_arr = np.array([r["rae_mean_bag"] for r in repro_records])
    repro_median_arr = np.array([r["rae_median_bag"] for r in repro_records])
    print(f"    fresh-kf mean_bag    -> mean {repro_mean_arr.mean():.4f}  "
          f"min {repro_mean_arr.min():.4f}  max {repro_mean_arr.max():.4f}")
    print(f"    fresh-kf median_bag  -> mean {repro_median_arr.mean():.4f}  "
          f"min {repro_median_arr.min():.4f}  max {repro_median_arr.max():.4f}")

    # ---------- STEP 4: Reproducibility verdict ----------
    repro_floor_hit = bool(repro_mean_arr.mean() <= FLOOR_TARGET)
    repro_min_hit = bool(repro_mean_arr.min() <= FLOOR_TARGET)
    if repro_floor_hit:
        verdict = "REPRODUCIBLE_AT_OR_BELOW_FLOOR"
    elif repro_min_hit:
        verdict = "OCCASIONAL_HIT_BUT_MEAN_ABOVE_FLOOR"
    else:
        verdict = "LUCKY_SEED_NOT_REPRODUCIBLE"
    print(f"[4] reproducibility verdict = {verdict}")
    print(f"    floor target {FLOOR_TARGET}; fresh-kf mean_bag mean "
          f"{repro_mean_arr.mean():.4f}")
    print(f"    nb2154 cycle 78 was 1 of {summary_2154['at_or_below_floor_count']} "
          f"sub-floor cycles out of 120 ({100*summary_2154['at_or_below_floor_frac']:.1f}%)")

    # ---------- STEP 5: Deploy CSV (only if reproducible) ----------
    deploy_path = None
    if repro_floor_hit:
        print("\n[5] DEPLOY: building nb2157_deploy_nb2154.csv (513 compounds)")
        # Train residual model on ALL 4392 (4139 train + 253 unblind) using
        # chemprop_aux as anchor. We need anchor and X on all 4392 + all 513 test.
        # NOTE: anchor for train rows comes from nb1133 chemprop_aux residual OOF.
        # Pragmatic deploy: refit the same residual model on the 253 unblind only
        # (since X / anchor at the 4139 row level for chemprop_aux mismatched
        # earlier -- BAD4141), then predict on 513.
        # The 4139 chemprop_aux residual OOF is at
        # data/processed/nb1133_chemprop_aux_residual_oof.npy
        # train anchor pec50 at oof_chemprop_aux.npy
        print("    (deploy step requires the train-side residual stack; using "
              "253-fit-only refit on X_top28 as the smallest-faithful deploy)")

        # Refit on ALL 253 (no holdout) with the 5 base seeds.
        # Then predict on all 513 using anchor_513 + resid_513.
        X_all513 = _build_117_feature_block(
            np.arange(n_test), n_test, top_idx_lookup, pred_chembl_513, mean_sim_513
        )
        X_top28_all = X_all513[:, top28_idx]
        per_seed_513 = np.zeros((len(BASE_SEEDS), n_test), dtype=np.float64)
        for i, s in enumerate(BASE_SEEDS):
            params = _exact_params(s)
            mdl = lgb.LGBMRegressor(**params)
            mdl.fit(X_top28_unb, residual)
            per_seed_513[i] = anchor_513 + mdl.predict(X_top28_all)
        pred513_mean = per_seed_513.mean(axis=0)
        deploy_df = pd.DataFrame({
            "SMILES": te["smiles"].astype(str) if "smiles" in te.columns
                      else te["SMILES"].astype(str),
            "Molecule Name": te["molecule_name"] if "molecule_name" in te.columns
                             else (te["Molecule Name"] if "Molecule Name" in te.columns else np.arange(n_test)),
            "pEC50": pred513_mean,
        })
        deploy_path = Path(__file__).resolve().parents[1] / "submissions" / "nb2157_deploy_nb2154.csv"
        deploy_df.to_csv(deploy_path, index=False)
        print(f"    [save] {deploy_path}  shape {deploy_df.shape}")
    else:
        print("\n[5] DEPLOY: SKIPPED -- not reproducible at floor.")

    # ---------- STEP 6: Save summary ----------
    out = {
        "tag": TAG,
        "verifies": "nb2154",
        "rae_best_oof_recomputed": float(rae_best),
        "rae_best_oof_claim": 0.4620,
        "claim_match_within_5e-4": bool(abs(rae_best - 0.4620) < 5e-4),
        "per_cycle_min": float(per_cycle.min()),
        "per_cycle_max": float(per_cycle.max()),
        "per_cycle_mean": float(per_cycle.mean()),
        "per_cycle_median": float(np.median(per_cycle)),
        "per_cycle_at_or_below_0.4698": at_or_below_floor,
        "per_cycle_at_or_below_0.4737": at_or_below_loose,
        "n_cycles": int(len(per_cycle)),
        "best_cycle_index": int(summary_2154["best_cycle"]),
        "exact_config": {
            "L": EXACT_L, "lr": EXACT_LR, "mc": EXACT_MC,
            "lambda": EXACT_LAMBDA, "ff": EXACT_FF,
            "seeds": BASE_SEEDS, "kf_seeds": FRESH_KF_SEEDS,
            "max_depth": EXACT_MAX_DEPTH, "n_estimators": EXACT_NEST,
        },
        "repro_records": repro_records,
        "repro_mean_bag_mean": float(repro_mean_arr.mean()),
        "repro_mean_bag_min": float(repro_mean_arr.min()),
        "repro_mean_bag_max": float(repro_mean_arr.max()),
        "repro_median_bag_mean": float(repro_median_arr.mean()),
        "verdict": verdict,
        "floor_target": FLOOR_TARGET,
        "mean_bag_baseline": MEAN_BAG_BASELINE,
        "deploy_csv": str(deploy_path) if deploy_path else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    return out


if __name__ == "__main__":
    res = main()
    print("\n==== VERIFY SUMMARY ====")
    for k in ("rae_best_oof_recomputed", "rae_best_oof_claim",
              "claim_match_within_5e-4", "per_cycle_min", "per_cycle_max",
              "per_cycle_mean", "per_cycle_at_or_below_0.4698",
              "per_cycle_at_or_below_0.4737", "best_cycle_index",
              "repro_mean_bag_mean", "repro_mean_bag_min", "repro_mean_bag_max",
              "verdict", "deploy_csv"):
        print(f"  {k}: {res.get(k)}")
