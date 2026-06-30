"""nb1410 -- DEPLOY of nb1403 BoB (outer-bagged nb1391 0.85*nb1373 + 0.15*nb1352 blend) to 513.

Per task spec:
    For 5 outer seeds {0, 1, 7, 42, 137}:
        inner_seeds(o) = [o*1000 + s for s in {0, 1, 7, 42, 137}]
        AtomPair path (top-30 + ChEMBL pred + sim = 32-col, nb1373 family):
            for each inner seed: fit shallow LGBM Huber on ALL 253 unblind,
            target = y_unb - nb1070_pred_oof, predict residual on full 513.
        MACCS path (top-20 + ChEMBL pred + sim = 22-col, nb1352 family):
            for each inner seed: fit shallow LGBM Huber on ALL 253 unblind,
            target = y_unb - nb1070_pred_oof, predict residual on full 513.
        per_outer_nb1373_o = mean of 5 inner AtomPair deploy resid_513
        per_outer_nb1352_o = mean of 5 inner MACCS    deploy resid_513
        blend_o = 0.85 * per_outer_nb1373_o + 0.15 * per_outer_nb1352_o

    Stack to (5, 513). Row-level BoB MEAN and MEDIAN across 5 outer seeds.
    te_nb1410_mean   = te_nb1070 + BoB_mean_residual_513
    te_nb1410_median = te_nb1070 + BoB_median_residual_513

Honest LB anchors (from nb1403 cross-fit):
    * BoB MEAN   0.5079
    * BoB MEDIAN 0.5073

Outputs:
    data/processed/te_nb1410_mean.npy                       (513,) float32
    data/processed/te_nb1410_median.npy                     (513,) float32
    data/processed/nb1410_per_outer_blend_resid_513.npy     (5, 513) float32
    data/processed/nb1410_per_outer_nb1373_resid_513.npy    (5, 513) float32
    data/processed/nb1410_per_outer_nb1352_resid_513.npy    (5, 513) float32
    data/processed/nb1410_summary.json
    submissions/nb1410_deploy_nb1403_mean.csv               (513 rows)
    submissions/nb1410_deploy_nb1403_median.csv             (513 rows)
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
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1410"
ANCHOR = "nb1070"
PARENT = "nb1403"
GP_ATOMPAIR = "nb1373"
GP_MACCS = "nb1352"

INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
OUTER_SEEDS = [0, 1, 7, 42, 137]

W_NB1373 = 0.85
W_NB1352 = 0.15

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

HONEST_LB_ANCHOR_MEAN = 0.5079
HONEST_LB_ANCHOR_MEDIAN = 0.5073


def _lgbm_params(seed: int) -> dict:
    # EXACT match to nb1373 / nb1352 / nb1381 / nb1361 / nb1380 / nb1370.
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1403 BoB (outer-bagged nb1391 blend) -> 513")
    print(f"          anchor={ANCHOR}  parent={PARENT}")
    print(f"          AtomPair grandparent={GP_ATOMPAIR}  weight={W_NB1373}")
    print(f"          MACCS    grandparent={GP_MACCS}     weight={W_NB1352}")
    print(f"          outer seeds = {OUTER_SEEDS}")
    print(f"          inner base seeds = {INNER_BASE_SEEDS}")
    print(f"          inner_seeds(o) = [o*1000 + s for s in base]")
    print(f"          deploy fit (NO KFold) on all 253 unblind per inner seed")
    print(f"          honest LB anchors: mean={HONEST_LB_ANCHOR_MEAN}  "
          f"median={HONEST_LB_ANCHOR_MEDIAN}")
    print("=" * 78)

    # ---- Load top-30 AtomPair bit indices from nb1373 summary ----
    p73_path = DATA_PROCESSED / f"{GP_ATOMPAIR}_summary.json"
    with open(p73_path) as f:
        p73 = json.load(f)
    top_ap_idx = np.array(p73["top_atompair_bit_indices_ranked"], dtype=int)
    top_k_ap = int(p73["top_k_atompair"])
    if len(top_ap_idx) != top_k_ap:
        raise ValueError(
            f"{GP_ATOMPAIR} top-{top_k_ap} indices mismatch: got {len(top_ap_idx)}")
    print(f"[load] {GP_ATOMPAIR} top-{top_k_ap} AtomPair bits (ranked) = "
          f"{top_ap_idx.tolist()}")

    # ---- Load top-20 MACCS bit indices from nb1352 summary ----
    p52_path = DATA_PROCESSED / f"{GP_MACCS}_summary.json"
    with open(p52_path) as f:
        p52 = json.load(f)
    top_mc_idx = np.array(p52["top_maccs_bit_indices_ranked"], dtype=int)
    top_k_mc = int(p52["top_k_maccs"])
    if len(top_mc_idx) != top_k_mc:
        raise ValueError(
            f"{GP_MACCS} top-{top_k_mc} indices mismatch: got {len(top_mc_idx)}")
    print(f"[load] {GP_MACCS} top-{top_k_mc} MACCS bits (ranked) = "
          f"{top_mc_idx.tolist()}")

    # ---- Load test ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    else:
        cand = [c for c in te.columns if "name" in c.lower()]
        if not cand:
            raise KeyError(
                f"No Molecule Name column found in test ({te.columns.tolist()})")
        mol_names = te[cand[0]].astype(str).tolist()
    print(f"[load] n_test={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Anchor (513) and anchor-OOF (253) ----
    te_anchor_513 = np.load(
        DATA_PROCESSED / f"te_{ANCHOR}.npy"
    ).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"te_{ANCHOR} shape mismatch: {te_anchor_513.shape}")
    anchor_oof_253 = np.load(
        DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    ).astype(np.float64)
    if anchor_oof_253.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} OOF shape mismatch: {anchor_oof_253.shape}")
    rae_anchor = float(rae(y_unb, anchor_oof_253))
    print(f"[anchor] {ANCHOR}_pred_oof RAE = {rae_anchor:.4f}")
    print(f"[anchor] te_{ANCHOR}  mean={te_anchor_513.mean():.4f}  "
          f"std={te_anchor_513.std():.4f}  "
          f"min={te_anchor_513.min():.4f}  max={te_anchor_513.max():.4f}")

    # ---- AtomPair-2048 cache (513) ----
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}")
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    if X_ap_te.shape[0] != n_test:
        raise ValueError(f"AtomPair cache shape mismatch: {X_ap_te.shape}")
    n_ap = int(X_ap_te.shape[1])
    print(f"[load] AtomPair cache shape = {X_ap_te.shape}  (n_bits={n_ap})")
    X_ap_te_pruned = X_ap_te[:, top_ap_idx].astype(np.float32)
    print(f"       AtomPair pruned (513) shape = {X_ap_te_pruned.shape}  "
          f"density = {X_ap_te_pruned.mean():.4f}")

    # ---- MACCS-167 cache (513) ----
    if not MACCS_TE_PATH.exists():
        raise FileNotFoundError(f"MACCS test cache missing: {MACCS_TE_PATH}")
    X_mc_te = np.load(MACCS_TE_PATH)
    if X_mc_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_mc_te.shape}")
    n_mc = int(X_mc_te.shape[1])
    print(f"[load] MACCS cache shape = {X_mc_te.shape}  (n_bits={n_mc})")
    X_mc_te_pruned = X_mc_te[:, top_mc_idx].astype(np.float32)
    print(f"       MACCS pruned (513) shape = {X_mc_te_pruned.shape}  "
          f"density = {X_mc_te_pruned.mean():.4f}")

    # ---- Cached ChEMBL features (513) ----
    if not PRED_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(
            f"pred_chembl_pec50_513 missing: {PRED_CHEMBL_513_PATH}")
    if not SIM_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(
            f"sim_chembl_513 missing: {SIM_CHEMBL_513_PATH}")
    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    if pred_chembl_513.shape[0] != n_test or sim_chembl_513.shape[0] != n_test:
        raise ValueError(
            f"ChEMBL feature shape mismatch: "
            f"pred {pred_chembl_513.shape}, sim {sim_chembl_513.shape}")
    print(f"[load] pred_chembl_pec50_513  mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[load] sim_chembl_513         mean={sim_chembl_513.mean():.3f}  "
          f"std={sim_chembl_513.std():.3f}")

    # ---- Build PRUNED matrices ----
    # AtomPair: 30 + 2 = 32-col
    X_ap_te_full = np.concatenate(
        [
            X_ap_te_pruned,
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat] X_ap_te_full (513) shape = {X_ap_te_full.shape}")

    X_ap_unb_pruned = X_ap_te[unb_idx][:, top_ap_idx].astype(np.float32)
    pred_chembl_unb = pred_chembl_513[unb_idx]
    sim_chembl_unb = sim_chembl_513[unb_idx]
    X_ap_unb_full = np.concatenate(
        [
            X_ap_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            sim_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat] X_ap_unb_full (253) shape = {X_ap_unb_full.shape}")

    # MACCS: 20 + 2 = 22-col
    X_mc_te_full = np.concatenate(
        [
            X_mc_te_pruned,
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat] X_mc_te_full (513) shape = {X_mc_te_full.shape}")

    X_mc_unb_pruned = X_mc_te[unb_idx][:, top_mc_idx].astype(np.float32)
    X_mc_unb_full = np.concatenate(
        [
            X_mc_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            sim_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat] X_mc_unb_full (253) shape = {X_mc_unb_full.shape}")

    feat_dim_ap = int(X_ap_te_full.shape[1])
    feat_dim_mc = int(X_mc_te_full.shape[1])

    # ---- Residual target on 253 ----
    residual_unb = y_unb - anchor_oof_253
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Outer-bag deploy loop ----
    print("\n" + "-" * 78)
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)
    n_fits_per_path = n_outer * n_inner
    n_fits_total = 2 * n_fits_per_path
    print(f"OUTER-BAG DEPLOY LOOP "
          f"({n_outer} outer x {n_inner} inner x 2 paths = {n_fits_total} deploy fits)")
    print("-" * 78)

    per_outer_nb1373_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_nb1352_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_blend_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_inner_seeds: list[list[int]] = []
    per_outer_records: list[dict] = []

    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [o * 1000 + s for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds.append([int(s) for s in inner_seeds])
        print(f"\n   outer seed {o}:  inner seeds = {inner_seeds}")

        # --- AtomPair (nb1373) path ---
        inner_ap_resid_513 = np.zeros((n_inner, n_test), dtype=np.float64)
        inner_ap_in_rae: list[float] = []
        for ii, s in enumerate(inner_seeds):
            mdl = LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_ap_unb_full, residual_unb)
            resid_in = mdl.predict(X_ap_unb_full)
            corr_in = anchor_oof_253 + resid_in
            in_rae_s = float(rae(y_unb, corr_in))
            inner_ap_in_rae.append(in_rae_s)
            inner_ap_resid_513[ii] = mdl.predict(X_ap_te_full)
        po_ap_mean = inner_ap_resid_513.mean(axis=0)
        per_outer_nb1373_resid_513[oi] = po_ap_mean
        po_ap_unb_corr = anchor_oof_253 + inner_ap_resid_513[:, unb_idx].mean(axis=0)
        po_ap_in_rae = float(rae(y_unb, po_ap_unb_corr))
        print(f"      [nb1373 AP path] inner in_RAE = "
              f"[{', '.join(f'{r:.4f}' for r in inner_ap_in_rae)}]")
        print(f"      [nb1373 AP path] outer mean_resid_513 mean={po_ap_mean.mean():+.4f}  "
              f"std={po_ap_mean.std():.4f}  in_RAE_253={po_ap_in_rae:.4f}")

        # --- MACCS (nb1352) path ---
        inner_mc_resid_513 = np.zeros((n_inner, n_test), dtype=np.float64)
        inner_mc_in_rae: list[float] = []
        for ii, s in enumerate(inner_seeds):
            mdl = LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_mc_unb_full, residual_unb)
            resid_in = mdl.predict(X_mc_unb_full)
            corr_in = anchor_oof_253 + resid_in
            in_rae_s = float(rae(y_unb, corr_in))
            inner_mc_in_rae.append(in_rae_s)
            inner_mc_resid_513[ii] = mdl.predict(X_mc_te_full)
        po_mc_mean = inner_mc_resid_513.mean(axis=0)
        per_outer_nb1352_resid_513[oi] = po_mc_mean
        po_mc_unb_corr = anchor_oof_253 + inner_mc_resid_513[:, unb_idx].mean(axis=0)
        po_mc_in_rae = float(rae(y_unb, po_mc_unb_corr))
        print(f"      [nb1352 MC path] inner in_RAE = "
              f"[{', '.join(f'{r:.4f}' for r in inner_mc_in_rae)}]")
        print(f"      [nb1352 MC path] outer mean_resid_513 mean={po_mc_mean.mean():+.4f}  "
              f"std={po_mc_mean.std():.4f}  in_RAE_253={po_mc_in_rae:.4f}")

        # --- Per-outer blend = 0.85 AP + 0.15 MC ---
        po_blend = W_NB1373 * po_ap_mean + W_NB1352 * po_mc_mean
        per_outer_blend_resid_513[oi] = po_blend
        po_blend_unb_corr = anchor_oof_253 + po_blend[unb_idx]
        po_blend_in_rae = float(rae(y_unb, po_blend_unb_corr))
        print(f"      [BLEND     0.85/0.15] outer blend_resid_513 mean={po_blend.mean():+.4f}  "
              f"std={po_blend.std():.4f}  in_RAE_253={po_blend_in_rae:.4f}")

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": [int(s) for s in inner_seeds],
            "inner_nb1373_in_sample_rae": inner_ap_in_rae,
            "inner_nb1352_in_sample_rae": inner_mc_in_rae,
            "po_nb1373_resid_513_mean": float(po_ap_mean.mean()),
            "po_nb1373_resid_513_std": float(po_ap_mean.std()),
            "po_nb1373_in_sample_rae_253": po_ap_in_rae,
            "po_nb1352_resid_513_mean": float(po_mc_mean.mean()),
            "po_nb1352_resid_513_std": float(po_mc_mean.std()),
            "po_nb1352_in_sample_rae_253": po_mc_in_rae,
            "po_blend_resid_513_mean": float(po_blend.mean()),
            "po_blend_resid_513_std": float(po_blend.std()),
            "po_blend_in_sample_rae_253": po_blend_in_rae,
        })

    # ---- BoB row-level aggregation (over 5 per-outer-blend vectors) ----
    bob_mean_resid_513 = per_outer_blend_resid_513.mean(axis=0)
    bob_median_resid_513 = np.median(per_outer_blend_resid_513, axis=0)
    te_nb1410_mean = te_anchor_513 + bob_mean_resid_513
    te_nb1410_median = te_anchor_513 + bob_median_resid_513

    # ---- In-sample RAE on unblind slice ----
    in_rae_mean = float(rae(y_unb, te_nb1410_mean[unb_idx]))
    in_rae_median = float(rae(y_unb, te_nb1410_median[unb_idx]))

    print("\n" + "-" * 78)
    print("513-ROW DEPLOY VECTOR DIAGNOSTICS")
    print("-" * 78)
    print(f"   bob_mean_resid_513    mean={bob_mean_resid_513.mean():+.4f}  "
          f"std={bob_mean_resid_513.std():.4f}  "
          f"min={bob_mean_resid_513.min():+.4f}  max={bob_mean_resid_513.max():+.4f}")
    print(f"   bob_median_resid_513  mean={bob_median_resid_513.mean():+.4f}  "
          f"std={bob_median_resid_513.std():.4f}  "
          f"min={bob_median_resid_513.min():+.4f}  max={bob_median_resid_513.max():+.4f}")
    print(f"   te_nb1410_mean    mean={te_nb1410_mean.mean():.4f}  "
          f"std={te_nb1410_mean.std():.4f}  "
          f"min={te_nb1410_mean.min():.4f}  max={te_nb1410_mean.max():.4f}")
    print(f"   te_nb1410_median  mean={te_nb1410_median.mean():.4f}  "
          f"std={te_nb1410_median.std():.4f}  "
          f"min={te_nb1410_median.min():.4f}  max={te_nb1410_median.max():.4f}")
    print(f"   in_RAE(unb, mean)   = {in_rae_mean:.4f}   "
          f"(honest LB anchor {HONEST_LB_ANCHOR_MEAN})")
    print(f"   in_RAE(unb, median) = {in_rae_median:.4f}   "
          f"(honest LB anchor {HONEST_LB_ANCHOR_MEDIAN})")

    # ---- Save NPY ----
    te_mean_path = DATA_PROCESSED / f"te_{TAG}_mean.npy"
    te_median_path = DATA_PROCESSED / f"te_{TAG}_median.npy"
    np.save(te_mean_path, te_nb1410_mean.astype(np.float32))
    np.save(te_median_path, te_nb1410_median.astype(np.float32))
    blend_path = DATA_PROCESSED / f"{TAG}_per_outer_blend_resid_513.npy"
    ap_path = DATA_PROCESSED / f"{TAG}_per_outer_nb1373_resid_513.npy"
    mc_path = DATA_PROCESSED / f"{TAG}_per_outer_nb1352_resid_513.npy"
    np.save(blend_path, per_outer_blend_resid_513.astype(np.float32))
    np.save(ap_path, per_outer_nb1373_resid_513.astype(np.float32))
    np.save(mc_path, per_outer_nb1352_resid_513.astype(np.float32))
    print(f"\n[save] {te_mean_path}")
    print(f"[save] {te_median_path}")
    print(f"[save] {blend_path}  shape={per_outer_blend_resid_513.shape}")
    print(f"[save] {ap_path}     shape={per_outer_nb1373_resid_513.shape}")
    print(f"[save] {mc_path}     shape={per_outer_nb1352_resid_513.shape}")

    # ---- Save CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    mean_csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1403_mean.csv"
    median_csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1403_median.csv"
    df_mean = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1410_mean.astype(np.float64),
    })
    df_median = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1410_median.astype(np.float64),
    })
    df_mean.to_csv(mean_csv_path, index=False)
    df_median.to_csv(median_csv_path, index=False)
    print(f"[save] {mean_csv_path}    rows={len(df_mean)}  "
          f"cols={list(df_mean.columns)}")
    print(f"[save] {median_csv_path}  rows={len(df_median)}  "
          f"cols={list(df_median.columns)}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "parent_method": PARENT,
        "grandparent_atompair": GP_ATOMPAIR,
        "grandparent_maccs": GP_MACCS,
        "top_atompair_bit_indices_ranked": top_ap_idx.tolist(),
        "top_maccs_bit_indices_ranked": top_mc_idx.tolist(),
        "top_k_atompair": top_k_ap,
        "top_k_maccs": top_k_mc,
        "feat_dim_atompair": feat_dim_ap,
        "feat_dim_maccs": feat_dim_mc,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_atompair_bits": int(n_ap),
        "n_maccs_bits": int(n_mc),
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds": per_outer_inner_seeds,
        "n_outer": int(n_outer),
        "n_inner": int(n_inner),
        "n_total_fits": int(n_fits_total),
        "w_nb1373": W_NB1373,
        "w_nb1352": W_NB1352,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070_oof_253": rae_anchor,
        "residual_unb_mean": float(residual_unb.mean()),
        "residual_unb_std": float(residual_unb.std()),
        "per_outer_records": per_outer_records,
        "te_nb1410_mean_stats": {
            "mean": float(te_nb1410_mean.mean()),
            "std": float(te_nb1410_mean.std()),
            "min": float(te_nb1410_mean.min()),
            "max": float(te_nb1410_mean.max()),
        },
        "te_nb1410_median_stats": {
            "mean": float(te_nb1410_median.mean()),
            "std": float(te_nb1410_median.std()),
            "min": float(te_nb1410_median.min()),
            "max": float(te_nb1410_median.max()),
        },
        "bob_mean_resid_513_stats": {
            "mean": float(bob_mean_resid_513.mean()),
            "std": float(bob_mean_resid_513.std()),
            "min": float(bob_mean_resid_513.min()),
            "max": float(bob_mean_resid_513.max()),
        },
        "bob_median_resid_513_stats": {
            "mean": float(bob_median_resid_513.mean()),
            "std": float(bob_median_resid_513.std()),
            "min": float(bob_median_resid_513.min()),
            "max": float(bob_median_resid_513.max()),
        },
        "in_rae_unb_mean": in_rae_mean,
        "in_rae_unb_median": in_rae_median,
        "honest_lb_anchor_mean": HONEST_LB_ANCHOR_MEAN,
        "honest_lb_anchor_median": HONEST_LB_ANCHOR_MEDIAN,
        "te_mean_npy_path": str(te_mean_path),
        "te_median_npy_path": str(te_median_path),
        "per_outer_blend_npy_path": str(blend_path),
        "per_outer_nb1373_npy_path": str(ap_path),
        "per_outer_nb1352_npy_path": str(mc_path),
        "mean_csv_path": str(mean_csv_path),
        "median_csv_path": str(median_csv_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_test", "n_unb", "feat_dim_atompair", "feat_dim_maccs",
        "n_outer", "n_inner", "n_total_fits",
        "w_nb1373", "w_nb1352",
        "rae_anchor_nb1070_oof_253",
        "te_nb1410_mean_stats",
        "te_nb1410_median_stats",
        "in_rae_unb_mean",
        "in_rae_unb_median",
        "honest_lb_anchor_mean",
        "honest_lb_anchor_median",
        "mean_csv_path",
        "median_csv_path",
        "te_mean_npy_path",
        "te_median_npy_path",
    ):
        print(f"  {k}: {res.get(k)}")
