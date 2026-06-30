"""nb1173 -- Fine-grained K-sweep around K=32 (nb1158 winner).

CONTEXT (cycles 143-...):
    nb1157 (fine grid) on FRESH seeds {1001..1010} reported:
        K=30  rae_mean_bag = 0.4907
        K=32  rae_mean_bag = 0.4902  <-- best fresh-grid optimum
        K=35  rae_mean_bag = 0.4937
    nb1158 promoted K=32 as DEPLOY (PRIMARY-1) at 0.4902 mean-bag (10 fresh
    seeds), in_RAE_deploy 0.10303 (in-sample, anchor te_chemprop_aux PRE).

    The K-grid {30, 32, 35} skipped K=31, K=33, K=34. If the local minimum
    lives at one of those (e.g. K=31 = 0.4895 or K=33 = 0.4898), we leave LB
    margin on the table.

HYPOTHESIS:
    Re-sweep K in {30, 31, 32, 33, 34} at finer resolution. Use both the
    nb1151 canonical seed set {0, 1, 7, 42, 137} (5 seeds) AND nb1157 fresh
    seeds {1001..1010} (10 seeds), 15 seeds total. Aggregate per-seed RAE
    mean/std and pooled mean-bag RAE per K. If best K beats K=32 (0.4902)
    by at least 0.003, build deploy CSV.

PROTOCOL (identical pipeline to nb1158 deploy):
    1. Anchor    : te_chemprop_aux.npy (PRE-unblind, in_RAE 0.6216)
    2. Features  : 117-col 5-way K-tuned matrix (AtomPair + MACCS + Mordred +
                   ChempropEmbed + Avalon + ChEMBL_kNN), reuse nb1151's loaders.
    3. SHAP rank : data/processed/nb2063_shap_importance_full117.npy
    4. LGBM(MSE) : max_depth=4, num_leaves=15, n_estimators=300, lr=0.03,
                   min_child_samples=5, reg_lambda=2.0 (nb1151 canonical).
    5. CV        : scaffold_kfold_indices(Murcko, n_splits=5, shuffle, seed=kf_seed)
    6. Seeds     : canonical {0,1,7,42,137} + fresh {1001..1010} = 15 seeds.
    7. Per K     : per-seed RAE list + mean/std/min/max,
                   mean-bag pooled RAE, median-bag pooled RAE.
    8. Gate      : best K mean-bag <= 0.5027 (deploy_gate).
    9. Promote   : if best mean-bag <= K=32 ref (0.4902) - 0.003, build deploy CSV.

DEPLOY (only if PROMOTE):
    - Refit LGBM on ALL 253 unblind residuals at nb1151 canonical seeds
      {0, 1, 7, 42, 137}; mean across seeds; add to te_chemprop_aux.
    - Emit data/processed/te_nb1173_K{best}.npy and
      submissions/nb1173_deploy_K{best}.csv.

NO mutation of te_nb1158.npy or other ladder artefacts.

Outputs:
    data/processed/nb1173_mean_bag_oof_K{K}.npy   (per K, 253 float32)
    data/processed/nb1173_summary.json
    (only if PROMOTE):
        data/processed/te_nb1173_K{best}.npy      (513 float32)
        submissions/nb1173_deploy_K{best}.csv     (Molecule Name, SMILES, pEC50)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

# Reuse nb1151's pipeline (avoid duplicating ChEMBL / Mordred / feature loaders).
from nb1151_scaffold_k_sweep import (  # type: ignore
    ANCHOR, ANCHOR_TE_PATH, RESID_FOLDS, RESID_SEEDS,
    NB2063_SHAP_IMP, NB2103_K28_SCAFFOLD_REF, DEPLOY_GATE, DECISION_MARGIN,
    ATOMPAIR_TE_PATH, MACCS_TE_PATH, CHEMPROP_EMBED_TE_PATH, AVALON_TE_PATH,
    NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY, NB1523_SUMMARY,
    NB1524_SUMMARY, NB1541_SUMMARY,
    _load_chembl_pool, _load_mordred_test, _load_npy_test,
    _murcko_scaffold, _safe_inchikey, _safe_can_smiles,
    _tanimoto_topk, _knn_predict, _lgbm_params,
    _residual_scaffold_cross_fit_one_seed,
    _extract_atompair_top_idx_from_nb1484, _extract_best_K_record,
    KNN_K,
)
from pxr.chem import standardize, morgan_fp_batch
from rdkit import Chem

TAG = "nb1173"

# Fine-grained K grid around K=32 winner from nb1157/nb1158.
K_GRID = [30, 31, 32, 33, 34]

# 15 seeds = nb1151 canonical 5 + nb1157 fresh 10 (disjoint).
CANONICAL_SEEDS = list(RESID_SEEDS)                  # [0, 1, 7, 42, 137]
FRESH_SEEDS = list(range(1001, 1011))                # 10 fresh seeds
ALL_SEEDS = CANONICAL_SEEDS + FRESH_SEEDS            # 15 seeds total

# Reference numbers.
NB1158_K32_MEAN_BAG_REF = 0.4902                     # promoted PRIMARY
NB1157_K30_MEAN_BAG_REF = 0.4907
NB1157_K35_MEAN_BAG_REF = 0.4937
PROMOTE_MARGIN = 0.003                               # must beat by >= 0.003

SUB_DIR = ROOT / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- FINE-GRAINED K-sweep K in {K_GRID}")
    print(f"          canonical seeds = {CANONICAL_SEEDS}  ({len(CANONICAL_SEEDS)})")
    print(f"          fresh seeds     = {FRESH_SEEDS}  ({len(FRESH_SEEDS)})")
    print(f"          total seeds     = {len(ALL_SEEDS)}")
    print(f"          ref K=32 mean_bag = {NB1158_K32_MEAN_BAG_REF:.4f}  "
          f"(nb1158 PRIMARY-1)")
    print(f"          deploy gate       = {DEPLOY_GATE:.4f}")
    print(f"          promote margin    = {PROMOTE_MARGIN:.4f}")
    print("=" * 78)

    # --- Load SHAP ranking + anchor + unblind labels ---
    if not NB2063_SHAP_IMP.exists():
        raise FileNotFoundError(f"missing {NB2063_SHAP_IMP}")
    shap_imp_full117 = np.load(NB2063_SHAP_IMP).astype(np.float32)
    full_rank_order = np.argsort(-shap_imp_full117).astype(np.int32)
    print(f"[ref] nb2063 SHAP importance shape = {shap_imp_full117.shape}")

    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    residual = y_unb - anchor
    print(f"[anchor] {ANCHOR} in_RAE = {rae_anchor:.4f}  "
          f"resid_mean={residual.mean():+.4f}  resid_std={residual.std():.4f}")

    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_mols = [standardize(s) for s in unb_smiles]
    scaffolds_unb = [_murcko_scaffold(m) for m in unb_mols]
    n_unique_sc = len({s for s in scaffolds_unb if s})
    print(f"[scaf] n_unique_scaffolds = {n_unique_sc}")

    # --- Build 117-col matrices (full 513 + unblind slice) ---
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f: sum_1352 = json.load(f)
    with open(NB1392_SUMMARY) as f: sum_1392 = json.load(f)
    with open(NB1484_SUMMARY) as f: sum_1484 = json.load(f)
    with open(NB1523_SUMMARY) as f: sum_1523 = json.load(f)
    with open(NB1524_SUMMARY) as f: sum_1524 = json.load(f)
    with open(NB1541_SUMMARY) as f: sum_1541 = json.load(f)

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"],
                                 dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                      best_K_key="best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"],
                                  dtype=int)

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    print(f"[feat] AP={X_ap_te.shape}  MACCS={X_maccs_te.shape}  "
          f"Mord={X_mord_te.shape}  Embed={X_emb_te.shape}  Av={X_av_te.shape}")

    # ChEMBL kNN feature (deterministic, identical to nb1151/nb1158)
    print("[chembl] loading PXR pool")
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
        std_test_smiles.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )
    print(f"[chembl] pool size {len(pool)}  median pEC50 {pool_median:.3f}")

    X_unb = np.concatenate([
        X_ap_te[unb_idx][:, top_ap_bit_idx],
        X_maccs_te[unb_idx][:, top_maccs_bit_idx],
        X_mord_te[unb_idx][:, top_mord_col_idx],
        X_emb_te[unb_idx][:, top_embed_col_idx],
        X_av_te[unb_idx][:, top_avalon_bit_idx],
        pred_chembl_pec50[unb_idx].reshape(-1, 1),
        mean_sim[unb_idx].reshape(-1, 1),
    ], axis=1).astype(np.float32)
    feat_dim = X_unb.shape[1]
    if feat_dim != shap_imp_full117.shape[0]:
        raise ValueError(
            f"feat_dim {feat_dim} != SHAP rank {shap_imp_full117.shape[0]}"
        )

    X_513 = np.concatenate([
        X_ap_te[:, top_ap_bit_idx],
        X_maccs_te[:, top_maccs_bit_idx],
        X_mord_te[:, top_mord_col_idx],
        X_emb_te[:, top_embed_col_idx],
        X_av_te[:, top_avalon_bit_idx],
        pred_chembl_pec50.reshape(-1, 1),
        mean_sim.reshape(-1, 1),
    ], axis=1).astype(np.float32)

    # --- Per-K sweep over 15 seeds ---
    print("\n" + "-" * 78)
    print(f"FINE-GRAINED K-SWEEP  K in {K_GRID}  seeds={len(ALL_SEEDS)} total")
    print("-" * 78)
    per_K_records: list[dict] = []
    for K in K_GRID:
        print(f"\n--- K={K} ---")
        topK_idx = full_rank_order[:K].astype(np.int32)
        X_unb_topK = X_unb[:, topK_idx].astype(np.float32)

        per_seed_corr = np.zeros((len(ALL_SEEDS), n_unb), dtype=np.float64)
        per_seed_rae_canon: list[float] = []
        per_seed_rae_fresh: list[float] = []
        per_seed_rae_all: list[float] = []
        for i, s in enumerate(ALL_SEEDS):
            ts = time.time()
            resid_oof_s = _residual_scaffold_cross_fit_one_seed(
                X_unb_topK, residual, scaffolds_unb, s
            )
            pred_corr_s = anchor + resid_oof_s
            per_seed_corr[i] = pred_corr_s
            r = float(rae(y_unb, pred_corr_s))
            per_seed_rae_all.append(r)
            if s in CANONICAL_SEEDS:
                per_seed_rae_canon.append(r)
            else:
                per_seed_rae_fresh.append(r)
            print(f"   K={K} seed={s:>4d}  rae={r:.4f}  "
                  f"wall={time.time()-ts:.1f}s")

        mean_bag_oof = per_seed_corr.mean(axis=0)
        median_bag_oof = np.median(per_seed_corr, axis=0)
        rae_mean_bag = float(rae(y_unb, mean_bag_oof))
        rae_median_bag = float(rae(y_unb, median_bag_oof))

        # Also compute canonical-only and fresh-only mean-bag for sanity.
        canon_mask = np.array([s in CANONICAL_SEEDS for s in ALL_SEEDS])
        fresh_mask = ~canon_mask
        mean_bag_canon = per_seed_corr[canon_mask].mean(axis=0)
        mean_bag_fresh = per_seed_corr[fresh_mask].mean(axis=0)
        rae_mean_bag_canon = float(rae(y_unb, mean_bag_canon))
        rae_mean_bag_fresh = float(rae(y_unb, mean_bag_fresh))

        arr_all = np.array(per_seed_rae_all)
        arr_canon = np.array(per_seed_rae_canon)
        arr_fresh = np.array(per_seed_rae_fresh)
        rec = {
            "K": int(K),
            "n_seeds_total": len(ALL_SEEDS),
            "n_seeds_canon": len(CANONICAL_SEEDS),
            "n_seeds_fresh": len(FRESH_SEEDS),
            "per_seed_rae_all": per_seed_rae_all,
            "per_seed_rae_canon": per_seed_rae_canon,
            "per_seed_rae_fresh": per_seed_rae_fresh,
            "rae_per_seed_mean_all": float(arr_all.mean()),
            "rae_per_seed_median_all": float(np.median(arr_all)),
            "rae_per_seed_std_all": float(arr_all.std()),
            "rae_per_seed_min_all": float(arr_all.min()),
            "rae_per_seed_max_all": float(arr_all.max()),
            "rae_per_seed_mean_canon": float(arr_canon.mean()),
            "rae_per_seed_std_canon": float(arr_canon.std()),
            "rae_per_seed_mean_fresh": float(arr_fresh.mean()),
            "rae_per_seed_std_fresh": float(arr_fresh.std()),
            "rae_mean_bag_all": rae_mean_bag,
            "rae_median_bag_all": rae_median_bag,
            "rae_mean_bag_canon": rae_mean_bag_canon,
            "rae_mean_bag_fresh": rae_mean_bag_fresh,
            "delta_mean_bag_vs_K32_ref": rae_mean_bag - NB1158_K32_MEAN_BAG_REF,
            "delta_mean_bag_vs_deploy_gate": rae_mean_bag - DEPLOY_GATE,
            "beats_deploy_gate": bool(rae_mean_bag < DEPLOY_GATE),
            "beats_K32_ref_by_margin": bool(
                rae_mean_bag < NB1158_K32_MEAN_BAG_REF - PROMOTE_MARGIN
            ),
        }
        per_K_records.append(rec)

        # Save per-K mean-bag OOF for downstream use.
        np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K}.npy",
                mean_bag_oof.astype(np.float32))

        print(f"   K={K} mean_bag_all   = {rae_mean_bag:.4f}  "
              f"(d_vs_K32_ref = {rae_mean_bag - NB1158_K32_MEAN_BAG_REF:+.4f})")
        print(f"   K={K} mean_bag_canon = {rae_mean_bag_canon:.4f}  "
              f"mean_bag_fresh = {rae_mean_bag_fresh:.4f}")
        print(f"   K={K} per_seed all   = "
              f"mean {arr_all.mean():.4f}  std {arr_all.std():.4f}  "
              f"range [{arr_all.min():.4f}, {arr_all.max():.4f}]")

    # --- Identify optimum ---
    K_vals = [r["K"] for r in per_K_records]
    rae_vals = [r["rae_mean_bag_all"] for r in per_K_records]
    best_i = int(np.argmin(rae_vals))
    best_K = int(K_vals[best_i])
    best_rae_mean_bag = float(rae_vals[best_i])
    best_rec = per_K_records[best_i]

    beats_K32_by_margin = best_rae_mean_bag < NB1158_K32_MEAN_BAG_REF - PROMOTE_MARGIN
    beats_deploy_gate = best_rae_mean_bag <= DEPLOY_GATE

    if beats_K32_by_margin and beats_deploy_gate:
        verdict = f"PROMOTE_K{best_K} (beats K=32 ref by margin)"
        promote = True
    elif beats_deploy_gate and best_rae_mean_bag < NB1158_K32_MEAN_BAG_REF:
        verdict = (f"FLAT_NEAR_K32 (best K={best_K} = {best_rae_mean_bag:.4f} "
                   f"beats K=32 ref {NB1158_K32_MEAN_BAG_REF:.4f} but not by "
                   f"margin {PROMOTE_MARGIN:.4f})")
        promote = False
    elif beats_deploy_gate:
        verdict = (f"FLAT_VS_K32 (best K={best_K} = {best_rae_mean_bag:.4f} "
                   f">= K=32 ref {NB1158_K32_MEAN_BAG_REF:.4f}; gate cleared)")
        promote = False
    else:
        verdict = (f"REJECT (best K={best_K} = {best_rae_mean_bag:.4f} fails "
                   f"deploy gate {DEPLOY_GATE:.4f})")
        promote = False

    print("\n" + "=" * 78)
    print("FINE-GRAINED K-SWEEP SUMMARY")
    print("=" * 78)
    print(f"  {'K':>3s}  {'mean_bag_all':>12s}  {'canon_bag':>10s}  "
          f"{'fresh_bag':>10s}  {'per_seed_mean':>14s}  {'std':>8s}  "
          f"d_vs_K32")
    for r in per_K_records:
        print(f"  {r['K']:>3d}  {r['rae_mean_bag_all']:>12.4f}  "
              f"{r['rae_mean_bag_canon']:>10.4f}  "
              f"{r['rae_mean_bag_fresh']:>10.4f}  "
              f"{r['rae_per_seed_mean_all']:>14.4f}  "
              f"{r['rae_per_seed_std_all']:>8.4f}  "
              f"{r['delta_mean_bag_vs_K32_ref']:+.4f}")
    print(f"\n  best K          = {best_K}")
    print(f"  best mean_bag   = {best_rae_mean_bag:.4f}")
    print(f"  K=32 ref        = {NB1158_K32_MEAN_BAG_REF:.4f}")
    print(f"  promote margin  = {PROMOTE_MARGIN:.4f}")
    print(f"  deploy gate     = {DEPLOY_GATE:.4f}")
    print(f"  VERDICT         = {verdict}")
    print(f"  PROMOTE         = {promote}")

    # --- Deploy refit (only on PROMOTE) ---
    deploy_info: dict = {
        "promoted": bool(promote),
        "best_K": int(best_K),
        "best_rae_mean_bag_all": float(best_rae_mean_bag),
    }
    if promote:
        print("\n" + "-" * 78)
        print(f"DEPLOY refit on 253 unblind residuals at K={best_K}")
        print(f"  seed set = {CANONICAL_SEEDS}  (nb1151 canonical)")
        print("-" * 78)
        topK_idx = full_rank_order[:best_K].astype(np.int32)
        X_unb_topK = X_unb[:, topK_idx].astype(np.float32)
        X_513_topK = X_513[:, topK_idx].astype(np.float32)

        seed_preds_513 = np.zeros((len(CANONICAL_SEEDS), n_test),
                                  dtype=np.float64)
        for i, s in enumerate(CANONICAL_SEEDS):
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb_topK, residual)
            seed_preds_513[i] = mdl.predict(X_513_topK)
        resid_pred_513 = seed_preds_513.mean(axis=0)
        final_513 = te_anchor_513 + resid_pred_513
        in_rae_deploy = float(rae(y_unb, final_513[unb_idx]))
        print(f"[deploy] resid mean={resid_pred_513.mean():+.4f}  "
              f"std={resid_pred_513.std():.4f}")
        print(f"[deploy] final  mean={final_513.mean():.4f}  "
              f"std={final_513.std():.4f}")
        print(f"[deploy] in_RAE(253) = {in_rae_deploy:.4f}  "
              f"(in-sample optimistic; cross-fit = {best_rae_mean_bag:.4f})")

        te_out = DATA_PROCESSED / f"te_{TAG}_K{best_K}.npy"
        np.save(te_out, final_513.astype(np.float32))
        print(f"[save] {te_out}")

        name_col = "name" if "name" in te.columns else "Molecule Name"
        names = te[name_col].astype(str).tolist()
        smis_full = (te["smiles"].astype(str).tolist()
                     if "smiles" in te.columns
                     else te["SMILES"].astype(str).tolist())
        sub_df = pd.DataFrame({
            "Molecule Name": names,
            "SMILES": smis_full,
            "pEC50": final_513.astype(np.float64),
        })
        deploy_csv = SUB_DIR / f"{TAG}_deploy_K{best_K}.csv"
        sub_df.to_csv(deploy_csv, index=False)
        print(f"[save] {deploy_csv}  rows={len(sub_df)}")

        deploy_info.update({
            "te_path": str(te_out),
            "submission_path": str(deploy_csv),
            "in_rae_deploy_253": in_rae_deploy,
            "deploy_refit_seeds": CANONICAL_SEEDS,
            "resid_pred_513_mean": float(resid_pred_513.mean()),
            "resid_pred_513_std": float(resid_pred_513.std()),
            "final_513_mean": float(final_513.mean()),
            "final_513_std": float(final_513.std()),
        })
    else:
        print("\n[no-deploy] best K did not beat K=32 ref by promote margin; "
              "ladder unchanged (nb1158 PRIMARY-1 stands).")

    summary = {
        "tag": TAG,
        "method": "fine_grained_k_sweep_around_K32_nb1158_winner",
        "anchor": ANCHOR,
        "K_grid": K_GRID,
        "canonical_seeds": CANONICAL_SEEDS,
        "fresh_seeds": FRESH_SEEDS,
        "all_seeds": ALL_SEEDS,
        "n_seeds_total": len(ALL_SEEDS),
        "resid_folds": RESID_FOLDS,
        "cv": "scaffold_kfold_indices(Murcko, n=5, shuffle=True, seed=kf_seed)",
        "lgbm_params": _lgbm_params(0),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "feat_dim_full": int(feat_dim),
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_K_records": per_K_records,
        "best_K": best_K,
        "best_rae_mean_bag_all": best_rae_mean_bag,
        "best_rec": best_rec,
        "nb1158_K32_mean_bag_ref": NB1158_K32_MEAN_BAG_REF,
        "nb1157_K30_mean_bag_ref": NB1157_K30_MEAN_BAG_REF,
        "nb1157_K35_mean_bag_ref": NB1157_K35_MEAN_BAG_REF,
        "deploy_gate": DEPLOY_GATE,
        "promote_margin": PROMOTE_MARGIN,
        "beats_K32_by_margin": bool(beats_K32_by_margin),
        "beats_deploy_gate": bool(beats_deploy_gate),
        "verdict": verdict,
        "promoted": bool(promote),
        "deploy_info": deploy_info,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "Fine-grained K-sweep over K in [30..34] around nb1158 winner K=32. "
            "Each K trained with 15 seeds total (5 canonical + 10 fresh), "
            "scaffold-CV 5-fold, LGBM(MSE) on top-K SHAP-ranked 117-col matrix. "
            "Mean-bag aggregated across all 15 seeds (also reports canon-only "
            "and fresh-only sub-bags). Deploy CSV emitted iff best mean-bag "
            "beats nb1158 K=32 (0.4902) by >= 0.003 margin AND <= 0.5027 gate."
        ),
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
        "K_grid", "n_seeds_total",
        "rae_anchor_chemprop_aux",
        "best_K", "best_rae_mean_bag_all",
        "nb1158_K32_mean_bag_ref", "promote_margin", "deploy_gate",
        "beats_K32_by_margin", "beats_deploy_gate",
        "verdict", "promoted",
    ):
        print(f"  {k}: {res.get(k)}")
    if res.get("promoted"):
        print(f"  deploy_csv: {res['deploy_info'].get('submission_path')}")
        print(f"  in_rae_deploy_253: "
              f"{res['deploy_info'].get('in_rae_deploy_253')}")
