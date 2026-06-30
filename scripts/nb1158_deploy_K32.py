"""nb1158 -- DEPLOY artifact for K=32 winner of nb1157 verify K-sweep.

CYCLE 143 CONTEXT:
    nb1157 (verify of nb1151's K-sweep claim) found that K=35 is NOT the
    scaffold-CV optimum -- K=32 wins with rae_mean_bag = 0.4902 (mean over
    fresh seeds 1001-1010), beating the deploy_gate 0.5027 by -0.013 and
    beating both K=30 (0.4907) and K=35 (0.4937).

    nb1151 only emits a deploy CSV for K=best within its hard-coded K_GRID
    {15, 20, 28, 35, 50} (skipping K=32). nb1158 closes that gap by running
    the IDENTICAL nb1151 protocol with K_GRID=[32] only and emitting the
    deploy CSV: submissions/nb1158_deploy_K32.csv.

PROTOCOL (verbatim nb1151 / nb1157):
    - Anchor: te_chemprop_aux.npy (PRE-unblind)
    - Feature stack: 117-col 5-way K-tuned matrix
      (AtomPair + MACCS + Mordred + ChempropEmbed + Avalon + ChEMBL_kNN)
    - SHAP ranking: data/processed/nb2063_shap_importance_full117.npy
    - LGBM(MSE): max_depth=4, num_leaves=15, n_estimators=300, lr=0.03,
      min_child_samples=5, reg_lambda=2.0
    - Scaffold KFold (n=5) cross-fit per seed, mean-bag across 5 seeds
    - Seed set: nb1157 fresh seeds {1001..1010} for diagnostic, deploy refit
      on ALL 253 unblind residuals at the FIXED nb1151 seed set
      {0, 1, 7, 42, 137} for parity with the K-sweep deploy contract.

DEPLOY GATE: K=32 mean_bag 0.4902 < 0.5027 (=0.5057 - 0.003) -> PASS
    nb1158 builds the deploy CSV unconditionally (gate already cleared in nb1157).

Outputs:
    data/processed/nb1158_mean_bag_oof_K32.npy        (253,) float32
    data/processed/te_nb1158.npy                      (513,) float32
    submissions/nb1158_deploy_K32.csv                 (Molecule Name, SMILES, pEC50)
    data/processed/nb1158_summary.json
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

# Reuse nb1151's loading + ranking pipeline (avoid duplicating ChEMBL/Mordred logic)
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

TAG = "nb1158"
K_DEPLOY = 32
NB1157_FRESH_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]

SUB_DIR = ROOT / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K={K_DEPLOY} deploy (replaces REJECTED nb1151_scaffold_K35)")
    print(f"          cycle-143 nb1157 fresh-optimal: K=32 mean_bag = 0.4902")
    print(f"          deploy gate = {DEPLOY_GATE:.4f}  -> PASS (-0.013)")
    print("=" * 78)

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

    # Scaffolds for the 253 unblind compounds
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_mols = [standardize(s) for s in unb_smiles]
    scaffolds_unb = [_murcko_scaffold(m) for m in unb_mols]
    n_unique_sc = len({s for s in scaffolds_unb if s})
    print(f"[scaf] n_unique_scaffolds = {n_unique_sc}")

    # Reuse the 5 K-tuned summary refs
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

    # Feature matrices (full 513 + unblind slice)
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    print(f"[feat] AP={X_ap_te.shape}  MACCS={X_maccs_te.shape}  "
          f"Mord={X_mord_te.shape}  Embed={X_emb_te.shape}  Av={X_av_te.shape}")

    # ChEMBL kNN feature (same as nb1151)
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
    pred_chembl_pec50, mean_sim = _knn_predict(top_idx_knn, top_sim_knn,
                                               pool_labels, fallback=pool_median)
    print(f"[chembl] pool size {len(pool)}  median pEC50 {pool_median:.3f}")

    # Build 117-col matrices (unblind + full)
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
        raise ValueError(f"feat_dim {feat_dim} != SHAP rank {shap_imp_full117.shape[0]}")

    X_513 = np.concatenate([
        X_ap_te[:, top_ap_bit_idx],
        X_maccs_te[:, top_maccs_bit_idx],
        X_mord_te[:, top_mord_col_idx],
        X_emb_te[:, top_embed_col_idx],
        X_av_te[:, top_avalon_bit_idx],
        pred_chembl_pec50.reshape(-1, 1),
        mean_sim.reshape(-1, 1),
    ], axis=1).astype(np.float32)

    # Slice to K=32 SHAP top
    topK_idx = full_rank_order[:K_DEPLOY].astype(np.int32)
    X_unb_topK = X_unb[:, topK_idx].astype(np.float32)
    X_513_topK = X_513[:, topK_idx].astype(np.float32)
    print(f"[slice] X_unb_K32={X_unb_topK.shape}  X_513_K32={X_513_topK.shape}")

    # --- Scaffold-CV cross-fit diagnostic at K=32 with nb1157 fresh seeds ---
    print("\n" + "-" * 78)
    print(f"DIAGNOSTIC scaffold-CV K={K_DEPLOY} (fresh seeds nb1157 set)")
    print("-" * 78)
    per_seed_corr = np.zeros((len(NB1157_FRESH_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    for i, s in enumerate(NB1157_FRESH_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_scaffold_cross_fit_one_seed(
            X_unb_topK, residual, scaffolds_unb, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corr[i] = pred_corr_s
        r = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(r)
        print(f"   seed {s:>4d}: rae = {r:.4f}  wall = {time.time()-ts:.1f}s")
    mean_bag_oof = per_seed_corr.mean(axis=0)
    median_bag_oof = np.median(per_seed_corr, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    rae_per_seed_arr = np.array(per_seed_rae)
    print(f"\n[diag] K={K_DEPLOY} mean_bag    RAE = {rae_mean_bag:.4f}")
    print(f"[diag] K={K_DEPLOY} median_bag  RAE = {rae_median_bag:.4f}")
    print(f"[diag] per-seed mean = {rae_per_seed_arr.mean():.4f}  "
          f"std = {rae_per_seed_arr.std():.4f}")
    print(f"[diag] vs deploy_gate {DEPLOY_GATE:.4f}: "
          f"{'PASS' if rae_mean_bag < DEPLOY_GATE else 'FAIL'} "
          f"({rae_mean_bag - DEPLOY_GATE:+.4f})")

    out_oof = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{K_DEPLOY}.npy"
    np.save(out_oof, mean_bag_oof.astype(np.float32))
    print(f"[save] {out_oof}")

    # --- Deploy: refit on ALL 253 residuals with nb1151 canonical seed set ---
    print("\n" + "-" * 78)
    print(f"DEPLOY refit on 253 unblind residuals at K={K_DEPLOY}")
    print(f"  seed set = {RESID_SEEDS}  (nb1151 canonical)")
    print("-" * 78)
    seed_preds_513 = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    for i, s in enumerate(RESID_SEEDS):
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb_topK, residual)
        seed_preds_513[i] = mdl.predict(X_513_topK)
    resid_pred_513 = seed_preds_513.mean(axis=0)
    final_513 = te_anchor_513 + resid_pred_513
    print(f"[deploy] resid mean={resid_pred_513.mean():+.4f}  "
          f"std={resid_pred_513.std():.4f}")
    print(f"[deploy] final  mean={final_513.mean():.4f}  std={final_513.std():.4f}")
    in_rae_deploy = float(rae(y_unb, final_513[unb_idx]))
    print(f"[deploy] in_RAE(253) = {in_rae_deploy:.4f}  "
          f"(in-sample; cross-fit is {rae_mean_bag:.4f})")

    te_out = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_out, final_513.astype(np.float32))
    print(f"[save] {te_out}")

    # CSV
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
    deploy_csv = SUB_DIR / f"{TAG}_deploy_K{K_DEPLOY}.csv"
    sub_df.to_csv(deploy_csv, index=False)
    print(f"[save] {deploy_csv}  rows={len(sub_df)}")

    summary = {
        "tag": TAG,
        "method": f"lgbm_mse_scaffold_KFold_K={K_DEPLOY}_deploy_from_nb1151_pipeline",
        "anchor": ANCHOR,
        "cycle": 143,
        "replaces": "nb1151_scaffold_K35.csv (REJECTED per nb1157 verify)",
        "K_deploy": K_DEPLOY,
        "fresh_seeds_diagnostic": NB1157_FRESH_SEEDS,
        "deploy_refit_seeds": RESID_SEEDS,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "feat_dim_full": int(feat_dim),
        "rae_anchor_chemprop_aux": rae_anchor,
        "rae_mean_bag_fresh": rae_mean_bag,
        "rae_median_bag_fresh": rae_median_bag,
        "rae_per_seed_mean_fresh": float(rae_per_seed_arr.mean()),
        "rae_per_seed_std_fresh": float(rae_per_seed_arr.std()),
        "rae_per_seed_fresh": per_seed_rae,
        "nb1157_claim_K32_mean_bag": 0.4902,
        "delta_vs_nb1157_claim": rae_mean_bag - 0.4902,
        "deploy_gate": DEPLOY_GATE,
        "beats_deploy_gate": bool(rae_mean_bag < DEPLOY_GATE),
        "nb2103_K28_scaffold_ref": NB2103_K28_SCAFFOLD_REF,
        "delta_vs_nb2103_K28_scaffold": rae_mean_bag - NB2103_K28_SCAFFOLD_REF,
        "in_rae_deploy_253": in_rae_deploy,
        "te_path": str(te_out),
        "submission_path": str(deploy_csv),
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "DEPLOY artefact for K=32 (nb1157 fresh-optimal). Diagnostic RAE "
            "is honest scaffold-CV cross-fit on 10 fresh seeds; deploy 513 "
            "vector is mean of 5 LGBM(MSE) refits on ALL 253 unblind residuals "
            "at the nb1151 canonical seed set {0,1,7,42,137} for parity. "
            "in_RAE on 253 is in-sample optimistic; LB-faithful number is "
            "rae_mean_bag_fresh."
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
        "K_deploy", "rae_anchor_chemprop_aux",
        "rae_mean_bag_fresh", "rae_median_bag_fresh",
        "rae_per_seed_mean_fresh", "rae_per_seed_std_fresh",
        "delta_vs_nb1157_claim", "deploy_gate",
        "beats_deploy_gate", "delta_vs_nb2103_K28_scaffold",
        "in_rae_deploy_253", "submission_path",
    ):
        print(f"  {k}: {res.get(k)}")
