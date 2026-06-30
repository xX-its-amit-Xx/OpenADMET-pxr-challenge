"""nb2562 -- LASSO meta-learner on PRE-clean anchor stack.

CONTEXT:
    Recent meta-stack work (LGBM nb1052/nb1281, XGBoost nb2473, CatBoost nb2552)
    is paradigm-matched to tree splits. We test a *sparse linear* meta-learner
    (LASSO with L1) on a small (3-4 col) PRE-clean anchor pool: L1 selection
    happens via the regularizer itself, so the meta picks which anchors carry
    signal vs which collapse to zero. This is a genuine model-class swap from
    every meta tried before.

SUBSTRATE (PRE-clean only):
    - nb2240_mean_bag_oof_K20.npy      (K=20 residual stack on chemprop_aux)
    - nb1133_chemprop_aux_pred_oof.npy (4139 PRE-unblind only)
    - nb1191_pred_oof if available     (deep-30 pyramid)
    - nb2490_pred_oof.npy              (counter_clean, nb730-free)

    nb730/nb562 EXCLUDED (POST contamination chain).
    nb1191 skipped here -- only te_nb1191.npy exists, no honest pred_oof on 253.

PROTOCOL:
    - 5-fold scaffold CV on 253 unblind, 5 kf_seeds {42, 1, 7, 137, 1009}
    - LassoCV(cv=3, max_iter=10000, random_state=42) per fold
    - Deploy: refit LassoCV on all 253, predict te 513
    - Anchor std-normalized before fit (LASSO is scale-sensitive)

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4601  -> MARGINAL_BEAT
    else                -> FAIL
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.linear_model import LassoCV

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2562"
N_FOLDS = 5
KF_SEEDS = [42, 1, 7, 137, 1009]
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# ---- PRE-clean anchors only ----
# Note: nb1191 has only te_nb1191.npy (deploy-refit), no pred_oof on 253 -> SKIPPED.
CANDIDATE_ANCHORS = [
    ("nb2240_K20",   DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
                     DATA_PROCESSED / "te_nb2240_K20.npy",
                     "PRE-clean (K=20 residual stack on chemprop_aux)"),
    ("chemprop_aux", DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
                     DATA_PROCESSED / "te_chemprop_aux.npy",
                     "PRE-clean (4139 PRE-unblind only)"),
    ("nb1191",       DATA_PROCESSED / "nb1191_pred_oof.npy",
                     DATA_PROCESSED / "te_nb1191.npy",
                     "PRE-clean deep-30 pyramid (if pred_oof exists)"),
    ("counter_clean",DATA_PROCESSED / "nb2490_pred_oof.npy",
                     DATA_PROCESSED / "te_nb2490.npy",
                     "PRE-clean counter-assay residual on chemprop_aux (nb730-free)"),
]

LASSO_PARAMS = dict(cv=3, max_iter=10000, random_state=42)


def _zscore_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def _zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LASSO meta-learner stack on PRE-clean anchor pool")
    print("=" * 78)

    # ---- Load ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test} n_unb={n_unb}")

    # ---- Resolve anchors (skip missing) ----
    anchor_names = []
    anchor_provenance = {}
    oof_cols, te_cols = [], []
    anchor_skipped = {}
    for name, oof_path, te_path, prov in CANDIDATE_ANCHORS:
        if not oof_path.exists():
            anchor_skipped[name] = f"pred_oof missing at {oof_path}"
            print(f"   SKIP {name}: pred_oof missing")
            continue
        if not te_path.exists():
            anchor_skipped[name] = f"te missing at {te_path}"
            print(f"   SKIP {name}: te missing")
            continue
        oof = np.load(oof_path).astype(np.float64)
        te_v = np.load(te_path).astype(np.float64)
        if oof.shape[0] != n_unb:
            anchor_skipped[name] = f"shape mismatch oof={oof.shape} expected ({n_unb},)"
            print(f"   SKIP {name}: shape mismatch")
            continue
        if te_v.shape[0] != n_test:
            anchor_skipped[name] = f"shape mismatch te={te_v.shape} expected ({n_test},)"
            print(f"   SKIP {name}: te shape mismatch")
            continue
        anchor_names.append(name)
        anchor_provenance[name] = prov
        oof_cols.append(oof)
        te_cols.append(te_v)
    K = len(anchor_names)
    if K < 2:
        raise RuntimeError(f"Need >=2 anchors, got {K}: {anchor_names}")

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    rae_anchors = {k: float(rae(y_unb, P_unb[:, i]))
                   for i, k in enumerate(anchor_names)}
    print(f"[anchors] K={K}")
    for k in anchor_names:
        print(f"   {k:14s}  unb_RAE={rae_anchors[k]:.4f}  [{anchor_provenance[k]}]")
    if anchor_skipped:
        print(f"[skipped] {anchor_skipped}")

    # ---- Scaffold folds: 5 kf_seeds ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}  n_folds={N_FOLDS}  kf_seeds={KF_SEEDS}")

    # ---- LASSO meta CV ----
    print("\n" + "-" * 78)
    print("LASSO meta-stack CV (5 kf_seeds x 5-fold scaffold)")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_mean_fold = []
    per_seed_alphas = []
    per_seed_nonzero = []
    oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)

    for s_idx, kf_seed in enumerate(KF_SEEDS):
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                        shuffle=True, seed=kf_seed)
        oof_lasso = np.full(n_unb, np.nan, dtype=np.float64)
        fold_rae = []
        fold_alphas = []
        fold_nnz = []
        for f_idx, (tr_loc, va_loc) in enumerate(splits):
            mu, sd = _zscore_fit(P_unb[tr_loc])
            Xtr = _zscore_apply(P_unb[tr_loc], mu, sd)
            Xva = _zscore_apply(P_unb[va_loc], mu, sd)
            model = LassoCV(**LASSO_PARAMS)
            model.fit(Xtr, y_unb[tr_loc])
            pred = model.predict(Xva)
            oof_lasso[va_loc] = pred
            r = float(rae(y_unb[va_loc], pred))
            fold_rae.append(r)
            fold_alphas.append(float(model.alpha_))
            fold_nnz.append(int(np.sum(np.abs(model.coef_) > 1e-10)))
        pooled = float(rae(y_unb, oof_lasso))
        mean_fold = float(np.mean(fold_rae))
        per_seed_pooled.append(pooled)
        per_seed_mean_fold.append(mean_fold)
        per_seed_alphas.append(fold_alphas)
        per_seed_nonzero.append(fold_nnz)
        oof_seed_stack[s_idx] = oof_lasso
        print(f"   kf_seed={kf_seed:5d}  pooled={pooled:.4f}  mean_fold={mean_fold:.4f}  "
              f"mean_nnz={np.mean(fold_nnz):.1f}/{K}  mean_alpha={np.mean(fold_alphas):.4f}")

    # Mean across seeds
    mean_pooled = float(np.mean(per_seed_pooled))
    std_pooled = float(np.std(per_seed_pooled))
    oof_final = oof_seed_stack.mean(axis=0)
    final_pooled_on_seed_mean = float(rae(y_unb, oof_final))

    print(f"\n[wide-seed] mean pooled = {mean_pooled:.4f} +/- {std_pooled:.4f} "
          f"(n_seeds={len(KF_SEEDS)})")
    print(f"[wide-seed] pooled on seed-mean OOF = {final_pooled_on_seed_mean:.4f}")

    # ---- Gate ----
    if mean_pooled < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_pooled < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_pooled {mean_pooled:.4f}  "
          f"(< {GATE_PROMOTE} PROMOTE / < {GATE_MARGINAL} MARGINAL)  ->  {verdict}")

    # ---- Deploy LASSO on all 253 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit LassoCV on all 253, predict 513")
    print("-" * 78)
    mu_full, sd_full = _zscore_fit(P_unb)
    Xtr_full = _zscore_apply(P_unb, mu_full, sd_full)
    Xte_full = _zscore_apply(P_te, mu_full, sd_full)
    model_deploy = LassoCV(**LASSO_PARAMS)
    model_deploy.fit(Xtr_full, y_unb)
    te_pred = model_deploy.predict(Xte_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    deploy_alpha = float(model_deploy.alpha_)
    deploy_coef = {k: float(v) for k, v in zip(anchor_names, model_deploy.coef_)}
    deploy_intercept = float(model_deploy.intercept_)
    deploy_nnz = int(np.sum(np.abs(model_deploy.coef_) > 1e-10))
    print(f"   te mean={te_pred.mean():.3f} std={te_pred.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}  (expected << pooled)")
    print(f"   deploy alpha={deploy_alpha:.5f}  nnz={deploy_nnz}/{K}  intercept={deploy_intercept:.4f}")
    for k, v in deploy_coef.items():
        flag = " (zeroed)" if abs(v) < 1e-10 else ""
        print(f"     coef[{k:14s}] = {v:+.4f}{flag}")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_final.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_lasso_meta_stack.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "lasso_meta_stack_preclean",
        "anchor_pool": anchor_names,
        "anchor_provenance": anchor_provenance,
        "anchor_skipped": anchor_skipped,
        "anchor_in_rae": rae_anchors,
        "lasso_params": LASSO_PARAMS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_mean_fold_rae": per_seed_mean_fold,
        "per_seed_alphas": per_seed_alphas,
        "per_seed_nonzero": per_seed_nonzero,
        "mean_pooled_rae": mean_pooled,
        "std_pooled_rae": std_pooled,
        "pooled_on_seed_mean_oof": final_pooled_on_seed_mean,
        "mean_rae": mean_pooled,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "deploy_alpha": deploy_alpha,
        "deploy_coef": deploy_coef,
        "deploy_intercept": deploy_intercept,
        "deploy_nonzero": deploy_nnz,
        "deploy_zscore_mu": {k: float(v) for k, v in zip(anchor_names, mu_full)},
        "deploy_zscore_sd": {k: float(v) for k, v in zip(anchor_names, sd_full)},
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean pooled RAE     = {mean_pooled:.4f} +/- {std_pooled:.4f}  ({verdict})")
    print(f"   K anchors used      = {K}  ({anchor_names})")
    print(f"   deploy nnz          = {deploy_nnz}/{K}  alpha={deploy_alpha:.5f}")
    print(f"   wall                = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("mean_pooled_rae", "std_pooled_rae", "verdict",
              "K_anchors", "deploy_nonzero", "deploy_alpha",
              "te_unb_in_sample_rae", "submission_csv"):
        print(f"  {k}: {res.get(k)}")
