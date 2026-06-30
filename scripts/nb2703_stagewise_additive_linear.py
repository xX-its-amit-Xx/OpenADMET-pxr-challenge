"""nb2703 -- Stagewise additive linear correction on LGBM K=20.

NEW PARADIGM (2-stage):
    Stage 1: K=20 LGBM (standard nb2240 config) predicts pEC50 on fold-val.
             Reuses nb2240's K=20 anchor+residual stack (chemprop_aux te[unb_idx]
             anchor + LGBM(MSE) mean-bag over 5 seeds {0,1,7,42,137} on K=20 RFE
             surviving features). The Stage-1 OOF on 253 is
             nb2240_mean_bag_oof_K20.npy; the deploy te(513) prediction is
             te_nb2240_K20.npy.
    Stage 2: residual_stage1 = y - stage1_pred. Fit Ridge(alpha=1.0) on the
             9 physchem features (MW, LogP, TPSA, HBD, HBA, rotbonds, fsp3,
             n_rings, n_aromatic_rings) predicting residual_stage1 under
             scaffold 5-fold CV. Standardize features with StandardScaler fit
             on train-fold only.
    Final  = stage1_pred + Ridge_residual_pred.

PROTOCOL:
    - 5-fold scaffold CV on the 253 unblind compounds across 5 kf_seeds
      {1001..1005} (matches nb2240/nb2171 cv harness).
    - Both Stage-1 and Stage-2 are cross-fit on the same scaffold splits per
      kf_seed (Stage-1 OOF reused from nb2240; Stage-2 Ridge re-fit per fold
      using train-fold indices, so we honor the scaffold split).

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4598  -> MARGINAL_BEAT
    else               -> FAIL

Outputs:
    data/processed/nb2703_summary.json
    data/processed/nb2703_pred_oof.npy   (253,) float32
    data/processed/te_nb2703.npy          (513,) float32
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
from rdkit import RDLogger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, compute_physchem
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2703"

PHYSCHEM_KEYS = [
    "mw", "logp", "tpsa", "hbd", "hba",
    "rotbonds", "fsp3", "n_rings", "n_aromatic_rings",
]

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
RIDGE_ALPHA = 1.0

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

LB_W_OOF = 0.51
LB_W_TE = 0.49

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
STAGE1_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
STAGE1_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"


def _build_physchem_matrix(smiles_list):
    rows = []
    for s in smiles_list:
        pc = compute_physchem(s)
        if pc is None:
            rows.append({k: np.nan for k in PHYSCHEM_KEYS})
        else:
            rows.append({k: float(pc[k]) for k in PHYSCHEM_KEYS})
    df = pd.DataFrame(rows, columns=PHYSCHEM_KEYS)
    X = df.to_numpy(dtype=np.float64)
    # median-impute any failures column-wise
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X.astype(np.float64)


def cv_run_for_seed(stage1_pred_unb, X_unb, y_unb, unb_scaffolds, kf_seed):
    """Stage-1 is given as cross-fit OOF (nb2240_mean_bag_oof_K20).
    Stage-2 Ridge fit on each scaffold-fold train portion, predicted on the
    val portion; final = stage1 + ridge_resid_pred.
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_final = np.full(n_unb, np.nan, dtype=np.float64)
    fold_coefs = []
    for tr_loc, va_loc in splits:
        # Stage-1 residual on train portion
        resid_tr = y_unb[tr_loc] - stage1_pred_unb[tr_loc]
        # Standardize physchem on train-fold only
        scl = StandardScaler()
        X_tr_std = scl.fit_transform(X_unb[tr_loc])
        X_va_std = scl.transform(X_unb[va_loc])
        # Ridge on residual
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, random_state=0)
        ridge.fit(X_tr_std, resid_tr)
        resid_pred_va = ridge.predict(X_va_std)
        oof_final[va_loc] = stage1_pred_unb[va_loc] + resid_pred_va
        fold_coefs.append(ridge.coef_.tolist())
    return float(rae(y_unb, oof_final)), oof_final, fold_coefs


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 2-stage: K=20 LGBM (nb2240) + Ridge on 9 physchem residual")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"] if "smiles" in te.columns else te["SMILES"]).astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique_unb_scaffolds = {n_unique_scaf}")

    # ---------------- Stage 1: reuse nb2240 K=20 cross-fit ----------------
    if not STAGE1_OOF_PATH.exists():
        raise FileNotFoundError(f"missing Stage-1 OOF: {STAGE1_OOF_PATH}")
    if not STAGE1_TE_PATH.exists():
        raise FileNotFoundError(f"missing Stage-1 te:  {STAGE1_TE_PATH}")
    stage1_oof_unb = np.load(STAGE1_OOF_PATH).astype(np.float64)
    stage1_te_513 = np.load(STAGE1_TE_PATH).astype(np.float64)
    assert stage1_oof_unb.shape == (n_unb,), f"stage1 OOF shape {stage1_oof_unb.shape}"
    assert stage1_te_513.shape == (n_test,), f"stage1 te shape  {stage1_te_513.shape}"
    rae_stage1 = float(rae(y_unb, stage1_oof_unb))
    print(f"[stage1] nb2240 K=20 OOF RAE = {rae_stage1:.4f}")

    # ---------------- Build 9 physchem matrix on unb + te ----------------
    print("\n[stage2] building 9 physchem features ...")
    X_unb = _build_physchem_matrix(unb_smiles)
    X_te = _build_physchem_matrix(te_smiles)
    print(f"   X_unb {X_unb.shape}  X_te {X_te.shape}  features={PHYSCHEM_KEYS}")

    # ---------------- 5-fold scaffold CV across 5 kf_seeds ----------------
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  ridge_alpha={RIDGE_ALPHA}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fold_coefs = cv_run_for_seed(
            stage1_oof_unb, X_unb, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_coefs_mean": [float(x) for x in np.mean(fold_coefs, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(f"   kf_seed={kf_seed}  pooled_RAE={pooled:.4f}")
    mean_oof_final = np.mean(np.column_stack(all_oofs), axis=1)
    rae_of_mean_oof = float(rae(y_unb, mean_oof_final))
    mean_rae = float(np.mean([r["pooled_rae"] for r in per_seed]))
    std_rae = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] mean_rae        = {mean_rae:.4f}  (+/- {std_rae:.4f})")
    print(f"[cv] rae_of_mean_oof = {rae_of_mean_oof:.4f}")
    print(f"[cv] stage1 anchor   = {rae_stage1:.4f}  (delta {mean_rae - rae_stage1:+.4f})")

    # ---------------- Deploy: refit Ridge on full 253, predict 513 ----------------
    print("\n" + "-" * 78)
    print("DEPLOY (refit Ridge on full 253 residual; predict 513)")
    print("-" * 78)
    resid_full = y_unb - stage1_oof_unb
    scl_deploy = StandardScaler()
    X_unb_std = scl_deploy.fit_transform(X_unb)
    X_te_std = scl_deploy.transform(X_te)
    ridge_deploy = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, random_state=0)
    ridge_deploy.fit(X_unb_std, resid_full)
    deploy_te = (stage1_te_513 + ridge_deploy.predict(X_te_std)).astype(np.float32)
    in_rae_final = float(rae(y_unb, stage1_oof_unb + ridge_deploy.predict(X_unb_std)))
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"   ridge coefs (std): {[f'{c:+.3f}' for c in ridge_deploy.coef_]}")
    print(f"   ridge intercept  : {ridge_deploy.intercept_:+.4f}")
    print(f"   in-sample RAE    : {in_rae_final:.4f}")
    print(f"   te[unb_idx] RAE  : {te_unb_rae:.4f}")
    print(f"   te(513) mean/std : {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    lb_band_est = LB_W_OOF * mean_rae + LB_W_TE * te_unb_rae
    print(f"\n[LB-band] {LB_W_OOF:.2f}*OOF + {LB_W_TE:.2f}*te_unb = {lb_band_est:.4f}")

    # ---------------- Gate ----------------
    print("\n" + "-" * 78)
    print(f"GATE  promote<{GATE_PROMOTE:.4f}  marginal<{GATE_MARGINAL:.4f}")
    print("-" * 78)
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   mean_rae = {mean_rae:.4f}  -> {verdict}")

    # ---------------- Save artefacts ----------------
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof_final.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "stagewise_additive_linear_on_K20_LGBM",
        "stage1_oof_path": str(STAGE1_OOF_PATH),
        "stage1_te_path": str(STAGE1_TE_PATH),
        "stage1_rae_unb": rae_stage1,
        "physchem_keys": PHYSCHEM_KEYS,
        "ridge_alpha": RIDGE_ALPHA,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_oof": rae_of_mean_oof,
        "delta_vs_stage1": mean_rae - rae_stage1,
        "deploy_ridge_coefs_std": [float(c) for c in ridge_deploy.coef_],
        "deploy_ridge_intercept": float(ridge_deploy.intercept_),
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "lb_band_estimate": lb_band_est,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   stage1 (K=20 LGBM) RAE   = {rae_stage1:.4f}")
    print(f"   2-stage mean_rae         = {mean_rae:.4f}  (+/- {std_rae:.4f})")
    print(f"   delta vs stage1          = {mean_rae - rae_stage1:+.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   LB band est              = {lb_band_est:.4f}")
    print(f"   wall                     = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "stage1_rae_unb",
        "mean_rae",
        "std_rae",
        "delta_vs_stage1",
        "verdict",
        "lb_band_estimate",
        "deploy_ridge_coefs_std",
    ):
        print(f"  {k}: {res.get(k)}")
