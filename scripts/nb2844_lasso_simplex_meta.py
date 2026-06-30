"""nb2844 -- Lasso meta with explicit simplex constraint (positive + sum-1 renorm).

NEW PARADIGM (vs cycle-134 paradigm exhaustion + nb2562 LASSO):
    Prior LASSO meta-stack (nb2562) used LassoCV unconstrained: coefficients
    were free to be positive or negative and were NOT renormalized. With L1
    regularization on a tiny 3-anchor pool the optimum was a sparse linear
    combination that did NOT respect the simplex (sum != 1, no non-negativity
    enforcement); when the meta picks negative weights on a regression target
    it implicitly does covariance-style debiasing which has no biological
    interpretation for an ensemble of *correlated* PXR pEC50 predictors.

    nb2844 swaps in Lasso(alpha=0.1, positive=True) -> coefficients are forced
    >= 0, then we explicitly RE-NORMALIZE the coefficients to sum to 1
    (projecting the positive-Lasso solution onto the unit simplex). This is
    the smallest capacity-add over SLSQP simplex: SLSQP minimises pure MSE
    on the simplex, while positive-Lasso + renorm minimises (MSE + alpha *
    L1) and the L1 acts as a dispersion regularizer that pulls weights
    toward zero before renorm, biasing the final simplex point toward the
    sparsest convex combination consistent with the data. Different inductive
    bias from SLSQP simplex (which has no sparsity prior) and different from
    nb2562 LassoCV (which has no simplex constraint).

SUBSTRATE (PRE-clean only, 3 anchors -- same as nb2820):
    - nb2240_K20      (K=20 residual stack on chemprop_aux)
    - chemprop_aux    (nb1133, 4139 PRE-unblind only)
    - counter_clean   (nb2490 counter-assay residual on chemprop_aux,
                       nb730-free)

    nb730/nb562/nb503 EXCLUDED (POST contamination chain / not PRE-clean).

PROTOCOL:
    - 5-fold scaffold CV on 253 unblind, 5 kf_seeds {42, 1, 7, 137, 1009}
    - Per fold:
        Lasso(alpha=0.1, positive=True, max_iter=10000) fit on standardized
        anchor stack -> coef on standardized scale
        Map standardized-coef back to raw-anchor weights via dividing by the
        train-fold per-anchor std, then clip negatives to zero (safety) and
        renormalize coefs to sum=1.
        Intercept absorbed by post-renorm calibration: after simplex weights
        are fixed, fit a single scalar bias = mean(y_tr) - mean(P_tr @ w)
        on train, applied to validation predictions.
    - Deploy: refit on all 253, predict te 513

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4598  -> MARGINAL_BEAT
    else                -> FAIL

Outputs:
    scripts/nb2844_lasso_simplex_meta.py
    data/processed/nb2844_summary.json
    data/processed/nb2844_pred_oof.npy   (253,) float32
    data/processed/te_nb2844.npy         (513,) float32
    submissions/nb2844_lasso_simplex_meta.csv
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
from sklearn.linear_model import Lasso

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2844"
N_FOLDS = 5
KF_SEEDS = [42, 1, 7, 137, 1009]
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

LASSO_ALPHA = 0.1
LASSO_MAX_ITER = 10000

# ---- PRE-clean anchors only (3 required) ----
CANDIDATE_ANCHORS = [
    ("nb2240_K20",   DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
                     DATA_PROCESSED / "te_nb2240_K20.npy",
                     "PRE-clean (K=20 residual stack on chemprop_aux)"),
    ("chemprop_aux", DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
                     DATA_PROCESSED / "te_chemprop_aux.npy",
                     "PRE-clean (4139 PRE-unblind only)"),
    ("counter_clean",DATA_PROCESSED / "nb2490_pred_oof.npy",
                     DATA_PROCESSED / "te_nb2490.npy",
                     "PRE-clean counter-assay residual on chemprop_aux (nb730-free)"),
]


def fit_lasso_positive_simplex(P_tr: np.ndarray, y_tr: np.ndarray, alpha: float):
    """Fit Lasso(positive=True) on standardized anchors, then map back to
    raw-anchor weights and project onto the unit simplex (clip neg + renorm).

    Returns (w_simplex, bias, raw_coef_pre_renorm, intercept_lasso, mu, sd)
    """
    mu = P_tr.mean(axis=0)
    sd = P_tr.std(axis=0)
    sd_safe = np.where(sd < 1e-12, 1.0, sd)
    P_tr_std = (P_tr - mu) / sd_safe
    lasso = Lasso(alpha=alpha, positive=True, max_iter=LASSO_MAX_ITER,
                  fit_intercept=True, random_state=42)
    lasso.fit(P_tr_std, y_tr)
    coef_std = lasso.coef_.astype(np.float64)  # on standardized scale
    intercept_lasso = float(lasso.intercept_)
    # Map standardized-coef back to raw-anchor weights:
    #   pred = intercept + sum_j coef_std[j] * (P[:,j] - mu[j]) / sd_safe[j]
    # Define raw_coef[j] = coef_std[j] / sd_safe[j]
    raw_coef = coef_std / sd_safe
    # Project onto simplex: clip negatives to 0 then renormalize to sum=1
    w = np.clip(raw_coef, 0.0, None)
    s = float(w.sum())
    if s < 1e-12:
        # Degenerate: Lasso zeroed everything -> equal weights fallback
        w = np.full_like(raw_coef, 1.0 / len(raw_coef))
        degen = True
    else:
        w = w / s
        degen = False
    # Post-renorm scalar bias: shift blend so mean matches y_tr mean
    blend_tr = P_tr @ w
    bias = float(y_tr.mean() - blend_tr.mean())
    return w, bias, raw_coef, intercept_lasso, mu, sd_safe, degen


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Lasso(alpha=0.1, positive=True) + simplex-renorm meta")
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

    # ---- Resolve anchors (strict: need all 3) ----
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
    if K < 3:
        raise RuntimeError(f"Need 3 PRE-clean anchors, got {K}: {anchor_names}")

    P_unb = np.column_stack(oof_cols)  # (253, K)
    P_te = np.column_stack(te_cols)    # (513, K)
    rae_anchors = {k: float(rae(y_unb, P_unb[:, i]))
                   for i, k in enumerate(anchor_names)}
    print(f"[anchors] K={K}  alpha={LASSO_ALPHA}")
    for k in anchor_names:
        print(f"   {k:14s}  unb_RAE={rae_anchors[k]:.4f}  [{anchor_provenance[k]}]")
    if anchor_skipped:
        print(f"[skipped] {anchor_skipped}")

    # ---- Scaffold folds ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}  n_folds={N_FOLDS}  kf_seeds={KF_SEEDS}")

    # ---- Lasso-positive + simplex-renorm CV ----
    print("\n" + "-" * 78)
    print("LASSO(positive=True) + SIMPLEX-RENORM CV (5 kf_seeds x 5-fold scaffold)")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_mean_fold = []
    per_seed_w_mean = []
    per_seed_bias_mean = []
    per_seed_degen_frac = []
    oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)

    for s_idx, kf_seed in enumerate(KF_SEEDS):
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                        shuffle=True, seed=kf_seed)
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        fold_rae, fold_w, fold_bias, fold_degen = [], [], [], []
        for f_idx, (tr_loc, va_loc) in enumerate(splits):
            w_f, bias_f, raw_coef, intc_lasso, mu, sd_safe, degen = \
                fit_lasso_positive_simplex(P_unb[tr_loc], y_unb[tr_loc],
                                           alpha=LASSO_ALPHA)
            blend_va = P_unb[va_loc] @ w_f + bias_f
            oof[va_loc] = blend_va
            r = float(rae(y_unb[va_loc], blend_va))
            fold_rae.append(r)
            fold_w.append(w_f)
            fold_bias.append(bias_f)
            fold_degen.append(int(degen))
        pooled = float(rae(y_unb, oof))
        mean_fold = float(np.mean(fold_rae))
        per_seed_pooled.append(pooled)
        per_seed_mean_fold.append(mean_fold)
        w_mean = np.mean(np.column_stack(fold_w), axis=1)
        per_seed_w_mean.append(w_mean.tolist())
        per_seed_bias_mean.append(float(np.mean(fold_bias)))
        per_seed_degen_frac.append(float(np.mean(fold_degen)))
        oof_seed_stack[s_idx] = oof
        print(f"   kf_seed={kf_seed:5d}  pooled={pooled:.4f}  "
              f"mean_fold={mean_fold:.4f}  w_mean={np.round(w_mean, 3).tolist()}  "
              f"bias_mean={np.mean(fold_bias):+.3f}  degen={np.mean(fold_degen):.2f}")

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

    # ---- Deploy: refit on all 253, predict 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit Lasso(positive)+simplex-renorm on all 253, predict 513")
    print("-" * 78)
    w_deploy, bias_deploy, raw_coef_deploy, intc_deploy_lasso, mu_deploy, sd_deploy, degen_deploy = \
        fit_lasso_positive_simplex(P_unb, y_unb, alpha=LASSO_ALPHA)
    te_pred = (P_te @ w_deploy + bias_deploy).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy raw_coef (pre-renorm) = {[float(v) for v in raw_coef_deploy]}")
    print(f"   deploy w (simplex)           = {[float(v) for v in w_deploy]}")
    print(f"   deploy bias                  = {bias_deploy:+.4f}")
    print(f"   deploy degenerate (all-zero) = {bool(degen_deploy)}")
    print(f"   te mean={te_pred.mean():.3f} std={te_pred.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE    = {te_unb_in:.4f}  (expected << pooled)")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_final.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_lasso_simplex_meta.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "lasso_positive_alpha_0p1_then_simplex_renorm_meta",
        "anchor_pool": anchor_names,
        "anchor_provenance": anchor_provenance,
        "anchor_skipped": anchor_skipped,
        "anchor_in_rae": rae_anchors,
        "lasso_alpha": LASSO_ALPHA,
        "lasso_max_iter": LASSO_MAX_ITER,
        "lasso_positive": True,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_mean_fold_rae": per_seed_mean_fold,
        "per_seed_w_mean": per_seed_w_mean,
        "per_seed_bias_mean": per_seed_bias_mean,
        "per_seed_degen_frac": per_seed_degen_frac,
        "mean_pooled_rae": mean_pooled,
        "std_pooled_rae": std_pooled,
        "pooled_on_seed_mean_oof": final_pooled_on_seed_mean,
        "mean_rae": mean_pooled,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "deploy_raw_coef_pre_renorm": [float(v) for v in raw_coef_deploy],
        "deploy_w_simplex": [float(v) for v in w_deploy],
        "deploy_bias": float(bias_deploy),
        "deploy_intercept_lasso": float(intc_deploy_lasso),
        "deploy_degenerate": bool(degen_deploy),
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
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
    print(f"   deploy w (simplex)  = {[round(float(v), 4) for v in w_deploy]}")
    print(f"   deploy bias         = {bias_deploy:+.4f}")
    print(f"   wall                = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("mean_pooled_rae", "std_pooled_rae", "verdict",
              "K_anchors", "deploy_w_simplex", "te_unb_in_sample_rae",
              "submission_csv"):
        print(f"  {k}: {res.get(k)}")
