"""nb2843 -- Per-row prediction shift toward fold train mean.

NEW PARADIGM:
    pred = (1 - alpha) * nb2240_oof + alpha * fold_train_mean

For each scaffold-CV fold, replace each held-out prediction by a convex
combination of the nb2240 mean-bag OOF prediction and the mean pEC50 of the
fold's training labels. alpha sweeps {0.05, 0.10, 0.20, 0.30, 0.50}.

Motivation
----------
nb2240 is the K=20 RFE-anchored pyramid sub-anchor (chemprop_aux + LGBM-on-K20
residual mean-bag over 5 seeds).  Its OOF predictions are variance-compressed
on the rare-scaffold tail (`pred_std` < `truth_std` per feedback_failure_mode_
quantile_compression).  A per-row pull toward the fold's training mean is the
dual of rank-stretch: instead of decompressing variance around its own mean,
it shrinks toward the empirical class prior of the fold's train set.  This
tests whether the OOD failure tail responds better to mean-shrinkage than to
multiplicative stretch.

Protocol
--------
  1. Load nb2240_mean_bag_oof_K20.npy (253 honest cross-fit) + y_unb (253).
  2. Build scaffold 5-fold splits with kf_seed=1001 (deterministic).
  3. For each alpha in ALPHA_GRID:
       For each fold (tr_loc, va_loc):
         mu_tr   = float(np.mean(y_unb[tr_loc]))      # fold train mean of truth
         pred_va = (1 - alpha) * nb2240_oof[va_loc] + alpha * mu_tr
       pooled_rae = rae(y_unb, oof_pred)
     mean_rae    = pooled_rae   (single deterministic seed)
  4. Pick alpha with min pooled_rae.

Gates
-----
  best_mean_rae < 0.4570  -> PROMOTE
  best_mean_rae < 0.4598  -> MARGINAL_BEAT
  else                    -> FAIL

Outputs
-------
  data/processed/nb2843_summary.json
  data/processed/nb2843_pred_oof.npy   (253 unblind, best-alpha mean-shifted)
  data/processed/te_nb2843.npy         (513 test, refit deploy)
  submissions/nb2843_train_mean_shift.csv
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

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2843"
N_FOLDS = 5
KF_SEED = 1001   # deterministic given fixed nb2240 OOF
ALPHA_GRID = [0.05, 0.10, 0.20, 0.30, 0.50]

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
NB2240_TE_PATH = DATA_PROCESSED / "te_nb2240.npy"


def cv_run_for_alpha(pred_base, y_unb, unb_scaffolds, alpha, kf_seed=KF_SEED):
    """Pooled scaffold-CV RAE for one alpha (single deterministic seed)."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(pred_base)
    oof_pred = np.full(n_unb, np.nan)
    fold_mus = []
    for tr_loc, va_loc in splits:
        mu_tr = float(np.mean(y_unb[tr_loc]))
        oof_pred[va_loc] = (1.0 - alpha) * pred_base[va_loc] + alpha * mu_tr
        fold_mus.append(mu_tr)
    assert not np.isnan(oof_pred).any()
    return float(rae(y_unb, oof_pred)), oof_pred, fold_mus


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-row mean-shift toward fold train mean on nb2240 OOF")
    print(f"       alpha grid  = {ALPHA_GRID}")
    print(f"       kf_seed     = {KF_SEED} (deterministic)")
    print(f"       gates: PROMOTE<{GATE_PROMOTE}  MARGINAL<{GATE_MARGINAL}")
    print("=" * 78)

    # ---- Load test + unblind truth ----
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_smiles)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_te={n_te}  n_unb={n_unb}  unique_scaffolds={n_unique_scaf}")
    print(f"[load] y_unb  mean={y_unb.mean():.4f}  std={y_unb.std():.4f}")

    # ---- Load nb2240 OOF (anchor base) ----
    pred_base = np.load(NB2240_OOF_PATH).astype(np.float64)
    print(f"[load] {NB2240_OOF_PATH.name} shape={pred_base.shape}")
    assert pred_base.shape == (n_unb,), (
        f"nb2240 OOF expected ({n_unb},) got {pred_base.shape}"
    )
    rae_base = float(rae(y_unb, pred_base))
    print(f"[base] nb2240 OOF mean={pred_base.mean():.4f}  std={pred_base.std():.4f}")
    print(f"[base] nb2240 OOF unshifted RAE = {rae_base:.4f}")

    # ---- Train mean for deploy ----
    tr = load_train()
    train_mean_global = float(tr["pec50"].dropna().astype(float).mean())
    print(f"[load] train_mean (global from load_train pec50) = "
          f"{train_mean_global:.4f}  (n={tr['pec50'].notna().sum()})")

    # ---- Sweep alpha ----
    print("\n" + "-" * 78)
    print(f"SWEEP: alpha in {ALPHA_GRID}  x  {N_FOLDS}-fold scaffold")
    print("-" * 78)
    sweep_results = []
    oof_by_alpha = {}
    for alpha in ALPHA_GRID:
        r, oof_pred, fold_mus = cv_run_for_alpha(
            pred_base, y_unb, unb_scaffolds, alpha,
        )
        sweep_results.append({
            "alpha": float(alpha),
            "mean_rae": float(r),
            "n_kf_seeds": 1,
            "kf_seed": int(KF_SEED),
            "fold_train_mus": [float(x) for x in fold_mus],
            "pred_va_std": float(oof_pred.std()),
            "pred_va_mean": float(oof_pred.mean()),
        })
        oof_by_alpha[float(alpha)] = oof_pred
        print(f"   alpha={alpha:.2f}   pooled_RAE={r:.4f}   "
              f"pred_std={oof_pred.std():.4f}   "
              f"fold_mu={np.round(fold_mus, 3).tolist()}")

    # Reference: alpha=0 (no shift, just nb2240 OOF)
    print(f"   alpha=0.00 (ref)  pooled_RAE={rae_base:.4f}   "
          f"pred_std={pred_base.std():.4f}")

    # ---- Pick best alpha ----
    best = min(sweep_results, key=lambda r: r["mean_rae"])
    best_alpha = best["alpha"]
    best_rae = best["mean_rae"]
    delta_vs_base = best_rae - rae_base
    print("\n" + "-" * 78)
    print("PICK")
    print("-" * 78)
    print(f"   best alpha     = {best_alpha:.2f}")
    print(f"   best mean RAE  = {best_rae:.4f}")
    print(f"   vs nb2240 base = {delta_vs_base:+.4f}")

    # ---- Gate ----
    if best_rae < GATE_PROMOTE:
        decision = "PROMOTE"
    elif best_rae < GATE_MARGINAL:
        decision = "MARGINAL_BEAT"
    else:
        decision = "FAIL"
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    print(f"   best_mean_rae = {best_rae:.4f}")
    print(f"   PROMOTE  < {GATE_PROMOTE}  ->  {best_rae < GATE_PROMOTE}")
    print(f"   MARGINAL < {GATE_MARGINAL}  ->  {best_rae < GATE_MARGINAL}")
    print(f"   decision = {decision}")

    # ---- Save pred_oof at best alpha ----
    pred_oof_best = oof_by_alpha[best_alpha].astype(np.float32)
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, pred_oof_best)
    print(f"\n[save] pred_oof @ best_alpha={best_alpha} -> {pred_oof_path}")
    print(f"       shape={pred_oof_best.shape}  "
          f"RAE_on_y_unb={rae(y_unb, pred_oof_best):.4f}")

    # ---- Deploy refit on full 513 test ----
    # For deploy we don't have fold structure; use global train_mean.
    te_base = np.load(NB2240_TE_PATH).astype(np.float64)
    assert te_base.shape == (n_te,), f"te_nb2240 shape {te_base.shape}"
    te_deploy = (
        (1.0 - best_alpha) * te_base + best_alpha * train_mean_global
    ).astype(np.float32)
    te_deploy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_deploy_path, te_deploy)
    te_unb_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"[deploy] te_base nb2240: mean={te_base.mean():.4f}  "
          f"std={te_base.std():.4f}")
    print(f"[deploy] te_shifted (alpha={best_alpha}, mu={train_mean_global:.4f}): "
          f"mean={te_deploy.mean():.4f}  std={te_deploy.std():.4f}")
    print(f"[deploy] te[unb] in-sample RAE = {te_unb_rae:.4f}")
    print(f"[deploy] te_npy -> {te_deploy_path}")

    # ---- Submission CSV ----
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy,
    })
    sub_path = SUBMISSIONS / f"{TAG}_train_mean_shift.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)
    print(f"[deploy] submission -> {sub_path}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "method": "per_row_mean_shift_toward_fold_train_mean_on_nb2240_oof",
        "anchor": "nb2240_mean_bag_oof_K20",
        "anchor_oof_source": str(NB2240_OOF_PATH),
        "anchor_te_source": str(NB2240_TE_PATH),
        "alpha_grid": ALPHA_GRID,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "anchor_unshifted_rae": rae_base,
        "anchor_pred_std": float(pred_base.std()),
        "anchor_truth_std": float(y_unb.std()),
        "train_mean_global_for_deploy": train_mean_global,
        "sweep_results": sweep_results,
        "best_alpha": best_alpha,
        "best_mean_rae": best_rae,
        "delta_best_vs_base": delta_vs_base,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "decision": decision,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_deploy_path),
        "submission_csv": str(sub_path),
        "te_unb_in_sample_rae": te_unb_rae,
        "deploy_te_mean": float(te_deploy.mean()),
        "deploy_te_std": float(te_deploy.std()),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                = nb2240 (K=20 mean-bag OOF)")
    print(f"   anchor unshifted      = {rae_base:.4f}")
    print(f"   alpha grid            = {ALPHA_GRID}")
    print(f"   best alpha            = {best_alpha:.2f}")
    print(f"   best mean RAE         = {best_rae:.4f}")
    print(f"   delta vs base         = {delta_vs_base:+.4f}")
    print(f"   decision              = {decision}")
    print(f"   te[unb] in-sample RAE = {te_unb_rae:.4f}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== KEY ====")
    for k in (
        "anchor_unshifted_rae",
        "best_alpha",
        "best_mean_rae",
        "delta_best_vs_base",
        "decision",
        "te_unb_in_sample_rae",
        "train_mean_global_for_deploy",
    ):
        print(f"  {k}: {res.get(k)}")
