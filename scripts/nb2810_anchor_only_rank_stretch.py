"""nb2810 -- Anchor-only rank-stretch sweep on chemprop_aux PRE-clean OOF.

Motivation
----------
Cycle 229: rebuild the simplest possible baseline. After 869+ methods over
145 cycles all converged to a 0.4682-0.4720 floor on top of nb2171's deep-30
blend, this script tests whether a *single anchor* (chemprop_aux, the only
verified-clean PRE-unblind anchor) plus per-fold scalar rank-stretch can
match or beat that floor without any blending machinery.

Protocol
--------
  1. pred_base = chemprop_aux honest OOF on the 253 unblind compounds
     (loaded from oof_chemprop_aux.npy[unb_idx]).
  2. For each s in {0.95, 1.00, 1.05, 1.10, 1.15, 1.20}:
       For each kf_seed in {1116, 1117, 1118, 1119, 1120}:
         For each scaffold fold (5-fold):
           mu = mean(pred_base[tr_loc])     # per-fold cross-fit mean
           pred_va = mu + s * (pred_base[va_loc] - mu)
         pool across folds -> seed_RAE
       mean_RAE_s = mean(seed_RAE across kf_seeds)
  3. Pick s with min mean_RAE; gate against fixed thresholds.

Gates
-----
  best_mean_rae < 0.4570  -> PROMOTE
  best_mean_rae < 0.4598  -> MARGINAL_BEAT
  else                    -> FAIL

Outputs
-------
  data/processed/nb2810_summary.json
  data/processed/nb2810_pred_oof.npy   (253 unblind, best-s stretched)
  data/processed/te_nb2810.npy         (513 test, refit deploy)
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
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2810"
N_FOLDS = 5
KF_BASE_SEEDS = [1116, 1117, 1118, 1119, 1120]
STRETCH_GRID = [0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598


def stretch_cv_per_seed(pred_base, y_unb, unb_scaffolds, kf_seed, s):
    """Pooled scaffold-CV RAE for one (s, kf_seed) on a single anchor."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(pred_base)
    oof_pred = np.full(n_unb, np.nan)
    for tr_loc, va_loc in splits:
        mu = float(np.mean(pred_base[tr_loc]))
        oof_pred[va_loc] = mu + s * (pred_base[va_loc] - mu)
    assert not np.isnan(oof_pred).any()
    return float(rae(y_unb, oof_pred)), oof_pred


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Anchor-only rank-stretch on chemprop_aux")
    print(f"       grid={STRETCH_GRID}")
    print(f"       kf_seeds={KF_BASE_SEEDS}")
    print(f"       gates: PROMOTE<{GATE_PROMOTE}  "
          f"MARGINAL<{GATE_MARGINAL}")
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

    # ---- Load chemprop_aux honest 253-unblind OOF (PRE-unblind clean) ----
    # nb1133_chemprop_aux_pred_oof.npy is the unblind-only honest cross-fit.
    oof_path = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
    pred_base = np.load(oof_path).astype(np.float64)
    print(f"[load] {oof_path.name} shape={pred_base.shape}")
    assert pred_base.shape == (n_unb,), (
        f"expected ({n_unb},) got {pred_base.shape}"
    )
    print(f"[base] pred_base shape={pred_base.shape}  "
          f"mean={pred_base.mean():.4f}  std={pred_base.std():.4f}")
    print(f"[base] truth: mean={y_unb.mean():.4f}  std={y_unb.std():.4f}")
    print(f"[base] anchor-only RAE (no stretch) = "
          f"{rae(y_unb, pred_base):.4f}")

    # ---- Sweep ----
    print("\n" + "-" * 78)
    print(f"SWEEP: s in {STRETCH_GRID}  x  {len(KF_BASE_SEEDS)} kf_seeds  "
          f"x  {N_FOLDS}-fold scaffold")
    print("-" * 78)
    sweep_results = []
    best_oof_by_s = {}
    for s in STRETCH_GRID:
        per_seed_rae = []
        per_seed_oof = []
        for kf_seed in KF_BASE_SEEDS:
            r, oof_pred = stretch_cv_per_seed(
                pred_base, y_unb, unb_scaffolds, kf_seed, s,
            )
            per_seed_rae.append(r)
            per_seed_oof.append(oof_pred)
        per_seed_rae = np.asarray(per_seed_rae)
        m = float(per_seed_rae.mean())
        sd = float(per_seed_rae.std(ddof=1))
        se = sd / np.sqrt(len(per_seed_rae))
        sweep_results.append({
            "s": float(s),
            "mean_rae": m,
            "std_ddof1": sd,
            "se": se,
            "n_kf_seeds": len(KF_BASE_SEEDS),
            "per_seed_rae": [float(x) for x in per_seed_rae.tolist()],
        })
        # Mean OOF across kf_seeds for this s (for diagnostic + deploy)
        best_oof_by_s[float(s)] = np.mean(per_seed_oof, axis=0)
        print(f"   s={s:.2f}   mean_RAE={m:.4f}  std={sd:.4f}  se={se:.4f}")

    # ---- Pick best s ----
    best = min(sweep_results, key=lambda r: r["mean_rae"])
    best_s = best["s"]
    best_rae = best["mean_rae"]
    baseline_s1 = next(
        (r for r in sweep_results if abs(r["s"] - 1.00) < 1e-6), None
    )
    delta_vs_s1 = best_rae - (
        baseline_s1["mean_rae"] if baseline_s1 else float("nan")
    )
    print("\n" + "-" * 78)
    print("PICK")
    print("-" * 78)
    print(f"   best s         = {best_s:.2f}")
    print(f"   best mean RAE  = {best_rae:.4f}")
    print(f"   vs s=1.00      = {delta_vs_s1:+.4f}")

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

    # ---- Save pred_oof at best s ----
    pred_oof_best = best_oof_by_s[best_s].astype(np.float32)
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, pred_oof_best)
    print(f"\n[save] pred_oof @ best_s={best_s} -> {pred_oof_path}")
    print(f"       shape={pred_oof_best.shape}  "
          f"RAE_on_y_unb={rae(y_unb, pred_oof_best):.4f}")

    # ---- Deploy refit on full 513 test ----
    te_path = DATA_PROCESSED / "te_chemprop_aux.npy"
    te_base = np.load(te_path).astype(np.float64)
    assert te_base.shape == (n_te,), f"te_chemprop_aux shape {te_base.shape}"
    mu_te = float(te_base.mean())
    te_deploy = (mu_te + best_s * (te_base - mu_te)).astype(np.float32)
    te_deploy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_deploy_path, te_deploy)
    te_unb_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"[deploy] te_base: mean={te_base.mean():.4f}  "
          f"std={te_base.std():.4f}")
    print(f"[deploy] te_stretched (s={best_s}): mean={te_deploy.mean():.4f}  "
          f"std={te_deploy.std():.4f}")
    print(f"[deploy] te[unb] in-sample RAE = {te_unb_rae:.4f}")
    print(f"[deploy] te_npy -> {te_deploy_path}")

    # ---- Submission CSV ----
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy,
    })
    sub_path = SUBMISSIONS / f"{TAG}_anchor_only_stretch.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)
    print(f"[deploy] submission -> {sub_path}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "method": "chemprop_aux_anchor_only_rank_stretch_sweep",
        "anchor": "chemprop_aux",
        "anchor_oof_source": "data/processed/nb1133_chemprop_aux_pred_oof.npy",
        "anchor_te_source": "data/processed/te_chemprop_aux.npy",
        "stretch_grid": STRETCH_GRID,
        "kf_base_seeds": KF_BASE_SEEDS,
        "n_kf_seeds": len(KF_BASE_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "anchor_unstretched_rae": float(rae(y_unb, pred_base)),
        "anchor_pred_std": float(pred_base.std()),
        "anchor_truth_std": float(y_unb.std()),
        "sweep_results": sweep_results,
        "best_s": best_s,
        "best_mean_rae": best_rae,
        "best_std_ddof1": best["std_ddof1"],
        "best_se": best["se"],
        "delta_best_vs_s1": delta_vs_s1,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "decision": decision,
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_deploy_path),
        "submission_csv": str(sub_path),
        "te_unb_in_sample_rae": te_unb_rae,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                = chemprop_aux (PRE-clean)")
    print(f"   anchor unstretched    = {summary['anchor_unstretched_rae']:.4f}")
    print(f"   sweep grid            = {STRETCH_GRID}")
    print(f"   best s                = {best_s:.2f}")
    print(f"   best mean RAE         = {best_rae:.4f}  "
          f"(std {best['std_ddof1']:.4f})")
    print(f"   delta vs s=1          = {delta_vs_s1:+.4f}")
    print(f"   decision              = {decision}")
    print(f"   te[unb] in-sample RAE = {te_unb_rae:.4f}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== KEY ====")
    for k in (
        "anchor_unstretched_rae",
        "best_s",
        "best_mean_rae",
        "best_std_ddof1",
        "delta_best_vs_s1",
        "decision",
        "te_unb_in_sample_rae",
    ):
        print(f"  {k}: {res.get(k)}")
