"""nb2850 -- Per-fold rank-stretch s in [1.0, 1.5] (decompression).

NEW PARADIGM (vs nb2843 shrinkage):
    For each scaffold-CV fold:
      mu_tr     = mean(pred_base[tr_loc])
      s_star    = argmin_{s in [1.0, 1.5]} RAE(y_tr, mu_tr + s * (pred_base[tr_loc] - mu_tr))
      pred_val  = mu_tr + s_star * (pred_base[va_loc] - mu_tr)

This is the dual of nb2843: instead of shrinking predictions toward fold-train
truth mean (alpha in [0, 1] shrinkage), we STRETCH around the fold-train
prediction mean by s in [1.0, 1.5] to decompress variance.  This targets the
documented variance-compression failure on the rare-scaffold tail of nb2240
(pred_std < truth_std).

Selection: per-fold golden-section search minimizing fold-TRAIN RAE on
shifted predictions, then applied to fold-VAL.  This is honest because the
held-out fold-val labels never enter s_star.

Protocol
--------
  1. Load nb2240_mean_bag_oof_K20.npy (253 honest cross-fit) + y_unb (253).
  2. For each kf_seed in {1001, 1002, 1003, 1004, 1005}:
       For each scaffold-CV fold (tr_loc, va_loc):
         golden-section search s in [1.0, 1.5] minimizing RAE on fold train
         apply best s to fold val
       pooled_rae[kf_seed] = rae(y_unb, oof_pred)
     mean_rae = mean(pooled_rae)
  3. Pick best mean_rae across 5 seeds (single global stretch policy).

Gates
-----
  best_mean_rae < 0.4570  -> PROMOTE
  best_mean_rae < 0.4598  -> MARGINAL_BEAT
  else                    -> FAIL

Outputs
-------
  data/processed/nb2850_summary.json
  data/processed/nb2850_pred_oof.npy   (253 unblind, kf_seed=1001 OOF)
  data/processed/te_nb2850.npy         (513 test, deploy with mean-s across folds/seeds)
  submissions/nb2850_per_fold_stretch.csv
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

TAG = "nb2850"
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
S_LO = 1.0
S_HI = 1.5
GS_TOL = 1e-4
GS_MAX_ITER = 60

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
NB2240_TE_PATH = DATA_PROCESSED / "te_nb2240.npy"


def _stretch(pred, mu, s):
    return mu + s * (pred - mu)


def golden_section_min(f, lo, hi, tol=GS_TOL, max_iter=GS_MAX_ITER):
    """Minimize unimodal f on [lo, hi] via golden-section search."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0  # 0.618...
    a, b = float(lo), float(hi)
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    if fc < fd:
        return c, fc
    return d, fd


def cv_run_for_seed(pred_base, y_unb, unb_scaffolds, kf_seed):
    """One scaffold-CV pass: golden-section per fold, apply to fold val.

    Returns (pooled_rae, oof_pred, per_fold_info).
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(pred_base)
    oof_pred = np.full(n_unb, np.nan)
    per_fold = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        p_tr = pred_base[tr_loc]
        y_tr = y_unb[tr_loc]
        mu_tr = float(np.mean(p_tr))

        def f(s, p_tr=p_tr, y_tr=y_tr, mu_tr=mu_tr):
            return float(rae(y_tr, _stretch(p_tr, mu_tr, s)))

        s_star, fold_train_rae = golden_section_min(f, S_LO, S_HI)
        # also record edge values
        f_lo = f(S_LO)
        f_hi = f(S_HI)
        oof_pred[va_loc] = _stretch(pred_base[va_loc], mu_tr, s_star)
        per_fold.append({
            "fold": int(fold_i),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "mu_tr_pred": float(mu_tr),
            "s_star": float(s_star),
            "fold_train_rae_at_s_star": float(fold_train_rae),
            "fold_train_rae_at_s_lo": float(f_lo),
            "fold_train_rae_at_s_hi": float(f_hi),
        })
    assert not np.isnan(oof_pred).any()
    return float(rae(y_unb, oof_pred)), oof_pred, per_fold


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-fold rank-stretch s in [{S_LO}, {S_HI}] on nb2240 OOF")
    print(f"       kf_seeds   = {KF_SEEDS}")
    print(f"       n_folds    = {N_FOLDS}")
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
    assert pred_base.shape == (n_unb,), (
        f"nb2240 OOF expected ({n_unb},) got {pred_base.shape}"
    )
    rae_base = float(rae(y_unb, pred_base))
    print(f"[base] nb2240 OOF mean={pred_base.mean():.4f}  std={pred_base.std():.4f}")
    print(f"[base] nb2240 OOF unshifted RAE = {rae_base:.4f}")
    print(f"[base] variance-compression ratio = "
          f"{pred_base.std() / y_unb.std():.4f} "
          f"(<1.0 => compressed; stretch motivated)")

    # ---- Sweep kf_seeds ----
    print("\n" + "-" * 78)
    print(f"PER-FOLD GOLDEN-SECTION s in [{S_LO}, {S_HI}]  x  {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    seed_results = []
    oof_by_seed = {}
    for kf_seed in KF_SEEDS:
        r, oof_pred, per_fold = cv_run_for_seed(
            pred_base, y_unb, unb_scaffolds, kf_seed,
        )
        s_vals = [f["s_star"] for f in per_fold]
        seed_results.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(r),
            "s_stars": [float(s) for s in s_vals],
            "s_mean": float(np.mean(s_vals)),
            "s_std": float(np.std(s_vals)),
            "pred_va_std": float(oof_pred.std()),
            "pred_va_mean": float(oof_pred.mean()),
            "per_fold": per_fold,
        })
        oof_by_seed[int(kf_seed)] = oof_pred
        print(f"   kf_seed={kf_seed}   pooled_RAE={r:.4f}   "
              f"s_stars={np.round(s_vals, 3).tolist()}   "
              f"s_mean={np.mean(s_vals):.3f}   "
              f"pred_std={oof_pred.std():.4f}")

    raes = np.array([r["pooled_rae"] for r in seed_results])
    mean_rae = float(raes.mean())
    std_rae = float(raes.std())
    min_rae = float(raes.min())
    max_rae = float(raes.max())
    best_seed = int(KF_SEEDS[int(np.argmin(raes))])
    print("\n" + "-" * 78)
    print("AGGREGATE")
    print("-" * 78)
    print(f"   mean RAE   = {mean_rae:.4f}")
    print(f"   std  RAE   = {std_rae:.4f}")
    print(f"   min  RAE   = {min_rae:.4f}  (kf_seed={best_seed})")
    print(f"   max  RAE   = {max_rae:.4f}")
    print(f"   delta_mean_vs_base = {mean_rae - rae_base:+.4f}")
    print(f"   delta_min_vs_base  = {min_rae  - rae_base:+.4f}")

    # ---- Gate (on MEAN across 5 kf_seeds) ----
    if mean_rae < GATE_PROMOTE:
        decision = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        decision = "MARGINAL_BEAT"
    else:
        decision = "FAIL"
    print("\n" + "-" * 78)
    print("GATE  (on mean across 5 kf_seeds)")
    print("-" * 78)
    print(f"   mean_rae       = {mean_rae:.4f}")
    print(f"   PROMOTE  < {GATE_PROMOTE}  ->  {mean_rae < GATE_PROMOTE}")
    print(f"   MARGINAL < {GATE_MARGINAL}  ->  {mean_rae < GATE_MARGINAL}")
    print(f"   decision = {decision}")

    # ---- Save pred_oof at kf_seed=1001 (canonical) ----
    pred_oof_canon = oof_by_seed[1001].astype(np.float32)
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, pred_oof_canon)
    print(f"\n[save] pred_oof @ kf_seed=1001 -> {pred_oof_path}")
    print(f"       shape={pred_oof_canon.shape}  "
          f"RAE_on_y_unb={rae(y_unb, pred_oof_canon):.4f}")

    # ---- Deploy on 513 test ----
    # Use global mean of per-fold per-seed s_stars as deploy stretch factor.
    # mu for deploy = mean(te_base) (single global stretch around test mean).
    all_s = np.array(
        [s for r in seed_results for s in r["s_stars"]], dtype=np.float64,
    )
    s_deploy = float(all_s.mean())
    s_deploy_std = float(all_s.std())
    te_base = np.load(NB2240_TE_PATH).astype(np.float64)
    assert te_base.shape == (n_te,), f"te_nb2240 shape {te_base.shape}"
    mu_te = float(te_base.mean())
    te_deploy = (mu_te + s_deploy * (te_base - mu_te)).astype(np.float32)
    te_deploy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_deploy_path, te_deploy)
    te_unb_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] s_deploy = mean(all per-fold s_stars) = "
          f"{s_deploy:.4f} +/- {s_deploy_std:.4f}  (n={len(all_s)})")
    print(f"[deploy] te_base nb2240: mean={te_base.mean():.4f}  "
          f"std={te_base.std():.4f}")
    print(f"[deploy] te_stretched  : mean={te_deploy.mean():.4f}  "
          f"std={te_deploy.std():.4f}")
    print(f"[deploy] te[unb] in-sample RAE = {te_unb_rae:.4f}")
    print(f"[deploy] te_npy -> {te_deploy_path}")

    # ---- Submission CSV ----
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_deploy,
    })
    sub_path = SUBMISSIONS / f"{TAG}_per_fold_stretch.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)
    print(f"[deploy] submission -> {sub_path}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "method": "per_fold_rank_stretch_golden_section_on_nb2240_oof",
        "anchor": "nb2240_mean_bag_oof_K20",
        "anchor_oof_source": str(NB2240_OOF_PATH),
        "anchor_te_source": str(NB2240_TE_PATH),
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "s_lo": S_LO,
        "s_hi": S_HI,
        "gs_tol": GS_TOL,
        "gs_max_iter": GS_MAX_ITER,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "anchor_unshifted_rae": rae_base,
        "anchor_pred_std": float(pred_base.std()),
        "anchor_truth_std": float(y_unb.std()),
        "anchor_variance_ratio": float(pred_base.std() / y_unb.std()),
        "seed_results": seed_results,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "best_seed": best_seed,
        "delta_mean_vs_base": mean_rae - rae_base,
        "delta_min_vs_base": min_rae - rae_base,
        "s_deploy": s_deploy,
        "s_deploy_std": s_deploy_std,
        "s_deploy_n_samples": int(len(all_s)),
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
    print(f"   anchor unshifted RAE  = {rae_base:.4f}")
    print(f"   s range               = [{S_LO}, {S_HI}]  (golden-section per fold)")
    print(f"   mean RAE (5 seeds)    = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   min  RAE              = {min_rae:.4f}  (kf_seed={best_seed})")
    print(f"   delta mean vs base    = {mean_rae - rae_base:+.4f}")
    print(f"   s_deploy (mean)       = {s_deploy:.4f} +/- {s_deploy_std:.4f}")
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
        "mean_rae",
        "std_rae",
        "min_rae",
        "best_seed",
        "delta_mean_vs_base",
        "s_deploy",
        "decision",
        "te_unb_in_sample_rae",
    ):
        print(f"  {k}: {res.get(k)}")
