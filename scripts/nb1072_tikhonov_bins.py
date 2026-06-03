"""nb1072 -- Tikhonov-regularized per-quantile-bin stretch on te_nb1014.

nb1053 fit one s_b per quantile bin (5 bins) and grid-scanned each on absolute
error. Across 5 folds the per-bin s averaged something like [1.93, 1.05, 0.95,
1.00, 0.80] -- tail bins (bin 0 and bin 4) were pushed far from 1.0 with ~50
points each, the prime overfit regime. nb1060 bagged 5 seeds and got
0.5798 pooled cross-fit; we want to test whether shrinking the per-bin
stretches toward identity (s_b -> 1.0) reduces the variance penalty.

Per-fold objective per bin b:
    L_b(s_b) = sum_{i in bin b} (y_i - (mu_b + s_b * (p_i - mu_b)))^2
             + lambda * (s_b - 1.0)^2

This decouples bins (each s_b is independent) so the closed-form per-bin
solution is straightforward least squares with an L2 anchor at 1.0:
    Let x_i = p_i - mu_b, r_i = y_i - mu_b.
    L_b(s) = sum(r_i - s*x_i)^2 + lambda*(s - 1)^2
    dL/ds = -2 sum x_i (r_i - s x_i) + 2 lambda (s - 1) = 0
    s_b = (sum(x_i r_i) + lambda) / (sum(x_i^2) + lambda)

(Note: nb1053 used absolute-error grid; switching to squared-error is needed
for a closed-form Tikhonov solution. The bin centers (mu_b) and bin edges
are still set from train-fold preds only.)

Lambda grid {0.0, 0.5, 1.0, 5.0, 10.0}.  lambda=0 = unregularized (≈ nb1053
LS variant).  Larger lambda pulls s_b toward 1.0 (identity), shrinking the
tail-bin amplification.

Cross-fit protocol (matches nb1053 / nb1060):
  - 5-fold KFold(seed=42) on the 253 unblind.
  - For each lambda:
      a. Train-fold edges = quantile(p_tr, [0.2, 0.4, 0.6, 0.8]).
      b. For each bin: closed-form s_b on train fold with this lambda.
      c. Apply per-bin (mu_b, s_b) to held-out fold using TRAIN edges.
      d. Pool 5 fold predictions and compute pooled RAE.
  - Pick lambda with lowest pooled RAE.
  - Refit on all 253 at chosen lambda for deploy.

Outputs:
  data/processed/te_nb1072.npy
  data/processed/nb1072_summary.json
  submissions/nb1072_tikhonov_bins.csv
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
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1072"
ANCHOR = "nb1014"
N_BINS = 5
LAMBDA_GRID = [0.0, 0.5, 1.0, 5.0, 10.0]
N_FOLDS = 5
SEED = 42

# Honest references from prior cross-fit work.
NB1014_BAGGED_HONEST_RAE = 0.5930
NB1053_HONEST_RAE = 0.5780      # nb1053 per-quantile stretch, seed42
NB1060_BAGGED_RAE = 0.5798      # nb1060 5-seed bagged per-quantile

# ----------------------------------------------------------------------
# Bin assignment + closed-form Tikhonov per bin
# ----------------------------------------------------------------------


def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin assignment 0..N_BINS-1 from internal edges of length N_BINS-1."""
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def fit_per_bin_tikhonov(p_train: np.ndarray, y_train: np.ndarray,
                         edges: np.ndarray, lam: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Tikhonov-regularized per-bin stretch.

    For each bin b solve
        s_b = (sum(x_i r_i) + lam) / (sum(x_i^2) + lam)
    where x_i = p_i - mu_b and r_i = y_i - mu_b on train-fold points in bin b.
    Empty / degenerate bins fall back to s_b = 1.0.
    """
    bins_tr = bin_assign(p_train, edges)
    mus = np.zeros(N_BINS, dtype=np.float64)
    ss = np.ones(N_BINS, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins_tr == b
        n_b = int(mask.sum())
        if n_b < 2:
            mus[b] = float(p_train.mean()) if n_b == 0 else float(p_train[mask][0])
            ss[b] = 1.0
            continue
        p_b = p_train[mask]
        y_b = y_train[mask]
        mu_b = float(p_b.mean())
        mus[b] = mu_b
        x = p_b - mu_b
        r = y_b - mu_b
        num = float((x * r).sum()) + lam
        den = float((x * x).sum()) + lam
        if den <= 1e-12:
            ss[b] = 1.0
        else:
            ss[b] = num / den
    return mus, ss


def apply_per_bin_stretch(p: np.ndarray, edges: np.ndarray,
                          mus: np.ndarray, ss: np.ndarray) -> np.ndarray:
    bins = bin_assign(p, edges)
    out = np.empty_like(p, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins == b
        if not mask.any():
            continue
        out[mask] = mus[b] + ss[b] * (p[mask] - mus[b])
    return out


# ----------------------------------------------------------------------
# Cross-fit driver
# ----------------------------------------------------------------------


def cross_fit_lambda(p_unb: np.ndarray, y_unb: np.ndarray,
                     lam: float, seed: int = SEED) -> dict:
    """Return per-lambda fold artifacts and pooled cross-fit RAE."""
    n_unb = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        p_va, y_va = p_unb[va_loc], y_unb[va_loc]
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = fit_per_bin_tikhonov(p_tr, y_tr, edges, lam)
        pred_va = apply_per_bin_stretch(p_va, edges, mus, ss)
        oof[va_loc] = pred_va
        rae_va = float(rae(y_va, pred_va))
        folds.append({
            "fold": k,
            "edges": edges.tolist(),
            "mus": mus.tolist(),
            "ss": ss.tolist(),
            "val_rae": rae_va,
            "n_va": int(len(va_loc)),
        })
    pooled = float(rae(y_unb, oof))
    s_mean = [float(np.mean([f["ss"][b] for f in folds])) for b in range(N_BINS)]
    s_std = [float(np.std([f["ss"][b] for f in folds])) for b in range(N_BINS)]
    return {
        "lambda": lam,
        "pooled_rae": pooled,
        "folds": folds,
        "oof": oof,
        "s_mean_per_bin": s_mean,
        "s_std_per_bin": s_std,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Tikhonov-regularized per-quantile stretch on te_{ANCHOR}")
    print("=" * 78)

    # ---- 513 test load ----
    te = load_test()
    te_names = te["name"].values
    preds_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    print(f"[load] te_{ANCHOR} shape = {preds_513.shape}  "
          f"mean={preds_513.mean():.3f}  std={preds_513.std():.3f}")

    # ---- 253 unblind subset ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] p_unb shape = {p_unb.shape}, y shape = {y_unb.shape}")
    print(f"[load] truth_std={y_unb.std():.4f}  pred_std={p_unb.std():.4f}  "
          f"(compression ratio = {p_unb.std() / y_unb.std():.3f})")

    in_rae_baseline = float(rae(y_unb, p_unb))
    print(f"\n[baseline] in_RAE(te_{ANCHOR} on 253 unblind) = "
          f"{in_rae_baseline:.4f}")
    print(f"[ref] nb1053 honest pooled (LS-grid)   = {NB1053_HONEST_RAE:.4f}")
    print(f"[ref] nb1060 5-seed bag pooled         = {NB1060_BAGGED_RAE:.4f}")
    print(f"[ref] nb1014 bagged honest             = {NB1014_BAGGED_HONEST_RAE:.4f}")

    # =================================================================
    # Sweep lambda grid via 5-fold cross-fit
    # =================================================================
    print("\n" + "-" * 78)
    print(f"LAMBDA SWEEP  (KFold seed={SEED}, N_BINS={N_BINS})")
    print(f"  closed-form: s_b = (Sxr + lam) / (Sxx + lam)")
    print(f"  lambda grid = {LAMBDA_GRID}")
    print("-" * 78)

    lam_results = {}
    for lam in LAMBDA_GRID:
        res = cross_fit_lambda(p_unb, y_unb, lam, seed=SEED)
        lam_results[lam] = res
        s_str = ",".join(f"{s:.2f}" for s in res["s_mean_per_bin"])
        print(f"   lambda={lam:>5.2f}  pooled={res['pooled_rae']:.4f}  "
              f"s_mean=[{s_str}]")

    best_lambda = min(LAMBDA_GRID, key=lambda l: lam_results[l]["pooled_rae"])
    best_pooled = lam_results[best_lambda]["pooled_rae"]
    print(f"\n[select] best_lambda = {best_lambda}  "
          f"pooled_rae = {best_pooled:.4f}")

    # =================================================================
    # Deploy: fit on all 253 at best lambda, apply to 513
    # =================================================================
    print("\n" + "-" * 78)
    print(f"DEPLOY  (fit on all 253 at lambda={best_lambda}, apply to 513)")
    print("-" * 78)
    qs_all = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
    deploy_edges = np.quantile(p_unb, qs_all)
    deploy_mus, deploy_ss = fit_per_bin_tikhonov(p_unb, y_unb,
                                                  deploy_edges, best_lambda)
    deploy_513 = apply_per_bin_stretch(preds_513, deploy_edges,
                                        deploy_mus, deploy_ss).astype(np.float32)
    in_rae_deploy = float(rae(y_unb, deploy_513[unb_idx].astype(np.float64)))
    print(f"   deploy edges  = {deploy_edges.tolist()}")
    print(f"   deploy mus    = "
          f"{[round(float(x), 3) for x in deploy_mus]}")
    print(f"   deploy ss     = "
          f"{[round(float(x), 3) for x in deploy_ss]}")
    print(f"   in-sample 253 = {in_rae_deploy:.4f}  (overfit lower bound)")
    print(f"   te(513) mean  = {deploy_513.mean():.3f}  "
          f"std={deploy_513.std():.3f}")
    print(f"   anchor te(513) mean = {preds_513.mean():.3f}  "
          f"std={preds_513.std():.3f}")

    # =================================================================
    # Save artifacts
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_tikhonov_bins.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta_vs_nb1060 = best_pooled - NB1060_BAGGED_RAE
    delta_vs_nb1053 = best_pooled - NB1053_HONEST_RAE
    delta_vs_nb1014 = best_pooled - NB1014_BAGGED_HONEST_RAE
    if delta_vs_nb1060 <= -0.005:
        verdict = "BEATS_NB1060"
    elif abs(delta_vs_nb1060) < 0.005:
        verdict = "TIES_NB1060"
    else:
        verdict = "WORSE_THAN_NB1060"
    print(f"\n[verdict] pooled CV vs nb1060 bagged ({NB1060_BAGGED_RAE}): "
          f"delta={delta_vs_nb1060:+.4f}  -> {verdict}")
    print(f"[verdict] pooled CV vs nb1053 ({NB1053_HONEST_RAE}): "
          f"delta={delta_vs_nb1053:+.4f}")

    # Per-lambda summary table for the JSON
    per_lambda = []
    for lam in LAMBDA_GRID:
        r = lam_results[lam]
        per_lambda.append({
            "lambda": lam,
            "pooled_rae": r["pooled_rae"],
            "s_mean_per_bin": r["s_mean_per_bin"],
            "s_std_per_bin": r["s_std_per_bin"],
            "fold_val_rae": [f["val_rae"] for f in r["folds"]],
        })

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_bins": N_BINS,
        "lambda_grid": LAMBDA_GRID,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "in_rae_anchor_on_253": in_rae_baseline,
        "best_lambda": best_lambda,
        "pooled_cross_fit_rae": best_pooled,
        "in_rae_deploy_on_253": in_rae_deploy,
        "per_lambda": per_lambda,
        "nb1014_bagged_honest_rae": NB1014_BAGGED_HONEST_RAE,
        "nb1053_honest_rae": NB1053_HONEST_RAE,
        "nb1060_bagged_rae": NB1060_BAGGED_RAE,
        "delta_vs_nb1014_bagged": delta_vs_nb1014,
        "delta_vs_nb1053": delta_vs_nb1053,
        "delta_vs_nb1060": delta_vs_nb1060,
        "verdict": verdict,
        "deploy_edges": deploy_edges.tolist(),
        "deploy_mus": deploy_mus.tolist(),
        "deploy_ss": deploy_ss.tolist(),
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "anchor_te_mean": float(preds_513.mean()),
        "anchor_te_std": float(preds_513.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                     = {ANCHOR}")
    print(f"   in_RAE anchor on 253       = {in_rae_baseline:.4f}")
    print(f"   best lambda                = {best_lambda}")
    print(f"   pooled cross-fit RAE       = {best_pooled:.4f}")
    print(f"   in-sample (deploy)         = {in_rae_deploy:.4f}")
    print(f"   delta vs nb1060 bagged     = {delta_vs_nb1060:+.4f}")
    print(f"   delta vs nb1053            = {delta_vs_nb1053:+.4f}")
    print(f"   delta vs nb1014 bagged     = {delta_vs_nb1014:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("in_rae_anchor_on_253", "best_lambda",
              "pooled_cross_fit_rae", "in_rae_deploy_on_253",
              "delta_vs_nb1060", "delta_vs_nb1053", "verdict",
              "deploy_ss", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
