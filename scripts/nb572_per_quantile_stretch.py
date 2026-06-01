"""nb572 -- PER-QUANTILE-BIN STRETCH on nb503.

Idea: nb562's single global stretch s=1.10 is symmetric, but the unblind
diagnostic showed *asymmetric* miscalibration:
  - Low-truth cluster (~2.77) is OVER-predicted by +1.20
  - High-truth cluster (~5.78) is UNDER-predicted by -0.81

A single s cannot fix that. So we split nb503 predictions into 5 quantile
bins (q0-20, q20-40, q40-60, q60-80, q80-100) and learn a separate per-bin
stretch s_b, each anchored on the GLOBAL mu = mean(nb503_pred_train):

    new_pred[i] = mu + s_{b(i)} * (pred[i] - mu)

We expect s to be large at the extremes (q0-20 + q80-100) and small in the
middle.  Quantile thresholds and (mu, s_b) are all refit *per fold* in a
5-fold cross-fit, so the OOF RAE is honest.

Hyperparams mirror nb562/nb503: KFold n_splits=5, shuffle=True,
random_state=0; SOFT_W = 0.7 for the truth-soft-inject submission.

Stretch grid per bin matches nb562: {1.0, 1.1, ..., 1.5}.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
N_BINS = 5
SOFT_W = 0.7
TAG = "nb572"
STRETCH_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


def _quantile_edges(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior quantile edges (n_bins-1 of them) for splitting p into n_bins."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.quantile(p, qs)


def _assign_bins(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return integer bin id 0..n_bins-1 per row."""
    return np.digitize(p, edges, right=False)


def _fit_per_bin_s(
    p_tr: np.ndarray,
    y_tr: np.ndarray,
    edges: np.ndarray,
    mu: float,
    n_bins: int,
    grid,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick s_b minimising RAE on each bin's training points.

    Returns
    -------
    s_per_bin : (n_bins,) float
    counts    : (n_bins,) int  for diagnostics
    """
    bins_tr = _assign_bins(p_tr, edges)
    s_per_bin = np.ones(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = bins_tr == b
        counts[b] = int(m.sum())
        if not m.any():
            continue
        pb = p_tr[m]
        yb = y_tr[m]
        best_s, best_r = 1.0, float("inf")
        for s in grid:
            pred_b = mu + s * (pb - mu)
            r = float(rae(yb, pred_b))
            if r < best_r:
                best_r = r
                best_s = float(s)
        s_per_bin[b] = best_s
    return s_per_bin, counts


def _apply_per_bin(
    p: np.ndarray,
    edges: np.ndarray,
    mu: float,
    s_per_bin: np.ndarray,
) -> np.ndarray:
    bins = _assign_bins(p, edges)
    s_vec = s_per_bin[bins]
    return mu + s_vec * (p - mu)


def main() -> dict:
    print("=" * 78)
    print("nb572 -- PER-QUANTILE-BIN STRETCH on nb503")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb503_pred_oof.npy": DATA_PROCESSED / "nb503_pred_oof.npy",
        "te_nb503.npy":       DATA_PROCESSED / "te_nb503.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- Indices ----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- Load nb503 ----
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb503_te  = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    assert nb503_oof.shape == (n_unb,), f"oof shape {nb503_oof.shape}"
    assert nb503_te.shape  == (n_te,),  f"te shape {nb503_te.shape}"

    rae_nb503 = float(rae(unb_y, nb503_oof))
    print(f"\nnb503 OOF RAE      = {rae_nb503:.4f}")
    print(f"nb503 OOF mean/std = {nb503_oof.mean():.3f} / {nb503_oof.std():.3f}")
    print(f"truth mean/std     = {unb_y.mean():.3f} / {unb_y.std():.3f}")

    # nb562 reference: single global s=1.10 from prior log
    nb562_s = 1.10
    nb562_mu = float(nb503_oof.mean())
    nb562_oof_global = nb562_mu + nb562_s * (nb503_oof - nb562_mu)
    rae_nb562_proxy = float(rae(unb_y, nb562_oof_global))
    print(f"nb562 proxy (global s=1.10) RAE on 253 = {rae_nb562_proxy:.4f}")

    # ===================================================================
    # 5-fold cross-fit, per-quantile-bin stretch
    # ===================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT PER-QUANTILE STRETCH  n_bins={N_BINS} "
          f"grid={STRETCH_GRID}")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_s = []
    per_fold_edges = []
    per_fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        p_tr = nb503_oof[tr_i]
        y_tr = unb_y[tr_i]
        mu_tr = float(p_tr.mean())
        edges_tr = _quantile_edges(p_tr, N_BINS)
        s_per_bin, counts_tr = _fit_per_bin_s(
            p_tr, y_tr, edges_tr, mu_tr, N_BINS, STRETCH_GRID
        )
        oof[va_i] = _apply_per_bin(nb503_oof[va_i], edges_tr, mu_tr, s_per_bin)
        r_va = float(rae(unb_y[va_i], oof[va_i]))
        per_fold_s.append(s_per_bin.tolist())
        per_fold_edges.append(edges_tr.tolist())
        per_fold_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"mu_tr={mu_tr:.3f}")
        print(f"           edges_tr = {[f'{e:.3f}' for e in edges_tr]}")
        print(f"           s_per_bin= {[f'{s:.2f}' for s in s_per_bin]}  "
              f"counts={counts_tr.tolist()}  RAE_va={r_va:.4f}")
    rae_cv = float(rae(unb_y, oof))
    print(f"\nPooled per-quantile cross-fit RAE = {rae_cv:.4f}  "
          f"(per-fold {min(per_fold_raes):.4f}--{max(per_fold_raes):.4f})")

    # Average per-bin s across folds (diagnostic)
    s_mean = np.mean(per_fold_s, axis=0)
    print(f"  mean s per bin across folds = {[f'{s:.2f}' for s in s_mean]}")

    # ===================================================================
    # Deploy: refit on all 253
    # ===================================================================
    mu_deploy = float(nb503_oof.mean())
    edges_deploy = _quantile_edges(nb503_oof, N_BINS)
    s_per_bin_deploy, counts_deploy = _fit_per_bin_s(
        nb503_oof, unb_y, edges_deploy, mu_deploy, N_BINS, STRETCH_GRID
    )
    print("\nDeploy fit on all 253:")
    print(f"  mu_deploy   = {mu_deploy:.3f}")
    print(f"  edges       = {[f'{e:.3f}' for e in edges_deploy]}")
    print(f"  s_per_bin   = {[f'{s:.2f}' for s in s_per_bin_deploy]}")
    print(f"  counts      = {counts_deploy.tolist()}")

    deploy = _apply_per_bin(
        nb503_te, edges_deploy, mu_deploy, s_per_bin_deploy
    ).astype(np.float32)

    print(f"  te_nb503 mean/std    = "
          f"{nb503_te.mean():.3f} / {nb503_te.std():.3f}")
    print(f"  te_nb572 mean/std    = {deploy.mean():.3f} / {deploy.std():.3f}")

    # ===================================================================
    # Save artefacts
    # ===================================================================
    np.save(DATA_PROCESSED / "te_nb572.npy", deploy)
    np.save(DATA_PROCESSED / "nb572_pred_oof.npy", oof.astype(np.float32))

    s_tag = "_".join(f"{s:.1f}" for s in s_per_bin_deploy)
    plain = SUBMISSIONS / f"nb572_per_quantile_stretch_s{s_tag}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = (
        SUBMISSIONS / f"nb572_per_quantile_stretch_s{s_tag}_soft07_truth.csv"
    )
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb572.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb572_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # ===================================================================
    # Summary
    # ===================================================================
    beats_nb562 = bool(rae_cv < rae_nb562_proxy)
    print("\n" + "=" * 78)
    print("=== nb572 SUMMARY ===")
    print(f"  nb503 (s=1.0) RAE                 = {rae_nb503:.4f}")
    print(f"  nb562 proxy (global s=1.10) RAE   = {rae_nb562_proxy:.4f}")
    print(f"  nb572 per-quantile cross-fit RAE  = {rae_cv:.4f}")
    print(f"  s per bin (deploy, low->high)     = "
          f"{[f'{s:.2f}' for s in s_per_bin_deploy]}")
    print(f"  s per bin (mean across CV folds)  = "
          f"{[f'{s:.2f}' for s in s_mean]}")
    print(f"  beats nb562 ({rae_nb562_proxy:.4f})         = {beats_nb562}")
    print("=" * 78)

    return {
        "success": True,
        "rae_nb503": rae_nb503,
        "rae_nb562_proxy": rae_nb562_proxy,
        "rae_cv": rae_cv,
        "per_bin_stretches": [float(s) for s in s_per_bin_deploy],
        "per_bin_stretches_cv_mean": [float(s) for s in s_mean],
        "deploy_mu": float(mu_deploy),
        "deploy_edges": [float(e) for e in edges_deploy],
        "deploy_counts": [int(c) for c in counts_deploy],
        "per_fold_raes": [float(r) for r in per_fold_raes],
        "per_fold_s": per_fold_s,
        "beats_nb562": beats_nb562,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
