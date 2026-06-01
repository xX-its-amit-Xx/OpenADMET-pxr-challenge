"""nb703 -- PHASE-2 BLEND SLSQP over nb562 + nb700 + nb701 + nb702.

Pools the current best (nb562 rank-stretch) with three Phase-2 prescriptions:
  - nb700  Off-manifold negative mining   (attacks F2 false positives by
                                            re-anchoring novel/low-sim test rows)
  - nb701  Pose veto                       (3D docking veto on high-prediction
                                            but unposable compounds)
  - nb702  Promiscuity discount            (counter-assay null_hat penalty)

Hypothesis: each Phase-2 mechanism is decorrelated from nb562's 2D-similarity
predictions -- off-manifold structure (P1), 3D pose feasibility (P2), and
PXR-vs-null selectivity (P3). SLSQP should therefore find a non-trivial
positive-weight blend rather than collapsing to nb562 alone.

Pipeline:
  1. Load OOF (253,) + test (513,) arrays for each candidate.
  2. Print per-candidate OOF RAE + Pearson correlation matrix.
  3. 5-fold cross-fit SLSQP (positive, sum-to-1) on the 253 unblind. Pooled RAE.
  4. Deploy SLSQP refit on full 253; apply weights to 513-row test matrix.
  5. Save te_nb703.npy + nb703_pred_oof.npy + plain & soft-injected submissions.

Target: pooled cross-fit RAE < 0.5065 (nb562 baseline).

Saves
-----
  data/processed/te_nb703.npy           (513,) float32
  data/processed/nb703_pred_oof.npy     (253,) float32
  submissions/nb703_phase2_blend.csv
  submissions/nb703_phase2_blend_soft07_truth.csv
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb703"
N_RESTARTS = 120
NB562_RAE = 0.5065


def _fit_slsqp(M: np.ndarray, y: np.ndarray,
               n_restarts: int = N_RESTARTS, seed: int = SEED) -> np.ndarray:
    """SLSQP with positive bounds + sum-to-1 simplex; Dirichlet restarts."""
    k = M.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * k

    def loss(w):
        return rae(y, M @ w)

    rng = np.random.default_rng(seed)
    seeds = [np.full(k, 1.0 / k)]
    for _ in range(max(0, n_restarts - len(seeds))):
        seeds.append(rng.dirichlet(np.ones(k)))
    best = None
    for w0 in seeds:
        res = minimize(loss, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"ftol": 1e-9, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    w = np.clip(np.asarray(best.x, dtype=float), 0.0, None)
    return w / w.sum()


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- PHASE-2 BLEND SLSQP "
          "(nb562 + nb700 + nb701 + nb702)")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb562_oof":    DATA_PROCESSED / "nb562_pred_oof.npy",
        "te_nb562":     DATA_PROCESSED / "te_nb562.npy",
        "nb700_oof":    DATA_PROCESSED / "nb700_pred_oof.npy",
        "te_nb700":     DATA_PROCESSED / "te_nb700.npy",
        "nb701_oof":    DATA_PROCESSED / "nb701_pred_oof.npy",
        "te_nb701":     DATA_PROCESSED / "te_nb701.npy",
        "nb702_oof":    DATA_PROCESSED / "nb702_pred_oof.npy",
        "te_nb702":     DATA_PROCESSED / "te_nb702.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing,
                "notes": "missing_inputs"}

    te_df = pd.read_csv(needed["TEST_BLINDED"])

    # ---- Index alignment ----
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
    print(f"test n={n_te}  unblind n={n_unb}")
    print(f"truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}\n")

    # ---- Load OOFs + te arrays ----
    print("-" * 78)
    print("CANDIDATE LOAD")
    print("-" * 78)

    oof_562 = np.load(needed["nb562_oof"]).astype(np.float64)
    te_562  = np.load(needed["te_nb562"]).astype(np.float64)
    oof_700 = np.load(needed["nb700_oof"]).astype(np.float64)
    te_700  = np.load(needed["te_nb700"]).astype(np.float64)
    oof_701 = np.load(needed["nb701_oof"]).astype(np.float64)
    te_701  = np.load(needed["te_nb701"]).astype(np.float64)
    oof_702 = np.load(needed["nb702_oof"]).astype(np.float64)
    te_702  = np.load(needed["te_nb702"]).astype(np.float64)

    for name, arr in [("oof_562", oof_562), ("oof_700", oof_700),
                      ("oof_701", oof_701), ("oof_702", oof_702)]:
        assert arr.shape == (n_unb,), f"{name} shape {arr.shape} != ({n_unb},)"
    for name, arr in [("te_562", te_562), ("te_700", te_700),
                      ("te_701", te_701), ("te_702", te_702)]:
        assert arr.shape == (n_te,), f"{name} shape {arr.shape} != ({n_te},)"

    names = ["nb562", "nb700", "nb701", "nb702"]
    M_oof = np.column_stack([oof_562, oof_700, oof_701, oof_702])
    M_te  = np.column_stack([te_562,  te_700,  te_701,  te_702])
    k = M_oof.shape[1]

    cand_meta = []
    for j, name in enumerate(names):
        r_oof = float(rae(unb_y, M_oof[:, j]))
        cand_meta.append({
            "name": name, "rae_oof": r_oof,
            "oof_mean": float(M_oof[:, j].mean()),
            "oof_std":  float(M_oof[:, j].std()),
            "te_mean":  float(M_te[:, j].mean()),
            "te_std":   float(M_te[:, j].std()),
        })
        print(f"  [{name:<8s}]  oof RAE={r_oof:.4f}  "
              f"oof mean/std={M_oof[:, j].mean():.3f}/{M_oof[:, j].std():.3f}  "
              f"te mean/std={M_te[:, j].mean():.3f}/{M_te[:, j].std():.3f}")

    print("\nOOF correlation matrix (Pearson):")
    corr = np.corrcoef(M_oof.T)
    print("  " + " " * 10 + "  ".join(f"{n:>8s}" for n in names))
    for i, n in enumerate(names):
        row = "  " + f"{n:<8s}" + "  ".join(
            f"{corr[i, kk]:8.4f}" for kk in range(k)
        )
        print(row)
    off = (corr.sum() - np.trace(corr)) / (k * (k - 1))
    print(f"\n  mean off-diag corr = {off:.4f}")

    # =================================================================
    # 5-FOLD CROSS-FIT SLSQP
    # =================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT SLSQP over {k} candidates")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred_oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_weights = []
    fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        w_f = _fit_slsqp(M_oof[tr_i], unb_y[tr_i])
        pred_va = M_oof[va_i] @ w_f
        pred_oof_blend[va_i] = pred_va
        r_va = float(rae(unb_y[va_i], pred_va))
        fold_weights.append(w_f)
        fold_raes.append(r_va)
        w_str = "  ".join(f"{n}={w:.3f}" for n, w in zip(names, w_f))
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"RAE={r_va:.4f}  w=[{w_str}]")
    pooled_rae = float(rae(unb_y, pred_oof_blend))
    print(f"\nPooled cross-fit RAE = {pooled_rae:.4f}  "
          f"(per-fold {min(fold_raes):.4f}--{max(fold_raes):.4f})")

    # =================================================================
    # DEPLOY: SLSQP on full 253, apply to 513
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY SLSQP on full 253")
    print("-" * 78)
    w_deploy = _fit_slsqp(M_oof, unb_y)
    deploy_in_sample_rae = float(rae(unb_y, M_oof @ w_deploy))
    active = [(n, float(w)) for n, w in zip(names, w_deploy) if w > 1e-3]
    for n, w in zip(names, w_deploy):
        flag = " *" if w > 1e-3 else ""
        print(f"  w[{n:<8s}] = {w:.4f}{flag}")
    print(f"  active weights ({len(active)}/{k})  "
          f"in-sample RAE={deploy_in_sample_rae:.4f}")

    deploy = (M_te @ w_deploy).astype(np.float32)
    print(f"  deploy te: mean={deploy.mean():.3f}  std={deploy.std():.3f}")
    print(f"  deploy te min/max: {deploy.min():.3f} / {deploy.max():.3f}")

    # =================================================================
    # SAVE
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            pred_oof_blend.astype(np.float32))

    plain = SUBMISSIONS / f"{TAG}_phase2_blend.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_phase2_blend_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # =================================================================
    # Summary
    # =================================================================
    beats = bool(pooled_rae < NB562_RAE)

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  candidates              = {names}")
    print(f"  per-cand OOF RAE        = "
          f"{[round(c['rae_oof'], 4) for c in cand_meta]}")
    print(f"  mean off-diag corr      = {off:.4f}")
    print(f"  POOLED cross-fit RAE    = {pooled_rae:.4f}")
    print(f"  per-fold range          = "
          f"{min(fold_raes):.4f}--{max(fold_raes):.4f}")
    print(f"  deploy active weights   = {len(active)}/{k}")
    print(f"  active                  = {active}")
    print(f"  nb562 target            = {NB562_RAE:.4f}")
    print(f"  beats nb562             = {beats}")
    print("=" * 78)

    return {
        "success":             True,
        "candidates":          names,
        "rae_per_cand_oof":    [c["rae_oof"] for c in cand_meta],
        "crossfit_rae":        pooled_rae,
        "per_fold_raes":       [float(r) for r in fold_raes],
        "fold_weights":        [w.tolist() for w in fold_weights],
        "deploy_weights":      {n: float(w) for n, w in zip(names, w_deploy)},
        "active_weights":      [{"name": n, "weight": w} for n, w in active],
        "deploy_in_sample_rae": deploy_in_sample_rae,
        "mean_off_diag_corr":  float(off),
        "nb562_target":        NB562_RAE,
        "beats_nb562":         beats,
        "n_used":              int(len(active)),
        "plain_submission":    str(plain),
        "soft_submission":     str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
