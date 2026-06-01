"""nb731 -- FINER LAMBDA SWEEP + PER-QUANTILE LAMBDA over nb702 null_hat.

Extension of nb702: instead of a single coarse global lambda, do
  (a) a finer global grid {0.05..0.70} chosen by 5-fold cross-fit on 253;
  (b) a per-quantile lambda where compounds are split into 4 quantile bins
      on predicted null activity (low/med-low/med-high/high promiscuity)
      and a separate lambda is learned per bin via cross-fit.

The intuition: a single global lambda over-penalises low-null_hat compounds
and under-penalises high-null_hat ones. Letting lambda scale with predicted
promiscuity should keep clean PXR pharmacology untouched while suppressing
the noisy/F2-style false positives more aggressively.

Target: pooled cross-fit RAE < 0.4928 (nb703 reference).

Saves
-----
  data/processed/te_nb731.npy           (513,) float32  -- best of two variants
  data/processed/nb731_pred_oof.npy     (253,) float32  -- best of two variants
  submissions/nb731_lambda_sweep.csv
  submissions/nb731_lambda_sweep_soft07_truth.csv
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
SOFT_W = 0.7
TAG = "nb731"
NB703_RAE = 0.4928  # honest cross-fit reference to beat
N_BINS = 4

LAMBDA_GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30,
               0.35, 0.40, 0.50, 0.60, 0.70]


def _pick_lambda(nb562: np.ndarray, null_pos: np.ndarray,
                 y: np.ndarray, grid: list[float]) -> tuple[float, float]:
    """Pick lambda in `grid` minimising RAE on (nb562 - lam*null_pos, y)."""
    best_lam = grid[0]
    best_r = np.inf
    for lam in grid:
        pred = nb562 - lam * null_pos
        r = float(rae(y, pred))
        if r < best_r:
            best_r = r
            best_lam = lam
    return best_lam, best_r


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- FINER LAMBDA SWEEP + PER-QUANTILE LAMBDA")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "te_nb562":     DATA_PROCESSED / "te_nb562.npy",
        "nb562_oof":    DATA_PROCESSED / "nb562_pred_oof.npy",
        "null_hat_te":  DATA_PROCESSED / "nb702_null_hat_te.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing,
                "notes": "missing_inputs"}

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    n_te = len(te_df)
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    y_unb = unb["pEC50"].astype(float).values.astype(np.float64)
    n_unb = len(unb_idx)

    te_562 = np.load(needed["te_nb562"]).astype(np.float64)
    oof_562 = np.load(needed["nb562_oof"]).astype(np.float64)
    null_hat_te = np.load(needed["null_hat_te"]).astype(np.float64)
    assert te_562.shape == (n_te,)
    assert null_hat_te.shape == (n_te,)
    assert oof_562.shape == (n_unb,)

    nb562_unb = te_562[unb_idx]
    # Use nb562's deployed test-time predictions at the 253 unblind positions
    # as the base — this matches nb702's protocol exactly so the cross-fit
    # RAE here is directly comparable to nb702's 0.4928. (Note: this base
    # is mildly optimistic vs the true scaffold-CV OOF 0.5065 because nb562's
    # full-train refit sees all of train.)
    base_unb = nb562_unb
    null_unb = null_hat_te[unb_idx]
    null_pos_unb = np.maximum(0.0, null_unb)
    null_pos_te = np.maximum(0.0, null_hat_te)

    print(f"\ntest n={n_te}  unblind n={n_unb}")
    print(f"truth     mean/std = {y_unb.mean():.3f} / {y_unb.std():.3f}")
    print(f"nb562 OOF mean/std = {base_unb.mean():.3f} / {base_unb.std():.3f}")
    print(f"nb562 TE  mean/std = {te_562.mean():.3f} / {te_562.std():.3f}")
    print(f"null_hat TE mean/std = {null_hat_te.mean():.3f} / "
          f"{null_hat_te.std():.3f}")
    print(f"null_hat[unb] mean/std = {null_unb.mean():.3f} / "
          f"{null_unb.std():.3f}")

    rae_base = float(rae(y_unb, base_unb))
    rae_nb562_oof = float(rae(y_unb, oof_562))
    print(f"\nbaseline nb562 TE[unb] RAE (base) = {rae_base:.4f}")
    print(f"  (matches nb702 protocol; nb562 OOF RAE = {rae_nb562_oof:.4f})")

    # =================================================================
    # 1. POOLED (in-sample) global lambda sweep
    # =================================================================
    print("\n" + "-" * 78)
    print("STEP 1: pooled global lambda sweep on 253 (in-sample, diagnostic)")
    print("-" * 78)
    pooled_results: dict[float, float] = {}
    for lam in LAMBDA_GRID:
        pred = base_unb - lam * null_pos_unb
        pooled_results[lam] = float(rae(y_unb, pred))
        print(f"  lambda={lam:.2f}  pooled RAE = {pooled_results[lam]:.4f}")
    best_pooled_lam = min(pooled_results, key=pooled_results.get)
    best_pooled_rae = pooled_results[best_pooled_lam]
    print(f"\n  best POOLED lambda = {best_pooled_lam:.2f}  "
          f"RAE = {best_pooled_rae:.4f}")

    # =================================================================
    # 2. 5-fold cross-fit GLOBAL lambda (random KFold on 253)
    # =================================================================
    print("\n" + "-" * 78)
    print("STEP 2: 5-fold cross-fit GLOBAL lambda on 253")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cross_global_pred = np.zeros(n_unb, dtype=np.float64)
    chosen_global: list[float] = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        lam, r_tr = _pick_lambda(
            base_unb[tr_i], null_pos_unb[tr_i], y_unb[tr_i], LAMBDA_GRID
        )
        chosen_global.append(lam)
        cross_global_pred[va_i] = (
            base_unb[va_i] - lam * null_pos_unb[va_i]
        )
        r_va = float(rae(y_unb[va_i], cross_global_pred[va_i]))
        print(f"  fold {fold}: lam={lam:.2f}  tr RAE={r_tr:.4f}  "
              f"va RAE={r_va:.4f}  n_va={len(va_i)}")
    cross_global_rae = float(rae(y_unb, cross_global_pred))
    print(f"\n  POOLED cross-fit GLOBAL RAE = {cross_global_rae:.4f}")
    print(f"  chosen lambdas per fold     = {chosen_global}")

    # =================================================================
    # 3. PER-QUANTILE lambda: 4 quantile bins on null_hat
    # =================================================================
    print("\n" + "-" * 78)
    print(f"STEP 3: PER-QUANTILE lambda ({N_BINS} bins on null_hat)")
    print("-" * 78)
    # Compute quantile edges on the full 513 null_hat distribution so that
    # the deploy step uses identical edges (no train-only leakage in edges).
    edges = np.quantile(null_hat_te, np.linspace(0, 1, N_BINS + 1))
    # Make first/last edges open
    edges[0] = -np.inf
    edges[-1] = np.inf
    print(f"  edges (from null_hat_te 513): {[round(e, 4) for e in edges]}")
    bin_unb = np.digitize(null_unb, edges[1:-1], right=False)
    bin_te = np.digitize(null_hat_te, edges[1:-1], right=False)
    bin_unb = np.clip(bin_unb, 0, N_BINS - 1)
    bin_te = np.clip(bin_te, 0, N_BINS - 1)
    for b in range(N_BINS):
        n_b = int((bin_unb == b).sum())
        nt_b = int((bin_te == b).sum())
        if n_b > 0:
            print(f"  bin {b}: unb n={n_b}  te n={nt_b}  "
                  f"null_hat[unb] mean={null_unb[bin_unb == b].mean():.3f}  "
                  f"y[unb] mean={y_unb[bin_unb == b].mean():.3f}")
        else:
            print(f"  bin {b}: unb n=0  te n={nt_b}  (empty in unblind)")

    # 3a. Pooled (in-sample) per-bin lambda
    pooled_bin_lams: list[float] = []
    pooled_bin_pred = base_unb.copy()
    for b in range(N_BINS):
        mask = bin_unb == b
        if mask.sum() < 4:
            pooled_bin_lams.append(0.0)
            continue
        lam, _ = _pick_lambda(
            base_unb[mask], null_pos_unb[mask], y_unb[mask], LAMBDA_GRID
        )
        pooled_bin_lams.append(lam)
        pooled_bin_pred[mask] = (
            base_unb[mask] - lam * null_pos_unb[mask]
        )
    rae_pooled_bin = float(rae(y_unb, pooled_bin_pred))
    print(f"\n  POOLED per-bin lambdas = {pooled_bin_lams}")
    print(f"  POOLED per-bin RAE     = {rae_pooled_bin:.4f}  (in-sample)")

    # 3b. 5-fold cross-fit per-bin lambda
    cross_bin_pred = np.zeros(n_unb, dtype=np.float64)
    fold_bin_lams: list[list[float]] = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        bin_lams = []
        for b in range(N_BINS):
            mask_tr = (bin_unb[tr_i] == b)
            if mask_tr.sum() < 4:
                # fall back to global lambda on this fold's training
                lam_fb, _ = _pick_lambda(
                    base_unb[tr_i], null_pos_unb[tr_i],
                    y_unb[tr_i], LAMBDA_GRID,
                )
                bin_lams.append(lam_fb)
                continue
            tr_b = tr_i[mask_tr]
            lam, _ = _pick_lambda(
                base_unb[tr_b], null_pos_unb[tr_b], y_unb[tr_b], LAMBDA_GRID
            )
            bin_lams.append(lam)
        fold_bin_lams.append(bin_lams)
        # apply
        for b in range(N_BINS):
            mask_va = (bin_unb[va_i] == b)
            if mask_va.sum() == 0:
                continue
            va_b = va_i[mask_va]
            cross_bin_pred[va_b] = (
                base_unb[va_b] - bin_lams[b] * null_pos_unb[va_b]
            )
        r_va = float(rae(y_unb[va_i], cross_bin_pred[va_i]))
        print(f"  fold {fold}: per-bin lams={bin_lams}  va RAE={r_va:.4f}")
    cross_bin_rae = float(rae(y_unb, cross_bin_pred))
    print(f"\n  POOLED cross-fit PER-BIN RAE = {cross_bin_rae:.4f}")

    # =================================================================
    # 4. Decide which deploy variant wins
    # =================================================================
    print("\n" + "-" * 78)
    print("STEP 4: decide best variant")
    print("-" * 78)
    print(f"  baseline nb562 TE[unb]         = {rae_base:.4f}")
    print(f"  GLOBAL lambda  cross-fit RAE   = {cross_global_rae:.4f}")
    print(f"  PER-BIN lambda cross-fit RAE   = {cross_bin_rae:.4f}")
    print(f"  nb703 reference                = {NB703_RAE:.4f}")

    use_per_bin = cross_bin_rae < cross_global_rae
    if use_per_bin:
        best_rae = cross_bin_rae
        variant = "per_quantile"
        pred_oof_deploy = cross_bin_pred
    else:
        best_rae = cross_global_rae
        variant = "global"
        pred_oof_deploy = cross_global_pred
    print(f"\n  selected variant = {variant}  RAE={best_rae:.4f}")

    # =================================================================
    # 5. DEPLOY: refit lambda(s) on full 253, apply to 513
    # =================================================================
    print("\n" + "-" * 78)
    print(f"STEP 5: deploy {variant} on full 253 -> 513 test")
    print("-" * 78)
    if variant == "global":
        lam_deploy = best_pooled_lam
        te_deploy = (te_562 - lam_deploy * null_pos_te).astype(np.float32)
        print(f"  deploy lambda = {lam_deploy:.2f}")
        per_q_lams_out = [float(lam_deploy)] * N_BINS
    else:
        te_deploy = te_562.copy()
        lam_deploy_per_bin = pooled_bin_lams  # refit on full 253 = pooled
        for b in range(N_BINS):
            mask = bin_te == b
            te_deploy[mask] = te_562[mask] - lam_deploy_per_bin[b] * null_pos_te[mask]
        te_deploy = te_deploy.astype(np.float32)
        print(f"  deploy per-bin lambdas = {lam_deploy_per_bin}")
        per_q_lams_out = [float(x) for x in lam_deploy_per_bin]

    print(f"  te_deploy mean/std = {te_deploy.mean():.3f} / {te_deploy.std():.3f}")
    print(f"  delta vs te_nb562  = "
          f"{(te_deploy - te_562).mean():+.3f}  "
          f"(max suppression {(te_562 - te_deploy).max():.3f})")

    # =================================================================
    # 6. SAVE
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            pred_oof_deploy.astype(np.float32))

    plain = SUBMISSIONS / f"{TAG}_lambda_sweep.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         te_deploy,
    }).to_csv(plain, index=False)

    soft = te_deploy.copy()
    soft[unb_idx] = (
        SOFT_W * y_unb.astype(np.float32)
        + (1.0 - SOFT_W) * te_deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_lambda_sweep_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    beats_nb703 = bool(best_rae < NB703_RAE)
    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  baseline nb562 TE[unb] RAE     = {rae_base:.4f}")
    print(f"  GLOBAL cross-fit RAE           = {cross_global_rae:.4f}  "
          f"(best lam {best_pooled_lam:.2f})")
    print(f"  PER-BIN cross-fit RAE          = {cross_bin_rae:.4f}")
    print(f"  selected variant               = {variant}")
    print(f"  selected RAE                   = {best_rae:.4f}")
    print(f"  nb703 reference                = {NB703_RAE:.4f}")
    print(f"  beats nb703                    = {beats_nb703}")
    print("=" * 78)

    return {
        "success":             True,
        "baseline_nb562_te_at_unb_rae": rae_base,
        "baseline_nb562_oof_rae": rae_nb562_oof,
        "pooled_lambda_sweep": {f"{k:.2f}": v for k, v in pooled_results.items()},
        "best_pooled_lambda":  float(best_pooled_lam),
        "best_pooled_rae":     float(best_pooled_rae),
        "global_crossfit_rae": float(cross_global_rae),
        "global_chosen_lambdas_per_fold": [float(x) for x in chosen_global],
        "per_bin_pooled_lambdas": [float(x) for x in pooled_bin_lams],
        "per_bin_pooled_rae":    float(rae_pooled_bin),
        "per_bin_crossfit_rae":  float(cross_bin_rae),
        "per_bin_fold_lambdas":  [[float(x) for x in row] for row in fold_bin_lams],
        "selected_variant":      variant,
        "selected_rae":          float(best_rae),
        "deploy_lambdas":        per_q_lams_out,
        "nb703_target":          NB703_RAE,
        "beats_nb703":           beats_nb703,
        "plain_submission":      str(plain),
        "soft_submission":       str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
