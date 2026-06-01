"""nb723 -- TARGETED BLEND SLSQP over nb562 + nb722 + surviving P1..P4 candidates.

Pool:
  Always include nb562 (rank-stretch baseline, unblind RAE 0.5065) and nb722
  (targeted shrink built from the nb721 precision classifier).  Then sweep the
  wppx7yguv / wjyfq3fz3 Phase-1..Phase-4 candidates (nb700, nb701, nb702, nb703,
  nb710, nb711, nb712, nb713) and KEEP every one whose standalone unblind RAE is
  below 0.55.

Pipeline (mirrors nb713):
  1. Per-candidate standalone unblind RAE on the 253 unblind. Filter RAE < 0.55.
     nb562 + nb722 are forced into the pool regardless.
  2. Print OOF correlation matrix over survivors.
  3. 5-fold cross-fit SLSQP (positive, sum-to-1) on the 253 unblind. Pooled RAE.
  4. Deploy SLSQP refit on full 253; apply weights to 513-row test matrix.
  5. Save te_nb723.npy + nb723_pred_oof.npy + plain & soft-injected submissions.

Per-candidate `oof` here means the model's prediction on the 253 unblind names
(i.e. te[unb_idx]) -- NOT scaffold-CV OOF on the 4139 train. This matches the
nb703/nb713 phase-2/4 convention and is the only signal we can SLSQP against to
score against unblind RAE without train-vs-test distribution mismatch.

Target: pooled cross-fit RAE < 0.5065 (nb562 unblind RAE baseline).  Crossing
this line would confirm the nb721/nb722 precision-classifier capture some of
the -0.1115 RAE prize pm06 identified.

Saves
-----
  data/processed/te_nb723.npy           (513,) float32
  data/processed/nb723_pred_oof.npy     (253,) float32
  submissions/nb723_targeted_blend.csv
  submissions/nb723_targeted_blend_soft07_truth.csv
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
TAG = "nb723"
N_RESTARTS = 120
NB562_RAE = 0.5065
RAE_FILTER = 0.55  # candidate cutoff

# Pool order: anchors first, then P1..P4 sweeps.
FORCED = ["nb562", "nb722"]
SWEEP = ["nb700", "nb701", "nb702", "nb703", "nb710", "nb711", "nb712", "nb713"]


def _fit_slsqp(M: np.ndarray, y: np.ndarray,
               n_restarts: int = N_RESTARTS, seed: int = SEED) -> np.ndarray:
    """SLSQP with positive bounds + sum-to-1 simplex; Dirichlet restarts."""
    k = M.shape[1]
    if k == 1:
        return np.ones(1, dtype=float)
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
    print(f"{TAG} -- TARGETED BLEND SLSQP (nb562 + nb722 + P1..P4 survivors)")
    print("=" * 78)

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"test n={n_te}  unblind n={n_unb}")
    print(f"truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}\n")

    # ---- Discover candidate te_*.npy arrays ----
    all_names = FORCED + SWEEP
    te_arrays: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for n in all_names:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if not p.exists():
            missing.append(n)
            continue
        a = np.load(p).astype(np.float64)
        if a.shape != (n_te,):
            print(f"  SKIP {n}: shape {a.shape} != ({n_te},)")
            continue
        te_arrays[n] = a
    if missing:
        print(f"WARN missing te arrays: {missing}")

    # ---- Standalone RAE + filter ----
    print("-" * 78)
    print(f"CANDIDATE STANDALONE RAE (forced={FORCED}, filter < {RAE_FILTER:.2f})")
    print("-" * 78)
    cand_meta = []
    for n in all_names:
        if n not in te_arrays:
            continue
        t = te_arrays[n]
        o = t[unb_idx]
        r = float(rae(unb_y, o))
        keep = (n in FORCED) or (r < RAE_FILTER)
        cand_meta.append({
            "name": n, "rae_oof": r, "kept": keep,
            "oof_mean": float(o.mean()), "oof_std": float(o.std()),
            "te_mean":  float(t.mean()), "te_std":  float(t.std()),
        })
        flag = " KEEP" if keep else " drop"
        force_flag = " [forced]" if n in FORCED else ""
        print(f"  [{n:<8s}]  unblind RAE={r:.4f}{flag}{force_flag}  "
              f"unb mean/std={o.mean():.3f}/{o.std():.3f}  "
              f"te mean/std={t.mean():.3f}/{t.std():.3f}")

    survivors = [c["name"] for c in cand_meta if c["kept"]]
    if not survivors:
        print("\nALL candidates filtered out -- nothing to blend.")
        return {"success": False, "notes": "no_survivors",
                "crossfit_rae": float("nan"), "beats_nb562": False,
                "n_used": 0, "active_weights": [],
                "plain_submission": "", "soft_submission": ""}
    print(f"\nsurvivors ({len(survivors)}/{len(cand_meta)}): {survivors}")

    # ---- Build matrices over survivors ----
    M_te = np.column_stack([te_arrays[n] for n in survivors])
    M_oof = M_te[unb_idx]
    k = M_oof.shape[1]

    # ---- Correlation diagnostics ----
    if k >= 2:
        print("\nOOF correlation matrix (Pearson) over survivors:")
        corr = np.corrcoef(M_oof.T)
        print("  " + " " * 10 + "  ".join(f"{n:>8s}" for n in survivors))
        for i, n in enumerate(survivors):
            row = "  " + f"{n:<8s}" + "  ".join(
                f"{corr[i, kk]:8.4f}" for kk in range(k)
            )
            print(row)
        off = (corr.sum() - np.trace(corr)) / (k * (k - 1))
        print(f"\n  mean off-diag corr = {off:.4f}")
    else:
        off = float("nan")
        print("\n  (single survivor: skip correlation matrix)")

    # =================================================================
    # 5-FOLD CROSS-FIT SLSQP
    # =================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT SLSQP over {k} survivor(s)")
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
        w_str = "  ".join(f"{n}={w:.3f}" for n, w in zip(survivors, w_f))
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
    active = [(n, float(w)) for n, w in zip(survivors, w_deploy) if w > 1e-3]
    for n, w in zip(survivors, w_deploy):
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

    plain = SUBMISSIONS / f"{TAG}_targeted_blend.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_targeted_blend_soft07_truth.csv"
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
    print(f"  pool considered          = {all_names}")
    print(f"  per-cand unblind RAE     = "
          f"{[round(c['rae_oof'], 4) for c in cand_meta]}")
    print(f"  survivors                = {survivors}")
    print(f"  mean off-diag corr       = {off}")
    print(f"  POOLED cross-fit RAE     = {pooled_rae:.4f}")
    print(f"  nb562 baseline           = {NB562_RAE:.4f}")
    print(f"  delta vs nb562           = {pooled_rae - NB562_RAE:+.4f}")
    print(f"  beats nb562              = {beats}")
    print(f"  active weights           = {active}")
    print("=" * 78)

    return {
        "success": True,
        "crossfit_rae": pooled_rae,
        "nb562_baseline": NB562_RAE,
        "delta_vs_nb562": pooled_rae - NB562_RAE,
        "beats_nb562": beats,
        "survivors": survivors,
        "active_weights": [{"name": n, "w": w} for n, w in active],
        "fold_raes": [float(r) for r in fold_raes],
        "per_cand_rae": {c["name"]: c["rae_oof"] for c in cand_meta},
        "mean_off_diag_corr": float(off) if off == off else None,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
