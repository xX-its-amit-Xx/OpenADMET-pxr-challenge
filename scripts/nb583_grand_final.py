"""nb583 -- GRAND FINAL SLSQP BLEND over nb562/nb580/nb581/nb582.

Pool the current best (nb562 = 0.5065 cross-fit) with the three follow-up
routers (nb580 multi-source SLSQP, nb581 iso-on-nb562, nb582 confidence-gated
shrink) and run an honest 5-fold cross-fit SLSQP (positive weights, sum-to-1)
over the 4 OOFs.  Deploy weights come from a SLSQP fit on the full 253.

Target: pooled cross-fit RAE < 0.5065.

Saves
-----
  data/processed/te_nb583.npy       (513,) float32
  data/processed/nb583_pred_oof.npy (253,) float32
  submissions/nb583_grand_final_slsqp.csv
  submissions/nb583_grand_final_slsqp_soft07_truth.csv
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
TAG = "nb583"
N_RESTARTS = 80

# (name, oof_file, te_file)
CANDIDATES = [
    ("nb562", "nb562_pred_oof.npy", "te_nb562.npy"),
    ("nb580", "nb580_pred_oof.npy", "te_nb580.npy"),
    ("nb581", "nb581_pred_oof.npy", "te_nb581.npy"),
    ("nb582", "nb582_pred_oof.npy", "te_nb582.npy"),
]


def _fit_slsqp(M: np.ndarray, y: np.ndarray,
               n_restarts: int = N_RESTARTS, seed: int = SEED) -> np.ndarray:
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
    print(f"{TAG} -- GRAND FINAL SLSQP (nb562 + nb580 + nb581 + nb582)")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for name, oof_f, te_f in CANDIDATES:
        needed[oof_f] = DATA_PROCESSED / oof_f
        needed[te_f]  = DATA_PROCESSED / te_f
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
    print(f"test n={n_te}  unblind n={n_unb}")
    print(f"truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}\n")

    # ---- Load candidate OOFs (253) + TE (513) ----
    print("-" * 78)
    print("CANDIDATE LOAD")
    print("-" * 78)
    names = [c[0] for c in CANDIDATES]
    k = len(CANDIDATES)
    M_oof = np.zeros((n_unb, k), dtype=np.float64)
    M_te  = np.zeros((n_te, k), dtype=np.float64)
    cand_meta = []
    for j, (name, oof_f, te_f) in enumerate(CANDIDATES):
        pred_oof = np.load(DATA_PROCESSED / oof_f).astype(np.float64)
        pred_te  = np.load(DATA_PROCESSED / te_f).astype(np.float64)
        assert pred_oof.shape == (n_unb,), \
            f"{oof_f} shape {pred_oof.shape} != ({n_unb},)"
        assert pred_te.shape == (n_te,), \
            f"{te_f} shape {pred_te.shape} != ({n_te},)"
        M_oof[:, j] = pred_oof
        M_te[:, j]  = pred_te
        r_oof = float(rae(unb_y, pred_oof))
        cand_meta.append({
            "name": name, "rae_oof": r_oof,
            "oof_mean": float(pred_oof.mean()), "oof_std": float(pred_oof.std()),
            "te_mean": float(pred_te.mean()), "te_std": float(pred_te.std()),
        })
        print(f"  [{name}]  oof RAE={r_oof:.4f}  "
              f"oof mean/std={pred_oof.mean():.3f}/{pred_oof.std():.3f}  "
              f"te mean/std={pred_te.mean():.3f}/{pred_te.std():.3f}")

    # ---- Correlation diagnostic ----
    print("\nOOF correlation matrix (Pearson):")
    corr = np.corrcoef(M_oof.T)
    hdr = "  " + " " * 12 + "  ".join(f"{n:>10s}" for n in names)
    print(hdr)
    for i, n in enumerate(names):
        row = "  " + f"{n:<12s}" + "  ".join(
            f"{corr[i, kk]:10.4f}" for kk in range(k)
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
        print(f"  w[{n:<10s}] = {w:.4f}{flag}")
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

    plain = SUBMISSIONS / f"{TAG}_grand_final_slsqp.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_grand_final_slsqp_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # =================================================================
    # Summary
    # =================================================================
    rae_nb562 = float(rae(unb_y, M_oof[:, names.index("nb562")]))
    beats = bool(pooled_rae < rae_nb562)

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  candidates              = {names}")
    print(f"  per-cand OOF RAE        = "
          f"{[round(c['rae_oof'], 4) for c in cand_meta]}")
    print(f"  POOLED cross-fit RAE    = {pooled_rae:.4f}")
    print(f"  per-fold range          = "
          f"{min(fold_raes):.4f}--{max(fold_raes):.4f}")
    print(f"  deploy active weights   = {len(active)}/{k}")
    print(f"  active                  = {active}")
    print(f"  nb562 OOF RAE           = {rae_nb562:.4f}")
    print(f"  beats nb562             = {beats}")
    print("=" * 78)

    return {
        "success": True,
        "candidates": names,
        "rae_per_cand_oof": [c["rae_oof"] for c in cand_meta],
        "crossfit_rae": pooled_rae,
        "per_fold_raes": [float(r) for r in fold_raes],
        "fold_weights": [w.tolist() for w in fold_weights],
        "deploy_weights": {n: float(w) for n, w in zip(names, w_deploy)},
        "active_weights": [{"name": n, "weight": w} for n, w in active],
        "deploy_in_sample_rae": deploy_in_sample_rae,
        "rae_nb562": rae_nb562,
        "beats_nb562": beats,
        "n_used": int(len(active)),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
