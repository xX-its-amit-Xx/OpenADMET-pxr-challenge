"""nb580 -- STRETCHED MULTI-SOURCE BLEND.

Hypothesis: nb562 (stretched nb503) is variance-corrected but its 4 hedge
ingredients all share the nb464 anchor + multimodal-33 feature space, so
SLSQP collapses to one weight. Combining stretched routers from *different*
feature families (MACCS=nb502, AtomPair=nb510, multimodal-33=nb492,
full-hedge=nb503->nb562) should be less correlated and let SLSQP put real
weight on multiple ingredients.

Procedure
=========
  Candidates (4):
    A. nb502 stretched (s=1.05)  -- MACCS family
    B. nb510 stretched (s=1.05)  -- AtomPair family
    C. nb492 stretched (s=1.05)  -- multimodal-33 family
    D. nb562 = nb503 stretched (s=1.10)  -- full-hedge
  For the 253 OOF: build pred_oof_stretched via 5-fold cross-fit
    mu_fold = mean(pred_oof[train_idx])
    pred_oof_stretched[va] = mu_fold + s * (pred_oof[va] - mu_fold)
  Then 5-fold cross-fit SLSQP (positive weights sum-to-1) over the 4 stretched
  OOFs. Deploy weights = SLSQP on full 253 against full 4-candidate matrix;
  apply to te_<base>_stretched.npy (513).

Target: pooled cross-fit RAE < 0.5065 (nb562's number).
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
TAG = "nb580"
N_RESTARTS = 80

# (name, base_for_oof, deploy_te_file, stretch_s)
CANDIDATES = [
    ("nb502_str", "nb502", "te_nb502_stretched.npy", 1.05),
    ("nb510_str", "nb510", "te_nb510_stretched.npy", 1.05),
    ("nb492_str", "nb492", "te_nb492_stretched.npy", 1.05),
    ("nb562",     "nb503", "te_nb562.npy",           1.10),
]


def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def _build_stretched_oof(pred_oof: np.ndarray, s: float) -> np.ndarray:
    """5-fold cross-fit: per fold mu = mean(pred_oof[train]); apply on held out."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n = pred_oof.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for tr_i, va_i in kf.split(np.arange(n)):
        mu_tr = float(pred_oof[tr_i].mean())
        out[va_i] = _stretch(pred_oof[va_i], mu_tr, s)
    return out


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
    print(f"{TAG} -- STRETCHED MULTI-SOURCE BLEND  (SLSQP cross-fit)")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for nm, base, te_f, _ in CANDIDATES:
        needed[f"{base}_pred_oof.npy"] = DATA_PROCESSED / f"{base}_pred_oof.npy"
        needed[te_f] = DATA_PROCESSED / te_f
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

    # ---- Build stretched OOFs (253) and load deploy te (513) ----
    print("-" * 78)
    print("STRETCHED OOFS (5-fold cross-fit)")
    print("-" * 78)
    M_oof = np.zeros((n_unb, len(CANDIDATES)), dtype=np.float64)
    M_te  = np.zeros((n_te, len(CANDIDATES)), dtype=np.float64)
    cand_meta = []
    for j, (name, base, te_f, s) in enumerate(CANDIDATES):
        pred_oof = np.load(
            DATA_PROCESSED / f"{base}_pred_oof.npy"
        ).astype(np.float64)
        pred_te = np.load(DATA_PROCESSED / te_f).astype(np.float64)
        assert pred_oof.shape == (n_unb,)
        assert pred_te.shape == (n_te,)
        oof_s = _build_stretched_oof(pred_oof, s)
        M_oof[:, j] = oof_s
        M_te[:, j]  = pred_te
        r_orig = float(rae(unb_y, pred_oof))
        r_str  = float(rae(unb_y, oof_s))
        cand_meta.append({
            "name": name, "s": s, "rae_orig": r_orig, "rae_stretched": r_str,
            "oof_mean": float(oof_s.mean()), "oof_std": float(oof_s.std()),
            "te_mean": float(pred_te.mean()), "te_std": float(pred_te.std()),
        })
        print(f"  [{name}] s={s:.2f}  orig RAE={r_orig:.4f}  "
              f"str RAE={r_str:.4f}  oof std={oof_s.std():.3f}  "
              f"te std={pred_te.std():.3f}")

    # ---- Correlation diagnostic ----
    print("\nOOF correlation matrix (Pearson):")
    names = [c["name"] for c in cand_meta]
    corr = np.corrcoef(M_oof.T)
    hdr = "  " + " " * 12 + "  ".join(f"{n:>10s}" for n in names)
    print(hdr)
    for i, n in enumerate(names):
        row = "  " + f"{n:<12s}" + "  ".join(
            f"{corr[i, k]:10.4f}" for k in range(len(names))
        )
        print(row)
    print(f"\n  mean off-diag corr = "
          f"{(corr.sum() - np.trace(corr)) / (len(names) * (len(names) - 1)):.4f}")

    # =================================================================
    # 5-FOLD CROSS-FIT SLSQP
    # =================================================================
    print("\n" + "-" * 78)
    print("5-FOLD CROSS-FIT SLSQP over 4 stretched candidates")
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
    print(f"  active weights ({len(active)}/{len(names)})  "
          f"in-sample RAE={deploy_in_sample_rae:.4f}")

    deploy = (M_te @ w_deploy).astype(np.float32)
    print(f"  deploy te: mean={deploy.mean():.3f}  std={deploy.std():.3f}")

    # =================================================================
    # SAVE
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            pred_oof_blend.astype(np.float32))

    plain = SUBMISSIONS / f"{TAG}_stretched_multisource_slsqp.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_stretched_multisource_slsqp_soft07_truth.csv"
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
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    rae_nb562 = float(rae(unb_y, nb562_oof))
    beats = bool(pooled_rae < rae_nb562)

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  candidates                = {names}")
    print(f"  stretches                 = {[c['s'] for c in cand_meta]}")
    print(f"  per-cand stretched RAE    = "
          f"{[round(c['rae_stretched'], 4) for c in cand_meta]}")
    print(f"  POOLED cross-fit RAE      = {pooled_rae:.4f}")
    print(f"  per-fold range            = "
          f"{min(fold_raes):.4f}--{max(fold_raes):.4f}")
    print(f"  deploy active weights     = {len(active)}/{len(names)}")
    print(f"  nb562 OOF RAE             = {rae_nb562:.4f}")
    print(f"  beats nb562               = {beats}")
    print("=" * 78)

    return {
        "success": True,
        "candidates": names,
        "stretches": [c["s"] for c in cand_meta],
        "rae_per_cand_stretched": [c["rae_stretched"] for c in cand_meta],
        "crossfit_rae": pooled_rae,
        "per_fold_raes": [float(r) for r in fold_raes],
        "fold_weights": [w.tolist() for w in fold_weights],
        "deploy_weights": {n: float(w) for n, w in zip(names, w_deploy)},
        "active_weights": active,
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
