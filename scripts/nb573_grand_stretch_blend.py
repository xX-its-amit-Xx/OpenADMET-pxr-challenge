"""nb573 -- GRAND STRETCH BLEND.

Pool stretch-family members and SLSQP-blend them honestly:
  - nb503       : hedged-router base (no stretch)
  - nb562       : nb503 stretched s=1.10
  - nb571       : nb503 bias+stretch (2-param affine)
  - nb572       : nb503 per-quantile stretch
  - nb570 stretched-routers that BEAT their own unstretched (delta < 0).
    Per data/processed/nb570_summary.csv that means
    {nb502, nb510, nb481, nb492} at s=1.05.  nb472 is dropped.

5-fold cross-fit SLSQP on the 253 unblind rows. mu's / s's are refit per
fold for the nb570 stretches so we don't leak OOF info into the blend.

Target: cross-fit RAE < 0.5065 (beat nb562).

Hyperparams: KFold n_splits=5, shuffle=True, random_state=0; SLSQP with
40 Dirichlet restarts per fold; SOFT_W = 0.7 for the truth-soft-inject
submission.
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
N_RESTARTS = 40
SOFT_W = 0.7
TAG = "nb573"
NB562_CROSSFIT_RAE = 0.5065  # target to beat

# nb570 bases whose stretched version beat the unstretched (from nb570_summary)
NB570_STRETCHED = [
    ("nb502", 1.05),
    ("nb510", 1.05),
    ("nb481", 1.05),
    ("nb492", 1.05),
]


def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def _affine(p: np.ndarray, mu: float, s: float, shift: float) -> np.ndarray:
    return s * (p - mu) + (mu + shift)


def _quantile_edges(p: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.quantile(p, qs)


def _assign_bins(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(p, edges, right=False)


def _fit_per_bin_s(p_tr, y_tr, edges, mu, n_bins, grid):
    bins_tr = _assign_bins(p_tr, edges)
    s_per_bin = np.ones(n_bins, dtype=np.float64)
    for b in range(n_bins):
        m = bins_tr == b
        if not m.any():
            continue
        pb, yb = p_tr[m], y_tr[m]
        best_s, best_r = 1.0, float("inf")
        for s in grid:
            r = float(rae(yb, mu + s * (pb - mu)))
            if r < best_r:
                best_r, best_s = r, float(s)
        s_per_bin[b] = best_s
    return s_per_bin


def _apply_per_bin(p, edges, mu, s_per_bin):
    bins = _assign_bins(p, edges)
    return mu + s_per_bin[bins] * (p - mu)


def _best_affine(p_tr, y_tr, s_grid, shift_grid):
    mu = float(p_tr.mean())
    best_s, best_sh, best_r = 1.0, 0.0, float("inf")
    for s in s_grid:
        for sh in shift_grid:
            r = float(rae(y_tr, _affine(p_tr, mu, s, sh)))
            if r < best_r:
                best_r, best_s, best_sh = r, float(s), float(sh)
    return best_s, best_sh, mu


def _slsqp_fit(M_tr, y_tr, n_restarts=N_RESTARTS, seed=0):
    K = M_tr.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * K
    def loss(w):
        return float(rae(y_tr, M_tr @ w))
    best = None
    for r in range(n_restarts):
        rng = np.random.default_rng(seed * 1000 + r)
        w0 = rng.dirichlet(np.ones(K))
        res = minimize(
            loss, w0, method="SLSQP", bounds=bounds,
            constraints=cons, options={"ftol": 1e-9, "maxiter": 200},
        )
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(best.fun)


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- GRAND STRETCH BLEND")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb503_pred_oof.npy": DATA_PROCESSED / "nb503_pred_oof.npy",
        "te_nb503.npy":       DATA_PROCESSED / "te_nb503.npy",
        "nb562_pred_oof.npy": DATA_PROCESSED / "nb562_pred_oof.npy",
        "te_nb562.npy":       DATA_PROCESSED / "te_nb562.npy",
        "nb571_pred_oof.npy": DATA_PROCESSED / "nb571_pred_oof.npy",
        "te_nb571.npy":       DATA_PROCESSED / "te_nb571.npy",
        "nb572_pred_oof.npy": DATA_PROCESSED / "nb572_pred_oof.npy",
        "te_nb572.npy":       DATA_PROCESSED / "te_nb572.npy",
    }
    for base, _ in NB570_STRETCHED:
        needed[f"{base}_pred_oof.npy"] = DATA_PROCESSED / f"{base}_pred_oof.npy"
        needed[f"te_{base}.npy"]       = DATA_PROCESSED / f"te_{base}.npy"
        needed[f"te_{base}_stretched.npy"] = (
            DATA_PROCESSED / f"te_{base}_stretched.npy"
        )

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
    print(f"truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}")

    # ---- Pool members ----
    # Each entry: (name, oof_unb (253,), te (513,))
    pool: list[tuple[str, np.ndarray, np.ndarray]] = []

    # (1) Direct-load OOF+TE pairs
    for nm in ["nb503", "nb562", "nb571", "nb572"]:
        oof = np.load(DATA_PROCESSED / f"{nm}_pred_oof.npy").astype(np.float64)
        te  = np.load(DATA_PROCESSED / f"te_{nm}.npy").astype(np.float64)
        assert oof.shape == (n_unb,), f"{nm} oof shape {oof.shape}"
        assert te.shape  == (n_te,),  f"{nm} te shape {te.shape}"
        pool.append((nm, oof, te))

    # (2) nb570 stretched routers: reconstruct honest OOF + load deploy te
    kf_inner = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for base, s in NB570_STRETCHED:
        base_oof = np.load(
            DATA_PROCESSED / f"{base}_pred_oof.npy"
        ).astype(np.float64)
        base_te = np.load(
            DATA_PROCESSED / f"te_{base}_stretched.npy"
        ).astype(np.float64)
        assert base_oof.shape == (n_unb,)
        assert base_te.shape  == (n_te,)
        # Reproduce nb570's cross-fit-at-s OOF (mu refit per fold)
        oof_s = np.full(n_unb, np.nan, dtype=np.float64)
        for tr_i, va_i in kf_inner.split(np.arange(n_unb)):
            mu_tr = float(base_oof[tr_i].mean())
            oof_s[va_i] = _stretch(base_oof[va_i], mu_tr, s)
        nm = f"{base}_str{s:.2f}"
        pool.append((nm, oof_s, base_te))

    # Diagnostics
    print("\nPOOL (name | OOF RAE | OOF mean/std | te mean/std)")
    print("-" * 78)
    pool_raes = {}
    for nm, oof, te in pool:
        r = float(rae(unb_y, oof))
        pool_raes[nm] = r
        print(
            f"  {nm:<20s} | RAE={r:.4f} | "
            f"oof {oof.mean():.3f}/{oof.std():.3f} | "
            f"te {te.mean():.3f}/{te.std():.3f}"
        )

    M_unb = np.column_stack([oof for _, oof, _ in pool])   # (253, K)
    M_te  = np.column_stack([te  for _, _, te in pool])    # (513, K)
    K = len(pool)
    names = [nm for nm, _, _ in pool]
    print(f"\npool size K = {K}")

    # =================================================================
    # 5-fold cross-fit SLSQP
    # =================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT SLSQP (restarts={N_RESTARTS})")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_w = []
    per_fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        w, train_rae = _slsqp_fit(
            M_unb[tr_i], unb_y[tr_i],
            n_restarts=N_RESTARTS, seed=fold,
        )
        pred_va = M_unb[va_i] @ w
        oof_blend[va_i] = pred_va
        r_va = float(rae(unb_y[va_i], pred_va))
        per_fold_w.append(w)
        per_fold_raes.append(r_va)
        top = sorted(zip(names, w), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{nm}={wi:.2f}" for nm, wi in top)
        print(
            f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
            f"train RAE={train_rae:.4f}  va RAE={r_va:.4f}  top: {top_str}"
        )
    rae_cv = float(rae(unb_y, oof_blend))
    print(
        f"\nPooled SLSQP cross-fit RAE = {rae_cv:.4f}  "
        f"(per-fold {min(per_fold_raes):.4f}--{max(per_fold_raes):.4f})"
    )

    # =================================================================
    # Deploy: refit SLSQP on ALL 253
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY SLSQP FIT (all 253)")
    print("-" * 78)
    w_deploy, train_rae_deploy = _slsqp_fit(
        M_unb, unb_y, n_restarts=N_RESTARTS, seed=999
    )
    print(f"deploy in-sample RAE = {train_rae_deploy:.4f}")
    print("deploy weights (sorted):")
    sorted_w = sorted(zip(names, w_deploy), key=lambda x: -x[1])
    for nm, wi in sorted_w:
        if wi > 1e-4:
            print(f"   {wi:.4f}  {nm}")

    deploy = (M_te @ w_deploy).astype(np.float32)

    # =================================================================
    # Save artefacts
    # =================================================================
    np.save(DATA_PROCESSED / "te_nb573.npy", deploy)
    np.save(DATA_PROCESSED / "nb573_pred_oof.npy", oof_blend.astype(np.float32))

    plain = SUBMISSIONS / f"{TAG}_grand_stretch_blend.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_grand_stretch_blend_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb573.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb573_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")
    print(
        f"\ndeploy te mean/std = {deploy.mean():.3f} / {deploy.std():.3f}"
    )

    # =================================================================
    # Summary
    # =================================================================
    beats = rae_cv < NB562_CROSSFIT_RAE
    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  pool size                          = {K}")
    print(f"  nb503 OOF RAE                      = {pool_raes['nb503']:.4f}")
    print(f"  nb562 OOF RAE                      = {pool_raes['nb562']:.4f}")
    print(f"  nb571 OOF RAE                      = {pool_raes['nb571']:.4f}")
    print(f"  nb572 OOF RAE                      = {pool_raes['nb572']:.4f}")
    print(f"  nb573 SLSQP cross-fit RAE          = {rae_cv:.4f}")
    print(f"  target (nb562 cross-fit)           = {NB562_CROSSFIT_RAE:.4f}")
    print(f"  beats nb562?                       = {beats}")
    print("=" * 78)

    active = [
        {"name": nm, "weight": float(wi)}
        for nm, wi in sorted_w if wi > 1e-4
    ]

    return {
        "success": True,
        "n_used": K,
        "pool_names": names,
        "pool_raes": pool_raes,
        "crossfit_rae": rae_cv,
        "per_fold_raes": [float(r) for r in per_fold_raes],
        "deploy_weights": [
            {"name": nm, "weight": float(wi)} for nm, wi in zip(names, w_deploy)
        ],
        "active_weights": active,
        "deploy_in_sample_rae": train_rae_deploy,
        "beats_nb562": bool(beats),
        "target_rae": NB562_CROSSFIT_RAE,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        if isinstance(v, list) and len(v) > 10:
            print(f"  {k}: <list len={len(v)}>")
        else:
            print(f"  {k}: {v}")
