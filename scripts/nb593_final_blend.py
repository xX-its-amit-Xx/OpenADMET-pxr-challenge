"""nb593 -- FINAL BLEND SLSQP over nb562/nb590/nb591/nb592.

Pool the four current best routers and run an honest 5-fold cross-fit SLSQP
(positive weights, sum-to-1) over their OOF predictions on the 253 unblind
rows. Deploy weights come from a SLSQP fit on the full 253. Apply deploy
weights to the 513-row test matrix.

Because nb592 does not persist its own cross-fit OOF, we reconstruct it
in-line using the same logic as nb592 (5-fold cross-fit rank-stretch over the
better of nb590/nb591). The stretch s is re-selected by pooled cross-fit RAE
on the train folds within nb593 so no test row leaks.

Target: pooled cross-fit RAE < 0.5065.

Saves
-----
  data/processed/te_nb593.npy       (513,) float32
  data/processed/nb593_pred_oof.npy (253,) float32
  submissions/nb593_final_blend_slsqp.csv
  submissions/nb593_final_blend_slsqp_soft07_truth.csv
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
TAG = "nb593"
N_RESTARTS = 80
NB562_RAE = 0.5065
S_GRID = (1.00, 1.05, 1.10, 1.15, 1.20, 1.25)


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


def _reconstruct_nb592_oof(
    oof_590: np.ndarray, oof_591: np.ndarray, y_unb: np.ndarray,
) -> tuple[np.ndarray, str, float]:
    """Replicate nb592 cross-fit stretch to get its OOF (253,).

    Picks the base (nb590 vs nb591) by full-OOF RAE, then runs a 5-fold
    KFold(SEED). Within each train fold, picks s* by pooled cross-fit on the
    *train* indices (inner kfold), then applies s* with train-fold median as
    center to the held-out fold. This mirrors nb592's logic while staying
    fully cross-fit at the nb593 fold level.
    """
    r590 = float(rae(y_unb, oof_590))
    r591 = float(rae(y_unb, oof_591))
    if r590 <= r591:
        base, base_name, base_rae = oof_590, "nb590", r590
    else:
        base, base_name, base_rae = oof_591, "nb591", r591

    n = len(y_unb)
    out = np.full(n, np.nan, dtype=np.float64)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr_i, va_i in kf.split(np.arange(n)):
        # Inner cross-fit on train to pick s*
        inner_pooled = {s: np.full(len(tr_i), np.nan) for s in S_GRID}
        kf_in = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for itr, iva in kf_in.split(np.arange(len(tr_i))):
            c_in = float(np.median(base[tr_i][itr]))
            for s in S_GRID:
                inner_pooled[s][iva] = c_in + s * (base[tr_i][iva] - c_in)
        rae_by_s = {
            s: float(rae(y_unb[tr_i], inner_pooled[s])) for s in S_GRID
        }
        s_star = min(rae_by_s, key=rae_by_s.get)
        c_tr = float(np.median(base[tr_i]))
        out[va_i] = c_tr + s_star * (base[va_i] - c_tr)
    return out, base_name, base_rae


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- FINAL BLEND SLSQP (nb562 + nb590 + nb591 + nb592)")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb562_oof":    DATA_PROCESSED / "nb562_pred_oof.npy",
        "nb590_oof":    DATA_PROCESSED / "nb590_pred_oof.npy",
        "nb591_oof":    DATA_PROCESSED / "nb591_pred_oof.npy",
        "te_nb562":     DATA_PROCESSED / "te_nb562.npy",
        "te_nb590":     DATA_PROCESSED / "te_nb590.npy",
        "te_nb591":     DATA_PROCESSED / "te_nb591.npy",
        "te_nb592":     DATA_PROCESSED / "te_nb592.npy",
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
    print(f"test n={n_te}  unblind n={n_unb}")
    print(f"truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}\n")

    # ---- Load candidates ----
    print("-" * 78)
    print("CANDIDATE LOAD")
    print("-" * 78)
    oof_562 = np.load(needed["nb562_oof"]).astype(np.float64)
    oof_590 = np.load(needed["nb590_oof"]).astype(np.float64)
    oof_591 = np.load(needed["nb591_oof"]).astype(np.float64)
    te_562 = np.load(needed["te_nb562"]).astype(np.float64)
    te_590 = np.load(needed["te_nb590"]).astype(np.float64)
    te_591 = np.load(needed["te_nb591"]).astype(np.float64)
    te_592 = np.load(needed["te_nb592"]).astype(np.float64)

    # Reconstruct nb592 OOF (cross-fit stretch)
    oof_592, base_used, base_rae = _reconstruct_nb592_oof(
        oof_590, oof_591, unb_y
    )
    print(f"  nb592 reconstructed OOF: base={base_used} "
          f"(base full-OOF RAE={base_rae:.4f})")

    names = ["nb562", "nb590", "nb591", "nb592"]
    M_oof = np.column_stack([oof_562, oof_590, oof_591, oof_592])
    M_te = np.column_stack([te_562, te_590, te_591, te_592])
    k = M_oof.shape[1]

    cand_meta = []
    for j, name in enumerate(names):
        r_oof = float(rae(unb_y, M_oof[:, j]))
        cand_meta.append({
            "name": name, "rae_oof": r_oof,
            "oof_mean": float(M_oof[:, j].mean()),
            "oof_std": float(M_oof[:, j].std()),
            "te_mean": float(M_te[:, j].mean()),
            "te_std": float(M_te[:, j].std()),
        })
        print(f"  [{name}]  oof RAE={r_oof:.4f}  "
              f"oof mean/std={M_oof[:, j].mean():.3f}/{M_oof[:, j].std():.3f}  "
              f"te mean/std={M_te[:, j].mean():.3f}/{M_te[:, j].std():.3f}")

    print("\nOOF correlation matrix (Pearson):")
    corr = np.corrcoef(M_oof.T)
    print("  " + " " * 12 + "  ".join(f"{n:>10s}" for n in names))
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

    plain = SUBMISSIONS / f"{TAG}_final_blend_slsqp.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_final_blend_slsqp_soft07_truth.csv"
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
    beats = bool(pooled_rae < NB562_RAE)

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
    print(f"  nb562 target            = {NB562_RAE:.4f}")
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
        "nb562_target": NB562_RAE,
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
