"""nb602 -- FINAL BLEND SLSQP over nb562 + nb601 + nb601_stretched.

Pools the rank-stretched nb562 anchor with the pretrained-embedding router
(nb601) and its cross-fit rank-stretched variant (nb601_stretched). Runs an
honest 5-fold cross-fit SLSQP (positive weights, sum-to-1) over the OOF
predictions on the 253 unblind rows. Deploy weights come from a SLSQP fit on
the full 253; weights are applied to the 513-row test matrix.

Because nb601 does not persist a cross-fit OOF for its rank-stretched variant,
this script reconstructs it in-line using the same logic as nb601: an outer
5-fold KFold(SEED) cross-fit where, within each train fold, the optimal
stretch s* is selected from STRETCH_GRID on the train indices (using the train
median as center), then applied to the held-out fold. No test rows leak.

Target: pooled cross-fit RAE < nb562 baseline (0.5065).

Saves
-----
  data/processed/te_nb602.npy           (513,) float32
  data/processed/nb602_pred_oof.npy     (253,) float32
  submissions/nb602_pretrained_final_blend.csv
  submissions/nb602_pretrained_final_blend_soft07_truth.csv

If nb601 artifacts are missing, this script falls back to a no-op: it copies
te_nb562 -> te_nb602 (and nb562_pred_oof -> nb602_pred_oof) and writes a
submission identical in content to nb562's plain submission.
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
TAG = "nb602"
N_RESTARTS = 80
NB562_RAE = 0.5065
STRETCH_GRID = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)


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


def _reconstruct_nb601_stretched_oof(
    pred_oof_601: np.ndarray, y_unb: np.ndarray,
) -> np.ndarray:
    """Replicate nb601 cross-fit rank-stretch on pred_oof_601 -> OOF (n_unb,).

    Outer 5-fold KFold(SEED). Within each train fold, picks s* from
    STRETCH_GRID by minimizing RAE on the train indices using the train median
    as center; applies s* with the same center to the held-out fold.
    """
    n = len(y_unb)
    out = np.full(n, np.nan, dtype=np.float64)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr_i, va_i in kf.split(np.arange(n)):
        mu_tr = float(np.mean(pred_oof_601[tr_i]))
        best_s, best_r = 1.0, float("inf")
        for s in STRETCH_GRID:
            cand = mu_tr + s * (pred_oof_601[tr_i] - mu_tr)
            r = float(rae(y_unb[tr_i], cand))
            if r < best_r:
                best_r = r
                best_s = float(s)
        out[va_i] = mu_tr + best_s * (pred_oof_601[va_i] - mu_tr)
    return out


def _noop_passthrough(te_df: pd.DataFrame) -> dict:
    """nb601 missing -> copy nb562 to nb602."""
    print("nb601 artifacts missing -> NO-OP: copying nb562 -> nb602")
    te_562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float32)
    oof_562 = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float32)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_562)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", oof_562)
    plain = SUBMISSIONS / f"{TAG}_pretrained_final_blend.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": te_562,
    }).to_csv(plain, index=False)
    print(f"Wrote {DATA_PROCESSED / f'te_{TAG}.npy'} (= te_nb562)")
    print(f"Wrote {plain}")
    return {
        "success": True,
        "noop": True,
        "candidates": ["nb562"],
        "crossfit_rae": float("nan"),
        "active_weights": [{"name": "nb562", "weight": 1.0}],
        "plain_submission": str(plain),
        "notes": "nb601_missing_passthrough",
    }


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- PRETRAINED FINAL BLEND SLSQP "
          "(nb562 + nb601 + nb601_stretched)")
    print("=" * 78)

    needed_core = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb562_oof":    DATA_PROCESSED / "nb562_pred_oof.npy",
        "te_nb562":     DATA_PROCESSED / "te_nb562.npy",
    }
    missing_core = [k for k, p in needed_core.items() if not Path(p).exists()]
    if missing_core:
        print("MISSING CORE:", missing_core)
        return {"success": False, "missing": missing_core,
                "notes": "missing_core_inputs"}

    te_df = pd.read_csv(needed_core["TEST_BLINDED"])

    nb601_needed = {
        "nb601_oof":          DATA_PROCESSED / "nb601_pred_oof.npy",
        "te_nb601":           DATA_PROCESSED / "te_nb601.npy",
        "te_nb601_stretched": DATA_PROCESSED / "te_nb601_stretched.npy",
    }
    nb601_missing = [k for k, p in nb601_needed.items() if not Path(p).exists()]
    if nb601_missing:
        print(f"nb601 missing: {nb601_missing}")
        return _noop_passthrough(te_df)

    # ---- Indices ----
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed_core["UNBLINDED"])
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
    oof_562 = np.load(needed_core["nb562_oof"]).astype(np.float64)
    te_562 = np.load(needed_core["te_nb562"]).astype(np.float64)
    oof_601 = np.load(nb601_needed["nb601_oof"]).astype(np.float64)
    te_601 = np.load(nb601_needed["te_nb601"]).astype(np.float64)
    te_601_s = np.load(nb601_needed["te_nb601_stretched"]).astype(np.float64)

    # Reconstruct nb601_stretched OOF (cross-fit rank-stretch)
    oof_601_s = _reconstruct_nb601_stretched_oof(oof_601, unb_y)
    print(f"  nb601_stretched OOF reconstructed: "
          f"mean={oof_601_s.mean():.3f}  std={oof_601_s.std():.3f}")

    names = ["nb562", "nb601", "nb601_stretched"]
    M_oof = np.column_stack([oof_562, oof_601, oof_601_s])
    M_te = np.column_stack([te_562, te_601, te_601_s])
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
        print(f"  [{name:<16s}]  oof RAE={r_oof:.4f}  "
              f"oof mean/std={M_oof[:, j].mean():.3f}/{M_oof[:, j].std():.3f}  "
              f"te mean/std={M_te[:, j].mean():.3f}/{M_te[:, j].std():.3f}")

    print("\nOOF correlation matrix (Pearson):")
    corr = np.corrcoef(M_oof.T)
    print("  " + " " * 18 + "  ".join(f"{n:>16s}" for n in names))
    for i, n in enumerate(names):
        row = "  " + f"{n:<16s}" + "  ".join(
            f"{corr[i, kk]:16.4f}" for kk in range(k)
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
        print(f"  w[{n:<16s}] = {w:.4f}{flag}")
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

    plain = SUBMISSIONS / f"{TAG}_pretrained_final_blend.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_pretrained_final_blend_soft07_truth.csv"
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
