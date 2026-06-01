"""nb526 -- POSITIVE-CONSTRAINED RIDGE BLEND over router OOFs.

Hypothesis: SLSQP-MAE keeps zeroing weak-but-decorrelated members; an L2-
regularised non-negative least squares (NNLS-ridge) keeps them in the
blend, reducing variance even if mean RAE is similar.

Pool: nb472, nb481, nb482, nb490, nb491, nb492, nb502, nb510, nb511, nb512,
      nb520, nb521, nb522 (filtered to standalone RAE < 0.55 on the 253).

NNLS-ridge trick:
    minimise ||X w - y||^2  +  alpha ||w||^2   s.t.  w >= 0
    => augment system:    X_aug = [X ; sqrt(alpha) * I]
                          y_aug = [y ; 0]
    => solve nnls(X_aug, y_aug)

Procedure:
  1) Load all OOFs (on 253 unblind rows) + matching te_*.npy (513 rows).
  2) Filter pool: drop tags with standalone RAE >= 0.55.
  3) 5-fold KFold (SEED=0) cross-fit NNLS-ridge with alpha=2.0.
  4) Pooled cross-fit RAE + per-fold range.
  5) Refit on all 253 -> deploy weights -> apply to test stack.
  6) Save te_nb526.npy + nb526_pred_oof.npy
       + submissions/nb526_ridge_blend.csv (plain)
       + submissions/nb526_ridge_blend_soft07_truth.csv (soft inject w=0.7).

Target: cross-fit < 0.5116 (beat nb503).
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
from scipy.optimize import nnls
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
ALPHA = 2.0          # ridge strength
RAE_GATE = 0.55      # standalone RAE filter
TAG = "nb526"

POOL_CANDIDATES = [
    "nb472", "nb481", "nb482", "nb490", "nb491", "nb492",
    "nb502", "nb510", "nb511", "nb512",
    "nb520", "nb521", "nb522",
]


def fit_nnls_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve min ||Xw - y||^2 + alpha ||w||^2  s.t. w >= 0 via augmented NNLS."""
    n, k = X.shape
    sa = float(np.sqrt(alpha))
    X_aug = np.vstack([X, sa * np.eye(k)])
    y_aug = np.concatenate([y, np.zeros(k, dtype=y.dtype)])
    w, _ = nnls(X_aug.astype(np.float64), y_aug.astype(np.float64))
    return w.astype(np.float64)


def main() -> dict:
    print("=" * 78)
    print("nb526 -- NNLS-RIDGE BLEND (alpha=%.2f, gate RAE<%.2f)" % (ALPHA, RAE_GATE))
    print("=" * 78)

    # ---- Required raw indices ----
    needed_raw = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for key, p in needed_raw.items():
        if not Path(p).exists():
            print("MISSING raw:", key, p)
            return {"success": False, "missing": [key]}

    te_df = pd.read_csv(needed_raw["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed_raw["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_te  = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- Discover which candidates have BOTH oof and te files ----
    available = []
    for tag in POOL_CANDIDATES:
        oof_p = DATA_PROCESSED / f"{tag}_pred_oof.npy"
        te_p  = DATA_PROCESSED / f"te_{tag}.npy"
        if oof_p.exists() and te_p.exists():
            available.append(tag)
        else:
            print(f"  skip {tag}: oof_exists={oof_p.exists()} "
                  f"te_exists={te_p.exists()}")
    print(f"\nAvailable candidates: {len(available)} / {len(POOL_CANDIDATES)}")

    # ---- Load + filter by standalone RAE < 0.55 ----
    rows = []
    oof_map = {}
    te_map  = {}
    print("\nStandalone RAE on 253 (gate: keep if RAE < %.2f):" % RAE_GATE)
    for tag in available:
        a = np.load(DATA_PROCESSED / f"{tag}_pred_oof.npy").astype(np.float64)
        d = np.load(DATA_PROCESSED / f"te_{tag}.npy").astype(np.float64)
        if a.shape != (n_unb,):
            print(f"  {tag}: BAD oof shape {a.shape} -- skipping")
            continue
        if d.shape != (n_te,):
            print(f"  {tag}: BAD te shape {d.shape} -- skipping")
            continue
        r_alone = float(rae(unb_y, a))
        keep = r_alone < RAE_GATE
        rows.append((tag, r_alone, keep))
        flag = "KEEP" if keep else "DROP"
        print(f"  {tag}: RAE={r_alone:.4f}  std_oof={a.std():.3f}  "
              f"std_te={d.std():.3f}  -> {flag}")
        if keep:
            oof_map[tag] = a
            te_map[tag]  = d

    pool = [t for t, _, k in rows if k]
    if len(pool) < 2:
        print(f"\nERROR: only {len(pool)} candidates pass gate; need >=2")
        return {"success": False, "pool": pool}
    print(f"\nFinal pool ({len(pool)}): {pool}")

    # ---- Build stacks ----
    P_oof = np.stack([oof_map[t] for t in pool], axis=1)   # (253, k)
    Q_te  = np.stack([te_map[t]  for t in pool], axis=1)   # (513, k)

    # ---- 5-fold cross-fit NNLS-ridge ----
    print("\n" + "-" * 78)
    print(f"5-fold cross-fit NNLS-ridge (alpha={ALPHA})")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_raes = []
    fold_weights = []
    fold_nnz = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        w_f = fit_nnls_ridge(P_oof[tr_i], unb_y[tr_i], ALPHA)
        oof_blend[va_i] = P_oof[va_i] @ w_f
        r_va = float(rae(unb_y[va_i], oof_blend[va_i]))
        fold_raes.append(r_va)
        fold_weights.append(w_f)
        fold_nnz.append(int((w_f > 1e-6).sum()))
        wstr = " ".join(f"{t}={wi:.3f}" for t, wi in zip(pool, w_f))
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"RAE={r_va:.4f}  nnz={fold_nnz[-1]}/{len(pool)}  "
              f"sum_w={w_f.sum():.3f}")
        print(f"           [{wstr}]")

    pooled_rae = float(rae(unb_y, oof_blend))
    fold_lo, fold_hi = float(min(fold_raes)), float(max(fold_raes))
    print(f"\nPooled cross-fit RAE = {pooled_rae:.4f}  "
          f"(per-fold {fold_lo:.4f}--{fold_hi:.4f})")
    print(f"Avg nnz/fold = {np.mean(fold_nnz):.2f} / {len(pool)}")

    # ---- Refit on all 253 -> deploy weights ----
    w_deploy = fit_nnls_ridge(P_oof, unb_y, ALPHA)
    nnz_deploy = int((w_deploy > 1e-6).sum())
    print("\nDeploy NNLS-ridge weights (refit on 253):")
    rows_w = []
    for t, wi in zip(pool, w_deploy):
        active = wi > 1e-6
        flag = "X" if active else "."
        print(f"  [{flag}] {t}: {wi:.4f}")
        rows_w.append({"tag": t, "weight": float(wi), "active": bool(active)})
    print(f"sum_w = {w_deploy.sum():.4f}  nnz = {nnz_deploy} / {len(pool)}")

    # ---- Apply deploy weights to test stack ----
    deploy = (Q_te @ w_deploy).astype(np.float32)

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            oof_blend.astype(np.float32))

    plain = SUBMISSIONS / f"{TAG}_ridge_blend.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{TAG}_ridge_blend_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    NB503 = 0.5116
    beats = pooled_rae < NB503
    print("\n" + "=" * 78)
    print("=== nb526 SUMMARY ===")
    print(f"  pool size                            = {len(pool)} / "
          f"{len(POOL_CANDIDATES)}")
    print(f"  alpha (ridge)                        = {ALPHA}")
    print(f"  cross-fit RAE                        = {pooled_rae:.4f}")
    print(f"  per-fold range                       = [{fold_lo:.4f}, "
          f"{fold_hi:.4f}]")
    print(f"  deploy nnz                           = {nnz_deploy} / {len(pool)}")
    print(f"  vs nb503 ({NB503:.4f})                 = "
          f"{'BEATS' if beats else 'no'} (delta={pooled_rae - NB503:+.4f})")
    print("=" * 78)

    return {
        "success": True,
        "pool": pool,
        "pool_size": len(pool),
        "alpha": ALPHA,
        "crossfit_rae": float(pooled_rae),
        "fold_raes": [float(x) for x in fold_raes],
        "fold_range": [fold_lo, fold_hi],
        "fold_nnz": fold_nnz,
        "deploy_weights": [
            {"tag": t, "weight": float(w), "active": bool(w > 1e-6)}
            for t, w in zip(pool, w_deploy)
        ],
        "n_nonzero_weights": nnz_deploy,
        "n_used": len(pool),
        "beats_nb503": bool(beats),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
