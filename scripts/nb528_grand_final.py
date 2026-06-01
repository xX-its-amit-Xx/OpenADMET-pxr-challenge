"""nb528 -- GRAND FINAL BLEND.

Pool all router OOFs (nb472, nb481, nb482, nb490, nb491, nb492, nb502,
nb510, nb511, nb512, nb520, nb521, nb522, nb526, nb527) on the 253 unblind
rows. Filter to standalone RAE < 0.55. Run BOTH:

  (A) 5-fold cross-fit SLSQP simplex (>=0, sum=1) minimising MAE.
  (B) 5-fold cross-fit NNLS-ridge (alpha=2.0) -- augmented NNLS trick.

Report pooled cross-fit RAE + deploy weights for both. Deploy the lower-RAE
winner -> te_nb528.npy + submissions/.

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
from scipy.optimize import minimize, nnls
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
ALPHA = 2.0          # NNLS-ridge strength
RAE_GATE = 0.55      # standalone RAE filter
TAG = "nb528"
NB503_TARGET = 0.5116

POOL_CANDIDATES = [
    "nb472", "nb481", "nb482", "nb490", "nb491", "nb492",
    "nb502", "nb510", "nb511", "nb512",
    "nb520", "nb521", "nb522",
    "nb526", "nb527",
]


def fit_slsqp_mae(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simplex weights (>=0, sum=1) minimising MAE of P @ w vs y."""
    k = P.shape[1]
    if k == 1:
        return np.array([1.0])
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return np.full(k, 1.0 / k) if s <= 0 else w / s


def fit_nnls_ridge(P: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve min ||Pw - y||^2 + alpha ||w||^2  s.t. w >= 0 via augmented NNLS.

    Note: NOT simplex-constrained (no sum=1) -- matches nb526.
    """
    n, k = P.shape
    sa = float(np.sqrt(alpha))
    P_aug = np.vstack([P, sa * np.eye(k)])
    y_aug = np.concatenate([y, np.zeros(k, dtype=y.dtype)])
    w, _ = nnls(P_aug.astype(np.float64), y_aug.astype(np.float64))
    return w.astype(np.float64)


def crossfit(
    P_oof: np.ndarray,
    y: np.ndarray,
    fit_fn,
    label: str,
    pool: list[str],
) -> tuple[np.ndarray, float, list[float], list[np.ndarray]]:
    """Generic 5-fold cross-fit. Returns (oof_preds, pooled_rae, fold_raes, fold_weights)."""
    print("\n" + "-" * 78)
    print(f"5-fold cross-fit ({label})")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_raes = []
    fold_weights = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n))):
        w_f = fit_fn(P_oof[tr_i], y[tr_i])
        oof[va_i] = P_oof[va_i] @ w_f
        r_va = float(rae(y[va_i], oof[va_i]))
        fold_raes.append(r_va)
        fold_weights.append(w_f)
        nnz = int((w_f > 1e-6).sum())
        wstr = " ".join(f"{t}={wi:.3f}" for t, wi in zip(pool, w_f))
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"RAE={r_va:.4f}  nnz={nnz}/{len(pool)}  sum_w={w_f.sum():.3f}")
        print(f"           [{wstr}]")
    pooled_rae = float(rae(y, oof))
    fold_lo, fold_hi = float(min(fold_raes)), float(max(fold_raes))
    print(f"\nPooled {label} cross-fit RAE = {pooled_rae:.4f}  "
          f"(per-fold {fold_lo:.4f}--{fold_hi:.4f})")
    return oof, pooled_rae, fold_raes, fold_weights


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- GRAND FINAL BLEND  (gate RAE<{RAE_GATE:.2f}, "
          f"SLSQP-MAE vs NNLS-ridge alpha={ALPHA})")
    print("=" * 78)

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

    # ---- Discover candidates with BOTH oof and te files ----
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
    print(f"\nStandalone RAE on 253 (gate: keep if RAE < {RAE_GATE:.2f}):")
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

    # ====================================================================
    # (A) SLSQP-MAE simplex cross-fit
    # ====================================================================
    oof_slsqp, rae_slsqp, fold_raes_s, fold_w_s = crossfit(
        P_oof, unb_y, fit_slsqp_mae, "SLSQP-MAE simplex", pool,
    )
    # Refit on all 253 -> deploy weights
    w_deploy_slsqp = fit_slsqp_mae(P_oof, unb_y)
    nnz_slsqp = int((w_deploy_slsqp > 1e-6).sum())
    print("\nDeploy SLSQP-MAE weights (refit on 253):")
    for t, wi in zip(pool, w_deploy_slsqp):
        active = wi > 1e-6
        flag = "X" if active else "."
        print(f"  [{flag}] {t}: {wi:.4f}")
    print(f"sum_w = {w_deploy_slsqp.sum():.4f}  nnz = {nnz_slsqp} / {len(pool)}")

    # ====================================================================
    # (B) NNLS-ridge cross-fit
    # ====================================================================
    oof_ridge, rae_ridge, fold_raes_r, fold_w_r = crossfit(
        P_oof, unb_y,
        lambda P, y: fit_nnls_ridge(P, y, ALPHA),
        f"NNLS-ridge alpha={ALPHA}", pool,
    )
    w_deploy_ridge = fit_nnls_ridge(P_oof, unb_y, ALPHA)
    nnz_ridge = int((w_deploy_ridge > 1e-6).sum())
    print("\nDeploy NNLS-ridge weights (refit on 253):")
    for t, wi in zip(pool, w_deploy_ridge):
        active = wi > 1e-6
        flag = "X" if active else "."
        print(f"  [{flag}] {t}: {wi:.4f}")
    print(f"sum_w = {w_deploy_ridge.sum():.4f}  nnz = {nnz_ridge} / {len(pool)}")

    # ====================================================================
    # Decide winner (lower cross-fit RAE)
    # ====================================================================
    print("\n" + "=" * 78)
    print("DEPLOY DECISION")
    print("=" * 78)
    print(f"  SLSQP-MAE   cross-fit RAE = {rae_slsqp:.4f}")
    print(f"  NNLS-ridge  cross-fit RAE = {rae_ridge:.4f}")
    if rae_slsqp <= rae_ridge:
        final_method = "slsqp_mae"
        final_rae = rae_slsqp
        w_deploy = w_deploy_slsqp
        oof_blend = oof_slsqp
        print(f"  -> SLSQP-MAE wins  RAE={final_rae:.4f}")
    else:
        final_method = "nnls_ridge"
        final_rae = rae_ridge
        w_deploy = w_deploy_ridge
        oof_blend = oof_ridge
        print(f"  -> NNLS-ridge wins  RAE={final_rae:.4f}")

    # ---- Apply winner's deploy weights to test stack ----
    deploy = (Q_te @ w_deploy).astype(np.float32)

    # ---- Save artefacts ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            oof_blend.astype(np.float32))

    plain = SUBMISSIONS / f"{TAG}_grand_final_{final_method}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{TAG}_grand_final_{final_method}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    beats = final_rae < NB503_TARGET
    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  pool size                            = {len(pool)} / "
          f"{len(POOL_CANDIDATES)}")
    print(f"  pool                                  = {pool}")
    print(f"  SLSQP-MAE   cross-fit RAE             = {rae_slsqp:.4f}")
    print(f"  NNLS-ridge  cross-fit RAE             = {rae_ridge:.4f}")
    print(f"  final_method                          = {final_method}")
    print(f"  final cross-fit RAE                   = {final_rae:.4f}")
    print(f"  vs nb503 ({NB503_TARGET:.4f})              = "
          f"{'BEATS' if beats else 'no'} (delta={final_rae - NB503_TARGET:+.4f})")
    print("=" * 78)

    return {
        "success": True,
        "pool": pool,
        "pool_size": len(pool),
        "crossfit_rae_slsqp": float(rae_slsqp),
        "crossfit_rae_ridge": float(rae_ridge),
        "fold_raes_slsqp": [float(x) for x in fold_raes_s],
        "fold_raes_ridge": [float(x) for x in fold_raes_r],
        "deploy_weights_slsqp": [
            {"tag": t, "weight": float(w), "active": bool(w > 1e-6)}
            for t, w in zip(pool, w_deploy_slsqp)
        ],
        "deploy_weights_ridge": [
            {"tag": t, "weight": float(w), "active": bool(w > 1e-6)}
            for t, w in zip(pool, w_deploy_ridge)
        ],
        "n_nonzero_slsqp": nnz_slsqp,
        "n_nonzero_ridge": nnz_ridge,
        "final_method": final_method,
        "final_rae": float(final_rae),
        "beats_nb503": bool(beats),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
