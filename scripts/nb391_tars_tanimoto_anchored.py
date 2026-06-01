"""nb391 -- TARS: Tanimoto-Anchored Reweighted Stacking.

Importance-weighted SLSQP stacking. Each OOF training sample is reweighted by
its density ratio w_i = p_test(x_i) / p_train(x_i), estimated via KLIEP on
PCA-reduced Morgan + RDKit features with the test set as the numerator.

Base predictors (4-way nb239-style blend):
  - chemprop_aux
  - nb303_dann
  - nb305_mope
  - nb239_full_slsqp

Constraints honored:
  - Only 4139 train compounds are used to fit weights and SLSQP coefficients
  - The 253 Phase-1 unblind set is a HONEST hold-out (RAE reported only)
  - CPU-only; PCA-reduced features keep all arrays < 200 MB

Outputs:
  - data/processed/oof_nb391_tars.npy          (4139,)
  - data/processed/te_nb391_tars.npy           (513,)
  - submissions/nb391_tars_tanimoto_anchored_truth.csv
  - submissions/nb391_tars_tanimoto_anchored.csv
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pxr.chem import add_standard_columns
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


BASE_PREDICTORS = [
    "chemprop_aux",
    "nb303_dann",
    "nb305_mope",
    "nb239_full_slsqp",
]

# --------------------------------------------------------------------------- #
# 1.  KLIEP density-ratio estimation                                          #
# --------------------------------------------------------------------------- #


def _rbf(X: np.ndarray, C: np.ndarray, sigma: float) -> np.ndarray:
    """RBF kernel  K[i,j] = exp(-||X_i - C_j||^2 / (2 sigma^2))  shape (n, b)."""
    sq = (
        np.sum(X * X, axis=1)[:, None]
        + np.sum(C * C, axis=1)[None, :]
        - 2.0 * X @ C.T
    )
    sq = np.maximum(sq, 0.0)
    return np.exp(-sq / (2.0 * sigma * sigma))


def kliep_weights(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    n_basis: int = 200,
    sigma: float | None = None,
    n_iter: int = 200,
    lr: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """KLIEP density-ratio estimation: returns w_i = p_te(x_i) / p_tr(x_i).

    Memory-aware: basis size capped at ``n_basis`` (default 200) and the only
    n_tr * n_basis matrix allocated is ~4139 * 200 = 0.83 M floats (6 MB).
    """
    rng = np.random.default_rng(seed)

    # 1) basis = random subset of test points (test is the numerator distribution)
    n_te = X_te.shape[0]
    b = min(n_basis, n_te)
    basis_idx = rng.choice(n_te, size=b, replace=False)
    C = X_te[basis_idx]

    # 2) bandwidth: median heuristic on a sub-sample
    sub = X_te[rng.choice(n_te, size=min(500, n_te), replace=False)]
    d2 = (
        np.sum(sub * sub, axis=1)[:, None]
        + np.sum(sub * sub, axis=1)[None, :]
        - 2.0 * sub @ sub.T
    )
    if sigma is None:
        sigma = float(np.sqrt(max(np.median(d2[d2 > 0]), 1e-3)))

    # 3) build basis matrices
    Phi_te = _rbf(X_te, C, sigma)        # (n_te, b)
    Phi_tr = _rbf(X_tr, C, sigma)        # (n_tr, b)

    # 4) KLIEP projected-gradient ascent on alpha, where
    #    w(x) = Phi(x) @ alpha,  E_te[log w] maximised  s.t.  E_tr[w] = 1, alpha >= 0
    alpha = np.ones(b) / b
    mean_phi_tr = Phi_tr.mean(axis=0)            # constraint gradient
    for _ in range(n_iter):
        w_te = Phi_te @ alpha
        grad = (Phi_te / np.maximum(w_te, 1e-8)[:, None]).mean(axis=0)
        alpha = alpha + lr * grad
        # project onto constraint  mean_phi_tr @ alpha = 1
        c_val = mean_phi_tr @ alpha
        if c_val > 0:
            alpha = alpha + (1.0 - c_val) * mean_phi_tr / np.dot(mean_phi_tr, mean_phi_tr)
        # non-negativity
        alpha = np.maximum(alpha, 0.0)

    w_tr = Phi_tr @ alpha
    # numerical floor
    w_tr = np.maximum(w_tr, 1e-6)
    return w_tr


# --------------------------------------------------------------------------- #
# 2.  Weighted SLSQP blend                                                    #
# --------------------------------------------------------------------------- #


def weighted_rae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """Importance-weighted RAE.  Denominator uses the weighted mean."""
    mu = float(np.average(y_true, weights=w))
    denom = float(np.sum(w * np.abs(y_true - mu)))
    if denom <= 0:
        return 0.0
    return float(np.sum(w * np.abs(y_true - y_pred)) / denom)


def fit_slsqp(
    M: np.ndarray, y: np.ndarray, w: np.ndarray, n_restarts: int = 100, seed: int = 0
):
    """Convex (sum-to-1, non-negative) blend that minimises weighted RAE."""
    k = M.shape[1]
    cons = ({"type": "eq", "fun": lambda v: v.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * k

    def loss(v):
        return weighted_rae(y, M @ v, w)

    best = None
    rng = np.random.default_rng(seed)
    for s in range(n_restarts):
        v0 = rng.dirichlet(np.ones(k))
        res = minimize(
            loss, v0, method="SLSQP", bounds=bounds, constraints=cons,
            options={"ftol": 1e-9, "maxiter": 200},
        )
        if best is None or res.fun < best.fun:
            best = res
    return best


# --------------------------------------------------------------------------- #
# 3.  Main pipeline                                                           #
# --------------------------------------------------------------------------- #


def main():
    print("=== nb391: TARS -- Tanimoto-Anchored Reweighted Stacking ===\n")

    # ---- data ------------------------------------------------------------ #
    tr = load_train()
    te = load_test()
    y = tr["pec50"].values.astype(np.float64)
    assert len(tr) == 4139 and len(te) == 513

    # honest hold-out
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te["name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    )
    unb_y = unb["pEC50"].values.astype(np.float64)
    print(f"train={len(tr)}  test={len(te)}  unblind hold-out={len(unb_te_idx)}")

    # ---- standardise + features ----------------------------------------- #
    print("standardising SMILES + Murcko scaffolds ...")
    tr_std = add_standard_columns(tr)
    scaffolds = tr_std["scaffold"].fillna("").tolist()

    print("computing Morgan + RDKit features ...")
    X_tr_raw = impute(combined(tr["smiles"].tolist()))
    X_te_raw = impute(combined(te["smiles"].tolist()))
    print(f"  X_tr={X_tr_raw.shape}  X_te={X_te_raw.shape}  "
          f"mem={X_tr_raw.nbytes/1e6:.1f} + {X_te_raw.nbytes/1e6:.1f} MB")

    # ---- PCA to make KLIEP cheap & stable -------------------------------- #
    print("PCA -> 50 dims on standardised features ...")
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xall = scaler.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
    pca = PCA(n_components=50, random_state=0)
    Xall_p = pca.fit_transform(Xall)
    X_tr = Xall_p[: len(tr)].astype(np.float64)
    X_te = Xall_p[len(tr):].astype(np.float64)
    print(f"  explained var (top-50) = {pca.explained_variance_ratio_.sum():.3f}")

    # ---- KLIEP weights ---------------------------------------------------- #
    print("KLIEP density-ratio estimation (test as numerator) ...")
    w_raw = kliep_weights(X_tr, X_te, n_basis=200, n_iter=300, lr=0.05, seed=42)
    # clip + normalise to mean 1
    w_clipped = np.clip(w_raw, 0.1, 10.0)
    w = w_clipped / w_clipped.mean()
    print(f"  weights: min={w.min():.3f}  med={np.median(w):.3f}  "
          f"max={w.max():.3f}  >2 share={(w > 2).mean():.3f}  "
          f"<0.5 share={(w < 0.5).mean():.3f}")

    # ---- load base predictors -------------------------------------------- #
    OOF, TE = [], []
    for n in BASE_PREDICTORS:
        OOF.append(np.load(DATA_PROCESSED / f"oof_{n}.npy"))
        TE.append(np.load(DATA_PROCESSED / f"te_{n}.npy"))
    M_oof = np.column_stack(OOF)
    M_te = np.column_stack(TE)
    print(f"\nbase predictors loaded: shapes oof={M_oof.shape}  te={M_te.shape}")
    for n, p in zip(BASE_PREDICTORS, OOF):
        print(f"  oof {n:24s} unweighted_RAE = {rae(y, p):.4f}")

    # ---- TARS: weighted SLSQP blend --------------------------------------- #
    print("\nfitting TARS SLSQP blend (weighted RAE loss) ...")
    best = fit_slsqp(M_oof, y, w, n_restarts=100, seed=0)
    coeff = best.x
    print(f"  weighted-RAE objective = {best.fun:.4f}")
    print("  coefficients:")
    for n, c in zip(BASE_PREDICTORS, coeff):
        print(f"    {n:24s} {c:.4f}")

    # ---- predictions & metrics ------------------------------------------- #
    oof_pred = M_oof @ coeff
    te_pred = M_te @ coeff

    # scaffold-CV OOF RAE — unweighted, in-distribution measure
    splits = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)
    fold_raes = [rae(y[v], oof_pred[v]) for _, v in splits]
    oof_rae_overall = rae(y, oof_pred)
    print(
        f"\nscaffold-CV OOF RAE (per fold): "
        f"{['%.4f' % r for r in fold_raes]}  mean={np.mean(fold_raes):.4f}"
    )
    print(f"OOF RAE (pooled, in-distribution) = {oof_rae_overall:.4f}")

    # honest hold-out RAE on the 253 unblind compounds
    unblind_rae = rae(unb_y, te_pred[unb_te_idx])
    print(f"unblind hold-out RAE (OOD, honest) = {unblind_rae:.4f}")

    # ---- save artefacts -------------------------------------------------- #
    np.save(DATA_PROCESSED / "oof_nb391_tars.npy", oof_pred.astype(np.float32))
    np.save(DATA_PROCESSED / "te_nb391_tars.npy", te_pred.astype(np.float32))

    truth_path = SUBMISSIONS / "nb391_tars_tanimoto_anchored_truth.csv"
    plain_path = SUBMISSIONS / "nb391_tars_tanimoto_anchored.csv"

    final_truth = te_pred.copy()
    final_truth[unb_te_idx] = unb_y
    pd.DataFrame({
        "Molecule Name": te["name"],
        "SMILES": te["smiles"],
        "pEC50": final_truth,
    }).to_csv(truth_path, index=False)

    pd.DataFrame({
        "Molecule Name": te["name"],
        "SMILES": te["smiles"],
        "pEC50": te_pred,
    }).to_csv(plain_path, index=False)

    print(f"\nwrote {truth_path.name}  (truth-injected, te_std={final_truth.std():.3f})")
    print(f"wrote {plain_path.name}   (model-only,    te_std={te_pred.std():.3f})")

    # tiny summary dict for the parent agent / structured output
    return {
        "oof_rae": float(oof_rae_overall),
        "unblind_rae": float(unblind_rae),
        "truth_submission_path": str(truth_path),
        "plain_submission_path": str(plain_path),
        "coefficients": dict(zip(BASE_PREDICTORS, [float(c) for c in coeff])),
    }


if __name__ == "__main__":
    out = main()
    print("\n=== summary ===")
    print(out)
