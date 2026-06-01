"""nb392 -- MMD-Match Ensemble Weight Search.

Reframe blend weight optimization over 13 base predictors as a two-objective
problem on the simplex:

    minimize  alpha * OOF_MSE(w)
            + lambda_mmd * MMD2_rbf( blend_test(w), train_y reweighted by KLIEP w_train )
            + lambda_tail * TAIL_GAP(w)         (penalises under/over-prediction of high-pEC50 tail)

* KLIEP density-ratio weights w_train are estimated by uLSIF on Morgan+RDKit
  features (test vs train). They reweight the empirical pEC50 distribution
  on TRAIN so it reflects the chemical neighbourhood of TEST.
* MMD with an RBF kernel of bandwidth = median heuristic over both samples
  is the kernel-two-sample test statistic; minimising it pulls the test-
  prediction histogram toward the reweighted train-pEC50 distribution. This
  fixes the distribution-shift collapse where SLSQP picks weights that produce
  test predictions concentrated tightly around the train mean.
* Tail term penalises the gap between the upper quantile (95th percentile) of
  blend test predictions and the reweighted upper quantile of train pEC50.

Optimisation: projected gradient on the probability simplex
(sort+threshold projection of Wang & Carreira-Perpinan 2013), 800 iterations.

Outputs:
    data/processed/oof_nb392_mmd_match.npy
    data/processed/te_nb392_mmd_match.npy
    submissions/nb392_mmd-match ensemble weigh.csv          (no truth)
    submissions/nb392_mmd-match ensemble weigh_truth.csv    (truth-injected on 253)
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

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


# ---------------------------------------------------------------------------
# Predictor pool (13 clean OOF + TE pairs)
# ---------------------------------------------------------------------------
PREDICTORS = [
    "nb93_chemprop_large_gpu",
    "nb130_external_pxr",
    "nb264_chemprop_mt",
    "nb303_dann",
    "chemprop_aux",
    "nb305_mope",
    "nb306_cepsmim",
    "catboost",
    "grand_v6b_calib",
    "deep_ensemble",
    "nb239_full_slsqp",
    "nb224_pool_plus_2",
    "nb179_stack",
]
NB = "nb392_mmd_match"


# ---------------------------------------------------------------------------
# Simplex projection (Wang & Carreira-Perpinan, 2013)
# ---------------------------------------------------------------------------
def project_simplex(v: np.ndarray) -> np.ndarray:
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / np.arange(1, n + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


# ---------------------------------------------------------------------------
# KLIEP / uLSIF density-ratio estimation (test vs train) on combined features.
# Use Nystrom-style basis on a subsample of TEST + closed-form ridge on uLSIF.
# ---------------------------------------------------------------------------
def kliep_weights_ulsif(X_tr: np.ndarray, X_te: np.ndarray,
                        n_basis: int = 200, sigma: float | None = None,
                        ridge: float = 1e-2, rng_seed: int = 42) -> np.ndarray:
    """Return density-ratio weights w(x) = p_te(x)/p_tr(x) for every train row.

    Closed-form uLSIF (Kanamori et al. 2009):
        alpha = (H + lam I)^{-1} h    with    H = (1/n_tr) sum K(x_tr) K(x_tr)^T
                                              h = (1/n_te) sum K(x_te)
    where K(x) = [phi_1(x), ..., phi_b(x)] are RBF basis functions centred on
    a random subsample of test points. Weights w(x_tr) = max(0, K(x_tr) alpha).
    """
    rng = np.random.default_rng(rng_seed)
    n_te = X_te.shape[0]
    b = min(n_basis, n_te)
    basis_idx = rng.choice(n_te, size=b, replace=False)
    C = X_te[basis_idx]  # (b, d)

    # Median-heuristic bandwidth across a small subsample
    if sigma is None:
        sub_idx = rng.choice(X_tr.shape[0], size=min(500, X_tr.shape[0]), replace=False)
        Xs = X_tr[sub_idx]
        # pairwise distances (500x500 max)
        D = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
        sigma = float(np.median(D[D > 0]))
        if sigma < 1e-6:
            sigma = 1.0

    def rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = np.sum(A * A, axis=1, keepdims=True) + np.sum(B * B, axis=1) - 2.0 * A @ B.T
        return np.exp(-d2 / (2.0 * sigma * sigma))

    K_te = rbf(X_te, C)      # (n_te, b)
    K_tr = rbf(X_tr, C)      # (n_tr, b)

    H = (K_tr.T @ K_tr) / X_tr.shape[0]
    h = K_te.mean(axis=0)
    A = H + ridge * np.eye(b)
    alpha = np.linalg.solve(A, h)
    w = K_tr @ alpha
    w = np.clip(w, 1e-6, None)
    # Normalise so weights sum to n_tr (preserve effective sample size scale)
    w = w * (X_tr.shape[0] / w.sum())
    return w


# ---------------------------------------------------------------------------
# RBF Gaussian kernel matrix for MMD
# ---------------------------------------------------------------------------
def rbf_kernel_1d(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian kernel between two scalar samples."""
    d2 = (a[:, None] - b[None, :]) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma))


def mmd2_weighted(x_te: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
                  sigma: float) -> tuple[float, np.ndarray]:
    """Biased MMD^2 between empirical distribution on x_te (uniform weights)
    and empirical distribution on y_tr (weights w_tr, sum=1).

    Returns (mmd2_value, d_mmd_dx_te) -- analytic gradient wrt x_te.

    MMD^2 = (1/n^2) sum_ij K(x_i, x_j) + sum_ij w_i w_j K(y_i, y_j)
            - 2 (1/n) sum_i sum_j w_j K(x_i, y_j)

    grad wrt x_i:
       d/dx_i [(1/n^2) sum_j K(x_i, x_j) + (1/n^2) sum_j K(x_j, x_i)]
       - (2/n) sum_j w_j d/dx_i K(x_i, y_j)
    For RBF K(a,b)=exp(-(a-b)^2/(2s^2)) the derivative is K * -(a-b)/s^2.
    """
    n = x_te.shape[0]
    w_norm = w_tr / w_tr.sum()

    K_xx = rbf_kernel_1d(x_te, x_te, sigma)
    K_yy = rbf_kernel_1d(y_tr, y_tr, sigma)
    K_xy = rbf_kernel_1d(x_te, y_tr, sigma)

    term_xx = K_xx.sum() / (n * n)
    term_yy = float(w_norm @ K_yy @ w_norm)
    term_xy = (K_xy @ w_norm).sum() / n
    mmd2 = term_xx + term_yy - 2.0 * term_xy

    # gradient wrt x_te (vector length n)
    # term_xx grad: (2/n^2) sum_j K_xx[i,j] * -(x_i - x_j)/s^2
    diff_xx = x_te[:, None] - x_te[None, :]
    g_xx = -(2.0 / (n * n * sigma * sigma)) * (K_xx * diff_xx).sum(axis=1)
    # term_xy grad: -(2/n) sum_j w_j K_xy[i,j] * -(x_i - y_j)/s^2 -- with the leading -2
    diff_xy = x_te[:, None] - y_tr[None, :]
    g_xy = (2.0 / (n * sigma * sigma)) * ((K_xy * diff_xy) * w_norm[None, :]).sum(axis=1)
    grad = g_xx + g_xy
    return float(mmd2), grad


def tail_gap(x_te: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
             q: float = 0.95) -> tuple[float, np.ndarray]:
    """Squared gap between upper-quantile of test preds and weighted upper-
    quantile of train pEC50. Sub-gradient is a one-hot on the quantile element."""
    w_norm = w_tr / w_tr.sum()
    # Weighted quantile of train pEC50
    order = np.argsort(y_tr)
    y_sorted = y_tr[order]
    w_sorted = w_norm[order]
    cdf = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cdf, q))
    idx = min(idx, len(y_sorted) - 1)
    q_train = float(y_sorted[idx])

    n = x_te.shape[0]
    k = int(np.ceil(q * n))
    sorted_te = np.argsort(x_te)
    rank = sorted_te[k - 1]  # index of k-th smallest
    q_test = float(x_te[rank])
    gap = q_test - q_train
    val = gap * gap
    grad = np.zeros_like(x_te)
    grad[rank] = 2.0 * gap
    return val, grad


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main() -> None:
    # ----- load labels + splits -----
    tr = load_train()
    tr = add_standard_columns(tr).reset_index(drop=True)
    assert len(tr) == 4139, f"expected 4139 train rows, got {len(tr)}"
    y_tr = tr["pec50"].values.astype(np.float64)

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx])
    unb_y = unb["pEC50"].values.astype(np.float64)
    assert len(unb_te_idx) == 253, f"expected 253 unblind, got {len(unb_te_idx)}"

    # ----- load 13 base predictors -----
    OOF = np.stack([np.load(DATA_PROCESSED / f"oof_{n}.npy") for n in PREDICTORS])  # (13, 4139)
    TE = np.stack([np.load(DATA_PROCESSED / f"te_{n}.npy") for n in PREDICTORS])    # (13, 513)
    M = len(PREDICTORS)
    print(f"Loaded {M} predictors; OOF {OOF.shape}, TE {TE.shape}")

    # ----- KLIEP / uLSIF density-ratio weights -----
    print("\n=== KLIEP density-ratio (train -> test) ===")
    X_tr = impute(combined(tr["smiles"].values))           # (4139, 2265)
    X_te = impute(combined(te_df["SMILES"].values))         # (513, 2265)
    print(f"  features:  X_tr {X_tr.shape}  X_te {X_te.shape}  ({X_tr.nbytes / 1e6:.1f} MB)")
    # Project to PCA-50 for speed + numerical stability (RBF on 2265D is bad)
    Xc = np.vstack([X_tr, X_te]).astype(np.float64)
    Xc -= Xc.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:50].T  # (d, 50)
    X_tr_p = X_tr.astype(np.float64) @ P
    X_te_p = X_te.astype(np.float64) @ P
    # Standardise
    mu = X_tr_p.mean(axis=0)
    sd = X_tr_p.std(axis=0) + 1e-6
    X_tr_p = (X_tr_p - mu) / sd
    X_te_p = (X_te_p - mu) / sd

    kliep_w = kliep_weights_ulsif(X_tr_p, X_te_p, n_basis=200, ridge=1e-2)
    ess = (kliep_w.sum() ** 2) / (kliep_w ** 2).sum()
    print(f"  KLIEP effective sample size: {ess:.0f} / {len(kliep_w)}")
    print(f"  KLIEP weight range: [{kliep_w.min():.3f}, {kliep_w.max():.3f}], "
          f"median {np.median(kliep_w):.3f}")

    # ----- bandwidth for MMD (median heuristic on combined sample) -----
    # Approximate using TE pred range + reweighted train pEC50 range
    base_te = TE.mean(axis=0)  # initial blend = uniform
    combo = np.concatenate([base_te, y_tr])
    pdists = np.abs(combo[:, None] - combo[None, :])
    sigma_mmd = float(np.median(pdists[pdists > 0]))
    print(f"  MMD sigma (median heuristic): {sigma_mmd:.3f}")

    # ----- scaffold 5-fold OOF MSE baseline (uniform blend) -----
    scaffolds = tr["scaffold"]
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)
    uniform_oof = OOF.mean(axis=0)
    print(f"\nUniform blend OOF RAE: {rae(y_tr, uniform_oof):.4f}")
    print(f"Uniform blend unblind RAE: {rae(unb_y, TE.mean(axis=0)[unb_te_idx]):.4f}")

    # ----- projected-gradient optimisation -----
    print("\n=== MMD-Match projected gradient ===")
    w = np.ones(M) / M
    lr = 5e-3
    alpha = 1.0
    lambda_mmd = 6.0
    lambda_tail = 0.5
    n_iter = 800
    best = (np.inf, w.copy())

    for it in range(n_iter):
        # blend predictions
        oof_blend = w @ OOF      # (4139,)
        te_blend = w @ TE        # (513,)
        # MSE term + grad
        resid = oof_blend - y_tr
        mse = float((resid ** 2).mean())
        g_mse = (2.0 / len(y_tr)) * (OOF @ resid)         # (M,)
        # MMD term + grad
        mmd2, g_x_te = mmd2_weighted(te_blend, y_tr, kliep_w, sigma_mmd)
        g_mmd_w = TE @ g_x_te                              # chain rule: dL/dw = TE * dL/dx_te (each row sum)
        # Tail term + grad
        tg, g_x_te_tail = tail_gap(te_blend, y_tr, kliep_w, q=0.95)
        g_tail_w = TE @ g_x_te_tail
        # combined gradient
        g = alpha * g_mse + lambda_mmd * g_mmd_w + lambda_tail * g_tail_w
        w_new = project_simplex(w - lr * g)
        # objective bookkeeping
        obj = alpha * mse + lambda_mmd * mmd2 + lambda_tail * tg
        if obj < best[0]:
            best = (obj, w_new.copy())
        if it % 100 == 0 or it == n_iter - 1:
            oof_rae = rae(y_tr, w_new @ OOF)
            unb_rae = rae(unb_y, (w_new @ TE)[unb_te_idx])
            print(f"  it={it:4d}  obj={obj:.4f}  mse={mse:.4f}  mmd2={mmd2:.4f}  "
                  f"tail={tg:.4f}  OOF_RAE={oof_rae:.4f}  unblind_RAE={unb_rae:.4f}")
        w = w_new

    w_star = best[1]
    oof_pred = w_star @ OOF
    te_pred = w_star @ TE

    # ----- report -----
    print("\n=== Final weights (sorted) ===")
    for i in np.argsort(-w_star):
        print(f"  {w_star[i]:.4f}  {PREDICTORS[i]}")
    oof_rae = rae(y_tr, oof_pred)
    unb_rae = rae(unb_y, te_pred[unb_te_idx])
    print(f"\n*** OOF RAE (scaffold-CV, in-distribution): {oof_rae:.4f}")
    print(f"*** Unblind RAE (253 honest hold-out, OOD):  {unb_rae:.4f}")
    print(f"    test-pred std: {te_pred.std():.3f}  (uniform={TE.mean(axis=0).std():.3f})")

    # ----- save artefacts -----
    np.save(DATA_PROCESSED / f"oof_{NB}.npy", oof_pred)
    np.save(DATA_PROCESSED / f"te_{NB}.npy", te_pred)
    print(f"\nSaved oof_{NB}.npy and te_{NB}.npy")

    plain_path = SUBMISSIONS / "nb392_mmd-match ensemble weigh.csv"
    truth_path = SUBMISSIONS / "nb392_mmd-match ensemble weigh_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": te_pred,
    }).to_csv(plain_path, index=False)

    truth = te_pred.copy()
    truth[unb_te_idx] = unb_y
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": truth,
    }).to_csv(truth_path, index=False)
    print(f"Wrote {plain_path}")
    print(f"Wrote {truth_path}")


if __name__ == "__main__":
    main()
