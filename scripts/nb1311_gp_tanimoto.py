"""nb1311 -- Gaussian Process (Tanimoto kernel) residual learner on nb1070.

Hypothesis:
    GP-Tanimoto provides Bayesian uncertainty AND non-parametric flexibility.
    Bayes-optimal kernel regression on chemistry-similarity may extract
    structure that parametric (LGBM/tree) residual learners miss.
    Compare to nb1183 (MACCS-residual bag, 0.5513) and nb1242 (ChEMBL-kNN
    residual bag, 0.5431).  Verdict at 0.003 margin.

Protocol (identical scaffolding to nb1183/nb1242 except GP replaces LGBM):
    1. Anchor   = nb1070_pred_oof on 253 unblind rows (constant across "seeds").
    2. Target   = y_unb - anchor (residual).
    3. Features = Morgan-2048 r=2 for the 513 test rows, sliced to 253 unblind.
    4. Kernel   = Tanimoto over Morgan-2048 -> precomputed (253, 253) NxN K.
    5. Bag      = 5 GPs with different noise level alpha in {0.05, 0.1, 0.2,
                  0.3, 0.5}.  GP with a precomputed kernel is otherwise
                  deterministic, so alpha plays the role of the seed factor.
    6. CV       = KFold(n=5, shuffle=True, random_state=0).  Same split every
                  alpha so per-alpha differences are PURELY the noise floor.
    7. Per-alpha pred_corrected = anchor + residual_oof_alpha; pooled RAE.
    8. Mean-bag pred_corrected  = anchor + mean over alpha of residual_oof_alpha.
    9. Verdict at DECISION_MARGIN=0.003 vs nb1183 (0.5513) and nb1242 (0.5431).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
    scripts/nb1311_gp_tanimoto_residual.py            (this file)
    data/processed/nb1311_summary.json
    data/processed/nb1311_K_tanimoto_unb.npy          (253, 253) float32
    data/processed/nb1311_per_seed_corrected_oof.npy  (5, 253)   float32
    data/processed/nb1311_per_seed_resid_std.npy      (5, 253)   float32
    data/processed/nb1311_mean_bag_oof.npy            (253,)     float32
    data/processed/nb1311_median_bag_oof.npy          (253,)     float32
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW

TAG = "nb1311"
ANCHOR = "nb1070"

RESID_FOLDS = 5
KFOLD_SEED = 0                       # one shared CV split across alphas
ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.50]   # "seed factor" for GP bag (noise level)

FP_RADIUS = 2
FP_BITS = 2048

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF = 0.5771
NB1183_REF = 0.5513   # MACCS residual bag on nb1070
NB1242_REF = 0.5431   # ChEMBL-kNN residual bag on nb1070
DECISION_MARGIN = 0.003


# ----------------------------------------------------------------------------
# Featurization
# ----------------------------------------------------------------------------
def _std_smi(smi: str) -> str | None:
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def morgan_bits(smiles: list[str]) -> np.ndarray:
    """Return (N, FP_BITS) bool Morgan-FP bits."""
    out = np.zeros((len(smiles), FP_BITS), dtype=bool)
    for i, smi in enumerate(smiles):
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_BITS)
        arr = np.zeros(FP_BITS, dtype=np.int8)
        ConvertToNumpyArray(fp, arr)
        out[i] = arr.astype(bool)
    return out


def tanimoto_kernel(fp_bool: np.ndarray) -> np.ndarray:
    """Symmetric (N,N) Tanimoto kernel from a (N, FP_BITS) bool fingerprint
    matrix.  K[i,i] = 1.0; K is float32."""
    X = fp_bool.astype(np.uint8)
    pop = X.sum(axis=1).astype(np.int32)
    inter = (X @ X.T).astype(np.int32, copy=False)
    union = pop[:, None] + pop[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    np.fill_diagonal(K, 1.0)
    return K


# ----------------------------------------------------------------------------
# GP fit on precomputed kernel
# ----------------------------------------------------------------------------
def _gp_cross_fit_one_alpha(
    K: np.ndarray, residual: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """5-fold cross-fit of a GP residual learner on a precomputed Tanimoto
    kernel.  Returns (oof_mean, oof_std) of shape (n,)."""
    n = K.shape[0]
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=KFOLD_SEED)
    oof_mean = np.full(n, np.nan, dtype=np.float64)
    oof_std = np.full(n, np.nan, dtype=np.float64)
    # Constant ConstantKernel(1.0, "fixed") just multiplies the precomputed
    # kernel by 1 -- effectively a no-op; GPR requires *some* sklearn Kernel
    # object when X is the precomputed Gram matrix.  alpha is the per-target
    # noise variance added to the kernel diagonal during fit.
    base_kernel = ConstantKernel(1.0, constant_value_bounds="fixed")
    for tr_loc, va_loc in kf.split(np.arange(n)):
        K_tr = K[np.ix_(tr_loc, tr_loc)].astype(np.float64, copy=False)
        K_va = K[np.ix_(va_loc, tr_loc)].astype(np.float64, copy=False)
        # We must hand sklearn an X-like object for fit / predict.  Pass the
        # precomputed Gram block directly; the ConstantKernel(1, fixed) treats
        # X as the kernel rows when called with (X, Y).  We bypass sklearn's
        # __call__ by precomputing -- so use a CustomPrecomputed wrapper.
        gp = _PrecomputedGPR(alpha=alpha)
        gp.fit(K_tr, residual[tr_loc])
        mu, sd = gp.predict(K_va)
        oof_mean[va_loc] = mu
        oof_std[va_loc] = sd
    return oof_mean, oof_std


class _PrecomputedGPR:
    """Thin GP regressor that takes precomputed Gram matrices.

    Faster and simpler than wrestling sklearn's GaussianProcessRegressor into
    a precomputed-kernel mode.  Mirrors nb910's Cholesky path:
        K_tt + alpha I = L L^T
        beta = solve(L^T, solve(L, y - mu_y))
        mean(*) = K_st @ beta + mu_y
        var(*)  = 1 - K_st^T (K_tt+alpha I)^-1 K_st^T  + alpha
    """

    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def fit(self, K_tt: np.ndarray, y: np.ndarray) -> "_PrecomputedGPR":
        from scipy.linalg import cho_factor
        n = K_tt.shape[0]
        A = K_tt + (self.alpha ** 2) * np.eye(n, dtype=np.float64)
        self._mu_y = float(np.mean(y))
        y0 = np.asarray(y, dtype=np.float64) - self._mu_y
        self._cho = cho_factor(A, lower=True, overwrite_a=True,
                               check_finite=False)
        from scipy.linalg import cho_solve
        self._beta = cho_solve(self._cho, y0, check_finite=False)
        return self

    def predict(self, K_st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.linalg import cho_solve
        K_st = np.asarray(K_st, dtype=np.float64)
        mean = K_st @ self._beta + self._mu_y
        V = cho_solve(self._cho, K_st.T, check_finite=False)
        var = 1.0 - np.einsum("ij,ji->i", K_st, V)
        var = np.clip(var, 1e-12, None) + self.alpha ** 2
        std = np.sqrt(var)
        return mean.astype(np.float64), std.astype(np.float64)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- GP (Tanimoto kernel) residual learner on top of "
          f"{ANCHOR}, {len(ALPHAS)}-alpha bag")
    print(f"          alphas (noise level) = {ALPHAS}")
    print(f"          features = Morgan-{FP_BITS} r={FP_RADIUS} (test only, "
          f"sliced to 253 unblind)")
    print(f"          residual target = y_unb - {ANCHOR}_pred_oof")
    print(f"          CV: KFold(n={RESID_FOLDS}, shuffle=True, "
          f"random_state={KFOLD_SEED}) -- shared across alphas")
    print("=" * 78)

    # ---- Load 513 test SMILES, 253 unblind indexing & truth ---------------
    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy").astype(int)
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Anchor ----------------------------------------------------------
    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; required anchor OOF (run nb1070 first)."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: "
            f"{anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ---- Morgan FP + Tanimoto kernel for 253 unblind ---------------------
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    smiles_te = te[smi_col].apply(_std_smi).tolist()
    smiles_unb = [smiles_te[i] for i in unb_idx]
    n_bad = sum(1 for s in smiles_unb if not s)
    print(f"[feat] standardized {n_unb} unblind SMILES "
          f"({n_bad} parse failures kept as empty)")

    print(f"[feat] computing Morgan FPs (r={FP_RADIUS}, "
          f"{FP_BITS}-bit) for 253 unblind ...")
    fp_unb = morgan_bits(smiles_unb)
    print(f"[feat] fp_unb shape={fp_unb.shape}  "
          f"popcount mean={fp_unb.sum(1).mean():.1f}  "
          f"density={fp_unb.mean():.4f}")

    print("[kern] building Tanimoto kernel (253 x 253) ...")
    K = tanimoto_kernel(fp_unb)
    off = K[np.triu_indices_from(K, k=1)]
    print(f"[kern] K shape={K.shape}  dtype={K.dtype}  "
          f"size={K.nbytes/1e6:.2f} MB  "
          f"off-diag mean={off.mean():.3f}  off-diag max={off.max():.3f}  "
          f"diag mean={np.diag(K).mean():.3f}")

    np.save(DATA_PROCESSED / f"{TAG}_K_tanimoto_unb.npy", K)

    # ---- Per-alpha GP residual cross-fit ---------------------------------
    print("\n" + "-" * 78)
    print(f"PER-ALPHA GP RESIDUAL CROSS-FIT (Tanimoto, {n_unb} rows)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(ALPHAS), n_unb), dtype=np.float64)
    per_seed_resid_std = np.zeros((len(ALPHAS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, a in enumerate(ALPHAS):
        resid_mean, resid_std = _gp_cross_fit_one_alpha(K, residual, alpha=a)
        pred_corr = anchor_oof + resid_mean
        per_seed_corrected[i] = pred_corr
        per_seed_resid_std[i] = resid_std
        rae_a = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_a)
        delta_a = rae_a - rae_anchor
        per_seed_records.append({
            "alpha": float(a),
            "rae_corrected": rae_a,
            "delta_vs_nb1070": delta_a,
            "resid_oof_mean": float(resid_mean.mean()),
            "resid_oof_std": float(resid_mean.std()),
            "gp_uncertainty_mean": float(resid_std.mean()),
            "gp_uncertainty_median": float(np.median(resid_std)),
        })
        print(f"   alpha {a:.2f}:  rae_corr = {rae_a:.4f}  "
              f"(d_vs_nb1070 = {delta_a:+.4f})  "
              f"|resid_oof|.std = {resid_mean.std():.3f}  "
              f"gp_unc.mean = {resid_std.mean():.3f}")

    # ---- Bag aggregations ------------------------------------------------
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    gp_unc_global_mean = float(per_seed_resid_std.mean())
    gp_unc_global_median = float(np.median(per_seed_resid_std))

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-alpha RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-alpha mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-alpha median        = {rae_per_seed_median:.4f}")
    print(f"   per-alpha std           = {rae_per_seed_std:.4f}")
    print(f"   per-alpha min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)    = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag)  = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   gp_uncertainty mean     = {gp_unc_global_mean:.4f}")
    print(f"   gp_uncertainty median   = {gp_unc_global_median:.4f}")
    print(f"   nb1183 mean_bag ref     = {NB1183_REF:.4f}  "
          f"(MACCS residual on nb1070)")
    print(f"   nb1242 mean_bag ref     = {NB1242_REF:.4f}  "
          f"(ChEMBL-kNN residual on nb1070)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_REF - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "GP_TANIMOTO_RESIDUAL_BEATS_NB1242_CHEMBL_KNN"
    elif beats_nb1183:
        verdict = "GP_TANIMOTO_RESIDUAL_BEATS_NB1183_BUT_NOT_NB1242"
    elif beats_nb1070:
        verdict = "GP_TANIMOTO_RESIDUAL_HELPS_NB1070_BUT_NOT_NB1183"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "GP_TANIMOTO_RESIDUAL_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "GP_TANIMOTO_RESIDUAL_HURTS_NB1070"
    print(f"   verdict                 = {verdict}")

    # ---- Persist artifacts -----------------------------------------------
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_resid_std.npy",
            per_seed_resid_std.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_seed_resid_std.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_K_tanimoto_unb.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": f"morgan_r{FP_RADIUS}_{FP_BITS}bit",
        "kernel": "tanimoto",
        "method": "precomputed_gp_cholesky_residual_learner",
        "n_unb": n_unb,
        "alphas": ALPHAS,
        "resid_folds": RESID_FOLDS,
        "kfold_seed": KFOLD_SEED,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "kernel_offdiag_mean": float(off.mean()),
        "kernel_offdiag_max": float(off.max()),
        "kernel_offdiag_median": float(np.median(off)),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "gp_uncertainty_mean": gp_unc_global_mean,
        "gp_uncertainty_median": gp_unc_global_median,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_median_bag_vs_nb1070": rae_median_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_REF,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1242": bool(beats_nb1242),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1183_mean_bag_ref": NB1183_REF,
        "nb1242_mean_bag_ref": NB1242_REF,
        "decision_margin": DECISION_MARGIN,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_anchor_nb1070",
        "per_seed_rae",
        "rae_per_seed_mean",
        "rae_per_seed_median",
        "rae_per_seed_std",
        "rae_mean_bag",
        "rae_median_bag",
        "gp_uncertainty_mean",
        "gp_uncertainty_median",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1183",
        "delta_mean_bag_vs_nb1242",
        "beats_nb1070",
        "beats_nb1183",
        "beats_nb1242",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
