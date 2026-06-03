"""nb910 -- Gaussian Process regression with a Tanimoto kernel.

Hypothesis: kernel methods give calibrated uncertainty that tree models
can't, and the Tanimoto kernel encodes the structural-similarity prior more
directly than fingerprint-fed boosting trees.

Pipeline:
  1) Morgan-FP (r=2, 2048-bit, bool) for 4139 train + 513 test.
  2) Build the full N x N (N=4652) Tanimoto kernel:
        K[i,j] = |A & B| / |A | B|
     using vectorized popcount on a packed uint8 matrix.
  3) Direct GP via Cholesky:
        L = chol(K_tr + sigma^2 I)
        alpha = solve(L^T, solve(L, y_tr_centered))
        mean(x*) = K(x*, X_tr) @ alpha + mu_y
        var(x*)  = 1 - k_*^T (K + sigma^2 I)^-1 k_*    (Tanimoto self-sim = 1)
  4) Hyperparameter: noise sigma = 0.4 (the pEC50 SE noise floor is ~0.24,
     but assay-to-analog variance dominates -> 0.4 is the empirical regularizer).
  5) Predict pEC50 for the 513 test compounds and score against the 253
     Phase-1-unblind labels.

Outputs:
  C:/pxr_artifacts/nb910_K_tanimoto.npy        # (4652,4652) float32
  C:/pxr_artifacts/nb910_te_pred.npy           # (513,)      float32
  C:/pxr_artifacts/nb910_te_std.npy            # (513,)      float32
  data/processed/te_nb910.npy                  # (513,)      float32
  submissions/nb910_gp_tanimoto.csv            # 513-row submission

Memory:
  - packed FP matrix:  4652 * 256 bytes  ~= 1.2 MB (bool->uint8 packed by 8)
  - kernel float32:    4652^2 * 4        ~= 86 MB
  - Cholesky of 4139:  68 MB             -> peak ~250 MB
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.linalg import cho_factor, cho_solve

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_train
from pxr.chem import add_standard_columns
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

TAG = "nb910"
SEED = 0
SIGMA = 0.4  # GP noise std on pEC50
FP_RADIUS = 2
FP_BITS = 2048

ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)


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
    """Return (N, FP_BITS) bool array of Morgan FP bits."""
    out = np.zeros((len(smiles), FP_BITS), dtype=bool)
    for i, smi in enumerate(smiles):
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_BITS)
        # fast bitvect -> numpy
        arr = np.zeros(FP_BITS, dtype=np.int8)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        out[i] = arr.astype(bool)
    return out


# ----------------------------------------------------------------------------
# Tanimoto kernel via packed-bit popcount (block matmul)
# ----------------------------------------------------------------------------
def tanimoto_kernel(fp_bool: np.ndarray) -> np.ndarray:
    """K[i,j] = |A_i & A_j| / |A_i | A_j|; fp_bool is (N, FP_BITS) bool.

    Uses dense matmul on uint8 (each bit is one column) -> intersection counts.
    Union = popA + popB - inter.  Returns float32 (N,N).
    """
    X = fp_bool.astype(np.uint8)                     # (N, B)
    pop = X.sum(axis=1).astype(np.int32)             # (N,)
    # Intersection counts via matmul; cast to int32 to fit (max = FP_BITS).
    inter = X @ X.T                                  # (N, N) int (sum of uint8)
    inter = inter.astype(np.int32, copy=False)
    union = pop[:, None] + pop[None, :] - inter
    # Tanimoto: handle empty fingerprints -> sim = 1 if both empty, else 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    np.fill_diagonal(K, 1.0)
    return K


# ----------------------------------------------------------------------------
# GP fit via Cholesky
# ----------------------------------------------------------------------------
def gp_predict(K_full: np.ndarray, n_tr: int, y_tr: np.ndarray, sigma: float):
    """Direct GP regression with a precomputed kernel.

    K_full layout: [train rows then test rows] x [train cols then test cols].
    Returns mean and predictive std for the test rows.
    """
    K_tt = K_full[:n_tr, :n_tr].astype(np.float64, copy=False)
    K_st = K_full[n_tr:, :n_tr].astype(np.float64, copy=False)
    # diag of K_test_test == 1.0 for Tanimoto self-similarity
    A = K_tt + (sigma ** 2) * np.eye(n_tr, dtype=np.float64)
    mu_y = float(y_tr.mean())
    y0 = y_tr.astype(np.float64) - mu_y

    c, low = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
    alpha = cho_solve((c, low), y0, check_finite=False)
    mean = (K_st @ alpha) + mu_y

    # predictive variance per test point: 1 - k_*^T A^-1 k_*
    V = cho_solve((c, low), K_st.T, check_finite=False)         # (n_tr, n_te)
    var = 1.0 - np.einsum("ij,ji->i", K_st, V)                  # (n_te,)
    var = np.clip(var, 1e-8, None) + sigma ** 2
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- GP regression with Tanimoto kernel (sigma={SIGMA})")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    miss = [k for k, p in needed.items() if not Path(p).exists()]
    if miss:
        print("MISSING raw inputs:", miss)
        return {"success": False, "missing": miss}

    tr = add_standard_columns(load_train())
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    smiles_tr = tr["std_smiles"].tolist()
    y_tr = tr["pec50"].astype(np.float64).values
    smiles_te = te_df["SMILES"].apply(_std_smi).tolist()
    n_tr, n_te = len(smiles_tr), len(smiles_te)
    print(f"Train n={n_tr}  Test n={n_te}")

    print("Computing Morgan FPs (r=2, 2048-bit)...")
    fp_tr = morgan_bits(smiles_tr)
    fp_te = morgan_bits(smiles_te)
    fp_all = np.concatenate([fp_tr, fp_te], axis=0)
    print(f"  fp_all shape={fp_all.shape}  popcount mean={fp_all.sum(1).mean():.1f}")

    print("Building Tanimoto kernel (4652 x 4652)...")
    K = tanimoto_kernel(fp_all)
    print(f"  K dtype={K.dtype}  size={K.nbytes/1e6:.1f} MB")
    print(f"  off-diag mean={K[np.triu_indices_from(K, k=1)][:200000].mean():.3f}  "
          f"diag mean={np.diag(K).mean():.3f}")
    np.save(ART / f"{TAG}_K_tanimoto.npy", K)

    print(f"Solving GP (sigma={SIGMA}) via Cholesky...")
    te_pred, te_std = gp_predict(K, n_tr, y_tr, sigma=SIGMA)
    print(f"  pred mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"  pred_std mean={te_std.mean():.3f}  min/max={te_std.min():.3f}/{te_std.max():.3f}")

    # ---- Unblind scoring ---------------------------------------------------
    te_names = te_df["Molecule Name"].tolist()
    n2i = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(n2i)].reset_index(drop=True)
    unb_idx = np.array([n2i[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    rae_u = float(rae(unb_y, te_pred[unb_idx]))

    print("=" * 78)
    print(f"UNBLIND (n={len(unb_idx)}): RAE = {rae_u:.4f}")
    print(f"  truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"  pred  mean/std = {te_pred[unb_idx].mean():.3f} / {te_pred[unb_idx].std():.3f}")
    print("=" * 78)

    # ---- Save artifacts ----------------------------------------------------
    np.save(ART / f"{TAG}_te_pred.npy", te_pred)
    np.save(ART / f"{TAG}_te_std.npy",  te_std)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_pred)

    sub = SUBMISSIONS / f"{TAG}_gp_tanimoto.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         te_pred,
    }).to_csv(sub, index=False)
    print(f"Wrote {sub}")

    return {
        "success": True,
        "in_rae": rae_u,
        "pred_std_mean": float(te_std.mean()),
        "submission": str(sub),
    }


if __name__ == "__main__":
    out = main()
    print("\nRESULT:", out)
