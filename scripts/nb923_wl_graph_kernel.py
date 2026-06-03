"""nb923 -- Weisfeiler-Lehman subtree graph kernel + kernel ridge.

Hypothesis: structural subtree counts (Morgan-like in spirit but at the graph
level, not bit-collisioned) give a different similarity prior than ECFP4 +
matmul Tanimoto. WL has been the workhorse graph-classification kernel for a
decade; it should at minimum contribute a decorrelated similarity view.

Pipeline:
  1) RDKit SMILES -> molecular graph (atom = atomic number, bond = bond order).
  2) Hand-coded 3-iteration WL relabeling. At each iteration:
        new_label[v] = hash( (old_label[v], sorted([ (bond, old_label[u]) ])) )
     Re-canonicalize globally so labels are dense int32.
  3) WL kernel = sum over iterations of phi_iter(x) . phi_iter(x'), where
     phi_iter(x) is the label-count histogram for iteration iter. We
     materialize a sparse (N x |labels|) matrix per iteration and add up
     the iter-wise Gram matrices.
  4) Kernel ridge: alpha = (K_tt + 0.1 I)^-1 (y - y.mean()); test mean is
     K_st @ alpha + y.mean(). No GP variance (kernel isn't unit-diag).
  5) Score against the 253 Phase-1 unblinded labels.

Grakel was attempted (uv pip install grakel) but its build requires MSVC; we
fall back to a NumPy/RDKit hand-coded WL, which is fast enough at N=4652.

Outputs:
  C:/pxr_artifacts/nb923_K_wl.npy       # (4652, 4652) float32
  C:/pxr_artifacts/nb923_te_pred.npy    # (513,) float32
  data/processed/te_nb923.npy           # (513,) float32
  submissions/nb923_wl_graph_kernel.csv # 513-row submission

Wall-time budget: < 12 min on CPU; N=4652 keeps the Gram matrix at ~86 MB
float32 and the Cholesky solve at ~140 MB float64.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_train
from pxr.chem import add_standard_columns
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

TAG = "nb923"
N_ITER = 3            # WL relabeling rounds (h=3 -> 4 hist incl. iter 0)
ALPHA = 0.1           # kernel-ridge regularizer

ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# SMILES -> initial WL graph (atomic-number labels + bond-order edges)
# ----------------------------------------------------------------------------
def smiles_to_graph(smi: str):
    """Return (init_labels int32 (n_atoms,), neighbors list of (j, bond_int)).

    Atom label = atomic number. Bond label = int(bond_type_as_double * 2) so
    aromatic (1.5) -> 3, single -> 2, double -> 4, triple -> 6 (distinct ints).
    """
    m = Chem.MolFromSmiles(str(smi)) if smi else None
    if m is None:
        return np.zeros(1, dtype=np.int32), [[]]
    n = m.GetNumAtoms()
    if n == 0:
        return np.zeros(1, dtype=np.int32), [[]]
    init = np.fromiter((a.GetAtomicNum() for a in m.GetAtoms()), dtype=np.int32,
                       count=n)
    nbrs: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = int(round(b.GetBondTypeAsDouble() * 2))
        nbrs[i].append((j, bt))
        nbrs[j].append((i, bt))
    return init, nbrs


# ----------------------------------------------------------------------------
# WL relabeling: per-iteration label histograms across all graphs
# ----------------------------------------------------------------------------
def wl_label_histograms(graphs):
    """For each WL iteration 0..N_ITER, return a sparse (N_graphs, |labels|)
    count matrix. The label dictionary is built globally per iteration so
    different molecules share columns when they share subtrees.

    Returns: list[csr_matrix] of length N_ITER+1.
    """
    n_graphs = len(graphs)
    cur_labels = [g[0].copy() for g in graphs]   # int32 per graph
    nbrs_all = [g[1] for g in graphs]

    histograms: list[csr_matrix] = []

    # iter 0: raw atomic-number counts -- canonicalize globally to dense ints
    hist0_rows, hist0_cols, hist0_data = [], [], []
    label_dict0: dict[int, int] = {}
    for gi, labs in enumerate(cur_labels):
        counts: dict[int, int] = defaultdict(int)
        for lab in labs:
            counts[int(lab)] += 1
        for lab, c in counts.items():
            if lab not in label_dict0:
                label_dict0[lab] = len(label_dict0)
            hist0_rows.append(gi)
            hist0_cols.append(label_dict0[lab])
            hist0_data.append(c)
    H0 = csr_matrix(
        (np.asarray(hist0_data, dtype=np.float32),
         (np.asarray(hist0_rows, dtype=np.int32),
          np.asarray(hist0_cols, dtype=np.int32))),
        shape=(n_graphs, len(label_dict0)),
    )
    histograms.append(H0)
    print(f"  iter 0: |labels|={len(label_dict0):,}  nnz={H0.nnz:,}")

    # iters 1..N_ITER: each atom relabels to hash(old, sorted[(bond,old_nbr)])
    for it in range(1, N_ITER + 1):
        label_dict: dict[bytes, int] = {}
        new_labels = [None] * n_graphs
        hist_rows, hist_cols, hist_data = [], [], []
        for gi in range(n_graphs):
            labs = cur_labels[gi]
            nbrs = nbrs_all[gi]
            n = len(labs)
            new = np.zeros(n, dtype=np.int32)
            counts: dict[int, int] = defaultdict(int)
            for v in range(n):
                # multi-set of (bond_label, neighbor_old_label)
                nb = sorted((bt, int(labs[u])) for (u, bt) in nbrs[v])
                # canonical bytes signature -> hash once -> dense int id
                key = repr((int(labs[v]), tuple(nb))).encode("utf-8")
                # short stable digest (8 bytes) for the lookup key
                key_h = hashlib.blake2b(key, digest_size=8).digest()
                idx = label_dict.get(key_h)
                if idx is None:
                    idx = len(label_dict)
                    label_dict[key_h] = idx
                new[v] = idx
                counts[idx] += 1
            new_labels[gi] = new
            for lab, c in counts.items():
                hist_rows.append(gi)
                hist_cols.append(lab)
                hist_data.append(c)
        cur_labels = new_labels
        H = csr_matrix(
            (np.asarray(hist_data, dtype=np.float32),
             (np.asarray(hist_rows, dtype=np.int32),
              np.asarray(hist_cols, dtype=np.int32))),
            shape=(n_graphs, len(label_dict)),
        )
        histograms.append(H)
        print(f"  iter {it}: |labels|={len(label_dict):,}  nnz={H.nnz:,}")

    return histograms


def wl_kernel(histograms) -> np.ndarray:
    """K = sum_iter H_iter @ H_iter.T  (dense float32)."""
    n = histograms[0].shape[0]
    K = np.zeros((n, n), dtype=np.float32)
    for it, H in enumerate(histograms):
        # sparse @ sparse.T -> sparse, then to dense
        G = (H @ H.T).toarray().astype(np.float32, copy=False)
        K += G
        print(f"  + iter {it} Gram added (diag mean={np.diag(G).mean():.1f})")
    return K


# ----------------------------------------------------------------------------
# Kernel ridge predict
# ----------------------------------------------------------------------------
def krr_predict(K: np.ndarray, n_tr: int, y_tr: np.ndarray, alpha: float):
    K_tt = K[:n_tr, :n_tr].astype(np.float64, copy=False)
    K_st = K[n_tr:, :n_tr].astype(np.float64, copy=False)
    A = K_tt + alpha * np.eye(n_tr, dtype=np.float64)
    mu_y = float(y_tr.mean())
    y0 = y_tr.astype(np.float64) - mu_y
    c, low = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
    coef = cho_solve((c, low), y0, check_finite=False)
    mean = (K_st @ coef) + mu_y
    return mean.astype(np.float32)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _std_smi(smi: str) -> str | None:
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- WL graph kernel (h={N_ITER}) + kernel ridge (alpha={ALPHA})")
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
    print(f"Train n={n_tr}  Test n={n_te}  Total n={n_tr + n_te}")

    print("Parsing SMILES to molecular graphs...")
    graphs = [smiles_to_graph(s) for s in smiles_tr + smiles_te]
    print(f"  parsed {len(graphs)} graphs  "
          f"(median |V|={int(np.median([len(g[0]) for g in graphs]))})  "
          f"[{time.time()-t0:.1f}s]")

    print(f"WL relabeling x {N_ITER} iterations...")
    hists = wl_label_histograms(graphs)
    print(f"  [{time.time()-t0:.1f}s]")

    print("Assembling WL kernel (sum of per-iter Grams)...")
    K = wl_kernel(hists)
    print(f"  K shape={K.shape}  dtype={K.dtype}  size={K.nbytes/1e6:.1f} MB  "
          f"[{time.time()-t0:.1f}s]")
    # Normalize so kernel-ridge is scale-stable across runs
    d = np.sqrt(np.clip(np.diag(K), 1e-9, None)).astype(np.float32)
    K_norm = (K / d[:, None]) / d[None, :]
    np.fill_diagonal(K_norm, 1.0)
    np.save(ART / f"{TAG}_K_wl.npy", K_norm)

    print(f"Solving kernel ridge (alpha={ALPHA}) via Cholesky...")
    te_pred = krr_predict(K_norm, n_tr, y_tr, alpha=ALPHA)
    print(f"  pred mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

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
    print(f"  pred  mean/std = {te_pred[unb_idx].mean():.3f} / "
          f"{te_pred[unb_idx].std():.3f}")
    print(f"  wall = {time.time()-t0:.1f}s")
    print("=" * 78)

    # ---- Save artifacts ----------------------------------------------------
    np.save(ART / f"{TAG}_te_pred.npy", te_pred)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_pred)
    sub = SUBMISSIONS / f"{TAG}_wl_graph_kernel.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         te_pred,
    }).to_csv(sub, index=False)
    print(f"Wrote {sub}")

    return {
        "success": True,
        "in_rae": rae_u,
        "wall_sec": time.time() - t0,
        "submission": str(sub),
    }


if __name__ == "__main__":
    out = main()
    print("\nRESULT:", out)
