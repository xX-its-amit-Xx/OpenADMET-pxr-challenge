"""nb361 -- Pairwise Pearson correlation across 12 honest top predictors on unblind set.

Outputs: 12x12 corr matrix, eigenvectors/values, saturated axes, wildcard, effective dim.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pxr.paths import DATA_PROCESSED

def main():
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    print(f"Unblind N = {len(unb_te_idx)}")

    names = [
        'nb320_phase2_top20', 'nb93_chemprop_large_gpu', 'nb130_external_pxr',
        'nb264_chemprop_mt', 'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141',
        'nb305_mope', 'nb306_cepsmim', 'catboost', 'grand_v6b_calib', 'deep_ensemble',
    ]
    preds = []
    used = []
    for n in names:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if not p.exists():
            print(f"MISSING: {n}"); continue
        arr = np.load(p)
        if arr.shape != (513,):
            print(f"BAD SHAPE {n}: {arr.shape}"); continue
        preds.append(arr[unb_te_idx])
        used.append(n)
    M = np.stack(preds)  # (k, N)
    print(f"Stacked shape: {M.shape}, predictors: {len(used)}")

    # Pearson corr matrix
    C = np.corrcoef(M)
    df_C = pd.DataFrame(C, index=used, columns=used)
    print("\n=== PAIRWISE PEARSON CORRELATION ===")
    print(df_C.round(3).to_string())

    # Eigendecomposition
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    total = evals.sum()
    frac = evals / total
    cum = np.cumsum(frac)
    print("\n=== EIGENVALUES (fraction var) ===")
    for i, (e, f, c) in enumerate(zip(evals, frac, cum)):
        print(f"  PC{i+1}: eval={e:.3f}  frac={f:.3%}  cum={c:.3%}")

    # 95% variance K
    K95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"\nEffective dimensionality (95% var): K = {K95}")

    # Top-3 saturated axes (load > 0.30 absolute)
    print("\n=== TOP-3 SATURATED AXES (loading predictors with |loading| > 0.30) ===")
    for k in range(3):
        v = evecs[:, k]
        loads = [(used[i], v[i]) for i in range(len(used))]
        loads.sort(key=lambda x: abs(x[1]), reverse=True)
        top_loaders = [n for n, val in loads if abs(val) > 0.30]
        print(f"PC{k+1} (frac={frac[k]:.3%}):")
        for n, val in loads[:6]:
            print(f"    {n:30s}  loading={val:+.3f}")
        print(f"  --> heavy loaders: {top_loaders}")

    # Least-correlated pair
    Coff = C.copy()
    np.fill_diagonal(Coff, np.nan)
    min_idx = np.nanargmin(Coff)
    i, j = np.unravel_index(min_idx, C.shape)
    print(f"\n=== LEAST-CORRELATED PAIR ===")
    print(f"  {used[i]} <-> {used[j]} : r = {C[i,j]:.3f}")

    # Wildcard: max corr with all others < 0.6
    print(f"\n=== WILDCARD CANDIDATES (max |corr| with others < 0.60) ===")
    max_corr = []
    for i, n in enumerate(used):
        others = [C[i,j] for j in range(len(used)) if j != i]
        m = max(others)
        max_corr.append((n, m))
        print(f"  {n:30s}  max_corr_with_others = {m:.3f}")
    wild = [n for n, m in max_corr if m < 0.60]
    print(f"  --> wildcard(s) (<0.60): {wild}")

    # Also report each predictor's avg correlation
    print("\n=== AVG CORR PER PREDICTOR ===")
    for i, n in enumerate(used):
        avg = (C[i].sum() - 1.0) / (len(used) - 1)
        print(f"  {n:30s}  avg_corr = {avg:.3f}")

if __name__ == "__main__":
    main()
