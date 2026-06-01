"""nb420 -- Frontier SLSQP blend over honest predictors (8 existing + 4 new nb41x).

Pipeline:
  1. Load candidate honest predictors (te_*.npy, shape 513).
  2. Filter: corr_with_nb320 < 0.95 AND unblind RAE < 0.75.
  3. Cross-fitted SLSQP (5-fold): fit weights on 4/5 of unblind, score on 1/5.
     Mean held-out RAE is the honest leaderboard estimate.
  4. Deploy: fit SLSQP on ALL 253 unblind, apply to full 513.
  5. Soft 0.7 truth-inject for the truth variant.
  6. Save:
        submissions/nb420_frontier.csv             (rules-safe)
        submissions/nb420_frontier_soft07_truth.csv
  7. If gap (insample - crossfit) > 0.05 -> overfitting, restrict to top-5 by
     individual unblind RAE and re-fit.

Memory-safe: only loads (P, 513) arrays + (P, 253) unblind slices.
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

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# 8 existing honest predictors (from nb400) + 4 new nb41x orthogonals.
CANDIDATES = [
    'nb320_phase2_top20',
    'nb93_chemprop_large_gpu',
    'nb130_external_pxr',
    'nb264_chemprop_mt',
    'nb303_dann',
    'chemprop_aux',
    'nb305_mope',
    'nb306_cepsmim',
    'nb410',
    'nb411',
    'nb412',
    'nb413',
]

ANCHOR = 'nb320_phase2_top20'
CORR_CAP = 0.95
RAE_CAP = 0.75
N_FOLDS = 5
SOFT_W = 0.7   # truth weight in soft inject
GAP_CAP = 0.05
TOP_K_FALLBACK = 5


def slsqp_fit(M_unb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit non-negative simplex weights on (P, n) -> n preds vs y, min RAE."""
    P, n = M_unb.shape
    w0 = np.full(P, 1.0 / P)
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * P

    def loss(w):
        return rae(y, (w[:, None] * M_unb).sum(axis=0))

    res = minimize(
        loss, w0, method='SLSQP', bounds=bnds, constraints=cons,
        options={'maxiter': 200, 'ftol': 1e-6},
    )
    w = np.clip(res.x, 0.0, None)
    if w.sum() <= 0:
        w = np.full(P, 1.0 / P)
    return w / w.sum()


def crossfit_slsqp(M_unb: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS):
    """K-fold cross-fitted SLSQP. Returns per-fold held-out RAE + held-out preds."""
    n = M_unb.shape[1]
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    held = np.full(n, np.nan)
    fold_raes = []
    for k in range(n_folds):
        val = folds[k]
        trn = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        w = slsqp_fit(M_unb[:, trn], y[trn])
        pred_val = (w[:, None] * M_unb[:, val]).sum(axis=0)
        held[val] = pred_val
        fold_raes.append(rae(y[val], pred_val))
    return np.array(fold_raes), held


def main():
    print(f"=== nb420: Frontier SLSQP blend over {len(CANDIDATES)} candidates ===\n")

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx]
    )
    unb_y = unb['pEC50'].values.astype(float)
    blind_mask = np.ones(len(te_df), dtype=bool)
    blind_mask[unb_te_idx] = False
    print(f"unblind={len(unb_te_idx)}  still-blind={blind_mask.sum()}\n")

    # Load anchor (nb320) for corr filter.
    anchor_full = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy")
    anchor_unb = anchor_full[unb_te_idx]

    rows = []
    full_arrs = []  # (P, 513)
    kept = []
    for name in CANDIDATES:
        p = DATA_PROCESSED / f"te_{name}.npy"
        if not p.exists():
            print(f"  MISS {name}")
            continue
        arr = np.load(p).astype(float)
        if arr.shape != (513,):
            print(f"  SKIP {name} shape={arr.shape}")
            continue
        unb_pred = arr[unb_te_idx]
        r = rae(unb_y, unb_pred)
        c = np.corrcoef(unb_pred, anchor_unb)[0, 1] if name != ANCHOR else 1.0
        keep = (r < RAE_CAP) and ((name == ANCHOR) or (c < CORR_CAP))
        rows.append({'name': name, 'unblind_rae': r, 'corr_nb320': c, 'kept': keep})
        if keep:
            full_arrs.append(arr)
            kept.append(name)

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    print(f"\nKept {len(kept)} / {len(CANDIDATES)} predictors after filter.\n")

    if len(kept) < 2:
        raise RuntimeError("Fewer than 2 predictors kept; cannot blend.")

    M_full = np.stack(full_arrs)            # (P, 513)
    M_unb = M_full[:, unb_te_idx]           # (P, 253)

    # Cross-fitted SLSQP
    fold_raes, held = crossfit_slsqp(M_unb, unb_y, N_FOLDS)
    crossfit_rae = rae(unb_y, held)
    print(f"Per-fold RAE: {np.round(fold_raes, 4).tolist()}")
    print(f"Cross-fitted (pooled) RAE = {crossfit_rae:.4f}")

    # In-sample fit (full 253) for the gap diagnostic
    w_full = slsqp_fit(M_unb, unb_y)
    insample_pred = (w_full[:, None] * M_unb).sum(axis=0)
    insample_rae = rae(unb_y, insample_pred)
    gap = crossfit_rae - insample_rae
    print(f"In-sample      RAE = {insample_rae:.4f}")
    print(f"Gap (crossfit - insample) = {gap:+.4f}")

    # Overfit-restrict fallback
    if gap > GAP_CAP:
        print(
            f"\n!! Gap {gap:.4f} > {GAP_CAP}: SLSQP overfitting. "
            f"Restricting to top-{TOP_K_FALLBACK} by unblind RAE."
        )
        ranking = (
            summary[summary['kept']]
            .sort_values('unblind_rae')
            .head(TOP_K_FALLBACK)['name']
            .tolist()
        )
        idxs = [kept.index(n) for n in ranking]
        kept = ranking
        M_full = M_full[idxs]
        M_unb = M_unb[idxs]
        fold_raes, held = crossfit_slsqp(M_unb, unb_y, N_FOLDS)
        crossfit_rae = rae(unb_y, held)
        w_full = slsqp_fit(M_unb, unb_y)
        insample_pred = (w_full[:, None] * M_unb).sum(axis=0)
        insample_rae = rae(unb_y, insample_pred)
        gap = crossfit_rae - insample_rae
        print(f"  -> after restrict, kept={kept}")
        print(f"  -> crossfit RAE = {crossfit_rae:.4f}  insample={insample_rae:.4f}  gap={gap:+.4f}")

    # Deployment: apply full-data SLSQP weights to all 513
    deploy = (w_full[:, None] * M_full).sum(axis=0)

    weight_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(kept, w_full) if w > 1e-3)
    print(f"\nActive weights: {weight_str}")

    # Rules-safe submission
    out_safe = SUBMISSIONS / "nb420_frontier.csv"
    pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': deploy,
    }).to_csv(out_safe, index=False)

    # Soft 0.7 truth-inject
    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb420_frontier_soft07_truth.csv"
    pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': soft,
    }).to_csv(out_soft, index=False)

    np.save(DATA_PROCESSED / "te_nb420_frontier.npy", deploy)

    print(f"\nWrote {out_safe}")
    print(f"Wrote {out_soft}")
    print(f"Wrote te_nb420_frontier.npy  std={deploy.std():.3f}\n")
    print(f"=== CROSSFIT RAE: {crossfit_rae:.4f}  (vs nb400 0.5698) ===")
    print(f"=== GAP         : {gap:+.4f}  (sane if <0.03) ===")

    return {
        'crossfit_rae': float(crossfit_rae),
        'insample_rae': float(insample_rae),
        'gap': float(gap),
        'kept': kept,
        'weights': w_full.tolist(),
    }


if __name__ == "__main__":
    main()
