"""nb319 -- Re-blend optimizing on Tanimoto-OOD holdout, not scaffold-CV.

nb318 revealed scaffold-CV under-estimates the true OOD gap by 0.25. The
Tanimoto-OOD holdout (413 most-dissimilar train compounds) better predicts
LB. Refit SLSQP using Tanimoto-OOD RAE as the objective, then check whether
the resulting blend differs from nb302 v4 and what its OOD RAE is.

Goal: identify a submission that minimises Tanimoto-OOD RAE (which should
correlate better with LB than scaffold-CV RAE).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb319: Tanimoto-OOD-optimised blend ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    n_tr = len(y); n_te = 513
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")

    holdout_idx = np.load(DATA_PROCESSED / "tanimoto_holdout_idx.npy")
    inscope_idx = np.load(DATA_PROCESSED / "tanimoto_inscope_idx.npy")
    print(f"Holdout: {len(holdout_idx)}  In-scope: {len(inscope_idx)}")

    # ============================
    # Build full pool of valid (oof, te) pairs
    # ============================
    print("\nEnumerating valid (oof, te) pairs...")
    pairs = []
    for oof_path in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        name = oof_path.stem.replace("oof_", "")
        te_path = DATA_PROCESSED / f"te_{name}.npy"
        if not te_path.exists():
            te_path = DATA_PROCESSED / f"te_oof_{name}.npy"
        if not te_path.exists(): continue
        try:
            o = np.load(oof_path); t = np.load(te_path)
            if o.shape != (n_tr,) or t.shape != (n_te,): continue
            if not (np.isfinite(o).all() and np.isfinite(t).all()): continue
            pairs.append((name, o, t))
        except Exception:
            continue
    print(f"Valid pool: {len(pairs)} models")

    # ============================
    # Filter to OOD-robust models: those whose Tanimoto-OOD RAE is reasonable
    # (i.e. NOT inflated relative to full-OOF RAE; gap < 0.35).
    # ALSO exclude train-only-feature traps (te_std < 0.3 = collapse)
    # ============================
    print("\nFiltering by OOD-robustness + te_std non-collapse...")
    filtered = []
    for nm, o, t in pairs:
        full_rae = rae(y, o)
        ood_rae = rae(y[holdout_idx], o[holdout_idx])
        gap = ood_rae - full_rae
        te_std = t.std()
        if te_std < 0.3:  # collapse trap
            continue
        if gap > 0.35:  # neighborhood-dependent trap (e.g. tiered delta-ML)
            continue
        filtered.append((nm, o, t, full_rae, ood_rae))
    print(f"After filter (te_std>=0.3, gap<=0.35): {len(filtered)} models")
    print(f"\nTop 15 by Tanimoto-OOD RAE (after filter):")
    filtered.sort(key=lambda x: x[4])
    for nm, o, t, fr, hr in filtered[:15]:
        print(f"  {nm[:32]:<33} full_rae={fr:.4f}  ood_rae={hr:.4f}  te_std={t.std():.3f}")

    # ============================
    # SLSQP on Tanimoto-OOD objective using top-25 filtered models
    # ============================
    top = filtered[:25]
    names = [t[0] for t in top]
    oofs = np.column_stack([t[1] for t in top])
    tes  = np.column_stack([t[2] for t in top])

    print(f"\n--- SLSQP on Tanimoto-OOD RAE (top-{len(top)} pool) ---")
    y_ho = y[holdout_idx]
    M_ho = oofs[holdout_idx]
    def loss_ood(w):
        pred = M_ho @ w
        return rae(y_ho, pred)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * len(names)
    best = None
    for seed in range(150):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(len(names)))
        res = minimize(loss_ood, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    pred_ho = M_ho @ best.x
    pred_full = oofs @ best.x
    pred_te = tes @ best.x
    ood_rae = rae(y_ho, pred_ho)
    full_rae = rae(y, pred_full)
    sp_ho, _ = spearmanr(y_ho, pred_ho)
    print(f"\nOOD-blend Tanimoto-OOD RAE={ood_rae:.4f}  full-OOF RAE={full_rae:.4f}  "
          f"OOD Spearman={sp_ho:.4f}  te_std={pred_te.std():.3f}  te_mean={pred_te.mean():.3f}")
    print("Active weights (>=0.01):")
    for nm, w in sorted(zip(names, best.x), key=lambda x: -x[1]):
        if w >= 0.01:
            print(f"  {w:.4f}  {nm}")

    np.save(DATA_PROCESSED / "oof_nb319_ood_blend.npy", pred_full)
    np.save(DATA_PROCESSED / "te_nb319_ood_blend.npy", pred_te)
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': pred_te,
    })
    out = SUBMISSIONS / "nb319_tanimoto_ood_blend.csv"
    sub.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # Multi-metric variant: OOD-RAE - 0.05*OOD-Spearman + collapse penalty
    print(f"\n--- SLSQP on multi-metric Tanimoto-OOD objective ---")
    def loss_mm(w):
        pred_h = M_ho @ w
        r = rae(y_ho, pred_h)
        sp, _ = spearmanr(y_ho, pred_h)
        te_p = tes @ w
        col_pen = max(0, 0.55 - te_p.std()) * 0.3
        return r - 0.05 * sp + col_pen
    best_mm = None
    for seed in range(150):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(len(names)))
        res = minimize(loss_mm, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best_mm is None or res.fun < best_mm.fun: best_mm = res
    pred_full_mm = oofs @ best_mm.x
    pred_te_mm  = tes @ best_mm.x
    ood_rae_mm = rae(y_ho, oofs[holdout_idx] @ best_mm.x)
    sp_mm, _ = spearmanr(y_ho, oofs[holdout_idx] @ best_mm.x)
    print(f"OOD-multimetric Tanimoto-OOD RAE={ood_rae_mm:.4f}  Spearman={sp_mm:.4f}  te_std={pred_te_mm.std():.3f}")
    print("Active weights (>=0.01):")
    for nm, w in sorted(zip(names, best_mm.x), key=lambda x: -x[1]):
        if w >= 0.01:
            print(f"  {w:.4f}  {nm}")

    sub_mm = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': pred_te_mm,
    })
    out_mm = SUBMISSIONS / "nb319_tanimoto_ood_multimetric.csv"
    sub_mm.to_csv(out_mm, index=False)
    print(f"Wrote {out_mm}")

    # ============================
    # STRICT variant: exclude known LB-underperformers (nb118/nb119 leak family,
    # all delta-3tier/4tier variants that overfit neighborhood)
    # ============================
    print(f"\n--- STRICT SLSQP (excludes known leaky candidates) ---")
    LEAKY = {'adaptive_delta_4tier', 'grand_v11', 'full_desc_delta_3tier',
             'allfp_delta_3tier', 'enhanced_delta_3tier', 'delta_similarity_tiers',
             'grand_v10', 'blend_optimizer', 'aux_features', 'grand_v6',
             'grand_v9', 'creative_mega_ensemble', 'nb118_delta_adaptive_k',
             'nb119_grand_v11', 'delta_5tiers',
             # also exclude self-derived nb319 outputs to avoid circularity
             'nb319_ood_blend', 'nb319_multimetric', 'nb319_strict_ood'}
    strict = [t for t in filtered if t[0] not in LEAKY]
    print(f"After LEAKY filter: {len(strict)} candidates")
    strict.sort(key=lambda x: x[4])
    top_s = strict[:20]
    names_s = [t[0] for t in top_s]
    oofs_s = np.column_stack([t[1] for t in top_s])
    tes_s  = np.column_stack([t[2] for t in top_s])
    M_ho_s = oofs_s[holdout_idx]
    def loss_s(w):
        pred = M_ho_s @ w
        r = rae(y_ho, pred)
        sp, _ = spearmanr(y_ho, pred)
        col_pen = max(0, 0.55 - (tes_s @ w).std()) * 0.3
        return r - 0.05 * sp + col_pen
    cons_s = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds_s = [(0, 1.0)] * len(names_s)
    best_s = None
    for seed in range(150):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(len(names_s)))
        res = minimize(loss_s, w0, method='SLSQP', bounds=bounds_s, constraints=cons_s, options={'ftol': 1e-9})
        if best_s is None or res.fun < best_s.fun: best_s = res
    pred_full_s = oofs_s @ best_s.x
    pred_te_s  = tes_s @ best_s.x
    ood_rae_s = rae(y_ho, oofs_s[holdout_idx] @ best_s.x)
    sp_s, _ = spearmanr(y_ho, oofs_s[holdout_idx] @ best_s.x)
    full_rae_s = rae(y, pred_full_s)
    print(f"STRICT OOD RAE={ood_rae_s:.4f}  Spearman={sp_s:.4f}  full-OOF={full_rae_s:.4f}  te_std={pred_te_s.std():.3f}")
    print("Active weights (>=0.01):")
    for nm, w in sorted(zip(names_s, best_s.x), key=lambda x: -x[1]):
        if w >= 0.01:
            print(f"  {w:.4f}  {nm}")
    np.save(DATA_PROCESSED / "oof_nb319_strict_ood.npy", pred_full_s)
    np.save(DATA_PROCESSED / "te_nb319_strict_ood.npy", pred_te_s)
    sub_s = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': pred_te_s,
    })
    out_s = SUBMISSIONS / "nb319_tanimoto_ood_strict.csv"
    sub_s.to_csv(out_s, index=False)
    print(f"Wrote {out_s}")


if __name__ == "__main__":
    main()
