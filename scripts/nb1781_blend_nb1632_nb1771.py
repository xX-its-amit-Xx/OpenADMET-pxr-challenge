"""nb1781 — Blend nb1632 BoB + nb1771 (cross-fit weight).

PROTOCOL:
1. Load nb1632_bob_mean_oof.npy (0.5107) and nb1771_mean_bag_oof.npy (0.5100).
2. Pearson correlation.
3. Grid w in {0.0..1.0 step 0.05}.
4. SLSQP cross-fit (5-fold scaffold).
5. Verdict at 0.003 margin vs nb1632 (0.5107) and nb1771 (0.5100).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import standardize

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"


def murcko_scaffold(smi: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s) if s is not None else ""
    except Exception:
        return ""


def main():
    print("=" * 70)
    print("nb1781 — Blend nb1632 BoB + nb1771")
    print("=" * 70)

    p_a = PROC / "nb1632_bob_mean_oof.npy"
    p_b = PROC / "nb1771_mean_bag_oof.npy"
    a = np.load(p_a)
    b = np.load(p_b)
    print(f"nb1632 BoB OOF: shape={a.shape}, mean={a.mean():.4f}, std={a.std():.4f}")
    print(f"nb1771 bag OOF: shape={b.shape}, mean={b.mean():.4f}, std={b.std():.4f}")

    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    # Load truth
    n = a.shape[0]
    if n == 253:
        y = np.load(PROC / "_audit_unblind_y.npy")
        unb_idx = np.load(PROC / "_audit_unblind_idx.npy")
        print(f"OOF over unblind 253 n={n}, y mean={y.mean():.4f} std={y.std():.4f}")
    else:
        train = load_train()
        if n == len(train):
            y = train["pec50"].values.astype(float)
            print(f"OOF over full train n={n}")
        else:
            raise RuntimeError(f"Cannot find truth for OOF length {n}")

    # Pearson
    r, p_val = pearsonr(a, b)
    print(f"\nPearson(nb1632, nb1771) = {r:.4f} (p={p_val:.2e})")

    rae_a = rae(y, a)
    rae_b = rae(y, b)
    print(f"\nStandalone RAE: nb1632={rae_a:.4f}, nb1771={rae_b:.4f}")

    # Grid search w*a + (1-w)*b
    print("\nGrid search w (nb1632) in {0.0..1.0 step 0.05}:")
    grid_rows = []
    for w in np.arange(0.0, 1.0001, 0.05):
        blend = w * a + (1 - w) * b
        s = rae(y, blend)
        grid_rows.append({"w_nb1632": float(round(w, 2)), "w_nb1771": float(round(1 - w, 2)), "rae": float(s)})
    grid_df = pd.DataFrame(grid_rows)
    print(grid_df.to_string(index=False))

    best_row = grid_df.iloc[grid_df["rae"].idxmin()]
    best_w = float(best_row["w_nb1632"])
    best_grid_rae = float(best_row["rae"])
    print(f"\nBest grid: w_nb1632={best_w}, w_nb1771={1 - best_w:.2f}, RAE={best_grid_rae:.4f}")

    # SLSQP cross-fit (5-fold scaffold)
    if n == 253:
        # Build scaffolds for the 253 unblind compounds (need their SMILES)
        # unb_idx points into the 513 test BLINDED rows; load that and pull SMILES
        print("\nSLSQP cross-fit 5-fold scaffold (n=253 unblind)...")
        try:
            test_blind = pd.read_csv(ROOT / "data" / "raw" / "pxr-challenge_TEST_BLINDED.csv")
            smi_col = "SMILES" if "SMILES" in test_blind.columns else test_blind.columns[1]
            smis_all = test_blind[smi_col].tolist()
            smis = [smis_all[i] for i in unb_idx]
        except Exception as e:
            print(f"  Could not load test SMILES ({e}); using KFold fallback")
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kf.split(np.arange(n)))
        else:
            smis = [standardize(s) for s in smis]
            scaffolds = [murcko_scaffold(s) for s in smis]
            scaffolds = [s if s else f"__nul_{i}" for i, s in enumerate(scaffolds)]
            folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)
    else:
        train = load_train()
        print("\nSLSQP cross-fit 5-fold scaffold...")
        smis = train["smiles"].apply(standardize).tolist()
        scaffolds = [murcko_scaffold(s) for s in smis]
        scaffolds = [s if s else f"__nul_{i}" for i, s in enumerate(scaffolds)]
        folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)

    cross_pred = np.zeros(n, dtype=float)
    fold_weights = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        def obj(w):
            w0 = w[0]
            blend = w0 * a[tr_idx] + (1 - w0) * b[tr_idx]
            return rae(y[tr_idx], blend)
        res = minimize(obj, x0=[0.5], bounds=[(0.0, 1.0)], method="SLSQP")
        w_fit = float(res.x[0])
        fold_weights.append(w_fit)
        cross_pred[va_idx] = w_fit * a[va_idx] + (1 - w_fit) * b[va_idx]
        print(f"  fold {k}: w_nb1632={w_fit:.4f} (tr_RAE={res.fun:.4f})")

    slsqp_rae = float(rae(y, cross_pred))
    mean_w = float(np.mean(fold_weights))
    print(f"\nSLSQP cross-fit RAE = {slsqp_rae:.4f}  (mean w_nb1632 = {mean_w:.4f})")

    # Also full-data SLSQP (in-sample, for reporting)
    def obj_full(w):
        return rae(y, w[0] * a + (1 - w[0]) * b)
    res_full = minimize(obj_full, x0=[0.5], bounds=[(0.0, 1.0)], method="SLSQP")
    full_w = float(res_full.x[0])
    full_rae = float(res_full.fun)
    print(f"In-sample SLSQP: w_nb1632={full_w:.4f}, RAE={full_rae:.4f}")

    # Verdict
    margin = 0.003
    base_best = min(rae_a, rae_b)
    best_eval = min(best_grid_rae, slsqp_rae)
    delta = best_eval - base_best
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"nb1632 standalone:        {rae_a:.4f}")
    print(f"nb1771 standalone:        {rae_b:.4f}")
    print(f"Best component:           {base_best:.4f}")
    print(f"Best blend (grid):        {best_grid_rae:.4f}  (w_nb1632={best_w})")
    print(f"SLSQP cross-fit blend:    {slsqp_rae:.4f}  (mean w_nb1632={mean_w:.4f})")
    print(f"Margin threshold:         {margin:.4f}")
    print(f"Delta (blend - best):     {delta:+.4f}")
    if delta <= -margin:
        verdict = "PROMOTE"
        print(f"VERDICT: {verdict} — blend beats best standalone by >= {margin}")
    elif delta <= 0:
        verdict = "MARGINAL"
        print(f"VERDICT: {verdict} — small improvement, below {margin} margin")
    else:
        verdict = "REJECT"
        print(f"VERDICT: {verdict} — blend does not beat best standalone")

    # Save summary
    summary = {
        "method": "nb1781_blend_nb1632_nb1771",
        "n": int(n),
        "pearson_r": float(r),
        "pearson_p": float(p_val),
        "rae_nb1632_standalone": float(rae_a),
        "rae_nb1771_standalone": float(rae_b),
        "grid": grid_rows,
        "best_grid_w_nb1632": best_w,
        "best_grid_rae": best_grid_rae,
        "slsqp_crossfit_rae": slsqp_rae,
        "slsqp_crossfit_mean_w_nb1632": mean_w,
        "slsqp_fold_weights": fold_weights,
        "slsqp_full_w_nb1632": full_w,
        "slsqp_full_rae": full_rae,
        "margin": margin,
        "delta_vs_best": float(delta),
        "verdict": verdict,
    }
    out = PROC / "nb1781_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
