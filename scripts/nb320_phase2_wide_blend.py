"""nb320 -- Phase 2 wide-pool SLSQP using TRUE Phase 1 unblind labels.

Now that 253 test compounds have true labels, we can fit SLSQP weights
DIRECTLY on actual OOD performance, not scaffold-CV proxy. Try top-K = 20,
50, 100 and report. Output multiple submission candidates spanning a
breadth-of-pool sweep.
"""
import os, sys, warnings, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize
from sklearn.metrics import r2_score

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb320: Phase 2 wide-pool SLSQP (true OOD labels) ===\n")
    te_df_blind = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    te_names = te_df_blind['Molecule Name'].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    print(f"Unblind: {len(unb)} compounds")
    # Build aligned arrays
    unb_idx = []
    unb_y = []
    for _, r in unb.iterrows():
        nm = r['Molecule Name']
        if nm in name_to_idx:
            unb_idx.append(name_to_idx[nm])
            unb_y.append(r['pEC50'])
    unb_idx = np.array(unb_idx); unb_y = np.array(unb_y)
    print(f"Matched {len(unb_idx)}/{len(unb)} to test set")

    # Load all te files
    preds = {}
    for f in sorted(DATA_PROCESSED.glob("te_*.npy")):
        try:
            arr = np.load(f)
            if arr.shape == (513,) and np.isfinite(arr).all():
                preds[f.stem.replace("te_", "")] = arr
        except Exception:
            pass
    print(f"Loaded {len(preds)} test prediction sets\n")

    # Score
    rows = []
    for n, p in preds.items():
        try:
            r = rae(unb_y, p[unb_idx])
            sp, _ = spearmanr(unb_y, p[unb_idx])
            kd, _ = kendalltau(unb_y, p[unb_idx])
            r2 = r2_score(unb_y, p[unb_idx])
            mae = np.abs(unb_y - p[unb_idx]).mean()
            rows.append(dict(model=n, rae=r, spearman=sp, kendall=kd, r2=r2, mae=mae,
                             te_std=float(p.std())))
        except Exception:
            continue
    sc = pd.DataFrame(rows).sort_values('rae').reset_index(drop=True)
    print(f"Top 30 by actual Phase 2 RAE:")
    print(sc.head(30).to_string(index=False))
    sc.to_csv(DATA_PROCESSED / "nb320_phase2_all_models_ranked.csv", index=False)

    # ============================
    # SLSQP over various top-K sizes
    # ============================
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    for K in [10, 20, 30, 50, 100]:
        top = sc.head(K)
        names = top['model'].tolist()
        M = np.column_stack([preds[n][unb_idx] for n in names])
        M_full = np.column_stack([preds[n] for n in names])
        bounds = [(0, 1.0)] * K
        def loss(w): return rae(unb_y, M @ w)
        best = None
        for seed in range(150):
            rng = np.random.default_rng(seed)
            w0 = rng.dirichlet(np.ones(K))
            res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
            if best is None or res.fun < best.fun: best = res
        blend = M_full @ best.x
        active = [(n, w) for n, w in zip(names, best.x) if w >= 0.005]
        active.sort(key=lambda x: -x[1])
        print(f"\n--- top-{K} blend: RAE={best.fun:.4f}  te_std={blend.std():.3f}  active={len(active)} ---")
        for n, w in active[:8]:
            print(f"  {w:.4f}  {n}")

        sub = pd.DataFrame({
            'Molecule Name': te_names,
            'SMILES': te_df_blind['SMILES'],
            'pEC50': blend,
        })
        out = SUBMISSIONS / f"nb320_phase2_top{K}_slsqp.csv"
        sub.to_csv(out, index=False)
        print(f"  Wrote {out.name}")

        if K == 20:
            np.save(DATA_PROCESSED / "te_nb320_phase2_top20.npy", blend)

    # ============================
    # Multi-metric with Spearman bonus + collapse penalty
    # ============================
    print("\n=== Multi-metric SLSQP (RAE - 0.1*Spearman + collapse_pen) — top-30 ===")
    K = 30
    top = sc.head(K)
    names = top['model'].tolist()
    M = np.column_stack([preds[n][unb_idx] for n in names])
    M_full = np.column_stack([preds[n] for n in names])
    bounds = [(0, 1.0)] * K
    def loss_mm(w):
        pred = M @ w
        r = rae(unb_y, pred)
        sp, _ = spearmanr(unb_y, pred)
        full_p = M_full @ w
        col_pen = max(0, 0.55 - full_p.std()) * 0.3
        return r - 0.1 * sp + col_pen
    best = None
    for seed in range(200):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(K))
        res = minimize(loss_mm, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    blend = M_full @ best.x
    blend_unb = M @ best.x
    r = rae(unb_y, blend_unb); sp, _ = spearmanr(unb_y, blend_unb)
    print(f"Multi-metric blend: RAE={r:.4f}  Spearman={sp:.4f}  te_std={blend.std():.3f}")
    print("Active (>=0.005):")
    for n, w in sorted(zip(names, best.x), key=lambda x: -x[1]):
        if w >= 0.005:
            print(f"  {w:.4f}  {n}")
    sub = pd.DataFrame({
        'Molecule Name': te_names,
        'SMILES': te_df_blind['SMILES'],
        'pEC50': blend,
    })
    out = SUBMISSIONS / "nb320_phase2_multimetric.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
