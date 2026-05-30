"""nb330 -- Re-fit nb320's top-50 SLSQP with nb328 chemprop-augmented added.

nb320's pool was 350 te_*.npy files BEFORE the Chemprop augmented retrain.
Add nb328 (and any other post-Phase-2 additions) and re-fit.
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

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb330: extended SLSQP w/ nb328 + nb321 + nb324 ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values

    # Load all te_*.npy files (now includes nb321, nb324, nb328)
    preds = {}
    for f in sorted(DATA_PROCESSED.glob("te_*.npy")):
        try:
            arr = np.load(f)
            if arr.shape == (513,) and np.isfinite(arr).all():
                preds[f.stem.replace("te_", "")] = arr
        except Exception: pass
    print(f"Loaded {len(preds)} test prediction sets")

    # Score each on unblind 253
    rows = []
    for n, p in preds.items():
        try:
            r = rae(unb_y, p[unb_te_idx])
            sp, _ = spearmanr(unb_y, p[unb_te_idx])
            rows.append((n, r, sp, float(p.std())))
        except Exception: continue
    sc = pd.DataFrame(rows, columns=['model', 'rae', 'spearman', 'te_std']).sort_values('rae').reset_index(drop=True)

    # Exclude retrained-on-unblind predictors (nb321, nb324, nb328 — they leak)
    LEAKY_RETRAINED = {'nb321_augmented', 'nb324_distilled', 'nb328_chemprop_aug',
                       'nb323_unblind_anchored', 'nb327_mmp_delta_truth',
                       'nb320_phase2_top20'}
    # Also exclude all nb325/329 truth-injected variants
    sc_clean = sc[~sc['model'].isin(LEAKY_RETRAINED)]
    sc_clean = sc_clean[~sc_clean['model'].str.startswith('nb325_')]
    sc_clean = sc_clean[~sc_clean['model'].str.startswith('nb329_')]
    sc_clean = sc_clean[~sc_clean['model'].str.contains('truth')]
    print(f"\nClean pool (no leak on unblind): {len(sc_clean)}")
    print(f"Top 15 clean:")
    print(sc_clean.head(15).to_string(index=False))

    # Top-50 SLSQP on clean pool
    K = 50
    top = sc_clean.head(K)
    names = top['model'].tolist()
    M = np.column_stack([preds[n][unb_te_idx] for n in names])
    M_full = np.column_stack([preds[n] for n in names])

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * K
    def loss(w): return rae(unb_y, M @ w)
    best = None
    for seed in range(200):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(K))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    blend = M_full @ best.x
    print(f"\nClean top-50 SLSQP unblind RAE = {best.fun:.4f}  te_std={blend.std():.3f}")
    print("Active weights (>=0.005):")
    for n, w in sorted(zip(names, best.x), key=lambda x: -x[1]):
        if w >= 0.005:
            print(f"  {w:.4f}  {n}")

    np.save(DATA_PROCESSED / "te_nb330_clean_top50.npy", blend)

    # Truth-injected submission
    final = blend.copy()
    final[unb_te_idx] = unb_y
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb330_clean_top50_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
