"""phase2_refit.py -- When Phase 2 analog labels drop, score every existing
model on the NEW true labels and refit SLSQP ensemble weights using the
~250 new analog labels as ground-truth held-out validation.

Phase 2 unblind = 2026-05-26: ~250 new analog labels become available. These
were previously in our TEST_BLINDED set, so every model already has predictions
for them (te_*.npy files). We now score those predictions against the new
truth and pick the ensemble that's optimal for OOD analog expansion (which
is what the leaderboard measures).

Usage:
  python scripts/phase2_refit.py [--phase2-csv PATH]

Detects new compounds by:
  - rows in updated TRAIN.csv that weren't in pre-Phase-2 TRAIN.csv, OR
  - explicit --phase2-csv with new labels
"""
import os, sys, warnings, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import spearmanr

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

DATA_RAW = Path("data/raw")
PHASE1_TRAIN_SNAPSHOT = DATA_PROCESSED / "phase1_train_snapshot_names.txt"


def detect_new_compounds(phase2_csv=None):
    """Return df with columns [Molecule Name, SMILES, pec50] of new Phase 2 labels."""
    if phase2_csv and Path(phase2_csv).exists():
        df = pd.read_csv(phase2_csv)
        pec_col = next((c for c in df.columns if 'pEC50' in c and 'std' not in c and 'ci' not in c), None)
        if pec_col is None:
            raise ValueError(f"No pEC50 column in {phase2_csv}")
        return df.rename(columns={pec_col: 'pec50'})[['Molecule Name', 'SMILES', 'pec50']]

    # Auto-detect by comparing current TRAIN to Phase 1 snapshot
    current = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    if not PHASE1_TRAIN_SNAPSHOT.exists():
        print(f"WARN: no Phase 1 snapshot at {PHASE1_TRAIN_SNAPSHOT}")
        print(f"  Saving current TRAIN names as snapshot (assuming pre-Phase-2)")
        current['Molecule Name'].to_csv(PHASE1_TRAIN_SNAPSHOT, index=False, header=False)
        return pd.DataFrame(columns=['Molecule Name', 'SMILES', 'pec50'])

    phase1_names = set(pd.read_csv(PHASE1_TRAIN_SNAPSHOT, header=None)[0].tolist())
    new_rows = current[~current['Molecule Name'].isin(phase1_names)].copy()
    new_rows['pec50'] = new_rows['pEC50']
    return new_rows[['Molecule Name', 'SMILES', 'pec50']]


def load_test_preds():
    """Load all te_*.npy files; return dict name -> array(513,)"""
    preds = {}
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    n_te = len(te_df)
    for f in sorted(DATA_PROCESSED.glob("te_*.npy")):
        try:
            arr = np.load(f)
            if arr.shape == (n_te,):
                preds[f.stem.replace("te_", "")] = arr
        except Exception:
            pass
    return preds, te_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-csv", default=None, help="Explicit Phase 2 labeled CSV")
    ap.add_argument("--top-k", type=int, default=20, help="Top-K candidate models to consider")
    args = ap.parse_args()

    print("=== phase2_refit ===\n")
    new_labels = detect_new_compounds(args.phase2_csv)
    print(f"Detected {len(new_labels)} new Phase 2 labels")
    if len(new_labels) == 0:
        print("Nothing to do. Save Phase 1 snapshot first if needed.")
        return

    preds, te_df = load_test_preds()
    print(f"Loaded {len(preds)} test prediction sets ({len(te_df)} compounds each)\n")

    # Find Phase 2 compounds in test set (they SHOULD be there, since analog set unblind = our test compounds)
    te_name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    new_idx, new_y = [], []
    for _, r in new_labels.iterrows():
        if r['Molecule Name'] in te_name_to_idx:
            new_idx.append(te_name_to_idx[r['Molecule Name']])
            new_y.append(r['pec50'])
    new_idx = np.array(new_idx); new_y = np.array(new_y)
    print(f"Phase 2 compounds matched to test set: {len(new_idx)}/{len(new_labels)}")
    if len(new_idx) < 30:
        print("WARN: few matches — Phase 2 labels may be a different cohort. Aborting refit.")
        return

    # ============================
    # Step 1: per-model leaderboard
    # ============================
    print(f"\n=== Per-model RAE on {len(new_idx)} Phase 2 compounds ===")
    scores = []
    for name, p in preds.items():
        try:
            r = rae(new_y, p[new_idx])
            mae = np.abs(new_y - p[new_idx]).mean()
            sp, _ = spearmanr(new_y, p[new_idx])
            scores.append((name, r, mae, sp))
        except Exception:
            continue
    scores.sort(key=lambda x: x[1])
    print(f"{'rank':<5}{'name':<55}{'RAE':<10}{'MAE':<10}{'Spearman':<10}")
    for i, (n, r, m, sp) in enumerate(scores[:30]):
        print(f"{i+1:<5}{n[:54]:<55}{r:<10.4f}{m:<10.4f}{sp:<10.4f}")

    # Save full leaderboard
    sc_df = pd.DataFrame(scores, columns=['model', 'rae', 'mae', 'spearman'])
    sc_df.to_csv(DATA_PROCESSED / "phase2_per_model_leaderboard.csv", index=False)

    # ============================
    # Step 2: SLSQP refit using top-K models
    # ============================
    top = [s[0] for s in scores[:args.top_k]]
    print(f"\n=== SLSQP refit using top-{args.top_k} models ===")
    M = np.column_stack([preds[n][new_idx] for n in top])

    def loss(w): return rae(new_y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * len(top)
    best = None
    for seed in range(200):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(len(top)))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    print(f"SLSQP best RAE on Phase 2: {best.fun:.4f}")
    print("Active weights (>0.01):")
    for n, w in sorted(zip(top, best.x), key=lambda x: -x[1]):
        if w > 0.01:
            print(f"  {w:.4f}  {n}")

    # ============================
    # Step 3: build final ensemble prediction for ALL 513 test compounds
    # ============================
    M_full = np.column_stack([preds[n] for n in top])
    final = M_full @ best.x
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final
    })
    out_path = SUBMISSIONS / "phase2_slsqp_refit.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"  te_mean={final.mean():.3f}  te_std={final.std():.3f}")

    # Save weights for reproducibility
    weights_path = DATA_PROCESSED / "phase2_slsqp_weights.json"
    with open(weights_path, 'w') as f:
        json.dump({n: float(w) for n, w in zip(top, best.x) if w > 0.001}, f, indent=2)
    print(f"Weights saved to {weights_path}")


if __name__ == "__main__":
    main()
