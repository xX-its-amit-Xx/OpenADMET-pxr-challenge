"""nb307 -- Substitution-Operator Counterfactual Inference (SOCI).

Idea E: medicinal-chemist-style mechanism prediction.
For each test compound:
  1. Find nearest train neighbour by scaffold + Tanimoto.
  2. Identify Morgan-bit DIFFERENCES (added/removed substructures).
  3. For each "operator" (bit pattern that differs), look up its average
     ΔpEC50 across the train MMP corpus (computed once, cached).
  4. Apply operators sequentially:
       test_pec50 = neighbour_pec50 + Σ Δ_op_i * confidence_i

This is the medicinal chemist mentally walking from a known analog to the
new compound: "ortho-Cl gives +0.3, N-methyl gives -0.1, replace phenyl
with pyridyl gives -0.4 ..."
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED


def std_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except: return None


def fp_bits(smi, r=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n_bits)


def fp_to_set(fp):
    return set(fp.GetOnBits())


def main():
    print("=== nb307: Substitution-Operator Counterfactual Inference ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    scaffolds = tr['scaffold'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()

    print("Computing Morgan FPs (r=2, n_bits=2048)...")
    fps_tr = [fp_bits(s) for s in smiles_tr]
    fps_te = [fp_bits(s) for s in smiles_te]
    bits_tr = [fp_to_set(f) if f else set() for f in fps_tr]
    bits_te = [fp_to_set(f) if f else set() for f in fps_te]

    # =====================================================================
    # Build the operator catalog: for each Morgan-bit that's ADDED in some
    # MMP pair, record the empirical mean delta_y. We mine pairs from ALL
    # train compound pairs with high Tanimoto similarity (analog-like).
    # =====================================================================
    print("\nMining operator catalog from train-train high-similarity pairs...")
    op_stats = defaultdict(lambda: [0.0, 0])  # bit_idx -> [sum_delta, n]
    SIM_LO = 0.5; SIM_HI = 0.95  # avoid near-identical and very-distant pairs
    for i in range(len(fps_tr)):
        if fps_tr[i] is None: continue
        sims = np.array(BulkTanimotoSimilarity(fps_tr[i], [f for f in fps_tr if f is not None]))
        valid_idx = [j for j, f in enumerate(fps_tr) if f is not None]
        for k, j in enumerate(valid_idx):
            if j <= i: continue
            s = sims[k]
            if not (SIM_LO <= s <= SIM_HI): continue
            added = bits_tr[j] - bits_tr[i]   # bits present in j but not i
            removed = bits_tr[i] - bits_tr[j]  # bits present in i but not j
            d = y_tr[j] - y_tr[i]
            # symmetric: each added bit in j contributes +d, each removed bit contributes -d
            for b in added:
                s_ = op_stats[("+", b)]
                s_[0] += d; s_[1] += 1
            for b in removed:
                s_ = op_stats[("-", b)]
                s_[0] += d; s_[1] += 1
        if (i + 1) % 500 == 0:
            print(f"  scanned {i+1}/{len(fps_tr)}; catalog size={len(op_stats)}")

    op_table = {k: (s / n, n) for k, (s, n) in op_stats.items() if n >= 3}
    print(f"\nOperator catalog: {len(op_table)} operators with >=3 supporting pairs")

    # =====================================================================
    # For each TEST compound, find nearest TRAIN neighbour, then apply
    # operators (with confidence weighting + clipping for stability).
    # =====================================================================
    def predict_one(target_bits, fps_pool, y_pool, exclude_idx=None):
        """Predict target's pec50 by neighbour + Σ Δ_operator."""
        sims = np.array(BulkTanimotoSimilarity(target_bits, fps_pool))
        if exclude_idx is not None: sims[exclude_idx] = -1
        nbr = int(np.argmax(sims))
        if sims[nbr] < 0.3:
            return float(np.mean(y_pool)), 0.0  # no good neighbour
        nbr_bits = fp_to_set(fps_pool[nbr])
        tg_bits = fp_to_set(target_bits)
        added = tg_bits - nbr_bits
        removed = nbr_bits - tg_bits
        delta = 0.0
        n_ops_used = 0
        for b in added:
            if ("+", b) in op_table:
                d, n = op_table[("+", b)]
                delta += d * min(n / 20.0, 1.0)  # confidence factor caps at n=20
                n_ops_used += 1
        for b in removed:
            if ("-", b) in op_table:
                d, n = op_table[("-", b)]
                delta += d * min(n / 20.0, 1.0)
                n_ops_used += 1
        # Clip delta — operators compound chaotically when n_ops is huge
        delta_clipped = np.clip(delta, -2.5, 2.5)
        return float(y_pool[nbr] + delta_clipped), float(sims[nbr])

    print("\nPredicting test compounds...")
    valid_tr_fps = [f for f in fps_tr if f is not None]
    valid_tr_y = np.array([y_tr[i] for i, f in enumerate(fps_tr) if f is not None])
    te_pred = np.zeros(len(smiles_te))
    for i, fp in enumerate(fps_te):
        if fp is None:
            te_pred[i] = y_tr.mean()
            continue
        p, _ = predict_one(fp, valid_tr_fps, valid_tr_y)
        te_pred[i] = p
        if (i + 1) % 100 == 0: print(f"  test {i+1}/{len(smiles_te)}")

    # OOF via LOO from train pool itself
    print("\nPredicting OOF (LOO)...")
    oof = np.zeros(len(y_tr))
    valid_idx_map = [i for i, f in enumerate(fps_tr) if f is not None]
    pos_in_pool = {orig: pos for pos, orig in enumerate(valid_idx_map)}
    for i, fp in enumerate(fps_tr):
        if fp is None:
            oof[i] = y_tr.mean()
            continue
        excl = pos_in_pool.get(i, None)
        p, _ = predict_one(fp, valid_tr_fps, valid_tr_y, exclude_idx=excl)
        oof[i] = p
        if (i + 1) % 500 == 0: print(f"  oof {i+1}/{len(fps_tr)}")

    r = rae(y_tr, oof)
    sp, _ = spearmanr(y_tr, oof)
    print(f"\nSOCI OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb307_soci.npy", oof)
    np.save(DATA_PROCESSED / "te_nb307_soci.npy", te_pred)

    # SLSQP blend
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    def loss_fn(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(80):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss_fn, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP: {best.fun:.4f}, weight(nb307)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
