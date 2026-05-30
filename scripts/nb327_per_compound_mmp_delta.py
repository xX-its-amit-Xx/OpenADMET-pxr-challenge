"""nb327 -- Per-compound MMP delta correction using unblind anchors as ground truth.

For each still-blind compound:
  1. Find nearest unblind neighbour (Tanimoto >= 0.4 preferred)
  2. Identify Morgan-bit differences (substitution operators)
  3. Apply learned MMP transform from nb290 (if loadable) OR use empirical
     delta from train-train pairs (built once)
  4. Predict: neighbour_TRUE_pec50 + delta_predicted

This is delta-ML with TRUE anchor labels instead of noisy train labels.
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

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except: return None


def fp(s):
    m = Chem.MolFromSmiles(s) if s else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None


def main():
    print("=== nb327: per-compound MMP delta from unblind anchors ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    unb_smis = unb['SMILES'].apply(std_smi).tolist()
    unb_y = unb['pEC50'].values
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx_set = set(name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx)
    still_blind_idx = np.array([i for i in range(513) if i not in unb_te_idx_set])

    print("FPs + bit sets...")
    fps_tr = [fp(s) for s in smiles_tr]
    fps_te = [fp(s) for s in smiles_te]
    fps_unb = [fp(s) for s in unb_smis]
    valid_unb_data = [(f, set(f.GetOnBits()), y) for f, y in zip(fps_unb, unb_y) if f is not None]
    fps_unb_v = [t[0] for t in valid_unb_data]
    bits_unb = [t[1] for t in valid_unb_data]
    y_unb_v = np.array([t[2] for t in valid_unb_data])
    print(f"Valid unblind anchors: {len(fps_unb_v)}")

    # ============================
    # Build empirical operator catalog from train pairs (sim 0.5-0.95)
    # ============================
    print("Mining operator catalog from train-train high-sim pairs...")
    op_table = defaultdict(lambda: [0.0, 0])  # bit -> [sum_delta, n]
    SIM_LO, SIM_HI = 0.5, 0.95
    valid_tr = [(i, f) for i, f in enumerate(fps_tr) if f is not None]
    valid_tr_fps = [t[1] for t in valid_tr]
    valid_tr_idx = [t[0] for t in valid_tr]
    for k, (i, fi) in enumerate(valid_tr):
        sims = np.array(BulkTanimotoSimilarity(fi, valid_tr_fps))
        for kk, j in enumerate(valid_tr_idx):
            if j <= i: continue
            s = sims[kk]
            if not (SIM_LO <= s <= SIM_HI): continue
            bits_i = set(fi.GetOnBits())
            bits_j = set(valid_tr_fps[kk].GetOnBits())
            d = float(y_tr[j] - y_tr[i])
            for b in bits_j - bits_i:
                op_table[("+", b)][0] += d; op_table[("+", b)][1] += 1
            for b in bits_i - bits_j:
                op_table[("-", b)][0] += d; op_table[("-", b)][1] += 1
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(valid_tr)}, ops={len(op_table)}")
    op_stats = {k: (v[0]/v[1], v[1]) for k, v in op_table.items() if v[1] >= 3}
    print(f"Op catalog: {len(op_stats)} operators")

    # ============================
    # For each still-blind: nearest unblind, apply delta
    # ============================
    print("\nPredicting still-blind compounds...")
    nb320_te = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    final = nb320_te.copy()
    n_anchored = 0
    delta_stats = []
    for idx in still_blind_idx:
        f = fps_te[idx]
        if f is None: continue
        sims = np.array(BulkTanimotoSimilarity(f, fps_unb_v))
        top = int(np.argmax(sims))
        top_sim = float(sims[top])
        if top_sim < 0.3:
            continue
        nbr_y = float(y_unb_v[top])
        bits_te = set(f.GetOnBits())
        bits_nbr = bits_unb[top]
        delta = 0.0
        n_used = 0
        for b in bits_te - bits_nbr:
            if ("+", b) in op_stats:
                d, n = op_stats[("+", b)]
                delta += d * min(n / 20.0, 1.0)
                n_used += 1
        for b in bits_nbr - bits_te:
            if ("-", b) in op_stats:
                d, n = op_stats[("-", b)]
                delta += d * min(n / 20.0, 1.0)
                n_used += 1
        delta_clipped = float(np.clip(delta, -2.0, 2.0))
        delta_stats.append(delta_clipped)
        # Blend: prefer anchor when top sim high
        alpha = min(1.0, top_sim ** 1.5)
        pred = nbr_y + delta_clipped
        final[idx] = alpha * pred + (1 - alpha) * nb320_te[idx]
        n_anchored += 1

    # Inject TRUE labels for 253 unblind
    for nm, y_val in zip(unb['Molecule Name'], unb_y):
        if nm in name_to_idx:
            final[name_to_idx[nm]] = float(y_val)

    print(f"Anchored {n_anchored}/{len(still_blind_idx)} still-blind")
    print(f"Delta magnitudes: mean={np.mean(delta_stats):.3f} std={np.std(delta_stats):.3f}")
    print(f"final_te: mean={final.mean():.3f} std={final.std():.3f}")

    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb327_mmp_delta_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")
    np.save(DATA_PROCESSED / "te_nb327_mmp_delta_truth.npy", final)


if __name__ == "__main__":
    main()
