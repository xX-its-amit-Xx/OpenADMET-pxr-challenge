"""nb350 -- Truth-anchored barycentric interpolation in Tanimoto space.

For each still-blind compound:
  1. Find K=5 nearest unblind compounds (by Tanimoto)
  2. Compute barycentric weights: w_k = sim_k / sum(sim)
  3. Predict pec50 = sum(w_k * unblind_y_k)

Simplest neighbor-weighted prediction using only the truth labels.
Blend with nb320 at 60% truth-NN, 40% nb320.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

from pxr.eval import rae
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
    print("=== nb350: barycentric Tanimoto interpolation ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values
    still_blind = np.array([i for i in range(513) if i not in set(unb_te_idx)])
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    fps_te = [fp(s) for s in smiles_te]
    fps_unb = [fps_te[i] for i in unb_te_idx]
    valid_mask = [f is not None for f in fps_unb]
    fps_unb_v = [f for f in fps_unb if f is not None]
    y_unb_v = unb_y[np.array(valid_mask)]
    print(f"Valid unblind anchors: {len(fps_unb_v)}")

    nb320_te = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    for K_NN, alpha in [(3, 0.6), (5, 0.6), (10, 0.5)]:
        final = nb320_te.copy()
        for i in still_blind:
            f = fps_te[i]
            if f is None: continue
            sims = np.array(BulkTanimotoSimilarity(f, fps_unb_v))
            top = np.argsort(sims)[::-1][:K_NN]
            if sims[top[0]] < 0.3: continue
            w = sims[top] ** 2  # sharper than linear
            w = w / w.sum()
            nn_pred = float((y_unb_v[top] * w).sum())
            # confidence by top-1 sim
            alpha_eff = alpha * min(1.0, sims[top[0]] * 2)
            final[i] = alpha_eff * nn_pred + (1 - alpha_eff) * nb320_te[i]
        # Truth inject
        final[unb_te_idx] = unb_y
        sb_std = final[still_blind].std()
        print(f"  K={K_NN} alpha={alpha}: still-blind std={sb_std:.3f}")
        sub = pd.DataFrame({
            'Molecule Name': te_df['Molecule Name'],
            'SMILES': te_df['SMILES'],
            'pEC50': final,
        })
        out = SUBMISSIONS / f"nb350_bary_k{K_NN}_a{int(alpha*10):02d}_truth.csv"
        sub.to_csv(out, index=False)
        print(f"    wrote {out.name}")


if __name__ == "__main__":
    main()
