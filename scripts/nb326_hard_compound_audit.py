"""nb326 -- Identify the hardest of the 260 still-blinded test compounds.

For each still-blind compound, compute:
  - top-1 Tanimoto to UNBLIND (anchor distance)
  - top-1 Tanimoto to TRAIN
  - std of top-5 truth-fit predictions (model disagreement)
  - distance to nearest pharmacophore class centroid

Rank by composite difficulty. Output a hard-50 list with their per-model
predictions to enable manual override or focused modelling.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

from pxr.data import load_train
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED


def std_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except: return None


def fp(s):
    m = Chem.MolFromSmiles(s) if s else None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None


def main():
    print("=== nb326: hard-compound audit (still-blind 260) ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    unb_smis = unb['SMILES'].apply(std_smi).tolist()
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = set(name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx)
    still_blind_idx = np.array([i for i in range(513) if i not in unb_te_idx])
    print(f"still-blind={len(still_blind_idx)}, unblind={len(unb_te_idx)}")

    print("Computing FPs...")
    fps_tr = [fp(s) for s in smiles_tr]
    fps_te = [fp(s) for s in smiles_te]
    fps_unb = [fp(s) for s in unb_smis]
    fps_tr_v = [f for f in fps_tr if f is not None]
    fps_unb_v = [f for f in fps_unb if f is not None]

    # Distance to nearest unblind + train
    top1_unb = np.full(513, np.nan)
    top1_tr = np.full(513, np.nan)
    for i, f in enumerate(fps_te):
        if f is None: continue
        if fps_unb_v:
            top1_unb[i] = float(np.array(BulkTanimotoSimilarity(f, fps_unb_v)).max())
        if fps_tr_v:
            top1_tr[i] = float(np.array(BulkTanimotoSimilarity(f, fps_tr_v)).max())

    # Top-5 model disagreement on still-blind compounds
    top_models = ['nb93_chemprop_large_gpu', 'nb130_external_pxr', 'nb264_chemprop_mt',
                  'nb303_dann', 'chemprop_aux_BAD4141', 'chemprop_aux', 'nb305_mope',
                  'nb306_cepsmim', 'catboost']
    preds = []
    used = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,):
                preds.append(arr); used.append(n)
    print(f"Loaded {len(used)} top models for disagreement: {used}")
    P = np.column_stack(preds)
    disagreement = P.std(axis=1)
    mean_pred = P.mean(axis=1)

    df = pd.DataFrame({
        'idx': np.arange(513),
        'name': te_df['Molecule Name'],
        'smiles': te_df['SMILES'],
        'top1_unb': top1_unb,
        'top1_tr':  top1_tr,
        'disagreement': disagreement,
        'mean_pred': mean_pred,
        'unblinded': [i in unb_te_idx for i in range(513)],
    })
    sb = df[~df['unblinded']].copy()
    print(f"\nStill-blind: {len(sb)}")
    print(f"top1_unb stats: min={sb['top1_unb'].min():.3f} median={sb['top1_unb'].median():.3f} max={sb['top1_unb'].max():.3f}")
    print(f"top1_tr stats:  min={sb['top1_tr'].min():.3f} median={sb['top1_tr'].median():.3f} max={sb['top1_tr'].max():.3f}")
    print(f"disagreement:   min={sb['disagreement'].min():.3f} median={sb['disagreement'].median():.3f} max={sb['disagreement'].max():.3f}")

    # Composite difficulty: low sim + high disagreement = hard
    sb['difficulty'] = (
        -sb['top1_unb'].fillna(0) +
        -sb['top1_tr'].fillna(0) +
        sb['disagreement'].fillna(0) * 1.5
    )
    sb = sb.sort_values('difficulty', ascending=False).reset_index(drop=True)
    print(f"\nHard-50 still-blind compounds:")
    print(sb[['name', 'top1_unb', 'top1_tr', 'disagreement', 'mean_pred']].head(50).to_string(index=False))
    sb.to_csv(DATA_PROCESSED / "nb326_hard_compounds.csv", index=False)
    print(f"\nWrote nb326_hard_compounds.csv")


if __name__ == "__main__":
    main()
