"""nb342 -- Per-compound trust-score blending.

For each still-blind test compound:
  1. Find K=20 nearest unblind compounds by Tanimoto.
  2. For each candidate predictor, compute its MAE on those K neighbors.
  3. Weight each predictor by 1 / (MAE + epsilon) for THIS specific compound.
  4. Blend.

This is per-compound expertise routing: a model that's accurate near
similar compounds gets more weight on this compound.
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
    print("=== nb342: per-compound trust score blending ===\n")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    unb_y = unb['pEC50'].values
    still_blind = np.array([i for i in range(513) if i not in set(unb_te_idx)])

    top_models = [
        'nb93_chemprop_large_gpu', 'nb130_external_pxr', 'nb264_chemprop_mt',
        'nb303_dann', 'chemprop_aux', 'chemprop_aux_BAD4141', 'nb305_mope',
        'nb306_cepsmim', 'catboost', 'grand_v6b_calib', 'deep_ensemble',
        'nb320_phase2_top20',
    ]
    preds = []
    names = []
    for n in top_models:
        p = DATA_PROCESSED / f"te_{n}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,):
                preds.append(arr); names.append(n)
    print(f"Pool: {len(names)} predictors")
    M = np.column_stack(preds)  # (513, K)

    # Compute FPs
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    fps_te = [fp(s) for s in smiles_te]
    fps_unb = [fps_te[i] for i in unb_te_idx]
    valid_unb_mask = [f is not None for f in fps_unb]
    fps_unb_v = [f for f in fps_unb if f is not None]
    unb_y_v = unb_y[np.array(valid_unb_mask)]
    M_unb = M[unb_te_idx][np.array(valid_unb_mask)]
    print(f"Valid unblind anchors: {len(fps_unb_v)}")

    K_NN = 20
    final = np.zeros(513)
    final[unb_te_idx] = unb_y  # truth inject
    n_routed = 0
    for i in still_blind:
        f = fps_te[i]
        if f is None:
            final[i] = M[i].mean()
            continue
        sims = np.array(BulkTanimotoSimilarity(f, fps_unb_v))
        top = np.argsort(sims)[::-1][:K_NN]
        # MAE of each predictor on the K nearest unblind compounds
        nbr_preds = M_unb[top]  # (K_NN, n_predictors)
        nbr_y = unb_y_v[top]
        mae_per_pred = np.abs(nbr_preds - nbr_y[:, None]).mean(axis=0)
        # Weight by 1/(MAE + 0.05)
        w = 1.0 / (mae_per_pred + 0.05)
        w = w / w.sum()
        final[i] = float((M[i] * w).sum())
        n_routed += 1
    print(f"Per-compound routing applied to {n_routed}/{len(still_blind)} still-blind")
    print(f"final: mean={final.mean():.3f} std={final.std():.3f}")

    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb342_per_compound_trust_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
