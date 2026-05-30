"""nb331 -- Per-compound predictor routing.

For each of the 253 unblind compounds, compute per-predictor absolute error.
Train a meta-classifier (LGBM-multiclass) that takes the COMPOUND'S features
(Morgan + RDKit + top-1 sim to train + top-1 sim to unblind) and predicts
WHICH of {nb320, nb321, nb324, nb328, mean} has the lowest error for that compound.
Apply to 260 still-blind to route each compound to its expected best predictor.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.stats import spearmanr

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
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
    print("=== nb331: per-compound predictor routing ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    unb_smis = unb['SMILES'].apply(std_smi).tolist()
    unb_y = unb['pEC50'].values
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])
    still_blind_idx = np.array([i for i in range(513) if i not in set(unb_te_idx)])

    # Load candidate predictors
    P = {
        'nb320': np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy"),
        'nb321': np.load(DATA_PROCESSED / "te_nb321_augmented.npy"),
        'nb324': np.load(DATA_PROCESSED / "te_nb324_distilled.npy"),
        'nb328': np.load(DATA_PROCESSED / "te_nb328_chemprop_aug.npy"),
    }
    P_names = list(P.keys())

    # Per-unblind-compound: which predictor has lowest |error|?
    errors = np.column_stack([np.abs(unb_y - P[n][unb_te_idx]) for n in P_names])  # (253, n_pred)
    winners = errors.argmin(axis=1)
    print(f"Winner distribution on 253 unblind:")
    for i, n in enumerate(P_names):
        print(f"  {n}: best on {(winners == i).sum()} compounds")

    # Features for the meta-classifier: compound features + sim metrics
    print("\nBuilding meta-classifier features...")
    X_unb = impute(combined(unb_smis)).astype(np.float32)
    X_blind = impute(combined([smiles_te[i] for i in still_blind_idx])).astype(np.float32)
    # Top-1 sim to train + sim to unblind
    fps_tr = [fp(s) for s in smiles_tr]
    fps_tr_v = [f for f in fps_tr if f is not None]
    def sim_feats(smis, fps_other=None):
        out = np.zeros((len(smis), 2), dtype=np.float32)
        for i, s in enumerate(smis):
            f = fp(s)
            if f is None: continue
            out[i, 0] = float(np.array(BulkTanimotoSimilarity(f, fps_tr_v)).max())
            if fps_other is not None:
                fps_other_v = [g for g in fps_other if g is not None]
                if fps_other_v:
                    out[i, 1] = float(np.array(BulkTanimotoSimilarity(f, fps_other_v)).max())
        return out
    sim_unb = sim_feats(unb_smis)
    sim_blind = sim_feats([smiles_te[i] for i in still_blind_idx])
    X_unb_full = np.column_stack([X_unb, sim_unb])
    X_blind_full = np.column_stack([X_blind, sim_blind])

    # Standardize
    mu = X_unb_full.mean(0); sd = X_unb_full.std(0) + 1e-6
    X_unb_full = ((X_unb_full - mu) / sd).clip(-5, 5).astype(np.float32)
    X_blind_full = ((X_blind_full - mu) / sd).clip(-5, 5).astype(np.float32)

    # LGBM multiclass classifier
    print("Training meta-classifier (LGBM)...")
    md = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                            min_child_samples=8, n_jobs=4, random_state=42, verbose=-1)
    md.fit(X_unb_full, winners)
    # In-sample accuracy
    tr_acc = md.score(X_unb_full, winners)
    print(f"In-sample (overfit) accuracy: {tr_acc:.3f}")

    # Soft prediction on still-blind: predict_proba then route
    proba_blind = md.predict_proba(X_blind_full)  # (260, n_pred)
    print(f"Avg routing probabilities (still-blind):")
    for i, n in enumerate(P_names):
        print(f"  {n}: {proba_blind[:, i].mean():.3f}")

    # Apply: blend per-compound with proba weights
    final = P['nb320'].copy()  # start with safest
    P_blind = np.column_stack([P[n][still_blind_idx] for n in P_names])  # (260, n_pred)
    routed = (P_blind * proba_blind).sum(axis=1)
    final[still_blind_idx] = routed
    # Inject truth on unblind
    final[unb_te_idx] = unb_y
    print(f"\nfinal: mean={final.mean():.3f} std={final.std():.3f}")
    print(f"still-blind portion std: {final[still_blind_idx].std():.3f}")

    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb331_routed_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")
    np.save(DATA_PROCESSED / "te_nb331_routed.npy", final)


if __name__ == "__main__":
    main()
