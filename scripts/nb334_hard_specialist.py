"""nb334 -- Per-hard-compound specialist using unblind anchors as KNN base.

For each of the top-50 hardest still-blind compounds (from nb326), train a
tiny Ridge regression on the K=50 most-similar TRAIN + UNBLIND neighbors.
Use combined features. Predict that specific compound.

For non-hard compounds (210 of 260), use nb332 meta-gbr or nb320.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

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
    print("=== nb334: Hard-compound specialist ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    smiles_tr = tr['std_smiles'].tolist()
    y_tr = tr['pec50'].values.astype(np.float64)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    unb_smis = unb['SMILES'].apply(std_smi).tolist()
    unb_y = unb['pEC50'].values
    name_to_idx = {n: i for i, n in enumerate(te_df['Molecule Name'])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb['Molecule Name'] if n in name_to_idx])

    # Load hard list from nb326
    hard_df = pd.read_csv(DATA_PROCESSED / "nb326_hard_compounds.csv")
    # hard_df has 'name' column; pick top-50
    hard_names = hard_df.head(50)['name'].tolist()
    hard_idx = [name_to_idx[n] for n in hard_names if n in name_to_idx]
    print(f"Hard-50: {len(hard_idx)} indices identified")

    # Combined pool: train + unblind (anchors)
    pool_smis = list(smiles_tr) + list(unb_smis)
    pool_y = np.concatenate([y_tr, unb_y])
    print(f"Pool: {len(pool_smis)} compounds (train+unblind)")

    # Compute pool FPs + features
    print("Featurising...")
    pool_fps = [fp(s) for s in pool_smis]
    te_fps = [fp(s) for s in smiles_te]
    X_pool = impute(combined(pool_smis)).astype(np.float32)
    X_te = impute(combined(smiles_te)).astype(np.float32)
    mu = X_pool.mean(0); sd = X_pool.std(0) + 1e-6
    X_pool = ((X_pool - mu) / sd).clip(-5, 5).astype(np.float32)
    X_te = ((X_te - mu) / sd).clip(-5, 5).astype(np.float32)

    # Base predictor for non-hard compounds: nb332 meta-gbr
    try:
        base = pd.read_csv(SUBMISSIONS / "nb332_meta_gbr_truth.csv")
        base_pred = base['pEC50'].values
        print(f"Base predictor: nb332_meta_gbr_truth")
    except Exception:
        base_pred = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
        # truth inject
        base_pred[unb_te_idx] = unb_y
        print(f"Base predictor: nb320 truth-injected (fallback)")

    # Specialist: for each hard compound, train Ridge on K=50 nearest pool neighbors
    K = 50
    final = base_pred.copy()
    n_changed = 0
    for i in hard_idx:
        f = te_fps[i]
        if f is None: continue
        sims = np.array(BulkTanimotoSimilarity(f, [g for g in pool_fps if g is not None]))
        valid_pool = [(k, g) for k, g in enumerate(pool_fps) if g is not None]
        valid_idx_pool = [t[0] for t in valid_pool]
        top = np.argsort(sims)[::-1][:K]
        nbr_pool_idx = [valid_idx_pool[k] for k in top]
        X_nbr = X_pool[nbr_pool_idx]; y_nbr = pool_y[nbr_pool_idx]
        if len(np.unique(y_nbr)) < 5: continue  # need variance
        try:
            r = Ridge(alpha=1.0); r.fit(X_nbr, y_nbr)
            pred = float(r.predict(X_te[i:i+1])[0])
            # Blend specialist with base 60/40
            final[i] = 0.6 * pred + 0.4 * base_pred[i]
            n_changed += 1
        except Exception: pass

    print(f"\nSpecialist applied to {n_changed}/{len(hard_idx)} hard compounds")
    print(f"final: mean={final.mean():.3f} std={final.std():.3f}")

    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': final,
    })
    out = SUBMISSIONS / "nb334_hard_specialist_truth.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out.name}")
    np.save(DATA_PROCESSED / "te_nb334_hard_specialist.npy", final)


if __name__ == "__main__":
    main()
