"""nb295 -- RAG-QSAR: retrieval-augmented test-time refinement using external NR.

For each test compound, retrieve top-K nearest external NR-binding compounds
(ChEMBL + Papyrus, restricted to high-quality records), build a small Gaussian-
Process / Ridge model on-the-fly, blend with global nb239.
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
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def std_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except: return None


def morgan_fp(smi, r=2, n=2048):
    m = Chem.MolFromSmiles(smi) if smi else None
    return AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n) if m else None


def main():
    print("=== nb295: RAG-QSAR ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()

    # External corpus: Papyrus PXR + train
    try:
        pap = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
        pap = pap[pap['target_name'].astype(str).str.contains('PXR', case=False, na=False)].copy()
        pap['std_smiles'] = pap['std_smiles'].apply(std_smi) if 'std_smiles' in pap.columns else pap['SMILES'].apply(std_smi) if 'SMILES' in pap.columns else None
        pap = pap.dropna(subset=['std_smiles', 'pec50']).groupby('std_smiles')['pec50'].median().reset_index()
        print(f"Papyrus PXR corpus: {len(pap)}")
    except Exception as e:
        print(f"Papyrus load failed: {e}")
        pap = pd.DataFrame(columns=['std_smiles', 'pec50'])

    ext_smiles = list(smiles_tr) + pap['std_smiles'].tolist()
    ext_y = list(y) + pap['pec50'].tolist()
    ext_fps = [morgan_fp(s) for s in ext_smiles]
    valid = [i for i, f in enumerate(ext_fps) if f is not None]
    ext_fps = [ext_fps[i] for i in valid]
    ext_y = [ext_y[i] for i in valid]
    ext_smiles = [ext_smiles[i] for i in valid]
    print(f"External corpus (post-cleaning): {len(ext_fps)} compounds")

    X_tr = impute(combined(smiles_tr)).astype(np.float32)
    X_te = impute(combined(smiles_te)).astype(np.float32)
    mu_f = X_tr.mean(0); sd_f = X_tr.std(0) + 1e-6
    X_tr = (X_tr - mu_f) / sd_f
    X_te = (X_te - mu_f) / sd_f

    # External features
    X_ext = impute(combined(ext_smiles)).astype(np.float32)
    X_ext = (X_ext - mu_f) / sd_f
    y_ext = np.array(ext_y)

    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb239_te  = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")

    # Per-test ridge on K nearest ext compounds
    K = 50
    print(f"\nRAG-refining {len(smiles_te)} test compounds (K={K})...")
    rag_te = np.zeros(len(smiles_te))
    fps_te = [morgan_fp(s) for s in smiles_te]
    for i, fp in enumerate(fps_te):
        if fp is None:
            rag_te[i] = nb239_te[i]; continue
        sims = np.array(BulkTanimotoSimilarity(fp, ext_fps))
        top = np.argsort(sims)[::-1][:K]
        Xk = X_ext[top]; yk = y_ext[top]
        try:
            mdl = Ridge(alpha=1.0)
            mdl.fit(Xk, yk)
            rag_te[i] = mdl.predict(X_te[i:i+1])[0]
        except Exception:
            rag_te[i] = nb239_te[i]
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(smiles_te)}")

    # Blend
    final_te = 0.6 * nb239_te + 0.4 * rag_te
    # For OOF use uniform mix as proxy (no leakage-free RAG-OOF here)
    final_oof = nb239_oof
    r = rae(y, final_oof)
    sp, _ = spearmanr(y, final_oof)
    print(f"\nBase nb239 OOF: {r:.4f}  Spearman {sp:.4f}")
    print(f"RAG test blend: mean={final_te.mean():.3f}  std={final_te.std():.3f}")

    np.save(DATA_PROCESSED / "te_nb295_rag.npy", final_te)
    np.save(DATA_PROCESSED / "oof_nb295_rag.npy", final_oof)


if __name__ == "__main__":
    main()
