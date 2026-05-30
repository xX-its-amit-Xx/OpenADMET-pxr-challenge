"""nb279 -- Pocket-aware fragment binding predictor.

Use pdb64 fragment-residue contact database + literature pec50 for those 70
ligands to build a SCORING FUNCTION:

  for each test compound:
    1. BRICS decompose into fragments
    2. For each fragment, find K nearest pdb64 fragments (Tanimoto)
    3. Inherit their residue-contact patterns + parent compound pec50
    4. Score test compound = weighted-average of (parent pec50 * fragment-similarity * pocket-match-quality)

Pocket-match-quality: emphasize contacts with TOP pocket residues (GLN167, MET125,
PHE170, TRP181, SER129, HIS289 — the residues most-contacted by known binders).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from collections import defaultdict

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def brics_fragments_for_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return []
    try:
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        if not brics_bonds:
            return [Chem.MolToSmiles(mol)]
        bond_indices = [mol.GetBondBetweenAtoms(*pair[0]).GetIdx() for pair in brics_bonds]
        fragmented = Chem.FragmentOnBonds(mol, bond_indices)
        return [Chem.MolToSmiles(fm) for fm in Chem.GetMolFrags(fragmented, asMols=True)]
    except Exception:
        return [Chem.MolToSmiles(mol)]


def morgan(smiles, radius=2, n_bits=2048):
    fps = []
    for s in smiles:
        # Strip BRICS markers [*:N] - replace with H for FP computation
        clean = s.replace('[*]', '[H]')
        for n in range(50):
            clean = clean.replace(f'[{n}*]', '[H]')
        mol = Chem.MolFromSmiles(clean)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


# Top 20 PXR pocket residues from data-driven extraction (nb277)
KEY_POCKET_RESIDUES = {
    ('GLN', '167'): 1.0, ('MET', '125'): 1.0, ('PHE', '170'): 1.0, ('TRP', '181'): 1.0,
    ('SER', '129'): 0.9, ('MET', '205'): 0.8, ('HIS', '289'): 0.9, ('LEU', '293'): 0.7,
    ('LEU', '91'): 0.7, ('TYR', '188'): 0.7, ('MET', '128'): 0.6, ('PHE', '163'): 0.6,
    ('PHE', '133'): 0.5, ('PHE', '302'): 0.5, ('HIS', '209'): 0.5, ('ARG', '292'): 0.5,
    ('MET', '307'): 0.4, ('LEU', '122'): 0.4, ('ILE', '296'): 0.4, ('CYS', '166'): 0.3,
}


def main():
    print("=== nb279: Pocket-aware fragment binding predictor ===\n")
    contacts_df = pd.read_parquet("data/processed/pxr_fragment_residue_contacts.parquet")
    smiles_df = pd.read_csv("data/external/dargason_cofolding/pdb64_raw_smiles.csv")
    print(f"Contacts: {len(contacts_df)}, pdb64 smiles: {len(smiles_df)}")

    # Filter out ligand-ligand contacts (res_name in CCD list)
    common_aa = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}
    contacts_df = contacts_df[contacts_df['res_name'].isin(common_aa)].reset_index(drop=True)
    print(f"After filtering to amino acids: {len(contacts_df)}")

    # Score pdb64 fragments by pocket residue alignment
    print("\n=== Building per-fragment pocket score ===")
    frag_pocket_score = {}
    for frag, group in contacts_df.groupby('fragment_smiles'):
        n_contacts_total = len(group)
        weighted_pocket = 0
        for _, c in group.iterrows():
            key = (c['res_name'], c['res_num'])
            weight = KEY_POCKET_RESIDUES.get(key, 0.1)  # 0.1 default for any AA contact
            weighted_pocket += weight
        frag_pocket_score[frag] = {
            'n_contacts': n_contacts_total,
            'n_pdbs': group['pdb_id'].nunique(),
            'pocket_score': weighted_pocket / max(n_contacts_total, 1),
            'total_weighted': weighted_pocket
        }

    # Load PDB literature pec50 (we don't have these for pdb64 directly; use Papyrus PXR for matching SMILES)
    # For pdb64 SMILES that match Papyrus/train, get their pec50
    papyrus_pxr = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
    papyrus_pxr = papyrus_pxr[papyrus_pxr['target_name'].str.contains('PXR', case=False, na=False)].copy()
    papyrus_pxr['std_smiles'] = papyrus_pxr['std_smiles'].apply(std_smi)
    papyrus_pxr = papyrus_pxr.dropna(subset=['std_smiles', 'pec50']).groupby('std_smiles')['pec50'].median()

    smiles_df['std_smiles'] = smiles_df['smiles'].apply(std_smi)
    smiles_df['lit_pec50'] = smiles_df['std_smiles'].map(papyrus_pxr)
    n_lit = smiles_df['lit_pec50'].notna().sum()
    print(f"PDB64 ligands with literature pec50 from Papyrus: {n_lit}/{len(smiles_df)}")
    # For others, use median of pec50 (pdb64 = known binders, likely high affinity ~5.5-6.5)
    smiles_df['lit_pec50'] = smiles_df['lit_pec50'].fillna(5.5)
    print(f"pec50 stats: mean={smiles_df['lit_pec50'].mean():.3f}")

    # Build pdb64 fragment FPs + per-fragment parent pec50
    pdb_frag_data = []
    for _, row in smiles_df.iterrows():
        frags = brics_fragments_for_smi(row['smiles'])
        for f in frags:
            pdb_frag_data.append({
                'fragment_smiles': f, 'parent_pec50': row['lit_pec50'],
                'pdb_id': row['pdb_id']
            })
    pdb_frag_df = pd.DataFrame(pdb_frag_data)
    pdb_frag_df = pdb_frag_df.merge(pd.DataFrame.from_dict(frag_pocket_score, orient='index').reset_index().rename(columns={'index': 'fragment_smiles'}), on='fragment_smiles', how='left')
    pdb_frag_df['pocket_score'] = pdb_frag_df['pocket_score'].fillna(0)
    pdb_frag_df['n_pdbs'] = pdb_frag_df['n_pdbs'].fillna(0)

    # Take unique fragments only
    pdb_frags = pdb_frag_df.groupby('fragment_smiles').agg(
        parent_pec50=('parent_pec50', 'mean'),
        n_pdbs=('n_pdbs', 'max'),
        pocket_score=('pocket_score', 'max')
    ).reset_index()
    print(f"\nUnique pdb64 fragments: {len(pdb_frags)}")
    pdb_frags_meaningful = pdb_frags[pdb_frags['n_pdbs'] >= 1]
    print(f"With contacts: {len(pdb_frags_meaningful)}")

    # Compute Morgan FPs for pdb64 fragments
    pdb_fps = morgan(pdb_frags_meaningful['fragment_smiles'].tolist())
    valid_mask = [fp is not None for fp in pdb_fps]
    pdb_frags_meaningful = pdb_frags_meaningful.reset_index(drop=True)[valid_mask].reset_index(drop=True)
    pdb_fps = [fp for fp in pdb_fps if fp is not None]
    print(f"With valid FPs: {len(pdb_fps)}")

    # Load test compounds + train
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    def predict_pocket_aware(smi_list, K=5):
        preds = np.zeros(len(smi_list))
        for i, smi in enumerate(smi_list):
            frags = brics_fragments_for_smi(smi)
            if not frags:
                preds[i] = 4.32; continue
            frag_fps = morgan(frags)
            scores = []
            weights = []
            for ff in frag_fps:
                if ff is None: continue
                sims = np.array(BulkTanimotoSimilarity(ff, pdb_fps))
                top_idx = np.argsort(sims)[::-1][:K]
                top_sims = sims[top_idx]
                top_pec50 = pdb_frags_meaningful.iloc[top_idx]['parent_pec50'].values
                top_pocket = pdb_frags_meaningful.iloc[top_idx]['pocket_score'].values
                w = top_sims * (1 + top_pocket)  # boost by pocket score
                if w.sum() > 0:
                    scores.append(np.average(top_pec50, weights=w))
                    weights.append(w.sum())
            if scores:
                preds[i] = np.average(scores, weights=weights)
            else:
                preds[i] = 4.32
        return preds

    print("\nPredicting on train (OOF substitute) and test...")
    oof_pred = predict_pocket_aware(smiles_tr)
    te_pred = predict_pocket_aware(smiles_te)
    r = rae(y_tr, oof_pred)
    print(f"Pocket-aware OOF RAE: {r:.4f}  (this is not real OOF, used pdb64 for training)")
    print(f"te: mean={te_pred.mean():.3f}, std={te_pred.std():.3f}")

    np.save(DATA_PROCESSED / "oof_nb279_pocket_aware.npy", oof_pred)
    np.save(DATA_PROCESSED / "te_nb279_pocket_aware.npy", te_pred)

    # Stack with nb239
    print("\n=== 5-way SLSQP w/ nb279 ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof_pred])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way OOF: {best.fun:.4f}, weight={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
