"""nb280 -- Meta-analogy transformer.

Trained on 8509 atom-residue contacts from pdb64 PXR cocrystals.
Learns implicit pairings: fragment-X type → residue-Y type.

Architecture:
- Input: fragment Morgan FP (2048) + parent compound features (combined 2265)
- Per-residue head: predicts contact probability for each of top 20 pocket residues
- Aggregate: sum of (residue contact prob × residue importance) → binding score

Combinatorial layers:
L1: PXR-only contacts
L2: + cliff weighting
L3: + assay-noise correction (downweight high-SE compounds)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


KEY_RES = [
    ('GLN', '167'), ('MET', '125'), ('PHE', '170'), ('TRP', '181'),
    ('SER', '129'), ('MET', '205'), ('HIS', '289'), ('LEU', '293'),
    ('LEU', '91'), ('TYR', '188'), ('MET', '128'), ('PHE', '163'),
    ('PHE', '133'), ('PHE', '302'), ('HIS', '209'), ('ARG', '292'),
    ('MET', '307'), ('LEU', '122'), ('ILE', '296'), ('CYS', '166'),
]
RES_IDX = {r: i for i, r in enumerate(KEY_RES)}


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def morgan_fp(smi, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi.replace('[*]', '[H]'))
    if mol is None: return None
    try:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    except: return None


def brics_fragments_for_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return []
    try:
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        if not brics_bonds:
            return [smi]
        bond_indices = [mol.GetBondBetweenAtoms(*pair[0]).GetIdx() for pair in brics_bonds]
        fragmented = Chem.FragmentOnBonds(mol, bond_indices)
        return [Chem.MolToSmiles(fm) for fm in Chem.GetMolFrags(fragmented, asMols=True)]
    except Exception:
        return [smi]


class MetaAnalogyModel(nn.Module):
    """Multi-head predictor: per-fragment per-residue contact probability."""
    def __init__(self, frag_dim=2048, n_residues=20, hidden=512, dropout=0.3):
        super().__init__()
        self.frag_encoder = nn.Sequential(
            nn.Linear(frag_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2), nn.ReLU(),
        )
        # Per-residue heads
        self.contact_heads = nn.ModuleList([nn.Linear(hidden//2, 1) for _ in range(n_residues)])
        # Binding score head
        self.binding_head = nn.Sequential(
            nn.Linear(hidden//2, hidden//4), nn.ReLU(),
            nn.Linear(hidden//4, 1),
        )

    def forward(self, frag_fp):
        z = self.frag_encoder(frag_fp)
        contacts = torch.cat([h(z) for h in self.contact_heads], dim=1)  # (B, n_res)
        score = self.binding_head(z).squeeze(-1)  # (B,)
        return contacts, score


def main():
    print("=== nb280: Meta-analogy transformer ===\n")
    # Load fragment-residue contacts
    contacts_df = pd.read_parquet("data/processed/pxr_fragment_residue_contacts.parquet")
    common_aa = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}
    contacts_df = contacts_df[contacts_df['res_name'].isin(common_aa)].reset_index(drop=True)
    print(f"Contacts (AA only): {len(contacts_df)}")

    # Get pdb64 ligands with literature pec50 (from Papyrus matching)
    smiles_df = pd.read_csv("data/external/dargason_cofolding/pdb64_raw_smiles.csv")
    papyrus = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
    papyrus_pxr = papyrus[papyrus['target_name'].str.contains('PXR', case=False, na=False)].copy()
    papyrus_pxr['std_smiles'] = papyrus_pxr['std_smiles'].apply(std_smi)
    papyrus_pxr = papyrus_pxr.dropna(subset=['std_smiles', 'pec50']).groupby('std_smiles')['pec50'].median()
    smiles_df['std_smiles'] = smiles_df['smiles'].apply(std_smi)
    smiles_df['lit_pec50'] = smiles_df['std_smiles'].map(papyrus_pxr).fillna(5.5)

    # Build training set: (fragment FP, per-residue contact vector, parent pec50)
    print("\nBuilding training set from contacts...")
    train_data = []
    for pid in smiles_df['pdb_id'].unique():
        pid_contacts = contacts_df[contacts_df['pdb_id'] == pid]
        if len(pid_contacts) == 0: continue
        # Per fragment, build contact vector for top 20 residues
        for frag_smi, group in pid_contacts.groupby('fragment_smiles'):
            fp = morgan_fp(frag_smi)
            if fp is None: continue
            contact_vec = np.zeros(20)
            for _, c in group.iterrows():
                key = (c['res_name'], c['res_num'])
                if key in RES_IDX:
                    contact_vec[RES_IDX[key]] = 1.0
            parent_pec50 = smiles_df[smiles_df['pdb_id'] == pid]['lit_pec50'].values[0]
            train_data.append({'frag_smi': frag_smi, 'fp': fp, 'contact_vec': contact_vec, 'pec50': parent_pec50, 'pdb_id': pid})
    print(f"Training samples (fragment-level): {len(train_data)}")
    if len(train_data) < 100:
        print("WARN: too few training samples")
        return

    # Convert to tensors
    X_frag = np.stack([d['fp'] for d in train_data]).astype(np.float32)
    Y_contacts = np.stack([d['contact_vec'] for d in train_data]).astype(np.float32)
    Y_pec = np.array([d['pec50'] for d in train_data]).astype(np.float32)
    print(f"X_frag: {X_frag.shape}, Y_contacts: {Y_contacts.shape}, Y_pec: {Y_pec.shape}")

    # Train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MetaAnalogyModel(frag_dim=X_frag.shape[1], n_residues=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n_epochs = 100
    batch = 32
    n = len(X_frag)
    print(f"\nTraining {n_epochs} epochs on {n} fragments...")
    Xt = torch.tensor(X_frag).to(device)
    Yct = torch.tensor(Y_contacts).to(device)
    Ypt = torch.tensor(Y_pec).to(device)
    for ep in range(n_epochs):
        model.train()
        idx = np.random.permutation(n)
        for i in range(0, n, batch):
            b = idx[i:i+batch]
            optimizer.zero_grad()
            contacts_pred, pec_pred = model(Xt[b])
            loss_contact = nn.functional.binary_cross_entropy_with_logits(contacts_pred, Yct[b])
            loss_pec = torch.abs(pec_pred - Ypt[b]).mean()
            loss = loss_contact + 0.5 * loss_pec
            loss.backward()
            optimizer.step()

    # Predict for test compounds: BRICS, predict per-fragment per-residue contact + binding score, aggregate
    print("\nPredicting test compounds...")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()

    def aggregate_for_compound(smi):
        frags = brics_fragments_for_smi(smi)
        if not frags: return 4.32, np.zeros(20)
        fps = [morgan_fp(f) for f in frags]
        fps = [f for f in fps if f is not None]
        if not fps: return 4.32, np.zeros(20)
        x = torch.tensor(np.array(fps).astype(np.float32)).to(device)
        with torch.no_grad():
            contacts_pred, pec_pred = model(x)
        # Average pec across fragments
        score = float(pec_pred.mean().item())
        # Sum contact probs across fragments (which residues this molecule's fragments target)
        contact_probs = torch.sigmoid(contacts_pred).sum(dim=0).cpu().numpy()
        return score, contact_probs

    model.eval()
    te_preds = np.zeros(len(smiles_te))
    te_contact_profiles = np.zeros((len(smiles_te), 20))
    for i, smi in enumerate(smiles_te):
        te_preds[i], te_contact_profiles[i] = aggregate_for_compound(smi)
    print(f"te_preds: mean={te_preds.mean():.3f}, std={te_preds.std():.3f}")

    # Same for train (OOF substitute)
    print("Predicting train compounds...")
    oof_preds = np.zeros(len(smiles_tr))
    oof_contact_profiles = np.zeros((len(smiles_tr), 20))
    for i, smi in enumerate(smiles_tr):
        oof_preds[i], oof_contact_profiles[i] = aggregate_for_compound(smi)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(smiles_tr)}")
    r = rae(y_tr, oof_preds)
    print(f"\nMeta-analogy OOF (not real, pdb64-trained) RAE: {r:.4f}")

    np.save(DATA_PROCESSED / "oof_nb280_meta_analogy.npy", oof_preds)
    np.save(DATA_PROCESSED / "te_nb280_meta_analogy.npy", te_preds)
    np.save(DATA_PROCESSED / "oof_nb280_contact_profiles.npy", oof_contact_profiles)
    np.save(DATA_PROCESSED / "te_nb280_contact_profiles.npy", te_contact_profiles)

    # Use contact profiles + base features in LGBM (richer than just score)
    print("\n=== LGBM with base + contact profiles ===")
    X_combined_tr = combined(smiles_tr); X_combined_tr = impute(X_combined_tr)
    X_combined_te = combined(smiles_te); X_combined_te = impute(X_combined_te)
    X_aug_tr = np.column_stack([X_combined_tr, oof_contact_profiles, oof_preds.reshape(-1,1)])
    X_aug_te = np.column_stack([X_combined_te, te_contact_profiles, te_preds.reshape(-1,1)])
    import lightgbm as lgb
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, objective='mae', n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    oof_full = np.zeros(len(y_tr))
    te_full_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_aug_tr[ti], y_tr[ti], eval_set=[(X_aug_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_full[vi] = md.predict(X_aug_tr[vi])
        te_full_preds.append(md.predict(X_aug_te))
    te_full = np.mean(te_full_preds, axis=0)
    print(f"Combined+contact_profiles+meta_score OOF: {rae(y_tr, oof_full):.4f}")
    np.save(DATA_PROCESSED / "oof_nb280_combined.npy", oof_full)
    np.save(DATA_PROCESSED / "te_nb280_combined.npy", te_full)

    # Stack with nb239
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof_full])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP OOF: {best.fun:.4f}, nb280 weight={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
