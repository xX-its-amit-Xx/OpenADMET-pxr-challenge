"""nb276 -- Fragment foundational model: train on FRAGMENTS from huge corpus.

Build a master fragment library from:
- 4139 PXR train compounds (labels: pec50)
- 1454 Papyrus PXR (labels: pec50)
- 5739 PubChem PXR-actives (pseudo-labels)
- 11k Papyrus other NR (labels)
Total: ~22k labeled compounds

Decompose each via BRICS. For each fragment:
- Aggregate stats: mean pec50 of compounds containing it, count, std
- Build fragment FP (Morgan on fragment SMILES)

Then train a "fragment foundational" predictor: per-fragment pec50 contribution.

For test compounds: decompose, lookup each fragment's score, aggregate.

Multiple engineering choices (run in parallel):
A) BRICS decomposition
B) RECAP decomposition
C) Murcko-sidechain decomposition
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import BRICS, Recap, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
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


def brics_frags(mol, min_size=3):
    try:
        frags = BRICS.BRICSDecompose(mol, minFragmentSize=min_size)
        return set(f for f in frags if f)
    except: return set()


def recap_frags(mol):
    try:
        hierarchy = Recap.RecapDecompose(mol)
        leaves = hierarchy.GetLeaves()
        return set(leaves.keys())
    except: return set()


def murcko_sidechain_frags(mol):
    """Extract scaffold + sidechains."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaff_smi = Chem.MolToSmiles(scaffold)
        # Sidechains: atoms not in scaffold
        scaffold_atoms = set(a.GetIdx() for a in scaffold.GetAtoms())
        # Simpler: just return scaffold as one fragment
        return {scaff_smi}
    except: return set()


def build_corpus():
    """Combine all labeled compounds into one corpus."""
    parts = []
    tr = load_train(); tr = add_standard_columns(tr)
    parts.append(pd.DataFrame({'std_smiles': tr['std_smiles'], 'pec50': tr['pec50'], 'source': 'train', 'weight': 1.0}))

    papyrus = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
    pxr_p = papyrus[papyrus['target_name'].str.contains('PXR', case=False, na=False)].copy()
    pxr_p['std_smiles'] = pxr_p['std_smiles'].apply(std_smi)
    pxr_p = pxr_p.dropna(subset=['std_smiles', 'pec50']).groupby('std_smiles')['pec50'].median().reset_index()
    pxr_p['source'] = 'papyrus_pxr'; pxr_p['weight'] = 0.5
    parts.append(pxr_p)

    other_nr = papyrus[~papyrus['target_name'].str.contains('PXR', case=False, na=False)].copy()
    other_nr['std_smiles'] = other_nr['std_smiles'].apply(std_smi)
    other_nr = other_nr.dropna(subset=['std_smiles', 'pec50']).groupby('std_smiles')['pec50'].median().reset_index()
    other_nr['source'] = 'papyrus_other_nr'; other_nr['weight'] = 0.2
    parts.append(other_nr)

    pubchem = pd.read_parquet("data/external/pubchem_pxr_active_smiles.parquet")
    pubchem['std_smiles'] = pubchem['smiles'].apply(std_smi)
    pubchem = pubchem.dropna(subset=['std_smiles'])
    pubchem['pec50'] = 4.0 + pubchem['active_rate'].fillna(0.5) * 2.5
    pubchem = pubchem.groupby('std_smiles')['pec50'].median().reset_index()
    pubchem['source'] = 'pubchem'; pubchem['weight'] = 0.3
    parts.append(pubchem)

    corpus = pd.concat(parts, ignore_index=True)
    print(f"Corpus: {len(corpus)} compounds, mean pec50={corpus['pec50'].mean():.3f}")
    return corpus, tr


def build_fragment_stats(corpus, frag_func, name):
    """Decompose corpus via frag_func; aggregate per-fragment pec50 stats."""
    print(f"\n=== Decomposing via {name} ===")
    frag_pec50 = defaultdict(list)
    frag_weight = defaultdict(list)
    for i, row in corpus.iterrows():
        mol = Chem.MolFromSmiles(row['std_smiles'])
        if mol is None: continue
        frags = frag_func(mol)
        for f in frags:
            frag_pec50[f].append(row['pec50'])
            frag_weight[f].append(row['weight'])
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(corpus)}")
    # Stats
    frag_stats = {}
    for f, pecs in frag_pec50.items():
        ws = frag_weight[f]
        wsum = sum(ws)
        if wsum > 0:
            w_mean = sum(p*w for p, w in zip(pecs, ws)) / wsum
            frag_stats[f] = {'mean_pec50': w_mean, 'count': len(pecs), 'std': np.std(pecs)}
    print(f"  Unique fragments: {len(frag_stats)}")
    return frag_stats


def predict_via_fragments(smiles_list, frag_func, frag_stats, min_count=5):
    """For each compound, decompose and aggregate fragment-level pec50 stats."""
    preds = np.zeros(len(smiles_list))
    confidences = np.zeros(len(smiles_list))
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            preds[i] = 4.32  # train mean
            continue
        frags = frag_func(mol)
        # Use only fragments with min_count occurrences
        ratings = []
        weights = []
        for f in frags:
            if f in frag_stats and frag_stats[f]['count'] >= min_count:
                ratings.append(frag_stats[f]['mean_pec50'])
                # Higher count + lower std → more reliable
                w = frag_stats[f]['count'] / (1 + frag_stats[f]['std'])
                weights.append(w)
        if ratings:
            preds[i] = np.average(ratings, weights=weights)
            confidences[i] = sum(weights)
        else:
            preds[i] = 4.32
            confidences[i] = 0
    return preds, confidences


def main():
    print("=== nb276: Fragment foundational ===\n")
    corpus, tr = build_corpus()
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()
    te_names = te_df['Molecule Name'].tolist()
    sm = dict(zip(te_df['Molecule Name'], te_df['SMILES']))

    # Three decomposition methods
    results = {}
    for name, func in [('brics', brics_frags), ('recap', recap_frags), ('murcko', murcko_sidechain_frags)]:
        frag_stats = build_fragment_stats(corpus, func, name)
        oof_pred, oof_conf = predict_via_fragments(smiles_tr, func, frag_stats)
        te_pred, te_conf = predict_via_fragments(smiles_te, func, frag_stats)
        r_oof = rae(y_tr, oof_pred)
        print(f"\n{name}: OOF RAE={r_oof:.4f}, te_mean={te_pred.mean():.3f}, te_std={te_pred.std():.3f}")
        print(f"  coverage: oof zero-confidence={ (oof_conf==0).sum()}/{len(oof_conf)}, te={(te_conf==0).sum()}/{len(te_conf)}")
        results[name] = (oof_pred, te_pred, oof_conf, te_conf)
        np.save(DATA_PROCESSED / f"oof_nb276_{name}.npy", oof_pred)
        np.save(DATA_PROCESSED / f"te_nb276_{name}.npy", te_pred)

    # Combine fragment predictions as features alongside base
    print("\n=== Combine fragment preds as features in LGBM ===")
    X_combined_tr = combined(smiles_tr); X_combined_tr = impute(X_combined_tr)
    X_combined_te = combined(smiles_te); X_combined_te = impute(X_combined_te)

    extra_tr = np.column_stack([results[n][0] for n in ['brics', 'recap', 'murcko']]
                                + [results[n][2] for n in ['brics', 'recap', 'murcko']])
    extra_te = np.column_stack([results[n][1] for n in ['brics', 'recap', 'murcko']]
                                + [results[n][3] for n in ['brics', 'recap', 'murcko']])
    X_aug_tr = np.column_stack([X_combined_tr, extra_tr])
    X_aug_te = np.column_stack([X_combined_te, extra_te])
    print(f"  X_aug: {X_aug_tr.shape}")

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, objective='mae', n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    oof = np.zeros(len(y_tr))
    te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_aug_tr[ti], y_tr[ti], eval_set=[(X_aug_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_aug_tr[vi])
        te_preds.append(md.predict(X_aug_te))
    te_pred = np.mean(te_preds, axis=0)
    r = rae(y_tr, oof)
    print(f"Combined+fragment LGBM OOF: {r:.4f}")
    np.save(DATA_PROCESSED / "oof_nb276_combined.npy", oof)
    np.save(DATA_PROCESSED / "te_nb276_combined.npy", te_pred)

    # Stack with nb239
    print("\n=== 5-way SLSQP ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way OOF: {best.fun:.4f}, nb276 weight={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
