"""nb150 -- Auto-discover SMARTS rules from active vs inactive PXR compounds.

Unlike nb228 (curated medchem SMARTS), this DERIVES rules by:
  1. Split train into actives (pEC50 >= 5.5) and inactives (pEC50 <= 4.0)
  2. For each active compound, extract circular fragments at multiple radii
  3. Compute enrichment ratio: P(fragment | active) / P(fragment | inactive)
  4. Keep top-K most enriched (and most depleted) fragments
  5. Use these as binary features for LGBM

This is essentially Murcko + structural alert mining, automated.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)


def get_circular_smarts(smi, radii=(1, 2, 3)):
    """Extract all unique circular fragments from a molecule at given radii."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return set()
    out = set()
    for r in radii:
        for atom in mol.GetAtoms():
            try:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom.GetIdx())
                if not env: continue
                atom_map = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
                smi_sub = Chem.MolToSmiles(submol, canonical=True)
                if smi_sub:
                    out.add(smi_sub)
            except Exception:
                continue
    return out


def main():
    print("=== nb150: SMARTS rule mining ===\n")
    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    # Split by activity
    actives = y_tr >= 5.5
    inactives = y_tr <= 4.0
    print(f"Actives (pEC50>=5.5): {actives.sum()}  Inactives (<=4.0): {inactives.sum()}")

    # Count fragment occurrences in active vs inactive
    print("Extracting fragments from train compounds...")
    import time; t0 = time.time()
    active_frags = Counter()
    inactive_frags = Counter()
    for i, smi in enumerate(smiles_tr):
        frags = get_circular_smarts(smi)
        if actives[i]:
            active_frags.update(frags)
        elif inactives[i]:
            inactive_frags.update(frags)
        if (i+1) % 1000 == 0:
            print(f"  {i+1}/{len(smiles_tr)}  ({time.time()-t0:.0f}s)")
    print(f"  Total unique active frags: {len(active_frags):,}")
    print(f"  Total unique inactive frags: {len(inactive_frags):,}")

    # Compute enrichment ratio (with smoothing)
    n_active = actives.sum()
    n_inactive = inactives.sum()
    enrichment = {}
    for frag in set(list(active_frags.keys()) + list(inactive_frags.keys())):
        a = active_frags.get(frag, 0)
        i = inactive_frags.get(frag, 0)
        # Only keep fragments that appear at least 10 times total
        if a + i < 10: continue
        p_a = (a + 1) / (n_active + 2)
        p_i = (i + 1) / (n_inactive + 2)
        enrichment[frag] = (np.log(p_a / p_i), a, i)

    # Top 100 enriched in actives + top 100 enriched in inactives
    en_df = pd.DataFrame([(f, e[0], e[1], e[2]) for f, e in enrichment.items()],
                          columns=["frag","log_ratio","n_act","n_inact"])
    en_df = en_df.sort_values("log_ratio", ascending=False)
    print(f"\nEnrichment ratios: {len(en_df)} fragments scored")
    print("Top 20 ACTIVE-enriched fragments:")
    print(en_df.head(20).to_string(index=False))
    print("\nTop 20 INACTIVE-enriched fragments (depleted in actives):")
    print(en_df.tail(20).to_string(index=False))

    # Select top 200 (100 each direction) as binary features
    top_active = en_df.head(100)["frag"].tolist()
    top_inactive = en_df.tail(100)["frag"].tolist()
    selected_frags = top_active + top_inactive
    print(f"\nTotal selected fragments as binary features: {len(selected_frags)}")

    # Build binary features for all train + test
    def featurize_frags(smiles_list, frags):
        n = len(smiles_list)
        F = np.zeros((n, len(frags)), dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            cs = get_circular_smarts(smi)
            for j, f in enumerate(frags):
                if f in cs:
                    F[i, j] = 1.0
        return F

    print("Computing fragment binary features for train + test...")
    F_tr = featurize_frags(smiles_tr, selected_frags)
    F_te = featurize_frags(smiles_te, selected_frags)
    print(f"  F shapes: train={F_tr.shape}  test={F_te.shape}")
    print(f"  Train: avg {F_tr.sum(axis=1).mean():.1f} matching frags per compound")

    # Augmented LGBM CV
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)
    X_tr_aug = np.hstack([X_tr_base, F_tr])
    X_te_aug = np.hstack([X_te_base, F_te])

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)
    for name, Xt, Xe in [("base_only", X_tr_base, X_te_base),
                          ("smarts_aug", X_tr_aug, X_te_aug)]:
        oof = np.zeros(len(y_tr)); te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(Xt[tr_idx], y_tr[tr_idx],
                  eval_set=[(Xt[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(Xt[va_idx])
            te_preds.append(m.predict(Xe))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof); ratio = te_pred.std() / oof.std()
        print(f"  {name:14s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")
        if name == "smarts_aug":
            np.save(DATA_PROCESSED / "oof_nb150_smarts.npy", oof)
            np.save(DATA_PROCESSED / "te_nb150_smarts.npy", te_pred)
            print("  Saved")


if __name__ == "__main__":
    main()
