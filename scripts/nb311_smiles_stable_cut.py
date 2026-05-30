"""nb311 -- Stable-core SMILES cut (Murcko scaffold + 1-bond shell).

Strip peripheral substituents that bloat the SMILES with noisy alkyl chains
and complex stereochem. Keep the Murcko scaffold heavy atoms plus any heavy
atoms one bond away from them. The result is a "stable core" SMILES that's
more reproducible across tautomers/rotamers and more docking-tractable while
preserving the original pec50 label.

Retrain combined+LGBM on the relabeled (cut) SMILES with scaffold 5-fold CV,
then SLSQP into the 4 nb239 base components.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from scipy.optimize import minimize
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def stable_core_smiles(smi):
    """Return SMILES for Murcko scaffold + 1-bond neighborhood.

    If the molecule has no rings/scaffold, return the original SMILES.
    """
    if not smi or not isinstance(smi, str): return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return smi
    if scaf is None or scaf.GetNumAtoms() == 0:
        return smi

    # Identify scaffold atom indices in the parent mol.
    try:
        scaf_match = mol.GetSubstructMatch(scaf)
    except Exception:
        scaf_match = ()
    if not scaf_match:
        # Fallback: detect ring atoms directly.
        scaf_match = tuple(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
        if not scaf_match:
            return smi

    keep = set(scaf_match)
    # Add atoms 1 bond from scaffold atoms.
    for idx in list(scaf_match):
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            keep.add(nb.GetIdx())

    # Mark
    for a in mol.GetAtoms():
        a.SetProp("_keep", "1" if a.GetIdx() in keep else "0")

    # Build edit-mol with only kept atoms.
    em = Chem.RWMol(mol)
    # Remove atoms in reverse-index order
    to_del = [a.GetIdx() for a in em.GetAtoms() if a.GetProp("_keep") == "0"]
    for idx in sorted(to_del, reverse=True):
        em.RemoveAtom(idx)
    try:
        out = em.GetMol()
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    except Exception:
        return smi


def main():
    print("=== nb311: Murcko + 1-shell stable core SMILES ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    scaffolds = tr['scaffold'].tolist()

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te_raw = te_df['SMILES'].tolist()
    smiles_te = []
    for s in smiles_te_raw:
        m = Chem.MolFromSmiles(str(s)) if s else None
        smiles_te.append(Chem.MolToSmiles(m) if m else s)

    # Cut SMILES
    print("Cutting train SMILES to stable core (Murcko + 1-bond)...")
    t0 = time.time()
    smiles_tr_cut = [stable_core_smiles(s) for s in smiles_tr]
    print(f"  done in {time.time()-t0:.1f}s")
    n_changed_tr = sum(1 for a, b in zip(smiles_tr, smiles_tr_cut) if a != b)
    print(f"  train changed: {n_changed_tr}/{len(smiles_tr)}")

    print("Cutting test SMILES...")
    smiles_te_cut = [stable_core_smiles(s) for s in smiles_te]
    n_changed_te = sum(1 for a, b in zip(smiles_te, smiles_te_cut) if a != b)
    print(f"  test changed: {n_changed_te}/{len(smiles_te)}")

    # Featurize
    print("\nFeaturizing combined (Morgan + RDKit) on cut SMILES...")
    X_tr = impute(combined(smiles_tr_cut))
    X_te = impute(combined(smiles_te_cut))
    print(f"  X_tr {X_tr.shape}  X_te {X_te.shape}")

    # Scaffold 5-fold CV (use ORIGINAL scaffolds for splitting; cut SMILES
    # are inputs, original scaffolds preserve generalization-test semantics).
    splits = scaffold_kfold_indices(scaffolds, n_splits=5)
    oof = np.zeros(len(y), dtype=np.float64)
    te_pred = np.zeros(len(smiles_te), dtype=np.float64)
    fold_raes = []
    for fold, (tr_idx, val_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(
            n_estimators=500, num_leaves=64, learning_rate=0.05,
            min_child_samples=10, subsample=0.9, colsample_bytree=0.9,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        m.fit(X_tr[tr_idx], y[tr_idx])
        oof[val_idx] = m.predict(X_tr[val_idx])
        te_pred += m.predict(X_te) / len(splits)
        fr = rae(y[val_idx], oof[val_idx])
        fold_raes.append(fr)
        print(f"  fold {fold}: RAE={fr:.4f}  n_val={len(val_idx)}")

    r = rae(y, oof); sp, _ = spearmanr(y, oof)
    print(f"\nOOF RAE={r:.4f}  Spearman={sp:.4f}")
    print(f"te_pred mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"Compare to nb239 base: 0.2838")

    np.save(DATA_PROCESSED / "oof_nb311_stable.npy", oof)
    np.save(DATA_PROCESSED / "te_nb311_stable.npy", te_pred)

    # 5-way SLSQP w/ nb239 base
    print("\n=== 5-way SLSQP with nb311 ===")
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd  = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M_oof = np.column_stack([nb224, nb179s, mtd, loso, oof])

    te224 = np.load(DATA_PROCESSED / "te_nb224_pool_plus_2.npy")
    te179s = np.load(DATA_PROCESSED / "te_nb179_stack.npy")
    temtd  = np.load(DATA_PROCESSED / "te_oof_multi_template_delta.npy")
    teloso = np.load(DATA_PROCESSED / "te_oof_delta_loso.npy")
    M_te = np.column_stack([te224, te179s, temtd, teloso, te_pred])

    nms = ['nb224', 'nb179s', 'mtd', 'loso', 'nb311']

    def loss(w): return rae(y, M_oof @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(150):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-10, 'maxiter': 300})
        if best is None or res.fun < best.fun: best = res
    pred_oof = M_oof @ best.x
    pred_te  = M_te  @ best.x
    rb = rae(y, pred_oof); spb, _ = spearmanr(y, pred_oof)
    print(f"\n5-way SLSQP: OOF RAE={rb:.4f}  Spearman={spb:.4f}  "
          f"te_std={pred_te.std():.3f}  te_mean={pred_te.mean():.3f}")
    for nm, w in zip(nms, best.x):
        print(f"  w[{nm}] = {w:.4f}")

    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': pred_te,
    })
    out = SUBMISSIONS / "nb311_stable_cut.csv"
    sub.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
