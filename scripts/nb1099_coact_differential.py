"""nb1099 — coactivator differential cofold: deploy-gate test (run AFTER the ternary cofold completes).

Ternary cofold (PXR + SRC-1 peptide + ligand) gives three z-interface blocks per test ligand:
  pxr_pep  = PXR x SRC-1 peptide AF-2 interface  -> THE activation-specific signal (self-contained; no diff needed)
  lig_pxr  = ligand x PXR in the coactivator-present context
  lig_pep  = ligand x SRC-1 peptide
Plus the differential  d_ligpxr = lig_pxr(ternary) - test_richz(binary)  = how the coactivator presence changes the
ligand-PXR interaction (activation coupling; caveat: cross-pipeline pooling alignment).

Deploy gate (the only metric that matters, per cycle-299): does the block ADD to nb3200? 30-seed scaffold-CV
residual-LGBM predicts (y - nb3200) from the block; judge mean corr(residual_pred, nb3200 error) and best blend delta.
Stable corr>0 AND blend-delta<~-0.005 over 30 seeds = real activation signal nb3200's 2D base can't see.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

P = "data/processed"; MO = "C:/pxr_struct/boltz/modal"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def resid_oof(X, resid, scaf, seed):
    oof = np.zeros(len(resid))
    for trn, val in scaffold_kfold_indices(scaf, n_splits=5, seed=seed):
        m = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=seed)
        m.fit(X[trn], resid[trn]); oof[val] = m.predict(X[val])
    return oof


def main():
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); resid = y - anchor
    te = load_test(); scaf = [murcko(s) for s in te["smiles"].to_numpy()[unb]]

    lig_pxr = np.load(f"{P}/coact_ligpxr.npy")[unb]
    pxr_pep = np.load(f"{P}/coact_pxrpep.npy")[unb]
    lig_pep = np.load(f"{P}/coact_ligpep.npy")[unb]
    bin_richz = np.load(f"{MO}/test_richz.npy")[unb]
    d_ligpxr = lig_pxr - bin_richz

    blocks = {"pxr_pep(AF2)": pxr_pep, "lig_pxr(coact)": lig_pxr, "lig_pep": lig_pep,
              "d_ligpxr(diff)": d_ligpxr, "all_concat": np.hstack([pxr_pep, lig_pxr, lig_pep])}
    print(f"nb3200 anchor RAE {rae(y, anchor):.4f} | n={len(unb)}\n")
    print(f"{'block':18s} {'corr(resid_pred,err)':>22s} {'blend_delta':>14s} {'frac_seeds_neg':>16s}")
    out = {}
    for name, B in blocks.items():
        B = np.where(np.isfinite(B), B, 0.0)
        B = StandardScaler().fit_transform(B)
        corrs, deltas = [], []
        for seed in range(1200, 1230):
            oof = resid_oof(B, resid, scaf, seed)
            corrs.append(np.corrcoef(oof, resid)[0, 1] if oof.std() > 1e-9 else 0.0)
            best = rae(y, anchor)
            for w in np.linspace(0, 1.5, 31):
                best = min(best, rae(y, anchor + w * oof))
            deltas.append(best - rae(y, anchor))
        c, d = float(np.mean(corrs)), float(np.mean(deltas))
        fn = float(np.mean(np.array(deltas) < -1e-6))
        out[name] = dict(corr=c, blend_delta=d, frac_seeds_improved=fn)
        print(f"{name:18s} {c:>+22.3f} {d:>+14.4f} {fn:>16.2f}")
    json.dump(out, open(f"{P}/nb1099_coact_deploy.json", "w"), indent=2)
    print("\nGATE: real activation signal needs stable corr>0 AND blend_delta<~-0.005 across the 30 seeds.")


if __name__ == "__main__":
    main()
