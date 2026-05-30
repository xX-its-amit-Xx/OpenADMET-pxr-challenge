"""nb250 -- H-bond density correction to 239.

Empirical finding: 239's worst predictions are OVERESTIMATIONS of pec50 for
compounds with high HBA+HBD (avg HBA 5.05 worst vs 4.04 typical). These
'drug-like' molecules don't bind PXR (which prefers hydrophobic).

Apply a correction: for high-HBA+HBD test compounds, pull prediction DOWN.

This is a STRUCTURED bias correction, derived from train OOF analysis but
applied to test based on chemistry alone. Should generalize.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Lipinski
from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def hbond_features(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return [0, 0, 0]
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    n_heavy = max(mol.GetNumHeavyAtoms(), 1)
    return [hba, hbd, (hba + hbd) / n_heavy]


def main():
    print("=== nb250: H-bond density correction ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].tolist()
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb239_te  = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")

    print("Computing H-bond features...")
    feats_tr = np.array([hbond_features(s) for s in smiles_tr])
    feats_te = np.array([hbond_features(s) for s in smiles_te])
    # 0=HBA, 1=HBD, 2=hbond_density

    # On train OOF, fit residual correction
    residuals = y_tr - nb239_oof
    hba = feats_tr[:, 0]; hbd = feats_tr[:, 1]; dens = feats_tr[:, 2]
    hba_total = hba + hbd

    # Look at residual vs hba_total
    from scipy.stats import pearsonr
    r_hba, _ = pearsonr(hba_total, residuals)
    r_dens, _ = pearsonr(dens, residuals)
    print(f"corr(residual, HBA+HBD) = {r_hba:.4f}")
    print(f"corr(residual, density)  = {r_dens:.4f}")
    # Negative = high H-bond compounds have LOWER actual pec50 than predicted (OVER-prediction)

    # Fit linear correction: residual = a + b * hba_total
    coefs = np.polyfit(hba_total, residuals, 1)
    print(f"Linear correction: residual = {coefs[1]:.4f} + {coefs[0]:.4f} * (HBA+HBD)")

    # Apply correction
    correction_oof = coefs[1] + coefs[0] * (feats_tr[:, 0] + feats_tr[:, 1])
    correction_te = coefs[1] + coefs[0] * (feats_te[:, 0] + feats_te[:, 1])
    print(f"\nCorrection on train: min={correction_oof.min():.3f}, max={correction_oof.max():.3f}, mean={correction_oof.mean():.3f}")
    print(f"Correction on test:  min={correction_te.min():.3f}, max={correction_te.max():.3f}, mean={correction_te.mean():.3f}")

    # Apply at various strengths
    print("\n=== OOF impact ===")
    r_base = rae(y_tr, nb239_oof)
    print(f"nb239 baseline OOF: {r_base:.4f}")
    for strength in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]:
        corrected_oof = nb239_oof + strength * correction_oof
        r = rae(y_tr, corrected_oof)
        sign = " ***" if r < r_base else ""
        print(f"  strength={strength}: {r:.4f}{sign}")

    # Find optimal strength
    best_s, best_r = 0, r_base
    for s in np.linspace(0, 2, 41):
        r = rae(y_tr, nb239_oof + s * correction_oof)
        if r < best_r:
            best_r, best_s = r, s
    print(f"\nBest strength: {best_s:.3f} -> OOF RAE {best_r:.4f}")

    if best_s > 0:
        # Apply to test
        te_corrected = nb239_te + best_s * correction_te
        print(f"\nFinal test: mean={te_corrected.mean():.3f}, std={te_corrected.std():.3f}")
        sub = pd.DataFrame({"SMILES": [sm[n] for n in te_names], "Molecule Name": te_names, "pEC50": te_corrected})
        sub.to_csv(SUBMISSIONS / f"250_hbond_corr_s{int(best_s*100):03d}.csv", index=False)
        print(f"Saved 250_hbond_corr_s{int(best_s*100):03d}.csv")
        np.save(DATA_PROCESSED / "oof_nb250_hbond_corr.npy", nb239_oof + best_s * correction_oof)
        np.save(DATA_PROCESSED / "te_nb250_hbond_corr.npy", te_corrected)

    # Also try with hbond_density (per-atom normalized)
    print("\n=== Same with density-based correction ===")
    coefs_dens = np.polyfit(dens, residuals, 1)
    print(f"correction = {coefs_dens[1]:.4f} + {coefs_dens[0]:.4f} * density")
    corr_oof_dens = coefs_dens[1] + coefs_dens[0] * feats_tr[:, 2]
    corr_te_dens = coefs_dens[1] + coefs_dens[0] * feats_te[:, 2]
    for s in [0.3, 0.5, 0.7, 1.0]:
        r = rae(y_tr, nb239_oof + s * corr_oof_dens)
        sign = " ***" if r < r_base else ""
        print(f"  density-corr strength={s}: {r:.4f}{sign}")


if __name__ == "__main__":
    main()
