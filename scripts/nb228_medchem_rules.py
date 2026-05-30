"""nb228 -- Medicinal chemistry rule engine: SMARTS pharmacophore features
specifically curated for PXR ligand binding.

PXR's binding pocket (~1200 Å³) is unusually large, flexible, and hydrophobic.
Key residues from co-crystal structures:
  - Hydrophobic shelf: Phe420, Phe288, Trp299, Met243, Leu239, Leu411, Leu209
  - Polar anchors:    His407 (H-donor), Gln285, Ser247, Arg410
  - Mouth residues:   Phe281, Leu411

A medicinal chemist looks at a molecule and asks:
  1. Does it have a lipophilic core that fills the cavity? (multi-ring system, fsp3<0.4)
  2. Does it have one or two H-bond anchors? (carbonyl, sulfonamide, OH)
  3. Are the anchors positioned to reach His407 / Arg410?
  4. Does it have a flexible linker?
  5. Are there aromatic stacking opportunities with Phe420/288?

We encode 50+ SMARTS rules capturing these features, compute counts per
compound, then learn weights via LASSO regression on the training set.
This is essentially Free-Wilson analysis with explicit medchem-defined rules.

Crucially, these features are SMARTS-pattern counts that depend on EXACT
substructure presence — not derived from Morgan fingerprints. They capture
medchem-level chemical concepts (e.g. "biaryl with sulfonyl meta to OH")
that ECFP4 doesn't isolate.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
import lightgbm as lgb
from scipy.stats import spearmanr

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

COLLAPSE_THRESH = 0.58

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)

# ── Curated PXR-relevant pharmacophore SMARTS rules ──
# Format: (rule_name, SMARTS, intent_string)
MEDCHEM_RULES = [
    # Lipophilic scaffolds (PXR loves multi-ring lipophiles)
    ("steroid_core",          "C1CCC2C(C1)CCC1C2CCC2(C)C1CCC2",     "tetracyclic steroid backbone (PXR loves these)"),
    ("biphenyl",              "c1ccc(-c2ccccc2)cc1",                 "biphenyl — bulky lipophile"),
    ("triphenylmethane",      "C(c1ccccc1)(c2ccccc2)c3ccccc3",       "triphenylmethane — Trp299 stacking"),
    ("naphthalene",           "c1ccc2ccccc2c1",                       "naphthalene — bulky aromatic"),
    ("indole",                "c1ccc2[nH]ccc2c1",                     "indole (NH H-donor + aromatic)"),
    ("benzimidazole",         "c1ccc2nc[nH]c2c1",                     "benzimidazole — H-donor in aromatic"),
    ("quinoline",             "c1ccc2ncccc2c1",                       "quinoline — aromatic + pyridine N"),
    ("benzofuran",            "c1ccc2occc2c1",                        "benzofuran"),
    ("benzothiophene",        "c1ccc2sccc2c1",                        "benzothiophene"),
    ("dihydropyridine",       "C1=CCC=CN1",                           "dihydropyridine"),

    # Polar anchors (His407/Arg410 H-bond targets)
    ("sulfonamide_aryl",      "c-S(=O)(=O)-N",                        "aryl sulfonamide (His407 anchor)"),
    ("aryl_sulfone",          "c-S(=O)(=O)-[#6]",                     "aryl sulfone"),
    ("aryl_amide_NH",         "c-C(=O)-[NH]-[#6]",                    "aryl amide with NH (H-donor)"),
    ("aryl_amide_NR2",        "c-C(=O)-N([#6])-[#6]",                 "aryl tertiary amide"),
    ("phenol",                "c-[OH]",                               "phenolic OH (H-donor/acceptor)"),
    ("aliphatic_OH",          "[C;!c]-[OH]",                          "aliphatic OH"),
    ("trifluoromethyl",       "C(F)(F)F",                             "CF3 (metabolism-resistant + lipophilic)"),
    ("para_CF3_aryl",         "c1ccc(C(F)(F)F)cc1",                   "para-CF3-aryl (T0901317-like)"),
    ("urea",                  "[NH]C(=O)[NH]",                        "urea — bidentate H-bond donor"),
    ("carbamate",             "[NH]C(=O)O",                           "carbamate"),

    # Hetero anchors
    ("morpholine",            "C1COCCN1",                             "morpholine (basic + acceptor)"),
    ("piperidine",            "C1CCNCC1",                             "piperidine"),
    ("piperazine",            "C1CNCCN1",                             "piperazine"),
    ("pyridine",              "c1ccncc1",                             "pyridine ring"),
    ("pyrimidine",            "c1cncnc1",                             "pyrimidine"),
    ("thiazole",              "c1scnc1",                              "thiazole"),
    ("imidazole",             "c1[nH]cnc1",                           "imidazole — His mimetic"),
    ("oxazole",               "c1ocnc1",                              "oxazole"),
    ("isoxazole",             "c1oncc1",                              "isoxazole"),

    # PXR-specific motifs from literature
    ("rifampicin_ansa",       "C1=CC=CC(=O)N",                        "ansa lactam mouth feature"),
    ("nitrile_aryl",          "c-C#N",                                "aryl nitrile (PCN-like)"),
    ("methylenedioxy",        "O1COC2=CC=CC=C12",                     "methylenedioxyphenyl (rifaximin-like)"),

    # Negative motifs (often hurt PXR)
    ("carboxylic_acid",       "C(=O)[OH]",                            "carboxylic acid (rarely tolerated)"),
    ("phosphonic",            "P(=O)([OH])[OH]",                      "phosphonic acid"),
    ("primary_amine",         "[NH2]",                                "primary amine (often deprotonated, polar)"),
    ("quaternary_N",          "[N+](=O)[O-]",                         "nitro / quaternary N"),

    # Connectivity / flexibility
    ("rotatable_bond",        "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]",       "rotatable bond"),

    # Halogens
    ("aryl_Cl",               "c-[Cl]",                               "aryl chloride"),
    ("aryl_F",                "c-[F]",                                "aryl fluoride"),
    ("aryl_Br",               "c-[Br]",                               "aryl bromide"),
    ("aliphatic_F",           "[C;!c]-F",                             "aliphatic F"),

    # Specific PXR co-crystal motifs
    ("aryl_dialkylamide",     "c-C(=O)-N([CH3])([CH3])",              "aryl N,N-dimethylamide"),
    ("phosphonate_ester",     "P(=O)(O[#6])(O[#6])",                  "phosphonate ester (SR12813)"),
    ("benzyl_ether",          "c-C-O-[#6]",                           "benzyl ether"),
    ("aryl_ketone",           "c-C(=O)-[#6]",                         "aryl ketone"),
    ("aryl_aldehyde",         "c-C=O",                                "aryl aldehyde"),

    # Pharmacophore "shape" rules — count fragments meeting criteria
    ("aromatic_ring_count",   "c1ccccc1",                             "any benzene-like ring"),
    ("aliphatic_ring",        "[C;R][C;R][C;R][C;R][C;R][C;R]",       "saturated 6-ring"),
]


def safe_smarts(smarts):
    try:
        return Chem.MolFromSmarts(smarts)
    except Exception:
        return None


def count_rule_matches(mol, smarts_mol):
    if mol is None or smarts_mol is None:
        return 0
    try:
        return len(mol.GetSubstructMatches(smarts_mol))
    except Exception:
        return 0


def compute_medchem_features(smiles_list):
    """For each compound, count occurrences of every SMARTS rule. Adds physchem too."""
    # Pre-parse SMARTS once
    smarts_mols = [(name, safe_smarts(smarts), intent) for name, smarts, intent in MEDCHEM_RULES]

    rule_names = [name for name, _, _ in MEDCHEM_RULES]
    n = len(smiles_list)
    rule_feats = np.zeros((n, len(rule_names)), dtype=np.float32)
    phys = np.zeros((n, 12), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, (_, smarts_mol, _) in enumerate(smarts_mols):
            rule_feats[i, j] = count_rule_matches(mol, smarts_mol)
        phys[i] = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.RingCount(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.MolMR(mol),
        ]
    phys_names = ["MW","LogP","TPSA","HBD","HBA","RotB","NArRing",
                  "NAlRing","HeavyAt","Rings","fsp3","MolMR"]
    return rule_feats, rule_names, phys, phys_names


def main():
    print("=== nb228: Medchem rule engine ===\n")

    tr = load_train()
    te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    base_rae = rae(y_tr, oof_base)
    print(f"Base nb197: OOF RAE={base_rae:.4f}\n")

    # Compute medchem features
    print("[A] Computing medchem rule features...")
    R_tr, rule_names, P_tr, phys_names = compute_medchem_features(smiles_tr)
    R_te, _,           P_te, _          = compute_medchem_features(smiles_te)
    print(f"  Rule features: {R_tr.shape[1]}  Phys features: {P_tr.shape[1]}")
    print(f"  Train shape: {R_tr.shape}  Test shape: {R_te.shape}")

    # Per-rule correlation
    print("\n[B] Rule correlations with PXR pEC50:")
    feature_corrs = []
    for j, name in enumerate(rule_names):
        if R_tr[:, j].std() > 0:
            rho, pval = spearmanr(R_tr[:, j], y_tr)
            feature_corrs.append((name, rho, pval, R_tr[:, j].sum()))
    feature_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, rho, pval, count in feature_corrs[:20]:
        print(f"  {name:25s} ρ={rho:+.3f}  p={pval:.2e}  n_hits={int(count)}")

    print("\n  Physchem correlations:")
    for j, name in enumerate(phys_names):
        if P_tr[:, j].std() > 0:
            rho, _ = spearmanr(P_tr[:, j], y_tr)
            print(f"  {name:12s}: ρ={rho:+.3f}")

    # Assemble features
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)
    X_tr_aug = np.hstack([X_tr_base, R_tr, P_tr])
    X_te_aug = np.hstack([X_te_base, R_te, P_te])
    print(f"\n[C] Augmented features: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    # Scaffold CV
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    results = {}
    for name, X_tr_use, X_te_use in [
        ("base_only",   X_tr_base, X_te_base),
        ("medchem_aug", X_tr_aug,  X_te_aug),
    ]:
        oof = np.zeros(len(y_tr))
        te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(X_tr_use[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr_use[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr_use[va_idx])
            te_preds.append(m.predict(X_te_use))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {name:14s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  [{flag}]")
        results[name] = (oof, te_pred, r, ratio)

    # Always save medchem features as ensemble candidate
    oof_aug, te_aug, r_aug, ratio_aug = results["medchem_aug"]
    np.save(DATA_PROCESSED / "oof_nb228_medchem.npy", oof_aug)
    np.save(DATA_PROCESSED / "te_nb228_medchem.npy",  te_aug)
    print(f"\n  [always-save] oof_nb228_medchem.npy + te_nb228_medchem.npy (OOF={r_aug:.4f})")

    # Blend with nb197
    print("\n[D] Blend with nb197:")
    best_blend, best_r_bl = None, 999
    for w in np.arange(0.05, 0.65, 0.05):
        oof_bl = (1-w)*oof_base + w*oof_aug
        te_bl  = (1-w)*te_base  + w*te_aug
        r_bl   = rae(y_tr, oof_bl)
        ratio_bl = te_bl.std() / oof_bl.std()
        flag = "PASS" if ratio_bl >= COLLAPSE_THRESH else "FAIL"
        print(f"  w={w:.2f}: OOF={r_bl:.4f}  ratio={ratio_bl:.3f}  [{flag}]")
        if ratio_bl >= COLLAPSE_THRESH and r_bl < best_r_bl:
            best_r_bl = r_bl; best_blend = (w, oof_bl, te_bl, ratio_bl)

    saved = []
    if best_blend and best_r_bl < base_rae:
        w_b, oof_b, te_b, ratio_b = best_blend
        name = f"228_blend_w{int(w_b*100):02d}"
        np.save(DATA_PROCESSED / f"oof_{name}.npy", oof_b)
        np.save(DATA_PROCESSED / f"te_{name}.npy", te_b)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_b})
        sub.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
        saved.append(f"{name} OOF={best_r_bl:.4f}")

    print(f"\n=== Saved: {saved or ['nb228_medchem.npy as ensemble candidate']}")


if __name__ == "__main__":
    main()
