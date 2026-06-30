"""nb1098 — reasoning-based GROUP-CONTRIBUTION pEC50 (user's additive idea) + LLM-pocket-reasoning test.

User: build a deterministic/reasoning function = SUM of contributions of the compound's binding groups; predict how
much each group adds to activity. This is a Free-Wilson additive model on interpretable functional-group features.

Three things, honestly:
1. PREDICTOR: additive (Ridge) model on RDKit fragment counts + key physchem -> honest scaffold-CV RAE on the 253.
   Expectation (cycle-299): a function of chemistry, additive+linear, will LOSE to nb3200 (0.4416). Report truth.
2. DEPLOY GATE: does the group-contribution prediction ADD to nb3200 (corr-with-error / blend)?
3. INTERPRETABILITY (the real value): the fitted per-group coefficients = the contribution table; AND a test of
   LLM-pocket-reasoning — I encode PXR-pharmacology priors (big hydrophobic pocket favors lipophilic/aromatic/halogen,
   disfavors charged/polar/acid) and check whether the DATA-FITTED signs agree with the REASONED signs.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

P = "data/processed"

# key physchem (interpretable "global" contributions)
PHYS = ["MolLogP", "TPSA", "MolWt", "NumAromaticRings", "FractionCSP3", "NumHDonors",
        "NumHAcceptors", "NumRotatableBonds", "RingCount", "NumAromaticHeterocycles"]
# fragment counts (rdkit Fragments.fr_*) = functional-group indicators
FRAGS = [n for n in dir(Fragments) if n.startswith("fr_")]

# --- LLM REASONING PRIOR: expected sign of each feature for PXR activation ---
# PXR LBD = ~1300 A^3 hydrophobic promiscuous pocket (Watkins 2001); lipophilicity is the dominant driver;
# charged/very-polar groups disfavored; aromatic/halogen/lipophilic favored.
REASON_PRIOR = {
    "MolLogP": +1, "TPSA": -1, "NumAromaticRings": +1, "FractionCSP3": -1, "NumHDonors": -1,
    "MolWt": +1, "NumAromaticHeterocycles": +1,
    "fr_benzene": +1, "fr_halogen": +1, "fr_ether": +1, "fr_aryl_methyl": +1, "fr_ketone": +1,
    "fr_COO": -1, "fr_COO2": -1, "fr_NH2": -1, "fr_NH1": -1, "fr_quatN": -1, "fr_sulfonamd": -1,
    "fr_amide": -1, "fr_guanido": -1, "fr_phos_acid": -1, "fr_nitro": -1, "fr_Al_OH": -1,
}


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            rows.append([np.nan] * (len(PHYS) + len(FRAGS))); continue
        phys = [getattr(Descriptors, n)(m) for n in PHYS]
        frag = [getattr(Fragments, n)(m) for n in FRAGS]
        rows.append(phys + frag)
    X = np.array(rows, np.float32)
    return np.where(np.isfinite(X), X, np.nanmedian(np.where(np.isfinite(X), X, np.nan), 0))


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = y - anchor
    names = PHYS + FRAGS
    smte = te["smiles"].to_numpy()[unb].tolist()

    Xtr = featurize(tr["smiles"].tolist()); ytr = tr["pec50"].to_numpy()
    Xte = featurize(smte)
    sc = StandardScaler().fit(Xtr); Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)

    # 1. additive predictor: honest scaffold-CV on TRAIN -> then fit full, predict 253
    scaf = tr["smiles"].map(murcko).tolist()
    folds = scaffold_kfold_indices(scaf, n_splits=5, seed=42)
    oof = np.zeros(len(tr))
    for trn, val in folds:
        m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Xtr_s[trn], ytr[trn]); oof[val] = m.predict(Xtr_s[val])
    cv_rae = rae(ytr, oof)
    model = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Xtr_s, ytr)
    pred = model.predict(Xte_s)
    test_rae = rae(y, pred)
    print(f"GROUP-CONTRIBUTION (additive Ridge): train scaffold-CV RAE {cv_rae:.4f} | 253 RAE {test_rae:.4f} "
          f"(nb3200 anchor 0.4416)", flush=True)

    # 2. deploy gate
    c = np.corrcoef(pred, err)[0, 1]
    bl = rae(y, anchor)
    for w in np.linspace(0, 1, 41):
        bl = min(bl, rae(y, (1 - w) * anchor + w * pred))
    print(f"DEPLOY GATE: corr(group-pred, nb3200 error) = {c:+.3f} | best blend {bl:.4f} (delta {bl-rae(y,anchor):+.4f})")

    # 3. contribution table (coef on standardized features = per-SD contribution) + reasoning-prior agreement
    coef = model.coef_
    contrib = pd.DataFrame({"group": names, "coef_perSD": coef}).sort_values("coef_perSD")
    nz = contrib[contrib["coef_perSD"].abs() > 1e-3]
    print(f"\n=== TOP ACTIVITY-INCREASING GROUPS ===")
    print(contrib.tail(10).iloc[::-1].to_string(index=False))
    print(f"\n=== TOP ACTIVITY-DECREASING GROUPS ===")
    print(contrib.head(10).to_string(index=False))

    # LLM reasoning-prior test
    agree = tot = 0; rows = []
    for g, prior in REASON_PRIOR.items():
        if g in names:
            data_sign = np.sign(coef[names.index(g)])
            ok = (data_sign == prior)
            agree += int(ok); tot += 1
            rows.append((g, prior, float(coef[names.index(g)]), "OK" if ok else "MISS"))
    print(f"\n=== LLM-POCKET-REASONING vs DATA (sign agreement {agree}/{tot} = {agree/tot:.0%}) ===")
    for g, pr, cf, st in sorted(rows, key=lambda r: -abs(r[2])):
        print(f"  {g:20s} reasoned {'+' if pr>0 else '-'}  data {cf:+.3f}  [{st}]")

    json.dump(dict(cv_rae=float(cv_rae), test_rae=float(test_rae), corr_err=float(c), blend=float(bl),
                   reason_agree=agree, reason_total=tot,
                   top_pos=contrib.tail(8)[["group", "coef_perSD"]].values.tolist(),
                   top_neg=contrib.head(8)[["group", "coef_perSD"]].values.tolist()),
              open(f"{P}/nb1098_group_contribution.json", "w"), indent=2)
    contrib.to_csv(f"{P}/nb1098_group_contributions.csv", index=False)
    print(f"\nwrote {P}/nb1098_group_contributions.csv")


if __name__ == "__main__":
    main()
