"""nb1107 — mechanistic interpretability pack: partial-dependence + gain importance on interpretable features.

User wants mechanistic info baked in. The group-contribution model (nb1098/nb1104) gave the LINEAR SAR; this adds the
NON-LINEAR shape: train an LGBM on interpretable physchem + fragment features and plot PARTIAL DEPENDENCE of pEC50 on
the top drivers (does activity rise-then-plateau with logP? is there a TPSA threshold?). More mechanistic than SHAP on
opaque Morgan bits. Figure -> data/processed/figures/nb1107_partial_dependence.png.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = "data/processed"; FIG = f"{P}/figures"
PHYS = ["MolLogP", "TPSA", "MolWt", "NumAromaticRings", "FractionCSP3", "NumHDonors",
        "NumHAcceptors", "NumRotatableBonds", "RingCount", "NumAromaticHeterocycles"]
FRAGS = [n for n in dir(Fragments) if n.startswith("fr_")]


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None: rows.append([np.nan] * (len(PHYS) + len(FRAGS))); continue
        rows.append([getattr(Descriptors, n)(m) for n in PHYS] + [getattr(Fragments, n)(m) for n in FRAGS])
    X = np.array(rows, np.float32)
    return np.where(np.isfinite(X), X, np.nanmedian(np.where(np.isfinite(X), X, np.nan), 0))


def pdp(model, X, j, grid):
    out = []
    Xc = X.copy()
    for v in grid:
        Xc[:, j] = v; out.append(model.predict(Xc).mean())
    return np.array(out)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    names = PHYS + FRAGS
    X = featurize(tr["smiles"].tolist()); y = tr["pec50"].to_numpy()
    m = lgb.LGBMRegressor(n_estimators=600, num_leaves=48, learning_rate=0.04, subsample=0.8,
                          colsample_bytree=0.8, n_jobs=6, verbose=-1).fit(X, y)
    imp = m.booster_.feature_importance(importance_type="gain")
    top = np.argsort(imp)[::-1][:12]
    print("=== top gain-importance features (non-linear drivers) ===")
    for j in top: print(f"  {names[j]:24s} gain {imp[j]:.0f}")

    drivers = [n for n in ["MolLogP", "TPSA", "NumAromaticRings", "MolWt", "FractionCSP3", "NumHAcceptors"] if n in names]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.ravel(), drivers):
        j = names.index(name)
        lo, hi = np.percentile(X[:, j], [2, 98]); grid = np.linspace(lo, hi, 40)
        ax.plot(grid, pdp(m, X, j, grid), color="#c0392b", lw=2)
        ax.scatter(X[:, j], y, s=4, alpha=0.06, color="gray")
        ax.set_xlabel(name); ax.set_ylabel("partial dependence pEC50"); ax.set_title(name)
    fig.suptitle("PXR activity — partial dependence on physicochemical drivers (mechanistic SAR shape)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{FIG}/nb1107_partial_dependence.png", dpi=115); plt.close()
    json.dump({"top_features": [[names[j], float(imp[j])] for j in top]}, open(f"{P}/nb1107_interpret.json", "w"), indent=2)
    print(f"\nwrote {FIG}/nb1107_partial_dependence.png")


if __name__ == "__main__":
    main()
