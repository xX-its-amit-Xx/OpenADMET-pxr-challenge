"""nb991 — feature importance + ablation studies.
(1) Pipeline-stage ablation: chemprop_aux anchor -> +K-feature residual+quantile (nb3090) -> +clip (nb3200).
(2) Feature-family ablation on the chemprop_aux residual model (drop Morgan / drop RDKit-desc).
(3) Top feature importances (LGBM gain), named where possible.
Figures -> C:/pxr_struct/diag/.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = "data/processed"; OUT = "C:/pxr_struct/diag"; os.makedirs(OUT, exist_ok=True)
# combined = Morgan(2048) + RDKit-desc(217)
N_MORGAN = 2048
RDKIT_NAMES = [n for n, _ in Descriptors._descList]


def main():
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist()
    anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb]
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in smiles]
    folds = scaffold_kfold_indices(scaf, n_splits=5, seed=42)

    # ---- (1) pipeline-stage ablation ----
    stages = {"chemprop_aux\nanchor": rae(y, anchor)}
    if os.path.exists(f"{D}/nb3090_pred_oof.npy"):
        stages["+ K-feat residual\n+ quantile blend\n(nb3090)"] = rae(y, np.load(f"{D}/nb3090_pred_oof.npy"))
    stages["+ range-clip\n(nb3200)"] = rae(y, np.load(f"{D}/nb3200_pred_oof.npy"))
    print("PIPELINE ABLATION:", {k.replace(chr(10), ' '): round(v, 4) for k, v in stages.items()})

    # ---- (2) feature-family ablation on the chemprop_aux residual ----
    X = impute(combined(smiles)).astype(np.float32)
    resid = y - anchor
    Xm = X[:, :N_MORGAN]; Xr = X[:, N_MORGAN:]

    def cv_rae(Xuse):
        pred = anchor.copy()
        gains = np.zeros(Xuse.shape[1])
        for tri, vai in folds:
            mdl = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1, importance_type="gain")
            mdl.fit(Xuse[tri], resid[tri]); pred[vai] = anchor[vai] + mdl.predict(Xuse[vai])
            gains += mdl.feature_importances_
        return rae(y, pred), gains

    r_all, gains = cv_rae(X)
    r_morgan, _ = cv_rae(Xm)
    r_rdkit, _ = cv_rae(Xr)
    fam = {"combined (Morgan+RDKit)": r_all, "Morgan only": r_morgan, "RDKit-desc only": r_rdkit, "anchor only (no resid)": rae(y, anchor)}
    print("FEATURE-FAMILY (residual model):", {k: round(v, 4) for k, v in fam.items()})

    # ---- (3) top feature importances (named) ----
    names = [f"Morgan_{i}" for i in range(N_MORGAN)] + (RDKIT_NAMES if len(RDKIT_NAMES) == X.shape[1] - N_MORGAN else [f"RDKit_{i}" for i in range(X.shape[1] - N_MORGAN)])
    top = np.argsort(-gains)[:20]
    top_named = [(names[i], float(gains[i])) for i in top]
    print("TOP FEATURES:", [n for n, _ in top_named[:10]])

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    ax = axes[0]
    ks = list(stages.keys()); vs = list(stages.values())
    ax.bar(range(len(ks)), vs, color=["#888", "#1f77b4", "#2ca02c"][:len(ks)])
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, fontsize=9)
    for i, v in enumerate(vs): ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_ylabel("RAE on 253"); ax.set_title("Pipeline ablation\n(what each stage wins)"); ax.set_ylim(0.4, 0.66)

    ax = axes[1]
    fk = list(fam.keys()); fv = list(fam.values())
    ax.barh(range(len(fk)), fv, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#888"])
    ax.set_yticks(range(len(fk))); ax.set_yticklabels(fk, fontsize=9)
    for i, v in enumerate(fv): ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("RAE on 253"); ax.set_title("Feature-family ablation\n(residual model)"); ax.invert_yaxis()

    ax = axes[2]
    tn = top_named[:15][::-1]
    ax.barh(range(len(tn)), [g for _, g in tn], color="#9467bd")
    ax.set_yticks(range(len(tn))); ax.set_yticklabels([n[:24] for n, _ in tn], fontsize=8)
    ax.set_xlabel("LGBM gain"); ax.set_title("Top-15 feature importances\n(chemprop_aux residual model)")
    plt.tight_layout(); plt.savefig(f"{OUT}/nb991_importance_ablation.png", dpi=140); plt.close()

    json.dump({"pipeline_ablation": {k.replace(chr(10), ' '): round(v, 4) for k, v in stages.items()},
               "family_ablation": {k: round(v, 4) for k, v in fam.items()},
               "top20_features": [(n, round(g, 1)) for n, g in top_named]},
              open(f"{OUT}/nb991_summary.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb991_importance_ablation.png + summary.json")


if __name__ == "__main__":
    main()
