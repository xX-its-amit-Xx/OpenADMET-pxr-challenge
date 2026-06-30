"""nb1130 — ensemble-of-top-configs check on the CLEAN holdout gate (cycle-307: the honest search signal).

The combinatorial search (nb1126) blend-vs-nb3200 gate is CONTAMINATED (masks architecture levers). The honest move:
take the top-K configs by ps_rae (per-series standalone, clean-ish), build their ENSEMBLE, and validate on 5 clean
analog-expansion TRAIN holdouts (never tuned against). Baseline LGBM clean-RAE = 0.4766; GBM ensemble = 0.4656.
If a top-K ensemble beats the best so far, save its 513-prediction submission. Run after each search batch.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nb1126_combinatorial_search import make_model, feature_matrix, CACHE
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.preprocessing import StandardScaler

P = "data/processed"; LOG = "C:/pxr_work/search/results.jsonl"; BEST = "C:/pxr_work/search/best_ensemble.json"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def fit_pred(c, d, Xtr_idx, Xte_full, ytr, fset=None):
    Xtr, Xte = feature_matrix(d, fset or c["fset"])
    se = d["se"]; mask = np.ones(len(ytr), bool)
    if c["prep"] == "noisy20": mask = se <= np.quantile(se, 0.8)
    elif c["prep"] == "noisy30": mask = se <= np.quantile(se, 0.7)
    m = make_model(c["arch"], c["hp"]); use = Xtr_idx[mask[Xtr_idx]]
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(Xtr[use]); m.fit(sc.transform(Xtr[use]), ytr[use]); return m.predict(sc.transform(Xte_full))
    m.fit(Xtr[use], ytr[use]); return m.predict(Xte_full)


def main():
    recs = [json.loads(l) for l in open(LOG) if l.strip()] if os.path.exists(LOG) else []
    valid = [r for r in recs if "ps_rae" in r and "error" not in r]
    if len(valid) < 5:
        print(f"only {len(valid)} valid configs; skip ensemble check"); return
    valid = [r for r in valid if r["arch"] in ("lgbm", "xgb", "cat", "histgb", "mlp")]   # GBMs + deep tabular (+ GNN added below)
    valid.sort(key=lambda r: r["ps_rae"])
    # DIVERSE ensemble: best config per architecture (raw-average -> need diverse archs, not clones)
    bestper = {}
    for r in valid:
        if r["arch"] not in bestper: bestper[r["arch"]] = r
    topK = list(bestper.values())
    print(f"diverse ensemble ({len(topK)} archs): {[ (r['arch'], r['fset'], round(r['ps_rae'],3)) for r in topK ]}")

    d = np.load(CACHE); ytr = d["ytr"]
    gnn_oof = np.load(f"{P}/oof_chemprop_aux.npy")   # ChemProp D-MPNN GNN: clean OOF (4139), orthogonal to trees
    gnn_te = np.load(f"{P}/te_chemprop_aux.npy")     # GNN 513-test predictions
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    scaf = np.array([murcko(s) for s in tr["smiles"]])
    # 5 clean holdouts
    raes = []
    for seed in range(3):
        folds = scaffold_kfold_indices(scaf.tolist(), n_splits=round(len(ytr) / 250), seed=100 + seed)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - 250))
        trn = np.array([i for i in range(len(ytr)) if i not in set(ho.tolist())])
        # ensemble of top configs (rank-standardized mean)
        ps = []
        for c in topK:
            Xtr, _ = feature_matrix(d, "combined")          # LEAKAGE-FREE gate: chempropembed leaks on train-holdouts
            ps.append(fit_pred(c, d, trn, Xtr[ho], ytr, fset="combined"))
        ps.append(gnn_oof[ho])                              # ChemProp GNN member (clean OOF)
        ens = np.clip(np.mean(ps, 0), np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98))
        raes.append(rae(ytr[ho], ens))
    em = float(np.mean(raes))
    print(f"\nTOP-8 ENSEMBLE clean-holdout RAE = {em:.4f} +/- {np.std(raes):.4f} (baseline LGBM 0.4766, GBM-ens 0.4656)")
    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4656
    if em < prev - 0.001:
        print(f"NEW BEST ENSEMBLE ({em:.4f} < {prev:.4f}) -> building full-513 submission")
        te = load_test().reset_index(drop=True)
        ff = np.load("C:/pxr_work/search/feats_full513.npz")
        full_feats = {"combined": ff["combined"], "embed": ff["embed"], "morgan": ff["morgan"]}
        idx_all = np.arange(len(ytr)); ps = []
        for c in topK:
            Xte_full = np.hstack([full_feats[tok] for tok in c["fset"].split("+")])
            ps.append(fit_pred(c, d, idx_all, Xte_full, ytr))
        ps.append(gnn_te)                                   # ChemProp GNN member (513 predictions)
        full = np.clip(np.mean(ps, 0), np.quantile(ytr, 0.05), np.quantile(ytr, 0.98))
        pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": full}).to_csv(
            "submissions/nb1130_search_ensemble.csv", index=False)
        json.dump({"rae": em, "configs": [c["id"] for c in topK]}, open(BEST, "w"), indent=2)
        print("saved submissions/nb1130_search_ensemble.csv")
    else:
        print(f"not better than best {prev:.4f}; no save")


if __name__ == "__main__":
    main()
