"""nb1133 — honestly test whether the nb1132 advanced members (custom GINE GNN + TabNet) HELP the ensemble.

Same leakage-free clean-holdout gate as nb1130 (combined-only for tree members; cached OOF for DL members).
Tries 4 ensemble variants and reports clean-holdout RAE for each:
  base    = topK GBMs + ChemProp D-MPNN              (current deployed = 0.4493)
  +gnn    = base + custom GINE GNN
  +tabnet = base + TabNet
  +both   = base + GINE + TabNet
Equal-weight mean + per-fold y-range clip. No deploy here — pure honest measurement.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nb1126_combinatorial_search import feature_matrix, CACHE
from nb1130_ensemble_check import fit_pred, murcko
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; LOG = "C:/pxr_work/search/results.jsonl"; SD = "C:/pxr_work/search"


def main():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in ("lgbm","xgb","cat","histgb","mlp")]
    valid.sort(key=lambda r: r["ps_rae"])
    bestper = {}
    for r in valid:
        bestper.setdefault(r["arch"], r)
    topK = list(bestper.values())

    d = np.load(CACHE); ytr = d["ytr"]
    gnn_aux = np.load(f"{P}/oof_chemprop_aux.npy")        # ChemProp D-MPNN (strong, already in base)
    gine = np.load(f"{SD}/gnn_oof.npy")                   # custom GINE GNN (nb1132)
    tabn = np.load(f"{SD}/tabnet_oof.npy")                # TabNet (nb1132)
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    scaf = np.array([murcko(s) for s in tr["smiles"]])

    variants = {"base": [], "+gine": [gine], "+tabnet": [tabn], "+both": [gine, tabn]}
    out = {k: [] for k in variants}
    for seed in range(3):
        folds = scaffold_kfold_indices(scaf.tolist(), n_splits=round(len(ytr)/250), seed=100+seed)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix)-250))
        trn = np.array([i for i in range(len(ytr)) if i not in set(ho.tolist())])
        treeps = []
        for c in topK:
            Xtr, _ = feature_matrix(d, "combined")
            treeps.append(fit_pred(c, d, trn, Xtr[ho], ytr, fset="combined"))
        base = treeps + [gnn_aux[ho]]
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        for k, extra in variants.items():
            ps = base + [e[ho] for e in extra]
            ens = np.clip(np.mean(ps, 0), lo, hi)
            out[k].append(rae(ytr[ho], ens))
    print(f"topK GBM archs: {[r['arch'] for r in topK]} + ChemProp D-MPNN base\n")
    for k in variants:
        m, s = np.mean(out[k]), np.std(out[k])
        print(f"  {k:9s} clean-holdout RAE = {m:.4f} +/- {s:.4f}")


if __name__ == "__main__":
    main()
