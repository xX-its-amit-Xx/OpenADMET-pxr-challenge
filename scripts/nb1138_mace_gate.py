"""nb1138 — honest-gate MACE-OFF23 embeddings (Blackwell GPU, molab) as an ensemble member.
Build MACE-LGBM scaffold-CV OOF + 513 preds, then nb1136-style add-member test on the clean never-tuned holdouts.
Decisive cheap screen: if MACE doesn't help the {topK GBMs + ChemProp-GNN + CheMeleon + TabPFN} base, it won't help the
richer deployed ensemble. Reports standalone RAE, corr-with-error, and base vs +mace clean-holdout delta.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nb1126_combinatorial_search import feature_matrix, CACHE
from nb1130_ensemble_check import fit_pred, murcko
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from lightgbm import LGBMRegressor
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"


def topK():
    recs = [json.loads(l) for l in open(f"{SD}/results.jsonl") if l.strip()]
    v = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in ("lgbm", "xgb", "cat", "histgb")]
    v.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in v: bp.setdefault(r["arch"], r)
    return list(bp.values())


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    y = tr["pec50"].to_numpy(); scaf = np.array([murcko(s) for s in tr["smiles"]])
    mtr = np.nan_to_num(np.load(f"{SD}/maceoff_tr.npy").astype(np.float32))

    # MACE-LGBM scaffold-CV OOF
    oof = np.zeros(len(y))
    for trn, val in scaffold_kfold_indices(scaf.tolist(), n_splits=5, seed=42):
        m = LGBMRegressor(n_estimators=400, num_leaves=48, learning_rate=0.05, n_jobs=4, verbose=-1)
        m.fit(mtr[trn], y[trn]); oof[val] = m.predict(mtr[val])
    np.save(f"{SD}/maceoff_oof.npy", oof)
    print(f"MACE-OFF23-LGBM scaffold-CV RAE {rae(y, oof):.4f}  (AIMNet2-scalars won at this axis; CheMeleon 0.577)")

    # corr of MACE-LGBM residual with the ChemProp-GNN residual (orthogonality proxy to the deployed anchor)
    gnn = np.load(f"{P}/oof_chemprop_aux.npy")
    print(f"corr(MACE-OOF, GNN-OOF) = {np.corrcoef(oof, gnn)[0,1]:.3f}  (lower = more orthogonal)")

    # nb1136-style add-member gate on clean holdouts
    d = np.load(CACHE); ytr = d["ytr"]
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    cfgs = topK()
    variants = {"base": [], "+mace": [oof]}
    out = {k: [] for k in variants}
    for seed in range(3):
        folds = scaffold_kfold_indices(scaf.tolist(), n_splits=round(len(ytr)/250), seed=100+seed)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix)-250))
        trn = np.array([i for i in range(len(ytr)) if i not in set(ho.tolist())])
        Xtr, _ = feature_matrix(d, "combined")
        base = [fit_pred(c, d, trn, Xtr[ho], ytr, fset="combined") for c in cfgs] + [gnn[ho], chem[ho], tab[ho]]
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        for k, extra in variants.items():
            ps = base + [e[ho] for e in extra]
            out[k].append(rae(ytr[ho], np.clip(np.mean(ps, 0), lo, hi)))
    bm, cm = np.mean(out["base"]), np.mean(out["+mace"])
    print(f"\nclean-holdout: base {bm:.4f} | +mace {cm:.4f} | delta {cm-bm:+.4f} "
          f"({'HELPS' if cm < bm-0.001 else 'no help'})")
    json.dump({"mace_oof_rae": round(rae(y, oof), 4), "base": round(bm, 4), "+mace": round(cm, 4),
               "delta": round(cm-bm, 4)}, open(f"{P}/nb1138_mace_gate.json", "w"), indent=2)


if __name__ == "__main__":
    main()
