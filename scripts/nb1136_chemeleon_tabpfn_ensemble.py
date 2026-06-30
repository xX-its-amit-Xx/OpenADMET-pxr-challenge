"""nb1136 — honest-gate test: do CheMeleon + TabPFN (the leaderboard winners we lacked) help the ensemble?

Same leakage-free clean-holdout gate as nb1130/nb1133. Base = topK GBMs (combined-only) + ChemProp D-MPNN.
Candidate members (fixed scaffold-CV OOF, like the GNN): CheMeleon-LGBM OOF, TabPFN OOF. Reports each variant;
if a variant beats the deployed 0.4493 by >0.001, builds the full-513 submission (CheMeleon/TabPFN use their own
foundation/in-context preds on the real 513 — no leak there). Run by the cron once nb1135 has cached the OOFs.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nb1126_combinatorial_search import feature_matrix, CACHE
from nb1130_ensemble_check import fit_pred, murcko
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; LOG = f"{SD}/results.jsonl"; BEST = f"{SD}/best_ensemble.json"
OUT = f"{P}/nb1136_chemeleon_tabpfn.json"


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in ("lgbm","xgb","cat","histgb","mlp")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def main():
    if not os.path.exists(f"{SD}/chemeleon_oof.npy"):
        print("waiting on chemeleon_oof.npy"); return
    has_tab = os.path.exists(f"{SD}/tabpfn_oof.npy")
    topK = topK_configs()
    d = np.load(CACHE); ytr = d["ytr"]
    gnn = np.load(f"{P}/oof_chemprop_aux.npy")
    chem = np.load(f"{SD}/chemeleon_oof.npy")
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    scaf = np.array([murcko(s) for s in tr["smiles"]])

    variants = {"base": [], "+chemeleon": [chem]}
    if has_tab:
        tab = np.load(f"{SD}/tabpfn_oof.npy")
        variants["+tabpfn"] = [tab]; variants["+both"] = [chem, tab]
    else:
        print("(TabPFN OOF absent — HF-gated; testing CheMeleon only)")
    out = {k: [] for k in variants}
    for seed in range(3):
        folds = scaffold_kfold_indices(scaf.tolist(), n_splits=round(len(ytr)/250), seed=100+seed)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix)-250))
        trn = np.array([i for i in range(len(ytr)) if i not in set(ho.tolist())])
        Xtr, _ = feature_matrix(d, "combined")
        treeps = [fit_pred(c, d, trn, Xtr[ho], ytr, fset="combined") for c in topK]
        base = treeps + [gnn[ho]]
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        for k, extra in variants.items():
            ps = base + [e[ho] for e in extra]
            out[k].append(rae(ytr[ho], np.clip(np.mean(ps, 0), lo, hi)))
    res = {k: round(float(np.mean(v)), 4) for k, v in out.items()}
    res_std = {k: round(float(np.std(v)), 4) for k, v in out.items()}
    print(f"topK {[r['arch'] for r in topK]} + ChemProp base")
    for k in variants: print(f"  {k:11s} {res[k]:.4f} +/- {res_std[k]:.4f}")
    best_k = min(res, key=res.get)
    json.dump({"variants": res, "std": res_std, "best": best_k}, open(OUT, "w"), indent=2)

    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4493
    if res[best_k] < prev - 0.001 and best_k != "base":
        print(f"NEW BEST via {best_k} ({res[best_k]:.4f} < {prev:.4f}) -> building 513 submission")
        te = load_test().reset_index(drop=True); ff = np.load(f"{SD}/feats_full513.npz")
        full_feats = {"combined": ff["combined"], "embed": ff["embed"], "morgan": ff["morgan"]}
        idx = np.arange(len(ytr)); ps = []
        for c in topK:
            Xte = np.hstack([full_feats[t] for t in c["fset"].split("+")])
            ps.append(fit_pred(c, d, idx, Xte, ytr))
        ps.append(np.load(f"{P}/te_chemprop_aux.npy"))
        if "chemeleon" in best_k: ps.append(np.load(f"{SD}/chemeleon_lgbm_te.npy"))
        if "tabpfn" in best_k or best_k == "+both": ps.append(np.load(f"{SD}/tabpfn_te.npy"))
        if best_k == "+chemeleon": ps = ps[:-1] + [np.load(f"{SD}/chemeleon_lgbm_te.npy")]
        full = np.clip(np.mean(ps, 0), np.quantile(ytr, 0.05), np.quantile(ytr, 0.98))
        pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": full}).to_csv(
            "submissions/nb1136_chemeleon_tabpfn_ensemble.csv", index=False)
        json.dump({"rae": res[best_k], "via": best_k}, open(BEST, "w"), indent=2)
        print("saved submissions/nb1136_chemeleon_tabpfn_ensemble.csv")
    else:
        print(f"no variant beats {prev:.4f} by >0.001 (best {best_k} {res[best_k]:.4f})")


if __name__ == "__main__":
    main()
