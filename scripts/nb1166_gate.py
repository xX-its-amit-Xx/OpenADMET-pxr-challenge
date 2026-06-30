"""nb1166_gate — honest ensemble gate for Octant main-head GNN vs deployed multitask GNN.

Mirrors nb1163_gate: full deployed ensemble (topK GBMs combined-only + CheMeleon + TabPFN + GNN-OOF) on the
SAME ~250 scaffold holdouts. Only the GNN member differs:
  control  = treat_oof (deployed multitask GNN, ext-EC50 aux)
  octant   = octant_oof (deployed GNN + 435 octant main-head rows)
GBM/CheMeleon/TabPFN OOFs identical between arms. Deploy gate: octant < deployed best - 0.001 AND < control.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model, StandardScaler
from src.pxr.eval import rae
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"; N_SEEDS = 3


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in ("lgbm", "xgb", "cat", "histgb", "mlp")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_one(c, Xtr, ytr, use_idx, Xho):
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(Xtr[use_idx]); m.fit(sc.transform(Xtr[use_idx]), ytr[use_idx]); return m.predict(sc.transform(Xho))
    m.fit(Xtr[use_idx], ytr[use_idx]); return m.predict(Xho)


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    topK = topK_configs()
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")

    arms = {"control": "treat_oof", "octant": "octant_oof"}
    res = {k: [] for k in arms}
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8); noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        Xho = Xtr[ho]
        gbm = []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            gbm.append(fit_one(c, Xtr, ytr, use, Xho))
        for arm, fname in arms.items():
            gnn = np.load(f"{MTL}/{fname}_seed{seed}.npy").ravel()
            ens = np.clip(np.mean(gbm + [chem[ho], tab[ho], gnn], 0), lo, hi)
            res[arm].append(rae(ytr[ho], ens))

    means = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    stds = {k: round(float(np.std(v)), 4) for k, v in res.items()}
    print(f"topK {[r['arch'] for r in topK]} + CheMeleon + TabPFN + GNN")
    for k in arms: print(f"  ensemble {k:8s} {means[k]:.4f} +/- {stds[k]:.4f}")
    delta = means["octant"] - means["control"]
    print(f"\n  octant - control = {delta:+.4f}  (- is better)")
    out = {"variants": means, "std": stds, "octant_minus_control": round(delta, 4)}
    json.dump(out, open(f"{P}/nb1166_summary.json", "w"), indent=2)

    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4421
    if means["octant"] < prev - 0.001 and means["octant"] < means["control"] - 0.0005:
        print(f"\nNEW BEST: octant {means['octant']:.4f} < {prev:.4f} AND < control -> DEPLOY")
    else:
        print(f"\nno deploy: octant {means['octant']:.4f} vs best {prev:.4f} / control {means['control']:.4f}")


if __name__ == "__main__":
    main()
