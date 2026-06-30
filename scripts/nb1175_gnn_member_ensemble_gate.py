"""nb1174 — honest gate: GNN-member ENSEMBLE (stack winning aux-head OOFs) vs deployed sister-NR.

Reuses nb1168 gate machinery. The GNN member of the deployed ensemble is currently a single
aux-head GNN (sn = octant-main + sisterNR aux, 4-GBM RAE 0.4361). We have, on IDENTICAL holdouts
(ho_idx_seed*), several GNN OOF variants that each carry a confirmed signal:
    sn_oof    = octant main + sisterNR aux  (DEPLOYED best)
    treat_oof = plain PXR + external-EC50 aux (nb1163 WIN, -0.0030)
    octant_oof= octant main only
Test whether averaging diverse winning GNNs as the ensemble's GNN member lowers RAE.
Deploy iff a combo beats deployed sn by < -0.001 on the 4-GBM(deployed) config.
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

# GNN-member definitions: name -> list of OOF basenames to average
COMBOS = {
    "sn (deployed)":      ["sn_oof"],
    "treat (extEC50)":    ["treat_oof"],
    "sn+treat":           ["sn_oof", "treat_oof"],
    "sn+octant":          ["sn_oof", "octant_oof"],
    "sn+treat+octant":    ["sn_oof", "treat_oof", "octant_oof"],
}


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_one(c, Xtr, ytr, use_idx, Xho):
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(Xtr[use_idx]); m.fit(sc.transform(Xtr[use_idx]), ytr[use_idx]); return m.predict(sc.transform(Xho))
    m.fit(Xtr[use_idx], ytr[use_idx]); return m.predict(Xho)


def gnn_member(combo, seed, ho):
    arrs = [np.load(f"{MTL}/{b}_seed{seed}.npy").ravel() for b in combo]
    return np.mean(arrs, 0)


def run_config(archs, label, Xtr, ytr, se, n_tr, chem, tab):
    topK = topK_configs(archs)
    res = {k: [] for k in COMBOS}
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
        for name, combo in COMBOS.items():
            gnn = gnn_member(combo, seed, ho)
            ens = np.clip(np.mean(gbm + [chem[ho], tab[ho], gnn], 0), lo, hi)
            res[name].append(rae(ytr[ho], ens))
    means = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    stds = {k: round(float(np.std(v)), 4) for k, v in res.items()}
    base = means["sn (deployed)"]
    print(f"[{label}] {[r['arch'] for r in topK]} + CheMeleon + TabPFN + GNN-member")
    for k in COMBOS:
        print(f"    {k:18s} {means[k]:.4f} +/- {stds[k]:.4f}   delta_vs_sn {means[k]-base:+.4f}")
    return means, stds


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")

    m5, s5 = run_config(("lgbm", "xgb", "cat", "histgb", "mlp"), "5-arch", Xtr, ytr, se, n_tr, chem, tab)
    m4, s4 = run_config(("lgbm", "xgb", "cat", "histgb"), "4-GBM(deployed)", Xtr, ytr, se, n_tr, chem, tab)

    base4 = m4["sn (deployed)"]
    best_combo = min(COMBOS, key=lambda k: m4[k])
    delta = m4[best_combo] - base4
    out = {"5arch": {"means": m5, "std": s5}, "4gbm_deployed": {"means": m4, "std": s4},
           "best_combo": best_combo, "delta_vs_sn_4gbm": round(delta, 4)}
    json.dump(out, open(f"{P}/nb1175_summary.json", "w"), indent=2)

    if best_combo != "sn (deployed)" and delta < -0.001:
        print(f"\nDEPLOY-WORTHY: '{best_combo}' beats deployed sn by {delta:+.4f} (4-GBM). Build deploy next.")
    else:
        print(f"\nno deploy: best combo '{best_combo}' delta {delta:+.4f} (need <-0.001); deployed sn stays best.")


if __name__ == "__main__":
    main()
