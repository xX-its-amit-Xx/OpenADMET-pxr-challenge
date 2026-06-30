"""nb1176_gate — honest ensemble gate for ERRγ-pooled sisterNR head vs the DEPLOYED sisterNR head.
control = sn_oof (deployed sisterNR), treatment = sn_errg_oof (sisterNR + 75 ERRγ EC50).
Deploy iff sn_errg < deployed best - 0.001 AND sn_errg < control (4-GBM deployed config).
Mirrors nb1168_gate exactly (same topK / CheMeleon+TabPFN members / clip).
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


def run_config(archs, label, Xtr, ytr, se, n_tr, chem, tab):
    topK = topK_configs(archs)
    arms = {"control": "sn_oof", "sn_errg": "sn_errg_oof"}
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
    delta = means["sn_errg"] - means["control"]
    print(f"[{label}] {[r['arch'] for r in topK]} + CheMeleon + TabPFN + GNN")
    for k in arms: print(f"    {k:9s} {means[k]:.4f} +/- {stds[k]:.4f}")
    print(f"    sn_errg - control = {delta:+.4f}  (- is better)")
    return means, stds, delta


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")

    m5, s5, d5 = run_config(("lgbm", "xgb", "cat", "histgb", "mlp"), "5-arch", Xtr, ytr, se, n_tr, chem, tab)
    m4, s4, d4 = run_config(("lgbm", "xgb", "cat", "histgb"), "4-GBM(deployed)", Xtr, ytr, se, n_tr, chem, tab)

    out = {"5arch": {"means": m5, "std": s5, "sn_errg_minus_control": round(d5, 4)},
           "4gbm_deployed": {"means": m4, "std": s4, "sn_errg_minus_control": round(d4, 4)}}
    json.dump(out, open(f"{P}/nb1176_summary.json", "w"), indent=2)

    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4367
    se4, ctl4 = m4["sn_errg"], m4["control"]
    if d4 < -0.001 and se4 < ctl4 and se4 < prev - 0.001:
        print(f"\nDEPLOY-WORTHY: 4-GBM matched sn_errg-control {d4:+.4f} (<-0.001), sn_errg {se4} < best {prev}. Build deploy next.")
    else:
        print(f"\nno deploy: 4-GBM matched delta {d4:+.4f} (need <-0.001); sn_errg {se4} vs best {prev}; best stays {prev}")


if __name__ == "__main__":
    main()
