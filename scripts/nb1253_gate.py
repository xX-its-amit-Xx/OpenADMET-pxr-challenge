"""nb1253_gate — Honest gate for test-adjacent oracle aux head (tadj) vs deployed sn GNN.

control   = sn_oof  (current deployed 3-head GNN: PXR-main + ext-EC50 + sisterNR)
treatment = tadj_oof (4-head GNN: above + oracle-labeled test-adjacent compounds)

Same ~250 scaffold holdouts, 3 seeds. Gate: ensemble swap test (GBM x4 + CheMeleon + TabPFN + GNN).
Deploy if: tadj_mean < sn_mean AND tadj_mean < best_rae - 0.001
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model, StandardScaler
from src.pxr.eval import rae
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P   = "data/processed"
SD  = "C:/pxr_work/search"
MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"
LOG  = f"{SD}/results.jsonl"
N_SEEDS = 3


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"])
    bp = {}
    for r in valid:
        bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_one(c, Xtr, ytr, use_idx, Xho):
    m = make_model(c["arch"], c["hp"])
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(Xtr[use_idx])
        m.fit(sc.transform(Xtr[use_idx]), ytr[use_idx])
        return m.predict(sc.transform(Xho))
    m.fit(Xtr[use_idx], ytr[use_idx])
    return m.predict(Xho)


def main():
    d   = np.load(CACHE)
    ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy")
    tab  = np.load(f"{SD}/tabpfn_oof.npy")

    archs = ("xgb", "lgbm", "histgb", "cat")
    topK  = topK_configs(archs)
    print(f"GBM configs: {[r['arch'] for r in topK]}", flush=True)

    arms = {"control_sn": "sn_oof", "tadj": "tadj_oof"}
    res  = {k: [] for k in arms}

    for seed in range(N_SEEDS):
        ho  = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8)
        noisy30 = se <= np.quantile(se, 0.7)
        lo = np.quantile(ytr[trn], 0.05)
        hi = np.quantile(ytr[trn], 0.98)
        Xho = Xtr[ho]

        gbm_preds = []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20":
                use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30":
                use = trn[noisy30[trn]]
            gbm_preds.append(fit_one(c, Xtr, ytr, use, Xho))

        for arm, fname in arms.items():
            fpath = f"{MTL}/{fname}_seed{seed}.npy"
            if not os.path.exists(fpath):
                print(f"  [seed {seed}] {arm} OOF not found: {fpath}", flush=True)
                continue
            gnn  = np.load(fpath).ravel()
            ens  = np.clip(np.mean(gbm_preds + [chem[ho], tab[ho], gnn], 0), lo, hi)
            r    = rae(ytr[ho], ens)
            res[arm].append(r)
            print(f"  [seed {seed}] {arm}: RAE={r:.4f}", flush=True)

    print("\n=== GATE RESULTS ===", flush=True)
    means = {k: round(float(np.mean(v)), 4) if v else None for k, v in res.items()}
    for k, m in means.items():
        print(f"  {k:15s}: {m}", flush=True)

    if means["control_sn"] and means["tadj"]:
        delta = means["tadj"] - means["control_sn"]
        print(f"  delta (tadj - sn): {delta:+.4f}  (- is better)", flush=True)

        with open(BEST) as f:
            best = json.load(f)
        best_rae = best["rae"]
        print(f"  current best RAE: {best_rae}", flush=True)
        gate = delta < -0.001 and means["tadj"] < best_rae - 0.001
        print(f"  DEPLOY: {gate}", flush=True)
        return delta, gate
    else:
        print("  Incomplete results (some seeds missing)", flush=True)
        return None, False


if __name__ == "__main__":
    main()
