"""nb1174_gate — honest ADD-MEMBER gate for TabICL on the DEPLOYED nb1168 ensemble.
control = deployed members (topK GBMs + CheMeleon + TabPFN + sisterNR-GNN).
treatment = control + TabICL member (equal-weight clipped mean).
Deploy iff treatment < deployed best - 0.001 AND treatment < control, on the matched 4-GBM deployed config.
Same ~250 scaffold holdouts (ho_idx_seed{0,1,2}) + sn_oof deployed GNN OOFs as nb1173_gate. NEVER the 253.
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


def run_config(archs, label, Xtr, ytr, se, n_tr, chem, tab, tic, err_nb3200):
    topK = topK_configs(archs)
    res = {"control": [], "tabicl": []}
    corrs = []
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
        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        base = gbm + [chem[ho], tab[ho], gnn]
        res["control"].append(rae(ytr[ho], np.clip(np.mean(base, 0), lo, hi)))
        res["tabicl"].append(rae(ytr[ho], np.clip(np.mean(base + [tic[ho]], 0), lo, hi)))
        if err_nb3200 is not None:
            corrs.append(np.corrcoef(tic[ho], err_nb3200[ho])[0, 1])
    means = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    stds = {k: round(float(np.std(v)), 4) for k, v in res.items()}
    delta = means["tabicl"] - means["control"]
    print(f"[{label}] {[r['arch'] for r in topK]} + CheMeleon + TabPFN + GNN (+TabICL)")
    for k in res: print(f"    {k:8s} {means[k]:.4f} +/- {stds[k]:.4f}")
    print(f"    tabicl - control = {delta:+.4f}  (- is better)")
    if corrs: print(f"    corr(TabICL, nb3200-err) = {np.nanmean(corrs):+.3f}")
    return means, stds, delta


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    tic = np.load(f"{SD}/tabicl_oof.npy")
    sc = rae  # noqa
    print(f"TabICL standalone scaffold-CV RAE {rae(ytr, tic):.4f}")
    # nb3200 OOF error for corr (optional; skip if absent)
    err = None
    for cand in [f"{SD}/nb3200_oof.npy", f"{P}/nb3200_oof.npy"]:
        if os.path.exists(cand):
            base = np.load(cand); err = np.abs(ytr - base); break

    m5, s5, d5 = run_config(("lgbm", "xgb", "cat", "histgb", "mlp"), "5-arch", Xtr, ytr, se, n_tr, chem, tab, tic, err)
    m4, s4, d4 = run_config(("lgbm", "xgb", "cat", "histgb"), "4-GBM(deployed)", Xtr, ytr, se, n_tr, chem, tab, tic, err)

    out = {"5arch": {"means": m5, "std": s5, "tabicl_minus_control": round(d5, 4)},
           "4gbm_deployed": {"means": m4, "std": s4, "tabicl_minus_control": round(d4, 4)},
           "tabicl_standalone_rae": round(float(rae(ytr, tic)), 4)}
    json.dump(out, open(f"{P}/nb1174_summary.json", "w"), indent=2)

    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4367
    t4, c4 = m4["tabicl"], m4["control"]
    if d4 < -0.001 and t4 < c4 and t4 < prev - 0.001:
        print(f"\nDEPLOY-WORTHY: 4-GBM matched tabicl-control {d4:+.4f} (<-0.001), {t4:.4f} < best {prev}. Build deploy submission.")
    else:
        print(f"\nno deploy: 4-GBM matched delta {d4:+.4f} (need <-0.001), tabicl {t4:.4f} vs best {prev}; best stays {prev}")


if __name__ == "__main__":
    main()
