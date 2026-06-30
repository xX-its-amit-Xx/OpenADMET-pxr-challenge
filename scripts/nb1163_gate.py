"""nb1163_gate — honest clean-holdout gate for the external-EC50 multitask head (nb1163).

Mirrors the deployed nb1136 ensemble (topK GBMs combined-only + CheMeleon + TabPFN + ChemProp), but swaps the
ChemProp GNN member between: base (orig oof_chemprop_aux), control (single-task, matched), treatment (PXR+ext-EC50
multitask). Same ~250 scaffold holdouts (seed=100+s) used to produce the OOFs. Reports RAE per variant + the
treatment-vs-control delta (the attributable aux-head effect) + corr-with-nb3200-error. Deploys only if treatment
beats deployed best 0.4451 by >0.001 AND beats matched control (so the lift is the head, not retrain noise).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nb1126_combinatorial_search import feature_matrix, CACHE
from nb1130_ensemble_check import fit_pred, murcko
from src.pxr.data import load_train
from src.pxr.eval import rae
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"
N_SEEDS = 3


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in ("lgbm", "xgb", "cat", "histgb", "mlp")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def main():
    # require all OOFs cached
    need = [f"{MTL}/{v}_oof_seed{s}.npy" for s in range(N_SEEDS) for v in ("control", "treat")]
    miss = [f for f in need if not os.path.exists(f)]
    if miss:
        print(f"waiting on {len(miss)} OOF files, e.g. {miss[0]}"); return

    d = np.load(CACHE); ytr = d["ytr"]
    topK = topK_configs()
    gnn = np.load(f"{P}/oof_chemprop_aux.npy")
    chem = np.load(f"{SD}/chemeleon_oof.npy")
    tab = np.load(f"{SD}/tabpfn_oof.npy")
    Xtr, _ = feature_matrix(d, "combined")

    # standalone-GNN RAE on holdout (does the head help the GNN itself, before ensembling)
    variants = {"base": [], "control": [], "treatment": []}
    solo = {"control": [], "treatment": []}
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(len(ytr)) if i not in set(ho.tolist())])
        treeps = [fit_pred(c, d, trn, Xtr[ho], ytr, fset="combined") for c in topK]
        common = treeps + [chem[ho], tab[ho]]
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        gnn_member = {"base": gnn[ho], "control": np.load(f"{MTL}/control_oof_seed{seed}.npy"),
                      "treatment": np.load(f"{MTL}/treat_oof_seed{seed}.npy")}
        for k in variants:
            pred = np.clip(np.mean(common + [gnn_member[k]], 0), lo, hi)
            variants[k].append(rae(ytr[ho], pred))
        for k in ("control", "treatment"):
            solo[k].append(rae(ytr[ho], np.clip(gnn_member[k], lo, hi)))

    res = {k: round(float(np.mean(v)), 4) for k, v in variants.items()}
    std = {k: round(float(np.std(v)), 4) for k, v in variants.items()}
    solo_r = {k: round(float(np.mean(v)), 4) for k, v in solo.items()}
    print(f"topK {[r['arch'] for r in topK]} + CheMeleon + TabPFN + <GNN>")
    for k in variants: print(f"  ensemble {k:10s} {res[k]:.4f} +/- {std[k]:.4f}")
    print(f"  standalone-GNN  control {solo_r['control']:.4f}  treatment {solo_r['treatment']:.4f}  (solo aux-head effect {solo_r['treatment']-solo_r['control']:+.4f})")
    delta = res["treatment"] - res["control"]
    print(f"\n  ensemble treatment - control = {delta:+.4f}  (attributable external-EC50 aux-head effect; - is better)")
    out = {"variants": res, "std": std, "treat_minus_control": round(delta, 4), "solo": solo_r}
    json.dump(out, open(f"{P}/nb1163_summary.json", "w"), indent=2)

    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4451
    if res["treatment"] < prev - 0.001 and res["treatment"] < res["control"] - 0.0005:
        print(f"\nNEW BEST: treatment {res['treatment']:.4f} < {prev:.4f} AND < control -> DEPLOY (build 513 in follow-up)")
        json.dump({"rae": res["treatment"], "via": "+ext_ec50_multitask_head"}, open(BEST, "w"), indent=2)
    else:
        print(f"\nno deploy: treatment {res['treatment']:.4f} vs best {prev:.4f} / control {res['control']:.4f}")


if __name__ == "__main__":
    main()
