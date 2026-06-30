"""nb1165 — honest clean-holdout gate for OCTANT HTChem pEC50 as EXTRA TRAINING ROWS.

Octant HTChem (454 crude) + 96 semi-pure pEC50s are the SAME Octant agonism reporter assay + SAME drug-like
chemspace as our 513 test → DIRECT pEC50 (no binding≠activation gap), so the natural mechanism is extra training
rows of the SAME target (not an aux head). 435 novel-vs-train compounds curated in octant_novel_curated.csv;
coverage to 513 test is the best of any external source (median octant→test Tanimoto 0.419; 48 test cmpds have a
nbr ≥0.5 vs 0 for bindingdb/cpi2m).

Mirrors nb1163_gate's deployed-ensemble structure (topK GBMs combined-only + CheMeleon + TabPFN + GNN) on the SAME
~250 scaffold holdouts (MTL/ho_idx_seed{s}.npy). Refits ONLY the GBM members with octant rows appended (control vs
treatment); CheMeleon/TabPFN/GNN OOFs are held fixed (they don't see octant — so this is a DILUTED, conservative
estimate of the extra-rows effect). Deploy gate: treatment < deployed best - 0.001 AND < matched control.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model, StandardScaler
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"; OC = "C:/pxr_work/octant_htchem"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"
N_SEEDS = 3


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
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]
    Xtr, _ = feature_matrix(d, "combined")          # (4139, 2265) imputed
    n_tr = len(ytr)

    # --- octant extra rows ---
    oc = pd.read_csv(f"{OC}/octant_novel_curated.csv")
    Xoc = impute(combined(oc["cs"].tolist())).astype(np.float32)
    yoc = oc["pec50"].to_numpy(np.float32)
    assert Xoc.shape[1] == Xtr.shape[1], (Xoc.shape, Xtr.shape)
    Xext = np.vstack([Xtr, Xoc]); yext = np.concatenate([ytr, yoc])
    oc_idx = np.arange(n_tr, n_tr + len(yoc))
    print(f"octant extra rows: {len(yoc)}  (ext matrix {Xext.shape})")

    topK = topK_configs()
    gnn = np.load(f"{P}/oof_chemprop_aux.npy")
    chem = np.load(f"{SD}/chemeleon_oof.npy")
    tab = np.load(f"{SD}/tabpfn_oof.npy")

    variants = {"control": [], "treatment": []}
    solo = {"control": [], "treatment": []}
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        hoset = set(ho.tolist())
        trn = np.array([i for i in range(n_tr) if i not in hoset])
        noisy20 = se <= np.quantile(se, 0.8)
        noisy30 = se <= np.quantile(se, 0.7)
        Xho = Xext[ho]
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        for label, train_idx in (("control", trn), ("treatment", np.concatenate([trn, oc_idx]))):
            preds = []
            for c in topK:
                use = train_idx
                if c["prep"] == "noisy20":
                    keep = noisy20[train_idx[train_idx < n_tr]]; orig = train_idx[train_idx < n_tr][keep]
                    use = np.concatenate([orig, train_idx[train_idx >= n_tr]])
                elif c["prep"] == "noisy30":
                    keep = noisy30[train_idx[train_idx < n_tr]]; orig = train_idx[train_idx < n_tr][keep]
                    use = np.concatenate([orig, train_idx[train_idx >= n_tr]])
                preds.append(fit_one(c, Xext, yext, use, Xho))
            ens = np.clip(np.mean(preds + [chem[ho], tab[ho], gnn[ho]], 0), lo, hi)
            variants[label].append(rae(ytr[ho], ens))
            solo[label].append(rae(ytr[ho], np.clip(np.mean(preds, 0), lo, hi)))

    res = {k: round(float(np.mean(v)), 4) for k, v in variants.items()}
    std = {k: round(float(np.std(v)), 4) for k, v in variants.items()}
    solo_r = {k: round(float(np.mean(v)), 4) for k, v in solo.items()}
    print(f"topK {[r['arch'] for r in topK]} + CheMeleon + TabPFN + GNN(fixed)")
    for k in variants: print(f"  ensemble {k:10s} {res[k]:.4f} +/- {std[k]:.4f}")
    print(f"  GBM-solo  control {solo_r['control']:.4f}  treatment {solo_r['treatment']:.4f}  (extra-rows effect {solo_r['treatment']-solo_r['control']:+.4f})")
    delta = res["treatment"] - res["control"]
    print(f"\n  ensemble treatment - control = {delta:+.4f}  (attributable octant-extra-rows effect; - is better)")
    out = {"variants": res, "std": std, "treat_minus_control": round(delta, 4), "solo": solo_r, "n_octant": int(len(yoc))}
    json.dump(out, open(f"{P}/nb1165_summary.json", "w"), indent=2)

    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4451
    if res["treatment"] < prev - 0.001 and res["treatment"] < res["control"] - 0.0005:
        print(f"\nNEW BEST: treatment {res['treatment']:.4f} < {prev:.4f} AND < control -> DEPLOY")
    else:
        print(f"\nno deploy: treatment {res['treatment']:.4f} vs best {prev:.4f} / control {res['control']:.4f}")


if __name__ == "__main__":
    main()
