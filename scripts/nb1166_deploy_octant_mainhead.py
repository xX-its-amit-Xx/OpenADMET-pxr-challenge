"""nb1166 deploy — DEPLOY the Octant main-head GNN (honest matched gate WIN).

nb1166_gate (deployed-matched 4-GBM config): octant 0.4448 vs control 0.4483 = matched -0.0035 over the
currently-deployed multitask GNN, on the SAME 3 scaffold holdouts. octant = deployed GNN + 435 novel Octant
HTChem pEC50 rows in the MAIN PXR head (same Octant agonism assay, on-manifold; median test-Tanimoto 0.419).
Strict superset of the deployed GNN's training data -> deploy by swapping the GNN ensemble member.

Mirrors nb1164_deploy_mtl_head but the GNN is trained on FULL 4139 train + 435 octant (main PXR head) + 959
ext-EC50 (aux head), predicting the real 513. GBM/CheMeleon/TabPFN members identical to the deployed ensemble
(4 GBMs lgbm/histgb/xgb/cat to match best_ensemble.json members -- NO mlp). Resumable: te_octant_seed{s}.npy.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nb1126_combinatorial_search import feature_matrix, CACHE
from nb1130_ensemble_check import fit_pred
from nb1163_external_multitask_head import train_predict
from src.pxr.data import load_train, load_test
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"; LOG = f"{SD}/results.jsonl"
EXT = "C:/pxr_work/cpi2m/ext_ec50_aux.csv"
OC = "C:/pxr_work/octant_htchem/octant_novel_curated.csv"
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))


def topK_configs():
    # match deployed ensemble (best_ensemble.json members: lgbm,histgb,xgb,cat -- NO mlp)
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in ("lgbm","xgb","cat","histgb")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(); smi = tr["smiles"].tolist()
    te = load_test().reset_index(drop=True); te_smi = te["smiles"].tolist()
    ext = pd.read_csv(EXT); ext_smi = ext["smiles"].tolist(); ext_p = ext["p"].to_numpy()
    oc = pd.read_csv(OC); oc_smi = oc["cs"].tolist(); oc_p = oc["pec50"].to_numpy(float)
    print(f"train {len(tr)} | octant main-head {len(oc)} | ext aux {len(ext)} | test {len(te)}", flush=True)

    # ---- octant main-head multitask GNN on FULL train + octant(main) + ext(aux) -> 513, 3 seeds avg ----
    for seed in range(N_SEEDS):
        sp = f"{MTL}/te_octant_seed{seed}.npy"
        if os.path.exists(sp):
            print(f"[seed {seed}] te_octant cached", flush=True); continue
        print(f"[seed {seed}] training octant main-head on full {len(tr)}+{len(oc)}oct+{len(ext)}ext -> 513...", flush=True)
        main_smi = smi + oc_smi
        main_Y = np.column_stack([np.concatenate([ytr, oc_p]), np.full(len(tr) + len(oc), np.nan)])
        aux_Y = np.column_stack([np.full(len(ext), np.nan), ext_p])
        allsmi = main_smi + ext_smi
        allY = np.vstack([main_Y, aux_Y])
        pred = train_predict(allsmi, allY, te_smi, 2, seed, EPOCHS)
        np.save(sp, pred); print(f"  saved {sp}", flush=True)

    te_oct = np.mean([np.load(f"{MTL}/te_octant_seed{s}.npy") for s in range(N_SEEDS)], 0)

    # ---- build ensemble: 4 GBMs(513) + octant GNN + CheMeleon + TabPFN, clip, mean ----
    d = np.load(CACHE); topK = topK_configs()
    ff = np.load(f"{SD}/feats_full513.npz")
    full_feats = {"combined": ff["combined"], "embed": ff["embed"], "morgan": ff["morgan"]}
    idx = np.arange(len(ytr)); ps = []
    for c in topK:
        Xte = np.hstack([full_feats[t] for t in c["fset"].split("+")])
        ps.append(fit_pred(c, d, idx, Xte, ytr))
    ps.append(te_oct)                                  # <-- swapped GNN member (octant main-head)
    ps.append(np.load(f"{SD}/chemeleon_lgbm_te.npy"))
    ps.append(np.load(f"{SD}/tabpfn_te.npy"))
    full = np.clip(np.mean(ps, 0), np.quantile(ytr, 0.05), np.quantile(ytr, 0.98))

    out = "submissions/nb1166_octant_mainhead_ensemble.csv"
    pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": full}).to_csv(out, index=False)
    print(f"saved {out}  (n={len(full)}, mean {full.mean():.3f}, range {full.min():.2f}-{full.max():.2f})", flush=True)
    # rae = prior deployed best (0.4421) + matched gate delta (-0.0035, octant vs deployed GNN, 4-GBM config)
    json.dump({"rae": 0.4386, "via": "+octant_mainhead (matched -0.0035 vs deployed GNN, nb1166_gate)",
               "submission": out,
               "members": [c["arch"] for c in topK] + ["octant_gnn", "chemeleon", "tabpfn"]},
              open(f"{SD}/best_ensemble.json", "w"), indent=2)
    print("updated best_ensemble.json -> 0.4386 (+octant_mainhead)", flush=True)


if __name__ == "__main__":
    main()
