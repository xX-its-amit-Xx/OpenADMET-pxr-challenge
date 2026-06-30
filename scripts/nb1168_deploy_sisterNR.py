"""nb1168 deploy — DEPLOY the pooled sister-NR 3-head GNN (honest matched gate WIN).

nb1168_gate (deployed-matched 4-GBM config): sn 0.4361 vs control 0.4380 = matched -0.0019 over the
currently-deployed octant main-head GNN (nb1166), on the SAME 3 scaffold holdouts AND < deployed best 0.4386.
sn = deployed GNN [PXR main(train+435 octant) + 959 ext-EC50 aux] + 4295 sister-NR (VDR/PPARg/GR/CAR…)
agonist EC50/AC50 as an ADDITIONAL activation aux head. Strict superset of the deployed GNN's training signal
-> deploy by swapping the GNN ensemble member.

Mirrors nb1166_deploy_octant_mainhead but the GNN is a 3-head model trained on FULL 4139 train + 435 octant
(main PXR head) + 959 ext-EC50 (aux) + 4295 sisterNR (aux), predicting the real 513. GBM/CheMeleon/TabPFN
members identical to the deployed ensemble (4 GBMs lgbm/histgb/xgb/cat -- NO mlp). Resumable: te_sn_seed{s}.npy.
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
SN = "C:/pxr_work/sisterNR/sisterNR_aux.csv"
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
    sn = pd.read_csv(SN); sn_smi = sn["cs"].tolist(); sn_p = sn["p"].to_numpy(float)
    print(f"train {len(tr)} | octant main {len(oc)} | ext aux {len(ext)} | sisterNR aux {len(sn)} | test {len(te)}", flush=True)

    # ---- sisterNR 3-head GNN on FULL train + octant(main) + ext(aux) + sisterNR(aux) -> 513, 3 seeds avg ----
    for seed in range(N_SEEDS):
        sp = f"{MTL}/te_sn_seed{seed}.npy"
        if os.path.exists(sp):
            print(f"[seed {seed}] te_sn cached", flush=True); continue
        print(f"[seed {seed}] training sisterNR 3-head on full {len(tr)}+{len(oc)}oct+{len(ext)}ext+{len(sn)}sn -> 513...", flush=True)
        main_smi = smi + oc_smi
        n_main = len(tr) + len(oc)
        main_Y = np.column_stack([np.concatenate([ytr, oc_p]), np.full(n_main, np.nan), np.full(n_main, np.nan)])
        ext_Y = np.column_stack([np.full(len(ext), np.nan), ext_p, np.full(len(ext), np.nan)])
        sn_Y = np.column_stack([np.full(len(sn), np.nan), np.full(len(sn), np.nan), sn_p])
        allsmi = main_smi + ext_smi + sn_smi
        allY = np.vstack([main_Y, ext_Y, sn_Y])
        pred = train_predict(allsmi, allY, te_smi, 3, seed, EPOCHS)
        np.save(sp, pred); print(f"  saved {sp}", flush=True)

    te_sn = np.mean([np.load(f"{MTL}/te_sn_seed{s}.npy") for s in range(N_SEEDS)], 0)

    # ---- build ensemble: 4 GBMs(513) + sisterNR GNN + CheMeleon + TabPFN, clip, mean ----
    d = np.load(CACHE); topK = topK_configs()
    ff = np.load(f"{SD}/feats_full513.npz")
    full_feats = {"combined": ff["combined"], "embed": ff["embed"], "morgan": ff["morgan"]}
    idx = np.arange(len(ytr)); ps = []
    for c in topK:
        Xte = np.hstack([full_feats[t] for t in c["fset"].split("+")])
        ps.append(fit_pred(c, d, idx, Xte, ytr))
    ps.append(te_sn)                                   # <-- swapped GNN member (sisterNR 3-head)
    ps.append(np.load(f"{SD}/chemeleon_lgbm_te.npy"))
    ps.append(np.load(f"{SD}/tabpfn_te.npy"))
    full = np.clip(np.mean(ps, 0), np.quantile(ytr, 0.05), np.quantile(ytr, 0.98))

    out = "submissions/nb1168_sisterNR_ensemble.csv"
    pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": full}).to_csv(out, index=False)
    print(f"saved {out}  (n={len(full)}, mean {full.mean():.3f}, range {full.min():.2f}-{full.max():.2f})", flush=True)
    # rae = prior deployed best (0.4386) + matched gate delta (-0.0019, sn vs deployed octant GNN, 4-GBM config)
    json.dump({"rae": 0.4367, "via": "+sisterNR_aux_head (matched -0.0019 vs deployed octant GNN, nb1168_gate)",
               "submission": out,
               "members": [c["arch"] for c in topK] + ["sisterNR_gnn", "chemeleon", "tabpfn"]},
              open(f"{SD}/best_ensemble.json", "w"), indent=2)
    print("updated best_ensemble.json -> 0.4367 (+sisterNR_aux_head)", flush=True)


if __name__ == "__main__":
    main()
