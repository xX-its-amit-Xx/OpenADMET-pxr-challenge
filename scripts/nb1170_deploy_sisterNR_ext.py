"""nb1170 deploy — DEPLOY the EXTENDED sister-NR pooled 3-head GNN (run ONLY if nb1170_gate wins).
Mirrors nb1168_deploy but the sister-NR aux head pools sisterNR_aux.csv UNION sisterNR_ext_aux.csv
(FXR/LXR/RORg/RXRa/ERa agonist EC50). GNN trained on FULL 4139 train + 435 octant (main) + 959 ext (aux)
+ (sisterNR UNION sisterNR_ext) (aux), predicting the real 513. Resumable: te_sne_seed{s}.npy.
Writes submissions/nb1170_sisterNR_ext_ensemble.csv + updates best_ensemble.json with the gate delta.
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
SNE = "C:/pxr_work/sisterNR_ext/sisterNR_ext_aux.csv"
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))
GATE_DELTA = float(os.environ.get("GATE_DELTA", "-0.0019"))  # set from nb1170_gate 4-GBM matched delta


def topK_configs():
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
    sn = pd.read_csv(SN); sne = pd.read_csv(SNE)
    pool_smi = sn["cs"].tolist() + sne["cs"].tolist()
    pool_p = np.concatenate([sn["p"].to_numpy(float), sne["p"].to_numpy(float)])
    print(f"train {len(tr)} | octant {len(oc)} | ext {len(ext)} | sisterNR {len(sn)}+ext {len(sne)}={len(pool_smi)} | test {len(te)}", flush=True)

    for seed in range(N_SEEDS):
        sp = f"{MTL}/te_sne_seed{seed}.npy"
        if os.path.exists(sp):
            print(f"[seed {seed}] te_sne cached", flush=True); continue
        print(f"[seed {seed}] training sisterNR-EXT 3-head on full -> 513...", flush=True)
        main_smi = smi + oc_smi
        n_main = len(tr) + len(oc)
        main_Y = np.column_stack([np.concatenate([ytr, oc_p]), np.full(n_main, np.nan), np.full(n_main, np.nan)])
        ext_Y = np.column_stack([np.full(len(ext), np.nan), ext_p, np.full(len(ext), np.nan)])
        sn_Y = np.column_stack([np.full(len(pool_smi), np.nan), np.full(len(pool_smi), np.nan), pool_p])
        allsmi = main_smi + ext_smi + pool_smi
        allY = np.vstack([main_Y, ext_Y, sn_Y])
        pred = train_predict(allsmi, allY, te_smi, 3, seed, EPOCHS)
        np.save(sp, pred); print(f"  saved {sp}", flush=True)

    te_sne = np.mean([np.load(f"{MTL}/te_sne_seed{s}.npy") for s in range(N_SEEDS)], 0)

    d = np.load(CACHE); topK = topK_configs()
    ff = np.load(f"{SD}/feats_full513.npz")
    full_feats = {"combined": ff["combined"], "embed": ff["embed"], "morgan": ff["morgan"]}
    idx = np.arange(len(ytr)); ps = []
    for c in topK:
        Xte = np.hstack([full_feats[t] for t in c["fset"].split("+")])
        ps.append(fit_pred(c, d, idx, Xte, ytr))
    ps.append(te_sne)
    ps.append(np.load(f"{SD}/chemeleon_lgbm_te.npy"))
    ps.append(np.load(f"{SD}/tabpfn_te.npy"))
    full = np.clip(np.mean(ps, 0), np.quantile(ytr, 0.05), np.quantile(ytr, 0.98))

    out = "submissions/nb1170_sisterNR_ext_ensemble.csv"
    pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": full}).to_csv(out, index=False)
    print(f"saved {out}  (n={len(full)}, mean {full.mean():.3f}, range {full.min():.2f}-{full.max():.2f})", flush=True)
    new_rae = round(0.4367 + GATE_DELTA, 4)
    json.dump({"rae": new_rae, "via": f"+sisterNR_EXT_aux_head (matched {GATE_DELTA:+.4f} vs deployed sisterNR GNN, nb1170_gate)",
               "submission": out,
               "members": [c["arch"] for c in topK] + ["sisterNR_ext_gnn", "chemeleon", "tabpfn"]},
              open(f"{SD}/best_ensemble.json", "w"), indent=2)
    print(f"updated best_ensemble.json -> {new_rae} (+sisterNR_EXT_aux_head)", flush=True)


if __name__ == "__main__":
    main()
