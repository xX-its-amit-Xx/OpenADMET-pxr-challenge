"""nb1164 — DEPLOY the external-EC50 multitask head (nb1163 honest-gate WIN, treatment 0.4421 vs best 0.4451).

Builds the full-513 submission by mirroring the deployed nb1136 ensemble (topK GBMs combined-only + ChemProp GNN
member + CheMeleon-LGBM + TabPFN, clip [q05,q98], mean) but SWAPS the ChemProp GNN member from the single-task
te_chemprop_aux.npy to the PXR+external-EC50 MULTITASK GNN trained on the FULL 4139 train + 959 novel external EC50
aux rows, predicting the real 513 test.

Only new artifact = the treatment multitask GNN on 513. Trained 3 seeds on full-train+ext and averaged (mirrors the
3-seed treatment OOF in the gate). Resumable: caches te_mtl_seed{s}.npy under C:/pxr_work/mtl/. CheMeleon/TabPFN use
their own foundation/in-context preds on the real 513 (no leak). EC50=activation (label match; binding caveat N/A).
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
N_SEEDS = 3
EPOCHS = int(os.environ.get("MTL_EPOCHS", "30"))


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in ("lgbm","xgb","cat","histgb","mlp")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(); smi = tr["smiles"].tolist()
    te = load_test().reset_index(drop=True); te_smi = te["smiles"].tolist()
    ext = pd.read_csv(EXT); ext_smi = ext["smiles"].tolist(); ext_p = ext["p"].to_numpy()
    print(f"train {len(tr)} | ext aux {len(ext)} | test {len(te)}", flush=True)

    # ---- treatment multitask GNN on FULL train + ext -> 513, 3 seeds averaged (resumable) ----
    for seed in range(N_SEEDS):
        sp = f"{MTL}/te_mtl_seed{seed}.npy"
        if os.path.exists(sp):
            print(f"[seed {seed}] te_mtl cached", flush=True); continue
        print(f"[seed {seed}] training treatment multitask on full {len(tr)}+{len(ext)} -> 513...", flush=True)
        main_Y = np.column_stack([ytr, np.full(len(tr), np.nan)])
        aux_Y = np.column_stack([np.full(len(ext), np.nan), ext_p])
        allsmi = smi + ext_smi
        allY = np.vstack([main_Y, aux_Y])
        pred = train_predict(allsmi, allY, te_smi, 2, seed, EPOCHS)
        np.save(sp, pred); print(f"  saved {sp}", flush=True)

    te_mtl = np.mean([np.load(f"{MTL}/te_mtl_seed{s}.npy") for s in range(N_SEEDS)], 0)

    # ---- build ensemble: topK GBMs(513) + treatment GNN + CheMeleon + TabPFN, clip, mean ----
    d = np.load(CACHE); topK = topK_configs()
    ff = np.load(f"{SD}/feats_full513.npz")
    full_feats = {"combined": ff["combined"], "embed": ff["embed"], "morgan": ff["morgan"]}
    idx = np.arange(len(ytr)); ps = []
    for c in topK:
        Xte = np.hstack([full_feats[t] for t in c["fset"].split("+")])
        ps.append(fit_pred(c, d, idx, Xte, ytr))
    ps.append(te_mtl)                                   # <-- swapped GNN member (multitask)
    ps.append(np.load(f"{SD}/chemeleon_lgbm_te.npy"))
    ps.append(np.load(f"{SD}/tabpfn_te.npy"))
    full = np.clip(np.mean(ps, 0), np.quantile(ytr, 0.05), np.quantile(ytr, 0.98))

    out = "submissions/nb1164_mtl_head_ensemble.csv"
    pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": full}).to_csv(out, index=False)
    print(f"saved {out}  (n={len(full)}, mean {full.mean():.3f}, range {full.min():.2f}-{full.max():.2f})", flush=True)
    json.dump({"rae": 0.4421, "via": "+ext_ec50_multitask_head", "submission": out,
               "members": [c["arch"] for c in topK] + ["mtl_gnn", "chemeleon", "tabpfn"]},
              open(f"{SD}/best_ensemble.json", "w"), indent=2)
    print("updated best_ensemble.json", flush=True)


if __name__ == "__main__":
    main()
