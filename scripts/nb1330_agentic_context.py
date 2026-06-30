"""nb1330 — build per-compound context for the agentic MedChem tweaker.

For each of the 253 test compounds, assemble everything a chemist would need to judge
whether our model's pEC50 prediction is too high / too low:
  - SMILES, our prediction
  - top-K nearest TRAINING neighbors (SMILES + MEASURED pEC50 + Tanimoto) -> cliff evidence
  - physchem (MW, logP, TPSA, HBD, HBA, rotbonds, fsp3, aromatic rings)
  - single-conc P(active) (orthogonal functional signal)
  - counter-assay nearest-neighbor pEC50 (selectivity)
The agent NEVER sees the true label -> honest test.

Output: data/processed/agentic_context_253.json (list of per-compound dicts)
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from src.pxr.data import load_train, load_test, load_counter
from src.pxr.chem import morgan_fp_batch, compute_physchem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
MS="C:/pxr_work/meta_stacking"

raw=pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc=next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc=next(c for c in raw.columns if "pec50" in c.lower())
raw=raw[[nc,pc]].dropna(); raw.columns=["name","pec50_true"]
te=load_test().reset_index(drop=True)
ub=te["name"].isin(set(raw["name"])); ub_idx=te.index[ub].tolist()
te_ub=te[ub].merge(raw,on="name").reset_index(drop=True)
y=te_ub["pec50_true"].to_numpy(float); N=len(y)

# our best prediction + corrections
meta=np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn=np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb")); comb=np.array(preds["combined_corrected"])[ub_idx]
boltz=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()
best=0.40*meta+0.20*knn+0.10*comb+0.30*boltz
p_active=np.load(f"{MS}/sc_pactive_253.npy")
final=np.load(f"{MS}/final_corrected_253.npy")

tr=load_train()
tr_bv=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in tr["smiles"]]
te_bv=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in te_ub["smiles"]]
tr_pec=tr["pec50"].to_numpy(); tr_emax=(tr["emax_rel"] if "emax_rel" in tr.columns else tr["emax"]).to_numpy()
tr_smi=tr["smiles"].tolist()

cnt=load_counter().dropna(subset=["pec50","smiles"]).reset_index(drop=True)
cnt_bv=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in cnt["smiles"]]
cnt_pec=cnt["pec50"].to_numpy()

ctx=[]
for i in range(N):
    sims=np.array(DataStructs.BulkTanimotoSimilarity(te_bv[i],tr_bv))
    top=np.argsort(sims)[::-1][:5]
    neigh=[{"smiles":tr_smi[j],"pec50":round(float(tr_pec[j]),2),
            "emax":round(float(tr_emax[j]),2) if not np.isnan(tr_emax[j]) else None,
            "tanimoto":round(float(sims[j]),3)} for j in top]
    csims=np.array(DataStructs.BulkTanimotoSimilarity(te_bv[i],cnt_bv))
    cj=int(csims.argmax())
    pcm=compute_physchem(te_ub["smiles"].iloc[i])
    ctx.append({
        "idx":i, "name":te_ub["name"].iloc[i], "smiles":te_ub["smiles"].iloc[i],
        "model_pred_pec50":round(float(best[i]),2),
        "single_conc_P_active":round(float(p_active[i]),3),
        "physchem":{k:round(float(v),2) for k,v in pcm.items()},
        "nearest_train_neighbors":neigh,
        "counter_assay_nn":{"pec50":round(float(cnt_pec[cj]),2),"tanimoto":round(float(csims[cj]),3)},
        # held-out truth (NOT shown to agent; used only for scoring)
        "_true_pec50":round(float(y[i]),2),
    })
json.dump(ctx, open("data/processed/agentic_context_253.json","w"), indent=1)
np.save(f"{MS}/agentic_best_253.npy", best)
print(f"Built context for {N} compounds -> data/processed/agentic_context_253.json")
print(f"Baseline best RAE on 253: {float(np.abs(y-best).sum()/np.abs(y-np.median(y)).sum()):.4f}")
# quick neighbor-disagreement (cliff) flag for blind subset selection
nbr_pec=np.array([np.mean([n['pec50'] for n in c['nearest_train_neighbors'][:3]]) for c in ctx])
top1=np.array([c['nearest_train_neighbors'][0]['tanimoto'] for c in ctx])
cliffish=(top1>=0.45)&(np.abs(best-nbr_pec)<0.6)  # similar to neighbors -> model trusts them
print(f"Compounds with high-sim neighbors (model leans on them): {cliffish.sum()}")
