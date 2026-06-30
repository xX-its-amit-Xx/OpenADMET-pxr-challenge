"""nb1332 — substructure-level public-PXR-SAR prior for blind compounds.

User idea: blind compounds are novel whole-molecule, but specific SUBSTRUCTURES carry
PXR-activity info present in BROAD public data. Build a substructure-enrichment prior +
a broad-public-KB model; validate on 253 (esp. novel-scaffold subset); apply to 260.

KB = single-conc(10870) + ChEMBL-PXR(883) + PubChem-PXR(5739) + Tox21-PXR-actives(1633).
Two priors:
 (A) broad-KB LGBM P(active) on combined features (vs single-conc-only sc_pactive)
 (B) substructure odds-ratio score: per Morgan bit, log-odds of activity in KB, summed
Test: corr-with-truth, corr-with-model-ERROR, RAE on FULL + NOVEL-SCAFFOLD 253.
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore"); os.environ["PYTHONUNBUFFERED"]="1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from src.pxr.data import load_train, load_test, load_single_conc
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch, bemis_murcko, standardize
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
MS="C:/pxr_work/meta_stacking"

raw=pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc=next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc=next(c for c in raw.columns if "pec50" in c.lower())
raw=raw[[nc,pc]].dropna(); raw.columns=["name","pec50_true"]
te=load_test().reset_index(drop=True)
ub=te["name"].isin(set(raw["name"])); ub_idx=te.index[ub].tolist()
te_ub=te[ub].merge(raw,on="name").reset_index(drop=True)
y=te_ub["pec50_true"].to_numpy(float); N=len(y)

meta=np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn=np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb")); comb=np.array(preds["combined_corrected"])[ub_idx]
boltz=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()
best=0.40*meta+0.20*knn+0.10*comb+0.30*boltz
err=best-y

# novel-scaffold flag (scaffold not in 4139 train)
tr=load_train()
def scaf(s):
    m=standardize(s); return bemis_murcko(m) if m else ""
tr_scaf=set(scaf(s) for s in tr["smiles"])
novel=np.array([scaf(s) not in tr_scaf and scaf(s)!="" for s in te_ub["smiles"]])
print(f"253: {novel.sum()}/{N} novel-scaffold", flush=True)

# ── Build BROAD public KB with binary active labels ─────────────────────────
rows=[]
sc=load_single_conc().dropna(subset=["smiles","log2_fc_estimate"])
agg=sc.groupby("smiles").agg(mx=("log2_fc_estimate","max"),mf=("fdr_bh","min")).reset_index()
agg["active"]=((agg["mx"]>0.5)&(agg["mf"]<0.1)).astype(int)
rows.append(agg[["smiles","active"]])
kb=pd.read_parquet("data/external/chembl_pxr_nr_kb.parquet")
ch=kb[kb["source_target"].isin(["NR_PXR","PXR_CHEMBL3401"])].dropna(subset=["smiles","pec50_chembl"])
rows.append(pd.DataFrame({"smiles":ch["smiles"],"active":(ch["pec50_chembl"]>5).astype(int)}))
pcb=pd.read_parquet("data/external/pubchem_pxr_active_smiles.parquet")
rows.append(pd.DataFrame({"smiles":pcb["smiles"],"active":(pcb["active_rate"]>=0.5).astype(int)}))
tx=pd.read_csv("data/external/tox21/pxr_actives_new.csv")
rows.append(pd.DataFrame({"smiles":tx["smiles"],"active":1}))
KB=pd.concat(rows,ignore_index=True).dropna(subset=["smiles"]).drop_duplicates("smiles").reset_index(drop=True)
print(f"Broad KB: {len(KB)} compounds, {KB['active'].mean()*100:.0f}% active", flush=True)

Xte_c=impute(combined(te_ub["smiles"].tolist()))
Xkb=impute(combined(KB["smiles"].tolist()))
ykb=KB["active"].to_numpy()

# (A) broad-KB LGBM P(active)
clf=LGBMClassifier(n_estimators=500,num_leaves=64,learning_rate=0.04,n_jobs=6,verbose=-1).fit(Xkb,ykb)
P_broad=clf.predict_proba(Xte_c)[:,1]
np.save(f"{MS}/P_broad_public_253.npy", P_broad)

# (B) substructure odds-ratio score from Morgan bits
FPkb=morgan_fp_batch(KB["smiles"].tolist()).astype(float)
FPte=morgan_fp_batch(te_ub["smiles"].tolist()).astype(float)
# per-bit log odds of activity (Laplace-smoothed)
act=ykb.astype(bool)
n_act=act.sum(); n_ina=(~act).sum()
bit_act=FPkb[act].sum(0)+1; bit_ina=FPkb[~act].sum(0)+1
logodds=np.log((bit_act/n_act)/(bit_ina/n_ina))  # >0 = activity-enriched substructure
# per-compound: mean logodds over PRESENT bits (substructure activity score)
sub_score=np.array([logodds[FPte[i]>0].mean() if (FPte[i]>0).any() else 0 for i in range(N)])
np.save(f"{MS}/substruct_score_253.npy", sub_score)

sc_pactive=np.load(f"{MS}/sc_pactive_253.npy")
print("\\n=== Correlations (corr_truth / corr_ERROR) ===", flush=True)
for nm,v in [("sc_pactive(single-conc)",sc_pactive),("P_broad(public-KB)",P_broad),("substruct_logodds",sub_score)]:
    print(f"  {nm:<26} truth={np.corrcoef(v,y)[0,1]:+.3f}  error={np.corrcoef(v,err)[0,1]:+.3f}")
print("  -- on NOVEL-scaffold subset --", flush=True)
for nm,v in [("sc_pactive",sc_pactive),("P_broad",P_broad),("substruct_logodds",sub_score)]:
    m=novel
    print(f"  {nm:<26} truth={np.corrcoef(v[m],y[m])[0,1]:+.3f}  error={np.corrcoef(v[m],err[m])[0,1]:+.3f}")

# ── Honest test: does broad/substructure prior beat single-conc shift? ───────
def shift_stack(pa):
    sh=best+0.10*(pa-0.5)*2; ig=(pa<0.15)&(boltz<3.2)&(sh<3.5)
    return np.where(ig,3.0,sh)
print(f"\\n=== RAE (honest) ===", flush=True)
print(f"  baseline best:            {rae(y,best):.4f}")
print(f"  single-conc stack (cur):  {rae(y,shift_stack(sc_pactive)):.4f}")
print(f"  broad-KB stack:           {rae(y,shift_stack(P_broad)):.4f}")
# nested: does adding P_broad + sub_score to the shift help?
from itertools import product
def nested(signals,name):
    allr=[]
    for seed in range(12):
        kf=KFold(n_splits=5,shuffle=True,random_state=seed); oof=np.zeros(N)
        for tri,va in kf.split(best):
            bw=None;br=9
            for ws in product(*[np.arange(-0.3,0.31,0.1)]*len(signals)):
                p=best.copy()
                for w,s in zip(ws,signals): p=p+w*(s-np.mean(s[tri]))
                r=rae(y[tri],p[tri])
                if r<br: br=r;bw=ws
            p=best.copy()
            for w,s in zip(bw,signals): p=p+w*(s-np.mean(s[tri]))
            oof[va]=p[va]
        allr.append(rae(y,oof))
    print(f"  {name}: {np.mean(allr):.4f}+-{np.std(allr):.4f}")
nested([sub_score],"best + substruct_logodds shift")
nested([P_broad],"best + P_broad shift")

json.dump({"novel_frac":round(float(novel.mean()),2),"kb_size":int(len(KB)),
  "corr_error":{"sc_pactive":round(float(np.corrcoef(sc_pactive,err)[0,1]),3),
                "P_broad":round(float(np.corrcoef(P_broad,err)[0,1]),3),
                "substruct":round(float(np.corrcoef(sub_score,err)[0,1]),3)},
  "rae":{"baseline":round(rae(y,best),4),"sc_stack":round(rae(y,shift_stack(sc_pactive)),4),
         "broad_stack":round(rae(y,shift_stack(P_broad)),4)}},
  open("data/processed/nb1332_substruct.json","w"),indent=2)
print("DONE", flush=True)
