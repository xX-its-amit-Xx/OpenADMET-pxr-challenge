"""nb1331 — assay-fusion: bake in ALL orthogonal biology (single-dose + counter + cross-NR).

The user's ask: incorporate single-conc, counter-screen, and related-NR data as
correlated signals / aux heads / features. Test honestly on the 253 whether a rich
biological-profile block beats the current single-conc-only stack (0.5799).

Bio signals per compound:
  - single-conc P(active) + predicted log2fc        [functional, my prior win]
  - counter-assay (PXR-null) predicted pEC50 + NN   [selectivity]
  - PXR-minus-counter selectivity gap
  - emax nearest-neighbor                            [efficacy]
  - cross-NR affinity: FXR/PPARg/VDR/RXRa/LXRa predicted pEC50 (5 sister receptors)
Tests: corr-with-truth, corr-with-model-ERROR, meta-feature LOOCV, directional fusion.
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore"); os.environ["PYTHONUNBUFFERED"]="1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from src.pxr.data import load_test, load_counter, load_single_conc
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor, LGBMClassifier
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
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
Xte=impute(combined(te_ub["smiles"].tolist()))

# Existing winning stack
meta=np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn=np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb")); comb=np.array(preds["combined_corrected"])[ub_idx]
boltz=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()
best=0.40*meta+0.20*knn+0.10*comb+0.30*boltz
err=best-y

bio={}
# single-conc
bio["sc_pactive"]=np.load(f"{MS}/sc_pactive_253.npy")
bio["sc_log2fc"]=np.load(f"{MS}/sc_log2fc_253.npy")
# counter-assay model
cnt=load_counter().dropna(subset=["pec50","smiles"]).reset_index(drop=True)
Xcnt=impute(combined(cnt["smiles"].tolist()))
cm=LGBMRegressor(n_estimators=500,num_leaves=64,learning_rate=0.04,n_jobs=6,verbose=-1).fit(Xcnt,cnt["pec50"].to_numpy())
bio["counter_pred"]=cm.predict(Xte)
bio["pxr_minus_counter"]=best-bio["counter_pred"]
# emax neighbor
import pandas as _pd
from src.pxr.data import load_train
tr=load_train(); tr_bv=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in tr["smiles"]]
te_bv=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in te_ub["smiles"]]
tr_emax=(tr["emax_rel"] if "emax_rel" in tr.columns else tr["emax"]).to_numpy()
en=np.zeros(N)
for i in range(N):
    s=np.array(DataStructs.BulkTanimotoSimilarity(te_bv[i],tr_bv)); j=int(np.nanargmax(s)); en[i]=tr_emax[j] if not np.isnan(tr_emax[j]) else np.nanmedian(tr_emax)
bio["emax_nn"]=en

# cross-NR affinity from chembl NR KB
kb=pd.read_parquet("data/external/chembl_pxr_nr_kb.parquet")
for nrkey in ["NR_FXR","NR_PPARg","NR_VDR","NR_RXRa","NR_LXRa"]:
    sub=kb[kb["source_target"]==nrkey].dropna(subset=["smiles","pec50_chembl"])
    if len(sub)<100: continue
    Xnr=impute(combined(sub["smiles"].tolist()))
    m=LGBMRegressor(n_estimators=400,num_leaves=48,learning_rate=0.04,n_jobs=6,verbose=-1).fit(Xnr,sub["pec50_chembl"].to_numpy())
    bio[f"xnr_{nrkey}"]=m.predict(Xte)
    print(f"cross-NR {nrkey}: trained on {len(sub)}", flush=True)

print("\\n=== Each bio signal: corr-with-TRUTH and corr-with-MODEL-ERROR ===")
for k,v in bio.items():
    print(f"  {k:<22} corr_truth={np.corrcoef(v,y)[0,1]:+.3f}  corr_error={np.corrcoef(v,err)[0,1]:+.3f}")

# Build bio block matrix
B=np.column_stack(list(bio.values()))
np.save(f"{MS}/bio_fusion_253.npy", B)

# TEST 1: bio block as meta-features alongside the base blend, honest LOOCV
base_feats=np.column_stack([meta,knn,comb,boltz])
def loocv(X):
    oof=np.zeros(N)
    for i in range(N):
        tr_i=np.delete(np.arange(N),i)
        sc=StandardScaler().fit(X[tr_i]); m=RidgeCV(alphas=np.logspace(-2,4,20)).fit(sc.transform(X[tr_i]),y[tr_i])
        oof[i]=m.predict(sc.transform(X[[i]]))[0]
    return rae(y,oof)
print("\\n=== Meta-feature LOOCV ===")
print(f"  base only:        {loocv(base_feats):.4f}")
print(f"  base + bio-fusion:{loocv(np.column_stack([base_feats,B])):.4f}")

# TEST 2: honest nested directional fusion (best + bio shifts), vs the 0.5799 stack
p_active=bio["sc_pactive"]
shifted=best+0.10*(p_active-0.5)*2
ig=(p_active<0.15)&(boltz<3.2)&(shifted<3.5); stack0799=np.where(ig,3.0,shifted)
print(f"\\n  current single-conc stack: {rae(y,stack0799):.4f}")
# honest: LGBM on [best + full bio] residual, nested
from itertools import product
allr=[]
Z=np.column_stack([best,B])
for seed in range(10):
    kf=KFold(n_splits=5,shuffle=True,random_state=seed); oof=np.zeros(N)
    for tr_i,va in kf.split(Z):
        sc=StandardScaler().fit(Z[tr_i]); m=RidgeCV(alphas=np.logspace(-1,4,15)).fit(sc.transform(Z[tr_i]),y[tr_i])
        oof[va]=m.predict(sc.transform(Z[va]))
    allr.append(rae(y,oof))
print(f"  Ridge[best+bio] nested: {np.mean(allr):.4f}+-{np.std(allr):.4f}")
json.dump({"base_loocv":round(loocv(base_feats),4),"base_bio_loocv":round(loocv(np.column_stack([base_feats,B])),4),
           "sc_stack":round(rae(y,stack0799),4),"ridge_best_bio":round(float(np.mean(allr)),4),
           "bio_signals":{k:{"corr_truth":round(float(np.corrcoef(v,y)[0,1]),3),"corr_error":round(float(np.corrcoef(v,err)[0,1]),3)} for k,v in bio.items()}},
          open("data/processed/nb1331_assay_fusion.json","w"),indent=2)
print("DONE", flush=True)
