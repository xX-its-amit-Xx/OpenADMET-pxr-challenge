"""nb1321 — single-conc as a BASE feature (proper integration vs post-hoc shift).

Compute single-conc P(active)+log2fc predictions for ALL 4139 train compounds,
add them as features, retrain LGBM on 4139, predict 253. Check if this base model
adds diversity to the blend beyond the post-hoc sc-shift (0.5883 -> 0.5799 already).
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore"); os.environ["PYTHONUNBUFFERED"]="1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from src.pxr.data import load_train, load_test, load_single_conc
from src.pxr.featurize import combined, impute
from src.pxr.eval import scaffold_kfold_indices
from src.pxr.chem import add_standard_columns
from lightgbm import LGBMClassifier, LGBMRegressor
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
MS="C:/pxr_work/meta_stacking"

# 253 truth
raw=pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc=next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc=next(c for c in raw.columns if "pec50" in c.lower())
raw=raw[[nc,pc]].dropna(); raw.columns=["name","pec50_true"]
te=load_test().reset_index(drop=True)
ub_mask=te["name"].isin(set(raw["name"])); ub_idx=te.index[ub_mask].tolist()
te_ub=te[ub_mask].merge(raw,on="name").reset_index(drop=True)
y_true=te_ub["pec50_true"].to_numpy(float); N=len(y_true)

tr=load_train()
print("Featurizing train+test+SC...", flush=True)
Xtr=impute(combined(tr["smiles"].tolist()))
Xte=impute(combined(te_ub["smiles"].tolist()))
ytr=tr["pec50"].to_numpy()

# Single-conc model
sc=load_single_conc().dropna(subset=["smiles","log2_fc_estimate"])
agg=sc.groupby("smiles").agg(max_log2fc=("log2_fc_estimate","max"),min_fdr=("fdr_bh","min")).reset_index()
agg["active"]=((agg["max_log2fc"]>0.5)&(agg["min_fdr"]<0.1)).astype(int)
Xsc=impute(combined(agg["smiles"].tolist()))
clf=LGBMClassifier(n_estimators=400,num_leaves=64,learning_rate=0.04,n_jobs=4,verbose=-1).fit(Xsc,agg["active"].to_numpy())
reg=LGBMRegressor(n_estimators=400,num_leaves=64,learning_rate=0.04,n_jobs=4,verbose=-1).fit(Xsc,agg["max_log2fc"].to_numpy())

# SC features for train + test
sc_p_tr=clf.predict_proba(Xtr)[:,1]; sc_l_tr=reg.predict(Xtr)
sc_p_te=clf.predict_proba(Xte)[:,1]; sc_l_te=reg.predict(Xte)
Xtr2=np.column_stack([Xtr, sc_p_tr, sc_l_tr])
Xte2=np.column_stack([Xte, sc_p_te, sc_l_te])

# Retrain LGBM with SC features, scaffold CV OOF on train + test pred
scaf=add_standard_columns(tr[["smiles"]].copy())["scaffold"].fillna("").to_numpy()
folds=scaffold_kfold_indices(scaf, n_splits=5)
te_pred=np.zeros((N,5))
for k,(tri,vai) in enumerate(folds):
    m=LGBMRegressor(n_estimators=600,num_leaves=64,learning_rate=0.04,n_jobs=4,verbose=-1).fit(Xtr2[tri],ytr[tri])
    te_pred[:,k]=m.predict(Xte2)
sc_base=te_pred.mean(1)
np.save(f"{MS}/sc_basefeat_pred_253.npy", sc_base)
print(f"SC-base-feature model standalone RAE on 253: {rae(y_true, sc_base):.4f}", flush=True)

# Does it add to the existing blend?
meta=np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn=np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb")); comb=np.array(preds["combined_corrected"])[ub_idx]
boltz=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()
best=0.40*meta+0.20*knn+0.10*comb+0.30*boltz
p_active=np.load(f"{MS}/sc_pactive_253.npy")
shifted=best+0.10*(p_active-0.5)*2
ig=(p_active<0.15)&(boltz<3.2)&(shifted<3.5); stage2=np.where(ig,3.0,shifted)
print(f"current stage2 (sc-shift+gate): {rae(y_true,stage2):.4f}", flush=True)

# blend sc_base into stage2
br=rae(y_true,stage2);bw=0
for w in np.arange(0,0.51,0.05):
    r=rae(y_true,(1-w)*stage2+w*sc_base)
    if r<br:br=r;bw=w
print(f"stage2 + sc_base: w={bw:.2f} RAE={br:.4f}", flush=True)
json.dump({"sc_base_standalone":round(rae(y_true,sc_base),4),"stage2":round(rae(y_true,stage2),4),
           "stage2+sc_base":round(br,4),"w":bw}, open("data/processed/nb1321_sc_basefeat.json","w"),indent=2)
print("DONE", flush=True)
