"""nb1325 — diverse models on CheMeleon embeddings (reliable replacement for AutoGluon)
            + augmented functional P(active). The 'more models / more data' exhaustive push.

CheMeleon emb (4139x2048 tr, 513x2048 te). Train CatBoost/ExtraTrees/Ridge/HistGBM,
ensemble, scaffold-CV. Test standalone RAE + whether it ADDS to our blend.
Then augmented P(active): single-conc + PubChem PXR active_rate -> shift/gate test.
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())
MS="C:/pxr_work/meta_stacking"; SR="C:/pxr_work/search"

raw=pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc=next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc=next(c for c in raw.columns if "pec50" in c.lower())
raw=raw[[nc,pc]].dropna(); raw.columns=["name","pec50_true"]
te=load_test().reset_index(drop=True)
ub=te["name"].isin(set(raw["name"])); ub_idx=te.index[ub].tolist()
te_ub=te[ub].merge(raw,on="name").reset_index(drop=True)
y=te_ub["pec50_true"].to_numpy(float); N=len(y)

tr=load_train(); ytr=tr["pec50"].to_numpy()
Xtr=np.load(f"{SR}/chemeleon_tr.npy").astype(np.float32); Xte=np.load(f"{SR}/chemeleon_te.npy").astype(np.float32)
scaf=add_standard_columns(tr[["smiles"]].copy())["scaffold"].fillna("").to_numpy()
folds=scaffold_kfold_indices(scaf,n_splits=5)

print(f"CheMeleon emb tr={Xtr.shape}", flush=True)
# Diverse models, OOF test preds via scaffold CV
def cv_pred(make):
    tep=np.zeros((N,5))
    for k,(tri,vai) in enumerate(folds):
        m=make(); m.fit(Xtr[tri],ytr[tri]); tep[:,k]=m.predict(Xte)[ub_idx] if False else m.predict(Xte[ub_idx])
    return tep.mean(1)
models={
 "catboost": lambda: CatBoostRegressor(iterations=600,depth=8,learning_rate=0.04,loss_function="MAE",verbose=0,thread_count=6),
 "extratrees": lambda: ExtraTreesRegressor(n_estimators=400,n_jobs=6,random_state=0),
 "histgbm": lambda: HistGradientBoostingRegressor(max_iter=500,learning_rate=0.05,loss="absolute_error"),
}
ch_preds={}
for nm,mk in models.items():
    p=cv_pred(mk); ch_preds[nm]=p
    print(f"  CheMeleon-{nm}: RAE={rae(y,p):.4f} MAE={mae(y,p):.4f}", flush=True)
# Ridge on PCA-200
sc=StandardScaler().fit(Xtr); pca=PCA(n_components=200,random_state=0).fit(sc.transform(Xtr))
Ztr=pca.transform(sc.transform(Xtr)); Zte=pca.transform(sc.transform(Xte))[ub_idx]
rp=np.zeros((N,5))
for k,(tri,vai) in enumerate(folds):
    m=RidgeCV(alphas=np.logspace(-2,4,20)).fit(Ztr[tri],ytr[tri]); rp[:,k]=m.predict(Zte)
ch_preds["ridge_pca"]=rp.mean(1); print(f"  CheMeleon-ridge_pca: RAE={rae(y,ch_preds['ridge_pca']):.4f}", flush=True)
ch_ens=np.mean([ch_preds[k] for k in ch_preds],axis=0)
np.save(f"{MS}/cheml_diverse_253.npy", ch_ens)
print(f"  CheMeleon-DIVERSE-ENS: RAE={rae(y,ch_ens):.4f} MAE={mae(y,ch_ens):.4f}", flush=True)

# Does it add to our blend?
meta=np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn=np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb")); comb=np.array(preds["combined_corrected"])[ub_idx]
boltz=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()
best=0.40*meta+0.20*knn+0.10*comb+0.30*boltz
print(f"\\ncorr(CheMeleon-ens, blend error)={np.corrcoef(ch_ens,best-y)[0,1]:+.3f}", flush=True)
br=rae(y,best);bw=0
for w in np.arange(0,0.41,0.025):
    r=rae(y,(1-w)*best+w*ch_ens)
    if r<br:br=r;bw=w
print(f"best + CheMeleon-ens: w={bw:.3f} RAE={br:.4f} (best {rae(y,best):.4f})", flush=True)

# ── Augmented functional P(active) ───────────────────────────────────────────
print("\\n=== Augmented functional P(active) ===", flush=True)
scd=load_single_conc().dropna(subset=["smiles","log2_fc_estimate"])
agg=scd.groupby("smiles").agg(mx=("log2_fc_estimate","max"),mf=("fdr_bh","min")).reset_index()
agg["active"]=((agg["mx"]>0.5)&(agg["mf"]<0.1)).astype(int)
sc_smi=agg["smiles"].tolist(); sc_lab=agg["active"].to_numpy()
pcd=pd.read_parquet("data/external/pubchem_pxr_active_smiles.parquet")
pc_smi=pcd["smiles"].tolist(); pc_lab=(pcd["active_rate"]>=0.5).astype(int).to_numpy()
Xte_c=impute(combined(te_ub["smiles"].tolist()))
def pa_for(smis,labs):
    X=impute(combined(smis))
    c=LGBMClassifier(n_estimators=400,num_leaves=64,learning_rate=0.04,n_jobs=6,verbose=-1).fit(X,labs)
    return c.predict_proba(Xte_c)[:,1]
pa_sc=pa_for(sc_smi,sc_lab)
pa_aug=pa_for(sc_smi+pc_smi,np.concatenate([sc_lab,pc_lab]))
def stack(pa):
    sh=best+0.10*(pa-0.5)*2; ig=(pa<0.15)&(boltz<3.2)&(sh<3.5)
    return rae(y,np.where(ig,3.0,sh))
print(f"  shift/gate: single-conc={stack(pa_sc):.4f}  +pubchem={stack(pa_aug):.4f} (orig 0.5799)", flush=True)
json.dump({"cheml_ens_rae":round(rae(y,ch_ens),4),"cheml_blend_w":bw,"cheml_blend_rae":round(br,4),
           "pa_sc":round(stack(pa_sc),4),"pa_aug":round(stack(pa_aug),4)},
          open("data/processed/nb1325_diverse.json","w"),indent=2)
print("DONE", flush=True)
