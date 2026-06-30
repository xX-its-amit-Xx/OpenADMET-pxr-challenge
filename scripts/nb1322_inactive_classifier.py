"""nb1322 — multi-signal inactive classifier for the low-end (the differentiator).

Builds the strongest possible "is this compound truly inactive (pEC50<3.5)" classifier
from ORTHOGONAL signals, then floors high-confidence inactives. Honest nested CV.

Signals (each orthogonal to the CRC-trained blend):
  - single-conc P(active) + pred log2fc (21k screen)
  - Boltz cofold prediction (target-aware structure)
  - counter-assay (PXR-null) model prediction (artifact detector)
  - AIMNet2 / physics scalars
  - best_blend itself
Goal: higher precision => safely floor MORE of the 37 inactives than the 3-signal gate (11).
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore"); os.environ["PYTHONUNBUFFERED"]="1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from src.pxr.data import load_train, load_test, load_counter
from src.pxr.featurize import combined, impute
from src.pxr.eval import scaffold_kfold_indices
from src.pxr.chem import add_standard_columns, morgan_fp_batch
from lightgbm import LGBMRegressor
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())
MS="C:/pxr_work/meta_stacking"

raw=pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc=next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc=next(c for c in raw.columns if "pec50" in c.lower())
raw=raw[[nc,pc]].dropna(); raw.columns=["name","pec50_true"]
te=load_test().reset_index(drop=True)
ub_mask=te["name"].isin(set(raw["name"])); ub_idx=te.index[ub_mask].tolist()
te_ub=te[ub_mask].merge(raw,on="name").reset_index(drop=True)
y=te_ub["pec50_true"].to_numpy(float); N=len(y)

# Existing signals
meta=np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn=np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb")); comb=np.array(preds["combined_corrected"])[ub_idx]
boltz=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()
best=0.40*meta+0.20*knn+0.10*comb+0.30*boltz
p_active=np.load(f"{MS}/sc_pactive_253.npy")
sc_log2fc=np.load(f"{MS}/sc_log2fc_253.npy")
sc_base=np.load(f"{MS}/sc_basefeat_pred_253.npy")

# Counter-assay model: predict PXR-null pEC50 for the 253
print("Building counter-assay model...", flush=True)
cnt=load_counter()
cnt_ok=cnt.dropna(subset=["pec50","smiles"]).reset_index(drop=True)
Xcnt=impute(combined(cnt_ok["smiles"].tolist())); ycnt=cnt_ok["pec50"].to_numpy(float)
Xte=impute(combined(te_ub["smiles"].tolist()))
cm=LGBMRegressor(n_estimators=500,num_leaves=64,learning_rate=0.04,n_jobs=4,verbose=-1).fit(Xcnt,ycnt)
counter_pred=cm.predict(Xte)

# Physics: AIMNet2 (load aligned)
def load_phys(path, ref_names):
    df=pd.read_csv(path)
    ncol="name" if "name" in df.columns else df.columns[0]
    cols=[c for c in df.columns if c not in ("name","src","smiles","status")]
    d=df.set_index(ncol)
    X=np.full((len(ref_names),len(cols)),np.nan)
    for i,nm in enumerate(ref_names):
        if nm in d.index: X[i]=pd.to_numeric(d.loc[nm,cols],errors="coerce").to_numpy(float)
    med=np.nanmedian(X,0); X[np.isnan(X)]=np.take(med,np.where(np.isnan(X))[1])
    return X
aim=load_phys("C:/pxr_work/aimnet2/aimnet_features.csv", te_ub["name"].tolist())

# Assemble inactive-classifier features
Z=np.column_stack([best, p_active, sc_log2fc, sc_base, boltz, counter_pred,
                   best-counter_pred, aim])
ylab=(y<3.5).astype(int)
print(f"Inactive label rate: {ylab.mean()*100:.1f}% ({ylab.sum()}/{N})", flush=True)

# Honest OOF inactive-probability via nested StratifiedKFold (logistic, regularized)
def oof_proba(seed):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    oof=np.zeros(N)
    for tr,va in skf.split(Z,ylab):
        sc=StandardScaler().fit(Z[tr])
        lr=LogisticRegressionCV(Cs=10,cv=4,max_iter=2000,scoring="neg_log_loss").fit(sc.transform(Z[tr]),ylab[tr])
        oof[va]=lr.predict_proba(sc.transform(Z[va]))[:,1]
    return oof

# average proba over seeds
P_inact=np.mean([oof_proba(s) for s in range(5)],axis=0)
np.save(f"{MS}/p_inactive_253.npy", P_inact)

# Precision/recall of the multi-signal inactive classifier
print("\\nMulti-signal inactive classifier precision/recall:")
for thr in [0.5,0.6,0.7,0.8]:
    flag=P_inact>thr; tp=(flag&(ylab==1)).sum(); fp=(flag&(ylab==0)).sum()
    fp_act=(flag&(y>=5)).sum()
    print(f"  P_inact>{thr}: flag={flag.sum():2d} TP={tp} FP={fp}(act={fp_act}) prec={tp/max(flag.sum(),1):.2f} recall={tp/ylab.sum():.2f}")

# Honest nested: floor compounds with high P_inact (start from sc-shifted)
shifted=best+0.10*(p_active-0.5)*2
from itertools import product
def apply_floor(bp,prm):
    thr,floor=prm
    return np.where(P_inact>thr, np.minimum(bp,floor), bp)
grids=[[0.5,0.6,0.7,0.8],[2.5,3.0,3.3]]
allr=[]
for seed in range(15):
    kf=KFold(n_splits=5,shuffle=True,random_state=seed); oof=np.zeros(N)
    for tr,va in kf.split(shifted):
        bp=None;brr=1e9
        for prm in product(*grids):
            r=rae(y[tr],apply_floor(shifted,prm)[tr])
            if r<brr:brr=r;bp=prm
        oof[va]=apply_floor(shifted,bp)[va]
    allr.append(rae(y,oof))
print(f"\\nHonest nested (sc-shift + multi-signal floor): RAE={np.mean(allr):.4f}+-{np.std(allr):.4f}")
print(f"  (prev: sc-shift 0.5883, +3-signal gate 0.5799)")
res=sorted([(rae(y,apply_floor(shifted,prm)),prm) for prm in product(*grids)])
print(f"  full-data optimal: RAE={res[0][0]:.4f} params={res[0][1]}")
json.dump({"nested":round(float(np.mean(allr)),4),"std":round(float(np.std(allr)),4),
           "fulldata_opt":round(res[0][0],4),"params":res[0][1]},
          open("data/processed/nb1322_inactive_clf.json","w"),indent=2)
print("DONE", flush=True)
