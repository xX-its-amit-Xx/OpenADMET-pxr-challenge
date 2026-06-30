"""nb1324 — AutoGluon on CheMeleon embeddings (the RyeCatcher / winner technique).

We have cached CheMeleon 2048-d embeddings (chemeleon_tr 4139x2048, chemeleon_te 513x2048).
Train AutoGluon TabularPredictor (auto-ensembles GBMs/RF/etc) -> predict 513 -> score 253.
Test: standalone RAE, corr with our blend error, does it ADD to the blend?
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore"); os.environ["PYTHONUNBUFFERED"]="1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from src.pxr.data import load_train, load_test
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())
MS="C:/pxr_work/meta_stacking"; SR="C:/pxr_work/search"

raw=pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc=next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc=next(c for c in raw.columns if "pec50" in c.lower())
raw=raw[[nc,pc]].dropna(); raw.columns=["name","pec50_true"]
te=load_test().reset_index(drop=True)
ub_mask=te["name"].isin(set(raw["name"])); ub_idx=te.index[ub_mask].tolist()
te_ub=te[ub_mask].merge(raw,on="name").reset_index(drop=True)
y=te_ub["pec50_true"].to_numpy(float); N=len(y)

tr=load_train(); ytr=tr["pec50"].to_numpy()
Xtr=np.load(f"{SR}/chemeleon_tr.npy"); Xte=np.load(f"{SR}/chemeleon_te.npy")
print(f"CheMeleon emb: tr={Xtr.shape} te={Xte.shape}", flush=True)

from autogluon.tabular import TabularPredictor
cols=[f"c{i}" for i in range(Xtr.shape[1])]
dtr=pd.DataFrame(Xtr,columns=cols); dtr["pec50"]=ytr
dte=pd.DataFrame(Xte,columns=cols)
pred=TabularPredictor(label="pec50",eval_metric="mean_absolute_error",path="C:/pxr_work/ag_cheml2",
                      ).fit(dtr,time_limit=600,presets="good_quality",verbosity=1)
ag513=pred.predict(dte).to_numpy()
np.save(f"{MS}/ag_chemeleon_513.npy", ag513)
ag253=ag513[ub_idx]
print(f"\\nAutoGluon-CheMeleon standalone on 253: RAE={rae(y,ag253):.4f} MAE={mae(y,ag253):.4f}", flush=True)

# Does it add to our blend?
meta=np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn=np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb")); comb=np.array(preds["combined_corrected"])[ub_idx]
boltz=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()
best=0.40*meta+0.20*knn+0.10*comb+0.30*boltz
err=best-y
print(f"corr(AG-CheMeleon, blend error)={np.corrcoef(ag253,err)[0,1]:+.3f}", flush=True)
br=rae(y,best);bw=0
for w in np.arange(0,0.51,0.025):
    r=rae(y,(1-w)*best+w*ag253)
    if r<br:br=r;bw=w
print(f"best + AG-CheMeleon: w={bw:.3f} RAE={br:.4f} (best alone {rae(y,best):.4f})", flush=True)
# Apply validated corrections on the new blend
blend2=(1-bw)*best+bw*ag253
p_active=np.load(f"{MS}/sc_pactive_253.npy")
sh=blend2+0.10*(p_active-0.5)*2
ig=(p_active<0.15)&(boltz<3.2)&(sh<3.5)
final=np.where(ig,3.0,sh)
print(f"+ sc-shift + gate: RAE={rae(y,final):.4f} (prev stack 0.5799)", flush=True)
json.dump({"ag_standalone":round(rae(y,ag253),4),"blend_w":bw,"blend_rae":round(br,4),
           "final_rae":round(rae(y,final),4)}, open("data/processed/nb1324_autogluon.json","w"),indent=2)
print("DONE", flush=True)
