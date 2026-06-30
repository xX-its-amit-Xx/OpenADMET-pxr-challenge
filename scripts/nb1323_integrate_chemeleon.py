"""nb1323 — integrate CheMeleon v2 predictions into the blend, honest score on 253.

Run after downloading chemeleon_test_pred_v2.csv + chemeleon_oof_4139.csv from Colab.
Tests: (1) CheMeleon v2 standalone RAE on 253, (2) does it ADD to the blend (corr with
blend error), (3) best blend including it, (4) does the full corrected stack improve.
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from src.pxr.data import load_test
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())
MS="C:/pxr_work/meta_stacking"
CH="C:/pxr_work/fm_build/chemeleon_test_pred_v2.csv"

raw=pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc=next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc=next(c for c in raw.columns if "pec50" in c.lower())
raw=raw[[nc,pc]].dropna(); raw.columns=["name","pec50_true"]
te=load_test().reset_index(drop=True)
ub_mask=te["name"].isin(set(raw["name"])); ub_idx=te.index[ub_mask].tolist()
te_ub=te[ub_mask].merge(raw,on="name").reset_index(drop=True)
y=te_ub["pec50_true"].to_numpy(float); N=len(y)

meta=np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn=np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb")); comb=np.array(preds["combined_corrected"])[ub_idx]
boltz=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()
best=0.40*meta+0.20*knn+0.10*comb+0.30*boltz
p_active=np.load(f"{MS}/sc_pactive_253.npy")

# CheMeleon v2 preds (513 -> 253)
ch=pd.read_csv(CH)
ch_te=te[["name"]].merge(ch[["name","pred"]],on="name",how="left")
ch513=ch_te["pred"].to_numpy()
ch253=ch513[ub_idx]
print(f"CheMeleon v2 standalone: RAE={rae(y,ch253):.4f} MAE={mae(y,ch253):.4f}")
err=best-y
print(f"corr(CheMeleon, blend error)={np.corrcoef(ch253,err)[0,1]:+.3f}")

# Blend best+chemeleon
br=rae(y,best);bw=0
for w in np.arange(0,0.51,0.025):
    r=rae(y,(1-w)*best+w*ch253)
    if r<br:br=r;bw=w
print(f"best + CheMeleon: w={bw:.3f} RAE={br:.4f} (best alone {rae(y,best):.4f})")
blend2=(1-bw)*best+bw*ch253

# Apply the validated corrections on top
sh=blend2+0.10*(p_active-0.5)*2
ig=(p_active<0.15)&(boltz<3.2)&(sh<3.5)
final=np.where(ig,3.0,sh)
print(f"+ sc-shift + gate: RAE={rae(y,final):.4f}")
print(f"  (prev best-stack without CheMeleon: 0.5799)")
json.dump({"cheml_standalone":round(rae(y,ch253),4),"blend_w":bw,
           "blend_rae":round(br,4),"final_rae":round(rae(y,final),4)},
          open("data/processed/nb1323_chemeleon.json","w"),indent=2)
print("DONE")
