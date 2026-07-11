"""nb1363 -- Biological fingerprint (NR read-across + cross-target transfer) on the REAL 260.

nb1123 built read-across bio-FP and found it absorbed on the 253; this re-scores the
mechanism on the now-unblinded 260, both as an added LGBM feature and as a residual on
the robust base (combined_corrected = 0.6318).

Two donor bio-fingerprints over the ChEMBL NR panel (FXR/PPARg/RXRa/CAR/VDR/...):
  READ-ACROSS : per target, Tanimoto-weighted mean of K=5 nearest *measured* donor pEC50
  TRANSFER    : per target, prediction of an LGBM donor model (learned SAR)
Feature block = [read-across pEC50 x T | max-sim x T | transfer pEC50 x T]  (~24 dims)

Honest: 5-fold scaffold-OOF on PXR train to gate; report 260 for
  (a) combined-LGBM  vs  combined-LGBM + bioFP        (does bioFP add to a plain model?)
  (b) base_260       vs  base_260 + bioFP-residual    (does it add to the deployed base?)
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
from src.pxr.featurize import combined, impute

OUT="C:/pxr_work/posthoc_creative"
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())
def mfp(s):
    m=Chem.MolFromSmiles(str(s)); return AllChem.GetMorganFingerprintAsBitVect(m,2,2048) if m else None

d=np.load(f"{OUT}/prep.npz",allow_pickle=True)
X_tr,y_tr,scaf_tr=d["X_tr"],d["y_tr"],d["scaf_tr"]; X_260,y_260,base_260=d["X_260"],d["y_260"],d["base_260"]
smi_tr=np.load(f"{OUT}/smi_tr.npy",allow_pickle=True); smi_260=d["smi_260"]

# donor panel
nr=pd.read_parquet("data/external/chembl_nr_targets.parquet")
nr=nr[nr["standard_type"].isin(["EC50","AC50"])].dropna(subset=["smiles","pec50"]).reset_index(drop=True)
top=nr["target_name"].value_counts().head(8).index.tolist(); nr=nr[nr["target_name"].isin(top)].reset_index(drop=True)
print(f"[bio] donor panel {len(nr)} rows, {len(top)} targets")

# donor fps per target
don={}
for t in top:
    sub=nr[nr["target_name"]==t]; fps=[mfp(s) for s in sub["smiles"]]
    keep=[i for i,f in enumerate(fps) if f is not None]
    don[t]=([fps[i] for i in keep], sub["pec50"].to_numpy()[keep])

def bio_block(smiles):
    Q=[mfp(s) for s in smiles]; T=len(top)
    RA=np.zeros((len(smiles),T)); SIM=np.zeros((len(smiles),T))
    for j,t in enumerate(top):
        dfps,dv=don[t]
        for i,q in enumerate(Q):
            if q is None or len(dfps)==0: continue
            s=np.array(DataStructs.BulkTanimotoSimilarity(q,dfps))
            k=np.argsort(s)[::-1][:5]; w=s[k]
            SIM[i,j]=s[k[0]]
            RA[i,j]=np.average(dv[k],weights=w+1e-6) if w.sum()>0 else dv[k].mean()
    return RA,SIM

# transfer: LGBM donor models -> predict for query (combined feats)
def transfer_block(Xq):
    T=len(top); P=np.zeros((len(Xq),T))
    for j,t in enumerate(top):
        sub=nr[nr["target_name"]==t]
        Xd=impute(combined(sub["smiles"].tolist())).astype(np.float32)
        Xd=np.nan_to_num(Xd,posinf=0,neginf=0)
        m=lgb.LGBMRegressor(n_estimators=300,num_leaves=31,learning_rate=0.05,n_jobs=-1,verbose=-1)
        m.fit(Xd,sub["pec50"].to_numpy()); P[:,j]=m.predict(Xq)
    return P

print("[bio] building read-across + transfer blocks (train/260)...")
RA_tr,SIM_tr=bio_block(smi_tr); RA_te,SIM_te=bio_block(smi_260)
TR_tr=transfer_block(X_tr); TR_te=transfer_block(X_260)
B_tr=np.hstack([RA_tr,SIM_tr,TR_tr]).astype(np.float32)
B_te=np.hstack([RA_te,SIM_te,TR_te]).astype(np.float32)
print(f"[bio] bioFP dim={B_tr.shape[1]}  mean max-sim to donors (260)={SIM_te.max(1).mean():.2f}")

# scaffold folds
uq=np.unique(scaf_tr); rng=np.random.RandomState(42); rng.shuffle(uq)
fold_of={s:i%5 for i,s in enumerate(uq)}; folds=np.array([fold_of[s] for s in scaf_tr])

def oof_lgb(X):
    oof=np.full(len(y_tr),np.nan)
    for f in range(5):
        m=folds!=f
        g=lgb.LGBMRegressor(n_estimators=500,num_leaves=64,learning_rate=0.05,n_jobs=-1,verbose=-1)
        g.fit(X[m],y_tr[m]); oof[~m]=g.predict(X[~m])
    return oof

# (a) plain combined vs combined+bioFP
oof_c=oof_lgb(X_tr); oof_cb=oof_lgb(np.hstack([X_tr,B_tr]))
def full_pred(X,Xte):
    g=lgb.LGBMRegressor(n_estimators=500,num_leaves=64,learning_rate=0.05,n_jobs=-1,verbose=-1)
    g.fit(X,y_tr); return g.predict(Xte)
p_c=full_pred(X_tr,X_260); p_cb=full_pred(np.hstack([X_tr,B_tr]),np.hstack([X_260,B_te]))

# (b) bioFP as residual on base: fit residual (y - base_proxy) on bioFP, add to base_260
#    base proxy on train = oof_c (honest). residual target = y_tr - oof_c
resid=y_tr-oof_c
oof_r=np.full(len(y_tr),np.nan)
for f in range(5):
    m=folds!=f
    g=lgb.LGBMRegressor(n_estimators=300,num_leaves=15,learning_rate=0.03,n_jobs=-1,verbose=-1)
    g.fit(B_tr[m],resid[m]); oof_r[~m]=g.predict(B_tr[~m])
# gate residual scale by train improvement
best_a=(rae(y_tr,oof_c),0.0)
for a in (0.25,0.5,0.75,1.0):
    r=rae(y_tr,oof_c+a*oof_r)
    if r<best_a[0]: best_a=(r,a)
gr=lgb.LGBMRegressor(n_estimators=300,num_leaves=15,learning_rate=0.03,n_jobs=-1,verbose=-1).fit(B_tr,resid)
r260=gr.predict(B_te)
base_plus=base_260+best_a[1]*r260

summary=dict(
  base_260_rae=round(rae(y_260,base_260),4),
  a_plain_combined_260=round(rae(y_260,p_c),4),
  a_combined_plus_bioFP_260=round(rae(y_260,p_cb),4),
  a_oof_combined=round(rae(y_tr,oof_c),4), a_oof_combined_bioFP=round(rae(y_tr,oof_cb),4),
  b_residual_scale=best_a[1], b_residual_oof_rae=round(best_a[0],4),
  b_base_260=round(rae(y_260,base_260),4), b_base_plus_bioFP_260=round(rae(y_260,base_plus),4),
  b_delta_260=round(rae(y_260,base_plus)-rae(y_260,base_260),4),
  mean_donor_sim_260=round(float(SIM_te.max(1).mean()),3))
json.dump(summary,open(f"{OUT}/nb1363_bio.json","w"),indent=2)
print("\n===== BIO-FINGERPRINT (read-across + transfer) on real 260 =====")
print(json.dumps(summary,indent=2))
