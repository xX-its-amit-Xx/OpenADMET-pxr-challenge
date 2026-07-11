"""nb1362 -- HIERARCHICAL GATED CURRICULUM finetune: broad -> NR family -> PXR.

The exact framing the user asked about and the campaign never built as a staged
curriculum (it had cross-target *features* and pooled MTL, not staged transfer).

Shared trunk (combined 2265 -> 512 -> 256), transferred across 3 stages:
  Stage 1 (BROAD):  tox21 12 NR/SR assays, multi-label BCE  -> generic xeno-sensing rep
  Stage 2 (NR fam): ChEMBL NR-target pEC50 (FXR/PPARg/RXRa/CAR/VDR/...), regression
  Stage 3 (PXR):    PXR pEC50 regression (the target)

'Gated': we train 4 variants and let honest scaffold-OOF on PXR PICK which transfer to keep:
  scratch (S3 only) | S2->S3 | S1->S3 | S1->S2->S3
Report each variant's scaffold-OOF RAE (gate) AND its real-260 RAE (truth). The delta
scratch->best is the curriculum value.
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from src.pxr.featurize import combined, impute

OUT="C:/pxr_work/posthoc_creative"
torch.manual_seed(0); np.random.seed(0); rng=np.random.RandomState(0)
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())

d=np.load(f"{OUT}/prep.npz",allow_pickle=True)
X_tr,y_tr,scaf_tr=d["X_tr"],d["y_tr"],d["scaf_tr"]; X_260,y_260,base_260=d["X_260"],d["y_260"],d["base_260"]
D=X_tr.shape[1]

# ---------- featurize panel sources (cached) ----------
def _san(X):
    X=np.asarray(X,np.float32)
    X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
    return np.clip(X,-1e6,1e6).astype(np.float32)

def feat_cache(name, smiles):
    p=f"{OUT}/feat_{name}.npy"
    if os.path.exists(p): return _san(np.load(p))
    print(f"[feat] {name}: {len(smiles)} smiles...")
    X=_san(impute(combined(list(smiles)))); np.save(p,X); return X

# Stage 1: tox21 broad panel
tox=pd.read_parquet("data/external/tox21_nr_data.parquet")
smi_col=[c for c in tox.columns if "smile" in c.lower() or c.lower()=="mol_id" or "smiles"==c.lower()]
smi_col=[c for c in tox.columns if "smile" in c.lower()]
assay_cols=[c for c in tox.columns if c.startswith(("NR-","SR-"))]
if not smi_col:
    # some tox21 parquets store smiles under 'smiles'
    smi_col=[c for c in tox.columns if tox[c].astype(str).str.contains("c1|C\\(|=O",regex=True,na=False).mean()>0.5]
sc1=smi_col[0]; tox=tox.dropna(subset=[sc1]).reset_index(drop=True)
X1=feat_cache("tox21",tox[sc1]); Y1=tox[assay_cols].to_numpy(np.float32)
M1=np.isfinite(Y1).astype(np.float32); Y1=np.nan_to_num(Y1)
print(f"[S1] tox21 {X1.shape} assays={len(assay_cols)} label-density={M1.mean():.2f}")

# Stage 2: ChEMBL NR-target pEC50 (regression, pooled per-target multitask)
nr=pd.read_parquet("data/external/chembl_nr_targets.parquet")
nr=nr[nr["standard_type"].isin(["EC50","AC50"])].dropna(subset=["smiles","pec50"]).reset_index(drop=True)
top_tgt=nr["target_name"].value_counts().head(8).index.tolist()
nr=nr[nr["target_name"].isin(top_tgt)].reset_index(drop=True)
X2=feat_cache("chembl_nr",nr["smiles"])
tgt_idx={t:i for i,t in enumerate(top_tgt)}
Y2=np.full((len(nr),len(top_tgt)),np.nan,np.float32)
for i,(t,v) in enumerate(zip(nr["target_name"],nr["pec50"])): Y2[i,tgt_idx[t]]=v
M2=np.isfinite(Y2).astype(np.float32);
# znorm each target col by observed mean/std
mu2=np.nanmean(Y2,0); sd2=np.nanstd(Y2,0)+1e-6; Y2z=np.nan_to_num((Y2-mu2)/sd2)
print(f"[S2] chembl-NR {X2.shape} targets={len(top_tgt)} ({top_tgt})")

# ---------- shared scalers: fit on PXR train (input space) ----------
scX=StandardScaler().fit(X_tr)
def T(X): return torch.tensor(scX.transform(X),dtype=torch.float32)

class Trunk(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(D,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.15),
                            nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.15))
    def forward(s,x): return s.net(x)

def head(out): return nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,out))

def fit_stage(trunk, X, Y, M, loss_kind, epochs, lr=1e-3):
    h=head(Y.shape[1]); params=list(trunk.parameters())+list(h.parameters())
    opt=torch.optim.Adam(params,lr=lr,weight_decay=1e-5)
    Xs=T(X); Yt=torch.tensor(Y,dtype=torch.float32); Mt=torch.tensor(M,dtype=torch.float32)
    bce=nn.BCEWithLogitsLoss(reduction="none"); bs=256; idx=np.arange(len(X))
    for ep in range(epochs):
        trunk.train(); rng.shuffle(idx)
        for b in range(0,len(idx),bs):
            j=idx[b:b+bs]
            opt.zero_grad(); z=trunk(Xs[j]); o=h(z)
            if loss_kind=="bce": l=(bce(o,Yt[j])*Mt[j]).sum()/(Mt[j].sum()+1e-6)
            else: l=(torch.abs(o-Yt[j])*Mt[j]).sum()/(Mt[j].sum()+1e-6)
            l.backward(); opt.step()
    return trunk

def fit_pxr(trunk, Xtr, ytr, epochs=60, lr=1e-3):
    h=head(1); opt=torch.optim.Adam(list(trunk.parameters())+list(h.parameters()),lr=lr,weight_decay=1e-5)
    Xs=T(Xtr); yt=torch.tensor(ytr,dtype=torch.float32).view(-1,1); l1=nn.L1Loss(); bs=256; idx=np.arange(len(Xtr))
    for ep in range(epochs):
        trunk.train(); rng.shuffle(idx)
        for b in range(0,len(idx),bs):
            j=idx[b:b+bs]; opt.zero_grad(); l=l1(h(trunk(Xs[j])),yt[j]); l.backward(); opt.step()
    return trunk,h

def predict(trunk,h,X):
    trunk.eval(); h.eval()
    with torch.no_grad(): return h(trunk(T(X))).numpy().ravel()

def build_variant(pre1,pre2,Xtr,ytr):
    trunk=Trunk()
    if pre1: trunk=fit_stage(trunk,X1,Y1,M1,"bce",epochs=25)
    if pre2: trunk=fit_stage(trunk,X2,Y2z,M2,"l1",epochs=25)
    return fit_pxr(trunk,Xtr,ytr)

# scaffold folds
uq=np.unique(scaf_tr); rng.shuffle(uq); fold_of={s:i%5 for i,s in enumerate(uq)}
folds=np.array([fold_of[s] for s in scaf_tr])

variants={"scratch":(0,0),"S2->S3":(0,1),"S1->S3":(1,0),"S1->S2->S3":(1,1)}
results={}
for name,(p1,p2) in variants.items():
    t0=time.time(); oof=np.full(len(y_tr),np.nan)
    for f in range(5):
        m=folds!=f
        trunk,h=build_variant(p1,p2,X_tr[m],y_tr[m]); oof[~m]=predict(trunk,h,X_tr[~m])
    trunk,h=build_variant(p1,p2,X_tr,y_tr); p260=predict(trunk,h,X_260)
    results[name]=dict(oof_rae=round(rae(y_tr,oof),4),
                       s260_rae=round(rae(y_260,p260),4),
                       blend_base_rae=round(rae(y_260,0.5*p260+0.5*base_260),4))
    np.save(f"{OUT}/nb1362_{name.replace('->','_')}_260.npy",p260)
    print(f"[{name}] scaffold-OOF RAE={results[name]['oof_rae']}  260 RAE={results[name]['s260_rae']}  "
          f"blend+base={results[name]['blend_base_rae']}  ({time.time()-t0:.0f}s)")

# gate: pick best by OOF, report its honest 260
best=min(results,key=lambda k:results[k]["oof_rae"])
summary=dict(base_260_rae=round(rae(y_260,base_260),4),variants=results,
             gate_pick_by_oof=best,gate_pick_260_rae=results[best]["s260_rae"],
             curriculum_value_oof=round(results["scratch"]["oof_rae"]-results[best]["oof_rae"],4),
             curriculum_value_260=round(results["scratch"]["s260_rae"]-results[best]["s260_rae"],4))
json.dump(summary,open(f"{OUT}/nb1362_curriculum.json","w"),indent=2)
print("\n===== CURRICULUM RESULTS (real 260) =====")
print(json.dumps(summary,indent=2))
