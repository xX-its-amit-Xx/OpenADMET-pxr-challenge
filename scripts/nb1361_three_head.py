"""nb1361 -- TRUE 3-head gated model: PXR pEC50 + assay-noise + activity-cliff.

The framing the campaign never built as such (it had heteroscedastic-NLL and MDN
pieces, but never a dedicated CLIFF head + a NOISE head jointly gating the PXR head).

Architecture (torch MLP, CPU):
    trunk:  X(2265) -> 512 -> 256           (ReLU, BN, dropout)
    head-P: 256 -> 64 -> 1                  pEC50            (L1 loss)
    head-N: 256 -> 64 -> 1                  log assay-noise  (MSE vs znorm log std-err)
    head-C: 256 -> 64 -> 1                  log cliff-hazard (MSE vs znorm log local-roughness)

Three honest tests on the REAL 260 (beat combined_corrected = 0.6318 RAE):
  H1  multi-task rep helps pEC50   -> 3-head pEC50 vs single-head MLP (same trunk)
  H2  CLIFF head gates the base    -> shrink high-cliff-hazard base preds toward median,
                                      threshold+strength calibrated on train OOF, applied to 260
  H3  NOISE head gates the base    -> same, using predicted assay-noise

cliff-hazard target (continuous, learnable; strict binary cliffs are only 20 compounds):
    roughness_i = max over neighbors j (Tanimoto>=0.5) of  Tan_ij * |y_i - y_j|
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

OUT = "C:/pxr_work/posthoc_creative"
torch.manual_seed(0); np.random.seed(0)
def rae(yt, yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt, yp): return float(np.abs(yt-yp).mean())

d = np.load(f"{OUT}/prep.npz", allow_pickle=True)
X_tr, y_tr, se_tr, scaf_tr = d["X_tr"], d["y_tr"], d["se_tr"], d["scaf_tr"]
X_260, y_260, base_260 = d["X_260"], d["y_260"], d["base_260"]
smi_tr = np.load(f"{OUT}/smi_tr.npy", allow_pickle=True)
n, D = X_tr.shape
print(f"[3head] train {X_tr.shape}  260 {X_260.shape}  base260 RAE={rae(y_260,base_260):.4f}")

# ---- continuous cliff-hazard target ----
fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
       if Chem.MolFromSmiles(s) else None for s in smi_tr]
rough = np.zeros(n, np.float32)
for i in range(n):
    if fps[i] is None: continue
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps)); sims[i] = 0
    m = sims >= 0.5
    if m.any(): rough[i] = np.max(sims[m] * np.abs(y_tr[m] - y_tr[i]))
cliff_t = np.log1p(rough)
noise_t = np.log(np.clip(se_tr, 1e-3, None))
# znorm aux targets
czm, czs = cliff_t.mean(), cliff_t.std()+1e-8
nzm, nzs = noise_t.mean(), noise_t.std()+1e-8
cliff_z = (cliff_t-czm)/czs; noise_z = (noise_t-nzm)/nzs
print(f"[3head] cliff-hazard>0: {(rough>0).mean()*100:.0f}%  noise med={np.median(se_tr):.3f}")

# ---- scaffold 5-fold ----
uq = np.unique(scaf_tr); rng = np.random.RandomState(42); rng.shuffle(uq)
fold_of = {s:i%5 for i,s in enumerate(uq)}
folds = np.array([fold_of[s] for s in scaf_tr])

class Net(nn.Module):
    def __init__(self, D, three=True):
        super().__init__(); self.three=three
        self.trunk = nn.Sequential(nn.Linear(D,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.15),
                                   nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.15))
        self.hp = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
        if three:
            self.hn = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
            self.hc = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x):
        z=self.trunk(x);
        return (self.hp(z), self.hn(z), self.hc(z)) if self.three else (self.hp(z),)

def train_net(Xtr,ytr,ntr,ctr,three,epochs=70):
    sc=StandardScaler().fit(Xtr); Xs=torch.tensor(sc.transform(Xtr),dtype=torch.float32)
    yt=torch.tensor(ytr,dtype=torch.float32).view(-1,1)
    nt=torch.tensor(ntr,dtype=torch.float32).view(-1,1); ct=torch.tensor(ctr,dtype=torch.float32).view(-1,1)
    net=Net(Xtr.shape[1],three); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
    l1=nn.L1Loss(); mse=nn.MSELoss(); bs=256; idx=np.arange(len(Xtr))
    for ep in range(epochs):
        net.train(); rng.shuffle(idx)
        for b in range(0,len(idx),bs):
            j=idx[b:b+bs]
            opt.zero_grad(); out=net(Xs[j])
            loss=l1(out[0],yt[j])
            if three: loss=loss+0.3*mse(out[1],nt[j])+0.3*mse(out[2],ct[j])
            loss.backward(); opt.step()
    return net, sc

def predict(net,sc,X,three):
    net.eval()
    with torch.no_grad():
        out=net(torch.tensor(sc.transform(X),dtype=torch.float32))
    p=out[0].numpy().ravel()
    if three: return p, out[1].numpy().ravel(), out[2].numpy().ravel()
    return p, None, None

# ---- OOF for both single & three head + full-fit 260 preds ----
res={}
for three in (False, True):
    tag="3head" if three else "1head"
    oof=np.full(n,np.nan); oof_c=np.full(n,np.nan); oof_nz=np.full(n,np.nan)
    t0=time.time()
    for f in range(5):
        tr_m=folds!=f; va=~tr_m
        net,sc=train_net(X_tr[tr_m],y_tr[tr_m],noise_z[tr_m],cliff_z[tr_m],three)
        p,nn_,cc_=predict(net,sc,X_tr[va],three)
        oof[va]=p
        if three: oof_c[va]=cc_; oof_nz[va]=nn_
    # full fit -> 260
    net,sc=train_net(X_tr,y_tr,noise_z,cliff_z,three)
    p260,n260,c260=predict(net,sc,X_260,three)
    res[tag]=dict(oof=oof, oof_c=oof_c, oof_nz=oof_nz, p260=p260, c260=c260, n260=n260)
    print(f"[{tag}] OOF RAE={rae(y_tr,oof):.4f}  260 standalone RAE={rae(y_260,p260):.4f} MAE={mae(y_260,p260):.4f}  ({time.time()-t0:.0f}s)")

# ================= H1: multi-task rep help pEC50? =================
p1=res["1head"]["p260"]; p3=res["3head"]["p260"]
H1={"1head_260_rae":rae(y_260,p1),"3head_260_rae":rae(y_260,p3),
    "blend50_1head_base":rae(y_260,0.5*p1+0.5*base_260),
    "blend50_3head_base":rae(y_260,0.5*p3+0.5*base_260)}

# ================= H2/H3: cliff/noise heads gate the BASE =================
# base proxy on TRAIN (scaffold-OOF plain combined-LGBM) to calibrate gate honestly
base_tr=np.full(n,np.nan)
for f in range(5):
    tr_m=folds!=f
    m=lgb.LGBMRegressor(n_estimators=500,num_leaves=64,learning_rate=0.05,n_jobs=-1,verbose=-1)
    m.fit(X_tr[tr_m],y_tr[tr_m]); base_tr[~tr_m]=m.predict(X_tr[~tr_m])
med=np.median(y_tr)

def calibrate_gate(signal_tr, base_tr):
    """pick top-quantile q and shrink s minimizing train RAE of gated base."""
    best=(rae(y_tr,base_tr),0.0,0.0)
    for q in (0.5,0.6,0.7,0.8,0.9):
        thr=np.quantile(signal_tr,q); hi=signal_tr>=thr
        for s in (0.2,0.35,0.5,0.7,1.0):
            g=base_tr.copy(); g[hi]=(1-s)*base_tr[hi]+s*med
            r=rae(y_tr,g)
            if r<best[0]: best=(r,q,s)
    return best  # (train_rae, q, s)

gate={}
for name,sig_tr,sig_260 in [("cliff",res["3head"]["oof_c"],res["3head"]["c260"]),
                            ("noise",res["3head"]["oof_nz"],res["3head"]["n260"])]:
    tr_rae,q,s=calibrate_gate(sig_tr,base_tr)
    if q==0 and s==0:
        gate[name]=dict(train_rae=tr_rae,q=0,s=0,base260=rae(y_260,base_260),gated260=rae(y_260,base_260),delta=0.0)
        continue
    thr=np.quantile(sig_260,q); hi=sig_260>=thr
    g=base_260.copy(); g[hi]=(1-s)*base_260[hi]+s*med
    gate[name]=dict(train_rae=round(tr_rae,4),q=q,s=s,n_gated_260=int(hi.sum()),
                    base260=round(rae(y_260,base_260),4),gated260=round(rae(y_260,g),4),
                    delta=round(rae(y_260,g)-rae(y_260,base_260),4))

summary=dict(base_260_rae=round(rae(y_260,base_260),4),
             H1={k:round(v,4) for k,v in H1.items()},
             H2_cliff_gate=gate["cliff"], H3_noise_gate=gate["noise"])
json.dump(summary, open(f"{OUT}/nb1361_three_head.json","w"), indent=2)
np.savez_compressed(f"{OUT}/nb1361_preds.npz", p3_260=p3, p1_260=p1,
                    cliff_260=res["3head"]["c260"], noise_260=res["3head"]["n260"])
print("\n===== 3-HEAD RESULTS (real 260) =====")
print(json.dumps(summary, indent=2))
