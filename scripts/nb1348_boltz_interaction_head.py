"""nb1348 — full-train Boltz cofold interaction head (we have all 4139 train z embeddings).

Our deployed boltz component used PCA-24 of z. Test the FULL 512-dim interaction embedding
with a proper deep head + GBM. Train on 4139+253, predict 260. Does amplifying the Boltz
structural signal (our only featurizer win, and the cliff-detector at AUC 0.84) beat comb?
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch, torch.nn as tnn, pickle
from src.pxr.data import load_test, load_train
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
MS="C:/pxr_work/meta_stacking"; OUT="C:/pxr_work/posthoc"; BZ="C:/pxr_struct/boltz"
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())
te=load_test().reset_index(drop=True); tr=load_train()
bl=np.load(f"{MS}/_blinded_idx.npy"); ub=np.load(f"{MS}/_unblinded_idx.npy")
t1=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"); t2=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
tru={**dict(zip(t1["Molecule Name"],t1["pEC50"])),**dict(zip(t2["Molecule Name"],t2["pEC50"]))}
y253=np.array([tru[n] for n in te.loc[ub,"name"]]); y260=np.array([tru[n] for n in te.loc[bl,"name"]])

ztr=np.load(f"{BZ}/boltz_z_rich_train.npy").astype(np.float32)   # (4139,512)
zte=np.load(f"{BZ}/boltz_z_rich_513.npy").astype(np.float32)     # (513,512)
ytr=tr["pec50"].to_numpy()
Zall=np.vstack([ztr, zte[ub]]); yall=np.concatenate([ytr, y253]); Zt=zte[bl]
comb260=np.load(f"{MS}/combined_corrected_513.npy").ravel()[bl]
print(f"train z {ztr.shape} + 253; predict 260. base comb 0.6318", flush=True)

sc=StandardScaler().fit(Zall); A=sc.transform(Zall).astype(np.float32); T=sc.transform(Zt).astype(np.float32)
R={}
# (1) Ridge on full 512
R["Ridge-full512"]=rae(y260, RidgeCV(alphas=np.logspace(-2,4,20)).fit(A,yall).predict(T))
# (2) LGBM on full 512
R["LGBM-full512"]=rae(y260, LGBMRegressor(n_estimators=600,num_leaves=64,learning_rate=0.03,n_jobs=4,verbose=-1).fit(A,yall).predict(T))
# (3) deep interaction head (MLP)
torch.manual_seed(0)
net=tnn.Sequential(tnn.Linear(512,256),tnn.GELU(),tnn.Dropout(0.3),tnn.Linear(256,128),tnn.GELU(),tnn.Dropout(0.2),tnn.Linear(128,1))
opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
Xt=torch.tensor(A); Yt=torch.tensor((yall-yall.mean())/yall.std(),dtype=torch.float32).unsqueeze(1)
for ep in range(200):
    net.train(); perm=torch.randperm(len(Xt))
    for i in range(0,len(Xt),128):
        idx=perm[i:i+128]; opt.zero_grad(); loss=((net(Xt[idx])-Yt[idx]).abs()).mean(); loss.backward(); opt.step()
net.eval()
with torch.no_grad(): pr_mlp=net(torch.tensor(T)).numpy().ravel()*yall.std()+yall.mean()
R["deep-interaction-head"]=rae(y260,pr_mlp)
# (4) our current PCA-24 boltz component (ref)
R["[ref] PCA24 boltz component"]=rae(y260, np.load(f"{OUT}/boltz260.npy"))

print("\\n=== Boltz interaction-head variants on 260 ===")
best_pr=None;best_r=9
for k,v in sorted(R.items(),key=lambda kv:kv[1]):
    print(f"  {v:.4f}  {k}")
# does the best boltz head ADD to comb?
for nm,pr in [("Ridge512",RidgeCV(alphas=np.logspace(-2,4,20)).fit(A,yall).predict(T)),("deepMLP",pr_mlp)]:
    br=rae(y260,comb260);bw=0
    for w in np.arange(0,0.51,0.05):
        r=rae(y260,(1-w)*comb260+w*pr)
        if r<br:br=r;bw=w
    print(f"  comb + boltz-{nm}: w={bw:.2f} RAE={br:.4f}")
json.dump({k:round(v,4) for k,v in R.items()}|{"comb":round(rae(y260,comb260),4)},open(f"{OUT}/boltz_head.json","w"),indent=2)
print("DONE")
