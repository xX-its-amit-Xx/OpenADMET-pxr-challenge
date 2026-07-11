"""nb1347 — Tanimoto-kernel regression + multitask MLP (torch works now), scored on 260."""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch, torch.nn as tnn
from src.pxr.data import load_test, load_train, load_counter, load_single_conc
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from sklearn.kernel_ridge import KernelRidge
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
MS="C:/pxr_work/meta_stacking"; OUT="C:/pxr_work/posthoc"
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())
te=load_test().reset_index(drop=True); tr=load_train()
bl=np.load(f"{MS}/_blinded_idx.npy"); ub=np.load(f"{MS}/_unblinded_idx.npy")
t1=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"); t2=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
tru={**dict(zip(t1["Molecule Name"],t1["pEC50"])),**dict(zip(t2["Molecule Name"],t2["pEC50"]))}
y253=np.array([tru[n] for n in te.loc[ub,"name"]]); y260=np.array([tru[n] for n in te.loc[bl,"name"]])
smc=[c for c in te.columns if "smile" in c.lower()][0]
comb260=np.load(f"{MS}/combined_corrected_513.npy").ravel()[bl]

# ---- (1) Tanimoto Kernel Ridge ----
print("Tanimoto KRR...", flush=True)
tr_smi=tr["smiles"].tolist()+list(t1["SMILES"]); ytr=np.concatenate([tr["pec50"].to_numpy(), y253])
FPtr=morgan_fp_batch(tr_smi).astype(np.float32); FPte=morgan_fp_batch(te.loc[bl,smc].tolist()).astype(np.float32)
def tanimoto_K(A,B):
    inter=A@B.T; a=A.sum(1,keepdims=True); b=B.sum(1,keepdims=True)
    return inter/(a+b.T-inter+1e-9)
Ktr=tanimoto_K(FPtr,FPtr); Kte=tanimoto_K(FPte,FPtr)
krr=KernelRidge(alpha=1.0,kernel="precomputed").fit(Ktr,ytr); pr_krr=krr.predict(Kte)
print(f"  Tanimoto-KRR: RAE={rae(y260,pr_krr):.4f} MAE={mae(y260,pr_krr):.4f}")

# ---- (2) Multitask MLP: pec50 + counter + single-conc heads ----
print("Multitask MLP...", flush=True)
cnt=load_counter().dropna(subset=["pec50","smiles"]); sc=load_single_conc().dropna(subset=["smiles","log2_fc_estimate"])
scagg=sc.groupby("smiles")["log2_fc_estimate"].max().reset_index()
# union table
main=pd.DataFrame({"smiles":tr_smi,"pec50":ytr})
tbl=main.merge(cnt[["smiles","pec50"]].rename(columns={"pec50":"counter"}),on="smiles",how="outer").merge(scagg.rename(columns={"log2_fc_estimate":"sc"}),on="smiles",how="outer").dropna(subset=["smiles"]).reset_index(drop=True)
X=impute(combined(tbl["smiles"].tolist())).astype(np.float32)
Y=tbl[["pec50","counter","sc"]].to_numpy(np.float32); Mmask=~np.isnan(Y)
mu=np.nanmean(Y,0); sd=np.nanstd(Y,0); Yn=(Y-mu)/sd; Yn[np.isnan(Yn)]=0
from sklearn.preprocessing import StandardScaler
xs=StandardScaler().fit(X); Xs=xs.transform(X).astype(np.float32)
Xte=xs.transform(impute(combined(te.loc[bl,smc].tolist()))).astype(np.float32)
torch.manual_seed(0)
net=tnn.Sequential(tnn.Linear(X.shape[1],512),tnn.ReLU(),tnn.Dropout(0.3),tnn.Linear(512,256),tnn.ReLU(),tnn.Dropout(0.2),tnn.Linear(256,3))
opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
Xt=torch.tensor(Xs); Yt=torch.tensor(Yn); Mt=torch.tensor(Mmask.astype(np.float32))
for ep in range(120):
    net.train(); perm=torch.randperm(len(Xt))
    for i in range(0,len(Xt),256):
        idx=perm[i:i+256]; opt.zero_grad()
        out=net(Xt[idx]); loss=(((out-Yt[idx])**2)*Mt[idx]).sum()/Mt[idx].sum()
        loss.backward(); opt.step()
net.eval()
with torch.no_grad(): pr_mlp=net(torch.tensor(Xte)).numpy()[:,0]*sd[0]+mu[0]
print(f"  Multitask-MLP (pec50 head): RAE={rae(y260,pr_mlp):.4f} MAE={mae(y260,pr_mlp):.4f}")

print(f"\\n=== vs base comb 0.6318 ===")
for nm,pr in [("Tanimoto-KRR",pr_krr),("Multitask-MLP",pr_mlp)]:
    br=rae(y260,comb260);bw=0
    for w in np.arange(0,0.51,0.05):
        r=rae(y260,(1-w)*comb260+w*pr)
        if r<br:br=r;bw=w
    print(f"  {nm} standalone {rae(y260,pr):.4f}; +comb best w={bw:.2f} RAE={br:.4f}")
json.dump({"tanimoto_krr":rae(y260,pr_krr),"mtl_mlp":rae(y260,pr_mlp),"comb":rae(y260,comb260)},open(f"{OUT}/krr_mtlnn.json","w"),indent=2)
print("DONE")
