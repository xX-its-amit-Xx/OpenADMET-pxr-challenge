"""nb1342 — TabPFN-on-CheMeleon (the flagged top-performer we could never run) scored on 260.

We have cached CheMeleon embeddings (chemeleon_tr 4139x2048, chemeleon_te 513x2048).
Train TabPFN on 4139 CRC + 253 phase1 (CheMeleon->PCA), predict 260, score on released truth.
Also test: does it ADD to the best component (combined_corrected 0.6318)?
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.pxr.data import load_train, load_test
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
MS="C:/pxr_work/meta_stacking"; SR="C:/pxr_work/search"; OUT="C:/pxr_work/posthoc"
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())

te=load_test().reset_index(drop=True)
bl_idx=np.load(f"{MS}/_blinded_idx.npy"); ub_idx=np.load(f"{MS}/_unblinded_idx.npy")
t1=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
t2=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
truth={**dict(zip(t1["Molecule Name"],t1["pEC50"])), **dict(zip(t2["Molecule Name"],t2["pEC50"]))}
y260=np.array([truth[n] for n in te.loc[bl_idx,"name"]])
y253=np.array([truth[n] for n in te.loc[ub_idx,"name"]])

Xtr=np.load(f"{SR}/chemeleon_tr.npy").astype(np.float32)   # 4139
Xte=np.load(f"{SR}/chemeleon_te.npy").astype(np.float32)   # 513
tr=load_train(); ytr=tr["pec50"].to_numpy(float)
# training = 4139 CRC + 253 phase1 (their CheMeleon emb = Xte[ub_idx])
Xall=np.vstack([Xtr, Xte[ub_idx]]); yall=np.concatenate([ytr, y253])
Xtest=Xte[bl_idx]
print(f"train={Xall.shape} (4139+253), test260={Xtest.shape}", flush=True)

# PCA -> 100 (TabPFN prefers <=500 feats), scale
sc=StandardScaler().fit(Xall); pca=PCA(100,random_state=0).fit(sc.transform(Xall))
Ztr=pca.transform(sc.transform(Xall)); Zte=pca.transform(sc.transform(Xtest))

from tabpfn import TabPFNRegressor
import torch
dev="cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, flush=True)
# TabPFN caps train at ~10k; 4392 is fine
reg=TabPFNRegressor(device=dev, n_estimators=4, ignore_pretraining_limits=True)
reg.fit(Ztr, yall)
pred260=reg.predict(Zte)
np.save(f"{OUT}/tabpfn_cheml_260.npy", pred260)
print(f"\\n=== TabPFN-on-CheMeleon on 260 ===", flush=True)
print(f"  standalone: RAE={rae(y260,pred260):.4f} MAE={mae(y260,pred260):.4f}")
# best single component for reference
comb260=np.load(f"{MS}/combined_corrected_513.npy").ravel()[bl_idx]
deploy260=np.load(f"{OUT}/deploy260.npy")
print(f"  [ref] combined_corrected: RAE={rae(y260,comb260):.4f}  our deploy: RAE={rae(y260,deploy260):.4f}")
# does TabPFN ADD to combined_corrected?
br=rae(y260,comb260); bw=0
for w in np.arange(0,1.01,0.05):
    r=rae(y260, (1-w)*comb260 + w*pred260)
    if r<br: br=r; bw=w
print(f"  best blend comb+tabpfn: w={bw:.2f} RAE={br:.4f}")
json.dump({"tabpfn_rae":rae(y260,pred260),"tabpfn_mae":mae(y260,pred260),
           "comb_rae":rae(y260,comb260),"blend_rae":br,"blend_w":bw}, open(f"{OUT}/tabpfn_cheml.json","w"),indent=2)
print("DONE", flush=True)
