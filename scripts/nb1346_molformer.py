"""nb1346 — MolFormer-XL embeddings (foundation model we never fully explored) scored on 260."""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from src.pxr.data import load_test, load_train
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
MS="C:/pxr_work/meta_stacking"; OUT="C:/pxr_work/posthoc"
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())

te=load_test().reset_index(drop=True); tr=load_train()
bl=np.load(f"{MS}/_blinded_idx.npy"); ub=np.load(f"{MS}/_unblinded_idx.npy")
t1=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"); t2=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
tru={**dict(zip(t1["Molecule Name"],t1["pEC50"])),**dict(zip(t2["Molecule Name"],t2["pEC50"]))}
y253=np.array([tru[n] for n in te.loc[ub,"name"]]); y260=np.array([tru[n] for n in te.loc[bl,"name"]])
smc=[c for c in te.columns if "smile" in c.lower()][0]

from transformers import AutoModel, AutoTokenizer
name="ibm/MoLFormer-XL-both-10pct"
tok=AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model=AutoModel.from_pretrained(name, trust_remote_code=True, deterministic_eval=True).eval()
@torch.no_grad()
def embed(smis, bs=64):
    out=[]
    for i in range(0,len(smis),bs):
        b=smis[i:i+bs]
        enc=tok(b, padding=True, truncation=True, max_length=200, return_tensors="pt")
        h=model(**enc).pooler_output if hasattr(model(**enc),"pooler_output") else None
        if h is None:
            hh=model(**enc).last_hidden_state; mask=enc["attention_mask"].unsqueeze(-1)
            h=(hh*mask).sum(1)/mask.sum(1)
        out.append(h.numpy())
        if i%640==0: print(f"  embed {i}/{len(smis)}", flush=True)
    return np.vstack(out)

print("Embedding train...", flush=True)
Etr=embed(tr["smiles"].tolist()); Ete=embed(te[smc].tolist())
np.save(f"{OUT}/molformer_tr.npy",Etr); np.save(f"{OUT}/molformer_te.npy",Ete)
ytr=tr["pec50"].to_numpy()
# train on 4139 + 253
Xall=np.vstack([Etr, Ete[ub]]); yall=np.concatenate([ytr, y253]); Xte=Ete[bl]
sc=StandardScaler().fit(Xall)
# Ridge on PCA
pca=PCA(200,random_state=0).fit(sc.transform(Xall))
Ztr=pca.transform(sc.transform(Xall)); Zte=pca.transform(sc.transform(Xte))
rg=RidgeCV(alphas=np.logspace(-2,4,20)).fit(Ztr,yall); pr_ridge=rg.predict(Zte)
lg=LGBMRegressor(n_estimators=500,num_leaves=64,learning_rate=0.04,n_jobs=4,verbose=-1).fit(sc.transform(Xall),yall); pr_lgbm=lg.predict(sc.transform(Xte))
comb260=np.load(f"{MS}/combined_corrected_513.npy").ravel()[bl]
print(f"\\n=== MolFormer-XL on 260 (base comb 0.6318) ===")
print(f"  MolFormer+Ridge: RAE={rae(y260,pr_ridge):.4f} MAE={mae(y260,pr_ridge):.4f}")
print(f"  MolFormer+LGBM:  RAE={rae(y260,pr_lgbm):.4f}")
# add to comb?
for nm,pr in [("ridge",pr_ridge),("lgbm",pr_lgbm)]:
    br=rae(y260,comb260);bw=0
    for w in np.arange(0,0.51,0.05):
        r=rae(y260,(1-w)*comb260+w*pr)
        if r<br:br=r;bw=w
    print(f"  comb + MolFormer-{nm}: w={bw:.2f} RAE={br:.4f}")
json.dump({"molformer_ridge":rae(y260,pr_ridge),"molformer_lgbm":rae(y260,pr_lgbm),"comb":rae(y260,comb260)},open(f"{OUT}/molformer.json","w"),indent=2)
print("DONE")
