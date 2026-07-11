"""nb1340 — POST-HOC scoreboard: score every method variant on the now-known 260 truth.

Challenge over -> we have data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv (260 truth).
Reconstruct all 260 predictions in deploy mode + score: base models, blend, our corrections,
and find what WOULD have been optimal. Output C:/pxr_work/posthoc/scoreboard.json + preds.
"""
import sys, os, warnings, json, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.pxr.data import load_test, load_single_conc
from src.pxr.featurize import combined, impute
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())
MS="C:/pxr_work/meta_stacking"; OUT="C:/pxr_work/posthoc"; os.makedirs(OUT,exist_ok=True)

te=load_test().reset_index(drop=True)
bl_idx=np.load(f"{MS}/_blinded_idx.npy"); ub_idx=np.load(f"{MS}/_unblinded_idx.npy")
t1=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
t2=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
truth={**dict(zip(t1["Molecule Name"],t1["pEC50"])), **dict(zip(t2["Molecule Name"],t2["pEC50"]))}
y260=np.array([truth[n] for n in te.loc[bl_idx,"name"]])
y253=np.array([truth[n] for n in te.loc[ub_idx,"name"]])

# ---- base 29 models, clean set ----
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb"))
keys=[];cols=[]
for k,v in preds.items():
    a=np.asarray(v).ravel().astype(float)
    if len(a)<513: continue
    if np.abs(a[ub_idx]-y253).mean()<0.05: continue  # leaker
    keys.append(k); cols.append(a)
BASE=np.column_stack(cols); B253=BASE[ub_idx]; B260=BASE[bl_idx]

# boltz F5 deploy (fit 253, predict 260)
BZP=next(q for q in ["data/processed/boltz_z_rich_513.npy","C:/pxr_struct/boltz/boltz_z_rich_513.npy"] if os.path.exists(q)); bz=np.load(BZP).astype(float)
pca=PCA(24,random_state=0).fit(bz[ub_idx])
X253=np.hstack([B253,pca.transform(bz[ub_idx])]); X260=np.hstack([B260,pca.transform(bz[bl_idx])])
scl=StandardScaler().fit(X253); rg=RidgeCV(alphas=np.logspace(-2,5,30)).fit(scl.transform(X253),y253)
boltz260=rg.predict(scl.transform(X260))

meta260=np.load(f"{MS}/meta_stacker_te_260.npy").ravel()
knn260=np.load(f"{MS}/knn_residual_loocv_te_260_v2.npy").ravel()
comb260=np.load(f"{MS}/combined_corrected_513.npy").ravel()[bl_idx]
best260=0.40*meta260+0.20*knn260+0.10*comb260+0.30*boltz260

# single-conc P(active) deploy
scd=load_single_conc().dropna(subset=["smiles","log2_fc_estimate"])
agg=scd.groupby("smiles").agg(mx=("log2_fc_estimate","max"),mf=("fdr_bh","min")).reset_index()
agg["active"]=((agg["mx"]>0.5)&(agg["mf"]<0.1)).astype(int)
clf=LGBMClassifier(n_estimators=400,num_leaves=64,learning_rate=0.04,n_jobs=4,verbose=-1).fit(impute(combined(agg["smiles"].tolist())),agg["active"].to_numpy())
P260=clf.predict_proba(impute(combined(te.loc[bl_idx,"smiles"].tolist())))[:,1]

sh260=best260+0.10*(P260-0.5)*2
gate=(P260<0.15)&(boltz260<3.2)&(sh260<3.5)
deploy260=np.where(gate,3.0,sh260)

# save all
np.save(f"{OUT}/boltz260.npy",boltz260); np.save(f"{OUT}/best260.npy",best260)
np.save(f"{OUT}/P260.npy",P260); np.save(f"{OUT}/deploy260.npy",deploy260)
np.save(f"{OUT}/y260.npy",y260)

# ---- SCOREBOARD ----
board={}
board["OUR_DEPLOYED (submitted)"]=rae(y260,deploy260)
board["base_blend (no corrections)"]=rae(y260,best260)
board["best_blend + sc-shift only"]=rae(y260,sh260)
for k,c in zip(keys,cols): board[f"base:{k}"]=rae(y260,c[bl_idx])
board["comp:meta_stacker"]=rae(y260,meta260)
board["comp:knn"]=rae(y260,knn260)
board["comp:combined_corrected"]=rae(y260,comb260)
board["comp:boltz_F5"]=rae(y260,boltz260)
sb=sorted(board.items(), key=lambda kv: kv[1])
print("=== 260 SCOREBOARD (RAE, lower=better) ===")
for k,v in sb[:20]: print(f"  {v:.4f}  {k}")
print(f"\\n  [truth: mean {y260.mean():.2f} std {y260.std():.2f}; our submitted MAE {mae(y260,deploy260):.4f}]")

# oracle: best simple 2-way blend of components
comps={"meta":meta260,"knn":knn260,"comb":comb260,"boltz":boltz260,"base_blend":best260}
best_bl=9;bestc=None
import itertools
for a,b in itertools.combinations(comps,2):
    for w in np.arange(0,1.01,0.1):
        r=rae(y260,w*comps[a]+(1-w)*comps[b])
        if r<best_bl: best_bl=r;bestc=(a,w,b,round(1-w,1))
print(f"\\n  best 2-way component blend (post-hoc oracle): {best_bl:.4f}  {bestc}")
json.dump({"scoreboard":dict(sb),"our_deployed_rae":rae(y260,deploy260),"our_deployed_mae":mae(y260,deploy260),
           "base_blend_rae":rae(y260,best260),"oracle_2way":{"rae":best_bl,"cfg":str(bestc)},
           "n260":260,"truth_std":float(y260.std())}, open(f"{OUT}/scoreboard.json","w"),indent=2)
print("\\nsaved", f"{OUT}/scoreboard.json")
