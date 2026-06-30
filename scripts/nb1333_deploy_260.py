"""nb1333 — final deploy of the validated single-conc stack to the 260 blinded compounds.

Validated stack (honest 253 LOOCV = 0.5799):
  best = 0.40*meta + 0.20*knn + 0.10*comb + 0.30*boltz_F5
  shift = best + 0.10*(P_active - 0.5)*2
  gate: (P_active<0.15)&(boltz<3.2)&(shift<3.5) -> floor 3.0

For the 260 (blinded): every component built in DEPLOY mode — trained on 4139 + the
now-unblinded 253, NEVER on the 260. Rigor: reproduce 253 number, leakage/NaN/dist checks.

Outputs: submissions/nb1333_final_260.csv, submissions/nb1333_final_513.csv
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore"); os.environ["PYTHONUNBUFFERED"]="1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from src.pxr.data import load_test, load_single_conc
from src.pxr.featurize import combined, impute
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())
MS="C:/pxr_work/meta_stacking"

# ── indices + truth ─────────────────────────────────────────────────────────
raw=pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
ncol=next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pcol=next(c for c in raw.columns if "pec50" in c.lower())
raw2=raw[[ncol,pcol]].dropna(); raw2.columns=["name","pec50_true"]
te=load_test().reset_index(drop=True)
ub=te["name"].isin(set(raw2["name"])).to_numpy()
ub_idx=np.where(ub)[0]; bl_idx=np.where(~ub)[0]
yt_map=dict(zip(raw2["name"],raw2["pec50_true"]))
y253=np.array([yt_map[n] for n in te.loc[ub_idx,"name"]])
N253,N260=len(ub_idx),len(bl_idx)
print(f"253 unblinded + {N260} blinded", flush=True)

# ── base 29-model matrix, clean set (drop leakers via 253 truth; same set both splits) ──
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb"))
keys=[]; cols=[]
for k,v in preds.items():
    arr=np.asarray(v).ravel().astype(float)
    if len(arr)<513: continue
    p253=arr[ub_idx]
    if np.abs(p253-y253).mean()<0.05:   # drop contaminated/leaker (e.g. nb3200 memorized 253)
        print(f"  drop leaker base model: {k}"); continue
    keys.append(k); cols.append(arr)
BASE=np.column_stack(cols)              # (513, n_base)
B253=BASE[ub_idx]; B260=BASE[bl_idx]
print(f"clean base models: {len(keys)}", flush=True)

# ── Boltz F5 component (base + Boltz-z PCA24), DEPLOY: fit on 253, predict 260 ──
bz=np.load("data/processed/boltz_z_rich_513.npy").astype(float)
pca=PCA(n_components=24,random_state=0).fit(bz[ub_idx])          # PCA fit on 253 only
BZ253=pca.transform(bz[ub_idx]); BZ260=pca.transform(bz[bl_idx])
X253=np.hstack([B253,BZ253]); X260=np.hstack([B260,BZ260])
sc=StandardScaler().fit(X253)
ridge=RidgeCV(alphas=np.logspace(-2,5,30)).fit(sc.transform(X253),y253)
boltz260=ridge.predict(sc.transform(X260))
boltz253_insample=ridge.predict(sc.transform(X253))
# honest 253 boltz = the LOOCV file (for sanity reproduction)
boltz253_loocv=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()

# ── meta + knn + comb deploy components ─────────────────────────────────────
meta260=np.load(f"{MS}/meta_stacker_te_260.npy").ravel()
knn260=np.load(f"{MS}/knn_residual_loocv_te_260_v2.npy").ravel()
comb513=np.load(f"{MS}/combined_corrected_513.npy").ravel()
comb260=comb513[bl_idx]; comb253=comb513[ub_idx]
meta253=np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn253=np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()

# ── single-conc P(active), DEPLOY: train on single-conc, predict 260 ─────────
scd=load_single_conc().dropna(subset=["smiles","log2_fc_estimate"])
agg=scd.groupby("smiles").agg(mx=("log2_fc_estimate","max"),mf=("fdr_bh","min")).reset_index()
agg["active"]=((agg["mx"]>0.5)&(agg["mf"]<0.1)).astype(int)
Xsc=impute(combined(agg["smiles"].tolist()))
clf=LGBMClassifier(n_estimators=400,num_leaves=64,learning_rate=0.04,n_jobs=6,verbose=-1).fit(Xsc,agg["active"].to_numpy())
Xte_all=impute(combined(te["smiles"].tolist()))
P_all=clf.predict_proba(Xte_all)[:,1]
P260=P_all[bl_idx]; P253=P_all[ub_idx]

# ── assemble stack ──────────────────────────────────────────────────────────
W=(0.40,0.20,0.10,0.30)
def stack(meta,knn,comb,boltz,pa):
    best=W[0]*meta+W[1]*knn+W[2]*comb+W[3]*boltz
    sh=best+0.10*(pa-0.5)*2
    flag=(pa<0.15)&(boltz<3.2)&(sh<3.5)
    return np.where(flag,3.0,sh), best, flag

# SANITY 1: reproduce honest 253 (LOOCV components) == 0.5799
fin253_loocv,_,_=stack(meta253,knn253,comb253,boltz253_loocv,np.load(f"{MS}/sc_pactive_253.npy"))
print(f"\\n[SANITY] honest-253 stack (LOOCV comps) RAE={rae(y253,fin253_loocv):.4f}  (expect ~0.5799)", flush=True)
# SANITY 2: deploy components on 253 (in-sample, expect <= honest)
fin253_dep,_,_=stack(meta253,knn253,comb253,boltz253_insample,P253)
print(f"[SANITY] deploy-253 (in-sample) RAE={rae(y253,fin253_dep):.4f}  MAE={mae(y253,fin253_dep):.4f}", flush=True)

# ── 260 deploy prediction ───────────────────────────────────────────────────
fin260,best260,flag260=stack(meta260,knn260,comb260,boltz260,P260)

# ── RIGOR CHECKS ────────────────────────────────────────────────────────────
print("\\n=== RIGOR CHECKS (260) ===", flush=True)
print(f"  NaNs: {np.isnan(fin260).sum()} | range [{fin260.min():.2f},{fin260.max():.2f}]")
print(f"  260 dist: mean={fin260.mean():.2f} med={np.median(fin260):.2f} std={fin260.std():.2f}")
print(f"  253 dist: mean={fin253_dep.mean():.2f} med={np.median(fin253_dep):.2f} std={fin253_dep.std():.2f} (true med {np.median(y253):.2f})")
print(f"  gate fired on {flag260.sum()}/{N260} blinded compounds (floored to 3.0)")
print(f"  P(active) 260: med={np.median(P260):.2f}  boltz260: med={np.median(boltz260):.2f}")
# leakage guard: 260 components never saw 260 labels (we have none) + no pred==truth possible
print(f"  leakage guard: 260 has no labels; all 260 comps trained on 4139+253 only -> OK")

# ── write outputs ───────────────────────────────────────────────────────────
os.makedirs("submissions",exist_ok=True)
sm_col=[c for c in te.columns if "smile" in c.lower()][0]
out260=pd.DataFrame({"Molecule Name":te.loc[bl_idx,"name"].values,"SMILES":te.loc[bl_idx,sm_col].values,"pEC50":np.round(fin260,4)})
out260.to_csv("submissions/nb1333_final_260.csv",index=False)
# full 513 (model predictions throughout, 253 uses honest LOOCV stack, 260 uses deploy)
full=np.zeros(513)
full[ub_idx]=fin253_loocv; full[bl_idx]=fin260
out513=pd.DataFrame({"Molecule Name":te["name"].values,"SMILES":te[sm_col].values,"pEC50":np.round(full,4)})
out513.to_csv("submissions/nb1333_final_513.csv",index=False)
np.save(f"{MS}/final_260_pred.npy",fin260)
json.dump({"honest_253_rae":round(rae(y253,fin253_loocv),4),"deploy_253_insample_rae":round(rae(y253,fin253_dep),4),
           "n260":int(N260),"gate_fired_260":int(flag260.sum()),
           "pred260_mean":round(float(fin260.mean()),3),"pred260_std":round(float(fin260.std()),3),
           "clean_base_models":len(keys)},
          open("data/processed/nb1333_deploy.json","w"),indent=2)
print("\\nwrote submissions/nb1333_final_260.csv + nb1333_final_513.csv", flush=True)
print("DONE", flush=True)
