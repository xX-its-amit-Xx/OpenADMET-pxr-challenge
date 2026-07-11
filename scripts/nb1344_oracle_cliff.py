"""nb1344 — post-hoc oracle ceiling + cliff-detectability (is the wall reducible?).

With the 260 truth known, ask:
 (A) ORACLE: best RAE achievable from our components (optimal blend, optimal calibration).
 (B) CLIFF-DETECTOR: can any feature combo PREDICT which compounds are cliffs (CV AUC)?
     If yes -> abstention/gating is viable. If no -> the wall is truly irreducible.
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
MS="C:/pxr_work/meta_stacking"; OUT="C:/pxr_work/posthoc"
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())

y=np.load(f"{OUT}/y260.npy")
bl_idx=np.load(f"{MS}/_blinded_idx.npy"); ub_idx=np.load(f"{MS}/_unblinded_idx.npy")
comb=np.load(f"{MS}/combined_corrected_513.npy").ravel()[bl_idx]
boltz=np.load(f"{OUT}/boltz260.npy"); best=np.load(f"{OUT}/best260.npy")
meta=np.load(f"{MS}/meta_stacker_te_260.npy").ravel(); knn=np.load(f"{MS}/knn_residual_loocv_te_260_v2.npy").ravel()
P=np.load(f"{OUT}/P260.npy"); deploy=np.load(f"{OUT}/deploy260.npy")

# all base models (260)
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb"))
y253=np.load(f"{MS}/_y260_truth.npy")  # placeholder; recompute 253 truth
import pandas as _pd
t1=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
from src.pxr.data import load_test
te=load_test().reset_index(drop=True)
y253=np.array([dict(zip(t1["Molecule Name"],t1["pEC50"]))[n] for n in te.loc[ub_idx,"name"]])
COMPS={}
for k,v in preds.items():
    a=np.asarray(v).ravel().astype(float)
    if len(a)<513: continue
    if np.abs(a[ub_idx]-y253).mean()<0.05: continue
    COMPS[k]=a[bl_idx]
COMPS.update({"meta":meta,"knn":knn,"comb":comb,"boltz":boltz,"best_blend":best})
names=list(COMPS); M=np.column_stack([COMPS[k] for k in names])

print("=== (A) POST-HOC ORACLE CEILING (260) ===")
print(f"  our deployed:                 {rae(y,deploy):.4f}")
print(f"  best single component:        {min(rae(y,COMPS[k]) for k in names):.4f} ({min(names,key=lambda k:rae(y,COMPS[k]))})")
# optimal convex blend (SLSQP) -- oracle upper bound
def obj(w): return rae(y, M@w)
w0=np.ones(len(names))/len(names)
cons=[{"type":"eq","fun":lambda w:w.sum()-1}]; bnds=[(0,1)]*len(names)
res=minimize(obj,w0,constraints=cons,bounds=bnds,method="SLSQP")
print(f"  optimal convex blend (oracle): {rae(y,M@res.x):.4f}")
# optimal isotonic calibration on best component (oracle)
bestk=min(names,key=lambda k:rae(y,COMPS[k])); bp=COMPS[bestk]
iso=IsotonicRegression(out_of_bounds="clip").fit(bp,y)
print(f"  + oracle isotonic on best:     {rae(y,iso.predict(bp)):.4f}")
# oracle: if inactives perfectly predicted
di=deploy.copy(); m_in=y<3.5; di[m_in]=y[m_in]
print(f"  + oracle: perfect inactives:   {rae(y,di):.4f} (headroom from the 28 inactives)")
print(f"  MAE oracle blend={mae(y,M@res.x):.4f} vs LB tie cluster 0.40-0.43")

# === (B) CLIFF DETECTABILITY ===
print("\\n=== (B) CAN WE PREDICT THE CLIFFS? (CV on 260) ===")
df=pd.read_csv("data/processed/posthoc/percompound_260.csv").set_index("name").loc[te.loc[bl_idx,"name"].values]
is_cliff=df["is_cliff"].values.astype(int)
# disagreement among base models (a natural cliff signal)
disagree=M.std(1)
feats=np.column_stack([P, boltz, df["top1_sim"].values, df["nbr_pec50"].values, disagree,
                       df["mw"].values, df["logp"].values, df["tpsa"].values, comb, best])
skf=StratifiedKFold(5,shuffle=True,random_state=0)
clf=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05)
oof=cross_val_predict(clf,feats,is_cliff,cv=skf,method="predict_proba")[:,1]
auc=roc_auc_score(is_cliff,oof)
print(f"  cliff-detector CV AUC (26 cliffs / 260): {auc:.3f}")
# individual signals
for nm,f in [("P_active",P),("boltz",boltz),("top1_sim",df['top1_sim'].values),("disagreement",disagree),("nbr_pec50",df['nbr_pec50'].values)]:
    try: print(f"    single-signal AUC {nm:<14}: {roc_auc_score(is_cliff,f if np.corrcoef(f,is_cliff)[0,1]>0 else -f):.3f}")
    except: pass
# if we COULD detect cliffs (oracle), floor them:
d_cliff=deploy.copy(); d_cliff[is_cliff.astype(bool)]=y[is_cliff.astype(bool)]
print(f"  IF cliffs perfectly floored (oracle): RAE={rae(y,d_cliff):.4f}")
json.dump({"our_deploy":rae(y,deploy),"best_single":min(rae(y,COMPS[k]) for k in names),
           "oracle_blend":rae(y,M@res.x),"oracle_iso":rae(y,iso.predict(bp)),
           "cliff_auc":auc,"n_cliff":int(is_cliff.sum())}, open(f"{OUT}/oracle_cliff.json","w"),indent=2)
print("\\nsaved oracle_cliff.json")
