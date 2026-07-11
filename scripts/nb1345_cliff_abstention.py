"""nb1345 — HONEST cliff-abstention + calibration + residual correctors.

The oracle showed cliffs are ~80% detectable (Boltz AUC 0.84). Build honest models:
train on the 253, apply to the 260 (different series - real transfer test).
 (1) learned cliff-detector -> floor detected cliffs
 (2) honest isotonic calibration (fit 253, apply 260)
 (3) honest residual-GBM corrector (fit 253, apply 260)
Does any BEAT combined_corrected (0.6318) on the blind 260?
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from src.pxr.data import load_test, load_train
from src.pxr.chem import morgan_fp_batch, standardize, bemis_murcko, compute_physchem
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
MS="C:/pxr_work/meta_stacking"; OUT="C:/pxr_work/posthoc"
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())

te=load_test().reset_index(drop=True)
bl=np.load(f"{MS}/_blinded_idx.npy"); ub=np.load(f"{MS}/_unblinded_idx.npy")
t1=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"); t2=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
tru={**dict(zip(t1["Molecule Name"],t1["pEC50"])),**dict(zip(t2["Molecule Name"],t2["pEC50"]))}
y253=np.array([tru[n] for n in te.loc[ub,"name"]]); y260=np.array([tru[n] for n in te.loc[bl,"name"]])

# predictions
comb=np.load(f"{MS}/combined_corrected_513.npy").ravel(); comb253=comb[ub]; comb260=comb[bl]
boltz253=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel(); boltz260=np.load(f"{OUT}/boltz260.npy")
P253=np.load(f"{MS}/sc_pactive_253.npy"); P260=np.load(f"{OUT}/P260.npy")
deploy260=np.load(f"{OUT}/deploy260.npy")

# feature builder (per compound): P, boltz, comb, physchem, nbr disagreement proxy
tr=load_train(); tr_bv=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in tr["smiles"]]; tr_pec=tr["pec50"].to_numpy()
sm_col=[c for c in te.columns if "smile" in c.lower()][0]
def feats(idxs,P,boltz,comb):
    smis=te.loc[idxs,sm_col].values; bv=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in smis]
    top1=[];nbr=[];pc=[]
    for i,b in enumerate(bv):
        s=np.array(DataStructs.BulkTanimotoSimilarity(b,tr_bv)); t=np.argsort(s)[::-1][:5]
        top1.append(s[t[0]]); nbr.append(tr_pec[t].mean())
        p=compute_physchem(smis[i]); pc.append([p["mw"],p["logp"],p["tpsa"],p["hbd"],p["fsp3"]])
    pc=np.array(pc)
    return np.column_stack([P,boltz,comb,np.array(top1),np.array(nbr),pc])
F253=feats(te.loc[ub].index.values,P253,boltz253,comb253)
F260=feats(te.loc[bl].index.values,P260,boltz260,comb260)

print("=== HONEST (train 253 -> apply 260). base comb260=%.4f, our deploy=%.4f ===" % (rae(y260,comb260),rae(y260,deploy260)))
R={}
# (1) cliff-detector: cliff = over-predicted-inactive on 253 (residual>1 AND truth<3.5)
resid253=comb253-y253
cliff253=((resid253>0.8)&(y253<4.0)).astype(int)
print(f"  253 cliff-label rate: {cliff253.mean()*100:.0f}% ({cliff253.sum()})")
clf=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05).fit(F253,cliff253)
pcliff=clf.predict_proba(F260)[:,1]
# floor high-cliff-prob compounds toward inactive
for thr,floor in [(0.5,3.0),(0.4,3.2),(0.6,2.8),(0.5,2.5)]:
    c=comb260.copy(); m=pcliff>thr; c[m]=np.minimum(c[m],floor)
    R[f"comb + cliff-floor(p>{thr},{floor})"]=rae(y260,c)
# (2) honest isotonic
iso=IsotonicRegression(out_of_bounds="clip").fit(comb253,y253)
R["comb + honest isotonic(253)"]=rae(y260,iso.predict(comb260))
# (3) honest residual-GBM corrector
gb=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=3).fit(F253,y253-comb253)
R["comb + honest residual-GBM"]=rae(y260,comb260+gb.predict(F260))
# (4) direct GBM on features -> pEC50 (train 253)
gb2=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05).fit(F253,y253)
R["direct GBM on cliff-feats(253)"]=rae(y260,gb2.predict(F260))
# (5) blend comb with cliff-shift (soft)
sh=comb260 - 1.0*np.maximum(0,pcliff-0.5)*2
R["comb + soft cliff-shift"]=rae(y260,sh)

print("\\n=== RESULTS on blind 260 (base comb 0.6318) ===")
for k,v in sorted(R.items(),key=lambda kv:kv[1]):
    print(f"  {v:.4f}  ({v-rae(y260,comb260):+.4f})  {k}")
# cliff-detector transfer AUC (253-trained on 260 true cliffs)
from sklearn.metrics import roc_auc_score
tc260=((comb260-y260>0.8)&(y260<4.0)).astype(int)
print(f"\\n  cliff-detector 253->260 transfer AUC: {roc_auc_score(tc260,pcliff):.3f} (260 true over-pred-inactive n={tc260.sum()})")
json.dump({k:round(v,4) for k,v in R.items()}|{"comb":round(rae(y260,comb260),4),"deploy":round(rae(y260,deploy260),4),
           "transfer_auc":round(roc_auc_score(tc260,pcliff),3)}, open(f"{OUT}/cliff_abstention.json","w"),indent=2)
print("DONE")
