"""nb1334 — executable, torch-free probe of the WATER/desolvation idea (Trick 3).

Full explicit-pocket-water (Lill GNN / GIST) needs GPU pose-regen + torch (poses not saved
locally). The runnable version: proper desolvation energetics from 3D conformers —
Eisenberg-McLachlan atomic solvation (ASP x SASA), buried polar/apolar SASA, a hydration
free-energy estimate. These are REAL hydration descriptors distinct from plain TPSA.
Gate honestly (corr-with-error, RAE) vs the deployed 0.5799 stack on the 253.
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore"); os.environ["PYTHONUNBUFFERED"]="1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pickle
from src.pxr.data import load_test
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA, Descriptors, Crippen
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
MS="C:/pxr_work/meta_stacking"

raw=pd.read_csv("C:/pxr_work/phase1_unblind/phase1_unblinded_raw.csv")
nc=next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc=next(c for c in raw.columns if "pec50" in c.lower())
raw=raw[[nc,pc]].dropna(); raw.columns=["name","pec50_true"]
te=load_test().reset_index(drop=True)
ub=te["name"].isin(set(raw["name"])); ub_idx=te.index[ub].tolist()
te_ub=te[ub].merge(raw,on="name").reset_index(drop=True)
y=te_ub["pec50_true"].to_numpy(float); N=len(y)

# Eisenberg-McLachlan atomic solvation parameters (cal/mol/A^2) by atom type
def desolv_features(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: return [np.nan]*6
    m=Chem.AddHs(m)
    if AllChem.EmbedMolecule(m,randomSeed=0xf00d)!=0:
        m2=Chem.MolFromSmiles(smi); return [Descriptors.TPSA(m2),Crippen.MolLogP(m2),0,0,0,0]
    AllChem.MMFFOptimizeMolecule(m)
    radii=rdFreeSASA.classifyAtoms(m)
    sasa=rdFreeSASA.CalcSASA(m,radii)
    # per-atom SASA
    asp={ "C":16,"S":21,"N":-6,"O":-6,"O-":-24,"N+":-50 }  # Eisenberg-McLachlan (cal/mol/A^2)
    polar_sasa=0.0; apolar_sasa=0.0; gsolv=0.0
    for a in m.GetAtoms():
        s=float(a.GetPropsAsDict().get("SASA",0.0))
        sym=a.GetSymbol(); ch=a.GetFormalCharge()
        if sym in ("N","O"):
            polar_sasa+=s
            key=sym+("-" if ch<0 else "")
            gsolv+=asp.get(key,asp.get(sym,0))*s
        elif sym in ("C","S"):
            apolar_sasa+=s; gsolv+=asp.get(sym,0)*s
    frac_polar=polar_sasa/max(sasa,1)
    return [sasa, polar_sasa, apolar_sasa, frac_polar, gsolv/1000.0, polar_sasa-apolar_sasa]

print("Computing desolvation/hydration descriptors (3D conformers)...", flush=True)
F=np.array([desolv_features(s) for s in te_ub["smiles"]])
# impute
med=np.nanmedian(F,0);
for j in range(F.shape[1]):
    F[np.isnan(F[:,j]),j]=med[j]
names=["sasa","polar_sasa","apolar_sasa","frac_polar","gsolv_EM","polar_minus_apolar"]
np.save(f"{MS}/desolv_253.npy", F)

# deployed stack
meta=np.load(f"{MS}/meta_stacker_loocv_253.npy").ravel()
knn=np.load(f"{MS}/knn_residual_loocv_253_correct.npy").ravel()
preds=pickle.load(open(f"{MS}/preds_513.pkl","rb")); comb=np.array(preds["combined_corrected"])[ub_idx]
boltz=np.load(f"{MS}/loocv_253_F5_base+boltz.npy").ravel()
best=0.40*meta+0.20*knn+0.10*comb+0.30*boltz
p_active=np.load(f"{MS}/sc_pactive_253.npy")
sh=best+0.10*(p_active-0.5)*2; ig=(p_active<0.15)&(boltz<3.2)&(sh<3.5); cur=np.where(ig,3.0,sh)
err=best-y

print("\\n=== Desolvation/water features: corr-with-TRUTH / corr-with-ERROR ===", flush=True)
for j,nm in enumerate(names):
    print(f"  {nm:<20} truth={np.corrcoef(F[:,j],y)[0,1]:+.3f}  error={np.corrcoef(F[:,j],err)[0,1]:+.3f}")

# Honest nested: add desolv block as a residual corrector on top of the deployed stack
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
Z=np.column_stack([cur,F])
allr=[]
for seed in range(15):
    kf=KFold(n_splits=5,shuffle=True,random_state=seed); oof=np.zeros(N)
    for tri,va in kf.split(Z):
        scl=StandardScaler().fit(Z[tri]); m=RidgeCV(alphas=np.logspace(-1,4,15)).fit(scl.transform(Z[tri]),y[tri])
        oof[va]=m.predict(scl.transform(Z[va]))
    allr.append(rae(y,oof))
print(f"\\n=== Honest RAE ===", flush=True)
print(f"  deployed stack:           {rae(y,cur):.4f}")
print(f"  Ridge[stack + desolv]:    {np.mean(allr):.4f}+-{np.std(allr):.4f}")
json.dump({"deployed":round(rae(y,cur),4),"with_desolv":round(float(np.mean(allr)),4),
           "corr_error":{names[j]:round(float(np.corrcoef(F[:,j],err)[0,1]),3) for j in range(len(names))}},
          open("data/processed/nb1334_water.json","w"),indent=2)
print("DONE", flush=True)
