"""nb1343 — retrospective: on the now-known 260, were our method decisions right?

Base = combined_corrected (the most robust component, 0.6318 on 260).
Test each lever we made a call on: single-conc corrections, broad-public-KB, desolvation.
Also the counterfactual: apply single-conc corrections to comb (not the overweighted blend).
"""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.pxr.data import load_test, load_single_conc
from src.pxr.featurize import combined, impute
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
MS="C:/pxr_work/meta_stacking"; OUT="C:/pxr_work/posthoc"
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())

te=load_test().reset_index(drop=True)
bl_idx=np.load(f"{MS}/_blinded_idx.npy")
t2=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
truth=dict(zip(t2["Molecule Name"],t2["pEC50"]))
y=np.array([truth[n] for n in te.loc[bl_idx,"name"]])
smis=te.loc[bl_idx,[c for c in te.columns if "smile" in c.lower()][0]].values

comb=np.load(f"{MS}/combined_corrected_513.npy").ravel()[bl_idx]
deploy=np.load(f"{OUT}/deploy260.npy"); best=np.load(f"{OUT}/best260.npy")
P=np.load(f"{OUT}/P260.npy"); boltz=np.load(f"{OUT}/boltz260.npy")
Xte=impute(combined(smis.tolist()))

print("=== BASELINES on 260 ===")
print(f"  combined_corrected (best component): {rae(y,comb):.4f}")
print(f"  our deployed stack:                  {rae(y,deploy):.4f}")
print(f"  base blend (no corr):                {rae(y,best):.4f}")

R={}
# (1) single-conc corrections applied to comb (the counterfactual)
sh=comb+0.10*(P-0.5)*2
gate=(P<0.15)&(boltz<3.2)&(sh<3.5); comb_sc=np.where(gate,3.0,sh)
R["comb + single-conc shift+gate"]=rae(y,comb_sc)

# (2) broad-KB skipped (parquets lost in wipe; already shown NEGATIVE on 253 nb1332)


# (3) desolvation corrector
def desolv(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: return 0.0
    m=Chem.AddHs(m)
    if AllChem.EmbedMolecule(m,randomSeed=0xf00d)!=0: return 0.0
    AllChem.MMFFOptimizeMolecule(m); r=rdFreeSASA.classifyAtoms(m); rdFreeSASA.CalcSASA(m,r)
    asp={"C":16,"S":21,"N":-6,"O":-6}; g=0.0
    for a in m.GetAtoms():
        s=float(a.GetPropsAsDict().get("SASA",0.0)); g+=asp.get(a.GetSymbol(),0)*s
    return g/1000.0
D=np.array([desolv(s) for s in smis]); D=(D-np.nanmean(D))/(np.nanstd(D)+1e-9)
best_d=rae(y,comb);
for w in np.arange(-0.4,0.41,0.1):
    best_d=min(best_d, rae(y, comb+w*D))
R["comb + desolvation (best w)"]=best_d

print("\\n=== RETROSPECTIVE: did each lever help on the BLIND 260? (base comb 0.6318) ===")
for k,v in sorted(R.items(),key=lambda kv:kv[1]):
    d=v-rae(y,comb); print(f"  {v:.4f}  ({d:+.4f})  {k}")
print("\\n  Interpretation: negative delta = would have helped; positive = our rejection was right.")

# Best achievable MAE for context vs LB tie cluster (0.40-0.426)
print(f"\\n  MAE: comb={mae(y,comb):.4f}  comb+sc={mae(y,comb_sc):.4f}  our_deploy={mae(y,deploy):.4f}  (LB tie cluster ~0.40-0.43)")
json.dump({k:round(v,4) for k,v in R.items()}|{"comb_base":round(rae(y,comb),4),"our_deploy":round(rae(y,deploy),4)},
          open(f"{OUT}/retrospective.json","w"),indent=2)
np.save(f"{OUT}/comb_sc_260.npy", comb_sc)
print("DONE")
