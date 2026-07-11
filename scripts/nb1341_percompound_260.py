"""nb1341 — per-compound + per-family error analysis for all 260 blind compounds."""
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.pxr.data import load_test, load_train
from src.pxr.chem import bemis_murcko, standardize, compute_physchem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
MS="C:/pxr_work/meta_stacking"; OUT="C:/pxr_work/posthoc"
def rae(yt,yp): return float(np.abs(yt-yp).sum()/np.abs(yt-np.median(yt)).sum())
def mae(yt,yp): return float(np.abs(yt-yp).mean())

te=load_test().reset_index(drop=True)
bl_idx=np.load(f"{MS}/_blinded_idx.npy")
sm_col=[c for c in te.columns if "smile" in c.lower()][0]
names=te.loc[bl_idx,"name"].values; smis=te.loc[bl_idx,sm_col].values
t2=pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
truth=dict(zip(t2["Molecule Name"],t2["pEC50"]))
emax=dict(zip(t2["Molecule Name"],t2["Emax_estimate (log2FC vs. baseline)"]))
se=dict(zip(t2["Molecule Name"],t2["pEC50_std.error (-log10(molarity))"]))
y=np.array([truth[n] for n in names])
pred=np.load(f"{OUT}/deploy260.npy"); P=np.load(f"{OUT}/P260.npy"); boltz=np.load(f"{OUT}/boltz260.npy")

# scaffolds + novelty + nearest-train-neighbor (cliff detection)
tr=load_train()
def scaf(s):
    m=standardize(s); return bemis_murcko(m) if m else ""
tr_scaf=set(scaf(s) for s in tr["smiles"])
tr_bv=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in tr["smiles"]]
te_bv=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in smis]
tr_pec=tr["pec50"].to_numpy()

rows=[]
for i,(nm,sm) in enumerate(zip(names,smis)):
    sims=np.array(DataStructs.BulkTanimotoSimilarity(te_bv[i],tr_bv))
    top=np.argsort(sims)[::-1][:3]
    sc=scaf(sm)
    pcm=compute_physchem(sm)
    err=pred[i]-y[i]
    nbr_pec=tr_pec[top].mean()
    rows.append(dict(name=nm, smiles=sm, truth=round(float(y[i]),2), pred=round(float(pred[i]),2),
        error=round(float(err),2), abs_error=round(float(abs(err)),2),
        emax=round(float(emax[nm]),2) if nm in emax else None,
        se=round(float(se[nm]),3) if nm in se else None,
        tier=("inactive" if y[i]<3.5 else "mid" if y[i]<5 else "active"),
        P_active=round(float(P[i]),3), boltz=round(float(boltz[i]),2),
        top1_sim=round(float(sims[top[0]]),3), nbr_pec50=round(float(nbr_pec),2),
        novel_scaffold=bool(sc not in tr_scaf and sc!=""),
        is_cliff=bool(sims[top[0]]>=0.45 and nbr_pec>=4.5 and y[i]<3.5),  # inactive but active neighbors
        mw=round(pcm["mw"],1), logp=round(pcm["logp"],2), tpsa=round(pcm["tpsa"],1), scaffold=sc))
df=pd.DataFrame(rows)
os.makedirs("data/processed/posthoc", exist_ok=True)
df.sort_values("abs_error",ascending=False).to_csv("data/processed/posthoc/percompound_260.csv", index=False)

# ---- Butina family clustering ----
dists=[]
for i in range(1,len(te_bv)):
    s=DataStructs.BulkTanimotoSimilarity(te_bv[i],te_bv[:i]); dists.extend([1-x for x in s])
clusters=Butina.ClusterData(dists,len(te_bv),0.6,isDistData=True)
fam=np.zeros(len(te_bv),int)
for ci,cl in enumerate(clusters):
    for idx in cl: fam[idx]=ci
df["family"]=fam
from collections import Counter
cnt=Counter(fam)
fam_stats=[]
for c,n in cnt.most_common():
    m=fam==c
    if n<4: continue
    fam_stats.append(dict(family=int(c), n=int(n),
        RAE=round(rae(y[m],pred[m]),3), MAE=round(mae(y[m],pred[m]),3),
        bias=round(float((pred[m]-y[m]).mean()),3),
        truth_range=f"[{y[m].min():.1f},{y[m].max():.1f}]",
        example=df[m].iloc[0]["name"]))
fam_df=pd.DataFrame(sorted(fam_stats,key=lambda d:-d["MAE"]))
fam_df.to_csv("data/processed/posthoc/family_260.csv", index=False)

print("=== WORST 15 COMPOUNDS (by abs error) ===")
print(df.sort_values("abs_error",ascending=False)[["name","truth","pred","error","tier","P_active","top1_sim","nbr_pec50","is_cliff","novel_scaffold"]].head(15).to_string(index=False))
print(f"\\n=== cliffs: {df['is_cliff'].sum()} | novel-scaffold: {df['novel_scaffold'].sum()}/{len(df)} ===")
print(f"\\n=== WORST 10 FAMILIES (>=4 cpds, by MAE) ===")
print(fam_df.head(10).to_string(index=False))
# summary stats
big=df[df["abs_error"]>=1.0]
print(f"\\n=== {len(big)} compounds with |error|>=1.0 ===")
print(f"  of these: {big['tier'].value_counts().to_dict()}")
print(f"  over-predicted (err>0): {(big['error']>0).sum()}, under: {(big['error']<0).sum()}")
print(f"  novel-scaffold among big errors: {big['novel_scaffold'].sum()}/{len(big)}")
json.dump({"n260":len(df),"cliffs":int(df["is_cliff"].sum()),"novel":int(df["novel_scaffold"].sum()),
           "n_big_error":int(len(big)),"worst_family_mae":float(fam_df.iloc[0]["MAE"]) if len(fam_df) else None},
          open(f"{OUT}/percompound_summary.json","w"),indent=2)
print("\\nsaved data/processed/posthoc/percompound_260.csv + family_260.csv")
