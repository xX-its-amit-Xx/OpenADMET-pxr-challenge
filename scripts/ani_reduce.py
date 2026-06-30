"""Reduce ANI-2x raw features -> deployable block.

Mean-pooled 1008-d AEV -> PCA 24 comps (unsupervised, like SOAP/CheMeleon/TabPFN
blocks), plus the NNP energy + energy-per-atom scalars. Writes
C:/pxr_work/ani/ani_features.csv (name, src, ani_energy, ani_epa, ani_0..ani_23).
Unsupported / failed mols get NaN -> imputed in the gate.
"""
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT = "C:/pxr_work/ani"
z = np.load(f"{OUT}/ani_raw.npz", allow_pickle=True)
names = z["names"]; aevs = z["aevs"].astype(float); en = z["energy"]; epa = z["epa"]; uns = z["unsupported"]
corpus = pd.read_csv("C:/pxr_work/xtb/corpus.csv").set_index("name")
src = corpus.reindex(names)["src"].values

ok = ~uns & np.isfinite(en) & (np.abs(aevs).sum(1) > 0)
print(f"total {len(names)} | usable {int(ok.sum())} | unsupported/failed {int((~ok).sum())}")

# standardize AEV on usable rows, PCA->24
sc = StandardScaler().fit(aevs[ok])
A = sc.transform(aevs)
pca = PCA(n_components=24, random_state=0).fit(A[ok])
comps = pca.transform(A)
print(f"PCA expl-var(24) = {pca.explained_variance_ratio_.sum():.3f}")

df = pd.DataFrame({"name": names, "src": src})
df["ani_energy"] = en
df["ani_epa"] = epa
for i in range(24):
    df[f"ani_{i}"] = comps[:, i]
# blank out non-usable rows -> NaN (gate imputes median)
bad = ~ok
for c in ["ani_energy", "ani_epa"] + [f"ani_{i}" for i in range(24)]:
    df.loc[bad, c] = np.nan
df.to_csv(f"{OUT}/ani_features.csv", index=False)
print(f"wrote {OUT}/ani_features.csv  rows={len(df)} bad={int(bad.sum())}")
print(df[df.src == 'train'].shape, df[df.src == 'test'].shape)
