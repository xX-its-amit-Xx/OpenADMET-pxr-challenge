"""PCA-compress the raw averaged SOAP vectors to a handful of scalars.

SOAP is 2244-d; the GBM blocks that win marginally (AIMNet2 9, strain 8) are a
handful of scalars. Reduce SOAP to K PCA comps (unsupervised, like the CheMeleon
/ TabPFN embedding blocks already in the deployed ensemble) so it gates as a
comparable compact physics block. Writes soap_pca.csv aligned to corpus.csv.

Run after soap_features.py:  .venv/Scripts/python.exe scripts/soap_reduce.py
"""
import os
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUTDIR = "C:/pxr_work/soap"
NPZ = os.path.join(OUTDIR, "soap_raw.npz")
META = os.path.join(OUTDIR, "soap_meta.csv")
OUT = os.path.join(OUTDIR, "soap_pca.csv")
K = 24


def main():
    d = np.load(NPZ, allow_pickle=True)
    names = list(d["names"]); X = d["X"].astype(float)
    meta = pd.read_csv(META).drop_duplicates(subset="name", keep="last").set_index("name")

    finite = np.isfinite(X).all(axis=1)
    med = np.nanmedian(X[finite], axis=0)
    inds = np.where(~np.isfinite(X)); X[inds] = np.take(med, inds[1])
    # drop zero-variance columns before PCA
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    k = min(K, Xs.shape[1])
    pca = PCA(n_components=k, random_state=0).fit(Xs)
    Z = pca.transform(Xs)
    print(f"SOAP {X.shape} -> PCA {Z.shape}  explained_var={pca.explained_variance_ratio_.sum():.3f}  err_rows={int((~finite).sum())}")

    cols = [f"soap_{i}" for i in range(k)]
    df = pd.DataFrame(Z, columns=cols)
    df.insert(0, "name", names)
    df["src"] = [meta.loc[n, "src"] if n in meta.index else "" for n in names]
    df["smiles"] = [meta.loc[n, "smiles"] if n in meta.index else "" for n in names]
    df = df[["name", "src", "smiles"] + cols]
    df.to_csv(OUT, index=False)
    print(f"saved {OUT}  ({len(df)} rows, {k} comps)")


if __name__ == "__main__":
    main()
