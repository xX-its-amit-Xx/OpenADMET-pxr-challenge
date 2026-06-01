"""nb551: Cluster the top-50 failure tail (from nb550) to find a coherent failure mode.

Inputs:
    data/processed/nb550_failure_tail.csv

Outputs:
    data/processed/nb551_failure_clusters.csv  -- per-row cluster assignment + features
    stdout: cluster sizes + characterization + coherence verdict
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "nb550_failure_tail.csv"
OUT = ROOT / "data" / "processed" / "nb551_failure_clusters.csv"

# 10 numeric features used for clustering
FEATS = [
    "truth_pec50",
    "nb503_pred",
    "error",
    "mw",
    "logp",
    "tpsa",
    "fsp3",
    "rotbonds",
    "top1_sim_train",
    "predictor_disagreement",
]


def main() -> None:
    df = pd.read_csv(IN)
    assert all(c in df.columns for c in FEATS), f"missing cols: {set(FEATS)-set(df.columns)}"
    n = len(df)
    print(f"loaded {n} rows from {IN.name}")

    X = df[FEATS].to_numpy(dtype=float)
    # impute any NaNs with column median
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

    Xs = StandardScaler().fit_transform(X)

    # K selection via silhouette + elbow (inertia)
    results = {}
    for k in (2, 3, 4, 5):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        results[k] = {"km": km, "sil": sil, "inertia": km.inertia_}
        print(f"  k={k}: silhouette={sil:.3f}  inertia={km.inertia_:.1f}")

    best_k = max(results, key=lambda k: results[k]["sil"])
    print(f"\nbest k by silhouette = {best_k}")
    km = results[best_k]["km"]
    df["cluster"] = km.labels_

    # cluster characterization
    print("\n=== cluster characterization ===")
    sizes = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        sizes.append(len(sub))
        # dominant scaffold (if scaf col exists)
        scaf_mode = sub["scaffold"].mode().iloc[0] if "scaffold" in sub.columns and len(sub) else ""
        scaf_top_n = (sub["scaffold"] == scaf_mode).sum() if scaf_mode else 0
        n_unique_scaf = sub["scaffold"].nunique() if "scaffold" in sub.columns else 0
        print(f"\ncluster {c}: n={len(sub)}")
        print(f"  median truth_pec50 = {sub['truth_pec50'].median():.2f}")
        print(f"  median pred_pec50  = {sub['nb503_pred'].median():.2f}")
        print(f"  median error       = {sub['error'].median():.2f}")
        print(f"  mean MW            = {sub['mw'].mean():.1f}")
        print(f"  mean logP          = {sub['logp'].mean():.2f}")
        print(f"  mean TPSA          = {sub['tpsa'].mean():.1f}")
        print(f"  mean fsp3          = {sub['fsp3'].mean():.2f}")
        print(f"  mean rotbonds      = {sub['rotbonds'].mean():.1f}")
        print(f"  mean top1_sim_train= {sub['top1_sim_train'].mean():.3f}")
        print(f"  mean disagreement  = {sub['predictor_disagreement'].mean():.3f}")
        print(f"  n_unique_scaffolds = {n_unique_scaf}")
        print(f"  dominant scaffold  = {scaf_mode!r} (count={scaf_top_n})")

    print(f"\ncluster sizes: {sizes}")
    print(f"largest cluster size = {max(sizes)} / {n} ({100*max(sizes)/n:.0f}%)")

    # coherence verdict: does ANY cluster contain >= 30 of the 50?
    max_size = max(sizes)
    coherent = max_size >= 30
    print(f"\nCOHERENT? {coherent}  (threshold: largest cluster >= 30)")

    df.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")
    print(f"best_k={best_k}  cluster_sizes={sizes}  coherent={coherent}")


if __name__ == "__main__":
    main()
