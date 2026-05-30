"""nb217 -- PXR binding mode clustering.

Hypothesis: PXR is promiscuous with multiple distinct binding modes (steroid-like,
hydrophobic, polar-anchor, etc.). A single global model averages over modes;
mode-aware features could let the downstream model learn mode-specific patterns.

Approach:
1. PCA(64) on combined morgan + rdkit features
2. KMeans (k in {6, 8, 10, 12}) - try multiple k values to add diversity
3. For each compound: distance to each centroid -> softmax -> mode probabilities
4. For each cluster: mean training pEC50 (centroid activity)
5. Features per compound: mode_probs (k features) + nearest_centroid_pec50 + cluster_pec50_var

Train LGBM on (combined + mode features). Save OOF + test.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.featurize import combined as feat_combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
SEED = 42
COLLAPSE_THRESH = 0.58
PREV_BEST = 0.296172
K_VALUES = [6, 8, 10, 12]   # multiple k for ensemble diversity

t0 = time.time()


def mode_features(X_tr_pca, X_te_pca, y_tr_subset, k, seed=SEED):
    """Returns (mode_tr_features, mode_te_features) for a given k.

    mode features per compound: k softmax probs + cluster_mean_pec50 + cluster_pec50_std.
    """
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(X_tr_pca)

    # Distances to each centroid
    d_tr = km.transform(X_tr_pca)  # (n, k)
    d_te = km.transform(X_te_pca)

    # Convert to softmax probabilities (negative distance / temperature)
    # Use 1/distance² as similarity, normalize
    sim_tr = 1.0 / (1.0 + d_tr ** 2)
    sim_te = 1.0 / (1.0 + d_te ** 2)
    prob_tr = sim_tr / sim_tr.sum(axis=1, keepdims=True)
    prob_te = sim_te / sim_te.sum(axis=1, keepdims=True)

    # Cluster activity stats (mean + std of train compounds in each cluster)
    labels_tr = km.labels_
    cluster_pec_mean = np.zeros(k)
    cluster_pec_std = np.zeros(k)
    for c in range(k):
        ms = labels_tr == c
        if ms.sum() > 0:
            cluster_pec_mean[c] = y_tr_subset[ms].mean()
            cluster_pec_std[c] = y_tr_subset[ms].std() if ms.sum() > 1 else 0.0

    # Per-compound aggregated cluster activity (weighted by prob)
    weighted_pec_tr = (prob_tr * cluster_pec_mean[None, :]).sum(axis=1, keepdims=True)
    weighted_pec_te = (prob_te * cluster_pec_mean[None, :]).sum(axis=1, keepdims=True)
    weighted_std_tr = (prob_tr * cluster_pec_std[None, :]).sum(axis=1, keepdims=True)
    weighted_std_te = (prob_te * cluster_pec_std[None, :]).sum(axis=1, keepdims=True)

    feat_tr = np.hstack([prob_tr, weighted_pec_tr, weighted_std_tr])
    feat_te = np.hstack([prob_te, weighted_pec_te, weighted_std_te])
    return feat_tr, feat_te


def main():
    print("=== nb217: PXR binding mode clustering ===\n", flush=True)
    tr_df = load_train()
    te_df = load_test()
    print(f"Train: {len(tr_df)}, Test: {len(te_df)}\n", flush=True)

    y_pec = tr_df["pec50"].values.astype(np.float64)
    n_tr = len(tr_df)

    scaffolds = tr_df["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Computing base features...", flush=True)
    X_tr = impute(feat_combined(tr_df["smiles"].tolist())).astype(np.float64)
    X_te = impute(feat_combined(te_df["smiles"].tolist())).astype(np.float64)
    print(f"  base shape: {X_tr.shape} ({time.time()-t0:.0f}s)", flush=True)

    print("PCA(64) ...", flush=True)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    pca = PCA(n_components=64, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)
    print(f"  explained variance: {pca.explained_variance_ratio_.sum():.3f} ({time.time()-t0:.0f}s)", flush=True)

    # Build mode features for each k value
    # Important: cluster on TRAIN ONLY (not test) - then transform test
    # For train OOF: use fold-aware clustering (cluster on tr_idx only, transform va_idx)
    print("\nBuilding mode features (multi-k, fold-aware for train)...", flush=True)
    mode_feat_tr_per_k = []
    mode_feat_te_per_k = []
    for k in K_VALUES:
        print(f"  k={k}...", flush=True)
        # Fold-aware train mode features
        feat_tr_k = np.zeros((n_tr, k + 2))
        for fi, (tr_idx, va_idx) in enumerate(splits):
            X_sub_pca = X_tr_pca[tr_idx]
            y_sub = y_pec[tr_idx]
            X_va_pca = X_tr_pca[va_idx]
            # Reuse mode_features but with custom inputs
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            km.fit(X_sub_pca)
            labels = km.labels_
            cluster_mean = np.zeros(k)
            cluster_std = np.zeros(k)
            for c in range(k):
                ms = labels == c
                if ms.sum() > 0:
                    cluster_mean[c] = y_sub[ms].mean()
                    cluster_std[c] = y_sub[ms].std() if ms.sum() > 1 else 0.0
            d_va = km.transform(X_va_pca)
            sim_va = 1.0 / (1.0 + d_va ** 2)
            prob_va = sim_va / sim_va.sum(axis=1, keepdims=True)
            wpec = (prob_va * cluster_mean[None, :]).sum(axis=1, keepdims=True)
            wstd = (prob_va * cluster_std[None, :]).sum(axis=1, keepdims=True)
            feat_tr_k[va_idx] = np.hstack([prob_va, wpec, wstd])
        # Test mode features: cluster on full train
        _, feat_te_k = mode_features(X_tr_pca, X_te_pca, y_pec, k)
        mode_feat_tr_per_k.append(feat_tr_k)
        mode_feat_te_per_k.append(feat_te_k)
        print(f"    k={k} done ({time.time()-t0:.0f}s)", flush=True)

    mode_feat_tr = np.hstack(mode_feat_tr_per_k)
    mode_feat_te = np.hstack(mode_feat_te_per_k)
    print(f"\n  mode features total dim: {mode_feat_tr.shape[1]}", flush=True)

    # Augmented feature matrix
    X_tr_full = np.hstack([X_tr, mode_feat_tr]).astype(np.float32)
    X_te_full = np.hstack([X_te, mode_feat_te]).astype(np.float32)
    print(f"  augmented shape: {X_tr_full.shape}\n", flush=True)

    print("Training LGBM (5-fold scaffold CV)...", flush=True)
    oof = np.full(n_tr, np.nan)
    te_pred = np.zeros(len(te_df))

    for fi, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(
            n_estimators=2000, num_leaves=64, learning_rate=0.03,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, objective="regression_l1",
            random_state=SEED, verbose=-1,
        )
        m.fit(
            X_tr_full[tr_idx], y_pec[tr_idx],
            eval_set=[(X_tr_full[va_idx], y_pec[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = m.predict(X_tr_full[va_idx])
        te_pred += m.predict(X_te_full) / N_FOLDS
        print(f"  fold {fi+1}/{N_FOLDS}: best_iter={m.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    r = rae(y_pec, oof)
    ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***BEATS PREV BEST***" if (ratio >= COLLAPSE_THRESH and r < PREV_BEST) else ""
    print(f"\n=== nb217 OOF: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat} ===", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

    out_stem = "nb217_pxr_modes"
    np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{out_stem}.npy", te_pred)
    sub = pd.DataFrame({
        "SMILES": te_df["smiles"].values,
        "Molecule Name": te_df["name"].values,
        "pEC50": te_pred,
    })
    sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
    print(f"Saved: {out_stem}", flush=True)


if __name__ == "__main__":
    main()
