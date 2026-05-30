"""nb132 -- Tanimoto-expanded training: leverage public-data NEIGHBORS of our
compounds even when our compounds themselves aren't in public DBs.

Background: only 12/4139 training compounds appear in ChEMBL by InChIKey
(nb218). But each training compound likely has structural NEIGHBORS in the
~20k Papyrus dataset with measured activity for PXR-related targets.

Strategy:
  1. Build reference fingerprint matrix from Papyrus (20k compounds, 41 targets)
  2. For each PXR train+test compound, find Tanimoto top-k (k=5, min_sim=0.30) neighbors
  3. For each compound, build a 41-dim "neighbor target profile" vector:
        feat[t] = similarity-weighted avg of neighbors' measured t-activity
        (NaN if no neighbor has measured t)
  4. Use as new LGBM features (added to base 2265 Morgan+RDKit)
  5. Save as ensemble candidate; integrate into SLSQP

This is read-across at scale: leveraging public bioactivity for compounds
similar to ours, when ours themselves aren't catalogued.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import lightgbm as lgb
from scipy.stats import spearmanr

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)


def morgan_fp(smi, radius=2, nbits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nbits)


def main():
    print("=== nb132: Tanimoto-expanded training ===\n")

    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    base_rae = rae(y_tr, oof_base)
    print(f"Base nb197: OOF RAE={base_rae:.4f}")

    # ── Load Papyrus reference ────────────────────────────────────────────────
    papy_path = DATA_EXTERNAL / "papyrus_pxr_related_filtered.parquet"
    if not papy_path.exists():
        print(f"Missing: {papy_path}")
        return
    papy = pd.read_parquet(papy_path)
    smi_col = "SMILES" if "SMILES" in papy.columns else "SMILES_Stripped"
    val_col = "pchembl_value_Mean" if "pchembl_value_Mean" in papy.columns else "pchembl_value"
    print(f"Papyrus: {len(papy):,} records, {papy['accession'].nunique()} targets, "
          f"{papy[smi_col].nunique()} unique compounds")

    # Aggregate per (compound, target): median activity
    papy_agg = papy.groupby([smi_col, "accession"])[val_col].median().reset_index()
    print(f"Aggregated: {len(papy_agg):,} (compound, target) pairs")

    # Wide pivot: compound × target → median activity
    wide = papy_agg.pivot(index=smi_col, columns="accession", values=val_col)
    targets = list(wide.columns)
    print(f"Wide pivot: {wide.shape}  ({len(targets)} target columns)")

    # ── Build reference fingerprint matrix ────────────────────────────────────
    print("\nBuilding reference fingerprints...")
    ref_smiles = wide.index.tolist()
    ref_fps, ref_valid = [], []
    for s in ref_smiles:
        fp = morgan_fp(s)
        if fp is not None:
            ref_fps.append(fp)
            ref_valid.append(True)
        else:
            ref_valid.append(False)
    wide = wide[ref_valid]
    ref_smiles = wide.index.tolist()
    print(f"  Valid reference compounds: {len(ref_fps):,}")

    target_matrix = wide.values  # shape (n_ref, n_targets), NaN for missing
    print(f"  Target matrix: {target_matrix.shape}  "
          f"NaN fraction: {np.isnan(target_matrix).mean()*100:.1f}%")

    # ── For each query, compute top-k Tanimoto neighbors → target profile ────
    def expand_query(query_smiles_list, label):
        print(f"\nExpanding {len(query_smiles_list)} {label} compounds...")
        n = len(query_smiles_list)
        n_t = target_matrix.shape[1]
        feat_avg = np.full((n, n_t), np.nan)     # similarity-weighted avg
        feat_max_sim = np.zeros(n)               # max similarity to any ref
        feat_n_neighbors = np.zeros(n)           # count of sim>=0.30 neighbors
        feat_topk_sim_mean = np.zeros(n)         # mean sim of top-k

        t0 = time.time()
        for i, qsmi in enumerate(query_smiles_list):
            qfp = morgan_fp(qsmi)
            if qfp is None: continue
            sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, ref_fps))
            feat_max_sim[i] = sims.max()

            # Top-5 neighbors with sim >= 0.30
            mask = sims >= 0.30
            feat_n_neighbors[i] = mask.sum()
            if mask.sum() == 0:
                continue

            # Get top-5 by similarity
            top_idx = np.argsort(sims)[::-1]
            top_idx = top_idx[:5]
            top_sims = sims[top_idx]
            feat_topk_sim_mean[i] = top_sims.mean()

            # For each target, similarity-weighted average over neighbors with measured value
            for t in range(n_t):
                vals = target_matrix[top_idx, t]
                valid = np.isfinite(vals)
                if valid.sum() == 0:
                    continue
                w = top_sims[valid]
                feat_avg[i, t] = np.dot(w, vals[valid]) / w.sum()

            if (i+1) % 500 == 0:
                print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)")

        # Replace NaN with column median of the matrix (per target)
        col_medians = np.nanmedian(feat_avg, axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 5.0)
        feat_avg_filled = np.where(np.isfinite(feat_avg),
                                    feat_avg,
                                    np.broadcast_to(col_medians, feat_avg.shape))

        feats = np.column_stack([
            feat_avg_filled,        # 41 cols
            feat_max_sim.reshape(-1, 1),     # 1 col
            feat_n_neighbors.reshape(-1, 1), # 1 col
            feat_topk_sim_mean.reshape(-1, 1), # 1 col
        ])
        print(f"  {label} expanded feature shape: {feats.shape}")
        print(f"  max_sim distribution: median={np.median(feat_max_sim):.3f}, "
              f"frac>0.30={np.mean(feat_max_sim >= 0.30)*100:.1f}%, "
              f"frac>0.50={np.mean(feat_max_sim >= 0.50)*100:.1f}%")
        return feats

    expand_tr = expand_query(smiles_tr, "train")
    expand_te = expand_query(smiles_te, "test")

    # ── Correlation analysis ──────────────────────────────────────────────────
    print("\nFeature correlations with PXR train labels (top 10):")
    rho_list = []
    for j, t in enumerate(targets):
        col = expand_tr[:, j]
        if np.unique(col).size > 1:
            rho, _ = spearmanr(col, y_tr)
            rho_list.append((t, rho))
    rho_list.sort(key=lambda x: abs(x[1]), reverse=True)
    for t, rho in rho_list[:10]:
        print(f"  {t:12s}: ρ={rho:+.3f}")

    # The summary features: max_sim, n_neighbors, top-k sim mean
    print("\nSummary feature correlations:")
    for j, name in enumerate(["max_sim", "n_neighbors", "topk_sim_mean"]):
        col = expand_tr[:, len(targets) + j]
        rho, _ = spearmanr(col, y_tr)
        print(f"  {name:18s}: ρ={rho:+.3f}")

    # ── Augmented LGBM ────────────────────────────────────────────────────────
    print("\n── Augmented LGBM scaffold CV ──")
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)
    X_tr_aug = np.hstack([X_tr_base, expand_tr])
    X_te_aug = np.hstack([X_te_base, expand_te])
    print(f"  Augmented shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    for name, Xt, Xe in [
        ("base_only", X_tr_base, X_te_base),
        ("tanimoto_aug", X_tr_aug, X_te_aug),
    ]:
        oof = np.zeros(len(y_tr))
        te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(Xt[tr_idx], y_tr[tr_idx],
                  eval_set=[(Xt[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(Xt[va_idx])
            te_preds.append(m.predict(Xe))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof); ratio = te_pred.std() / oof.std()
        print(f"  {name:15s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")
        if name == "tanimoto_aug":
            np.save(DATA_PROCESSED / "oof_nb132_tanimoto.npy", oof)
            np.save(DATA_PROCESSED / "te_nb132_tanimoto.npy", te_pred)
            print(f"  Saved oof_nb132_tanimoto.npy + te_nb132_tanimoto.npy")


if __name__ == "__main__":
    main()
