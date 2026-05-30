"""nb146 -- Analogy chain on 2.84M ChEMBL bulk records (pillar 3 at full scale).

For each PXR compound, find top-K Tanimoto neighbors in the 1.02M ChEMBL
compound universe. Each neighbor has measurements across some of 6,710 targets
in 166k assays. Build per-query 'neighbor activity profile'.

Then on PXR train: identify which ChEMBL targets best correlate with PXR pEC50.
These become PROXY ASSAYS — features for PXR prediction.

This is the user's 'any assay is fair game' approach at full scale.

Memory: 1M compounds × ~6.7k targets sparse matrix ≈ 10M nnz × 8 bytes = ~80MB.
Manageable. Tanimoto via BulkTanimoto = ~1 sec per query × 4652 queries = ~80 min.
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
from scipy.sparse import csr_matrix
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
K = 10
MIN_SIM = 0.30
MIN_ASSAY_DENSITY = 30  # only consider targets with >= 30 measured compounds


def morgan_fp(s, radius=2, nbits=2048):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nbits)


def main():
    print("=== nb146: Analogy chain on 2.84M ChEMBL bulk ===\n")

    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    print("Loading ChEMBL bulk...")
    bulk = pd.read_parquet(DATA_EXTERNAL / "chembl_bulk_activities.parquet",
                            columns=["canonical_smiles","target_chembl_id","pchembl_value"])
    bulk = bulk.dropna(subset=["canonical_smiles","target_chembl_id","pchembl_value"])
    print(f"  {len(bulk):,} rows  {bulk['canonical_smiles'].nunique():,} compounds  "
          f"{bulk['target_chembl_id'].nunique()} targets")

    # Per (compound, target) median
    print("Aggregating per (compound, target) ...")
    bulk = bulk.groupby(["canonical_smiles","target_chembl_id"])["pchembl_value"].median().reset_index()
    print(f"  After median agg: {len(bulk):,} pairs")

    # Filter to targets with sufficient density
    tgt_count = bulk["target_chembl_id"].value_counts()
    good_tgts = tgt_count[tgt_count >= MIN_ASSAY_DENSITY].index
    bulk = bulk[bulk["target_chembl_id"].isin(good_tgts)].reset_index(drop=True)
    print(f"  Targets w/ >= {MIN_ASSAY_DENSITY} compounds: {len(good_tgts)}")

    ref_smiles = sorted(bulk["canonical_smiles"].unique().tolist())
    target_list = sorted(bulk["target_chembl_id"].unique().tolist())
    smi_idx = {s: i for i, s in enumerate(ref_smiles)}
    tgt_idx = {t: i for i, t in enumerate(target_list)}
    print(f"  Final ref: {len(ref_smiles):,} compounds × {len(target_list)} targets")

    # Build sparse matrix
    rows = bulk["canonical_smiles"].map(smi_idx).values
    cols = bulk["target_chembl_id"].map(tgt_idx).values
    vals = bulk["pchembl_value"].values
    A = csr_matrix((vals, (rows, cols)), shape=(len(ref_smiles), len(target_list)))
    M = csr_matrix(([1.0]*len(vals), (rows, cols)), shape=(len(ref_smiles), len(target_list)))
    print(f"  Sparse matrix: {A.shape}, nnz={A.nnz:,}")

    # Build reference fingerprints
    print("Building reference fingerprints...")
    t0 = time.time()
    ref_fps = []
    keep_mask = []
    for s in ref_smiles:
        fp = morgan_fp(s)
        ref_fps.append(fp)
        keep_mask.append(fp is not None)
    print(f"  Valid: {sum(keep_mask):,}/{len(ref_smiles):,}  in {time.time()-t0:.0f}s")
    keep = np.array(keep_mask)
    ref_fps_valid = [f for f, v in zip(ref_fps, keep_mask) if v]
    A_valid = A[keep]
    M_valid = M[keep]
    n_ref_valid = A_valid.shape[0]
    n_t = A_valid.shape[1]
    print(f"  Active reference: {n_ref_valid:,} compounds")

    # Expand queries
    def expand(query_smiles, label):
        print(f"\nExpanding {len(query_smiles)} {label}...")
        n = len(query_smiles)
        feat_avg = np.full((n, n_t), np.nan, dtype=np.float32)
        feat_eng = np.zeros((n, n_t), dtype=np.float32)
        feat_max_sim = np.zeros(n)
        feat_n = np.zeros(n)
        t0 = time.time()
        for i, qs in enumerate(query_smiles):
            qfp = morgan_fp(qs)
            if qfp is None: continue
            sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, ref_fps_valid), dtype=np.float32)
            feat_max_sim[i] = sims.max()
            top_idx = np.argsort(sims)[::-1][:K]
            top_idx = top_idx[sims[top_idx] >= MIN_SIM]
            feat_n[i] = len(top_idx)
            if len(top_idx) == 0: continue
            top_sims = sims[top_idx]
            sub_A = A_valid[top_idx].toarray()  # K' x n_t
            sub_M = M_valid[top_idx].toarray()
            # For each target, sim-weighted avg of measured values
            mask = sub_M > 0  # K' x n_t
            for t_idx in range(n_t):
                col_mask = mask[:, t_idx]
                if col_mask.sum() > 0:
                    w = top_sims[col_mask]
                    feat_avg[i, t_idx] = np.dot(w, sub_A[col_mask, t_idx]) / w.sum()
                    feat_eng[i, t_idx] = col_mask.sum() / len(top_idx)
            if (i+1) % 200 == 0:
                print(f"  {label}: {i+1}/{n}  ({time.time()-t0:.0f}s)")
        return feat_avg, feat_eng, feat_max_sim, feat_n

    tr_avg, tr_eng, tr_msim, tr_nn = expand(smiles_tr, "train")
    te_avg, te_eng, te_msim, te_nn = expand(smiles_te, "test")

    print(f"\nMax sim distribution (train): median={np.median(tr_msim):.3f}  "
          f"frac>=0.50={np.mean(tr_msim >= 0.50)*100:.1f}%  "
          f"frac>=0.70={np.mean(tr_msim >= 0.70)*100:.1f}%")
    print(f"Max sim distribution (test):  median={np.median(te_msim):.3f}  "
          f"frac>=0.50={np.mean(te_msim >= 0.50)*100:.1f}%  "
          f"frac>=0.70={np.mean(te_msim >= 0.70)*100:.1f}%")

    # Find top assays correlating with PXR pEC50
    print("\nFinding top ChEMBL targets that correlate with PXR pEC50...")
    rho_list = []
    for t_idx in range(n_t):
        col = tr_avg[:, t_idx]
        mask = np.isfinite(col)
        if mask.sum() < 100: continue
        if col[mask].std() < 0.01: continue
        rho, pval = spearmanr(col[mask], y_tr[mask])
        if np.isnan(rho): continue
        rho_list.append((target_list[t_idx], t_idx, rho, pval, mask.sum()))
    rho_df = pd.DataFrame(rho_list, columns=["target","idx","rho","pval","n"]).sort_values("rho", key=abs, ascending=False)
    rho_df.to_csv(DATA_PROCESSED / "nb146_assay_corr.csv", index=False)
    print(f"  Assays with valid correlation: {len(rho_df)}")
    print(f"  Top 20 by |rho|:")
    print(rho_df.head(20).to_string(index=False))

    # Select top-K most correlated assays as feature columns
    TOP_K_ASSAYS = 200
    selected = rho_df.head(TOP_K_ASSAYS)["idx"].values
    print(f"\nSelected top {TOP_K_ASSAYS} correlated assays as features")

    # Fill NaN per column with the column median (from train), then take only selected cols
    tr_feats = np.empty((len(smiles_tr), TOP_K_ASSAYS), dtype=np.float32)
    te_feats = np.empty((len(smiles_te), TOP_K_ASSAYS), dtype=np.float32)
    for j, t_idx in enumerate(selected):
        col_tr = tr_avg[:, t_idx]
        col_te = te_avg[:, t_idx]
        med = np.nanmedian(col_tr)
        if not np.isfinite(med): med = 5.0
        tr_feats[:, j] = np.where(np.isfinite(col_tr), col_tr, med)
        te_feats[:, j] = np.where(np.isfinite(col_te), col_te, med)

    # Augmented LGBM CV
    print("\n── Augmented LGBM scaffold CV ──")
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)
    X_tr_aug = np.hstack([X_tr_base, tr_feats, tr_msim.reshape(-1,1), tr_nn.reshape(-1,1)])
    X_te_aug = np.hstack([X_te_base, te_feats, te_msim.reshape(-1,1), te_nn.reshape(-1,1)])

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)
    for name, Xt, Xe in [("base_only", X_tr_base, X_te_base),
                          ("chembl_analogy", X_tr_aug, X_te_aug)]:
        oof = np.zeros(len(y_tr)); te_preds = []
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
        if name == "chembl_analogy":
            np.save(DATA_PROCESSED / "oof_nb146_chembl_analogy.npy", oof)
            np.save(DATA_PROCESSED / "te_nb146_chembl_analogy.npy", te_pred)
            print(f"  Saved oof_nb146_chembl_analogy.npy + te_nb146_chembl_analogy.npy")


if __name__ == "__main__":
    main()
