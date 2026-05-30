"""nb144 -- Tanimoto expansion on FULL Papyrus (423k compounds, 1.5M activities, 41 targets).

vs nb132 which used Papyrus++ filtered (6k compounds). 70x more reference data
means much higher chance of finding good Tanimoto neighbors for our compounds.

Same logic as nb132: for each PXR compound (train + test), find top-k=10 Tanimoto
neighbors in Papyrus full, build 41-dim per-target sim-weighted activity profile.

Save as ensemble candidate.
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
    print("=== nb144: Tanimoto expansion on FULL Papyrus ===\n")

    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    # Use the WIDE pivot (compound × target activity matrix)
    print("Loading Papyrus wide pivot...")
    wide = pd.read_parquet(DATA_EXTERNAL / "papyrus_full_wide.parquet")
    print(f"  Shape: {wide.shape}")
    smi_col = "SMILES" if "SMILES" in wide.columns else "SMILES_Stripped"
    target_cols = [c for c in wide.columns if c not in ("SMILES","SMILES_Stripped","connectivity","index")]
    print(f"  smi_col={smi_col}  {len(target_cols)} targets")
    wide = wide.dropna(subset=[smi_col]).reset_index(drop=True)
    ref_smiles = wide[smi_col].tolist()
    activity_matrix = wide[target_cols].values  # shape (N, 41), NaN for missing
    print(f"  Reference compounds: {len(ref_smiles):,}")
    print(f"  Coverage: {(~np.isnan(activity_matrix)).sum():,} measured cells "
          f"({(~np.isnan(activity_matrix)).mean()*100:.2f}%)")

    print("\nBuilding reference fingerprints...")
    t0 = time.time()
    ref_fps, ref_valid = [], []
    for s in ref_smiles:
        fp = morgan_fp(s)
        ref_fps.append(fp); ref_valid.append(fp is not None)
    print(f"  Valid: {sum(ref_valid):,}/{len(ref_smiles):,} in {time.time()-t0:.0f}s")
    valid_mask = np.array(ref_valid)
    ref_fps_valid = [f for f, v in zip(ref_fps, ref_valid) if v]
    activity_valid = activity_matrix[valid_mask]
    print(f"  Active reference set: {len(ref_fps_valid):,} compounds × {len(target_cols)} targets")

    def expand(query_smiles_list, label):
        print(f"\nExpanding {len(query_smiles_list)} {label}...")
        n = len(query_smiles_list)
        n_t = len(target_cols)
        K = 10
        MIN_SIM = 0.30
        feat_avg = np.full((n, n_t), np.nan, dtype=np.float32)
        feat_eng = np.zeros((n, n_t), dtype=np.float32)
        feat_max_sim = np.zeros(n)
        feat_n_neighbors = np.zeros(n)
        t0 = time.time()
        for i, qs in enumerate(query_smiles_list):
            qfp = morgan_fp(qs)
            if qfp is None: continue
            sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, ref_fps_valid), dtype=np.float32)
            feat_max_sim[i] = sims.max()
            top_idx = np.argsort(sims)[::-1][:K]
            top_idx = top_idx[sims[top_idx] >= MIN_SIM]
            feat_n_neighbors[i] = len(top_idx)
            if len(top_idx) == 0: continue
            top_sims = sims[top_idx]
            sub_acts = activity_valid[top_idx]  # K' x n_t
            for t_idx in range(n_t):
                col = sub_acts[:, t_idx]
                mask = np.isfinite(col)
                if mask.sum() > 0:
                    w = top_sims[mask]
                    feat_avg[i, t_idx] = np.dot(w, col[mask]) / w.sum()
                    feat_eng[i, t_idx] = mask.sum() / len(top_idx)
            if (i+1) % 200 == 0:
                print(f"  {label}: {i+1}/{n}  ({time.time()-t0:.0f}s)")
        # Fill NaN with column median
        for t_idx in range(n_t):
            col = feat_avg[:, t_idx]
            med = np.nanmedian(col)
            feat_avg[:, t_idx] = np.where(np.isfinite(col), col, med if np.isfinite(med) else 5.0)
        feats = np.column_stack([feat_avg, feat_eng,
                                  feat_max_sim.reshape(-1,1),
                                  feat_n_neighbors.reshape(-1,1)])
        print(f"  {label} feature shape: {feats.shape}")
        print(f"  max_sim distribution: median={np.median(feat_max_sim):.3f}  "
              f"frac>=0.50={np.mean(feat_max_sim >= 0.50)*100:.1f}%  "
              f"frac>=0.70={np.mean(feat_max_sim >= 0.70)*100:.1f}%")
        return feats

    tr_feat = expand(smiles_tr, "train")
    te_feat = expand(smiles_te, "test")

    # Correlations with PXR pEC50
    print("\nTop-10 target activity correlations with PXR pEC50:")
    n_t = len(target_cols)
    rho_list = []
    for t_idx in range(n_t):
        col = tr_feat[:, t_idx]
        if col.std() > 0:
            rho, pval = spearmanr(col, y_tr)
            rho_list.append((target_cols[t_idx], rho, pval))
    rho_list.sort(key=lambda x: abs(x[1]), reverse=True)
    for t, rho, p in rho_list[:10]:
        print(f"  {t:12s}: ρ={rho:+.3f}  p={p:.2e}")

    # Augmented LGBM
    print("\n── Augmented LGBM scaffold CV ──")
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)
    X_tr_aug = np.hstack([X_tr_base, tr_feat])
    X_te_aug = np.hstack([X_te_base, te_feat])

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)
    for name, Xt, Xe in [("base_only", X_tr_base, X_te_base),
                         ("full_papyrus_aug", X_tr_aug, X_te_aug)]:
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
        print(f"  {name:20s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")
        if name == "full_papyrus_aug":
            np.save(DATA_PROCESSED / "oof_nb144_full_papyrus.npy", oof)
            np.save(DATA_PROCESSED / "te_nb144_full_papyrus.npy", te_pred)
            print(f"  Saved oof_nb144_full_papyrus.npy + te_nb144_full_papyrus.npy")


if __name__ == "__main__":
    main()
