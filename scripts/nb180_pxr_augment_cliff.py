"""nb180 -- PXR data augmentation + cliff-aware retraining + relabel-correction.

Components:
1. Augment train with 945 Papyrus PXR records (downweight + remove overlaps).
2. Cliff-aware sample weighting: compounds in known activity cliffs get HIGHER
   weight (force model to learn the discriminator).
3. Relabel suspect compounds via neighborhood consensus (from nb179).

This is the full "build everything in" attempt informed by all prior findings.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns, standardize, bemis_murcko
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS


def morgan_fps(smiles, radius=2, n_bits=2048):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


def main():
    print("=== nb180: Papyrus augmentation + cliff + relabel ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    inchikey_tr = set(tr["inchikey"].tolist())
    smiles_te = te_df["smiles"].tolist()
    te_names = te_df["Molecule Name"].tolist() if "Molecule Name" in te_df.columns else te_df["name"].tolist()
    print(f"Train: {len(smiles_tr)}, Test: {len(smiles_te)}")

    # --- Load Papyrus PXR ---
    print("\n--- Loading Papyrus PXR (945 records) ---")
    papyrus = pd.read_parquet(DATA_EXTERNAL / "papyrus_pxr_nr.parquet")
    papyrus_pxr = papyrus[papyrus['target_name'].str.contains('PXR', case=False, na=False)].copy()
    print(f"Total Papyrus PXR rows: {len(papyrus_pxr)}")

    # Filter: remove any already in our train (by inchikey)
    papyrus_pxr = papyrus_pxr[~papyrus_pxr['inchikey'].isin(inchikey_tr)].reset_index(drop=True)
    print(f"After train-overlap filter: {len(papyrus_pxr)}")
    # Drop duplicate inchikeys (keep median pec50)
    papyrus_pxr = papyrus_pxr.groupby('inchikey').agg({'std_smiles': 'first', 'pec50': 'median'}).reset_index()
    print(f"After dedup: {len(papyrus_pxr)}")
    print(f"Papyrus pec50: mean={papyrus_pxr.pec50.mean():.3f} std={papyrus_pxr.pec50.std():.3f}")

    # --- Compute features for train + test + augmented ---
    print("\nFeaturizing...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    smiles_aug = papyrus_pxr['std_smiles'].tolist()
    y_aug = papyrus_pxr['pec50'].values.astype(np.float64)
    X_aug = combined(smiles_aug); X_aug = impute(X_aug)
    print(f"X_aug: {X_aug.shape}")

    # --- Identify activity cliffs in train ---
    print("\n--- Identifying activity cliffs in train ---")
    fps_tr = morgan_fps(smiles_tr)
    cliff_score = np.zeros(len(smiles_tr))  # max |dy| for any same-scaffold-or-similar neighbor
    # Use scaffold + Tanimoto similarity to find cliff pairs
    scaffolds = tr["scaffold"].tolist()
    from collections import defaultdict
    scaff_groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        scaff_groups[s].append(i)
    n_cliff_pairs = 0
    for scaff, idx_list in scaff_groups.items():
        if len(idx_list) < 2:
            continue
        for i, ai in enumerate(idx_list):
            for bj in idx_list[i+1:]:
                sim = float(np.array(BulkTanimotoSimilarity(fps_tr[ai], [fps_tr[bj]]))[0])
                dy = abs(y_tr[ai] - y_tr[bj])
                if sim >= 0.7 and dy >= 1.0:
                    n_cliff_pairs += 1
                    cliff_score[ai] = max(cliff_score[ai], dy)
                    cliff_score[bj] = max(cliff_score[bj], dy)
    print(f"Found {n_cliff_pairs} cliff pairs in train; {(cliff_score > 0).sum()} compounds involved")

    # --- Relabel suspect compounds (carry forward from nb179 logic) ---
    print("\n--- Relabel suspect compounds via neighborhood consensus ---")
    # Compute neighbor-avg pec50 for each train (top 5 NN excluding self)
    nb_avg = np.zeros(len(smiles_tr))
    nb_spread = np.zeros(len(smiles_tr))
    for i in range(len(smiles_tr)):
        sims = np.array(BulkTanimotoSimilarity(fps_tr[i], fps_tr))
        sims[i] = -1
        top_idx = np.argsort(sims)[::-1][:5]
        nb_avg[i] = y_tr[top_idx].mean()
        nb_spread[i] = y_tr[top_idx].max() - y_tr[top_idx].min()

    residual = y_tr - nb_avg
    # Suspect mislabel: residual large AND neighborhood spread small
    suspect_mask = (np.abs(residual) > 1.5) & (nb_spread < 1.0)
    print(f"Suspect mislabel: {suspect_mask.sum()}")
    relabeled_y = y_tr.copy()
    relabeled_y[suspect_mask] = 0.5 * y_tr[suspect_mask] + 0.5 * nb_avg[suspect_mask]

    # --- Sample weighting ---
    # train: weight=1.0 always
    # cliffs: weight=1.5 (force model to learn discriminators)
    # papyrus aug: weight=0.3 (downweight as semi-supervised)
    weights_tr = np.ones(len(y_tr))
    weights_tr[cliff_score > 0] = 1.5  # cliff compounds higher weight
    weights_aug = np.full(len(y_aug), 0.3)

    # Combined training set: original train + papyrus
    X_full = np.vstack([X_tr, X_aug])
    y_full = np.concatenate([relabeled_y, y_aug])
    w_full = np.concatenate([weights_tr, weights_aug])
    print(f"\nCombined training: {X_full.shape}")

    # --- OOF training ---
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    # Strategy: fold validation only on original train compounds. Use augmented as additional train.
    # OOF compares to ORIGINAL labels (so we measure if model learned to predict the truth, not relabeled targets).
    print("\n=== OOF training (eval against ORIGINAL labels) ===")
    n_tr = len(y_tr)
    n_aug = len(y_aug)
    oof = np.zeros(n_tr)
    te_preds = []
    for ti, vi in folds:
        # ti is indices in train. Add ALL papyrus indices (offset by n_tr).
        ti_full = np.concatenate([ti, np.arange(n_tr, n_tr + n_aug)])
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_full[ti_full], y_full[ti_full], sample_weight=w_full[ti_full],
               eval_set=[(X_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr[vi])
        te_preds.append(md.predict(X_te))
    te_pred = np.mean(te_preds, axis=0)
    print(f"\n=== Combined approach OOF RAE: {rae(y_tr, oof):.4f}  vs nb224 baseline: 0.2891 ===")
    print(f"te_std: {te_pred.std():.4f}  te_mean: {te_pred.mean():.3f}")

    # --- Variations: ablation study ---
    print("\n=== Ablation ===")
    for name, x, y_, w_ in [
        ("base (orig train, no aug)", X_tr, y_tr, np.ones(n_tr)),
        ("orig+relabel", X_tr, relabeled_y, np.ones(n_tr)),
        ("orig+cliff_weight", X_tr, y_tr, weights_tr),
        ("orig+papyrus", X_full, np.concatenate([y_tr, y_aug]), np.concatenate([np.ones(n_tr), np.full(n_aug, 0.3)])),
    ]:
        oof_t = np.zeros(n_tr)
        for ti, vi in folds:
            if len(x) > n_tr:
                ti_full = np.concatenate([ti, np.arange(n_tr, len(x))])
            else:
                ti_full = ti
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(x[ti_full], y_[ti_full], sample_weight=w_[ti_full],
                   eval_set=[(X_tr[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof_t[vi] = md.predict(X_tr[vi])
        print(f"  {name:40s}: OOF RAE = {rae(y_tr, oof_t):.4f}")

    np.save(DATA_PROCESSED / "oof_nb180_full.npy", oof)
    np.save(DATA_PROCESSED / "te_nb180_full.npy", te_pred)
    np.save(DATA_PROCESSED / "nb180_cliff_score.npy", cliff_score)
    np.save(DATA_PROCESSED / "nb180_suspect_mask.npy", suspect_mask)
    np.save(DATA_PROCESSED / "nb180_relabeled_y.npy", relabeled_y)

    # --- Stack with nb224 ---
    print("\n=== Stack nb180 + nb224 ===")
    nb224_oof = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb224_te = np.load(DATA_PROCESSED / "te_nb224_pool_plus_2.npy")
    r_nb224 = rae(y_tr, nb224_oof)
    for w in [0.05, 0.10, 0.15, 0.20, 0.30]:
        blend = (1-w)*nb224_oof + w*oof
        r = rae(y_tr, blend)
        print(f"  w_nb180={w:.2f}: OOF RAE={r:.4f} (delta {r-r_nb224:+.4f})")

    best_w, best_r = 0, r_nb224
    for w in np.linspace(0.01, 0.30, 30):
        blend = (1-w)*nb224_oof + w*oof
        r = rae(y_tr, blend)
        if r < best_r: best_r, best_w = r, w
    print(f"\nBest blend: w={best_w:.3f}  OOF RAE={best_r:.4f}  delta={best_r-r_nb224:+.4f}")
    if best_w > 0:
        blend_te = (1-best_w)*nb224_te + best_w*te_pred
        sub = pd.DataFrame({'Molecule Name': te_names, 'pEC50': blend_te})
        sub.to_csv(SUBMISSIONS / f"237_nb180_blend_w{int(best_w*1000):03d}.csv", index=False)
        print(f"Saved 237_nb180_blend_w{int(best_w*1000):03d}.csv  te_std={blend_te.std():.3f}")
        print("*** IMPROVEMENT FOUND ***")


if __name__ == "__main__":
    main()
