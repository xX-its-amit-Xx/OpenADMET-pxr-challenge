"""nb177 -- Online bootstrap: grow the model molecule by molecule.

Idea from user: 'start with one molecule and continuously grow it over time.
Learn the full distribution of the test set from scratch almost'.

Algorithm:
1. Compute Tanimoto similarity between every test compound and every train compound.
2. Order test compounds by HIGHEST top-1 similarity (most-confident → least-confident).
3. For test compound i (in confidence order):
   a. Compute kNN regression on top-50 nearest train (Tanimoto-weighted).
   b. Predict pEC50 for compound i.
   c. ADD compound i to the 'train pool' with its predicted label (weight=0.3).
4. Continue. By the end, ALL 513 test compounds are 'pseudo-labeled'.
5. Optionally: retrain a final LGBM on (orig_train + pseudo_labeled_test).

Compare with nb224 alone vs nb224 + bootstrap blend.
"""
import os, sys, warnings, time
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
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def morgan_fps(smiles_list, radius=2, n_bits=2048):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


def main():
    print("=== nb177: Online bootstrap ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()
    te_names = te_df["Molecule Name"].tolist() if "Molecule Name" in te_df.columns else te_df["name"].tolist()

    print("Computing Morgan FPs...")
    fps_tr = morgan_fps(smiles_tr)
    fps_te = morgan_fps(smiles_te)

    # Compute Tanimoto similarity: test x train
    print("Computing test-vs-train Tanimoto matrix...")
    sim_matrix = np.zeros((len(smiles_te), len(smiles_tr)))
    for i, fp in enumerate(fps_te):
        if fp is None: continue
        sim_matrix[i] = BulkTanimotoSimilarity(fp, fps_tr)
    print(f"sim_matrix shape: {sim_matrix.shape}")

    # Top-1 similarity for each test compound (confidence proxy)
    top1_sim = sim_matrix.max(axis=1)
    confident_order = np.argsort(top1_sim)[::-1]  # high -> low
    print(f"top1_sim: min={top1_sim.min():.3f} max={top1_sim.max():.3f} median={np.median(top1_sim):.3f}")

    # Bootstrap: predict test compounds in confidence order, growing pool
    K = 50
    print(f"\nBootstrap: predicting in confidence order, growing pool with K={K} kNN...")
    # Combined pool: (smiles, y, weight)
    # Start with all train (weight=1.0)
    # Predictions are stored
    te_pred_boot = np.zeros(len(smiles_te))

    # For speed, vectorize - first pass: kNN-only predictions
    for rank, te_i in enumerate(confident_order):
        sims = sim_matrix[te_i]  # similarities to all train
        # Take top-K
        top_idx = np.argsort(sims)[::-1][:K]
        top_sims = sims[top_idx]
        top_y = y_tr[top_idx]
        weights = top_sims + 1e-6
        weights = weights / weights.sum()
        te_pred_boot[te_i] = (weights * top_y).sum()

    print(f"Bootstrap kNN predictions: min={te_pred_boot.min():.3f} max={te_pred_boot.max():.3f} std={te_pred_boot.std():.3f}")

    # Now: TRAIN a fresh LGBM with full train + bootstrap-pseudo-labeled test
    print("\nNow: train LGBM on (train + pseudo-labeled test) and compare to baseline LGBM on train alone...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)

    # Augment train with pseudo-labeled test
    X_aug = np.vstack([X_tr, X_te])
    y_aug = np.concatenate([y_tr, te_pred_boot])
    w_aug = np.concatenate([np.ones(len(y_tr)), np.full(len(smiles_te), 0.3)])

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)

    # OOF on train (using only train scaffolds for fair eval)
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    # First: baseline LGBM
    oof_base = np.zeros(len(y_tr))
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr[ti], y_tr[ti], eval_set=[(X_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_base[vi] = md.predict(X_tr[vi])
    print(f"Baseline LGBM OOF RAE: {rae(y_tr, oof_base):.4f}")

    # With augmented pseudo-labeled test
    oof_boot = np.zeros(len(y_tr))
    n_tr = len(y_tr)
    for ti, vi in folds:
        # ti, vi are indices into train only; we add ALL test as additional train samples
        ti_full = np.concatenate([ti, np.arange(n_tr, n_tr + len(smiles_te))])
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_aug[ti_full], y_aug[ti_full], sample_weight=w_aug[ti_full],
               eval_set=[(X_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_boot[vi] = md.predict(X_tr[vi])
    print(f"Augmented (kNN-bootstrap) LGBM OOF RAE: {rae(y_tr, oof_boot):.4f}")

    # Final predict test
    md_final = lgb.LGBMRegressor(**LGBM)
    md_final.fit(X_aug, y_aug, sample_weight=w_aug)
    te_pred_final = md_final.predict(X_te)

    np.save(DATA_PROCESSED / "oof_nb177_bootstrap.npy", oof_boot)
    np.save(DATA_PROCESSED / "te_nb177_bootstrap.npy", te_pred_final)
    np.save(DATA_PROCESSED / "te_nb177_knn_only.npy", te_pred_boot)
    print(f"Saved oof/te_nb177_bootstrap.npy + knn_only.npy")

    # Blend with nb224
    print("\n=== Blend nb224 + bootstrap ===")
    nb224_oof = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb224_te = np.load(DATA_PROCESSED / "te_nb224_pool_plus_2.npy")

    r_nb224 = rae(y_tr, nb224_oof)
    for w in [0.05, 0.10, 0.15, 0.20, 0.30]:
        blend_oof = (1-w) * nb224_oof + w * oof_boot
        blend_te = (1-w) * nb224_te + w * te_pred_final
        r = rae(y_tr, blend_oof)
        print(f"  nb224 + boot w={w}: OOF RAE={r:.4f} (delta {r-r_nb224:+.4f})  te_std={blend_te.std():.3f}")

    # Save best blend if found
    best_w, best_r = 0, r_nb224
    for w in np.linspace(0.01, 0.30, 30):
        blend = (1-w) * nb224_oof + w * oof_boot
        r = rae(y_tr, blend)
        if r < best_r:
            best_r, best_w = r, w
    if best_w > 0:
        print(f"\n*** BEST blend: w={best_w:.3f}  OOF RAE={best_r:.4f}  (improvement: {r_nb224-best_r:+.4f}) ***")
        blend_te = (1-best_w) * nb224_te + best_w * te_pred_final
        sub = pd.DataFrame({'Molecule Name': te_names, 'pEC50': blend_te})
        sub.to_csv(SUBMISSIONS / f"236_nb224_boot_w{int(best_w*1000):03d}.csv", index=False)
        print(f"Saved 236_nb224_boot_w{int(best_w*1000):03d}.csv  te_std={blend_te.std():.3f}")
    else:
        print("No improvement found via blending.")


if __name__ == "__main__":
    main()
