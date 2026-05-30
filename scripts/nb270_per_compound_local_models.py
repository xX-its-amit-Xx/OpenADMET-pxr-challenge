"""nb270 -- Per-compound local model: train K=200 nearest-neighbor specialized LGBM.

User idea: at extreme limit, train 513 models — one per test compound. Each
model trained on most-similar train + external PXR compounds.

Cluster levels:
1. Hyper-local: 513 models (one per test compound)
2. Family-level: ~50 families via Murcko scaffolds → 50 models
3. Coarse: 5-10 families

Use Tanimoto-expansion validation: for each test compound, generate randomized
SMILES OR similar molecules from external library to test prediction stability.

Tracking: markdown file `data/processed/per_compound_models.md` logs each model.
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
from pathlib import Path

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def morgan(smiles, radius=2, n_bits=2048):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


def main():
    print("=== nb270: Per-compound local models ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    # Combine training pool: PXR train + Papyrus PXR + PubChem actives
    print("Building combined training pool...")
    papyrus = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
    papyrus_pxr = papyrus[papyrus["target_name"].str.contains("PXR", case=False, na=False)].copy()
    papyrus_pxr["std_smiles"] = papyrus_pxr["std_smiles"].apply(std_smi)
    papyrus_pxr = papyrus_pxr.dropna(subset=["std_smiles", "pec50"])
    papyrus_pxr = papyrus_pxr.groupby("std_smiles")["pec50"].median().reset_index()
    print(f"  Papyrus PXR (deduped): {len(papyrus_pxr)}")

    pubchem = pd.read_parquet("data/external/pubchem_pxr_active_smiles.parquet")
    pubchem["std_smiles"] = pubchem["smiles"].apply(std_smi)
    pubchem = pubchem.dropna(subset=["std_smiles"])
    # Pseudo-pec50 from active_rate: 1->6.5, 0->4.0
    pubchem["pec50"] = 4.0 + pubchem["active_rate"].fillna(0.5) * 2.5
    pubchem = pubchem.groupby("std_smiles")["pec50"].median().reset_index()
    print(f"  PubChem actives: {len(pubchem)}")

    # Exclude train and test from external
    excl = set(smiles_tr) | set(smiles_te)
    papyrus_pxr = papyrus_pxr[~papyrus_pxr["std_smiles"].isin(excl)]
    pubchem = pubchem[~pubchem["std_smiles"].isin(excl)]

    # Combine
    train_pool_smi = smiles_tr + papyrus_pxr["std_smiles"].tolist() + pubchem["std_smiles"].tolist()
    train_pool_y = np.concatenate([y_tr, papyrus_pxr["pec50"].values, pubchem["pec50"].values])
    train_pool_w = np.concatenate([np.ones(len(y_tr)),
                                     np.full(len(papyrus_pxr), 0.5),  # papyrus medium
                                     np.full(len(pubchem), 0.3)])      # pubchem soft
    print(f"  Total training pool: {len(train_pool_smi)}")

    # Featurize once
    print("Computing FPs and features (this is slow)...")
    fps_pool = morgan(train_pool_smi)
    fps_te = morgan(smiles_te)
    print(f"  Pool FPs: {len(fps_pool)}, test FPs: {len(fps_te)}")

    X_pool = combined(train_pool_smi); X_pool = impute(X_pool)
    X_te = combined(smiles_te); X_te = impute(X_te)
    print(f"  X_pool: {X_pool.shape}")

    # Markdown tracker
    md_path = DATA_PROCESSED / "per_compound_models.md"
    md_file = open(md_path, "w")
    md_file.write("# Per-compound local models\n\n")
    md_file.write("| Test compound | K NN used | Top1 sim | NN mean pec50 | NN std | Local pred |\n")
    md_file.write("|---|---|---|---|---|---|\n")

    # For each test compound: top-K NN -> local LGBM
    K = 200
    print(f"\nBuilding {len(smiles_te)} local models (K={K} NN each)...")
    te_local_pred = np.zeros(len(smiles_te))
    te_local_top1 = np.zeros(len(smiles_te))
    te_local_nn_mean = np.zeros(len(smiles_te))

    LGBM = dict(n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=10,
                objective="mae", n_jobs=2, random_state=42, verbose=-1)
    t0 = time.time()

    for i in range(len(smiles_te)):
        if fps_te[i] is None:
            te_local_pred[i] = train_pool_y[:len(y_tr)].mean()
            continue
        sims = np.array(BulkTanimotoSimilarity(fps_te[i], fps_pool))
        top_idx = np.argsort(sims)[::-1][:K]
        top_sims = sims[top_idx]
        nn_y = train_pool_y[top_idx]
        nn_w = train_pool_w[top_idx] * (top_sims + 0.01)  # similarity-weighted
        te_local_top1[i] = top_sims[0]
        te_local_nn_mean[i] = nn_y.mean()
        # Train LGBM
        try:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X_pool[top_idx], nn_y, sample_weight=nn_w)
            te_local_pred[i] = float(md.predict(X_te[i:i+1])[0])
        except Exception as e:
            te_local_pred[i] = nn_y.mean()

        # Markdown row (every 50)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(smiles_te) - i - 1)
            print(f"  {i+1}/{len(smiles_te)}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.0f}m")
            md_file.write(f"| {te_names[i]} | {K} | {top_sims[0]:.3f} | {nn_y.mean():.2f} | {nn_y.std():.2f} | {te_local_pred[i]:.3f} |\n")

    # Final markdown summary
    md_file.write(f"\n## Summary\n")
    md_file.write(f"- Total models: {len(smiles_te)}\n")
    md_file.write(f"- Mean prediction: {te_local_pred.mean():.3f}\n")
    md_file.write(f"- Std prediction: {te_local_pred.std():.3f}\n")
    md_file.write(f"- Top1 sim distribution: min {te_local_top1.min():.3f}, median {np.median(te_local_top1):.3f}, max {te_local_top1.max():.3f}\n")
    md_file.close()
    print(f"\nMarkdown saved to {md_path}")

    print(f"\nLocal model predictions: mean={te_local_pred.mean():.3f}, std={te_local_pred.std():.3f}")
    print(f"Top1-sim stats: min={te_local_top1.min():.3f}, median={np.median(te_local_top1):.3f}")

    np.save(DATA_PROCESSED / "te_nb270_per_compound.npy", te_local_pred)
    np.save(DATA_PROCESSED / "te_nb270_top1sim.npy", te_local_top1)

    # Also generate a validation: re-run on TRAIN compounds (each train compound, find K nearest in pool excluding self)
    # Compare to nb239 OOF
    print("\nValidation on train (subset of 200 random compounds)...")
    rng = np.random.default_rng(42)
    val_idx = rng.choice(len(y_tr), size=200, replace=False)
    val_pred = np.zeros(len(val_idx))
    fps_tr_sub = morgan([smiles_tr[i] for i in val_idx])
    for j, i in enumerate(val_idx):
        if fps_tr_sub[j] is None:
            val_pred[j] = train_pool_y.mean()
            continue
        sims = np.array(BulkTanimotoSimilarity(fps_tr_sub[j], fps_pool))
        sims[i] = -1  # exclude self
        top_idx = np.argsort(sims)[::-1][:K]
        top_sims = sims[top_idx]
        nn_y = train_pool_y[top_idx]
        nn_w = train_pool_w[top_idx] * (top_sims + 0.01)
        try:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X_pool[top_idx], nn_y, sample_weight=nn_w)
            val_pred[j] = float(md.predict(X_pool[i:i+1])[0])
        except:
            val_pred[j] = nn_y.mean()
    val_mae = np.abs(y_tr[val_idx] - val_pred).mean()
    val_rae = rae(y_tr[val_idx], val_pred)
    print(f"Local-model val MAE (200 random): {val_mae:.4f}, RAE: {val_rae:.4f}")

    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb239_val_rae = rae(y_tr[val_idx], nb239_oof[val_idx])
    print(f"nb239 OOF on same 200: RAE = {nb239_val_rae:.4f}")

    # Submit
    blend_te = 0.5 * te_local_pred + 0.5 * np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")
    sub = pd.DataFrame({"SMILES": [sm[n] for n in te_names], "Molecule Name": te_names, "pEC50": blend_te})
    sub.to_csv(SUBMISSIONS / "270_local_models_blend.csv", index=False)
    print(f"\nSaved 270_local_models_blend.csv (50/50 with nb239)")
    sub2 = pd.DataFrame({"SMILES": [sm[n] for n in te_names], "Molecule Name": te_names, "pEC50": te_local_pred})
    sub2.to_csv(SUBMISSIONS / "270_local_models_pure.csv", index=False)
    print(f"Saved 270_local_models_pure.csv")


if __name__ == "__main__":
    main()
