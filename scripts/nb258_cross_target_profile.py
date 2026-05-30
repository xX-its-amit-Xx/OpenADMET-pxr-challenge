"""nb258 -- Cross-target NR activity profile as feature.

Papyrus has 11k records across 7 NR targets (PXR, FXR, PPARg, LXR, RXRa, VDR, CAR).
For each compound, build a 7-dim 'promiscuity vector' showing predicted activity
across all NRs.

PXR ligands often hit multiple NRs (rifampicin → PXR + CAR; T0901317 → LXR + PXR).
This cross-target signature is GENUINELY orthogonal to single-target features.

Approach:
1. For each NR target, train a kNN/LGBM on its Papyrus records
2. For each train + test compound, predict activity at each NR
3. Use 7-dim vector as new feature
4. Train LGBM with these + base features
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

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def main():
    print("=== nb258: Cross-target NR activity profile ===\n")

    # Load Papyrus
    papyrus = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
    print("Papyrus targets:")
    print(papyrus["target_name"].value_counts())

    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()

    # Build per-target reference set
    targets = papyrus["target_name"].value_counts().head(7).index.tolist()
    print(f"\nUsing targets: {targets}")

    # Featurize FPs
    def morgan(smiles):
        fps = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s) if s else None
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None)
        return fps

    fps_tr = morgan(smiles_tr)
    fps_te = morgan(smiles_te)

    profile_tr = np.zeros((len(smiles_tr), len(targets)))
    profile_te = np.zeros((len(smiles_te), len(targets)))

    train_set = set(smiles_tr); test_set = set(smiles_te)

    for t_idx, target in enumerate(targets):
        target_df = papyrus[papyrus["target_name"] == target].copy()
        target_df["std_smiles"] = target_df["std_smiles"].apply(std_smi)
        target_df = target_df[target_df["std_smiles"].notna() & target_df["pec50"].notna()]
        # Exclude train + test compounds for fairness
        target_df = target_df[~target_df["std_smiles"].isin(train_set | test_set)]
        # Dedupe
        target_df = target_df.groupby("std_smiles")["pec50"].median().reset_index()
        ref_smiles = target_df["std_smiles"].tolist()
        ref_y = target_df["pec50"].values
        ref_fps = morgan(ref_smiles)
        print(f"  {target}: {len(ref_smiles)} compounds (mean pec50 {ref_y.mean():.3f})")
        if len(ref_smiles) < 50:
            continue
        # kNN
        K = 5
        for i, fp in enumerate(fps_tr):
            if fp is None: continue
            sims = np.array(BulkTanimotoSimilarity(fp, ref_fps))
            top_idx = np.argsort(sims)[::-1][:K]
            top_sims = sims[top_idx]
            if top_sims[0] < 0.2:
                profile_tr[i, t_idx] = ref_y.mean()  # fallback
            else:
                w = top_sims + 0.01; w = w / w.sum()
                profile_tr[i, t_idx] = (w * ref_y[top_idx]).sum()
        for i, fp in enumerate(fps_te):
            if fp is None: continue
            sims = np.array(BulkTanimotoSimilarity(fp, ref_fps))
            top_idx = np.argsort(sims)[::-1][:K]
            top_sims = sims[top_idx]
            if top_sims[0] < 0.2:
                profile_te[i, t_idx] = ref_y.mean()
            else:
                w = top_sims + 0.01; w = w / w.sum()
                profile_te[i, t_idx] = (w * ref_y[top_idx]).sum()

    # Show correlations with pec50
    print("\nCorrelation of each cross-target prediction with PXR pec50:")
    for t_idx, target in enumerate(targets):
        if profile_tr[:, t_idx].std() > 0:
            r = np.corrcoef(profile_tr[:, t_idx], y_tr)[0, 1]
            print(f"  {target}: r={r:.4f}, mean={profile_tr[:, t_idx].mean():.3f}")

    # Also compute promiscuity: mean activity across all targets
    promiscuity_tr = profile_tr.mean(axis=1)
    promiscuity_te = profile_te.mean(axis=1)
    max_other_tr = np.array([np.max(np.delete(profile_tr[i], 0)) for i in range(len(profile_tr))])  # exclude PXR if it's at index 0
    max_other_te = np.array([np.max(np.delete(profile_te[i], 0)) for i in range(len(profile_te))])

    # Base + cross-target profile + promiscuity
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    X_tr_aug = np.column_stack([X_tr, profile_tr, promiscuity_tr.reshape(-1,1), max_other_tr.reshape(-1,1)])
    X_te_aug = np.column_stack([X_te, profile_te, promiscuity_te.reshape(-1,1), max_other_te.reshape(-1,1)])

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)

    print("\n=== OOF comparison ===")
    for name, X in [("base", X_tr), ("base+cross_target", X_tr_aug)]:
        oof = np.zeros(len(y_tr))
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X[ti], y_tr[ti], eval_set=[(X[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(X[vi])
        print(f"  {name:25s}: OOF RAE = {rae(y_tr, oof):.4f}")

    # Final
    final_oof = np.zeros(len(y_tr))
    final_te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr_aug[ti], y_tr[ti], eval_set=[(X_tr_aug[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        final_oof[vi] = md.predict(X_tr_aug[vi])
        final_te_preds.append(md.predict(X_te_aug))
    final_te = np.mean(final_te_preds, axis=0)
    print(f"\nFinal nb258 OOF: {rae(y_tr, final_oof):.4f}  te_std={final_te.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb258_cross_target.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb258_cross_target.npy", final_te)
    np.save(DATA_PROCESSED / "profile_tr_nb258.npy", profile_tr)
    np.save(DATA_PROCESSED / "profile_te_nb258.npy", profile_te)

    # Stack with 239 components
    print("\n=== 5-way SLSQP w/ nb258 ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, final_oof])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(150):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"5-way SLSQP OOF: {best.fun:.4f}")
    for n, w in zip(['nb224', 'nb179s', 'mtd', 'loso', 'nb258'], best.x):
        print(f"  {n}: {w:.4f}")


if __name__ == "__main__":
    main()
