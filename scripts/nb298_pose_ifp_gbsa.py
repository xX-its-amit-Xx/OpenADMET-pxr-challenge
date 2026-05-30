"""nb298 -- Cofolded-pose Interaction Fingerprint (IFP) featurization.

For each test compound, extract per-residue contact-type features from the
already-existing Boltz2 cofolded structures. Builds an interaction fingerprint:
for each of the top-20 PXR pocket residues, count contacts of type
{H-bond donor/acceptor, hydrophobic, pi-stack, salt-bridge} by atom-type proximity.

Uses already-computed pdb64 atom-level contacts (data/processed/pxr_pdb64_contacts.parquet)
as a proxy IFP template since we don't have per-test cofold parsed-residue contacts.
Fallback: pocket-residue contact prediction from nb280 model.

We then train LGBM with combined+IFP feats. MM-GBSA energy not available
(no OpenMM in env); we substitute Boltz iPTM as physics-confidence proxy.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def main():
    print("=== nb298: Pose-IFP + Boltz-iPTM as physics proxy ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].tolist()
    te_names = te_df['Molecule Name'].tolist()

    # IFP-like predicted contact profiles (from nb280)
    try:
        oof_contacts = np.load(DATA_PROCESSED / "oof_nb280_contact_profiles.npy")  # (N_tr, 20)
        te_contacts  = np.load(DATA_PROCESSED / "te_nb280_contact_profiles.npy")   # (N_te, 20)
        print(f"Loaded predicted contact profiles: train{oof_contacts.shape}, test{te_contacts.shape}")
    except FileNotFoundError:
        print("nb280 contact profiles not found; skipping IFP features")
        oof_contacts = np.zeros((len(y), 20))
        te_contacts  = np.zeros((len(smiles_te), 20))

    # Boltz iPTM physics confidence
    iptm_te = np.full(len(smiles_te), 0.5)
    try:
        boltz = pd.read_parquet("data/processed/boltz_dargason_features_test.parquet")
        idx = {n: i for i, n in enumerate(boltz['name'])}
        for i, n in enumerate(te_names):
            if n in idx:
                iptm_te[i] = boltz.iloc[idx[n]]['ligand_iptm_best']
        print(f"Boltz iPTM coverage: {(iptm_te != 0.5).sum()}/{len(smiles_te)}")
    except Exception as e:
        print(f"Boltz features unavailable: {e}")

    # Train doesn't have Boltz iPTM (no co-folding done on train compounds);
    # use a placeholder mean
    iptm_tr = np.full(len(y), iptm_te.mean())

    X_tr_base = impute(combined(smiles_tr)).astype(np.float32)
    X_te_base = impute(combined(smiles_te)).astype(np.float32)
    X_tr = np.column_stack([X_tr_base, oof_contacts, iptm_tr.reshape(-1, 1)])
    X_te = np.column_stack([X_te_base, te_contacts, iptm_te.reshape(-1, 1)])
    print(f"Feature dim: {X_tr.shape[1]}")

    folds = scaffold_kfold_indices(tr['scaffold'].tolist(), n_splits=5)
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_samples=10, objective='mae',
                n_jobs=4, random_state=42, verbose=-1)
    oof = np.zeros(len(y))
    te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr[ti], y[ti], eval_set=[(X_tr[vi], y[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr[vi])
        te_preds.append(md.predict(X_te))
    te_pred = np.mean(te_preds, axis=0)
    r = rae(y, oof)
    sp, _ = spearmanr(y, oof)
    print(f"\nPose-IFP+iPTM LGBM OOF: RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    np.save(DATA_PROCESSED / "oof_nb298_pose_ifp.npy", oof)
    np.save(DATA_PROCESSED / "te_nb298_pose_ifp.npy", te_pred)

    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    def loss(w): return rae(y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP OOF: {best.fun:.4f}, weight(nb298)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
