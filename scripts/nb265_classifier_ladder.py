"""nb265 -- Classifier ladder: binary → 5-class → 10-class → relabel-regression.

User's idea: move from EASY (binary) to HARD (regression) progressively, with
bin denoising and class balancing.

Stage A: Binary classifier (above/below pec50=4.5).
Stage B: 5-class quintile classifier.
Stage C: 10-class decile classifier.
Stage D: Each compound predicted bin centroid (denoised regression).

For each stage:
- LGBM multi-class
- Class-balanced training (upsample minority classes)
- Pad high bins with PubChem PXR active SMILES (predicted high)
- Pad low bins with random ZINC-like compounds (predicted low)

Compare OOF RAE of bin-centroid predictions vs nb239.
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


def main():
    print("=== nb265: Classifier ladder ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    print(f"Train: {len(y_tr)}, pec50 range {y_tr.min():.2f}-{y_tr.max():.2f}")

    # Featurize
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)

    # PubChem actives for padding
    print("Loading PubChem actives for padding...")
    lib = pd.read_parquet("data/external/pubchem_pxr_active_smiles.parquet")
    lib["std_smiles"] = lib["smiles"].apply(std_smi)
    lib = lib.dropna(subset=["std_smiles"]).reset_index(drop=True)
    excl = set(smiles_tr) | set(smiles_te)
    lib = lib[~lib["std_smiles"].isin(excl)].reset_index(drop=True)
    print(f"  {len(lib)} PubChem actives (for high-bin padding)")

    # Featurize PubChem actives
    print("Featurizing PubChem actives...")
    X_lib = combined(lib["std_smiles"].tolist()); X_lib = impute(X_lib)
    # Assign pseudo pec50: high active_rate → higher pec50
    # active_rate=1 → pec50=6.5, active_rate=0.5 → 5.0, active_rate=0 → 4.0
    lib_pec50 = 4.0 + lib["active_rate"].values * 2.5
    print(f"  PubChem active pseudo-pec50: mean={lib_pec50.mean():.3f}, std={lib_pec50.std():.3f}")

    LGBM_CLS = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=10, n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)

    # ====================
    # Stage A: Binary classifier (>= 4.65 = median)
    # ====================
    print("\n=== Stage A: Binary classifier (above/below 4.65) ===")
    threshold = 4.65
    y_bin = (y_tr >= threshold).astype(int)
    print(f"  Active rate: {y_bin.mean():.3f}")

    # Pad PubChem actives as POSITIVE class (likely active)
    X_full_a = np.vstack([X_tr, X_lib])
    y_full_a = np.concatenate([y_bin, np.ones(len(X_lib), dtype=int)])
    w_full_a = np.concatenate([np.ones(len(X_tr)), np.full(len(X_lib), 0.3)])
    print(f"  Padded train: {len(X_full_a)} (PubChem actives weight 0.3)")

    oof_a_prob = np.zeros(len(y_tr))
    te_a_prob = []
    for ti, vi in folds:
        ti_full = np.concatenate([ti, np.arange(len(X_tr), len(X_full_a))])
        md = lgb.LGBMClassifier(objective="binary", **LGBM_CLS)
        md.fit(X_full_a[ti_full], y_full_a[ti_full], sample_weight=w_full_a[ti_full],
               eval_set=[(X_tr[vi], y_bin[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_a_prob[vi] = md.predict_proba(X_tr[vi])[:, 1]
        te_a_prob.append(md.predict_proba(X_te)[:, 1])
    te_a_prob = np.mean(te_a_prob, axis=0)

    # Convert binary prob to pec50 estimate: linear blend of class means
    pec_low = y_tr[y_bin == 0].mean()
    pec_high = y_tr[y_bin == 1].mean()
    oof_a_reg = oof_a_prob * pec_high + (1 - oof_a_prob) * pec_low
    te_a_reg = te_a_prob * pec_high + (1 - te_a_prob) * pec_low
    print(f"  Binary OOF as regression RAE: {rae(y_tr, oof_a_reg):.4f}")
    print(f"  te: mean={te_a_reg.mean():.3f}, std={te_a_reg.std():.3f}")

    # ====================
    # Stage B: 5-class quintile classifier
    # ====================
    print("\n=== Stage B: 5-class quintile classifier ===")
    quintile_edges = np.quantile(y_tr, [0.2, 0.4, 0.6, 0.8])
    print(f"  Quintile edges: {quintile_edges}")
    y_5cls = np.digitize(y_tr, quintile_edges)  # 0..4
    bin_centroids_5 = np.array([y_tr[y_5cls == c].mean() for c in range(5)])
    print(f"  Bin centroids (5): {bin_centroids_5}")
    print(f"  Class distribution: {np.bincount(y_5cls)}")

    # Pad: PubChem actives → top 2 bins (4 or 3)
    # Random "likely inactive" from any held-out lib (with active_rate<0.2) → bin 0
    lib_5cls = np.where(lib["active_rate"] >= 0.8, 4,
              np.where(lib["active_rate"] >= 0.5, 3,
              np.where(lib["active_rate"] >= 0.2, 2, 0)))
    X_full_b = np.vstack([X_tr, X_lib])
    y_full_b = np.concatenate([y_5cls, lib_5cls])
    w_full_b = np.concatenate([np.ones(len(X_tr)), np.full(len(X_lib), 0.3)])

    oof_b_classes = np.zeros((len(y_tr), 5))
    te_b_classes = []
    for ti, vi in folds:
        ti_full = np.concatenate([ti, np.arange(len(X_tr), len(X_full_b))])
        md = lgb.LGBMClassifier(objective="multiclass", num_class=5, **LGBM_CLS)
        md.fit(X_full_b[ti_full], y_full_b[ti_full], sample_weight=w_full_b[ti_full],
               eval_set=[(X_tr[vi], y_5cls[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_b_classes[vi] = md.predict_proba(X_tr[vi])
        te_b_classes.append(md.predict_proba(X_te))
    te_b_classes = np.mean(te_b_classes, axis=0)

    # Convert to pec50: expected value over bin centroids
    oof_b_reg = oof_b_classes @ bin_centroids_5
    te_b_reg = te_b_classes @ bin_centroids_5
    print(f"  5-class OOF as regression RAE: {rae(y_tr, oof_b_reg):.4f}")
    print(f"  te: mean={te_b_reg.mean():.3f}, std={te_b_reg.std():.3f}")

    # ====================
    # Stage C: 10-class decile classifier
    # ====================
    print("\n=== Stage C: 10-class decile classifier ===")
    decile_edges = np.quantile(y_tr, np.arange(0.1, 1.0, 0.1))
    y_10cls = np.digitize(y_tr, decile_edges)  # 0..9
    bin_centroids_10 = np.array([y_tr[y_10cls == c].mean() for c in range(10)])
    print(f"  10-class centroids: {bin_centroids_10}")

    lib_10cls = np.clip((lib["active_rate"].values * 8).astype(int) + 1, 0, 9)
    y_full_c = np.concatenate([y_10cls, lib_10cls])

    oof_c_classes = np.zeros((len(y_tr), 10))
    te_c_classes = []
    for ti, vi in folds:
        ti_full = np.concatenate([ti, np.arange(len(X_tr), len(X_full_b))])
        md = lgb.LGBMClassifier(objective="multiclass", num_class=10, **LGBM_CLS)
        md.fit(X_full_b[ti_full], y_full_c[ti_full], sample_weight=w_full_b[ti_full],
               eval_set=[(X_tr[vi], y_10cls[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_c_classes[vi] = md.predict_proba(X_tr[vi])
        te_c_classes.append(md.predict_proba(X_te))
    te_c_classes = np.mean(te_c_classes, axis=0)
    oof_c_reg = oof_c_classes @ bin_centroids_10
    te_c_reg = te_c_classes @ bin_centroids_10
    print(f"  10-class OOF as regression RAE: {rae(y_tr, oof_c_reg):.4f}")
    print(f"  te: mean={te_c_reg.mean():.3f}, std={te_c_reg.std():.3f}")

    # ====================
    # Save all + check blends
    # ====================
    np.save(DATA_PROCESSED / "oof_nb265_binary.npy", oof_a_reg)
    np.save(DATA_PROCESSED / "te_nb265_binary.npy", te_a_reg)
    np.save(DATA_PROCESSED / "oof_nb265_5cls.npy", oof_b_reg)
    np.save(DATA_PROCESSED / "te_nb265_5cls.npy", te_b_reg)
    np.save(DATA_PROCESSED / "oof_nb265_10cls.npy", oof_c_reg)
    np.save(DATA_PROCESSED / "te_nb265_10cls.npy", te_c_reg)

    # Stack with 239
    print("\n=== Stack with 239 ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")

    for cls_name, oof_c, te_c in [("binary", oof_a_reg, te_a_reg),
                                    ("5cls", oof_b_reg, te_b_reg),
                                    ("10cls", oof_c_reg, te_c_reg)]:
        M = np.column_stack([nb224, nb179s, mtd, loso, oof_c])
        def loss(w): return rae(y_tr, M @ w)
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0, 1.0)] * 5
        best = None
        for seed in range(100):
            rng = np.random.default_rng(seed)
            w0 = rng.dirichlet(np.ones(5))
            res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
            if best is None or res.fun < best.fun: best = res
        print(f"  +{cls_name}: 5-way OOF = {best.fun:.4f}, weight on classifier = {best.x[4]:.4f}")


if __name__ == "__main__":
    main()
