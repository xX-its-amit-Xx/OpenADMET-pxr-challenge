"""nb245 -- 1000-LGBM massive ensemble.

Wildly unconventional: train 1000 LGBMs with random hyperparameters, random
row subsets, random feature subsets, random scaffold-fold seeds, random label
noise injection. Each model contributes a tiny prediction; mean = ensemble.

Key tricks:
- Random bootstrap of train (50-100% size)
- Random feature subsetting (50-100% of 2265 features)
- Random hyperparameter sampling
- Random scaffold fold seed
- Optional Gaussian noise (sigma 0-0.2) added to labels for regularization

Aggregations: mean, median, trimmed mean, quantile predictions.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def random_hparams(rng):
    return dict(
        n_estimators=int(rng.choice([500, 800, 1200, 1500, 2000])),
        num_leaves=int(rng.choice([15, 31, 47, 63, 95, 127])),
        learning_rate=float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        subsample=float(rng.uniform(0.5, 0.95)),
        colsample_bytree=float(rng.uniform(0.4, 0.95)),
        min_child_samples=int(rng.choice([5, 10, 15, 20, 30, 50])),
        reg_alpha=float(rng.choice([0, 0.001, 0.01, 0.1, 1.0])),
        reg_lambda=float(rng.choice([0, 0.001, 0.01, 0.1, 1.0])),
        objective=str(rng.choice(["mae", "huber", "regression"])),
        n_jobs=2,
        random_state=42 + int(rng.integers(0, 100000)),
        verbose=-1,
    )


def main():
    print("=== nb245: 1000-LGBM massive ensemble ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()
    print(f"Train: {len(smiles_tr)}, Test: {len(smiles_te)}")

    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    scaffolds = tr["scaffold"].tolist()

    N_MODELS = 1000
    n_tr = len(y_tr); n_te = len(smiles_te); n_feats = X_tr.shape[1]

    rng_master = np.random.default_rng(42)
    oof_all = np.zeros((N_MODELS, n_tr), dtype=np.float32)
    te_all = np.zeros((N_MODELS, n_te), dtype=np.float32)
    oof_raes = np.zeros(N_MODELS)
    valid_mask = np.ones(N_MODELS, dtype=bool)

    t0 = time.time()
    for i in range(N_MODELS):
        seed = int(rng_master.integers(0, 1_000_000))
        rng = np.random.default_rng(seed)
        params = random_hparams(rng)

        # Random row bootstrap
        n_sample = int(rng.uniform(0.5, 1.0) * n_tr)
        rows = rng.choice(n_tr, size=n_sample, replace=False)
        # Random feature subset
        feat_frac = rng.uniform(0.4, 0.95)
        cols = np.sort(rng.choice(n_feats, size=int(feat_frac * n_feats), replace=False))
        # Optional label noise
        noise_sigma = rng.uniform(0.0, 0.15)
        y_noisy = y_tr.copy()
        if noise_sigma > 0:
            y_noisy = y_noisy + rng.normal(0, noise_sigma, size=n_tr)

        # Scaffold CV (single split, since 1000 models is enough)
        # Use a single seed, only get OOF for the val fold of THIS model
        fold_seed = int(rng.integers(0, 1000000))
        folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=fold_seed)
        # Pick ONE fold for OOF eval
        fold_idx = int(rng.integers(0, 5))
        ti, vi = folds[fold_idx]
        # Use bootstrap rows that are in ti
        ti_use = np.intersect1d(rows, ti)
        if len(ti_use) < 50:
            valid_mask[i] = False
            continue
        try:
            md = lgb.LGBMRegressor(**params)
            md.fit(X_tr[ti_use][:, cols], y_noisy[ti_use],
                   eval_set=[(X_tr[vi][:, cols], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof_all[i, vi] = md.predict(X_tr[vi][:, cols])
            te_all[i] = md.predict(X_te[:, cols])
        except Exception:
            valid_mask[i] = False
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_MODELS - i - 1) / 60
            print(f"  {i+1}/{N_MODELS}  elapsed={elapsed/60:.1f}m  ETA={eta:.0f}m  valid={valid_mask[:i+1].sum()}")

    valid = valid_mask
    print(f"\nValid models: {valid.sum()}/{N_MODELS}")

    # Compute ensembled OOF (mean of all val predictions where they exist)
    oof_sum = np.zeros(n_tr); oof_count = np.zeros(n_tr)
    for i in np.where(valid)[0]:
        nonzero = np.abs(oof_all[i]) > 1e-9
        oof_sum[nonzero] += oof_all[i, nonzero]
        oof_count[nonzero] += 1
    oof_mean = np.where(oof_count > 0, oof_sum / np.maximum(oof_count, 1), y_tr.mean())
    print(f"OOF coverage: {(oof_count > 0).sum()}/{n_tr}")

    r_oof = rae(y_tr, oof_mean)
    print(f"Mean ensemble OOF RAE: {r_oof:.4f}")

    # Trimmed mean
    te_trim = np.zeros(n_te)
    for j in range(n_te):
        vals = te_all[valid, j]
        sorted_v = np.sort(vals)
        cut = int(len(sorted_v) * 0.15)
        te_trim[j] = sorted_v[cut:-cut].mean() if cut > 0 else sorted_v.mean()
    print(f"Test trim-mean: mean={te_trim.mean():.3f} std={te_trim.std():.3f}")

    # Mean test
    te_mean = te_all[valid].mean(axis=0)
    print(f"Test mean: mean={te_mean.mean():.3f} std={te_mean.std():.3f}")

    np.save(DATA_PROCESSED / "oof_nb245_massive_mean.npy", oof_mean)
    np.save(DATA_PROCESSED / "te_nb245_massive_mean.npy", te_mean)
    np.save(DATA_PROCESSED / "te_nb245_massive_trim.npy", te_trim)
    print("Saved nb245 arrays")


if __name__ == "__main__":
    main()
