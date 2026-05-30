"""nb172 -- Brute-force distribution learning.

Three sub-strategies in one script:

A) Train 100 LGBM variants with random hyperparameters (different seeds, leaves,
   learning rates, feature subsets). Save all OOF + test predictions. Compute
   per-test-compound prediction distribution. Pick test predictions that maximize
   ensemble consensus (lowest prediction variance).

B) Mislabel detection: identify train compounds where OOF residual is large AND
   pec50_se is large (high-confidence-of-noise). Mask them, retrain, compare.

C) Self-training: pseudo-label the K most-confident test predictions (smallest
   ensemble variance), add to train, retrain. Iterate up to 5 rounds.
"""
import os, sys, warnings, time, json
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def random_lgb_params(rng, base_seed):
    """Sample diverse LGBM hyperparameters."""
    return dict(
        n_estimators=int(rng.choice([800, 1200, 1500, 2000, 2500])),
        num_leaves=int(rng.choice([31, 47, 63, 95, 127])),
        learning_rate=float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        subsample=float(rng.uniform(0.6, 0.95)),
        colsample_bytree=float(rng.uniform(0.5, 0.95)),
        min_child_samples=int(rng.choice([5, 10, 15, 20, 30])),
        reg_alpha=float(rng.choice([0, 0.01, 0.1, 1.0])),
        reg_lambda=float(rng.choice([0, 0.01, 0.1, 1.0])),
        objective=str(rng.choice(["mae", "huber", "regression"])),
        n_jobs=4,
        random_state=base_seed,
        verbose=-1,
    )


def main():
    print("=== nb172: brute-force distribution learning ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    pec50_se = tr["pec50_se"].values.astype(np.float64) if "pec50_se" in tr.columns else np.full(len(tr), np.nan)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    print("Featurizing...")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    print(f"X_tr={X_tr.shape}, X_te={X_te.shape}")

    scaffolds = tr["scaffold"].tolist()

    # ------ A) 100 LGBM variants ------
    N_MODELS = 100
    print(f"\n[A] Training {N_MODELS} LGBM variants with random hyperparameters...")
    rng = np.random.default_rng(42)

    oof_all = np.zeros((N_MODELS, len(y_tr)))
    te_all = np.zeros((N_MODELS, len(smiles_te)))
    oof_rae_per_model = np.zeros(N_MODELS)

    t0 = time.time()
    for m_idx in range(N_MODELS):
        seed = 42 + m_idx
        params = random_lgb_params(rng, seed)
        folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=seed)

        # Subsample feature columns to add diversity
        n_feats = X_tr.shape[1]
        col_keep = rng.choice(n_feats, size=int(n_feats * rng.uniform(0.6, 1.0)), replace=False)
        col_keep = np.sort(col_keep)
        Xt = X_tr[:, col_keep]
        Xe = X_te[:, col_keep]

        oof = np.zeros(len(y_tr))
        te_preds = []
        for ti, vi in folds:
            md = lgb.LGBMRegressor(**params)
            md.fit(Xt[ti], y_tr[ti], eval_set=[(Xt[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(Xt[vi])
            te_preds.append(md.predict(Xe))
        oof_all[m_idx] = oof
        te_all[m_idx] = np.mean(te_preds, axis=0)
        oof_rae_per_model[m_idx] = rae(y_tr, oof)
        if (m_idx + 1) % 10 == 0:
            print(f"  {m_idx+1}/{N_MODELS}  best OOF RAE so far: {oof_rae_per_model[:m_idx+1].min():.4f}  elapsed={time.time()-t0:.0f}s")

    print(f"\n  Best single model OOF RAE: {oof_rae_per_model.min():.4f}")
    print(f"  Mean OOF RAE: {oof_rae_per_model.mean():.4f}")
    print(f"  Worst OOF RAE: {oof_rae_per_model.max():.4f}")

    # Ensemble: mean of models with OOF RAE < median
    threshold = np.median(oof_rae_per_model)
    keep = oof_rae_per_model < threshold
    print(f"\n  Keeping top {keep.sum()} models with OOF RAE < {threshold:.4f}")
    oof_ens = oof_all[keep].mean(axis=0)
    te_ens = te_all[keep].mean(axis=0)
    print(f"  Ensemble OOF RAE: {rae(y_tr, oof_ens):.4f}")

    # Compute per-test variance (confidence)
    te_var = te_all.std(axis=0)
    print(f"  Test prediction variance: min={te_var.min():.3f} max={te_var.max():.3f} median={np.median(te_var):.3f}")

    np.save(DATA_PROCESSED / "oof_nb172_brute_ensemble.npy", oof_ens)
    np.save(DATA_PROCESSED / "te_nb172_brute_ensemble.npy", te_ens)
    np.save(DATA_PROCESSED / "te_nb172_variance.npy", te_var)
    print("  Saved oof/te_nb172_brute_ensemble.npy + te_variance.npy")

    # ------ B) Mislabel detection ------
    print("\n[B] Mislabel detection via OOF residuals × SE...")
    residual = np.abs(y_tr - oof_ens)
    # High-residual AND high-SE = likely noise. High-residual AND low-SE = likely model failure (not mislabel).
    se_valid = np.isfinite(pec50_se)
    print(f"  pec50_se valid for {se_valid.sum()}/{len(y_tr)} train rows")
    if se_valid.sum() > 100:
        # Z-score residual and SE separately
        res_z = (residual - residual.mean()) / residual.std()
        se_z = np.where(se_valid, (pec50_se - np.nanmean(pec50_se)) / np.nanstd(pec50_se), 0)
        # Composite: high residual + high SE → likely mislabel
        composite = res_z + 0.5 * se_z
        # Mask top 5% as likely mislabel
        mask_threshold = np.percentile(composite, 95)
        suspect = composite >= mask_threshold
        print(f"  Suspect-mislabel count (composite>=95%): {suspect.sum()}")

        # Retrain without suspects
        keep_idx = ~suspect
        print(f"  Retraining on {keep_idx.sum()} clean train compounds...")
        clean_oof = np.zeros(len(y_tr))
        folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)
        median_params = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                             objective="mae", n_jobs=4, random_state=42, verbose=-1)
        for ti, vi in folds:
            ti = np.array([i for i in ti if keep_idx[i]])
            md = lgb.LGBMRegressor(**median_params)
            md.fit(X_tr[ti], y_tr[ti], eval_set=[(X_tr[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            clean_oof[vi] = md.predict(X_tr[vi])
        # Compare RAE on CLEAN vs FULL
        r_clean_full = rae(y_tr, clean_oof)
        r_full_full = rae(y_tr, oof_ens)
        print(f"  After mislabel masking - OOF RAE on ALL train: {r_clean_full:.4f}")
        print(f"  vs ensemble RAE on all: {r_full_full:.4f}")
        # On suspects only: did predictions improve?
        if suspect.sum() > 0:
            r_suspect_orig = rae(y_tr[suspect], oof_ens[suspect])
            r_suspect_clean = rae(y_tr[suspect], clean_oof[suspect])
            print(f"  RAE on SUSPECTS only - ensemble: {r_suspect_orig:.4f}, clean-retrain: {r_suspect_clean:.4f}")

        np.save(DATA_PROCESSED / "nb172_suspect_mislabel.npy", suspect)
        np.save(DATA_PROCESSED / "oof_nb172_clean.npy", clean_oof)
        print("  Saved nb172_suspect_mislabel.npy + oof_nb172_clean.npy")

    # ------ C) Self-training (pseudo-label most-confident test) ------
    print("\n[C] Self-training: pseudo-label most-confident test predictions...")
    # Confidence = inverse of test prediction variance
    confidence = -te_var
    confident_order = np.argsort(confidence)[::-1]
    n_pseudo = min(50, len(smiles_te))  # add top 50 most confident
    pseudo_idx = confident_order[:n_pseudo]
    pseudo_y = te_ens[pseudo_idx]
    print(f"  Pseudo-labeling {n_pseudo} most-confident test compounds (var threshold={te_var[pseudo_idx[-1]]:.3f})")
    print(f"  Pseudo y range: {pseudo_y.min():.2f} to {pseudo_y.max():.2f}")

    # Featurize pseudo-train (already done as part of X_te)
    X_aug = np.vstack([X_tr, X_te[pseudo_idx]])
    y_aug = np.concatenate([y_tr, pseudo_y])
    # Weight: pseudo labels 0.3
    w_aug = np.concatenate([np.ones(len(y_tr)), np.full(n_pseudo, 0.3)])
    # Need scaffolds for new compounds - reuse first scaffold for simplicity
    scaff_aug = scaffolds + [scaffolds[0]] * n_pseudo

    folds_aug = scaffold_kfold_indices(scaff_aug, n_splits=5, seed=42)
    # Map back to original train rows only (last n_pseudo are pseudo, evaluate on first len(y_tr))
    selftr_oof = np.zeros(len(y_tr))
    median_params = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                         objective="mae", n_jobs=4, random_state=42, verbose=-1)
    for ti, vi in folds_aug:
        # Skip evaluation on pseudo rows (they have weak labels)
        vi_orig = vi[vi < len(y_tr)]
        md = lgb.LGBMRegressor(**median_params)
        md.fit(X_aug[ti], y_aug[ti], sample_weight=w_aug[ti],
               eval_set=[(X_aug[vi], y_aug[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        if len(vi_orig) > 0:
            selftr_oof[vi_orig] = md.predict(X_aug[vi_orig])
    r_selftr = rae(y_tr, selftr_oof)
    print(f"  Self-training OOF RAE: {r_selftr:.4f}  vs ensemble {rae(y_tr, oof_ens):.4f}")
    np.save(DATA_PROCESSED / "oof_nb172_selftrain.npy", selftr_oof)
    print("  Saved oof_nb172_selftrain.npy")

    print("\n=== nb172 done ===")


if __name__ == "__main__":
    main()
