"""nb312 -- Label Denoising via Noise Score + Neighborhood Relabeling.

Idea: many of the 4139 PXR train labels are noisy (median pEC50 SE ~0.24,
max ~1.0). Identify the noisiest compounds by combining three signals:
  1. pEC50_std.error (assay-derived noise)
  2. Disagreement with k-NN (sim>=0.5) neighborhood mean
  3. Residual under nb239 (the current best blended OOF)

Top 10% noisy compounds are relabeled with a shrinkage toward the
similarity-weighted neighborhood mean (shrinkage = 0.7). We then retrain
LGBM on the relabeled training set and SCORE THE OOF AGAINST THE ORIGINAL y
so the OOF metric stays honest.
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
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW


def fp_bits(smi, r=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n_bits)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def main():
    print("=== nb312: Label Denoising + Relabeling ===\n")
    # ---- 1. load original y + SE + smiles -----------------------------------
    tr = load_train()
    tr = add_standard_columns(tr)
    raw = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    se = raw["pEC50_std.error (-log10(molarity))"].values.astype(np.float64)
    # Some rows have NaN SE -- impute with median.
    se = np.where(np.isnan(se), np.nanmedian(se), se)
    y_orig = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    scaffolds = tr["scaffold"].tolist()
    print(f"Train: {len(smiles_tr)}, median SE={np.median(se):.3f}, max SE={se.max():.3f}")

    te_df = load_test()
    smiles_te = te_df["smiles"].tolist()

    # ---- 2. Morgan FPs ------------------------------------------------------
    print("\nComputing Morgan FPs ...")
    fps = [fp_bits(s) for s in smiles_tr]
    valid_idx = [i for i, f in enumerate(fps) if f is not None]
    valid_fps = [fps[i] for i in valid_idx]

    # ---- 3. Neighborhood mean (sim>=0.4 top-5) -----------------------------
    print("Computing neighborhood means (top-5, sim>=0.4) ...")
    nbr_mean = np.copy(y_orig)
    nbr_disagree = np.zeros(len(y_orig))
    for i, f in enumerate(fps):
        if f is None:
            nbr_disagree[i] = 0.0
            continue
        sims = np.array(BulkTanimotoSimilarity(f, valid_fps))
        # exclude self
        pos_self = valid_idx.index(i) if i in valid_idx else None
        if pos_self is not None:
            sims[pos_self] = -1
        # top-5 with sim>=0.4
        order = np.argsort(-sims)
        chosen = [j for j in order[:30] if sims[j] >= 0.4][:5]
        if len(chosen) == 0:
            nbr_disagree[i] = 0.0
            continue
        nbr_y = np.array([y_orig[valid_idx[k]] for k in chosen])
        nbr_w = np.array([sims[k] for k in chosen])
        nbr_w = np.maximum(nbr_w, 1e-6)
        m = float((nbr_y * nbr_w).sum() / nbr_w.sum())
        nbr_mean[i] = m
        nbr_disagree[i] = abs(y_orig[i] - m)
        if (i + 1) % 500 == 0:
            print(f"  scanned {i+1}/{len(fps)}")

    # ---- 4. nb239 residual -------------------------------------------------
    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    residual = np.abs(y_orig - nb239_oof)
    print(f"nb239 OOF RAE={rae(y_orig, nb239_oof):.4f}")

    # ---- 5. noise score = z(SE) + z(|residual|) + z(|y - nbr_mean|) ---------
    noise_score = zscore(se) + zscore(residual) + zscore(nbr_disagree)
    pct = np.quantile(noise_score, 0.90)
    noisy_mask = noise_score >= pct
    print(f"\nNoisy compounds (top 10%): {noisy_mask.sum()} / {len(y_orig)}")
    print(f"  mean original |y - nbr_mean| for noisy: {nbr_disagree[noisy_mask].mean():.3f}")
    print(f"  mean original |y - nbr_mean| for clean: {nbr_disagree[~noisy_mask].mean():.3f}")

    # ---- 6. relabel noisy with shrinkage=0.7 toward nbr_mean ---------------
    SHRINK = 0.7
    y_relabel = y_orig.copy()
    y_relabel[noisy_mask] = (1 - SHRINK) * y_orig[noisy_mask] + SHRINK * nbr_mean[noisy_mask]
    print(f"Relabeled. y_orig std={y_orig.std():.3f}, y_relabel std={y_relabel.std():.3f}")

    # ---- 7. retrain LGBM (combined features), scaffold 5-fold ---------------
    print("\nFeaturizing combined + impute ...")
    X_tr = impute(combined(smiles_tr))
    X_te = impute(combined(smiles_te))
    print(f"X_tr shape: {X_tr.shape}, X_te shape: {X_te.shape}")

    splits = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)
    oof = np.zeros(len(y_orig))
    te_pred_folds = np.zeros((5, len(smiles_te)))
    for k, (ti, vi) in enumerate(splits):
        md = lgb.LGBMRegressor(
            n_estimators=2000, num_leaves=63, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.8,
            min_child_samples=10, reg_alpha=0.01, reg_lambda=0.01,
            objective="mae", n_jobs=4, random_state=42, verbose=-1)
        # Train on RELABELED y; validate against ORIGINAL y for OOF reporting
        md.fit(X_tr[ti], y_relabel[ti],
               eval_set=[(X_tr[vi], y_orig[vi])],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr[vi])
        te_pred_folds[k] = md.predict(X_te)
        print(f"  fold {k}: RAE={rae(y_orig[vi], oof[vi]):.4f}")
    te_pred = te_pred_folds.mean(axis=0)

    r = rae(y_orig, oof)
    sp = spearmanr(y_orig, oof).statistic
    print(f"\nnb312 OOF RAE={r:.4f}  Spearman={sp:.4f}  te_std={te_pred.std():.3f}")
    print(f"  te mean={te_pred.mean():.3f}, min={te_pred.min():.3f}, max={te_pred.max():.3f}")

    np.save(DATA_PROCESSED / "oof_nb312_relabeled.npy", oof)
    np.save(DATA_PROCESSED / "te_nb312_relabeled.npy", te_pred)

    # ---- 8. SLSQP 5-way with nb239 base ------------------------------------
    print("\nSLSQP 5-way blend with nb239 base components ...")
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, oof])
    def loss_fn(w): return rae(y_orig, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(80):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss_fn, w0, method='SLSQP',
                       bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    print(f"5-way SLSQP RAE={best.fun:.4f}")
    print(f"  weights: nb224={best.x[0]:.4f}, nb179s={best.x[1]:.4f}, "
          f"mtd={best.x[2]:.4f}, loso={best.x[3]:.4f}, nb312={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
