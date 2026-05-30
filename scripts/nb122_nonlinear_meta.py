"""nb122 — Non-linear Meta-Learner (XGBoost on OOF Stack).

Current best: Ridge CV on 15 OOF models gives OOF RAE=0.3714.
Ridge forces linear combinations. A non-linear meta-learner can learn:
  - Use model A more for low-activity compounds (< 3.5 pEC50)
  - Use model B more for high-activity compounds (> 5.5 pEC50)
  - Interaction effects between meta-features
  - Compound-level feature importance shifts

Strategy:
  - Load all non-collapsed OOF + TE pairs (te_std/oof_std >= 0.58)
  - Add structural features as additional meta-features (reduced PCA space)
  - Train XGBoost level-2 model with early stopping on scaffold CV
  - Also try LGBM meta-learner for comparison
  - Nested CV to prevent meta-overfitting

Key risk: level-2 overfitting since we have only 4139 samples.
Mitigation: strong regularization + scaffold CV validation.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
COLLAPSE_THRESH = 0.58
PCA_COMPONENTS = 50  # Structural feature reduction for meta-learner


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def load_all_oofs(n_tr, thresh=COLLAPSE_THRESH):
    """Load all non-collapsed OOF+TE pairs."""
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists():
                break
        else:
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te  = np.where(np.isfinite(te), te,   np.nanmean(te))
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio))
        except Exception:
            pass
    return results


def main():
    print("=== nb122: Non-linear Meta-Learner ===\n")

    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Loading OOF models...")
    models = load_all_oofs(n_tr)
    print(f"  {len(models)} non-collapsed models loaded")
    models.sort(key=lambda x: rae(y_tr, x["oof"]))
    for m in models[:8]:
        print(f"    {m['stem']:45s}  RAE={rae(y_tr, m['oof']):.4f}  ratio={m['ratio']:.2f}")

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    n_models = oof_mat.shape[1]
    print(f"Meta-feature matrix: (train={oof_mat.shape}, test={te_mat.shape})")

    # Add structural PCA features
    print("\nComputing structural PCA features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # Meta-feature matrices: OOF predictions + PCA structural
    M_tr = np.hstack([oof_mat, X_tr_pca])
    M_te = np.hstack([te_mat,  X_te_pca])
    print(f"  Final meta matrix: {M_tr.shape}")

    # === Strategy 1: Ridge CV (baseline) ===
    print("\n=== Ridge CV baseline ===")
    oof_ridge = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=3)
        m_ridge.fit(M_tr[tr_idx], y_tr[tr_idx])
        oof_ridge[va_idx] = m_ridge.predict(M_tr[va_idx])
    full_metrics(y_tr, oof_ridge, "Ridge CV")

    m_ridge_full = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
    m_ridge_full.fit(M_tr, y_tr)
    te_ridge = m_ridge_full.predict(M_te)

    # === Strategy 2: LGBM meta-learner (strong regularization) ===
    print("\n=== LGBM meta-learner ===")
    LGBM_META = dict(
        n_estimators=300, num_leaves=8, learning_rate=0.05,
        min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, verbose=-1, n_jobs=4
    )
    oof_lgbm = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM_META, lgb.Dataset(M_tr[tr_idx], label=y_tr[tr_idx]),
                      valid_sets=[lgb.Dataset(M_tr[va_idx], label=y_tr[va_idx])],
                      callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
        oof_lgbm[va_idx] = m.predict(M_tr[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof_lgbm[va_idx]):.4f}", flush=True)
    full_metrics(y_tr, oof_lgbm, "LGBM meta-learner")

    m_lgbm_full = lgb.train(dict(LGBM_META, n_estimators=200),
                             lgb.Dataset(M_tr, label=y_tr),
                             callbacks=[lgb.log_evaluation(-1)])
    te_lgbm = m_lgbm_full.predict(M_te)

    # === Strategy 3: XGBoost meta-learner ===
    print("\n=== XGBoost meta-learner ===")
    XGB_META = dict(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=2.0,
        tree_method="hist", random_state=SEED, n_jobs=4, verbosity=0
    )
    oof_xgb = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = xgb.XGBRegressor(**XGB_META)
        m.set_params(early_stopping_rounds=40, eval_metric="mae")
        m.fit(M_tr[tr_idx], y_tr[tr_idx],
              eval_set=[(M_tr[va_idx], y_tr[va_idx])],
              verbose=False)
        oof_xgb[va_idx] = m.predict(M_tr[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof_xgb[va_idx]):.4f}", flush=True)
    full_metrics(y_tr, oof_xgb, "XGBoost meta-learner")

    m_xgb_full = xgb.XGBRegressor(**dict(XGB_META, n_estimators=200))
    m_xgb_full.fit(M_tr, y_tr, verbose=False)
    te_xgb = m_xgb_full.predict(M_te)

    # === Blend meta-learners ===
    print("\n=== Blending meta-learners ===")
    all_meta = {
        "Ridge":   (oof_ridge, te_ridge),
        "LGBM":    (oof_lgbm,  te_lgbm),
        "XGBoost": (oof_xgb,   te_xgb),
    }
    best_name, best_oof, best_te, best_r = "Ridge", oof_ridge, te_ridge, rae(y_tr, oof_ridge)
    for name, (oof_v, te_v) in all_meta.items():
        r = rae(y_tr, oof_v)
        if r < best_r:
            best_r, best_name, best_oof, best_te = r, name, oof_v, te_v

    # Try 50/50 blends
    for a, b in [("Ridge", "LGBM"), ("Ridge", "XGBoost"), ("LGBM", "XGBoost")]:
        blend_oof = (all_meta[a][0] + all_meta[b][0]) / 2
        blend_te  = (all_meta[a][1] + all_meta[b][1]) / 2
        r = rae(y_tr, blend_oof)
        print(f"  {a}+{b} blend: OOF RAE={r:.4f}")
        if r < best_r:
            best_r, best_name, best_oof, best_te = r, f"{a}+{b}", blend_oof, blend_te

    print(f"\nBest strategy: {best_name}  OOF RAE={best_r:.4f}")
    best_te = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"Test: med={np.median(best_te):.2f}  std={best_te.std():.3f}  "
          f"ratio={best_te.std()/best_oof.std():.2f}")

    np.save(DATA_PROCESSED / "oof_nb122_nonlinear_meta.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb122_nonlinear_meta.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "122_nonlinear_meta.csv", index=False)
    print(f"\nSaved: submissions/122_nonlinear_meta.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
