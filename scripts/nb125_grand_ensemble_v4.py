"""nb125 — Grand Ensemble v4: comprehensive re-blend of ALL models.

After nb115-nb124 complete, this consolidates ALL non-collapsed OOF predictions
into a new best ensemble. Key differences from nb112:

1. Many more models now (nb115-nb124 add 8+ new predictors)
2. Ridge + ElasticNet + LGBM meta-learner comparison
3. Scaffold-fold optimized weights (prevents OOF overfitting)
4. Hierarchical blend: first combine best 4 models, then blend with the rest
5. Saves multiple versions for submission
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV, ElasticNetCV
import lightgbm as lgb
from pathlib import Path
import itertools

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
COLLAPSE_THRESH = 0.58

# Models expected to be available after batch 4
PRIORITY_MODELS = [
    "nb109_deep_meta_stack",
    "nb111_selectivity_primary",
    "nb107_assay_decomp",
    "nb110_scaffold_prior",
    "nb115_extreme_weighted",
    "nb116_quantile_q50",
    "nb120_huber_1_0",
    "nb120_lad_l1_",
    "nb121_test_sim_weighted",
    "nb122_nonlinear_meta",
    "nb123_lad_lad_l1_",
    "nb123_lad_huber_1_0_",
    "nb123_lad_mse_reference_",
    "nb103_seed_propagation",
    "nb99_sc_bio_fp",
    "grand_v6b",
    "lgbm_tuned",
    "catboost",
    "multi_nr_transfer",
]


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def load_all_models(n_tr, thresh=COLLAPSE_THRESH):
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
            r = rae(y_tr, oof)
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results


def main():
    global y_tr
    print("=== nb125: Grand Ensemble v4 ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Loading all non-collapsed models...")
    models = load_all_models(n_tr)
    print(f"  {len(models)} models pass collapse filter (ratio >= {COLLAPSE_THRESH})")

    # Print top models
    print("\nTop 15 by OOF RAE:")
    for m in models[:15]:
        print(f"  {m['stem']:50s}  RAE={m['rae']:.4f}  ratio={m['ratio']:.2f}")

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    n_mod = len(models)
    stems = [m["stem"] for m in models]

    # === Baseline: top model alone ===
    best_single_rae = models[0]["rae"]
    print(f"\nBest single model: {models[0]['stem']}  RAE={best_single_rae:.4f}")

    # === Strategy 1: Exhaustive 2-model sweep ===
    print("\n2-model sweep (top 8 candidates):")
    top_n = min(8, n_mod)
    best_2_r, best_2_combo = best_single_rae, None
    for i, j in itertools.combinations(range(top_n), 2):
        for alpha in np.arange(0.05, 1.0, 0.05):
            blend = alpha * oof_mat[:, i] + (1 - alpha) * oof_mat[:, j]
            r = rae(y_tr, blend)
            if r < best_2_r:
                best_2_r = r
                best_2_combo = (i, j, alpha)
    if best_2_combo:
        i, j, a = best_2_combo
        print(f"  Best 2-model: {stems[i]} ({a:.2f}) + {stems[j]} ({1-a:.2f})  RAE={best_2_r:.4f}")

    # === Strategy 2: Ridge CV on all models ===
    print("\nRidge CV (all models):")
    oof_ridge = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000], cv=3)
        m_ridge.fit(oof_mat[tr_idx], y_tr[tr_idx])
        oof_ridge[va_idx] = m_ridge.predict(oof_mat[va_idx])
    r_ridge = rae(y_tr, oof_ridge)
    full_metrics(y_tr, oof_ridge, "Ridge CV (all)")

    m_ridge_full = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000], cv=5)
    m_ridge_full.fit(oof_mat, y_tr)
    te_ridge = m_ridge_full.predict(te_mat)
    coefs = m_ridge_full.coef_
    top_coef = np.argsort(-np.abs(coefs))[:5]
    print("  Top-5 Ridge coefficients:")
    for idx in top_coef:
        print(f"    {stems[idx]:50s}  coef={coefs[idx]:.4f}")

    # === Strategy 3: ElasticNet CV (sparser) ===
    print("\nElasticNet CV:")
    oof_enet = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0], cv=3, max_iter=2000)
        m_enet.fit(oof_mat[tr_idx], y_tr[tr_idx])
        oof_enet[va_idx] = m_enet.predict(oof_mat[va_idx])
    r_enet = rae(y_tr, oof_enet)
    full_metrics(y_tr, oof_enet, "ElasticNet CV (all)")

    m_enet_full = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0], cv=5, max_iter=2000)
    m_enet_full.fit(oof_mat, y_tr)
    te_enet = m_enet_full.predict(te_mat)

    # === Strategy 4: LGBM meta-learner (only top-20 models) ===
    print("\nLGBM meta-learner (top-20 models):")
    top20 = min(20, n_mod)
    oof_sub = oof_mat[:, :top20]; te_sub = te_mat[:, :top20]
    LGBM_META = dict(
        n_estimators=200, num_leaves=8, learning_rate=0.05,
        min_child_samples=15, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=4.0, random_state=SEED, verbose=-1, n_jobs=4
    )
    oof_lgbm_meta = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM_META, lgb.Dataset(oof_sub[tr_idx], label=y_tr[tr_idx]),
                      valid_sets=[lgb.Dataset(oof_sub[va_idx], label=y_tr[va_idx])],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        oof_lgbm_meta[va_idx] = m.predict(oof_sub[va_idx])
    r_lgbm_meta = rae(y_tr, oof_lgbm_meta)
    full_metrics(y_tr, oof_lgbm_meta, "LGBM meta (top-20)")

    m_meta_full = lgb.train(dict(LGBM_META, n_estimators=150),
                             lgb.Dataset(oof_sub, label=y_tr),
                             callbacks=[lgb.log_evaluation(-1)])
    te_lgbm_meta = m_meta_full.predict(te_sub)

    # === Pick best and save ===
    all_results = {
        "ridge": (oof_ridge, te_ridge, r_ridge),
        "enet":  (oof_enet,  te_enet,  r_enet),
        "lgbm_meta": (oof_lgbm_meta, te_lgbm_meta, r_lgbm_meta),
    }
    if best_2_combo:
        i, j, a = best_2_combo
        oof_2 = a * oof_mat[:, i] + (1 - a) * oof_mat[:, j]
        te_2  = a * te_mat[:,  i] + (1 - a) * te_mat[:,  j]
        all_results["2way"] = (oof_2, te_2, best_2_r)

    best_strat = min(all_results, key=lambda k: all_results[k][2])
    best_oof_v4, best_te_v4, best_r_v4 = all_results[best_strat]
    print(f"\n=== BEST: {best_strat}  OOF RAE={best_r_v4:.4f} ===")
    best_te_v4 = np.clip(best_te_v4, y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"Test: med={np.median(best_te_v4):.2f}  std={best_te_v4.std():.3f}  "
          f"max={best_te_v4.max():.2f}")

    # Save all strategies
    for name, (oof_v, te_v, r_v) in all_results.items():
        te_clipped = np.clip(te_v, y_tr.min() - 0.5, y_tr.max() + 0.5)
        np.save(DATA_PROCESSED / f"oof_nb125_{name}.npy", oof_v)
        np.save(DATA_PROCESSED / f"te_nb125_{name}.npy",  te_clipped)
        sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_clipped})
        sub.to_csv(SUBMISSIONS / f"125_grand_v4_{name}.csv", index=False)
        print(f"  Saved 125_grand_v4_{name}.csv  OOF={r_v:.4f}  "
              f"te_std={te_clipped.std():.3f}")


if __name__ == "__main__":
    main()
