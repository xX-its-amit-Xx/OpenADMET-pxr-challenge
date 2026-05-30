"""nb132 — Diverse Seed Ensemble (Model Soup for LGBM).

Train many LGBM instances with different random seeds, slight hyperparameter
perturbations, and data augmentation variations. Average all predictions.

Motivation: Each seed produces a slightly different model due to:
  - Different column/row subsampling
  - Different tree split ordering at ties
  - Small numerical differences in leaf splits

With 20 seeds × 3 hyperparameter configs = 60 models, the averaged prediction
should have lower variance than any single model. This is the LGBM equivalent
of "model soup" / SWA.

Key: OOF computed via averaging OOF predictions of all 60 models, so the
OOF estimate is honest (no leakage from different seeds seeing same fold data).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
N_SEEDS = 15
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    verbose=-1, n_jobs=4
)

# 3 hyperparameter configs (slightly varied)
CONFIGS = [
    dict(n_estimators=500, num_leaves=48, learning_rate=0.04, min_child_samples=8,
         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, verbose=-1, n_jobs=4),
    dict(n_estimators=600, num_leaves=32, learning_rate=0.05, min_child_samples=10,
         subsample=0.75, colsample_bytree=0.75, reg_alpha=0.05, verbose=-1, n_jobs=4),
    dict(n_estimators=700, num_leaves=64, learning_rate=0.03, min_child_samples=6,
         subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, verbose=-1, n_jobs=4),
]


def full_metrics(y_true, y_pred, label=""):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return rae_v


def main():
    print(f"=== nb132: Diverse Seed Ensemble ({N_SEEDS} seeds x {len(CONFIGS)} configs) ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw    = raw_train[emax_col].values.astype(np.float64)
    emax_log    = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_imputed = np.where(np.isnan(pec50_null), np.nanmedian(pec50_null), pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    print("Computing structural features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    print("Aux OOF (assay decomposition)...")
    oof_emax = np.full(n_tr, np.nan)
    oof_null = np.full(n_tr, np.nan)
    oof_sel  = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        aux_seed = 42
        m_em = lgb.train(dict(LGBM_AUX, random_state=aux_seed),
                         lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_tr[va_idx])
        m_nl = lgb.train(dict(LGBM_AUX, random_state=aux_seed),
                         lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_tr[va_idx])
        m_sl = lgb.train(dict(LGBM_AUX, random_state=aux_seed),
                         lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx] = m_sl.predict(X_tr[va_idx])

    m_em_f = lgb.train(dict(LGBM_AUX, random_state=42), lgb.Dataset(X_tr, label=emax_log),
                       callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(dict(LGBM_AUX, random_state=42), lgb.Dataset(X_tr, label=null_imputed),
                       callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(dict(LGBM_AUX, random_state=42), lgb.Dataset(X_tr, label=selectivity),
                       callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null,
                                  np.log1p(np.clip(oof_emax, 0, None))])
    assay_te  = np.column_stack([te_emax, te_null, te_sel,
                                  np.zeros(len(X_te)),
                                  np.log1p(np.clip(te_emax, 0, None))])
    X_aug = np.hstack([X_tr, assay_oof])
    X_aug_te = np.hstack([X_te, assay_te])

    # === Diverse seed ensemble ===
    total_models = N_SEEDS * len(CONFIGS)
    all_oofs = np.zeros((n_tr, total_models))
    all_tes  = np.zeros((len(X_te), total_models))

    model_idx = 0
    for cfg_idx, cfg in enumerate(CONFIGS):
        for seed in range(N_SEEDS):
            rng_seed = seed * 100 + cfg_idx * 1000 + 42
            params = dict(cfg, random_state=rng_seed)

            oof = np.full(n_tr, np.nan)
            for fold, (tr_idx, va_idx) in enumerate(splits):
                m = lgb.train(params, lgb.Dataset(X_aug[tr_idx], label=y_tr[tr_idx]),
                              callbacks=[lgb.log_evaluation(-1)])
                oof[va_idx] = m.predict(X_aug[va_idx])

            m_full = lgb.train(params, lgb.Dataset(X_aug, label=y_tr),
                               callbacks=[lgb.log_evaluation(-1)])
            te_pred = m_full.predict(X_aug_te)

            all_oofs[:, model_idx] = oof
            all_tes[:, model_idx]  = te_pred
            model_idx += 1

            if model_idx % 5 == 0:
                avg_oof_so_far = np.nanmean(all_oofs[:, :model_idx], axis=1)
                r_so_far = rae(y_tr, avg_oof_so_far)
                print(f"  [{model_idx:3d}/{total_models}]  cfg={cfg_idx}  seed={seed:2d}  "
                      f"avg_RAE={r_so_far:.4f}", flush=True)

    oof_mean = np.nanmean(all_oofs, axis=1)
    te_mean  = all_tes.mean(axis=1)
    te_mean  = np.clip(te_mean, y_tr.min() - 0.5, y_tr.max() + 0.5)

    full_metrics(y_tr, oof_mean, f"Diverse seed ensemble ({total_models} models)")
    ratio = te_mean.std() / oof_mean.std()
    print(f"  Test: med={np.median(te_mean):.2f}  std={te_mean.std():.3f}  ratio={ratio:.2f}")

    np.save(DATA_PROCESSED / "oof_nb132_seed_ensemble.npy", oof_mean)
    np.save(DATA_PROCESSED / "te_nb132_seed_ensemble.npy",  te_mean)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_mean})
    sub.to_csv(SUBMISSIONS / "132_diverse_seed_ensemble.csv", index=False)
    print(f"\nSaved: submissions/132_diverse_seed_ensemble.csv")
    print(f"OOF RAE: {rae(y_tr, oof_mean):.4f}")


if __name__ == "__main__":
    main()
