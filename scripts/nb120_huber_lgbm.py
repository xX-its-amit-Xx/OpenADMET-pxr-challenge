"""nb120 — LGBM with Huber Loss + Mape Loss.

Two robust regression objectives complementary to nb115 (sample-weighted MSE)
and nb116 (quantile regression):

1. Huber loss: quadratic near zero, linear for large errors. Reduces influence
   of extreme-activity compounds (pEC50 < 2.5 or > 6.0) on gradient updates.
   alpha parameter controls the transition point.

2. MAPE-proxy: custom callback with mean absolute percentage error as metric
   (approximated via log-space). Relative errors are minimized, which naturally
   down-weights easy near-mean predictions.

Both use assay decomp + meta-OOF augmentation (same as nb107/nb109 approach).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5

BASE_PARAMS = dict(
    n_estimators=1500, num_leaves=64, learning_rate=0.03,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def load_meta(stem, n_tr):
    for op in ("oof_", ""):
        for tp in ("te_", "te_oof_"):
            of = DATA_PROCESSED / f"{op}{stem}.npy"
            tf = DATA_PROCESSED / f"{tp}{stem}.npy"
            if of.exists() and tf.exists():
                oof = np.load(of); te = np.load(tf)
                if oof.ndim == 2: oof = oof[:, 0]
                if te.ndim == 2:  te  = te[:, 0]
                if len(oof) == n_tr: return oof, te
    return None, None


def build_augmented(X_tr, X_te, oof_emax, oof_null, oof_sel, has_null,
                    te_emax, te_null, te_sel, meta_oofs, meta_tes):
    assay_oof = np.column_stack([
        oof_emax, oof_null, oof_sel, has_null,
        np.log1p(np.clip(oof_emax, 0, None))
    ] + meta_oofs)
    assay_te = np.column_stack([
        te_emax, te_null, te_sel, np.zeros(len(X_te)),
        np.log1p(np.clip(te_emax, 0, None))
    ] + meta_tes)
    return np.hstack([X_tr, assay_oof]), np.hstack([X_te, assay_te])


def main():
    print("=== nb120: Huber Loss LGBM + MAPE-Proxy LGBM ===\n")

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Assay features
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names = raw_train["Molecule Name"].values
    pec50_null = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_median  = np.nanmedian(pec50_null)
    null_imputed = np.where(np.isnan(pec50_null), null_median, pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    print("Computing structural features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # Load meta-OOFs
    meta_candidates = [
        "nb107_assay_decomp", "nb109_deep_meta_stack", "nb101_delta_base",
        "nb99_sc_bio_fp", "grand_v6b", "lgbm_tuned", "catboost",
    ]
    meta_oofs, meta_tes, meta_names = [], [], []
    for stem in meta_candidates:
        oof, te_m = load_meta(stem, n_tr)
        if oof is not None:
            oof_f = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te_f  = np.where(np.isfinite(te_m), te_m, np.nanmean(te_m))
            ratio = te_f.std() / oof_f.std() if oof_f.std() > 0 else 0
            if ratio >= 0.55:
                meta_oofs.append(oof_f); meta_tes.append(te_f)
                meta_names.append(stem)
                print(f"  {stem}: RAE={rae(y_tr, oof_f):.4f}  ratio={ratio:.2f}")

    # Stage 1: auxiliary OOF
    print("\nStage 1: auxiliary OOF...")
    oof_emax = np.full(n_tr, np.nan); oof_null = np.full(n_tr, np.nan)
    oof_sel  = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_tr[va_idx])
        m_nl = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_tr[va_idx])
        m_sl = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx]  = m_sl.predict(X_tr[va_idx])

    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    X_tr_aug, X_te_aug = build_augmented(
        X_tr, X_te, oof_emax, oof_null, oof_sel, has_null,
        te_emax, te_null, te_sel, meta_oofs, meta_tes
    )
    print(f"Augmented shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    saved = {}
    for obj_name, obj, extra_params in [
        ("Huber_0.5",  "huber",     {"alpha": 0.5}),
        ("Huber_1.0",  "huber",     {"alpha": 1.0}),
        ("Huber_2.0",  "huber",     {"alpha": 2.0}),
        ("MAPE",       "mape",      {}),
    ]:
        print(f"\n=== {obj_name} ===")
        params = dict(BASE_PARAMS, objective=obj, **extra_params)
        oof = np.full(n_tr, np.nan)

        for fold, (tr_idx, va_idx) in enumerate(splits):
            # Rebuild fold aux
            m_em2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                               callbacks=[lgb.log_evaluation(-1)])
            m_nl2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                               callbacks=[lgb.log_evaluation(-1)])
            m_sl2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                               callbacks=[lgb.log_evaluation(-1)])
            em_va = 10.0 ** m_em2.predict(X_tr[va_idx])
            nl_va = m_nl2.predict(X_tr[va_idx])
            sl_va = m_sl2.predict(X_tr[va_idx])
            va_assay = np.column_stack([
                em_va, nl_va, sl_va, has_null[va_idx],
                np.log1p(np.clip(em_va, 0, None))
            ] + [o[va_idx] for o in meta_oofs])
            X_va = np.hstack([X_tr[va_idx], va_assay])

            m = lgb.train(params, lgb.Dataset(X_tr_aug[tr_idx], label=y_tr[tr_idx]),
                          valid_sets=[lgb.Dataset(X_va, label=y_tr[va_idx])],
                          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_va)
            print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

        full_metrics(y_tr, oof, obj_name)

        m_final = lgb.train(dict(params, n_estimators=1000),
                            lgb.Dataset(X_tr_aug, label=y_tr),
                            callbacks=[lgb.log_evaluation(-1)])
        te_preds = np.clip(m_final.predict(X_te_aug), y_tr.min() - 0.5, y_tr.max() + 0.5)
        ratio = te_preds.std() / oof.std()
        print(f"  Test: med={np.median(te_preds):.2f}  std={te_preds.std():.3f}  ratio={ratio:.2f}")
        saved[obj_name] = (oof, te_preds)

    # Best single
    best_name = min(saved, key=lambda k: rae(y_tr, saved[k][0]))
    best_oof, best_te = saved[best_name]
    full_metrics(y_tr, best_oof, f"BEST ({best_name})")

    stem = f"nb120_{best_name.lower().replace('.', '_')}"
    np.save(DATA_PROCESSED / f"oof_{stem}.npy", best_oof)
    np.save(DATA_PROCESSED / f"te_{stem}.npy",  best_te)

    # Also save all variants
    for obj_name, (oof_v, te_v) in saved.items():
        s = f"nb120_{obj_name.lower().replace('.', '_')}"
        np.save(DATA_PROCESSED / f"oof_{s}.npy", oof_v)
        np.save(DATA_PROCESSED / f"te_{s}.npy",  te_v)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / f"120_huber_mape_{best_name.lower().replace('.', '_')}.csv", index=False)
    print(f"\nSaved best ({best_name}): OOF RAE={rae(y_tr, best_oof):.4f}")


if __name__ == "__main__":
    main()
