"""nb128 — Augmented Random Forest + ExtraTrees.

nb118 ran RF/ET on structural features only (2265 dims) and got OOF ~0.57.
Key insight: nb107/nb109 achieved major improvement (0.38) by adding
predicted Emax/null/selectivity as meta-features.

This version adds the same assay augmentation to RF/ET:
  - Structural features (2265 dims)
  - Predicted Emax (from LGBM auxiliary)
  - Predicted null pEC50 (from LGBM auxiliary)
  - Predicted selectivity (from LGBM auxiliary)
  - has_null indicator
  - Top-4 meta-OOF predictions (nb107, nb109, nb111, nb99)

Expected: RF/ET with augmentation should reach 0.40-0.45 RAE instead of 0.57.
The diverse prediction style (averaging over 300 trees) provides different
error patterns from LGBM — good ensemble diversity.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
RF_PARAMS = dict(n_estimators=300, max_features=0.33, min_samples_leaf=3,
                 n_jobs=4, random_state=SEED)
ET_PARAMS = dict(n_estimators=300, max_features=0.33, min_samples_leaf=3,
                 n_jobs=4, random_state=SEED)


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


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def main():
    print("=== nb128: Augmented Random Forest + ExtraTrees ===\n")

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names = raw_train["Molecule Name"].values
    pec50_null = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_imputed = np.where(np.isnan(pec50_null), np.nanmedian(pec50_null), pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    print("Computing structural features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    meta_candidates = ["nb107_assay_decomp", "nb109_deep_meta_stack",
                       "nb111_selectivity_primary", "nb99_sc_bio_fp"]
    meta_oofs, meta_tes = [], []
    for stem in meta_candidates:
        oof, te_m = load_meta(stem, n_tr)
        if oof is not None:
            oof_f = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te_f  = np.where(np.isfinite(te_m), te_m, np.nanmean(te_m))
            if te_f.std() / oof_f.std() >= 0.55:
                meta_oofs.append(oof_f); meta_tes.append(te_f)
                print(f"  {stem}: RAE={rae(y_tr, oof_f):.4f}")

    print("\nAux OOF...")
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

    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null,
                                  np.log1p(np.clip(oof_emax, 0, None))] + meta_oofs)
    assay_te  = np.column_stack([te_emax, te_null, te_sel, np.zeros(len(X_te)),
                                  np.log1p(np.clip(te_emax, 0, None))] + meta_tes)
    X_tr_aug = np.hstack([X_tr, assay_oof])
    X_te_aug = np.hstack([X_te, assay_te])
    print(f"Augmented: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    results = {}
    for name, ModelClass, params in [
        ("RF_aug",  RandomForestRegressor, RF_PARAMS),
        ("ET_aug",  ExtraTreesRegressor,   ET_PARAMS),
    ]:
        print(f"\n=== {name} ===")
        oof = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m_em2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]),
                               callbacks=[lgb.log_evaluation(-1)])
            m_nl2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]),
                               callbacks=[lgb.log_evaluation(-1)])
            m_sl2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]),
                               callbacks=[lgb.log_evaluation(-1)])
            em_va = 10.0 ** m_em2.predict(X_tr[va_idx])
            nl_va = m_nl2.predict(X_tr[va_idx])
            sl_va = m_sl2.predict(X_tr[va_idx])
            va_assay = np.column_stack([em_va, nl_va, sl_va, has_null[va_idx],
                                         np.log1p(np.clip(em_va, 0, None))] + [o[va_idx] for o in meta_oofs])
            X_va = np.hstack([X_tr[va_idx], va_assay])

            m = ModelClass(**params)
            m.fit(X_tr_aug[tr_idx], y_tr[tr_idx])
            oof[va_idx] = m.predict(X_va)
            print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

        m_full = ModelClass(**params)
        m_full.fit(X_tr_aug, y_tr)
        te_preds = np.clip(m_full.predict(X_te_aug), y_tr.min() - 0.5, y_tr.max() + 0.5)
        ratio = te_preds.std() / oof.std()
        full_metrics(y_tr, oof, name)
        print(f"  Test: med={np.median(te_preds):.2f}  std={te_preds.std():.3f}  ratio={ratio:.2f}")
        results[name] = (oof, te_preds)
        np.save(DATA_PROCESSED / f"oof_nb128_{name.lower()}.npy", oof)
        np.save(DATA_PROCESSED / f"te_nb128_{name.lower()}.npy",  te_preds)

    # 50/50 blend
    oof_blend = (results["RF_aug"][0] + results["ET_aug"][0]) / 2
    te_blend  = (results["RF_aug"][1] + results["ET_aug"][1]) / 2
    te_blend  = np.clip(te_blend, y_tr.min() - 0.5, y_tr.max() + 0.5)
    full_metrics(y_tr, oof_blend, "RF_aug+ET_aug blend")
    np.save(DATA_PROCESSED / "oof_nb128_rf_et_aug.npy", oof_blend)
    np.save(DATA_PROCESSED / "te_nb128_rf_et_aug.npy",  te_blend)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_blend})
    sub.to_csv(SUBMISSIONS / "128_rf_et_augmented.csv", index=False)
    print(f"\nSaved: submissions/128_rf_et_augmented.csv  OOF={rae(y_tr, oof_blend):.4f}")


if __name__ == "__main__":
    main()
