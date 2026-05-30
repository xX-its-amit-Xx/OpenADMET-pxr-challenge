"""nb126 — Classifier-Conditioned Regression.

The main failure mode of all current models: extreme-pEC50 compounds
(very inactive < 2.5, very active > 6.0) are systematically predicted
as near-mean (~4.3). This causes the largest individual errors.

Strategy: build a 3-class activity classifier:
  - Class 0: inactive (pEC50 <= 2.5)  — 415 training compounds (10%)
  - Class 1: moderate (2.5 < pEC50 <= 5.5)  — 3,558 compounds (86%)
  - Class 2: active (pEC50 > 5.5)  — 166 compounds (4%)

Then train three separate regressors:
  - Regressor_inactive: predict pEC50 given class=inactive
  - Regressor_moderate: predict pEC50 given class=moderate
  - Regressor_active: predict pEC50 given class=active

Final prediction = sum_c P(class=c) * regressor_c(x)

This forces the model to learn that some compounds are truly extreme,
and the active/inactive regressors can focus on their sub-population.
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

SEED = 42
N_FOLDS = 5
INACTIVE_THRESH = 2.5
ACTIVE_THRESH   = 5.5

LGBM_CLF = dict(
    objective="multiclass", num_class=3,
    n_estimators=800, num_leaves=32, learning_rate=0.04,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.2, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_REG = dict(
    n_estimators=1000, num_leaves=32, learning_rate=0.04,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.2, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_MAIN = dict(
    n_estimators=1500, num_leaves=64, learning_rate=0.03,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def pec50_to_class(y):
    """0=inactive, 1=moderate, 2=active."""
    c = np.ones(len(y), dtype=int)
    c[y <= INACTIVE_THRESH] = 0
    c[y >  ACTIVE_THRESH]   = 2
    return c


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
    print("=== nb126: Classifier-Conditioned Regression ===\n")

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    classes = pec50_to_class(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Class distribution:")
    for c, name in enumerate(["inactive (<=2.5)", "moderate", "active (>5.5)"]):
        print(f"  {name}: {(classes == c).sum()} ({100*(classes == c).mean():.1f}%)")

    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names = raw_train["Molecule Name"].values
    pec50_null = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_imputed = np.where(np.isnan(pec50_null), np.nanmedian(pec50_null), pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    print("\nComputing features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    meta_candidates = ["nb109_deep_meta_stack", "nb107_assay_decomp",
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

    # Stage 1: aux OOF
    print("\nAuxiliary OOF...")
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

    # Scaffold CV: classifier + conditioned regressors
    print("\n=== Scaffold CV: classifier-conditioned regression ===")
    oof_cond = np.full(n_tr, np.nan)
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
        va_assay = np.column_stack([em_va, nl_va, sl_va, has_null[va_idx],
                                     np.log1p(np.clip(em_va, 0, None))] + [o[va_idx] for o in meta_oofs])
        X_va = np.hstack([X_tr[va_idx], va_assay])
        X_tr_fold = X_tr_aug[tr_idx]
        y_tr_fold = y_tr[tr_idx]
        cls_fold  = classes[tr_idx]

        # Multiclass classifier
        m_clf = lgb.train(LGBM_CLF,
                          lgb.Dataset(X_tr_fold, label=cls_fold),
                          valid_sets=[lgb.Dataset(X_va, label=classes[va_idx])],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        proba = m_clf.predict(X_va)  # (N_va, 3)

        # Per-class regressors
        class_preds = np.full((len(va_idx), 3), np.nan)
        for c in range(3):
            c_mask = cls_fold == c
            if c_mask.sum() < 5:
                class_preds[:, c] = y_tr_fold.mean()
                continue
            m_reg = lgb.train(LGBM_REG,
                              lgb.Dataset(X_tr_fold[c_mask], label=y_tr_fold[c_mask]),
                              valid_sets=[lgb.Dataset(X_va, label=y_tr[va_idx])],
                              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            class_preds[:, c] = m_reg.predict(X_va)

        oof_cond[va_idx] = (proba * class_preds).sum(axis=1)
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof_cond[va_idx]):.4f}", flush=True)

    full_metrics(y_tr, oof_cond, "Classifier-conditioned")

    # Also train a simple augmented LGBM baseline for comparison
    print("\n=== Baseline: augmented LGBM ===")
    oof_base = np.full(n_tr, np.nan)
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
        m = lgb.train(LGBM_MAIN, lgb.Dataset(X_tr_aug[tr_idx], label=y_tr[tr_idx]),
                      valid_sets=[lgb.Dataset(X_va, label=y_tr[va_idx])],
                      callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_base[va_idx] = m.predict(X_va)
    full_metrics(y_tr, oof_base, "Baseline LGBM (same features)")

    # Blend conditioned + baseline
    print("\nBlend sweep (conditioned + baseline):")
    best_alpha, best_r = 0.0, rae(y_tr, oof_base)
    for alpha in np.arange(0.0, 1.01, 0.1):
        blend = alpha * oof_cond + (1 - alpha) * oof_base
        r = rae(y_tr, blend)
        if r < best_r: best_r, best_alpha = r, alpha
        if alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            print(f"  alpha(cond)={alpha:.1f}  RAE={r:.4f}")
    print(f"  Best: alpha={best_alpha:.1f}  RAE={best_r:.4f}")

    # Final predictions
    best_oof = best_alpha * oof_cond + (1 - best_alpha) * oof_base

    # Train final models on full data
    m_clf_f = lgb.train(dict(LGBM_CLF, n_estimators=600),
                         lgb.Dataset(X_tr_aug, label=classes),
                         callbacks=[lgb.log_evaluation(-1)])
    proba_te = m_clf_f.predict(X_te_aug)  # (N_te, 3)

    class_preds_te = np.full((len(te["smiles"]), 3), np.nan)
    for c in range(3):
        c_mask = classes == c
        if c_mask.sum() < 5:
            class_preds_te[:, c] = y_tr.mean(); continue
        m_reg_f = lgb.train(dict(LGBM_REG, n_estimators=800),
                             lgb.Dataset(X_tr_aug[c_mask], label=y_tr[c_mask]),
                             callbacks=[lgb.log_evaluation(-1)])
        class_preds_te[:, c] = m_reg_f.predict(X_te_aug)
    te_cond = (proba_te * class_preds_te).sum(axis=1)

    m_base_f = lgb.train(dict(LGBM_MAIN, n_estimators=1200),
                          lgb.Dataset(X_tr_aug, label=y_tr),
                          callbacks=[lgb.log_evaluation(-1)])
    te_base = m_base_f.predict(X_te_aug)

    te_final = np.clip(best_alpha * te_cond + (1 - best_alpha) * te_base,
                       y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"\nTest: med={np.median(te_final):.2f}  std={te_final.std():.3f}  "
          f"max={te_final.max():.2f}  ratio={te_final.std()/best_oof.std():.2f}")

    np.save(DATA_PROCESSED / "oof_nb126_classifier_conditioned.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb126_classifier_conditioned.npy",  te_final)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_final})
    sub.to_csv(SUBMISSIONS / "126_classifier_conditioned.csv", index=False)
    print(f"\nSaved: submissions/126_classifier_conditioned.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
