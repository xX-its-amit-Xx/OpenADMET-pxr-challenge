"""Fix for nb109: re-run only the final model + save OOF + submission.
The OOF from scaffold CV is already in memory from the failed run.
This script re-runs nb109 from scratch but with the fixed early_stopping."""
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

SEED = 42; N_FOLDS = 5
LGBM_PARAMS = dict(
    n_estimators=1200, num_leaves=64, learning_rate=0.03,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)

def load_meta(stem, n_tr):
    for op in ("oof_",""):
        for tp in ("te_","te_oof_"):
            of = DATA_PROCESSED / f"{op}{stem}.npy"
            tf = DATA_PROCESSED / f"{tp}{stem}.npy"
            if of.exists() and tf.exists():
                oof = np.load(of); te = np.load(tf)
                if oof.ndim==2: oof=oof[:,0]
                if te.ndim==2: te=te[:,0]
                if len(oof)==n_tr: return oof, te
    return None, None

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
null_median = np.nanmedian(pec50_null)
null_imputed = np.where(np.isnan(pec50_null), null_median, pec50_null)
selectivity = y_tr - null_imputed
has_null = (~np.isnan(pec50_null)).astype(np.float32)

print("Computing features...")
X_tr = impute(combined(tr["smiles"].tolist()))
X_te = impute(combined(te["smiles"].tolist()))

meta_candidates = [
    "nb107_assay_decomp", "nb101_delta_base", "nb103_seed_propagation",
    "nb99_sc_bio_fp", "grand_v6b", "grand25", "lgbm_tuned", "catboost",
    "multi_nr_transfer", "xgboost_dart", "chemprop_aux", "nb97_pxr_features", "grover_large"
]
oof_metas, te_metas, meta_names = [], [], []
for stem in meta_candidates:
    oof, te_p = load_meta(stem, n_tr)
    if oof is not None:
        oof_f = np.where(np.isfinite(oof), oof, np.nanmean(oof))
        te_f = np.where(np.isfinite(te_p), te_p, np.nanmean(te_p))
        ratio = te_f.std() / oof_f.std() if oof_f.std() > 0 else 0
        if ratio >= 0.50:
            oof_metas.append(oof_f); te_metas.append(te_f); meta_names.append(stem)
            print(f"  {stem}: RAE={rae(y_tr,oof_f):.4f}  ratio={ratio:.2f}")

M_tr = np.column_stack(oof_metas); M_te = np.column_stack(te_metas)

# Stage 1 OOF aux
print("\nStage 1: auxiliary OOF...")
oof_emax = np.full(n_tr, np.nan); oof_null = np.full(n_tr, np.nan); oof_sel = np.full(n_tr, np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    for lbl, X_lbl, tgt, dst in [
        ("emax", X_tr[tr_idx], emax_log[tr_idx], oof_emax),
        ("null", X_tr[tr_idx], null_imputed[tr_idx], oof_null),
        ("sel",  X_tr[tr_idx], selectivity[tr_idx], oof_sel)
    ]:
        m = lgb.train(LGBM_AUX, lgb.Dataset(X_lbl, label=tgt), callbacks=[lgb.log_evaluation(-1)])
        if lbl == "emax":   oof_emax[va_idx] = 10.0 ** m.predict(X_tr[va_idx])
        elif lbl == "null": oof_null[va_idx] = m.predict(X_tr[va_idx])
        else:               oof_sel[va_idx]  = m.predict(X_tr[va_idx])

# Full data aux for test
m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
te_emax = 10.0 ** m_em_f.predict(X_te)
te_null = m_nl_f.predict(X_te)
te_sel  = m_sl_f.predict(X_te)

assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null, np.log1p(np.clip(oof_emax,0,None))])
assay_te  = np.column_stack([te_emax, te_null, te_sel, np.zeros(len(X_te)), np.log1p(np.clip(te_emax,0,None))])

X_tr_full = np.hstack([X_tr, assay_oof, M_tr])
X_te_full  = np.hstack([X_te, assay_te, M_te])
print(f"Augmented shape: train={X_tr_full.shape}  test={X_te_full.shape}")

# Scaffold CV
print("\nScaffold CV...")
oof_deep = np.full(n_tr, np.nan)
for fold, (tr_idx, va_idx) in enumerate(splits):
    m_em2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=emax_log[tr_idx]), callbacks=[lgb.log_evaluation(-1)])
    m_nl2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=null_imputed[tr_idx]), callbacks=[lgb.log_evaluation(-1)])
    m_sl2 = lgb.train(LGBM_AUX, lgb.Dataset(X_tr[tr_idx], label=selectivity[tr_idx]), callbacks=[lgb.log_evaluation(-1)])
    em_va = 10.0 ** m_em2.predict(X_tr[va_idx])
    nl_va = m_nl2.predict(X_tr[va_idx])
    sl_va = m_sl2.predict(X_tr[va_idx])
    assay_va = np.column_stack([em_va, nl_va, sl_va, has_null[va_idx], np.log1p(np.clip(em_va,0,None))])
    X_va = np.hstack([X_tr[va_idx], assay_va, M_tr[va_idx]])
    m = lgb.train(LGBM_PARAMS, lgb.Dataset(X_tr_full[tr_idx], label=y_tr[tr_idx]),
                  valid_sets=[lgb.Dataset(X_va, label=y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_deep[va_idx] = m.predict(X_va)
    print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof_deep[va_idx]):.4f}", flush=True)

r_oof = rae(y_tr, oof_deep)
print(f"\nOOF RAE = {r_oof:.4f}")

# Final model (fixed: no early stopping)
print("Training final model...")
m_final = lgb.train(LGBM_PARAMS, lgb.Dataset(X_tr_full, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
te_preds = m_final.predict(X_te_full)
te_preds = np.clip(te_preds, y_tr.min()-0.5, y_tr.max()+0.5)
print(f"Test: med={np.median(te_preds):.2f}  std={te_preds.std():.3f}  max={te_preds.max():.2f}")

# Feature importances
fi = m_final.feature_importance(importance_type="gain")
n_struct = X_tr.shape[1] + 5
meta_fi = fi[n_struct:]
print("\nTop meta-feature importances:")
for nm, imp in sorted(zip(meta_names, meta_fi), key=lambda x: -x[1])[:6]:
    print(f"  {nm:45s}  gain={imp:.1f}")

np.save(DATA_PROCESSED / "oof_nb109_deep_meta_stack.npy", oof_deep)
np.save(DATA_PROCESSED / "te_nb109_deep_meta_stack.npy", te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
sub.to_csv(SUBMISSIONS / "109_deep_meta_stack.csv", index=False)
print(f"\nSaved: 109_deep_meta_stack.csv  OOF={r_oof:.4f}")
