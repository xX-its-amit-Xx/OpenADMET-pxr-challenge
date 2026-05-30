"""nb140 save_best — Reproduces only LGBM_meta config_1 (OOF 0.3300) and saves.

nb140 crashed on Unicode char in print. All training completed; this script
reproduces only the best config (LGBM_meta config_1) to save OOF/test arrays.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
COLLAPSE_THRESH = 0.58
LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_BEST = dict(
    n_estimators=700, num_leaves=48, learning_rate=0.04, min_child_samples=8,
    subsample=0.75, colsample_bytree=0.5, reg_alpha=0.2, verbose=-1, n_jobs=4, random_state=SEED
)


def load_model_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists(): break
        else:
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
            r = rae(y_tr, oof)
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results


def main():
    print("=== nb140 save_best: LGBM_meta config_1 ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

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
    X_str = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))

    print("Loading OOF predictions...")
    models = load_model_oofs(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded")
    oof_all = np.column_stack([m["oof"] for m in models])
    te_all  = np.column_stack([m["te"]  for m in models])

    print("Aux OOF...")
    oof_emax = np.full(n_tr, np.nan); oof_null = np.full(n_tr, np.nan); oof_sel = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_str[va_idx])
        m_nl = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=null_imputed[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_str[va_idx])
        m_sl = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=selectivity[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx] = m_sl.predict(X_str[va_idx])

    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_str_te)
    te_null = m_nl_f.predict(X_str_te)
    te_sel  = m_sl_f.predict(X_str_te)

    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null,
                                  np.log1p(np.clip(oof_emax, 0, None))])
    assay_te  = np.column_stack([te_emax, te_null, te_sel,
                                  np.zeros(len(X_str_te)),
                                  np.log1p(np.clip(te_emax, 0, None))])

    X_meta_tr = np.hstack([oof_all, X_str, assay_oof])
    X_meta_te = np.hstack([te_all,  X_str_te, assay_te])
    print(f"Meta features: {X_meta_tr.shape}")

    print("Training LGBM_meta config_1...")
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM_BEST, lgb.Dataset(X_meta_tr[tr_idx], label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_meta_tr[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)
    m_full = lgb.train(LGBM_BEST, lgb.Dataset(X_meta_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_pred = m_full.predict(X_meta_te)
    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    print(f"OOF RAE={r:.4f}  ratio={ratio:.2f}")

    te_out = np.clip(te_pred, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb140_xgb_lgbm_meta.npy", oof)
    np.save(DATA_PROCESSED / "te_nb140_xgb_lgbm_meta.npy",  te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_out})
    sub.to_csv(SUBMISSIONS / "140_xgb_lgbm_meta.csv", index=False)
    print(f"Saved: submissions/140_xgb_lgbm_meta.csv  OOF RAE={r:.4f}")


if __name__ == "__main__":
    main()
