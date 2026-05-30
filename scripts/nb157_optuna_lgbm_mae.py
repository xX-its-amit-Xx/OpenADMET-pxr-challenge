"""nb157 — Optuna HPO for LGBM MAE on Clean Base OOF.

nb149 found LGBM_MAE with manual config: OOF 0.3069.
nb154 found A (base-only) OOF ~0.30xx (pending).

Goal: use Optuna to systematically search LGBM_MAE hyperparameter space
on clean base OOF (excluding meta-stack stems).

Hyperparameters to tune:
  - n_estimators: [400, 1500]
  - num_leaves: [31, 255]
  - learning_rate: [0.01, 0.10]
  - min_child_samples: [3, 30]
  - subsample: [0.5, 1.0]
  - colsample_bytree: [0.5, 1.0]
  - reg_alpha: [0.0, 0.5]
  - reg_lambda: [0.0, 1.0]

n_trials=50, pruning enabled (MedianPruner).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58
N_TRIALS = 60

META_STEMS = {
    "nb136_xgb_meta", "nb138_elnet_blend", "nb139_adaptive_blend",
    "nb140_xgb_lgbm_meta", "nb141_xgb_ablation", "nb142_xgb_calibrated",
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb146_pca_oof_meta", "nb147_oofrdkit_meta", "nb148_meta_disagreement",
    "nb149_meta_maeloss", "nb150_residual_ensemble", "nb151_grand_v11",
    "nb152_lgbm_mae_tuned", "nb153_grand_v12", "nb154_lgbm_mae_filtered",
    "nb155_grand_v13", "nb156_catboost_mae",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
}

LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
)


def load_base_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH, exclude_stems=None):
    exclude_stems = exclude_stems or set()
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in exclude_stems:
            continue
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


def build_assay_features(tr, te, splits, y_tr, n_tr):
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

    X_str = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))

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
    return assay_oof, assay_te


def main():
    print("=== nb157: Optuna HPO for LGBM_MAE on Clean Base OOF ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    print("Loading base model OOFs (excluding meta-stacks)...")
    mods = load_base_oofs(n_tr, y_tr, exclude_stems=META_STEMS)
    print(f"  {len(mods)} base models loaded")
    oof_mat = np.column_stack([m["oof"] for m in mods])
    te_mat  = np.column_stack([m["te"]  for m in mods])
    X_tr_full = np.hstack([oof_mat, assay_oof])
    X_te_full  = np.hstack([te_mat, assay_te])
    print(f"  Meta features: {X_tr_full.shape}")

    def objective(trial):
        cfg = dict(
            n_estimators=trial.suggest_int("n_estimators", 400, 1600),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 3, 30),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 0.5),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 1.0),
            objective="regression_l1",
            verbose=-1, n_jobs=4, random_state=42
        )
        n_tr_local = len(y_tr)
        oof = np.full(n_tr_local, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = lgb.train(cfg, lgb.Dataset(X_tr_full[tr_idx], label=y_tr[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr_full[va_idx])
            # Pruning: report intermediate value
            inter = rae(y_tr[va_idx], oof[va_idx])
            trial.report(inter, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return rae(y_tr, oof)

    print(f"\nRunning Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    )
    # Seed with the nb149 known-good config
    study.enqueue_trial({
        "n_estimators": 800, "num_leaves": 63, "learning_rate": 0.03,
        "min_child_samples": 5, "subsample": 0.8, "colsample_bytree": 1.0,
        "reg_alpha": 0.05, "reg_lambda": 0.0
    })
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nBest trial: RAE={study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Re-train best config fully and get test predictions
    best_cfg = {**study.best_params, "objective": "regression_l1",
                "verbose": -1, "n_jobs": 4, "random_state": 42}
    oof_best = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(best_cfg, lgb.Dataset(X_tr_full[tr_idx], label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof_best[va_idx] = m.predict(X_tr_full[va_idx])
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], oof_best[va_idx]):.4f}", flush=True)

    m_full = lgb.train(best_cfg, lgb.Dataset(X_tr_full, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_best = m_full.predict(X_te_full)
    final_rae = rae(y_tr, oof_best)
    ratio = te_best.std() / oof_best.std()
    print(f"\nFinal refit  RAE={final_rae:.4f}  ratio={ratio:.2f}  n_feats={X_tr_full.shape[1]}")

    # Multi-seed ensemble with best config (5 seeds)
    print("\nMulti-seed ensemble (5 seeds, best config)...")
    seed_oofs = [oof_best]; seed_tes = [te_best]
    for seed in [123, 456, 789, 1234]:
        cfg_s = {**best_cfg, "random_state": seed}
        oof_s = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = lgb.train(cfg_s, lgb.Dataset(X_tr_full[tr_idx], label=y_tr[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
            oof_s[va_idx] = m.predict(X_tr_full[va_idx])
        m_f = lgb.train(cfg_s, lgb.Dataset(X_tr_full, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
        seed_oofs.append(oof_s); seed_tes.append(m_f.predict(X_te_full))
        print(f"  seed {seed}: RAE={rae(y_tr, oof_s):.4f}", flush=True)

    oof_ens = np.mean(seed_oofs, axis=0)
    te_ens  = np.mean(seed_tes, axis=0)
    r_ens = rae(y_tr, oof_ens)
    ratio_ens = te_ens.std() / oof_ens.std()
    print(f"  Multi-seed ensemble: RAE={r_ens:.4f}  ratio={ratio_ens:.2f}")

    # Choose best (single vs ensemble)
    if ratio >= COLLAPSE_THRESH and (r_ens < final_rae or ratio_ens < COLLAPSE_THRESH):
        best_oof_out = oof_best; best_te_out_raw = te_best; save_rae = final_rae
        print("Using single best config")
    elif ratio_ens >= COLLAPSE_THRESH and r_ens < final_rae:
        best_oof_out = oof_ens; best_te_out_raw = te_ens; save_rae = r_ens
        print("Using multi-seed ensemble")
    else:
        # Fallback: whichever passes threshold
        if ratio >= COLLAPSE_THRESH:
            best_oof_out = oof_best; best_te_out_raw = te_best; save_rae = final_rae
        else:
            best_oof_out = oof_ens; best_te_out_raw = te_ens; save_rae = r_ens

    best_te_clipped = np.clip(best_te_out_raw, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb157_optuna_lgbm_mae.npy", best_oof_out)
    np.save(DATA_PROCESSED / "te_nb157_optuna_lgbm_mae.npy",  best_te_clipped)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_clipped})
    sub.to_csv(SUBMISSIONS / "157_optuna_lgbm_mae.csv", index=False)
    print(f"\nSaved: submissions/157_optuna_lgbm_mae.csv  OOF RAE={save_rae:.4f}")
    print(f"(nb149 manual config: 0.3069; this target: < 0.3069)")


if __name__ == "__main__":
    main()
