"""nb121 — Test-Similarity-Weighted LGBM.

Key insight: the test set is an analog expansion — there are 513 test
compounds that have *some* chemical similarity to training, but many are
in unexplored scaffold regions. If we weight training compounds by their
maximum Tanimoto similarity to any test compound, we bias the model toward
learning from training examples that are most relevant to the test set.

This is different from uncertainty sampling or difficulty weighting:
- We're not re-weighting by distance from the mean
- We're re-weighting by relevance to the actual inference target

Strategy:
  1. Compute ECFP4 Tanimoto similarities between all training and all test compounds
  2. For each training compound, compute max_sim = max Tanimoto to any test compound
  3. weight = max_sim^gamma (gamma > 0 → stronger emphasis on test-similar train compounds)
  4. Train LGBM with these sample weights
  5. Sweep gamma: 0.5, 1.0, 2.0, 3.0
  6. Report OOF RAE and test predictions

Note: training OOF uses fold-level sim computation (no leakage — the val
compounds' similarities to test are still computed using ALL test compounds,
which is fine since we don't know test labels).
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
from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0]

LGBM_PARAMS = dict(
    n_estimators=1500, num_leaves=64, learning_rate=0.03,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def tanimoto_maxsim(fps_q, fps_db, batch=256):
    """For each query, compute max Tanimoto to any DB compound."""
    fps_q = fps_q.astype(np.float32)
    fps_db = fps_db.astype(np.float32)
    db_norm = (fps_db ** 2).sum(axis=1)
    n_q = len(fps_q)
    max_sims = np.zeros(n_q, dtype=np.float32)
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        q_f = fps_q[start:end]
        q_norm = (q_f ** 2).sum(axis=1, keepdims=True)
        dot = q_f @ fps_db.T
        union = q_norm + db_norm[None, :] - dot
        sim = np.where(union > 0, dot / union, 0.0)
        max_sims[start:end] = sim.max(axis=1)
    return max_sims


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


def main():
    print("=== nb121: Test-Similarity-Weighted LGBM ===\n")

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

    print("Computing structural features + fingerprints...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    fps_tr = morgan_fp_batch(tr["smiles"].tolist())
    fps_te = morgan_fp_batch(te["smiles"].tolist())
    print(f"  Feature shape: train={X_tr.shape}  test={X_te.shape}")

    # Compute max-sim of each training compound to any test compound
    print("Computing training->test max Tanimoto similarity...")
    tr_max_sim = tanimoto_maxsim(fps_tr, fps_te, batch=256)
    print(f"  Training max-sim stats: min={tr_max_sim.min():.3f}  "
          f"mean={tr_max_sim.mean():.3f}  max={tr_max_sim.max():.3f}")
    print(f"  Percentiles: 25%={np.percentile(tr_max_sim,25):.3f}  "
          f"50%={np.percentile(tr_max_sim,50):.3f}  75%={np.percentile(tr_max_sim,75):.3f}")

    # Load meta-OOF
    meta_oofs, meta_tes, meta_names = [], [], []
    for stem in ["nb107_assay_decomp", "nb109_deep_meta_stack", "nb101_delta_base",
                 "nb99_sc_bio_fp", "grand_v6b", "lgbm_tuned"]:
        oof, te_m = load_meta(stem, n_tr)
        if oof is not None:
            oof_f = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te_f  = np.where(np.isfinite(te_m), te_m, np.nanmean(te_m))
            if te_f.std() / oof_f.std() >= 0.55:
                meta_oofs.append(oof_f); meta_tes.append(te_f)
                meta_names.append(stem)

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

    assay_oof_full = np.column_stack([
        oof_emax, oof_null, oof_sel, has_null,
        np.log1p(np.clip(oof_emax, 0, None))
    ] + meta_oofs)
    assay_te = np.column_stack([
        te_emax, te_null, te_sel, np.zeros(len(X_te)),
        np.log1p(np.clip(te_emax, 0, None))
    ] + meta_tes)
    X_tr_aug = np.hstack([X_tr, assay_oof_full])
    X_te_aug = np.hstack([X_te, assay_te])
    print(f"Augmented shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    # Gamma sweep
    print("\n=== Sweeping gamma (test-similarity weight exponent) ===")
    results = {}
    for gamma in GAMMAS:
        w = tr_max_sim ** gamma  # (N_tr,) weights
        w = w / w.mean()  # normalize so average weight = 1.0

        oof = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            # Fold-level aux rebuild (no data leakage on OOF)
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

            ds_tr = lgb.Dataset(X_tr_aug[tr_idx], label=y_tr[tr_idx],
                                weight=w[tr_idx])
            m = lgb.train(LGBM_PARAMS, ds_tr,
                          valid_sets=[lgb.Dataset(X_va, label=y_tr[va_idx])],
                          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_va)

        r = rae(y_tr, oof)
        print(f"  gamma={gamma:.1f}  OOF RAE={r:.4f}", flush=True)
        results[gamma] = (oof, r)

    # Best gamma
    best_gamma = min(results, key=lambda g: results[g][1])
    best_oof, best_r = results[best_gamma]
    print(f"\nBest: gamma={best_gamma:.1f}  OOF RAE={best_r:.4f}")

    # Train final model with best gamma on full data
    best_w = tr_max_sim ** best_gamma
    best_w = best_w / best_w.mean()
    m_final = lgb.train(dict(LGBM_PARAMS, n_estimators=1200),
                        lgb.Dataset(X_tr_aug, label=y_tr, weight=best_w),
                        callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_final.predict(X_te_aug), y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio = te_preds.std() / best_oof.std()
    print(f"Test: med={np.median(te_preds):.2f}  std={te_preds.std():.3f}  "
          f"max={te_preds.max():.2f}  ratio={ratio:.2f}")

    np.save(DATA_PROCESSED / "oof_nb121_test_sim_weighted.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb121_test_sim_weighted.npy",  te_preds)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    sub.to_csv(SUBMISSIONS / "121_test_sim_weighted.csv", index=False)
    print(f"\nSaved: submissions/121_test_sim_weighted.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
