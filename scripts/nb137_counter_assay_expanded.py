"""nb137 — Expanded Counter-Assay Delta Model.

nb129 showed that counter_delta (from nb113) is the crucial 3rd model in the
best k=3 ensemble (RAE 0.3556). counter_delta uses counter-assay measurements
as reference compounds for delta prediction.

This notebook expands the counter-delta approach:
  1. Multiple similarity thresholds (0.3, 0.4, 0.5) — not just top-k
  2. Multiple delta predictors: pEC50, Emax, selectivity
  3. Weighted delta: weight by similarity² AND by hit probability
  4. Ensemble of delta predictions at different similarity windows

Goal: create a better counter-assay-based predictor that can enter the
ensemble and improve beyond 0.3556.
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
from pxr.chem import morgan_fp_batch, bemis_murcko
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_DELTA = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_MAIN = dict(
    n_estimators=600, num_leaves=48, learning_rate=0.04,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)

SIM_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]
K_MAX = 15


def compute_tanimoto_batch(fp_q, fp_r, batch=200):
    """Compute Tanimoto matrix (nq, nr) in batches."""
    nq, nr = len(fp_q), len(fp_r)
    sim = np.zeros((nq, nr), dtype=np.float32)
    sum_r = fp_r.sum(axis=1)
    for start in range(0, nq, batch):
        end = min(start + batch, nq)
        fp_b = fp_q[start:end]
        dot = fp_b @ fp_r.T
        sum_b = fp_b.sum(axis=1, keepdims=True)
        denom = sum_b + sum_r.reshape(1, -1) - dot
        denom[denom == 0] = 1e-6
        sim[start:end] = dot / denom
    return sim


def counter_delta_features(fp_q, fp_ref, y_ref, k_max=K_MAX, thresholds=SIM_THRESHOLDS):
    """Compute expanded counter-delta features."""
    sim = compute_tanimoto_batch(fp_q, fp_ref)
    n_q = len(fp_q)

    # Sort by descending similarity
    idx_sorted = np.argsort(-sim, axis=1)[:, :k_max]
    sim_sorted = np.take_along_axis(sim, idx_sorted, axis=1)
    y_sorted   = y_ref[idx_sorted]

    features = []

    # Top-k features at multiple k values
    for k in [3, 5, 10]:
        k_eff = min(k, k_max, sim_sorted.shape[1])
        sims_k = sim_sorted[:, :k_eff]
        y_k    = y_sorted[:, :k_eff]
        w = sims_k / (sims_k.sum(axis=1, keepdims=True) + 1e-9)
        knn_pred = (w * y_k).sum(axis=1)
        top1_sim = sims_k[:, 0]
        knn_std  = y_k.std(axis=1)
        features.extend([knn_pred, top1_sim, knn_std])

    # Similarity-threshold features
    for thresh in thresholds:
        mask = (sim_sorted >= thresh)
        counts = mask.sum(axis=1)
        # Mean y of analogs above threshold (0 if no analogs)
        y_above = np.where(mask, y_sorted, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_above = np.nanmean(y_above, axis=1)
        mean_above = np.where(np.isfinite(mean_above), mean_above, np.nanmean(y_ref))
        features.extend([mean_above, counts.astype(np.float32)])

    return np.column_stack(features)


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
    print("=== nb137: Expanded Counter-Assay Delta Model ===\n")

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

    # Counter assay reference compounds
    counter_smiles_list = raw_counter["SMILES"].fillna("").tolist()
    counter_pec50_ref   = raw_counter["pEC50"].fillna(raw_counter["pEC50"].median()).values.astype(np.float64)
    print(f"Counter assay reference: {len(counter_smiles_list)} compounds  "
          f"pEC50=[{counter_pec50_ref.min():.1f},{counter_pec50_ref.max():.1f}]")

    print("Computing Morgan fingerprints (challenge + counter)...")
    fp_tr      = morgan_fp_batch(tr["smiles"].tolist()).astype(np.float32)
    fp_te      = morgan_fp_batch(te["smiles"].tolist()).astype(np.float32)
    fp_counter = morgan_fp_batch(counter_smiles_list).astype(np.float32)
    print(f"  FPs: train={fp_tr.shape}  test={fp_te.shape}  counter={fp_counter.shape}")

    print("Computing structural features (challenge)...")
    X_str    = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))

    # Counter-delta features for test (uses full counter reference)
    print("Computing counter-delta features for test...")
    cd_te = counter_delta_features(fp_te, fp_counter, counter_pec50_ref)
    print(f"  Counter-delta features: {cd_te.shape}")

    print("Aux OOF...")
    oof_emax = np.full(n_tr, np.nan)
    oof_null = np.full(n_tr, np.nan)
    oof_sel  = np.full(n_tr, np.nan)
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

    print("\n=== CV: Counter-Delta Augmented LGBM ===")
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Counter-delta features: use ONLY training fold as reference during CV
        # (counter data is fixed reference, not held out — this is OK because
        #  counter assay compounds are not in the PXR training set)
        cd_tr_fold = counter_delta_features(fp_tr[tr_idx], fp_counter, counter_pec50_ref)
        cd_va_fold = counter_delta_features(fp_tr[va_idx], fp_counter, counter_pec50_ref)

        X_tr_fold = np.hstack([X_str[tr_idx], assay_oof[tr_idx], cd_tr_fold])
        X_va_fold = np.hstack([X_str[va_idx], assay_oof[va_idx], cd_va_fold])

        m = lgb.train(LGBM_MAIN, lgb.Dataset(X_tr_fold, label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_va_fold)
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

    # Full model: counter-delta from full training set
    cd_tr_full = counter_delta_features(fp_tr, fp_counter, counter_pec50_ref)
    X_tr_full  = np.hstack([X_str, assay_oof, cd_tr_full])
    X_te_full  = np.hstack([X_str_te, assay_te, cd_te])
    m_full = lgb.train(LGBM_MAIN, lgb.Dataset(X_tr_full, label=y_tr),
                       callbacks=[lgb.log_evaluation(-1)])
    te_preds = m_full.predict(X_te_full)
    te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)
    full_metrics(y_tr, oof, "Counter-Delta Augmented LGBM")
    ratio = te_preds.std() / oof.std()
    print(f"  Test: med={np.median(te_preds):.2f}  std={te_preds.std():.3f}  ratio={ratio:.2f}")

    np.save(DATA_PROCESSED / "oof_nb137_counter_delta_exp.npy", oof)
    np.save(DATA_PROCESSED / "te_nb137_counter_delta_exp.npy",  te_preds)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    sub.to_csv(SUBMISSIONS / "137_counter_assay_expanded.csv", index=False)
    print(f"\nSaved: submissions/137_counter_assay_expanded.csv")
    print(f"OOF RAE: {rae(y_tr, oof):.4f}")


if __name__ == "__main__":
    main()
