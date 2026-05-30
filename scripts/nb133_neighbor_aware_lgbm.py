"""nb133 — Neighbor-Aware LGBM.

Augment the feature vector with nearest-neighbor statistics from the
training set. For each compound, compute:
  - Top-k Tanimoto similarities to training set
  - Weighted mean/std of k-NN pEC50 values
  - Fraction of k-NN with pEC50 > 5.5 (hit rate among neighbors)
  - Mean similarity-weighted pEC50 (k-NN interpolation)

These neighbor features encode the local activity landscape and are
highly complementary to global LGBM features. They help the model
extrapolate in regions with high training similarity.

For test compounds: compute neighbor stats using all training data.
For OOF validation: compute neighbor stats from the training fold only
(to avoid leakage from the validation fold).
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
K_NEIGHBORS = 10
LGBM_AUX = dict(
    n_estimators=500, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_MAIN = dict(
    n_estimators=600, num_leaves=48, learning_rate=0.04,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def tanimoto_topk(fp_query, fp_ref, k=10):
    """Compute top-k Tanimoto similarities. Returns (n_query, k) sims and indices."""
    # fp_query: (nq, d)  fp_ref: (nr, d)  values in [0,1] (Morgan)
    fp_q = fp_query.astype(np.float32)
    fp_r = fp_ref.astype(np.float32)
    # Tanimoto = dot / (sum_q + sum_r - dot)
    dot = fp_q @ fp_r.T  # (nq, nr)
    sum_q = fp_q.sum(axis=1, keepdims=True)  # (nq, 1)
    sum_r = fp_r.sum(axis=1, keepdims=True).T  # (1, nr)
    denom = sum_q + sum_r - dot
    denom = np.where(denom == 0, 1e-6, denom)
    sim = dot / denom  # (nq, nr)
    # Get top-k
    k = min(k, sim.shape[1])
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    sims_topk = np.take_along_axis(sim, idx, axis=1)
    # Sort within top-k
    order = np.argsort(-sims_topk, axis=1)
    idx_sorted = np.take_along_axis(idx, order, axis=1)
    sims_sorted = np.take_along_axis(sims_topk, order, axis=1)
    return sims_sorted, idx_sorted


def compute_neighbor_features(fp_q, y_ref, fp_ref, k=K_NEIGHBORS):
    """Compute k-NN neighbor features for query set."""
    sims, idx = tanimoto_topk(fp_q, fp_ref, k)
    y_nbr = y_ref[idx]  # (nq, k)
    # Similarity-weighted pEC50
    w = sims / (sims.sum(axis=1, keepdims=True) + 1e-9)
    knn_pred = (w * y_nbr).sum(axis=1)
    knn_mean = y_nbr.mean(axis=1)
    knn_std  = y_nbr.std(axis=1)
    knn_max  = y_nbr.max(axis=1)
    top1_sim = sims[:, 0]
    mean_sim = sims.mean(axis=1)
    hit_rate = (y_nbr > 5.5).mean(axis=1)
    return np.column_stack([knn_pred, knn_mean, knn_std, knn_max,
                             top1_sim, mean_sim, hit_rate])


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
    print("=== nb133: Neighbor-Aware LGBM ===\n")

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

    print("Computing structural features + Morgan FP...")
    X_str = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))
    fp_tr = morgan_fp_batch(tr["smiles"].tolist()).astype(np.float32)
    fp_te = morgan_fp_batch(te["smiles"].tolist()).astype(np.float32)
    print(f"  Structural: {X_str.shape}  FP: {fp_tr.shape}")

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

    # Full neighbor features for test (using all training data)
    print(f"\nComputing test neighbor features (k={K_NEIGHBORS}) from all training...")
    nbr_te = compute_neighbor_features(fp_te, y_tr, fp_tr)
    print(f"  Test kNN: knn_pred med={np.median(nbr_te[:,0]):.2f}  "
          f"top1_sim med={np.median(nbr_te[:,4]):.3f}")

    # Build augmented matrices for test (known)
    assay_te = np.column_stack([te_emax, te_null, te_sel,
                                 np.zeros(len(X_str_te)),
                                 np.log1p(np.clip(te_emax, 0, None))])
    X_te_aug = np.hstack([X_str_te, assay_te, nbr_te])

    # === CV with fold-aware neighbor features ===
    print("\n=== CV: Neighbor-Aware LGBM ===")
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Neighbor features for validation fold using only training fold
        nbr_va = compute_neighbor_features(fp_tr[va_idx], y_tr[tr_idx], fp_tr[tr_idx])

        assay_oof_tr = np.column_stack([oof_emax[tr_idx], oof_null[tr_idx], oof_sel[tr_idx],
                                         has_null[tr_idx],
                                         np.log1p(np.clip(oof_emax[tr_idx], 0, None))])
        assay_oof_va = np.column_stack([oof_emax[va_idx], oof_null[va_idx], oof_sel[va_idx],
                                         has_null[va_idx],
                                         np.log1p(np.clip(oof_emax[va_idx], 0, None))])

        # Training fold: neighbor features from within training fold
        nbr_tr = compute_neighbor_features(fp_tr[tr_idx], y_tr[tr_idx], fp_tr[tr_idx])

        X_tr_fold = np.hstack([X_str[tr_idx], assay_oof_tr, nbr_tr])
        X_va_fold = np.hstack([X_str[va_idx], assay_oof_va, nbr_va])

        m = lgb.train(LGBM_MAIN, lgb.Dataset(X_tr_fold, label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_va_fold)
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

    full_metrics(y_tr, oof, f"Neighbor-Aware LGBM (k={K_NEIGHBORS})")

    # Full model: neighbor features from full training set
    nbr_tr_full = compute_neighbor_features(fp_tr, y_tr, fp_tr)
    assay_oof_full = np.column_stack([oof_emax, oof_null, oof_sel, has_null,
                                       np.log1p(np.clip(oof_emax, 0, None))])
    X_tr_full = np.hstack([X_str, assay_oof_full, nbr_tr_full])
    m_full = lgb.train(LGBM_MAIN, lgb.Dataset(X_tr_full, label=y_tr),
                       callbacks=[lgb.log_evaluation(-1)])
    te_preds = m_full.predict(X_te_aug)
    te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio = te_preds.std() / oof.std()
    print(f"  Test: med={np.median(te_preds):.2f}  std={te_preds.std():.3f}  ratio={ratio:.2f}")

    np.save(DATA_PROCESSED / "oof_nb133_neighbor_aware.npy", oof)
    np.save(DATA_PROCESSED / "te_nb133_neighbor_aware.npy",  te_preds)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    sub.to_csv(SUBMISSIONS / "133_neighbor_aware_lgbm.csv", index=False)
    print(f"\nSaved: submissions/133_neighbor_aware_lgbm.csv")
    print(f"OOF RAE: {rae(y_tr, oof):.4f}")


if __name__ == "__main__":
    main()
