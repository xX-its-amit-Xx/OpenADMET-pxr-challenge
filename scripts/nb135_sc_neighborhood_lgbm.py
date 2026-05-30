"""nb135 — Single-Concentration Neighborhood LGBM.

The 21k single-concentration (SC) screen provides log2FC measurements for
2,763 unique compounds. These provide a "PXR activity landscape" signal.

Strategy:
  1. For each challenge train/test compound, compute Tanimoto similarity to
     all SC compounds.
  2. Compute neighborhood features: weighted log2FC, weighted t-statistic,
     max SC similarity, fraction of SC hits (log2FC > 1) among k-NN.
  3. Augment the standard assay-decomposition LGBM with these SC features.

Key: for compounds also present in the SC data (by SMILES), use their
direct SC measurements. Otherwise, use k-NN imputation from SC.

This provides orthogonal signal to the CRC-based models (nb107-nb109).
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
K_SC = 10  # SC neighbors
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


def tanimoto_row(fp_q, fp_ref):
    """Compute Tanimoto similarity of one query fp against all ref fps."""
    dot = fp_q @ fp_ref.T
    sum_q = fp_q.sum()
    sum_r = fp_ref.sum(axis=1)
    denom = sum_q + sum_r - dot
    denom = np.where(denom == 0, 1e-6, denom)
    return dot / denom


def compute_sc_features(smiles_query, smiles_sc, sc_fc, sc_tstat, k=K_SC):
    """Compute SC neighborhood features for query compounds."""
    fp_q  = morgan_fp_batch(smiles_query).astype(np.float32)
    fp_sc = morgan_fp_batch(smiles_sc).astype(np.float32)
    n_q, n_sc = len(fp_q), len(fp_sc)

    # Process in batches of 500 to manage memory
    batch = 500
    feats = np.zeros((n_q, 7), dtype=np.float32)
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        fp_batch = fp_q[start:end]
        # Tanimoto sim: (batch, n_sc)
        dot = fp_batch @ fp_sc.T
        sum_q = fp_batch.sum(axis=1, keepdims=True)
        sum_r = fp_sc.sum(axis=1, keepdims=True).T
        denom = np.where(sum_q + sum_r - dot == 0, 1e-6, sum_q + sum_r - dot)
        sim = dot / denom
        # Top-k
        k_eff = min(k, n_sc)
        idx = np.argpartition(-sim, k_eff, axis=1)[:, :k_eff]
        sims_k = np.take_along_axis(sim, idx, axis=1)
        fc_k   = sc_fc[idx]
        ts_k   = sc_tstat[idx]
        # Features
        w = sims_k / (sims_k.sum(axis=1, keepdims=True) + 1e-9)
        wfc  = (w * fc_k).sum(axis=1)
        wts  = (w * ts_k).sum(axis=1)
        top1_sim = sims_k.max(axis=1)
        mean_sim = sims_k.mean(axis=1)
        hit_rate = (fc_k > 1.0).mean(axis=1)
        max_fc   = fc_k.max(axis=1)
        mean_fc  = fc_k.mean(axis=1)
        feats[start:end] = np.column_stack([wfc, wts, top1_sim, mean_sim,
                                             hit_rate, max_fc, mean_fc])
    return feats


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
    print("=== nb135: Single-Concentration Neighborhood LGBM ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    raw_sc      = pd.read_csv("data/raw/pxr-challenge_single_concentration_TRAIN.csv")
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw    = raw_train[emax_col].values.astype(np.float64)
    emax_log    = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_imputed = np.where(np.isnan(pec50_null), np.nanmedian(pec50_null), pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    # Process SC data: aggregate by compound (take highest-concentration measurement)
    print("Processing SC data...")
    # Use highest concentration per compound
    raw_sc_sorted = raw_sc.sort_values("concentration_M", ascending=False)
    sc_agg = raw_sc_sorted.groupby("Molecule Name").first().reset_index()
    sc_smiles = sc_agg["SMILES"].fillna("").tolist()
    sc_fc   = sc_agg["log2_fc_estimate"].fillna(0).values.astype(np.float32)
    sc_ts   = sc_agg["t_statistic"].fillna(0).values.astype(np.float32)
    print(f"  SC unique compounds: {len(sc_smiles)}")
    print(f"  SC hits (fc>1): {(sc_fc > 1).sum()} / {len(sc_fc)}")

    print("Computing structural features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    print("Computing SC neighborhood features for train/test...")
    sc_feats_tr = compute_sc_features(tr["smiles"].tolist(), sc_smiles, sc_fc, sc_ts)
    sc_feats_te = compute_sc_features(te["smiles"].tolist(), sc_smiles, sc_fc, sc_ts)
    print(f"  SC feats: train={sc_feats_tr.shape}  test={sc_feats_te.shape}")
    print(f"  Train: wfc med={np.median(sc_feats_tr[:,0]):.3f}  "
          f"top1_sim med={np.median(sc_feats_tr[:,2]):.3f}")

    print("Aux OOF (assay decomposition)...")
    oof_emax = np.full(n_tr, np.nan)
    oof_null = np.full(n_tr, np.nan)
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
        oof_sel[va_idx] = m_sl.predict(X_tr[va_idx])

    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_tr, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_te)
    te_null = m_nl_f.predict(X_te)
    te_sel  = m_sl_f.predict(X_te)

    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null,
                                  np.log1p(np.clip(oof_emax, 0, None))])
    assay_te  = np.column_stack([te_emax, te_null, te_sel,
                                  np.zeros(len(X_te)),
                                  np.log1p(np.clip(te_emax, 0, None))])

    # Augmented matrices
    X_aug_tr = np.hstack([X_tr, assay_oof, sc_feats_tr])
    X_aug_te = np.hstack([X_te, assay_te, sc_feats_te])
    print(f"Augmented: train={X_aug_tr.shape}  test={X_aug_te.shape}")

    print("\n=== CV: SC-Augmented LGBM ===")
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(LGBM_MAIN, lgb.Dataset(X_aug_tr[tr_idx], label=y_tr[tr_idx]),
                      callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_aug_tr[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

    m_full = lgb.train(LGBM_MAIN, lgb.Dataset(X_aug_tr, label=y_tr),
                       callbacks=[lgb.log_evaluation(-1)])
    te_preds = m_full.predict(X_aug_te)
    te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)
    full_metrics(y_tr, oof, "SC-Augmented LGBM")
    ratio = te_preds.std() / oof.std()
    print(f"  Test: med={np.median(te_preds):.2f}  std={te_preds.std():.3f}  ratio={ratio:.2f}")

    # Baseline without SC features
    print("\n=== Baseline (no SC features) ===")
    X_base_tr = np.hstack([X_tr, assay_oof])
    X_base_te = np.hstack([X_te, assay_te])
    oof_base = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        mb = lgb.train(LGBM_MAIN, lgb.Dataset(X_base_tr[tr_idx], label=y_tr[tr_idx]),
                       callbacks=[lgb.log_evaluation(-1)])
        oof_base[va_idx] = mb.predict(X_base_tr[va_idx])
    full_metrics(y_tr, oof_base, "Baseline (no SC)")
    mb_full = lgb.train(LGBM_MAIN, lgb.Dataset(X_base_tr, label=y_tr), callbacks=[lgb.log_evaluation(-1)])
    te_base = mb_full.predict(X_base_te)

    np.save(DATA_PROCESSED / "oof_nb135_sc_neighborhood.npy", oof)
    np.save(DATA_PROCESSED / "te_nb135_sc_neighborhood.npy",  te_preds)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    sub.to_csv(SUBMISSIONS / "135_sc_neighborhood_lgbm.csv", index=False)
    print(f"\nSaved: submissions/135_sc_neighborhood_lgbm.csv")
    print(f"OOF RAE: {rae(y_tr, oof):.4f}")


if __name__ == "__main__":
    main()
