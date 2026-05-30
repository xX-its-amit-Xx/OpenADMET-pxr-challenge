"""nb131 — Transductive Pseudo-Label Refinement.

Use best-model predictions on the 513 test compounds as pseudo-labels.
Train refined model on (train + pseudo-labeled test), using test compounds
as training examples with their predicted pEC50 values.

Strategy:
  - Round 0: Use nb127 blend predictions as initial pseudo-labels
  - Round 1: Train new model on (train + pseudo-test), compute OOF
  - Round 2: Use Round 1 test predictions as new pseudo-labels, repeat
  - Stop after 3 rounds (diminishing returns)

Risk mitigation:
  - Pseudo-labels weighted at 0.5 of training compound weight
  - OOF computed on challenge train only (honest estimate)
  - Track test_std/oof_std ratio — if test distribution collapses, abort
  - This exploits the test compound structures without leaking labels
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
N_ROUNDS = 3
PSEUDO_WEIGHT = 0.5  # down-weight test pseudo-labels relative to training

LGBM_BASE = dict(
    n_estimators=600, num_leaves=48, learning_rate=0.04,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1, n_jobs=4
)


def full_metrics(y_true, y_pred, label=""):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return rae_v


def train_round(X_tr, y_tr, splits, X_aug_tr, X_te_aug,
                pseudo_te, pseudo_weight, lgbm_params):
    """One round of transductive training. Returns (oof, te_preds)."""
    n_tr = len(y_tr)
    n_te = len(pseudo_te)
    X_pseudo = X_te_aug
    y_pseudo = pseudo_te

    # Sample weights: 1.0 for training, pseudo_weight for test
    w_tr  = np.ones(n_tr, dtype=np.float32)
    w_te  = np.full(n_te, pseudo_weight, dtype=np.float32)

    oof = np.full(n_tr, np.nan)
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        X_fold = np.vstack([X_aug_tr[tr_idx], X_pseudo])
        y_fold = np.concatenate([y_tr[tr_idx], y_pseudo])
        w_fold = np.concatenate([w_tr[tr_idx], w_te])
        ds = lgb.Dataset(X_fold, label=y_fold, weight=w_fold)
        m = lgb.train(lgbm_params, ds, callbacks=[lgb.log_evaluation(-1)])
        oof[va_idx] = m.predict(X_aug_tr[va_idx])
        print(f"    fold {fold_idx+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

    # Full model (train + pseudo)
    X_full = np.vstack([X_aug_tr, X_pseudo])
    y_full = np.concatenate([y_tr, y_pseudo])
    w_full = np.concatenate([w_tr, w_te])
    ds_full = lgb.Dataset(X_full, label=y_full, weight=w_full)
    m_full = lgb.train(lgbm_params, ds_full, callbacks=[lgb.log_evaluation(-1)])
    te_preds = m_full.predict(X_te_aug)
    return oof, te_preds


def main():
    print("=== nb131: Transductive Pseudo-Label Refinement ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw = raw_train[emax_col].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_imputed = np.where(np.isnan(pec50_null), np.nanmedian(pec50_null), pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)

    print("Computing structural features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

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

    # Augmented feature matrices
    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null,
                                  np.log1p(np.clip(oof_emax, 0, None))])
    assay_te  = np.column_stack([te_emax, te_null, te_sel,
                                  np.zeros(len(X_te)),
                                  np.log1p(np.clip(te_emax, 0, None))])
    X_aug_tr = np.hstack([X_tr, assay_oof])
    X_aug_te = np.hstack([X_te, assay_te])
    print(f"Augmented: train={X_aug_tr.shape}  test={X_aug_te.shape}")

    # Initial pseudo-labels: load best existing model (nb127 blend)
    nb127_te = DATA_PROCESSED / "te_nb127_exhaustive_blend.npy"
    if nb127_te.exists():
        pseudo_init = np.load(nb127_te).astype(np.float64)
        if pseudo_init.ndim == 2: pseudo_init = pseudo_init[:, 0]
        print(f"Initial pseudo-labels from nb127: med={np.median(pseudo_init):.2f}  "
              f"std={pseudo_init.std():.3f}")
    else:
        # Fall back to nb109
        nb109_te = DATA_PROCESSED / "te_nb109_deep_meta_stack.npy"
        pseudo_init = np.load(nb109_te).astype(np.float64)
        if pseudo_init.ndim == 2: pseudo_init = pseudo_init[:, 0]
        print(f"Initial pseudo-labels from nb109: med={np.median(pseudo_init):.2f}")

    pseudo_te = np.clip(pseudo_init, y_tr.min() - 0.5, y_tr.max() + 0.5)
    best_oof, best_te = None, None
    best_rae = 1.0

    for round_idx in range(N_ROUNDS):
        print(f"\n=== Round {round_idx + 1} / {N_ROUNDS} ===")
        print(f"  Pseudo-label stats: med={np.median(pseudo_te):.2f}  "
              f"std={pseudo_te.std():.3f}  "
              f"range=[{pseudo_te.min():.1f},{pseudo_te.max():.1f}]")

        oof, te_preds = train_round(
            X_tr, y_tr, splits, X_aug_tr, X_aug_te,
            pseudo_te, PSEUDO_WEIGHT, LGBM_BASE
        )
        r = full_metrics(y_tr, oof, f"Round {round_idx+1} pseudo (w={PSEUDO_WEIGHT})")
        ratio = te_preds.std() / oof.std()
        print(f"  Test: std={te_preds.std():.3f}  ratio={ratio:.2f}")

        if ratio < 0.50:
            print("  WARNING: test_std collapsed — stopping refinement")
            break

        if r < best_rae:
            best_rae = r; best_oof = oof.copy(); best_te = te_preds.copy()

        # Update pseudo-labels for next round
        pseudo_te = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)

    if best_oof is None:
        best_oof, best_te = oof, te_preds

    best_te = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb131_pseudo_label.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb131_pseudo_label.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "131_pseudo_label_refinement.csv", index=False)
    print(f"\nSaved: submissions/131_pseudo_label_refinement.csv")
    print(f"OOF RAE: {best_rae:.4f}")


if __name__ == "__main__":
    main()
