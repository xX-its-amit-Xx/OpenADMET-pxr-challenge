"""nb972 - Long-training-with-slow-LR optimization hypothesis test.

Hypothesis: a slow learning rate (0.005), very long max-iteration budget
(10000 trees), wide early-stopping patience (500) and modest leaf size
(64) find a *flatter* loss minimum that generalizes better than the fast/
short LGBM recipes used so far on this dataset.

Recipe:
  - LightGBM Huber (alpha=2.0)
  - n_estimators=10000, learning_rate=0.005, num_leaves=64
  - min_child_samples=20, reg_lambda=0.2
  - early_stopping_rounds=500
  - Features: Morgan (2048) + RDKit desc (217)  -> combined (2265)
  - Scaffold 5-fold CV on 4139 CRC; 1 held-out fold serves as the
    early-stopping validation set within each split.
  - Predict 513 test; report in_RAE on 253 Phase-1 unblind index.

Outputs:
  data/processed/oof_nb972_long_train.npy
  data/processed/te_nb972_long_train.npy
  data/processed/nb972_summary.json
  submissions/nb972_long_train_optim.csv
"""
import os, sys, json, time, warnings
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

PARAMS = dict(
    objective="huber",
    alpha=2.0,
    n_estimators=10000,
    learning_rate=0.005,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=4,
)
EARLY_STOP = 500


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def main():
    t0 = time.time()
    print("=== nb972: long-train slow-LR Huber LGBM ===\n")

    # ---- Truth / unblind for in_RAE ----
    unblind_idx = np.load(DATA_PROCESSED / "nb472_unblind_idx.npy")
    y_unblind = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    assert len(unblind_idx) == len(y_unblind) == 253, "unblind shapes off"

    # ---- Data ----
    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Computing combined features (Morgan + RDKit)...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}")

    # ---- Scaffold CV with long-train + early stop ----
    oof = np.full(n_tr, np.nan)
    best_iters = []
    fold_raes = []
    print(f"\nTraining {N_FOLDS} folds (LR={PARAMS['learning_rate']}, "
          f"max_iter={PARAMS['n_estimators']}, patience={EARLY_STOP})...")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(
            PARAMS,
            lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof[va_idx] = m.predict(X_tr[va_idx], num_iteration=m.best_iteration)
        fr = rae(y_tr[va_idx], oof[va_idx])
        fold_raes.append(fr)
        best_iters.append(int(m.best_iteration or PARAMS["n_estimators"]))
        elapsed = time.time() - t0
        print(f"  fold {fold+1}  best_iter={best_iters[-1]:5d}  "
              f"RAE={fr:.4f}  elapsed={elapsed:6.1f}s", flush=True)

    oof_rae = rae(y_tr, oof)
    mean_best = int(np.mean(best_iters))
    print(f"\nOOF RAE = {oof_rae:.4f}")
    print(f"mean best_iteration across folds = {mean_best}")

    # ---- Final fit on full train, capped at mean best_iter ----
    print(f"\nFinal fit on full 4139 train, n_estimators={mean_best}...")
    final_params = dict(PARAMS, n_estimators=mean_best)
    m_final = lgb.train(
        final_params,
        lgb.Dataset(X_tr, label=y_tr),
        callbacks=[lgb.log_evaluation(-1)],
    )
    te_preds = np.clip(
        m_final.predict(X_te),
        y_tr.min() - 0.5, y_tr.max() + 0.5,
    )
    ratio = te_preds.std() / oof.std() if oof.std() > 0 else 0.0
    in_r = in_rae(y_unblind, te_preds[unblind_idx])
    print(f"TEST  med={np.median(te_preds):.2f}  std={te_preds.std():.3f}  "
          f"ratio(te/oof)={ratio:.2f}")
    print(f"in_RAE(253) = {in_r:.4f}")

    # ---- Persist ----
    np.save(DATA_PROCESSED / "oof_nb972_long_train.npy", oof)
    np.save(DATA_PROCESSED / "te_nb972_long_train.npy", te_preds)
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_preds,
    }).to_csv(SUBMISSIONS / "nb972_long_train_optim.csv", index=False)

    summary = {
        "params": {k: v for k, v in PARAMS.items() if k != "verbose"},
        "early_stopping_rounds": EARLY_STOP,
        "fold_best_iters": best_iters,
        "fold_raes": [float(x) for x in fold_raes],
        "mean_best_iter": mean_best,
        "oof_rae": float(oof_rae),
        "in_rae_253": float(in_r),
        "test_std": float(te_preds.std()),
        "te_oof_std_ratio": float(ratio),
        "wall_time_sec": float(time.time() - t0),
    }
    with open(DATA_PROCESSED / "nb972_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {DATA_PROCESSED/'nb972_summary.json'}")
    print(f"wall_time = {summary['wall_time_sec']:.1f}s")


if __name__ == "__main__":
    main()
