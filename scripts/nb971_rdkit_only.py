"""nb971 — RDKit-descriptor-only Huber LGBM ablation.

Hypothesis: Are Morgan FPs net-positive on this dataset, or do they add noise
that descriptor-only LGBM can match?

Recipe:
  * Features: ONLY RDKit ~217-d descriptors (NO Morgan, NO meta-OOFs, NO assay)
  * LGBM Huber alpha=2.0, n_est=1500, num_leaves=64, lr=0.03
  * Scaffold 5-fold CV on 4139 CRC
  * Predict 513 test; compute in_RAE on 253 unblind

Saves to C:/pxr_artifacts/nb971/:
  - oof_nb971_rdkit_only.npy        (4139,)
  - te_nb971_rdkit_only.npy         (513,)
  - nb971_summary.json
Also drops submissions/nb971_rdkit_only.csv for parity.
"""
import os, sys, json, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import rdkit_desc, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
ALPHA = 2.0

ART_DIR = Path("C:/pxr_artifacts/nb971")
ART_DIR.mkdir(parents=True, exist_ok=True)

BASE_PARAMS = dict(
    objective="huber", alpha=ALPHA,
    n_estimators=1500, num_leaves=64, learning_rate=0.03,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1,
    random_state=SEED, verbose=-1, n_jobs=4,
)


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def main():
    t0 = time.time()
    print("=== nb971: RDKit-only Huber LGBM (alpha=2.0) ablation ===\n")

    # ---- Load truth + unblind index ----
    unblind_idx = np.load(DATA_PROCESSED / "nb472_unblind_idx.npy")
    y_unblind   = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    assert len(unblind_idx) == len(y_unblind) == 253

    # ---- Data ----
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ---- RDKit-ONLY features ----
    print("Computing RDKit-only descriptors (NO Morgan)...")
    X_tr = impute(rdkit_desc(tr["smiles"].tolist()))
    X_te = impute(rdkit_desc(te["smiles"].tolist()))
    print(f"  train shape: {X_tr.shape}  test shape: {X_te.shape}\n")

    # ---- Scaffold 5-fold CV ----
    oof = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(
            BASE_PARAMS,
            lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof[va_idx] = m.predict(X_tr[va_idx])
        print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof[va_idx]):.4f}", flush=True)

    oof_rae = rae(y_tr, oof)
    print(f"\n  OOF RAE = {oof_rae:.4f}")

    # ---- Final fit on full train; predict 513 test ----
    final_params = dict(BASE_PARAMS, n_estimators=1000)
    m_final = lgb.train(final_params, lgb.Dataset(X_tr, label=y_tr),
                        callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_final.predict(X_te),
                       y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio = te_preds.std() / oof.std() if oof.std() > 0 else 0.0
    in_r = in_rae(y_unblind, te_preds[unblind_idx])
    print(f"  TEST med={np.median(te_preds):.2f} std={te_preds.std():.3f} "
          f"ratio={ratio:.2f}  in_RAE(253)={in_r:.4f}")

    # ---- Persist artifacts ----
    np.save(ART_DIR / "oof_nb971_rdkit_only.npy", oof)
    np.save(ART_DIR / "te_nb971_rdkit_only.npy",  te_preds)

    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_preds,
    })
    sub.to_csv(SUBMISSIONS / "nb971_rdkit_only.csv", index=False)

    # ---- Compare against Morgan+RDKit baseline (nb120_huber_2_0) ----
    ref_path = DATA_PROCESSED / "te_nb120_huber_2_0.npy"
    ref_in_r = None
    if ref_path.exists():
        ref_te = np.load(ref_path)
        ref_in_r = in_rae(y_unblind, ref_te[unblind_idx])
        delta = in_r - ref_in_r
        print(f"\nReference (Morgan+RDKit+meta, nb120_huber_2.0): in_RAE={ref_in_r:.4f}")
        print(f"  Delta (RDKit-only - full): {delta:+.4f}  "
              f"({'RDKit-only WORSE' if delta>0 else 'RDKit-only better'})")

    wall = time.time() - t0
    summary = {
        "variant": "rdkit_only_huber_2_0",
        "n_features": int(X_tr.shape[1]),
        "oof_rae": float(oof_rae),
        "in_rae_253": float(in_r),
        "ref_nb120_huber_2_0_in_rae": float(ref_in_r) if ref_in_r is not None else None,
        "delta_vs_full": float(in_r - ref_in_r) if ref_in_r is not None else None,
        "wall_seconds": wall,
        "test_std": float(te_preds.std()),
        "oof_std": float(oof.std()),
        "std_ratio": float(ratio),
    }
    with open(ART_DIR / "nb971_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {ART_DIR/'nb971_summary.json'}")
    print(f"Wall time: {wall:.1f}s")


if __name__ == "__main__":
    main()
