"""nb118 — Random Forest on Combined Features.

Random Forest provides a structurally different predictor from LGBM:
- Different splitting criterion (all features at each node vs row-sampling)
- Averaged over 300 full trees (vs boosted shallow trees)
- Tends to have better calibrated uncertainty in extrapolation regions

This serves as an ensemble member with genuinely different error patterns.
Also provides implicit feature importance via MDI (mean decrease impurity).
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5

RF_PARAMS = dict(
    n_estimators=300,
    max_features=0.33,
    min_samples_leaf=3,
    n_jobs=4,
    random_state=SEED,
)
ET_PARAMS = dict(
    n_estimators=300,
    max_features=0.33,
    min_samples_leaf=3,
    n_jobs=4,
    random_state=SEED,
)


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:55s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def main():
    print("=== nb118: Random Forest + ExtraTrees ===\n")

    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Computing features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"  shape: train={X_tr.shape}  test={X_te.shape}")

    results = {}
    for name, ModelClass, params in [
        ("RandomForest", RandomForestRegressor, RF_PARAMS),
        ("ExtraTrees",   ExtraTreesRegressor,   ET_PARAMS),
    ]:
        print(f"\n=== {name} ===")
        oof = np.full(n_tr, np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = ModelClass(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof[va_idx] = m.predict(X_tr[va_idx])
            fold_rae = rae(y_tr[va_idx], oof[va_idx])
            print(f"  fold {fold+1}  RAE={fold_rae:.4f}", flush=True)

        m_full = ModelClass(**params)
        m_full.fit(X_tr, y_tr)
        te_preds = m_full.predict(X_te)

        te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)
        oof_ratio = te_preds.std() / oof.std()
        full_metrics(y_tr, oof, name)
        print(f"  Test: med={np.median(te_preds):.2f}  std={te_preds.std():.3f}  "
              f"max={te_preds.max():.2f}  ratio={oof_ratio:.2f}")

        stem = "nb118_rf" if name == "RandomForest" else "nb118_et"
        np.save(DATA_PROCESSED / f"oof_{stem}.npy", oof)
        np.save(DATA_PROCESSED / f"te_{stem}.npy", te_preds)
        results[name] = (oof, te_preds)

    # 50/50 blend of RF + ET
    oof_blend = (results["RandomForest"][0] + results["ExtraTrees"][0]) / 2
    te_blend  = (results["RandomForest"][1] + results["ExtraTrees"][1]) / 2
    full_metrics(y_tr, oof_blend, "RF+ET blend")
    te_blend  = np.clip(te_blend, y_tr.min() - 0.5, y_tr.max() + 0.5)

    np.save(DATA_PROCESSED / "oof_nb118_rf_et_blend.npy", oof_blend)
    np.save(DATA_PROCESSED / "te_nb118_rf_et_blend.npy", te_blend)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_blend})
    sub.to_csv(SUBMISSIONS / "118_rf_et_blend.csv", index=False)
    print(f"\nSaved: submissions/118_rf_et_blend.csv")
    print(f"OOF RAE blend: {rae(y_tr, oof_blend):.4f}")


if __name__ == "__main__":
    main()
