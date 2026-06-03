"""nb961 -- ExtraTreesRegressor on combined Morgan + RDKit features.

Distinct from prior RF/LGBM/XGB/CatBoost ensembles: ExtraTrees randomizes the
split THRESHOLDS (not just feature subsets), producing a lower-variance
ensemble with a different inductive bias. Expectation: predictions
slightly smoother than RF, with a different OOD failure mode -- a candidate
for residual-orthogonality to the existing tree-based PRIMARY anchors.

Pipeline:
  1) load 4139 CRC train + 513 blinded test.
  2) combined Morgan + RDKit features (2265-dim) with median imputation.
  3) ExtraTreesRegressor(n_estimators=1000, max_depth=20, min_samples_leaf=3,
     max_features='sqrt', n_jobs=-1, random_state=42).
  4) predict on 513; compute in_RAE on 253 unblind subset.
  5) save te_nb961.npy + submissions/nb961_extra_trees.csv.

Artifacts: C:/pxr_artifacts/nb961/
Wall-time budget: < 5 min.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

ART = Path("C:/pxr_artifacts/nb961")
ART.mkdir(parents=True, exist_ok=True)

SEED = 42
N_EST = 1000
MAX_DEPTH = 20
MIN_LEAF = 3
WALL_BUDGET_S = 300  # 5 min


def main():
    t0 = time.time()
    print("=== nb961: ExtraTreesRegressor (1000 / depth=20 / min_leaf=3) ===")

    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    print(f"train={len(tr)}  test={len(te)}")

    print("Featurizing (combined Morgan + RDKit)...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}  "
          f"mem~{X_tr.nbytes/1e6:.1f}MB")

    print(f"Fitting ExtraTrees (n_est={N_EST}, depth={MAX_DEPTH}, "
          f"min_leaf={MIN_LEAF}, max_features='sqrt')...")
    t_fit = time.time()
    et = ExtraTreesRegressor(
        n_estimators=N_EST,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_LEAF,
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED,
    )
    et.fit(X_tr, y_tr)
    print(f"  fit done in {time.time()-t_fit:.1f}s")

    if time.time() - t0 > WALL_BUDGET_S * 0.8:
        print("WARN: fit consumed >80% of wall budget")

    pred = et.predict(X_te).astype(np.float64)
    print(f"  pred mean={pred.mean():.3f}  std={pred.std():.3f}  "
          f"min={pred.min():.3f}  max={pred.max():.3f}")

    # Save artifacts
    np.save(ART / "te_nb961.npy", pred)
    np.save(DATA_PROCESSED / "te_nb961.npy", pred)

    # In-RAE on 253 unblind
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te["name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    )
    unb_y = unb.loc[unb["Molecule Name"].isin(name_to_idx), "pEC50"].values

    in_rae_et = rae(unb_y, pred[unb_te_idx])
    print(f"\nUnblind n={len(unb_te_idx)}")
    print(f"  in_RAE ExtraTrees = {in_rae_et:.4f}")

    # Submission CSV (SMILES + Molecule Name + pEC50)
    sub = pd.DataFrame({
        "Molecule Name": te["name"].values,
        "SMILES": te["smiles"].values,
        "pEC50": pred,
    })
    out_csv = SUBMISSIONS / "nb961_extra_trees.csv"
    sub.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    metrics = {
        "in_rae_extra_trees": float(in_rae_et),
        "n_unblind": int(len(unb_te_idx)),
        "n_estimators": N_EST,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_LEAF,
        "wall_time_s": float(time.time() - t0),
    }
    pd.Series(metrics).to_json(ART / "metrics.json")
    print(f"Done in {time.time()-t0:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
