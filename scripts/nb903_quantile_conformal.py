"""nb903 -- Three-quantile LGBM + adaptive conformal correction.

Distinct from prior pinball work (nb710 / nb191 / nb267):
  (a) Uses THREE quantiles (q10, q50, q90), not two.
  (b) Adaptive conformal correction is a separate calibration step
      computed from held-out residuals on one fold.
  (c) Final test prediction = q50 + interval-width-based shrinkage,
      where wider (q90-q10) intervals get pulled toward the global mean
      (higher epistemic uncertainty -> stronger shrinkage).

Pipeline:
  1) Featurize 4139 train + 513 test with combined Morgan + RDKit.
  2) Scaffold 5-fold CV.  Fold 0 = conformal calibration set; folds 1-4
     train all 3 quantile LGBMs (objective='quantile', alpha in {.1,.5,.9}).
  3) For each quantile, OOF predictions over all 5 folds are computed.
  4) On fold 0 calibration residuals (truth - q50_fold0), fit a small
     1-D map:   correction(w) = E[truth - q50 | w-bucket]
     where w = q90 - q10 (interval width).  We use 5 width buckets.
  5) At test:  pred = q50_te + correction(w_te)  with the correction
     additionally shrunk by (1 - w_te / w_max) toward zero.
  6) Score on 253 unblind; save submission + artifacts.

Outputs:
  submissions/nb903_quantile_conformal.csv
  C:/pxr_artifacts/nb903_*.npy
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
N_BUCKETS = 5
TAG = "nb903"
ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)

LGBM_PARAMS = dict(
    n_estimators=500,
    num_leaves=64,
    learning_rate=0.05,
    min_child_samples=20,
    reg_lambda=1.0,
    n_jobs=4,
    random_state=SEED,
    verbose=-1,
)


def _std_smi(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _train_quantile(X_tr, y_tr, X_te, folds, alpha: float, tag: str):
    oof = np.zeros_like(y_tr, dtype=np.float64)
    te_folds = []
    params = dict(LGBM_PARAMS, objective="quantile", alpha=alpha)
    for k, (ti, vi) in enumerate(folds):
        md = lgb.LGBMRegressor(**params)
        md.fit(X_tr[ti], y_tr[ti])
        oof[vi] = md.predict(X_tr[vi])
        te_folds.append(md.predict(X_te))
        print(f"  [{tag}] fold {k}: n_tr={len(ti)} n_va={len(vi)} "
              f"fold RAE={rae(y_tr[vi], oof[vi]):.4f}")
    return oof, np.mean(te_folds, axis=0)


def _bucket_correction(width_cal, resid_cal, n_buckets=N_BUCKETS):
    """Empirical conformal-style correction: E[resid | width-bucket]."""
    edges = np.quantile(width_cal, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bucket_means = np.zeros(n_buckets, dtype=np.float64)
    for b in range(n_buckets):
        m = (width_cal > edges[b]) & (width_cal <= edges[b + 1])
        bucket_means[b] = resid_cal[m].mean() if m.any() else 0.0
    return edges, bucket_means


def _apply_correction(width, edges, bucket_means):
    n_b = len(bucket_means)
    out = np.zeros_like(width, dtype=np.float64)
    for b in range(n_b):
        m = (width > edges[b]) & (width <= edges[b + 1])
        out[m] = bucket_means[b]
    out[width <= edges[0]] = bucket_means[0]
    out[width > edges[-1]] = bucket_means[-1]
    return out


def main() -> dict:
    print("=" * 78)
    print("nb903 -- 3-quantile LGBM + adaptive conformal correction")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    miss = [k for k, p in needed.items() if not Path(p).exists()]
    if miss:
        print("MISSING:", miss)
        return {"success": False, "missing": miss}

    tr = load_train()
    tr = add_standard_columns(tr)
    te_df = pd.read_csv(needed["TEST_BLINDED"])

    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(_std_smi).tolist()
    print(f"Train n={len(y_tr)}  Test n={len(smiles_te)}")

    print("Featurizing combined (Morgan+RDKit)...")
    X_tr = impute(combined(smiles_tr)).astype(np.float32)
    X_te = impute(combined(smiles_te)).astype(np.float32)
    print(f"X_tr={X_tr.shape}  X_te={X_te.shape}")

    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=N_FOLDS)
    cal_ti, cal_vi = folds[0]  # fold 0 = calibration set

    print("\nTraining q10...")
    oof_q10, te_q10 = _train_quantile(X_tr, y_tr, X_te, folds, 0.1, "q10")
    print("Training q50...")
    oof_q50, te_q50 = _train_quantile(X_tr, y_tr, X_te, folds, 0.5, "q50")
    print("Training q90...")
    oof_q90, te_q90 = _train_quantile(X_tr, y_tr, X_te, folds, 0.9, "q90")

    print(f"\n[OOF] RAE q10={rae(y_tr, oof_q10):.4f}  "
          f"q50={rae(y_tr, oof_q50):.4f}  q90={rae(y_tr, oof_q90):.4f}")

    # ---- Conformal calibration on fold 0 ----
    width_cal = oof_q90[cal_vi] - oof_q10[cal_vi]
    resid_cal = y_tr[cal_vi] - oof_q50[cal_vi]
    edges, bucket_means = _bucket_correction(width_cal, resid_cal)
    print(f"\nCalibration n={len(cal_vi)}  width range "
          f"[{width_cal.min():.3f}, {width_cal.max():.3f}]")
    print("Bucket-mean residuals (truth - q50) by width quantile:")
    for b, bm in enumerate(bucket_means):
        print(f"  bucket {b}  (w in ({edges[b]:.3f}, {edges[b+1]:.3f}])  "
              f"correction = {bm:+.4f}")

    # ---- Apply adaptive correction at test ----
    width_te = te_q90 - te_q10
    corr_te = _apply_correction(width_te, edges, bucket_means)
    w_max = max(width_te.max(), width_cal.max(), 1e-6)
    shrink_te = 1.0 - (width_te / w_max)  # wider -> closer to 0
    shrink_te = np.clip(shrink_te, 0.0, 1.0)
    te_pred = te_q50 + corr_te * shrink_te

    # ---- Diagnostic: also score OOF with same scheme on folds 1-4 ----
    oof_pred = oof_q50.copy()
    other = np.concatenate([folds[k][1] for k in range(1, N_FOLDS)])
    width_oof = oof_q90[other] - oof_q10[other]
    corr_oof = _apply_correction(width_oof, edges, bucket_means)
    shrink_oof = np.clip(1.0 - width_oof / w_max, 0.0, 1.0)
    oof_pred[other] = oof_q50[other] + corr_oof * shrink_oof
    rae_oof = float(rae(y_tr[other], oof_pred[other]))
    print(f"\n[OOF folds1-4] RAE q50         = {rae(y_tr[other], oof_q50[other]):.4f}")
    print(f"[OOF folds1-4] RAE conformal   = {rae_oof:.4f}")

    # ---- Honest unblind ----
    te_names = te_df["Molecule Name"].tolist()
    n2i = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(n2i)].reset_index(drop=True)
    unb_idx = np.array([n2i[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)

    rae_u_q50 = float(rae(unb_y, te_q50[unb_idx]))
    rae_u_pred = float(rae(unb_y, te_pred[unb_idx]))
    print("\n" + "=" * 78)
    print(f"UNBLIND RAE (n={len(unb_idx)})")
    print(f"  q50 alone       = {rae_u_q50:.4f}")
    print(f"  conformal pred  = {rae_u_pred:.4f}")
    print(f"  truth mean/std  = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"  pred mean/std   = {te_pred.mean():.3f} / {te_pred.std():.3f}")
    print("=" * 78)

    # ---- Save artifacts ----
    np.save(ART / f"{TAG}_te_pred.npy", te_pred.astype(np.float32))
    np.save(ART / f"{TAG}_te_q10.npy",  te_q10.astype(np.float32))
    np.save(ART / f"{TAG}_te_q50.npy",  te_q50.astype(np.float32))
    np.save(ART / f"{TAG}_te_q90.npy",  te_q90.astype(np.float32))
    np.save(ART / f"{TAG}_oof_q10.npy", oof_q10.astype(np.float32))
    np.save(ART / f"{TAG}_oof_q50.npy", oof_q50.astype(np.float32))
    np.save(ART / f"{TAG}_oof_q90.npy", oof_q90.astype(np.float32))
    np.save(ART / f"{TAG}_edges.npy",   edges.astype(np.float64))
    np.save(ART / f"{TAG}_buckets.npy", bucket_means.astype(np.float64))
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_pred.astype(np.float32))

    sub = SUBMISSIONS / f"{TAG}_quantile_conformal.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": te_pred.astype(np.float32),
    }).to_csv(sub, index=False)
    print(f"\nWrote {sub}")

    return {
        "success": True,
        "rae_oof_q50_subset":   float(rae(y_tr[other], oof_q50[other])),
        "rae_oof_conformal":    rae_oof,
        "rae_unblind_q50":      rae_u_q50,
        "rae_unblind_conformal": rae_u_pred,
        "in_rae":               rae_u_pred,
        "submission":           str(sub),
    }


if __name__ == "__main__":
    r = main()
    print("\n==== SUMMARY ====")
    for k, v in r.items():
        print(f"  {k}: {v}")
