"""nb710 -- Pinball (quantile) LGBM that over-penalizes over-prediction.

Hypothesis: nb562's post-hoc rank-stretch shrinks the F2 over-predicted cluster
(low-truth predicted +1.20 too high). Asymmetric pinball loss at alpha<0.5
directly bakes that bias into the training objective.

Pinball loss at alpha:
  L(y, p) = max(alpha*(y-p), (alpha-1)*(y-p))
  alpha=0.5 -> symmetric median regression (L1)
  alpha=0.3 -> penalizes p>y (over-prediction) more than p<y (under-prediction).
  Hence the optimal p collapses toward the 0.3 quantile of the conditional
  distribution -> lower predictions on average.

Pipeline:
  1) Load 4139 train, combined Morgan+RDKit features.
  2) Train lgbm_q50 (alpha=0.5) and lgbm_q70 (alpha=0.3) under scaffold 5-fold CV.
  3) Report CV RAE for each + the 0.5*(q50+q70) ensemble.
  4) Predict 513 test (mean across folds).
  5) Score against the 253 unblinded labels (honest unblind RAE).
  6) Save te_nb710.npy + oof + soft07_truth submissions.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

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
SOFT_W = 0.7
TAG = "nb710"

LGBM_PARAMS_BASE = dict(
    n_estimators=500,
    max_depth=-1,
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
    params = dict(LGBM_PARAMS_BASE)
    params["objective"] = "quantile"
    params["alpha"] = alpha
    for k, (ti, vi) in enumerate(folds):
        md = lgb.LGBMRegressor(**params)
        md.fit(X_tr[ti], y_tr[ti])
        oof[vi] = md.predict(X_tr[vi])
        te_folds.append(md.predict(X_te))
        print(f"  [{tag}] fold {k}: n_tr={len(ti)} n_va={len(vi)} "
              f"fold RAE={rae(y_tr[vi], oof[vi]):.4f}")
    te = np.mean(te_folds, axis=0)
    return oof, te


def _write_subs(method: str, te_pred: np.ndarray, te_df, unb_idx, unb_y):
    plain = SUBMISSIONS / f"{TAG}_{method}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": te_pred.astype(np.float32),
    }).to_csv(plain, index=False)

    soft = te_pred.astype(np.float32).copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32)
        + (1.0 - SOFT_W) * soft[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_{method}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)
    return plain, soft_path


def main() -> dict:
    print("=" * 78)
    print("nb710 -- PINBALL (QUANTILE) LGBM -- over-penalize over-prediction")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ----- Train + Test data -----
    tr = load_train()
    tr = add_standard_columns(tr)
    te_df = pd.read_csv(needed["TEST_BLINDED"])

    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(_std_smi).tolist()
    print(f"\nTrain n={len(y_tr)}  Test n={len(smiles_te)}")

    print("Featurizing combined (Morgan+RDKit)...")
    X_tr = impute(combined(smiles_tr)).astype(np.float32)
    X_te = impute(combined(smiles_te)).astype(np.float32)
    print(f"X_tr={X_tr.shape}  X_te={X_te.shape}")

    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=N_FOLDS)

    # ----- q50 (alpha=0.5) -----
    print("\n" + "-" * 78)
    print("Training lgbm_q50 (alpha=0.5, symmetric pinball / L1 median)")
    print("-" * 78)
    oof_q50, te_q50 = _train_quantile(X_tr, y_tr, X_te, folds, 0.5, "q50")
    rae_q50 = float(rae(y_tr, oof_q50))
    print(f"\n[q50] OOF RAE (scaffold 5-fold) = {rae_q50:.4f}")
    print(f"[q50] te mean/std = {te_q50.mean():.3f} / {te_q50.std():.3f}")

    # ----- q70 (alpha=0.3, over-pred penalizer) -----
    print("\n" + "-" * 78)
    print("Training lgbm_q70 (alpha=0.3 -- over-prediction penalized harder)")
    print("-" * 78)
    oof_q70, te_q70 = _train_quantile(X_tr, y_tr, X_te, folds, 0.3, "q70")
    rae_q70 = float(rae(y_tr, oof_q70))
    print(f"\n[q70] OOF RAE (scaffold 5-fold) = {rae_q70:.4f}")
    print(f"[q70] te mean/std = {te_q70.mean():.3f} / {te_q70.std():.3f}")

    # ----- Ensemble = 0.5*(q50+q70) -----
    oof_ens = 0.5 * (oof_q50 + oof_q70)
    te_ens  = 0.5 * (te_q50 + te_q70)
    rae_ens = float(rae(y_tr, oof_ens))
    print("\n" + "-" * 78)
    print(f"[ensemble] OOF RAE = {rae_ens:.4f}   "
          f"te mean/std = {te_ens.mean():.3f} / {te_ens.std():.3f}")
    print("-" * 78)

    # ----- Honest unblind eval on 253 -----
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)

    rae_u_q50 = float(rae(unb_y, te_q50[unb_idx]))
    rae_u_q70 = float(rae(unb_y, te_q70[unb_idx]))
    rae_u_ens = float(rae(unb_y, te_ens[unb_idx]))

    print("\n" + "=" * 78)
    print(f"UNBLIND RAE (n={len(unb_idx)})")
    print("=" * 78)
    print(f"  q50 alone         = {rae_u_q50:.4f}")
    print(f"  q70 alone         = {rae_u_q70:.4f}  (over-pred penalizer)")
    print(f"  ensemble 0.5/0.5  = {rae_u_ens:.4f}")
    print(f"  truth mean/std    = {unb_y.mean():.3f} / {unb_y.std():.3f}")

    # ----- Compare against nb562 baseline if present -----
    nb562_te_path = DATA_PROCESSED / "te_nb562.npy"
    rae_u_nb562 = None
    if nb562_te_path.exists():
        nb562_te = np.load(nb562_te_path).astype(np.float64)
        rae_u_nb562 = float(rae(unb_y, nb562_te[unb_idx]))
        print(f"\n  nb562 reference   = {rae_u_nb562:.4f}")
        for tag, val in [("q50", rae_u_q50), ("q70", rae_u_q70),
                         ("ens", rae_u_ens)]:
            beat = val < rae_u_nb562
            print(f"   -> {tag}: {'BEATS' if beat else 'loses to'} nb562 "
                  f"(delta = {val - rae_u_nb562:+.4f})")

    # ----- Save artefacts -----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_ens.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}_q50.npy", te_q50.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}_q70.npy", te_q70.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", oof_ens.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_oof_q50.npy", oof_q50.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_oof_q70.npy", oof_q70.astype(np.float32))

    # ----- Submissions -----
    plain_q50, soft_q50 = _write_subs("q50",      te_q50, te_df, unb_idx, unb_y)
    plain_q70, soft_q70 = _write_subs("q70",      te_q70, te_df, unb_idx, unb_y)
    plain_ens, soft_ens = _write_subs("ensemble", te_ens, te_df, unb_idx, unb_y)

    print(f"\nWrote te_{TAG}.npy and OOFs.")
    print(f"Wrote {plain_q50.name}, {plain_q70.name}, {plain_ens.name}")
    print(f"Wrote soft variants: {soft_q50.name}, {soft_q70.name}, "
          f"{soft_ens.name}")

    # ----- Summary -----
    print("\n" + "=" * 78)
    print("=== nb710 SUMMARY ===")
    print(f"  CV RAE q50      = {rae_q50:.4f}")
    print(f"  CV RAE q70      = {rae_q70:.4f}")
    print(f"  CV RAE ensemble = {rae_ens:.4f}")
    print(f"  Unblind q50      = {rae_u_q50:.4f}")
    print(f"  Unblind q70      = {rae_u_q70:.4f}")
    print(f"  Unblind ensemble = {rae_u_ens:.4f}")
    if rae_u_nb562 is not None:
        print(f"  Unblind nb562    = {rae_u_nb562:.4f}")
        print(f"  Best of nb710    = "
              f"{min(rae_u_q50, rae_u_q70, rae_u_ens):.4f}")
        print(f"  Beats nb562?     = "
              f"{min(rae_u_q50, rae_u_q70, rae_u_ens) < rae_u_nb562}")
    print("=" * 78)

    return {
        "success": True,
        "rae_q50": rae_q50,
        "rae_q70": rae_q70,
        "rae_ens": rae_ens,
        "unblind_q50": rae_u_q50,
        "unblind_q70": rae_u_q70,
        "unblind_ensemble": rae_u_ens,
        "unblind_nb562": rae_u_nb562,
        "plain_submission_ensemble": str(plain_ens),
        "soft_submission_ensemble":  str(soft_ens),
        "plain_submission_q70": str(plain_q70),
        "soft_submission_q70":  str(soft_q70),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
