"""nb711 -- Tail-weighted LGBM that emphasizes under-represented pEC50 tails.

Hypothesis: today's models (incl. nb562 & nb710 q70) compress to the
bulk-mean. The pEC50 distribution is concentrated near the inactive floor
(~4.5-5.0) with sparse tails at very-low (<4.0) and very-high (>6.5) values.
Inverse-density sample weights force LGBM to fit those extremes -- which is
exactly where RAE on the unblind 253 is dominated.

Pipeline:
  1) Load 4139 train, combined Morgan+RDKit features.
  2) Compute inverse-density sample weights:
       - 20 quantile bins of pec50
       - w_i = 1 / (density(bin_i) + eps)
       - normalize so mean(w)=1
  3) Train LGBM with sample_weight under scaffold 5-fold CV
     (n_estimators=500, max_depth=-1, num_leaves=64, lr=0.05,
      min_child_samples=20, reg_lambda=1.0, seed=0).
  4) Report CV RAE.
  5) Predict 513 test (mean across folds).
  6) Score against the 253 unblinded labels (honest unblind RAE).
  7) Save te_nb711.npy + oof + submissions.
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
N_BINS = 20
EPS = 1e-6
SOFT_W = 0.7
TAG = "nb711"

LGBM_PARAMS = dict(
    n_estimators=500,
    max_depth=-1,
    num_leaves=64,
    learning_rate=0.05,
    min_child_samples=20,
    reg_lambda=1.0,
    objective="regression",
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


def _inverse_density_weights(y: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Inverse-density weights via quantile bins; normalized to mean 1.0."""
    # Quantile edges. duplicates='drop' guards against ties in pec50.
    edges = np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    # Right edge inclusive: shift last edge a hair to catch the max value.
    edges[-1] = edges[-1] + 1e-9
    bins = np.digitize(y, edges[1:-1], right=False)  # bin idx in [0, len(edges)-2]
    counts = np.bincount(bins, minlength=len(edges) - 1).astype(np.float64)
    density = counts / counts.sum()
    w = 1.0 / (density[bins] + EPS)
    w *= len(w) / w.sum()  # normalize so mean(w)=1
    return w, bins, density


def _train_weighted(X_tr, y_tr, w_tr, X_te, folds):
    oof = np.zeros_like(y_tr, dtype=np.float64)
    te_folds = []
    for k, (ti, vi) in enumerate(folds):
        md = lgb.LGBMRegressor(**LGBM_PARAMS)
        md.fit(X_tr[ti], y_tr[ti], sample_weight=w_tr[ti])
        oof[vi] = md.predict(X_tr[vi])
        te_folds.append(md.predict(X_te))
        print(f"  fold {k}: n_tr={len(ti)} n_va={len(vi)} "
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
    print("nb711 -- TAIL-WEIGHTED LGBM -- emphasize low/high pEC50 tails")
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

    # ----- Sample weights -----
    w_tr, bin_idx, density = _inverse_density_weights(y_tr, N_BINS)
    print("\n" + "-" * 78)
    print(f"Inverse-density weights ({N_BINS} quantile bins)")
    print("-" * 78)
    print(f"  weight mean={w_tr.mean():.3f}  "
          f"min={w_tr.min():.3f}  max={w_tr.max():.3f}  "
          f"std={w_tr.std():.3f}")
    # Print weights by pec50 decile for diagnostic.
    deciles = np.quantile(y_tr, np.linspace(0, 1, 11))
    print("  decile  pec50-edges     n     mean_w")
    for d in range(10):
        m = (y_tr >= deciles[d]) & (y_tr <= deciles[d + 1] + 1e-9)
        if m.sum():
            print(f"   {d:>2d}  [{deciles[d]:.2f},{deciles[d+1]:.2f}]  "
                  f"{m.sum():>4d}   {w_tr[m].mean():.3f}")

    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=N_FOLDS)

    # ----- Train weighted LGBM -----
    print("\n" + "-" * 78)
    print("Training tail-weighted LGBM (scaffold 5-fold CV)")
    print("-" * 78)
    oof, te = _train_weighted(X_tr, y_tr, w_tr, X_te, folds)
    rae_cv = float(rae(y_tr, oof))
    print(f"\n[tail-weighted] OOF RAE (scaffold 5-fold) = {rae_cv:.4f}")
    print(f"[tail-weighted] te mean/std = {te.mean():.3f} / {te.std():.3f}")
    print(f"[tail-weighted] te p05/p50/p95 = "
          f"{np.quantile(te,0.05):.3f}/{np.quantile(te,0.5):.3f}/"
          f"{np.quantile(te,0.95):.3f}")

    # ----- Honest unblind eval on 253 -----
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)

    rae_u = float(rae(unb_y, te[unb_idx]))

    print("\n" + "=" * 78)
    print(f"UNBLIND RAE (n={len(unb_idx)})")
    print("=" * 78)
    print(f"  tail-weighted LGBM = {rae_u:.4f}")
    print(f"  truth mean/std     = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"  pred  mean/std     = {te[unb_idx].mean():.3f} / "
          f"{te[unb_idx].std():.3f}")

    # ----- Compare against nb562 baseline if present -----
    nb562_te_path = DATA_PROCESSED / "te_nb562.npy"
    rae_u_nb562 = None
    if nb562_te_path.exists():
        nb562_te = np.load(nb562_te_path).astype(np.float64)
        rae_u_nb562 = float(rae(unb_y, nb562_te[unb_idx]))
        print(f"\n  nb562 reference    = {rae_u_nb562:.4f}")
        beat = rae_u < rae_u_nb562
        print(f"   -> tail-weighted: {'BEATS' if beat else 'loses to'} nb562 "
              f"(delta = {rae_u - rae_u_nb562:+.4f})")

    # ----- Save artefacts -----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_sample_weights.npy",
            w_tr.astype(np.float32))

    # ----- Submissions -----
    plain, soft = _write_subs("tailw", te, te_df, unb_idx, unb_y)
    print(f"\nWrote te_{TAG}.npy + {TAG}_pred_oof.npy + "
          f"{TAG}_sample_weights.npy")
    print(f"Wrote {plain.name}")
    print(f"Wrote soft variant: {soft.name}")

    # ----- Summary -----
    print("\n" + "=" * 78)
    print("=== nb711 SUMMARY ===")
    print(f"  CV RAE (scaffold 5-fold) = {rae_cv:.4f}")
    print(f"  Unblind RAE (n=253)      = {rae_u:.4f}")
    if rae_u_nb562 is not None:
        print(f"  Unblind nb562            = {rae_u_nb562:.4f}")
        print(f"  Beats nb562?             = {rae_u < rae_u_nb562}")
    print("=" * 78)

    return {
        "success": True,
        "cv_rae": rae_cv,
        "unblind_rae": rae_u,
        "unblind_nb562": rae_u_nb562,
        "beats_nb562": (rae_u_nb562 is not None) and (rae_u < rae_u_nb562),
        "plain_submission": str(plain),
        "soft_submission":  str(soft),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
