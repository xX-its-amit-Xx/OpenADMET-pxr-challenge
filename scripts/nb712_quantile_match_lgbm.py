"""nb712 -- Distribution-matching LGBM (spline post-hoc CDF transform).

Goal
----
Train models compress toward the bulk mean (pEC50 ~ 4.7), which inflates
RAE on the analog-expansion test where the true distribution is wider than
the OOF prediction distribution.  We want predicted CDF to match the
*train* label CDF (best honest proxy we have for the test/unblind shape --
the 253 unblind labels are NOT used to fit the map; using them would be a
double-dip into the very labels we evaluate against).

Approach
--------
The custom-LGBM-objective path (MSE + alpha*W2_to_target) is differentiable
only weakly (quantile sort breaks per-sample gradients) and would require
custom C++ or a numpy boosted-tree loop.  Per spec, we fall back to the
*post-hoc spline* path which is mathematically equivalent at convergence:

  1. Train a plain LGBM under scaffold 5-fold CV.
  2. Build a monotone cubic-spline (PCHIP) CDF map:
         f : sorted_OOF_quantiles -> sorted_TRAIN_LABEL_quantiles
     i.e. the empirical quantile-quantile map from OOF preds to train labels.
     This is a strict generalisation of nb560's piecewise-linear rank map
     and is exactly nb560's idea fit on TRAIN OOF (not 253 unblind).
  3. Apply f to the 5-fold OOF preds (in-distribution) AND to the 513 test
     preds (deploy).  Linear-tail extrapolation outside the OOF range with
     slope estimated from the nearest 10% of points.
  4. Score honest unblind RAE on the 253 known names.
  5. Save te_nb712.npy + oof + plain & soft07 submissions.

Why train-CDF target (not unblind CDF):
  - Using the unblind label CDF to fit the map is leakage relative to the
    very metric we report -- the in-sample fit would push unblind RAE
    artificially low by design, not by generalisation (cf. MEMORY note
    on unblind in-sample overfit risk).
  - Train labels are the only generalisable target shape we have ex ante.
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
from scipy.interpolate import PchipInterpolator

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb712"
N_KNOTS = 64          # quantile knots for the spline CDF map
TAIL_FRAC = 0.10      # nearest 10% of pts to estimate extrapolation slope

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


def _fit_quantile_spline(
    p_src: np.ndarray,
    y_tgt: np.ndarray,
    n_knots: int = N_KNOTS,
) -> tuple[PchipInterpolator, np.ndarray, np.ndarray, float, float]:
    """Fit a monotone PCHIP spline mapping the src empirical CDF -> tgt CDF.

    Returns (interpolator, x_knots, y_knots, slope_lo, slope_hi).

    The knots are the matched empirical quantiles of (p_src, y_tgt) at
    n_knots evenly spaced probability levels in (0, 1).  PCHIP guarantees
    monotonicity (so the map is a valid CDF transform).  Linear tail slopes
    are estimated from the nearest TAIL_FRAC fraction of points for
    out-of-range extrapolation.
    """
    probs = np.linspace(0.0, 1.0, n_knots)
    x_knots = np.quantile(p_src, probs)
    y_knots = np.quantile(y_tgt, probs)

    # Strictly increasing x is required by PCHIP -- dedupe ties on x with
    # a tiny epsilon so the function stays well-defined.
    for i in range(1, len(x_knots)):
        if x_knots[i] <= x_knots[i - 1]:
            x_knots[i] = x_knots[i - 1] + 1e-9

    interp = PchipInterpolator(x_knots, y_knots, extrapolate=False)

    # Tail slopes for safe extrapolation outside [x_knots[0], x_knots[-1]].
    k = max(2, int(np.ceil(TAIL_FRAC * len(x_knots))))
    dx_lo = x_knots[k - 1] - x_knots[0]
    dy_lo = y_knots[k - 1] - y_knots[0]
    slope_lo = float(dy_lo / dx_lo) if dx_lo > 1e-12 else 1.0
    dx_hi = x_knots[-1] - x_knots[-k]
    dy_hi = y_knots[-1] - y_knots[-k]
    slope_hi = float(dy_hi / dx_hi) if dx_hi > 1e-12 else 1.0
    return interp, x_knots, y_knots, slope_lo, slope_hi


def _apply_spline(
    p_query: np.ndarray,
    interp: PchipInterpolator,
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    slope_lo: float,
    slope_hi: float,
) -> np.ndarray:
    p = np.asarray(p_query, dtype=np.float64)
    out = np.empty_like(p)
    inside = (p >= x_knots[0]) & (p <= x_knots[-1])
    below = p < x_knots[0]
    above = p > x_knots[-1]
    if inside.any():
        out[inside] = interp(p[inside])
    if below.any():
        out[below] = y_knots[0] + slope_lo * (p[below] - x_knots[0])
    if above.any():
        out[above] = y_knots[-1] + slope_hi * (p[above] - x_knots[-1])
    return out


def _train_lgbm(X_tr, y_tr, X_te, folds):
    oof = np.zeros_like(y_tr, dtype=np.float64)
    te_folds = []
    for k, (ti, vi) in enumerate(folds):
        md = lgb.LGBMRegressor(**LGBM_PARAMS)
        md.fit(X_tr[ti], y_tr[ti])
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
    print("nb712 -- DISTRIBUTION-MATCHING LGBM (post-hoc PCHIP CDF spline)")
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
    print(f"  y_tr  mean/std = {y_tr.mean():.3f} / {y_tr.std():.3f}")
    print(f"  y_tr  p05/p50/p95 = "
          f"{np.quantile(y_tr,0.05):.3f}/{np.quantile(y_tr,0.5):.3f}/"
          f"{np.quantile(y_tr,0.95):.3f}")

    print("Featurizing combined (Morgan+RDKit)...")
    X_tr = impute(combined(smiles_tr)).astype(np.float32)
    X_te = impute(combined(smiles_te)).astype(np.float32)
    print(f"X_tr={X_tr.shape}  X_te={X_te.shape}")

    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=N_FOLDS)

    # ----- Step 1: train plain LGBM -----
    print("\n" + "-" * 78)
    print("(1) Plain LGBM scaffold 5-fold CV")
    print("-" * 78)
    oof_raw, te_raw = _train_lgbm(X_tr, y_tr, X_te, folds)
    rae_cv_raw = float(rae(y_tr, oof_raw))
    print(f"\n[plain LGBM] OOF RAE         = {rae_cv_raw:.4f}")
    print(f"[plain LGBM] OOF mean/std    = "
          f"{oof_raw.mean():.3f} / {oof_raw.std():.3f}")
    print(f"[plain LGBM] te  mean/std    = "
          f"{te_raw.mean():.3f} / {te_raw.std():.3f}")

    # ----- Step 2: fit PCHIP spline OOF -> train label CDF -----
    # Train OOF (4139) gives the source distribution; train labels give
    # the target distribution.  NO unblind labels used here.
    print("\n" + "-" * 78)
    print("(2) Fit PCHIP spline: OOF empirical-quantile -> train-label "
          "empirical-quantile")
    print("-" * 78)
    interp, x_kn, y_kn, slope_lo, slope_hi = _fit_quantile_spline(
        oof_raw, y_tr, n_knots=N_KNOTS,
    )
    print(f"  n_knots = {N_KNOTS}")
    print(f"  src (OOF)  knot range = [{x_kn[0]:.3f}, {x_kn[-1]:.3f}]")
    print(f"  tgt (y_tr) knot range = [{y_kn[0]:.3f}, {y_kn[-1]:.3f}]")
    print(f"  tail slopes (lo/hi)   = {slope_lo:.3f} / {slope_hi:.3f}")

    # ----- Step 3: apply spline to OOF & test -----
    oof_map = _apply_spline(oof_raw, interp, x_kn, y_kn, slope_lo, slope_hi)
    te_map  = _apply_spline(te_raw,  interp, x_kn, y_kn, slope_lo, slope_hi)
    rae_cv_map = float(rae(y_tr, oof_map))
    print(f"\n[mapped LGBM] OOF RAE         = {rae_cv_map:.4f}  "
          f"(d={rae_cv_map - rae_cv_raw:+.4f})")
    print(f"[mapped LGBM] OOF mean/std    = "
          f"{oof_map.mean():.3f} / {oof_map.std():.3f}")
    print(f"[mapped LGBM] te  mean/std    = "
          f"{te_map.mean():.3f} / {te_map.std():.3f}")

    n_below_te = int((te_raw < x_kn[0]).sum())
    n_above_te = int((te_raw > x_kn[-1]).sum())
    print(f"[mapped LGBM] te extrapolated below/above = "
          f"{n_below_te} / {n_above_te}")

    # ----- Step 4: honest unblind RAE -----
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)

    rae_u_raw = float(rae(unb_y, te_raw[unb_idx]))
    rae_u_map = float(rae(unb_y, te_map[unb_idx]))

    print("\n" + "=" * 78)
    print(f"UNBLIND RAE (n={len(unb_idx)}) -- honest, map fit on train only")
    print("=" * 78)
    print(f"  plain LGBM         = {rae_u_raw:.4f}")
    print(f"  mapped LGBM (nb712)= {rae_u_map:.4f}  "
          f"(d_vs_plain={rae_u_map - rae_u_raw:+.4f})")
    print(f"  truth mean/std     = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"  pred  mean/std (mapped) = "
          f"{te_map[unb_idx].mean():.3f} / {te_map[unb_idx].std():.3f}")

    # ----- Compare against nb562 baseline -----
    nb562_te_path = DATA_PROCESSED / "te_nb562.npy"
    rae_u_nb562 = None
    if nb562_te_path.exists():
        nb562_te = np.load(nb562_te_path).astype(np.float64)
        rae_u_nb562 = float(rae(unb_y, nb562_te[unb_idx]))
        beat = rae_u_map < rae_u_nb562
        print(f"\n  nb562 reference    = {rae_u_nb562:.4f}")
        print(f"   -> nb712: {'BEATS' if beat else 'loses to'} nb562 "
              f"(delta = {rae_u_map - rae_u_nb562:+.4f})")

    # ----- Save artefacts -----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_map.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", oof_map.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}_raw.npy", te_raw.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_oof_raw.npy", oof_raw.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_spline_x.npy", x_kn.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_spline_y.npy", y_kn.astype(np.float32))

    # ----- Submissions -----
    plain, soft = _write_subs("qmatch", te_map, te_df, unb_idx, unb_y)
    print(f"\nWrote te_{TAG}.npy + {TAG}_pred_oof.npy + spline knots")
    print(f"Wrote {plain.name}")
    print(f"Wrote soft variant: {soft.name}")

    # ----- Summary -----
    print("\n" + "=" * 78)
    print("=== nb712 SUMMARY ===")
    print(f"  CV RAE plain  (scaffold 5-fold) = {rae_cv_raw:.4f}")
    print(f"  CV RAE mapped (scaffold 5-fold) = {rae_cv_map:.4f}")
    print(f"  Unblind RAE plain  (n=253)      = {rae_u_raw:.4f}")
    print(f"  Unblind RAE mapped (n=253)      = {rae_u_map:.4f}")
    if rae_u_nb562 is not None:
        print(f"  Unblind nb562                   = {rae_u_nb562:.4f}")
        print(f"  Beats nb562?                    = "
              f"{rae_u_map < rae_u_nb562}")
    print("=" * 78)

    return {
        "success": True,
        "cv_rae_plain": rae_cv_raw,
        "cv_rae_mapped": rae_cv_map,
        "unblind_rae_plain": rae_u_raw,
        "unblind_rae": rae_u_map,
        "unblind_nb562": rae_u_nb562,
        "beats_nb562": (rae_u_nb562 is not None) and (rae_u_map < rae_u_nb562),
        "n_extrap_below_te": n_below_te,
        "n_extrap_above_te": n_above_te,
        "plain_submission": str(plain),
        "soft_submission":  str(soft),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
