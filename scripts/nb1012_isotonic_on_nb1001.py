"""nb1012 -- Cross-fit isotonic calibration ON TOP of nb1001 (chemprop_aux +
nb972 + stretch).

nb1001 scalar rank-stretch (s=1.25) lifted the (chemprop_aux + nb972) blend
from raw 0.6216 -> 0.5994 honest cross-fit on 253.  Scalar stretch only fixes
linear variance compression.  Residual non-linearity (e.g. mid-range
flattening, tail saturation) is unaddressed.

Hypothesis: a monotone (isotonic) calibration on top of the stretched blend
may correct residual non-linearity that the 1-parameter stretch cannot.

Procedure:
  1. Reconstruct the nb1001 honest 5-fold cross-fit on the 253 unblind:
       - candidates: [chemprop_aux, nb972_long_train]
       - per fold: SLSQP w0 + stretch grid -> nb1001 OOF.
       - save nb1001_pred_oof.npy (253,).
  2. 5-fold cross-fit IsotonicRegression(out_of_bounds='clip'):
       - per fold f: fit on (nb1001_oof[tr] -> y[tr]),
         apply to nb1001_oof[va] -> nb1012_oof[va].
  3. Pooled cross-fit RAE on the 253.
  4. Deploy: fit IsotonicRegression on ALL 253 (nb1001_oof_all -> y_unb),
     apply to te_nb1001 (513,) -> te_nb1012.

Outputs:
  data/processed/nb1001_pred_oof.npy       (created if missing)
  data/processed/nb1012_pred_oof.npy
  data/processed/te_nb1012.npy
  data/processed/nb1012_summary.json
  submissions/nb1012_isotonic_on_nb1001.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1012"
CANDIDATES = ["chemprop_aux", "nb972_long_train"]
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEED = 42
NB1001_CROSSFIT = 0.5994
CHEMPROP_AUX_RAW = 0.6216


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def slsqp_w0(p0: np.ndarray, p1: np.ndarray, y: np.ndarray) -> float:
    """Fit w0 in [0,1] minimizing SSE of w0*p0 + (1-w0)*p1 vs y."""
    P = np.column_stack([p0, p1])
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0), (0.0, 1.0)]
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.array([0.5, 0.5]),
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return float(res.x[0])


def best_stretch_on(blend_train: np.ndarray, y_train: np.ndarray,
                    mu: float) -> tuple[float, float]:
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        stretched = mu + s * (blend_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def regen_nb1001_oof(P_unb: np.ndarray, y_unb: np.ndarray
                     ) -> tuple[np.ndarray, list]:
    """Honest 5-fold cross-fit of (SLSQP w0 + stretch). Returns OOF (253,)."""
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n, np.nan)
    fold_rows = []
    for k, (tr, va) in enumerate(kf.split(np.arange(n))):
        w0 = slsqp_w0(P_unb[tr, 0], P_unb[tr, 1], y_unb[tr])
        blend_tr = w0 * P_unb[tr, 0] + (1.0 - w0) * P_unb[tr, 1]
        mu = float(blend_tr.mean())
        s, _ = best_stretch_on(blend_tr, y_unb[tr], mu)
        blend_va = w0 * P_unb[va, 0] + (1.0 - w0) * P_unb[va, 1]
        oof[va] = mu + s * (blend_va - mu)
        fold_rows.append({"fold": k, "w0": w0, "s": s, "mu_tr": mu,
                          "val_rae": float(rae(y_unb[va], oof[va])),
                          "n_va": int(len(va))})
    return oof, fold_rows


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- cross-fit ISOTONIC on top of nb1001 (chemprop_aux + "
          f"nb972 + stretch)")
    print("=" * 78)

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    preds_513 = np.column_stack([load_te(c, te_names) for c in CANDIDATES])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] P_unb shape = {P_unb.shape}, y shape = {y_unb.shape}")

    # ---- Step 1: get nb1001 OOF (load or regenerate) ----
    nb1001_oof_path = DATA_PROCESSED / "nb1001_pred_oof.npy"
    if nb1001_oof_path.exists():
        nb1001_oof = np.load(nb1001_oof_path).astype(np.float64)
        print(f"[load] nb1001_pred_oof.npy  shape={nb1001_oof.shape}")
        nb1001_fold_rows = "loaded_from_disk"
    else:
        print("[regen] nb1001_pred_oof.npy missing -- "
              "regenerating via nb1001 cross-fit protocol")
        nb1001_oof, nb1001_fold_rows = regen_nb1001_oof(P_unb, y_unb)
        np.save(nb1001_oof_path, nb1001_oof.astype(np.float32))
        print(f"[save] {nb1001_oof_path}")
    nb1001_oof_rae = float(rae(y_unb, nb1001_oof))
    print(f"[chk]  nb1001 OOF RAE on 253 = {nb1001_oof_rae:.4f}  "
          f"(expected ~{NB1001_CROSSFIT})")

    # =================================================================
    # Step 2: 5-fold cross-fit IsotonicRegression on nb1001 OOF
    # =================================================================
    print("\n" + "-" * 78)
    print("5-FOLD CROSS-FIT ISOTONIC  (fit on train OOF -> truth, "
          "apply to held-out OOF)")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    iso_oof = np.full(n_unb, np.nan)
    iso_fold_rows = []
    for k, (tr, va) in enumerate(kf.split(np.arange(n_unb))):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(nb1001_oof[tr], y_unb[tr])
        pred_va = ir.predict(nb1001_oof[va])
        iso_oof[va] = pred_va
        rae_tr = float(rae(y_unb[tr], ir.predict(nb1001_oof[tr])))
        rae_va = float(rae(y_unb[va], pred_va))
        # capture knots: distinct x values where isotonic is anchored
        n_knots = int(len(np.unique(ir.X_thresholds_))) \
            if hasattr(ir, "X_thresholds_") else -1
        iso_fold_rows.append({"fold": k, "train_rae": rae_tr,
                              "val_rae": rae_va,
                              "n_va": int(len(va)),
                              "n_knots": n_knots})
        print(f"   fold {k}: train_RAE={rae_tr:.4f}  val_RAE={rae_va:.4f}  "
              f"n_knots={n_knots}")

    pooled_rae = float(rae(y_unb, iso_oof))
    print(f"\n[cv] pooled honest cross-fit RAE on 253 = {pooled_rae:.4f}")

    # ---- save nb1012 oof ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", iso_oof.astype(np.float32))
    print(f"[save] {TAG}_pred_oof.npy")

    # =================================================================
    # Comparison
    # =================================================================
    delta_nb1001 = pooled_rae - NB1001_CROSSFIT
    delta_raw = pooled_rae - CHEMPROP_AUX_RAW
    print("\n" + "-" * 78)
    print("COMPARISON")
    print("-" * 78)
    print(f"   chemprop_aux raw                  = {CHEMPROP_AUX_RAW:.4f}")
    print(f"   nb1001 (stretch only)             = {NB1001_CROSSFIT:.4f}")
    print(f"   nb1012 (stretch + isotonic) [CV]  = {pooled_rae:.4f}")
    print(f"   delta (nb1012 - nb1001)           = {delta_nb1001:+.4f}")
    print(f"   delta (nb1012 - raw)              = {delta_raw:+.4f}")
    if delta_nb1001 <= -0.005:
        verdict = "ISOTONIC_HELPS"
        print("   verdict = ISOTONIC HELPS  "
              "(residual non-linearity captured beyond scalar stretch)")
    elif delta_nb1001 < 0.005:
        verdict = "ISOTONIC_INERT"
        print("   verdict = ISOTONIC INERT  "
              "(stretch already captured the calibration signal)")
    else:
        verdict = "ISOTONIC_OVERFIT"
        print("   verdict = ISOTONIC OVERFIT  "
              "(per-fold non-parametric fit hurts cross-fit)")

    # =================================================================
    # Deploy: fit IsotonicRegression on full 253 (nb1001_oof -> y_unb),
    # apply to te_nb1001 (513,) -> te_nb1012
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (fit isotonic on full 253 nb1001_oof, "
          "apply to te_nb1001)")
    print("-" * 78)
    te_nb1001_path = DATA_PROCESSED / "te_nb1001.npy"
    if not te_nb1001_path.exists():
        raise FileNotFoundError(f"missing {te_nb1001_path}")
    te_nb1001 = np.load(te_nb1001_path).astype(np.float64)
    print(f"[load] te_nb1001.npy  shape={te_nb1001.shape}  "
          f"mean={te_nb1001.mean():.3f}  std={te_nb1001.std():.3f}")

    ir_deploy = IsotonicRegression(out_of_bounds="clip")
    ir_deploy.fit(nb1001_oof, y_unb)
    in_rae_deploy = float(rae(y_unb, ir_deploy.predict(nb1001_oof)))
    deploy_513 = ir_deploy.predict(te_nb1001).astype(np.float32)
    n_knots_deploy = int(len(np.unique(ir_deploy.X_thresholds_)))
    print(f"   deploy n_knots      = {n_knots_deploy}")
    print(f"   in-sample RAE (253) = {in_rae_deploy:.4f}  "
          "(overfit lower bound)")
    print(f"   te(513) mean/std    = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}  "
          f"(nb1001 was {te_nb1001.mean():.3f}/{te_nb1001.std():.3f})")

    # =================================================================
    # Save te + submission CSV
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_isotonic_on_nb1001.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    summary = {
        "tag": TAG,
        "anchor": "nb1001 (chemprop_aux + nb972 + stretch)",
        "candidates": CANDIDATES,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "nb1001_oof_in_rae_253": nb1001_oof_rae,
        "nb1001_crossfit_baseline": NB1001_CROSSFIT,
        "chemprop_aux_raw_baseline": CHEMPROP_AUX_RAW,
        "nb1001_oof_regenerated": (not nb1001_oof_path.exists()
                                   if False else
                                   (nb1001_fold_rows != "loaded_from_disk")),
        "nb1001_regen_folds": (nb1001_fold_rows
                               if nb1001_fold_rows != "loaded_from_disk"
                               else None),
        "iso_fold_results": iso_fold_rows,
        "pooled_cv_rae_253": pooled_rae,
        "delta_vs_nb1001": delta_nb1001,
        "delta_vs_raw": delta_raw,
        "verdict": verdict,
        "deploy_n_knots": n_knots_deploy,
        "in_sample_rae_overfit_bound": in_rae_deploy,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "te_nb1001_mean": float(te_nb1001.mean()),
        "te_nb1001_std": float(te_nb1001.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                     = nb1001 OOF (253,)")
    print(f"   nb1001 baseline            = {NB1001_CROSSFIT:.4f}")
    print(f"   nb1012 cross-fit RAE (253) = {pooled_rae:.4f}")
    print(f"   delta vs nb1001            = {delta_nb1001:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   deploy n_knots             = {n_knots_deploy}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("pooled_cv_rae_253", "delta_vs_nb1001", "delta_vs_raw",
              "verdict", "deploy_n_knots",
              "in_sample_rae_overfit_bound", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
