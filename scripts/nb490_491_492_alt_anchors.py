"""nb490 / nb491 / nb492 -- RESIDUAL ROUTERS WITH ALTERNATIVE BASE ANCHORS.

Same architecture as nb481 (signed-residual LGBM cross-fit + sigmoid err_hat
gate, EXTENDED 33-col feature matrix) but with DIFFERENT base anchors:

    nb490 -> chemprop_aux  anchor   (te_chemprop_aux.npy)
    nb491 -> nb420         anchor   (te_nb420_frontier.npy)
    nb492 -> nb464         anchor   (te_nb464.npy)

Process per anchor:
  1. Load anchor te_*.npy (513 rows).
  2. Compute residual = truth - anchor_pred on 253 unblind.
  3. 5-fold KFold cross-fit shallow LGBM residual prediction on the 253 rows.
  4. Gate: alpha = sigmoid((err_hat - median(err_hat)) * 4.0).
  5. pred_oof = anchor[unb_idx] + alpha[unb_idx] * resid_oof  (HONEST).
  6. Refit residual LGBM on all 253 unblind -> deploy resid_hat for 513.
  7. Final deploy = anchor + alpha * resid_hat (513-vec).
  8. Save te_{nb490,nb491,nb492}.npy + pred_oof + resid_oof; emit plain and
     soft07_truth submissions.

Hyperparams identical to nb472 / nb481:
    LGBM(max_depth=3, n_est=80, lr=0.05, num_leaves=8, min_child_samples=20,
         reg_lambda=1.0, random_state=0).

Memory note: feature matrix is built ONCE (most expensive step: NN scan over
~4k train fps) and reused across all three anchors.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SCALE = 4.0
SOFT_W = 0.7

LGBM_PARAMS = dict(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=8,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)

ANCHORS = [
    ("nb490", "chemprop_aux", "te_chemprop_aux.npy"),
    ("nb491", "nb420",        "te_nb420_frontier.npy"),
    ("nb492", "nb464",        "te_nb464.npy"),
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _import_nb481():
    """Dynamically import build_feature_matrix from nb481 script."""
    nb481_path = Path(__file__).parent / "nb481_residual_router_extended.py"
    spec = importlib.util.spec_from_file_location("nb481_mod", nb481_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_anchor(
    tag: str,
    anchor_name: str,
    anchor: np.ndarray,
    X: np.ndarray,
    err_hat: np.ndarray,
    unb_idx: np.ndarray,
    unb_y: np.ndarray,
    te_df: pd.DataFrame,
) -> dict:
    n_te = anchor.shape[0]
    n_unb = unb_idx.shape[0]

    X_unb = X[unb_idx]
    y_resid = (unb_y - anchor[unb_idx]).astype(np.float32)

    print(f"\n--- {tag} ({anchor_name}) ---")
    print(f"  anchor stats: mean={anchor.mean():.3f} std={anchor.std():.3f}")
    print(f"  residual target: mean={y_resid.mean():+.3f} "
          f"std={y_resid.std():.3f} |mean|={np.abs(y_resid).mean():.3f}")

    # 5-fold cross-fit residual prediction (honest)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    resid_oof = np.zeros(n_unb, dtype=np.float32)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_unb[tr_i], y_resid[tr_i])
        resid_oof[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
    del mdl

    # Soft gate using err_hat
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)

    pred_oof = (anchor[unb_idx] + alpha_513[unb_idx] * resid_oof).astype(np.float32)

    # Refit on all 253 -> deploy to 513
    deploy_mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_mdl.fit(X_unb, y_resid)
    resid_hat_513 = deploy_mdl.predict(X).astype(np.float32)
    deploy = (anchor + alpha_513 * resid_hat_513).astype(np.float32)
    del deploy_mdl

    rae_anchor = float(rae(unb_y, anchor[unb_idx]))
    rae_routed = float(rae(unb_y, pred_oof))
    print(f"  unblind RAE anchor (standalone) = {rae_anchor:.4f}")
    print(f"  unblind RAE routed (cross-fit)  = {rae_routed:.4f}")
    print(f"  delta vs anchor                 = {rae_routed - rae_anchor:+.4f}")

    # Save arrays
    np.save(DATA_PROCESSED / f"te_{tag}.npy", deploy)
    np.save(DATA_PROCESSED / f"{tag}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{tag}_resid_oof.npy", resid_oof)

    # Submissions
    plain_path = SUBMISSIONS / f"{tag}_alt_anchor_{anchor_name}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain_path, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{tag}_alt_anchor_{anchor_name}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"  wrote te_{tag}.npy + {plain_path.name} + {soft_path.name}")

    return {
        "tag": tag,
        "anchor": anchor_name,
        "rae_anchor": rae_anchor,
        "rae_routed": rae_routed,
        "plain": str(plain_path),
        "soft": str(soft_path),
    }


def main() -> dict:
    print("=" * 78)
    print("nb490/491/492 -- ALT-ANCHOR RESIDUAL ROUTERS")
    print("=" * 78)

    # ---- Validate required inputs ----
    needed = {
        "te_nb432.npy": DATA_PROCESSED / "te_nb432.npy",   # used to build features
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for tag, anchor_name, fname in ANCHORS:
        needed[fname] = DATA_PROCESSED / fname
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING (required):", missing)
        return {"success": False, "missing": missing}

    nb432 = np.load(needed["te_nb432.npy"]).astype(np.float32)
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    n_te = nb432.shape[0]
    assert err_hat.shape == (n_te,)

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    print(f"test n={n_te}  unblind n={len(unb_idx)}")
    print(f"err_hat: mean={err_hat.mean():.3f} median={np.median(err_hat):.3f} "
          f"std={err_hat.std():.3f}")

    # ---- Build feature matrix ONCE (uses nb432-based feature engineering) ----
    print("\nImporting build_feature_matrix from nb481 ...")
    nb481_mod = _import_nb481()
    print("Building extended feature matrix (this is the expensive step):")
    X, feat_names = nb481_mod.build_feature_matrix(te_df, nb432)
    print(f"  X shape: {X.shape}  cols={len(feat_names)}")
    if nb481_mod.MISSING_LOG:
        print("  Missing feature sources:")
        for f in nb481_mod.MISSING_LOG:
            print(f"    - {f}")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---- Run each anchor ----
    results: list[dict] = []
    for tag, anchor_name, fname in ANCHORS:
        anchor = np.load(needed[fname]).astype(np.float32)
        if anchor.shape != (n_te,):
            print(f"  SKIP {tag} ({anchor_name}): bad shape {anchor.shape}")
            continue
        res = run_anchor(
            tag, anchor_name, anchor, X, err_hat, unb_idx, unb_y, te_df
        )
        results.append(res)

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("=== nb490/491/492 SUMMARY ===")
    print(f"  {'tag':6s} {'anchor':14s} {'RAE_anchor':>11s} {'RAE_routed':>11s} "
          f"{'delta':>8s}")
    for r in results:
        print(f"  {r['tag']:6s} {r['anchor']:14s} {r['rae_anchor']:11.4f} "
              f"{r['rae_routed']:11.4f} "
              f"{r['rae_routed'] - r['rae_anchor']:+8.4f}")
    print("=" * 78)

    summary = {"success": True}
    for r in results:
        summary[f"rae_{r['anchor']}_base"] = r["rae_anchor"]
        summary[f"rae_{r['anchor']}_routed"] = r["rae_routed"]
        summary[r["tag"]] = r
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
