"""nb1261 -- DEPLOY artifact for the nb1252 (BoB-of-BoBs ChEMBL kNN + MACCS
residual) + nb1211 (BoB-of-BoBs blend) naive 0.5/0.5 blend on 513 test.

PRECEDENT
---------
nb1252 (diagnostic) outer-bag validation rebuilt the nb1242 residual bag
5 times with disjoint inner-seed quintets (inner = outer * 1000 + base),
each producing its own 5-seed mean-bag corrected OOF.  The row-level mean
of these 5 per-outer corrected OOFs (BoB-of-BoBs mean) gave pooled RAE
0.5446 on 253 unblind -- slightly worse than nb1242's 0.5431 alone, but
when 0.5/0.5-mean-blended with nb1211 it dropped to 0.5407 (vs nb1251's
0.55/0.45 blend of (nb1242, nb1211) at 0.5394).  nb1252 is a stronger
blend partner with nb1211 than nb1242 -- bigger variance reduction.

PROTOCOL
--------
Step A -- Build te_nb1252.npy (513,) deploy:
  - For each outer seed o in {0, 1, 7, 42, 137}:
      inner_seeds(o) = [o*1000 + s for s in {0, 1, 7, 42, 137}]
      For each inner seed s in inner_seeds(o):
          Fit shallow LGBM Huber (depth=3, leaves=7, n_est=80, lr=0.05,
          min_child=20, alpha=1.0, random_state=s) on ALL 253 unblind rows
          of the (MACCS-167 + pred_chembl_pec50 + sim_chembl) feature matrix
          predicting residual = y_unb - nb1070_pred_oof.  Predict on the
          513-row test slice -> resid_513_seed_s.
      Mean-bag across 5 inner seeds -> resid_513_outer_o (513,).
  - Row-bag mean across 5 outer seeds -> te_nb1252_residual_513 (513,).
  - te_nb1252 = te_nb1070 + te_nb1252_residual_513.
  - Save data/processed/te_nb1252.npy.

Step B -- Blend with nb1211 deploy:
  - Load te_nb1220.npy (nb1211 deploy).
  - te_nb1261 = 0.5 * te_nb1252 + 0.5 * te_nb1220.
  - Save data/processed/te_nb1261.npy and
    submissions/nb1261_deploy_nb1252plus1211.csv.

NOTE
----
This is a POST-unblind deploy (each LGBM fit on ALL 253 unblind rows).
in_RAE on te[unb_idx] is in-sample optimistic.  LB-faithful anchor is the
nb1252 BoB-of-BoBs + nb1211 cross-fit blend RAE = 0.5407.

Outputs:
  data/processed/te_nb1252.npy             (513,) float32
  data/processed/te_nb1261.npy             (513,) float32
  submissions/nb1261_deploy_nb1252plus1211.csv     (513 rows)
  data/processed/nb1261_summary.json
"""
from __future__ import annotations

import json
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
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1261"
NB1252_TAG = "nb1252"
ANCHOR = "nb1070"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE = [0, 1, 7, 42, 137]  # inner = outer * 1000 + base

W_NB1252 = 0.5
W_NB1211 = 0.5
NB1261_HONEST_CROSSFIT = 0.5407

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"

SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _save_submission_csv(te_pred, te_smiles, te_names, csv_path, label):
    assert te_pred.shape[0] == 513
    assert np.all(np.isfinite(te_pred))
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred.astype(np.float64),
    })
    assert len(sub) == 513
    assert list(sub.columns) == ["SMILES", "Molecule Name", "pEC50"]
    assert sub.isna().sum().sum() == 0
    sub.to_csv(csv_path, index=False)
    return {"csv_path": csv_path, "n_rows": int(len(sub)),
            "columns": list(sub.columns)}


def _build_te_nb1252(
    X_unb: np.ndarray,
    X_test: np.ndarray,
    residual_target: np.ndarray,
    te_anchor: np.ndarray,
    y_unb: np.ndarray,
    unb_idx: np.ndarray,
):
    """Build te_nb1252 (513,) following the nb1252 BoB-of-BoBs recipe.

    Returns
    -------
    te_nb1252       : (513,) float64
    per_outer_te    : (5, 513) float64 (each row is the mean-bag deploy for one
                                        outer seed)
    per_outer_in_rae: list[float] (in_RAE on te[unb_idx] for each per-outer
                                   mean-bag deploy, in-sample)
    inner_in_rae    : list[list[float]] (per inner LGBM in_RAE per outer)
    """
    n_test = X_test.shape[0]
    per_outer_te = np.zeros((len(OUTER_SEEDS), n_test), dtype=np.float64)
    per_outer_in_rae = []
    inner_in_rae = []
    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [int(o) * 1000 + int(s) for s in INNER_BASE]
        bag_513 = np.zeros((len(inner_seeds), n_test), dtype=np.float64)
        inner_rae_list = []
        t_outer = time.time()
        for j, s in enumerate(inner_seeds):
            mdl = LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb, residual_target)
            resid_pred_513 = mdl.predict(X_test).astype(np.float64)
            bag_513[j] = resid_pred_513
            te_seed = te_anchor + resid_pred_513
            in_r = float(rae(y_unb, te_seed[unb_idx]))
            inner_rae_list.append(in_r)
        outer_resid_513 = bag_513.mean(axis=0)
        outer_te_513 = te_anchor + outer_resid_513
        in_rae_outer = float(rae(y_unb, outer_te_513[unb_idx]))
        per_outer_te[oi] = outer_te_513
        per_outer_in_rae.append(in_rae_outer)
        inner_in_rae.append(inner_rae_list)
        print(f"   outer {o:5d}  inner={inner_seeds}")
        print(f"     per-inner in_RAE(unb) = "
              f"[{', '.join(f'{r:.4f}' for r in inner_rae_list)}]")
        print(f"     outer mean-bag in_RAE(unb) = {in_rae_outer:.4f}   "
              f"elapsed {time.time() - t_outer:.1f}s")

    te_nb1252 = per_outer_te.mean(axis=0)
    return te_nb1252, per_outer_te, per_outer_in_rae, inner_in_rae


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY 0.5 * te_nb1252 (BoB-of-BoBs ChEMBL+MACCS resid) +")
    print(f"           0.5 * te_nb1220 (nb1211 BoB-of-BoBs)  on 513 test")
    print(f"          honest cross-fit anchor = {NB1261_HONEST_CROSSFIT:.4f}")
    print(f"          OUTER seeds  = {OUTER_SEEDS}")
    print(f"          inner family = outer*1000 + base, base = {INNER_BASE}")
    print("=" * 78)

    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_test = len(te_smiles)
    assert n_test == 513

    te_nb1070 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    nb1070_oof = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert te_nb1070.shape == (n_test,)
    assert nb1070_oof.shape == (n_unb,)

    rae_anchor_oof = float(rae(y_unb, nb1070_oof))
    print(f"[load] te_{ANCHOR}.npy shape={te_nb1070.shape}")
    print(f"[load] {ANCHOR}_pred_oof.npy pooled RAE = {rae_anchor_oof:.4f}")

    residual_target = y_unb - nb1070_oof
    print(f"[resid] target mean={residual_target.mean():+.4f}  "
          f"std={residual_target.std():.4f}")

    # ChEMBL kNN feature caches.
    pred_chembl_path = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
    sim_chembl_path = DATA_PROCESSED / "sim_chembl_513.npy"
    if not pred_chembl_path.exists() or not sim_chembl_path.exists():
        raise FileNotFoundError(
            "Required ChEMBL feature caches missing -- run nb1250 first to "
            "generate pred_chembl_pec50_513.npy and sim_chembl_513.npy"
        )
    pred_chembl = np.load(pred_chembl_path).astype(np.float32)
    sim_chembl = np.load(sim_chembl_path).astype(np.float32)
    if pred_chembl.shape[0] != n_test or sim_chembl.shape[0] != n_test:
        raise ValueError("ChEMBL feature cache shape mismatch")
    print(f"[load] pred_chembl_pec50_513 shape={pred_chembl.shape}  "
          f"mean={pred_chembl.mean():.3f}  std={pred_chembl.std():.3f}")
    print(f"[load] sim_chembl_513        shape={sim_chembl.shape}  "
          f"mean={sim_chembl.mean():.3f}  std={sim_chembl.std():.3f}")

    # Build (513, 169) and (253, 169) feature matrices.
    X_maccs_te = np.load(MACCS_TE_PATH).astype(np.float32)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_test = np.concatenate(
        [X_maccs_te,
         pred_chembl.reshape(-1, 1).astype(np.float32),
         sim_chembl.reshape(-1, 1).astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    X_unb = X_test[unb_idx]
    print(f"[feat] X_test shape={X_test.shape}  X_unb shape={X_unb.shape}")

    # ---- Step A: build te_nb1252 (513) via BoB-of-BoBs deploy ----
    print("\n" + "-" * 78)
    print(f"STEP A: BUILD te_{NB1252_TAG}.npy (513) -- "
          f"{len(OUTER_SEEDS)} outer x {len(INNER_BASE)} inner = "
          f"{len(OUTER_SEEDS) * len(INNER_BASE)} LGBM fits on ALL 253 unblind")
    print("-" * 78)
    te_nb1252, per_outer_te, per_outer_in_rae, inner_in_rae = _build_te_nb1252(
        X_unb, X_test, residual_target, te_nb1070, y_unb, unb_idx,
    )
    in_rae_nb1252 = float(rae(y_unb, te_nb1252[unb_idx]))
    print("\n   te_nb1252  (row-bag mean across 5 outer mean-bags)")
    print(f"     mean={te_nb1252.mean():.4f}  std={te_nb1252.std():.4f}  "
          f"min={te_nb1252.min():.4f}  max={te_nb1252.max():.4f}")
    print(f"     in_RAE(unb) = {in_rae_nb1252:.4f}")

    te_1252_path = DATA_PROCESSED / f"te_{NB1252_TAG}.npy"
    np.save(te_1252_path, te_nb1252.astype(np.float32))
    print(f"[save] {te_1252_path}")

    # ---- Step B: blend with te_nb1220 ----
    print("\n" + "-" * 78)
    print(f"STEP B: blend  {W_NB1252} * te_{NB1252_TAG} + {W_NB1211} * te_nb1220")
    print("-" * 78)
    te_1220 = np.load(DATA_PROCESSED / "te_nb1220.npy").astype(np.float64)
    assert te_1220.shape == (n_test,)
    assert np.all(np.isfinite(te_1220))
    rae_1220_in = float(rae(y_unb, te_1220[unb_idx]))
    print(f"[load] te_nb1220  mean={te_1220.mean():.4f}  std={te_1220.std():.4f}  "
          f"in_RAE(unb) = {rae_1220_in:.4f}")

    te_nb1261 = W_NB1252 * te_nb1252 + W_NB1211 * te_1220
    in_rae_blend = float(rae(y_unb, te_nb1261[unb_idx]))
    pred_corr = float(np.corrcoef(te_nb1252, te_1220)[0, 1])

    print("\n" + "=" * 78)
    print("BLEND")
    print("=" * 78)
    print(f"   w_nb1252 = {W_NB1252:.4f}")
    print(f"   w_nb1211 = {W_NB1211:.4f}")
    print(f"   pred corr (te_nb1252, te_nb1220) = {pred_corr:.4f}")
    print(f"   te_nb1261  mean={te_nb1261.mean():.4f}  std={te_nb1261.std():.4f}  "
          f"min={te_nb1261.min():.4f}  max={te_nb1261.max():.4f}")
    print(f"   in_RAE(unb) = {in_rae_blend:.4f}  "
          f"(honest cross-fit anchor = {NB1261_HONEST_CROSSFIT:.4f})")

    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb1261.astype(np.float32))
    print(f"\n[save] {te_path}")

    csv_path = os.path.join(SUBMISSIONS_DIR, f"{TAG}_deploy_nb1252plus1211.csv")
    csv_info = _save_submission_csv(
        te_nb1261, te_smiles, te_names, csv_path, "nb1261"
    )
    print(f"[save] {csv_path}  rows={csv_info['n_rows']}  "
          f"cols={csv_info['columns']}")

    summary = {
        "tag": TAG,
        "recipe": f"{W_NB1252}*te_nb1252 + {W_NB1211}*te_nb1220",
        "step_a_te_1252_path": str(te_1252_path),
        "step_a_components": "outer-bag(5x5) ChEMBL+MACCS residual on nb1070 anchor",
        "step_b_components": ["te_nb1252 (BoB-of-BoBs deploy)",
                              "te_nb1220 (nb1211 BoB-of-BoBs deploy)"],
        "outer_seeds": OUTER_SEEDS,
        "inner_base": INNER_BASE,
        "inner_family_rule": "inner = outer*1000 + base  (matches nb1252)",
        "w_nb1252": W_NB1252,
        "w_nb1211": W_NB1211,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "rae_anchor_oof_253": rae_anchor_oof,
        "residual_target_mean": float(residual_target.mean()),
        "residual_target_std": float(residual_target.std()),
        "per_outer_in_rae_te_unb": per_outer_in_rae,
        "inner_in_rae_te_unb": inner_in_rae,
        "te_nb1252_stats": {
            "mean": float(te_nb1252.mean()), "std": float(te_nb1252.std()),
            "min": float(te_nb1252.min()),   "max": float(te_nb1252.max()),
        },
        "in_rae_nb1252_unb": in_rae_nb1252,
        "te_nb1220_stats": {
            "mean": float(te_1220.mean()), "std": float(te_1220.std()),
            "min": float(te_1220.min()),   "max": float(te_1220.max()),
        },
        "in_rae_nb1220_unb": rae_1220_in,
        "te_nb1261_stats": {
            "mean": float(te_nb1261.mean()), "std": float(te_nb1261.std()),
            "min": float(te_nb1261.min()),   "max": float(te_nb1261.max()),
        },
        "in_rae_nb1261_unb": in_rae_blend,
        "pred_corr_nb1252_nb1220": pred_corr,
        "nb1261_honest_crossfit_anchor": NB1261_HONEST_CROSSFIT,
        "te_path": str(te_path),
        "csv_path": csv_path,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "POST-unblind deploy: every LGBM is fit on ALL 253 unblind rows, "
            "so in_RAE on te[unb_idx] is in-sample optimistic. The LB-faithful "
            "anchor is the nb1252 BoB-of-BoBs + nb1211 0.5/0.5 cross-fit blend "
            "RAE 0.5407."
        ),
    }
    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {summary_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== STRUCTURED SUMMARY ====")
    print(f"  te_nb1252_mean: {res['te_nb1252_stats']['mean']:.4f}")
    print(f"  te_nb1252_std:  {res['te_nb1252_stats']['std']:.4f}")
    print(f"  te_nb1252_min:  {res['te_nb1252_stats']['min']:.4f}")
    print(f"  te_nb1252_max:  {res['te_nb1252_stats']['max']:.4f}")
    print(f"  in_rae_nb1252_unb: {res['in_rae_nb1252_unb']:.4f}")
    print(f"  te_nb1261_mean: {res['te_nb1261_stats']['mean']:.4f}")
    print(f"  te_nb1261_std:  {res['te_nb1261_stats']['std']:.4f}")
    print(f"  te_nb1261_min:  {res['te_nb1261_stats']['min']:.4f}")
    print(f"  te_nb1261_max:  {res['te_nb1261_stats']['max']:.4f}")
    print(f"  in_rae_nb1261_unb: {res['in_rae_nb1261_unb']:.4f}")
    print(f"  honest_anchor: {res['nb1261_honest_crossfit_anchor']:.4f}")
    print(f"  te_nb1252_path: {res['step_a_te_1252_path']}")
    print(f"  te_path: {res['te_path']}")
    print(f"  csv_path: {res['csv_path']}")
