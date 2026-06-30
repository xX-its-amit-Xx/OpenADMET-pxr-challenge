"""nb1420 -- DEPLOY of nb1411 (3-way naive 1/3 blend of nb1373 + nb1352 + nb1364) to 513.

Loads the three component 513-row deploy vectors (one of them, nb1364, may need
to be built inline) and averages them with equal weights 1/3 each.

Honest LB anchors (from nb1411 cross-fit / grid):
    * naive 1/3 mean    0.5037
    * SLSQP cross-fit   0.5045

Outputs:
    data/processed/te_nb1420.npy                       (513,) float32
    data/processed/te_nb1364.npy   (only if newly built; (513,) float32)
    data/processed/nb1420_summary.json
    submissions/nb1420_deploy_nb1411.csv               (513 rows: SMILES,Molecule Name,pEC50)
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

TAG = "nb1420"
ANCHOR = "nb1070"

RESID_SEEDS = [0, 1, 7, 42, 137]

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

HONEST_LB_ANCHOR_NAIVE = 0.5037
HONEST_LB_ANCHOR_SLSQP = 0.5045


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


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    """Load cached Mordred test matrix (513 x 1533). Median-impute NaN/Inf."""
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _build_te_nb1364(n_test: int, unb_idx: np.ndarray, y_unb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Build nb1364 deploy vector (513) using the recipe in nb1364_summary.json.

    Recipe:
      * Top-30 Mordred col indices from nb1364 summary
      * + pred_chembl_pec50_513 + sim_chembl_513 (cached)
      * 5-seed bag, shallow LGBM Huber, fit on ALL 253 unblind rows
        with residual target = y_unb - nb1070_pred_oof
      * Predict residual on 513, add to te_nb1070
    """
    info: dict = {}
    print("-" * 78)
    print("BUILD te_nb1364 (Mordred-30 + ChEMBL) inline")
    print("-" * 78)

    # Load nb1364 summary for top-30 Mordred indices
    parent_summary_path = DATA_PROCESSED / "nb1364_summary.json"
    with open(parent_summary_path) as f:
        parent_summary = json.load(f)
    top_col_idx_ranked = np.array(
        parent_summary["top_mordred_col_indices_ranked"], dtype=int
    )
    top_k = int(parent_summary["top_k_mordred"])
    if len(top_col_idx_ranked) != top_k:
        raise ValueError(
            f"nb1364 top-{top_k} indices mismatch: got {len(top_col_idx_ranked)}"
        )
    print(f"[load] nb1364 top-{top_k} Mordred cols (ranked) = "
          f"{top_col_idx_ranked.tolist()}")
    info["top_mordred_col_indices_ranked"] = top_col_idx_ranked.tolist()
    info["top_k_mordred"] = top_k

    # Load Mordred test cache (513 x 1533)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    n_mord = int(X_mord_te.shape[1])
    print(f"[feat] X_mord_te shape = {X_mord_te.shape}  (n_mordred={n_mord})")
    info["n_mordred_cols"] = n_mord

    # Anchor
    te_anchor_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    anchor_oof_253 = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)

    # Cached ChEMBL features (513)
    pred_chembl_513 = np.load(DATA_PROCESSED / "pred_chembl_pec50_513.npy").astype(np.float32)
    sim_chembl_513 = np.load(DATA_PROCESSED / "sim_chembl_513.npy").astype(np.float32)
    print(f"[load] pred_chembl_pec50_513  mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[load] sim_chembl_513         mean={sim_chembl_513.mean():.3f}  "
          f"std={sim_chembl_513.std():.3f}")

    # Prune Mordred to top-30 cols
    X_mord_te_pruned = X_mord_te[:, top_col_idx_ranked].astype(np.float32)
    print(f"[feat] X_mord_te_pruned shape = {X_mord_te_pruned.shape}")

    # Build 32-col feature matrices on 513 and on 253 unblind
    X_te_pruned = np.concatenate(
        [
            X_mord_te_pruned,
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat] X_te_pruned (513) shape = {X_te_pruned.shape}")

    X_mord_unb_pruned = X_mord_te[unb_idx][:, top_col_idx_ranked].astype(np.float32)
    pred_chembl_unb = pred_chembl_513[unb_idx]
    sim_chembl_unb = sim_chembl_513[unb_idx]
    X_unb_pruned = np.concatenate(
        [
            X_mord_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            sim_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat] X_unb_pruned (253) shape = {X_unb_pruned.shape}")
    info["feat_dim_pruned"] = int(X_te_pruned.shape[1])

    # Residual target
    residual_unb = y_unb - anchor_oof_253
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # 5-seed deploy bag: fit on ALL 253 unblind
    print("[deploy] 5-seed bag fit on full 253 unblind")
    n_seeds = len(RESID_SEEDS)
    seed_resid_513 = np.zeros((n_seeds, n_test), dtype=np.float64)
    per_seed_records = []
    for si, s in enumerate(RESID_SEEDS):
        mdl = LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb_pruned, residual_unb)
        resid_in = mdl.predict(X_unb_pruned)
        corr_in = anchor_oof_253 + resid_in
        in_rae_s = float(rae(y_unb, corr_in))
        resid_513 = mdl.predict(X_te_pruned)
        seed_resid_513[si] = resid_513
        per_seed_records.append({
            "seed": int(s),
            "in_sample_rae_253": in_rae_s,
            "resid_513_mean": float(resid_513.mean()),
            "resid_513_std": float(resid_513.std()),
        })
        print(f"   seed {s:3d}:  in-sample RAE_253 = {in_rae_s:.4f}  "
              f"resid_513 mean={resid_513.mean():+.4f}  std={resid_513.std():.4f}")

    mean_resid_513 = seed_resid_513.mean(axis=0)
    te_nb1364 = te_anchor_513 + mean_resid_513
    print(f"[deploy] te_nb1364  mean={te_nb1364.mean():.4f}  std={te_nb1364.std():.4f}  "
          f"min={te_nb1364.min():.4f}  max={te_nb1364.max():.4f}")
    info["per_seed_records"] = per_seed_records
    info["te_nb1364_stats"] = {
        "mean": float(te_nb1364.mean()),
        "std": float(te_nb1364.std()),
        "min": float(te_nb1364.min()),
        "max": float(te_nb1364.max()),
    }
    return te_nb1364.astype(np.float32), info


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1411 (1/3 mean of nb1373 + nb1352 + nb1364) -> 513")
    print(f"          honest LB anchors: naive={HONEST_LB_ANCHOR_NAIVE}  "
          f"slsqp={HONEST_LB_ANCHOR_SLSQP}")
    print("=" * 78)

    # ---- Load test ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    else:
        cand = [c for c in te.columns if "name" in c.lower()]
        if not cand:
            raise KeyError(
                f"No Molecule Name column found in test ({te.columns.tolist()})"
            )
        mol_names = te[cand[0]].astype(str).tolist()
    print(f"[load] n_test={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Component 1: te_nb1380_mean (nb1373 deploy) ----
    te_nb1380_path = DATA_PROCESSED / "te_nb1380_mean.npy"
    if not te_nb1380_path.exists():
        raise FileNotFoundError(f"Missing component: {te_nb1380_path}")
    te_nb1380_mean = np.load(te_nb1380_path).astype(np.float64)
    if te_nb1380_mean.shape[0] != n_test:
        raise ValueError(f"te_nb1380_mean shape mismatch: {te_nb1380_mean.shape}")
    in_rae_1380 = float(rae(y_unb, te_nb1380_mean[unb_idx]))
    print(f"[comp] te_nb1380_mean (nb1373 deploy)  mean={te_nb1380_mean.mean():.4f}  "
          f"std={te_nb1380_mean.std():.4f}  in_RAE={in_rae_1380:.4f}")

    # ---- Component 2: te_nb1360_mean (nb1352 deploy) ----
    te_nb1360_path = DATA_PROCESSED / "te_nb1360_mean.npy"
    if not te_nb1360_path.exists():
        raise FileNotFoundError(f"Missing component: {te_nb1360_path}")
    te_nb1360_mean = np.load(te_nb1360_path).astype(np.float64)
    if te_nb1360_mean.shape[0] != n_test:
        raise ValueError(f"te_nb1360_mean shape mismatch: {te_nb1360_mean.shape}")
    in_rae_1360 = float(rae(y_unb, te_nb1360_mean[unb_idx]))
    print(f"[comp] te_nb1360_mean (nb1352 deploy)  mean={te_nb1360_mean.mean():.4f}  "
          f"std={te_nb1360_mean.std():.4f}  in_RAE={in_rae_1360:.4f}")

    # ---- Component 3: te_nb1364 (Mordred-30 + ChEMBL); build inline if missing ----
    te_nb1364_path = DATA_PROCESSED / "te_nb1364.npy"
    nb1364_build_info: dict | None = None
    nb1364_built_now = False
    if te_nb1364_path.exists():
        te_nb1364 = np.load(te_nb1364_path).astype(np.float64)
        if te_nb1364.shape[0] != n_test:
            raise ValueError(f"te_nb1364 shape mismatch: {te_nb1364.shape}")
        print(f"[comp] te_nb1364 loaded from cache  mean={te_nb1364.mean():.4f}  "
              f"std={te_nb1364.std():.4f}")
    else:
        print(f"[comp] te_nb1364 missing -- building inline")
        te_nb1364_f32, nb1364_build_info = _build_te_nb1364(n_test, unb_idx, y_unb)
        np.save(te_nb1364_path, te_nb1364_f32)
        print(f"[save] {te_nb1364_path}")
        te_nb1364 = te_nb1364_f32.astype(np.float64)
        nb1364_built_now = True
    in_rae_1364 = float(rae(y_unb, te_nb1364[unb_idx]))
    print(f"[comp] te_nb1364 (nb1364 deploy)       mean={te_nb1364.mean():.4f}  "
          f"std={te_nb1364.std():.4f}  in_RAE={in_rae_1364:.4f}")

    # ---- Naive 1/3 mean blend ----
    te_nb1420 = (te_nb1380_mean + te_nb1360_mean + te_nb1364) / 3.0
    in_rae_blend = float(rae(y_unb, te_nb1420[unb_idx]))

    print("\n" + "-" * 78)
    print("BLEND DIAGNOSTICS (1/3 mean of 3 components)")
    print("-" * 78)
    print(f"   te_nb1420  mean={te_nb1420.mean():.4f}  "
          f"std={te_nb1420.std():.4f}  "
          f"min={te_nb1420.min():.4f}  max={te_nb1420.max():.4f}")
    print(f"   in_RAE(unb)      = {in_rae_blend:.4f}")
    print(f"   honest LB anchor (naive)  = {HONEST_LB_ANCHOR_NAIVE}")
    print(f"   honest LB anchor (SLSQP)  = {HONEST_LB_ANCHOR_SLSQP}")

    # ---- Save NPY ----
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb1420.astype(np.float32))
    print(f"\n[save] {te_path}")

    # ---- Save CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1411.csv"
    df = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1420.astype(np.float64),
    })
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}  rows={len(df)}  cols={list(df.columns)}")

    summary = {
        "tag": TAG,
        "parent_method": "nb1411",
        "blend_recipe": "naive_1_3_mean",
        "components": ["te_nb1380_mean", "te_nb1360_mean", "te_nb1364"],
        "component_paths": {
            "te_nb1380_mean": str(te_nb1380_path),
            "te_nb1360_mean": str(te_nb1360_path),
            "te_nb1364": str(te_nb1364_path),
        },
        "nb1364_built_now": bool(nb1364_built_now),
        "nb1364_build_info": nb1364_build_info,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "te_nb1380_mean_stats": {
            "mean": float(te_nb1380_mean.mean()),
            "std": float(te_nb1380_mean.std()),
            "min": float(te_nb1380_mean.min()),
            "max": float(te_nb1380_mean.max()),
        },
        "te_nb1360_mean_stats": {
            "mean": float(te_nb1360_mean.mean()),
            "std": float(te_nb1360_mean.std()),
            "min": float(te_nb1360_mean.min()),
            "max": float(te_nb1360_mean.max()),
        },
        "te_nb1364_stats": {
            "mean": float(te_nb1364.mean()),
            "std": float(te_nb1364.std()),
            "min": float(te_nb1364.min()),
            "max": float(te_nb1364.max()),
        },
        "te_nb1420_stats": {
            "mean": float(te_nb1420.mean()),
            "std": float(te_nb1420.std()),
            "min": float(te_nb1420.min()),
            "max": float(te_nb1420.max()),
        },
        "in_rae_unb_nb1380": in_rae_1380,
        "in_rae_unb_nb1360": in_rae_1360,
        "in_rae_unb_nb1364": in_rae_1364,
        "in_rae_unb_blend": in_rae_blend,
        "honest_lb_anchor_naive": HONEST_LB_ANCHOR_NAIVE,
        "honest_lb_anchor_slsqp": HONEST_LB_ANCHOR_SLSQP,
        "te_npy_path": str(te_path),
        "csv_path": str(csv_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_test", "n_unb",
        "nb1364_built_now",
        "te_nb1380_mean_stats",
        "te_nb1360_mean_stats",
        "te_nb1364_stats",
        "te_nb1420_stats",
        "in_rae_unb_nb1380",
        "in_rae_unb_nb1360",
        "in_rae_unb_nb1364",
        "in_rae_unb_blend",
        "honest_lb_anchor_naive",
        "honest_lb_anchor_slsqp",
        "csv_path",
        "te_npy_path",
    ):
        print(f"  {k}: {res.get(k)}")
