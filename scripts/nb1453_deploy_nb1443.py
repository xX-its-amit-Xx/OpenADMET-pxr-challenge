"""nb1453 -- DEPLOY of nb1443 (75-model BoB-of-Bags) to 513.

For each (outer in {0,1,7,42,137}, inner = outer*1000 + base_seed,
component in {nb1373, nb1352, nb1364}) -- 75 models total -- fit a shallow
LGBM Huber on the appropriate PRUNED feature matrix over ALL 253 unblind
rows (no KFold, deploy mode) with target

    residual = y_unb - nb1070_pred_oof

and predict residual on the 513 PRUNED test rows.  Stack the 75 residual_513
vectors into (75, 513), take row-MEDIAN, add to te_nb1070.

Honest cross-fit LB anchor (nb1443 median): 0.4999

Outputs:
    data/processed/te_nb1453.npy            (513,) float32
    data/processed/nb1453_summary.json
    submissions/nb1453_deploy_nb1443.csv    (513 rows: SMILES, Molecule Name, pEC50)
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
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1453"
ANCHOR = "nb1070"
PARENT = "nb1443"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
COMPONENTS = ["nb1373", "nb1352", "nb1364"]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

HONEST_LB_ANCHOR_MEDIAN = 0.4999


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
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
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


def _build_component_matrices(component: str, n_test: int, unb_idx: np.ndarray,
                              pred_chembl_513: np.ndarray,
                              sim_chembl_513: np.ndarray):
    """Return (X_unb_pruned, X_te_pruned, meta) for one component.

    Uses CACHED top-K indices from each component's parent summary (nb1373
    AtomPair top-30, nb1352 MACCS top-20, nb1364 Mordred top-30).  This is
    consistent with how the deploy variants nb1380 / nb1360 / nb1354
    re-deploy from their cached pruned indices.
    """
    if component == "nb1373":
        summary_path = DATA_PROCESSED / "nb1373_summary.json"
        with open(summary_path) as f:
            s = json.load(f)
        top_bit_idx = np.array(
            s["top_atompair_bit_indices_ranked"], dtype=int
        )
        top_k = int(s["top_k_atompair"])
        if not ATOMPAIR_TE_PATH.exists():
            raise FileNotFoundError(f"Missing {ATOMPAIR_TE_PATH}")
        X_ap_te = np.load(ATOMPAIR_TE_PATH)
        if X_ap_te.shape[0] != n_test:
            raise ValueError(f"AtomPair cache shape mismatch: {X_ap_te.shape}")
        X_te_feat = X_ap_te[:, top_bit_idx].astype(np.float32)
        meta = {
            "src": "nb1373_summary_top_atompair",
            "top_k": top_k,
            "n_full": int(X_ap_te.shape[1]),
            "top_first10": [int(b) for b in top_bit_idx[:10].tolist()],
        }
    elif component == "nb1352":
        summary_path = DATA_PROCESSED / "nb1352_summary.json"
        with open(summary_path) as f:
            s = json.load(f)
        top_bit_idx = np.array(
            s["top_maccs_bit_indices_ranked"], dtype=int
        )
        top_k = int(s["top_k_maccs"])
        if not MACCS_TE_PATH.exists():
            raise FileNotFoundError(f"Missing {MACCS_TE_PATH}")
        X_ma_te = np.load(MACCS_TE_PATH)
        if X_ma_te.shape[0] != n_test:
            raise ValueError(f"MACCS cache shape mismatch: {X_ma_te.shape}")
        X_te_feat = X_ma_te[:, top_bit_idx].astype(np.float32)
        meta = {
            "src": "nb1352_summary_top_maccs",
            "top_k": top_k,
            "n_full": int(X_ma_te.shape[1]),
            "top_first10": [int(b) for b in top_bit_idx[:10].tolist()],
        }
    elif component == "nb1364":
        summary_path = DATA_PROCESSED / "nb1364_summary.json"
        with open(summary_path) as f:
            s = json.load(f)
        top_col_idx = np.array(
            s["top_mordred_col_indices_ranked"], dtype=int
        )
        top_k = int(s["top_k_mordred"])
        X_mord_te = _load_mordred_test(n_test_expected=n_test)
        n_mord = int(X_mord_te.shape[1])
        X_te_feat = X_mord_te[:, top_col_idx].astype(np.float32)
        meta = {
            "src": "nb1364_summary_top_mordred",
            "top_k": top_k,
            "n_full": n_mord,
            "top_first10": [int(b) for b in top_col_idx[:10].tolist()],
        }
    else:
        raise ValueError(f"Unknown component {component}")

    # Append ChEMBL features (pred_chembl + sim) -- shared across components.
    X_te_pruned = np.concatenate(
        [
            X_te_feat,
            pred_chembl_513.reshape(-1, 1).astype(np.float32),
            sim_chembl_513.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    X_unb_pruned = X_te_pruned[unb_idx].astype(np.float32)
    meta["feat_dim_pruned"] = int(X_te_pruned.shape[1])
    return X_unb_pruned, X_te_pruned, meta


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1443 (75-model BoB-of-Bags) -> 513")
    print(f"          anchor={ANCHOR}  parent={PARENT}")
    print(f"          outer seeds      = {OUTER_SEEDS}")
    print(f"          inner base seeds = {INNER_BASE_SEEDS}")
    print(f"          components       = {COMPONENTS}")
    print(f"          honest LB anchor (median): {HONEST_LB_ANCHOR_MEDIAN}")
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
            raise KeyError(f"No Molecule Name column found in test ({te.columns.tolist()})")
        mol_names = te[cand[0]].astype(str).tolist()
    print(f"[load] n_test={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Anchor (513) and anchor-OOF (253) ----
    te_anchor_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"te_{ANCHOR} shape mismatch: {te_anchor_513.shape}")
    anchor_oof_253 = np.load(DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy").astype(np.float64)
    if anchor_oof_253.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} OOF shape mismatch: {anchor_oof_253.shape}")
    rae_anchor = float(rae(y_unb, anchor_oof_253))
    print(f"[anchor] {ANCHOR}_pred_oof RAE = {rae_anchor:.4f}")
    print(f"[anchor] te_{ANCHOR}  mean={te_anchor_513.mean():.4f}  "
          f"std={te_anchor_513.std():.4f}  "
          f"min={te_anchor_513.min():.4f}  max={te_anchor_513.max():.4f}")

    # ---- Cached ChEMBL features (513) ----
    if not PRED_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(f"pred_chembl_pec50_513 missing: {PRED_CHEMBL_513_PATH}")
    if not SIM_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(f"sim_chembl_513 missing: {SIM_CHEMBL_513_PATH}")
    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    if pred_chembl_513.shape[0] != n_test or sim_chembl_513.shape[0] != n_test:
        raise ValueError(
            f"ChEMBL feature shape mismatch: "
            f"pred {pred_chembl_513.shape}, sim {sim_chembl_513.shape}"
        )
    print(f"[load] pred_chembl_pec50_513  mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[load] sim_chembl_513         mean={sim_chembl_513.mean():.3f}  "
          f"std={sim_chembl_513.std():.3f}")

    # ---- Residual target on 253 ----
    residual_unb = y_unb - anchor_oof_253
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Build per-component pruned matrices (unb + 513) ----
    per_component_meta: dict[str, dict] = {}
    comp_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for comp in COMPONENTS:
        print("\n" + "-" * 78)
        print(f"BUILD pruned matrices for {comp}")
        print("-" * 78)
        X_unb_p, X_te_p, meta = _build_component_matrices(
            comp, n_test, unb_idx, pred_chembl_513, sim_chembl_513
        )
        per_component_meta[comp] = meta
        comp_matrices[comp] = (X_unb_p, X_te_p)
        print(f"   {comp}: src={meta['src']}  top_k={meta['top_k']}  "
              f"feat_dim_pruned={meta['feat_dim_pruned']}  "
              f"X_unb={X_unb_p.shape}  X_te={X_te_p.shape}")

    # ---- Fit 75 deploy models: (component x outer x inner) -> resid_513 ----
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)
    n_comp = len(COMPONENTS)
    n_total = n_outer * n_inner * n_comp
    assert n_total == 75, f"expected 75, got {n_total}"

    stack_resid_513 = np.zeros((n_total, n_test), dtype=np.float64)
    per_model_records: list[dict] = []

    model_idx = 0
    for comp in COMPONENTS:
        X_unb_p, X_te_p = comp_matrices[comp]
        print("\n" + "-" * 78)
        print(f"COMPONENT {comp}  --  fit 5 outer x 5 inner = 25 deploy models")
        print("-" * 78)
        for oi, o in enumerate(OUTER_SEEDS):
            inner_seeds = [o * 1000 + s for s in INNER_BASE_SEEDS]
            for ii, isd in enumerate(inner_seeds):
                mdl = LGBMRegressor(**_lgbm_params(isd))
                mdl.fit(X_unb_p, residual_unb)
                # In-sample 253
                resid_in = mdl.predict(X_unb_p)
                corr_in = anchor_oof_253 + resid_in
                in_rae = float(rae(y_unb, corr_in))
                # Deploy 513
                resid_513 = mdl.predict(X_te_p)
                stack_resid_513[model_idx] = resid_513
                per_model_records.append({
                    "model_idx": int(model_idx),
                    "component": comp,
                    "outer_seed": int(o),
                    "inner_seed": int(isd),
                    "in_sample_rae_253": in_rae,
                    "resid_513_mean": float(resid_513.mean()),
                    "resid_513_std": float(resid_513.std()),
                })
                print(f"   [{model_idx:2d}] {comp}  outer={o:3d}  "
                      f"inner={isd:6d}  in_RAE={in_rae:.4f}  "
                      f"resid_513 mean={resid_513.mean():+.4f}  "
                      f"std={resid_513.std():.4f}")
                model_idx += 1
    assert model_idx == n_total

    # ---- Row-level MEDIAN aggregation across 75 vectors ----
    median_resid_513 = np.median(stack_resid_513, axis=0)
    te_nb1453 = te_anchor_513 + median_resid_513

    # ---- In-sample RAE on unblind slice ----
    in_rae_median = float(rae(y_unb, te_nb1453[unb_idx]))

    print("\n" + "=" * 78)
    print("75-MODEL DEPLOY MEDIAN-BAG DIAGNOSTICS")
    print("=" * 78)
    print(f"   median_resid_513  mean={median_resid_513.mean():+.4f}  "
          f"std={median_resid_513.std():.4f}  "
          f"min={median_resid_513.min():+.4f}  max={median_resid_513.max():+.4f}")
    print(f"   te_nb1453         mean={te_nb1453.mean():.4f}  "
          f"std={te_nb1453.std():.4f}  "
          f"min={te_nb1453.min():.4f}  max={te_nb1453.max():.4f}")
    print(f"   in_RAE(unb, te_nb1453[unb_idx]) = {in_rae_median:.4f}   "
          f"(honest LB anchor {HONEST_LB_ANCHOR_MEDIAN})")

    # ---- Save NPY ----
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, te_nb1453.astype(np.float32))
    print(f"\n[save] {te_npy_path}")

    # ---- Save CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1443.csv"
    df = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1453.astype(np.float64),
    })
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}    rows={len(df)}  cols={list(df.columns)}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "parent_method": PARENT,
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "components": COMPONENTS,
        "n_total_fits": int(n_total),
        "per_component_meta": per_component_meta,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070_oof_253": rae_anchor,
        "residual_unb_mean": float(residual_unb.mean()),
        "residual_unb_std": float(residual_unb.std()),
        "per_model_records": per_model_records,
        "te_nb1453_stats": {
            "mean": float(te_nb1453.mean()),
            "std": float(te_nb1453.std()),
            "min": float(te_nb1453.min()),
            "max": float(te_nb1453.max()),
        },
        "median_resid_513_stats": {
            "mean": float(median_resid_513.mean()),
            "std": float(median_resid_513.std()),
            "min": float(median_resid_513.min()),
            "max": float(median_resid_513.max()),
        },
        "in_rae_unb_median": in_rae_median,
        "honest_lb_anchor_median": HONEST_LB_ANCHOR_MEDIAN,
        "te_npy_path": str(te_npy_path),
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
        "n_test", "n_unb", "n_total_fits",
        "rae_anchor_nb1070_oof_253",
        "te_nb1453_stats",
        "median_resid_513_stats",
        "in_rae_unb_median",
        "honest_lb_anchor_median",
        "csv_path",
        "te_npy_path",
    ):
        print(f"  {k}: {res.get(k)}")
