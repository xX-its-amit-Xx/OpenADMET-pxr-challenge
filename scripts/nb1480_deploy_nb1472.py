"""nb1480 -- Deploy nb1472 (PRE-unblind 3-way SHAP-pruned residual blend)
to a 513-row submission CSV.

Anchor    : te_chemprop_aux.npy (513,) -- PRE-unblind.
Residuals : three SHAP-pruned families (top-K indices loaded from nb1472):
              A AtomPair-30 + ChEMBL  (32 cols)
              B MACCS-20  + ChEMBL    (22 cols)
              C Mordred-30 + ChEMBL   (32 cols)
            Each: 5-seed bag shallow LGBM Huber trained on ALL 253 unblind
            with target = y_unb - chemprop_aux[unb_idx]; predict 513.
Blend     : te_nb1480 = chemprop_aux + (resid_AP + resid_MACCS + resid_Mordred) / 3

Honest LB anchor 0.5330 (PRE-unblind cross-fit -> predicted LB ~0.536).
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

TAG = "nb1480"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
MORDRED_TE_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_test.npy")
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"
NB1472_SUMMARY = DATA_PROCESSED / "nb1472_summary.json"

SEEDS = [0, 1, 7, 42, 137]
TE_OUT = DATA_PROCESSED / f"te_{TAG}.npy"

SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUB_OUT = SUB_DIR / f"{TAG}_deploy_nb1472.csv"


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


def _load_family_te(family: str, n_test: int) -> np.ndarray:
    if family == "AtomPair":
        X = np.load(ATOMPAIR_TE_PATH)
    elif family == "MACCS":
        X = np.load(MACCS_TE_PATH)
    elif family == "Mordred":
        X = np.load(MORDRED_TE_PATH).astype(np.float32)
        # NaN-impute via column median (mirror nb1472)
        X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
        col_med = np.nanmedian(X, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
        bad = ~np.isfinite(X)
        if bad.any():
            idx_r, idx_c = np.where(bad)
            X[idx_r, idx_c] = col_med[idx_c]
        return X
    else:
        raise ValueError(family)
    if X.shape[0] != n_test:
        raise ValueError(f"{family} cache shape mismatch: {X.shape}")
    return X.astype(np.float32)


def _build_family_X513(family: str, X_fam_te: np.ndarray,
                       top_idx: np.ndarray,
                       pred_chembl: np.ndarray,
                       sim_chembl: np.ndarray) -> np.ndarray:
    X_fam_pruned = X_fam_te[:, top_idx].astype(np.float32)
    X = np.concatenate(
        [
            X_fam_pruned,
            pred_chembl.reshape(-1, 1),
            sim_chembl.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat][{family}] X_513 shape = {X.shape}")
    return X


def _bag_fit_predict(family: str, X_513: np.ndarray, X_unb: np.ndarray,
                     residual_unb: np.ndarray, n_test: int, n_unb: int):
    print("-" * 78)
    print(f"FAMILY = {family}  5-SEED BAG TRAIN ON {n_unb} UNB; "
          f"PREDICT RESID ON {n_test}")
    print("-" * 78)
    per_seed_resid_513 = np.zeros((len(SEEDS), n_test), dtype=np.float64)
    per_seed_resid_unb_in = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    for i, s in enumerate(SEEDS):
        mdl = LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb, residual_unb)
        r_513 = mdl.predict(X_513)
        r_unb = mdl.predict(X_unb)
        per_seed_resid_513[i] = r_513
        per_seed_resid_unb_in[i] = r_unb
        print(f"   seed {s:3d}:  resid_513 mean={r_513.mean():+.4f}  "
              f"std={r_513.std():.4f}   |  resid_unb_insample "
              f"std={r_unb.std():.4f}")
    mean_bag_513 = per_seed_resid_513.mean(axis=0)
    print(f"   [bag {family}] mean_bag_resid_513: "
          f"mean={mean_bag_513.mean():+.4f}  std={mean_bag_513.std():.4f}")
    return mean_bag_513


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Deploy nb1472 PRE-unblind 3-way naive 1/3 mean to 513 CSV")
    print(f"       anchor = chemprop_aux   seeds = {SEEDS}")
    print("=" * 78)

    # ---- Load test + truth slice ----
    te_df = load_test()
    n_test = len(te_df)
    print(f"[load] load_test() -> {n_test} rows")

    smiles_col = "smiles" if "smiles" in te_df.columns else "SMILES"
    name_col = None
    for cand in ("molecule_name", "Molecule Name", "name", "compound_id"):
        if cand in te_df.columns:
            name_col = cand
            break
    if name_col is None:
        raise KeyError(
            f"No Molecule-Name column found; cols = {list(te_df.columns)}"
        )
    print(f"[load] smiles col = {smiles_col!r}   name col = {name_col!r}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape != (n_test,):
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape}"
        )
    print(f"[load] te_chemprop_aux: shape={te_anchor_513.shape}  "
          f"mean={te_anchor_513.mean():.4f}  std={te_anchor_513.std():.4f}")

    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] unb_idx: {unb_idx.shape}   y_unb: {y_unb.shape}")

    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] chemprop_aux in_RAE (unb) = {rae_anchor:.4f}")
    residual_unb = y_unb - anchor_unb
    print(f"[resid] mean={residual_unb.mean():+.4f}  "
          f"std={residual_unb.std():.4f}")

    # ---- nb1472 SHAP-pruned indices per family ----
    with open(NB1472_SUMMARY) as f:
        nb1472 = json.load(f)
    fam_top_idx = {}
    for fam_entry in nb1472["families"]:
        fam_top_idx[fam_entry["family"]] = np.asarray(
            fam_entry["top_idx_ranked"], dtype=int
        )
        print(f"[bits][{fam_entry['family']}] top-{len(fam_top_idx[fam_entry['family']])} "
              f"indices loaded (n_fam_bits={fam_entry['n_fam_bits']}, "
              f"crossfit RAE={fam_entry['rae_mean_bag']:.4f})")

    # ---- ChEMBL kNN cached features for 513 ----
    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    if pred_chembl_513.shape != (n_test,) or sim_chembl_513.shape != (n_test,):
        raise ValueError(
            f"chembl cache shapes mismatch: pred={pred_chembl_513.shape}, "
            f"sim={sim_chembl_513.shape}"
        )
    print(f"[chembl] pred_chembl_513: mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[chembl] sim_chembl_513:  mean={sim_chembl_513.mean():.3f}  "
          f"std={sim_chembl_513.std():.3f}")

    # ---- Per-family build, fit, deploy ----
    residual_deploys = []
    per_family_diag = {}
    for family in ("AtomPair", "MACCS", "Mordred"):
        X_fam_te = _load_family_te(family, n_test)
        X_513 = _build_family_X513(
            family, X_fam_te, fam_top_idx[family],
            pred_chembl_513, sim_chembl_513
        )
        X_unb = X_513[unb_idx]
        print(f"[feat][{family}] X_unb shape = {X_unb.shape}")
        mean_bag_resid_513 = _bag_fit_predict(
            family, X_513, X_unb, residual_unb, n_test, n_unb
        )
        residual_deploys.append(mean_bag_resid_513)
        # per-family in-sample diagnostic on unblind
        te_fam = te_anchor_513 + mean_bag_resid_513
        per_family_diag[family] = {
            "in_RAE_unb_deploy_in_sample": float(rae(y_unb, te_fam[unb_idx])),
            "resid_513_mean": float(mean_bag_resid_513.mean()),
            "resid_513_std": float(mean_bag_resid_513.std()),
        }
        print(f"[diag][{family}] te_family in_RAE on unblind = "
              f"{per_family_diag[family]['in_RAE_unb_deploy_in_sample']:.4f}")

    # ---- Naive 1/3 mean blend ----
    print("\n" + "=" * 78)
    print("NAIVE 1/3 MEAN BLEND ACROSS 3 FAMILY RESIDUALS")
    print("=" * 78)
    resid_stack = np.stack(residual_deploys, axis=0)  # (3, 513)
    mean_resid_513 = resid_stack.mean(axis=0)
    print(f"[blend] mean_resid_513: mean={mean_resid_513.mean():+.4f}  "
          f"std={mean_resid_513.std():.4f}")

    te_nb1480 = te_anchor_513 + mean_resid_513
    print(f"[te]  te_{TAG}: mean={te_nb1480.mean():.4f}  "
          f"std={te_nb1480.std():.4f}  "
          f"min={te_nb1480.min():.4f}  max={te_nb1480.max():.4f}")

    in_RAE_unb = float(rae(y_unb, te_nb1480[unb_idx]))
    print(f"[in]  in_RAE on unblind (deploy-on-all, in-sample) = "
          f"{in_RAE_unb:.4f}   (cross-fit anchor 0.5330 from nb1472)")

    # ---- Save te + submission CSV ----
    np.save(TE_OUT, te_nb1480.astype(np.float32))
    print(f"[save] {TE_OUT}")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "SMILES": te_df[smiles_col].astype(str).to_numpy(),
        "Molecule Name": te_df[name_col].astype(str).to_numpy(),
        "pEC50": te_nb1480.astype(np.float64),
    })
    if len(out) != n_test:
        raise ValueError(f"CSV row count mismatch: {len(out)} vs {n_test}")
    out.to_csv(SUB_OUT, index=False)
    print(f"[save] {SUB_OUT}  (rows={len(out)}, cols={list(out.columns)})")

    summary = {
        "tag": TAG,
        "anchor": "chemprop_aux",
        "anchor_kind": "PRE_unblind_te",
        "deploy_of": "nb1472",
        "seeds": SEEDS,
        "n_test": n_test,
        "n_unb": n_unb,
        "rae_anchor_chemprop_aux_in_RAE_unb": rae_anchor,
        "per_family_top_k": {
            "AtomPair": len(fam_top_idx["AtomPair"]),
            "MACCS": len(fam_top_idx["MACCS"]),
            "Mordred": len(fam_top_idx["Mordred"]),
        },
        "per_family_diag": per_family_diag,
        "per_family_crossfit_RAE_from_nb1472": nb1472["per_family_mean_bag_rae"],
        "te_mean": float(te_nb1480.mean()),
        "te_std": float(te_nb1480.std()),
        "te_min": float(te_nb1480.min()),
        "te_max": float(te_nb1480.max()),
        "in_RAE_unb_deploy_in_sample": in_RAE_unb,
        "honest_crossfit_RAE_nb1472": float(nb1472["rae_mean_blend"]),
        "predicted_LB": float(nb1472["rae_mean_blend"]) + 0.003,
        "te_path": str(TE_OUT),
        "csv_path": str(SUB_OUT),
        "wall_sec": round(time.time() - t0, 2),
    }
    sum_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {sum_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_test", "n_unb",
        "rae_anchor_chemprop_aux_in_RAE_unb",
        "per_family_top_k",
        "per_family_diag",
        "per_family_crossfit_RAE_from_nb1472",
        "te_mean", "te_std", "te_min", "te_max",
        "in_RAE_unb_deploy_in_sample",
        "honest_crossfit_RAE_nb1472", "predicted_LB",
        "te_path", "csv_path",
    ):
        print(f"  {k}: {res.get(k)}")
