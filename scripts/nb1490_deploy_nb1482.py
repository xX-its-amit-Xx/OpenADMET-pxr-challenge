"""nb1490 -- Deploy nb1482 (PRE-unblind 3-way outer-bag, BoB MEDIAN 0.5310)
to a 513-row submission CSV.

Anchor    : te_chemprop_aux.npy (513,) -- PRE-unblind.
Outer-bag : for each o in OUTER_SEEDS = {0, 1, 7, 42, 137}:
              inner_seeds(o) = [o*1000 + s for s in {0, 1, 7, 42, 137}]
              For each FAMILY in {AtomPair-30, MACCS-20, Mordred-30}:
                top-K cols loaded from nb1472_summary.json (anchor-only).
                5-inner-seed bag of shallow LGBM Huber fit on ALL 253 unblind
                with target = y_unb - chemprop_aux[unb_idx]; predict residual
                on 513.   resid_fam_o_513 = mean across inner seeds.
              blend_o_513 = (resid_AP_o + resid_MACCS_o + resid_Mord_o) / 3
Stack     : 5 per-outer blend vectors (5, 513)
Aggregate : row-level BoB MEAN and BoB MEDIAN across outer axis -> (513,)
Output    : te_nb1490_mean   = te_chemprop_aux + BoB_mean_residual_513
            te_nb1490_median = te_chemprop_aux + BoB_median_residual_513
            submissions/nb1490_deploy_nb1482_mean.csv
            submissions/nb1490_deploy_nb1482_median.csv

Honest LB anchors 0.5319 mean / 0.5310 median (PRE-unblind, predicted LB 0.534).
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

TAG = "nb1490"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
MORDRED_TE_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_test.npy")
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"
NB1472_SUMMARY = DATA_PROCESSED / "nb1472_summary.json"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
FAMILIES = ("AtomPair", "MACCS", "Mordred")

TE_OUT_MEAN = DATA_PROCESSED / f"te_{TAG}_mean.npy"
TE_OUT_MEDIAN = DATA_PROCESSED / f"te_{TAG}_median.npy"

SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUB_OUT_MEAN = SUB_DIR / f"{TAG}_deploy_nb1482_mean.csv"
SUB_OUT_MEDIAN = SUB_DIR / f"{TAG}_deploy_nb1482_median.csv"

HONEST_LB_MEAN = 0.5319
HONEST_LB_MEDIAN = 0.5310


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
    return X


def _inner_bag_resid_513(family: str, X_513: np.ndarray, X_unb: np.ndarray,
                         residual_unb: np.ndarray, inner_seeds: list[int],
                         outer_seed: int, n_test: int) -> np.ndarray:
    """For a single outer seed, run 5 inner-seed shallow LGBM Huber fits on
    ALL 253 unblind with target=residual_unb; predict on 513 and mean across
    inner seeds.  Returns (n_test,) residual vector.
    """
    n_inner = len(inner_seeds)
    per_inner = np.zeros((n_inner, n_test), dtype=np.float64)
    for ii, isd in enumerate(inner_seeds):
        ts = time.time()
        mdl = LGBMRegressor(**_lgbm_params(isd))
        mdl.fit(X_unb, residual_unb)
        r_513 = mdl.predict(X_513)
        per_inner[ii] = r_513
        print(f"      outer {outer_seed:3d}  {family:<9s}  inner seed "
              f"{isd:6d}:  r_513 mean={r_513.mean():+.4f}  std={r_513.std():.4f}  "
              f"wall={time.time() - ts:.1f}s")
    mean_bag_513 = per_inner.mean(axis=0)
    print(f"      outer {outer_seed:3d}  {family:<9s} mean_bag_resid_513: "
          f"mean={mean_bag_513.mean():+.4f}  std={mean_bag_513.std():.4f}")
    return mean_bag_513


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Deploy nb1482 PRE-unblind 3-way outer-bag to 513 CSV")
    print(f"        outer seeds      = {OUTER_SEEDS}")
    print(f"        inner base seeds = {INNER_BASE_SEEDS}")
    print(f"        inner_seeds(o)   = [o*1000 + s for s in base]")
    print(f"        families         = {list(FAMILIES)}")
    print(f"        honest LB anchors: mean={HONEST_LB_MEAN:.4f}  "
          f"median={HONEST_LB_MEDIAN:.4f}")
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

    # ---- Per-family X_513 + X_unb built once (column choice is anchor-only) ----
    fam_X513: dict[str, np.ndarray] = {}
    fam_Xunb: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        X_fam_te = _load_family_te(family, n_test)
        X_513 = _build_family_X513(
            family, X_fam_te, fam_top_idx[family],
            pred_chembl_513, sim_chembl_513
        )
        fam_X513[family] = X_513
        fam_Xunb[family] = X_513[unb_idx]
        print(f"[feat][{family}] X_513={X_513.shape}  X_unb={X_513[unb_idx].shape}")

    # ---- Outer x Family x Inner deploy bag ----
    print("\n" + "=" * 78)
    print("OUTER x FAMILY x INNER LGBM HUBER DEPLOY BAG (TRAIN ON 253 UNB)")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)
    n_fam = len(FAMILIES)

    # Per-outer 3-way blended residual on 513 -> stack (n_outer, n_test)
    outer_blend_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_records: list[dict] = []

    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [int(o * 1000 + s) for s in INNER_BASE_SEEDS]
        t_outer = time.time()
        print("-" * 78)
        print(f"OUTER {o:3d}  inner_seeds = {inner_seeds}")
        print("-" * 78)
        fam_resid_513 = np.zeros((n_fam, n_test), dtype=np.float64)
        per_family_diag: dict[str, dict] = {}
        for fi, family in enumerate(FAMILIES):
            mean_bag_513 = _inner_bag_resid_513(
                family=family,
                X_513=fam_X513[family],
                X_unb=fam_Xunb[family],
                residual_unb=residual_unb,
                inner_seeds=inner_seeds,
                outer_seed=o,
                n_test=n_test,
            )
            fam_resid_513[fi] = mean_bag_513
            te_fam_unb = (te_anchor_513 + mean_bag_513)[unb_idx]
            in_RAE_fam_unb = float(rae(y_unb, te_fam_unb))
            per_family_diag[family] = {
                "in_RAE_unb_deploy_in_sample": in_RAE_fam_unb,
                "resid_513_mean": float(mean_bag_513.mean()),
                "resid_513_std": float(mean_bag_513.std()),
            }
            print(f"      outer {o:3d}  {family:<9s} te_family in_RAE on unblind "
                  f"= {in_RAE_fam_unb:.4f}")

        blend_o_513 = fam_resid_513.mean(axis=0)
        outer_blend_resid_513[oi] = blend_o_513
        te_blend_unb = (te_anchor_513 + blend_o_513)[unb_idx]
        in_RAE_blend_unb = float(rae(y_unb, te_blend_unb))
        print(f"   outer {o:3d}  blend_o_513 (1/3 mean): mean={blend_o_513.mean():+.4f}  "
              f"std={blend_o_513.std():.4f}")
        print(f"   outer {o:3d}  te_blend in_RAE on unblind = "
              f"{in_RAE_blend_unb:.4f}   (outer wall = "
              f"{time.time() - t_outer:.1f}s)")
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "per_family_diag": per_family_diag,
            "blend_in_RAE_unb_deploy_in_sample": in_RAE_blend_unb,
            "blend_resid_513_mean": float(blend_o_513.mean()),
            "blend_resid_513_std": float(blend_o_513.std()),
            "wall_sec": round(time.time() - t_outer, 2),
        })

    # ---- BoB row-level aggregation across outer axis ----
    print("\n" + "=" * 78)
    print("BoB ROW-LEVEL AGGREGATION ACROSS 5 OUTER BLEND VECTORS (513,)")
    print("=" * 78)
    bob_mean_resid_513 = outer_blend_resid_513.mean(axis=0)
    bob_median_resid_513 = np.median(outer_blend_resid_513, axis=0)

    te_nb1490_mean = te_anchor_513 + bob_mean_resid_513
    te_nb1490_median = te_anchor_513 + bob_median_resid_513

    in_RAE_unb_mean = float(rae(y_unb, te_nb1490_mean[unb_idx]))
    in_RAE_unb_median = float(rae(y_unb, te_nb1490_median[unb_idx]))

    print(f"[bob.mean]   resid_513 mean={bob_mean_resid_513.mean():+.4f}  "
          f"std={bob_mean_resid_513.std():.4f}")
    print(f"[bob.median] resid_513 mean={bob_median_resid_513.mean():+.4f}  "
          f"std={bob_median_resid_513.std():.4f}")
    print(f"[te.mean]   mean={te_nb1490_mean.mean():.4f}  "
          f"std={te_nb1490_mean.std():.4f}  "
          f"min={te_nb1490_mean.min():.4f}  max={te_nb1490_mean.max():.4f}")
    print(f"[te.median] mean={te_nb1490_median.mean():.4f}  "
          f"std={te_nb1490_median.std():.4f}  "
          f"min={te_nb1490_median.min():.4f}  max={te_nb1490_median.max():.4f}")
    print(f"[in.mean]   in_RAE on unblind (in-sample) = {in_RAE_unb_mean:.4f}   "
          f"(honest cross-fit anchor {HONEST_LB_MEAN:.4f} from nb1482)")
    print(f"[in.median] in_RAE on unblind (in-sample) = {in_RAE_unb_median:.4f}   "
          f"(honest cross-fit anchor {HONEST_LB_MEDIAN:.4f} from nb1482)")

    # ---- Save te + submission CSVs ----
    np.save(TE_OUT_MEAN, te_nb1490_mean.astype(np.float32))
    np.save(TE_OUT_MEDIAN, te_nb1490_median.astype(np.float32))
    print(f"[save] {TE_OUT_MEAN}")
    print(f"[save] {TE_OUT_MEDIAN}")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    smiles_513 = te_df[smiles_col].astype(str).to_numpy()
    name_513 = te_df[name_col].astype(str).to_numpy()

    out_mean = pd.DataFrame({
        "SMILES": smiles_513,
        "Molecule Name": name_513,
        "pEC50": te_nb1490_mean.astype(np.float64),
    })
    out_median = pd.DataFrame({
        "SMILES": smiles_513,
        "Molecule Name": name_513,
        "pEC50": te_nb1490_median.astype(np.float64),
    })
    if len(out_mean) != n_test or len(out_median) != n_test:
        raise ValueError(
            f"CSV row count mismatch: mean={len(out_mean)} median={len(out_median)} "
            f"expected={n_test}"
        )
    out_mean.to_csv(SUB_OUT_MEAN, index=False)
    out_median.to_csv(SUB_OUT_MEDIAN, index=False)
    print(f"[save] {SUB_OUT_MEAN}   (rows={len(out_mean)}, cols={list(out_mean.columns)})")
    print(f"[save] {SUB_OUT_MEDIAN}  (rows={len(out_median)}, cols={list(out_median.columns)})")

    summary = {
        "tag": TAG,
        "anchor": "chemprop_aux",
        "anchor_kind": "PRE_unblind_te",
        "deploy_of": "nb1482",
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "families": list(FAMILIES),
        "n_test": n_test,
        "n_unb": n_unb,
        "rae_anchor_chemprop_aux_in_RAE_unb": rae_anchor,
        "per_family_top_k": {
            fam: int(len(fam_top_idx[fam])) for fam in FAMILIES
        },
        "per_outer_records": per_outer_records,
        "honest_LB_anchor_mean_from_nb1482": HONEST_LB_MEAN,
        "honest_LB_anchor_median_from_nb1482": HONEST_LB_MEDIAN,
        "predicted_LB_mean": HONEST_LB_MEAN + 0.003,
        "predicted_LB_median": HONEST_LB_MEDIAN + 0.003,
        # mean variant
        "te_mean_mean": float(te_nb1490_mean.mean()),
        "te_mean_std": float(te_nb1490_mean.std()),
        "te_mean_min": float(te_nb1490_mean.min()),
        "te_mean_max": float(te_nb1490_mean.max()),
        "in_RAE_unb_deploy_in_sample_mean": in_RAE_unb_mean,
        "te_path_mean": str(TE_OUT_MEAN),
        "csv_path_mean": str(SUB_OUT_MEAN),
        # median variant
        "te_median_mean": float(te_nb1490_median.mean()),
        "te_median_std": float(te_nb1490_median.std()),
        "te_median_min": float(te_nb1490_median.min()),
        "te_median_max": float(te_nb1490_median.max()),
        "in_RAE_unb_deploy_in_sample_median": in_RAE_unb_median,
        "te_path_median": str(TE_OUT_MEDIAN),
        "csv_path_median": str(SUB_OUT_MEDIAN),
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
        "outer_seeds",
        "te_mean_mean", "te_mean_std", "te_mean_min", "te_mean_max",
        "in_RAE_unb_deploy_in_sample_mean",
        "te_median_mean", "te_median_std", "te_median_min", "te_median_max",
        "in_RAE_unb_deploy_in_sample_median",
        "honest_LB_anchor_mean_from_nb1482",
        "honest_LB_anchor_median_from_nb1482",
        "predicted_LB_mean", "predicted_LB_median",
        "te_path_mean", "csv_path_mean",
        "te_path_median", "csv_path_median",
    ):
        print(f"  {k}: {res.get(k)}")
