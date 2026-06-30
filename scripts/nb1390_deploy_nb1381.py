"""nb1390 -- DEPLOY of nb1381 BoB (outer-bagged nb1373) to 513.

Per task spec:
    For 5 outer seeds {0, 1, 7, 42, 137}:
        inner_seeds(o) = [o*1000 + s for s in {0, 1, 7, 42, 137}]
        for each inner seed:
            fit shallow LGBM Huber on ALL 253 unblind rows
            (top-30 AtomPair + ChEMBL pred + sim = 32-col PRUNED matrix),
            target = y_unb - nb1070_pred_oof,
            predict residual on full 513.
        per_outer_mean[o] = mean of 5 inner deploy resid_513 vectors
    BoB MEAN  = mean of 5 per-outer-mean vectors
    BoB MEDIAN = median of 5 per-outer-mean vectors

    te_nb1390_mean   = te_nb1070 + BoB_mean_resid_513
    te_nb1390_median = te_nb1070 + BoB_median_resid_513

Honest LB anchors (from nb1381 cross-fit):
    * BoB MEAN   0.5095
    * BoB MEDIAN 0.5088

Outputs:
    data/processed/te_nb1390_mean.npy                       (513,) float32
    data/processed/te_nb1390_median.npy                     (513,) float32
    data/processed/nb1390_per_outer_mean_resid_513.npy      (5, 513) float32
    data/processed/nb1390_seed_resid_513.npy                (25, 513) float32
    data/processed/nb1390_summary.json
    submissions/nb1390_deploy_nb1381_mean.csv               (513 rows)
    submissions/nb1390_deploy_nb1381_median.csv             (513 rows)
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

TAG = "nb1390"
ANCHOR = "nb1070"
PARENT = "nb1381"
GRANDPARENT = "nb1373"

INNER_BASE_SEEDS = [0, 1, 7, 42, 137]
OUTER_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"          # (513, 2048) uint8
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

HONEST_LB_ANCHOR_MEAN = 0.5095
HONEST_LB_ANCHOR_MEDIAN = 0.5088


def _lgbm_params(seed: int) -> dict:
    # EXACT match to nb1381 / nb1373 / nb1380 shallow Huber config.
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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1381 BoB (outer-bagged nb1373) -> 513")
    print(f"          anchor={ANCHOR}  parent={PARENT}  grandparent={GRANDPARENT}")
    print(f"          outer seeds = {OUTER_SEEDS}")
    print(f"          inner base seeds = {INNER_BASE_SEEDS}")
    print(f"          inner_seeds(o) = [o*1000 + s for s in base]")
    print(f"          deploy fit (NO KFold) on all 253 unblind per inner seed")
    print(f"          honest LB anchors: mean={HONEST_LB_ANCHOR_MEAN}  "
          f"median={HONEST_LB_ANCHOR_MEDIAN}")
    print("=" * 78)

    # ---- Top-30 AtomPair bit indices from nb1373 / nb1381 summary ----
    # (nb1381 reuses nb1373 verbatim; either summary works -- use nb1373 to be safe.)
    parent_summary_path = DATA_PROCESSED / f"{GRANDPARENT}_summary.json"
    with open(parent_summary_path) as f:
        parent_summary = json.load(f)
    top_bit_idx_ranked = np.array(
        parent_summary["top_atompair_bit_indices_ranked"], dtype=int
    )
    top_k = int(parent_summary["top_k_atompair"])
    if len(top_bit_idx_ranked) != top_k:
        raise ValueError(
            f"{GRANDPARENT} top-{top_k} indices mismatch: "
            f"got {len(top_bit_idx_ranked)}"
        )
    print(f"[load] {GRANDPARENT} top-{top_k} AtomPair bits (ranked) = "
          f"{top_bit_idx_ranked.tolist()}")

    # ---- Sanity: confirm nb1381 reused these same bits ----
    nb1381_summary_path = DATA_PROCESSED / f"{PARENT}_summary.json"
    if nb1381_summary_path.exists():
        with open(nb1381_summary_path) as f:
            p81 = json.load(f)
        p81_bits = np.array(p81.get("top_atompair_bit_indices_ranked", []), dtype=int)
        if p81_bits.shape == top_bit_idx_ranked.shape \
           and np.array_equal(p81_bits, top_bit_idx_ranked):
            print(f"[check] nb1381 top-k bits match nb1373 verbatim")
        else:
            print(f"[WARN] nb1381 bit list differs from nb1373; "
                  f"using nb1373 ({top_k} bits) as canonical")

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

    # ---- Anchor (513) and anchor-OOF (253) ----
    te_anchor_513 = np.load(
        DATA_PROCESSED / f"te_{ANCHOR}.npy"
    ).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"te_{ANCHOR} shape mismatch: {te_anchor_513.shape}")
    anchor_oof_253 = np.load(
        DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    ).astype(np.float64)
    if anchor_oof_253.shape[0] != n_unb:
        raise ValueError(f"{ANCHOR} OOF shape mismatch: {anchor_oof_253.shape}")
    rae_anchor = float(rae(y_unb, anchor_oof_253))
    print(f"[anchor] {ANCHOR}_pred_oof RAE = {rae_anchor:.4f}")
    print(f"[anchor] te_{ANCHOR}  mean={te_anchor_513.mean():.4f}  "
          f"std={te_anchor_513.std():.4f}  "
          f"min={te_anchor_513.min():.4f}  max={te_anchor_513.max():.4f}")

    # ---- AtomPair-2048 cache (513) ----
    if not ATOMPAIR_TE_PATH.exists():
        raise FileNotFoundError(f"AtomPair test cache missing: {ATOMPAIR_TE_PATH}")
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    if X_ap_te.shape[0] != n_test:
        raise ValueError(f"AtomPair cache shape mismatch: {X_ap_te.shape}")
    n_ap = int(X_ap_te.shape[1])
    print(f"[load] AtomPair cache shape = {X_ap_te.shape}  (n_bits={n_ap})")
    X_ap_te_pruned = X_ap_te[:, top_bit_idx_ranked].astype(np.float32)
    print(f"       AtomPair pruned (513) shape = {X_ap_te_pruned.shape}  "
          f"density = {X_ap_te_pruned.mean():.4f}")

    # ---- Cached ChEMBL features (513) ----
    if not PRED_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(
            f"pred_chembl_pec50_513 missing: {PRED_CHEMBL_513_PATH}"
        )
    if not SIM_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(
            f"sim_chembl_513 missing: {SIM_CHEMBL_513_PATH}"
        )
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

    # ---- Build PRUNED 32-col matrices ----
    X_te_pruned = np.concatenate(
        [
            X_ap_te_pruned,
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat] X_te_pruned (513) shape = {X_te_pruned.shape}")

    X_ap_unb_pruned = X_ap_te[unb_idx][:, top_bit_idx_ranked].astype(np.float32)
    pred_chembl_unb = pred_chembl_513[unb_idx]
    sim_chembl_unb = sim_chembl_513[unb_idx]
    X_unb_pruned = np.concatenate(
        [
            X_ap_unb_pruned,
            pred_chembl_unb.reshape(-1, 1),
            sim_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat] X_unb_pruned (253) shape = {X_unb_pruned.shape}")

    feat_dim_pruned = int(X_te_pruned.shape[1])

    # ---- Residual target on 253 ----
    residual_unb = y_unb - anchor_oof_253
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Outer-bag deploy loop ----
    print("\n" + "-" * 78)
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)
    n_fits = n_outer * n_inner
    print(f"OUTER-BAG DEPLOY LOOP "
          f"({n_outer} outer x {n_inner} inner = {n_fits} deploy fits)")
    print("-" * 78)

    seed_resid_513 = np.zeros((n_fits, n_test), dtype=np.float64)
    per_outer_mean_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_inner_seeds: list[list[int]] = []
    per_outer_inner_in_rae: list[list[float]] = []
    per_outer_inner_resid_stats: list[list[dict]] = []
    per_outer_mean_in_rae: list[float] = []
    per_outer_mean_resid_stats: list[dict] = []
    flat_seed_records: list[dict] = []

    fit_i = 0
    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [o * 1000 + s for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds.append([int(s) for s in inner_seeds])
        print(f"\n   outer seed {o}:  inner seeds = {inner_seeds}")
        inner_resid_513 = np.zeros((n_inner, n_test), dtype=np.float64)
        inner_in_rae_list: list[float] = []
        inner_resid_stats_list: list[dict] = []
        for ii, s in enumerate(inner_seeds):
            mdl = LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb_pruned, residual_unb)
            # In-sample residual on 253
            resid_in = mdl.predict(X_unb_pruned)
            corr_in = anchor_oof_253 + resid_in
            in_rae_s = float(rae(y_unb, corr_in))
            # Predict residual on 513
            resid_513 = mdl.predict(X_te_pruned)
            inner_resid_513[ii] = resid_513
            seed_resid_513[fit_i] = resid_513
            stats_s = {
                "outer_seed": int(o),
                "inner_seed": int(s),
                "in_sample_rae_253": in_rae_s,
                "resid_513_mean": float(resid_513.mean()),
                "resid_513_std": float(resid_513.std()),
                "resid_513_min": float(resid_513.min()),
                "resid_513_max": float(resid_513.max()),
            }
            inner_in_rae_list.append(in_rae_s)
            inner_resid_stats_list.append(stats_s)
            flat_seed_records.append(stats_s)
            print(f"      inner seed {s:7d}:  in-sample RAE_253 = {in_rae_s:.4f}  "
                  f"resid_513 mean={resid_513.mean():+.4f}  "
                  f"std={resid_513.std():.4f}")
            fit_i += 1
        per_outer_inner_in_rae.append(inner_in_rae_list)
        per_outer_inner_resid_stats.append(inner_resid_stats_list)

        # Per-outer: mean-bag of 5 inner deploy resid_513 vectors
        po_mean_resid = inner_resid_513.mean(axis=0)
        per_outer_mean_resid_513[oi] = po_mean_resid
        po_corr_unb = te_anchor_513[unb_idx] + po_mean_resid[unb_idx]
        po_in_rae = float(rae(y_unb, po_corr_unb))
        per_outer_mean_in_rae.append(po_in_rae)
        per_outer_mean_resid_stats.append({
            "outer_seed": int(o),
            "mean_resid_513_mean": float(po_mean_resid.mean()),
            "mean_resid_513_std": float(po_mean_resid.std()),
            "mean_resid_513_min": float(po_mean_resid.min()),
            "mean_resid_513_max": float(po_mean_resid.max()),
            "in_sample_rae_253_on_per_outer_mean": po_in_rae,
        })
        print(f"      [per-outer]  mean_resid_513 mean={po_mean_resid.mean():+.4f}  "
              f"std={po_mean_resid.std():.4f}  in_RAE_253={po_in_rae:.4f}")

    # ---- BoB row-level aggregation (over 5 per-outer-mean vectors) ----
    bob_mean_resid_513 = per_outer_mean_resid_513.mean(axis=0)
    bob_median_resid_513 = np.median(per_outer_mean_resid_513, axis=0)
    te_nb1390_mean = te_anchor_513 + bob_mean_resid_513
    te_nb1390_median = te_anchor_513 + bob_median_resid_513

    # ---- In-sample RAE on unblind slice (deploy fit -> in-sample) ----
    in_rae_mean = float(rae(y_unb, te_nb1390_mean[unb_idx]))
    in_rae_median = float(rae(y_unb, te_nb1390_median[unb_idx]))

    print("\n" + "-" * 78)
    print("513-ROW DEPLOY VECTOR DIAGNOSTICS")
    print("-" * 78)
    print(f"   bob_mean_resid_513    mean={bob_mean_resid_513.mean():+.4f}  "
          f"std={bob_mean_resid_513.std():.4f}  "
          f"min={bob_mean_resid_513.min():+.4f}  max={bob_mean_resid_513.max():+.4f}")
    print(f"   bob_median_resid_513  mean={bob_median_resid_513.mean():+.4f}  "
          f"std={bob_median_resid_513.std():.4f}  "
          f"min={bob_median_resid_513.min():+.4f}  max={bob_median_resid_513.max():+.4f}")
    print(f"   te_nb1390_mean    mean={te_nb1390_mean.mean():.4f}  "
          f"std={te_nb1390_mean.std():.4f}  "
          f"min={te_nb1390_mean.min():.4f}  max={te_nb1390_mean.max():.4f}")
    print(f"   te_nb1390_median  mean={te_nb1390_median.mean():.4f}  "
          f"std={te_nb1390_median.std():.4f}  "
          f"min={te_nb1390_median.min():.4f}  max={te_nb1390_median.max():.4f}")
    print(f"   in_RAE(unb, mean)   = {in_rae_mean:.4f}   "
          f"(honest LB anchor {HONEST_LB_ANCHOR_MEAN})")
    print(f"   in_RAE(unb, median) = {in_rae_median:.4f}   "
          f"(honest LB anchor {HONEST_LB_ANCHOR_MEDIAN})")

    # ---- Save NPY ----
    te_mean_path = DATA_PROCESSED / f"te_{TAG}_mean.npy"
    te_median_path = DATA_PROCESSED / f"te_{TAG}_median.npy"
    np.save(te_mean_path, te_nb1390_mean.astype(np.float32))
    np.save(te_median_path, te_nb1390_median.astype(np.float32))
    per_outer_path = DATA_PROCESSED / f"{TAG}_per_outer_mean_resid_513.npy"
    seeds_resid_path = DATA_PROCESSED / f"{TAG}_seed_resid_513.npy"
    np.save(per_outer_path, per_outer_mean_resid_513.astype(np.float32))
    np.save(seeds_resid_path, seed_resid_513.astype(np.float32))
    print(f"\n[save] {te_mean_path}")
    print(f"[save] {te_median_path}")
    print(f"[save] {per_outer_path}     shape={per_outer_mean_resid_513.shape}")
    print(f"[save] {seeds_resid_path}   shape={seed_resid_513.shape}")

    # ---- Save CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    mean_csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1381_mean.csv"
    median_csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1381_median.csv"
    df_mean = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1390_mean.astype(np.float64),
    })
    df_median = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1390_median.astype(np.float64),
    })
    df_mean.to_csv(mean_csv_path, index=False)
    df_median.to_csv(median_csv_path, index=False)
    print(f"[save] {mean_csv_path}    rows={len(df_mean)}  "
          f"cols={list(df_mean.columns)}")
    print(f"[save] {median_csv_path}  rows={len(df_median)}  "
          f"cols={list(df_median.columns)}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "parent_method": PARENT,
        "grandparent_method": GRANDPARENT,
        "top_atompair_bit_indices_ranked": top_bit_idx_ranked.tolist(),
        "feat_dim_pruned": feat_dim_pruned,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "n_atompair_bits": int(n_ap),
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds": per_outer_inner_seeds,
        "n_outer": int(n_outer),
        "n_inner": int(n_inner),
        "n_total_fits": int(n_fits),
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070_oof_253": rae_anchor,
        "residual_unb_mean": float(residual_unb.mean()),
        "residual_unb_std": float(residual_unb.std()),
        "per_outer_inner_in_rae_253": per_outer_inner_in_rae,
        "per_outer_inner_resid_stats": per_outer_inner_resid_stats,
        "per_outer_mean_in_rae_253": per_outer_mean_in_rae,
        "per_outer_mean_resid_stats": per_outer_mean_resid_stats,
        "flat_seed_records": flat_seed_records,
        "te_nb1390_mean_stats": {
            "mean": float(te_nb1390_mean.mean()),
            "std": float(te_nb1390_mean.std()),
            "min": float(te_nb1390_mean.min()),
            "max": float(te_nb1390_mean.max()),
        },
        "te_nb1390_median_stats": {
            "mean": float(te_nb1390_median.mean()),
            "std": float(te_nb1390_median.std()),
            "min": float(te_nb1390_median.min()),
            "max": float(te_nb1390_median.max()),
        },
        "bob_mean_resid_513_stats": {
            "mean": float(bob_mean_resid_513.mean()),
            "std": float(bob_mean_resid_513.std()),
            "min": float(bob_mean_resid_513.min()),
            "max": float(bob_mean_resid_513.max()),
        },
        "bob_median_resid_513_stats": {
            "mean": float(bob_median_resid_513.mean()),
            "std": float(bob_median_resid_513.std()),
            "min": float(bob_median_resid_513.min()),
            "max": float(bob_median_resid_513.max()),
        },
        "in_rae_unb_mean": in_rae_mean,
        "in_rae_unb_median": in_rae_median,
        "honest_lb_anchor_mean": HONEST_LB_ANCHOR_MEAN,
        "honest_lb_anchor_median": HONEST_LB_ANCHOR_MEDIAN,
        "te_mean_npy_path": str(te_mean_path),
        "te_median_npy_path": str(te_median_path),
        "per_outer_npy_path": str(per_outer_path),
        "seed_resid_npy_path": str(seeds_resid_path),
        "mean_csv_path": str(mean_csv_path),
        "median_csv_path": str(median_csv_path),
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
        "n_test", "n_unb", "feat_dim_pruned",
        "n_outer", "n_inner", "n_total_fits",
        "rae_anchor_nb1070_oof_253",
        "per_outer_mean_in_rae_253",
        "te_nb1390_mean_stats",
        "te_nb1390_median_stats",
        "in_rae_unb_mean",
        "in_rae_unb_median",
        "honest_lb_anchor_mean",
        "honest_lb_anchor_median",
        "mean_csv_path",
        "median_csv_path",
        "te_mean_npy_path",
        "te_median_npy_path",
    ):
        print(f"  {k}: {res.get(k)}")
