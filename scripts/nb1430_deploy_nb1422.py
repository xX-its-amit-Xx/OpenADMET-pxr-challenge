"""nb1430 -- Outer-bag DEPLOY of nb1422 (BoB MEAN + MEDIAN of 3-way nb1411 blend) to 513.

Mirrors the nb1422 outer-bag VALIDATION protocol, but on the 513-row deploy
fits (not 253-row OOF). For each OUTER seed o in {0,1,7,42,137}:

    inner_seeds(o) = [o*1000 + s for s in {0,1,7,42,137}]

    nb1373_o deploy: top-30 AtomPair SHAP-pruned + ChEMBL kNN features,
                     5-inner-seed bag, each inner fit on ALL 253 unblind,
                     predict residual on 513, mean over 5 inner -> (513,)
    nb1352_o deploy: top-20 MACCS SHAP-pruned + ChEMBL kNN features,
                     same 5-inner-seed bag protocol -> (513,)
    nb1364_o deploy: top-30 Mordred SHAP-pruned + ChEMBL kNN features,
                     same 5-inner-seed bag protocol -> (513,)

    blend_o_deploy_513 = (nb1373_o + nb1352_o + nb1364_o) / 3        (513,)

Stack 5 outer blend_o vectors -> (5, 513). Row-level BoB:
    BoB_mean_residual_513   = (5, 513).mean(axis=0)   -> (513,)
    BoB_median_residual_513 = np.median((5, 513), axis=0) -> (513,)

(Equivalently, since each blend_o already includes te_nb1070 once, the row-mean
across 5 outer-seed deploy blend_o vectors equals te_nb1070 + mean over the 5
outer-seed residual-blend vectors. We compute on residuals directly to keep
the +anchor step explicit.)

    te_nb1430_mean   = te_nb1070 + BoB_mean_residual_513
    te_nb1430_median = te_nb1070 + BoB_median_residual_513

Honest LB anchors (from nb1422 cross-fit on 253):
    * BoB MEAN   pooled RAE = 0.5022
    * BoB MEDIAN pooled RAE = 0.5016

Outputs:
    data/processed/te_nb1430_mean.npy                  (513,) float32
    data/processed/te_nb1430_median.npy                (513,) float32
    data/processed/nb1430_per_outer_blend_513.npy      (5, 513) float32
    data/processed/nb1430_per_outer_nb1373_513.npy     (5, 513) float32
    data/processed/nb1430_per_outer_nb1352_513.npy     (5, 513) float32
    data/processed/nb1430_per_outer_nb1364_513.npy     (5, 513) float32
    data/processed/nb1430_summary.json
    submissions/nb1430_deploy_nb1422_mean.csv          (513 rows: SMILES,Molecule Name,pEC50)
    submissions/nb1430_deploy_nb1422_median.csv        (513 rows: SMILES,Molecule Name,pEC50)
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

TAG = "nb1430"
ANCHOR = "nb1070"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

HONEST_LB_ANCHOR_MEAN = 0.5022
HONEST_LB_ANCHOR_MEDIAN = 0.5016


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


def _build_component_pruned(
    component_tag: str,
    top_indices: np.ndarray,
    X_full_te: np.ndarray,
    pred_chembl_513: np.ndarray,
    sim_chembl_513: np.ndarray,
    unb_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build pruned feature matrices on 513 (X_te_pruned) and on 253 (X_unb_pruned)
    for a given component family using its top-k feature indices.

    The pruned matrix is [top-k cols | pred_chembl | sim_chembl].
    """
    X_te_pruned_base = X_full_te[:, top_indices].astype(np.float32)
    X_te_pruned = np.concatenate(
        [
            X_te_pruned_base,
            pred_chembl_513.reshape(-1, 1).astype(np.float32),
            sim_chembl_513.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    X_unb_pruned_base = X_full_te[unb_idx][:, top_indices].astype(np.float32)
    pred_chembl_unb = pred_chembl_513[unb_idx].astype(np.float32)
    sim_chembl_unb = sim_chembl_513[unb_idx].astype(np.float32)
    X_unb_pruned = np.concatenate(
        [
            X_unb_pruned_base,
            pred_chembl_unb.reshape(-1, 1),
            sim_chembl_unb.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"   [{component_tag}] X_te_pruned  shape = {X_te_pruned.shape}")
    print(f"   [{component_tag}] X_unb_pruned shape = {X_unb_pruned.shape}")
    return X_te_pruned, X_unb_pruned


def _inner_bag_mean_resid_513(
    X_unb_pruned: np.ndarray,
    X_te_pruned: np.ndarray,
    residual_unb: np.ndarray,
    anchor_oof_253: np.ndarray,
    y_unb: np.ndarray,
    inner_seeds: list[int],
) -> tuple[np.ndarray, list[float]]:
    """Inner 5-seed bag. Each seed fits LGBM on all 253 unblind rows
    (residual target), predicts residual on 513. Mean over 5 inner seeds.

    Returns (mean_resid_513, inner_in_sample_rae_list).
    """
    n_test = X_te_pruned.shape[0]
    n_inner = len(inner_seeds)
    inner_resid_513 = np.zeros((n_inner, n_test), dtype=np.float64)
    inner_in_sample_rae: list[float] = []
    for ii, isd in enumerate(inner_seeds):
        mdl = LGBMRegressor(**_lgbm_params(isd))
        mdl.fit(X_unb_pruned, residual_unb)
        resid_in = mdl.predict(X_unb_pruned)
        corr_in = anchor_oof_253 + resid_in
        in_rae = float(rae(y_unb, corr_in))
        inner_in_sample_rae.append(in_rae)
        resid_513 = mdl.predict(X_te_pruned)
        inner_resid_513[ii] = resid_513
    mean_resid_513 = inner_resid_513.mean(axis=0)
    return mean_resid_513, inner_in_sample_rae


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY outer-bag nb1422 (BoB MEAN + MEDIAN of 3-way nb1411 blend) -> 513")
    print(f"          anchor       = {ANCHOR}")
    print(f"          outer seeds  = {OUTER_SEEDS}")
    print(f"          inner base   = {INNER_BASE_SEEDS}")
    print(f"          inner(o)     = [o*1000 + s for s in base]")
    print(f"          honest LB anchors: mean={HONEST_LB_ANCHOR_MEAN}  "
          f"median={HONEST_LB_ANCHOR_MEDIAN}")
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

    # ---- Component A: nb1373 (top-30 AtomPair + ChEMBL) ----
    with open(DATA_PROCESSED / "nb1373_summary.json") as f:
        nb1373_summary = json.load(f)
    top_ap_idx = np.array(nb1373_summary["top_atompair_bit_indices_ranked"], dtype=int)
    print(f"[load] nb1373 top-{len(top_ap_idx)} AtomPair bits = "
          f"{top_ap_idx[:10].tolist()} ...")

    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    if X_ap_te.shape[0] != n_test:
        raise ValueError(f"AtomPair cache shape mismatch: {X_ap_te.shape}")
    print(f"[load] AtomPair cache shape = {X_ap_te.shape}")
    X_te_pruned_A, X_unb_pruned_A = _build_component_pruned(
        "nb1373", top_ap_idx, X_ap_te.astype(np.float32),
        pred_chembl_513, sim_chembl_513, unb_idx,
    )

    # ---- Component B: nb1352 (top-20 MACCS + ChEMBL) ----
    with open(DATA_PROCESSED / "nb1352_summary.json") as f:
        nb1352_summary = json.load(f)
    top_maccs_idx = np.array(nb1352_summary["top_maccs_bit_indices_ranked"], dtype=int)
    print(f"[load] nb1352 top-{len(top_maccs_idx)} MACCS bits = "
          f"{top_maccs_idx.tolist()}")

    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    print(f"[load] MACCS cache shape = {X_maccs_te.shape}")
    X_te_pruned_B, X_unb_pruned_B = _build_component_pruned(
        "nb1352", top_maccs_idx, X_maccs_te.astype(np.float32),
        pred_chembl_513, sim_chembl_513, unb_idx,
    )

    # ---- Component C: nb1364 (top-30 Mordred + ChEMBL) ----
    with open(DATA_PROCESSED / "nb1364_summary.json") as f:
        nb1364_summary = json.load(f)
    top_mord_idx = np.array(nb1364_summary["top_mordred_col_indices_ranked"], dtype=int)
    print(f"[load] nb1364 top-{len(top_mord_idx)} Mordred cols = "
          f"{top_mord_idx[:10].tolist()} ...")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    print(f"[load] Mordred cache shape = {X_mord_te.shape}")
    X_te_pruned_C, X_unb_pruned_C = _build_component_pruned(
        "nb1364", top_mord_idx, X_mord_te,
        pred_chembl_513, sim_chembl_513, unb_idx,
    )

    # ---- Residual target on 253 ----
    residual_unb = y_unb - anchor_oof_253
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- OUTER loop: 5 outer seeds, 5 inner each per component, 3 components ----
    print("\n" + "-" * 78)
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)
    print(f"DEPLOY OUTER x INNER BAG  ({n_outer} outer x {n_inner} inner x 3 components "
          f"= {n_outer*n_inner*3} fits)")
    print("-" * 78)

    per_outer_nb1373_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_nb1352_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_nb1364_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_blend_513 = np.zeros((n_outer, n_test), dtype=np.float64)

    per_outer_records: list[dict] = []
    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [o * 1000 + s for s in INNER_BASE_SEEDS]
        print(f"\n   outer {o:3d}   inner_seeds = {inner_seeds}")

        # nb1373_o: AtomPair top-30
        mean_resid_A, inner_rae_A = _inner_bag_mean_resid_513(
            X_unb_pruned_A, X_te_pruned_A, residual_unb,
            anchor_oof_253, y_unb, inner_seeds,
        )
        nb1373_o_513 = te_anchor_513 + mean_resid_A
        per_outer_nb1373_513[oi] = nb1373_o_513
        in_rae_A = float(rae(y_unb, nb1373_o_513[unb_idx]))
        print(f"     [nb1373_o] inner per-seed in-RAE = "
              f"[{', '.join(f'{r:.4f}' for r in inner_rae_A)}]   "
              f"mean-bag in-RAE_253 = {in_rae_A:.4f}")

        # nb1352_o: MACCS top-20
        mean_resid_B, inner_rae_B = _inner_bag_mean_resid_513(
            X_unb_pruned_B, X_te_pruned_B, residual_unb,
            anchor_oof_253, y_unb, inner_seeds,
        )
        nb1352_o_513 = te_anchor_513 + mean_resid_B
        per_outer_nb1352_513[oi] = nb1352_o_513
        in_rae_B = float(rae(y_unb, nb1352_o_513[unb_idx]))
        print(f"     [nb1352_o] inner per-seed in-RAE = "
              f"[{', '.join(f'{r:.4f}' for r in inner_rae_B)}]   "
              f"mean-bag in-RAE_253 = {in_rae_B:.4f}")

        # nb1364_o: Mordred top-30
        mean_resid_C, inner_rae_C = _inner_bag_mean_resid_513(
            X_unb_pruned_C, X_te_pruned_C, residual_unb,
            anchor_oof_253, y_unb, inner_seeds,
        )
        nb1364_o_513 = te_anchor_513 + mean_resid_C
        per_outer_nb1364_513[oi] = nb1364_o_513
        in_rae_C = float(rae(y_unb, nb1364_o_513[unb_idx]))
        print(f"     [nb1364_o] inner per-seed in-RAE = "
              f"[{', '.join(f'{r:.4f}' for r in inner_rae_C)}]   "
              f"mean-bag in-RAE_253 = {in_rae_C:.4f}")

        # blend_o_deploy = (nb1373_o + nb1352_o + nb1364_o) / 3
        blend_o = (nb1373_o_513 + nb1352_o_513 + nb1364_o_513) / 3.0
        per_outer_blend_513[oi] = blend_o
        in_rae_blend_o = float(rae(y_unb, blend_o[unb_idx]))
        print(f"     [blend_o]  in-RAE_253 = {in_rae_blend_o:.4f}")

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": [int(s) for s in inner_seeds],
            "nb1373_inner_in_sample_rae": inner_rae_A,
            "nb1352_inner_in_sample_rae": inner_rae_B,
            "nb1364_inner_in_sample_rae": inner_rae_C,
            "nb1373_in_sample_rae_253": in_rae_A,
            "nb1352_in_sample_rae_253": in_rae_B,
            "nb1364_in_sample_rae_253": in_rae_C,
            "blend_in_sample_rae_253": in_rae_blend_o,
        })

    # ---- Row-level BoB mean/median across 5 outer deploy blend vectors ----
    # Equivalently in residual space:
    #   blend_o_513 = te_anchor + (residA_o + residB_o + residC_o)/3
    # so the row-mean/median across outer blends shifts only the residual term.
    per_outer_blend_residual = per_outer_blend_513 - te_anchor_513  # (5, 513)
    bob_mean_residual_513 = per_outer_blend_residual.mean(axis=0)
    bob_median_residual_513 = np.median(per_outer_blend_residual, axis=0)

    te_nb1430_mean = te_anchor_513 + bob_mean_residual_513
    te_nb1430_median = te_anchor_513 + bob_median_residual_513

    # In-sample RAE on unblind slice (deploy fit -> in-sample!)
    in_rae_mean = float(rae(y_unb, te_nb1430_mean[unb_idx]))
    in_rae_median = float(rae(y_unb, te_nb1430_median[unb_idx]))

    print("\n" + "=" * 78)
    print("513-ROW DEPLOY VECTOR DIAGNOSTICS")
    print("=" * 78)
    print(f"   bob_mean_residual_513    mean={bob_mean_residual_513.mean():+.4f}  "
          f"std={bob_mean_residual_513.std():.4f}  "
          f"min={bob_mean_residual_513.min():+.4f}  max={bob_mean_residual_513.max():+.4f}")
    print(f"   bob_median_residual_513  mean={bob_median_residual_513.mean():+.4f}  "
          f"std={bob_median_residual_513.std():.4f}  "
          f"min={bob_median_residual_513.min():+.4f}  max={bob_median_residual_513.max():+.4f}")
    print(f"   te_nb1430_mean    mean={te_nb1430_mean.mean():.4f}  "
          f"std={te_nb1430_mean.std():.4f}  "
          f"min={te_nb1430_mean.min():.4f}  max={te_nb1430_mean.max():.4f}")
    print(f"   te_nb1430_median  mean={te_nb1430_median.mean():.4f}  "
          f"std={te_nb1430_median.std():.4f}  "
          f"min={te_nb1430_median.min():.4f}  max={te_nb1430_median.max():.4f}")
    print(f"   in_RAE(unb, mean)   = {in_rae_mean:.4f}   "
          f"(honest LB anchor {HONEST_LB_ANCHOR_MEAN})")
    print(f"   in_RAE(unb, median) = {in_rae_median:.4f}   "
          f"(honest LB anchor {HONEST_LB_ANCHOR_MEDIAN})")

    # ---- Save NPY artifacts ----
    te_mean_path = DATA_PROCESSED / f"te_{TAG}_mean.npy"
    te_median_path = DATA_PROCESSED / f"te_{TAG}_median.npy"
    np.save(te_mean_path, te_nb1430_mean.astype(np.float32))
    np.save(te_median_path, te_nb1430_median.astype(np.float32))
    print(f"\n[save] {te_mean_path}")
    print(f"[save] {te_median_path}")

    per_outer_paths = {
        "blend": DATA_PROCESSED / f"{TAG}_per_outer_blend_513.npy",
        "nb1373": DATA_PROCESSED / f"{TAG}_per_outer_nb1373_513.npy",
        "nb1352": DATA_PROCESSED / f"{TAG}_per_outer_nb1352_513.npy",
        "nb1364": DATA_PROCESSED / f"{TAG}_per_outer_nb1364_513.npy",
    }
    np.save(per_outer_paths["blend"], per_outer_blend_513.astype(np.float32))
    np.save(per_outer_paths["nb1373"], per_outer_nb1373_513.astype(np.float32))
    np.save(per_outer_paths["nb1352"], per_outer_nb1352_513.astype(np.float32))
    np.save(per_outer_paths["nb1364"], per_outer_nb1364_513.astype(np.float32))
    for k, p in per_outer_paths.items():
        print(f"[save] {p}")

    # ---- Save CSVs ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    mean_csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1422_mean.csv"
    median_csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1422_median.csv"
    df_mean = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1430_mean.astype(np.float64),
    })
    df_median = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1430_median.astype(np.float64),
    })
    df_mean.to_csv(mean_csv_path, index=False)
    df_median.to_csv(median_csv_path, index=False)
    print(f"[save] {mean_csv_path}    rows={len(df_mean)}  cols={list(df_mean.columns)}")
    print(f"[save] {median_csv_path}  rows={len(df_median)}  cols={list(df_median.columns)}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "parent_method": "nb1422",
        "blend_recipe": "BoB (mean / median) over 5 outer seeds of (nb1373 + nb1352 + nb1364)/3",
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "n_outer": int(n_outer),
        "n_inner_per_outer": int(n_inner),
        "n_components": 3,
        "n_total_fits": int(n_outer * n_inner * 3),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "nb1373_top_atompair_bits_count": int(len(top_ap_idx)),
        "nb1352_top_maccs_bits_count": int(len(top_maccs_idx)),
        "nb1364_top_mordred_cols_count": int(len(top_mord_idx)),
        "rae_anchor_nb1070_oof_253": rae_anchor,
        "residual_unb_mean": float(residual_unb.mean()),
        "residual_unb_std": float(residual_unb.std()),
        "per_outer_records": per_outer_records,
        "te_nb1430_mean_stats": {
            "mean": float(te_nb1430_mean.mean()),
            "std": float(te_nb1430_mean.std()),
            "min": float(te_nb1430_mean.min()),
            "max": float(te_nb1430_mean.max()),
        },
        "te_nb1430_median_stats": {
            "mean": float(te_nb1430_median.mean()),
            "std": float(te_nb1430_median.std()),
            "min": float(te_nb1430_median.min()),
            "max": float(te_nb1430_median.max()),
        },
        "bob_mean_residual_513_stats": {
            "mean": float(bob_mean_residual_513.mean()),
            "std": float(bob_mean_residual_513.std()),
            "min": float(bob_mean_residual_513.min()),
            "max": float(bob_mean_residual_513.max()),
        },
        "bob_median_residual_513_stats": {
            "mean": float(bob_median_residual_513.mean()),
            "std": float(bob_median_residual_513.std()),
            "min": float(bob_median_residual_513.min()),
            "max": float(bob_median_residual_513.max()),
        },
        "in_rae_unb_mean": in_rae_mean,
        "in_rae_unb_median": in_rae_median,
        "honest_lb_anchor_mean": HONEST_LB_ANCHOR_MEAN,
        "honest_lb_anchor_median": HONEST_LB_ANCHOR_MEDIAN,
        "te_mean_npy_path": str(te_mean_path),
        "te_median_npy_path": str(te_median_path),
        "per_outer_blend_513_path": str(per_outer_paths["blend"]),
        "per_outer_nb1373_513_path": str(per_outer_paths["nb1373"]),
        "per_outer_nb1352_513_path": str(per_outer_paths["nb1352"]),
        "per_outer_nb1364_513_path": str(per_outer_paths["nb1364"]),
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
        "n_test", "n_unb",
        "n_outer", "n_inner_per_outer", "n_total_fits",
        "rae_anchor_nb1070_oof_253",
        "te_nb1430_mean_stats",
        "te_nb1430_median_stats",
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
