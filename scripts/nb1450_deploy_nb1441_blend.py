"""nb1450 -- DEPLOY nb1441 (CatBoost on 82-col 3-way pruned) + 50/50 blend with nb1422 BoB.

PROTOCOL:
    1. Build the 82-col 3-way pruned feature matrix on ALL 513 test rows:
         top-30 AtomPair + top-20 MACCS + top-30 Mordred + pred_chembl + sim
       Reuse cached features:
         data/processed/te_atompair.npy            (513, 2048)
         data/processed/te_maccs.npy               (513, 167)
         C:/pxr_artifacts/nb1030/X_mordred_test.npy (513, ~1600)
         data/processed/pred_chembl_pec50_513.npy  (513,)
         data/processed/sim_chembl_513.npy         (513,)
       Top indices from nb1352 / nb1364 / nb1373 summaries.

    2. Same 82-col matrix sliced on the 253 unblind rows. Residual target =
       y_unb - nb1070_pred_oof.

    3. 5-seed bag CatBoost(loss=MAE, depth=4, n_est=200, lr=0.05, l2=5,
       seeds [0,1,7,42,137]). Each seed fits on ALL 253 unblind (residual
       target), predicts residual on 513. Mean across 5 seeds -> (513,)
       mean_bag_residual_513.

    4. te_nb1441 = te_nb1070 + mean_bag_residual_513.

    5. Load te_nb1430_mean.npy (nb1422 BoB deploy mean).

    6. te_nb1450 = 0.5 * te_nb1441 + 0.5 * te_nb1430_mean.

    7. Save:
         data/processed/te_nb1441.npy                       (513,) float32
         data/processed/te_nb1450.npy                       (513,) float32
         submissions/nb1450_deploy_nb1441_blend.csv         (513 rows: SMILES, Molecule Name, pEC50)
         data/processed/nb1450_summary.json

Honest LB anchor: 0.4990 (nb1441 BoB 50/50 cross-fit on 253).
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
from catboost import CatBoostRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1450"
ANCHOR = "nb1070"

SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1364_SUMMARY = DATA_PROCESSED / "nb1364_summary.json"
NB1373_SUMMARY = DATA_PROCESSED / "nb1373_summary.json"

TE_NB1430_MEAN_PATH = DATA_PROCESSED / "te_nb1430_mean.npy"

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

HONEST_LB_ANCHOR = 0.4990


def _cat_params(seed: int) -> dict:
    return dict(
        loss_function="MAE",
        depth=4,
        iterations=200,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1441 (CatBoost 82-col 3-way pruned) + 50/50 blend with nb1422 BoB")
    print(f"          anchor       = {ANCHOR}")
    print(f"          seeds        = {SEEDS}")
    print(f"          honest LB anchor = {HONEST_LB_ANCHOR}")
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
            f"ChEMBL feature shape mismatch: pred {pred_chembl_513.shape}, "
            f"sim {sim_chembl_513.shape}"
        )
    print(f"[load] pred_chembl_pec50_513 mean={pred_chembl_513.mean():.3f} "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[load] sim_chembl_513        mean={sim_chembl_513.mean():.3f} "
          f"std={sim_chembl_513.std():.3f}")

    # ---- Load SHAP-picked feature indices ----
    for p in (NB1352_SUMMARY, NB1364_SUMMARY, NB1373_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1364_SUMMARY) as f:
        sum_1364 = json.load(f)
    with open(NB1373_SUMMARY) as f:
        sum_1373 = json.load(f)
    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    top_mord_col_idx = np.array(sum_1364["top_mordred_col_indices_ranked"], dtype=int)
    top_ap_bit_idx = np.array(sum_1373["top_atompair_bit_indices_ranked"], dtype=int)
    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    print(f"[reuse] top-{n_top_maccs} MACCS  top-{n_top_mord} Mordred  "
          f"top-{n_top_ap} AtomPair")

    # ---- Load full feature caches (513) ----
    X_maccs_te = np.load(MACCS_TE_PATH).astype(np.float32)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(f"MACCS cache shape mismatch: {X_maccs_te.shape}")
    X_ap_te = np.load(ATOMPAIR_TE_PATH).astype(np.float32)
    if X_ap_te.shape[0] != n_test:
        raise ValueError(f"AtomPair cache shape mismatch: {X_ap_te.shape}")
    X_mord_te = _load_mordred_test(n_test_expected=n_test)

    # ---- Build TRIPLE-PRUNED 82-col feature matrix on 513 ----
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx]
    X_mord_te_top = X_mord_te[:, top_mord_col_idx]
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx]
    X_te_82 = np.concatenate(
        [
            X_maccs_te_top,
            X_mord_te_top,
            X_ap_te_top,
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_te_82.shape[1]
    expected_dim = n_top_maccs + n_top_mord + n_top_ap + 2
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    print(f"[feat] X_te_82 shape  = {X_te_82.shape}")

    # ---- Slice to 253 unblind rows ----
    X_unb_82 = X_te_82[unb_idx].astype(np.float32)
    print(f"[feat] X_unb_82 shape = {X_unb_82.shape}")

    # ---- Residual target on 253 ----
    residual_unb = y_unb - anchor_oof_253
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- 5-seed bag CatBoost: fit on ALL 253 unblind, predict on 513 ----
    print("\n" + "-" * 78)
    print(f"DEPLOY 5-SEED CATBOOST RESIDUAL BAG  (dim={feat_dim})")
    print("-" * 78)
    per_seed_resid_513 = np.zeros((len(SEEDS), n_test), dtype=np.float64)
    per_seed_in_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(SEEDS):
        ts = time.time()
        mdl = CatBoostRegressor(**_cat_params(s))
        mdl.fit(X_unb_82, residual_unb)
        # in-sample residual on 253 -> in-sample RAE check
        resid_in = mdl.predict(X_unb_82)
        corr_in = anchor_oof_253 + resid_in
        in_rae_s = float(rae(y_unb, corr_in))
        per_seed_in_rae.append(in_rae_s)
        # predict residual on 513
        resid_513 = mdl.predict(X_te_82)
        per_seed_resid_513[i] = resid_513
        per_seed_records.append({
            "seed": int(s),
            "in_sample_rae_253": in_rae_s,
            "resid_513_mean": float(resid_513.mean()),
            "resid_513_std": float(resid_513.std()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed {s:3d}: in-RAE_253 = {in_rae_s:.4f}  "
              f"resid_513 mean={resid_513.mean():+.4f} std={resid_513.std():.4f}  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_residual_513 = per_seed_resid_513.mean(axis=0)
    print(f"\n   mean_bag_residual_513  mean={mean_bag_residual_513.mean():+.4f}  "
          f"std={mean_bag_residual_513.std():.4f}  "
          f"min={mean_bag_residual_513.min():+.4f}  max={mean_bag_residual_513.max():+.4f}")

    # ---- te_nb1441 = te_nb1070 + mean_bag_residual_513 ----
    te_nb1441 = te_anchor_513 + mean_bag_residual_513
    in_rae_nb1441 = float(rae(y_unb, te_nb1441[unb_idx]))

    # ---- Load nb1430 mean deploy ----
    if not TE_NB1430_MEAN_PATH.exists():
        raise FileNotFoundError(f"missing {TE_NB1430_MEAN_PATH}")
    te_nb1430_mean = np.load(TE_NB1430_MEAN_PATH).astype(np.float64)
    if te_nb1430_mean.shape[0] != n_test:
        raise ValueError(f"te_nb1430_mean shape mismatch: {te_nb1430_mean.shape}")
    in_rae_nb1430 = float(rae(y_unb, te_nb1430_mean[unb_idx]))

    # ---- 50/50 blend ----
    te_nb1450 = 0.5 * te_nb1441 + 0.5 * te_nb1430_mean
    in_rae_nb1450 = float(rae(y_unb, te_nb1450[unb_idx]))

    print("\n" + "=" * 78)
    print("513-ROW DEPLOY VECTOR DIAGNOSTICS")
    print("=" * 78)
    print(f"   te_nb1441   mean={te_nb1441.mean():.4f}  std={te_nb1441.std():.4f}  "
          f"min={te_nb1441.min():.4f}  max={te_nb1441.max():.4f}")
    print(f"   te_nb1430m  mean={te_nb1430_mean.mean():.4f}  std={te_nb1430_mean.std():.4f}  "
          f"min={te_nb1430_mean.min():.4f}  max={te_nb1430_mean.max():.4f}")
    print(f"   te_nb1450   mean={te_nb1450.mean():.4f}  std={te_nb1450.std():.4f}  "
          f"min={te_nb1450.min():.4f}  max={te_nb1450.max():.4f}")
    print(f"   in_RAE(unb, te_nb1441)  = {in_rae_nb1441:.4f}")
    print(f"   in_RAE(unb, te_nb1430m) = {in_rae_nb1430:.4f}")
    print(f"   in_RAE(unb, te_nb1450)  = {in_rae_nb1450:.4f}   "
          f"(honest LB anchor {HONEST_LB_ANCHOR})")

    # ---- Save NPY artifacts ----
    te_nb1441_path = DATA_PROCESSED / "te_nb1441.npy"
    te_nb1450_path = DATA_PROCESSED / "te_nb1450.npy"
    np.save(te_nb1441_path, te_nb1441.astype(np.float32))
    np.save(te_nb1450_path, te_nb1450.astype(np.float32))
    print(f"\n[save] {te_nb1441_path}")
    print(f"[save] {te_nb1450_path}")

    # ---- Save CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1441_blend.csv"
    df = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1450.astype(np.float64),
    })
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}  rows={len(df)}  cols={list(df.columns)}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "parent_method": "nb1441 + nb1422 BoB",
        "blend_recipe": "0.5 * te_nb1441 + 0.5 * te_nb1430_mean",
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "feat_dim": int(feat_dim),
        "feat_breakdown": {
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "atompair": n_top_ap,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "catboost_loss": "MAE",
        "catboost_depth": 4,
        "catboost_iterations": 200,
        "catboost_learning_rate": 0.05,
        "catboost_l2_leaf_reg": 5.0,
        "rae_anchor_nb1070_oof_253": rae_anchor,
        "residual_unb_mean": float(residual_unb.mean()),
        "residual_unb_std": float(residual_unb.std()),
        "per_seed_records": per_seed_records,
        "per_seed_in_sample_rae": per_seed_in_rae,
        "mean_bag_residual_513_stats": {
            "mean": float(mean_bag_residual_513.mean()),
            "std": float(mean_bag_residual_513.std()),
            "min": float(mean_bag_residual_513.min()),
            "max": float(mean_bag_residual_513.max()),
        },
        "te_nb1441_stats": {
            "mean": float(te_nb1441.mean()),
            "std": float(te_nb1441.std()),
            "min": float(te_nb1441.min()),
            "max": float(te_nb1441.max()),
        },
        "te_nb1430_mean_stats": {
            "mean": float(te_nb1430_mean.mean()),
            "std": float(te_nb1430_mean.std()),
            "min": float(te_nb1430_mean.min()),
            "max": float(te_nb1430_mean.max()),
        },
        "te_nb1450_stats": {
            "mean": float(te_nb1450.mean()),
            "std": float(te_nb1450.std()),
            "min": float(te_nb1450.min()),
            "max": float(te_nb1450.max()),
        },
        "in_rae_unb_nb1441": in_rae_nb1441,
        "in_rae_unb_nb1430_mean": in_rae_nb1430,
        "in_rae_unb_nb1450": in_rae_nb1450,
        "honest_lb_anchor": HONEST_LB_ANCHOR,
        "te_nb1441_path": str(te_nb1441_path),
        "te_nb1450_path": str(te_nb1450_path),
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
        "n_test", "n_unb", "feat_dim", "feat_breakdown",
        "rae_anchor_nb1070_oof_253",
        "per_seed_in_sample_rae",
        "te_nb1441_stats",
        "te_nb1430_mean_stats",
        "te_nb1450_stats",
        "in_rae_unb_nb1441",
        "in_rae_unb_nb1430_mean",
        "in_rae_unb_nb1450",
        "honest_lb_anchor",
        "csv_path",
        "te_nb1441_path",
        "te_nb1450_path",
    ):
        print(f"  {k}: {res.get(k)}")
