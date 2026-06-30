"""nb1470 -- Deploy nb1460 (PRE-unblind chemprop_aux anchor + AtomPair-pruned
+ ChEMBL residual learner) to a 513-row submission CSV.

Protocol:
    1.  Anchor = te_chemprop_aux.npy (513,) -- PRE-unblind.
    2.  Build features for ALL 513 test rows:
            - top-30 AtomPair bits (from nb1460/nb1373 SHAP ranking).
            - pred_chembl_pec50_513 + sim_chembl_513 (cached from nb1242).
            = 32 cols.
    3.  Fit 5-seed bag shallow LGBM Huber (depth=3, leaves=7, n_est=80,
        lr=0.05, min_child=20, alpha=1.0) on ALL 253 unblind rows with
        residual target = y_unb - te_chemprop_aux[unb_idx].
    4.  Predict residual on the 513 test rows; mean-bag across seeds.
    5.  te_nb1470 = te_chemprop_aux + mean_bag_residual_513.
    6.  Save data/processed/te_nb1470.npy and
        submissions/nb1470_deploy_nb1460.csv
        (513 rows, SMILES + Molecule Name + pEC50).

Honest LB anchor 0.5550 (PRE-unblind cross-fit -> predicted LB ~0.558).
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

TAG = "nb1470"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"
NB1460_SUMMARY = DATA_PROCESSED / "nb1460_summary.json"

SEEDS = [0, 1, 7, 42, 137]
TE_OUT = DATA_PROCESSED / f"te_{TAG}.npy"

SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUB_OUT = SUB_DIR / f"{TAG}_deploy_nb1460.csv"


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


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Deploy nb1460 (PRE-unblind chemprop_aux anchor + "
          f"AtomPair-30 + ChEMBL) to 513 CSV")
    print(f"       seeds = {SEEDS}")
    print("=" * 78)

    # ---- Load anchor (513) + truth slice (253) ----
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
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
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

    # ---- Top-30 AtomPair bit indices (from nb1460 summary) ----
    with open(NB1460_SUMMARY) as f:
        nb1460 = json.load(f)
    top_bit_idx = np.asarray(nb1460["top_atompair_bit_indices_ranked"],
                             dtype=int)
    print(f"[bits] top-{len(top_bit_idx)} AtomPair bit indices loaded "
          f"from nb1460_summary.json")

    # ---- Build 32-col features on 513 + 253 slice ----
    X_ap_te_513 = np.load(ATOMPAIR_TE_PATH)
    if X_ap_te_513.shape[0] != n_test:
        raise ValueError(
            f"AtomPair cache shape mismatch: {X_ap_te_513.shape}"
        )
    n_ap = int(X_ap_te_513.shape[1])
    print(f"[ap]   te_atompair: shape={X_ap_te_513.shape}")

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

    X_ap_pruned_513 = X_ap_te_513[:, top_bit_idx].astype(np.float32)
    X_513 = np.concatenate(
        [
            X_ap_pruned_513,
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat] X_513 shape = {X_513.shape}  (top-30 AP + chembl_pred + sim)")

    X_unb = X_513[unb_idx]
    print(f"[feat] X_unb shape = {X_unb.shape}")

    # ---- 5-seed bag fit on ALL 253; predict residual on 513 ----
    print("-" * 78)
    print(f"5-SEED BAG TRAIN ON {n_unb} UNB; PREDICT RESIDUAL ON {n_test}")
    print("-" * 78)
    per_seed_resid_513 = np.zeros((len(SEEDS), n_test), dtype=np.float64)
    per_seed_resid_unb_insample = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    for i, s in enumerate(SEEDS):
        mdl = LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb, residual_unb)
        r_513 = mdl.predict(X_513)
        r_unb_in = mdl.predict(X_unb)
        per_seed_resid_513[i] = r_513
        per_seed_resid_unb_insample[i] = r_unb_in
        print(f"   seed {s:3d}:  resid_513 mean={r_513.mean():+.4f}  "
              f"std={r_513.std():.4f}   |  "
              f"resid_unb_insample std={r_unb_in.std():.4f}")

    mean_bag_resid_513 = per_seed_resid_513.mean(axis=0)
    mean_bag_resid_unb_in = per_seed_resid_unb_insample.mean(axis=0)
    print(f"\n[bag] mean_bag_resid_513:  mean={mean_bag_resid_513.mean():+.4f}  "
          f"std={mean_bag_resid_513.std():.4f}")

    # ---- Compose deploy te ----
    te_nb1470 = te_anchor_513 + mean_bag_resid_513
    print(f"[te]  te_{TAG}: mean={te_nb1470.mean():.4f}  "
          f"std={te_nb1470.std():.4f}  "
          f"min={te_nb1470.min():.4f}  max={te_nb1470.max():.4f}")

    # In-sample RAE on the 253 unblind rows (deploy-on-all in_RAE, NOT cross-fit)
    in_RAE_unb = float(rae(y_unb, te_nb1470[unb_idx]))
    print(f"[in]  in_RAE on unblind (deploy-on-all, in-sample) = "
          f"{in_RAE_unb:.4f}   (cross-fit anchor 0.5550 from nb1460)")

    # ---- Save te + submission CSV ----
    np.save(TE_OUT, te_nb1470.astype(np.float32))
    print(f"[save] {TE_OUT}")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "SMILES": te_df[smiles_col].astype(str).to_numpy(),
        "Molecule Name": te_df[name_col].astype(str).to_numpy(),
        "pEC50": te_nb1470.astype(np.float64),
    })
    if len(out) != n_test:
        raise ValueError(f"CSV row count mismatch: {len(out)} vs {n_test}")
    out.to_csv(SUB_OUT, index=False)
    print(f"[save] {SUB_OUT}  (rows={len(out)}, cols={list(out.columns)})")

    summary = {
        "tag": TAG,
        "anchor": "chemprop_aux",
        "anchor_kind": "PRE_unblind_te",
        "seeds": SEEDS,
        "n_test": n_test,
        "n_unb": n_unb,
        "n_features": int(X_513.shape[1]),
        "top_atompair_bit_indices_ranked":
            [int(b) for b in top_bit_idx.tolist()],
        "rae_anchor_chemprop_aux_in_RAE_unb": rae_anchor,
        "te_mean": float(te_nb1470.mean()),
        "te_std": float(te_nb1470.std()),
        "te_min": float(te_nb1470.min()),
        "te_max": float(te_nb1470.max()),
        "in_RAE_unb_deploy_in_sample": in_RAE_unb,
        "honest_crossfit_RAE_nb1460": float(nb1460["rae_mean_bag"]),
        "predicted_LB": float(nb1460["rae_mean_bag"]) + 0.003,
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
        "n_test", "n_unb", "n_features",
        "rae_anchor_chemprop_aux_in_RAE_unb",
        "te_mean", "te_std", "te_min", "te_max",
        "in_RAE_unb_deploy_in_sample",
        "honest_crossfit_RAE_nb1460", "predicted_LB",
        "te_path", "csv_path",
    ):
        print(f"  {k}: {res.get(k)}")
