"""nb1481 -- DEPLOY of nb1471 (chemprop embed + AtomPair blend w=0.5) to 513.

Builds the deploy 513-row te for nb1471:
    te_nb1481 = 0.5 * te_nb1462_deploy + 0.5 * te_nb1380_mean

Where:
    te_nb1462_deploy:  5-seed shallow LGBM Huber on top-30 chemprop embed
                       dims + pred_chembl + sim (32 cols) over residual
                       y_unb - nb1070_pred_oof. Fit on ALL 253 unblind rows
                       (deploy fit, no KFold), predict residual on 513,
                       add to te_nb1070.

    te_nb1380_mean:    already exists from nb1380 deploy of nb1373
                       (SHAP-pruned AtomPair top-30 + ChEMBL).

Honest LB anchor 0.4995 (POST-unblind, uncertain transfer).

Outputs:
    data/processed/te_nb1462.npy             (513,) float32
    data/processed/te_nb1481.npy             (513,) float32
    submissions/nb1481_deploy_nb1471.csv     (513 rows: SMILES, Molecule Name, pEC50)
    data/processed/nb1481_summary.json
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

TAG = "nb1481"
ANCHOR = "nb1070"
PARENT_EMB = "nb1462"
PARENT_AP = "nb1380"   # deploy mean-bag for nb1373

RESID_SEEDS = [0, 1, 7, 42, 137]
BLEND_W_EMB = 0.5  # weight for chemprop embed component
BLEND_W_AP = 0.5   # weight for AtomPair component

ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb1070.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"
NB1462_SUMMARY = DATA_PROCESSED / "nb1462_summary.json"
TE_NB1380_MEAN_PATH = DATA_PROCESSED / "te_nb1380_mean.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

TE_NB1462_OUT = DATA_PROCESSED / "te_nb1462.npy"
TE_OUT = DATA_PROCESSED / f"te_{TAG}.npy"

SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUB_OUT = SUB_DIR / f"{TAG}_deploy_nb1471.csv"


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
    print(f"{TAG} -- Deploy nb1471 (chemprop embed + AtomPair blend w=0.5) "
          f"to 513 CSV")
    print(f"       blend: 0.5 * te_nb1462_deploy + 0.5 * te_nb1380_mean")
    print(f"       seeds = {RESID_SEEDS}")
    print("=" * 78)

    # ---- Load test frame + anchor + unblind truth ----
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
            f"te_nb1070 shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    print(f"[load] te_nb1070: shape={te_anchor_513.shape}  "
          f"mean={te_anchor_513.mean():.4f}  std={te_anchor_513.std():.4f}")

    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] unb_idx: {unb_idx.shape}   y_unb: {y_unb.shape}")

    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] nb1070 in_RAE (unb) = {rae_anchor:.4f}")
    residual_unb = y_unb - anchor_unb
    print(f"[resid] mean={residual_unb.mean():+.4f}  "
          f"std={residual_unb.std():.4f}")

    # ---- Load top-30 chemprop embed dim indices from nb1462 summary ----
    with open(NB1462_SUMMARY) as f:
        nb1462 = json.load(f)
    top_dim_idx = np.asarray(nb1462["top_embed_dim_indices_ranked"], dtype=int)
    print(f"[dims] top-{len(top_dim_idx)} chemprop embed dim indices loaded "
          f"from nb1462_summary.json")

    # ---- Build 32-col feature matrix on 513 (and slice to 253) ----
    X_embed_te_513 = np.load(CHEMPROP_EMBED_TE_PATH)
    if X_embed_te_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop embed cache shape mismatch: {X_embed_te_513.shape}"
        )
    n_embed = int(X_embed_te_513.shape[1])
    print(f"[emb]  te_chemprop_embed_300: shape={X_embed_te_513.shape}")

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

    X_embed_pruned_513 = X_embed_te_513[:, top_dim_idx].astype(np.float32)
    X_513 = np.concatenate(
        [
            X_embed_pruned_513,
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    print(f"[feat] X_513 shape = {X_513.shape}  "
          f"(top-30 chemprop embed + chembl_pred + sim)")

    X_unb = X_513[unb_idx]
    print(f"[feat] X_unb shape = {X_unb.shape}")

    # ---- 5-seed bag fit on ALL 253; predict residual on 513 ----
    print("-" * 78)
    print(f"5-SEED BAG TRAIN ON {n_unb} UNB; PREDICT RESIDUAL ON {n_test}")
    print("-" * 78)
    per_seed_resid_513 = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_resid_unb_in = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    for i, s in enumerate(RESID_SEEDS):
        mdl = LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb, residual_unb)
        r_513 = mdl.predict(X_513)
        r_unb_in = mdl.predict(X_unb)
        per_seed_resid_513[i] = r_513
        per_seed_resid_unb_in[i] = r_unb_in
        print(f"   seed {s:3d}:  resid_513 mean={r_513.mean():+.4f}  "
              f"std={r_513.std():.4f}   |  "
              f"resid_unb_in std={r_unb_in.std():.4f}")

    mean_bag_resid_513 = per_seed_resid_513.mean(axis=0)
    print(f"\n[bag] mean_bag_resid_513:  mean={mean_bag_resid_513.mean():+.4f}  "
          f"std={mean_bag_resid_513.std():.4f}")

    # ---- te_nb1462_deploy = anchor + mean_bag_resid_513 ----
    te_nb1462_deploy = te_anchor_513 + mean_bag_resid_513
    print(f"[te]  te_nb1462_deploy: mean={te_nb1462_deploy.mean():.4f}  "
          f"std={te_nb1462_deploy.std():.4f}  "
          f"min={te_nb1462_deploy.min():.4f}  max={te_nb1462_deploy.max():.4f}")

    in_RAE_nb1462 = float(rae(y_unb, te_nb1462_deploy[unb_idx]))
    print(f"[in]  in_RAE nb1462_deploy on unblind (in-sample) = "
          f"{in_RAE_nb1462:.4f}   (cross-fit anchor 0.5144 from nb1462)")

    np.save(TE_NB1462_OUT, te_nb1462_deploy.astype(np.float32))
    print(f"[save] {TE_NB1462_OUT}")

    # ---- Load te_nb1380_mean (deploy of nb1373) ----
    te_nb1380_mean = np.load(TE_NB1380_MEAN_PATH).astype(np.float64)
    if te_nb1380_mean.shape != (n_test,):
        raise ValueError(
            f"te_nb1380_mean shape mismatch: {te_nb1380_mean.shape} vs {n_test}"
        )
    print(f"[load] te_nb1380_mean: shape={te_nb1380_mean.shape}  "
          f"mean={te_nb1380_mean.mean():.4f}  std={te_nb1380_mean.std():.4f}")
    in_RAE_nb1380 = float(rae(y_unb, te_nb1380_mean[unb_idx]))
    print(f"[in]  in_RAE nb1380_mean on unblind (in-sample) = "
          f"{in_RAE_nb1380:.4f}")

    # ---- Blend ----
    te_nb1481 = (
        BLEND_W_EMB * te_nb1462_deploy
        + BLEND_W_AP * te_nb1380_mean
    )
    te_mean = float(te_nb1481.mean())
    te_std = float(te_nb1481.std())
    te_min = float(te_nb1481.min())
    te_max = float(te_nb1481.max())
    print(f"\n[blend] te_nb1481 = {BLEND_W_EMB} * te_nb1462_deploy + "
          f"{BLEND_W_AP} * te_nb1380_mean")
    print(f"[te]  te_nb1481: mean={te_mean:.4f}  std={te_std:.4f}  "
          f"min={te_min:.4f}  max={te_max:.4f}")

    in_RAE_blend = float(rae(y_unb, te_nb1481[unb_idx]))
    print(f"[in]  in_RAE nb1481 blend on unblind (in-sample) = "
          f"{in_RAE_blend:.4f}   (honest LB anchor 0.4995 POST-unblind, "
          f"uncertain transfer)")

    # ---- Save te + submission CSV ----
    np.save(TE_OUT, te_nb1481.astype(np.float32))
    print(f"[save] {TE_OUT}")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "SMILES": te_df[smiles_col].astype(str).to_numpy(),
        "Molecule Name": te_df[name_col].astype(str).to_numpy(),
        "pEC50": te_nb1481.astype(np.float64),
    })
    if len(out) != n_test:
        raise ValueError(f"CSV row count mismatch: {len(out)} vs {n_test}")
    out.to_csv(SUB_OUT, index=False)
    print(f"[save] {SUB_OUT}  (rows={len(out)}, cols={list(out.columns)})")

    summary = {
        "tag": TAG,
        "parent_blend": "nb1471",
        "components": {
            "nb1462_deploy": "chemprop embed top-30 + ChEMBL residual (LGBM Huber 5-seed)",
            "nb1380_mean": "AtomPair-30 SHAP + ChEMBL residual mean-bag (nb1373 deploy)",
        },
        "anchor": ANCHOR,
        "seeds": RESID_SEEDS,
        "blend_w_emb": BLEND_W_EMB,
        "blend_w_ap": BLEND_W_AP,
        "n_test": n_test,
        "n_unb": n_unb,
        "n_features_nb1462": int(X_513.shape[1]),
        "top_embed_dim_indices_ranked":
            [int(b) for b in top_dim_idx.tolist()],
        "rae_anchor_nb1070_in_RAE_unb": rae_anchor,
        "te_nb1462_in_RAE_unb": in_RAE_nb1462,
        "te_nb1380_mean_in_RAE_unb": in_RAE_nb1380,
        "te_mean": te_mean,
        "te_std": te_std,
        "te_min": te_min,
        "te_max": te_max,
        "in_RAE_unb_blend_in_sample": in_RAE_blend,
        "honest_crossfit_RAE_nb1471": 0.4995,
        "te_nb1462_path": str(TE_NB1462_OUT),
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
        "n_test", "n_unb", "n_features_nb1462",
        "blend_w_emb", "blend_w_ap",
        "rae_anchor_nb1070_in_RAE_unb",
        "te_nb1462_in_RAE_unb",
        "te_nb1380_mean_in_RAE_unb",
        "te_mean", "te_std", "te_min", "te_max",
        "in_RAE_unb_blend_in_sample",
        "honest_crossfit_RAE_nb1471",
        "te_nb1462_path", "te_path", "csv_path",
    ):
        print(f"  {k}: {res.get(k)}")
