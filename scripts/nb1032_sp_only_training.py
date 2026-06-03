"""nb1032 -- Single-concentration ("SP-only") training with crude pEC50 mapping.

Hypothesis: the 21,003-row single-concentration primary screen lives on a
VASTLY different chemistry distribution than the 4,139 CRC training set
(the SP set contains 8,126 compounds not present in any CRC config).
Even a crude log2FC -> pEC50 mapping may inject a new signal axis that the
existing CRC-trained LGBM Huber model (nb972) cannot see.

Crude pEC50 mapping for SP rows:
    pec50 = 4.0 + 0.5 * log2_fc * (-log10(C) / -log10(1e-5))
where C is the assay concentration in molar. The (-log10 C / 5) factor
re-scales log2FC by how informative the assay concentration is (rows
screened near 10 uM map close to 1.0, near 1 mM map close to ~0.6).
Predictions clipped to [2, 8].

Training set:
    - 4,139 CRC compounds (true pec50, weight 1.0)
    - 21,003 SP-only compounds (crude pec50, weight 0.2)
    -> 25,142 rows total

Model: LightGBM Huber alpha=2.0 on combined (Morgan + RDKit) features.
Validation: in_RAE on 253 Phase-1 unblind.

Outputs:
  data/processed/te_nb1032.npy
  data/processed/nb1032_summary.json
  submissions/nb1032_sp_only_training.csv
  (optional) submissions/nb1032_bag_chemprop_aux_nb1032.csv  if Pearson<0.95
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import pearsonr

from pxr.data import load_train, load_test, load_single_conc
from pxr.featurize import combined, impute
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1032"
SEED = 42

PARAMS = dict(
    objective="huber",
    alpha=2.0,
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=4,
)
SP_WEIGHT = 0.2
CRC_WEIGHT = 1.0
PEC50_CLIP = (2.0, 8.0)
NB1014_REF = 0.5994  # bag mean cross-fit RAE proxy


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def sp_crude_pec50(log2fc: np.ndarray, conc_M: np.ndarray) -> np.ndarray:
    """Map (log2_fc, concentration_M) -> crude pEC50.

    pec50 = 4.0 + 0.5 * log2fc * (-log10(C) / -log10(1e-5))
    """
    neg_log_c = -np.log10(np.clip(conc_M, 1e-12, None))
    norm = neg_log_c / 5.0   # -log10(1e-5) = 5
    pec50 = 4.0 + 0.5 * log2fc * norm
    return np.clip(pec50, PEC50_CLIP[0], PEC50_CLIP[1])


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"=== {TAG}: SP-only training with crude pEC50 mapping ===")
    print("=" * 78)

    # ---- Unblind truth ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    print(f"[load] unblind n={len(y_unb)}")

    # ---- Load CRC + SP + test ----
    tr = load_train()
    sp = load_single_conc()
    te = load_test()

    print(f"[load] CRC train rows = {len(tr)}")
    print(f"[load] SP rows         = {len(sp)}")
    print(f"[load] test rows       = {len(te)}")

    # ---- Build SP-only pool (drop SP rows already in CRC by SMILES) ----
    crc_smiles = set(tr["smiles"].astype(str))
    sp_mask = ~sp["smiles"].astype(str).isin(crc_smiles)
    sp_only = sp.loc[sp_mask].copy()
    sp_dups_per_smiles = sp_only.groupby("smiles").size()
    # collapse duplicate SP rows (same SMILES, different plates) by mean log2_fc
    # but keep them as separate weighted rows for now to keep code simple.
    print(f"[sp]   SP rows not in CRC: {len(sp_only)}  "
          f"(unique SMILES: {sp_only['smiles'].nunique()})")

    log2fc = sp_only["log2_fc_estimate"].astype(float).values
    conc = sp_only["concentration_M"].astype(float).values
    valid_mask = np.isfinite(log2fc) & np.isfinite(conc) & (conc > 0)
    sp_only = sp_only.iloc[valid_mask].reset_index(drop=True)
    log2fc = log2fc[valid_mask]
    conc = conc[valid_mask]
    crude = sp_crude_pec50(log2fc, conc)
    print(f"[sp]   crude pec50  min/med/max = "
          f"{crude.min():.2f} / {np.median(crude):.2f} / {crude.max():.2f}  "
          f"std={crude.std():.3f}")

    # ---- Featurize ----
    print("\n[feat] computing combined features (Morgan + RDKit) ...")
    X_crc = impute(combined(tr["smiles"].tolist()))
    X_sp = impute(combined(sp_only["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"[feat] X_crc={X_crc.shape}  X_sp={X_sp.shape}  X_te={X_te.shape}")

    # ---- Combine with weights ----
    y_crc = tr["pec50"].astype(float).values
    X_full = np.vstack([X_crc, X_sp])
    y_full = np.concatenate([y_crc, crude])
    w_full = np.concatenate([
        np.full(len(y_crc), CRC_WEIGHT, dtype=np.float64),
        np.full(len(crude), SP_WEIGHT, dtype=np.float64),
    ])
    print(f"[fit]  combined rows = {len(y_full)}  "
          f"(CRC={len(y_crc)}, SP={len(crude)})  "
          f"total weight = {w_full.sum():.0f}")

    # ---- Train one LGBM Huber on combined pool, no CV ----
    print(f"\n[fit]  LGBM Huber alpha={PARAMS['alpha']}  "
          f"n_estimators={PARAMS['n_estimators']}  "
          f"LR={PARAMS['learning_rate']}  num_leaves={PARAMS['num_leaves']}")
    ds = lgb.Dataset(X_full, label=y_full, weight=w_full)
    m = lgb.train(PARAMS, ds, callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(
        m.predict(X_te),
        y_crc.min() - 0.5, y_crc.max() + 0.5,
    ).astype(np.float64)

    # ---- in_RAE on 253 ----
    in_r = in_rae(y_unb, te_preds[unb_idx])
    print(f"\n[val]  in_RAE(253) = {in_r:.4f}")
    print(f"[val]  te(513) mean/std = "
          f"{te_preds.mean():.3f} / {te_preds.std():.3f}")

    # ---- Pearson vs nb972 ----
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(
        np.float64)
    te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(
        np.float64)
    pear_972 = float(pearsonr(te_preds, te_nb972)[0])
    pear_cp = float(pearsonr(te_preds, te_chemprop)[0])
    print(f"\n[corr] Pearson(te_nb1032, te_nb972)        = {pear_972:.4f}")
    print(f"[corr] Pearson(te_nb1032, te_chemprop_aux) = {pear_cp:.4f}")

    # ---- Save SP-only deploy ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_preds.astype(np.float32))
    plain = SUBMISSIONS / f"{TAG}_sp_only_training.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_preds,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    # ---- Optional 3-way bag if orthogonal enough ----
    bag_path = None
    bag_in_rae = None
    bag_w = None
    if pear_972 < 0.95:
        # equal-weight 3-way bag of (chemprop_aux, nb972, nb1032)
        bag = (te_chemprop + te_nb972 + te_preds) / 3.0
        bag_in_rae = in_rae(y_unb, bag[unb_idx])
        bag_w = "equal_3way"
        print(f"\n[bag]  Pearson<0.95  -> 3-way bag "
              f"(chemprop_aux + nb972 + nb1032)/3")
        print(f"[bag]  in_RAE(253) = {bag_in_rae:.4f}")
        bag_path = SUBMISSIONS / f"{TAG}_bag_chemprop_aux_nb972.csv"
        pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": bag,
        }).to_csv(bag_path, index=False)
        np.save(DATA_PROCESSED / f"te_{TAG}_bag.npy",
                bag.astype(np.float32))
        print(f"[save] {bag_path}")
    else:
        print(f"\n[bag]  Pearson>=0.95 ({pear_972:.4f}) -- skipping bag")

    delta = in_r - NB1014_REF
    if delta < -0.005:
        verdict = "BEATS_NB1014"
    elif abs(delta) <= 0.005:
        verdict = "TIES_NB1014"
    else:
        verdict = "WORSE_THAN_NB1014"

    summary = {
        "tag": TAG,
        "params": {k: v for k, v in PARAMS.items() if k != "verbose"},
        "sp_weight": SP_WEIGHT,
        "crc_weight": CRC_WEIGHT,
        "pec50_clip": list(PEC50_CLIP),
        "n_crc_rows": int(len(y_crc)),
        "n_sp_rows": int(len(crude)),
        "n_total_rows": int(len(y_full)),
        "sp_unique_smiles": int(sp_only["smiles"].nunique()),
        "crude_pec50_min": float(crude.min()),
        "crude_pec50_median": float(np.median(crude)),
        "crude_pec50_max": float(crude.max()),
        "crude_pec50_std": float(crude.std()),
        "in_rae_253": float(in_r),
        "te_mean_513": float(te_preds.mean()),
        "te_std_513": float(te_preds.std()),
        "pearson_with_nb972": float(pear_972),
        "pearson_with_chemprop_aux": float(pear_cp),
        "bagged_3way_in_rae": (None if bag_in_rae is None
                                else float(bag_in_rae)),
        "bag_weights": bag_w,
        "bag_submission": (str(bag_path) if bag_path else None),
        "plain_submission": str(plain),
        "nb1014_ref": NB1014_REF,
        "delta_vs_nb1014": float(delta),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pool                  = CRC(w=1.0)+SP_only(w={SP_WEIGHT}) "
          f"= {len(y_crc)}+{len(crude)} rows")
    print(f"   in_RAE(253)           = {in_r:.4f}")
    print(f"   pearson(nb1032,nb972) = {pear_972:.4f}")
    print(f"   pearson(nb1032,chemprop_aux) = {pear_cp:.4f}")
    if bag_in_rae is not None:
        print(f"   3-way bag in_RAE(253) = {bag_in_rae:.4f}")
    print(f"   delta vs nb1014       = {delta:+.4f}   -> {verdict}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
