"""nb1043 -- Hill-equation-based SP -> pEC50 mapping (vs nb1032 crude mapping).

Hypothesis: per-compound 2-parameter Hill fit on multi-point SP rows yields
much higher-quality pec50 than the crude log2FC mapping in nb1032 (0.755 in_RAE).

Pipeline:
1. Load 21k SP rows.
2. Group by SMILES. For compounds with >=2 distinct concentrations, fit a
   2-parameter Hill curve  log2FC = Lmax / (1 + (EC50/C)^n)  via scipy curve_fit.
3. For compounds with only 1 concentration: derive pec50 from log2FC magnitude
   and assay concentration (single-point conservative mapping).
4. Train LightGBM Huber on CRC (w=1.0) + Hill-derived SP-only (w=0.4).
5. Pearson vs nb972; blend if Pearson<0.95.
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
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from pxr.data import load_train, load_test, load_single_conc
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1043"
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
SP_WEIGHT = 0.4
CRC_WEIGHT = 1.0
PEC50_CLIP = (2.0, 8.0)
NB1014_REF = 0.5994


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def hill(C, Lmax, ec50, n):
    return Lmax / (1.0 + (ec50 / np.clip(C, 1e-15, None)) ** n)


def fit_hill_pec50(C: np.ndarray, log2fc: np.ndarray) -> float | None:
    """Fit log2FC = Lmax / (1 + (EC50/C)^n). Return pec50 or None."""
    try:
        C = np.asarray(C, float)
        y = np.asarray(log2fc, float)
        keep = np.isfinite(C) & np.isfinite(y) & (C > 0)
        C = C[keep]; y = y[keep]
        if len(C) < 2 or np.unique(C).size < 2:
            return None
        # Initial guesses
        Lmax0 = float(np.sign(np.nanmean(y)) * max(abs(np.nanmax(y)), abs(np.nanmin(y)), 0.5))
        if Lmax0 == 0:
            Lmax0 = float(np.sign(y[np.argmax(np.abs(y))]) * 1.0)
        ec0 = float(np.median(C))
        n0 = 1.0
        # If only 2 points, fix n=1 and fit (Lmax, ec50)
        if len(C) == 2:
            def hill_n1(C_, Lmax, ec50):
                return hill(C_, Lmax, ec50, 1.0)
            popt, _ = curve_fit(
                hill_n1, C, y,
                p0=[Lmax0, ec0],
                bounds=([-10, 1e-10], [10, 1e-2]),
                maxfev=2000,
            )
            ec50 = float(popt[1])
        else:
            popt, _ = curve_fit(
                hill, C, y,
                p0=[Lmax0, ec0, n0],
                bounds=([-10, 1e-10, 0.3], [10, 1e-2, 4.0]),
                maxfev=4000,
            )
            ec50 = float(popt[1])
        if not np.isfinite(ec50) or ec50 <= 0:
            return None
        pec50 = -np.log10(ec50)
        return float(np.clip(pec50, PEC50_CLIP[0], PEC50_CLIP[1]))
    except Exception:
        return None


def single_point_pec50(log2fc: float, conc_M: float) -> float:
    """Single-point heuristic: bigger |log2FC| -> stronger -> higher pec50.

    Anchor: log2FC = 2.0 at 10 uM -> pec50 ~ 5.5.
    pec50 = -log10(C) + 0.25 * log2FC, clipped.
    """
    if not (np.isfinite(log2fc) and np.isfinite(conc_M) and conc_M > 0):
        return 4.0
    base = -np.log10(conc_M)
    val = base + 0.25 * log2fc
    return float(np.clip(val, PEC50_CLIP[0], PEC50_CLIP[1]))


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"=== {TAG}: Hill-equation SP -> pec50 mapping ===")
    print("=" * 78)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    print(f"[load] unblind n={len(y_unb)}")

    tr = load_train()
    sp = load_single_conc()
    te = load_test()
    print(f"[load] CRC={len(tr)}  SP={len(sp)}  test={len(te)}")

    # Drop SP rows whose SMILES is in CRC (avoid double-counting)
    crc_smiles = set(tr["smiles"].astype(str))
    sp = sp[~sp["smiles"].astype(str).isin(crc_smiles)].copy()
    print(f"[sp]   SP-only rows (not in CRC) = {len(sp)}  unique SMILES={sp['smiles'].nunique()}")

    # Clean conc/log2fc
    sp["concentration_M"] = pd.to_numeric(sp["concentration_M"], errors="coerce")
    sp["log2_fc_estimate"] = pd.to_numeric(sp["log2_fc_estimate"], errors="coerce")
    sp = sp.dropna(subset=["concentration_M", "log2_fc_estimate", "smiles"])
    sp = sp[sp["concentration_M"] > 0]
    print(f"[sp]   clean rows = {len(sp)}")

    # Group by SMILES, fit Hill or single-point
    t1 = time.time()
    print("\n[hill] fitting per-compound Hill curves ...")
    pec50_map: dict[str, float] = {}
    n_multi = 0
    n_hill_ok = 0
    n_single = 0
    grouped = sp.groupby("smiles", sort=False)
    for smi, g in grouped:
        C = g["concentration_M"].values
        y = g["log2_fc_estimate"].values
        n_distinct = np.unique(C).size
        if n_distinct >= 2:
            n_multi += 1
            p = fit_hill_pec50(C, y)
            if p is not None:
                pec50_map[smi] = p
                n_hill_ok += 1
            else:
                # fallback: best-log2FC single point
                idx = int(np.argmax(np.abs(y)))
                pec50_map[smi] = single_point_pec50(float(y[idx]), float(C[idx]))
                n_single += 1
        else:
            n_single += 1
            pec50_map[smi] = single_point_pec50(float(y[0]), float(C[0]))
    print(f"[hill] multi-conc compounds  = {n_multi}  (Hill OK: {n_hill_ok}, fallback: {n_multi - n_hill_ok})")
    print(f"[hill] single-conc compounds = {n_single}")
    print(f"[hill] total mapped          = {len(pec50_map)}")
    print(f"[hill] fit wall = {time.time() - t1:.1f}s")

    sp_unique = pd.DataFrame({
        "smiles": list(pec50_map.keys()),
        "pec50": [pec50_map[s] for s in pec50_map.keys()],
    })
    print(f"[hill] sp pec50 distribution: "
          f"min={sp_unique['pec50'].min():.2f} "
          f"med={sp_unique['pec50'].median():.2f} "
          f"max={sp_unique['pec50'].max():.2f} "
          f"std={sp_unique['pec50'].std():.3f}")

    # Featurize
    print("\n[feat] computing combined features ...")
    X_crc = impute(combined(tr["smiles"].tolist()))
    X_sp = impute(combined(sp_unique["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))
    print(f"[feat] X_crc={X_crc.shape}  X_sp={X_sp.shape}  X_te={X_te.shape}")

    y_crc = tr["pec50"].astype(float).values
    y_sp = sp_unique["pec50"].astype(float).values

    X_full = np.vstack([X_crc, X_sp])
    y_full = np.concatenate([y_crc, y_sp])
    w_full = np.concatenate([
        np.full(len(y_crc), CRC_WEIGHT),
        np.full(len(y_sp), SP_WEIGHT),
    ])
    print(f"[fit]  rows={len(y_full)}  weight_sum={w_full.sum():.0f}")

    ds = lgb.Dataset(X_full, label=y_full, weight=w_full)
    m = lgb.train(PARAMS, ds)
    te_preds = np.clip(m.predict(X_te),
                       y_crc.min() - 0.5, y_crc.max() + 0.5).astype(np.float64)

    in_r = in_rae(y_unb, te_preds[unb_idx])
    print(f"\n[val]  in_RAE(253) = {in_r:.4f}")
    print(f"[val]  te(513) mean/std = {te_preds.mean():.3f} / {te_preds.std():.3f}")

    # Pearson vs nb972, chemprop_aux
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    pear_972 = float(pearsonr(te_preds, te_nb972)[0])
    pear_cp = float(pearsonr(te_preds, te_chemprop)[0])
    print(f"[corr] Pearson(nb1043, nb972)        = {pear_972:.4f}")
    print(f"[corr] Pearson(nb1043, chemprop_aux) = {pear_cp:.4f}")

    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_preds.astype(np.float32))
    plain = SUBMISSIONS / f"{TAG}_better_sp_mapping.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": te_preds,
    }).to_csv(plain, index=False)
    print(f"[save] te_{TAG}.npy  +  {plain}")

    # Bag if orthogonal
    bag_in_r = None
    bag_path = None
    if pear_972 < 0.95:
        bag = (te_chemprop + te_nb972 + te_preds) / 3.0
        bag_in_r = in_rae(y_unb, bag[unb_idx])
        bag_path = SUBMISSIONS / f"{TAG}_bag_chemprop_nb972.csv"
        pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": bag,
        }).to_csv(bag_path, index=False)
        np.save(DATA_PROCESSED / f"te_{TAG}_bag.npy", bag.astype(np.float32))
        print(f"[bag]  Pearson<0.95 -> 3-way bag in_RAE = {bag_in_r:.4f}")
        print(f"[bag]  saved {bag_path}")
    else:
        print(f"[bag]  Pearson>=0.95 ({pear_972:.4f}) -> skip bag")

    delta = in_r - NB1014_REF
    verdict = ("BEATS_NB1014" if delta < -0.005
               else "TIES_NB1014" if abs(delta) <= 0.005
               else "WORSE_THAN_NB1014")

    summary = {
        "tag": TAG,
        "params": {k: v for k, v in PARAMS.items() if k != "verbose"},
        "sp_weight": SP_WEIGHT,
        "crc_weight": CRC_WEIGHT,
        "n_crc_rows": int(len(y_crc)),
        "n_sp_rows": int(len(y_sp)),
        "n_with_hill_fit": int(n_hill_ok),
        "n_multi_conc_compounds": int(n_multi),
        "n_single_conc_compounds": int(n_single),
        "sp_pec50_min": float(sp_unique["pec50"].min()),
        "sp_pec50_median": float(sp_unique["pec50"].median()),
        "sp_pec50_max": float(sp_unique["pec50"].max()),
        "sp_pec50_std": float(sp_unique["pec50"].std()),
        "in_rae_253": float(in_r),
        "te_mean_513": float(te_preds.mean()),
        "te_std_513": float(te_preds.std()),
        "pearson_with_nb972": float(pear_972),
        "pearson_with_chemprop_aux": float(pear_cp),
        "bagged_3way_in_rae": (None if bag_in_r is None else float(bag_in_r)),
        "plain_submission": str(plain),
        "bag_submission": (str(bag_path) if bag_path else None),
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
    print(f"   in_RAE(253)           = {in_r:.4f}")
    print(f"   Hill fits OK          = {n_hill_ok}/{n_multi}")
    print(f"   pearson(nb972)        = {pear_972:.4f}")
    if bag_in_r is not None:
        print(f"   3-way bag in_RAE      = {bag_in_r:.4f}")
    print(f"   delta vs nb1014       = {delta:+.4f} -> {verdict}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    main()
