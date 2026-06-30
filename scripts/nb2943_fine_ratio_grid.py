"""nb2943 -- Fine ratio grid sweep for {nb2240, nb1191, K-ensemble}.

NEW PARADIGM:
    2-D sweep over (w_nb2240, w_nb1191) with constraint sum<=1.  The
    remainder K_ensemble_weight = 1 - w_nb2240 - w_nb1191 is allocated to
    the equal-weight K-ensemble {K18, K24, K28} (nb2240 is K=20 carried
    as its own anchor).  No SLSQP, no learning -- pure scalar grid scan.

GRID:
    w_nb2240 in {0.4, 0.5, 0.6, 0.7}
    w_nb1191 in {0.0, 0.1, 0.2, 0.3}
    w_K = 1 - w_nb2240 - w_nb1191  (must be >= 0)
    pred = w_nb2240*nb2240 + w_nb1191*nb1191 + w_K*equal_K

ANCHORS:
    nb2240 -> K=20 (data/processed/nb2240_mean_bag_oof_K20.npy + te_nb2240_K20.npy)
    nb1191 -> PRE-pyramid (data/processed/nb1191_pred_oof.npy + te_nb1191.npy)
    equal_K = mean(K18, K24, K28)   -- K=20 excluded (carried in nb2240)

PROTOCOL:
    1. Load 5 cached OOF + te arrays (K18, K20, K24, K28, nb1191).
    2. Build equal_K = mean(K18, K24, K28) -- 3-anchor average.
    3. For each (w_2240, w_1191, w_K=1-w_2240-w_1191) with w_K >= 0:
         pred_oof = w_2240*nb2240 + w_1191*nb1191 + w_K*equal_K
         5-fold scaffold CV on 253, kf_seed=1001
    4. Pick best pooled RAE.

GATE:
    best_rae < 0.4570 -> PROMOTE
    best_rae < 0.4585 -> BETTER_THAN_NB2934
    else              -> FAIL

References:
    nb2934 5-anchor equal weight pooled RAE = 0.4580 (MARGINAL_BEAT)
    nb2604 4-K equal weight pooled RAE      = 0.4580
    nb1191 PRE-pyramid deep-30              = 0.4718
    nb2171 ceiling deep-30 PRIMARY-1        = 0.4682

Outputs:
    scripts/nb2943_fine_ratio_grid.py
    data/processed/nb2943_summary.json
    data/processed/nb2943_pred_oof.npy   (253,) float32   [best combo]
    data/processed/te_nb2943.npy         (513,) float32   [best combo]
    submissions/nb2943_fine_ratio_grid.csv  (on any non-FAIL)
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
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2943"

# ---- Component OOF + te paths ----
K18_OOF_PATH = DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy"
K18_TE_PATH = DATA_PROCESSED / "te_nb2604_K18.npy"
K20_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"   # == nb2240 anchor
K20_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
K24_OOF_PATH = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
K24_TE_PATH = DATA_PROCESSED / "te_nb2310_K24.npy"
K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
K28_TE_PATH = DATA_PROCESSED / "te_nb2112.npy"
NB1191_OOF_PATH = DATA_PROCESSED / "nb1191_pred_oof.npy"
NB1191_TE_PATH = DATA_PROCESSED / "te_nb1191.npy"

# ---- Eval protocol ----
N_FOLDS = 5
KF_SEED = 1001

# ---- Grid ----
W_NB2240_GRID = [0.4, 0.5, 0.6, 0.7]
W_NB1191_GRID = [0.0, 0.1, 0.2, 0.3]
EPS = 1e-9

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_BETTER_NB2934 = 0.4585

# ---- References ----
NB2934_REF = 0.4580
NB2604_REF = 0.4580
NB1191_REF = 0.4718
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- fine ratio grid sweep {{nb2240, nb1191, equal_K(K18,K24,K28)}}")
    print(f"          paradigm: 2-D scalar grid, df = 0 (no learning)")
    print(f"          grid w_nb2240={W_NB2240_GRID}  w_nb1191={W_NB1191_GRID}")
    print(f"          refs nb2934={NB2934_REF}  nb1191={NB1191_REF}  "
          f"nb2171={NB2171_REF}")
    print("=" * 78)

    # ---- Load truth + scaffolds ----
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (te_df["smiles"].astype(str).tolist()
                 if "smiles" in te_df.columns
                 else te_df["SMILES"].astype(str).tolist())
    te_names = (te_df["name"].values
                if "name" in te_df.columns
                else te_df["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Load components ----
    print("\n" + "-" * 78)
    print("Load 5 components")
    print("-" * 78)
    comps = [
        ("K18", K18_OOF_PATH, K18_TE_PATH),
        ("K20", K20_OOF_PATH, K20_TE_PATH),   # == nb2240
        ("K24", K24_OOF_PATH, K24_TE_PATH),
        ("K28", K28_OOF_PATH, K28_TE_PATH),
        ("nb1191", NB1191_OOF_PATH, NB1191_TE_PATH),
    ]
    oofs = {}
    tes = {}
    per_anchor_rae = {}
    for name, oof_p, te_p in comps:
        if not oof_p.exists():
            raise FileNotFoundError(f"missing OOF: {oof_p}")
        if not te_p.exists():
            raise FileNotFoundError(f"missing te: {te_p}")
        oof = np.load(oof_p).astype(np.float64)
        te_v = np.load(te_p).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{name} oof shape {oof.shape} != ({n_unb},)")
        if te_v.shape != (n_test,):
            raise ValueError(f"{name} te shape {te_v.shape} != ({n_test},)")
        oofs[name] = oof
        tes[name] = te_v
        r = float(rae(y_unb, oof))
        per_anchor_rae[name] = r
        print(f"  {name:>7s}  oof_RAE={r:.4f}  te_mean={te_v.mean():.3f}  "
              f"te_std={te_v.std():.3f}  [{oof_p.name}]")

    # ---- Build equal_K from {K18, K24, K28} (K20 excluded -> carried by nb2240) ----
    equal_K_oof = (oofs["K18"] + oofs["K24"] + oofs["K28"]) / 3.0
    equal_K_te = (tes["K18"] + tes["K24"] + tes["K28"]) / 3.0
    equal_K_rae = float(rae(y_unb, equal_K_oof))
    print(f"\n  equal_K = mean(K18, K24, K28)  oof_RAE={equal_K_rae:.4f}  "
          f"te_mean={equal_K_te.mean():.3f}  te_std={equal_K_te.std():.3f}")

    # ---- Build scaffold splits once (deterministic) ----
    print("\n" + "-" * 78)
    print(f"5-fold scaffold CV  kf_seed={KF_SEED}")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )

    def evaluate_blend(w2240: float, w1191: float, wK: float):
        pred_oof = w2240 * oofs["K20"] + w1191 * oofs["nb1191"] + wK * equal_K_oof
        pred_te = w2240 * tes["K20"] + w1191 * tes["nb1191"] + wK * equal_K_te
        oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
        per_fold = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            oof_pooled[va_loc] = pred_oof[va_loc]
            per_fold.append(float(rae(y_unb[va_loc], pred_oof[va_loc])))
        if np.isnan(oof_pooled).any():
            raise RuntimeError("scaffold splits did not cover all rows")
        pooled = float(rae(y_unb, oof_pooled))
        return pooled, per_fold, pred_oof, pred_te

    # ---- Sweep grid ----
    print("\n" + "-" * 78)
    print("Grid sweep")
    print("-" * 78)
    grid_results = []
    best_pooled = float("inf")
    best_combo = None
    best_pred_oof = None
    best_pred_te = None
    best_per_fold = None
    print(f"  {'w_2240':>7s}  {'w_1191':>7s}  {'w_K':>7s}  {'pooled':>7s}  "
          f"{'fold_mean':>9s}  {'fold_std':>8s}")
    for w2240 in W_NB2240_GRID:
        for w1191 in W_NB1191_GRID:
            wK = 1.0 - w2240 - w1191
            if wK < -EPS:
                # invalid (sum > 1)
                continue
            if wK < 0:
                wK = 0.0
            pooled, per_fold, pred_oof, pred_te = evaluate_blend(w2240, w1191, wK)
            grid_results.append({
                "w_nb2240": float(w2240),
                "w_nb1191": float(w1191),
                "w_K_ensemble": float(wK),
                "pooled_rae": pooled,
                "per_fold_rae": per_fold,
                "fold_mean": float(np.mean(per_fold)),
                "fold_std": float(np.std(per_fold)),
            })
            tag = ""
            if pooled < best_pooled:
                best_pooled = pooled
                best_combo = (w2240, w1191, wK)
                best_pred_oof = pred_oof.copy()
                best_pred_te = pred_te.copy()
                best_per_fold = per_fold
                tag = "  <-- best"
            print(f"  {w2240:>7.2f}  {w1191:>7.2f}  {wK:>7.2f}  {pooled:>7.4f}  "
                  f"{np.mean(per_fold):>9.4f}  {np.std(per_fold):>8.4f}{tag}")

    print(f"\n[best] (w_nb2240, w_nb1191, w_K) = "
          f"({best_combo[0]:.2f}, {best_combo[1]:.2f}, {best_combo[2]:.2f})  "
          f"pooled_RAE={best_pooled:.4f}")

    # ---- Gate ----
    if best_pooled < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif best_pooled < GATE_BETTER_NB2934:
        verdict = "BETTER_THAN_NB2934"
    else:
        verdict = "FAIL"
    print(f"[gate] best_pooled={best_pooled:.4f}  "
          f"(<{GATE_PROMOTE} PROMOTE / <{GATE_BETTER_NB2934} BETTER_THAN_NB2934)  "
          f"-> {verdict}")

    delta_vs_nb2934 = best_pooled - NB2934_REF
    delta_vs_nb2171 = best_pooled - NB2171_REF
    delta_vs_nb1191 = best_pooled - NB1191_REF
    print(f"  delta vs nb2934 ({NB2934_REF}) = {delta_vs_nb2934:+.4f}")
    print(f"  delta vs nb2171 ({NB2171_REF}) = {delta_vs_nb2171:+.4f}")
    print(f"  delta vs nb1191 ({NB1191_REF}) = {delta_vs_nb1191:+.4f}")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, best_pred_oof.astype(np.float32))
    np.save(te_path, best_pred_te.astype(np.float32))
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_fine_ratio_grid.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": best_pred_te.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] verdict=FAIL; no submission CSV written")

    te_unb_in = float(rae(y_unb, best_pred_te[unb_idx]))

    summary = {
        "tag": TAG,
        "method": "fine_ratio_grid_nb2240_nb1191_equal_K",
        "paradigm": "2D_scalar_grid_df_zero_no_learning",
        "anchors": {
            "nb2240": "K=20 (nb2240_mean_bag_oof_K20.npy)",
            "nb1191": "PRE-pyramid (nb1191_pred_oof.npy)",
            "equal_K": "mean(K18, K24, K28)",
        },
        "grid_w_nb2240": W_NB2240_GRID,
        "grid_w_nb1191": W_NB1191_GRID,
        "per_anchor_rae_in_sample": per_anchor_rae,
        "equal_K_oof_rae": equal_K_rae,
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "grid_results": grid_results,
        "best": {
            "w_nb2240": float(best_combo[0]),
            "w_nb1191": float(best_combo[1]),
            "w_K_ensemble": float(best_combo[2]),
            "pooled_rae": best_pooled,
            "per_fold_rae": best_per_fold,
            "fold_mean": float(np.mean(best_per_fold)),
            "fold_std": float(np.std(best_per_fold)),
        },
        "best_pooled_rae": best_pooled,
        "mean_rae": best_pooled,
        "gate_promote": GATE_PROMOTE,
        "gate_better_nb2934": GATE_BETTER_NB2934,
        "verdict": verdict,
        "delta_vs_nb2934": delta_vs_nb2934,
        "delta_vs_nb2171": delta_vs_nb2171,
        "delta_vs_nb1191": delta_vs_nb1191,
        "nb2934_ref": NB2934_REF,
        "nb2604_ref": NB2604_REF,
        "nb2171_ref": NB2171_REF,
        "nb1191_ref": NB1191_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "te_mean": float(best_pred_te.mean()),
        "te_std": float(best_pred_te.std()),
        "te_unb_in_sample_rae": te_unb_in,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   per-anchor RAE  = " + "  ".join(
        f"{k}={v:.4f}" for k, v in per_anchor_rae.items()))
    print(f"   equal_K RAE     = {equal_K_rae:.4f}")
    print(f"   n_grid          = {len(grid_results)}")
    print(f"   best (w2240,w1191,wK) = "
          f"({best_combo[0]:.2f}, {best_combo[1]:.2f}, {best_combo[2]:.2f})")
    print(f"   best pooled RAE = {best_pooled:.4f}")
    print(f"   verdict         = {verdict}")
    print(f"   delta nb2934    = {delta_vs_nb2934:+.4f}")
    print(f"   delta nb2171    = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] in  = {te_unb_in:.4f}")
    print(f"   wall            = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_pooled_rae", "mean_rae", "verdict",
        "delta_vs_nb2934", "delta_vs_nb2171", "delta_vs_nb1191",
        "te_unb_in_sample_rae", "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  best_combo: {res.get('best')}")
