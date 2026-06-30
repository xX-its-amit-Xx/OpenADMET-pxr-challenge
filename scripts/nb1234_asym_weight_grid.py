"""nb1234 -- Asymmetric BoB blend grid search.

Hypothesis:
    nb1211 settled on naive 0.5/0.5 mean of (nb1190_bob_mean, nb1200_bob_mean)
    at pooled RAE 0.5451. If nb1200 BoB (MACCS) is genuinely a touch better
    than nb1190 BoB (triple-FP) -- 0.5495 vs 0.5499 -- an asymmetric weight may
    extract a sliver more. Sweep w in {0.30..0.70 step 0.05} for
        blend_w = w * nb1190 + (1 - w) * nb1200.

Protocol:
  1. Load nb1190_bob_mean_oof.npy and nb1200_bob_mean_oof.npy (both 253 rows).
  2. In-sample grid: for each w, compute pooled RAE on full 253.
  3. Cross-fit grid: 5-fold (same KFold(shuffle=True, random_state=42) used in
     nb1211); on each train fold pick best w from the same grid, apply to val.
     Pool cross-fit RAE.
  4. Verdict at 0.003 margin vs nb1211 (0.5451).

Outputs:
  scripts/nb1234_asym_weight_grid.py
  data/processed/nb1234_summary.json
  data/processed/nb1234_best_oof.npy   (253,) float32 -- in-sample best-w blend
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
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1234"
SLSQP_FOLDS = 5       # match nb1211 fold scheme
SLSQP_SEED = 42       # match nb1211 seed
NB1211_REF_RAE = 0.5451     # naive 0.5/0.5 mean (nb1190_mean + nb1200_mean)
NB1190_BOB_MEAN_REF = 0.5499
NB1200_BOB_MEAN_REF = 0.5495
W_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- asymmetric BoB blend grid search on (nb1190_mean, nb1200_mean)")
    print(f"     grid w in {W_GRID}")
    print(f"     blend_w = w * nb1190 + (1 - w) * nb1200")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = int(len(y_unb))

    p1_path = DATA_PROCESSED / "nb1190_bob_mean_oof.npy"
    p2_path = DATA_PROCESSED / "nb1200_bob_mean_oof.npy"
    if not p1_path.exists():
        raise FileNotFoundError(f"missing {p1_path}")
    if not p2_path.exists():
        raise FileNotFoundError(f"missing {p2_path}")

    p1 = np.load(p1_path).astype(np.float64)   # nb1190 BoB mean
    p2 = np.load(p2_path).astype(np.float64)   # nb1200 BoB mean
    if p1.shape[0] != n_unb or p2.shape[0] != n_unb:
        raise ValueError(f"shape mismatch p1={p1.shape} p2={p2.shape} y={y_unb.shape}")

    rae_p1 = float(rae(y_unb, p1))
    rae_p2 = float(rae(y_unb, p2))
    print(f"\n[load] standalone pooled RAE on n={n_unb}")
    print(f"   nb1190 BoB mean : {rae_p1:.4f}  (ref {NB1190_BOB_MEAN_REF:.4f})")
    print(f"   nb1200 BoB mean : {rae_p2:.4f}  (ref {NB1200_BOB_MEAN_REF:.4f})")

    # ------------------------------------------------------------------ #
    # Step 2: in-sample (pooled) grid over W_GRID                        #
    # ------------------------------------------------------------------ #
    print("\n" + "-" * 78)
    print("  STEP 2  in-sample pooled RAE on full 253 (w grid)")
    print("-" * 78)
    in_sample_table = {}
    for w in W_GRID:
        pred = w * p1 + (1.0 - w) * p2
        r = float(rae(y_unb, pred))
        in_sample_table[f"{w:.2f}"] = r
        print(f"   w(nb1190)={w:.2f}  w(nb1200)={1-w:.2f}   RAE = {r:.6f}")
    best_w_in = min(in_sample_table, key=in_sample_table.get)
    best_rae_in = in_sample_table[best_w_in]
    best_w_in_val = float(best_w_in)
    print(f"\n   best in-sample w(nb1190) = {best_w_in_val:.2f}  RAE = {best_rae_in:.6f}")

    best_blend = best_w_in_val * p1 + (1.0 - best_w_in_val) * p2

    # ------------------------------------------------------------------ #
    # Step 3: 5-fold cross-fit grid search (1-parameter calibration)     #
    # ------------------------------------------------------------------ #
    print("\n" + "-" * 78)
    print("  STEP 3  5-fold cross-fit w grid (pick w on train fold, apply to val)")
    print("-" * 78)
    kf = KFold(n_splits=SLSQP_FOLDS, shuffle=True, random_state=SLSQP_SEED)
    cross_fit_pred = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        # pick best w on training fold (same grid)
        best_w_f = None
        best_rae_f = float("inf")
        per_w = {}
        for w in W_GRID:
            pred_tr = w * p1[tr_loc] + (1.0 - w) * p2[tr_loc]
            r_tr = float(rae(y_unb[tr_loc], pred_tr))
            per_w[f"{w:.2f}"] = r_tr
            if r_tr < best_rae_f:
                best_rae_f = r_tr
                best_w_f = w
        # apply frozen w to held-out fold
        pred_va = best_w_f * p1[va_loc] + (1.0 - best_w_f) * p2[va_loc]
        cross_fit_pred[va_loc] = pred_va
        rae_va = float(rae(y_unb[va_loc], pred_va))
        fold_records.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "best_w_train": float(best_w_f),
            "rae_train_at_best_w": float(best_rae_f),
            "rae_val_at_best_w": rae_va,
            "per_w_train_rae": per_w,
        })
        print(f"   fold {f}: best w(nb1190)={best_w_f:.2f}  "
              f"train RAE={best_rae_f:.4f}   val RAE={rae_va:.4f}")
    rae_cross_fit = float(rae(y_unb, cross_fit_pred))
    print(f"\n   pooled cross-fit RAE   = {rae_cross_fit:.6f}")

    # ------------------------------------------------------------------ #
    # Step 4: verdict vs nb1211 at 0.003 margin                          #
    # ------------------------------------------------------------------ #
    rae_naive_half = float(rae(y_unb, 0.5 * p1 + 0.5 * p2))
    print(f"   sanity: naive 0.5/0.5 RAE (recomputed) = {rae_naive_half:.6f}  "
          f"(nb1211 ref {NB1211_REF_RAE:.4f})")

    delta_in_sample_vs_nb1211 = best_rae_in - NB1211_REF_RAE
    delta_cross_fit_vs_nb1211 = rae_cross_fit - NB1211_REF_RAE
    delta_in_sample_vs_naive_half = best_rae_in - rae_naive_half
    delta_cross_fit_vs_naive_half = rae_cross_fit - rae_naive_half

    beats_nb1211_in_sample = best_rae_in < NB1211_REF_RAE - 0.003
    beats_nb1211_cross_fit = rae_cross_fit < NB1211_REF_RAE - 0.003
    beats_naive_half_in_sample = best_rae_in < rae_naive_half - 0.003
    beats_naive_half_cross_fit = rae_cross_fit < rae_naive_half - 0.003

    if beats_nb1211_cross_fit:
        verdict = (f"ASYM_BEATS_NB1211_CROSS_FIT (best w={best_w_in_val:.2f}, "
                   f"cross-fit {rae_cross_fit:.4f})")
    elif beats_nb1211_in_sample:
        verdict = (f"ASYM_BEATS_NB1211_IN_SAMPLE_ONLY (best w={best_w_in_val:.2f}, "
                   f"in-sample {best_rae_in:.4f}, cross-fit {rae_cross_fit:.4f})")
    elif abs(rae_cross_fit - NB1211_REF_RAE) < 0.003:
        verdict = (f"ASYM_FLAT_VS_NB1211 (best w={best_w_in_val:.2f}, "
                   f"cross-fit {rae_cross_fit:.4f})")
    else:
        verdict = (f"ASYM_HURTS_VS_NB1211 (best w={best_w_in_val:.2f}, "
                   f"cross-fit {rae_cross_fit:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1190 standalone        : {rae_p1:.4f}")
    print(f"   nb1200 standalone        : {rae_p2:.4f}")
    print(f"   nb1211 ref (0.5/0.5 mean): {NB1211_REF_RAE:.4f}")
    print(f"   naive 0.5/0.5 recomputed : {rae_naive_half:.4f}")
    print(f"   best in-sample w(nb1190) : {best_w_in_val:.2f}  RAE = {best_rae_in:.4f}")
    print(f"   pooled cross-fit RAE     : {rae_cross_fit:.4f}")
    print(f"   delta(in-sample vs nb1211)   : {delta_in_sample_vs_nb1211:+.4f}")
    print(f"   delta(cross-fit vs nb1211)   : {delta_cross_fit_vs_nb1211:+.4f}")
    print(f"   delta(in-sample vs 0.5/0.5)  : {delta_in_sample_vs_naive_half:+.4f}")
    print(f"   delta(cross-fit vs 0.5/0.5)  : {delta_cross_fit_vs_naive_half:+.4f}")
    print(f"   beats_nb1211 (in-sample)     : {beats_nb1211_in_sample}")
    print(f"   beats_nb1211 (cross-fit)     : {beats_nb1211_cross_fit}")
    print(f"   verdict                      : {verdict}")

    # ---- persist artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy", best_blend.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_cross_fit_oof.npy",
            cross_fit_pred.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}  "
          f"(in-sample best-w blend)")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_cross_fit_oof.npy'}  "
          f"(5-fold cross-fit blend)")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "w_grid": W_GRID,
        "components": ["nb1190_bob_mean_oof", "nb1200_bob_mean_oof"],
        "standalone_rae": {
            "nb1190_bob_mean": rae_p1,
            "nb1200_bob_mean": rae_p2,
        },
        "in_sample_w_table": in_sample_table,
        "best_w_in_sample": best_w_in_val,
        "best_rae_in_sample": best_rae_in,
        "rae_naive_half_recomputed": rae_naive_half,
        "cross_fit_folds": SLSQP_FOLDS,
        "cross_fit_seed": SLSQP_SEED,
        "cross_fit_fold_records": fold_records,
        "rae_cross_fit": rae_cross_fit,
        "nb1211_ref_rae": NB1211_REF_RAE,
        "delta_in_sample_vs_nb1211": delta_in_sample_vs_nb1211,
        "delta_cross_fit_vs_nb1211": delta_cross_fit_vs_nb1211,
        "delta_in_sample_vs_naive_half": delta_in_sample_vs_naive_half,
        "delta_cross_fit_vs_naive_half": delta_cross_fit_vs_naive_half,
        "beats_nb1211_in_sample_0p003": bool(beats_nb1211_in_sample),
        "beats_nb1211_cross_fit_0p003": bool(beats_nb1211_cross_fit),
        "beats_naive_half_in_sample_0p003": bool(beats_naive_half_in_sample),
        "beats_naive_half_cross_fit_0p003": bool(beats_naive_half_cross_fit),
        "beats_nb1211": bool(beats_nb1211_cross_fit),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.2f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("standalone_rae", "in_sample_w_table",
              "best_w_in_sample", "best_rae_in_sample",
              "rae_naive_half_recomputed", "rae_cross_fit",
              "delta_in_sample_vs_nb1211", "delta_cross_fit_vs_nb1211",
              "beats_nb1211_in_sample_0p003",
              "beats_nb1211_cross_fit_0p003",
              "beats_nb1211", "verdict"):
        print(f"  {k}: {res.get(k)}")
