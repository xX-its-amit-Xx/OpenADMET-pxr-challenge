"""nb1011 -- Honest 5-fold cross-fit of (chemprop_aux ALONE + stretch).

Isolation experiment: does the nb1001 gain come from BLENDING or from STRETCHING?

nb1001 = honest cross-fit of (chemprop_aux + nb972 + stretch) -> 0.5994 on 253.
chemprop_aux raw on 253 = 0.6216.

If chemprop_aux + stretch alone gets to ~0.59, then nb972 wasn't actually
needed -- the stretch was doing all the work.  If it stays at ~0.62, then the
nb972 blend was the lever.

Procedure:
  Pool = [chemprop_aux] on the 253 unblind.
  5-fold KFold on the 253:
    For each fold f:
      a. mu_tr = mean of chemprop_aux predictions on the 4 train folds.
      b. Scan s in {1.00..2.00 step 0.05}.  For each s, compute
         mu_tr + s * (p_tr - mu_tr) and pick the s minimizing train-fold RAE.
      c. Apply (s_f, mu_tr) to the held-out fold.
  Pooled honest cross-fit RAE on the 253.
  Compare to chemprop_aux raw 0.6216 and nb1001 0.5994.

Deploy: refit s on all 253, apply to the 513 te file.

Outputs:
  data/processed/te_nb1011.npy
  data/processed/nb1011_summary.json
  submissions/nb1011_crossfit_chempropaux_stretch_only.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1011"
ANCHOR = "chemprop_aux"
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEED = 42
CHEMPROP_AUX_RAW = 0.6216
NB1001_CROSSFIT = 0.5994


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def best_stretch_on(p_train: np.ndarray, y_train: np.ndarray,
                    mu: float) -> tuple[float, float]:
    """Grid-scan s on the train-fold; return (best_s, best_train_rae)."""
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        stretched = mu + s * (p_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- honest 5-fold cross-fit of (chemprop_aux ALONE + stretch)")
    print("=" * 78)

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    p_513 = load_te(ANCHOR, te_names)
    print(f"[load] p_513({ANCHOR}) shape = {p_513.shape}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = p_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] p_unb shape = {p_unb.shape}, y shape = {y_unb.shape}")

    # ---- Anchor in_RAE sanity ----
    raw_in_rae = float(rae(y_unb, p_unb))
    print(f"\n[indiv] in_RAE on 253 unblind:")
    print(f"   {ANCHOR:30s}: {raw_in_rae:.4f}  (expected ~{CHEMPROP_AUX_RAW})")

    # =================================================================
    # 5-fold cross-fit (stretch grid only, fit on train folds)
    # =================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT  (stretch grid {STRETCH_GRID[0]}"
          f"..{STRETCH_GRID[-1]}, no blending)")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    fold_rows = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        # a) mu_tr from train fold anchor predictions
        mu_tr = float(p_unb[tr_loc].mean())
        # b) stretch grid on train fold
        s_f, rae_tr = best_stretch_on(p_unb[tr_loc], y_unb[tr_loc], mu_tr)
        # c) apply to held-out fold using SAME mu_tr (no peeking)
        oof[va_loc] = mu_tr + s_f * (p_unb[va_loc] - mu_tr)
        rae_va = float(rae(y_unb[va_loc], oof[va_loc]))
        fold_rows.append({"fold": k, "s": s_f, "mu_tr": mu_tr,
                          "train_rae": rae_tr, "val_rae": rae_va,
                          "n_va": int(len(va_loc))})
        print(f"   fold {k}: s={s_f:.2f}  mu_tr={mu_tr:.3f}  "
              f"train_RAE={rae_tr:.4f}  val_RAE={rae_va:.4f}")

    pooled_rae = float(rae(y_unb, oof))
    print(f"\n[cv] pooled honest cross-fit RAE on 253 = {pooled_rae:.4f}")

    # =================================================================
    # Comparison
    # =================================================================
    delta_vs_raw = pooled_rae - CHEMPROP_AUX_RAW
    delta_vs_nb1001 = pooled_rae - NB1001_CROSSFIT
    print("\n" + "-" * 78)
    print("COMPARISON")
    print("-" * 78)
    print(f"   chemprop_aux raw (no stretch)     = {CHEMPROP_AUX_RAW:.4f}")
    print(f"   nb1011 stretch-only (cross-fit)   = {pooled_rae:.4f}")
    print(f"   nb1001 chemprop+nb972+stretch     = {NB1001_CROSSFIT:.4f}")
    print(f"   delta (nb1011 - raw)              = {delta_vs_raw:+.4f}  "
          "(stretch alone)")
    print(f"   delta (nb1011 - nb1001)           = "
          f"{delta_vs_nb1001:+.4f}  (blend contribution)")
    if pooled_rae <= NB1001_CROSSFIT + 0.005:
        verdict = "STRETCH_DOES_THE_WORK"
        print("   verdict = STRETCH DOES THE WORK  "
              "(nb972 blend was NOT the lever)")
    elif pooled_rae <= CHEMPROP_AUX_RAW - 0.01:
        verdict = "STRETCH_HELPS_BLEND_HELPS_MORE"
        print("   verdict = STRETCH HELPS, BUT BLEND HELPS MORE")
    else:
        verdict = "STRETCH_INERT"
        print("   verdict = STRETCH INERT on chemprop_aux alone")

    # =================================================================
    # Deploy: refit s on all 253
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (refit s on all 253)")
    print("-" * 78)
    mu_deploy = float(p_unb.mean())
    s_deploy, _ = best_stretch_on(p_unb, y_unb, mu_deploy)
    in_rae_final = float(rae(y_unb,
                              mu_deploy + s_deploy * (p_unb - mu_deploy)))
    print(f"   deploy mu (anchor)  = {mu_deploy:.4f}")
    print(f"   deploy s            = {s_deploy:.2f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}  "
          "(overfit lower bound)")

    # Apply to all 513 using deploy mu (computed from 253 anchor preds)
    deploy_513 = (mu_deploy + s_deploy * (p_513 - mu_deploy)).astype(np.float32)
    print(f"   te(513) mean/std    = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}")

    # =================================================================
    # Save
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_crossfit_chempropaux_stretch_only.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_raw_in_rae_253": raw_in_rae,
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "fold_results": fold_rows,
        "pooled_cv_rae_253": pooled_rae,
        "chemprop_aux_raw_baseline": CHEMPROP_AUX_RAW,
        "nb1001_crossfit_baseline": NB1001_CROSSFIT,
        "delta_vs_raw": delta_vs_raw,
        "delta_vs_nb1001": delta_vs_nb1001,
        "verdict": verdict,
        "deploy_mu_anchor": mu_deploy,
        "deploy_s": float(s_deploy),
        "in_sample_rae_overfit_bound": in_rae_final,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                     = {ANCHOR}")
    print(f"   raw                        = {CHEMPROP_AUX_RAW:.4f}")
    print(f"   honest cross-fit RAE (253) = {pooled_rae:.4f}")
    print(f"   nb1001 baseline            = {NB1001_CROSSFIT:.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   deploy s                   = {s_deploy:.2f}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("pooled_cv_rae_253", "delta_vs_raw", "delta_vs_nb1001",
              "verdict", "deploy_s",
              "in_sample_rae_overfit_bound", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
