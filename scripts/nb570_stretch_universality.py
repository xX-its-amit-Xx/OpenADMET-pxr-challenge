"""nb570 -- STRETCH UNIVERSALITY across top routers.

Hypothesis: every top router is variance-compressed by hedged blending, so the
nb562 rank-stretch recipe should generalise. For each base in
{nb502, nb510, nb481, nb492, nb472}:

  1) Load <base>_pred_oof.npy (253) and te_<base>.npy (513).
  2) For s in {1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30}:
       - 5-fold cross-fit: per fold, fit mu = mean(pred_oof[tr]),
         apply pred_va = mu + s*(pred_oof[va] - mu) on held out.
       - Pool the 5 held-out predictions, score RAE.
  3) Pick best s; report delta vs unstretched (s=1.0).
  4) Save te_{base}_stretched.npy with deploy mu = mean(pred_oof_all),
     deploy s = best_s applied to te_<base>.npy.

Output: a single table base | original RAE | best s | stretched RAE | delta.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW

SEED = 0
N_FOLDS = 5
TAG = "nb570"
STRETCH_GRID = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
BASES = ["nb502", "nb510", "nb481", "nb492", "nb472"]


def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def _crossfit_rae_at_s(pred_oof: np.ndarray, y: np.ndarray, s: float,
                       n_splits: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Return pooled held-out cross-fit predictions at stretch factor s."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = pred_oof.shape[0]
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_i, va_i in kf.split(np.arange(n)):
        mu_tr = float(pred_oof[tr_i].mean())
        oof[va_i] = _stretch(pred_oof[va_i], mu_tr, s)
    return oof


def _eval_base(base: str, unb_idx: np.ndarray, unb_y: np.ndarray,
               n_te: int) -> dict:
    oof_path = DATA_PROCESSED / f"{base}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{base}.npy"
    if not oof_path.exists() or not te_path.exists():
        return {"base": base, "ok": False, "missing": str(oof_path)
                if not oof_path.exists() else str(te_path)}

    pred_oof = np.load(oof_path).astype(np.float64)
    pred_te = np.load(te_path).astype(np.float64)
    if pred_oof.shape != (unb_y.shape[0],):
        return {"base": base, "ok": False,
                "err": f"oof shape {pred_oof.shape} != ({unb_y.shape[0]},)"}
    if pred_te.shape != (n_te,):
        return {"base": base, "ok": False,
                "err": f"te shape {pred_te.shape} != ({n_te},)"}

    rae_orig = float(rae(unb_y, pred_oof))

    per_s = []
    best_s, best_rae, best_oof = 1.0, float("inf"), pred_oof.copy()
    for s in STRETCH_GRID:
        oof_s = _crossfit_rae_at_s(pred_oof, unb_y, s)
        r = float(rae(unb_y, oof_s))
        per_s.append((s, r))
        if r < best_rae:
            best_rae, best_s, best_oof = r, float(s), oof_s

    # Deploy on te: mu fit on full 253 pred_oof, s = best
    mu_deploy = float(pred_oof.mean())
    te_stretched = _stretch(pred_te, mu_deploy, best_s).astype(np.float32)
    out_path = DATA_PROCESSED / f"te_{base}_stretched.npy"
    np.save(out_path, te_stretched)

    return {
        "base": base,
        "ok": True,
        "rae_orig": rae_orig,
        "best_s": best_s,
        "rae_stretched": best_rae,
        "delta": best_rae - rae_orig,
        "per_s": per_s,
        "pred_oof_mean": float(pred_oof.mean()),
        "pred_oof_std": float(pred_oof.std()),
        "te_stretched_mean": float(te_stretched.mean()),
        "te_stretched_std": float(te_stretched.std()),
        "out_path": str(out_path),
    }


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- STRETCH UNIVERSALITY across top routers")
    print(f"  bases: {BASES}")
    print(f"  grid : {STRETCH_GRID}")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")
    print(f"truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}\n")

    results = []
    for base in BASES:
        r = _eval_base(base, unb_idx, unb_y, n_te)
        results.append(r)
        if not r.get("ok"):
            print(f"[{base}] SKIPPED: {r}")
            continue
        print("-" * 78)
        print(f"[{base}] orig RAE = {r['rae_orig']:.4f}  "
              f"oof mean/std = {r['pred_oof_mean']:.3f}/{r['pred_oof_std']:.3f}")
        for s, rs in r["per_s"]:
            marker = " <- best" if s == r["best_s"] else ""
            print(f"   s={s:.2f}  cross-fit RAE = {rs:.4f}{marker}")
        print(f"   delta vs orig = {r['delta']:+.4f}   "
              f"saved {r['out_path']}")

    # ---- Final table ----
    print("\n" + "=" * 78)
    print("FINAL TABLE")
    print("=" * 78)
    header = f"  {'base':<8} | {'orig RAE':>9} | {'best s':>7} | " \
             f"{'stretched RAE':>14} | {'delta':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows_table = []
    best_overall_rae = float("inf")
    best_overall_router = None
    universal_works = True
    for r in results:
        if not r.get("ok"):
            print(f"  {r['base']:<8} | -- skipped --")
            universal_works = False
            continue
        print(f"  {r['base']:<8} | {r['rae_orig']:9.4f} | "
              f"{r['best_s']:7.2f} | {r['rae_stretched']:14.4f} | "
              f"{r['delta']:+8.4f}")
        rows_table.append({
            "base": r["base"],
            "rae_orig": r["rae_orig"],
            "best_s": r["best_s"],
            "rae_stretched": r["rae_stretched"],
            "delta": r["delta"],
        })
        if r["rae_stretched"] < best_overall_rae:
            best_overall_rae = r["rae_stretched"]
            best_overall_router = r["base"]
        # "universal" = best s > 1.0 AND delta < 0 (strictly improved)
        if not (r["best_s"] > 1.0 and r["delta"] < 0):
            universal_works = False

    table_csv = DATA_PROCESSED / f"{TAG}_summary.csv"
    pd.DataFrame(rows_table).to_csv(table_csv, index=False)
    print(f"\nWrote {table_csv}")

    print("\n" + "=" * 78)
    print(f"best stretched router : {best_overall_router}  "
          f"RAE={best_overall_rae:.4f}")
    print(f"universal stretch (s>1 AND delta<0 for every base) = "
          f"{universal_works}")
    print("=" * 78)

    return {
        "success": True,
        "results": rows_table,
        "best_overall_router": best_overall_router,
        "best_overall_rae": float(best_overall_rae),
        "universal_stretch_works": bool(universal_works),
        "summary_csv": str(table_csv),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
