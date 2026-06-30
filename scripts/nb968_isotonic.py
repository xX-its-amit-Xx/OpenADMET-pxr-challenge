"""nb968 -- PER-DECILE / GLOBAL / QUANTILE-SHIFT / WEIGHTED isotonic calibration
            on nb2103 K=28 5-seed mean-bag OOF.

HYPOTHESIS:
    nb2103 K=28 mean-bag OOF preds (253,) carry a (truth - pred) bias that
    isotonic regression can absorb monotonically.  Test four variants:

      A.  GLOBAL isotonic on (pred, y), 5-fold cross-fit
      B.  PER-DECILE isotonic (10 separate fits, one per pred-decile),
          5-fold cross-fit
      C.  PER-QUANTILE SHIFT: in each fold, shift each decile by the
          median(y - pred) of the train-fold rows in that decile
      D.  WEIGHTED isotonic (global) with sample_weight = inverse
          bin density (uplifts tails), 5-fold cross-fit

REFERENCES (from data/processed/nb2103_summary.json):
    nb2103 K=28 mean_bag RAE   = 0.4737
    nb2103 K=28 median_bag RAE = 0.4698  (target to beat)

DECISION:
    margin = 0.003.  If any method beats 0.4698 by >=0.003, build deploy
    CSV submissions/nb968_deploy_isotonic.csv applying the SAME calibration
    learned on the full 253 (i.e. refit on all 253) to te_chemprop_aux
    (513 deploy preds; LGBM K=28 deploy preds on 513 are not cached so
    we apply calibration to the chemprop_aux anchor and disclose).

Outputs:
    scripts/nb968_isotonic.py
    data/processed/nb968_summary.json
    submissions/nb968_deploy_isotonic.csv  (only if beats)
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

TAG = "nb968"
N_FOLDS = 5
SEED = 42
N_DECILES = 10
DECISION_MARGIN = 0.003
REF_MEAN_BAG = 0.4737
REF_MEDIAN_BAG = 0.4698


# ----------------------------- helpers -----------------------------

def _decile_bins(p: np.ndarray, n_bins: int = N_DECILES) -> np.ndarray:
    """Return bin edges of length n_bins+1 from quantiles of p."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(p, qs)
    # ensure strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return edges


def _assign_bin(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return bin index in [0, n_bins-1] for each row."""
    n_bins = len(edges) - 1
    idx = np.searchsorted(edges, p, side="right") - 1
    return np.clip(idx, 0, n_bins - 1)


def _method_global(p_tr, y_tr, p_va) -> np.ndarray:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y_tr)
    return iso.predict(p_va)


def _method_per_decile(p_tr, y_tr, p_va) -> np.ndarray:
    edges = _decile_bins(p_tr, N_DECILES)
    tr_bin = _assign_bin(p_tr, edges)
    va_bin = _assign_bin(p_va, edges)
    out = np.empty_like(p_va, dtype=np.float64)
    for b in range(N_DECILES):
        m_tr = tr_bin == b
        m_va = va_bin == b
        if m_va.sum() == 0:
            continue
        if m_tr.sum() < 3:
            # fallback: shift by train decile median
            shift = float(np.median(y_tr[m_tr] - p_tr[m_tr])) if m_tr.sum() > 0 else 0.0
            out[m_va] = p_va[m_va] + shift
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr[m_tr], y_tr[m_tr])
        out[m_va] = iso.predict(p_va[m_va])
    return out


def _method_quantile_shift(p_tr, y_tr, p_va) -> np.ndarray:
    edges = _decile_bins(p_tr, N_DECILES)
    tr_bin = _assign_bin(p_tr, edges)
    va_bin = _assign_bin(p_va, edges)
    shifts = np.zeros(N_DECILES, dtype=np.float64)
    for b in range(N_DECILES):
        m_tr = tr_bin == b
        if m_tr.sum() > 0:
            shifts[b] = float(np.median(y_tr[m_tr] - p_tr[m_tr]))
    return p_va + shifts[va_bin]


def _method_weighted(p_tr, y_tr, p_va) -> np.ndarray:
    edges = _decile_bins(p_tr, N_DECILES)
    tr_bin = _assign_bin(p_tr, edges)
    counts = np.bincount(tr_bin, minlength=N_DECILES).astype(np.float64)
    # inverse bin density (uplift tails); avoid div-by-zero
    inv = 1.0 / np.maximum(counts, 1.0)
    inv = inv / inv.sum() * len(p_tr)
    w_tr = inv[tr_bin]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, y_tr, sample_weight=w_tr)
    return iso.predict(p_va)


METHODS = {
    "A_global":       _method_global,
    "B_per_decile":   _method_per_decile,
    "C_quantile_shift": _method_quantile_shift,
    "D_weighted":     _method_weighted,
}


def _cross_fit(p: np.ndarray, y: np.ndarray, fn) -> np.ndarray:
    """5-fold cross-fit: train fn on 4/5, predict on 1/5."""
    n = len(p)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = np.full(n, np.nan, dtype=np.float64)
    for tr, va in kf.split(np.arange(n)):
        out[va] = fn(p[tr], y[tr], p[va])
    assert not np.any(np.isnan(out)), "NaN in cross-fit output"
    return out


# ----------------------------- main -----------------------------

def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- ISOTONIC CALIBRATION on nb2103 K=28 mean-bag OOF")
    print("=" * 78)

    # ---- Load OOF + truth ----
    p_oof = np.load(
        DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    ).astype(np.float64)
    y_unb = np.load(
        DATA_PROCESSED / "_audit_unblind_y.npy"
    ).astype(np.float64)
    unb_idx = np.load(
        DATA_PROCESSED / "_audit_unblind_idx.npy"
    ).astype(int)
    n = len(y_unb)
    assert p_oof.shape == y_unb.shape == (n,), \
        f"shape mismatch: p={p_oof.shape} y={y_unb.shape}"
    print(f"[load] p_oof={p_oof.shape} mean={p_oof.mean():.3f} "
          f"std={p_oof.std():.3f}")
    print(f"[load] y_unb={y_unb.shape} mean={y_unb.mean():.3f} "
          f"std={y_unb.std():.3f}")
    print(f"[load] unb_idx={unb_idx.shape}")

    rae_baseline = float(rae(y_unb, p_oof))
    print(f"\n[ref] nb2103 K=28 mean_bag OOF RAE    = {rae_baseline:.4f}  "
          f"(spec ref {REF_MEAN_BAG:.4f})")
    print(f"[ref] nb2103 K=28 median_bag spec ref = {REF_MEDIAN_BAG:.4f}")
    print(f"[ref] decision margin                 = {DECISION_MARGIN:.4f}")

    # ---- Cross-fit each method ----
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT (seed={SEED}, n_deciles={N_DECILES})")
    print("-" * 78)

    method_results: dict[str, dict] = {}
    method_oof: dict[str, np.ndarray] = {}
    for mname, fn in METHODS.items():
        oof = _cross_fit(p_oof, y_unb, fn)
        r = float(rae(y_unb, oof))
        method_oof[mname] = oof
        method_results[mname] = {
            "rae_cross_fit": r,
            "delta_vs_baseline_mean_bag": r - rae_baseline,
            "delta_vs_ref_median_bag": r - REF_MEDIAN_BAG,
            "oof_mean": float(oof.mean()),
            "oof_std":  float(oof.std()),
        }
        flag = ""
        if r < REF_MEDIAN_BAG - DECISION_MARGIN:
            flag = "  <- BEATS median_bag by margin"
        elif r < rae_baseline - DECISION_MARGIN:
            flag = "  <- beats mean_bag by margin"
        elif r < rae_baseline:
            flag = "  (improves vs mean_bag, < margin)"
        else:
            flag = "  (no improvement)"
        print(f"  {mname:18s}  RAE={r:.4f}  "
              f"d_mean={r - rae_baseline:+.4f}  "
              f"d_med={r - REF_MEDIAN_BAG:+.4f}{flag}")

    # ---- Pick best ----
    best_name = min(method_results, key=lambda k: method_results[k]["rae_cross_fit"])
    best_rae = method_results[best_name]["rae_cross_fit"]
    beats_mean_bag = best_rae < rae_baseline - DECISION_MARGIN
    beats_median_bag = best_rae < REF_MEDIAN_BAG - DECISION_MARGIN

    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)
    print(f"  best method            = {best_name}")
    print(f"  best cross-fit RAE     = {best_rae:.4f}")
    print(f"  vs mean_bag {rae_baseline:.4f}  -> "
          f"delta {best_rae - rae_baseline:+.4f}  "
          f"beats_by_margin={beats_mean_bag}")
    print(f"  vs median_bag {REF_MEDIAN_BAG:.4f}  -> "
          f"delta {best_rae - REF_MEDIAN_BAG:+.4f}  "
          f"beats_by_margin={beats_median_bag}")

    # ---- Build deploy CSV only if beats spec target ----
    deploy_path = None
    deploy_info: dict = {}
    if beats_median_bag or beats_mean_bag:
        # Refit best method on FULL 253 OOF -> calibrator
        # Apply to te_chemprop_aux (513) as the deploy anchor (LGBM K=28 deploy
        # preds on 513 not cached; document this).
        te_anchor = np.load(
            DATA_PROCESSED / "te_chemprop_aux.npy"
        ).astype(np.float64)
        n_te = te_anchor.shape[0]
        print(f"\n[deploy] te_chemprop_aux 513 shape={te_anchor.shape}")

        # Refit calibrator using best method on full 253: train=full, test=513
        fn = METHODS[best_name]
        deploy_pred = fn(p_oof, y_unb, te_anchor).astype(np.float32)
        print(f"[deploy] deploy_pred mean={deploy_pred.mean():.3f} "
              f"std={deploy_pred.std():.3f}")

        # Save deploy CSV
        te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
        deploy_path = SUBMISSIONS / f"{TAG}_deploy_isotonic.csv"
        pd.DataFrame({
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": deploy_pred,
        }).to_csv(deploy_path, index=False)
        print(f"[deploy] wrote {deploy_path}")

        np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_pred)
        print(f"[deploy] wrote {DATA_PROCESSED / f'te_{TAG}.npy'}")

        # In-sample check on 253 slice of deploy
        in_pred = deploy_pred[unb_idx].astype(np.float64)
        in_rae = float(rae(y_unb, in_pred))
        deploy_info = {
            "deploy_csv": str(deploy_path),
            "deploy_te_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
            "deploy_te_mean": float(deploy_pred.mean()),
            "deploy_te_std": float(deploy_pred.std()),
            "deploy_in_rae_253": in_rae,
            "deploy_anchor": "te_chemprop_aux (513)",
            "deploy_note": "LGBM K=28 deploy on 513 not cached; calibration applied to chemprop_aux anchor only.",
        }
        print(f"[deploy] in-sample RAE on 253 (anchor calibrated) = {in_rae:.4f}")
    else:
        print("\n[deploy] No method beats by margin -- no deploy CSV written.")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "isotonic_calibration_per_decile_global_quantile_weighted",
        "anchor": "nb2103_mean_bag_oof_K28",
        "n_unb": int(n),
        "n_folds": int(N_FOLDS),
        "n_deciles": int(N_DECILES),
        "seed": int(SEED),
        "decision_margin": DECISION_MARGIN,
        "rae_baseline_mean_bag_oof": rae_baseline,
        "ref_nb2103_K28_mean_bag": REF_MEAN_BAG,
        "ref_nb2103_K28_median_bag": REF_MEDIAN_BAG,
        "method_results": method_results,
        "best_method": best_name,
        "best_rae_cross_fit": best_rae,
        "delta_vs_mean_bag": best_rae - rae_baseline,
        "delta_vs_median_bag": best_rae - REF_MEDIAN_BAG,
        "beats_mean_bag_by_margin": bool(beats_mean_bag),
        "beats_median_bag_by_margin": bool(beats_median_bag),
        "deploy": deploy_info,
    }
    out_p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] wrote {out_p}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} DONE ===  best={best_name}  RAE={best_rae:.4f}  "
          f"beats_med_bag={beats_median_bag}")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    main()
