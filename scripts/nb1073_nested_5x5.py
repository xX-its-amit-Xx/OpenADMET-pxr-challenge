"""nb1073 -- Nested 5x5 CV for honest s_b selection cost on te_nb1014.

Across the nb562 / nb1053 / nb1060 / nb1070 / nb1072 series, every stretch
variant tunes s_b on the SAME 253 unblind rows it is later evaluated on.
Standard 5-fold cross-fit hides one layer of optimism: the *grid choice* on
the training fold still uses 80% of the same 253 to pick s_b, then we score
on the remaining 20%. The selection-cost gap (how much s_b grid scanning
itself overfits) is invisible in single-layer CV.

Nested 5x5 makes that cost honest:

  Outer 5-fold KFold on 253 unblind:
    For each outer fold (~202 train / ~51 val):
      Inner 5-fold cross-fit on the ~202 outer-train rows:
        For each candidate s in STRETCH_GRID:
          Average inner-val RAE across 5 inner folds.
        Pick s* = argmin inner-CV-RAE.
      Apply (mu fit on full outer-train, s*) to the outer-val rows.
    Accumulate outer-val predictions -> pooled outer-CV RAE.

Hypothesis: nested CV pooled RAE is the most honest possible estimate of
deployment performance for the scalar-stretch family on this 253-row anchor.
If nested RAE > flat 5-fold RAE by >= 0.005, the grid-selection step itself
is meaningfully overfit and the flat CV numbers in nb1053/1060/1070/1072
overstate true generalisation; if the gap is <0.003 we have evidence that
flat single-layer CV is already honest at this n.

Procedure (single seed for stable comparison):
  - mu is fit on each outer-train (or inner-train) slice as p.mean(); s_b is
    a single scalar drawn from STRETCH_GRID = [0.80, 0.85, ..., 1.50].
  - inner-CV objective = sum of fold abs errors / sum of fold |y - mean(y)|
    pooled (rae) on the inner held-out fold, averaged over 5 inner folds.
  - outer pooled RAE = rae(y_unb, oof_outer).

Deploy (informational only -- nb1072 / nb1070 own deploy):
  - Refit s_global on ALL 253 via flat 5-fold (matches nb562 protocol).
  - te_nb1073.npy = mu_all + s_global * (preds_513 - mu_all).
  - The deploy submission is logged but NOT meant to overtake nb1072; the
    headline result of nb1073 is the nested-vs-flat gap.

Outputs:
  data/processed/te_nb1073.npy
  data/processed/nb1073_summary.json
  submissions/nb1073_nested_5x5.csv
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

TAG = "nb1073"
ANCHOR = "nb1014"
N_OUTER = 5
N_INNER = 5
SEED = 42
# Stretch grid covers the s<1 (shrink) and s>1 (decompress) regimes seen
# across nb1053/1060/1072 fits; coarse 0.05 step matches nb1070's bag.
STRETCH_GRID = np.round(np.arange(0.80, 1.501, 0.05), 2).tolist()

# Honest reference numbers from prior cross-fit work.
NB1014_BAGGED_HONEST_RAE = 0.5930
NB1053_HONEST_RAE = 0.5780          # per-quantile, seed 42, flat 5-fold
NB1060_BAGGED_RAE = 0.5798          # 5-seed mean bag
NB1070_MEDIAN_BAG_RAE = None        # filled if present, else left None
NB1072_BEST_LAMBDA_RAE = None


# ----------------------------------------------------------------------
# Scalar stretch helpers
# ----------------------------------------------------------------------


def stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def flat_pick_s(p_tr: np.ndarray, y_tr: np.ndarray, grid) -> tuple[float, float]:
    """Pick s on a single training slice by direct RAE minimisation.

    Used for the FLAT (single-layer) reference number and for deploy.
    Returns (best_s, best_rae) computed on the very same training rows
    (in-sample on the inner train slice).
    """
    mu = float(p_tr.mean())
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        r = float(rae(y_tr, stretch(p_tr, mu, s)))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def inner_cv_pick_s(p_outer_tr: np.ndarray, y_outer_tr: np.ndarray,
                    grid, n_inner: int, seed: int) -> tuple[float, dict]:
    """Cross-fit s* on an outer-train slice via n_inner-fold KFold.

    For each candidate s, compute the pooled inner-CV RAE (apply mu_inner_tr
    + s * (p_val - mu_inner_tr) to each inner-val fold; pool predictions
    across inner folds; rae against pooled inner-val y).  Pick argmin.
    """
    n = len(y_outer_tr)
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
    fold_splits = list(kf.split(np.arange(n)))

    per_s_pooled = {}
    per_s_fold_rae = {s: [] for s in grid}
    for s in grid:
        oof_inner = np.full(n, np.nan, dtype=np.float64)
        for tr_loc, va_loc in fold_splits:
            mu_in = float(p_outer_tr[tr_loc].mean())
            oof_inner[va_loc] = stretch(p_outer_tr[va_loc], mu_in, s)
            per_s_fold_rae[s].append(
                float(rae(y_outer_tr[va_loc], oof_inner[va_loc]))
            )
        per_s_pooled[s] = float(rae(y_outer_tr, oof_inner))
    best_s = min(grid, key=lambda s: per_s_pooled[s])
    return float(best_s), {
        "pooled_per_s": per_s_pooled,
        "fold_per_s_rae": per_s_fold_rae,
        "best_s": float(best_s),
        "best_inner_rae": per_s_pooled[best_s],
    }


# ----------------------------------------------------------------------
# Nested 5x5 driver
# ----------------------------------------------------------------------


def nested_cross_fit(p_unb: np.ndarray, y_unb: np.ndarray,
                     n_outer: int, n_inner: int,
                     grid, seed: int) -> dict:
    n = len(y_unb)
    outer_kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
    oof_outer_nested = np.full(n, np.nan, dtype=np.float64)
    oof_outer_flat = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    for k, (otr_loc, ova_loc) in enumerate(outer_kf.split(np.arange(n))):
        p_otr, y_otr = p_unb[otr_loc], y_unb[otr_loc]
        p_ova, y_ova = p_unb[ova_loc], y_unb[ova_loc]
        # Inner CV pick (nested honest path)
        s_nested, inner_info = inner_cv_pick_s(
            p_otr, y_otr, grid, n_inner=n_inner, seed=seed
        )
        # Flat single-layer reference: pick s on the full outer-train rows
        # using in-sample RAE (matches nb562 flat protocol).
        s_flat, _ = flat_pick_s(p_otr, y_otr, grid)
        # mu is fit on the outer-train rows (no leak) for BOTH variants
        mu_otr = float(p_otr.mean())
        pred_nested = stretch(p_ova, mu_otr, s_nested)
        pred_flat = stretch(p_ova, mu_otr, s_flat)
        oof_outer_nested[ova_loc] = pred_nested
        oof_outer_flat[ova_loc] = pred_flat
        rae_nested = float(rae(y_ova, pred_nested))
        rae_flat = float(rae(y_ova, pred_flat))
        fold_records.append({
            "outer_fold": k,
            "n_train": int(len(otr_loc)),
            "n_val": int(len(ova_loc)),
            "mu_train": mu_otr,
            "s_nested": s_nested,
            "s_flat": s_flat,
            "outer_val_rae_nested": rae_nested,
            "outer_val_rae_flat": rae_flat,
            "inner_pooled_per_s": inner_info["pooled_per_s"],
            "inner_best_rae": inner_info["best_inner_rae"],
        })
    pooled_nested = float(rae(y_unb, oof_outer_nested))
    pooled_flat = float(rae(y_unb, oof_outer_flat))
    return {
        "pooled_nested_rae": pooled_nested,
        "pooled_flat_rae": pooled_flat,
        "oof_outer_nested": oof_outer_nested,
        "oof_outer_flat": oof_outer_flat,
        "folds": fold_records,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _maybe_load_ref(tag: str, key: str = "pooled_cross_fit_rae"):
    path = DATA_PROCESSED / f"{tag}_summary.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            blob = json.load(f)
        return float(blob.get(key))
    except Exception:
        return None


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Nested {N_OUTER}x{N_INNER} CV on te_{ANCHOR} (scalar s_b)")
    print("=" * 78)

    nb1070_rae = _maybe_load_ref("nb1070", key="pooled_cross_fit_rae")
    nb1072_rae = _maybe_load_ref("nb1072", key="pooled_cross_fit_rae")

    te = load_test()
    te_names = te["name"].values
    preds_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    print(f"[load] te_{ANCHOR}  shape={preds_513.shape}  "
          f"mean={preds_513.mean():.3f}  std={preds_513.std():.3f}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] p_unb {p_unb.shape}  y {y_unb.shape}")
    print(f"[load] truth_std={y_unb.std():.4f}  pred_std={p_unb.std():.4f}  "
          f"(compression = {p_unb.std() / y_unb.std():.3f})")

    in_rae_anchor = float(rae(y_unb, p_unb))
    print(f"\n[ref] in_RAE anchor (s=1)        = {in_rae_anchor:.4f}")
    print(f"[ref] nb1014 bagged honest        = {NB1014_BAGGED_HONEST_RAE:.4f}")
    print(f"[ref] nb1053 honest flat 5-fold   = {NB1053_HONEST_RAE:.4f}")
    print(f"[ref] nb1060 5-seed mean bag      = {NB1060_BAGGED_RAE:.4f}")
    if nb1070_rae is not None:
        print(f"[ref] nb1070 median bag           = {nb1070_rae:.4f}")
    if nb1072_rae is not None:
        print(f"[ref] nb1072 Tikhonov best lambda = {nb1072_rae:.4f}")

    # ------------------------------------------------------------------
    # Nested 5x5
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"NESTED CV  outer={N_OUTER}, inner={N_INNER}, seed={SEED}")
    print(f"  STRETCH_GRID = {STRETCH_GRID}")
    print("-" * 78)
    res = nested_cross_fit(
        p_unb, y_unb,
        n_outer=N_OUTER, n_inner=N_INNER,
        grid=STRETCH_GRID, seed=SEED,
    )
    print()
    for f in res["folds"]:
        print(
            f"  outer {f['outer_fold']}  "
            f"n_tr={f['n_train']:3d}  n_va={f['n_val']:3d}  "
            f"s_nested={f['s_nested']:.2f}  s_flat={f['s_flat']:.2f}  "
            f"val_rae_nested={f['outer_val_rae_nested']:.4f}  "
            f"val_rae_flat={f['outer_val_rae_flat']:.4f}  "
            f"inner_best_rae={f['inner_best_rae']:.4f}"
        )

    pooled_nested = res["pooled_nested_rae"]
    pooled_flat = res["pooled_flat_rae"]
    selection_cost = pooled_nested - pooled_flat
    print()
    print(f"[result] pooled NESTED outer-CV RAE   = {pooled_nested:.4f}")
    print(f"[result] pooled FLAT  outer-CV RAE    = {pooled_flat:.4f}")
    print(f"[result] selection-cost (nested-flat) = {selection_cost:+.4f}")

    # Quick verdict for the nested-vs-flat gap.
    if selection_cost <= 0.003:
        gap_verdict = "FLAT_CV_IS_HONEST"
    elif selection_cost <= 0.01:
        gap_verdict = "MILD_SELECTION_OVERFIT"
    else:
        gap_verdict = "LARGE_SELECTION_OVERFIT"
    print(f"[verdict] {gap_verdict}")

    # ------------------------------------------------------------------
    # Deploy (informational): refit on all 253 with flat 5-fold s pick
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("DEPLOY  (flat-CV s on full 253, apply to 513)")
    print("-" * 78)
    s_deploy, _ = flat_pick_s(p_unb, y_unb, STRETCH_GRID)
    mu_deploy = float(p_unb.mean())
    deploy_513 = stretch(preds_513, mu_deploy, s_deploy).astype(np.float32)
    in_rae_deploy = float(rae(y_unb, deploy_513[unb_idx].astype(np.float64)))
    print(f"  s_deploy        = {s_deploy:.2f}")
    print(f"  mu_deploy       = {mu_deploy:.3f}")
    print(f"  in-sample 253   = {in_rae_deploy:.4f}  (overfit lower bound)")
    print(f"  te(513) mean    = {deploy_513.mean():.3f}  "
          f"std={deploy_513.std():.3f}")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_nested_5x5.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta_vs_nb1060 = pooled_nested - NB1060_BAGGED_RAE
    delta_vs_nb1053 = pooled_nested - NB1053_HONEST_RAE
    delta_vs_anchor = pooled_nested - NB1014_BAGGED_HONEST_RAE
    if pooled_nested <= NB1060_BAGGED_RAE - 0.005:
        verdict = "BEATS_NB1060"
    elif abs(delta_vs_nb1060) < 0.005:
        verdict = "TIES_NB1060"
    else:
        verdict = "WORSE_THAN_NB1060"
    print(f"\n[verdict] nested pooled vs nb1060 ({NB1060_BAGGED_RAE}): "
          f"delta={delta_vs_nb1060:+.4f}  -> {verdict}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "n_outer": N_OUTER,
        "n_inner": N_INNER,
        "seed": SEED,
        "stretch_grid": STRETCH_GRID,
        "in_rae_anchor_on_253": in_rae_anchor,
        "pooled_nested_rae": pooled_nested,
        "pooled_flat_rae": pooled_flat,
        "selection_cost_nested_minus_flat": selection_cost,
        "gap_verdict": gap_verdict,
        "s_deploy": s_deploy,
        "mu_deploy": mu_deploy,
        "in_rae_deploy_on_253": in_rae_deploy,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "anchor_te_mean": float(preds_513.mean()),
        "anchor_te_std": float(preds_513.std()),
        "nb1014_bagged_honest_rae": NB1014_BAGGED_HONEST_RAE,
        "nb1053_honest_rae": NB1053_HONEST_RAE,
        "nb1060_bagged_rae": NB1060_BAGGED_RAE,
        "nb1070_median_bag_rae": nb1070_rae,
        "nb1072_best_lambda_rae": nb1072_rae,
        "delta_vs_nb1060": delta_vs_nb1060,
        "delta_vs_nb1053": delta_vs_nb1053,
        "delta_vs_anchor_bagged": delta_vs_anchor,
        "verdict_vs_nb1060": verdict,
        "folds": [
            {k: v for k, v in f.items()
             if k != "inner_pooled_per_s"}
            for f in res["folds"]
        ],
        "fold_inner_pooled_per_s": [
            f["inner_pooled_per_s"] for f in res["folds"]
        ],
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                       = {ANCHOR}")
    print(f"   in_RAE anchor on 253         = {in_rae_anchor:.4f}")
    print(f"   pooled NESTED outer-CV RAE   = {pooled_nested:.4f}")
    print(f"   pooled FLAT  outer-CV RAE    = {pooled_flat:.4f}")
    print(f"   selection-cost (nested-flat) = {selection_cost:+.4f}")
    print(f"   gap verdict                  = {gap_verdict}")
    print(f"   s_deploy                     = {s_deploy:.2f}")
    print(f"   in-sample (deploy)           = {in_rae_deploy:.4f}")
    print(f"   delta nested vs nb1060       = {delta_vs_nb1060:+.4f}")
    print(f"   delta nested vs nb1053       = {delta_vs_nb1053:+.4f}")
    print(f"   verdict vs nb1060            = {verdict}")
    print(f"   wall                         = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "in_rae_anchor_on_253",
        "pooled_nested_rae",
        "pooled_flat_rae",
        "selection_cost_nested_minus_flat",
        "gap_verdict",
        "s_deploy",
        "in_rae_deploy_on_253",
        "delta_vs_nb1060",
        "verdict_vs_nb1060",
        "plain_submission",
    ):
        print(f"  {k}: {res.get(k)}")
