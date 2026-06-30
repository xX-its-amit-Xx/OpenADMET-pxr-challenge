"""nb2504 -- Per-decile isotonic regression on K=20 anchor.

DIFFERENT from prior rank-stretch / global isotonic:
    - Prior: single global IsotonicRegression on whole 253 (e.g. nb1464),
      or single scalar rank-stretch s (e.g. nb562, nb2200).
    - HERE: split fold-train predictions into 10 deciles by anchor value,
      fit a SEPARATE IsotonicRegression(y_min=3.0, y_max=8.0) on each
      decile (per-bin monotone map over conditional residual subgroups).
    - At inference, each val/test row is routed to its anchor decile
      (decile-boundary lookup) and calibrated with that decile's
      isotonic function.

The hypothesis: global isotonic flattens to scalar shrink on n=253;
per-decile isotonic adds 10 piecewise monotone shapes -- each capturing
local quantile-specific variance compression -- without sharing the
shrink coefficient across the predicted-pEC50 axis.  If the conditional
residual structure varies by anchor value (e.g. compressed at high
predicted pEC50 = potency tail, normal at mid), per-decile regression
captures it where global isotonic cannot.

PROTOCOL (exact):
    1. anchor = nb2240_mean_bag_oof_K20.npy on 253.
    2. 5-fold scaffold CV outer, 5 kf_seeds {1001..1005}.
    3. For each fold (tr, va):
         a. Compute decile edges from anchor[tr] (np.quantile q=0..1 step 0.1)
         b. Assign tr rows to deciles via np.digitize on those edges
         c. For each decile d in 0..9:
              if n_tr_decile >= 2 and y has spread:
                  fit IsotonicRegression(y_min=3.0, y_max=8.0,
                                         out_of_bounds='clip')
                                  on (anchor[tr][d], y[tr][d])
              else:
                  fallback = identity (return anchor value clipped)
         d. Predict val rows: for each va row, find its decile by
            digitizing anchor[va] against the SAME edges; route to
            that decile's isotonic; predict.
    4. Pool val predictions for each kf_seed -> per-seed RAE.
    5. mean_rae = mean over 5 kf_seeds.
    6. Deploy te: refit on ALL 253 (same protocol but single global
       partition), apply to 513 chemprop_aux anchor refit.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4601  -> "MARGINAL_BEAT"
    else               -> "FAIL"

Outputs:
    data/processed/nb2504_summary.json
    data/processed/nb2504_pred_oof.npy     (253,) float32
    data/processed/te_nb2504.npy           (513,) float32
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
from sklearn.isotonic import IsotonicRegression

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2504"

# ------------------------------ paths --------------------------------
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"  # 513 deploy refit
# Fallback if te_nb2240_K20.npy is not available
ANCHOR_TE_PATH_FALLBACK = DATA_PROCESSED / "te_nb2240.npy"

# ------------------------------ knobs --------------------------------
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5
N_DECILES = 10
Y_MIN = 3.0
Y_MAX = 8.0

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601


# ============================================================================
# per-decile isotonic helpers
# ============================================================================

def _build_decile_edges(values: np.ndarray) -> np.ndarray:
    """Return 9 interior cut points (deciles 10%..90%).

    np.digitize with these edges yields bins 0..9 (10 bins total).
    """
    # interior quantiles only -- digitize handles the open ends
    qs = np.linspace(0.1, 0.9, N_DECILES - 1)
    edges = np.quantile(values, qs)
    # ensure strictly increasing to avoid digitize collapse on ties
    edges = np.maximum.accumulate(edges)
    return edges.astype(np.float64)


def _assign_deciles(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return integer decile index in 0..9 for each row."""
    return np.digitize(values, edges, right=False).astype(np.int32)


def _fit_one_decile_iso(anchor_d: np.ndarray, y_d: np.ndarray):
    """Fit a per-decile isotonic regression, or return None if degenerate."""
    if len(y_d) < 2:
        return None
    if float(np.ptp(anchor_d)) < 1e-9:
        # all train anchors in this decile are identical -> no monotone slope
        return None
    if float(np.ptp(y_d)) < 1e-9:
        # all train y identical -> constant map; still build for clip
        pass
    iso = IsotonicRegression(
        y_min=Y_MIN, y_max=Y_MAX, increasing=True, out_of_bounds="clip"
    )
    try:
        iso.fit(anchor_d, y_d)
    except Exception:
        return None
    return iso


def _predict_per_decile(
    anchor_va: np.ndarray,
    edges: np.ndarray,
    per_decile_iso: dict,
    anchor_global_mean: float,
) -> np.ndarray:
    """Route each va row by anchor decile -> per-decile isotonic.

    Fallback for empty deciles: clip-to-(Y_MIN, Y_MAX) of anchor.
    """
    bins = _assign_deciles(anchor_va, edges)
    out = np.empty(len(anchor_va), dtype=np.float64)
    for i, b in enumerate(bins):
        iso = per_decile_iso.get(int(b))
        if iso is None:
            # fallback: identity-with-clip on the anchor value
            out[i] = float(np.clip(anchor_va[i], Y_MIN, Y_MAX))
        else:
            out[i] = float(iso.predict(np.array([anchor_va[i]]))[0])
    return out


# ============================================================================
# scaffold-CV per-decile isotonic
# ============================================================================

def scaffold_cv_per_decile(anchor: np.ndarray, y: np.ndarray,
                           scaffolds, kf_seeds, n_folds):
    """Returns per-kf-seed pooled RAE + bag-mean OOF prediction."""
    n = len(y)
    per_seed_oofs = []
    per_seed_rae = []
    per_seed_diag = []
    for kf_seed in kf_seeds:
        splits = scaffold_kfold_indices(
            scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n, np.nan, dtype=np.float64)
        fold_decile_counts = []
        fold_n_thresholds = []
        for tr_loc, va_loc in splits:
            edges = _build_decile_edges(anchor[tr_loc])
            tr_bins = _assign_deciles(anchor[tr_loc], edges)
            per_decile_iso = {}
            per_decile_n_train = {}
            per_decile_n_thr = {}
            for d in range(N_DECILES):
                mask_d = tr_bins == d
                n_d = int(mask_d.sum())
                per_decile_n_train[d] = n_d
                if n_d == 0:
                    per_decile_iso[d] = None
                    per_decile_n_thr[d] = 0
                    continue
                iso = _fit_one_decile_iso(anchor[tr_loc][mask_d],
                                          y[tr_loc][mask_d])
                per_decile_iso[d] = iso
                per_decile_n_thr[d] = (
                    int(len(iso.X_thresholds_)) if iso is not None else 0
                )
            fold_decile_counts.append(per_decile_n_train)
            fold_n_thresholds.append(per_decile_n_thr)
            val_pred = _predict_per_decile(
                anchor[va_loc], edges, per_decile_iso,
                anchor_global_mean=float(anchor[tr_loc].mean()),
            )
            oof[va_loc] = val_pred
        if np.isnan(oof).any():
            raise RuntimeError("OOF has NaN -- scaffold splits did not cover all rows")
        seed_rae = float(rae(y, oof))
        per_seed_oofs.append(oof)
        per_seed_rae.append(seed_rae)
        per_seed_diag.append({
            "kf_seed": kf_seed,
            "rae": seed_rae,
            "per_fold_decile_n_train": [
                {str(k): int(v) for k, v in d.items()} for d in fold_decile_counts
            ],
            "per_fold_decile_n_thresholds": [
                {str(k): int(v) for k, v in d.items()} for d in fold_n_thresholds
            ],
        })
    stack = np.column_stack(per_seed_oofs)
    mean_oof = stack.mean(axis=1)
    return per_seed_rae, mean_oof, per_seed_diag


def deploy_te_per_decile(anchor_unb: np.ndarray, y_unb: np.ndarray,
                         anchor_te: np.ndarray) -> np.ndarray:
    """Refit per-decile isotonic on ALL 253 then apply to 513 deploy anchor."""
    edges = _build_decile_edges(anchor_unb)
    tr_bins = _assign_deciles(anchor_unb, edges)
    per_decile_iso = {}
    for d in range(N_DECILES):
        mask_d = tr_bins == d
        if int(mask_d.sum()) == 0:
            per_decile_iso[d] = None
            continue
        per_decile_iso[d] = _fit_one_decile_iso(
            anchor_unb[mask_d], y_unb[mask_d]
        )
    te_pred = _predict_per_decile(
        anchor_te, edges, per_decile_iso,
        anchor_global_mean=float(anchor_unb.mean()),
    )
    return te_pred.astype(np.float32), edges, per_decile_iso


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-decile isotonic regression on K=20 anchor")
    print("=" * 78)

    # ---- truth ----
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- anchor (K=20 OOF) ----
    if not ANCHOR_OOF_PATH.exists():
        raise FileNotFoundError(ANCHOR_OOF_PATH)
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"anchor shape mismatch: {anchor_oof.shape}  expected ({n_unb},)"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] anchor nb2240_K20 OOF in_RAE = {rae_anchor:.4f}")
    print(f"[diag] anchor mean={anchor_oof.mean():.3f}  std={anchor_oof.std():.3f}  "
          f"truth std={y_unb.std():.3f}  "
          f"ratio={anchor_oof.std()/y_unb.std():.3f}")
    print(f"[diag] anchor range=[{anchor_oof.min():.3f}, {anchor_oof.max():.3f}]  "
          f"truth range=[{y_unb.min():.3f}, {y_unb.max():.3f}]")

    # ---- deploy anchor on 513 ----
    if ANCHOR_TE_PATH.exists():
        anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
        anchor_te_src = str(ANCHOR_TE_PATH)
    elif ANCHOR_TE_PATH_FALLBACK.exists():
        anchor_te = np.load(ANCHOR_TE_PATH_FALLBACK).astype(np.float64)
        anchor_te_src = str(ANCHOR_TE_PATH_FALLBACK)
        print(f"[warn] te_nb2240_K20.npy missing, using fallback {ANCHOR_TE_PATH_FALLBACK.name}")
    else:
        raise FileNotFoundError(
            f"No K=20 deploy te found: tried {ANCHOR_TE_PATH} and "
            f"{ANCHOR_TE_PATH_FALLBACK}"
        )
    if anchor_te.shape[0] != n_test:
        raise ValueError(
            f"anchor_te shape mismatch: {anchor_te.shape}  expected ({n_test},)"
        )
    print(f"[load] deploy anchor te = {anchor_te_src}")
    print(f"[diag] te anchor mean={anchor_te.mean():.3f}  "
          f"std={anchor_te.std():.3f}")

    # ---- scaffold-CV per-decile isotonic ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  kf_seeds={KF_SEEDS}  n_folds={N_FOLDS}  "
          f"n_deciles={N_DECILES}")
    print("-" * 78)
    per_seed_rae, mean_oof, per_seed_diag = scaffold_cv_per_decile(
        anchor_oof, y_unb, unb_scaffolds, KF_SEEDS, N_FOLDS,
    )
    for ks, r in zip(KF_SEEDS, per_seed_rae):
        print(f"   kf_seed={ks}  pooled_RAE={r:.4f}")
    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    rae_mean_oof = float(rae(y_unb, mean_oof))
    print(f"\n[bag] mean_rae across {len(KF_SEEDS)} kf_seeds = "
          f"{mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"[bag] rae(mean_oof)                            = "
          f"{rae_mean_oof:.4f}")
    print(f"[diag] mean_oof std = {mean_oof.std():.3f}  "
          f"(anchor {anchor_oof.std():.3f}, truth {y_unb.std():.3f})")

    # ---- gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE}/<{GATE_MARGINAL})  verdict={verdict}")

    # ---- deploy te ----
    print("\n[deploy] refitting per-decile isotonic on full 253 "
          "for 513 prediction...")
    te_pred, deploy_edges, deploy_iso = deploy_te_per_decile(
        anchor_oof, y_unb, anchor_te
    )
    te_unb_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"[deploy] te_unb_rae(in-sample)={te_unb_rae:.4f}  "
          f"te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")

    deploy_decile_counts = {
        int(d): int((_assign_deciles(anchor_oof, deploy_edges) == d).sum())
        for d in range(N_DECILES)
    }
    deploy_decile_n_thr = {
        int(d): (int(len(deploy_iso[d].X_thresholds_))
                 if deploy_iso[d] is not None else 0)
        for d in range(N_DECILES)
    }

    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "per_decile_isotonic_K20_anchor_nb2240",
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_oof_in_rae_253": rae_anchor,
        "anchor_oof_mean": float(anchor_oof.mean()),
        "anchor_oof_std": float(anchor_oof.std()),
        "anchor_te_path": anchor_te_src,
        "anchor_te_mean": float(anchor_te.mean()),
        "anchor_te_std": float(anchor_te.std()),
        "truth_mean": float(y_unb.mean()),
        "truth_std": float(y_unb.std()),
        "anchor_pre_unblind": False,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_deciles": N_DECILES,
        "y_min_isotonic": Y_MIN,
        "y_max_isotonic": Y_MAX,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "per_kf_seed_rae": [float(x) for x in per_seed_rae],
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_oof": rae_mean_oof,
        "mean_oof_std": float(mean_oof.std()),
        "per_seed_diag": per_seed_diag,
        "deploy_decile_edges": [float(e) for e in deploy_edges],
        "deploy_decile_n_train": deploy_decile_counts,
        "deploy_decile_n_thresholds": deploy_decile_n_thr,
        "gate_promote_below": GATE_PROMOTE,
        "gate_marginal_below": GATE_MARGINAL,
        "verdict": verdict,
        "te_unb_rae_in_sample": te_unb_rae,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor (nb2240 K=20) in_RAE    = {rae_anchor:.4f}")
    print(f"   per-kf-seed RAE                = "
          f"{[float('%.4f' % r) for r in per_seed_rae]}")
    print(f"   MEAN RAE (5 kf_seeds)          = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   RAE of mean-OOF                = {rae_mean_oof:.4f}")
    print(f"   gate thresholds                = <{GATE_PROMOTE} PROMOTE | "
          f"<{GATE_MARGINAL} MARGINAL")
    print(f"   verdict                        = {verdict}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "anchor_oof_in_rae_253",
        "mean_rae",
        "std_rae",
        "rae_of_mean_oof",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
