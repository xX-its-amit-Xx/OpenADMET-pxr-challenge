"""nb2572 -- Per-anchor isotonic calibration then linear blend (SLSQP simplex).

NEW PARADIGM vs prior calibration work:
    - nb562 / nb2200: single scalar rank-stretch s on a single blend.
    - nb1464 / nb2504: single global isotonic on a single anchor (or per-decile).
    - nb1150 / nb2171: SLSQP simplex blend across raw anchors (no per-anchor
      calibration).
    - HERE: 3 PRE-clean anchors are EACH calibrated by their OWN
      IsotonicRegression(y_min=3.0, y_max=8.0) fold-wise. The 3 calibrated
      streams then enter a SLSQP simplex blend (w >= 0, sum = 1). Each anchor
      gets its own monotone variance-decompression map BEFORE blending,
      decoupling the per-anchor compression slope from the blend coefficients.

HYPOTHESIS:
    Different anchors have different variance-compression slopes vs truth
    (chemprop_aux is undercompressed, counter_clean is wildly overcompressed,
    K=20 sits in between). A single shared stretch / iso / blend cannot
    simultaneously decompress each axis. Per-anchor iso + simplex blend gives
    3 free monotone maps plus 2 free simplex coords = exactly enough capacity
    to capture per-axis compression and inter-axis routing separately.

SUBSTRATE (PRE-clean only):
    1. nb2240_K20    -> nb2240_mean_bag_oof_K20.npy / te_nb2240_K20.npy
    2. chemprop_aux  -> nb1133_chemprop_aux_pred_oof.npy / te_chemprop_aux.npy
    3. counter_clean -> nb2490_pred_oof.npy / te_nb2490.npy
       (nb2490 is the K=20 RFE residual on counter_clean joint anchor;
        nb730-free, anchor_pre_unblind=True)

PROTOCOL:
    - 5-fold scaffold CV on 253 unblind, 5 kf_seeds {42, 1, 7, 137, 1009}
    - Per fold per anchor: IsotonicRegression(y_min=3.0, y_max=8.0,
      out_of_bounds='clip') fit on (anchor_i[tr], y[tr])
    - Linear blend: w_1 * iso_1(a_1) + w_2 * iso_2(a_2) + w_3 * iso_3(a_3)
      via SLSQP simplex (w >= 0, sum = 1) fit on the 3 calibrated
      fold-train streams, applied to fold-val.
    - Per-seed pooled RAE; mean_rae = mean across 5 kf_seeds.
    - Deploy: refit per-anchor iso on all 253, fit deploy SLSQP simplex on
      calibrated stack, apply to 513 anchor refits.

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4601  -> MARGINAL_BEAT
    else                -> FAIL

Outputs:
    data/processed/nb2572_summary.json
    data/processed/nb2572_pred_oof.npy   (253,) float32
    data/processed/te_nb2572.npy         (513,) float32
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
from rdkit import RDLogger
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2572"

# ---- CV ----
N_FOLDS = 5
KF_SEEDS = [42, 1, 7, 137, 1009]

# ---- Isotonic clip range ----
Y_MIN = 3.0
Y_MAX = 8.0

# ---- Gate ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# ---- PRE-clean anchor pool (exactly 3, fixed-order) ----
ANCHORS = [
    ("nb2240_K20",
     DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
     DATA_PROCESSED / "te_nb2240_K20.npy",
     "PRE-clean K=20 RFE residual stack on chemprop_aux"),
    ("chemprop_aux",
     DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
     DATA_PROCESSED / "te_chemprop_aux.npy",
     "PRE-clean (4139 PRE-unblind only; verified anchor)"),
    ("counter_clean",
     DATA_PROCESSED / "nb2490_pred_oof.npy",
     DATA_PROCESSED / "te_nb2490.npy",
     "PRE-clean counter-assay K=20 residual joint anchor (nb730-free)"),
]


# ============================================================================
# helpers
# ============================================================================

def _fit_iso(x: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
    if len(y) < 2 or float(np.ptp(x)) < 1e-9:
        return None
    iso = IsotonicRegression(
        y_min=Y_MIN, y_max=Y_MAX, increasing=True, out_of_bounds="clip",
    )
    try:
        iso.fit(x, y)
    except Exception:
        return None
    return iso


def _apply_iso(iso: IsotonicRegression | None, x: np.ndarray) -> np.ndarray:
    """If degenerate fold, fall back to clipped identity (preserve rank)."""
    if iso is None:
        return np.clip(x, Y_MIN, Y_MAX).astype(np.float64)
    return iso.predict(x).astype(np.float64)


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-anchor isotonic calibration then SLSQP simplex blend")
    print("=" * 78)

    # ---- load test + unblind ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- load anchors (all 3 required; abort if any missing) ----
    anchor_names = []
    anchor_prov = {}
    oof_cols, te_cols = [], []
    for name, oof_path, te_path, prov in ANCHORS:
        if not oof_path.exists():
            raise FileNotFoundError(f"anchor {name} pred_oof missing: {oof_path}")
        if not te_path.exists():
            raise FileNotFoundError(f"anchor {name} te missing: {te_path}")
        oof = np.load(oof_path).astype(np.float64)
        te_v = np.load(te_path).astype(np.float64)
        if oof.shape[0] != n_unb:
            raise ValueError(f"{name} pred_oof shape {oof.shape} != ({n_unb},)")
        if te_v.shape[0] != n_test:
            raise ValueError(f"{name} te shape {te_v.shape} != ({n_test},)")
        anchor_names.append(name)
        anchor_prov[name] = prov
        oof_cols.append(oof)
        te_cols.append(te_v)
    K = len(anchor_names)
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)

    rae_anchors = {k: float(rae(y_unb, P_unb[:, i]))
                   for i, k in enumerate(anchor_names)}
    std_anchors = {k: float(P_unb[:, i].std()) for i, k in enumerate(anchor_names)}
    truth_std = float(y_unb.std())
    print(f"[anchors] K={K}  truth_std={truth_std:.3f}")
    for k in anchor_names:
        ratio = std_anchors[k] / truth_std if truth_std > 0 else float("nan")
        print(f"   {k:14s}  in_RAE={rae_anchors[k]:.4f}  std={std_anchors[k]:.3f}  "
              f"std/truth_std={ratio:.3f}  [{anchor_prov[k]}]")

    # ---- scaffold CV ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  kf_seeds={KF_SEEDS}  n_folds={N_FOLDS}  K_anchors={K}")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_fold_weights = []
    per_seed_fold_thresholds = []
    oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)

    for s_idx, kf_seed in enumerate(KF_SEEDS):
        t_seed = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
        fold_weights = []
        fold_thresholds = []
        for f_idx, (tr_loc, va_loc) in enumerate(splits):
            # Per-anchor isotonic on fold-train
            iso_list = []
            iso_n_thr = []
            cal_tr = np.empty((len(tr_loc), K), dtype=np.float64)
            cal_va = np.empty((len(va_loc), K), dtype=np.float64)
            for j in range(K):
                iso_j = _fit_iso(P_unb[tr_loc, j], y_unb[tr_loc])
                iso_list.append(iso_j)
                iso_n_thr.append(
                    int(len(iso_j.X_thresholds_)) if iso_j is not None else 0
                )
                cal_tr[:, j] = _apply_iso(iso_j, P_unb[tr_loc, j])
                cal_va[:, j] = _apply_iso(iso_j, P_unb[va_loc, j])
            # SLSQP simplex blend on calibrated train stack
            w = slsqp_simplex(cal_tr, y_unb[tr_loc])
            oof_blend[va_loc] = cal_va @ w
            fold_weights.append(w.tolist())
            fold_thresholds.append(iso_n_thr)

        if np.isnan(oof_blend).any():
            raise RuntimeError("OOF has NaN -- scaffold splits did not cover all rows")
        pooled = float(rae(y_unb, oof_blend))
        per_seed_pooled.append(pooled)
        per_seed_fold_weights.append(fold_weights)
        per_seed_fold_thresholds.append(fold_thresholds)
        oof_seed_stack[s_idx] = oof_blend
        mean_w = np.mean(np.asarray(fold_weights), axis=0)
        print(
            f"   kf_seed={kf_seed:5d}  pooled={pooled:.4f}  "
            f"mean_w=[" + ", ".join(f"{nm}={mw:.3f}" for nm, mw in zip(anchor_names, mean_w))
            + f"]  wall={time.time()-t_seed:.1f}s"
        )

    mean_rae = float(np.mean(per_seed_pooled))
    std_rae = float(np.std(per_seed_pooled))
    oof_final = oof_seed_stack.mean(axis=0)
    rae_mean_oof = float(rae(y_unb, oof_final))
    print(f"\n[wide-seed] mean pooled RAE = {mean_rae:.4f} +/- {std_rae:.4f} "
          f"(n_seeds={len(KF_SEEDS)})")
    print(f"[wide-seed] RAE(seed-mean OOF) = {rae_mean_oof:.4f}")
    print(f"[diag] oof_final std = {oof_final.std():.3f}  (truth {truth_std:.3f})")

    # ---- gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  "
          f"thresholds(<{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL)  "
          f"->  {verdict}")

    # ---- deploy: refit on all 253, apply to 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit per-anchor iso on all 253, fit deploy SLSQP simplex, predict 513")
    print("-" * 78)
    deploy_iso_list = []
    deploy_iso_n_thr = []
    cal_unb_full = np.empty_like(P_unb)
    cal_te_full = np.empty_like(P_te)
    for j in range(K):
        iso_j = _fit_iso(P_unb[:, j], y_unb)
        deploy_iso_list.append(iso_j)
        deploy_iso_n_thr.append(
            int(len(iso_j.X_thresholds_)) if iso_j is not None else 0
        )
        cal_unb_full[:, j] = _apply_iso(iso_j, P_unb[:, j])
        cal_te_full[:, j] = _apply_iso(iso_j, P_te[:, j])
    w_deploy = slsqp_simplex(cal_unb_full, y_unb)
    te_pred = (cal_te_full @ w_deploy).astype(np.float32)
    te_pred = np.clip(te_pred, Y_MIN, Y_MAX).astype(np.float32)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy weights = "
          + ", ".join(f"{nm}={w:.4f}" for nm, w in zip(anchor_names, w_deploy)))
    print(f"   deploy iso n_thresholds = "
          + ", ".join(f"{nm}={n}" for nm, n in zip(anchor_names, deploy_iso_n_thr)))
    print(f"   te mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}  (expected << mean_rae)")

    # ---- save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_final.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_per_anchor_iso_blend.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "per_anchor_isotonic_then_slsqp_simplex_blend",
        "anchors": anchor_names,
        "anchor_provenance": anchor_prov,
        "anchor_in_rae": rae_anchors,
        "anchor_std": std_anchors,
        "anchor_pre_unblind": True,
        "truth_std": truth_std,
        "y_min_isotonic": Y_MIN,
        "y_max_isotonic": Y_MAX,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_fold_weights": per_seed_fold_weights,
        "per_seed_fold_iso_n_thresholds": per_seed_fold_thresholds,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_oof": rae_mean_oof,
        "oof_final_std": float(oof_final.std()),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "deploy_weights": {nm: float(w) for nm, w in zip(anchor_names, w_deploy)},
        "deploy_iso_n_thresholds": {
            nm: int(n) for nm, n in zip(anchor_names, deploy_iso_n_thr)
        },
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K anchors                = {K}  ({anchor_names})")
    print(f"   per-seed pooled RAE      = "
          f"{[float('%.4f' % r) for r in per_seed_pooled]}")
    print(f"   MEAN pooled RAE (5 sd)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   RAE(seed-mean OOF)       = {rae_mean_oof:.4f}")
    print(f"   gate                     = <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL  ->  {verdict}")
    print(f"   deploy weights           = "
          + ", ".join(f"{nm}={w:.3f}" for nm, w in zip(anchor_names, w_deploy)))
    print(f"   te[unb_idx] in-sample    = {te_unb_in:.4f}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "std_rae",
        "rae_of_mean_oof",
        "verdict",
        "deploy_weights",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
