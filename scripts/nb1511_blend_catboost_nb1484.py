"""nb1511 -- Blend nb1501 CatBoost + nb1484 (cross-fit weight blend).

HYPOTHESIS:
    nb1501 (CatBoost mean-bag on the 4-way 112-col pruned matrix, anchored
    on chemprop_aux te[unb_idx]) reaches RAE 0.5223 on the 253-vector
    PRE-unblind slice -- effectively flat vs nb1484 (0.5231) whose
    LGBM-Huber residual-stack also targets the chemprop_aux residual on
    the same 4-family pruned matrix.  Pearson(nb1501 mean-bag vs nb1484
    best) is reported in nb1501 summary as 0.9887, so they are
    near-collinear but not identical -- the loss shape (CatBoost-MAE vs
    LGBM-Huber) and SLSQP fold splitting introduce real residual delta.

    Test whether a 2-way convex blend on the 253 PRE-unblind slice can
    push under nb1501 (0.5223) by 0.003 margin.

PROTOCOL:
    1.  Load nb1501_mean_bag_oof.npy and nb1484_best_oof.npy (each 253-vec
        on the PRE-unblind slice).
    2.  Pred Pearson + residual Pearson (a-y vs b-y) for orthogonality.
    3.  1D grid w in {0.00, 0.05, ..., 1.00} for blend
            p = w * nb1501 + (1 - w) * nb1484
        and tabulate RAE.
    4.  5-fold cross-fit SLSQP convex blend (w >= 0, sum-to-1) on the
        2-vec matrix for an honest blend RAE.
    5.  Verdict at 0.003 margin vs nb1501 (0.5223) and nb1484 (0.5231).

Outputs:
    scripts/nb1511_blend_catboost_nb1484.py        (this file)
    data/processed/nb1511_summary.json
    data/processed/nb1511_best_oof.npy             (253,) float32
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1511"

NB1501_REF = 0.5223  # nb1501 CatBoost mean-bag PRE-unblind (per spec)
NB1484_REF = 0.5231  # nb1484 LGBM-Huber 4way SLSQP best PRE-unblind (per spec)
DECISION_MARGIN = 0.003

GRID_STEP = 0.05
SLSQP_FOLDS = 5
SLSQP_SEED = 42

INPUTS = [
    ("A_nb1501", DATA_PROCESSED / "nb1501_mean_bag_oof.npy"),  # CatBoost
    ("B_nb1484", DATA_PROCESSED / "nb1484_best_oof.npy"),      # LGBM-Huber
]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if den < 1e-12:
        return float("nan")
    return float((a * b).sum() / den)


def _slsqp_one_fold(preds_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """SLSQP: minimize RAE on training fold; weights sum-to-1 nonnegative."""
    K = preds_tr.shape[1]

    def loss(w):
        p = preds_tr @ w
        return float(rae(y_tr, p))

    w0 = np.full(K, 1.0 / K, dtype=np.float64)
    cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
    bounds = [(0.0, 1.0)] * K
    res = minimize(loss, w0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 200, "ftol": 1e-10})
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    if s < 1e-12:
        w = np.full(K, 1.0 / K)
    else:
        w = w / s
    return w


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- blend nb1501 CatBoost + nb1484 (cross-fit weight blend)")
    print(f"   inputs: {[name for name, _ in INPUTS]}")
    print(f"   ref nb1501 = {NB1501_REF:.4f}   "
          f"ref nb1484 = {NB1484_REF:.4f}   margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb = {n_unb}")

    # ---- Load OOFs ----
    arrays = []
    raes_solo = {}
    for name, path in INPUTS:
        if not path.exists():
            raise FileNotFoundError(f"missing input: {path}")
        a = np.load(path).astype(np.float64).ravel()
        if a.shape[0] != n_unb:
            raise ValueError(f"{name} shape {a.shape} != {n_unb}")
        arrays.append(a)
        r = float(rae(y_unb, a))
        raes_solo[name] = r
        print(f"[load] {name:<10s} from {path.name:<35s}  RAE = {r:.4f}")

    A = arrays[0]  # nb1501
    B = arrays[1]  # nb1484
    P = np.stack([A, B], axis=1)  # (253, 2)
    names = [name for name, _ in INPUTS]

    # ---- Pearson diagnostics ----
    print("\n" + "-" * 78)
    print("PEARSON DIAGNOSTICS")
    print("-" * 78)
    pred_pearson = _pearson(A, B)
    res_a = A - y_unb
    res_b = B - y_unb
    resid_pearson = _pearson(res_a, res_b)
    print(f"   pred  Pearson(A nb1501, B nb1484)     r = {pred_pearson:+.4f}")
    print(f"   resid Pearson(A-y,    B-y)            r = {resid_pearson:+.4f}")
    print(f"   solo RAE   A nb1501 = {raes_solo['A_nb1501']:.4f}")
    print(f"   solo RAE   B nb1484 = {raes_solo['B_nb1484']:.4f}")

    # ---- 1D grid w in {0..1 step 0.05} ----
    print("\n" + "-" * 78)
    print(f"1D GRID  w in {{0..1 step {GRID_STEP}}}  blend = w*nb1501 + (1-w)*nb1484")
    print("-" * 78)
    ws = np.round(np.arange(0.0, 1.0 + 1e-9, GRID_STEP), 4)
    grid_records = []
    for w in ws:
        pred = w * A + (1.0 - w) * B
        r = float(rae(y_unb, pred))
        grid_records.append({"w": float(w), "rae": r})
    # print full table
    print("       w        RAE")
    for rec in grid_records:
        print(f"     {rec['w']:.2f}     {rec['rae']:.4f}")
    grid_records_sorted = sorted(grid_records, key=lambda d: d["rae"])
    grid_best = grid_records_sorted[0]
    w_grid_best = float(grid_best["w"])
    pred_grid_best = w_grid_best * A + (1.0 - w_grid_best) * B
    rae_grid_best = float(grid_best["rae"])
    print(f"\n   GRID BEST: w = {w_grid_best:.2f}   RAE = {rae_grid_best:.4f}")
    print("   top-5 (sorted):")
    for rec in grid_records_sorted[:5]:
        print(f"     w = {rec['w']:.2f}  RAE = {rec['rae']:.4f}")

    # ---- 5-fold cross-fit SLSQP ----
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT SLSQP  seed = {SLSQP_SEED}")
    print("-" * 78)
    kf = KFold(n_splits=SLSQP_FOLDS, shuffle=True, random_state=SLSQP_SEED)
    cf_oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records = []
    for fi, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w_star = _slsqp_one_fold(P[tr_loc], y_unb[tr_loc])
        pred_va = P[va_loc] @ w_star
        cf_oof[va_loc] = pred_va
        rae_tr = float(rae(y_unb[tr_loc], P[tr_loc] @ w_star))
        rae_va = float(rae(y_unb[va_loc], pred_va))
        fold_records.append({
            "fold": int(fi),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "w": [float(x) for x in w_star],
            "rae_tr": rae_tr,
            "rae_va": rae_va,
        })
        print(f"   fold {fi}: w = ({w_star[0]:.3f}, {w_star[1]:.3f})  "
              f"rae_tr = {rae_tr:.4f}  rae_va = {rae_va:.4f}")
    rae_slsqp_cf = float(rae(y_unb, cf_oof))
    w_mean = np.mean([fr["w"] for fr in fold_records], axis=0).tolist()
    w_std = np.std([fr["w"] for fr in fold_records], axis=0).tolist()
    print(f"   SLSQP cross-fit RAE = {rae_slsqp_cf:.4f}")
    print(f"   w_mean = ({w_mean[0]:.3f}, {w_mean[1]:.3f})")
    print(f"   w_std  = ({w_std[0]:.3f}, {w_std[1]:.3f})")

    # ---- Pick best variant ----
    candidates = {
        "grid_best": (rae_grid_best, pred_grid_best),
        "slsqp_5fold": (rae_slsqp_cf, cf_oof),
    }
    best_tag = min(candidates, key=lambda k: candidates[k][0])
    rae_best, oof_best = candidates[best_tag]

    # ---- Verdict ----
    delta_grid_vs_nb1501 = rae_grid_best - NB1501_REF
    delta_slsqp_vs_nb1501 = rae_slsqp_cf - NB1501_REF
    delta_best_vs_nb1501 = rae_best - NB1501_REF
    delta_grid_vs_nb1484 = rae_grid_best - NB1484_REF
    delta_slsqp_vs_nb1484 = rae_slsqp_cf - NB1484_REF
    delta_best_vs_nb1484 = rae_best - NB1484_REF

    beats_nb1501_grid = rae_grid_best < NB1501_REF - DECISION_MARGIN
    beats_nb1501_slsqp = rae_slsqp_cf < NB1501_REF - DECISION_MARGIN
    beats_nb1501_best = rae_best < NB1501_REF - DECISION_MARGIN
    flat_vs_nb1501_best = abs(rae_best - NB1501_REF) < DECISION_MARGIN

    beats_nb1484_grid = rae_grid_best < NB1484_REF - DECISION_MARGIN
    beats_nb1484_slsqp = rae_slsqp_cf < NB1484_REF - DECISION_MARGIN
    beats_nb1484_best = rae_best < NB1484_REF - DECISION_MARGIN
    flat_vs_nb1484_best = abs(rae_best - NB1484_REF) < DECISION_MARGIN

    if beats_nb1501_best:
        verdict = f"NB1511_BLEND_BEATS_NB1501_NEW_PRIMARY_via_{best_tag}"
    elif flat_vs_nb1501_best:
        verdict = (f"NB1511_BLEND_FLAT_VS_NB1501  ({best_tag} "
                   f"RAE {rae_best:.4f})")
    else:
        verdict = (f"NB1511_BLEND_HURTS_NB1501  ({best_tag} "
                   f"RAE {rae_best:.4f})")

    # PRE-unblind cleanliness: both inputs anchored on chemprop_aux te
    # (PRE-unblind), and the blend is computed on the 253 slice -- in-sample
    # for grid, cross-fit for SLSQP.
    pre_unblind_clean = True

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   solo RAE: {raes_solo}")
    print(f"   grid_best    RAE = {rae_grid_best:.4f}  "
          f"vs nb1501 d = {delta_grid_vs_nb1501:+.4f}  "
          f"vs nb1484 d = {delta_grid_vs_nb1484:+.4f}")
    print(f"   slsqp_5fold  RAE = {rae_slsqp_cf:.4f}  "
          f"vs nb1501 d = {delta_slsqp_vs_nb1501:+.4f}  "
          f"vs nb1484 d = {delta_slsqp_vs_nb1484:+.4f}")
    print(f"   BEST = {best_tag}   RAE = {rae_best:.4f}")
    print(f"   beats_nb1501_best  = {beats_nb1501_best}    "
          f"flat_vs_nb1501_best  = {flat_vs_nb1501_best}")
    print(f"   beats_nb1484_best  = {beats_nb1484_best}    "
          f"flat_vs_nb1484_best  = {flat_vs_nb1484_best}")
    print(f"   verdict = {verdict}")
    print(f"   pre_unblind_clean = {pre_unblind_clean}")

    # ---- Save ----
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy",
            oof_best.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}")

    summary = {
        "tag": TAG,
        "inputs": [
            {"name": name, "path": str(path), "rae_solo": raes_solo[name]}
            for name, path in INPUTS
        ],
        "n_unb": int(n_unb),
        "pred_pearson_A_vs_B": float(pred_pearson),
        "resid_pearson_A_vs_B": float(resid_pearson),
        "grid_step": GRID_STEP,
        "grid_records": grid_records,
        "grid_top5_sorted": grid_records_sorted[:5],
        "grid_best_w_nb1501": float(w_grid_best),
        "grid_best_w_nb1484": float(1.0 - w_grid_best),
        "grid_best_rae": float(rae_grid_best),
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "slsqp_fold_records": fold_records,
        "slsqp_w_mean": w_mean,
        "slsqp_w_std": w_std,
        "slsqp_crossfit_rae": float(rae_slsqp_cf),
        "candidate_rae": {k: float(v[0]) for k, v in candidates.items()},
        "best_variant": best_tag,
        "rae_best": float(rae_best),
        "nb1501_ref": NB1501_REF,
        "nb1484_ref": NB1484_REF,
        "decision_margin": DECISION_MARGIN,
        "delta_grid_vs_nb1501": float(delta_grid_vs_nb1501),
        "delta_slsqp_vs_nb1501": float(delta_slsqp_vs_nb1501),
        "delta_best_vs_nb1501": float(delta_best_vs_nb1501),
        "delta_grid_vs_nb1484": float(delta_grid_vs_nb1484),
        "delta_slsqp_vs_nb1484": float(delta_slsqp_vs_nb1484),
        "delta_best_vs_nb1484": float(delta_best_vs_nb1484),
        "beats_nb1501_grid": bool(beats_nb1501_grid),
        "beats_nb1501_slsqp": bool(beats_nb1501_slsqp),
        "beats_nb1501_best": bool(beats_nb1501_best),
        "flat_vs_nb1501_best": bool(flat_vs_nb1501_best),
        "beats_nb1484_grid": bool(beats_nb1484_grid),
        "beats_nb1484_slsqp": bool(beats_nb1484_slsqp),
        "beats_nb1484_best": bool(beats_nb1484_best),
        "flat_vs_nb1484_best": bool(flat_vs_nb1484_best),
        "verdict": verdict,
        "pre_unblind_clean": pre_unblind_clean,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_unb",
        "pred_pearson_A_vs_B",
        "resid_pearson_A_vs_B",
        "grid_best_w_nb1501",
        "grid_best_w_nb1484",
        "grid_best_rae",
        "slsqp_crossfit_rae",
        "best_variant",
        "rae_best",
        "delta_best_vs_nb1501",
        "delta_best_vs_nb1484",
        "beats_nb1501_best",
        "flat_vs_nb1501_best",
        "beats_nb1484_best",
        "verdict",
        "pre_unblind_clean",
    ):
        print(f"  {k}: {res.get(k)}")
