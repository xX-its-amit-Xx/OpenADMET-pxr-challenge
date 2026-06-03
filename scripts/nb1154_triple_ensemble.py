"""nb1154 -- Triple-OOF SLSQP cross-fit blend.

Three OOF predictors (all on the 253 unblind):
  0. nb1014   -- chemprop_aux + nb972 deep+stretch bagged (baseline)
                 (nb1133_nb1014_pred_oof.npy)
  1. nb1130   -- residual-corrected mean-bag of nb1014 family
                 (nb1130_mean_bag_oof.npy)
  2. nb1143   -- per-quantile-bin median bag on top of bag anchors
                 (nb1143_pred_oof.npy)

Protocol:
  KFold(n_splits=5, shuffle=True, random_state=0) on the 253 unblind.
  For each fold f:
    - SLSQP fit w = (w0, w1, w2) on the simplex (sum=1, w_i in [0,1])
      minimizing SSE(P_tr @ w - y_tr) on the 4 train folds.
    - Apply w to held-out fold: oof[va] = P_va @ w.
  Pooled cross-fit RAE = rae(y_unb, oof).

Hypothesis: each subsequent predictor adds a small orthogonal correction
on top of the previous (baseline -> residual -> per-quantile). If
orthogonality is real, SLSQP picks a non-trivial mix and pooled RAE
strictly improves on nb1143's standalone 0.5649. Likely outcome: SLSQP
collapses to ~100% nb1143 since it dominates the individual RAE column.

Outputs:
  data/processed/nb1154_summary.json
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1154"
OOF_FILES = [
    ("nb1014", "nb1133_nb1014_pred_oof.npy"),
    ("nb1130_mean_bag", "nb1130_mean_bag_oof.npy"),
    ("nb1143", "nb1143_pred_oof.npy"),
]
K = len(OOF_FILES)
N_FOLDS = 5
SEED = 0

# Honest anchor references.
NB1143_STANDALONE_RAE = 0.5649


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SLSQP fit of K weights on the simplex (sum=1, w_i in [0,1])
    minimizing SSE(P @ w - y). Returns w (K,)."""
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)
    cons = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},)
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        w0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = float(w.sum())
    if s <= 0.0:
        return np.full(k, 1.0 / k)
    return w / s


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- triple-OOF SLSQP cross-fit blend (K={K}, KFold={N_FOLDS}, "
          f"seed={SEED})")
    print("=" * 78)

    # ---- Load 253 unblind truth ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape = {y_unb.shape}  "
          f"mean={y_unb.mean():.3f}  std={y_unb.std():.3f}")

    # ---- Load three OOFs ----
    P = np.zeros((n_unb, K), dtype=np.float64)
    labels = []
    for j, (lab, fname) in enumerate(OOF_FILES):
        path = DATA_PROCESSED / fname
        arr = np.load(path).astype(np.float64)
        assert arr.shape == (n_unb,), f"{fname} shape {arr.shape} != ({n_unb},)"
        P[:, j] = arr
        labels.append(lab)
        print(f"   oof[{j}] {lab:20s}: mean={arr.mean():.3f}  std={arr.std():.3f}")

    # ---- Individual in-RAE on 253 ----
    indiv_rae = {}
    print("\n[indiv] standalone RAE on 253 unblind:")
    for j, lab in enumerate(labels):
        r = float(rae(y_unb, P[:, j]))
        indiv_rae[lab] = r
        print(f"   {lab:20s}: {r:.4f}")

    # ---- Pairwise correlations (orthogonality check) ----
    print("\n[corr] Pearson corr matrix (OOF columns):")
    cor = np.corrcoef(P.T)
    for j in range(K):
        row = "  ".join(f"{cor[j, k]:+.3f}" for k in range(K))
        print(f"   {labels[j]:20s}: {row}")

    # ---- Residual-orthogonality matrix: corr of (P_j - y) ----
    res_mat = P - y_unb[:, None]
    res_cor = np.corrcoef(res_mat.T)
    print("\n[corr] Pearson corr matrix (residuals P_j - y):")
    for j in range(K):
        row = "  ".join(f"{res_cor[j, k]:+.3f}" for k in range(K))
        print(f"   {labels[j]:20s}: {row}")

    # ---- 5-fold SLSQP cross-fit ----
    print("\n" + "-" * 78)
    print(f"SLSQP cross-fit (KFold n_splits={N_FOLDS}, seed={SEED})")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    all_w: list[list[float]] = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w_f = slsqp_simplex(P[tr_loc], y_unb[tr_loc])
        all_w.append(w_f.tolist())
        # Train RAE under this w.
        tr_blend = P[tr_loc] @ w_f
        tr_rae = float(rae(y_unb[tr_loc], tr_blend))
        # Apply to validation fold.
        oof[va_loc] = P[va_loc] @ w_f
        va_rae = float(rae(y_unb[va_loc], oof[va_loc]))
        fold_records.append({
            "fold": k,
            "w": w_f.tolist(),
            "train_rae": tr_rae,
            "val_rae": va_rae,
            "n_va": int(len(va_loc)),
        })
        w_str = ",".join(f"{x:.3f}" for x in w_f)
        print(f"   fold {k}: w=[{w_str}]  tr_RAE={tr_rae:.4f}  "
              f"va_RAE={va_rae:.4f}  n_va={len(va_loc)}")

    pooled_rae = float(rae(y_unb, oof))
    all_w_arr = np.array(all_w)
    mean_w = all_w_arr.mean(axis=0)
    std_w = all_w_arr.std(axis=0)

    print(f"\n[blend] pooled cross-fit RAE = {pooled_rae:.4f}")
    print(f"[blend] mean weights across {N_FOLDS} folds:")
    for j, lab in enumerate(labels):
        print(f"   w[{j}] {lab:20s}: mean={mean_w[j]:.3f}  std={std_w[j]:.3f}")

    # ---- Verdict vs nb1143 standalone ----
    delta_vs_nb1143 = pooled_rae - NB1143_STANDALONE_RAE
    beats_nb1143 = pooled_rae < NB1143_STANDALONE_RAE
    if delta_vs_nb1143 < -0.005:
        verdict = "BEATS_NB1143"
    elif abs(delta_vs_nb1143) <= 0.005:
        verdict = "TIES_NB1143"
    else:
        verdict = "WORSE_THAN_NB1143"

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1143 standalone   = {NB1143_STANDALONE_RAE:.4f}")
    print(f"   triple SLSQP blend  = {pooled_rae:.4f}")
    print(f"   delta vs nb1143     = {delta_vs_nb1143:+.4f}  -> {verdict}")

    # ---- Deploy weights = mean over folds (renormalized) ----
    deploy_w = mean_w / float(mean_w.sum())

    summary = {
        "tag": TAG,
        "oof_files": [{"label": lab, "file": fn} for lab, fn in OOF_FILES],
        "K": K,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "n_unb": n_unb,
        "indiv_in_rae_on_253": indiv_rae,
        "pred_corr": cor.tolist(),
        "residual_corr": res_cor.tolist(),
        "fold_records": fold_records,
        "mean_w": mean_w.tolist(),
        "std_w": std_w.tolist(),
        "deploy_w": deploy_w.tolist(),
        "pooled_cross_fit_rae": pooled_rae,
        "anchor_nb1143_standalone": NB1143_STANDALONE_RAE,
        "delta_vs_nb1143": delta_vs_nb1143,
        "beats_nb1143": bool(beats_nb1143),
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "indiv_in_rae_on_253",
        "mean_w",
        "deploy_w",
        "pooled_cross_fit_rae",
        "anchor_nb1143_standalone",
        "delta_vs_nb1143",
        "beats_nb1143",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
