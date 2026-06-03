"""nb985 -- Extended 5-way SLSQP blend of PRE-unblind candidates.

Extends nb982 (3-way) by adding two structurally-orthogonal predictors:
    - chemprop_aux              (in_RAE 0.6216)  -- anchor
    - nb901_nr_multitask        (in_RAE 0.6765)  -- NR-family multitask
    - nb972_long_train          (in_RAE 0.6898)  -- long-train LGBM
    - nb914_persistence_homology(in_RAE 0.7062)  -- gudhi topology features
    - nb960_pseudo_self_train   (in_RAE 0.6951)  -- pseudo-label self-train

Hypothesis: nb914 (PH topology, different feature manifold) and nb960
(different training data via pseudo-labels) contribute orthogonal signal that
nb982's 3-way may not capture. SLSQP can zero out useless components.

Target: pooled 5-fold cross-fit RAE < 0.6181 (nb982 baseline).

Artifacts: data/processed/te_nb985.npy, data/processed/nb985_summary.json
Submission: submissions/nb985_extended_blend.csv
"""
from __future__ import annotations
import json, os, sys, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# Candidate name -> (npy_stem, csv_stem)
# npy_stem is checked first under data/processed/te_<stem>.npy
# csv_stem falls back to submissions/<stem>.csv with Molecule Name + pEC50
CANDIDATES = [
    ("chemprop_aux",              "chemprop_aux",              "chemprop_aux"),
    ("nb901_nr_multitask",        "nb901_nr_multitask",        "nb901_nr_multitask"),
    ("nb972_long_train",          "nb972_long_train",          "nb972_long_train_optim"),
    ("nb914_persistence_homology","nb914",                     "nb914_persistence_homology"),
    ("nb960_pseudo_self_train",   "nb960",                     "nb960_pseudo_self_train"),
]


def load_te(label: str, npy_stem: str, csv_stem: str, te_names: np.ndarray) -> np.ndarray:
    """Load a 513-vector candidate; reconstruct from submission CSV if no .npy."""
    npy = DATA_PROCESSED / f"te_{npy_stem}.npy"
    if npy.exists():
        v = np.load(npy).astype(np.float64)
        if v.shape[0] != te_names.shape[0]:
            raise ValueError(f"{label}: te_{npy_stem}.npy shape {v.shape} != {te_names.shape}")
        return v
    sub_path = SUBMISSIONS / f"{csv_stem}.csv"
    sub = pd.read_csv(sub_path)
    if not (sub["Molecule Name"].values == te_names).all():
        # try align by name lookup
        sub = sub.set_index("Molecule Name").loc[te_names].reset_index()
    return sub["pEC50"].values.astype(np.float64)


def slsqp_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Positive weights summing to 1 that minimize SSE of (P @ w - y)."""
    K = P.shape[1]
    w0 = np.full(K, 1.0 / K)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return res.x


def main():
    t_start = time.time()
    print("=== nb985: SLSQP 5-way blend (extends nb982) ===")
    te = load_test()
    te_names = te["name"].values

    # --- Load 513-vectors ---
    cols = []
    labels = []
    for label, npy_stem, csv_stem in CANDIDATES:
        v = load_te(label, npy_stem, csv_stem, te_names)
        cols.append(v)
        labels.append(label)
        print(f"[load] {label:32s} mean={v.mean():.3f} std={v.std():.3f}")
    preds_513 = np.column_stack(cols)
    print(f"[load] preds_513 shape = {preds_513.shape}")

    idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[idx]
    print(f"[load] unblind preds shape = {P_unb.shape}, y shape = {y_unb.shape}")

    # --- Individual in_RAEs (sanity) ---
    print("\n[indiv] in_RAE on 253 unblind:")
    indiv = {}
    for j, c in enumerate(labels):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv[c] = r
        print(f"   {c:32s}: {r:.4f}")

    # --- Residual correlations ---
    print("\n[corr] residual correlation matrix:")
    R = P_unb - y_unb[:, None]
    cc = np.corrcoef(R.T)
    hdr = "  ".join(f"{c[:10]:>10s}" for c in labels)
    print(f"   {'':32s} {hdr}")
    for i, ci in enumerate(labels):
        row = "  ".join(f"{cc[i, j]:+10.3f}" for j in range(len(labels)))
        print(f"   {ci:32s} {row}")

    # --- 5-fold cross-fit SLSQP ---
    print("\n[cv] 5-fold cross-fit SLSQP on 253 unblind:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y_unb)
    fold_weights = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(len(y_unb)))):
        w_k = slsqp_weights(P_unb[tr_loc], y_unb[tr_loc])
        oof[va_loc] = P_unb[va_loc] @ w_k
        rae_k = rae(y_unb[va_loc], oof[va_loc])
        fold_weights.append(w_k.tolist())
        wstr = ", ".join(f"{w:.3f}" for w in w_k)
        print(f"   fold {k}: w=[{wstr}]  val_RAE={rae_k:.4f}")

    pooled_rae = float(rae(y_unb, oof))
    print(f"\n[cv] pooled 5-fold cross-fit RAE = {pooled_rae:.4f}")

    # --- DEPLOY weights (single SLSQP on all 253) ---
    w_deploy = slsqp_weights(P_unb, y_unb)
    print("\n[deploy] weights from full 253:")
    for c, w in zip(labels, w_deploy):
        print(f"   {c:32s}: {w:.4f}")
    in_rae_deploy = float(rae(y_unb, P_unb @ w_deploy))
    print(f"[deploy] in-sample RAE (overfit lower bound) = {in_rae_deploy:.4f}")

    # --- Compare to nb982 ---
    NB982_POOLED = 0.6181
    delta = pooled_rae - NB982_POOLED
    sign = "BEATS" if delta < 0 else "WORSE"
    print(f"\n[cmp] nb982 pooled cross-fit = {NB982_POOLED:.4f}")
    print(f"[cmp] nb985 pooled cross-fit = {pooled_rae:.4f}  ({sign} by {abs(delta):.4f})")

    # --- Apply to 513 test set ---
    te_blend = (preds_513 @ w_deploy).astype(np.float32)
    np.save(DATA_PROCESSED / "te_nb985.npy", te_blend)
    print(f"[save] te_nb985.npy shape = {te_blend.shape}, "
          f"mean={te_blend.mean():.3f} std={te_blend.std():.3f}")

    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": te_blend,
    })
    out_csv = SUBMISSIONS / "nb985_extended_blend.csv"
    sub.to_csv(out_csv, index=False)
    print(f"[sub] wrote {out_csv}  ({len(sub)} rows)")

    # --- Summary ---
    summary = {
        "candidates": labels,
        "indiv_in_rae": indiv,
        "residual_corr": cc.tolist(),
        "fold_weights": fold_weights,
        "pooled_cv_rae": pooled_rae,
        "nb982_pooled_cv_rae": NB982_POOLED,
        "delta_vs_nb982": delta,
        "beats_nb982": bool(delta < 0),
        "deploy_weights": dict(zip(labels, w_deploy.tolist())),
        "in_sample_rae_overfit_bound": in_rae_deploy,
        "wall_sec": round(time.time() - t_start, 2),
    }
    with open(DATA_PROCESSED / "nb985_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE  pooled_cv_RAE={pooled_rae:.4f}  "
          f"deploy_w={dict(zip(labels, [round(w, 3) for w in w_deploy]))}  "
          f"wall={time.time()-t_start:.1f}s")
    return summary


if __name__ == "__main__":
    main()
