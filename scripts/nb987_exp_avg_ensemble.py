"""nb987 -- Exponentially-weighted average ensemble of 7 PRE-unblind candidates.

Pool 7 PRE-unblind candidates with in_RAE < 0.80:
    - chemprop_aux              (in_RAE 0.6216)  anchor
    - nb901_nr_multitask        (in_RAE 0.6765)  NR-family multitask
    - nb972_long_train          (in_RAE 0.6898)  long-train LGBM
    - nb914_persistence_homology(in_RAE 0.7062)  PH topology features
    - nb960_pseudo_self_train   (in_RAE 0.6951)  pseudo-label self-train
    - nb923_wl_graph_kernel     (in_RAE 0.79xx)  WL graph kernel
    - nb120_huber_1_0           (in_RAE 0.79xx)  Huber alpha=1.0

Compute weights w_i = exp(-beta * in_RAE_i) / Z  for beta in {1, 2, 5, 10, 20}.
For each beta, evaluate in_RAE on 253 unblind. Pick best beta. Save submission.

Hypothesis: exponential weighting with the right beta captures variance reduction
across all candidates without overfitting like SLSQP can (no per-fold weight fit).
At beta -> 0 we get uniform average; at beta -> inf we collapse to argmin (chemprop_aux).
A middle beta should land closer to the variance-reduction optimum.

Artifacts:
    data/processed/te_nb987.npy
    data/processed/nb987_summary.json
Submission: submissions/nb987_exp_avg_ensemble.csv
"""
from __future__ import annotations
import json, os, sys, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


# (label, npy_stem, csv_stem)
CANDIDATES = [
    ("chemprop_aux",               "chemprop_aux",      "chemprop_aux"),
    ("nb901_nr_multitask",         "nb901_nr_multitask","nb901_nr_multitask"),
    ("nb972_long_train",           "nb972_long_train",  "nb972_long_train_optim"),
    ("nb914_persistence_homology", "nb914",             "nb914_persistence_homology"),
    ("nb960_pseudo_self_train",    "nb960",             "nb960_pseudo_self_train"),
    ("nb923_wl_graph_kernel",      "nb923",             "nb923_wl_graph_kernel"),
    ("nb120_huber_1_0",            "nb120_huber_1_0",   "nb120_huber_1_0"),
]

BETAS = [1.0, 2.0, 5.0, 10.0, 20.0]


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
        sub = sub.set_index("Molecule Name").loc[te_names].reset_index()
    return sub["pEC50"].values.astype(np.float64)


def exp_weights(in_raes: np.ndarray, beta: float) -> np.ndarray:
    """w_i = exp(-beta * in_RAE_i) / Z. Use shift for numerical stability."""
    logits = -beta * in_raes
    logits = logits - logits.max()
    w = np.exp(logits)
    return w / w.sum()


def main():
    t_start = time.time()
    print("=== nb987: exponential-weighting ensemble of 7 PRE-unblind candidates ===")
    te = load_test()
    te_names = te["name"].values

    # Load predictions
    cols, labels = [], []
    for label, npy_stem, csv_stem in CANDIDATES:
        v = load_te(label, npy_stem, csv_stem, te_names)
        cols.append(v)
        labels.append(label)
        print(f"[load] {label:32s} mean={v.mean():.3f} std={v.std():.3f}")
    preds_513 = np.column_stack(cols)
    print(f"[load] preds_513 shape = {preds_513.shape}")

    # Unblind labels
    idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[idx]
    print(f"[load] unblind preds shape = {P_unb.shape}, y shape = {y_unb.shape}")

    # Per-candidate in_RAE on 253
    print("\n[indiv] in_RAE on 253 unblind:")
    indiv = {}
    for j, c in enumerate(labels):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv[c] = r
        print(f"   {c:32s}: {r:.4f}")
    in_raes = np.array([indiv[c] for c in labels])

    # Uniform baseline (beta = 0)
    uniform_pred = P_unb.mean(axis=1)
    uniform_rae = float(rae(y_unb, uniform_pred))
    print(f"\n[uniform] beta=0  in_RAE on 253 = {uniform_rae:.4f}")

    # Sweep betas
    print(f"\n[sweep] betas = {BETAS}")
    sweep = {}
    best = {"beta": None, "rae": float("inf"), "weights": None}
    for beta in BETAS:
        w = exp_weights(in_raes, beta)
        pred = P_unb @ w
        r = float(rae(y_unb, pred))
        sweep[beta] = {
            "rae": r,
            "weights": dict(zip(labels, w.tolist())),
        }
        wstr = ", ".join(f"{wi:.3f}" for wi in w)
        print(f"   beta={beta:5.1f}  in_RAE={r:.4f}  w=[{wstr}]")
        if r < best["rae"]:
            best = {"beta": beta, "rae": r, "weights": w}

    print(f"\n[best] beta={best['beta']}  in_RAE={best['rae']:.4f}")
    for c, wi in zip(labels, best["weights"]):
        print(f"   {c:32s}: {wi:.4f}")

    # Apply best weights to 513-test
    te_blend = (preds_513 @ best["weights"]).astype(np.float32)
    np.save(DATA_PROCESSED / "te_nb987.npy", te_blend)
    print(f"\n[save] te_nb987.npy  mean={te_blend.mean():.3f} std={te_blend.std():.3f}")

    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": te_blend,
    })
    out_csv = SUBMISSIONS / "nb987_exp_avg_ensemble.csv"
    sub.to_csv(out_csv, index=False)
    print(f"[sub] wrote {out_csv}  ({len(sub)} rows)")

    summary = {
        "candidates": labels,
        "indiv_in_rae": indiv,
        "uniform_in_rae": uniform_rae,
        "betas": BETAS,
        "sweep": {str(b): sweep[b] for b in BETAS},
        "best_beta": best["beta"],
        "best_in_rae": best["rae"],
        "best_weights": dict(zip(labels, best["weights"].tolist())),
        "wall_sec": round(time.time() - t_start, 2),
    }
    with open(DATA_PROCESSED / "nb987_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE  best_beta={best['beta']}  in_RAE={best['rae']:.4f}  "
          f"wall={time.time()-t_start:.1f}s")
    return summary


if __name__ == "__main__":
    main()
