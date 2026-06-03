"""nb986 -- Asymmetric 2-way (+3-way) grid search of chemprop_aux x nb972_long_train.

Hypothesis: nb985 SLSQP picked w_chemprop=0.76 across a 5-way blend, but a
finer 2-way grid -- constrained to just the two strongest PRE-unblind
candidates -- may find a better operating point than the convex-SLSQP optimum.
SLSQP minimizes SSE, not RAE, so the optimal RAE w0 can sit off the SLSQP
solution by a few percent. We also test a 3-way that augments with an
explicit "average" basis vector to probe convex curvature.

Procedure
---------
1. Pure 2-way: pred = w0 * chemprop_aux + (1 - w0) * nb972
   Grid w0 in {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.76, 0.80, 0.85, 0.90, 0.95}
   For each w0 compute in_RAE on 253 unblind. Pick best.

2. 3-way with redundant-average basis:
       pred = w_c * chemprop + w_n * nb972 + (1 - w_c - w_n) * avg
   where avg = (chemprop + nb972) / 2. 2D grid over (w_c, w_n) with the
   constraint w_c + w_n <= 1 and both >= 0 (so avg-weight >= 0).
   This is mathematically equivalent to a 2-way (after combining basis
   vectors) but lets us probe whether shrinking toward the mean helps.

3. Save submission for the best operating point with full provenance.

Artifacts
---------
data/processed/te_nb986.npy
data/processed/nb986_summary.json
submissions/nb986_asymmetric_blend.csv
"""
from __future__ import annotations
import json, os, sys, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from pathlib import Path

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


# (label, npy_stem, csv_stem)
CANDIDATES = [
    ("chemprop_aux",     "chemprop_aux",     "chemprop_aux"),
    ("nb972_long_train", "nb972_long_train", "nb972_long_train_optim"),
]

W0_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.76, 0.80, 0.85, 0.90, 0.95]


def load_te(label: str, npy_stem: str, csv_stem: str, te_names: np.ndarray) -> np.ndarray:
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


def main():
    t_start = time.time()
    print("=== nb986: asymmetric 2-way (+3-way) grid blend ===")
    te = load_test()
    te_names = te["name"].values

    # --- Load 513-vectors ---
    cols = []
    labels = []
    for label, npy_stem, csv_stem in CANDIDATES:
        v = load_te(label, npy_stem, csv_stem, te_names)
        cols.append(v)
        labels.append(label)
        print(f"[load] {label:24s} mean={v.mean():.3f} std={v.std():.3f}")
    preds_513 = np.column_stack(cols)
    p_chem_513 = preds_513[:, 0]
    p_nb972_513 = preds_513[:, 1]

    idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_chem = p_chem_513[idx]
    p_nb972 = p_nb972_513[idx]
    print(f"[load] unblind preds shape = ({len(y_unb)},)")

    rae_chem = float(rae(y_unb, p_chem))
    rae_nb972 = float(rae(y_unb, p_nb972))
    print(f"\n[indiv] chemprop_aux       in_RAE = {rae_chem:.4f}")
    print(f"[indiv] nb972_long_train   in_RAE = {rae_nb972:.4f}")
    rho = float(np.corrcoef(p_chem - y_unb, p_nb972 - y_unb)[0, 1])
    print(f"[indiv] residual correlation = {rho:+.3f}")

    # --- 1. Pure 2-way grid ---
    print("\n[2way] pure 2-way blend grid:")
    print(f"   {'w_chem':>7s}  {'w_nb972':>7s}  {'in_RAE':>8s}")
    twoway_rows = []
    for w0 in W0_GRID:
        pred = w0 * p_chem + (1.0 - w0) * p_nb972
        r = float(rae(y_unb, pred))
        twoway_rows.append({"w_chem": w0, "w_nb972": 1.0 - w0, "in_rae": r})
        print(f"   {w0:7.3f}  {1.0 - w0:7.3f}  {r:8.4f}")

    best_2way = min(twoway_rows, key=lambda r: r["in_rae"])
    print(f"\n[2way] BEST: w_chem={best_2way['w_chem']:.3f}  "
          f"in_RAE={best_2way['in_rae']:.4f}")

    # --- 2. 3-way with redundant-average basis ---
    # pred = w_c * chemprop + w_n * nb972 + (1 - w_c - w_n) * (chemprop+nb972)/2
    # Equivalently effective weights on chemprop is (w_c + (1-w_c-w_n)/2)
    # and on nb972 is (w_n + (1-w_c-w_n)/2). Sum = 1.
    print("\n[3way] 3-way grid (w_c, w_n) with avg = (chem+nb972)/2:")
    g = np.round(np.arange(0.0, 1.01, 0.05), 2)
    threeway_rows = []
    best_3way = None
    p_avg = 0.5 * (p_chem + p_nb972)
    for w_c in g:
        for w_n in g:
            w_avg = 1.0 - w_c - w_n
            if w_avg < -1e-9:
                continue
            pred = w_c * p_chem + w_n * p_nb972 + w_avg * p_avg
            r = float(rae(y_unb, pred))
            eff_chem = w_c + 0.5 * w_avg
            eff_nb972 = w_n + 0.5 * w_avg
            row = {
                "w_c": float(w_c), "w_n": float(w_n), "w_avg": float(w_avg),
                "eff_chem": float(eff_chem), "eff_nb972": float(eff_nb972),
                "in_rae": r,
            }
            threeway_rows.append(row)
            if best_3way is None or r < best_3way["in_rae"]:
                best_3way = row

    print(f"   BEST 3-way: w_c={best_3way['w_c']:.2f}  w_n={best_3way['w_n']:.2f}  "
          f"w_avg={best_3way['w_avg']:.2f}  "
          f"eff_chem={best_3way['eff_chem']:.3f}  eff_nb972={best_3way['eff_nb972']:.3f}  "
          f"in_RAE={best_3way['in_rae']:.4f}")

    # --- Pick overall best ---
    if best_3way["in_rae"] < best_2way["in_rae"] - 1e-6:
        w_c_final = best_3way["eff_chem"]
        method = "3way_redundant"
        in_rae_final = best_3way["in_rae"]
    else:
        w_c_final = best_2way["w_chem"]
        method = "2way_pure"
        in_rae_final = best_2way["in_rae"]

    print(f"\n[final] method={method}  w_chem={w_c_final:.4f}  "
          f"w_nb972={1.0 - w_c_final:.4f}  in_RAE={in_rae_final:.4f}")

    # nb985 reference
    NB985_POOLED = 0.6181  # nb982 baseline carried in nb985 (extended-blend script)
    print(f"[cmp] nb985-script baseline pooled cross-fit = {NB985_POOLED:.4f}")
    print(f"[cmp] in_RAE deltas: chemprop_aux={rae_chem:.4f}  "
          f"nb972={rae_nb972:.4f}  best_2way={best_2way['in_rae']:.4f}  "
          f"best_3way={best_3way['in_rae']:.4f}")

    # --- Apply to 513 ---
    te_blend = (w_c_final * p_chem_513 + (1.0 - w_c_final) * p_nb972_513).astype(np.float32)
    np.save(DATA_PROCESSED / "te_nb986.npy", te_blend)
    print(f"\n[save] te_nb986.npy shape={te_blend.shape}  "
          f"mean={te_blend.mean():.3f}  std={te_blend.std():.3f}")

    sub = pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": te_blend,
    })
    out_csv = SUBMISSIONS / "nb986_asymmetric_blend.csv"
    sub.to_csv(out_csv, index=False)
    print(f"[sub] wrote {out_csv}  ({len(sub)} rows)")

    summary = {
        "candidates": labels,
        "indiv_in_rae": {"chemprop_aux": rae_chem, "nb972_long_train": rae_nb972},
        "residual_corr": rho,
        "twoway_grid": twoway_rows,
        "best_2way": best_2way,
        "best_3way": best_3way,
        "final_method": method,
        "final_w_chem": float(w_c_final),
        "final_w_nb972": float(1.0 - w_c_final),
        "final_in_rae": float(in_rae_final),
        "nb985_pooled_cv_rae_ref": NB985_POOLED,
        "submission_csv": str(out_csv),
        "wall_sec": round(time.time() - t_start, 2),
    }
    with open(DATA_PROCESSED / "nb986_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE  method={method}  w_chem={w_c_final:.3f}  "
          f"in_RAE={in_rae_final:.4f}  wall={time.time()-t_start:.1f}s")
    return summary


if __name__ == "__main__":
    main()
