"""nb989 -- 2-PARAM AFFINE RECALIBRATION on te_chemprop_aux.

Hypothesis: chemprop_aux may carry a small linear-shift bias that a simple
2-parameter affine `pred = a * chemprop_aux + b` can correct, beyond what
the 1-parameter scalar rank-stretch can capture.

Procedure (per user spec):
  1) Load te_chemprop_aux.npy (513 deploy preds).
  2) 2-param affine grid:
       a in {0.7, 0.8, 0.9, 1.0, 1.1, 1.2}
       b in {-0.5, -0.25, 0, +0.25, +0.5}
  3) For each (a, b): compute pred = a*chemprop_aux + b, then in_RAE on
     the 253 unblind subset.
  4) Pick best (a, b); save submission CSV + te_nb989.npy + summary JSON.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

TAG = "nb989"
ANCHOR = "chemprop_aux"
A_GRID = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
B_GRID = [-0.5, -0.25, 0.0, 0.25, 0.5]


def _affine(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * p + b


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- 2-PARAM AFFINE RECALIB on {ANCHOR}")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        f"te_{ANCHOR}.npy":
            DATA_PROCESSED / f"te_{ANCHOR}.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- Indices ----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- Load anchor ----
    te_anchor = np.load(
        DATA_PROCESSED / f"te_{ANCHOR}.npy"
    ).astype(np.float64)
    assert te_anchor.shape == (n_te,), f"te shape {te_anchor.shape}"

    in_pred = te_anchor[unb_idx]
    rae_anchor_in = float(rae(unb_y, in_pred))
    truth_std = float(unb_y.std())
    truth_mean = float(unb_y.mean())
    pred_std_in = float(in_pred.std())
    pred_mean_in = float(in_pred.mean())
    pred_std_full = float(te_anchor.std())
    pred_mean_full = float(te_anchor.mean())
    print(f"\n{ANCHOR} in_RAE(253)         = {rae_anchor_in:.4f}")
    print(f"{ANCHOR} te mean/std (513)    = "
          f"{pred_mean_full:.3f} / {pred_std_full:.3f}")
    print(f"{ANCHOR} te[unb] mean/std     = "
          f"{pred_mean_in:.3f} / {pred_std_in:.3f}")
    print(f"truth mean/std (253)         = "
          f"{truth_mean:.3f} / {truth_std:.3f}")
    print(f"mean-shift (truth - pred_in) = "
          f"{truth_mean - pred_mean_in:+.4f}")
    print(f"variance-ratio (truth/pred_in)= "
          f"{truth_std / max(pred_std_in, 1e-9):.4f}")

    # =================================================================
    # Grid sweep, in-sample on 253 (per spec)
    # =================================================================
    print("\n" + "-" * 78)
    print(f"GRID AFFINE IN-SAMPLE ON 253  "
          f"|A|={len(A_GRID)}  |B|={len(B_GRID)}  "
          f"total={len(A_GRID) * len(B_GRID)}")
    print("-" * 78)

    rows = []
    best_a, best_b, best_rae = 1.0, 0.0, float("inf")
    best_full = te_anchor.copy()

    for a in A_GRID:
        for b in B_GRID:
            full = _affine(te_anchor, a, b)
            r = float(rae(unb_y, full[unb_idx]))
            rows.append({
                "a": float(a), "b": float(b),
                "in_rae_253": r,
                "te_mean": float(full.mean()),
                "te_std":  float(full.std()),
            })
            flag = ""
            if r < best_rae:
                best_rae = r
                best_a = float(a)
                best_b = float(b)
                best_full = full
                flag = "  <- best"
            print(f"  a={a:.2f}  b={b:+.2f}  in_RAE={r:.4f}  "
                  f"te_std={full.std():.3f}{flag}")

    sweep_df = pd.DataFrame(rows)
    sweep_path = DATA_PROCESSED / f"{TAG}_affine_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nWrote sweep: {sweep_path}")

    # =================================================================
    # Decide
    # =================================================================
    deploy = best_full.astype(np.float32)
    print("\n" + "=" * 78)
    print("DEPLOY")
    print("=" * 78)
    print(f"  {ANCHOR} (a=1,b=0) in_RAE       = {rae_anchor_in:.4f}")
    print(f"  best (a, b)                     = "
          f"({best_a:.2f}, {best_b:+.2f})")
    print(f"  best in_RAE                     = {best_rae:.4f}")
    print(f"  improvement vs (1,0)            = "
          f"{rae_anchor_in - best_rae:+.4f}")
    print(f"  deploy te mean/std              = "
          f"{deploy.mean():.3f} / {deploy.std():.3f}")

    # =================================================================
    # Save artefacts
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)

    plain = (
        SUBMISSIONS
        / f"{TAG}_{ANCHOR}_affine_a{best_a:.2f}_b{best_b:+.2f}.csv"
    )
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)
    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {plain}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_in_rae_253": rae_anchor_in,
        "anchor_te_mean": pred_mean_full,
        "anchor_te_std": pred_std_full,
        "anchor_in_pred_mean": pred_mean_in,
        "anchor_in_pred_std": pred_std_in,
        "truth_mean": truth_mean,
        "truth_std": truth_std,
        "a_grid": A_GRID,
        "b_grid": B_GRID,
        "sweep": rows,
        "best_a": float(best_a),
        "best_b": float(best_b),
        "best_in_rae": float(best_rae),
        "delta_vs_anchor": float(rae_anchor_in - best_rae),
        "deploy_te_mean": float(deploy.mean()),
        "deploy_te_std": float(deploy.std()),
        "beats_anchor": bool(best_rae < rae_anchor_in),
        "plain_submission": str(plain),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  anchor                    = {ANCHOR}")
    print(f"  anchor in_RAE (a=1,b=0)   = {rae_anchor_in:.4f}")
    print(f"  best (a, b)               = "
          f"({best_a:.2f}, {best_b:+.2f})")
    print(f"  best in_RAE               = {best_rae:.4f}")
    print(f"  delta                     = {rae_anchor_in - best_rae:+.4f}")
    print(f"  beats anchor              = {best_rae < rae_anchor_in}")
    print("=" * 78)

    return {
        "success": True,
        "anchor": ANCHOR,
        "anchor_in_rae": rae_anchor_in,
        "best_a": float(best_a),
        "best_b": float(best_b),
        "best_in_rae": float(best_rae),
        "beats_anchor": bool(best_rae < rae_anchor_in),
        "plain_submission": str(plain),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
