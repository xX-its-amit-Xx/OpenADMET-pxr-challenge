"""nb1170 -- DEPLOY artifact for nb1161 naive mean blend.

nb1161 (combine residuals) finds that the naive mean of
nb1130 mean-bag (Morgan+RDKit residual) and nb1153 mean-bag
(Mordred residual) achieves honest 5-fold cross-fit RAE = 0.5600
on the 253 unblind, beating nb1153 standalone (0.5640) by -0.004
and becoming the current best honest cross-fit. This script writes
the 513-row deploy companion by averaging the deploy vectors:

    te_nb1170 = 0.5 * te_nb1140 + 0.5 * te_nb1162

where:
    te_nb1140.npy = deploy artifact for nb1130 (Morgan+RDKit residual mean-bag)
    te_nb1162.npy = deploy artifact for nb1153 (Mordred residual mean-bag)

Note on naming: the task prompt referenced ``te_nb1130.npy``; the actual
deploy companion to nb1130 is ``te_nb1140.npy`` (see nb1140 docstring).

Outputs:
  data/processed/te_nb1170.npy
  submissions/nb1170_deploy_nb1161_mean.csv  (SMILES, Molecule Name, pEC50)
  data/processed/nb1170_summary.json

Caveat (per feedback_lb_two_regime_calibration): this is a POST-unblind
deploy artifact -- in_RAE on te_nb1170[unb_idx] is in-sample and
optimistic. The LB-faithful number is the nb1161 honest cross-fit RAE
of 0.5600.
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

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1170"
P1_TAG = "nb1140"   # deploy companion for nb1130 (Morgan+RDKit residual)
P2_TAG = "nb1162"   # deploy companion for nb1153 (Mordred residual)
NB1161_HONEST_CROSSFIT_RAE = 0.5600

SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY naive mean blend of {P1_TAG} + {P2_TAG}")
    print(f"          te_nb1170 = 0.5 * te_{P1_TAG} + 0.5 * te_{P2_TAG}")
    print(f"          nb1161 honest cross-fit RAE = "
          f"{NB1161_HONEST_CROSSFIT_RAE:.4f} (LB-faithful number)")
    print("=" * 78)

    # ---- Load 513 test SMILES/names + unblind index/truth ----
    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_test = len(te_smiles)
    print(f"[load] test set: n_test = {n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    # ---- Load deploy vectors for nb1130 (nb1140) and nb1153 (nb1162) ----
    te_p1_path = DATA_PROCESSED / f"te_{P1_TAG}.npy"
    te_p2_path = DATA_PROCESSED / f"te_{P2_TAG}.npy"
    if not te_p1_path.exists():
        raise FileNotFoundError(
            f"{te_p1_path} not found (run nb1140 first -- deploy companion to nb1130)."
        )
    if not te_p2_path.exists():
        raise FileNotFoundError(
            f"{te_p2_path} not found (run nb1162 first -- deploy companion to nb1153)."
        )

    te_p1 = np.load(te_p1_path).astype(np.float64)
    te_p2 = np.load(te_p2_path).astype(np.float64)

    assert te_p1.shape == (n_test,), (
        f"te_{P1_TAG} shape {te_p1.shape} mismatch n_test={n_test}"
    )
    assert te_p2.shape == (n_test,), (
        f"te_{P2_TAG} shape {te_p2.shape} mismatch n_test={n_test}"
    )

    rae_p1_in = float(rae(y_unb, te_p1[unb_idx]))
    rae_p2_in = float(rae(y_unb, te_p2[unb_idx]))
    print(f"[load] te_{P1_TAG}: shape={te_p1.shape}  "
          f"mean={te_p1.mean():.4f}  std={te_p1.std():.4f}  "
          f"in_RAE(unb_idx)={rae_p1_in:.4f}")
    print(f"[load] te_{P2_TAG}: shape={te_p2.shape}  "
          f"mean={te_p2.mean():.4f}  std={te_p2.std():.4f}  "
          f"in_RAE(unb_idx)={rae_p2_in:.4f}")

    # ---- Naive mean blend ----
    te_nb1170 = 0.5 * te_p1 + 0.5 * te_p2

    # ---- Validate ----
    assert te_nb1170.shape == (n_test,), (
        f"te_nb1170 shape {te_nb1170.shape} != ({n_test},)"
    )
    if not np.isfinite(te_nb1170).all():
        n_bad = int((~np.isfinite(te_nb1170)).sum())
        raise ValueError(f"te_nb1170 has {n_bad} non-finite entries.")

    te_mean = float(te_nb1170.mean())
    te_std = float(te_nb1170.std())
    te_min = float(te_nb1170.min())
    te_max = float(te_nb1170.max())
    print(f"\n[blend] te_nb1170 shape={te_nb1170.shape}  "
          f"mean={te_mean:.4f}  std={te_std:.4f}  "
          f"min={te_min:.4f}  max={te_max:.4f}")

    # ---- In-sample RAE on the 253 unblind subset (optimistic) ----
    in_rae_253 = float(rae(y_unb, te_nb1170[unb_idx]))
    print("\n" + "-" * 78)
    print("IN-SAMPLE DIAGNOSTIC (optimistic)")
    print("-" * 78)
    print(f"   in_RAE(te_nb1170[unb_idx]) = {in_rae_253:.4f}  "
          f"(in-sample; biased)")
    print(f"   nb1161 honest cross-fit RAE = "
          f"{NB1161_HONEST_CROSSFIT_RAE:.4f}  (LB-faithful)")

    # ---- Save te artefact ----
    te_out = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_out, te_nb1170.astype(np.float32))
    print(f"\n[save] {te_out}")

    # ---- Save submission CSV (3 cols: SMILES, Molecule Name, pEC50) ----
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_nb1170.astype(np.float64),
    })
    # ---- Validate CSV structure ----
    assert len(sub) == 513, f"submission row count {len(sub)} != 513"
    assert list(sub.columns) == ["SMILES", "Molecule Name", "pEC50"], (
        f"unexpected columns: {list(sub.columns)}"
    )
    assert sub["pEC50"].notna().all(), "pEC50 has NaN entries"
    assert sub["pEC50"].dtype == np.float64, (
        f"pEC50 dtype {sub['pEC50'].dtype} != float64"
    )

    sub_path = os.path.join(SUBMISSIONS_DIR, f"{TAG}_deploy_nb1161_mean.csv")
    sub.to_csv(sub_path, index=False)
    print(f"[save] {sub_path}  rows={len(sub)}  cols={list(sub.columns)}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "deploy_companion_of": "nb1161_naive_mean",
        "blend_recipe": f"0.5 * te_{P1_TAG} + 0.5 * te_{P2_TAG}",
        "components": {
            "p1": {
                "tag": P1_TAG,
                "describes": "deploy companion to nb1130 (Morgan+RDKit residual mean-bag)",
                "in_rae_unb": rae_p1_in,
                "mean": float(te_p1.mean()),
                "std": float(te_p1.std()),
            },
            "p2": {
                "tag": P2_TAG,
                "describes": "deploy companion to nb1153 (Mordred residual mean-bag)",
                "in_rae_unb": rae_p2_in,
                "mean": float(te_p2.mean()),
                "std": float(te_p2.std()),
            },
        },
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "te_nb1170_mean": te_mean,
        "te_nb1170_std": te_std,
        "te_nb1170_min": te_min,
        "te_nb1170_max": te_max,
        "in_rae_253": in_rae_253,
        "nb1161_honest_crossfit_rae": NB1161_HONEST_CROSSFIT_RAE,
        "te_path": str(te_out),
        "submission_path": sub_path,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "DEPLOY artifact: naive 50/50 mean of te_nb1140 (Morgan+RDKit "
            "residual mean-bag) and te_nb1162 (Mordred residual mean-bag). "
            "in_RAE is in-sample and optimistic; the LB-faithful number is "
            "the nb1161 honest cross-fit RAE = 0.5600."
        ),
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
        "te_nb1170_mean",
        "te_nb1170_std",
        "te_nb1170_min",
        "te_nb1170_max",
        "in_rae_253",
        "nb1161_honest_crossfit_rae",
        "te_path",
        "submission_path",
    ):
        print(f"  {k}: {res.get(k)}")
