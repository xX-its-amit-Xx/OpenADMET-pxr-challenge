"""nb1713 -- DEPLOY of nb1411 (3-way naive 1/3 mean) as ALT POST candidate.

This is an explicit re-deployment of nb1411 (the 3-way SHAP-pruned residual blend
of nb1373 + nb1352 + nb1364) using the residual-on-anchor decomposition.

PROTOCOL (per task spec):
    1. Load 3 deploy residuals (each built as 5-seed bag on ALL 253 unblind on
       the nb1070 anchor):
         * residual_AP    = te_nb1380_mean - te_nb1070    (nb1373 AP top-30)
         * residual_MACCS = te_nb1360_mean - te_nb1070    (nb1352 MACCS top-20)
         * residual_Mord  = te_nb1364      - te_nb1070    (nb1364 Mordred top-30)
    2. te_nb1713 = te_nb1070 + (residual_AP + residual_MACCS + residual_Mord) / 3
    3. Save submissions/nb1713_deploy_nb1411.csv

Mathematically equivalent to nb1420 (naive 1/3 mean of the three deploy vectors),
since all three deploys share the same nb1070 anchor.  Re-deploy is justified as
an ALT POST candidate so the candidate file lives in submissions/ under the
nb1713 tag for ladder bookkeeping.

POST-unblind:  LB transfer uncertain (residuals fit on 253 unblind labels).

Outputs:
    data/processed/te_nb1713.npy                    (513,) float32
    data/processed/nb1713_summary.json
    submissions/nb1713_deploy_nb1411.csv            (SMILES, Molecule Name, pEC50)
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
import pandas as pd

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1713"
ANCHOR = "nb1070"
PARENT = "nb1411"

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

# Honest LB anchors carried from nb1411 cross-fit / nb1420 deploy
HONEST_LB_ANCHOR_NAIVE = 0.5037
HONEST_LB_ANCHOR_SLSQP = 0.5045


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY {PARENT} (3-way naive 1/3 mean residual blend) "
          f"as ALT POST candidate")
    print(f"          anchor={ANCHOR}")
    print(f"          honest LB anchors: naive={HONEST_LB_ANCHOR_NAIVE}  "
          f"slsqp={HONEST_LB_ANCHOR_SLSQP}")
    print("=" * 78)

    # ---- Load test ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    else:
        cand = [c for c in te.columns if "name" in c.lower()]
        if not cand:
            raise KeyError(
                f"No Molecule Name column found in test ({te.columns.tolist()})"
            )
        mol_names = te[cand[0]].astype(str).tolist()
    print(f"[load] n_test={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Anchor (513) ----
    te_anchor_path = DATA_PROCESSED / f"te_{ANCHOR}.npy"
    if not te_anchor_path.exists():
        raise FileNotFoundError(f"Missing anchor: {te_anchor_path}")
    te_anchor = np.load(te_anchor_path).astype(np.float64)
    if te_anchor.shape[0] != n_test:
        raise ValueError(f"te_{ANCHOR} shape mismatch: {te_anchor.shape}")
    in_rae_anchor = float(rae(y_unb, te_anchor[unb_idx]))
    print(f"[anchor] te_{ANCHOR}  mean={te_anchor.mean():.4f}  "
          f"std={te_anchor.std():.4f}  "
          f"min={te_anchor.min():.4f}  max={te_anchor.max():.4f}  "
          f"in_RAE={in_rae_anchor:.4f}")

    # ---- Component deploy vectors (each = te_nb1070 + 5-seed bag residual) ----
    comp_paths = {
        "AP":    DATA_PROCESSED / "te_nb1380_mean.npy",  # nb1373 AP top-30
        "MACCS": DATA_PROCESSED / "te_nb1360_mean.npy",  # nb1352 MACCS top-20
        "Mord":  DATA_PROCESSED / "te_nb1364.npy",       # nb1364 Mordred top-30
    }
    comps: dict[str, np.ndarray] = {}
    resids: dict[str, np.ndarray] = {}
    comp_stats: dict[str, dict] = {}
    resid_stats: dict[str, dict] = {}
    in_rae_comps: dict[str, float] = {}

    print("\n" + "-" * 78)
    print("LOAD COMPONENT DEPLOY VECTORS (each = nb1070 anchor + 5-seed residual)")
    print("-" * 78)
    for tag, path in comp_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing component {tag}: {path}")
        v = np.load(path).astype(np.float64)
        if v.shape[0] != n_test:
            raise ValueError(f"{tag} shape mismatch: {v.shape}")
        comps[tag] = v
        resids[tag] = v - te_anchor
        in_rae_c = float(rae(y_unb, v[unb_idx]))
        in_rae_comps[tag] = in_rae_c
        comp_stats[tag] = {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
        }
        resid_stats[tag] = {
            "mean": float(resids[tag].mean()),
            "std": float(resids[tag].std()),
            "min": float(resids[tag].min()),
            "max": float(resids[tag].max()),
        }
        print(f"   [{tag:>5s}]  comp mean={v.mean():.4f}  std={v.std():.4f}  "
              f"in_RAE={in_rae_c:.4f}  | resid mean={resids[tag].mean():+.4f}  "
              f"std={resids[tag].std():.4f}")

    # ---- Mean residual (1/3 each) and final deploy ----
    mean_resid = (resids["AP"] + resids["MACCS"] + resids["Mord"]) / 3.0
    te_nb1713 = te_anchor + mean_resid
    in_rae_blend = float(rae(y_unb, te_nb1713[unb_idx]))

    te_stats = {
        "mean": float(te_nb1713.mean()),
        "std": float(te_nb1713.std()),
        "min": float(te_nb1713.min()),
        "max": float(te_nb1713.max()),
    }
    mean_resid_stats = {
        "mean": float(mean_resid.mean()),
        "std": float(mean_resid.std()),
        "min": float(mean_resid.min()),
        "max": float(mean_resid.max()),
    }

    print("\n" + "-" * 78)
    print("FINAL  te_nb1713 = te_nb1070 + (resid_AP + resid_MACCS + resid_Mord)/3")
    print("-" * 78)
    print(f"   mean_resid    mean={mean_resid.mean():+.4f}  "
          f"std={mean_resid.std():.4f}  "
          f"min={mean_resid.min():.4f}  max={mean_resid.max():.4f}")
    print(f"   te_nb1713     mean={te_nb1713.mean():.4f}  "
          f"std={te_nb1713.std():.4f}  "
          f"min={te_nb1713.min():.4f}  max={te_nb1713.max():.4f}")
    print(f"   in_RAE(unb)   = {in_rae_blend:.4f}")
    print(f"   honest LB anchor (naive 1/3 mean)  = {HONEST_LB_ANCHOR_NAIVE}")
    print(f"   honest LB anchor (SLSQP cross-fit) = {HONEST_LB_ANCHOR_SLSQP}")

    # ---- Save NPY ----
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb1713.astype(np.float32))
    print(f"\n[save] {te_path}")

    # ---- Save CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1411.csv"
    df = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1713.astype(np.float64),
    })
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}  rows={len(df)}  cols={list(df.columns)}")

    summary = {
        "tag": TAG,
        "parent_method": PARENT,
        "anchor": ANCHOR,
        "blend_recipe": "naive_1_3_mean_residual",
        "components": {
            "AP":    "te_nb1380_mean (nb1373 AtomPair top-30, 5-seed bag)",
            "MACCS": "te_nb1360_mean (nb1352 MACCS top-20, 5-seed bag)",
            "Mord":  "te_nb1364      (nb1364 Mordred top-30 + ChEMBL, 5-seed bag)",
        },
        "component_paths": {k: str(v) for k, v in comp_paths.items()},
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "in_rae_unb_anchor": in_rae_anchor,
        "in_rae_unb_components": in_rae_comps,
        "in_rae_unb_blend": in_rae_blend,
        "component_stats": comp_stats,
        "residual_stats": resid_stats,
        "mean_residual_stats": mean_resid_stats,
        "te_nb1713_stats": te_stats,
        "honest_lb_anchor_naive": HONEST_LB_ANCHOR_NAIVE,
        "honest_lb_anchor_slsqp": HONEST_LB_ANCHOR_SLSQP,
        "te_npy_path": str(te_path),
        "csv_path": str(csv_path),
        "post_unblind": True,
        "lb_transfer_uncertain": True,
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
        "n_test", "n_unb",
        "in_rae_unb_anchor",
        "in_rae_unb_components",
        "in_rae_unb_blend",
        "te_nb1713_stats",
        "honest_lb_anchor_naive",
        "honest_lb_anchor_slsqp",
        "csv_path",
        "te_npy_path",
    ):
        print(f"  {k}: {res.get(k)}")
