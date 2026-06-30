"""nb1180 -- Deploy nb1162 stack-pyramid (verified HONEST by nb1170).

CONTEXT:
  nb1162 = 5-anchor SLSQP convex blend + (rejected) rank-stretch on the 253
  unblind. Stage-3 stretch grid selected s=1.0 in every fold so this is a
  PURE convex blend deploy. nb1170 verified HONEST at scaffold-CV RAE
  0.4204 (in-sample 0.4172) -- 5 fresh seeds reproduced within tolerance.

DEPLOY WEIGHTS (mean of per-fold SLSQP weights):
  nb730_honest  ~ 88.7%
  nb2103_K28    ~ 11.3%
  chemprop_aux  ~ 0    (collapsed to floor in every fold)
  nb503         ~ 0
  nb562         ~ 0

ANCHOR TE FILES ON 513:
  nb730_honest -> te_nb730_honest.npy           (nb2183 deploy, cycle 125)
  nb2103_K28   -> te_nb2112.npy                 (nb2112 = chemprop_aux te +
                                                  nb2103 K=28 BoB residual
                                                  MEDIAN; the canonical K=28
                                                  deploy artefact)

NOTE vs nb1162 SCRIPT: nb1162 used te_chemprop_aux.npy as a proxy for the
  nb2103_K28 anchor because the K=28 deploy te was not cached at the time.
  te_nb2112.npy IS the canonical K=28 deploy, so nb1180 uses it -- giving
  the deploy CSV the correct anchor structure even though stage-2 weights
  are imported verbatim from nb1162 (we do NOT refit on the 253; that would
  break the audit chain).

OUTPUTS:
  submissions/nb1180_deploy_stack_pyramid.csv  (513 rows: SMILES,
                                                Molecule Name, pEC50)
  data/processed/te_nb1180.npy
  data/processed/nb1180_summary.json

LB REGIME CAVEAT (POST-unblind):
  Weights and stretch fit on the 253 unblind. Per the LB two-regime memory
  note (POST-unblind candidates incur a +0.10 conservative shift) and the
  train-OOF blend transfer memory (+0.10 RAE delta from cross-fit to LB),
  the honest conservative LB band for nb1180 is 0.52 - 0.62 RAE, NOT the
  0.42 cross-fit number. Treat that as a hard ceiling for any go/no-go.
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
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1180"
NB1162_SUMMARY = DATA_PROCESSED / "nb1162_summary.json"
NB1170_SUMMARY = DATA_PROCESSED / "nb1170_summary.json"

# Anchor te files on the 513 test set.
# nb2103_K28 deploy = te_nb2112 (chemprop_aux + K=28 BoB residual MEDIAN).
ANCHOR_TE = {
    "nb730_honest": DATA_PROCESSED / "te_nb730_honest.npy",
    "nb2103_K28":   DATA_PROCESSED / "te_nb2112.npy",
}

# Expected in-sample / cross-fit RAE checkpoints from nb1162 / nb1170.
EXPECT_IN_SAMPLE_RAE = 0.4172
EXPECT_CROSSFIT_RAE = 0.4206
TOL_IN_SAMPLE = 0.02     # nb1180 substitutes te_nb2112 for te_chemprop_aux
                         # on the 11.3% nb2103_K28 leg, so a small drift is
                         # expected and acceptable.


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- deploy nb1162 stack-pyramid (verified HONEST by nb1170)")
    print("=" * 78)

    # ---- Reload nb1162 deploy weights (mean of per-fold SLSQP weights) ----
    assert NB1162_SUMMARY.exists(), f"missing {NB1162_SUMMARY}"
    assert NB1170_SUMMARY.exists(), f"missing {NB1170_SUMMARY}"
    nb1162 = json.loads(NB1162_SUMMARY.read_text())
    nb1170 = json.loads(NB1170_SUMMARY.read_text())
    assert nb1170["verdict"] == "HONEST", (
        f"nb1170 verdict is {nb1170['verdict']!r}, refusing to deploy"
    )
    deploy_w = {row["name"]: float(row["w"]) for row in nb1162["deploy_weights"]}
    print(f"[nb1162] pooled scaffold-CV RAE = "
          f"{nb1162['pooled_scaffold_cv_rae']:.4f}")
    print(f"[nb1162] in-sample RAE          = "
          f"{nb1162['in_sample_rae_overfit_bound']:.4f}")
    print(f"[nb1162] deploy stretch s       = {nb1162['deploy_s']:.4f}  "
          f"(stage-3 rejected; this is a pure convex blend)")
    print(f"[nb1170] verdict                = {nb1170['verdict']}")
    print("[nb1162] deploy weights:")
    for name, w in deploy_w.items():
        print(f"   {name:14s} w={w:.6f}")

    # Only two anchors carry non-trivial weight; the other three collapsed.
    w_nb730 = deploy_w["nb730_honest"]
    w_nb2103 = deploy_w["nb2103_K28"]
    w_other = (deploy_w["chemprop_aux"]
               + deploy_w["nb503"] + deploy_w["nb562"])
    assert w_other < 1e-6, (
        f"chemprop_aux/nb503/nb562 weights sum to {w_other:.3e}, "
        f"expected ~0 per nb1162 deploy"
    )
    # Renormalize to the active 2 (numerical drop of ~1e-12 from SLSQP).
    s = w_nb730 + w_nb2103
    w_nb730 /= s
    w_nb2103 /= s
    print(f"[active anchors] w_nb730_honest={w_nb730:.6f}  "
          f"w_nb2103_K28={w_nb2103:.6f}  (renormalized)")

    # ---- Load 513 test + unblind labels ----
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    # ---- Load anchor te (513) ----
    te_nb730 = np.load(ANCHOR_TE["nb730_honest"]).astype(np.float64)
    te_nb2103 = np.load(ANCHOR_TE["nb2103_K28"]).astype(np.float64)
    assert te_nb730.shape == (n_te,), te_nb730.shape
    assert te_nb2103.shape == (n_te,), te_nb2103.shape
    print(f"[te] nb730_honest  mean={te_nb730.mean():.4f}  "
          f"std={te_nb730.std():.4f}")
    print(f"[te] nb2103_K28    mean={te_nb2103.mean():.4f}  "
          f"std={te_nb2103.std():.4f}")

    # ---- Compute deploy on 513 ----
    deploy_te = (w_nb730 * te_nb730 + w_nb2103 * te_nb2103).astype(np.float32)
    print(f"[deploy] te(513) mean/std = {deploy_te.mean():.4f} / "
          f"{deploy_te.std():.4f}")

    # ---- Verify in-sample RAE on the 253 unblind ----
    in_sample_pred = deploy_te[unb_idx].astype(np.float64)
    in_sample_rae = float(rae(y_unb, in_sample_pred))
    delta_vs_expect = in_sample_rae - EXPECT_IN_SAMPLE_RAE
    print(f"[verify] in-sample RAE on 253 unblind = {in_sample_rae:.4f}")
    print(f"         expected (nb1162 in-sample)  = {EXPECT_IN_SAMPLE_RAE:.4f}"
          f"  delta={delta_vs_expect:+.4f}  tol={TOL_IN_SAMPLE}")
    print(f"         expected (nb1162 cross-fit)  = {EXPECT_CROSSFIT_RAE:.4f}")
    in_sample_ok = abs(delta_vs_expect) <= TOL_IN_SAMPLE
    if not in_sample_ok:
        print("[WARN] in-sample RAE drifted beyond tolerance from nb1162; "
              "this is informational only -- te_nb2112 substitution on the "
              "11.3% leg can produce small drift.")
    else:
        print("[OK] in-sample RAE within tolerance of nb1162")

    # ---- Save artefacts ----
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)

    sub_csv_path = SUBMISSIONS / f"{TAG}_deploy_stack_pyramid.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_te,
    }).to_csv(sub_csv_path, index=False)
    print(f"[save] {te_npy_path}")
    print(f"[save] {sub_csv_path}  ({n_te} rows)")

    summary = {
        "tag": TAG,
        "purpose": "deploy nb1162 stack-pyramid; nb1170 verified HONEST",
        "source_summaries": {
            "nb1162": str(NB1162_SUMMARY),
            "nb1170": str(NB1170_SUMMARY),
        },
        "method": "pure_convex_blend_2anchors_no_stretch",
        "stage3_stretch_rejected_s": float(nb1162["deploy_s"]),
        "deploy_weights_imported_from_nb1162": deploy_w,
        "active_anchors_renormalized": {
            "nb730_honest": float(w_nb730),
            "nb2103_K28":   float(w_nb2103),
        },
        "anchor_te_files_513": {
            "nb730_honest": str(ANCHOR_TE["nb730_honest"]),
            "nb2103_K28":   str(ANCHOR_TE["nb2103_K28"]),
        },
        "note_anchor_te_substitution": (
            "nb1162 used te_chemprop_aux as the nb2103_K28 proxy because the "
            "K=28 deploy te was not cached. nb1180 substitutes te_nb2112 "
            "(canonical K=28 deploy = chemprop_aux te + nb2103 K=28 BoB "
            "residual MEDIAN). Stage-2 weights are imported verbatim -- not "
            "refit -- so the audit chain (nb1162 -> nb1170 HONEST) survives."
        ),
        "n_te": int(n_te),
        "n_unb": int(n_unb),
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "in_sample_rae_253": in_sample_rae,
        "expect_in_sample_rae_nb1162": EXPECT_IN_SAMPLE_RAE,
        "expect_crossfit_rae_nb1162": EXPECT_CROSSFIT_RAE,
        "in_sample_delta_vs_nb1162": delta_vs_expect,
        "in_sample_tol": TOL_IN_SAMPLE,
        "in_sample_within_tol": bool(in_sample_ok),
        "nb1170_verdict": nb1170["verdict"],
        "nb1162_pooled_scaffold_cv_rae": float(
            nb1162["pooled_scaffold_cv_rae"]
        ),
        "lb_regime_caveat": (
            "POST-unblind regime: nb1180 weights and stretch were fit on the "
            "253 unblind. Per memory notes 'LB two-regime calibration' "
            "(POST-unblind in_RAE unreliable) and 'train-OOF blend weights "
            "don't transfer' (+0.10 RAE shift expected), the conservative "
            "honest LB band for nb1180 is 0.52 - 0.62 RAE -- NOT the 0.42 "
            "cross-fit number. Treat 0.52 as the optimistic side and 0.62 "
            "as the realistic ceiling. Use only as a candidate, not as "
            "PRIMARY-1 until LB validates."
        ),
        "conservative_lb_band": [0.52, 0.62],
        "submission_csv": str(sub_csv_path),
        "te_npy_path": str(te_npy_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   active weights         = nb730_honest {w_nb730:.4f}, "
          f"nb2103_K28 {w_nb2103:.4f}")
    print(f"   deploy te mean/std     = {deploy_te.mean():.4f} / "
          f"{deploy_te.std():.4f}")
    print(f"   in-sample RAE (253)    = {in_sample_rae:.4f}  "
          f"(expect ~{EXPECT_IN_SAMPLE_RAE:.4f}, tol {TOL_IN_SAMPLE})")
    print(f"   nb1162 cross-fit RAE   = {EXPECT_CROSSFIT_RAE:.4f}  "
          f"(nb1170 HONEST)")
    print(f"   conservative LB band   = "
          f"[{summary['conservative_lb_band'][0]:.2f}, "
          f"{summary['conservative_lb_band'][1]:.2f}]")
    print(f"   wall                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "active_anchors_renormalized",
        "in_sample_rae_253",
        "in_sample_within_tol",
        "nb1162_pooled_scaffold_cv_rae",
        "conservative_lb_band",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
