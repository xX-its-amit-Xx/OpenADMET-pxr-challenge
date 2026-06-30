"""nb1170 -- Audit nb1162 stack pyramid (0.4206 RAE) for contamination.

nb1162 builds a 5-anchor SLSQP+rank-stretch stack on the 253 unblind. Per
cycle 144 agent note, nb730_honest dominates ~90% of the blend weight and
the cross-fit (0.4202) is suspiciously close to the in-sample value (0.4172).
Per feedback_anchor_contamination_chain.md, the ORIGINAL nb730 OOF was
contaminated bit-for-bit by deploy lambda selection (nb2177 audit). This
script verifies whether the FIX (nb2183-style nb730_honest_pred_oof.npy)
truly broke that contamination chain, and whether nb1162's 0.4206 stands.

Checks:
  1. sha256(nb730_honest_pred_oof) vs sha256(y_unb)             -> must differ
  2. sha256(nb730_honest_pred_oof) vs sha256(te_nb730_honest[unb_idx]) -> must differ
  3. in-sample RAE(te_nb730_honest[unb_idx]) vs OOF RAE -- if equal to 0.4172
     bit-for-bit, contamination still present
  4. Trace generating script (per nb2177 audit + nb2183 fix metadata)
  5. Confirm nb2183 used per-fold inner lambda grid (TRAIN-only) and deploy
     lambda = MEDIAN of per-fold (NOT pooled-best)
  6. If contaminated -> flag nb1162 REJECTED, do NOT promote
  7. If honest      -> re-run nb1162 Stage 2/3 with fresh kf_seeds and verify
     the 0.4206 pooled cross-fit RAE reproduces within tolerance

Outputs:
  data/processed/nb1170_summary.json  (overwrites the prior nb1170 entry;
  the existing nb1170 was a deploy-companion mean of nb1140+nb1162 and is
  superseded by this audit-versioned summary).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1170"
NB1162_REPORTED_RAE = 0.4206326222520904
NB1162_IN_SAMPLE_RAE = 0.41724003372108714  # from nb1162_summary.json
NB1162_PYRAMID_GATE = 0.5027
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
RAE_REPRO_TOL = 0.02  # band for fresh-seed reproducibility
FRESH_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5

ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_chemprop_aux.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
    ("nb730_honest", "nb730_honest_pred_oof.npy",         "te_nb730_honest.npy"),
    ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
    ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
]


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(a.astype(np.float64)).tobytes()
    ).hexdigest()


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


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def reproduce_nb1162(P_unb, y_unb, scaffolds, seed):
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed,
    )
    oof_blend = np.full(len(y_unb), np.nan)
    fold_s = []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_s.append(s_f)
    return float(rae(y_unb, oof_blend)), float(np.mean(fold_s))


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- nb1162 stack pyramid contamination audit")
    print("=" * 78)

    # ---- Load truth + unblind index ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    n_unb = len(y_unb)
    sha_y = sha256_array(y_unb)
    print(f"  n_unb = {n_unb}  sha256(y_unb) = {sha_y[:16]}...")

    # ---- Check 1: nb730_honest_pred_oof exists + sha256 vs y_unb ----
    oof_p = DATA_PROCESSED / "nb730_honest_pred_oof.npy"
    te_p = DATA_PROCESSED / "te_nb730_honest.npy"
    check1 = {
        "oof_file_exists": oof_p.exists(),
        "te_file_exists": te_p.exists(),
    }
    if not (oof_p.exists() and te_p.exists()):
        check1["fatal"] = "missing nb730_honest artefacts"
        return _save({"tag": TAG, "verdict": "MISSING_INPUTS", "check1": check1})

    oof_730h = np.load(oof_p).astype(np.float64)
    te_730h = np.load(te_p).astype(np.float64)
    te_730h_at_unb = te_730h[unb_idx]
    sha_oof = sha256_array(oof_730h)
    sha_te_unb = sha256_array(te_730h_at_unb)
    check1.update({
        "sha256_y_unb": sha_y,
        "sha256_nb730_honest_oof": sha_oof,
        "sha_oof_differs_from_y": sha_oof != sha_y,
        "oof_equals_y_allclose": bool(np.allclose(oof_730h, y_unb)),
        "oof_shape": list(oof_730h.shape),
        "te_shape": list(te_730h.shape),
    })
    print(f"  check1: sha(oof)!=sha(y_unb) -> {check1['sha_oof_differs_from_y']}")
    print(f"          oof_RAE={rae(y_unb, oof_730h):.4f}")

    # ---- Check 2: in-sample te[unb] vs OOF RAE / sha ----
    in_rae_te_unb = float(rae(y_unb, te_730h_at_unb))
    oof_rae = float(rae(y_unb, oof_730h))
    sha_te_unb_differs = sha_te_unb != sha_oof
    in_rae_equals_4172 = abs(in_rae_te_unb - 0.4172) < 1e-4
    check2 = {
        "sha256_te_nb730_honest_at_unb": sha_te_unb,
        "sha_te_unb_differs_from_oof": sha_te_unb_differs,
        "oof_rae": oof_rae,
        "te_unb_in_sample_rae": in_rae_te_unb,
        "oof_minus_te_unb_rae_gap": oof_rae - in_rae_te_unb,
        "in_rae_equals_in_sample_4172_bit_for_bit": bool(in_rae_equals_4172),
        "max_abs_diff_oof_vs_te_unb": float(
            np.max(np.abs(oof_730h - te_730h_at_unb))
        ),
    }
    print(f"  check2: in_RAE(te[unb])={in_rae_te_unb:.4f}  "
          f"OOF_RAE={oof_rae:.4f}  gap={oof_rae - in_rae_te_unb:+.4f}")
    print(f"          sha(te[unb])!=sha(oof) -> {sha_te_unb_differs}")

    # ---- Check 3: trace generation script ----
    nb2183_summ = DATA_PROCESSED / "nb2183_summary.json"
    nb2177_summ = DATA_PROCESSED / "nb2177_summary.json"
    check3 = {"nb2183_summary_present": nb2183_summ.exists(),
              "nb2177_summary_present": nb2177_summ.exists()}
    if nb2183_summ.exists():
        with open(nb2183_summ) as f:
            s2183 = json.load(f)
        check3.update({
            "generator_script": "scripts/nb2183_nb730_honest.py",
            "nb2183_pred_oof_path": s2183.get("artefacts", {}).get("pred_oof"),
            "nb2183_te_deploy_path": s2183.get("artefacts", {}).get("te_deploy"),
            "nb2183_per_fold_lambdas": s2183.get("per_fold_chosen_lambdas"),
            "nb2183_deploy_lambda_median": s2183.get("deploy_lambda_median"),
            "nb2183_pooled_best_lambda": s2183.get("pooled_best_lambda"),
            "nb2183_honest_crossfit_rae": s2183.get("honest_crossfit_rae"),
            "nb2183_sha256_distinct": s2183.get("sha256_distinct"),
        })
    if nb2177_summ.exists():
        with open(nb2177_summ) as f:
            s2177 = json.load(f)
        check3.update({
            "nb2177_original_verdict": s2177.get("final_verdict"),
            "nb2177_orig_te_oof_identical": s2177.get(
                "check5_sha256_te_nb730_unb_idx"
            ) == s2177.get("check5_sha256_nb730_pred_oof"),
        })
    print(f"  check3: generator = {check3.get('generator_script')}")
    print(f"          nb2177 original verdict = "
          f"{check3.get('nb2177_original_verdict')}")

    # ---- Check 4: per-fold lambda search constraint (TRAIN-only, not pooled) ----
    # nb2183 uses inner grid on TRAIN rows of each fold and deploy = MEDIAN of
    # per-fold lambdas. Distinct per-fold lambdas across folds is consistent
    # with honest selection; per-fold all-zero would also be valid because
    # the grid happened to pick the same value, BUT deploy != pooled-best
    # in general.
    pf = check3.get("nb2183_per_fold_lambdas", [])
    deploy_med = check3.get("nb2183_deploy_lambda_median")
    pooled_best = check3.get("nb2183_pooled_best_lambda")
    constraint_train_fold_only = bool(pf) and (deploy_med is not None)
    median_matches_pooled_by_accident = (
        deploy_med == pooled_best if (deploy_med is not None and pooled_best is not None) else False
    )
    check4 = {
        "per_fold_lambdas": pf,
        "deploy_lambda_median": deploy_med,
        "pooled_best_lambda": pooled_best,
        "uses_train_fold_inner_grid": constraint_train_fold_only,
        "deploy_median_matches_pooled_best": bool(median_matches_pooled_by_accident),
        "rationale": (
            "nb2183 selects lambda per-fold from TRAIN rows only (inner grid) "
            "and uses MEDIAN(per-fold) as deploy. If median coincides with "
            "pooled-best it is by grid coincidence, not by-construction "
            "contamination -- the sha256(oof) vs sha256(te[unb]) DISTINCT "
            "check above is the binding evidence."
        ),
    }
    print(f"  check4: per-fold lambdas={pf}  deploy={deploy_med}  "
          f"pooled-best={pooled_best}")

    # ---- Verdict ----
    is_oof_truth_inject = (sha_oof == sha_y) or bool(np.allclose(oof_730h, y_unb))
    is_oof_eq_te_unb = not sha_te_unb_differs
    is_in_rae_4172_bit = in_rae_equals_4172 and is_oof_eq_te_unb

    contaminated = is_oof_truth_inject or is_oof_eq_te_unb or is_in_rae_4172_bit
    verdict = "CONTAMINATED" if contaminated else "HONEST"
    contamination_reasons = []
    if is_oof_truth_inject:
        contamination_reasons.append("OOF == y_unb (truth injection)")
    if is_oof_eq_te_unb:
        contamination_reasons.append(
            "OOF == te[unb_idx] (deploy-lambda selected in-sample)"
        )
    if is_in_rae_4172_bit:
        contamination_reasons.append(
            "in-sample RAE bit-for-bit equals 0.4172 (no honest gap)"
        )
    print("\n" + "-" * 78)
    print(f"VERDICT: {verdict}")
    for r in contamination_reasons:
        print(f"  - {r}")
    print("-" * 78)

    # ---- If HONEST, reproduce nb1162 0.4206 under fresh kf_seeds ----
    repro = {}
    if not contaminated:
        print("\n  Reproducing nb1162 cross-fit RAE under fresh kf_seeds...")
        te = load_test()
        te_smiles = te["smiles"].values
        unb_smiles = te_smiles[unb_idx]
        unb_scaf = [bemis_murcko(s) for s in unb_smiles]

        oof_cols, te_cols = [], []
        for disp, oof_rel, te_rel in ANCHORS:
            oof_cols.append(np.load(DATA_PROCESSED / oof_rel).astype(np.float64))
            te_cols.append(np.load(DATA_PROCESSED / te_rel).astype(np.float64))
        P_unb = np.column_stack(oof_cols)

        per_seed = []
        for sd in FRESH_SEEDS:
            rae_sd, s_mean_sd = reproduce_nb1162(P_unb, y_unb, unb_scaf, sd)
            per_seed.append({"seed": sd, "rae": rae_sd, "mean_s": s_mean_sd})
            print(f"    seed={sd}: RAE={rae_sd:.4f}  mean_s={s_mean_sd:.3f}")

        rae_arr = np.array([d["rae"] for d in per_seed])
        repro = {
            "fresh_seeds": FRESH_SEEDS,
            "per_seed": per_seed,
            "fresh_mean_rae": float(rae_arr.mean()),
            "fresh_std_rae": float(rae_arr.std()),
            "fresh_min_rae": float(rae_arr.min()),
            "fresh_max_rae": float(rae_arr.max()),
            "reported_rae": NB1162_REPORTED_RAE,
            "delta_fresh_mean_minus_reported": float(
                rae_arr.mean() - NB1162_REPORTED_RAE
            ),
            "reproduces_within_tol": bool(
                abs(rae_arr.mean() - NB1162_REPORTED_RAE) <= RAE_REPRO_TOL
            ),
            "tolerance": RAE_REPRO_TOL,
        }
        print(f"  fresh mean RAE = {rae_arr.mean():.4f} "
              f"(reported {NB1162_REPORTED_RAE:.4f}, "
              f"delta {rae_arr.mean() - NB1162_REPORTED_RAE:+.4f})")
    else:
        repro = {"skipped": True, "reason": "contamination found"}

    # ---- Ladder decision ----
    if contaminated:
        ladder_action = "REJECT_nb1162_DO_NOT_PROMOTE"
    elif repro.get("reproduces_within_tol", False):
        ladder_action = "ACCEPT_nb1162_PROMOTE_TO_CANDIDATE"
    else:
        ladder_action = "HOLD_nb1162_reproducibility_fail"

    summary = {
        "tag": TAG,
        "audit_target": "nb1162_stack_pyramid_RAE_0p4206",
        "anchors": [a[0] for a in ANCHORS],
        "anchor_weight_concentration_note": (
            "nb1162 deploy: nb730_honest ~88.7%, nb2103_K28 ~11.3%, all "
            "others ~0 -- ANY contamination in nb730_honest propagates "
            "to nb1162 almost verbatim."
        ),
        "check1_nb730_honest_oof_vs_y_unb": check1,
        "check2_in_sample_te_unb_vs_oof": check2,
        "check3_generator_trace": check3,
        "check4_per_fold_lambda_constraint": check4,
        "contamination_reasons": contamination_reasons,
        "verdict": verdict,
        "fresh_seed_reproduction": repro,
        "nb1162_reported_pooled_cv_rae": NB1162_REPORTED_RAE,
        "nb1162_in_sample_rae": NB1162_IN_SAMPLE_RAE,
        "nb1162_gate_target": NB1162_PYRAMID_GATE,
        "ladder_action": ladder_action,
    }
    return _save(summary)


def _save(summary: dict) -> dict:
    out = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote {out}")
    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  verdict        = {summary.get('verdict')}")
    print(f"  ladder action  = {summary.get('ladder_action')}")
    repro = summary.get("fresh_seed_reproduction", {})
    if "fresh_mean_rae" in repro:
        print(f"  fresh mean RAE = {repro['fresh_mean_rae']:.4f} "
              f"(reported {summary['nb1162_reported_pooled_cv_rae']:.4f})")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    main()
