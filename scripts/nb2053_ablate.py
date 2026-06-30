"""nb2053 -- Ablate nb1162 anchors one at a time; identify dominant signal carrier.

nb1162 has 5 anchors (5-anchor stacking pyramid):
    0. nb2103_K28      data/processed/nb2103_mean_bag_oof_K28.npy
    1. chemprop_aux    data/processed/nb1133_chemprop_aux_pred_oof.npy
    2. nb730_honest    data/processed/nb730_honest_pred_oof.npy
    3. nb503           data/processed/nb503_pred_oof.npy
    4. nb562           data/processed/nb562_pred_oof.npy

Plan:
    (a) BASELINE: 5-anchor SLSQP simplex under scaffold 5-fold CV
        (mirrors nb1162 weights & RAE; sanity check).
    (b) LEAVE-ONE-OUT (5 runs): drop each anchor in turn, re-run SLSQP
        simplex on the remaining 4, compute pooled scaffold-CV RAE.
        The anchor whose removal causes the largest RAE jump is the
        dominant signal carrier.
    (c) TOP-3 STRESS: drop the 2 weakest anchors (by baseline deploy
        weight) and re-run SLSQP simplex on the remaining 3.
        Confirms that the dominant trio carries the signal.

Cycle-145 audit reported nb730_honest at ~88% deploy weight; we expect
its removal to cause the largest jump. nb730 is a POST-unblind anchor
(trained on labels that include the 253 unblind), so this ablation
calibrates how much of the nb1162 honest cross-fit RAE is actually
in-sample optimism rather than transferable signal.

Outputs:
    scripts/nb2053_ablate.py (this file)
    data/processed/nb2053_summary.json
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2053"
N_FOLDS = 5
SEED = 42

# Mirror nb1162 anchor order EXACTLY so weight indices line up.
ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
    ("nb730_honest", "nb730_honest_pred_oof.npy"),
    ("nb503",        "nb503_pred_oof.npy"),
    ("nb562",        "nb562_pred_oof.npy"),
]


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convex blend weights minimizing SSE of (P @ w) vs y."""
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


def cv_slsqp_blend(P: np.ndarray, y: np.ndarray, scaffolds: list,
                   n_folds: int = N_FOLDS, seed: int = SEED
                   ) -> tuple[float, np.ndarray, list]:
    """Run scaffold k-fold CV with SLSQP simplex. Return (pooled_rae,
    deploy_weights_on_all, per-fold weight list)."""
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=n_folds, shuffle=True, seed=seed,
    )
    n = len(y)
    oof = np.full(n, np.nan)
    fold_w = []
    for k, (tr_loc, va_loc) in enumerate(splits):
        w_f = slsqp_simplex(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w_f
        fold_w.append([float(x) for x in w_f])
    pooled_rae = float(rae(y, oof))
    w_deploy = slsqp_simplex(P, y)
    return pooled_rae, w_deploy, fold_w


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ablate nb1162 anchors (LOO + top-3)")
    print("=" * 78)

    # ---- Load 253 unblind targets + scaffolds ----
    te = load_test()
    te_smiles = te["smiles"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    print(f"[load] n_unb={n_unb}  scaffolds={len(set(unb_scaffolds))}")

    # ---- Load 5 anchor OOFs ----
    P_cols, indiv_rae = [], {}
    for disp, oof_rel in ANCHORS:
        oof = np.load(DATA_PROCESSED / oof_rel).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} oof shape {oof.shape}"
        indiv_rae[disp] = float(rae(y_unb, oof))
        P_cols.append(oof)
        print(f"   {disp:14s} oof_RAE={indiv_rae[disp]:.4f}")
    P_all = np.column_stack(P_cols)
    names = [a[0] for a in ANCHORS]

    # =================================================================
    # (a) BASELINE: 5-anchor SLSQP simplex
    # =================================================================
    print("\n" + "-" * 78)
    print("BASELINE (5 anchors, mirrors nb1162)")
    print("-" * 78)
    base_rae, base_w, _ = cv_slsqp_blend(P_all, y_unb, unb_scaffolds)
    print(f"   pooled scaffold-CV RAE = {base_rae:.4f}")
    print(f"   deploy weights         = "
          + ", ".join(f"{n}={w:.4f}" for n, w in zip(names, base_w)))

    # =================================================================
    # (b) LEAVE-ONE-OUT: drop each anchor in turn
    # =================================================================
    print("\n" + "-" * 78)
    print("LEAVE-ONE-OUT (5 runs, 4-anchor SLSQP each)")
    print("-" * 78)
    loo_rows = []
    for i, drop_name in enumerate(names):
        keep_idx = [j for j in range(5) if j != i]
        keep_names = [names[j] for j in keep_idx]
        P_loo = P_all[:, keep_idx]
        loo_rae, loo_w, _ = cv_slsqp_blend(P_loo, y_unb, unb_scaffolds)
        delta = loo_rae - base_rae
        w_dict = {kn: float(w) for kn, w in zip(keep_names, loo_w)}
        loo_rows.append({
            "dropped": drop_name,
            "kept": keep_names,
            "pooled_rae": loo_rae,
            "delta_rae_vs_baseline": delta,
            "deploy_weights": w_dict,
        })
        print(f"   drop {drop_name:14s}  RAE={loo_rae:.4f}  "
              f"delta={delta:+.4f}  w_kept="
              + ", ".join(f"{n}={w:.3f}" for n, w in w_dict.items()))

    # Rank by delta_rae (largest jump = dominant carrier)
    loo_sorted = sorted(
        loo_rows, key=lambda r: r["delta_rae_vs_baseline"], reverse=True,
    )
    dominant = loo_sorted[0]["dropped"]
    dominant_delta = loo_sorted[0]["delta_rae_vs_baseline"]
    print(f"\n[verdict] dominant signal carrier = {dominant}  "
          f"(drop -> delta RAE = {dominant_delta:+.4f})")
    if dominant == "nb730_honest":
        print(f"[verdict] CONFIRMS cycle-145 audit: nb730_honest carries "
              f"~{base_w[2]*100:.0f}% deploy weight; POST-unblind risk LIVE.")

    # =================================================================
    # (c) TOP-3 STRESS: drop the 2 weakest anchors by baseline deploy w
    # =================================================================
    print("\n" + "-" * 78)
    print("TOP-3 STRESS (drop 2 weakest by baseline deploy weight)")
    print("-" * 78)
    w_rank = sorted(
        list(enumerate(base_w)), key=lambda x: x[1],
    )
    drop2_idx = [w_rank[0][0], w_rank[1][0]]
    drop2_names = [names[j] for j in drop2_idx]
    keep3_idx = [j for j in range(5) if j not in drop2_idx]
    keep3_names = [names[j] for j in keep3_idx]
    P_top3 = P_all[:, keep3_idx]
    top3_rae, top3_w, _ = cv_slsqp_blend(P_top3, y_unb, unb_scaffolds)
    top3_delta = top3_rae - base_rae
    top3_wdict = {kn: float(w) for kn, w in zip(keep3_names, top3_w)}
    print(f"   dropped weakest 2 = {drop2_names}")
    print(f"   kept top 3        = {keep3_names}")
    print(f"   pooled RAE        = {top3_rae:.4f}  delta={top3_delta:+.4f}")
    print(f"   top-3 weights     = "
          + ", ".join(f"{n}={w:.4f}" for n, w in top3_wdict.items()))

    # =================================================================
    # Summary table (concise)
    # =================================================================
    print("\n" + "=" * 78)
    print("ABLATION TABLE")
    print("=" * 78)
    print(f"{'dropped':16s}  {'pooled RAE':>11s}  {'delta':>8s}  "
          f"{'verdict':s}")
    print("-" * 78)
    print(f"{'(none, base)':16s}  {base_rae:>11.4f}  "
          f"{0.0:>8.4f}  baseline")
    for r in loo_sorted:
        tag = " <- DOMINANT" if r["dropped"] == dominant else ""
        print(f"{r['dropped']:16s}  {r['pooled_rae']:>11.4f}  "
              f"{r['delta_rae_vs_baseline']:>+8.4f}{tag}")
    print(f"{'drop2_weakest':16s}  {top3_rae:>11.4f}  "
          f"{top3_delta:>+8.4f}  (kept {','.join(keep3_names)})")

    summary = {
        "tag": TAG,
        "method": "5-anchor_LOO_ablation_plus_top3_stress",
        "anchors": names,
        "indiv_oof_rae": indiv_rae,
        "n_unb": n_unb,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "baseline_pooled_rae": base_rae,
        "baseline_deploy_weights": {
            n: float(w) for n, w in zip(names, base_w)
        },
        "loo_results": loo_rows,
        "loo_sorted_by_delta_desc": [
            {"dropped": r["dropped"],
             "pooled_rae": r["pooled_rae"],
             "delta": r["delta_rae_vs_baseline"]}
            for r in loo_sorted
        ],
        "dominant_signal_carrier": dominant,
        "dominant_delta_rae": dominant_delta,
        "post_unblind_risk_confirmed": dominant == "nb730_honest",
        "top3_stress": {
            "dropped_2_weakest": drop2_names,
            "kept_top_3": keep3_names,
            "pooled_rae": top3_rae,
            "delta_vs_baseline": top3_delta,
            "deploy_weights": top3_wdict,
        },
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")
    print(f"   wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "baseline_pooled_rae",
        "dominant_signal_carrier",
        "dominant_delta_rae",
        "post_unblind_risk_confirmed",
    ):
        print(f"  {k}: {res.get(k)}")
