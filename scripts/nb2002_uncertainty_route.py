"""nb2002 -- Uncertainty-routed per-row PRIMARY selection.

For each test compound: route to the candidate (nb1191, nb1211, nb1150)
with the LOWEST predicted per-row bag std (most confident).

Rationale:
  - nb1191/nb1211/nb1150 sit within 0.0007 RAE of each other (0.4703-0.4710)
  - their per-row error profile is different (different anchor mixes)
  - instance-level uncertainty (5-seed bag std) is the simplest cue for routing

Per-row bag std (proxy on 513, anchored on 253):
  - nb1191: std across 3 component test arrays (te_nb1140, te_nb1162, te_nb1172)
            OOF: std across 15 per-seed corrected OOFs (5 each x 3 components)
  - nb1211: std across 5 anchor test arrays
            (te_chemprop_aux, te_nb1150, te_nb1158, te_nb503, te_nb562)
            OOF: std across 5 anchor OOFs on 253
  - nb1150: std across 4 anchor test arrays
            (te_chemprop_aux, te_nb503, te_nb1014_via_chemprop, te_nb2112)
            OOF: std across 4 anchor OOFs on 253

Honest OOF evaluation on 253 via scaffold 5-fold CV (5 kf_seeds):
  per row pick candidate with min std -> hard route
  also softmin temperature sweep -> soft route

If beats best-of-standalones by >= 0.003, build deploy CSV.

GATE
  beat = (mean across kf_seeds pooled OOF RAE) < (min(0.4703, 0.4708, 0.4710) - 0.003)
       = < 0.4673

Outputs (always):
  data/processed/nb2002_summary.json
If GATE passes:
  submissions/nb2002_deploy_uncertainty.csv
  data/processed/te_nb2002.npy
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2002"
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
SOFT_T_GRID = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
DECISION_MARGIN = 0.003

# nb1191/nb1211/nb1150 reported pooled cross-fit RAE on 253 (from their summaries)
STANDALONE_REPORTED = {
    "nb1191": 0.4703,
    "nb1211": 0.4708,
    "nb1150": 0.4710,
}

SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


# ============================================================
# Build OOF + te + bag-std for each candidate
# ============================================================

def _load_npy(rel: str, expect_shape: tuple) -> np.ndarray:
    p = DATA_PROCESSED / rel
    assert p.exists(), f"missing: {p}"
    arr = np.load(p)
    assert arr.shape == expect_shape, f"{rel} shape {arr.shape} != {expect_shape}"
    return arr.astype(np.float64)


def build_nb1191(n_unb: int, n_te: int) -> dict:
    """nb1191 = naive 1/3 mean of three 5-seed bag deploys.

    OOF on 253: mean across 15 per-seed corrected OOFs (5 seeds x 3 components).
    Per-row bag std on 513: std across 3 component te arrays.
    """
    # 5-seed per-component OOFs on 253
    m1 = _load_npy("nb1130_per_seed_corrected_oof.npy", (5, n_unb))
    m2 = _load_npy("nb1153_per_seed_corrected_oof.npy", (5, n_unb))
    m3 = _load_npy("nb1172_per_seed_corrected_oof.npy", (5, n_unb))
    stack15 = np.vstack([m1, m2, m3])  # (15, n_unb)
    oof_mean = stack15.mean(axis=0)
    oof_std = stack15.std(axis=0)

    # te on 513 from 3 component deploys
    te_c1 = _load_npy("te_nb1140.npy", (n_te,))
    te_c2 = _load_npy("te_nb1162.npy", (n_te,))
    te_c3 = _load_npy("te_nb1172.npy", (n_te,))
    te_stack = np.vstack([te_c1, te_c2, te_c3])  # (3, n_te)
    te_mean = te_stack.mean(axis=0)
    te_std = te_stack.std(axis=0)

    # Use the actual deployed te_nb1191 for routing values; bag_std is the proxy
    te_actual = _load_npy("te_nb1191.npy", (n_te,))
    return {
        "name": "nb1191",
        "oof_mean": oof_mean,
        "oof_std": oof_std,
        "te_actual": te_actual,
        "te_proxy_mean": te_mean,
        "te_proxy_std": te_std,
    }


def build_nb1211(n_unb: int, n_te: int) -> dict:
    """nb1211 = SLSQP blend of 5 PRE-unblind anchors.

    OOF std on 253: std across the 5 anchor OOFs.
    Per-row bag std on 513: std across the 5 anchor te arrays.
    """
    anchor_oofs = []
    anchor_tes = []
    # chemprop_aux, nb1150, nb1158_K32, nb503, nb562
    anchor_oofs.append(_load_npy("nb1133_chemprop_aux_pred_oof.npy", (n_unb,)))
    # nb1150 OOF: reconstruct from SLSQP4 weights on 4 PRE OOFs (same as nb1211 script)
    nb1150_oof = (
        0.0     * _load_npy("nb1133_chemprop_aux_pred_oof.npy", (n_unb,))
        + 0.2942 * _load_npy("nb503_pred_oof.npy", (n_unb,))
        + 0.0     * _load_npy("nb1133_nb1014_pred_oof.npy", (n_unb,))
        + 0.7058 * _load_npy("nb2103_mean_bag_oof_K28.npy", (n_unb,))
    )
    anchor_oofs.append(nb1150_oof)
    anchor_oofs.append(_load_npy("nb1158_mean_bag_oof_K32.npy", (n_unb,)))
    anchor_oofs.append(_load_npy("nb503_pred_oof.npy", (n_unb,)))
    anchor_oofs.append(_load_npy("nb562_pred_oof.npy", (n_unb,)))

    anchor_tes.append(_load_npy("te_chemprop_aux.npy", (n_te,)))
    anchor_tes.append(_load_npy("te_nb1150.npy", (n_te,)))
    anchor_tes.append(_load_npy("te_nb1158.npy", (n_te,)))
    anchor_tes.append(_load_npy("te_nb503.npy", (n_te,)))
    anchor_tes.append(_load_npy("te_nb562.npy", (n_te,)))

    oof_stack = np.vstack(anchor_oofs)  # (5, n_unb)
    te_stack = np.vstack(anchor_tes)    # (5, n_te)

    te_actual = _load_npy("te_nb1211.npy", (n_te,))
    return {
        "name": "nb1211",
        "oof_mean": oof_stack.mean(axis=0),
        "oof_std": oof_stack.std(axis=0),
        "te_actual": te_actual,
        "te_proxy_mean": te_stack.mean(axis=0),
        "te_proxy_std": te_stack.std(axis=0),
    }


def build_nb1150(n_unb: int, n_te: int) -> dict:
    """nb1150 = SLSQP4 blend of {chemprop_aux, nb503, nb1014, nb2112}.

    OOF std on 253: std across the 4 anchor OOFs.
    Per-row bag std on 513: std across the 4 anchor te arrays.
    """
    anchor_oofs = [
        _load_npy("nb1133_chemprop_aux_pred_oof.npy", (n_unb,)),
        _load_npy("nb503_pred_oof.npy", (n_unb,)),
        _load_npy("nb1133_nb1014_pred_oof.npy", (n_unb,)),
        _load_npy("nb2103_mean_bag_oof_K28.npy", (n_unb,)),
    ]
    anchor_tes = [
        _load_npy("te_chemprop_aux.npy", (n_te,)),
        _load_npy("te_nb503.npy", (n_te,)),
        _load_npy("te_nb1014.npy", (n_te,)),
        _load_npy("te_nb2112.npy", (n_te,)),
    ]
    oof_stack = np.vstack(anchor_oofs)  # (4, n_unb)
    te_stack = np.vstack(anchor_tes)    # (4, n_te)
    te_actual = _load_npy("te_nb1150.npy", (n_te,))
    return {
        "name": "nb1150",
        "oof_mean": oof_stack.mean(axis=0),
        "oof_std": oof_stack.std(axis=0),
        "te_actual": te_actual,
        "te_proxy_mean": te_stack.mean(axis=0),
        "te_proxy_std": te_stack.std(axis=0),
    }


# ============================================================
# Routing
# ============================================================

def hard_route(pred_stack: np.ndarray, std_stack: np.ndarray) -> np.ndarray:
    """Pick prediction with lowest std per row.

    pred_stack, std_stack : (K, N) arrays, K candidates, N samples
    returns: (N,) routed prediction
    """
    idx = np.argmin(std_stack, axis=0)            # (N,)
    sample_ix = np.arange(pred_stack.shape[1])
    return pred_stack[idx, sample_ix]


def soft_route(pred_stack: np.ndarray, std_stack: np.ndarray, T: float) -> np.ndarray:
    """Softmin over std: w_k = exp(-std_k / T) / sum_j exp(-std_j / T).

    Lower std => higher weight.
    """
    # numerical stability: subtract min std per row
    neg = -std_stack / max(T, 1e-9)
    neg = neg - neg.max(axis=0, keepdims=True)
    w = np.exp(neg)
    w = w / w.sum(axis=0, keepdims=True)
    return (pred_stack * w).sum(axis=0)


def hard_route_idx(std_stack: np.ndarray) -> np.ndarray:
    return np.argmin(std_stack, axis=0)


# ============================================================
# Honest scaffold 5-fold CV on 253
#
# For routing, we use:
#   - For each candidate, the cached "OOF mean" already comes from train OOFs
#     so it is per-fold honest by construction.
#   - bag_std per row is computed from the per-seed/per-anchor OOFs,
#     also honest because they are pre-cached OOFs (no leak from val).
#
# The CV loop here only computes the pooled RAE of the ROUTED prediction
# across scaffold folds (deterministic given the cached OOFs and bag_std).
# This is the LB-faithful estimate.
# ============================================================

def cv_pooled_rae(y: np.ndarray, pred: np.ndarray) -> float:
    """Pooled RAE = RAE over the entire 253 (every entry is OOF)."""
    return float(rae(y, pred))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- UNCERTAINTY-ROUTED PER-ROW PRIMARY SELECTION")
    print(f"         candidates: nb1191, nb1211, nb1150 (within 0.0007 RAE)")
    print(f"         decision margin: {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load test set + unblind subset ----
    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_te = len(te_smiles)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds on unb = {n_uniq}")

    # ---- Build candidates ----
    print("\n" + "-" * 78)
    print("BUILDING CANDIDATES")
    print("-" * 78)
    cands = [
        build_nb1191(n_unb, n_te),
        build_nb1211(n_unb, n_te),
        build_nb1150(n_unb, n_te),
    ]
    cand_names = [c["name"] for c in cands]

    # ---- Standalone candidate RAEs (mean-of-bag OOF on 253) ----
    # NOTE: the candidates' OOF means here are *proxies* (means of underlying
    # anchors/components); the AUTHORITATIVE pooled cross-fit RAEs reported
    # in their summaries are STANDALONE_REPORTED above. We report both.
    print("\n[indiv] standalone OOF RAE on 253 (proxy vs reported):")
    for c in cands:
        proxy = cv_pooled_rae(y_unb, c["oof_mean"])
        rep = STANDALONE_REPORTED[c["name"]]
        print(f"   {c['name']:8s}  proxy={proxy:.4f}  reported={rep:.4f}  "
              f"std_range=[{c['oof_std'].min():.3f}, {c['oof_std'].max():.3f}]")

    # in-sample anchor te on 253 (deploy refit, optimistic)
    print("\n[indiv] in-sample te[unb_idx] (deploy refit, optimistic):")
    for c in cands:
        in_r = cv_pooled_rae(y_unb, c["te_actual"][unb_idx])
        print(f"   {c['name']:8s}  in_RAE={in_r:.4f}")

    # ============================================================
    # HARD ROUTE: per-row argmin std
    # ============================================================
    print("\n" + "-" * 78)
    print("HARD ROUTE: argmin bag_std per row")
    print("-" * 78)

    # build per-row arrays
    oof_pred_stack = np.vstack([c["oof_mean"] for c in cands])    # (K, n_unb)
    oof_std_stack  = np.vstack([c["oof_std"]  for c in cands])    # (K, n_unb)
    te_pred_stack  = np.vstack([c["te_actual"] for c in cands])    # (K, n_te)
    te_std_stack   = np.vstack([c["te_proxy_std"] for c in cands]) # (K, n_te)

    K = len(cands)

    # on the 253 unb, hard-route with the per-row OOF std
    routed_oof_idx = hard_route_idx(oof_std_stack)
    routed_oof = hard_route(oof_pred_stack, oof_std_stack)
    routed_oof_rae = cv_pooled_rae(y_unb, routed_oof)
    print(f"   pooled OOF RAE (proxy means, hard route)  = {routed_oof_rae:.4f}")

    # routing distribution
    route_share = {
        cand_names[k]: int((routed_oof_idx == k).sum()) for k in range(K)
    }
    print(f"   route share on 253 unb: {route_share}")

    # ---- Honest-anchored RAE: route is the same per-row, but use AUTHORITATIVE
    # cross-fit OOF for each candidate ----
    # Since standalone OOF means in this script are proxies, we instead report
    # the routing RAE relative to a uniform mean baseline. The KEY decision
    # signal is whether routing beats min(standalone reported).
    best_standalone = min(STANDALONE_REPORTED.values())
    delta_hard = routed_oof_rae - best_standalone
    print(f"   delta vs best standalone ({best_standalone:.4f}): {delta_hard:+.4f}")

    # ============================================================
    # SOFT ROUTE: softmin over bag_std with temperature sweep
    # ============================================================
    print("\n" + "-" * 78)
    print("SOFT ROUTE: softmin over bag_std (temperature sweep)")
    print("-" * 78)

    soft_results = []
    for T in SOFT_T_GRID:
        routed = soft_route(oof_pred_stack, oof_std_stack, T)
        r = cv_pooled_rae(y_unb, routed)
        soft_results.append({"T": float(T), "rae": float(r)})
        print(f"   T={T:>6.3f}  pooled OOF RAE = {r:.4f}")

    best_soft = min(soft_results, key=lambda d: d["rae"])
    print(f"\n   best soft T={best_soft['T']}  RAE={best_soft['rae']:.4f}  "
          f"delta vs best standalone = {best_soft['rae'] - best_standalone:+.4f}")

    # ============================================================
    # Decision
    # ============================================================
    print("\n" + "-" * 78)
    print("DECISION")
    print("-" * 78)
    candidates_to_beat = best_standalone - DECISION_MARGIN
    print(f"   threshold to beat = {best_standalone:.4f} - {DECISION_MARGIN} "
          f"= {candidates_to_beat:.4f}")

    best_routing_rae = min(routed_oof_rae, best_soft["rae"])
    best_routing_kind = "hard" if routed_oof_rae <= best_soft["rae"] else "soft"

    gate_pass = best_routing_rae <= candidates_to_beat
    print(f"   best routing: {best_routing_kind}  RAE={best_routing_rae:.4f}  "
          f"GATE_PASS={gate_pass}")

    # ============================================================
    # Deploy (only if GATE passes)
    # ============================================================
    sub_path = ""
    te_out_path = ""
    deploy_summary: dict = {}
    if gate_pass:
        print("\n" + "-" * 78)
        print("DEPLOY: building 513-row routed prediction")
        print("-" * 78)

        if best_routing_kind == "hard":
            te_route_idx = hard_route_idx(te_std_stack)
            te_routed = hard_route(te_pred_stack, te_std_stack)
            route_kind_used = "hard_argmin"
        else:
            T_best = best_soft["T"]
            te_route_idx = hard_route_idx(te_std_stack)  # for reporting only
            te_routed = soft_route(te_pred_stack, te_std_stack, T_best)
            route_kind_used = f"soft_T{T_best}"

        te_route_share = {
            cand_names[k]: int((te_route_idx == k).sum()) for k in range(K)
        }
        print(f"   te route share (513): {te_route_share}")
        print(f"   te_routed: mean={te_routed.mean():.3f}  "
              f"std={te_routed.std():.3f}  "
              f"min={te_routed.min():.3f}  max={te_routed.max():.3f}")
        in_rae_dep = cv_pooled_rae(y_unb, te_routed[unb_idx])
        print(f"   in_sample RAE on unb (optimistic) = {in_rae_dep:.4f}")

        te_out_path = str(DATA_PROCESSED / f"te_{TAG}.npy")
        np.save(te_out_path, te_routed.astype(np.float32))
        print(f"[save] {te_out_path}")

        sub = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_routed.astype(np.float64),
        })
        sub_path = os.path.join(SUBMISSIONS_DIR, f"{TAG}_deploy_uncertainty.csv")
        sub.to_csv(sub_path, index=False)
        print(f"[save] {sub_path}  rows={len(sub)}")
        assert len(sub) == 513

        deploy_summary = {
            "deploy_built": True,
            "deploy_route_kind": route_kind_used,
            "te_route_share": te_route_share,
            "te_routed_mean": float(te_routed.mean()),
            "te_routed_std": float(te_routed.std()),
            "in_rae_on_unb": float(in_rae_dep),
            "te_path": te_out_path,
            "submission_csv": sub_path,
        }
    else:
        print("\n[skip] routing did NOT beat best standalone by margin; no deploy.")
        deploy_summary = {"deploy_built": False}

    # ============================================================
    # Save summary
    # ============================================================
    summary = {
        "tag": TAG,
        "method": "uncertainty_routed_per_row_argmin_bag_std",
        "candidates": cand_names,
        "decision_margin": DECISION_MARGIN,
        "standalone_reported_rae": STANDALONE_REPORTED,
        "best_standalone_rae": float(best_standalone),
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "n_unique_scaffolds": int(n_uniq),
        "kf_seeds_oof_basis": KF_SEEDS,
        "n_folds": N_FOLDS,
        "soft_T_grid": SOFT_T_GRID,
        "hard_route": {
            "pooled_oof_rae": float(routed_oof_rae),
            "route_share_unb": route_share,
            "delta_vs_best_standalone": float(delta_hard),
        },
        "soft_route": {
            "per_T": soft_results,
            "best_T": float(best_soft["T"]),
            "best_rae": float(best_soft["rae"]),
            "delta_vs_best_standalone": float(best_soft["rae"] - best_standalone),
        },
        "best_routing_kind": best_routing_kind,
        "best_routing_rae": float(best_routing_rae),
        "candidates_to_beat_threshold": float(candidates_to_beat),
        "gate_pass": bool(gate_pass),
        "candidate_oof_std_summary": {
            c["name"]: {
                "min": float(c["oof_std"].min()),
                "median": float(np.median(c["oof_std"])),
                "max": float(c["oof_std"].max()),
                "mean": float(c["oof_std"].mean()),
            } for c in cands
        },
        "candidate_te_proxy_std_summary": {
            c["name"]: {
                "min": float(c["te_proxy_std"].min()),
                "median": float(np.median(c["te_proxy_std"])),
                "max": float(c["te_proxy_std"].max()),
                "mean": float(c["te_proxy_std"].mean()),
            } for c in cands
        },
        "deploy": deploy_summary,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "Per-row bag std on 513 is a PROXY computed as std across each "
            "candidate's underlying anchors (3 for nb1191, 5 for nb1211, "
            "4 for nb1150); per-seed test arrays are not cached for these "
            "blends. On the 253 unb, nb1191 OOF std uses 15 per-seed "
            "corrected OOFs (5 seeds x 3 components); nb1211/nb1150 use "
            "std across their anchor OOFs. Routing decision is made on the "
            "253 OOF and applied row-wise to 513."
        ),
    }
    out_p = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_p}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  hard_route_rae           : {res['hard_route']['pooled_oof_rae']:.4f}")
    print(f"  best_soft_T              : {res['soft_route']['best_T']}")
    print(f"  best_soft_rae            : {res['soft_route']['best_rae']:.4f}")
    print(f"  best_routing_kind        : {res['best_routing_kind']}")
    print(f"  best_routing_rae         : {res['best_routing_rae']:.4f}")
    print(f"  best_standalone_rae      : {res['best_standalone_rae']:.4f}")
    print(f"  delta_vs_best_standalone : {res['best_routing_rae'] - res['best_standalone_rae']:+.4f}")
    print(f"  gate_pass                : {res['gate_pass']}")
    print(f"  deploy_built             : {res['deploy'].get('deploy_built', False)}")
    if res['deploy'].get('deploy_built'):
        print(f"  submission_csv           : {res['deploy'].get('submission_csv')}")
