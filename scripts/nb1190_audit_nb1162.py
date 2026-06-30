"""nb1190 -- Reproducibility audit of nb1162 stack pyramid (PRIMARY-1).

Re-runs the nb1162 SLSQP+rank-stretch stacking pipeline under FIVE FRESH
scaffold-CV seeds {1006, 1007, 1008, 1009, 1010} to test whether the
reported 0.4206 pooled cross-fit RAE is stable or kf_seed-lucky.

Pipeline (mirrors nb1162_stack_pyramid.py):
    Stage 1 anchors (5 cached OOFs on the 253 unblind):
        nb2103_K28, chemprop_aux, nb730_honest, nb503, nb562
    Stage 2: SLSQP convex blend (w>=0, sum=1) per TRAIN fold
    Stage 3: rank-stretch grid {1.000, 1.025, ..., 1.150} per TRAIN fold
    OOF blend = (mu_tr + s_tr * (P_val @ w_tr - mu_tr)) gathered across folds
    Pooled scaffold-CV RAE = rae(y_unb, oof_blend)

Contamination decomposition (drop-nb730):
    Re-fit the same pipeline without the nb730_honest anchor.
    Use the 4 OTHER nb1162 anchors that have honest OOFs on the 253:
        chemprop_aux, nb2103_K28, nb503, nb562
    (Note: the prompt named "chemprop_aux+nb1150+nb1158" as PRE-unblind
    anchors, but nb1150 and nb1158 only have te(513) cached -- no OOF
    on the 253 -- so we use the 4 PRE-unblind OOFs that ARE available
    on the 253 for an apples-to-apples comparison. The signal we want
    -- "does dropping the dominant contaminated anchor blow up the RAE"
    -- is preserved.)

Gating rules (applied to the FRESH 5-seed reproduction):
    A) mean RAE <= 0.42
    B) std  RAE <= 0.010
    C) drop-nb730 mean RAE <= 0.50
    => keep nb1162 PRIMARY-1

    If std > 0.015        -> nb1162 was kf_seed-lucky -> demote, revert to nb1150
    If drop-nb730 > 0.52  -> contamination dominates -> flag for caution

Outputs:
    data/processed/nb1190_summary.json
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
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1190"
FRESH_SEEDS = [1006, 1007, 1008, 1009, 1010]
N_FOLDS = 5
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]

NB1162_REPORTED_RAE = 0.4206326222520904

# Gate thresholds (from prompt)
GATE_MEAN = 0.42
GATE_STD = 0.010
GATE_DROP_NB730 = 0.50
DEMOTE_STD = 0.015
FLAG_DROP_NB730 = 0.52

# Full 5-anchor set (matches nb1162_stack_pyramid.py)
ANCHORS_FULL = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
    ("nb730_honest", "nb730_honest_pred_oof.npy"),
    ("nb503",        "nb503_pred_oof.npy"),
    ("nb562",        "nb562_pred_oof.npy"),
]

# Drop-nb730: same 5 anchors minus nb730_honest -- all PRE-unblind clean.
ANCHORS_DROP = [a for a in ANCHORS_FULL if a[0] != "nb730_honest"]


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


def best_stretch_on(blend_tr: np.ndarray, y_tr: np.ndarray,
                    mu: float, grid) -> tuple[float, float]:
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def stack_one_seed(P: np.ndarray, y: np.ndarray,
                    scaffolds: list[str], seed: int) -> dict:
    """Run nb1162's full SLSQP+stretch scaffold-CV at one kf_seed."""
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed,
    )
    oof = np.full(len(y), np.nan)
    fold_s = []
    fold_w = []
    for tr, va in splits:
        w_f = slsqp_simplex(P[tr], y[tr])
        blend_tr = P[tr] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y[tr], mu_tr, STRETCH_GRID)
        blend_va = P[va] @ w_f
        oof[va] = mu_tr + s_f * (blend_va - mu_tr)
        fold_s.append(float(s_f))
        fold_w.append([float(x) for x in w_f])
    return {
        "seed": int(seed),
        "rae": float(rae(y, oof)),
        "mean_s": float(np.mean(fold_s)),
        "fold_s": fold_s,
        "fold_w_max": float(np.max([np.max(w) for w in fold_w])),
    }


def load_anchors(anchor_list, n_unb):
    cols = []
    indiv = {}
    for disp, oof_rel in anchor_list:
        p = DATA_PROCESSED / oof_rel
        assert p.exists(), f"missing OOF: {p}"
        a = np.load(p).astype(np.float64)
        assert a.shape == (n_unb,), f"{disp} shape {a.shape}"
        cols.append(a)
        indiv[disp] = float("nan")  # filled later
    return np.column_stack(cols), indiv


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- nb1162 reproducibility audit "
          f"(fresh kf_seeds={FRESH_SEEDS})")
    print("=" * 78)

    # ---- Load 513 + 253 substrate ----
    te = load_test()
    te_smiles = te["smiles"].values
    n_te = len(te_smiles)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = te_smiles[unb_idx]
    scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in scaffolds if s})
    n_singleton = sum(1 for s in scaffolds if not s)
    print(f"[load] n_te={n_te}  n_unb={n_unb}  "
          f"unique_scaffolds={n_unique_scaf}  singleton={n_singleton}")

    # ---- Load full + drop-nb730 anchor matrices ----
    P_full, indiv_full = load_anchors(ANCHORS_FULL, n_unb)
    for (disp, oof_rel), col in zip(ANCHORS_FULL, P_full.T):
        indiv_full[disp] = float(rae(y_unb, col))
    print("\n[anchors-FULL]")
    for disp, _ in ANCHORS_FULL:
        print(f"   {disp:14s}  oof_RAE={indiv_full[disp]:.4f}")

    P_drop, indiv_drop = load_anchors(ANCHORS_DROP, n_unb)
    for (disp, _), col in zip(ANCHORS_DROP, P_drop.T):
        indiv_drop[disp] = float(rae(y_unb, col))
    print("\n[anchors-DROP-nb730]")
    for disp, _ in ANCHORS_DROP:
        print(f"   {disp:14s}  oof_RAE={indiv_drop[disp]:.4f}")

    # ============================================================
    # Step 1-3: Fresh-seed reproduction (FULL 5-anchor pipeline)
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 1-3: FULL 5-anchor stack across fresh kf_seeds")
    print("-" * 78)
    per_seed_full = []
    for sd in FRESH_SEEDS:
        rec = stack_one_seed(P_full, y_unb, scaffolds, sd)
        per_seed_full.append(rec)
        print(f"   seed={sd}: RAE={rec['rae']:.4f}  "
              f"mean_s={rec['mean_s']:.3f}  max_w={rec['fold_w_max']:.3f}")

    rae_full = np.array([d["rae"] for d in per_seed_full])
    mean_full = float(rae_full.mean())
    std_full = float(rae_full.std(ddof=0))
    min_full = float(rae_full.min())
    max_full = float(rae_full.max())
    print(f"\n[full] mean={mean_full:.4f}  std={std_full:.4f}  "
          f"range=[{min_full:.4f}, {max_full:.4f}]")
    print(f"       reported nb1162 RAE = {NB1162_REPORTED_RAE:.4f}  "
          f"(delta = {mean_full - NB1162_REPORTED_RAE:+.4f})")

    # ============================================================
    # Step 4: Drop-nb730 contamination decomposition
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 4: DROP-nb730 stack across fresh kf_seeds")
    print("        anchors = chemprop_aux + nb2103_K28 + nb503 + nb562")
    print("        (PRE-unblind clean substitutes -- see header note)")
    print("-" * 78)
    per_seed_drop = []
    for sd in FRESH_SEEDS:
        rec = stack_one_seed(P_drop, y_unb, scaffolds, sd)
        per_seed_drop.append(rec)
        print(f"   seed={sd}: RAE={rec['rae']:.4f}  "
              f"mean_s={rec['mean_s']:.3f}  max_w={rec['fold_w_max']:.3f}")

    rae_drop = np.array([d["rae"] for d in per_seed_drop])
    mean_drop = float(rae_drop.mean())
    std_drop = float(rae_drop.std(ddof=0))
    print(f"\n[drop-nb730] mean={mean_drop:.4f}  std={std_drop:.4f}  "
          f"delta_vs_full=+{mean_drop - mean_full:.4f}")

    # ============================================================
    # Step 5-7: Gate evaluation + ladder action
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 5-7: GATE EVALUATION")
    print("-" * 78)
    gate_a = mean_full <= GATE_MEAN
    gate_b = std_full <= GATE_STD
    gate_c = mean_drop <= GATE_DROP_NB730
    gate_keep_primary1 = gate_a and gate_b and gate_c

    flag_seed_lucky = std_full > DEMOTE_STD
    flag_contamination_dominant = mean_drop > FLAG_DROP_NB730

    if gate_keep_primary1:
        ladder_action = "KEEP_nb1162_PRIMARY1"
    elif flag_seed_lucky:
        ladder_action = "DEMOTE_nb1162_REVERT_TO_nb1150_PRIMARY1"
    elif flag_contamination_dominant:
        ladder_action = "FLAG_CONTAMINATION_DOMINATES_HOLD_nb1162"
    else:
        ladder_action = "HOLD_nb1162_partial_gate_fail"

    print(f"   gate A: mean({mean_full:.4f}) <= {GATE_MEAN}      -> "
          f"{'PASS' if gate_a else 'FAIL'}")
    print(f"   gate B: std({std_full:.4f})  <= {GATE_STD}      -> "
          f"{'PASS' if gate_b else 'FAIL'}")
    print(f"   gate C: drop({mean_drop:.4f}) <= {GATE_DROP_NB730}      -> "
          f"{'PASS' if gate_c else 'FAIL'}")
    print(f"   keep PRIMARY-1: {gate_keep_primary1}")
    print(f"   demote-flag (std > {DEMOTE_STD})       = {flag_seed_lucky}")
    print(f"   contamination-flag (drop > {FLAG_DROP_NB730}) = "
          f"{flag_contamination_dominant}")
    print(f"   LADDER ACTION = {ladder_action}")

    summary = {
        "tag": TAG,
        "method": "nb1162_reproducibility_audit_5fresh_seeds_with_drop_nb730",
        "audit_target": "nb1162_stack_pyramid_PRIMARY1_0p4206",
        "fresh_seeds": FRESH_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_unique_scaffolds": n_unique_scaf,
        "n_singleton_scaffolds": n_singleton,
        "nb1162_reported_pooled_cv_rae": NB1162_REPORTED_RAE,
        "anchors_full": [a[0] for a in ANCHORS_FULL],
        "anchors_drop_nb730": [a[0] for a in ANCHORS_DROP],
        "anchors_drop_substitution_note": (
            "Prompt named 'chemprop_aux+nb1150+nb1158' as PRE-unblind anchors "
            "for the drop-nb730 swap, but nb1150 and nb1158 only have te(513) "
            "cached -- no OOFs on the 253. Using the 4 PRE-unblind OOFs from "
            "nb1162 minus nb730 (chemprop_aux + nb2103_K28 + nb503 + nb562) "
            "preserves the contamination-decomposition signal."
        ),
        "indiv_anchor_oof_rae_full": indiv_full,
        "indiv_anchor_oof_rae_drop": indiv_drop,
        "full_per_seed": per_seed_full,
        "full_mean_rae": mean_full,
        "full_std_rae": std_full,
        "full_min_rae": min_full,
        "full_max_rae": max_full,
        "full_delta_vs_reported": mean_full - NB1162_REPORTED_RAE,
        "drop_nb730_per_seed": per_seed_drop,
        "drop_nb730_mean_rae": mean_drop,
        "drop_nb730_std_rae": std_drop,
        "drop_nb730_delta_vs_full": mean_drop - mean_full,
        "gate_mean_target": GATE_MEAN,
        "gate_std_target": GATE_STD,
        "gate_drop_nb730_target": GATE_DROP_NB730,
        "demote_std_threshold": DEMOTE_STD,
        "flag_drop_nb730_threshold": FLAG_DROP_NB730,
        "gate_a_mean_le_target": bool(gate_a),
        "gate_b_std_le_target": bool(gate_b),
        "gate_c_drop_le_target": bool(gate_c),
        "gate_keep_primary1": bool(gate_keep_primary1),
        "flag_seed_lucky": bool(flag_seed_lucky),
        "flag_contamination_dominant": bool(flag_contamination_dominant),
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   FULL  mean RAE = {mean_full:.4f}  +/- {std_full:.4f}")
    print(f"   DROP  mean RAE = {mean_drop:.4f}  +/- {std_drop:.4f}")
    print(f"   gates: A={gate_a}  B={gate_b}  C={gate_c}  "
          f"-> keep={gate_keep_primary1}")
    print(f"   action = {ladder_action}")
    print(f"   wall   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    main()
