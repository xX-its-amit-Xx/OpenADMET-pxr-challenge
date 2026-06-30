"""
nb954_trajectory.py - Cycle 129 integration summary + trajectory.

Records the planned cycle-129 sequence (5 distinct methods: nb950-953 + this
integrator) and its outcomes. The cycle-129 thesis was to build "v2" variants
of the four canonical anchors (chemprop_aux, nb562, nb730, nb2189/nb2112) on
top of the augmented data substrate (BACE / ChEMBL PXR / Tox21 / counter
multi-impute) introduced in cycle 128, and check whether the augmented
anchors translate the honest cross-fit gain to actual deploy.

PRECONDITION CHECK (2026-06-08):
The four cycle-129 method scripts nb950/nb951/nb952/nb953 do NOT yet exist on
disk; their summary JSONs are not present in data/processed/. This integrator
records the slate as PENDING and refuses to overwrite the established
honest-floor numbers. The honest floor 0.4698 (nb2189 K=28 median bag /
nb2103 K=28 median bag) is preserved as the active anchor.

Inputs (read-only):
  data/processed/nb2189_summary.json   (current honest residual, K=20 best 0.4556 / K=28 median 0.4698)
  data/processed/nb2202_summary.json   (chemprop_aux clean rebuild)
  data/processed/nb2203_summary.json   (train-holdout transfer sim)
Outputs:
  data/processed/nb954_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

CYCLE = 129
PRIOR_HONEST_FLOOR_RAE = 0.4698          # nb2189 / nb2103 K=28 median bag (TRULY HONEST)
PRIOR_HONEST_FLOOR_ID = "nb2189_truly_honest_residual_K28_median"

# --- 1. Five cycle-129 methods ---------------------------------------------
# Each method is a planned v2 rebuild of a canonical anchor on the augmented
# substrate. Status is PENDING because the upstream scripts have not been
# executed in this repository state.
CYCLE_129_METHODS = [
    {
        "id": "nb950_chemprop_aux_v2",
        "axis": "chemprop_aux retrained on augmented substrate (BACE+ChEMBL+Tox21)",
        "parent_anchor": "chemprop_aux",
        "parent_rae": 0.6216,
        "status": "PENDING",
        "v2_rae": None,
        "expected_summary_path": str(PROC / "nb950_summary.json"),
    },
    {
        "id": "nb951_nb2112_v2_residual",
        "axis": "nb2112-style residual on chemprop_aux_v2 with refreshed K=28 SHAP",
        "parent_anchor": "nb2112_residual",
        "parent_rae": 0.4698,
        "status": "PENDING",
        "v2_rae": None,
        "expected_summary_path": str(PROC / "nb951_summary.json"),
    },
    {
        "id": "nb952_nb562_v2_stretch",
        "axis": "nb562 rank-stretch on chemprop_aux_v2 with refreshed scalar s",
        "parent_anchor": "nb562_rank_stretch",
        "parent_rae": 0.5065,
        "status": "PENDING",
        "v2_rae": None,
        "expected_summary_path": str(PROC / "nb952_summary.json"),
    },
    {
        "id": "nb953_nb730_v2_planning",
        "axis": "nb730 multi-seed null-ensemble v2 (PLANNED for cycle 130 conditional on v2 floor moving)",
        "parent_anchor": "nb730_multi_seed_null_ens",
        "parent_rae": 0.4603,
        "status": "PENDING_CYCLE_130",
        "v2_rae": None,
        "expected_summary_path": str(PROC / "nb953_summary.json"),
    },
    {
        "id": "nb954_trajectory",
        "axis": "this integrator -- records cycle-129 slate + ladder implications",
        "parent_anchor": None,
        "parent_rae": None,
        "status": "DONE",
        "v2_rae": None,
        "expected_summary_path": str(PROC / "nb954_summary.json"),
    },
]

# --- 2. Floor-move check ----------------------------------------------------
# A v2 candidate "moves" the floor only if its honest cross-fit RAE on the
# 253 unblind beats 0.4698 by >= 0.005 (one assay-noise-floor unit). With all
# inputs PENDING this check is vacuously False and the floor is unchanged.
candidate_rae = [m["v2_rae"] for m in CYCLE_129_METHODS if m["v2_rae"] is not None]
floor_moved = bool(candidate_rae) and min(candidate_rae) <= PRIOR_HONEST_FLOOR_RAE - 0.005
new_floor_rae = min(candidate_rae) if floor_moved else PRIOR_HONEST_FLOOR_RAE
new_floor_id  = (
    next(m["id"] for m in CYCLE_129_METHODS if m["v2_rae"] == new_floor_rae)
    if floor_moved else PRIOR_HONEST_FLOOR_ID
)

FLOOR_TRANSITION = {
    "old_floor_rae": PRIOR_HONEST_FLOOR_RAE,
    "old_floor_id": PRIOR_HONEST_FLOOR_ID,
    "new_floor_rae": new_floor_rae,
    "new_floor_id": new_floor_id,
    "floor_moved": floor_moved,
    "delta_rae": round(new_floor_rae - PRIOR_HONEST_FLOOR_RAE, 4),
    "reason": "all v2 candidates PENDING -- floor preserved; no candidate beats 0.4698" if not floor_moved else "v2 candidate beat 0.4698 by >= 0.005",
}

# --- 3. Memory implications (v1 -> v2 updates) -----------------------------
MEMORY_IMPLICATIONS = [
    {"key": "chemprop_aux", "v1_rae": 0.6216, "v2_label": "chemprop_aux_v2", "v2_rae": None, "status": "PENDING_nb950"},
    {"key": "nb562",        "v1_rae": 0.5065, "v2_label": "nb562_v2",        "v2_rae": None, "status": "PENDING_nb952"},
    {"key": "nb730",        "v1_rae": 0.4603, "v2_label": "nb730_v2",        "v2_rae": None, "status": "PLANNED_cycle130_conditional"},
    {"key": "nb2112",       "v1_rae": 0.4698, "v2_label": "nb951_residual",  "v2_rae": None, "status": "PENDING_nb951"},
    {
        "key": "feedback_lb_two_regime_calibration",
        "v1_state": "PRE-unblind in_RAE ~= LB + 0.003; POST-unblind unreliable",
        "v2_check": "augmented anchors (cycle 128) are still PRE-unblind w.r.t. the 513 -- 2 regime model should hold; CONFIRM with nb950 LB submission delta",
        "status": "UNCHANGED_pending_LB_evidence",
    },
]

# --- 4. Cycle 130 plan (5 distinct methods) --------------------------------
# Conditional on cycle 129 outcome: if v2 floor moves, build orthogonal
# residual + abstention on top; if not, pivot to external-data axes.
CYCLE_130_PLAN = [
    {
        "id": "nb1010_chemprop_aux_v2_pose_features",
        "axis": "anchor v2 + 3D pose-quality features (Boltz-2 + DOCK6 cross-score per ligand)",
        "depends_on": "nb950_chemprop_aux_v2",
    },
    {
        "id": "nb1011_residual_v2_isotonic",
        "axis": "v2 residual + isotonic per-decile calibration on the 253 unblind",
        "depends_on": "nb951_nb2112_v2_residual",
    },
    {
        "id": "nb1012_OOD_aware_router",
        "axis": "scaffold-novelty router: route Tan<0.35 to mean-predictor abstention, Tan>=0.35 to v2 stack",
        "depends_on": "nb951 + nb950",
    },
    {
        "id": "nb1013_abstention_conformal",
        "axis": "conformal-shrink on chemprop_aux_v2 residuals (already trial'd via nb2190; rerun on v2 substrate)",
        "depends_on": "nb950_chemprop_aux_v2",
    },
    {
        "id": "nb1014_nb730_v2_multi_seed_null",
        "axis": "promoted from cycle 129 PENDING_CYCLE_130 (nb953); only fires if floor moves",
        "depends_on": "FLOOR_TRANSITION.floor_moved == True",
    },
]

# --- 5. STILL OPEN axes (cross-checked against nb2193 still-open list) -----
STILL_OPEN_AXES = [
    "NN / Transformer on K=28 augmented v2 (Graphormer / MAT / ChemBERTa fine-tune on 4139+253+aug)",
    "Structure-track integration (pose-quality of activity ligands via Boltz-2 + DOCK6 -> per-row gate)",
    "External-data abstention on F2 tail (scaffold-novelty + counter-assay disagreement gate)",
    "2-stage stack (PRE-unblind chemprop_aux_v2 base -> honest meta on 253)",
    "Novel feature paradigm (3D-pose-quality, docking score, MD descriptors from cycle-128 boltz outputs)",
]

# --- 6. Closed axes (carried forward from nb2193) --------------------------
CLOSED_AXES_INHERITED = [
    "K/L/lr/mc/ff hyperparam sweep",
    "monotone constraints",
    "DART boosting",
    "SHAP-seeded feature selection beyond K=28",
    "row-bootstrap ensembling",
    "pooled-120 cross-fit",
    "XGB-SHAP transfer",
    "family-ablation (drop NR-family)",
    "residual-cascade depth>2",
    "tanh-target reparameterization",
    "sklearn-stack meta-learner",
    "contaminated-anchor swap (te_nb730 <-> te_nb562)",
    "alt-honest-anchor swap (nb503 <-> nb464)",
    "loss-level decompression (nb710-712 pinball/tail/PCHIP)",
    "unblind augmentation alone (nb590-593)",
]

# --- 7. Decision summary ---------------------------------------------------
DECISION = {
    "next_action": "Execute nb950 first (chemprop_aux v2 retrain on augmented substrate); only then build nb951/nb952 residuals on the new anchor.",
    "rationale": (
        "Cycle 129 cannot integrate -- nb950/nb951/nb952/nb953 are PENDING. "
        "Floor stays at 0.4698 (nb2189 K=28 median). The v2 thesis (augmented "
        "training substrate translates to honest deploy gain) is unfalsified "
        "but also unconfirmed. The cheapest first move is nb950 because every "
        "downstream cycle-129 method depends on a refreshed chemprop_aux v2."
    ),
    "abort_condition": "if nb950 v2 RAE on 4139 OOF degrades by >0.02 vs chemprop_aux v1 (0.6216), abort cycle 129 and pivot to cycle 130 external-data axes only.",
}

# --- Assemble summary -------------------------------------------------------
summary = {
    "tag": "nb954",
    "cycle": CYCLE,
    "method": "cycle_129_integration_summary_plus_trajectory",
    "precondition_check": {
        "nb950_summary_present": (PROC / "nb950_summary.json").exists(),
        "nb951_summary_present": (PROC / "nb951_summary.json").exists(),
        "nb952_summary_present": (PROC / "nb952_summary.json").exists(),
        "nb953_summary_present": (PROC / "nb953_summary.json").exists(),
        "all_inputs_present": all(
            (PROC / f"nb{n}_summary.json").exists() for n in (950, 951, 952, 953)
        ),
    },
    "cycle_129_methods": CYCLE_129_METHODS,
    "n_distinct_methods_cycle_129": 5,
    "prior_honest_floor": {"id": PRIOR_HONEST_FLOOR_ID, "rae": PRIOR_HONEST_FLOOR_RAE},
    "floor_transition": FLOOR_TRANSITION,
    "memory_implications": MEMORY_IMPLICATIONS,
    "cycle_130_plan": CYCLE_130_PLAN,
    "n_distinct_methods_cycle_130_plan": 5,
    "still_open_axes": STILL_OPEN_AXES,
    "closed_axes_inherited_from_nb2193": CLOSED_AXES_INHERITED,
    "decision": DECISION,
}


def main() -> None:
    out = PROC / "nb954_summary.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out}")
    print(f"cycle={CYCLE}  cycle129_methods={len(CYCLE_129_METHODS)}  cycle130_plan={len(CYCLE_130_PLAN)}")
    print(f"prior honest floor: {PRIOR_HONEST_FLOOR_ID}  RAE={PRIOR_HONEST_FLOOR_RAE}")
    print(f"floor_moved={FLOOR_TRANSITION['floor_moved']}  new_floor={FLOOR_TRANSITION['new_floor_id']}  RAE={FLOOR_TRANSITION['new_floor_rae']}")
    pending = [m["id"] for m in CYCLE_129_METHODS if m["status"].startswith("PENDING")]
    print(f"PENDING methods: {pending}")
    print(f"next action: {DECISION['next_action']}")


if __name__ == "__main__":
    main()
