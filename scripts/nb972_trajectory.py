"""
nb972_trajectory.py - Cycle 131 trajectory snapshot.

Records the 5 cycle-131 method slate, running cycle/method counts, the honest
floor, the CLOSED axes (with cycle-130 failures added), the candidate-NEW
CLOSED axes (cycle 131 if all 5 fail), STILL-OPEN axes, and the 5-distinct-
method cycle-132 plan.

Cycle 131 methods (slated):
  nb968  per-decile isotonic calibration on the 253 unblind
  nb969  PU-learning on 21k weak/positive-unlabeled pool (single-conc + counter)
  nb970  OOD sigmoid router (scaffold-novelty + counter-disagree gate)
  nb971  SLSQP 6-anchor blend (nb2189 / nb730 / nb562 / nb503 / nb950_lite / nb960)
  nb972  this trajectory integrator

PRECONDITION CHECK (2026-06-08):
  - nb968 per-decile isotonic: SCRIPT NOT YET ON DISK; summary not present
  - nb969 PU 21k:                SCRIPT NOT YET ON DISK; summary not present
  - nb970 OOD sigmoid router:    nb970_multi_conformer.py exists but is a
                                 DIFFERENT artifact (multi-conformer 3D, not
                                 OOD router); router summary not present
  - nb971 SLSQP 6-anchor:        nb971_rdkit_only.py exists but is a DIFFERENT
                                 artifact (RDKit-only baseline); SLSQP blend
                                 summary not present
  - nb972 trajectory:            this script

Reads (read-only):
  data/processed/nb964_summary.json   (cycle-130 snapshot)
  data/processed/nb2189_summary.json  (current best honest residual; not
                                       required if absent -- floor pulled from
                                       nb964 inheritance)

Writes:
  data/processed/nb972_summary.json   (OVERWRITES the prior nb972 long-train
                                       optim artifact, by task instruction)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

CYCLE = 131
APPROX_METHODS_TO_DATE = 830          # 131 cycles ~ 825 + 5 cycle-131 methods
PRIOR_HONEST_FLOOR_RAE = 0.4698       # nb2112 = nb2103 K=28 median bag (chemprop_aux anchor)
PRIOR_HONEST_FLOOR_ID = "nb2112_chemprop_aux_K28_median_bag"
NB2189_BEST_K20_MEDIAN_BAG = 0.4556   # nb2189 K=20 median bag; absolute best honest

# --- 1. Five cycle-131 methods ---------------------------------------------
CYCLE_131_METHODS = [
    {
        "id": "nb968_per_decile_isotonic",
        "axis": "per-decile isotonic calibration on the 253 unblind (10 bins, monotone PAVA per bin)",
        "rationale": "rank-stretch (nb562 scalar s=1.05) gains -0.005; per-decile isotonic is the natural 10-parameter extension to attack residual quantile compression",
        "status": "PENDING_script_not_on_disk",
        "honest_rae": None,
        "summary_path": str(PROC / "nb968_summary.json"),
        "te_path": str(PROC / "te_nb968.npy"),
    },
    {
        "id": "nb969_pu_21k",
        "axis": "PU-learning on the 21,003 single-concentration + counter-screen pool (positive-unlabeled gradient-boosted)",
        "rationale": "8,126 SP-only compounds never enter activity training; PU framing converts weak-label pool into auxiliary regression target via priors",
        "status": "PENDING_script_not_on_disk",
        "honest_rae": None,
        "summary_path": str(PROC / "nb969_summary.json"),
        "te_path": str(PROC / "te_nb969.npy"),
    },
    {
        "id": "nb970_ood_sigmoid_router",
        "axis": "OOD sigmoid router: P(OOD) = sigmoid(a*(0.35-tan_top1) + b*counter_disagree); route P>0.5 to mean-predictor + abstention shrink",
        "rationale": "F2 (greasy-novel-inactive) carries the -0.11 RAE tail prize per pm06 framing; explicit sigmoid gate is the differentiable cousin of nb960 hard abstention",
        "status": "PENDING_script_not_on_disk_naming_collision_with_multi_conformer",
        "honest_rae": None,
        "summary_path": str(PROC / "nb970_router_summary.json"),
        "te_path": str(PROC / "te_nb970_router.npy"),
    },
    {
        "id": "nb971_slsqp_6_anchor",
        "axis": "SLSQP convex 6-anchor blend: nb2189 + nb730 + nb562 + nb503 + nb950_lite + nb960 (honest cross-fit weights, l2=0.01)",
        "rationale": "nb239 4-way breakthrough memory says SLSQP DOES break ceilings at 4-6 anchors but degrades past ~5-7; 6-way is the upper edge of useful capacity",
        "status": "PENDING_script_not_on_disk_naming_collision_with_rdkit_only",
        "honest_rae": None,
        "summary_path": str(PROC / "nb971_slsqp_summary.json"),
        "te_path": str(PROC / "te_nb971_slsqp.npy"),
    },
    {
        "id": "nb972_trajectory",
        "axis": "this integrator -- snapshot cycle-131 slate + closed/open axes + cycle-132 plan",
        "rationale": None,
        "status": "DONE",
        "honest_rae": None,
        "summary_path": str(PROC / "nb972_summary.json"),
        "te_path": None,
    },
]

# --- 2. Floor check --------------------------------------------------------
candidate_rae = [m["honest_rae"] for m in CYCLE_131_METHODS if m["honest_rae"] is not None]
floor_broken = bool(candidate_rae) and min(candidate_rae) <= PRIOR_HONEST_FLOOR_RAE - 0.005
new_floor_rae = min(candidate_rae) if floor_broken else PRIOR_HONEST_FLOOR_RAE
new_floor_id = (
    next(m["id"] for m in CYCLE_131_METHODS if m["honest_rae"] == new_floor_rae)
    if floor_broken else PRIOR_HONEST_FLOOR_ID
)
FLOOR_TRANSITION = {
    "old_floor_rae": PRIOR_HONEST_FLOOR_RAE,
    "old_floor_id": PRIOR_HONEST_FLOOR_ID,
    "new_floor_rae": new_floor_rae,
    "new_floor_id": new_floor_id,
    "floor_broken": floor_broken,
    "delta_rae": round(new_floor_rae - PRIOR_HONEST_FLOOR_RAE, 4),
    "nb2189_best_honest_rae": NB2189_BEST_K20_MEDIAN_BAG,
    "reason": (
        "all 4 cycle-131 method scripts (nb968/nb969/nb970-router/nb971-slsqp) are PENDING_script_not_on_disk; "
        "no candidate honest cross-fit RAE recorded; floor preserved at nb2112 0.4698 (nb2189 best 0.4556)."
    ) if not floor_broken else "cycle-131 candidate beat 0.4698 by >= 0.005",
}

# --- 3. CLOSED axes (inherited + cycle-130 new closures) -------------------
# Cycle-130 closures added per nb964 snapshot + post-130 honest-cross-fit
# evaluation:
#   - F2 abstention (nb960): te_only honest summary never produced acceptable
#     cross-fit RAE; hard abstention on novel scaffolds OVER-shrinks and pushes
#     pred_std collapse (RAE worse than baseline pass-through).
#   - ChEMBL train-aug (nb962): summary materialized but cross-fit RAE > 0.55
#     -- external-data corpus does NOT transfer; OOD wall confirmed.
#   - heteroscedastic SE-weighted (nb963): NLL head produces tighter sigma but
#     no RAE gain; 1/sigma^2 loss weighting collapses to flat weights once
#     sigma_floor is enforced; closed as a deploy axis.
CLOSED_AXES_CYCLE_130_NEW = [
    "F2 abstention (hard scaffold-novelty gate -- nb960 self-train; te-only no honest gain; over-shrinks)",
    "ChEMBL training augmentation (nb962 -- external corpus does NOT transfer; OOD wall confirmed)",
    "heteroscedastic SE-weighted Gaussian-NLL head (nb963 -- sigma-weighted loss collapses to flat once floored)",
]

CLOSED_AXES_INHERITED_FROM_NB964 = [
    "K (top-K SHAP sweep)",
    "L (n_leaves)",
    "lr (learning_rate)",
    "mc (min_child_samples)",
    "ff (feature_fraction)",
    "monotone constraints",
    "DART boosting",
    "SHAP-seed (5 seeds, all bagged)",
    "row-bootstrap",
    "pooled-120 cross-fit",
    "XGB-SHAP feature import",
    "family-ablation (drop NR-family)",
    "residual-cascade (depth>2)",
    "tanh-target reparam",
    "sklearn-stack meta-learner",
    "conformal-shrink (nb2190 vacuous on K=28)",
    "contaminated-anchor swap (nb730 <-> nb562 te)",
    "alt-honest-anchor swap (nb503 <-> nb464)",
    "ensemble-v2-augmented (nb953 collapse to nb503_v1)",
]

CLOSED_AXES = CLOSED_AXES_INHERITED_FROM_NB964 + CLOSED_AXES_CYCLE_130_NEW

# --- 4. CANDIDATE-NEW CLOSED axes (only if cycle 131 ALL FAIL) -------------
# These will be appended to CLOSED_AXES at cycle-132 integration only if every
# nb968/nb969/nb970-router/nb971-slsqp honest cross-fit RAE >= 0.4698 - 0.005.
# Tracked here as PENDING_CLOSURE to keep the closure decision explicit.
CANDIDATE_NEW_CLOSED_AXES_IF_CYCLE_131_FAILS = [
    {"axis": "per-decile isotonic calibration (nb968)", "promotion_rule": "close if nb968 honest RAE >= 0.465"},
    {"axis": "PU-learning on 21k weak-label pool (nb969)", "promotion_rule": "close if nb969 honest RAE >= 0.465"},
    {"axis": "OOD sigmoid router (nb970)", "promotion_rule": "close if nb970 router honest RAE >= 0.465"},
    {"axis": "SLSQP 6-anchor blend (nb971)", "promotion_rule": "close if nb971 SLSQP honest RAE >= 0.465 (memory says SLSQP overfits past ~5 anchors)"},
]

# --- 5. STILL OPEN axes ----------------------------------------------------
STILL_OPEN_AXES = [
    "chemprop_v2 full retrain (nb950 background; lite proxy 0.587; full chemprop pending)",
    "2-stage stack on chemprop_v2 (PRE-unblind base -> honest meta on 253)",
    "NN / Transformer head (Graphormer / MAT / ChemBERTa fine-tune on aug corpus)",
    "structure-track features routed to activity (pose-quality / docking score / RMSD-to-template from Boltz-2 v1 outputs)",
    "ChemBERTa 384-dim K=28 (foundation embeddings with the SHAP-K=28 reduction recipe; deferred to cycle 132 from cycle-130-deep substrate)",
]

# --- 6. Cycle 132 plan (5 distinct methods) --------------------------------
# Branch logic: if cycle 131 closes its 4 NEW axes, cycle 132 must pivot to the
# 5 STILL_OPEN_AXES verbatim. If any of cycle 131 lands below 0.465, cycle 132
# extends that axis with a v2 variant instead of pivoting.
CYCLE_132_PLAN = [
    {
        "id": "nb1030_chemprop_v2_K28_residual",
        "axis": "chemprop_v2 (nb950 full) + nb2189-style K=28 SHAP residual on top; honest cross-fit on 253",
        "depends_on": "nb950 chemprop_v2 background completion + honest summary",
    },
    {
        "id": "nb1031_2stage_stack_clean_anchor",
        "axis": "PRE-unblind base = chemprop_v2 OOF; honest meta-stack on 253 = (nb2189, nb730, nb503, nb950_lite); Huber SLSQP, l2=0.02",
        "depends_on": "nb950 chemprop_v2 OOF on 4139",
    },
    {
        "id": "nb1032_chemberta_384d_K28",
        "axis": "ChemBERTa-77M-MTR 384-dim CLS embeddings + the SHAP-K=28 reduction recipe; LGBM head; honest cross-fit",
        "depends_on": "cycle-130-deep substrate (cached embeddings) -- ready",
    },
    {
        "id": "nb1033_structure_features_routed_to_activity",
        "axis": "pose-quality + RMSD-to-template + docking-score from Boltz-2 v1 outputs as 6-7 numerical features into nb2189 residual",
        "depends_on": "structure-track v1 outputs cached locally",
    },
    {
        "id": "nb1034_NN_transformer_head_aug_corpus",
        "axis": "small Graphormer or MAT FT-head on (4139 train + 901 aug); honest cross-fit on 253; -> blend candidate for nb1031 stack",
        "depends_on": "nb1030 release of Kaggle GPU slot",
    },
]

# --- 7. Decision summary ---------------------------------------------------
DECISION = {
    "next_action": "Hold floor at nb2112 0.4698 (nb2189 best 0.4556). Write nb968/nb969/nb970-router/nb971-slsqp scripts; rename existing nb970/nb971 artifacts to nb970_multi_conformer / nb971_rdkit_only to clear the namespace.",
    "rationale": (
        "Cycle 131 has 0 of 4 method scripts on disk under the planned semantics. "
        "nb970 and nb971 namespaces are taken by unrelated multi-conformer and RDKit-only artifacts; "
        "writing the router/SLSQP work requires either a name-suffix (e.g., nb970_router, nb971_slsqp) or a renumber. "
        "Floor preserved at nb2112 0.4698. Cycle-132 plan stages the 5 STILL-OPEN axes (chemprop_v2 residual, "
        "2-stage stack, ChemBERTa 384d, structure-features routed, NN/Transformer head) so that, regardless of "
        "cycle-131 outcomes, cycle-132 has 5 distinct disjoint methods queued."
    ),
    "abort_condition": "if nb968 per-decile isotonic OVER-shrinks pred_std to <0.55 (vs truth_std ~1.03), retire isotonic calibration axis and pivot to nb1032 ChemBERTa 384d in cycle 132 slot 1.",
}

# --- 8. Counts -------------------------------------------------------------
# Cycle count: cycle 131 in flight; method count approx since we don't enumerate
# every cached XGB/LGBM hyperparam variant -- ~5 distinct methods per cycle x
# 131 cycles + ~175 ad-hoc integrators, audits, calibration scripts, and
# structure-track variants = ~830 methods.
COUNTS = {
    "n_cycles": 131,
    "n_methods_approx": APPROX_METHODS_TO_DATE,
    "n_distinct_methods_cycle_131": 5,
    "n_distinct_methods_cycle_132_plan": 5,
}

summary = {
    "tag": "nb972",
    "cycle": CYCLE,
    "method": "cycle_131_trajectory_snapshot",
    "counts": COUNTS,
    "precondition_check": {
        "nb968_summary_present": (PROC / "nb968_summary.json").exists(),
        "nb969_summary_present": (PROC / "nb969_summary.json").exists(),
        "nb970_router_summary_present": (PROC / "nb970_router_summary.json").exists(),
        "nb971_slsqp_summary_present": (PROC / "nb971_slsqp_summary.json").exists(),
        "nb964_summary_present": (PROC / "nb964_summary.json").exists(),
        "nb2189_summary_present": (PROC / "nb2189_summary.json").exists(),
    },
    "cycle_131_methods": CYCLE_131_METHODS,
    "prior_honest_floor": {"id": PRIOR_HONEST_FLOOR_ID, "rae": PRIOR_HONEST_FLOOR_RAE},
    "nb2189_best_honest": {"id": "nb2189_K20_median_bag", "rae": NB2189_BEST_K20_MEDIAN_BAG},
    "floor_transition": FLOOR_TRANSITION,
    "closed_axes_inherited_from_nb964": CLOSED_AXES_INHERITED_FROM_NB964,
    "closed_axes_cycle_130_new": CLOSED_AXES_CYCLE_130_NEW,
    "closed_axes_all": CLOSED_AXES,
    "candidate_new_closed_axes_if_cycle_131_fails": CANDIDATE_NEW_CLOSED_AXES_IF_CYCLE_131_FAILS,
    "still_open_axes": STILL_OPEN_AXES,
    "cycle_132_plan": CYCLE_132_PLAN,
    "decision": DECISION,
}


def main() -> None:
    out = PROC / "nb972_summary.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out}")
    print(f"cycle={CYCLE}  n_methods_approx~{APPROX_METHODS_TO_DATE}")
    print(f"cycle131_methods={len(CYCLE_131_METHODS)}  cycle132_plan={len(CYCLE_132_PLAN)}")
    print(f"honest floor: {PRIOR_HONEST_FLOOR_ID} RAE={PRIOR_HONEST_FLOOR_RAE}")
    print(f"nb2189 best honest: K=20 median bag RAE={NB2189_BEST_K20_MEDIAN_BAG}")
    print(f"floor_broken={FLOOR_TRANSITION['floor_broken']}")
    print(f"next: {DECISION['next_action']}")


if __name__ == "__main__":
    main()
