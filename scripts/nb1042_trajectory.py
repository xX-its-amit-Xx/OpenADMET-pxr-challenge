"""
nb1042_trajectory.py - Cycle 133 trajectory snapshot.

This integrator records cycle 133 (relaunch + verify + PXR-direct + LB poll),
ratifies the cycle-131 closures that nb972 had only proposed, posts the
nb971_verify verdict, and stages 5 distinct methods for cycle 134.

Cycle 133 activities (this cycle):
  RELAUNCH:  nb950 chemprop_aux v2 - crashed 2x (nb950_v2.log: pandas dtype
             coercion fail, nb950_v3.log: fallback also fails on same string
             coercion). 2nd patch applied; 3rd attempt staged from bash this
             cycle. STATUS: STILL_PENDING (no honest summary).
  VERIFY:    nb971_verify_summary.json -- OUTER 5-fold cross-fit reproduced
             RAE 0.4675 (mae, seed=0) verbatim. Multi-seed mean=0.4696 std=0.0026
             max=0.4746. claim_matches_reproduction=true. Beats bare floor
             0.4698 but FAILS 0.005 delta target (0.4648) and worst seed 0.4746
             exceeds floor -- VERDICT MARGINAL/HOLD, not in-sample-overfit.
  PXR-DIRECT: ChEMBL PXR-direct external data pipe (te_oof_lgbm_chembl_pxr_direct
             cached); no honest cross-fit RAE recorded as a deploy candidate
             this cycle.
  LB POLL:   activity LB still FROZEN at rank 262 / RAE 0.7655 / submitted
             2026-05-26 04:45 UTC (visible n=328). No new activity submission
             graded since 2026-05-26 despite 9+ PRIMARY-* submissions queued
             through 2026-06-07 20:25:03 UTC. Structure LB rank 10 / LDDT-PLI
             0.4996 / submitted 2026-06-02 16:50 UTC (visible n=50).

Reads (read-only):
  data/processed/nb964_summary.json
  data/processed/nb972_summary.json
  data/processed/nb971_summary.json
  data/processed/nb971_verify_summary.json
  data/processed/nb968_summary.json
  data/processed/nb969_summary.json
  data/processed/nb970_summary.json
  data/processed/nb966_summary.json
  data/processed/leaderboard_log.csv (tail)
  data/processed/submission_log.csv  (tail)

Writes:
  data/processed/nb1042_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

CYCLE = 133
APPROX_METHODS_TO_DATE = 845      # 131 (nb972) -> 132 (~5 more) -> 133 (~5 more)
PRIOR_HONEST_FLOOR_RAE = 0.4698   # nb2112 = nb2103 K=28 median bag
PRIOR_HONEST_FLOOR_ID = "nb2112_chemprop_aux_K28_median_bag"
NB2189_BEST_K20_MEDIAN_BAG = 0.4556

# --- 1. Cycle 133 activities (NOT 5 new methods this cycle) ----------------
# This is a verify + relaunch cycle; the deliverable is consolidating the
# nb968/nb969/nb970/nb971 outcomes from cycle 131 into ratified closures, then
# verdicting nb971's claim.
CYCLE_133_ACTIVITIES = [
    {
        "id": "relaunch_nb950_chemprop_v2",
        "axis": "chemprop_aux v2 -- 3rd bash retry after 2 prior crashes",
        "prior_crashes": [
            {"log": "scripts/nb950_v2.log", "fail": "pandas ValueError 'could not convert string to float: -' in fit_chemprop"},
            {"log": "scripts/nb950_v3.log", "fail": "fallback LGBM hit same string coercion in featurization path"},
        ],
        "patch_applied": "2nd patch (cleans semi_pure/crudes/counter '-' tokens) is committed; current attempt is the 3rd",
        "status": "STILL_PENDING",
        "honest_rae": None,
    },
    {
        "id": "verify_nb971_outer_cv",
        "axis": "independent OUTER 5-fold reproduction of nb971_slsqp_6anchor cross-fit RAE 0.4675",
        "claimed_rae": 0.4675,
        "reproduced_rae_mae_seed0": 0.4675,
        "reproduced_rae_huber_seed0": 0.4685,
        "multi_seed_mean_mae": 0.4696,
        "multi_seed_max_mae": 0.4746,
        "multi_seed_std_mae": 0.0026,
        "claim_matches_reproduction": True,
        "beats_floor_strict_005": False,
        "robust_across_seeds": False,
        "verdict": "HONEST_BUT_MARGINAL",
        "rationale": (
            "nb971 IS NOT in-sample-overfit; reproduced verbatim by independent OUTER 5-fold cross-fit. "
            "However it beats the bare floor 0.4698 by only 0.0023 (target 0.005). Worst-seed CV is 0.4746 "
            "(WORSE than floor). Policy: HOLD; do NOT deploy until either floor moves or multi-seed mean "
            "lands below 0.4648."
        ),
    },
    {
        "id": "pxr_direct_chembl_external",
        "axis": "ChEMBL PXR-direct external corpus -- featurized + oof recorded; no honest deploy candidate",
        "artifacts_present": [
            "data/processed/chembl_pxr_new_external.parquet",
            "data/processed/oof_lgbm_chembl_pxr_direct.npy",
            "data/processed/te_oof_lgbm_chembl_pxr_direct.npy",
        ],
        "honest_rae": None,
        "status": "DATA_CACHED_NOT_SUBMITTED",
    },
    {
        "id": "lb_poll",
        "axis": "scripts/log_lb_scores.py read-only poll",
        "activity": {
            "rank": 262,
            "rae": 0.7655,
            "visible_n": 328,
            "submitted_utc": "2026-05-26 04:45",
            "frozen": True,
            "frozen_since_2026_05_26": True,
            "n_queued_since_05_26": 9,  # 9 PRIMARY-* submissions (cycle 1-2-3 + mm-audit set) all ungraded
        },
        "structure": {
            "rank": 10,
            "lddt_pli": 0.4996,
            "visible_n": 50,
            "submitted_utc": "2026-06-02 16:50",
            "frozen": False,
        },
    },
]

# --- 2. CLOSED axes update (ratify cycle-131 candidates as CLOSED) ---------
# nb972 cycle-131 trajectory listed 4 CANDIDATE closures pending the actual
# cross-fit numbers; all 4 cycle-131 method summaries now exist, so they get
# ratified as CLOSED.
CLOSED_AXES_CYCLE_131_RATIFIED = [
    {
        "axis": "per-decile isotonic calibration (nb968)",
        "best_variant_rae": 0.4793,           # A_global (best of 4 sub-methods)
        "baseline_ref": 0.4698,
        "delta_vs_ref": 0.0095,
        "verdict": "FLAT_VS_NB2103_MEDIAN -- per-decile MAKES IT WORSE; A_global ~+0.006, B_per_decile +0.028",
        "closure_rule_passed": True,
    },
    {
        "axis": "PU-learning on 21k weak-label pool (nb969)",
        "best_variant_rae": 0.6446,           # B_PU_AUX (best of 3 sub-methods)
        "baseline_ref": 0.4698,
        "delta_vs_ref": 0.1748,
        "verdict": "PU_DOES_NOT_BEAT_NB2103_K28 -- best (B_PU_AUX) 0.6446 worse by +0.175",
        "closure_rule_passed": True,
    },
    {
        "axis": "OOD sigmoid router (nb970)",
        "best_variant_rae": 0.4716,           # train_mean, alpha=20, threshold=0.30
        "baseline_ref": 0.4698,
        "delta_vs_ref": 0.0018,
        "verdict": "FLAT_VS_NB2103_MEDIAN -- best variant collapses to pass-through (w_mean=0.976)",
        "closure_rule_passed": True,
    },
    {
        "axis": "ChemBERTa-77M-MTR residual (nb966)",
        "best_variant_rae": 0.5028,           # D_shap_top28_of_501
        "baseline_ref": 0.4698,
        "delta_vs_ref": 0.0330,
        "verdict": "NO_BERTA_METHOD_BEATS_NB2103_K28_BY_MARGIN -- best (D_shap_top28) 0.5028 worse by +0.033",
        "closure_rule_passed": True,
    },
]

# nb971 NOT closed yet -- verdict is MARGINAL/HOLD, axis stays OPEN (the
# SLSQP-6-anchor pattern is the only cycle-131 method to actually beat the
# bare floor, even if it falls short of the 0.005 delta target).
NB971_STATUS = {
    "axis": "SLSQP 6-anchor convex blend (nb971)",
    "best_rae_crossfit_mae": 0.4675,
    "baseline_ref": 0.4698,
    "delta_vs_ref": -0.0023,                # negative = BETTER
    "delta_target_required": -0.005,        # must beat floor by at least 5 mRAE
    "verdict": "HONEST_BUT_MARGINAL",
    "verify_summary_path": str(PROC / "nb971_verify_summary.json"),
    "axis_status": "OPEN_KEPT_FOR_CYCLE_134_EXTENSION",
}

# --- 3. Full CLOSED axes (inherited from nb972 + cycle-131 ratifications) --
CLOSED_AXES_INHERITED_FROM_NB972 = [
    # cycle-130 + earlier inheritance
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
    "F2 abstention (nb960 self-train)",
    "ChEMBL training augmentation (nb962)",
    "heteroscedastic SE-weighted Gaussian-NLL (nb963)",
]

CLOSED_AXES_NEW_CYCLE_131_RATIFIED_NAMES = [
    "per-decile isotonic calibration (nb968 -- A_global 0.4793, B_per_decile 0.5018; ALL 4 sub-methods FLAT or WORSE)",
    "PU-learning 21k weak-label (nb969 -- best B_PU_AUX 0.6446; +0.175 vs floor)",
    "OOD sigmoid router (nb970 -- best 0.4716; collapses to pass-through at high alpha)",
    "ChemBERTa-77M-MTR residual (nb966 -- best D_shap_top28 0.5028; +0.033 vs floor)",
]

CLOSED_AXES_ALL = CLOSED_AXES_INHERITED_FROM_NB972 + CLOSED_AXES_NEW_CYCLE_131_RATIFIED_NAMES

# --- 4. STILL-OPEN axes ----------------------------------------------------
STILL_OPEN_AXES = [
    "chemprop_v2 full retrain (nb950 -- crashed 2x, 3rd bash attempt staged; lite proxy 0.587)",
    "SLSQP 6-anchor blend (nb971 -- HONEST 0.4675 but marginal; needs anchor diversification to clear 0.005 delta)",
    "2-stage stack on chemprop_v2 (PRE-unblind base -> honest meta on 253)",
    "structure-track features routed to activity (pose-quality / RMSD-to-template from Boltz-2 v1)",
    "external scaffold-diverse data (PXR-direct ChEMBL cached but no deploy candidate yet)",
]

# --- 5. Honest floor (unchanged) -------------------------------------------
# Floor moves only if a candidate beats 0.4698 by >=0.005.
# nb971 honest cross-fit 0.4675 is BETTER than 0.4698 but only by 0.0023,
# fails 0.005 delta. Floor stays at nb2112 0.4698.
FLOOR_TRANSITION = {
    "old_floor_rae": PRIOR_HONEST_FLOOR_RAE,
    "old_floor_id": PRIOR_HONEST_FLOOR_ID,
    "new_floor_rae": PRIOR_HONEST_FLOOR_RAE,
    "new_floor_id": PRIOR_HONEST_FLOOR_ID,
    "floor_broken_strict_005": False,
    "floor_broken_loose": True,             # nb971 0.4675 < 0.4698 by 0.0023
    "nb971_loose_break_delta": -0.0023,
    "nb2189_best_honest_rae": NB2189_BEST_K20_MEDIAN_BAG,
    "reason": (
        "Honest floor preserved at nb2112 0.4698. nb971 0.4675 beats it by only 2.3 mRAE "
        "(target 5 mRAE), worst-seed CV 0.4746 exceeds floor, so HOLD per the 0.005 policy delta. "
        "nb950 chemprop_v2 still pending (3rd bash attempt). No cycle-131 closure candidate moved floor."
    ),
}

# --- 6. Cycle 134 plan (5 distinct methods; pivots on nb971 verdict) -------
# Branch logic from this cycle's evidence:
#   * nb971 verify is HONEST (not in-sample-overfit), so the SLSQP 6-anchor
#     axis stays OPEN. Cycle 134 method 1 EXTENDS it: nb971_v2 with a 7th
#     anchor (chemprop_v2 when nb950 lands) and weight-shrinkage.
#   * 4 of cycle-131's 5 methods CLOSED, so cycle 134 must seed 4 new axes
#     (NOT just 1 extension).
#   * nb1014 multi-seed bag is the dominant PRIMARY-1 from cycle 5 (RAE 0.5930)
#     but LB never graded; cycle 134 must include a method whose honest
#     cross-fit is at least competitive with nb2189 best 0.4556.
CYCLE_134_PLAN = [
    {
        "id": "nb1050_slsqp_7_anchor_chemprop_v2",
        "axis": "EXTEND nb971: add chemprop_v2 (nb950 full) as 7th anchor with weight-shrinkage l2=0.02 on SLSQP; only fires if nb950 honest <0.55",
        "rationale": "nb971 0.4675 is honest; adding a 7th decorrelated anchor is the single highest-prior shot at clearing 0.005 delta target (0.4648)",
        "depends_on": "nb950 chemprop_v2 retry #3 honest summary",
        "axis_status": "EXTENSION_OF_OPEN_AXIS",
    },
    {
        "id": "nb1051_pxr_direct_chembl_anchor",
        "axis": "use cached PXR-direct ChEMBL OOF as a NEW PRE-unblind anchor; honest cross-fit on 253; if RAE<0.55 add to nb971_v2 8-anchor blend",
        "rationale": "external scaffold-diverse corpus is the explicit P3 lever from pm06 framing; data already cached this cycle, just needs honest summary",
        "depends_on": "no upstream",
        "axis_status": "NEW_OPEN_AXIS",
    },
    {
        "id": "nb1052_2stage_stack_clean_anchor",
        "axis": "PRE-unblind base = chemprop_aux_v1 OOF; honest meta-stack on 253 = (nb2189, nb730, nb503, nb464); Huber SLSQP l2=0.02",
        "rationale": "STILL_OPEN axis from nb972; nb971's 6-anchor convex result proves 4-6 anchor stacks have honest capacity; 2-stage protects against meta overfit",
        "depends_on": "no upstream",
        "axis_status": "NEW_OPEN_AXIS",
    },
    {
        "id": "nb1053_structure_features_routed_to_activity",
        "axis": "Boltz-2 v1 outputs (pose-quality, RMSD-to-template, docking-score) for the 513 -> 6-7 numerical features into nb2189 residual",
        "rationale": "STILL_OPEN axis from nb972; structure-track features are the only NEW substrate that hasn't been tested against the OOD wall yet",
        "depends_on": "structure v1 outputs cached locally",
        "axis_status": "NEW_OPEN_AXIS",
    },
    {
        "id": "nb1054_trajectory",
        "axis": "this integrator -- snapshot cycle-134 outcomes + closures + cycle-135 plan",
        "rationale": "ratify whichever of nb1050-nb1053 fires; closure rules: any axis with honest RAE>=0.465 closes",
        "depends_on": "nb1050-nb1053 honest summaries",
        "axis_status": "INTEGRATOR",
    },
]

# --- 7. Counts -------------------------------------------------------------
COUNTS = {
    "n_cycles": CYCLE,
    "n_methods_approx": APPROX_METHODS_TO_DATE,
    "n_distinct_methods_cycle_133": 4,        # 4 activities (relaunch, verify, pxr-direct, lb poll); NOT a new-method cycle
    "n_distinct_methods_cycle_134_plan": 5,
    "n_axes_closed_this_cycle": 4,            # nb968/nb969/nb970/nb966 ratified
    "n_axes_still_open": len(STILL_OPEN_AXES),
}

# --- 8. LB poll summary ----------------------------------------------------
LB_POLL = {
    "activity": {
        "rank": 262,
        "rae": 0.7655,
        "visible_n": 328,
        "submitted_utc": "2026-05-26 04:45",
        "frozen_since_2026_05_26": True,
        "n_ungraded_submissions_since": 9,
        "implication": "LB queue backlog; cannot trust any 'predicted LB' from the 9 mm-audit / cycle-1/2/3 submissions until grading resumes",
    },
    "structure": {
        "rank": 10,
        "lddt_pli": 0.4996,
        "visible_n": 50,
        "submitted_utc": "2026-06-02 16:50",
        "frozen_since_2026_06_02": True,
    },
    "any_new_activity_graded_since_2026_05_26": False,
    "last_3_leaderboard_log_rows_at_poll_time": [
        "2026-06-07 21:24:34 UTC,activity,262,0.7655,328 (submitted 2026-05-26 04:45)",
        "2026-06-07 21:24:35 UTC,structure,10,0.4996,50 (submitted 2026-06-02 16:50)",
        "2026-06-07 21:27:08 UTC,activity,262,0.7655,328 (submitted 2026-05-26 04:45)",
    ],
    "last_3_submission_log_rows_at_poll_time": [
        "2026-06-07 04:25:02 UTC,54_deep_ensemble_uncertainty.csv,in_RAE 0.6657",
        "2026-06-07 12:25:02 UTC,82_selectivity_aware.csv,in_RAE 0.6730",
        "2026-06-07 20:25:03 UTC,67_lgbm_chembl_all_nr_weighted.csv,in_RAE 0.6746",
    ],
}

# --- 9. Decision summary ---------------------------------------------------
DECISION = {
    "next_action": (
        "Hold floor at nb2112 0.4698 (nb2189 best 0.4556). nb971 0.4675 stays HOLD per 0.005 policy delta. "
        "Wait for nb950 chemprop_v2 retry #3 honest summary, then execute cycle-134 plan."
    ),
    "rationale": (
        "Cycle 131 fully closed: nb968 (per-decile iso) FLAT, nb969 (PU 21k) +0.175, nb970 (OOD router) FLAT, "
        "nb966 (BERTa) +0.033. ONLY nb971 (SLSQP 6-anchor) cleared bare floor, and only marginally; verify is honest. "
        "Cycle 133 LB poll confirms activity grading is FROZEN since 2026-05-26 -- 9 queued submissions ungraded, "
        "so 'predicted LB' estimates from the mm-audit / cycle-1/2/3 batches cannot be ladder-trusted. "
        "Cycle 134 must SEED 4 NEW axes (PXR-direct ChEMBL anchor, 2-stage stack, structure features routed, "
        "+ 1 extension of nb971) so that, regardless of nb950 outcome, the integrator has 5 distinct disjoint methods."
    ),
    "abort_condition": (
        "if nb950 chemprop_v2 retry #3 fails again (4th crash), retire the chemprop_v2 substrate, "
        "drop nb1050 from cycle-134 plan, and promote nb1051 (PXR-direct ChEMBL anchor) to slot 1."
    ),
}

summary = {
    "tag": "nb1042",
    "cycle": CYCLE,
    "method": "cycle_133_relaunch_verify_pxr_direct_lb_poll_snapshot",
    "counts": COUNTS,
    "precondition_check": {
        "nb950_v3_log_present": (ROOT / "scripts" / "nb950_v3.log").exists(),
        "nb971_summary_present": (PROC / "nb971_summary.json").exists(),
        "nb971_verify_summary_present": (PROC / "nb971_verify_summary.json").exists(),
        "nb968_summary_present": (PROC / "nb968_summary.json").exists(),
        "nb969_summary_present": (PROC / "nb969_summary.json").exists(),
        "nb970_summary_present": (PROC / "nb970_summary.json").exists(),
        "nb966_summary_present": (PROC / "nb966_summary.json").exists(),
        "nb964_summary_present": (PROC / "nb964_summary.json").exists(),
        "nb972_summary_present": (PROC / "nb972_summary.json").exists(),
        "pxr_direct_artifacts_present": (PROC / "te_oof_lgbm_chembl_pxr_direct.npy").exists(),
        "leaderboard_log_present": (PROC / "leaderboard_log.csv").exists(),
        "submission_log_present": (PROC / "submission_log.csv").exists(),
    },
    "cycle_133_activities": CYCLE_133_ACTIVITIES,
    "lb_poll": LB_POLL,
    "honest_floor": {
        "id": PRIOR_HONEST_FLOOR_ID,
        "rae": PRIOR_HONEST_FLOOR_RAE,
        "broken_cycle_133": False,
        "broken_loose_by_nb971": True,
        "policy_delta_required": 0.005,
    },
    "nb971_verify_verdict": {
        "claim_matches_reproduction": True,
        "best_outer_cv_rae": 0.4675,
        "is_in_sample_overfit": False,
        "verdict": "HONEST_BUT_MARGINAL",
        "ladder_action": "DO NOT DEPLOY (within margin); keep nb2112 as PRIMARY-1",
    },
    "nb950_chemprop_v2_status": {
        "crashes": 2,
        "log_v2_error": "ValueError: could not convert string to float: '-' in chemprop fit",
        "log_v3_error": "fallback LGBM hit same string coercion in featurization path",
        "patches_applied": 2,
        "current_attempt": "3rd bash attempt this cycle",
        "honest_summary_present": (PROC / "nb950_summary.json").exists(),
        "verdict": "STILL_PENDING",
    },
    "nb2189_best_honest": {"id": "nb2189_K20_median_bag", "rae": NB2189_BEST_K20_MEDIAN_BAG},
    "floor_transition": FLOOR_TRANSITION,
    "closed_axes_inherited_from_nb972": CLOSED_AXES_INHERITED_FROM_NB972,
    "closed_axes_cycle_131_ratified_detail": CLOSED_AXES_CYCLE_131_RATIFIED,
    "closed_axes_cycle_131_ratified_names": CLOSED_AXES_NEW_CYCLE_131_RATIFIED_NAMES,
    "closed_axes_all": CLOSED_AXES_ALL,
    "nb971_open_axis_status": NB971_STATUS,
    "still_open_axes": STILL_OPEN_AXES,
    "cycle_134_plan": CYCLE_134_PLAN,
    "decision": DECISION,
}


def main() -> None:
    out = PROC / "nb1042_summary.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out}")
    print(f"cycle={CYCLE}  n_methods_approx~{APPROX_METHODS_TO_DATE}")
    print(f"cycle133_activities={len(CYCLE_133_ACTIVITIES)}  cycle134_plan={len(CYCLE_134_PLAN)}")
    print(f"honest floor: {PRIOR_HONEST_FLOOR_ID} RAE={PRIOR_HONEST_FLOOR_RAE} (unbroken)")
    print(f"nb971 verify: HONEST cross-fit 0.4675; verdict=MARGINAL/HOLD")
    print(f"LB activity: rank=262 RAE=0.7655 frozen since 2026-05-26 (9 ungraded since)")
    print(f"LB structure: rank=10 LDDT-PLI=0.4996 (2026-06-02)")
    print(f"closed this cycle: nb968/nb969/nb970/nb966 (4 axes)")
    print(f"next: {DECISION['next_action']}")


if __name__ == "__main__":
    main()
