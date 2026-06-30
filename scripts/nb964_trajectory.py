"""
nb964_trajectory.py - Cycle 130 trajectory snapshot.

Records the 5 cycle-130 method slate, the running cycle/method counts
(~130 cycles, ~825 methods), the top-10 HONEST cross-fit RAE leaderboard with
contaminated entries explicitly excluded per
memory/feedback_anchor_contamination_chain, and the cycle-131 plan.

Cycle 130 methods (slated):
  nb950  chemprop_aux v2 (BACE+ChEMBL+Tox21 augmented training) - background launch
  nb960  F2 abstain / pseudo-label self-train on SP-only compounds
  nb962  ChEMBL training-data augmentation
  nb963  heteroscedastic NLL head
  nb964  this trajectory integrator

PRECONDITION CHECK (2026-06-08):
  - nb950 chemprop full retrain: BACKGROUND_LAUNCHED (lite LGBM proxy nb950_lite already
    completed, RAE_crossfit 0.5870)
  - nb951 LGBM v2 (morgan-only proxy): COMPLETED but BLOCKED for promotion (RAE 0.6880)
  - nb952 Tanimoto-OOD diagnostic: COMPLETED -- naive 1-NN replacement HURTS RAE on 4
    variants for both anchors (verdict: augmentation alone does NOT break OOD wall)
  - nb953 SLSQP v2 blend: COMPLETED, best candidate nb503_v2 = 0.5124 (worse than
    nb2189 0.4556 honest floor; deploy weight collapses to nb503_v1 = 1.0)
  - nb960 F2 abstain pseudo-label: te_nb960.npy present, summary not yet snapshotted
  - nb961 ExtraTrees: te_nb961.npy present
  - nb962 ChEMBL aug: SCRIPT NOT YET ON DISK
  - nb963 heteroscedastic: SCRIPT NOT YET ON DISK

Reads (read-only):
  data/processed/nb2189_summary.json    (current honest floor anchor)
  data/processed/nb953_summary.json     (cycle 130 v2 blend outcomes)
  data/processed/nb952_summary.json     (OOD wall diagnostic)

Writes:
  data/processed/nb964_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

CYCLE = 130
APPROX_METHODS_TO_DATE = 825   # 130 cycles x ~5 methods + integrator drift + structure track
PRIOR_HONEST_FLOOR_RAE = 0.4698          # nb2112 = nb2103 K=28 median bag, chemprop_aux anchor
PRIOR_HONEST_FLOOR_ID = "nb2112_chemprop_aux_K28_median_bag"
NB2189_BEST_K20_MEDIAN_BAG = 0.4556     # nb2189 best (K=20 median bag); used as best-honest
NB2189_K28_MEDIAN_BAG = 0.4646          # nb2189 K=28 median bag (cross-checks nb2112 0.4698)

# --- 1. Five cycle-130 methods ---------------------------------------------
CYCLE_130_METHODS = [
    {
        "id": "nb950_chemprop_aux_v2",
        "axis": "chemprop_aux retrained on augmented substrate (4139 train + 95 semi-pure + 456 crudes)",
        "status": "BACKGROUND_LAUNCHED",
        "proxy_lite_rae_crossfit": 0.5870,    # LGBM(combined) proxy from nb953
        "honest_rae": None,                    # full chemprop pending
        "summary_path": str(PROC / "nb950_summary.json"),
        "te_path": str(PROC / "te_nb950_lite.npy"),
    },
    {
        "id": "nb960_f2_abstain_self_train",
        "axis": "F2 abstain / pseudo-label self-train on SP-only compounds (~8.5k SMILES, std<0.3 gate)",
        "status": "COMPLETED_te_only",
        "honest_rae": None,                    # summary JSON not snapshotted yet
        "summary_path": str(PROC / "nb960_summary.json"),
        "te_path": str(PROC / "te_nb960.npy"),
    },
    {
        "id": "nb962_chembl_aug",
        "axis": "ChEMBL PXR + Tox21 + BACE training-set augmentation as a fresh anchor refit",
        "status": "PENDING_script_not_on_disk",
        "honest_rae": None,
        "summary_path": str(PROC / "nb962_summary.json"),
        "te_path": str(PROC / "te_nb962.npy"),
    },
    {
        "id": "nb963_heteroscedastic_nll",
        "axis": "heteroscedastic Gaussian-NLL head (predict mu + sigma; loss weights by 1/sigma^2)",
        "status": "PENDING_script_not_on_disk",
        "honest_rae": None,
        "summary_path": str(PROC / "nb963_summary.json"),
        "te_path": str(PROC / "te_nb963.npy"),
    },
    {
        "id": "nb964_trajectory",
        "axis": "this integrator -- snapshot cycle-130 outcomes + ladder + cycle-131 plan",
        "status": "DONE",
        "honest_rae": None,
        "summary_path": str(PROC / "nb964_summary.json"),
        "te_path": None,
    },
]

# --- 2. Top-10 HONEST cross-fit RAE leaderboard ----------------------------
# CONTAMINATION POLICY (per memory feedback_anchor_contamination_chain):
#   - Any candidate whose anchor is trained on the 253 unblind labels is EXCLUDED
#     (te_X.npy reports in-sample optimism, not honest LB-faithful RAE).
#   - Only PRE-unblind anchors with HONEST cross-fit (5-fold on the 253) survive.
#   - nb1014 / nb1133 / nb1101 / nb1103 contamination chain is EXCLUDED.
#   - chemprop_aux_v1 IS clean (PRE-unblind training); used as anchor by nb2112/nb2189.
TOP10_HONEST = [
    {"rank":  1, "id": "nb2189_K20_median_bag",    "rae": 0.4556, "anchor": "nb562_pred_oof",       "clean": True,  "note": "best honest residual; 5-seed median bag, K=20 SHAP top features"},
    {"rank":  2, "id": "nb2103_K28_median_bag",    "rae": 0.4698, "anchor": "chemprop_aux_v1",      "clean": True,  "note": "deployed as nb2112; matches PRIOR_HONEST_FLOOR_RAE"},
    {"rank":  3, "id": "nb703_p2_p3_slsqp",        "rae": 0.4928, "anchor": "nb562 + nb702 + nb730","clean": True,  "note": "p2+p3 phase-2 blend; superseded but still honest"},
    {"rank":  4, "id": "nb730_multi_seed_null_ens","rae": 0.4603, "anchor": "nb562_lambda_discount","clean": True,  "note": "5-seed LGBM + MACCS, lambda discount on nb562; was PRIMARY-1 cycle 5"},
    {"rank":  5, "id": "nb562_rank_stretch",       "rae": 0.5065, "anchor": "self (scalar s=1.05)", "clean": True,  "note": "post-hoc rank-stretch on grand_v6b_calib base"},
    {"rank":  6, "id": "nb503_curriculum_v1",      "rae": 0.5116, "anchor": "self",                 "clean": True,  "note": "curriculum + residual stack; OOF anchor used by nb953"},
    {"rank":  7, "id": "nb503_v2_slsqp_aug",       "rae": 0.5124, "anchor": "nb950_lite+nb951_lite","clean": True,  "note": "cycle-130 SLSQP on augmented pool; weights collapse to nb503_v1"},
    {"rank":  8, "id": "nb432_grand_v6b_calib",    "rae": 0.5519, "anchor": "self",                 "clean": True,  "note": "grand_v6b_calib base; PRIMARY-2 candidate"},
    {"rank":  9, "id": "nb950_lite_lgbm_v2",       "rae": 0.5870, "anchor": "LGBM(combined) on train+aug","clean": True, "note": "lite proxy for nb950 chemprop full retrain"},
    {"rank": 10, "id": "chemprop_aux_v1_insample", "rae": 0.6216, "anchor": "chemprop_aux",         "clean": True,  "note": "v1 PRE-unblind base; honest in_RAE on 253; ANCHOR of nb2112/nb2189"},
]

EXCLUDED_CONTAMINATED = [
    {"id": "nb1014_cycle4_nb932",           "te_rae_apparent": 0.32, "reason": "nb932-anchored chain; trained on 253 unblind labels"},
    {"id": "nb1133_nb1014_residual",        "te_rae_apparent": 0.30, "reason": "nb1014 -> residual; same contamination chain"},
    {"id": "nb1101_nb1014_bag",             "te_rae_apparent": 0.31, "reason": "nb1014 bagging variant; chain inherited"},
    {"id": "nb1103_nb1014_bag",             "te_rae_apparent": 0.31, "reason": "nb1014 bagging variant; chain inherited"},
    {"id": "nb540_train_oof_slsqp",         "te_rae_apparent": 0.518, "reason": "train-OOF blend weights; jumps to 0.6234 unblind (+0.105 transfer shift)"},
    {"id": "any_anchor_swap_nb730_nb562",   "te_rae_apparent": "<0.40", "reason": "post-unblind te files swapped; voided per data-integrity audit 2026-06-01"},
]

# --- 3. Floor check --------------------------------------------------------
candidate_rae = [m["honest_rae"] for m in CYCLE_130_METHODS if m["honest_rae"] is not None]
floor_moved = bool(candidate_rae) and min(candidate_rae) <= PRIOR_HONEST_FLOOR_RAE - 0.005
new_floor_rae = min(candidate_rae) if floor_moved else PRIOR_HONEST_FLOOR_RAE
FLOOR_TRANSITION = {
    "old_floor_rae": PRIOR_HONEST_FLOOR_RAE,
    "old_floor_id": PRIOR_HONEST_FLOOR_ID,
    "new_floor_rae": new_floor_rae,
    "floor_moved": floor_moved,
    "delta_rae": round(new_floor_rae - PRIOR_HONEST_FLOOR_RAE, 4),
    "nb2189_best_honest_rae": NB2189_BEST_K20_MEDIAN_BAG,
    "reason": (
        "nb950 (chemprop v2 full) still BACKGROUND, nb951 (LGBM v2 0.6880) WORSE than v1, "
        "nb952 OOD diagnostic confirms augmentation alone does NOT close OOD wall, "
        "nb953 SLSQP collapses to nb503_v1, nb960/961 te-only (no honest summary). "
        "Floor unchanged at nb2112 0.4698."
    ) if not floor_moved else "cycle-130 candidate beat 0.4698",
}

# --- 4. NEW data-integration status ----------------------------------------
DATA_INTEGRATION_STATUS = {
    "chemprop_v2_full": {
        "id": "nb950_chemprop_aux_v2",
        "status": "BACKGROUND_LAUNCHED",
        "proxy_lite_rae_crossfit": 0.5870,
        "verdict": "full chemprop retrain pending; lite LGBM proxy is 5.9 RAE -- below v1 chemprop in_RAE (0.6216) but well above honest floor 0.4698",
    },
    "lgbm_v2": {
        "id": "nb951_lite",
        "status": "BLOCKED_pending_nb950",
        "rae_crossfit": 0.6880,
        "verdict": "morgan-only LGBM v2 on aug corpus is +0.018 WORSE than v1 -- aug payload does not transfer; blocked from promotion",
    },
    "augmentation_alone": {
        "ids": ["nb952_tanimoto_ood", "nb953_slsqp_aug"],
        "verdict": "CONFIRMED augmentation alone does NOT break the OOD wall; nb952 diagnostic shows naive 1-NN-from-NEW HURTS RAE on all 4 variants; nb953 SLSQP weights collapse to nb503_v1=1.0",
    },
}

# --- 5. CLOSED axes (updated from nb954) -----------------------------------
CLOSED_AXES = [
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

# --- 6. STILL OPEN axes ----------------------------------------------------
STILL_OPEN_AXES = [
    "chemprop_v2 full retrain (nb950 background; lite proxy 0.587)",
    "F2 abstention (nb960 cycle-130 te only; needs honest cross-fit summary)",
    "ChEMBL training augmentation (nb962 PENDING -- script not on disk)",
    "heteroscedastic NLL head (nb963 PENDING -- script not on disk)",
    "NN / Transformer head (Graphormer / MAT / ChemBERTa-FT on aug corpus)",
    "2-stage stack on truly clean anchor (PRE-unblind base -> honest meta on 253)",
    "external-data abstention via Tanimoto-novelty gate (P3 partial unlock)",
]

# --- 7. Cycle 131 plan (5 distinct methods conditional on 130 outcomes) ----
# Branch logic:
#   IF nb950 chemprop_v2 full lands below 0.55 honest -> build nb1020-series residuals on it.
#   ELIF nb960/961 honest summary surfaces and RAE < 0.50 -> push F2 abstention router.
#   ELSE -> pivot to pose-quality + NN head (cycle-131 plan below).
CYCLE_131_PLAN = [
    {
        "id": "nb1020_chemprop_v2_residual_K28",
        "axis": "if nb950 chemprop_v2 honest RAE < 0.55, fit nb2189-style K=28 SHAP residual on top",
        "depends_on": "nb950_chemprop_aux_v2 honest summary",
    },
    {
        "id": "nb1021_f2_abstention_router",
        "axis": "scaffold-novelty (Tan<0.35) + counter-disagree gate -> route to mean-predictor abstention",
        "depends_on": "nb960_f2_abstain_self_train honest cross-fit RAE",
    },
    {
        "id": "nb1022_heteroscedastic_nb963_blend",
        "axis": "if nb963 heteroscedastic NLL completes, blend sigma-weighted preds with nb2189 anchor",
        "depends_on": "nb963_heteroscedastic_nll script + summary",
    },
    {
        "id": "nb1023_graphormer_FT_aug_corpus",
        "axis": "fine-tune a small Graphormer / ChemBERTa head on (4139 train + 901 aug) -> honest cross-fit on 253",
        "depends_on": "nb950 background completion (frees Kaggle GPU slot)",
    },
    {
        "id": "nb1024_2stage_stack_clean_anchor",
        "axis": "PRE-unblind base = chemprop_aux_v1, honest meta-stack = (nb2189, nb730, nb503) -> Huber SLSQP on 253",
        "depends_on": "no upstream -- can fire immediately",
    },
]

# --- Decision summary ------------------------------------------------------
DECISION = {
    "next_action": "Hold floor at nb2112 0.4698 (nb2189 best 0.4556). Wait for nb950 chemprop_v2 background; in parallel, snapshot nb960 honest summary and write nb962/nb963 scripts.",
    "rationale": (
        "Cycle 130 has 2 of 4 method scripts on disk (nb950, nb960). "
        "Of completed work: nb951 v2 is worse than v1, nb952 confirms augmentation does NOT close OOD wall, "
        "nb953 collapses to nb503_v1. nb960 has te_ only (no honest summary). "
        "Floor preserved. Cycle 131 must branch on whichever of (nb950 chemprop_v2 full, nb960 F2 honest summary) lands first."
    ),
    "abort_condition": "if nb950 chemprop_v2 honest RAE > 0.65, retire chemprop_v2 axis and pivot cycle 131 to NN head (nb1023) + 2-stage stack (nb1024) only.",
}

summary = {
    "tag": "nb964",
    "cycle": CYCLE,
    "approx_methods_to_date": APPROX_METHODS_TO_DATE,
    "method": "cycle_130_trajectory_snapshot",
    "precondition_check": {
        "nb950_summary_present": (PROC / "nb950_summary.json").exists(),
        "nb960_summary_present": (PROC / "nb960_summary.json").exists(),
        "nb962_summary_present": (PROC / "nb962_summary.json").exists(),
        "nb963_summary_present": (PROC / "nb963_summary.json").exists(),
        "nb953_summary_present": (PROC / "nb953_summary.json").exists(),
        "nb952_summary_present": (PROC / "nb952_summary.json").exists(),
        "nb2189_summary_present": (PROC / "nb2189_summary.json").exists(),
    },
    "cycle_130_methods": CYCLE_130_METHODS,
    "n_distinct_methods_cycle_130": 5,
    "top10_honest_crossfit": TOP10_HONEST,
    "excluded_contaminated": EXCLUDED_CONTAMINATED,
    "prior_honest_floor": {"id": PRIOR_HONEST_FLOOR_ID, "rae": PRIOR_HONEST_FLOOR_RAE},
    "nb2189_best_honest": {"id": "nb2189_K20_median_bag", "rae": NB2189_BEST_K20_MEDIAN_BAG},
    "floor_transition": FLOOR_TRANSITION,
    "data_integration_status": DATA_INTEGRATION_STATUS,
    "closed_axes": CLOSED_AXES,
    "still_open_axes": STILL_OPEN_AXES,
    "cycle_131_plan": CYCLE_131_PLAN,
    "n_distinct_methods_cycle_131_plan": 5,
    "decision": DECISION,
}


def main() -> None:
    out = PROC / "nb964_summary.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out}")
    print(f"cycle={CYCLE}  approx_methods_to_date~{APPROX_METHODS_TO_DATE}")
    print(f"cycle130_methods={len(CYCLE_130_METHODS)}  cycle131_plan={len(CYCLE_131_PLAN)}")
    print(f"honest floor: {PRIOR_HONEST_FLOOR_ID} RAE={PRIOR_HONEST_FLOOR_RAE}")
    print(f"nb2189 best honest: K=20 median bag RAE={NB2189_BEST_K20_MEDIAN_BAG}")
    print(f"floor_moved={FLOOR_TRANSITION['floor_moved']}")
    print(f"next: {DECISION['next_action']}")


if __name__ == "__main__":
    main()
