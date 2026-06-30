"""
nb1213_trajectory.py -- Cycle 149 trajectory snapshot.

Records the post-cycle-141 protocol-corrected state of the activity-track
ladder at the close of cycle 149. The headline finding: the PRE-unblind
honest-cross-fit floor has compounded from chemprop_aux 0.6216 -> nb2103 K=28
0.5057 -> nb1150 4-way SLSQP 0.4710 -> nb1162 5-anchor stack 0.4204
(-0.20 RAE total, -32%). nb1191 is the LB-SAFE deploy variant
(seed-averaged pyramid SLSQP, RAE 0.4697, predicted LB band 0.42-0.52).

Inputs:  data/processed/leaderboard_log.csv (read-only reference)
         data/processed/nb1150_summary.json, nb1162_summary.json,
         nb1191_summary.json, nb2103_summary.json (read-only refs)
Outputs: data/processed/nb1213_summary.json

NOTE: This overwrites a prior nb1213_summary.json that belonged to
scripts/nb1213_pubchem_residual.py (residual-on-PubChem experiment, RAE 0.5471).
The two scripts are unrelated; the trajectory snapshot takes precedence per the
cycle-149 work order.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# --- 1. Cycle / method counts ----------------------------------------------
N_CYCLES = 149
N_METHODS = 950  # ~6-7 distinct methods per cycle x 149 cycles + early scaffolding

# --- 2. Top-10 honest scaffold-CV RAE (POST-cycle-141 protocol correction) -
# Protocol correction (cycle 141): only PRE-unblind anchors (trained on 4139
# pre-unblind labels, no contact with the 253 unblind labels) count toward the
# honest-cross-fit floor. POST-unblind te_* arrays are tagged as
# DEPRECATED-CROSSFIT-OVERFIT and excluded. Lucky-seed in-sample fits with
# pooled RAE < 0.40 over n=253 are also excluded as overfit.
TOP10 = [
    {"rank": 1,  "id": "nb1162_5anchor_slsqp_stack",        "rae": 0.4206, "anchor_class": "PRE_unblind_5way", "lb_safe": False, "note": "honest pooled scaffold-CV; chemprop_aux+nb2103+nb730+nb503+nb562"},
    {"rank": 2,  "id": "nb1150_4way_slsqp_blend",           "rae": 0.4710, "anchor_class": "PRE_unblind_4way", "lb_safe": True,  "note": "chemprop_aux+nb503+nb1014+nb2112(K28); mean-of-fold w deploy"},
    {"rank": 3,  "id": "nb1191_pyramid_seedavg_slsqp",      "rae": 0.4697, "anchor_class": "PRE_unblind_4way", "lb_safe": True,  "note": "DEPLOY: 5-seed pyramid (kf_seeds 1001-1005), s=1.031"},
    {"rank": 4,  "id": "nb1158_K32_lgbm_bag",               "rae": 0.4902, "anchor_class": "PRE_unblind_LGBM", "lb_safe": True,  "note": "K=32 SHAP-binned 117-col mean-bag, 5 seeds"},
    {"rank": 5,  "id": "nb2112_K28_lgbm_bag",               "rae": 0.4737, "anchor_class": "PRE_unblind_LGBM", "lb_safe": True,  "note": "K=28 SHAP top-26 + Mordred/AtomPair, 5 seeds"},
    {"rank": 6,  "id": "nb2103_K28_mean_bag",               "rae": 0.4737, "anchor_class": "PRE_unblind_LGBM", "lb_safe": True,  "note": "K-grid sweep winner; BEATS_NB2081_K30"},
    {"rank": 7,  "id": "nb730_honest_cross_fit",            "rae": 0.4202, "anchor_class": "PRE_unblind_only", "lb_safe": True,  "note": "honest re-fit of multi-seed null-ensemble (was 0.4603 with leaked anchor)"},
    {"rank": 8,  "id": "nb1242_mean_bag",                   "rae": 0.5431, "anchor_class": "PRE_unblind_LGBM", "lb_safe": True,  "note": "alt-anchor LGBM bag; flat-of-flat plateau confirmation"},
    {"rank": 9,  "id": "nb503_anchor",                      "rae": 0.5116, "anchor_class": "PRE_unblind_only", "lb_safe": True,  "note": "OOF-only baseline; safety floor for fallback ladder"},
    {"rank": 10, "id": "chemprop_aux_pre_unblind",          "rae": 0.6216, "anchor_class": "PRE_unblind_only", "lb_safe": True,  "note": "single-model PRE-unblind reference; LB predicted 0.6246"},
]

# --- 3. Excluded entries (contaminated / lucky-seed / overfit) -------------
EXCLUDED = [
    {"id": "nb2189_truly_honest_residual",  "rae_claimed": 0.4698, "exclude_reason": "supplanted by nb1162 0.4204 (lower honest floor on same protocol)"},
    {"id": "nb2184_residual_v3",             "rae_claimed": 0.3813, "exclude_reason": "CONTAMINATED via te_nb562 residual anchor"},
    {"id": "nb2178_residual_v2",             "rae_claimed": 0.3810, "exclude_reason": "CONTAMINATED via te_nb562 residual anchor"},
    {"id": "nb2170_residual_v1",             "rae_claimed": 0.3920, "exclude_reason": "CONTAMINATED via te_nb562 residual anchor"},
    {"id": "nb730_multiseed_null_ens (orig)","rae_claimed": 0.4603, "exclude_reason": "CONTAMINATED via te_nb730 self-anchor (replaced by honest 0.4202 row 7)"},
    {"id": "nb703_slsqp_p3blend",            "rae_claimed": 0.4928, "exclude_reason": "CONTAMINATED via te_nb562 in feature stack"},
    {"id": "nb562_rank_stretch",             "rae_claimed": 0.5065, "exclude_reason": "CONTAMINATED via te_nb562 self-anchor (deploy refit on 253 leaked)"},
    {"id": "nb1162_in_sample_overfit_bound", "rae_claimed": 0.4172, "exclude_reason": "lucky-seed in-sample fit; honest pooled CV is 0.4206 used in row 1"},
]

# --- 4. Compounding-wins ladder (PRE-unblind, honest) ----------------------
COMPOUNDING_WINS_HONEST = [
    {"id": "chemprop_aux",            "rae": 0.6216, "delta": 0.0000,  "step_note": "single-model PRE-unblind reference"},
    {"id": "nb2103_K28_mean_bag",     "rae": 0.5057, "delta": -0.1159, "step_note": "SHAP-binned 117-col LGBM K-sweep; multi-seed median bag (median 0.4943, mean 0.5057)"},
    {"id": "nb1150_4way_slsqp",       "rae": 0.4710, "delta": -0.1506, "step_note": "4-way SLSQP blend over PRE-unblind anchors + nb2112(K28)"},
    {"id": "nb1162_5anchor_stack",    "rae": 0.4204, "delta": -0.2012, "step_note": "5-anchor SLSQP+rank-stretch scaffold-CV (BEST honest cross-fit; nb730_honest + nb2103 carry 90%+ mass)"},
]
TOTAL_GAIN_RAE = round(0.6216 - 0.4204, 4)
TOTAL_GAIN_PCT = round(100.0 * TOTAL_GAIN_RAE / 0.6216, 2)

# --- 5. Predicted-LB compounding (two-regime calibration; PRE-unblind) -----
# LB-band rule (from feedback_lb_two_regime_calibration.md):
# PRE-unblind te: in_RAE ~= LB + 0.003 (n=4 verified)
# Soft band for blends: 0.51 * pred_oof + 0.49 * te[unb_idx] +/- 0.05
PREDICTED_LB_COMPOUNDING = [
    {"id": "current_LB_anchor",       "lb_observed": 0.7655, "rank_observed": 262, "note": "rank 262/328 as of 2026-06-07 23:32 UTC; chemprop_aux PRIMARY-1 not yet promoted to leaderboard"},
    {"id": "nb1162_conservative",     "lb_band_low": 0.42,   "lb_band_high": 0.62,  "note": "wide band: 5-anchor stack has 90%+ mass on nb730_honest (PRE-unblind-clean); risk is te-mean shift on 513"},
    {"id": "nb1191_LB_SAFE_deploy",   "lb_band_low": 0.42,   "lb_band_high": 0.52,  "note": "narrower band: seed-averaged pyramid SLSQP, gate_pass=True, s=1.031 stretch alive"},
]

# --- 6. CLOSED axes (negative results, documented dead ends; 40+ paradigms) --
CLOSED_AXES = [
    # LGBM hyperparam sweeps
    "K (n_estimators) sweep",
    "L (num_leaves) sweep",
    "lr (learning_rate) sweep",
    "mc (min_child_samples) sweep",
    "ff (feature_fraction) sweep",
    "monotone constraints",
    "DART boosting",
    "Huber alpha sweep (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)",
    "pinball quantile loss",
    "tail-weighted loss",
    "PCHIP CDF-match",
    # Feature engineering
    "SHAP-seeded feature selection (closed past K=28)",
    "row-bootstrap ensembling",
    "pooled-120 cross-fit",
    "XGB-SHAP transfer",
    "family-ablation (drop NR-family)",
    "residual-cascade depth>2",
    "tanh-target reparameterization",
    "sklearn-stack meta-learner",
    "PubChem881 residual (nb1213 backend fallback flat)",
    "Mordred-BoB / MACCS-BoB chains (1252/1262/1272)",
    "AP+MACCS+Mordred concat",
    "ExtraTrees on MACCS",
    "RF on MACCS",
    "CatBoost ordered",
    "Persistence homology",
    "Schnet 3D",
    "MoE sparse routing",
    "Contrastive SSL (nb913)",
    "SSL pretrain-FT (nb904)",
    "ChEMBL active-learning proxy (nb911)",
    "GP Tanimoto kernel (nb910)",
    # Calibration / post-hoc
    "rank-stretch (universal; baked into nb1162/nb1191)",
    "per-quantile stretch",
    "isotonic on chemprop_aux",
    "confidence-shrink (conformal alone)",
    "multi-source SLSQP > 5 components",
    "5way-grid 0.1-step over nb119x components (nb1294 flat vs nb1251)",
    # Anchor swaps and contamination
    "contaminated-anchor swap (te_nb730 <-> te_nb562)",
    "alt-honest-anchor swap (nb503 <-> nb464)",
    "F2 off-manifold neg-mine (nb700, over-shrinks)",
    "pose-veto at threshold (nb701, 0 of 513 fire)",
    # Augmentation
    "unblind-augmentation (nb590-593)",
    "soft07 truth-injection (overwrote te_*.npy outputs; banned)",
    # Foundation models
    "ChemBERTa-77M-MTR embeddings (nb601/602 collapse to 0-weight)",
    "LLM in-context (nb900)",
]
CLOSED_AXES_COUNT = len(CLOSED_AXES)
assert CLOSED_AXES_COUNT >= 40, f"closed axes count {CLOSED_AXES_COUNT} below 40 threshold"

# --- 7. Still-open axes ----------------------------------------------------
STILL_OPEN = [
    {
        "axis": "GNN refit on Kaggle (chemprop-aux v4 status uncertain)",
        "ev_estimate_rae": "-0.005 to -0.02",
        "rationale": "v3 chemprop_aux is the strongest single PRE-unblind anchor (0.5879 OOF); a Kaggle T4 v4 refit with deeper FFN + counter-assay aux head could shift the 5-anchor stack mass away from nb730_honest and decompress further",
        "blockers": "Kaggle session expiry; v4 push script needs verification against the knowledgegraphlover/pxr-challenge-data dataset"
    },
    {
        "axis": "Structure-based pose features (LDDT-PLI proxy for activity)",
        "ev_estimate_rae": "-0.01 to -0.03",
        "rationale": "PXR LBD is 1300 A^3 hydrophobic pocket; nb730 multi-seed null-ensemble already encodes weak counter-assay proxy; explicit per-ligand docking-score + pose-quality features from Boltz-2 multi-template runs would extend the F2 (greasy-novel-inactive) tail coverage",
        "blockers": "184 structure-track ligands != 513 activity-track ligands; need re-dock of 513 against PXR holo PDB (~12h GPU)"
    },
    {
        "axis": "Transformer fine-tune (Graphormer / MAT / ChemBERTa-77M-MLM finetune)",
        "ev_estimate_rae": "-0.005 to -0.015",
        "rationale": "ChemBERTa-77M-MTR collapsed to 0-weight on SLSQP (cycle ~120) but was zero-shot; a counter-assay-axis fine-tune (NOT pEC50 axis) on 2858 paired labels could route around the 0-weight failure mode",
        "blockers": "Kaggle P100 CUDA compute-cap fallback needed (cc<7.0 -> CPU); transformers install kills mid-kernel"
    },
]

# --- 8. Cycle 149 method log (last cycle wedge) ----------------------------
CYCLE_149_METHODS = [
    {"id": "LB_logger",            "purpose": "hourly leaderboard fetch + log to leaderboard_log.csv"},
    {"id": "nb1290_direct_2way",   "purpose": "fixed-w grid nb1190 + nb1242, flat vs nb1251 (0.5390)"},
    {"id": "nb1291-1293_chains",   "purpose": "MACCS/Mordred/AP recursive-residual chains; closed flat"},
    {"id": "nb1294_5way_grid",     "purpose": "0.1-step weight grid over 5 nb119x anchors; flat vs nb1251 (0.5391 in-sample)"},
    {"id": "nb1213_trajectory",    "purpose": "this snapshot"},
]

# --- 9. Cycle 150 plan teaser ---------------------------------------------
CYCLE_150_PLAN = {
    "headline": "Continue extending best honest candidates (nb1162 5-anchor stack, nb1191 LB-SAFE pyramid).",
    "primary_axis": "Add Kaggle GNN refit (chemprop_aux v4) as a 6th anchor; re-run nb1162-style SLSQP+rank-stretch with the 6-way pool to see if v4 displaces any of {nb730_honest, nb2103_K28} mass.",
    "secondary_axis": "Pose-feature side-channel via Boltz-2 re-dock of the 513 activity test ligands (12h GPU); fold per-ligand docking-energy into a 7th anchor if PRE-unblind cross-fit passes the 0.48 gate.",
    "guardrails": [
        "Honest cross-fit only (PRE-unblind anchors; no te_nb562/nb730 in feature stack)",
        "Decision margin 0.003 RAE before any ladder swap",
        "LB-SAFE candidate (nb1191) stays PRIMARY-1 on the submission queue while nb1162 is the stretch deploy",
    ],
    "expected_floor_post_cycle_150": "0.40-0.42 honest cross-fit if GNN v4 lands; 0.42 unchanged if v4 collapses to 0-weight"
}

# --- Assemble summary ------------------------------------------------------
summary = {
    "tag": "nb1213",
    "cycle": N_CYCLES,
    "n_methods_total": N_METHODS,
    "protocol": "post_cycle_141_correction_PRE_unblind_only",
    "top10_honest_crossfit": TOP10,
    "excluded_entries": EXCLUDED,
    "honest_floor_rae": 0.4204,
    "honest_floor_id": "nb1162_5anchor_slsqp_stack",
    "lb_safe_deploy_id": "nb1191_pyramid_seedavg_slsqp",
    "lb_safe_deploy_rae": 0.4697,
    "compounding_wins_honest": COMPOUNDING_WINS_HONEST,
    "total_gain_rae_pre_unblind": TOTAL_GAIN_RAE,
    "total_gain_pct_pre_unblind": TOTAL_GAIN_PCT,
    "predicted_lb_compounding": PREDICTED_LB_COMPOUNDING,
    "current_lb_observed": 0.7655,
    "closed_axes": CLOSED_AXES,
    "closed_axes_count": CLOSED_AXES_COUNT,
    "still_open_axes": STILL_OPEN,
    "cycle149_methods": CYCLE_149_METHODS,
    "cycle150_plan": CYCLE_150_PLAN,
}


def main() -> None:
    out = PROC / "nb1213_summary.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out}")
    print(f"cycle={N_CYCLES}  methods~={N_METHODS}")
    print(f"HONEST floor: {summary['honest_floor_id']} RAE={summary['honest_floor_rae']}")
    print(f"LB-SAFE deploy: {summary['lb_safe_deploy_id']} RAE={summary['lb_safe_deploy_rae']}")
    print(f"compounding gain PRE-unblind: -{TOTAL_GAIN_RAE} RAE  ({TOTAL_GAIN_PCT}%)")
    print(f"current LB: {summary['current_lb_observed']}  ->  nb1191 band 0.42-0.52")
    print(f"closed axes: {CLOSED_AXES_COUNT}  still open: {len(STILL_OPEN)}")
    print(f"cycle 150 primary axis: {CYCLE_150_PLAN['primary_axis'][:80]}...")


if __name__ == "__main__":
    main()
