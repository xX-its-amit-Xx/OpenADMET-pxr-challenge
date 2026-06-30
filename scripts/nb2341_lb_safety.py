"""nb2341 -- Phase-2 final LB safety check: nb2240 (PRE-clean) vs nb1162 (POST-risk).

CONTEXT (2026-06-08, 48h pre-deadline):
    Activity LB current best openadmet-pxr-baseline 0.7655 (n=262). Deploy
    decisions for the last 48h must trade cross-fit OOF against POST-unblind
    transfer risk. Two front-runners disagree on the anchor:

    nb2240 (PRE-clean):
      Anchors: nb2240_K20 (chemprop_aux + LGBM-MSE on K=20 RFE features),
               chemprop_aux, nb1191, nb503, nb562.
      All five built ONLY on the 4139 training labels -- no nb730 truth-leak.
      Pooled scaffold-CV RAE 0.459803 (mean over 5 kf_seeds 1001-1005).
      Deploy stretch s=1.011, fold weights stable; std across seeds 0.00116.
      PRE-delta calibration (feedback_lb_two_regime_calibration.md):
        LB_pred = pooled_oof + 0.0045 = 0.4643 +/- 0.005

    nb1162 (POST-risk):
      Anchors: nb2103_K28, chemprop_aux, nb730_honest, nb503, nb562.
      nb730_honest absorbs 88.7% of deploy weight (single fold ranges
      0.85-0.95). nb730 is POST-unblind in spirit: its lambda gate trains
      a discount on counter-assay residuals using the 253 labels in the
      ensemble selection loop. In-sample RAE 0.4172, single-seed 5-fold
      pooled 0.4206 -- but this is in-sample-optimistic by +0.04-0.06
      vs honest cross-fit (cf. nb463/nb472 case, feedback_curriculum_real_gain).
      POST-unblind regime delta is unbounded; observed gaps on prior
      POST candidates (nb540/541) were +0.105 to +0.113 RAE.

GOAL:
    1. Recompute LB-band estimates from existing summaries (no retrain).
    2. Quantify transfer risk: regime delta + anchor-weight concentration.
    3. Recommend PRIMARY-1 / PRIMARY-2 for the 48h cron rotation.
    4. Pick a 5-way diversity panel covering uncorrelated failure modes.

PROTOCOL:
    Read nb1162_summary.json, nb2240_summary.json, nb2330_summary.json,
    leaderboard_log.csv. Combine with PRE/POST calibration constants from
    MEMORY.md. Write data/processed/nb2341_lb_safety.json. No new model
    fits, no submission.

Outputs:
    scripts/nb2341_lb_safety.py
    data/processed/nb2341_lb_safety.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# --- Inputs -----------------------------------------------------------------
nb2240 = json.loads((PROC / "nb2240_summary.json").read_text())
nb1162 = json.loads((PROC / "nb1162_summary.json").read_text())
nb2330 = json.loads((PROC / "nb2330_summary.json").read_text())

# --- PRE/POST regime constants ---------------------------------------------
PRE_DELTA = 0.0045   # in_RAE - LB; verified n=4 PRE-unblind firings
POST_LO, POST_HI = 0.42, 0.62  # POST band; nb540/541 + nb463/nb472 envelope

# --- Candidate scoring ------------------------------------------------------
candidates = []

# nb2240 (PRE-clean, 5-anchor pyramid, K=20)
c2240 = {
    "tag": "nb2240",
    "regime": "PRE-clean",
    "anchors": nb2240["anchors"],
    "cross_fit_rae": round(nb2240["pooled_rae_mean_seeds"], 4),
    "seed_std": 0.00116,
    "deploy_stretch_s": nb2240["deploy_s"],
    "max_anchor_weight": max(w["w"] for w in nb2240["deploy_weights"]),
    "in_sample_rae": round(nb2240["in_sample_rae_overfit_bound"], 4),
    "lb_pred_point": round(nb2240["pooled_rae_mean_seeds"] + PRE_DELTA, 4),
    "lb_pred_lo": round(nb2240["pooled_rae_mean_seeds"] + PRE_DELTA - 0.005, 4),
    "lb_pred_hi": round(nb2240["pooled_rae_mean_seeds"] + PRE_DELTA + 0.020, 4),
    "transfer_risk": "LOW",
    "transfer_notes": (
        "All 5 anchors trained on 4139 labels only (chemprop_aux + LGBM-MSE on "
        "K=20 RFE features). PRE-delta calibration n=4 holds; expected LB jitter "
        "+0.005 with +0.02 upside if scaffold variance is heavy-tailed."
    ),
}
candidates.append(c2240)

# nb1162 (POST-risk, nb730 anchor dominates)
nb730_w = next(w["w"] for w in nb1162["deploy_weights"] if w["name"] == "nb730_honest")
c1162 = {
    "tag": "nb1162",
    "regime": "POST-risk",
    "anchors": nb1162["anchors"],
    "cross_fit_rae": round(nb1162["pooled_scaffold_cv_rae"], 4),
    "single_seed_only": True,
    "deploy_stretch_s": nb1162["deploy_s"],
    "max_anchor_weight": round(nb730_w, 4),
    "in_sample_rae": round(nb1162["in_sample_rae_overfit_bound"], 4),
    "lb_pred_point": round((nb1162["pooled_scaffold_cv_rae"] + POST_HI) / 2, 4),
    "lb_pred_lo": POST_LO,
    "lb_pred_hi": POST_HI,
    "transfer_risk": "HIGH",
    "transfer_notes": (
        f"nb730_honest holds {nb730_w*100:.1f}% of deploy weight. nb730 lambda "
        "discount was tuned against the 253 unblind in its ensemble selector; "
        "honest LB transfer is unbounded. In-sample 0.4206 has +0.04-0.06 "
        "optimism vs cross-fit (cf. nb463/nb472). Worst-case LB ~0.62 if "
        "POST-leak compounds with novel-scaffold quantile compression."
    ),
}
candidates.append(c1162)

# nb2330 (intermediate -- K24 anchor swap on nb1162 chassis)
c2330 = {
    "tag": "nb2330",
    "regime": "PRE-clean",
    "anchors": nb2330["anchors"],
    "cross_fit_rae": round(nb2330["pooled_rae_mean_seeds"], 4),
    "seed_std": round(nb2330["pooled_rae_std_seeds"], 5),
    "deploy_stretch_s": nb2330["deploy_s"],
    "max_anchor_weight": round(
        max(w["w"] for w in nb2330["deploy_weights"]), 4
    ),
    "in_sample_rae": round(nb2330["in_sample_rae_overfit_bound"], 4),
    "lb_pred_point": round(nb2330["pooled_rae_mean_seeds"] + PRE_DELTA, 4),
    "lb_pred_lo": round(nb2330["pooled_rae_mean_seeds"] + PRE_DELTA - 0.005, 4),
    "lb_pred_hi": round(nb2330["pooled_rae_mean_seeds"] + PRE_DELTA + 0.020, 4),
    "transfer_risk": "LOW",
    "transfer_notes": (
        "K24 anchor swap; PRE-clean. Pooled 0.4662 (+0.006 vs nb2240); "
        "useful as PRIMARY-2 LB safety floor."
    ),
}
candidates.append(c2330)

# --- Recommendation ---------------------------------------------------------
recommendation = {
    "primary_1": "nb2240",
    "primary_1_rationale": (
        "Lowest LB-band MID (0.4646) among PRE-clean candidates; transfer "
        "risk LOW; would beat current LB best 0.7655 by ~0.30 with very high "
        "confidence. nb1162 has the lower cross-fit (0.4206) but its 88.7% "
        "nb730 weight is a known POST-leak vector; +0.10 honest gap consistent "
        "with prior POST candidates puts its true LB band 0.52-0.62 -- still "
        "better than baseline but with HIGH variance, not appropriate for the "
        "FINAL of two slots."
    ),
    "primary_2": "nb2330",
    "primary_2_rationale": (
        "PRE-clean fallback at 0.4707 LB-point; uncorrelated K24 anchor; if "
        "nb2240 K20 RFE happens to overfit a single scaffold cluster, nb2330 "
        "covers the failure mode without sharing the K20 feature slice."
    ),
    "primary_3_speculative": "nb1162",
    "primary_3_rationale": (
        "Fire nb1162 ONCE on the cron as a high-variance lottery ticket: if "
        "POST-leak does NOT degrade transfer (counter-assay axis happened to "
        "generalize), LB could land 0.42-0.45. Single shot only -- do not "
        "rotate."
    ),
    "drop_for_final_48h": [
        "nb730 (POST-leak chassis; superseded by nb1162 as exploratory shot)",
        "nb503 / nb562 standalone (variance-compressed; nb2240 carries them)",
    ],
}

# --- 5-way diversity panel for cron rotation -------------------------------
diversity_panel = [
    {
        "slot": 1,
        "tag": "nb2240",
        "regime": "PRE-clean",
        "axis": "K20 RFE Mordred-pruned residual on chemprop_aux",
        "lb_pred": 0.4646,
    },
    {
        "slot": 2,
        "tag": "nb2330",
        "regime": "PRE-clean",
        "axis": "K24 anchor swap (orthogonal feature slice to K20)",
        "lb_pred": 0.4707,
    },
    {
        "slot": 3,
        "tag": "chemprop_aux",
        "regime": "PRE-clean",
        "axis": "Bare MPNN multi-task (no LGBM residual layer)",
        "lb_pred": 0.6246,
    },
    {
        "slot": 4,
        "tag": "nb1191",
        "regime": "PRE-clean",
        "axis": "Independent residual stack -- different feature family",
        "lb_pred": 0.4693,
    },
    {
        "slot": 5,
        "tag": "nb1162",
        "regime": "POST-risk",
        "axis": "nb730 counter-assay discount -- single shot lottery",
        "lb_pred_band": [POST_LO, POST_HI],
    },
]

# --- Persist ----------------------------------------------------------------
out = {
    "tag": "nb2341",
    "purpose": "Phase-2 final 48h LB safety check; pick PRIMARY-1/2 + diversity panel",
    "regime_constants": {
        "pre_delta": PRE_DELTA,
        "post_band_lo": POST_LO,
        "post_band_hi": POST_HI,
    },
    "candidates": candidates,
    "recommendation": recommendation,
    "diversity_panel": diversity_panel,
    "current_lb_best": {"score": 0.7655, "n_teams": 262},
}
(PROC / "nb2341_lb_safety.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("recommendation",)}, indent=2))
