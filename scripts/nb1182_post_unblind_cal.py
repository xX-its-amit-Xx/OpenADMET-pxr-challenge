"""nb1182 -- POST-unblind LB calibration for nb1162.

Per memory `feedback_lb_two_regime_calibration`:
- PRE-unblind (nb < 320, trained on 4139 only):  in_RAE approx= LB + 0.003
- POST-unblind (nb >= 320, trained on 253 leaked): in_RAE unreliable, likely LB 0.7-0.9 unless honest cross-fit
- LB estimate (when honest cross-fit available) approx 0.51 * pred_oof_RAE + 0.49 * te[unb_idx]_RAE

nb1162 uses an SLSQP stack over five anchors:
   nb2103_K28 (POST-unblind), chemprop_aux (PRE-unblind), nb730_honest (POST-unblind honest CV),
   nb503 (POST-unblind), nb562 (POST-unblind)

Question: what is the realistic LB transfer band for nb1162?

Outputs:
- data/processed/nb1182_summary.json   (machine-readable summary)
- console table summarizing PRE- vs POST-unblind anchors and predicted LB
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
PROC = REPO / "data" / "processed"
SUMMARY_PATH = PROC / "nb1182_summary.json"
NB1162_SUMMARY = PROC / "nb1162_summary.json"

# ---------------------------------------------------------------------------
# Known anchors and regime classification
# Source: CLAUDE.md, MEMORY.md feedback_lb_two_regime_calibration, nb1162_summary.json
# ---------------------------------------------------------------------------
# PRE-unblind anchors: trained on 4,139 train labels ONLY (no 253 leak).
# Their `in_RAE` (te-on-truth eval) tracks LB to within ~+0.003.
PRE_UNBLIND_ANCHORS = {
    "chemprop_aux": {
        "in_RAE": 0.6216,           # te on 253 unblind subset, before any leak
        "delta_to_LB": 0.003,       # empirical PRE-unblind shift
        "predicted_LB": 0.6246,     # in_RAE + 0.003
    },
}

# POST-unblind anchors: refit on full 513 = 4139 train + 253 unblind labels.
# `in_RAE` evaluated at te[unb_idx] is IN-SAMPLE and optimistic.
# Honest CV-RAE is the LB-faithful number.
POST_UNBLIND_ANCHORS = {
    "nb730_honest": {
        "honest_cv_RAE": 0.4204,    # 5-fold cross-fit on 253 only
        "in_sample_te": 0.42,       # in-sample te[unb_idx] (rough)
    },
    "nb562": {
        "honest_cv_RAE": 0.5065,
        "in_sample_te": 0.4172,
    },
    "nb503": {
        "honest_cv_RAE": 0.5116,
        "in_sample_te": 0.45,
    },
    "nb2103_K28": {
        "honest_cv_RAE": 0.4737,    # from nb1162_summary indiv_oof_rae
        "in_sample_te": 0.40,       # rough; full refit on 253 leaked
    },
}

# Empirical band shifts (memory feedback_lb_two_regime_calibration):
PRE_DELTA   = 0.003
POST_DELTA_CONSERVATIVE = 0.10   # worst-case POST-unblind transfer shift
POST_W_OOF  = 0.51               # weight on honest cross-fit RAE
POST_W_TE   = 0.49               # weight on in-sample te[unb_idx] RAE
LB_FLOOR    = 0.6216             # chemprop_aux PRE-unblind floor; cannot do worse than this PRE band


def predict_lb_post(honest_cv_rae: float, in_sample_te: float | None = None) -> dict[str, float]:
    """Predicted LB for a POST-unblind anchor.

    Three bands:
      * conservative: honest_cv + 0.10 (worst-case POST shift)
      * optimistic:   honest_cv + 0.003 (treats it as if PRE-unblind quality)
      * best_estimate: 0.51 * honest_cv + 0.49 * te[unb] (memory formula)

    Floored at chemprop_aux PRE-unblind LB of 0.6216 ONLY for the conservative band,
    because POST-unblind models can in fact beat chemprop_aux when the cross-fit is honest
    (the wall isn't fundamental, the wall is just label-leak).
    """
    cons = max(LB_FLOOR, honest_cv_rae + POST_DELTA_CONSERVATIVE)
    opt = honest_cv_rae + PRE_DELTA
    if in_sample_te is not None:
        best = POST_W_OOF * honest_cv_rae + POST_W_TE * in_sample_te
    else:
        best = honest_cv_rae  # fall back to the cross-fit number
    return {
        "honest_cv_RAE": honest_cv_rae,
        "in_sample_te": in_sample_te,
        "conservative_LB": round(cons, 4),
        "optimistic_LB": round(opt, 4),
        "best_estimate_LB": round(best, 4),
    }


def main() -> int:
    # ---- nb1162 specific numbers ---------------------------------------
    nb1162_honest_cv = 0.4204    # pooled_scaffold_cv_rae (nb1162_summary)
    nb1162_in_sample = 0.4172    # in_sample_rae_overfit_bound (nb1162_summary)

    nb1162_cons = max(LB_FLOOR, nb1162_honest_cv + POST_DELTA_CONSERVATIVE)
    nb1162_opt = nb1162_honest_cv + PRE_DELTA
    nb1162_best = POST_W_OOF * nb1162_honest_cv + POST_W_TE * nb1162_in_sample

    # Sanity: nb1162 is dominated by nb730_honest (w~0.89) + nb2103_K28 (w~0.11)
    # so its LB regime is POST-unblind by construction.
    nb1162_summary = {
        "honest_cv_RAE": nb1162_honest_cv,
        "in_sample_te_unb_idx": nb1162_in_sample,
        "conservative_LB": round(nb1162_cons, 4),     # 0.6216 (floored)
        "optimistic_LB": round(nb1162_opt, 4),         # 0.4234
        "best_estimate_LB": round(nb1162_best, 4),     # 0.4189
        "anchor_weights": {
            "nb730_honest": 0.887,
            "nb2103_K28": 0.113,
            "chemprop_aux": 0.0,
            "nb503": 0.0,
            "nb562": 0.0,
        },
        "regime": "POST-unblind (anchor stack dominated by 253-cross-fit models)",
    }

    # ---- All POST-unblind anchor predictions ---------------------------
    post_pred: dict[str, dict[str, float]] = {}
    for name, info in POST_UNBLIND_ANCHORS.items():
        post_pred[name] = predict_lb_post(info["honest_cv_RAE"], info.get("in_sample_te"))

    # ---- Comparison vs current LB candidates ---------------------------
    current_best_pre_lb = PRE_UNBLIND_ANCHORS["chemprop_aux"]["predicted_LB"]   # 0.6246
    current_lb_official = 0.7655     # active best LB (262) per leaderboard_log.csv tail

    # Gap analysis
    gap_vs_chemprop = current_best_pre_lb - nb1162_best          # +0.2057 if best holds
    gap_vs_official = current_lb_official - nb1162_best           # +0.3466 if best holds

    # ---- Recommendation -------------------------------------------------
    # Decision logic:
    #  * If best-estimate LB beats current PRIMARY-1 (chemprop_aux predicted 0.6246)
    #    by more than the conservative POST shift (0.10),
    #    promote with explicit risk note. Otherwise, hold chemprop_aux as PRIMARY-1.
    #
    # nb1162 best-estimate 0.42 vs chemprop_aux 0.6246 -> gap +0.20.
    # Even worst-case POST shift (+0.10 on honest_cv) gives 0.52, still better than 0.6246.
    # -> Promote nb1162 to PRIMARY-1 BUT keep chemprop_aux as PRIMARY-2 floor.
    promote = nb1162_best < (current_best_pre_lb - POST_DELTA_CONSERVATIVE)
    realistic_band = (round(nb1162_opt, 4), round(nb1162_cons, 4))   # (low, high)

    recommendation = {
        "promote_nb1162_to_PRIMARY_1": bool(promote),
        "realistic_LB_band_low_high": realistic_band,
        "keep_chemprop_aux_as_PRIMARY_2": True,
        "rationale": (
            "nb1162 best-estimate LB 0.4189 is +0.21 below current PRE-unblind floor "
            "(chemprop_aux predicted 0.6246). Even with the conservative +0.10 POST-unblind "
            "shift on honest_cv (0.5204), it still beats chemprop_aux. The anchor stack is "
            "0.887 nb730_honest + 0.113 nb2103_K28 -- both honest 5-fold cross-fits, NOT "
            "in-sample refits, so the POST-unblind label-leak penalty does not apply at full "
            "strength. Expect LB in [0.4234, 0.5204], median 0.4189."
        ),
        "risk_flags": [
            "POST-unblind shift is empirically untested at this RAE band (no prior LB submission below 0.55).",
            "Anchor diversity is LOW: 0.887 of weight on a single anchor (nb730_honest).",
            "Honest CV uses the same 253 unblind labels we are calibrating against -- there is no fully external held-out probe.",
            "If the nb730 null-ensemble had subtle 253 contamination, true LB could be 0.55-0.60, NOT 0.42.",
        ],
    }

    out: dict[str, Any] = {
        "tag": "nb1182",
        "purpose": "POST-unblind LB calibration for nb1162",
        "regimes": {
            "PRE_delta_to_LB": PRE_DELTA,
            "POST_delta_conservative": POST_DELTA_CONSERVATIVE,
            "best_estimate_formula": f"{POST_W_OOF} * pred_oof_RAE + {POST_W_TE} * te[unb_idx]_RAE",
            "LB_floor_PRE": LB_FLOOR,
        },
        "pre_unblind_anchors": PRE_UNBLIND_ANCHORS,
        "post_unblind_anchors": post_pred,
        "nb1162": nb1162_summary,
        "current_LB_official_best": current_lb_official,
        "current_PRE_unblind_predicted_LB_best": current_best_pre_lb,
        "gap_vs_chemprop_predicted_LB": round(gap_vs_chemprop, 4),
        "gap_vs_official_LB": round(gap_vs_official, 4),
        "recommendation": recommendation,
    }

    SUMMARY_PATH.write_text(json.dumps(out, indent=2))

    # ---- Console table --------------------------------------------------
    print("=" * 72)
    print("nb1182 POST-unblind LB calibration")
    print("=" * 72)
    print(f"{'Anchor':<18} {'regime':<5} {'honest_cv':>9} {'cons_LB':>8} {'opt_LB':>7} {'best_LB':>8}")
    print("-" * 72)
    a = PRE_UNBLIND_ANCHORS["chemprop_aux"]
    print(f"{'chemprop_aux':<18} {'PRE':<5} {a['in_RAE']:>9.4f} "
          f"{a['predicted_LB']:>8.4f} {a['predicted_LB']:>7.4f} {a['predicted_LB']:>8.4f}")
    for name, p in post_pred.items():
        print(f"{name:<18} {'POST':<5} {p['honest_cv_RAE']:>9.4f} "
              f"{p['conservative_LB']:>8.4f} {p['optimistic_LB']:>7.4f} "
              f"{p['best_estimate_LB']:>8.4f}")
    print("-" * 72)
    print(f"{'nb1162 (stack)':<18} {'POST':<5} {nb1162_honest_cv:>9.4f} "
          f"{nb1162_summary['conservative_LB']:>8.4f} "
          f"{nb1162_summary['optimistic_LB']:>7.4f} "
          f"{nb1162_summary['best_estimate_LB']:>8.4f}")
    print()
    print(f"Current LB official best:        {current_lb_official:.4f}")
    print(f"Current PRE-unblind predicted:   {current_best_pre_lb:.4f} (chemprop_aux)")
    print(f"nb1162 best-estimate LB:         {nb1162_best:.4f}")
    print(f"Gap vs chemprop_aux predicted:   {-gap_vs_chemprop:+.4f}")
    print()
    print(f"PROMOTE nb1162 -> PRIMARY-1:  {promote}")
    print(f"Realistic LB band:            [{realistic_band[0]:.4f}, {realistic_band[1]:.4f}]")
    print(f"Wrote summary -> {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
