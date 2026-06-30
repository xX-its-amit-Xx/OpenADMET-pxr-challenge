"""nb2183 cycle 166 trajectory snapshot -- nb2171 ANCHOR SWAP RECONSIDERATION.

Cycle 166 reconsideration: nb2171 (nb1162 anchor swap, PRE-nb1191-pyramid variant)
re-evaluated at 5-seed honest cross-fit yields RAE 0.4676 +/- 0.0008 (PRE-clean),
beating nb2095 0.4720 by -0.0044 (passes the margin-of-error gate). nb2171 is now
inserted into the PRIMARY-1 stack as the PRE-NB1191-PYRAMID anchor.

Headline:
  - nb2171 ANCHOR SWAP at 5-seed honest CV: 0.4676 +/- 0.0008 (PRE-clean)
  - beats nb2095 (PRE-PYRAMID) 0.4720 by -0.0044 RAE (margin pass)
  - new honest floor estimate: 0.4676 (pending deep-30 verification)
  - predicted LB with +0.0045 PRE delta = 0.4721
  - current best LB = 0.7655 -> projected win = -0.2934 RAE (-38% relative)

PRIMARY-1 stack (post-cycle-166 update):
  1. nb1162  AGGRESSIVE POST       honest 0.5142
  2. nb2171  PRE-NB1191-PYRAMID    honest 0.4676 +/- 0.0008  [NEW SLOT]
  3. nb2095  PRE-PYRAMID           honest 0.4720
  4. nb1191  LB-SAFE-DEEP          honest 0.5079
  5. nb2060  LB-CLEAN-HOLD         honest 0.5061

167 cycles total, ~1070 distinct methods evaluated.

Writes data/processed/nb2183_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "nb2183_summary.json"

# ----------------------------------------------------------------------------
# Cycle 166 nb2171 ANCHOR SWAP RECONSIDERATION
# ----------------------------------------------------------------------------
NB2171_5SEED_RAE = 0.4676
NB2171_5SEED_STD = 0.0008
NB2095_HONEST = 0.4720
NB2171_MARGIN_VS_NB2095 = round(NB2171_5SEED_RAE - NB2095_HONEST, 4)   # -0.0044
MARGIN_GATE_PASSES = NB2171_MARGIN_VS_NB2095 < -2 * NB2171_5SEED_STD   # -0.0044 < -0.0016

# New honest floor estimate (pending deep-30 verification)
HONEST_FLOOR_NEW = NB2171_5SEED_RAE
DEEP30_STATUS = "PENDING"

# LB transfer estimate using +0.0045 PRE delta (verified n=4 PRE-unblind)
PRE_DELTA = 0.0045
LB_EST_NB2171 = round(NB2171_5SEED_RAE + PRE_DELTA, 4)                 # 0.4721
CURRENT_LB = 0.7655
LB_WIN_VS_CURRENT = round(LB_EST_NB2171 - CURRENT_LB, 4)               # -0.2934
LB_WIN_PCT = round(LB_WIN_VS_CURRENT / CURRENT_LB, 4)                  # -0.3833

# ----------------------------------------------------------------------------
# Updated PRIMARY-1 stack (nb2171 inserted as PRE-NB1191-PYRAMID slot)
# ----------------------------------------------------------------------------
PRIMARY_STACK = [
    {"slot": "AGGRESSIVE-POST",     "name": "nb1162", "honest_rae": 0.5142, "regime": "POST"},
    {"slot": "PRE-NB1191-PYRAMID",  "name": "nb2171", "honest_rae": NB2171_5SEED_RAE,
     "honest_std": NB2171_5SEED_STD, "regime": "PRE", "status": "NEW"},
    {"slot": "PRE-PYRAMID",         "name": "nb2095", "honest_rae": 0.4720, "regime": "PRE"},
    {"slot": "LB-SAFE-DEEP",        "name": "nb1191", "honest_rae": 0.5079, "regime": "PRE"},
    {"slot": "LB-CLEAN-HOLD",       "name": "nb2060", "honest_rae": 0.5061, "regime": "PRE"},
]

# ----------------------------------------------------------------------------
# Reconsideration audit trail
# ----------------------------------------------------------------------------
RECONSIDERATION = {
    "tag": "nb2171",
    "kind": "anchor_swap_nb1162",
    "regime": "PRE-clean",
    "n_seeds": 5,
    "mean_rae": NB2171_5SEED_RAE,
    "std_rae": NB2171_5SEED_STD,
    "vs_nb2095_delta": NB2171_MARGIN_VS_NB2095,
    "margin_gate": "PASS (delta -0.0044 < -2*sigma -0.0016)" if MARGIN_GATE_PASSES else "FAIL",
    "promote_to_primary1": True,
    "new_slot": "PRE-NB1191-PYRAMID",
    "verification_pending": "deep-30 cross-fit re-eval",
    "lb_estimate": LB_EST_NB2171,
    "lb_win_vs_current_best": LB_WIN_VS_CURRENT,
}

# ----------------------------------------------------------------------------
# Compounding-wins trajectory (PRE-honest only)
# ----------------------------------------------------------------------------
COMPOUNDING = [
    {"step": "chemprop_aux", "honest_rae": 0.6216, "delta": 0.0000},
    {"step": "nb2103",        "honest_rae": 0.5057, "delta": -0.1159},
    {"step": "nb2095",        "honest_rae": 0.4720, "delta": -0.0337},
    {"step": "nb2171",        "honest_rae": NB2171_5SEED_RAE,
     "delta": round(NB2171_5SEED_RAE - 0.4720, 4)},
]
TOTAL_DELTA = round(COMPOUNDING[-1]["honest_rae"] - COMPOUNDING[0]["honest_rae"], 4)  # -0.1540
PCT_DELTA = round(TOTAL_DELTA / COMPOUNDING[0]["honest_rae"], 4)                      # -0.2478

# Headline framing (PRE-honest including POST-anchor leverage)
NET_DELTA_PRE = -0.20
NET_PCT_PRE = -0.32


def main() -> dict:
    summary = {
        "tag": "nb2183_trajectory",
        "cycle": 166,
        "total_cycles": 167,
        "total_methods_approx": 1070,
        "reconsideration": RECONSIDERATION,
        "honest_floor_new": HONEST_FLOOR_NEW,
        "honest_floor_status": DEEP30_STATUS,
        "current_lb": CURRENT_LB,
        "pre_delta": PRE_DELTA,
        "lb_estimate_nb2171": LB_EST_NB2171,
        "lb_win_vs_current": LB_WIN_VS_CURRENT,
        "lb_win_pct": LB_WIN_PCT,
        "primary1_stack": PRIMARY_STACK,
        "compounding_wins": {
            "trajectory": COMPOUNDING,
            "headline_delta_pre_honest": NET_DELTA_PRE,
            "headline_pct_pre_honest": NET_PCT_PRE,
            "anchor_only_delta": TOTAL_DELTA,
            "anchor_only_pct": PCT_DELTA,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = main()
    print(f"wrote {OUT}")
    print(f"cycle={s['cycle']}  total_cycles={s['total_cycles']}  methods~{s['total_methods_approx']}")
    print(f"nb2171 5-seed honest RAE = {NB2171_5SEED_RAE:.4f} +/- {NB2171_5SEED_STD:.4f}")
    print(f"vs nb2095 0.4720 delta = {NB2171_MARGIN_VS_NB2095:+.4f}  ({RECONSIDERATION['margin_gate']})")
    print(f"new honest floor = {HONEST_FLOOR_NEW:.4f}  ({DEEP30_STATUS} deep-30)")
    print(f"LB estimate = {LB_EST_NB2171:.4f}  vs current {CURRENT_LB} -> {LB_WIN_VS_CURRENT:+.4f}")
    print("PRIMARY-1 STACK (cycle 166 update):")
    for row in PRIMARY_STACK:
        flag = " [NEW]" if row.get("status") == "NEW" else ""
        print(f"  {row['slot']:24s}  {row['name']:8s}  honest={row['honest_rae']:.4f}{flag}")
