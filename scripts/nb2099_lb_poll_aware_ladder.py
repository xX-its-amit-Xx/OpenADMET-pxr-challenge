"""
nb2099 — LB-poll-aware ladder rebuilder
========================================
Rebuilds auto_submit_ladder.py decision rule to alternate AGGRESSIVE vs LB-SAFE
based on the MOST RECENT LB score observed for that track:
  - if last LB <= 0.55 (we're winning): try AGGRESSIVE next
  - if last LB >  0.55 (we're losing or unknown): force LB-SAFE next

Produces a patch (does NOT overwrite live ladder) at:
  scripts/auto_submit_ladder.proposed.py

User reviews and copies to live if approved.

Output: data/processed/nb2099_summary.json
        scripts/auto_submit_ladder.proposed.py (patch suggestion)
"""
import json, pandas as pd
from pathlib import Path

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"

lb_log = pd.read_csv(PROC / "leaderboard_log.csv")
# Filter to activity-track rows with a real lb_score
act = lb_log[lb_log["track"] == "activity"].dropna(subset=["lb_score"]).copy()
act["submitted_at"] = pd.to_datetime(act["submitted_at"], errors="coerce")
act = act.sort_values("submitted_at")

if act.empty:
    last_lb = None
    last_rank = None
else:
    last_lb = float(act.iloc[-1]["lb_score"])
    last_rank = float(act.iloc[-1]["user_rank"])

# Decision rule
AGGRESSIVE = "nb1162_deploy_nb1153.csv"
LB_SAFE = "nb1191_deploy_pre_pyramid.csv"
SAFETY_FLOOR = "chemprop_aux.csv"

if last_lb is None:
    next_fire = LB_SAFE
    reason = "no prior LB observed -> default LB-SAFE"
elif last_lb <= 0.55:
    next_fire = AGGRESSIVE
    reason = f"last LB {last_lb:.4f} <= 0.55 -> we're winning, try AGGRESSIVE"
elif last_lb <= 0.70:
    next_fire = LB_SAFE
    reason = f"last LB {last_lb:.4f} in (0.55, 0.70] -> hedge, fire LB-SAFE"
else:
    next_fire = SAFETY_FLOOR
    reason = f"last LB {last_lb:.4f} > 0.70 -> fall back to SAFETY_FLOOR chemprop_aux"

print(f"Last activity LB: {last_lb}  rank: {last_rank}")
print(f"Next fire decision: {next_fire}")
print(f"Reason: {reason}")

# Emit proposed patch
patch_src = f'''# ====================================================
# CYCLE 162 LB-POLL-AWARE LADDER PROPOSAL (nb2099) 2026-06-08
# Last observed activity LB: {last_lb}  (rank {last_rank})
# Decision rule:
#   - last LB <= 0.55: fire AGGRESSIVE ({AGGRESSIVE})
#   - last LB in (0.55, 0.70]: fire LB-SAFE ({LB_SAFE})
#   - last LB > 0.70 or None: fire SAFETY_FLOOR ({SAFETY_FLOOR})
#
# Computed next fire: {next_fire}
# Reason: {reason}
#
# Apply by replacing the PRIMARY_LADDER block in auto_submit_ladder.py with:
PRIMARY_LADDER = [
    "{next_fire}",                          # CYCLE 162 nb2099 LB-aware next
    "nb1191_deploy_pre_pyramid.csv",        # LB-SAFE backstop
    "nb1150_deploy_slsqp4.csv",             # PRIMARY-2
    "nb1158_deploy_K32.csv",                # PRIMARY-3
    "nb2112_deploy_shap28.csv",             # PRIMARY-4
    "chemprop_aux.csv",                     # SAFETY_FLOOR
]
# ====================================================
'''
patch_path = ROOT / "scripts" / "auto_submit_ladder.proposed.py"
patch_path.write_text(patch_src)

summary = {
    "method": "nb2099_lb_poll_aware_ladder",
    "last_activity_lb": last_lb,
    "last_activity_rank": last_rank,
    "next_fire": next_fire,
    "decision_reason": reason,
    "proposed_patch": str(patch_path),
    "rule": {
        "lb_<=_0.55": "AGGRESSIVE",
        "lb_in_(0.55,0.70]": "LB_SAFE",
        "lb_>_0.70_or_none": "SAFETY_FLOOR"
    },
    "live_ladder_unchanged": True,
}
with open(PROC / "nb2099_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
