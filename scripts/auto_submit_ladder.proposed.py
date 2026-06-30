# ====================================================
# CYCLE 162 LB-POLL-AWARE LADDER PROPOSAL (nb2099) 2026-06-08
# Last observed activity LB: 0.7655  (rank 262.0)
# Decision rule:
#   - last LB <= 0.55: fire AGGRESSIVE (nb1162_deploy_nb1153.csv)
#   - last LB in (0.55, 0.70]: fire LB-SAFE (nb1191_deploy_pre_pyramid.csv)
#   - last LB > 0.70 or None: fire SAFETY_FLOOR (chemprop_aux.csv)
#
# Computed next fire: chemprop_aux.csv
# Reason: last LB 0.7655 > 0.70 -> fall back to SAFETY_FLOOR chemprop_aux
#
# Apply by replacing the PRIMARY_LADDER block in auto_submit_ladder.py with:
PRIMARY_LADDER = [
    "chemprop_aux.csv",                          # CYCLE 162 nb2099 LB-aware next
    "nb1191_deploy_pre_pyramid.csv",        # LB-SAFE backstop
    "nb1150_deploy_slsqp4.csv",             # PRIMARY-2
    "nb1158_deploy_K32.csv",                # PRIMARY-3
    "nb2112_deploy_shap28.csv",             # PRIMARY-4
    "chemprop_aux.csv",                     # SAFETY_FLOOR
]
# ====================================================
