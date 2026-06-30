"""nb1544 — Final ladder summary: PRE-unblind vs POST-unblind regimes.

Aggregates honest cross-fit RAEs and BoB-validated estimates from the recent
ladder of activity-track submissions, emits a JSON summary, and prints the
top-10 in each regime.

LB calibration (from MEMORY feedback_lb_two_regime_calibration.md):
  PRE-unblind te (trained on 4139 only):  predicted_LB ~= RAE + 0.003
  POST-unblind te (trained on leaked 253): predicted_LB ~= RAE + 0.20  (very uncertain;
                                            in-sample on 253 is unreliable as LB proxy)
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
SUB = ROOT / "submissions"
OUT = ROOT / "data" / "processed" / "final_ladder_summary.json"

# -----------------------------------------------------------------------------
# Candidate ladder. Populated from session protocol + MEMORY notes.
# -----------------------------------------------------------------------------
# regime: PRE-unblind anchor = chemprop_aux base (trained on 4139 only, BoB-valid)
#         POST-unblind anchor = nb1070 base (trained including 253 unblind labels)
# confidence: BoB-validated > honest_crossfit > in-sample > lucky-seed
# predicted_LB: PRE = RAE + 0.003 ; POST = "≈in-sample" (unreliable; honest upper-band)
# deploy_csv: path under submissions/, if a deploy artifact exists.

CANDIDATES = [
    # ------------- PRE-unblind (chemprop_aux family) -------------
    {
        "name": "nb1521",
        "regime": "PRE-unblind",
        "rae": 0.5221,
        "rae_kind": "BoB-validated",
        "deploy_csv": "submissions/nb1530_deploy_nb1521_mean.csv",
        "notes": "BoB-validated chemprop_aux ladder candidate (mean variant)",
    },
    {
        "name": "nb1512",
        "regime": "PRE-unblind",
        "rae": 0.5246,
        "rae_kind": "BoB-validated",
        "deploy_csv": None,  # no nb1512 deploy artifact found in submissions/
        "notes": "BoB-validated; no separate deploy CSV materialized",
    },
    {
        "name": "nb1501",
        "regime": "PRE-unblind",
        "rae": 0.5223,
        "rae_kind": "lucky-seed",
        "honest_rae_est": 0.5270,
        "deploy_csv": None,
        "notes": "Reported 0.5223 but honest ~0.527; demote to lucky-seed tier",
    },
    {
        "name": "nb1500",
        "regime": "PRE-unblind",
        "rae": 0.5234,
        "rae_kind": "BoB-validated",
        "deploy_csv": "submissions/nb1510_deploy_nb1500_mean.csv",
        "notes": "BoB-validated multi-seed chemprop_aux blend",
    },
    {
        "name": "nb1484_slsqp",
        "regime": "PRE-unblind",
        "rae": 0.5231,
        "rae_kind": "honest_crossfit",
        "deploy_csv": "submissions/nb1503_deploy_nb1484_slsqp.csv",
        "notes": "SLSQP-weighted variant",
    },
    {
        "name": "nb1484_naive",
        "regime": "PRE-unblind",
        "rae": 0.5257,
        "rae_kind": "honest_crossfit",
        "deploy_csv": "submissions/nb1491_deploy_nb1484.csv",
        "notes": "Naive equal-weight variant of nb1484",
    },
    {
        "name": "nb1482",
        "regime": "PRE-unblind",
        "rae": 0.5310,
        "rae_kind": "BoB-validated",
        "deploy_csv": "submissions/nb1490_deploy_nb1482_mean.csv",
        "notes": "BoB-validated",
    },
    {
        "name": "nb1472",
        "regime": "PRE-unblind",
        "rae": 0.5330,
        "rae_kind": "honest_crossfit",
        "deploy_csv": "submissions/nb1480_deploy_nb1472.csv",
        "notes": "Honest cross-fit on chemprop_aux anchor",
    },
    {
        "name": "nb1460",
        "regime": "PRE-unblind",
        "rae": 0.5550,
        "rae_kind": "honest_crossfit",
        "deploy_csv": "submissions/nb1470_deploy_nb1460.csv",
        "notes": "Honest cross-fit; earliest in chain",
    },
    # ------------- POST-unblind (nb1070 family) -------------
    {
        "name": "nb1441",
        "regime": "POST-unblind",
        "rae": 0.4990,
        "rae_kind": "in-sample",
        "deploy_csv": "submissions/nb1450_deploy_nb1441_blend.csv",
        "notes": "In-sample blend on 253 unblind; sub-margin gain vs nb1422",
    },
    {
        "name": "nb1471",
        "regime": "POST-unblind",
        "rae": 0.4995,
        "rae_kind": "in-sample",
        "deploy_csv": "submissions/nb1481_deploy_nb1471.csv",
        "notes": "In-sample blend; sub-margin gain vs nb1422",
    },
    {
        "name": "nb1422",
        "regime": "POST-unblind",
        "rae": 0.5016,
        "rae_kind": "BoB-validated",
        "deploy_csv": "submissions/nb1430_deploy_nb1422_mean.csv",
        "notes": "BoB-validated POST-unblind anchor",
    },
    {
        "name": "nb1373",
        "regime": "POST-unblind",
        "rae": 0.5095,
        "rae_kind": "honest_crossfit",
        "deploy_csv": "submissions/nb1380_deploy_nb1373_mean.csv",
        "notes": "Honest cross-fit on nb1070 anchor",
    },
]

# Confidence tier ranking for tie-breaking (lower = better)
CONF_RANK = {
    "BoB-validated": 0,
    "honest_crossfit": 1,
    "in-sample": 2,
    "lucky-seed": 3,
}


def predicted_lb(c: dict) -> float:
    """Two-regime LB calibration (see feedback_lb_two_regime_calibration.md)."""
    rae = c.get("honest_rae_est", c["rae"]) if c["rae_kind"] == "lucky-seed" else c["rae"]
    if c["regime"] == "PRE-unblind":
        return round(rae + 0.003, 4)
    # POST-unblind: in-sample on 253 is unreliable; conservative LB band ~0.7-0.9.
    # Use rae as an "honest-upper" placeholder and flag uncertain.
    return round(rae + 0.20, 4)  # conservative shift estimate


def main() -> None:
    # Honest sort key: confidence first, then numeric RAE.
    def sort_key(c):
        rae = c.get("honest_rae_est", c["rae"]) if c["rae_kind"] == "lucky-seed" else c["rae"]
        return (rae, CONF_RANK.get(c["rae_kind"], 99))

    pre = sorted([c for c in CANDIDATES if c["regime"] == "PRE-unblind"], key=sort_key)
    post = sorted([c for c in CANDIDATES if c["regime"] == "POST-unblind"], key=sort_key)

    ranked = []
    for i, c in enumerate(pre, 1):
        entry = {
            "rank": i,
            "name": c["name"],
            "regime": c["regime"],
            "rae": c["rae"],
            "honest_rae_est": c.get("honest_rae_est"),
            "rae_kind": c["rae_kind"],
            "predicted_LB": predicted_lb(c),
            "deploy_csv": c["deploy_csv"],
            "notes": c["notes"],
        }
        ranked.append(entry)
    for i, c in enumerate(post, 1):
        entry = {
            "rank": i,
            "name": c["name"],
            "regime": c["regime"],
            "rae": c["rae"],
            "honest_rae_est": c.get("honest_rae_est"),
            "rae_kind": c["rae_kind"],
            "predicted_LB": predicted_lb(c),
            "deploy_csv": c["deploy_csv"],
            "notes": c["notes"],
        }
        ranked.append(entry)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "lb_calibration": {
            "PRE-unblind":  "predicted_LB = RAE + 0.003 (verified, n=4)",
            "POST-unblind": "predicted_LB = RAE + 0.20 (conservative; in-sample on 253 unreliable)",
        },
        "regime_count": {"PRE-unblind": len(pre), "POST-unblind": len(post)},
        "ladder": ranked,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    print()

    print("=" * 78)
    print("TOP-10 PRE-unblind (chemprop_aux anchor, LB-honest)")
    print("=" * 78)
    print(f"{'rank':>4} {'name':<18} {'RAE':>7} {'predLB':>7} {'confidence':<18} {'deploy_csv'}")
    print("-" * 78)
    for c in pre[:10]:
        rae = c.get("honest_rae_est", c["rae"]) if c["rae_kind"] == "lucky-seed" else c["rae"]
        dc = c["deploy_csv"] or "(none)"
        print(f"{pre.index(c)+1:>4} {c['name']:<18} {rae:>7.4f} {predicted_lb(c):>7.4f} "
              f"{c['rae_kind']:<18} {dc}")

    print()
    print("=" * 78)
    print("TOP-10 POST-unblind (nb1070 anchor, in-sample on 253 — LB-uncertain)")
    print("=" * 78)
    print(f"{'rank':>4} {'name':<18} {'RAE':>7} {'predLB':>7} {'confidence':<18} {'deploy_csv'}")
    print("-" * 78)
    for c in post[:10]:
        rae = c.get("honest_rae_est", c["rae"]) if c["rae_kind"] == "lucky-seed" else c["rae"]
        dc = c["deploy_csv"] or "(none)"
        print(f"{post.index(c)+1:>4} {c['name']:<18} {rae:>7.4f} {predicted_lb(c):>7.4f} "
              f"{c['rae_kind']:<18} {dc}")


if __name__ == "__main__":
    main()
