"""
nb2351_trajectory.py - Cycle 184 trajectory snapshot.

Records the state of the PXR activity track ladder after 184 cycles and
~1210 methods. Anchors the ROBUST honest floor (nb2240 0.4601) and
documents the cycle 175-183 plateau (all ~20 extensions failed).

Outputs:
    data/processed/nb2351_summary.json
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "processed" / "nb2351_summary.json"


CYCLE = 184
TOTAL_METHODS = 1210
ROBUST_FLOOR_SCRIPT = "nb2240"
ROBUST_FLOOR_RAE = 0.4601

PHASE2_DEADLINE = date(2026, 6, 24)
SNAPSHOT_DATE = date(2026, 6, 3)
DAYS_REMAINING = (PHASE2_DEADLINE - SNAPSHOT_DATE).days  # 21

PLATEAU_CYCLES = list(range(175, 184))  # 175..183 inclusive
PLATEAU_EXTENSIONS_FAILED = 20  # all attempted variants failed to beat 0.4601

LOCK_RECOMMENDATION = {
    "activity_primary_1": ROBUST_FLOOR_SCRIPT,
    "activity_primary_1_rae": ROBUST_FLOOR_RAE,
    "activity_primary_2": "nb730",
    "activity_primary_2_rae": 0.4603,
    "policy": "lock_nb2240_as_phase2_final",
    "rationale": (
        "9 consecutive cycles (175-183) of plateau-break attempts produced "
        "zero improvement; further search has negative expected value vs "
        "the 21-day deadline and submission rate limit."
    ),
    "deprecate": [
        "post-unblind train-OOF SLSQP variants",
        "delta-ML on sim<0.35 tier",
        "loss-level decompression (pinball/tail-weight/CDF-match)",
    ],
}

BACKGROUND_JOBS = [
    {
        "script": "nb2263_rfe",
        "description": "Recursive feature elimination on combined feature set",
        "status": "running",
        "expected_outcome": "feature-pruned LGBM variant; low probability of beating nb2240",
    },
    {
        "script": "nb2291_mordred",
        "description": "Mordred descriptor expansion (1800+ features)",
        "status": "running",
        "expected_outcome": "alternate feature axis; orthogonality check vs nb730 ensemble",
    },
]


def build_summary() -> dict:
    return {
        "snapshot": {
            "date": SNAPSHOT_DATE.isoformat(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "cycle": CYCLE,
            "total_methods_explored": TOTAL_METHODS,
        },
        "robust_honest_floor": {
            "script": ROBUST_FLOOR_SCRIPT,
            "honest_cross_fit_rae": ROBUST_FLOOR_RAE,
            "label": "ROBUST honest floor",
        },
        "plateau": {
            "cycles": PLATEAU_CYCLES,
            "extensions_tested": PLATEAU_EXTENSIONS_FAILED,
            "extensions_improving": 0,
            "verdict": "all_failed_to_beat_floor",
        },
        "phase2_final_lock": LOCK_RECOMMENDATION,
        "deadline": {
            "phase2_close": PHASE2_DEADLINE.isoformat(),
            "days_remaining": DAYS_REMAINING,
        },
        "background_jobs": BACKGROUND_JOBS,
    }


def main() -> None:
    summary = build_summary()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
