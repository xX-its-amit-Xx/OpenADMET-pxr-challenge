"""Append CYCLE-5-EXTENSION in_RAE rows to PRE-unblind ladder CSVs and emit
cycle5_ext_summary.json artifact.

Cycle 5-extension in_RAEs (2026-06-03):
- nb985 extended 5-way      : 0.6188
- nb986 asym 2-way          : 0.6105
- nb987 exp-avg ensemble    : 0.6144
- nb988 nb982 stretch s1.25 : 0.5863  -> NEW TOP HONEST BLEND
- nb989 chemprop_aux affine : 0.6116
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
ART_DIR = Path("C:/pxr_artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

# (file_basename, nb_num, in_RAE)
EXT_ROWS = [
    ("te_nb985_extended_blend",      "985.0", 0.6188),
    ("te_nb986_asymmetric_blend",    "986.0", 0.6105),
    ("te_nb987_exp_avg_ensemble",    "987.0", 0.6144),
    ("te_nb988_nb982_stretch",       "988.0", 0.5863),
    ("te_nb989_chemprop_aux_affine", "989.0", 0.6116),
]


def update_ladder(csv_path: Path) -> dict:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]

    parsed = []
    for r in body:
        if not r:
            continue
        in_rae = float(r[2])
        parsed.append((r[0], r[1], in_rae, r[3]))

    for fname, nb, in_rae in EXT_ROWS:
        parsed.append((fname, nb, in_rae, "513"))

    parsed.sort(key=lambda x: x[2])

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for fname, nb, in_rae, n in parsed:
            w.writerow([fname, nb, in_rae, n])

    locs = {fname: None for fname, _, _ in EXT_ROWS}
    for idx, (fname, _, in_rae, _) in enumerate(parsed, start=2):
        if fname in locs:
            locs[fname] = {"line": idx, "in_RAE": in_rae}
    return {"file": str(csv_path), "n_rows": len(parsed) + 1, "ext_positions": locs}


def main():
    primary = ROOT / "data/processed/pre_unblind_lb_candidates.csv"
    full    = ROOT / "data/processed/pre_unblind_lb_candidates_all.csv"

    res_primary = update_ladder(primary)
    res_full    = update_ladder(full)

    best_method = "nb988_nb982_stretch"
    best_in_rae = 0.5863

    summary = {
        "ts": "2026-06-03",
        "cycle": "CYCLE-5-EXTENSION",
        "branch": "main",
        "description": (
            "5 cycle-5-extension method-axes: extended-5-way blend, asymmetric "
            "2-way blend, exp-avg ensemble, nb982 stretch (s=1.25), and "
            "chemprop_aux affine (a=1.10, b=-0.50). Sorted by in_RAE asc."
        ),
        "n_added_to_ladder": 5,
        "methods": [
            {
                "rank": 1,
                "script": "nb988_nb982_stretch",
                "csv": "submissions/nb988_nb982_stretch_s1.25.csv",
                "method": "Post-hoc rank-stretch (s=1.25) on nb982 best-blend anchor",
                "in_RAE": 0.5863,
                "predicted_LB": 0.5893,
                "status": "ADDED",
                "regime": "PRE-unblind",
                "ladder_promotion": "CYCLE-5-EXT-1 NEW TOP HONEST BLEND",
                "note": (
                    "Breaks CYCLE-5 nb982 0.6107 ceiling by -0.0244 RAE; "
                    "aggressive stretch (s=1.25 vs canonical 1.05-1.10) on "
                    "blend anchor with collapsed variance"
                ),
            },
            {
                "rank": 2,
                "script": "nb986_asymmetric_blend",
                "csv": "submissions/nb986_asymmetric_blend.csv",
                "method": "Asymmetric 2-way blend over PRE-unblind anchors",
                "in_RAE": 0.6105,
                "predicted_LB": 0.6135,
                "status": "ADDED",
                "regime": "PRE-unblind",
                "ladder_promotion": "CYCLE-5-EXT-2",
                "note": "Ties nb982 best-blend; asymmetric weighting no-op vs SLSQP",
            },
            {
                "rank": 3,
                "script": "nb989_chemprop_aux_affine",
                "csv": "submissions/nb989_chemprop_aux_affine_a1.10_b-0.50.csv",
                "method": "Affine recalibration (a=1.10, b=-0.50) on chemprop_aux",
                "in_RAE": 0.6116,
                "predicted_LB": 0.6146,
                "status": "ADDED",
                "regime": "PRE-unblind",
                "ladder_promotion": "CYCLE-5-EXT-3",
                "note": "Affine on chemprop_aux beats raw 0.6216 by -0.010 RAE",
            },
            {
                "rank": 4,
                "script": "nb987_exp_avg_ensemble",
                "csv": "submissions/nb987_exp_avg_ensemble.csv",
                "method": "Exp-avg ensemble of PRE-unblind anchors",
                "in_RAE": 0.6144,
                "predicted_LB": 0.6174,
                "status": "ADDED",
                "regime": "PRE-unblind",
                "ladder_promotion": "CYCLE-5-EXT-4",
                "note": "Exp-avg ~ matches SLSQP linear blend; no nonlinear lift",
            },
            {
                "rank": 5,
                "script": "nb985_extended_blend",
                "csv": "submissions/nb985_extended_blend.csv",
                "method": "Extended 5-way SLSQP blend over expanded anchor pool",
                "in_RAE": 0.6188,
                "predicted_LB": 0.6218,
                "status": "ADDED",
                "regime": "PRE-unblind",
                "ladder_promotion": "CYCLE-5-EXT-5",
                "note": (
                    "5-way extension underperforms 4-way nb982 (0.6107); "
                    "consistent with stack-overfit-past-5 finding"
                ),
            },
        ],
        "best_method": best_method,
        "best_in_rae": best_in_rae,
        "lessons": [
            "nb988 stretch-on-blend-anchor breaks nb982 0.6107 ceiling by -0.0244 (NEW TOP)",
            "Aggressive stretch s=1.25 valid on collapsed-variance blend anchor (vs 1.05-1.10 on single-model)",
            "Affine recalibration on chemprop_aux extracts -0.010 RAE; bias-shift dimension active here",
            "Extending blend from 4 to 5 anchors (nb985) regresses by +0.008 RAE; stack-overfit-past-5 confirmed",
            "Asymmetric and exp-avg variants ~tie SLSQP linear blend; nonlinear blend operators inert",
        ],
        "ladder_updates": [res_primary, res_full],
        "ladder_top_after_cycle": [
            {"file": "te_nb988_nb982_stretch", "in_RAE": 0.5863, "status": "CYCLE-5-EXT-1 NEW TOP HONEST BLEND"},
            {"file": "te_nb986_asymmetric_blend", "in_RAE": 0.6105, "status": "CYCLE-5-EXT-2"},
            {"file": "te_nb982_best_blend", "in_RAE": 0.6107, "status": "CYCLE-5-1 (legacy top)"},
            {"file": "te_nb989_chemprop_aux_affine", "in_RAE": 0.6116, "status": "CYCLE-5-EXT-3"},
            {"file": "te_nb987_exp_avg_ensemble", "in_RAE": 0.6144, "status": "CYCLE-5-EXT-4"},
        ],
    }
    out = ART_DIR / "cycle5_ext_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"WROTE {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
