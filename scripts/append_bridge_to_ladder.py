"""Append bridge in_RAE rows to PRE-unblind ladder CSVs (in_RAE-sorted) and
emit bridge_summary.json artifact.

Bridge in_RAEs (2026-06-03):
- nb900 LLM retry        : 0.0000  -> top of ladder (PRIMARY-1 candidate)
- nb914 PH retry         : 0.7062
- nb970 Multi-conformer  : N/A     -> skipped (no CSV)
- nb971 RDKit-only       : 0.7504
- nb972 Long-train       : 0.6898
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
ART_DIR = Path("C:/pxr_artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

# (file_basename, nb_num, in_RAE)
BRIDGE_ROWS = [
    ("te_nb900_llm_retry",     "900.0", 0.0000),
    ("te_nb914_ph_retry",      "914.0", 0.7062),
    ("te_nb972_long_train",    "972.0", 0.6898),
    ("te_nb971_rdkit_only",    "971.0", 0.7504),
]
SKIPPED = [
    ("te_nb970_multi_conformer", "970.0", None, "no CSV / N/A"),
]


def update_ladder(csv_path: Path) -> dict:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]

    # parse existing into tuples
    parsed = []
    for r in body:
        if not r:
            continue
        in_rae = float(r[2])
        parsed.append((r[0], r[1], in_rae, r[3]))

    # add bridge rows
    for fname, nb, in_rae in BRIDGE_ROWS:
        parsed.append((fname, nb, in_rae, "513"))

    # sort ascending by in_RAE
    parsed.sort(key=lambda x: x[2])

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for fname, nb, in_rae, n in parsed:
            w.writerow([fname, nb, in_rae, n])

    # locate bridge rows after sort
    locs = {fname: None for fname, _, _ in BRIDGE_ROWS}
    for idx, (fname, _, in_rae, _) in enumerate(parsed, start=2):  # +1 header, +1 1-indexed
        if fname in locs:
            locs[fname] = {"line": idx, "in_RAE": in_rae}
    return {"file": str(csv_path), "n_rows": len(parsed) + 1, "bridge_positions": locs}


def main():
    primary = ROOT / "data/processed/pre_unblind_lb_candidates.csv"
    full    = ROOT / "data/processed/pre_unblind_lb_candidates_all.csv"

    res_primary = update_ladder(primary)
    res_full    = update_ladder(full)

    summary = {
        "ts": "2026-06-03",
        "phase": "bridge",
        "bridge_rows": [
            {"file": fname, "nb_num": nb, "in_RAE": in_rae}
            for fname, nb, in_rae in BRIDGE_ROWS
        ],
        "skipped": [
            {"file": fname, "nb_num": nb, "reason": reason}
            for fname, nb, _, reason in SKIPPED
        ],
        "ladder_updates": [res_primary, res_full],
        "notes": (
            "Bridge cycle: nb900 LLM retry, nb914 PH retry, nb970 multi-conformer "
            "(skipped, N/A), nb971 RDKit-only, nb972 long-train. "
            "Sorted by in_RAE ascending. nb900_llm_retry in_RAE=0.0000 surfaces as "
            "ladder-row #2 (PRIMARY-1 candidate) but the 0.0000 value is "
            "suspiciously low; treat as in-sample contamination flag until "
            "audit_ladder_integrity passes."
        ),
    }
    out = ART_DIR / "bridge_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"WROTE {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
