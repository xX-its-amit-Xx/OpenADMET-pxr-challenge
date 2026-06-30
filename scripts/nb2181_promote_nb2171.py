"""nb2181 -- conditional promotion of nb2171 to PRIMARY-1-PRE-NB1191-PYRAMID.

Per cycle 166 spec: if nb2180 deep-30 verify returns PROMOTE, prepend nb2171
into scripts/auto_submit_ladder.LADDER as PRIMARY-1-PRE-NB1191-PYRAMID,
slotted between nb1162 (AGGRESSIVE) and nb2095 (PRE-PYRAMID). Otherwise HOLD
and document the gating reason.

Pipeline:
    1. Verify (or build) submissions/nb2171_deploy_anchor_swap.csv from
       data/processed/te_nb2171.npy using the canonical TEST_BLINDED template.
    2. Read data/processed/nb2180_summary.json verdict.
    3. If verdict == "PROMOTE": rewrite scripts/auto_submit_ladder.py to
       prepend a cycle-166 header comment and insert the new PRIMARY-1
       (PRE-NB1191-PYRAMID) tuple between the nb1162 AGGRESSIVE row and the
       nb2095 PRE-PYRAMID row.
    4. Else: leave LADDER unchanged and write HOLD reason to summary.
    5. Run scripts/audit_ladder_integrity.py.

Writes:
    submissions/nb2171_deploy_anchor_swap.csv         (always, if missing)
    data/processed/nb2181_promote_nb2171_summary.json (always)
    scripts/auto_submit_ladder.py                     (PROMOTE only)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxr.paths import DATA_PROCESSED, SUBMISSIONS  # type: ignore

TAG = "nb2181_promote_nb2171"
TEST_CSV = ROOT / "data" / "raw" / "pxr-challenge_TEST_BLINDED.csv"
TE_NB2171 = DATA_PROCESSED / "te_nb2171.npy"
NB2180_SUMMARY = DATA_PROCESSED / "nb2180_summary.json"
NB2171_SUMMARY = DATA_PROCESSED / "nb2171_summary.json"
DEPLOY_CSV = SUBMISSIONS / "nb2171_deploy_anchor_swap.csv"
LADDER_PATH = ROOT / "scripts" / "auto_submit_ladder.py"
AUDIT_PATH = ROOT / "scripts" / "audit_ladder_integrity.py"
SUMMARY_PATH = DATA_PROCESSED / f"{TAG}_summary.json"

# Ladder editing anchors (must match current cycle-162 layout)
NEW_LADDER_ROW = (
    '    ("nb2171_deploy_anchor_swap.csv",                   '
    '"PRIMARY-1-PRE-NB1191-PYRAMID (cycle-166): nb2171 nb1162 anchor-swap '
    'replacing nb730_honest POST-anchor with nb1191 PRE-pyramid; SLSQP '
    'deploy weights 93.6% nb1191 + 6.4% nb562 (chemprop_aux/nb503/nb2103 '
    'pruned to 0); seed-mean scaffold-CV OOF RAE 0.4676 / mean-of-seeds '
    'OOF 0.4673; deploy stretch s=1.009; nb2180 deep-30 verify PROMOTE; '
    'predicted LB band 0.32-0.42 (mid 0.37); cleanest PRE-only anchor '
    'swap on the nb1162 stack-pyramid family"),\n'
)
NB1162_LINE_FRAGMENT = '("nb1162_deploy_nb1153.csv",'
NB2095_LINE_FRAGMENT = '("nb2095_deploy_pre_pyramid.csv",'

CYCLE_166_HEADER = """# ====================================================
# CYCLE 166 LADDER PROMOTE 2026-06-08 -- nb2171 added as PRIMARY-1-PRE-NB1191-PYRAMID
# nb2180 deep-30 verify (stretch grid + scaffold-fold SLSQP over 30 seeds) returned
# PROMOTE for nb2171 (nb1162 anchor-swap: drops nb730_honest POST-anchor and
# replaces it with the nb1191 PRE-pyramid deploy vector). Cycle-159 ablation:
# nb2171 honest seed-mean scaffold-CV RAE 0.4676 (vs nb2095 0.4703, -0.0027);
# nb1191 carries 93.6% of the deploy weight + nb562 6.4%. Predicted LB band
# mid 0.37 per 0.51*pred_oof + 0.49*te[unb] calibration. New ordering:
#   PRIMARY-1 (AGGRESSIVE, POST risk):    nb1162_deploy_nb1153.csv       scaffold-CV 0.4206 (LB queue pending)
#   PRIMARY-1 (PRE-NB1191-PYRAMID, NEW):  nb2171_deploy_anchor_swap.csv  scaffold-CV 0.4676 (PRE-anchor only)
#   PRIMARY-1 (PRE-PYRAMID):              nb2095_deploy_pre_pyramid.csv  scaffold-CV 0.4703 (drops nb730)
#   PRIMARY-1 (LB-SAFE-DEEP):             nb1191_deploy_pre_pyramid.csv  scaffold-CV 0.4718 wide-seed verified
#   PRIMARY-1 (LB-CLEAN-HOLD):            nb2060_deploy_no_nb730.csv     deep-30 sub-noise miss
# Rotation rule: AGGRESSIVE fires first if open, then PRE-NB1191-PYRAMID nb2171,
# then PRE-PYRAMID nb2095, then LB-SAFE-DEEP nb1191, then LB-CLEAN-HOLD nb2060.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_deploy_csv() -> dict:
    """Build submissions/nb2171_deploy_anchor_swap.csv from te_nb2171.npy if absent."""
    if DEPLOY_CSV.exists():
        return {
            "deploy_csv_status": "EXISTS",
            "deploy_csv_path": str(DEPLOY_CSV),
            "deploy_csv_built": False,
        }
    if not TE_NB2171.exists():
        return {
            "deploy_csv_status": "MISSING_TE_NPY",
            "deploy_csv_path": str(TE_NB2171),
            "deploy_csv_built": False,
        }
    te = np.load(TE_NB2171)
    if not TEST_CSV.exists():
        return {
            "deploy_csv_status": "MISSING_TEST_CSV",
            "deploy_csv_path": str(TEST_CSV),
            "deploy_csv_built": False,
        }
    tdf = pd.read_csv(TEST_CSV)
    smi_col = "SMILES" if "SMILES" in tdf.columns else [c for c in tdf.columns if "smile" in c.lower()][0]
    name_col = "Molecule Name" if "Molecule Name" in tdf.columns else [c for c in tdf.columns if "name" in c.lower()][0]
    if len(tdf) != len(te):
        return {
            "deploy_csv_status": "LENGTH_MISMATCH",
            "deploy_csv_path": str(DEPLOY_CSV),
            "deploy_csv_built": False,
            "test_n": int(len(tdf)),
            "te_n": int(len(te)),
        }
    out = pd.DataFrame({
        "SMILES": tdf[smi_col].values,
        "Molecule Name": tdf[name_col].values,
        "pEC50": te.astype(float),
    })
    DEPLOY_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DEPLOY_CSV, index=False)
    return {
        "deploy_csv_status": "BUILT",
        "deploy_csv_path": str(DEPLOY_CSV),
        "deploy_csv_built": True,
        "te_mean": float(te.mean()),
        "te_std": float(te.std()),
    }


def read_nb2180_verdict() -> dict:
    if not NB2180_SUMMARY.exists():
        return {
            "nb2180_status": "MISSING",
            "verdict": "HOLD",
            "reason": f"nb2180_summary.json not found at {NB2180_SUMMARY}",
        }
    try:
        with open(NB2180_SUMMARY, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "nb2180_status": "READ_ERROR",
            "verdict": "HOLD",
            "reason": f"failed to read nb2180_summary.json: {e}",
        }
    raw = (data.get("verdict") or data.get("decision") or data.get("gate") or "").upper()
    if "PROMOTE" in raw:
        verdict = "PROMOTE"
    elif "HOLD" in raw or "REJECT" in raw or "FAIL" in raw:
        verdict = "HOLD"
    else:
        verdict = "HOLD"
    return {
        "nb2180_status": "READ",
        "verdict": verdict,
        "raw_verdict": raw,
        "nb2180_keys": sorted(list(data.keys()))[:20],
    }


def insert_into_ladder() -> dict:
    text = LADDER_PATH.read_text(encoding="utf-8")
    if "nb2171_deploy_anchor_swap.csv" in text and "PRIMARY-1-PRE-NB1191-PYRAMID" in text:
        return {"ladder_status": "ALREADY_PRESENT"}
    lines = text.splitlines(keepends=True)
    # Find nb1162 row index inside the LADDER list
    idx_nb1162 = None
    idx_nb2095 = None
    for i, ln in enumerate(lines):
        if NB1162_LINE_FRAGMENT in ln and idx_nb1162 is None:
            idx_nb1162 = i
        elif NB2095_LINE_FRAGMENT in ln and idx_nb2095 is None:
            idx_nb2095 = i
    if idx_nb1162 is None or idx_nb2095 is None:
        return {
            "ladder_status": "ANCHOR_LINES_NOT_FOUND",
            "idx_nb1162": idx_nb1162,
            "idx_nb2095": idx_nb2095,
        }
    if idx_nb2095 != idx_nb1162 + 1:
        return {
            "ladder_status": "ANCHOR_LINES_NOT_ADJACENT",
            "idx_nb1162": idx_nb1162,
            "idx_nb2095": idx_nb2095,
        }
    # Find the first LADDER = [ line to attach the cycle-166 header just above it.
    ladder_open_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("LADDER = ["):
            ladder_open_idx = i
            break
    if ladder_open_idx is None:
        return {"ladder_status": "LADDER_OPEN_NOT_FOUND"}
    new_lines = (
        lines[:ladder_open_idx]
        + [CYCLE_166_HEADER]
        + lines[ladder_open_idx:idx_nb2095]
        + [NEW_LADDER_ROW]
        + lines[idx_nb2095:]
    )
    LADDER_PATH.write_text("".join(new_lines), encoding="utf-8")
    return {
        "ladder_status": "INSERTED",
        "insert_after_line": idx_nb1162 + 1,
        "header_at_line": ladder_open_idx,
    }


def run_audit() -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, str(AUDIT_PATH)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        return {
            "audit_returncode": proc.returncode,
            "audit_stdout_tail": proc.stdout[-2000:],
            "audit_stderr_tail": proc.stderr[-1000:],
        }
    except Exception as e:
        return {"audit_returncode": -1, "audit_error": str(e)}


def main() -> int:
    out = {"tag": TAG, "utc": utcnow()}
    out.update(ensure_deploy_csv())
    verdict_info = read_nb2180_verdict()
    out.update(verdict_info)
    # Pull supporting metrics from nb2171_summary.json for traceability.
    if NB2171_SUMMARY.exists():
        try:
            with open(NB2171_SUMMARY, "r", encoding="utf-8") as f:
                s = json.load(f)
            out["nb2171_pooled_rae_mean_seeds"] = s.get("pooled_rae_mean_seeds")
            out["nb2171_w_nb1191_deploy"] = s.get("w_nb1191_deploy")
            out["nb2171_deploy_s"] = s.get("deploy_s")
            out["nb2171_lb_band_estimate"] = s.get("lb_band_estimate")
        except Exception as e:
            out["nb2171_summary_error"] = str(e)
    if verdict_info["verdict"] == "PROMOTE":
        out.update(insert_into_ladder())
        out.update(run_audit())
        out["action"] = "PROMOTED"
    else:
        out["action"] = "HELD"
        out["hold_reason"] = verdict_info.get("reason") or (
            f"nb2180 verdict={verdict_info.get('raw_verdict')!r} did not == PROMOTE; "
            "keeping nb2171 as PRIMARY-2 backup (entry remains in DEPRECATED line per "
            "the cycle-125 audit warning; ladder unchanged)"
        )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
