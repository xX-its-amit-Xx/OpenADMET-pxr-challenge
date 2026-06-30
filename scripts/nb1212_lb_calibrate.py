"""nb1212 -- LB calibration ground-truth using verified anchor.

Per feedback_lb_two_regime_calibration memory:
  PRE-unblind in_RAE approx LB + 0.003  (verified n=4 pre-freeze, 2026-05-13..20)
  POST-unblind in_RAE unreliable, LB likely 0.7-0.9 because models leak the 253

This script:
  1. Lists every Phase-1-close (>= 2026-05-26) activity submission in
     submission_log.csv
  2. Reports (file, predicted LB, actual LB) for each
  3. Re-polls the LB to detect any newly-graded entries
  4. Computes observed PRE-unblind delta (in_RAE - LB) for any new entries
  5. Recalibrates nb1191 LB estimate using the observed delta
  6. Recalibrates nb1162 LB estimate using the observed delta
  7. Identifies the queued submission most likely to score first

Output: data/processed/nb1212_summary.json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LB_LOG = ROOT / "data" / "processed" / "leaderboard_log.csv"
SUB_LOG = ROOT / "data" / "processed" / "submission_log.csv"
SUMMARY = ROOT / "data" / "processed" / "nb1212_summary.json"

PHASE1_CLOSE = "2026-05-26"
USER_HANDLE = "xX-its-amit-Xx"

# Verified PRE-unblind anchor (only graded entry on LB as of cycle):
#   submitted 2026-05-26 04:45 UTC, in_RAE 0.7625 -> LB 0.7655
#   gap = LB - in_RAE = +0.003 (memory)
# In submission_log: file=FINAL_phase1_nb120_huber_2_0.csv local OOF=0.3839
# but the in_RAE on 513 from pre_unblind_lb_candidates is 0.7502 for
# nb120_huber_2_0 -> implies actual delta is LB(0.7655) - in_RAE(0.7502) = +0.0153
ANCHOR_FILE = "FINAL_phase1_nb120_huber_2_0.csv"
ANCHOR_LB = 0.7655
# Use the in_RAE from data/processed/pre_unblind_lb_candidates.csv as the
# verified PRE-unblind in_RAE for the anchor
ANCHOR_IN_RAE = 0.7501893781890057  # te_nb120_huber_2_0 in pre_unblind candidates


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_pre_unblind_candidates() -> pd.DataFrame:
    p = ROOT / "data" / "processed" / "pre_unblind_lb_candidates.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def lookup_in_rae(file_stem: str, cand: pd.DataFrame) -> float | None:
    """Find in_RAE on 513 for a submission file by stem matching."""
    if cand.empty:
        return None
    stem = file_stem.replace(".csv", "")
    # try several stems
    for key in (f"te_{stem}", f"te_{stem.replace('_', '')}",
                f"te_oof_{stem}", stem):
        hit = cand[cand["file"].astype(str).str.lower() == key.lower()]
        if not hit.empty:
            v = hit["in_RAE"].iloc[0]
            if pd.notna(v) and v > 0:
                return float(v)
    # fallback: substring
    hit = cand[cand["file"].astype(str).str.contains(stem, case=False, na=False)]
    if not hit.empty:
        v = hit.sort_values("in_RAE").iloc[0]["in_RAE"]
        if pd.notna(v) and v > 0:
            return float(v)
    return None


def list_phase1_submissions() -> pd.DataFrame:
    """Return all activity submissions since Phase-1 close, parsed."""
    df = pd.read_csv(SUB_LOG)
    # robust parse of submitted_utc -> date string
    def to_date(s: str) -> str | None:
        if not isinstance(s, str):
            return None
        m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
        return m.group(1) if m else None

    df["date"] = df["submitted_utc"].apply(to_date)
    keep = df[(df["date"].notna()) & (df["date"] >= PHASE1_CLOSE)].copy()
    # filter out audit_complete / structure logs
    keep = keep[~keep["file"].astype(str).str.contains("audit_complete", na=False)]
    keep = keep[~keep["file"].astype(str).str.contains(".zip", na=False)]
    return keep.reset_index(drop=True)


def extract_predicted_lb(notes: str) -> float | None:
    """Pull predicted LB from the notes column."""
    if not isinstance(notes, str):
        return None
    m = re.search(r"predicted\s*LB\s*[~]?(\d+\.\d+)", notes, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"in_RAE\s+(\d+\.\d+)", notes, re.I)
    if m:
        # fall back to in_RAE if no explicit predicted LB
        return float(m.group(1)) + 0.003
    return None


def poll_leaderboard() -> dict:
    """Run log_lb_scores.py to refresh LB; return parsed best activity LB."""
    out = {"polled": False, "best_activity_lb": None,
           "best_activity_submitted": None,
           "n_visible_activity": None, "error": None}
    try:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "log_lb_scores.py")],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        out["polled"] = True
        out["stdout_tail"] = r.stdout[-500:] if r.stdout else ""
        out["stderr_tail"] = r.stderr[-300:] if r.stderr else ""
    except Exception as e:
        out["error"] = repr(e)
    # read latest LB log row regardless of poll success
    if LB_LOG.exists():
        lb = pd.read_csv(LB_LOG)
        act = lb[lb["track"] == "activity"].dropna(subset=["lb_score"])
        if not act.empty:
            last = act.sort_values("timestamp_utc").iloc[-1]
            out["best_activity_lb"] = float(last["lb_score"])
            out["best_activity_submitted"] = str(last.get("submitted_utc_on_lb"))
            out["n_visible_activity"] = int(last.get("n_submissions_visible") or 0)
    return out


def recalibrate(in_rae: float, observed_delta: float, default_delta: float = 0.003) -> dict:
    """Apply observed_delta if reasonable, else fall back to memory delta."""
    used = observed_delta if observed_delta is not None else default_delta
    return {
        "in_rae": in_rae,
        "delta_used": used,
        "delta_source": "observed" if observed_delta is not None else "memory_default",
        "predicted_lb": in_rae + used,
    }


def main() -> None:
    cand = load_pre_unblind_candidates()
    subs = list_phase1_submissions()

    # 1+2: enrich each row with in_RAE + predicted_LB
    rows = []
    for _, r in subs.iterrows():
        f = str(r["file"])
        notes = str(r.get("notes", ""))
        in_rae = lookup_in_rae(f, cand)
        pred_lb = extract_predicted_lb(notes)
        rows.append({
            "submitted_utc": str(r["submitted_utc"]),
            "file": f,
            "in_rae_513": in_rae,
            "predicted_lb": pred_lb,
            "actual_lb": None,  # filled in below if matched on poll
        })
    sub_table = pd.DataFrame(rows)

    # 3: poll LB
    poll = poll_leaderboard()
    current_lb = poll.get("best_activity_lb")
    current_lb_subdt = poll.get("best_activity_submitted")

    # try to attribute current LB to a specific submission by timestamp match
    matched_file = None
    if current_lb_subdt and not sub_table.empty:
        # current_lb_subdt format: "2026-05-26 04:45 UTC"
        # submitted_utc format: "2026-05-26 04:45" or "2026-06-01 04:35:02 UTC"
        prefix = str(current_lb_subdt).split(" UTC")[0][:16]  # YYYY-MM-DD HH:MM
        for i, row in sub_table.iterrows():
            ts = row["submitted_utc"][:16]
            if ts == prefix:
                matched_file = row["file"]
                sub_table.at[i, "actual_lb"] = current_lb
                break

    # 4: PRE-unblind delta
    # Anchor: FINAL_phase1_nb120_huber_2_0 (verified PRE-unblind file)
    anchor_in_rae = ANCHOR_IN_RAE
    anchor_lb = current_lb if current_lb else ANCHOR_LB
    observed_delta = None
    new_pre_unblind_graded = []

    # If the LB shows the anchor file, use it
    if matched_file == ANCHOR_FILE and current_lb is not None:
        observed_delta = anchor_lb - anchor_in_rae
        new_pre_unblind_graded.append({
            "file": ANCHOR_FILE,
            "in_rae_513": anchor_in_rae,
            "lb": anchor_lb,
            "delta": observed_delta,
        })
    else:
        # If LB updated to something other than the legacy anchor and it's
        # one of our PRE-unblind submissions, treat that as the new anchor
        if matched_file is not None and current_lb is not None:
            in_rae = lookup_in_rae(matched_file, cand)
            if in_rae is not None:
                observed_delta = current_lb - in_rae
                new_pre_unblind_graded.append({
                    "file": matched_file,
                    "in_rae_513": in_rae,
                    "lb": current_lb,
                    "delta": observed_delta,
                })

    # 5+6: recalibrate nb1191 and nb1162
    # in_RAE on the 513 test set, in-sample (model trained on all 4139 incl 253)
    # nb1191 in_sample is te_unb_rae_in_sample = 0.2631 (very optimistic, leaky)
    # nb1191 honest cross-fit pooled_rae_mean_seeds = 0.4703
    # The PRE-unblind PRE-freeze rule (in_RAE = LB + 0.003) was for models
    # trained PRE-unblind. nb1191 is POST-unblind (trained on 253). So the
    # honest LB prediction uses the cross-fit number 0.4703, not in_sample.
    nb1191_summary = json.loads((ROOT / "data" / "processed" / "nb1191_summary.json").read_text())
    nb1162_summary = json.loads((ROOT / "data" / "processed" / "nb1162_summary.json").read_text())

    nb1191_crossfit = nb1191_summary["pooled_rae_mean_seeds"]      # 0.4703
    nb1191_in_sample = nb1191_summary["in_sample_rae_overfit_bound"]  # 0.4647
    nb1162_crossfit = nb1162_summary["pooled_scaffold_cv_rae"]     # 0.4206
    nb1162_in_sample = nb1162_summary["in_sample_rae_overfit_bound"]  # 0.4172

    # Per the memory, POST-unblind in_RAE estimates aren't trustworthy at all
    # because the 253 leak is in the model itself. Honest LB estimate is the
    # cross-fit number itself, with a +0.10 "train->unblind" shift that has
    # been observed historically (feedback_train_oof_blend_transfer).
    POST_UNBLIND_SHIFT = 0.10  # conservative

    nb1191_recal = {
        "honest_cross_fit": nb1191_crossfit,
        "in_sample_513": nb1191_in_sample,
        "predicted_lb_optimistic": nb1191_crossfit + 0.003,  # if PRE-unblind rule held
        "predicted_lb_conservative": nb1191_crossfit + POST_UNBLIND_SHIFT,
        "predicted_lb_band": [nb1191_crossfit + 0.003,
                               nb1191_crossfit + POST_UNBLIND_SHIFT],
        "delta_used_if_observed": observed_delta,
        "note": "POST-unblind refit; honest LB likely in [0.473, 0.570]; do NOT trust in_sample 0.4647",
    }
    if observed_delta is not None:
        nb1191_recal["predicted_lb_calibrated"] = nb1191_crossfit + max(0.003, observed_delta)

    nb1162_recal = {
        "honest_cross_fit": nb1162_crossfit,
        "in_sample_513": nb1162_in_sample,
        "predicted_lb_optimistic": nb1162_crossfit + 0.003,
        "predicted_lb_conservative": nb1162_crossfit + POST_UNBLIND_SHIFT,
        "predicted_lb_band": [nb1162_crossfit + 0.003,
                               nb1162_crossfit + POST_UNBLIND_SHIFT],
        "delta_used_if_observed": observed_delta,
        "note": "POST-unblind refit; honest LB likely in [0.424, 0.521]; do NOT trust in_sample 0.4172",
    }
    if observed_delta is not None:
        nb1162_recal["predicted_lb_calibrated"] = nb1162_crossfit + max(0.003, observed_delta)

    # 7: identify which queued submission scores first
    # The earliest unmatched submission after Phase-1 close is the next to grade
    queued = sub_table[sub_table["actual_lb"].isna()].copy()
    # parse submitted_utc into sortable
    queued["dt"] = pd.to_datetime(queued["submitted_utc"], errors="coerce", utc=True)
    queued = queued.dropna(subset=["dt"]).sort_values("dt")
    likely_first = None
    if not queued.empty:
        first = queued.iloc[0]
        likely_first = {
            "submitted_utc": str(first["submitted_utc"]),
            "file": str(first["file"]),
            "predicted_lb": first.get("predicted_lb"),
            "in_rae_513": first.get("in_rae_513"),
        }

    # chemprop_aux specific check (per memory: predicted LB 0.6246)
    chemprop_aux_in_rae = lookup_in_rae("chemprop_aux", cand)
    chemprop_aux_lb_pred = None
    if chemprop_aux_in_rae is not None:
        chemprop_aux_lb_pred = {
            "in_rae_513": chemprop_aux_in_rae,
            "predicted_lb_pre_unblind_rule": chemprop_aux_in_rae + 0.003,
            "note": "chemprop_aux is PRE-unblind (trained on 4139 only); PRE-unblind +0.003 rule applies",
        }

    summary = {
        "ts": utc_now(),
        "phase1_close": PHASE1_CLOSE,
        "anchor": {
            "file": ANCHOR_FILE,
            "expected_in_rae_513": anchor_in_rae,
            "expected_lb": ANCHOR_LB,
            "memo_delta_pre_unblind": 0.003,
        },
        "poll": poll,
        "current_lb_attribution": {
            "matched_file": matched_file,
            "current_lb": current_lb,
            "current_lb_submitted_on_lb": current_lb_subdt,
        },
        "newly_graded_pre_unblind": new_pre_unblind_graded,
        "observed_delta": observed_delta,
        "observed_delta_used": observed_delta is not None,
        "nb1191_recalibrated": nb1191_recal,
        "nb1162_recalibrated": nb1162_recal,
        "chemprop_aux": chemprop_aux_lb_pred,
        "likely_first_to_grade": likely_first,
        "n_phase1_submissions": int(len(sub_table)),
        "n_with_in_rae": int(sub_table["in_rae_513"].notna().sum()),
        "n_with_pred_lb": int(sub_table["predicted_lb"].notna().sum()),
        "n_with_actual_lb": int(sub_table["actual_lb"].notna().sum()),
        "phase1_submissions": sub_table.to_dict(orient="records"),
    }

    # convert NaN -> None for valid JSON
    def _scrub(o):
        if isinstance(o, dict):
            return {k: _scrub(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_scrub(v) for v in o]
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    SUMMARY.write_text(json.dumps(_scrub(summary), indent=2, default=str))
    print(f"[nb1212] wrote {SUMMARY.relative_to(ROOT)}")
    print(f"  n_phase1_subs={len(sub_table)}  n_graded={summary['n_with_actual_lb']}")
    print(f"  current LB best (activity): {current_lb} (attributed: {matched_file})")
    print(f"  observed delta (in_RAE-LB): {observed_delta}")
    print(f"  nb1191 LB band: {nb1191_recal['predicted_lb_band']}")
    print(f"  nb1162 LB band: {nb1162_recal['predicted_lb_band']}")
    if chemprop_aux_lb_pred:
        print(f"  chemprop_aux predicted LB: "
              f"{chemprop_aux_lb_pred['predicted_lb_pre_unblind_rule']:.4f} (PRE-unblind rule)")
    if likely_first:
        print(f"  likely next to grade: {likely_first['file']} "
              f"@ {likely_first['submitted_utc']}")


if __name__ == "__main__":
    main()
