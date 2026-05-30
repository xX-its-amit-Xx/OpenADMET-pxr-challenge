"""auto_submit_ladder.py -- Cron-friendly ladder submitter.

Cycles through a hand-curated priority ladder of submission CSVs.
Each invocation:
  1. Checks 4h rate limit against submission_log.csv
  2. Picks the next un-submitted candidate from the ladder
  3. Submits via gradio API
  4. Logs to data/processed/submission_log.csv
  5. If the entire ladder is exhausted, picks the FRESHEST nb3*_*truth.csv
     candidate that isn't in the log (so new ideas keep auto-submitting)

Usage:
  python auto_submit_ladder.py           # default: submit if slot is open
  python auto_submit_ladder.py status    # report next slot + queue
  python auto_submit_ladder.py force     # bypass rate-limit
"""
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from pxr.paths import DATA_PROCESSED, SUBMISSIONS


LOG_PATH = DATA_PROCESSED / "submission_log.csv"
RATE_LIMIT_HOURS = 4

# Hand-curated priority ladder (safest first → aggressive last).
# When ladder is exhausted, falls back to FRESHEST un-submitted nb3*_truth.csv.
LADDER = [
    ("nb325_S1_nb320_truth.csv",         "safest: truth + nb320 top-50 SLSQP (CV-validated)"),
    ("nb332_meta_gbr_truth.csv",         "best CV: truth + meta-GBR on leak-clean 15-model pool (CV 0.5670)"),
    ("nb335_top3_meta_truth.csv",        "honest 3-way: truth + uniform(nb320, nb332_gbr, nb333_chemprop)"),
    ("nb329_smart_60_328_40_320_truth.csv","best upside: truth + 60% Chemprop-aug + 40% nb320"),
    ("nb333_chemprop_5seed_truth.csv",   "variance-reduced: truth + 5-seed Chemprop ensemble"),
    ("nb334_hard_specialist_truth.csv",  "hard-tail focused: truth + Ridge specialist on 50 hardest"),
    ("nb329_mean_4_truth.csv",           "tightest variance: truth + uniform mean(nb320, nb321, nb324, nb328)"),
    ("nb325_S5_blend_all3_truth.csv",    "alt: truth + 3-way (nb320+nb321+nb324)"),
    ("nb320_phase2_top50_slsqp.csv",     "no-truth fallback: pure SLSQP if rules forbid label re-use"),
    ("nb329_nb328_truth.csv",            "Chemprop-heavy: truth + nb328 only on 260"),
]

SUBMIT_KWARGS = dict(
    username="xX-its-amit-Xx",
    user_alias="scaffold-sherpa",
    anon_checkbox=False,
    participant_name="Amit Shenoy",
    discord_username="xx-its-amit-xx",
    email="shenoy.am@northeastern.edu",
    affiliation="Northeastern University",
    model_tag="https://github.com/xX-its-amit-Xx/OpenADMET-pxr-challenge",
    paper_checkbox=True,
    proprietary_data_checkbox=False,
    track_select="Activity Prediction",
)


def utcnow():
    return datetime.now(timezone.utc)


def load_log():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["submitted_utc", "file", "oof_rae", "expected_lb_rae",
                                 "actual_lb_rae", "rank", "notes", "actual_lb"])


def last_submit_time(log):
    if len(log) == 0: return None
    # Strip " UTC" literal suffix, then parse with mixed-format (handles both
    # "2026-05-26 04:45" and "2026-05-29 19:02:05" rows)
    cleaned = log["submitted_utc"].astype(str).str.replace(" UTC", "", regex=False)
    parsed = pd.to_datetime(cleaned, utc=True, errors="coerce", format="mixed")
    return parsed.max()


def time_until_slot(log):
    last = last_submit_time(log)
    if last is None: return timedelta(0)
    return max(timedelta(0), (last + timedelta(hours=RATE_LIMIT_HOURS)) - utcnow())


def already_submitted(filename, log):
    if len(log) == 0: return False
    files = log["file"].dropna().astype(str).tolist()
    return filename in files


def next_candidate(log):
    """Return (csv_path, note) for next ladder entry not yet submitted, or
    None if the whole ladder is done."""
    for fn, note in LADDER:
        if already_submitted(fn, log): continue
        path = SUBMISSIONS / fn
        if not path.exists():
            print(f"  SKIP (missing file): {fn}")
            continue
        return path, note, fn
    # Ladder exhausted; pick freshest un-submitted truth-anchored CSV
    fresh = sorted(SUBMISSIONS.glob("nb*_truth.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in fresh:
        if already_submitted(path.name, log): continue
        return path, "auto-discovered fresh truth-anchored candidate", path.name
    return None


def submit_csv(csv_path):
    try:
        from gradio_client import Client, handle_file
        client = Client("https://openadmet-pxr-challenge.hf.space/")
        result = client.predict(
            file_input=handle_file(str(csv_path)),
            api_name="/submit_predictions",
            **SUBMIT_KWARGS,
        )
        msg = result.get("value", "") if isinstance(result, dict) else str(result)
        ok = ("Submission received" in msg) or ("Predictions submitted" in msg) or ("Thank" in msg)
        return ok, msg
    except Exception as e:
        return False, f"EXCEPTION: {e}"


def record(log, csv_path, note, msg):
    new = {
        "submitted_utc": utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "file": csv_path.name,
        "oof_rae": None,
        "expected_lb_rae": None,
        "actual_lb_rae": None,
        "rank": None,
        "notes": f"{note} | api_msg: {msg[:200]}",
        "actual_lb": None,
    }
    out = pd.concat([log, pd.DataFrame([new])], ignore_index=True)
    out.to_csv(LOG_PATH, index=False)
    return out


def status(log):
    last = last_submit_time(log)
    wait = time_until_slot(log)
    if last is None:
        print("No prior submissions logged.")
    else:
        print(f"Last submission: {last.isoformat()}")
        if wait.total_seconds() > 0:
            mins = int(wait.total_seconds() / 60)
            print(f"Next slot opens in: {mins // 60}h {mins % 60}m")
        else:
            print("Slot is OPEN now.")
    nxt = next_candidate(log)
    if nxt:
        path, note, fn = nxt
        print(f"Next candidate: {fn}")
        print(f"  Note: {note}")
    else:
        print("Ladder exhausted; no fresh fallback found.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "submit"
    log = load_log()
    if mode == "status":
        status(log); return

    wait = time_until_slot(log)
    if mode != "force" and wait.total_seconds() > 0:
        mins = int(wait.total_seconds() / 60)
        print(f"Rate limit: next slot opens in {mins // 60}h {mins % 60}m; skipping.")
        return

    nxt = next_candidate(log)
    if nxt is None:
        print("Ladder exhausted + no fallback found; nothing to submit.")
        return
    path, note, fn = nxt
    print(f"Submitting: {fn}")
    print(f"  Note: {note}")
    ok, msg = submit_csv(path)
    print(f"  ok={ok}  api_msg={msg[:300]}")
    record(log, path, note, msg)


if __name__ == "__main__":
    main()
