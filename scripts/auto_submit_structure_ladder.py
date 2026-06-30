"""auto_submit_structure_ladder.py -- Cron-friendly ladder submitter (Structure track).

Parallel to scripts/auto_submit_ladder.py but for the Structure Prediction track.
Cycles through a hand-curated priority ladder of submission ZIP bundles.

Each invocation:
  1. Checks 4h rate limit against data/processed/structure_submission_log.csv
  2. Picks the next un-submitted candidate from STRUCTURE_LADDER
  3. Submits via Gradio API with track_select="Structure Prediction"
  4. Appends to structure_submission_log.csv (preserves existing schema)

Usage:
  python auto_submit_structure_ladder.py           # submit if slot is open
  python auto_submit_structure_ladder.py status    # report next slot + queue
  python auto_submit_structure_ladder.py force     # bypass rate-limit
"""
import os, sys, hashlib
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd

from pxr.paths import DATA_PROCESSED, SUBMISSIONS


LOG_PATH = DATA_PROCESSED / "structure_submission_log.csv"
RATE_LIMIT_HOURS = 4
TRACK = "Structure Prediction"

# Hand-curated ladder.
# ===========================================================================
# CYCLE 161 REGRESSION (2026-06-08 01:29 UTC):
#   v6 (per-ligand QSel, nb2040) submitted 2026-06-08 01:17 UTC -> LB 0.4632
#   (rank 27/50). Collapsed to openadmet-boltz-baseline level, regressing
#   -0.0364 vs v5's verified 0.4996 (rank 10/50, 2026-06-02 16:50 UTC).
#   Diagnosis: per-ligand QSel (clash+pocket-distance) over-rotated the 14
#   disputed positions to v1/v2/v3 pure-baseline geometries, erasing v5's
#   template-bias gains. The QSel score is NOT correlated with LDDT-PLI on
#   this dataset -- it picked the wrong "best" pose 14/14 times.
#   ACTION: revert PRIMARY-1 to v5 (LB-verified 0.4996), v1 fallback,
#   demote v6 to DEPRECATED-AUDIT-REGRESSION.
# ===========================================================================
# 2026-06-01 history: v4 (170 Boltz + 14 RDKit redocks) scored LB 0.4583
# (rank 29/48), below openadmet-boltz-baseline 0.4632 -- the RDKit-placed
# centroid redocks HURT. v2/v3 (same redock failure mode) DEPRECATED.
STRUCTURE_LADDER = [
    # ===========================================================================
    # CYCLE 294 CROSS-REPO RECONCILE (2026-06-12): the STRUCTURE REPO
    # (OpenADMET-pxr-structure) overtook this ladder. Its ai_3model_zhybrid
    # (Boltz-2 + OpenFold3 + AF3 z-hybrid) scored LB 0.5414 rank 2/54 on
    # 2026-06-11 21:04 UTC (AF3 won 61/184; leader 0.5612). That zip is copied
    # here as structure_3model_zhybrid_0p5414.zip and is the new PRIMARY-1 so
    # this stale cron RESTORES rank-2 (0.5414) instead of clobbering it back to
    # v5's 0.4996. Do NOT resubmit v5 over 0.5414. New structure work happens in
    # the structure repo; this ladder is now a safety net only.
    # ===========================================================================
    {"file": "structure_3model_zhybrid_0p5414.zip", "note": "PRIMARY-1: LB-VERIFIED 0.5414 (rank 2/54, 2026-06-11 21:04 UTC); B2+OF3+AF3 z-hybrid from structure repo; 184 PDBs verified LIG; 8694557 bytes = submitted bytes"},
    {"file": "structure_baseline_v5_resubmit.zip", "note": "PRIMARY-2: LB-VERIFIED 0.4996 (rank 10/50, 2026-06-02 16:50 UTC); SUPERSEDED by 0.5414 3-model; keep as deep fallback only"},
    {"file": "structure_baseline_v5.zip", "note": "PRIMARY-2-REFERENCE: LB-VERIFIED 0.4996; original submission; identical bytes to _resubmit copy"},
    {"file": "structure_baseline_v1.zip", "note": "PRIMARY-2: LB-VERIFIED 0.4632 baseline (rank 27/48); 184 pure Boltz-2 cofolds; matches openadmet-boltz-baseline"},
    {"file": "structure_v6_perlig_qsel.zip", "note": "DEPRECATED-AUDIT-REGRESSION: Cycle 161 LB scored 0.4632 vs v5 0.4996 (rank 10->27); per-ligand QSel collapsed to baseline; do not resubmit"},
    # DEPRECATED below: redock variants underperformed pure Boltz (v4=0.4583 < baseline 0.4632).
    {"file": "structure_baseline_v3.zip", "note": "DEPRECATED: 170 Boltz + 14 per-ligand template redocks (RDKit redocks hurt; see v4 LB 0.4583)"},
    {"file": "structure_baseline_v2.zip", "note": "DEPRECATED: 170 Boltz + 14 global 8R81 redocks (RDKit redocks hurt; see v4 LB 0.4583)"},
    {"file": "structure_baseline_v4.zip", "note": "DEPRECATED: LB 0.4583 (rank 29/48); 170 Boltz + 14 RDKit centroid redocks"},
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
    track_select=TRACK,
)

LOG_COLUMNS = ["timestamp_utc", "track", "submission_csv", "sha256",
               "api_response", "lb_score_if_returned"]


def utcnow():
    return datetime.now(timezone.utc)


def load_log():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def last_submit_time(log):
    if len(log) == 0:
        return None
    cleaned = log["timestamp_utc"].astype(str).str.replace(" UTC", "", regex=False)
    parsed = pd.to_datetime(cleaned, utc=True, errors="coerce", format="mixed")
    return parsed.max()


def time_until_slot(log):
    last = last_submit_time(log)
    if last is None:
        return timedelta(0)
    return max(timedelta(0), (last + timedelta(hours=RATE_LIMIT_HOURS)) - utcnow())


def already_submitted(filename, log):
    if len(log) == 0:
        return False
    files = log["submission_csv"].dropna().astype(str).tolist()
    return filename in files


def next_candidate(log):
    """Return (path, note, filename) for next ladder entry not yet submitted."""
    for entry in STRUCTURE_LADDER:
        fn, note = entry["file"], entry["note"]
        if already_submitted(fn, log):
            continue
        path = SUBMISSIONS / fn
        if not path.exists():
            print(f"  SKIP (missing file): {fn}")
            continue
        return path, note, fn
    return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def submit_zip(zip_path: Path):
    try:
        from gradio_client import Client, handle_file
        client = Client("https://openadmet-pxr-challenge.hf.space/")
        result = client.predict(
            file_input=handle_file(str(zip_path)),
            api_name="/submit_predictions",
            **SUBMIT_KWARGS,
        )
        msg = result.get("value", "") if isinstance(result, dict) else str(result)
        ok = ("Submission received" in msg) or ("Predictions submitted" in msg) or ("Thank" in msg)
        return ok, msg
    except Exception as e:
        return False, f"EXCEPTION: {e}"


def record(log, zip_path: Path, note: str, msg: str):
    new = {
        "timestamp_utc": utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "track": TRACK,
        "submission_csv": zip_path.name,
        "sha256": sha256_of(zip_path),
        "api_response": f"{note} | api_msg: {msg[:300]}",
        "lb_score_if_returned": "",
    }
    out = pd.concat([log, pd.DataFrame([new])], ignore_index=True)
    out.to_csv(LOG_PATH, index=False)
    return out


def status(log):
    last = last_submit_time(log)
    wait = time_until_slot(log)
    if last is None:
        print("No prior structure submissions logged.")
    else:
        print(f"Last submission: {last.isoformat()}")
        if wait.total_seconds() > 0:
            mins = int(wait.total_seconds() / 60)
            print(f"Next slot opens in: {mins // 60}h {mins % 60}m")
        else:
            print("Slot is OPEN now.")

    submitted_files = (log["submission_csv"].dropna().astype(str).tolist()
                       if len(log) else [])
    print("\nLadder state:")
    for i, entry in enumerate(STRUCTURE_LADDER, 1):
        fn = entry["file"]
        mark = "[X] DONE" if fn in submitted_files else "[ ] TODO"
        print(f"  {mark}  {i}. {fn}")
        print(f"          {entry['note']}")

    nxt = next_candidate(log)
    if nxt:
        path, note, fn = nxt
        print(f"\nNext candidate: {fn}")
        print(f"  Note: {note}")
    else:
        print("\nLadder exhausted (all 4 zips submitted).")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "submit"
    log = load_log()
    if mode == "status":
        status(log)
        return

    # ===========================================================================
    # HARD-DISABLED 2026-06-12 (user directive): the STRUCTURE track is owned by
    # the SEPARATE OpenADMET-pxr-structure repo. This main-repo ladder must NEVER
    # submit to the structure track -- a submit here would burn a 1/4h API slot
    # the structure repo needs and interfere with the user's process. This repo's
    # only purpose is the compound pEC50 (Activity) track. Status-only from here.
    # ===========================================================================
    print("DISABLED: structure submissions are owned by the separate structure repo; refusing to submit. (Use 'status' to inspect.)")
    return

    wait = time_until_slot(log)
    if mode != "force" and wait.total_seconds() > 0:
        mins = int(wait.total_seconds() / 60)
        print(f"Rate limit: next slot opens in {mins // 60}h {mins % 60}m; skipping.")
        return

    nxt = next_candidate(log)
    if nxt is None:
        print("Structure ladder exhausted; nothing to submit.")
        return
    path, note, fn = nxt
    print(f"Submitting (Structure track): {fn}")
    print(f"  Note: {note}")
    ok, msg = submit_zip(path)
    print(f"  ok={ok}  api_msg={msg[:300]}")
    record(log, path, note, msg)


if __name__ == "__main__":
    main()
