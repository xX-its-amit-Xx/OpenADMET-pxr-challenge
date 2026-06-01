"""submit_structure_v4.py -- fire structure-track submission immediately.

Submits submissions/structure_baseline_v4.zip to the Gradio Space using the
same API call pattern as auto_submit_ladder.py, but with track_select set to
"Structure Prediction" and the zip handed off via gradio_client.handle_file.

Records to data/processed/structure_submission_log.csv:
  timestamp_utc, track, submission_csv, sha256, api_response, lb_score_if_returned
"""
import os, sys, hashlib, re
os.environ["PYTHONIOENCODING"] = "utf-8"

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

LOG_PATH = DATA_PROCESSED / "structure_submission_log.csv"

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
    track_select="Structure Prediction",
)


def utcnow_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def submit_zip(zip_path: Path):
    from gradio_client import Client, handle_file
    client = Client("https://openadmet-pxr-challenge.hf.space/")
    try:
        result = client.predict(
            file_input=handle_file(str(zip_path)),
            api_name="/submit_predictions",
            **SUBMIT_KWARGS,
        )
    except Exception as e:
        return False, f"EXCEPTION: {e}", None
    msg = result.get("value", "") if isinstance(result, dict) else str(result)
    ok = ("Submission received" in msg) or ("Predictions submitted" in msg) or ("Thank" in msg) or ("LDDT" in msg)
    # Try to scrape a score from the response
    score = None
    m = re.search(r"(?:LDDT[- _]PLI|score)[^0-9\-]*([0-9]*\.?[0-9]+)", msg, re.IGNORECASE)
    if m:
        try: score = float(m.group(1))
        except: score = None
    return ok, msg, score


def append_log(zip_path: Path, msg: str, score, sha: str):
    row = {
        "timestamp_utc": utcnow_str(),
        "track": "Structure Prediction",
        "submission_csv": zip_path.name,  # column name from spec; here it's the zip
        "sha256": sha,
        "api_response": msg[:1000],
        "lb_score_if_returned": score,
    }
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log.to_csv(LOG_PATH, index=False)


def main():
    primary = SUBMISSIONS / "structure_baseline_v4.zip"
    fallback = SUBMISSIONS / "structure_baseline_v1.zip"

    target = primary if primary.exists() else fallback
    size_mb = target.stat().st_size / (1024 * 1024)
    sha = sha256_of(target)
    print(f"Submitting: {target.name}  size={size_mb:.2f} MB  sha256={sha[:16]}...")

    ok, msg, score = submit_zip(target)
    print(f"  ok={ok}")
    print(f"  api_msg={msg[:500]}")
    print(f"  score_parsed={score}")

    # Fallback to v1 if v4 fails for structural reason
    if not ok and target.name == "structure_baseline_v4.zip" and fallback.exists():
        bad_signs = ("InvalidZip", "validation", "invalid", "format", "missing", "184")
        if any(s.lower() in msg.lower() for s in bad_signs):
            print("v4 failed for format reason; falling back to v1.")
            append_log(target, f"[v4 FAILED] {msg}", score, sha)
            sha1 = sha256_of(fallback)
            ok, msg, score = submit_zip(fallback)
            print(f"  [v1] ok={ok}")
            print(f"  [v1] api_msg={msg[:500]}")
            append_log(fallback, msg, score, sha1)
            return
    append_log(target, msg, score, sha)


if __name__ == "__main__":
    main()
