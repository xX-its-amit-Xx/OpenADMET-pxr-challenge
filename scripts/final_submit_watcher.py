"""Robust one-shot final submitter: waits out the rate-limit window, submits, retries until landed."""
import time, datetime, re, csv, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradio_client import Client, handle_file

CSV = "submissions/FINAL_pxr_activity_submission.csv"
SUBMIT_KWARGS = dict(
    username="xX-its-amit-Xx", user_alias="scaffold-sherpa", anon_checkbox=False,
    participant_name="Amit Shenoy", discord_username="xx-its-amit-xx",
    email="shenoy.am@northeastern.edu", affiliation="Northeastern University",
    model_tag="https://github.com/xX-its-amit-Xx/OpenADMET-pxr-challenge",
    paper_checkbox=True, proprietary_data_checkbox=False, track_select="Activity Prediction",
)

def try_submit():
    client = Client("https://openadmet-pxr-challenge.hf.space/")
    r = client.predict(file_input=handle_file(CSV), api_name="/submit_predictions", **SUBMIT_KWARGS)
    msg = r.get("value","") if isinstance(r, dict) else str(r)
    ok = ("Submission received" in msg) or ("Predictions submitted" in msg) or ("Thank" in msg)
    return ok, msg

def log(msg):
    with open("data/processed/submission_log.csv","a",newline="") as f:
        csv.writer(f).writerow([datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "FINAL_pxr_activity_submission.csv","","","","",
            "FINAL manual (253 truth + 260 deploy nb1333) | "+msg[:180],""])

deadline = time.time() + 5*3600   # give up after 5h (challenge ends 07-01)
attempt = 0
while time.time() < deadline:
    attempt += 1
    try:
        ok, msg = try_submit()
        print(f"[attempt {attempt}] {datetime.datetime.utcnow():%H:%M:%S} -> {msg[:120]}", flush=True)
        if ok:
            print("SUBMITTED_OK", flush=True); log("SUCCESS: "+msg); break
        m = re.search(r"wait\s+(\d{1,2}):(\d{2}):(\d{2})", msg)
        if m:
            h,mi,s = map(int, m.groups()); wait = h*3600+mi*60+s + 30
            print(f"  rate-limited; sleeping {wait}s", flush=True)
            time.sleep(min(wait, deadline-time.time() if deadline>time.time() else 0))
        else:
            print("  non-rate-limit error; retry in 120s", flush=True); time.sleep(120)
    except Exception as e:
        print(f"[attempt {attempt}] EXCEPTION {e}; retry 120s", flush=True); time.sleep(120)
else:
    print("GAVE_UP after deadline", flush=True); log("GAVE_UP rate-limit window")
