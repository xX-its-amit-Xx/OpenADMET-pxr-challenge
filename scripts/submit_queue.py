"""Submission queue manager for the OpenADMET PXR leaderboard.

Maintains a queue of candidate submissions. Each invocation:
1. Checks if the 4-hour rate limit has elapsed.
2. Selects the next-best UNSUBMITTED candidate (ratio >= 0.58, lowest OOF RAE).
3. If nothing fresh, generates a perturbation of the current best by SLSQP
   over (current best + nb221 + any new models from data/processed/).
4. Submits via the gradio API.
5. Logs the result to data/processed/leaderboard_log.csv.

Run modes:
  python submit_queue.py status     # report next slot + queue
  python submit_queue.py submit     # try to submit; respects rate limit
  python submit_queue.py force      # submit even if too early (manual override)
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

LOG_PATH = DATA_PROCESSED / "leaderboard_log.csv"
RATE_LIMIT_HOURS = 4
COLLAPSE_THRESH = 0.58

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
    return pd.DataFrame(columns=["submitted_at","model_stem","local_oof_rae",
                                  "local_ratio","leaderboard_rae","leaderboard_rank","note"])


def last_submission_time(log):
    if len(log) == 0: return None
    parsed = pd.to_datetime(log["submitted_at"], utc=True, errors="coerce")
    return parsed.max()


def time_until_next_slot(log):
    last = last_submission_time(log)
    if last is None: return timedelta(0)
    next_slot = last + timedelta(hours=RATE_LIMIT_HOURS)
    return max(timedelta(0), next_slot - utcnow())


def candidate_score(stem, y):
    """Returns (rae, ratio, csv_path) or None if unavailable."""
    op = DATA_PROCESSED / f"oof_{stem}.npy"
    tp = DATA_PROCESSED / f"te_{stem}.npy"
    csv = SUBMISSIONS / f"{stem}.csv"
    if not (op.exists() and tp.exists() and csv.exists()):
        return None
    oof = np.load(op).flatten().astype(np.float64)
    te  = np.load(tp).flatten().astype(np.float64)
    if len(oof) != len(y) or len(te) != 513:
        return None
    return rae(y, oof), te.std() / oof.std(), csv


def build_queue(log, y, max_corr=0.999):
    """Find candidates: stems with oof_/te_ files NOT in submission log,
    ratio >= 0.58, AND not nearly identical to an already-submitted model
    (OOF correlation < max_corr). Sorted by RAE ascending."""
    submitted_stems = set(log["model_stem"].dropna().astype(str).tolist())
    # Load OOFs of all submitted models for de-duplication
    submitted_oofs = {}
    for s in submitted_stems:
        op = DATA_PROCESSED / f"oof_{s}.npy"
        if op.exists():
            submitted_oofs[s] = np.load(op).flatten().astype(np.float64)

    candidates = []
    for op in sorted(DATA_PROCESSED.glob("oof_nb*.npy")):
        stem = op.stem[4:]
        if stem in submitted_stems:
            continue
        score = candidate_score(stem, y)
        if score is None:
            continue
        r, ratio, csv = score
        if ratio < COLLAPSE_THRESH: continue
        # Check duplicate via OOF correlation with any already-submitted
        oof = np.load(op).flatten().astype(np.float64)
        if any(np.corrcoef(oof, prev)[0, 1] > max_corr for prev in submitted_oofs.values()):
            continue  # too similar to something already submitted
        candidates.append({"stem": stem, "rae": r, "ratio": ratio, "csv": csv})
    return sorted(candidates, key=lambda c: c["rae"])


def submit_csv(csv_path):
    """Submit a CSV to the leaderboard. Returns (success, message)."""
    try:
        from gradio_client import Client, handle_file
        client = Client("https://openadmet-pxr-challenge.hf.space/")
        result = client.predict(
            file_input=handle_file(str(csv_path)),
            api_name="/submit_predictions",
            **SUBMIT_KWARGS,
        )
        msg = result.get("value", "") if isinstance(result, dict) else str(result)
        ok = "Submission received" in msg
        return ok, msg
    except Exception as e:
        return False, f"EXCEPTION: {e}"


def record_submission(log, stem, csv_path, y, note=""):
    score = candidate_score(stem, y)
    if score is None:
        rae_v, ratio_v = None, None
    else:
        rae_v, ratio_v = round(score[0], 6), round(score[1], 4)
    new_row = {
        "submitted_at": utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_stem": stem,
        "local_oof_rae": rae_v,
        "local_ratio": ratio_v,
        "leaderboard_rae": None,
        "leaderboard_rank": None,
        "note": note,
    }
    log2 = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)
    log2.to_csv(LOG_PATH, index=False)
    return log2


def perturb_blend(log, y, te_df):
    """If no fresh candidate is available, generate a perturbed variant of the best.

    Method: take the most recently submitted model's OOF/test, blend with
    nb221_singleconc_aux at small weight to create a slightly different submission.
    This produces a 'variant' that's near the current best but distinct enough
    to be informative on the leaderboard.
    """
    if len(log) == 0: return None
    # Use the model with the BEST local_oof_rae among submitted as the base
    submitted = log.dropna(subset=["local_oof_rae"])
    if len(submitted) == 0: return None
    best_row = submitted.sort_values("local_oof_rae").iloc[0]
    base_stem = best_row["model_stem"]
    base_oof = np.load(DATA_PROCESSED / f"oof_{base_stem}.npy").flatten().astype(np.float64)
    base_te  = np.load(DATA_PROCESSED / f"te_{base_stem}.npy").flatten().astype(np.float64)

    perturbations = []
    # Add small weight on nb221 (high ratio, RAE 0.45)
    nb221_op = DATA_PROCESSED / "oof_nb221_singleconc_aux.npy"
    nb221_tp = DATA_PROCESSED / "te_nb221_singleconc_aux.npy"
    if nb221_op.exists() and nb221_tp.exists():
        o221 = np.load(nb221_op).flatten().astype(np.float64)
        t221 = np.load(nb221_tp).flatten().astype(np.float64)
        for w in [0.02, 0.03, 0.05, 0.08]:
            oof_p = (1 - w) * base_oof + w * o221
            te_p  = (1 - w) * base_te  + w * t221
            r = rae(y, oof_p)
            ratio = te_p.std() / oof_p.std()
            if ratio >= COLLAPSE_THRESH:
                perturbations.append((r, ratio, w, oof_p, te_p, "nb221"))

    if not perturbations: return None
    # Pick the one with lowest RAE that passes
    perturbations.sort(key=lambda x: x[0])
    r, ratio, w, oof_p, te_p, mix = perturbations[0]
    # Build CSV
    stem = f"perturb_{base_stem}_{mix}_w{w:.2f}"
    csv_path = SUBMISSIONS / f"{stem}.csv"
    sub = pd.DataFrame({
        "SMILES": te_df["smiles"].values,
        "Molecule Name": te_df["name"].values,
        "pEC50": te_p,
    })
    sub.to_csv(csv_path, index=False)
    np.save(DATA_PROCESSED / f"oof_{stem}.npy", oof_p)
    np.save(DATA_PROCESSED / f"te_{stem}.npy",  te_p)
    return {"stem": stem, "rae": r, "ratio": ratio, "csv": csv_path,
            "note": f"perturb of {base_stem} + {w:.2f}*{mix}"}


def status(log, y):
    last = last_submission_time(log)
    until = time_until_next_slot(log)
    print(f"Last submission: {last}")
    print(f"Next slot in: {until}  (current UTC: {utcnow()})")
    print(f"Submissions logged: {len(log)}")
    if len(log):
        print("Most recent:")
        print(log.tail(3).to_string(index=False))
    print("\nQueue (unsubmitted candidates with ratio >= 0.58):")
    q = build_queue(log, y)
    if q:
        for c in q[:10]:
            print(f"  {c['stem']}  RAE={c['rae']:.6f}  ratio={c['ratio']:.4f}")
    else:
        print("  (none — would need perturbation)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    tr = load_train(); te_df = load_test()
    y = tr["pec50"].values.astype(np.float64)
    log = load_log()

    if mode == "status":
        status(log, y); return

    until = time_until_next_slot(log)
    if mode != "force" and until.total_seconds() > 0:
        mins = until.total_seconds() / 60
        print(f"Too early: {mins:.1f} min until next slot. Use 'force' to override.")
        return

    queue = build_queue(log, y)
    chosen = None
    if queue:
        chosen = queue[0]
        chosen["note"] = "fresh-queue"
    else:
        print("No fresh candidate. Generating perturbation...")
        chosen = perturb_blend(log, y, te_df)
    if chosen is None:
        print("Could not select or generate a candidate.")
        return

    print(f"Submitting: {chosen['stem']}  RAE={chosen['rae']:.6f}  "
          f"ratio={chosen['ratio']:.4f}")
    ok, msg = submit_csv(chosen["csv"])
    print(f"Result: {msg}")
    log = record_submission(log, chosen["stem"], chosen["csv"], y, note=chosen.get("note", ""))
    print(f"Logged. Total submissions: {len(log)}")


if __name__ == "__main__":
    main()
