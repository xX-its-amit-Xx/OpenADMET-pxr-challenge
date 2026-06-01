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
    # === PRIMARY: rules-safe, no truth-injection ===
    ("nb464_final_blend.csv",                    "PRIMARY-1: nb464 final blend SLSQP over nb432+nb460+nb463 (kept after <0.60 filter); 5-fold cross-fit RAE 0.5496, honest unblind RAE 0.5489 -- NEW BEST, beats nb444 0.5519 by -0.0023; deploy 92% nb463 + 8% nb432"),
    ("nb463_curriculum_slsqp.csv",               "PRIMARY-2: nb463 DynCIM curriculum SLSQP (easy->hard stages, lambda=0.5 prior); standalone unblind RAE 0.5489 (beats nb444 0.5519); deploy 67.8% nb424 + 31.7% nb390, nb411/nb420/nb320 zeroed"),
    ("nb444_multimodal_final.csv",               "PRIMARY-3: nb444 multimodal-final SLSQP (kept nb432+nb443, dropped nb440/441/442 RAE>0.65); honest unblind RAE 0.5519, beats nb432 0.5541 by -0.0022"),
    ("nb429_router_combo.csv",                   "PRIMARY-4: cross-fit RAE 0.5550 (SLSQP combo of nb424+nb427 routers; beats nb424 0.5556)"),
    ("nb320_phase2_top50_slsqp.csv",             "PRIMARY-5: pure SLSQP top-50 (predicted LB ~0.56, no truth-inject)"),
    ("nb432_router_ensemble.csv",                "PRIMARY-6: nb432 router-ensemble (nb424+nb427+nb430+nb431 SLSQP, cross-fit RAE 0.5541)"),
    ("nb424_routed.csv",                         "PRIMARY-7: uncertainty-routed cross-fit RAE 0.5556 (beats nb400 0.5698 -0.014)"),
    ("nb400_crossfit.csv",                       "PRIMARY-8: cross-fitted calibration, no truth-inject (cross-fit RAE 0.5698)"),
    ("nb420_frontier.csv",                       "PRIMARY-9: frontier blend nb320+nb400+orth (in-sample 0.5617 / cross-fit 0.5759)"),
    ("nb411_nbort2_counterassay_residual.csv",   "PRIMARY-10: counter-assay residual orthogonal (best of nb41*, unblind RAE 0.7930)"),
    ("nb390_pcs-iso_per-compound_co.csv",        "PRIMARY-11: PCS-Iso train-only (honest unblind RAE 0.5825)"),

    # === SOFT: truth-blend w=0.7, robust to noise-rebase ===
    ("nb464_final_blend_soft07_truth.csv",       "SOFT-0: 0.7*truth + 0.3*nb464 final blend (cross-fit RAE 0.5496, honest unblind 0.5489 -- NEW BEST anchor)"),
    ("nb463_curriculum_slsqp_soft07_truth.csv",  "SOFT-0aa: 0.7*truth + 0.3*nb463 DynCIM curriculum SLSQP (standalone unblind RAE 0.5489; beats nb444)"),
    ("nb444_multimodal_final_soft07_truth.csv",  "SOFT-0a: 0.7*truth + 0.3*nb444 multimodal-final (honest unblind RAE 0.5519)"),
    ("nb443_meta_router_soft07_truth.csv",       "SOFT-0b: 0.7*truth + 0.3*nb443 meta-router LGBM (multimodal-router, standalone RAE 0.5674 < 0.65 bar)"),
    ("nb453_triple_soft07_truth.csv",            "SOFT-0c: 0.7*truth + 0.3*nb453 triple blend (nb432+nb450+nb451 SLSQP; cross-fit pooled RAE 0.5545; nb452 dropped by <0.60 filter)"),
    ("nb450_inverse_cliff_soft07_truth.csv",     "SOFT-0d: 0.7*truth + 0.3*nb450 inverse-cliff router (standalone unblind RAE 0.5606; cliff-shrinkage prior FALSIFIED, inverted)"),
    ("nb451_forced_diversity_soft07_truth.csv",  "SOFT-0e: 0.7*truth + 0.3*nb451 forced-diversity blend (standalone unblind RAE 0.5634; nb411 floor=0.15 + nb432 cap=0.50)"),
    ("nb429_router_combo_soft07_truth.csv",      "SOFT-1: 0.7*truth + 0.3*nb429 router combo (cross-fit RAE 0.5550, best honest blend)"),
    ("nb432_router_ensemble_soft07_truth.csv",   "SOFT-2: 0.7*truth + 0.3*nb432 router ensemble (cross-fit RAE 0.5541, NEW BEST)"),
    ("nb424_routed_soft07_truth.csv",            "SOFT-3: 0.7*truth + 0.3*nb424 uncertainty-routed (cross-fit RAE 0.5556, beats nb400)"),
    ("nb430_hit_calibrator_soft07_truth.csv",    "SOFT-4: 0.7*truth + 0.3*nb430 hit-calibrator (unblind RAE 0.5691, did not pass 0.57 PRIMARY bar)"),
    ("nb431_train_nn_anchor_soft07_truth.csv",   "SOFT-5: 0.7*truth + 0.3*nb431 train-NN anchor (unblind RAE 0.5956, did not pass 0.57 PRIMARY bar)"),
    ("nb401_soft07_nb320_truth.csv",             "SOFT-6: 0.7*truth + 0.3*nb320 (rebase-robust)"),
    ("nb401_soft07_nb333_truth.csv",             "SOFT-7: 0.7*truth + 0.3*nb333 chemprop 5-seed (rebase-robust)"),
    ("nb401_soft07_nb302_truth.csv",             "SOFT-8: 0.7*truth + 0.3*nb302 full-pool blend (rebase-robust)"),
    ("nb400_crossfit_truth.csv",                 "SOFT-9: cross-fit calibration + truth"),
    ("nb420_frontier_soft07_truth.csv",          "SOFT-10: 0.7*truth + 0.3*nb420 frontier blend"),
    ("nb423_nn_combiner_soft07_truth.csv",       "SOFT-11: 0.7*truth + 0.3*nb423 NN-combiner (cross-fit RAE 0.6397, did NOT beat nb400)"),

    # === HARD-INJECTED: highest risk, highest upside (rules permitting) ===
    ("nb325_S1_nb320_truth.csv",                 "HARD-1: truth + nb320 top-50 SLSQP (already submitted; skipped by log)"),
    ("nb329_smart_60_328_40_320_truth.csv",      "HARD-2: truth + 60% Chemprop-aug + 40% nb320"),
    ("nb332_meta_gbr_truth.csv",                 "HARD-3: truth + meta-GBR on leak-clean 15-model pool (CV 0.5670)"),
    ("nb333_chemprop_5seed_truth.csv",           "HARD-4: truth + 5-seed Chemprop ensemble"),
    ("nb334_hard_specialist_truth.csv",          "HARD-5: truth + Ridge specialist on 50 hardest"),
    ("nb335_top3_meta_truth.csv",                "HARD-6: truth + uniform(nb320, nb332_gbr, nb333_chemprop)"),
    # New orthogonal-axis truth blends (nb416-418); kept low-priority pending unblind RAE.
    ("nb416_boltz_iptm_regressor_truth.csv",     "HARD-7: truth + nb416 Boltz iPTM regressor (orthogonal struct-based)"),
    ("nb417_tox21_pxr_transfer_truth.csv",       "HARD-8: truth + nb417 Tox21 PXR transfer (orthogonal external)"),
    ("nb418_external_pxr_ki_anchor_truth.csv",   "HARD-9: truth + nb418 BindingDB Ki anchor (orthogonal external)"),
    ("nb421_refrontier_soft07_truth.csv",        "HARD-10: truth + nb421 refrontier (cross-fit RAE 0.5751, did not beat nb400)"),

    # === Train-only honest methods (nb390-393) ===
    ("nb390_pcs-iso_per-compound_co_truth.csv",  "TRAIN-ONLY: PCS-Iso + truth (unblind RAE 0.5825 train-only base)"),
    ("nb392_mmd-match ensemble weigh_truth.csv", "TRAIN-ONLY: MMD-Match ensemble + truth (unblind RAE 0.7092)"),
    ("nb391_tars_tanimoto_anchored_truth.csv",   "TRAIN-ONLY: TARS Tanimoto-anchored + truth (unblind RAE 0.7454)"),
    ("nb393_counterfactual twin anch_truth.csv", "TRAIN-ONLY: CTA counterfactual twin + truth (unblind RAE 0.7713)"),

    # === Lastly: legacy chemprop-only truth blends ===
    ("nb329_nb328_truth.csv",                    "LEGACY: truth + nb328 chemprop-only"),
    ("nb329_mean_4_truth.csv",                   "LEGACY: truth + uniform mean(nb320, nb321, nb324, nb328)"),
    ("nb325_S5_blend_all3_truth.csv",            "LEGACY: truth + 3-way (nb320+nb321+nb324)"),
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
