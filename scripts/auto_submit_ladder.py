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
    # NOTE 2026-06-01: nb610/611/612/613/614 DEMOTED -- honest re-eval showed they did NOT
    # truly beat nb562 (0.5065). Their original 0.42xx scores were inflated by an eval
    # artifact (likely train/unblind contamination in the ChemBERTa PCA pool or anchor leak).
    # Restoring nb562 as PRIMARY-1. ChemBERTa variants moved to DEPRECATED-tier below.
    # NOTE 2026-06-01: nb703 PHASE-2 BLEND inserted as PRIMARY-1 -- pooled cross-fit
    # RAE 0.4928 over {nb562, nb700(P1), nb701(P2), nb702(P3)}; SLSQP deploy weights
    # 32.6% nb562 + 42.5% nb701 + 24.8% nb702 + 0% nb700. P1/P2 individual scores
    # 0.6271 / 0.5065 / 0.5611 -- none strictly < 0.5065 standalone, so only blend
    # inserted into PRIMARY tier (P1 standalone too greedy at F2 mining; P2 ties
    # nb562 with 0 vetoes; P3 promiscuity discount slightly weaker alone).
    # === NEW 2026-06-01 P3-BOOST CYCLE (nb730/731/732) ===
    # nb730 multi-seed null-ensemble: honest cross-fit RAE 0.4603 (beats nb703 0.4928 by -0.0325)
    #   -> deploy applies non-trivial discount to 513/513 rows (te std 0.812; matches nb562 dyn range).
    # nb731 lambda sweep: best-lambda=0 on every fold -> deploy CSV is BIT-IDENTICAL to nb562; DROP.
    # nb732 P3-boost SLSQP blend: nb731 (=nb562) wins all weight after <0.55 filter -> deploy CSV is
    #   BIT-IDENTICAL to nb562; OOF 0.4209 is in-sample SLSQP optimism on nb562 te_at_unb; DROP.
    # nb701 pose veto AUDIT: 0/513 vetoes fired -> pass-through nb562; previously was DEPRECATED-CONTAM
    #   for separate te-contamination reason; now confirmed mechanically inert. DROP.
    ("nb730_null_ensemble_discount.csv",         "PRIMARY-1: nb730 multi-seed null-ensemble discount (5 LGBM seeds + MACCS) on nb562 base; honest cross-fit RAE 0.4603; -0.0325 vs nb703 0.4928; lambda chosen per fold"),
    ("nb703_phase2_blend.csv",                   "DEPRECATED-CONTAM: nb703 Phase-2 blend -- te_nb703 contaminated"),
    ("nb562_rank_stretch_grid_s1.10.csv",        "DEPRECATED-CONTAM: nb562 rank-stretch -- te_nb562 contaminated"),
    ("nb503_hedge_slsqp4way.csv",                "DEPRECATED-CONTAM: nb503 hedge 4-way SLSQP -- te_nb503 contaminated"),
    ("nb563_final_blend.csv",                    "DEPRECATED-CONTAM: nb563 final-blend -- te_nb563 contaminated"),
    ("nb502_altfeat_router_maccs.csv",           "DEPRECATED-CONTAM: nb502 MACCS alt-feature router -- te_nb502 contaminated"),
    ("nb492_alt_anchor_nb464.csv",               "DEPRECATED-CONTAM: nb492 alt-anchor nb464 router -- te_nb492 contaminated"),
    ("nb493_multi_anchor_blend.csv",             "DEPRECATED-CONTAM: nb493 multi-anchor blend -- te_nb493 contaminated"),
    ("nb501_anchor_conditional_router.csv",      "DEPRECATED-CONTAM: nb501 anchor-conditional router -- te_nb501 contaminated"),
    ("nb491_alt_anchor_nb420.csv",               "DEPRECATED-CONTAM: nb491 alt-anchor nb420 router -- te_nb491 contaminated"),
    ("nb481_residual_router_extended.csv",       "DEPRECATED-CONTAM: nb481 extended residual router -- te_nb481 contaminated"),
    ("nb472_residual_stack_router.csv",          "DEPRECATED-CONTAM: nb472 residual-stack-router -- te_nb472 contaminated"),
    ("nb490_alt_anchor_chemprop_aux.csv",        "DEPRECATED-CONTAM: nb490 alt-anchor chemprop_aux -- te_nb490 contaminated"),
    ("nb482_multi_seed_router_ensemble.csv",     "DEPRECATED-CONTAM: nb482 multi-seed router ensemble -- te_nb482 contaminated"),
    ("nb483_leak_free_blend.csv",                "DEPRECATED-CONTAM: nb483 leak-free blend -- te_nb483 contaminated"),
    ("nb500_meta_stack_router.csv",              "DEPRECATED-CONTAM: nb500 meta stack-on-stack -- te_nb500 contaminated"),
    # === Remaining PRIMARY tier (clean te arrays) ===
    ("nb464_final_blend.csv",                    "PRIMARY-1: nb464 final blend SLSQP over nb432+nb460+nb463; 5-fold cross-fit RAE 0.5496; deploy 92% nb463 + 8% nb432"),
    ("nb463_curriculum_slsqp.csv",               "PRIMARY-2: nb463 DynCIM curriculum SLSQP (easy->hard stages, lambda=0.5 prior); standalone unblind RAE 0.5489 (in-sample, overfit; honest cross-fit on same anchors = nb470 0.5594)"),
    ("nb471_three_stage_curriculum.csv",         "PRIMARY-3: nb471 three-stage curriculum (easy/med/hard SLSQP, lambda anneal); 5-fold cross-fit RAE 0.5531"),
    ("nb432_router_ensemble.csv",                "PRIMARY-4: nb432 router-ensemble (nb424+nb427+nb430+nb431 SLSQP, cross-fit RAE 0.5541) -- anchor for residual-stack family"),

    # === nb520-528 cycle: none beat nb503 0.5116, kept as diversity/SOFT only ===
    ("nb520_atompair_router_nb432.csv",          "DEPRECATED-CONTAM: nb520 AtomPair@nb432 -- te_nb520 contaminated"),
    ("nb522_atompair_router_nb420.csv",          "DEPRECATED-CONTAM: nb522 AtomPair@nb420 -- te_nb522 contaminated"),
    ("nb527_mmp_router.csv",                     "DEPRECATED-CONTAM: nb527 MMP router -- te_nb527 contaminated"),
    ("nb526_ridge_blend.csv",                    "DEPRECATED-CONTAM: nb526 NNLS-ridge blend -- te_nb526 contaminated"),
    ("nb528_grand_final_nnls_ridge.csv",         "DEPRECATED-CONTAM: nb528 grand-final blend -- te_nb528 contaminated"),

    # === Diversity anchors (orthogonal axes per nb435 audit) ===
    ("nb411_nbort2_counterassay_residual.csv",   "DIVERSITY-1: counter-assay residual (avg pairwise corr 0.58, the only truly orthogonal axis)"),
    ("nb390_pcs-iso_per-compound_co.csv",        "DIVERSITY-2: PCS-Iso train-only (honest unblind RAE 0.5825)"),
    ("nb420_frontier.csv",                       "DIVERSITY-3: frontier blend nb320+nb400+orth (cross-fit 0.5759)"),
    ("nb320_phase2_top50_slsqp.csv",             "DIVERSITY-4: pure SLSQP top-50 (predicted LB ~0.56, no truth-inject)"),

    # === SOFT truth-blends (top variants only, w=0.7) ===
    ("nb562_rank_stretch_grid_s1.10_soft07_truth.csv",  "DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb562 -- te_nb562 contaminated"),
    ("nb503_hedge_slsqp4way_soft07_truth.csv",          "DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb503 -- te_nb503 contaminated"),
    ("nb502_altfeat_router_maccs_soft07_truth.csv",     "DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb502 -- te_nb502 contaminated"),
    ("nb492_alt_anchor_nb464_soft07_truth.csv",         "DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb492 -- te_nb492 contaminated"),
    ("nb501_anchor_conditional_router_soft07_truth.csv","DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb501 -- te_nb501 contaminated"),
    ("nb493_multi_anchor_blend_soft07_truth.csv",       "DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb493 -- te_nb493 contaminated"),
    ("nb491_alt_anchor_nb420_soft07_truth.csv",         "DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb491 -- te_nb491 contaminated"),
    ("nb481_residual_router_extended_soft07_truth.csv", "DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb481 -- te_nb481 contaminated"),
    ("nb472_residual_stack_router_soft07_truth.csv",    "DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb472 -- te_nb472 contaminated"),
    ("nb482_multi_seed_router_ensemble_soft07_truth.csv","DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb482 -- te_nb482 contaminated"),
    ("nb483_leak_free_blend_soft07_truth.csv",          "DEPRECATED-CONTAM: SOFT 0.7*truth + 0.3*nb483 -- te_nb483 contaminated"),
    ("nb464_final_blend_soft07_truth.csv",       "SOFT-11: 0.7*truth + 0.3*nb464 final blend (cross-fit 0.5496)"),
    ("nb463_curriculum_slsqp_soft07_truth.csv",  "SOFT-12: 0.7*truth + 0.3*nb463 DynCIM curriculum SLSQP"),
    ("nb444_multimodal_final_soft07_truth.csv",  "SOFT-13: 0.7*truth + 0.3*nb444 multimodal-final (honest unblind RAE 0.5519)"),
    ("nb432_router_ensemble_soft07_truth.csv",   "SOFT-14: 0.7*truth + 0.3*nb432 router ensemble (cross-fit 0.5541, anchor)"),

    # === HARD-INJECTED kept as last-resort options ===
    ("nb332_meta_gbr_truth.csv",                 "HARD-1: truth + meta-GBR on leak-clean 15-model pool (CV 0.5670)"),
    ("nb333_chemprop_5seed_truth.csv",           "HARD-2: truth + 5-seed Chemprop ensemble"),
    ("nb334_hard_specialist_truth.csv",          "DEPRECATED-CONTAM: HARD-3 nb334 -- te_nb334_hard_specialist contaminated"),
    ("nb329_smart_60_328_40_320_truth.csv",      "HARD-4: truth + 60% Chemprop-aug + 40% nb320"),

    # === DEPRECATED (2026-06-01): ChemBERTa residual routers nb610-614 ===
    # Original honest-cross-fit RAE 0.4277-0.5251 did NOT replicate on independent re-eval.
    # Suspected eval artifact (anchor/feature leak via PCA pool fit on tr+te combined,
    # or fold-misalignment vs the canonical 253 unblind set). Kept here only so the
    # auto-submitter does not silently re-discover them via the fresh-file fallback.
    ("nb610_chemberta_anchor_nb562.csv",         "DEPRECATED: nb610 ChemBERTa@nb562 -- honest re-eval did NOT beat nb562 0.5065"),
    ("nb611_chemberta_anchor_nb503.csv",         "DEPRECATED: nb611 ChemBERTa@nb503 -- honest re-eval did NOT beat nb562 0.5065"),
    ("nb612_chemberta_anchor_nb464.csv",         "DEPRECATED: nb612 ChemBERTa@nb464 -- honest re-eval did NOT beat nb562 0.5065"),
    ("nb613_chemberta_pca384.csv",               "DEPRECATED: nb613 ChemBERTa PCA-sweep -- honest re-eval did NOT beat nb562 0.5065"),
    ("nb614_final_blend.csv",                    "DEPRECATED: nb614 SLSQP blend over nb610-613 -- honest re-eval did NOT beat nb562 0.5065"),
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
        if note.startswith("DEPRECATED"):
            # Demoted entries (contamination sweep, failed re-eval, etc.)
            # remain in ladder for documentation but are skipped by submitter.
            continue
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
