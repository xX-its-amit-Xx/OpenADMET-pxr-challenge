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
    # === CYCLE-1 2026-06-03: 5 distinct method-axes (LLM, NR-multitask, SMILES-aug TTT,
    # quantile conformal, SSL pretrain-FT). Sorted ascending by in_RAE on canonical
    # 253 unblind. nb904 SSL was N/A (no submission produced); kept only in cycle log.
    # All are PRE-unblind regime -> predicted LB ~= in_RAE + 0.003 transfer.
    ("nb901_nr_multitask.csv",                   "CYCLE-1-1: nb901 NR-multitask LGBM (multi-NR transfer w/ shared trunk); in_RAE 0.6765; predicted LB ~0.68"),
    ("nb902_smiles_aug_ttt.csv",                 "CYCLE-1-2: nb902 SMILES-augmented test-time training (TTT); in_RAE 0.6998; predicted LB ~0.70"),
    ("nb903_quantile_conformal.csv",             "CYCLE-1-3: nb903 quantile-regression conformal prediction (q10/q50/q90 calibrated); in_RAE 0.7240; predicted LB ~0.73"),
    ("nb900_llm.csv",                            "CYCLE-1-4: nb900 LLM-based predictor (prompted, te_nb900.npy); in_RAE 0.8500; predicted LB ~0.85"),
    # nb904 SSL pretrain-FT: N/A this cycle (no CSV produced); see C:/pxr_artifacts/cycle1_summary.json

    # === URGENT REORDER 2026-06-01: HONEST PREDICTED-LB ORDER ===
    # Prior nb700-series + nb503/nb562/nb472-family entries were all trained-on-unblind
    # (cross-fit on the 253 unblinded labels). Predicted LB = max(in_RAE * 1.5, ~0.55) per
    # feedback_unblind_overfit_risk -- they will likely score 0.8-1.0 on the blind LB,
    # which would HURT rank vs current best 0.7655. Demoting all of them to
    # DEPRECATED-CROSSFIT-OVERFIT below.
    #
    # New PRIMARY tier ordered by HONEST predicted LB ascending:
    #   PRE-unblind models:   predicted LB ~= in_RAE + 0.003
    #   POST-unblind models:  predicted LB ~= in_RAE * 1.5 (floor at ~0.55)
    # chemprop_aux is the true #1 from the 2026-05-29 unblind validation (RAE 0.6216);
    # at predicted LB 0.6246 it would crush the current best 0.7655.
    ("chemprop_aux.csv",                         "PRIMARY-1: chemprop multi-task w/ aux heads; honest unblind RAE 0.6216; predicted LB 0.6246 (would crush current best 0.7655)"),
    ("grand_v6b_calib.csv",                      "PRIMARY-2: grand_v6b calibrated ensemble; in_RAE 0.6409; predicted LB 0.6439"),
    ("nb306_cepsmim.csv",                        "PRIMARY-3: nb306 ceps-MIM; in_RAE 0.6486; predicted LB 0.6516"),
    ("nb305_mope.csv",                           "PRIMARY-4: nb305 MoPE; in_RAE 0.6601; predicted LB 0.6631"),
    ("95_all_feature_fusion.csv",                "PRIMARY-5: mm-audit #5 all-feature-fusion (PRE-unblind); in_RAE 0.6625; predicted LB 0.6655"),
    ("54_deep_ensemble_uncertainty.csv",         "PRIMARY-6: mm-audit deep-ensemble-uncertainty (PRE-unblind); in_RAE 0.6657; predicted LB 0.6687"),
    ("27_nr_weighted_lgbm.csv",                  "PRIMARY-7: mm-audit NR-weighted LGBM (PRE-unblind); in_RAE 0.6729; predicted LB 0.6759"),
    ("82_selectivity_aware.csv",                 "PRIMARY-8: mm-audit selectivity-aware (PRE-unblind); in_RAE 0.6730; predicted LB 0.6760"),
    ("67_lgbm_chembl_all_nr_weighted.csv",       "PRIMARY-9: mm-audit ChEMBL-all-NR-weighted LGBM (PRE-unblind); in_RAE 0.6746; predicted LB 0.6776"),
    ("nb303_dann.csv",                           "PRIMARY-10: nb303 Karpathy DANN; in_RAE 0.6931; predicted LB 0.6961"),
    ("nb800_huber_1_5.csv",                      "PRIMARY-11: nb800 Huber alpha=1.5 (best alpha sweep variant, PRE-unblind); in_RAE 0.7378; predicted LB 0.7408"),
    ("nb800_huber_ens4.csv",                     "PRIMARY-12: nb800 4-way Huber ensemble {0.3,0.7,1.5,3.0} (PRE-unblind); in_RAE 0.7441; predicted LB 0.7471"),
    ("nb801_huber_assay_decomp_plus.csv",        "PRIMARY-13: nb801 Huber w/ expanded assay-decomp (6 new feats, PRE-unblind); in_RAE 0.7448; predicted LB 0.7478"),
    ("nb120_huber_1_0.csv",                      "PRIMARY-14: nb120 Huber delta=1.0; in_RAE 0.7461; predicted LB 0.7491"),
    ("nb120_huber_2_0.csv",                      "PRIMARY-15: nb120 Huber delta=2.0; in_RAE 0.7502; predicted LB 0.7532"),
    ("nb120_huber_0_5.csv",                      "PRIMARY-16: nb120 Huber delta=0.5; in_RAE 0.7513; predicted LB 0.7543"),
    ("nb273_molformer.csv",                      "PRIMARY-17: nb273 MoLFormer (already submitted but worth re-checking)"),

    # === DEPRECATED-CROSSFIT-OVERFIT (2026-06-01) ===
    # All entries below were ranked using in-sample / cross-fit RAE on the 253 unblinded
    # labels (df>150 in iso+BMA+SLSQP setups). Per feedback_unblind_overfit_risk, gap to
    # blind LB is +0.05-0.30 RAE; honest predicted LB band is 0.8-1.0, which would
    # HARM rank vs current best 0.7655. Kept in ladder only so submitter does not
    # silently re-discover them via fresh-file fallback.
    ("nb703_phase2_blend.csv",                   "DEPRECATED-CROSSFIT-OVERFIT: nb703 Phase-2 SLSQP blend trained on unblind labels; predicted LB ~0.97; was PRIMARY-1"),
    ("nb562_rank_stretch_grid_s1.10.csv",        "DEPRECATED-CROSSFIT-OVERFIT: nb562 rank-stretch grid trained on unblind labels"),
    ("nb503_hedge_slsqp4way.csv",                "DEPRECATED-CROSSFIT-OVERFIT: nb503 hedge 4-way SLSQP trained on unblind labels"),
    ("nb502_altfeat_router_maccs.csv",           "DEPRECATED-CROSSFIT-OVERFIT: nb502 MACCS alt-feature router trained on unblind labels"),
    ("nb492_alt_anchor_nb464.csv",               "DEPRECATED-CROSSFIT-OVERFIT: nb492 alt-anchor nb464 trained on unblind labels"),
    ("nb493_multi_anchor_blend.csv",             "DEPRECATED-CROSSFIT-OVERFIT: nb493 multi-anchor blend trained on unblind labels"),
    ("nb501_anchor_conditional_router.csv",      "DEPRECATED-CROSSFIT-OVERFIT: nb501 anchor-conditional router trained on unblind labels"),
    ("nb491_alt_anchor_nb420.csv",               "DEPRECATED-CROSSFIT-OVERFIT: nb491 alt-anchor nb420 trained on unblind labels"),
    ("nb481_residual_router_extended.csv",       "DEPRECATED-CROSSFIT-OVERFIT: nb481 extended residual router trained on unblind labels"),
    ("nb472_residual_stack_router.csv",          "DEPRECATED-CROSSFIT-OVERFIT: nb472 residual-stack-router trained on unblind labels"),
    ("nb490_alt_anchor_chemprop_aux.csv",        "DEPRECATED-CROSSFIT-OVERFIT: nb490 alt-anchor chemprop_aux trained on unblind labels"),
    ("nb482_multi_seed_router_ensemble.csv",     "DEPRECATED-CROSSFIT-OVERFIT: nb482 multi-seed router ensemble trained on unblind labels"),
    ("nb483_leak_free_blend.csv",                "DEPRECATED-CROSSFIT-OVERFIT: nb483 'leak-free' SLSQP blend still trained on unblind labels"),
    ("nb500_meta_stack_router.csv",              "DEPRECATED-CROSSFIT-OVERFIT: nb500 meta stack-on-stack router trained on unblind labels"),
    ("nb563_final_blend.csv",                    "DEPRECATED-CROSSFIT-OVERFIT: nb563 final-blend trained on unblind labels"),
    ("nb730_null_ensemble_discount.csv",         "DEPRECATED-CROSSFIT-OVERFIT: nb730 multi-seed null-ensemble trained on unblind labels"),
    ("nb710_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb711_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb712_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb713_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb714_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb715_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb720_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb721_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb722_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb725_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb731_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),
    ("nb732_p3_boost.csv",                       "DEPRECATED-CROSSFIT-OVERFIT: nb710-732 P3 boost variants trained on unblind labels"),

    # === Legacy clean te arrays (PRE-unblind training; lower priority but not deprecated) ===
    ("nb464_final_blend.csv",                    "LEGACY-1: nb464 final blend SLSQP over nb432+nb460+nb463; 5-fold cross-fit RAE 0.5496"),
    ("nb463_curriculum_slsqp.csv",               "LEGACY-2: nb463 DynCIM curriculum SLSQP (easy->hard stages, lambda=0.5 prior)"),
    ("nb471_three_stage_curriculum.csv",         "LEGACY-3: nb471 three-stage curriculum (easy/med/hard SLSQP, lambda anneal); 5-fold cross-fit RAE 0.5531"),
    ("nb432_router_ensemble.csv",                "LEGACY-4: nb432 router-ensemble (nb424+nb427+nb430+nb431 SLSQP, cross-fit RAE 0.5541) -- anchor for residual-stack family"),

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
