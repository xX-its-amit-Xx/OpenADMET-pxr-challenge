# OpenADMET PXR Challenge — Activity Leaderboard Freeze Escalation

**User:** `xX-its-amit-Xx`
**Track affected:** Activity Prediction (`pEC50` regression, 513 test compounds)
**Track unaffected:** Structure Prediction (refreshes normally)
**Frozen since:** 2026-05-26 04:45 UTC
**Hours frozen as of 2026-06-08 04:45 UTC:** 336 h (14 days)
**Status as draft, not yet sent.** Submit at user discretion.

---

## 1. Freeze evidence

The `lb_score` column for the Activity Prediction track has not changed in any
hourly probe since the 2026-05-26 04:45 UTC snapshot. The hourly LB-logger
cron (`f4f166e1`, `scripts/log_lb_scores.py`) writes one row per track per
fire to `data/processed/leaderboard_log.csv`. Every activity row written in
the trailing 14-day window is byte-identical on the scoring columns:

```
ts_utc                    track     user_rank  lb_score  n_visible  mae     rae     r2      spearman  kendall  submitted_utc_on_lb
2026-06-08 02:24:08 UTC   activity  262        0.7655    328        0.58    0.7655  0.2878  0.6315    0.4502   2026-05-26 04:45 UTC
2026-06-08 02:28:19 UTC   activity  262        0.7655    328        0.58    0.7655  0.2878  0.6315    0.4502   2026-05-26 04:45 UTC
... (every subsequent hourly probe, identical) ...
2026-06-08 03:39:16 UTC   activity  262        0.7655    328        0.58    0.7655  0.2878  0.6315    0.4502   2026-05-26 04:45 UTC
```

Key field: `submitted_utc_on_lb = 2026-05-26 04:45 UTC` — the LB itself
reports the scored submission timestamp, and the value has not advanced for
14 days despite continuous successful submission acks (see §2).

## 2. Submission history while frozen

`data/processed/submission_log.csv` records **26 successful Activity
Prediction acks** with API response `"Submission received from
'xX-its-amit-Xx' for the Activity Prediction track. Your predictions are
being processed and will appear on the leaderboard within 2 hours."`
between 2026-05-29 19:02 UTC and 2026-06-08 01:04 UTC.

Recent representative tail (PRIMARY ladder, all acked, none scored):

| submitted_utc          | csv                                              | local OOF / predicted LB |
|------------------------|--------------------------------------------------|--------------------------|
| 2026-06-01 20:25 UTC   | chemprop_aux.csv                                 | honest 0.6216 / pred 0.6246 |
| 2026-06-02 00:25 UTC   | grand_v6b_calib.csv                              | in 0.6409 / pred 0.6439 |
| 2026-06-02 08:25 UTC   | nb306_cepsmim.csv                                | in 0.6486 / pred 0.6516 |
| 2026-06-02 16:25 UTC   | nb305_mope.csv                                   | in 0.6601 / pred 0.6631 |
| 2026-06-03 00:25 UTC   | 95_all_feature_fusion.csv                        | in 0.6625 / pred 0.6655 |
| 2026-06-04 02:07 UTC   | nb1660_deploy_nb1632_mean.csv                    | honest cross-fit 0.5107 / pred ~0.514 |
| 2026-06-04 12:25 UTC   | nb2112_deploy_shap28.csv                         | predicted ~0.47 |
| 2026-06-06 12:25 UTC   | nb1014_multi_seed_bag.csv                        | honest 0.5930 / pred ~0.596 |
| 2026-06-06 20:25 UTC   | nb1001_crossfit_chempropaux_nb972_stretch.csv    | honest 0.5994 / pred ~0.602 |
| 2026-06-07 04:25 UTC   | 54_deep_ensemble_uncertainty.csv                 | in 0.6657 / pred 0.6687 |
| 2026-06-07 08:25 UTC   | 27_nr_weighted_lgbm.csv                          | in 0.6729 / pred 0.6759 |
| 2026-06-07 12:25 UTC   | 82_selectivity_aware.csv                         | in 0.6730 / pred 0.6760 |
| 2026-06-07 20:25 UTC   | 67_lgbm_chembl_all_nr_weighted.csv               | in 0.6746 / pred 0.6776 |
| 2026-06-08 01:04 UTC   | nb1162_deploy_nb1153.csv                         | scaffold-CV 0.4206 / calibrated band 0.436–0.521 |

All 26 acks include a SHA256 and were emitted via
`scripts/auto_submit_ladder.py` to the Gradio endpoint
`https://openadmet-pxr-challenge.hf.space/`. Several would, if scored at
parity with their cross-fit estimate, materially change `n_submissions_visible=328`
and `user_rank=262`. None of them have changed the LB.

## 3. Comparison — structure track refreshes normally

Same logger cron, same Gradio endpoint, same user. Structure submissions
are scored within the advertised ≤2h window:

| submitted_utc        | structure csv                     | LB scored at                | lddt_pli |
|----------------------|-----------------------------------|-----------------------------|----------|
| 2026-06-01 16:43 UTC | structure_baseline_v4.zip         | superseded                  | 0.4583   |
| 2026-06-01 20:46 UTC | structure_baseline_v5.zip        | scored, visible             | --       |
| 2026-06-08 01:17 UTC | structure_v6_perlig_qsel.zip      | **2026-06-08 01:17 UTC**    | 0.4632   |

The structure LB advanced 6× during the activity freeze window, with the
most recent advancement < 3 h before this document. Same auth, same client,
same upload mechanism — only activity track is stuck.

## 4. Diagnostic — no error, no rejection

Every activity POST returns the canonical `Submission received from
'xX-its-amit-Xx' for the Activity Prediction track. Your predictions are
being processed and will appear on the leaderboard within 2 hours.`
The only non-success replies in the 14-day window are the in-script
4 h-rate-limit messages (`Error: You submitted an activity prediction on …
Please wait HH:MM:SS before submitting again.`) when the auto-submit cron
fires too close to a prior fire — these are local-rate-limit, not server
rejection. There has been **no validator error, no schema mismatch, no
authentication failure, no 4xx/5xx**. The job appears to be entering an
ingest queue that is not draining.

## 5. Local OOF projections (what should land if scoring resumes)

Honest 5-fold scaffold-CV cross-fit (LB-faithful, train-only, no unblind
leakage) for several queued candidates:

| candidate                        | cross-fit RAE | predicted LB | vs frozen 0.7655 |
|----------------------------------|---------------|--------------|------------------|
| `nb1162_deploy_nb1153.csv`       | 0.4206 (scaffold pooled) | 0.436–0.521 band | -0.244 to -0.330 |
| `nb2240_nb2171_k20.csv` (queued) | ~0.4646 (k=20 anchor stack, PRE-unblind)    | ~0.4646 expected | -0.301 |
| `nb1660_deploy_nb1632_mean.csv`  | 0.5107        | ~0.514       | -0.252 |
| `nb1014_multi_seed_bag.csv`      | 0.5930        | ~0.596       | -0.170 |
| `chemprop_aux.csv`               | 0.6216 honest | 0.6246       | -0.141 |

If any of the recent PRIMARY-1 candidates scored at parity with cross-fit,
displayed rank should improve materially from 262/328. Multi-day silence on
this magnitude of expected delta is the strongest indicator the activity
scorer is not running.

## 6. Contact points

- **HF Space (submission endpoint):**
  https://openadmet-pxr-challenge.hf.space/
- **HF Space repo (where to file an Issue / Discussion):**
  https://huggingface.co/spaces/openadmet/pxr-challenge — open the
  *Community* tab → *New discussion* (preferred channel; ties the report to
  the Space the auto-submitter actually hits).
- **OpenADMET GitHub org (alt issue tracker):**
  https://github.com/OpenADMET — file in the most relevant active repo
  (e.g. `OpenADMET/PXR-Challenge-Tutorial` issues if no dedicated
  challenge-ops repo exists).
- **OpenADMET tutorial / challenge docs:**
  https://github.com/OpenADMET/PXR-Challenge-Tutorial
- **Training data dataset card** (organizer contact often listed here):
  https://huggingface.co/datasets/openadmet/pxr-challenge-train-test
- **Discord:** if OpenADMET has a community Discord, link is typically on
  the HF Space README — confirm before referencing.

## 7. The ask

1. **Confirm intent vs incident.** Is Activity Prediction scoring
   intentionally paused for the Phase-1 → Phase-2 cutover (Phase-1 closed
   2026-05-25, Analog Set 1 unblinded 2026-05-26 — i.e. the freeze
   timestamp coincides with the documented cutover), or is the scoring job
   broken?
2. **If paused:** publish the expected resume date so we can stop firing
   into a black hole and reclaim API rate-limit budget for the Phase-2
   re-open.
3. **If broken:** ETA for backlog catch-up. We have 26 acked-but-unscored
   submissions on the user account; please confirm whether the queue will
   be re-processed in order, only the latest will be scored, or all will
   be silently dropped (so we can resubmit the intended PRIMARY-1).
4. **Either way:** acknowledge whether `submission_log.csv` API-acks
   constitute durable receipt, or whether they are best-effort. If
   best-effort, please document a retry SLA so the auto-submitter can
   implement an idempotency / replay layer.

---

## Suggested message body for paste

> Hi OpenADMET team,
>
> The Activity Prediction leaderboard for user `xX-its-amit-Xx` has been
> frozen at rank 262 / RAE 0.7655 with
> `submitted_utc_on_lb = 2026-05-26 04:45 UTC` for 14 days as of
> 2026-06-08 04:45 UTC. During that window I have 26 activity submissions
> that returned the canonical "Submission received … will appear on the
> leaderboard within 2 hours" message from the HF Space Gradio endpoint;
> none have been scored or displayed. There is no validator error, no
> 4xx/5xx, no schema rejection — the receipts look healthy.
>
> The Structure Prediction track on the same account, hitting the same
> endpoint with the same credentials, has refreshed 6× in the same window
> (most recent score 0.4632 LDDT-PLI on 2026-06-08 01:17 UTC), so this
> appears isolated to activity scoring.
>
> Could you confirm whether activity scoring is intentionally paused for
> the Phase-1 → Phase-2 cutover (the freeze timestamp matches the
> 2026-05-26 unblind date), or whether the scoring job is down? If the
> latter, an ETA for backlog catch-up and confirmation that queued acks
> will be replayed (vs latest-wins vs dropped) would be very helpful so I
> can stop saturating the 4 h rate limit and queue the intended PRIMARY-1
> submission cleanly when scoring resumes.
>
> Happy to share my `submission_log.csv` + `leaderboard_log.csv` (with
> SHA256s) on request.
>
> Thanks,
> Amit (HF username `xX-its-amit-Xx`)

---

*Draft generated 2026-06-08. Local evidence sources:*
- `data/processed/leaderboard_log.csv` (316 rows, hourly LB probes)
- `data/processed/submission_log.csv` (67 rows, every POST + API response)
- `data/processed/structure_submission_log.csv` (9 rows, structure POSTs)
- `scripts/auto_submit_ladder.py` (submission client wiring)
- `scripts/log_lb_scores.py` (hourly LB scrape)
