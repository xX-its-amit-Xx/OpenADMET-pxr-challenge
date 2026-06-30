# Phase-2 Final Submission Strategy — v2

**Drafted:** 2026-06-08 (cycle 173–178 close)
**Supersedes:** `phase2_final_strategy.md` (v1, cycle 134; nb2112 0.4737)
**Deadline:** 2026-07-01 (~23 days, ~138 cron fires at 4h cadence)
**LB freeze note:** activity track score 0.7655 has not moved since 2026-05-26 05:45 UTC (rank 262/328); LB logger `f4f166e1` confirms frozen across ~16 hourly polls/day. Organisers are expected to do a **bulk re-grade at Phase-2 close (~2026-07-01)** against the 253-unblind labels. All Phase-2 submissions queue invisibly until that re-grade — so the right move is to **maximise diverse honest candidates in the queue**, not to chase incremental LB feedback.

## 1. NEW honest floor — nb2240

| Metric | Value | Source |
|---|---|---|
| OOF pooled RAE (5 kf-seeds, K=20 RFE pyramid) | **0.4598** ± 0.0009 | `nb2240_summary.json:pooled_rae_mean_seeds` |
| Deep-30-seed verify | **0.4601** ± 0.0017 (min 0.4574, max 0.4651) | `deep_30_verify.mean_rae` |
| Deploy CSV | `submissions/nb2240_nb2171_k20.csv` (513 rows) | written 2026-06-08 02:56 |
| Stretch s | 1.011 (per-fold cross-fit median) | `deploy_s` |
| K=20 family mix | Mordred 4, ChempropEmbed 8, Avalon 2, MACCS 1, AtomPair 4, ChEMBL kNN 1 | `k20_family_counts` |
| Deploy blend | 80.8% K=20 LGBM + 19.2% nb562 (chemprop_aux 0%, nb1191 0%, nb503 0%) | `deploy_weights` |
| Δ vs nb2171 (prev best) | **−0.0078** | `delta_vs_nb2171`; verdict `BEATS_NB2171` |
| Δ vs anchor chemprop_aux | **−0.1586** | `delta_K20_vs_anchor` |

Key recipe: drop nb730 from the anchor pool (POST-unblind contamination), refit Karpathy K-RFE on the cleaned 117-col 5-way matrix down to K=20.

## 2. Predicted LB band

PRE-unblind calibration (`feedback_lb_two_regime_calibration`, gap ≈ +0.0045 in_RAE → LB on n=4 verified pairs):

| Quantity | Value |
|---|---|
| Honest floor | 0.4601 |
| +PRE delta | +0.0045 |
| **Predicted LB at re-grade** | **0.4646** |
| Current frozen LB | 0.7655 (best of 262 entries) |
| Expected Δ at re-grade | **−0.3009 RAE** (≈ −39% relative) |

Caveat: this is a per-submission estimate; under the bulk re-grade only **one** of our queued CSVs becomes "the" score per leaderboard rule (best-of). Rotating the full PRIMARY stack maximises probability that the chosen score is nb2240-class.

## 3. PRIMARY ladder (locked, deploy CSVs verified)

| Rank | CSV | Honest RAE | Predicted LB | Notes |
|---|---|---|---|---|
| PRIMARY-1 | `submissions/nb2240_nb2171_k20.csv` | **0.4601** | 0.4646 | K=20 RFE pyramid, drop-nb730 clean |
| PRIMARY-2 | `submissions/nb2171_deploy_anchor_swap.csv` | 0.4676 | ~0.472 | prev pyramid baseline; K=28 |
| PRIMARY-3 | `submissions/nb2112_deploy_shap28.csv` | 0.4737 | ~0.478 | chemprop_aux + LGBM K=28 (v1 PRIMARY-1) |
| PRIMARY-4 | `submissions/nb1660_deploy.csv` | 0.479x | ~0.484 | PRE-unblind stack |
| PRIMARY-5 | `submissions/nb1014_deploy.csv` | 0.482x | ~0.487 | LGBM combined + counter-feat |
| **SAFETY** | `submissions/chemprop_aux.csv` | **0.6216 LB-verified** | 0.6246 | only LB-anchored entry — fallback if the K=20 honest signal collapses |

All six are PRE-unblind (trained on 4,139 only). Integrity audit (`scripts/audit_ladder_integrity.py`) passes for every entry; max in-sample vs cross-fit gap +0.16 (within nb703 expected band).

## 4. Cycle 173–178 closed axes (negative results to NOT revisit)

| Axis | Notebook | Pooled RAE | Verdict |
|---|---|---|---|
| XGBoost K=20 substitute | nb2270 | — | residual stack HURTS_NB2240 |
| Random Forest K=20 substitute | nb2281 | — | HURTS_NB2240 |
| Morgan radius {3, 5} SHAP-top15 add-on | nb2280 | 0.4648 | +0.0018 vs nb2240, NO_BEAT |
| Mordred-substrate expansion | nb2291 | — | NO_BEAT |
| Persistence homology features | nb2222 | 0.5060 | BEATS_ANCHOR_BUT_WORSE_THAN_K28 |
| Graph spectral K=48 | nb2201 | 0.5050 | BEATS_ANCHOR_BUT_WORSE_THAN_K28 (prune skipped) |
| Structure-pose-conditioned features | nb2271 | 0.4711 | HURTS_NB2240 |
| Sim-matrix nearest-neighbor pool | nb2292 | 0.471 | HURTS_NB2240 |
| Per-fold K-pyramid vs global | nb2290 | 0.4607 perfold vs 0.4574 global | GLOBAL_BETTER (no gain) |
| Bag-of-bags meta | nb2272 | 0.4644 | HURTS_NB2240 (+0.0046) |

All nine attack vectors closed against the nb2240 0.4601 wall. The K=20 RFE pyramid is the current honest ceiling for the available feature corpus.

## 5. Final-week strategy

a) **Lock nb2240 as PRIMARY-1** for all remaining auto-fires through 2026-06-29 18:00 UTC.
b) **Rotate the full PRIMARY-1..PRIMARY-5 stack every 4h** via cron `83570bdf` — at ~138 fires we cycle each candidate ~27 times. Maximises queue diversity for the Phase-2 re-grade.
c) **Insert SAFETY (chemprop_aux) every 6th rotation** (~23 fires) as our LB-anchored insurance — the only entry with a real graded baseline (0.6216).
d) **Final 48h freeze** 2026-06-29 18:00 UTC → 2026-07-01: disable auto-rotation, hard-pin PRIMARY-1 + SAFETY only, manual approval required for any deviation, last fire ≥2h before deadline.

## 6. Risk mitigation

| Risk | Mitigation |
|---|---|
| nb2240 cross-fit-to-LB transfer fails | SAFETY `chemprop_aux.csv` LB-verified band 0.6216 → 0.6246; rotated into the queue regularly |
| K=20 RFE manifold is too narrow for the 513 test set | PRIMARY-2..PRIMARY-5 cover K=28, stack, LGBM+counter, RF axes; one of them grades if K=20 generalises poorly |
| POST-unblind contamination sneaks back in | every `te_*.npy` checked by `audit_ladder_integrity.py`; gap >0.10 between `te[unb_idx]` and `pred_oof` auto-flags |
| Bulk re-grade picks a stale (pre-nb2240) submission | rotation puts nb2240 into the queue every ~24h until freeze (~23 fires before deadline) |
| Cron silently stops | LB logger `f4f166e1` writes hourly to `data/processed/leaderboard_log.csv`; we'll detect missing submission_log entries within one cron cycle |

## 7. Memory hygiene

Every 10 cycles (180, 190, …): prune MEMORY entries superseded by nb2240. Keep contamination-chain warnings, two-regime calibration, integrity audit rule, and the closed-axis list in §4 (so we don't re-attempt XGB/CatBoost/RF/Mordred-substrate/PH/spectral/Morgan-radius/sim-matrix). Drop cycle-by-cycle blend experiments that didn't beat 0.4601.

**Files:**
- `d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/scripts/phase2_final_strategy_v2.md` (this file)
- `d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/scripts/phase2_final_strategy.md` (v1, preserved for diff)
- `d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/submissions/nb2240_nb2171_k20.csv` (PRIMARY-1 deploy)
- `d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/processed/nb2240_summary.json` (honest floor evidence)
