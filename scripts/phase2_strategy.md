# Phase-2 Strategy — Frozen Activity LB → 2026-07-01 Final Deadline

Window: ~23 days. Activity LB frozen; submissions queue for Phase-2 re-grade against full 513 ground truth (253 unblind + 260 new Phase-2 labels). No live LB feedback. All decisions must rely on honest cross-fit + holdout simulation.

## 1. Submission Cadence

- Activity cron `83570bdf` (4h :23) STAYS LIVE. Every fire queues a candidate; re-grade re-scores the most recent submission per team at deadline.
- Structure cron `2699b648` (4h :47) STAYS LIVE.
- Rotate PRIMARY-1..PRIMARY-5 each fire (round-robin, mod 5) so 5 diverse candidates remain in the queue at any time. Final re-grade picks LAST submission, but the queued history protects against last-hour bugs.
- In-script 4h rate-limit enforced; no per-fire prompt (user authorized 2026-06-01).

## 2. Iteration Goals

a) **Final ensemble candidate maximizes RAE on all 513** (not 253-OOD only). Optimize against simulated 5-fold scaffold-CV on the full ~4939 labeled set (4139 train + 547 new Phase-2 + 253 unblind).
b) **F2 failure-tail attack**: 172 still-unrescued novel scaffolds (scaf_train_freq=0, ΔRAE > 0.6 vs corpus). Targeted external-data kNN + abstention + per-row OOD gating.
c) **Honest 5-fold scaffold-CV on ~4939**: rebuild OOF matrix end-to-end. 253 unblind enters TRAINING for the final-deploy refit, but ALSO held out in dedicated 5-fold to preserve a 253-OOD honest gate (no double-dipping).

## 3. Risk Factors

a) **DRC noise drift**: downweight rows with `CAD_CV > 5%` in the per-fold loss (sample_weight = 1/(1+CAD_CV/5)). Already validated in nb472 residual-stack.
b) **POST-unblind cross-fit overfit**: 253 phase1 unblind labels stay as VALIDATION ONLY for ladder gating; only enter training in the final-deploy refit at the very end. Preserves the 0.5065/0.4928/0.4603 honest gate.
c) **No LB feedback for 23 days**: rely on (i) honest 5-fold cross-fit, (ii) leave-one-scaffold-out on rare-scaffold tail, (iii) PRE-unblind LB calibration (in_RAE ≈ LB + 0.003 for nb<320). LB estimate = `0.51*pred_oof + 0.49*te[unb_idx]`.

## 4. Cycle Plan (5 distinct methods per cycle, every 4-8h)

- **Cycles 130-140** (now → +5d): build on c129v2 anchor (nb730 multi-seed null-ensemble, 0.4603). Test abstention head, isotonic calibration on rare-scaffold residuals, small MLP head on combined features, split-conformal prediction intervals, rank-stretch sweep on c129v2.
- **Cycles 141-150** (+5d → +12d): focused F2 attack. Per-row external-data kNN (ChEMBL PXR + counter-assay), abstention only on rare-scaffold (scaf_train_freq<2), OOD-aware blend weights (per-row gating by pred-uncertainty + Tanimoto-to-train).
- **Cycles 151-160** (+12d → +20d): final ensemble candidate selection + diversity-aware deploy. SLSQP cap at 5 components; reject components with residual corr > 0.85 to current ensemble.

## 5. PRIMARY-1..5 Rotation

Maintain 5-slot ladder in `data/processed/leaderboard_log.csv`:
- PRIMARY-1: best honest cross-fit (currently nb730 0.4603)
- PRIMARY-2: best LB-calibrated PRE-unblind (chemprop_aux 0.6246 LB)
- PRIMARY-3..5: orthogonal candidates (residual corr < 0.85 vs PRIMARY-1)

Rotate fire order each cycle; update `auto_submit_ladder.py` to read `submission_watch.csv` index modulo 5.

## 6. Memory Checkpoint

Every 10 cycles, run `scripts/audit_ladder_integrity.py` + prune MEMORY.md entries older than 14 days that have been superseded by newer findings. Keep only landmark feedback (top-5 plateau-breakers + active failure modes).

## 7. Final-Deadline Trigger

T-24h before 2026-07-01: lock PRIMARY-1, disable cron rotation, pin chosen candidate to a single deterministic submission, fire once at T-12h and T-2h as redundancy. Confirm structure track has a final candidate (Boltz-2 multi-seed v1 LB 0.4632 baseline; upgrade target 0.55-0.75 via template-biased multi-seed).
