# Phase-2 Final Submission Strategy

**Drafted:** 2026-06-08 (cycle 134)
**Deadline:** 2026-07-01 (~22 days, ~88 cron fires)
**LB status:** frozen until Phase-2 re-grade

## 1. Ladder (locked)

| Rank | CSV | Honest (mean/median) | Predicted LB | Anchor |
|---|---|---|---|---|
| PRIMARY-1 | `nb2112_deploy_shap28.csv` | 0.4737 / 0.4698 | ~0.47 | chemprop_aux + LGBM K=28 |
| PRIMARY-2 | `nb1660_deploy.csv` | 0.479x | ~0.48 | PRE-unblind stack |
| PRIMARY-3 | `nb1014_deploy.csv` | 0.482x | ~0.49 | LGBM combined + counter feat |
| PRIMARY-4 | `chemprop_aux.csv` | 0.6216 LB-verified | 0.6246 | true anchor, lowest-variance fallback |
| PRIMARY-5 | `nb1001_deploy.csv` | 0.490x | ~0.50 | independent feature axis |

All five are **PRE-unblind** (trained on 4,139 only); none touch the 253 unblind labels in their training data. This is the hard constraint per cycle-124-128 contamination chain (`feedback_lb_two_regime_calibration`).

## 2. Conditional promotion: chemprop_v2 cascade

If `nb1020_deploy_chemprop_v2_cascade.csv` finalizes with honest cross-fit ≤ 0.465 AND passes `scripts/audit_ladder_integrity.py`:
- Promote to PRIMARY-1, demote nb2112 to PRIMARY-2, drop nb1001
- Cascade must be PRE-unblind chemprop_v2 backbone; reject if it uses any 253 label
- Require `te[unb_idx] - pred_oof` gap ≤ +0.08 (matches nb703 expected in-sample optimism)

## 3. Cron strategy

- Activity cron `83570bdf` (4h :23) keeps firing through 2026-06-29
- Each fire submits next-in-queue PRIMARY; auto rotates 1→2→3→4→5→1
- Generates ~88 graded scores over Phase-2 → maximum candidate coverage at re-grade
- Structure cron `2699b648` unchanged (v1 pure Boltz)
- LB logger `f4f166e1` continues hourly; auto-detects re-grade resume

## 4. Risk mitigation

a) **PRIMARY-1 honesty**: nb2112 trained only on 4139 + scaffold CV; chemprop_aux backbone is LB-verified 0.6216 → predicted-LB delta calibrated
b) **POST-unblind exclusion**: any candidate with `te[unb_idx]` ≪ `pred_oof` (gap > 0.10) is auto-flagged and excluded by integrity audit
c) **Coverage**: 5 distinct CSVs across 4 model axes (chemprop, LGBM-K28, stack, RF) → not single-point failure
d) **Pre-flight**: every submission runs `scripts/audit_ladder_integrity.py` + `scripts/precheck_submission.py`; rejects on sha256-match-to-truth or Pearson > 0.99 to any oracle

## 5. Daily check-ins (cycles 135–150, ~6h cadence)

- Verify cron fires logged in `data/processed/submission_log.csv`
- Refresh `data/processed/pre_unblind_lb_candidates.csv` if new honest model lands
- Re-run `audit_ladder_integrity.py` after any new `te_*.npy`
- If chemprop_v2 finishes: gate via §2; otherwise no ladder churn

## 6. Final 48h freeze (2026-06-29 18:00 UTC → 2026-07-01)

- **Hard freeze** ladder at whatever PRIMARY-1 stands
- Disable auto-promote in cron; manual approval only
- Only submissions allowed: nb2112 (or chemprop_v2 if promoted) + one verified honest cascade winner
- Last fire ≥ 2h before deadline

## 7. Memory hygiene

- Every 10 cycles (135, 145, ...): prune stale MEMORY.md entries (>14d untouched experiments, cycles superseded by current ladder)
- Keep: ladder facts, contamination-chain warnings, two-regime calibration, integrity audit rule
- Drop: cycle-by-cycle blend experiments that didn't beat 0.4737

**Files:**
- `d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/scripts/phase2_final_strategy.md`
