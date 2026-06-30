# New Data Integration Plan (Cycle 129)

**Status:** PLAN ONLY — execution deferred to cycle 129.
**Input dependency:** `data/processed/discover_data_inventory.json` (from discovery phase — not yet present; new raw files observed: `pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv`, `pxr-challenge_htchem-libraries_TRAIN.csv`, `pxr-challenge_TEST_PHASE_1_UNBLINDED.csv`).
**Date:** 2026-06-08

---

## 1. Anchors needing refit
| Anchor | Refit? | Why |
|---|---|---|
| `chemprop_aux` (te 0.6216 LB-honest) | YES | LB-honest PRIMARY-1 source; biggest win-per-hour from enlarged train |
| `nb562` (rank-stretch, 0.5065 cross-fit) | YES | Stretch scalar must be re-grid-searched on enlarged unblind set |
| `nb503` (anchor for nb562, 0.5116) | YES | Underlies nb562; OOD wall may shift with new scaffolds |
| LGBM baselines (nb02 combined; nb2112 K=28 residual) | YES | Cheap, must be re-anchored on new chemprop_aux_v2 |
| `nb730` (multi-seed null-ensemble, 0.4603 cross-fit) | YES | POST-unblind anchor; orthogonality to chemprop_aux_v2 must be re-measured |
| `grand_v6b_calib` | YES | Components shift; re-blend SLSQP weights |

## 2. Compute budget
- Chemprop_aux v2 refit: **2-3 h CPU** (5-fold scaffold CV + final refit on full); single-threaded D-drive bound
- LGBM K=28 residual refit: **5-10 min** per fold; **~30 min total**
- nb562 stretch grid-search: **<5 min**
- nb730 multi-seed (5 LGBM seeds + MACCS): **~15 min**
- Tanimoto-kNN baseline on enlarged train: **~10 min**
- Grand_v6b SLSQP re-blend: **<5 min**
- **Total cycle 129 budget: ~4 h wall, dominated by chemprop_aux v2**

## 3. PRIMARY-1 transition
- Current PRIMARY-1: `nb2112` (chemprop_aux + LGBM K=28 residual, 0.4698 cross-fit)
- **Immediate refit required** because nb2112 = f(te_chemprop_aux)
- After refit: `te_chemprop_aux_v2` becomes the LB-honest anchor; `nb2112_v2` = chemprop_aux_v2 + LGBM K=28 residual on enlarged train
- Until v2 lands, keep `nb2112` as PRIMARY-1 (do NOT submit any v2 candidate to LB without honest cross-fit ≤ 0.4698)
- All POST-unblind candidates (nb562/nb730/nb703) remain CROSSFIT-OVERFIT-flagged until two-regime calibration is re-derived on enlarged unblind

## 4. Stale memory entries (flag for update post-cycle-129)
| Memory entry | Stale field | New value source |
|---|---|---|
| `feedback_lb_actual_scores` | chemprop_aux 0.6216 | te_chemprop_aux_v2 honest cross-fit |
| `feedback_phase2_p3_winner` | nb730 0.4603 (orthogonality vs nb562) | re-measure residual corr to chemprop_aux_v2 |
| `feedback_rank_stretch_universal` | nb562 0.5065, stretch s≈1.07 | re-grid on enlarged unblind |
| `feedback_lb_two_regime_calibration` | PRE/POST cutoff at nb<320 | recalibrate; new train horizon resets cutoff |
| `project_239_blend_breakthrough` | 4-way SLSQP 0.2838 OOF | obsolete if grand_v6b_v2 dominates |
| `feedback_te_vs_pred_oof_protocol` | nb562/nb703 in-sample gaps | refresh delta table |

## 5. Honest evaluation protocol
- **Train:** old 4139 + new train rows (96-compound-uscale + htchem-libraries) + 253 unblind
- **Holdout:** original 513 test (unchanged) — still the LB target
- **CV:** 5-fold scaffold-CV on the enlarged train; old 513 held out across all folds
- **LB delta proxy:** cross-fit RAE on the new unblind subset (if `TEST_PHASE_1_UNBLINDED` contains new labels) — establishes a fresh PRE-unblind regression `in_RAE → LB`
- **Acceptance rule:** any v2 candidate must beat its v1 ancestor on BOTH the original 5-fold scaffold-CV AND the new unblind subset; otherwise revert

## 6. Cycle 129 — 5 distinct methods
a) **`nb950_chemprop_aux_v2`** — retrain chemprop_aux on (4139 + new_train + 253 unblind); 2 heads (PXR + counter); produce `te_chemprop_aux_v2.npy` (513) and `pred_oof_chemprop_aux_v2.npy` on enlarged train OOF
b) **`nb951_lgbm_k28_residual_v2`** — refit K=28 LGBM residual on `chemprop_aux_v2` anchor across enlarged train; new `nb2112_v2` candidate
c) **`nb952_knn_tanimoto_diag`** — Tanimoto-kNN (k=5, ECFP4) on enlarged train; diagnostic only — does novel-scaffold tail RAE compress vs 0.5116 OOD wall? Reports tail-cluster RAE delta
d) **`nb953_nb562_nb503_refit`** — re-grid stretch scalar s and re-fit nb503 anchor on (253 + new_unblind if any); new POST-unblind anchors `nb562_v2` / `nb503_v2`; report both cross-fit and in-sample RAE
e) **`nb954_grand_v6b_v2`** — SLSQP re-blend of {chemprop_aux_v2, nb2112_v2, nb730_v2, nb562_v2, knn_tanimoto_v2}; weights cross-fit, never in-sample-optimized

## 7. Risk flags
- **Label drift:** new train may re-test compounds already in old 4139 with refined pEC50; before merging run InChIKey concordance check — flag any |Δ pEC50| > 0.30 (≈ 2× noise floor 0.15); for drifted compounds keep NEW label only, log to `data/processed/label_drift_audit.csv`
- **Assay batch effect:** uscale-semi-pure and htchem-libraries are different platforms from CRC; add `assay_source` categorical to LGBM K=28 feature set; do NOT pool single-conc-style with CRC pEC50 without first checking distribution match
- **Scaffold overlap:** if new train shares ≥ 80% scaffolds with old 4139, novel-scaffold OOD wall (0.5116) will NOT move; gate optimism accordingly
- **Counter-assay coverage:** if new train lacks counter-assay labels, chemprop_aux v2 second head becomes NaN-masked majority — measure auxiliary-head signal-to-noise before deploy
- **Unblind augmentation precedent:** `feedback_unblind_augmentation` showed +6.1% rows on covered scaffolds did NOT move LB; expect external-scaffold-diverse rows (htchem-libraries plausibly) to be the only real-LB mover

---
END PLAN — execution gated on cycle 129 trigger.
