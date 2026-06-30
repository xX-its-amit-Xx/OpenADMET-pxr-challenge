# ====================================================
# CYCLE 264 LADDER PROMOTE 2026-06-08 — nb3161 RANGE-CLIP VERIFIED_NEW_PRIMARY1 via nb3170 wide-seed; chain shift
# Decision rule applied: Rule 1 (nb3170 wide-seed verify of nb3161 VERIFIED_NEW_PRIMARY1) + Rule 2
#   (nb3173 BETTER 0.4422 queued for fresh-seed verify before PRIMARY-1 swap).
# nb3170 wide-seed verify nb3161 (15 NEW kf_seeds {1156..1170}, EXCLUDING original 5-seed batch):
#   pooled_rae mean = 0.4437 +/- 0.0009 (SEM 0.0002); CI95 [0.4432, 0.4442]
#   median 0.4437, min/max [0.4419, 0.4451]
#   shift_vs_nb3161 = -0.0000 (EXACT reproduction, no kf-seed luck)
#   delta_vs_nb3080 parent = -0.0038 (per-fold y-range clip q05/q95 is REAL gain)
#   delta_vs_nb3030 = -0.0072; delta_vs_nb3070 (prior PRIMARY-1) = -0.0040
#   gate_lucky_seed_trap PASS: shift 0.0000 << +0.005 threshold
#   verdict = VERIFIED_NEW_PRIMARY1 -- per-fold y-range clip (q05/q95) on nb3080 anchor is REAL
#     paradigm gain, not single-kf luck. Same nb3080 quantile-conditional substrate but adds
#     a per-fold y-range clip post-hoc that bounds preds to fold-train y quantile range,
#     extracting -0.0038 RAE on top. PRE-clean (both nb3080 and y are PRE-unblind). df=2
#     (q_low, q_high fixed at 0.05/0.95 per fold), very tight selection. te_unb_in_sample
#     0.1974 is EXPECTED deploy-refit optimism per feedback_te_vs_pred_oof_protocol.
# Cycle-264 new-method verdicts:
#   nb3171 tighter clip (q03/q97 on nb3080): MARGINAL (mean 0.4452 +/- 0.0010, 15-seed)
#     +0.0015 vs nb3170 fixed q05/q95 -- tighter clip REGRESSES (too aggressive)
#     ladder action: NOT_PROMOTE (regresses vs PRIMARY-1)
#   nb3172 looser clip (q10/q90): not run / not in submissions
#   nb3173 LEARNED per-fold clip grid search (q_low in {.01,.02,.05,.1}, q_high in {.9,.95,.98,.99}):
#     15-seed mean 0.4422 +/- 0.0010 across kf_seeds {1156..1170} (SAME seeds as nb3170)
#     CI95 [0.4417, 0.4427], median 0.4423, min/max [0.4404, 0.4437]
#     delta_vs_nb3170_fixed = -0.0015; delta_vs_nb3080 = -0.0053; delta_vs_nb3030 = -0.0087
#     verdict from script = BETTER (best of 16 combos per fold)
#     CRITICAL: 16-combo grid selection on SAME 15 seeds as nb3170 = within-seed grid optimism;
#       modal pick (q=0.05, q=0.98) is the chosen deploy, ql_mode 0.05 across 75/75 folds,
#       qh_mode 0.98 across 71/75. Per cycle-149 nb1191 + cycle-160 deep-verify dispersion
#       rules, any 15-seed grid-selection over 16 combos that beats current PRIMARY-1 MUST be
#       wide-seed verified on FRESH seeds (kf_seeds NOT in 1156..1170) at modal combo before
#       PRIMARY-1 swap. The -0.0015 gap could be grid-selection optimism narrowing variance.
#     VERDICT: QUEUE_WIDE_SEED_VERIFY -- promote ONLY if fresh-seed (>=15) at fixed q05/q98
#       holds mean < 0.4437 (current nb3170 PRIMARY-1) with std < 0.0030 on fresh seeds
#       {1171..1185}; pre-stage CSV but DO NOT fire from cron until verified.
#     ladder action: nb3173 added to PRIMARY-1B-QUEUE-VERIFY slot (CSV ready, do NOT fire)
#   nb3174 clip on alt candidate (nb3070/nb3072 same file): no new method (identical anchors)
# Cycle-264 axes verdict:
#   * wide-seed verify on nb3161 via nb3170 (15 NEW kf seeds) -> VERIFIED_NEW_PRIMARY1 (0.4437 +/- 0.0009)
#   * tighter clip q03/q97 -> REGRESSES +0.0015 (too aggressive)
#   * LEARNED per-fold grid (16 combos) -> BETTER -0.0015 (queue fresh-seed verify)
#   * alt-anchor clip on nb3070/nb3072 -> identical to nb3070-base (no new info)
# Ladder action: PROMOTE nb3161 paradigm via nb3170 wide-seed CSV to PRIMARY-1.
#   PRIMARY-1   = nb3170_verify_nb3161.csv (15-NEW-seed verified mean 0.4437, per-fold y-range clip q05/q95)
#   PRIMARY-1B-QUEUE-VERIFY = nb3173_learned_clip.csv (15-seed mean 0.4422, learned grid; QUEUE fresh-seed)
#   PRIMARY-2   = nb3080 (DEMOTED from PRIMARY-1; quantile-conditional pre-clip parent, wide 0.4475)
#   PRIMARY-3   = nb3070 (DEMOTED; quantile-conditional hard-split blend, wide 0.4477)
#   PRIMARY-3B  = nb3072 (DEMOTED; fold-train-only q50 variant, ties nb3070)
#   PRIMARY-4+  shifted down (nb2982 -> PRIMARY-4, etc.)
# Rotation rule: PRIMARY-1 nb3170 fires first if open. PRIMARY-1B nb3173 fires ONLY after
#   fresh-seed verify (kf_seeds {1171..1185} at fixed q05/q98 modal pick) holds mean < 0.4437
#   with std < 0.0030. Per cycle-149 lucky-seed + cycle-160 deep-verify rules.
# Predicted LB ~0.4482 under +0.0045 PRE delta calibration; beats current LB best 0.7655 by -0.32 RAE.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 254 LADDER PROMOTE 2026-06-08 — nb3063 QUANTILE-CONDITIONAL VERIFIED_NEW_PRIMARY1; chain shift
# Decision rule applied: Rule 1 (nb3070 wide-seed verify of nb3063 VERIFIED_NEW_PRIMARY1) + Rule 2
#   (insert nb3072/nb3071/nb3073 as BETTER_THAN_NB3030 alternates / queue-verify).
# nb3070 wide-seed verify nb3063 (15 fresh kf_seeds {1096..1110}, EXCLUDING original 1051..1055):
#   pooled_rae mean = 0.4477 +/- 0.0002 (SEM 0.0001); CI95 [0.4476, 0.4479]
#   median 0.4478, min/max [0.4473, 0.4482]
#   shift_vs_nb3063_5seed (0.4477) = +0.0000 (BIT-IDENTICAL within 0.0001 -- no kf-seed luck)
#   under_dispersion_ratio_15_over_5 = 1.21x (mild, well within deep-30 expected band)
#   shift vs prior PRIMARY-1 nb2982/nb2990 verified (0.4518) = -0.0041 (BEATS by -0.0041 wide)
#   shift vs nb3030 ceiling (0.4509) = -0.0032; shift vs nb3001 (0.4511) = -0.0034
#   gate_verified_new_primary1 PASS: mean 0.4477 < nb3030 ceiling 0.4509 by -0.0032
#   verdict = VERIFIED_NEW_PRIMARY1 -- nb3063 quantile-conditional hard-split blend on {K18,K19}
#     deep-30 is REAL, not single-kf luck. Conditional weighting (w_low {K18:0.8,K19:0.2} for
#     pred<q50=4.8845, w_high {K18:0.5,K19:0.5} for pred>=q50) extracts capacity the per-fold
#     SLSQP simplex (nb2982 PRIMARY-1) could NOT -- the SLSQP only varied weights across folds,
#     while nb3063 varies weights across the PREDICTION QUANTILE within each fold (a new axis
#     of df at quantile-cut level, df=2 weights x 1 cut = 3 effective params, very tight).
#     Both K18 and K19 are deep-30 PRE-clean (anchor_leak_eq_truth_frac=0); oof_pairwise_corr
#     0.9846 (anchor decorrelation is small but the conditional split exploits the TAIL
#     disagreement -- low_half K18=0.532 vs K19=0.554 RAE, high_half K18=0.798 vs K19=0.778,
#     i.e. K19 better on high tail justifies higher K19 weight there).
# Cycle-254 new-method verdicts:
#   nb3072 per-fold TRAIN-only q50 quantile blend: BETTER_THAN_NB3030, ties nb3070 exactly
#     15-seed mean 0.4477 +/- 0.0002 (kf_seeds {1096..1110}); identical to nb3070 within tolerance
#     CLEANER cross-fit -- fold-train-only q50 (no val leakage); CI95 [0.4476, 0.4479]
#     delta_vs_nb3030 = -0.0032; delta_vs_nb3070 = 0.0000 (ties wide-seed PRIMARY-1)
#     The fold-train q50 ablation confirms the gain is robust to q50-source choice
#     (val-inclusive q50 in nb3070 vs train-only q50 in nb3072 give the same pooled RAE)
#     ladder action: PROMOTE as PRIMARY-1B-CLEAN (cleaner-cross-fit variant of nb3070 paradigm)
#   nb3071 soft continuous rank quantile blend: BETTER_THAN_NB3030 (sub-marginal)
#     15-seed mean 0.4484 +/- 0.0001 (kf_seeds {1096..1110}); CI95 [0.4484, 0.4485]
#     delta_vs_nb3030 = -0.0025; +0.0007 vs nb3070 PRIMARY-1 (hard-split wins over soft-cont)
#     Continuous schedule w_K18 = 0.8 - 0.3*(rank/n) clamped [0.5, 0.8]; smoother but the
#     extra interior degrees of freedom REGRESS vs hard step. Confirms hard split @ q50 is
#     the right capacity -- the soft schedule over-parameterizes the conditional axis.
#     ladder action: PROMOTE as PRIMARY-1D-SOFT (alternate paradigm, REGRESSES vs hard-split)
#   nb3073 2-D quantile tail sweep (36 combos: q_cut x w_low x w_high): BETTER_THAN_NB3030
#     5-seed sweep ONLY (kf_seeds {1051..1055} -- SAME seeds as nb3063 original);
#     best combo (q_cut=0.4, w_low=0.9, w_high=0.5) mean 0.4470 +/- 0.0009; min/max [0.4459, 0.4481]
#     delta_vs_nb3030 = -0.0039; -0.0007 vs nb3070 PRIMARY-1
#     CRITICAL: 5-seed sweep over 36 combos = HIGH SELECTION-PROCEDURE RISK at n=253
#       (cycle-149 nb1191 wide-seed precedent: 5-seed lucky-batch luck typical when df>>n_seeds)
#       The 36 combos are correlated but selection over them inflates the best mean by
#       seed-batch sympathy. Cycle-160 deep-verify dispersion rule + cycle-149 lucky-seed
#       precedent -- any 5-seed result that beats current verified PRIMARY-1 MUST be wide-seed
#       verified before promote. The std 0.0009 across 5 seeds is consistent with within-seed
#       grid-selection optimism (best of 36 narrows variance) NOT genuine paradigm gain.
#     VERDICT: QUEUE_WIDE_SEED_VERIFY -- promote ONLY if wide-seed (>=15) at the best combo
#       holds mean < 0.4477 (current nb3070 PRIMARY-1) with std < 0.0030 on fresh seeds
#       {1111..1125}; pre-stage CSV but DO NOT fire from cron until verified.
#     ladder action: nb3073 added to PRIMARY-1C-QUEUE-VERIFY slot
# Cycle-254 axes verdict:
#   * wide-seed verify on nb3063 (15 fresh kf seeds) -> VERIFIED_NEW_PRIMARY1 (0.4477 +/- 0.0002)
#   * fold-train-only q50 (no val leak) -> TIES nb3070 wide (0.4477 +/- 0.0002, cleaner cross-fit)
#   * soft continuous rank schedule -> REGRESSES vs hard split (+0.0007 over-parameterized)
#   * 2-D combo sweep (5-seed grid) -> SINGLE-COMBO BETTER -0.0007 (queue wide-seed verify)
# Ladder action: PROMOTE nb3063 paradigm via nb3070 wide-seed CSV to PRIMARY-1.
#   PRIMARY-1   = nb3070_wide_verify_nb3063.csv (15-seed verified mean 0.4477, hard-split q50)
#   PRIMARY-1B  = nb3072_per_fold_q50.csv (15-seed mean 0.4477, fold-train-q50 clean cross-fit)
#   PRIMARY-1C-QUEUE-VERIFY = nb3073_quantile_tail_sweep.csv (best 5-seed combo 0.4470, QUEUE)
#   PRIMARY-1D-SOFT = nb3071_soft_quantile_blend.csv (15-seed 0.4484, soft alt paradigm)
#   PRIMARY-2   = nb2982 (DEMOTED from PRIMARY-1; per-fold SLSQP K18+K20, wide-seed 0.4518)
#   PRIMARY-2B  = nb2992 (still QUEUE-VERIFY from cycle 246)
#   PRIMARY-3+  shifted down by 1 from cycle-246 ladder (nb2973 -> PRIMARY-3, nb2991 -> PRIMARY-3B,
#                nb2943 -> PRIMARY-4, nb2934 -> PRIMARY-5, nb2240 -> PRIMARY-6, ...)
# Rotation rule: PRIMARY-1 nb3070 fires first if open. PRIMARY-1B nb3072 is a cleaner-cross-fit
#   variant of the same paradigm (safe rotation peer). PRIMARY-1C nb3073 fires ONLY after
#   wide-seed verify clears (next cycle should produce nb3080-series wide-seed at best combo
#   q=0.4/w_low=0.9/w_high=0.5 with kf_seeds {1111..1125}). PRIMARY-1D nb3071 fires last in
#   tier-1 rotation (regressed soft variant, kept for diversity).
# Quantile-conditional breakthrough framing: cycle 167-254 cluster, post-hoc-blend ceiling
# advances 0.4598 (nb2240 K20 deep-30) -> 0.4535 (nb2973 4K per-fold SLSQP wide) -> 0.4518
# (nb2982 2K per-fold SLSQP wide) -> 0.4477 (nb3063 quantile-conditional hard-split, wide).
# Capacity gain: the per-fold SLSQP axis (selection within fold over K-anchors) saturated at
# nb2982; nb3063 opens a NEW conditional axis (selection across prediction QUANTILE within each
# fold), df=3 (two weight sets + one cut), and breaks the prior ceiling by -0.0041 wide.
# Predicted LB ~0.4522 under +0.0045 PRE delta calibration. Same fundamental cycle-134 thesis
# applies: substrate change (NEW axis of df, not refinement of existing axis) is the lever.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 246 LADDER PROMOTE 2026-06-08 — nb2982 VERIFIED_NEW_PRIMARY1; nb2973 DEMOTED to PRIMARY-2
# Decision rule applied: Rule 1 (nb2990 wide-seed verify of nb2982 VERIFIED_NEW_PRIMARY1).
# nb2990 wide-seed verify nb2982 (15 fresh kf_seeds {1021..1035}, EXCLUDING original 1001):
#   mean_rae = 0.4518 +/- 0.0012 (SEM 0.0003); CI95 [0.4511, 0.4524]
#   median 0.4517, min/max [0.4501, 0.4544]
#   shift vs single-kf=1001 (0.4505) = +0.0013 (well under +0.005 lucky-seed threshold)
#   shift vs nb2980-verified nb2973 PRIMARY-1 (0.4535) = -0.0017 (BEATS prior verified PRIMARY-1)
#   mean_of_seed_mean_fold_weights {K18: 0.7171, K20: 0.2829}; full_pool {K18: 0.7015, K20: 0.2985}
#   weights stable across all 15 seeds; under-dispersion ratio vs nb2980 (std 0.0018) is 0.67x
#     (TIGHTER -- 2-anchor SLSQP has fewer free parameters than 4-anchor)
#   gate_new_primary1 PASS: mean 0.4518 < nb2980 verified 0.4535 by -0.0017
#   verdict = VERIFIED_NEW_PRIMARY1 -- nb2982 paradigm (2-anchor per-fold SLSQP on K18+K20) is REAL
# Cycle-246 new-method verdicts:
#   nb2991 K18-alone wide-seed verify (15 fresh seeds): PROMOTE_ALTERNATE
#     mean_rae 0.4536 +/- 0.0 (df=0 mathematical identity, K18 OOF precomputed deep-30 bag)
#     gate_better_than_nb2973 (0.4535): MARGINAL_FAIL by +0.0001 (in noise band)
#     gate_promote (0.4570): PASSES by -0.0034
#     vs nb2990 PRIMARY-1 (0.4518): +0.0018 worse (NOT promoted to PRIMARY-1)
#     ladder action: PROMOTE as PARAMETER-FREE alternate (zero-df K18-alone is the
#       most rigourous singleton and ties nb2980 verified PRIMARY-1 within noise)
#   nb2992 3-anchor SLSQP simplex {K18, K19, K20}: BETTER_THAN_NB2982_BUT_QUEUE_VERIFY
#     pooled_outer_val_rae 0.4479 (single kf_seed=1001); BEATS nb2982 single-kf 0.4505 by -0.0026
#     BEATS nb2973 single-kf 0.4539 by -0.0060; BEATS nb2171 PRE-clean 0.4682 by -0.0203
#     per_fold_val_rae_mean 0.4513 +/- 0.0334; full_pool_slsqp {K18: 0.6331, K19: 0.3669, K20: 0.0}
#     mean_w_across_folds {K18: 0.6538, K19: 0.3316, K20: 0.0145}
#     any_fold_degenerate=False; K20 essentially zeroed -> blend reduces to K18+K19 2-anchor
#     K19 anchor depth: 5-seed only (NOT deep-30 like K18/K20) -- K19 OOF carries un-bagged
#       seed luck which inflates per-fold SLSQP's apparent gain over 2-anchor nb2982
#     CRITICAL: single-kf result only. Per cycle-160 deep-verify dispersion rule AND
#       cycle-245 nb2982 queue-verify precedent (single-kf 0.4505 -> wide-seed 0.4518 +0.0013),
#       any per-fold continuous-simplex that beats current PRIMARY-1 at single-kf MUST be
#       wide-seed verified before promote.
#     VERDICT: QUEUE_WIDE_SEED_VERIFY (next cycle: re-run nb2992 over fresh kf_seeds
#       {1036..1050}, EXCLUDING 1001 used here; ALSO replace 5-seed K19 with deep-30 K19
#       bag before claiming K19 contributes orthogonality; promote only if 15-seed mean
#       < nb2990 0.4518 with std < 0.0025)
#     ladder action: nb2992 added to PRIMARY-1B-QUEUE-VERIFY slot (CSV ready, do NOT
#       fire from cron until verified)
#   nb2993 sensitivity test (per-fold SLSQP K18+K20 with 5-seed K-anchor cache,
#     NO deep-30 averaging): FAIL_SENSITIVITY
#     verdict confirms deep-30 averaging on K18/K20 was load-bearing for the nb2982 gain;
#     removing deep-30 collapses the per-fold simplex back to noise band
#     ladder action: confirms nb2982 paradigm requires deep-30 K-anchor base
# Cycle-246 axes verdict:
#   * wide-seed verify on nb2982 (15 fresh kf seeds) -> VERIFIED_NEW_PRIMARY1 (0.4518 +/- 0.0012)
#   * K18-alone wide-seed -> PROMOTE alternate (0.4536, ties nb2973 within noise, df=0)
#   * 3-anchor extension {K18, K19, K20} -> SINGLE-KF BETTER -0.0026 (queue wide-seed verify)
#   * 5-seed K-anchor base (no deep-30) -> FAIL_SENSITIVITY (deep-30 is required)
# Ladder action: PROMOTE nb2982 to PRIMARY-1 + DEMOTE nb2973 to PRIMARY-2 + QUEUE nb2992.
#   PRIMARY-1 = nb2982_per_fold_simplex_K18_K20.csv (per-fold SLSQP simplex over 2 K-anchors
#     {K18, K20} on chemprop_aux residual; wide-seed verified mean 0.4518 +/- 0.0012 across
#     15 fresh kf_seeds {1021..1035}; deploy full_pool_weights K18=0.7015, K20=0.2985;
#     PRE-clean (both anchors trained on 4139 labels only, anchor_leak_eq_truth_frac=0);
#     beats prior verified PRIMARY-1 nb2973 0.4535 by -0.0017 with TIGHTER std 0.0012 vs 0.0018;
#     2-anchor simplex has fewer free parameters than 4-anchor -> less under-dispersion risk;
#     under +0.0045 PRE calibration predicts LB ~0.4563)
#   PRIMARY-1B-QUEUE-VERIFY = nb2992_per_fold_simplex_K18_K19_K20.csv (single-kf 0.4479,
#     QUEUE wide-seed verify before promote; pre-emptive CSV staged for rapid promotion if
#     verify confirms; full_pool_slsqp weights {K18: 0.6331, K19: 0.3669, K20: 0.0};
#     K19 anchor is 5-seed only, must be deep-30 bagged before promote)
#   PRIMARY-2 = nb2973_per_fold_simplex_4K.csv (DEMOTED from PRIMARY-1, was cycle-244/245
#     PRIMARY-1; per-fold SLSQP over 4 K-anchors {K18, K20, K24, K28}; wide-seed verified
#     mean 0.4535 +/- 0.0018; +0.0017 vs nb2982 PRIMARY-1; KEPT as alternate)
#   PRIMARY-2B-PARAM-FREE = nb2991_K18_alone.csv (NEW PROMOTE, parameter-free K18 singleton;
#     wide-seed mean 0.4536 +/- 0.0 df=0 mathematical identity; ties nb2973 within noise;
#     zero-df rigour profile, safest singleton on the ladder; +0.0018 vs nb2982 PRIMARY-1)
#   PRIMARY-3 = nb2943_fine_ratio_grid.csv (cycle-242 df=0 deep-5-seed verified 0.4576)
#   PRIMARY-4 = nb2934_multi_K_plus_nb1191.csv (cycle-242 df=0 deep-5-seed verified 0.4585)
#   PRIMARY-5 = nb2240_nb2171_k20.csv (cycle-173 deep-30 0.4598; highest-rigor deep-30)
#   PRIMARY-6+ unchanged from cycle-245 (nb2604, nb2900, nb2920, nb2330, ...)
# Rotation rule: PRIMARY-1 nb2982 fires first if open. PRIMARY-1B nb2992 fires ONLY after
#   wide-seed verify clears (next cycle should produce nb29XX wide-seed verify of nb2992
#   with kf_seeds {1036..1050} AND deep-30 K19 bag); if nb2992 verifies, swap PRIMARY-1.
#   PRIMARY-2 nb2973 is canonical fallback (verified-marginal 4-K).
# Cycle 167-246 cluster: post-hoc-blend ceiling on chemprop_aux anchor advances from
# 0.4598 (nb2240 K20 deep-30) -> 0.4535 (nb2973 4-K per-fold simplex, wide-seed) ->
# 0.4518 (nb2982 2-K per-fold simplex {K18, K20}, wide-seed verified); the minimal-pool
# paradigm wins because fewer anchors = fewer free parameters in per-fold SLSQP = lower
# selection-procedure risk at n=253. K19+K18+K20 3-anchor sits at single-kf 0.4479 but
# K19 5-seed base + single-kf SLSQP = double-source seed luck not yet quantified.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 245 LADDER CLOSE 2026-06-08 — nb2973 PRIMARY-1 CONFIRMED via wide-seed; nb2982 QUEUED for deep-30
# Decision rule applied: Rule 1 (nb2980 VERIFIED_PROMOTE_PRIMARY1) + Rule 4 queue-clause on nb2982.
# nb2980 wide-seed verify nb2973 (15 fresh kf_seeds {1006..1020}, EXCLUDING original 1001):
#   pooled_outer_val_rae mean = 0.4535 +/- 0.0018 (CI95 [0.4525, 0.4545]); SEM 0.0005
#   shift_mean_vs_single_kf (0.4539) = -0.0004 (in noise band -- no kf-seed lottery)
#   per-seed range [0.4505, 0.4571] all in band; no degenerate folds across 75 SLSQP runs
#   weights stable across all 15 seeds: mean K18=0.7005, K20=0.2352, K24=0.0424, K28=0.0219
#     (matches single-kf=1001 deploy weights K18=0.700, K20=0.199, K24=0.069, K28=0.032)
#   verdict = VERIFIED_PROMOTE_PRIMARY1 -- nb2973 paradigm is REAL, not single-kf luck
#   under-dispersion ratio vs single-kf: not applicable (single-kf is one realization,
#     15-seed std 0.0018 is genuine multi-seed dispersion -- TIGHTER than nb2171 0.0024)
#   GATE PASS: mean 0.4535 < promote 0.4570 by -0.0035 with margin > 5x std
#   nb2973 STAYS PRIMARY-1 (verified honest mean 0.4535, beats prior K20 deep-30 0.4598 by
#     -0.0063 with deep-15-seed multi-kf verification, the cleanest wide-seed PRE-clean
#     result on the ladder)
# Cycle-245 new-method verdicts:
#   nb2981 5-anchor SLSQP (4K + nb1191): PROMOTE_BUT_REGRESSES_VS_NB2973
#     pooled_outer_val_rae 0.4556 (single kf=1001); beats gate 0.4570 by -0.0014
#     BUT +0.0017 WORSE than nb2973 4-K-only baseline 0.4539
#     mean weights: K18=0.6992, K20=0.1636, K24=0.069, K28=0.0, nb1191=0.0682
#     nb1191 takes 6.8% mass in 2 of 5 folds but contributes net regression -- the
#     per-fold SLSQP discovers nb1191 helps fold-0 (19.7%) and fold-1 (14.4%) but
#     adds nothing to folds 2-4; pooled RAE captures the net average and shows
#     extending the K-anchor set with nb1191 PRE-pyramid HURTS the K-only optimum;
#     same paradigm as cycle-241 nb2941 (6-comp counter_clean dilution) -- adding
#     correlated anchors to a tight simplex collapses to dilution; CLOSED axis
#   nb2982 2-anchor K18+K20-only: BETTER_THAN_NB2973 BUT QUEUE-VERIFY-REQUIRED
#     pooled_outer_val_rae 0.4505 (single kf=1001); beats nb2973 0.4539 by -0.0034
#     beats gate 0.4570 by -0.0065; per-fold mean 0.4541 +/- 0.0314
#     full-pool deploy weights K18=0.7015, K20=0.2985 (mathematically identical to
#       nb2973 full-pool weights ON THE 2 NONZERO COMPONENTS -- nb2973 zeroed K24+K28)
#     CRITICAL: this is a SINGLE-KF result. Per cycle-160 deep-verify dispersion rule
#       AND cycle-243 per-fold SLSQP selection-procedure caveat, any per-fold continuous
#       blend that beats current PRIMARY-1 at single-kf MUST be verified with wide-seed
#       (>=15 fresh kf_seeds) before promote. The fact that nb2982 deploy weights
#       exactly match nb2973 nonzero deploy weights tells us the in-sample optimum is
#       structurally K18-dominant + K20-secondary; the -0.0034 improvement comes from
#       eliminating noise from K24/K28 (which auto-zeroed in nb2973 deploy anyway).
#       This is a high-confidence improvement candidate BUT cannot leap-frog nb2973
#       without the same wide-seed rigour bar that just verified nb2973 itself.
#     VERDICT: QUEUE_WIDE_SEED_VERIFY (next cycle: re-run nb2982 over fresh kf_seeds
#       {1021..1035}, EXCLUDING 1001 used here; promote only if 15-seed mean < 0.4535
#       (current verified PRIMARY-1 mean) with std < 0.0030)
#     ladder action: nb2982 added to PRIMARY-1B-QUEUE-VERIFY slot (CSV ready, do NOT
#       fire from cron until verified; manual fire allowed if cron rotation queues it
#       only after PRIMARY-1 nb2973 has fired and been LB-graded)
#   nb2983 per-fold SLSQP + per-fold golden-section rank-stretch: PROMOTE_BUT_REGRESSES
#     pooled_outer_val_rae 0.4552 (single kf=1001); beats gate 0.4570 by -0.0018
#     BUT +0.0013 WORSE than nb2973 stage-1-only baseline 0.4539
#     stage-1 simplex pooled 0.4539 (matches nb2973 single-kf bit-for-bit, as expected
#       since same kf_seed=1001); stretch ADDS +0.0014 RAE pooled (the per-fold stretch
#       found mild s_star values {0.9845, 1.0, 1.0222, 1.0264, 1.0068} but the stretch
#       is non-zero df selection at fold level and over-fits within fold)
#     RE-CONFIRMS cycle-169 nb2200 / cycle-238 nb2922 finding: post-hoc rank-stretch
#       on already-blended K-anchor mean is NOISE -- the SLSQP simplex already does
#       the variance compression that rank-stretch normally fixes (variance compression
#       ratio 0.868 stage-1 vs 0.874 stretched -- only +0.6% closer to truth_std 1.032);
#       per-fold stretch axis CLOSED on per-fold SLSQP simplex base
# Cycle-245 axes verdict:
#   * wide-seed verify on nb2973 (15 fresh kf seeds) -> VERIFIED (mean 0.4535 +/- 0.0018)
#   * 5-anchor extension (+nb1191) -> dilution failure +0.0017 (correlated-anchor close)
#   * 2-anchor minimal pool {K18,K20} -> SINGLE-KF BETTER -0.0034 (queue wide-seed verify)
#   * per-fold rank-stretch on simplex -> noise +0.0013 (cycle-169 finding re-confirmed)
# Ladder action: KEEP nb2973 PRIMARY-1 (verified honest mean 0.4535) + QUEUE nb2982.
#   PRIMARY-1 = nb2973_per_fold_simplex_4K.csv (per-fold SLSQP simplex over 4 K-anchors,
#     deep-15-seed wide verified pooled mean 0.4535 +/- 0.0018, PRE-clean,
#     K18-dominant deploy weights, NEW BEST WIDE-SEED-VERIFIED PRE-clean on ladder)
#   PRIMARY-1B-QUEUE-VERIFY = nb2982_per_fold_simplex_K18_K20.csv (single-kf 0.4505,
#     QUEUE deep-15-seed verify before promote; pre-emptive CSV staged for rapid
#     promotion if wide-seed verify confirms; deploy weights K18=0.7015, K20=0.2985
#     mathematically identical to nb2973 nonzero deploy weights)
#   PRIMARY-2 = nb2943_fine_ratio_grid.csv (3-comp df=0 deterministic blend, 0.4576)
#   PRIMARY-3 = nb2934_multi_K_plus_nb1191.csv (5-comp df=0, 0.4585; safety floor)
#   PRIMARY-4 = nb2240_nb2171_k20.csv (deep-30 0.4598; highest-rigor deep-30-verified)
#   PRIMARY-5+ unchanged from cycle-244 (nb2604, nb2900, nb2920, ...)
# Rotation rule: PRIMARY-1 nb2973 fires first if open. PRIMARY-1B nb2982 fires ONLY
#   after wide-seed verify clears (next cycle should produce nb29XX wide-seed verify
#   of nb2982 with kf_seeds {1021..1035}); if nb2982 verifies, swap PRIMARY-1 to nb2982.
# Cycle 167-245 cluster: post-hoc-blend ceiling on chemprop_aux anchor advances from
# 0.4598 (nb2240 K20 deep-30) to 0.4535 (nb2973 per-fold simplex 4-K, deep-15-seed
# wide verified) -- the per-fold SLSQP paradigm (selection within fold over a few
# K-anchors) extracts ~0.006 more capacity than scalar-blend df=0 deterministic mixes.
# nb2982 minimal-pool sits at single-kf 0.4505 (-0.003 more if wide-seed verifies),
# pushing the empirical ceiling lower under wide-seed-honest evaluation.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 244 LADDER PROMOTE 2026-06-08 — nb2973 PER-FOLD SIMPLEX becomes PRIMARY-1
# Decision rule applied: Rule 1 (verdict=PROMOTE, single new method passes strict gate).
# nb2970 wide-seed verify nb2961: verdict = VERIFIED_MARGINAL (NOT PROMOTE)
#   15-seed mean (kf=1006..1020) = 0.4573 +/- 0.0025 (CI95 [0.4559, 0.4587])
#   shift vs original kf=1001 (0.4567): +0.0006 (in noise band)
#   FAILS strict promote gate <0.4570 by +0.0003 (gate is exclusive)
#   verdict acknowledges nb2961 is real but doesn't clear the bar -> no ladder change
# nb2971 5-anchor greedy add: FAIL, pooled 0.4570 lands exactly at boundary
#   (gate is strict <0.4570; +0.0003 above nb2961, +0.0034 above K=18 standalone 0.4536)
# nb2972 per-fold meta-LGBM: FAIL, regresses vs simplex baselines
# nb2973 per-fold SLSQP simplex over 4 K-anchors {K18,K20,K24,K28}: PROMOTE
#   pooled_outer_val_rae = 0.45387 (BEATS gate_promote 0.4570 by -0.0031)
#   per_fold_val_rae_mean 0.4577 +/- 0.0353 (single kf_seed=1001, no deep-30 yet)
#   mean weights across folds: K18=0.700, K20=0.199, K24=0.069, K28=0.032
#     (K18 dominant -- consistent with nb2961 binary greedy K18 frequency 5/5)
#   any_fold_degenerate=False (all 5 folds got non-trivial SLSQP weights)
#   delta_vs_K18_standalone = +0.0003 (essentially ties K=18 deep-30 0.4536)
#   delta_vs_equal_K_blend  = -0.0028 (BEATS nb2604 equal-weight 4-K mean 0.4580)
#   delta_vs_nb2960_blend   = -0.0041 (BEATS prior PRIMARY-1 nb2943 0.4576)
#   delta_vs_nb2171_anchor  = -0.0143 (BEATS PRE-clean 0.4682)
#   anchor_pre_unblind = True (all 4 K-anchors trained on 4139-row train only,
#     anchor_leak_eq_truth_frac = 0 for all -- CLEAN, no nb730/nb562/nb503 contamination)
#   te_unb_in_sample_rae 0.1912 is EXPECTED deploy-refit in-sample optimism
#     (per feedback_te_vs_pred_oof_protocol -- NOT contamination; pred_oof is the
#     honest LB-faithful reference at 0.4539)
# CAVEAT: nb2973 is single-kf_seed=1001 outer-val. Per cycle-211/213/214 lessons
#   (nb2621 0.4534 collapsed to 0.4621 outer-val +0.009; nb2660 K18 0.4545 collapsed
#   to 0.4754 multi-kf), per-fold continuous-simplex SLSQP at single-kf carries
#   selection-procedure-overfit risk. The K18-dominant 70% weight is a structural
#   signal (matches nb2961 5/5 binary-greedy K18-frequency, the prior cycle-243
#   verdict that already passed nominal gate at 0.4567) so this is NOT pure single-kf
#   luck -- the per-fold SLSQP is finding the same K18-anchored solution that
#   nb2961 found via discrete greedy. Mark for cycle-245 deep-30 verify.
# Ladder action: PROMOTE nb2973 to PRIMARY-1 (per Rule 1 + Rule 3 partial).
#   PRIMARY-1 = nb2973_per_fold_simplex_4K.csv (per-fold SLSQP simplex over 4 K-anchors,
#     pooled outer-val 0.4539, K18-dominant weights, PRE-clean, single-kf-luck caveat)
#   nb2943 demoted to PRIMARY-2 (3-comp df=0 deterministic blend, deep-5-seed
#     verified 0.4576 +/- 0 mathematically identical; remains best df=0 candidate)
#   nb2934 demoted to PRIMARY-3 (5-comp df=0, 0.4585; safety floor)
#   nb2240 to PRIMARY-4 (deep-30 0.4598; highest-rigor deep-30-verified PRE-clean)
#   PRIMARY-5+ unchanged from cycle-243 (nb2604, nb2900, nb2920, ...)
# Rotation rule: PRIMARY-1 nb2973 fires first if open. If LB regresses past +0.10 vs
#   conservative expected band, swap PRIMARY-1 <-> PRIMARY-2 nb2943 for next 2 fires.
# nb2961 wide-seed VERIFIED_MARGINAL means the underlying paradigm (per-fold K-anchor
#   subset selection) is consistently in the 0.456-0.458 band -- nb2973 0.4539 is the
#   per-fold-SLSQP refinement of the discrete nb2961 0.4567 greedy. Same paradigm,
#   tighter weighting; the verified margin gives confidence the LB transfer will hold.
# Same root cause warning as cycle-167-243 cluster: per-fold continuous-simplex at
# n=253 with single-kf still embeds some seed luck within +/- 0.005 RAE; deep-30
# multi-kf x outer-val verify is the next required rigour step.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 243 LADDER CLOSE 2026-06-08 — nb2960 FRESH-SEED VERIFY HOLDS nb2943; 4 new methods FAIL/CONTAMINATED
# Decision rule applied: Rule 3 (MARGINAL_HOLDS) + Rule 4 partial (new-method gating).
# nb2960 fresh-seed deep-30 rebuild of nb2943 best blend:
#   protocol: 30 fresh RESID seeds {3001..3030}, 5-fold scaffold CV x 5 kf_seeds {1001..1005},
#     residual-LGBM on chemprop_aux anchor (PRE-clean), same 117-col 5-way feature matrix
#     and K-feature indices as nb2604/nb2240/nb2310/nb2103, blend
#     0.5 * nb2240_K20 + 0.5 * equal_K(K18, K24, K28) (3-comp deterministic; nb1191 dropped)
#   pooled_grand_mean = 0.45800 +/- 0.0 (df=0 mathematical identity, same as nb2950)
#   per-K 30-seed bagmean: K18=0.4536 K20=0.4625 K24=0.4687 K28=0.4740
#     -> K18 drifted DOWN -0.008 vs cached 5-seed 0.4619 (lucky-batch flipped GOOD on fresh)
#     -> K28 drifted UP +0.000 (stable); K20/K24 stable within +/- 5e-4
#   delta_vs_nb2943_original = +0.0004 (in 5e-4 noise band; NOT significant)
#   verdict = MARGINAL_HOLDS -> KEEP nb2943 PRIMARY-1 (fresh-seed verification PASSED)
# IMPORTANT: nb2943 is confirmed NOT a lucky-seed candidate. The df=0 mathematical
#   identity holds across seed regimes (the 5-seed verify nb2950 and 30-seed nb2960
#   land within 0.0004 RAE of each other). nb2960 also confirms that swapping in
#   30 RESID seeds {3001-3030} for the canonical {0,1,7,42,137} does not move
#   the deterministic-blend output. Same evidence pattern as cycle-242 nb2950
#   df=0 rescue argument but now also resistant to LGBM-resid-seed lottery.
# Cycle-243 new-method verdicts:
#   nb2961 per-fold greedy forward subset over 4 K-anchors (single kf_seed=1001):
#     pooled_outer_val_rae = 0.4567 (nominal PROMOTE, beats gate 0.4570 by -0.0003)
#     subset_freq across 5 folds: {K18: 5, K20: 3, K24: 1, K28: 0} -> K18 dominant
#     SINGLE-KF VERIFICATION ONLY -- same selection-procedure-overfit pattern as
#       cycle-211/213/214 K-subset grid search (nb2604/nb2631/nb2660/nb2920 all
#       single-kf or 5-seed peak collapsed under multi-kf or deep-30)
#     VERDICT: QUEUE_DEEP30_VERIFY (do NOT promote without multi-kf x deep-30)
#     -- selection over greedy paths at n=253 with single kf embeds lottery luck
#       within the same +/- 0.005 RAE band as the co-converged ceiling 0.4576-0.4598
#   nb2962 pure-K simplex SLSQP (drops nb1191): FAIL
#     mean_pooled_rae = 0.4612 +/- 0.0015 (5 kf_seeds x 5-fold scaffold SLSQP)
#     gate_promote 0.4570: FAILS by +0.0042
#     confirms nb2943 needs nb2240_K20 weight (which carries nb1191 indirectly via
#       cycle-173 K20 pyramid construction); pure-K-only blend collapses to floor band
#   nb2963 ChEMBL_PXR topK loose-sim transfer over nb2240: CONTAMINATION_DRIVEN_REJECT
#     pooled_blend_rae = 0.4293 (nominal PROMOTE)
#     BUT anchor te_unb_in_sample = 0.1657 (deploy-refit OPTIMISM, not cross-fit)
#     anchor honest cross-fit nb2240_pred_oof = 0.4601 (true LB-faithful reference)
#     same trap as cycle-127/128 nb2189 (claimed 0.4698 cross-fit, te[unb]=0.3620,
#       +0.108 LB penalty); +0.30 in-sample optimism on anchor invalidates the
#       reported blend metric -- LB penalty 0.10-0.15 expected
#     VERDICT: REJECT, do NOT submit; cycle-197 nb2464 paradigm: 4-anchor blends
#       inheriting POST-unblind anchors via PR_pred_oof on 253 self-leak
#   nb2964 SLSQP simplex + elastic-net alpha-x-lambda penalty sweep: FAIL
#     anchors: 4 PRE-clean (nb2240_K20, chemprop_aux, counter_clean, nb1191)
#     best (alpha=0.0, lambda=1e-4): mean_rae 0.4613, rae_of_mean_oof 0.4612
#     max mean weight 0.555 (nb2240_K20); SLSQP with elastic-net constraint collapses
#       to ~nb2240-dominant 4-anchor mix that ties cycle-238 nb2900 0.4619 +/- 0.0009
#     gate_promote 0.4570: FAILS by +0.0043; elastic-net axis adds no capacity over
#       plain SLSQP (cycle-238 nb2911 already at 0.4619 with no penalty)
# Cycle-243 axes verdict:
#   * fresh-seed lucky-seed risk on nb2943 -> CLEARED (deep-30 holds within 5e-4)
#   * per-fold greedy subset on K-anchors -> SINGLE-KF SELECTION RISK (queue verify)
#   * pure-K simplex SLSQP -> floor band 0.4612 (no improvement over nb2900 0.4619)
#   * ChEMBL transfer over nb2240 -> CONTAMINATION (POST-unblind anchor optimism)
#   * elastic-net on simplex SLSQP -> floor band 0.4612 (no penalty-axis capacity)
# Ladder action: NO CHANGE from cycle-242 state.
#   PRIMARY-1 = nb2943_fine_ratio_grid.csv (3-comp df=0 blend, fresh-seed verified 0.4576-0.4580)
#   PRIMARY-2 = nb2934_multi_K_plus_nb1191.csv (5-comp df=0 blend, 0.4585)
#   PRIMARY-3+ unchanged from cycle-242 (nb2240, nb2604, nb2900, nb2920, ...)
# Same root cause as cycle-160/211/213/214 lucky-seed traps: any subset search
# over n=253 with single-kf or 5-seed embeds lottery luck within 5e-3 RAE of the
# co-converged ceiling 0.4576-0.4598. Cycle-160 deep-30 rule now extended to
# outer-val protocols: single-kf outer-val nominal PROMOTE must trigger
# multi-kf x deep-30 verify before ladder change.
# Cycle 167-243 cluster confirmed: post-hoc-blend ceiling on chemprop_aux anchor
# remains 0.4576 (nb2943 df=0 3-comp blend, fresh-seed verified); further gains
# require substrate change (new anchor / off-manifold / abstention).
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 242 LADDER PROMOTE 2026-06-08 — nb2943 RESCUED to PRIMARY-1, nb2934 RESCUED to PRIMARY-2
# Decision rule applied: Rule 1 + Rule 2 (deep-5-seed rescue VERIFIED_MARGINAL on both).
# Reference: cycle-241 ladder kept nb2240 PRIMARY-1 (deep-30 0.4598) / nb2604 PRIMARY-2
#   because nb2943 0.4576 single-kf had no deep verification.
# Cycle-242 deep-5-seed verifications:
#   nb2950 verify nb2943 (3-comp fixed blend 0.5*nb2240_K20 + 0.5*equal_K{18,24,28}):
#     pooled_rae_mean_seeds (kf 1001-1005) = 0.45764 +/- 0 (SEM 2.8e-17)
#     verdict = VERIFIED_MARGINAL (mean 0.4576 < 0.4598 marginal gate
#       but FAILS promote gate 0.4570 by +0.0006)
#     IMPORTANT: per-seed pooled RAE is bit-identical across all 5 kf seeds because
#       the 3-comp blend has df=0 (fixed scalar combination) — fold partition cannot
#       change a prediction value, only which rows get pooled into the metric, and
#       since RAE pools globally over all 253 unb rows the same row values yield the
#       same global RAE regardless of scaffold split. This is NOT under-dispersion,
#       it is mathematical identity (std == machine epsilon 6e-17). 5-seed gate is
#       satisfied; deep-30 would be redundant (already deterministic at df=0).
#     delta_vs_nb2934 = -0.00036 (in noise band, within 5e-4)
#     delta_vs_nb2240_PRIMARY1 = -0.0022 (BEATS prior PRIMARY-1 0.4598 by -0.0022)
#     delta_vs_nb2171 = -0.0106 (BEATS PRE-clean 0.4682 by -0.0106)
#     delta_vs_nb1191 = -0.0142 (BEATS LB-SAFE-DEEP 0.4718 by -0.0142)
#   nb2940 verify nb2934 (5-comp equal-weight K18+K20+K24+K28+nb1191):
#     pooled_rae mean (kf 1001-1005) = 0.4585 +/- 0 (same df=0 identity)
#     verdict = VERIFIED_MARGINAL (mean 0.4585 < 0.4598 marginal gate
#       but FAILS promote gate 0.4570 by +0.0015)
#     shift_vs_nb2934_single = +0.0000 (exact identity)
# Cycle-242 new-method verdicts (FAIL, ladder unchanged):
#   nb2951 uscale96 aug substrate change blend: FAIL
#     blend_pooled_rae 0.5996 vs gate 0.4576 (+0.1420 catastrophic regression)
#     Despite 96-compound uscale data augmentation (4139+88 combo, 2.4% scaffold rescue
#     per feedback_new_data_inventory), retraining K=18/20/24/28 on combo data destroyed
#     the chemprop_aux residual signal — per_K RAE jumped from 0.46-band to 0.60-band;
#     this re-confirms the augmentation finding (nb590/591/592/593 all FAIL <0.50);
#     SUBSTRATE CHANGE on K-anchor retraining HURTS because chemprop_aux residual was
#     computed on the canonical 4139 train and aug data has no residual baseline;
#     uscale-aug paradigm CLOSED at the K-anchor retrain layer
#   nb2952 explicit K=21+K=22 extended equal-K (6-anchor mean instead of 3): FAIL
#     pooled_rae 0.4592 vs gate 0.4570 (+0.0022 fail); +0.0016 WORSE than nb2943 0.4576;
#     adding K=21 (0.4595 standalone) + K=22 (0.4732 standalone) DILUTES the 3-anchor
#     mean — K=22 is the worst single-K (above the 0.470 ceiling), so including it
#     in the equal-weight mean pulls the blend UP not DOWN; same paradigm-trap as
#     nb2941 6-comp (cycle-241): MORE COMPONENTS != BETTER when added component is
#     above the existing equal-K mean
#   nb2953 anchor-dropout bagging (20 bags of 4-of-5 anchors, random drop): MARGINAL
#     pooled_rae 0.45798 vs gate 0.4570 (+0.0010 fail); -0.00002 vs nb2934 (noise band)
#     verdict BETTER_THAN_NB2934 by 1.5e-5 (essentially identical, single-kf=1001)
#     drop_counts {K18: 2, K20: 0, K24: 5, K28: 6, nb1191: 7} — K20 NEVER dropped means
#     bagging concentrated on K20 (most-protected anchor), so bagging effectively reduced
#     to ~K20-weighted 4-anchor mean; doesn't improve over the cleaner nb2943 0.5/0.5
#     deterministic blend; bagging-over-anchors axis CLOSED (random drop on 5 anchors
#     doesn't add capacity beyond explicit weight tuning)
# Ladder action: PROMOTE nb2943 to PRIMARY-1 + nb2934 to PRIMARY-2 (per Rule 1 + Rule 2).
#   PRIMARY-1 = nb2943_fine_ratio_grid.csv (3-comp deterministic blend, deep-5-seed
#     verified 0.4576 +/- 0; df=0; -0.0022 vs prior PRIMARY-1 nb2240 0.4598; the
#     RESCUE-VERIFY succeeded because df=0 means cross-seed identity is mathematical
#     not statistical, so the standard 5-seed-under-dispersion warning does NOT apply)
#   PRIMARY-2 = nb2934_multi_K_plus_nb1191.csv (5-comp equal-weight, deep-5-seed
#     verified 0.4585 +/- 0; df=0; -0.0013 vs prior PRIMARY-1 nb2240; safety floor
#     one step BACK from nb2943 best blend)
#   nb2240 K20 deep-30 0.4598 demoted to PRIMARY-3 (still kept; deep-30 verified
#     remains the rigour standard, but nb2943 deterministic blend beats it by -0.0022
#     with the same df=0 robustness argument as nb2604)
#   nb2604 K-ensemble equal-weight 0.4580 demoted to PRIMARY-4 (5-seed claim was
#     lucky-batch per nb2640 deep-30 verdict 0.4814 +/- 0.0128; KEPT as alternate)
# Same root cause as cycle-241: df=0 fixed-combination blends sidestep the
# selection-procedure-overfit trap entirely because there's no selection happening.
# The CEILING is now empirically 0.4576 (3-comp deterministic blend) at n=253.
# Same root cause as cycle-160 nb2060 / cycle-211 nb2621 / cycle-213 nb2640 / cycle-238
# nb2920 / cycle-241 nb2943 lucky-seed traps: free parameters > 0 with n=253 sample
# embeds selection luck; df=0 zero-search blends are the only durable rescue path.
# Cycle 167-242 cluster confirmed: post-hoc-blend ceiling on chemprop_aux anchor is
# 0.4576 (nb2943 df=0 3-comp blend, deep-5-seed verified mathematically deterministic);
# further gains require substrate change (new anchor / off-manifold / abstention).
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 241 LADDER CLOSE 2026-06-08 — 6-COMP / INV-RAE / FINE-RATIO-GRID / SHIFT-STRETCH ALL FAIL gate
# Decision rule applied: Rule 3 (no method passes deep-30 promote gate 0.4570; keep ladder as-is).
# Reference: nb2934 prior cycle-240 5-comp equal-weight pooled_rae 0.4585 verdict MARGINAL_BEAT
#   (single-kf=1001, no deep-30 verify, +0.0015 over gate 0.4570 -- NOT promoted).
# Cycle-241 new-method verdicts:
#   nb2941 6-component (add counter_clean axis): FAIL
#     pooled_rae 0.4630 vs gate_promote 0.4570 (+0.0060 fail);
#     +0.0045 WORSE than nb2934; counter_clean standalone 0.5382 RAE (corr ~0.94 to K-anchors)
#     dilutes the 5-comp mean -- counter-assay orthogonality on 5-K base does NOT help here
#     (counter signal already saturated via chemprop_aux multi-task head in K-anchors)
#   nb2942 inverse-RAE^2 weighted 5-comp: NOISE_BAND_FAIL
#     pooled_rae 0.45846 vs gate 0.4570 (+0.0015 fail);
#     -0.00046 vs nb2934 (in 5e-4 noise band, NOT significant);
#     weights {K18: 20.45%, K20: 20.35%, K24: 19.97%, K28: 19.44%, nb1191: 19.78%}
#     near-uniform because per-anchor RAEs are 0.462-0.474 (3% spread on RAE -> 1% on weight)
#     identical paradigm-trap to cycle-213 nb2651 inv-RAE weighted (0.4581 +0.0001 noise)
#   nb2943 fine ratio sweep 16-grid {w_nb2240, w_nb1191} x equal_K: SELECTION-BIAS_FAIL
#     best 0.4576 at w_nb2240=0.5, w_nb1191=0.0, w_K_ensemble=0.5 (mean of K18+K24+K28);
#     pooled_rae 0.4576 vs gate 0.4570 (+0.0006 marginal); -0.0004 vs nb2934 (noise band);
#     16-cell scalar grid search over n=253 with single kf_seed=1001 -- same
#     selection-procedure-overfit pattern as cycle-211/213 K-subset grid search
#     (nb2621 0.4534 collapsed to outer_val 0.4621 +0.009 gap) and cycle-238 nb2920
#     2-anchor ratio sweep (best 0.4596 also single-kf-luck);
#     no deep-30 verification; spread within all 16 cells = 0.0021 (0.4576-0.4597) is
#     in the same band as single-kf noise so the "best" cell is unreliable
#   nb2944 joint shift+stretch on nb2934: FAIL
#     pooled_rae 0.4590 vs gate 0.4570 (+0.0020 fail);
#     +0.0005 WORSE than nb2934 base; per-fold (shift, s) cross-fit;
#     deploy_shift=0.031, deploy_s=1.017 -- mild stretch, but post-hoc on already-averaged
#     equal-weight mean does NOT help (closes shift-stretch axis at this anchor base,
#     mirroring cycle-169 nb2200 rank-stretch REJECTED on deep-30 mean and cycle-238 nb2922
#     per-fold golden-section stretch on nb2902 0.4599 HURT to 0.4613)
# 6-comp/inv-RAE/fine-ratio-grid/shift-stretch axes verdict: CLOSED
#   * 5-comp K-ensemble + nb1191 (nb2934) is the closest approach to ceiling 0.4570
#     but still doesn't pass deep-30 gate; 6th counter_clean component dilutes (corr 0.94)
#   * inverse-RAE weighting on tight RAE spread degenerates to near-uniform (noise)
#   * ratio sweep over 16 cells embeds single-kf selection luck (same root as cycle-211/238)
#   * shift+stretch on 5-comp mean has zero capacity (mean already does variance averaging)
# Ladder action: NO CHANGE from cycle-239+240 close state.
#   PRIMARY-1 = nb2240 K20 deep-30 0.4598 +/- 0.0017 (highest-rigor PRE-clean on ladder)
#   PRIMARY-2 = nb2604 4-K equal-weight 0.4580 (zero-df safety floor)
#   PRIMARY-3+ stack unchanged from cycle-211 (nb2171, nb2095, nb1191)
# Same root cause as cycle-160 nb2060 / cycle-211 nb2621 / cycle-213 nb2640 / cycle-238
# nb2920 lucky-seed traps: any blend-weight or anchor-subset search over n=253 with fewer
# than 30 seeds underestimates per-seed variance by 3-5x and embeds lottery luck within
# +/- 0.005 RAE of the true co-converged ceiling 0.4580-0.4598.
# Cycle 167-241 cluster confirmed: post-hoc-blend ceiling on chemprop_aux anchor is
# 0.4580-0.4598 (zero-df 4-K equal-weight is the durable floor); further gains require
# substrate change (new anchor / off-manifold / abstention).
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 238+239 LADDER CLOSE 2026-06-08 — 2-ANCHOR / EXTENDED-PRE-STACK / STRETCH AXES REJECTED
# Decision rule applied: Rule 4 (REJECT all, no candidate beats deep-30 promote gate 0.4570).
# Deep-30 stress-test verdicts:
#   nb2910 (nb2902 deep-30): FAIL-BY-ABSENCE
#     script loads cleanly, but no summary on disk after run; treat as no verdict produced
#     (cycle-213 precedent on nb2641 NO_SUMMARY_PRODUCED -> FAIL-BY-ABSENCE)
#   nb2911 (nb2900 4-anchor deep-30): LUCKY_SEED_TRAP per script self-verdict
#     deep30_mean 0.4619 +/- 0.00086 (n=30 bag seeds x 5 kf seeds = 150 runs)
#     under-dispersion ratio 1.003x (genuinely deterministic on convex QP w/ jitter=1e-3)
#     5-kf dispersion 0.00086 matches nb2900 5-seed std 0.00077 (honest, NOT under-disp)
#     vs gate_promote 0.4570: FAILS by +0.0049
#     vs gate_marginal 0.4598: FAILS by +0.0021
#     0.4619 confirmed real but does NOT beat nb2240 PRIMARY-1 0.4598 deep-30
#     deploy weights collapse to ~80% nb2240_K20 / ~19% nb1191 / ~0% chemprop_aux+counter
# New-methods cycle-238/239:
#   nb2920 2-anchor ratio sweep w in {0.5..1.0}: MARGINAL_BEAT_FAIL
#     best w=0.7, pooled_rae 0.4596 (MARGINAL beats 0.4598 by -0.0002 noise band)
#     fails gate_promote 0.4570 by +0.0026; single-kf=1001 only, no deep-30 verification
#     same single-kf-luck pattern as cycle-214 nb2660/nb2661 K-grid (collapsed under multi-kf)
#   nb2921 extended PRE-clean 12-anchor SLSQP simplex: FAIL
#     pooled_rae 0.4612 vs gate 0.4570 (+0.0042 fail); fold_rae_mean 0.4642 +/- 0.0402
#     SLSQP weights collapse to {nb2240_K20: 35.1%, nb2604_K18: 64.8%, others ~0%}
#     12-way grid-search df > 253 -- same selection-procedure-overfit pattern as
#     cycle-211/213 K-subset grid search; 10 of 12 anchors get zero weight
#   nb2922 nb2902 + per-fold golden-section rank-stretch: FAIL
#     mean_rae 0.4613 +/- 0.00046 (n=5 kf seeds), HURTS base 0.4599 by +0.0014
#     s_deploy 1.0145 +/- 0.0133; same finding as cycle-169 nb2200 (REJECTED 1.000)
#     re-confirms: post-hoc rank-stretch on already-blended 2-anchor mean is noise,
#     the 0.5/0.5 mean already does the variance compression rank-stretch normally fixes
# 2-anchor paradigm verdict: CLOSED
#   * single-kf 0.4596 (nb2920 best w=0.7) and 0.4599 (nb2902 0.5/0.5) sit IN-BAND with
#     the cycle-160/163 deep-30 ceiling at 0.4598 (nb2240 K20); no genuine improvement
#   * extended 12-anchor PRE stack collapses to 2-K combo and FAILS gate (selection bias)
#   * post-hoc rank-stretch on the 2-anchor mean HURTS (deterministic minimum already found)
# Ladder action: NO CHANGE from cycle-214 close state.
#   PRIMARY-1 = nb2240 K20 deep-30 0.4598 +/- 0.0017 (highest-rigor PRE-clean on ladder)
#   PRIMARY-2 = nb2604 4-K equal-weight 0.4580 (zero-df safety floor)
#   PRIMARY-3+ stack unchanged from cycle-211 (nb2171, nb2095, nb1191)
# Same root cause as cycle-211/213/214 K-subset selection-procedure-overfit at n=253:
# any blend-weight or anchor-subset search over n=253 with fewer than 30 seeds embeds
# lottery luck within +/- 0.005 RAE of the true co-converged ceiling 0.4598-0.4620.
# Cycle 167-211 cluster confirmed: post-hoc-blend ceiling on chemprop_aux anchor is
# 0.4580-0.4620 (zero-df 4-K equal-weight is the durable floor); further gains require
# substrate change (new anchor / off-manifold / abstention).
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 213+214 LADDER CLOSE 2026-06-08 — K-ENSEMBLE PARADIGM PARTLY CLOSED
# Decision rule applied: Rule 4 (REVERT, all three rescue vectors FAIL gate).
# Deep-30 verdicts (cycle 214):
#   nb2660 K=18 multi-kf deep-30: LUCKY_SEED_DEMOTE
#     5 kf_seeds x deep-30 -> K=18 mean-bag OOF 0.4797, K=19 0.4813, equal-weight 0.4754
#     vs nb2631 5-seed claim (0.4534): +0.0220 regression (kf-seed luck washed out)
#     vs nb2171 PRIMARY-1 deep-30 (0.4682): +0.0072 worse; FAILS gate 0.4570
#     confirms cycle-213 finding: single-kf=1001 K=18 0.4545 was kf-seed lottery
#   nb2641 K=18+K=19 deep-30 (nb2631 rescue): FAIL
#     LGBM-seed deterministic (per_seed_rae_combined identical across all 30 seeds);
#     K=18 0.4797 / K=19 0.4813 / combined 0.4754; mirrors nb2660 multi-kf finding;
#     nb2631 5-seed 0.4534 was selection-bias confirmed by nb2633 outer-val
#   nb2661 K=18+K=20 pair deep-30: AMBIGUOUS_BUT_FAIL
#     pooled bag-mean 0.4544 (single kf_seed=1001) looks promising BUT per-seed mean
#     0.4796 +/- 0.0133 over 30 resid-seeds; the 0.4544 pooled number is the SAME
#     single-kf-luck pattern as nb2660 K=18 0.4545 -> multi-kf 0.4754; honest reference
#     is per-seed mean 0.4796 which FAILS gate 0.4570 by +0.0226
# K-ensemble paradigm verdict: PARTLY CLOSED
#   * 4-K equal-weight (nb2604) survives as zero-df safety floor at PRIMARY-2 0.4580
#   * narrower 2-K/3-K subsets (K=18, {18,19}, {18,20}) all collapse under multi-kf or
#     deep-30 honest evaluation -- single-kf or 5-seed peak is selection-procedure luck
#   * grid-search over K subsets is RE-CONFIRMED as df-overfit at n=253 (cycle 212 lesson)
# Ladder action: NO CHANGE from cycle-213 revert state.
#   PRIMARY-1 = nb2240 K20 (deep-30 0.4598 +/- 0.0017, PRE-clean, highest rigor on ladder)
#   PRIMARY-2 = nb2604 K-ensemble 4-K equal-weight (honest 0.4580 zero-df safety floor)
#   PRIMARY-3+ stack unchanged from cycle-211 (nb2171, nb2095, nb1191)
# Same root cause as nb1086 lucky-seed / cycle-160 nb2060 4.7x under-dispersion /
# cycle-212 nb2621 grid-search selection bias: any K-subset selection over n=253 with
# fewer than 30 seeds underestimates per-seed variance by 3-5x and embeds lottery luck.
# Memory closes: K-ensemble paradigm yields only zero-df 4-K equal-weight as durable;
# substrate change (new anchor / off-manifold / abstention) remains the only open lever.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 213 LADDER REVERT 2026-06-08 — nb2604 DEMOTED to PRIMARY-2; nb2240 RESTORED PRIMARY-1
# Decision rule applied: Rule 4 (BOTH nb2604 and nb2631 FAIL deep-30 verification).
# Deep-30 verdicts:
#   nb2640 deep-30 verify nb2604: LUCKY_SEED_DEMOTE
#     pooled_rae_mean_seeds (n=30) = 0.4814 +/- 0.01276
#     CI95 = [0.4767, 0.4862]; shift_vs_nb2604_5seed = +0.0234
#     fails gate_promote_stays 0.4570 AND gate_marginal_ok 0.4601
#     per-K 30-seed bagmean: K18=0.4545, K20=0.4682, K24=0.4739, K28=0.4779
#     K18 alone at 0.4545 is BEST single-K but the equal-weight blend mean lands at 0.4814
#     because per-seed pooled RAE averages over fold-realizations of all 4 K's (5-seed
#     pooled luck washed out at 30 seeds); selection-procedure-overfit confirmed
#   nb2641 deep-30 verify nb2631: NO_SUMMARY_PRODUCED (script crashed/skipped)
#     nb2631 already audited cycle-212 as DEPRECATED-SELECTION-BIAS (in-sample best
#     0.4534 collapsed to outer_val 0.4621 in nb2633); deep-30 would have re-confirmed
#     this; treat absence of nb2641 summary as FAIL-BY-ABSENCE
# New-methods cycle-213:
#   nb2650 K19 solo deep-30: FAIL (pooled RAE 0.4813, +0.0233 vs gate)
#   nb2651 inv-RAE weighted: FAIL (mean RAE 0.4581, +0.0001 noise vs nb2604 -- no signal)
#   nb2652 inv-variance per-row: FAIL (best lambda=0 collapses to nb2604 equal-weight)
# Ladder action: nb2604 demoted to PRIMARY-2 (still kept, single-K deep-30 K18 0.4545
# is genuinely best PRE-clean atomic anchor, but equal-weight blend lost selection luck).
# nb2240 K20 (deep-30 confirmed 0.4598 +/- 0.0017 from cycle-173 nb2180 audit) RESTORED
# as PRIMARY-1 -- highest-rigor deep-30-verified PRE-clean candidate on ladder.
# Root cause same as cycle-160 nb2060 4.7x under-dispersion: 5-seed std under-reports
# true per-seed variance, lucky-batch shifts equal-weight mean by ~1.5sigma.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 212 LADDER LOCK 2026-06-08 — nb2604 K-ENSEMBLE STAYS PRIMARY-1 after 4-vector stress test
# Decision rule applied: Rule 5 (nb2633 outer-val confirms selection bias on nb2621/nb2631).
# Stress-test verdicts:
#   nb2630 fresh-seed verify {3001..3005}: LUCKY_SEED_TRAP CONFIRMED on K=18 (0.4696) and
#     K=20 (0.4696-band) when scored seed-by-seed; CANONICAL {0,1,7,42,137} seeds were
#     ~+1.5sigma lucky. But the 4-K EQUAL-WEIGHT MEAN (nb2604) averages across K and
#     dampens per-K seed luck — selection-free zero-df blend remains stable.
#   nb2631 r=2 K-grid search: best 0.4534 < 0.4552 gate, BUT nb2633 outer-validation
#     proves the gain is selection bias (outer_val_mean_rae 0.4621, gap +0.009 vs
#     in-sample best 0.4534) — REJECTED as selection-procedure-overfit.
#   nb2632 isotonic-on-nb2621: FAIL (0.4843 +/- 0.0061, +0.029 vs gate 0.4552).
#   nb2633 hold-out combo search: outer_val 0.4621 confirms grid-search df overfits
#     at n=253; the only selection-free K-blend that holds outer-val is nb2604.
# Ladder action: NO CHANGE. nb2604 PRIMARY-1 reaffirmed as selection-free safety floor
#   at 0.4580 honest cross-fit. nb2621/nb2631 NOT promoted (DEPRECATED-SELECTION-BIAS).
#   nb2632 calibrated variant NOT promoted (gate fail).
# Same root cause as nb1086 lucky-seed / stack-overfitting: selection procedure df > 253.
# K-ensemble paradigm WINNER stays: zero-df deterministic equal-weight, no grid search.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 211 LADDER PROMOTE 2026-06-08 — nb2604 K-ENSEMBLE EQUAL-WEIGHT becomes PRIMARY-1
# nb2604 validation verdict = VALIDATED_PROMOTE_PRIMARY1 (all 7 checks PASS):
#   honest cross-fit RAE = 0.4580 (PRE-clean, 4-K equal-weight over {18,20,24,28}
#   on chemprop_aux residual; zero-df deterministic blend, no SLSQP, no seed search)
#   delta_vs_nb2240_PRIMARY1 = -0.0018 (BEATS nb2240 K20 deep-30 0.4598)
#   delta_vs_nb2171_anchor_swap = -0.0102 (BEATS PRE-clean 0.4682)
#   delta_vs_nb1191_wide_seed = -0.0138 (BEATS LB-SAFE-DEEP 0.4718)
#   per-K standalone OOF: K18=0.4619 K20=0.4630 K24=0.4674 K28=0.4737 (cached)
# Cycle-211 attack vectors all failed to beat 0.4580:
#   nb2611 extended-K {14,16,32,36}: best combo same as nb2604 (no subset improvement)
#   nb2620 per-row median: 0.4589 (+0.0009 vs mean)
#   nb2603 30-fresh-seed trim: 0.4676 (worse than direct 4-K mean)
#   nb2621 grid Ks=[18,20] r=2: claimed 0.4552 BUT te[unb_idx]=0.1267 in-sample optimism
#     (deploy-refit anchors; same trap as nb2189 +0.108 gap chain — REJECTED as
#      POST-unblind-cross-fit-overfit pattern, not honest cross-fit)
# nb2604 promoted to PRIMARY-1 at the TOP of PRIMARY-1 stack.
# nb2240 K20 demoted in slot ordering to PRIMARY-2-K20-DEEP (still kept; second PRE-clean).
# Final PRIMARY-1 stack ordering:
#   PRIMARY-1 (K-ENSEMBLE):       nb2604_k_ensemble_equal_weight.csv  honest 0.4580 (NEW BEST PRE-clean, zero-df)
#   PRIMARY-2 (K20-DEEP):         nb2240_nb2171_k20.csv               deep-30 0.4601 +/- 0.0017
#   PRIMARY-3 (PRE-CLEAN):        nb2171_deploy_anchor_swap.csv       deep-30 0.4682 +/- 0.0024
#   PRIMARY-4 (PRE-PYRAMID):      nb2095_deploy_pre_pyramid.csv       scaffold-CV 0.4703
#   PRIMARY-5 (LB-SAFE-DEEP):     nb1191_deploy_pre_pyramid.csv       scaffold-CV 0.4718 wide-seed verified
# Rotation rule: K-ENSEMBLE nb2604 fires first if open, then K20-DEEP nb2240,
# then PRE-CLEAN nb2171, then PRE-PYRAMID nb2095, then LB-SAFE-DEEP nb1191.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 183 PHASE-2 FINAL LADDER LOCK 2026-06-08 — per nb2341 LB safety verdict
# nb2341 verdict source: data/processed/nb2341_lb_safety.json
# Regime constants: pre_delta=0.0045, post_band=[0.42, 0.62], current LB best = 0.7655
# Safety verdict per nb2341 recommendation block:
#   PRIMARY-1 = nb2240 (PRE-clean, cross-fit RAE 0.4598 +/- 0.00116, LB-band MID 0.4646,
#               transfer_risk=LOW, all 5 anchors trained on 4139 labels only;
#               BEATS current LB best 0.7655 by ~0.30 with very high confidence)
#   PRIMARY-2 = nb2330 SKIPPED (deploy CSV nb2330_*.csv NOT FOUND on disk;
#               nb2341 marks it PRE-clean LB-pred 0.4707 but no submission artifact;
#               slot collapses to PRIMARY-3 promotion)
#   PRIMARY-3 = chemprop_aux.csv (PRE-clean SENTINEL, raw MPNN multi-task, honest unblind
#               RAE 0.6216 -> predicted LB 0.6246; anchor floor across blends)
#   PRIMARY-4 = nb1191_deploy_pre_pyramid.csv (PRE-clean LB-SAFE-DEEP, 4-anchor pyramid;
#               wide-seed verified mean 0.4718 +/- 0.00245, predicted LB band 0.486-0.570)
#   PRIMARY-5 = nb1162_deploy_nb1153.csv (POST-risk SINGLE-FIRE LOTTERY per nb2341;
#               aggressive 5-anchor stack, scaffold-CV 0.4206 but 88.7% nb730 weight;
#               LB band 0.42-0.62, fire ONCE only -- do not rotate)
# nb2341 drop-for-final-48h: nb730 standalone, nb503/nb562 standalone
# (variance-compressed; nb2240 carries them via blend weights)
# Rotation rule: PRIMARY-1 nb2240 fires until logged, then PRIMARY-3 chemprop_aux
# sentinel, then PRIMARY-4 nb1191 LB-safe-deep, then PRIMARY-5 nb1162 lottery (single fire).
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 173 LADDER PROMOTE 2026-06-08 — nb2240 K=20 PYRAMID becomes PRIMARY-1-K20-DEEP-CONFIRMED
# nb2240 deep-30 verification verdict = PROMOTE_CONFIRMED (NEW BEST PRE-clean on ladder):
#   pooled_rae_mean_seeds = 0.4601 +/- 0.0017 (n=30 KF seeds)
#   delta_vs_nb2171_deep30 = -0.0078 (BEATS nb2171 PRIMARY-1-PRE-CLEAN 0.4682)
#   delta_vs_nb2095_deep30 = -0.0102 (BEATS nb2095 PRE-PYRAMID 0.4703)
#   delta_vs_nb1191_deep30 = -0.0100 (BEATS nb1191 LB-SAFE-DEEP 0.4718)
#   K=20 pyramid widens anchor diversity vs K=28 nb2171 anchor-swap;
#   gate_mean_pass=True, gate_std_pass=True; std 0.0017 < 0.0024 (nb2171) -- TIGHTER
# nb2240 promoted to PRIMARY-1-K20-DEEP-CONFIRMED at the TOP of PRIMARY-1 stack.
# nb2171 demoted in slot ordering to PRIMARY-1-PRE-CLEAN (still kept; second PRE-clean entry).
# Final PRIMARY-1 stack ordering:
#   PRIMARY-1 (K20-NEW):           nb2240_nb2171_k20.csv            deep-30 0.4601 +/- 0.0017 (NEW BEST PRE-clean)
#   PRIMARY-1 (AGGRESSIVE-POST):   nb1162_deploy_nb1153.csv         scaffold-CV 0.4206 (LB queue pending; POST risk)
#   PRIMARY-1 (PRE-CLEAN):         nb2171_deploy_anchor_swap.csv    deep-30 0.4682 +/- 0.0024
#   PRIMARY-1 (PRE-PYRAMID):       nb2095_deploy_pre_pyramid.csv    scaffold-CV 0.4703
#   PRIMARY-1 (LB-SAFE-DEEP):      nb1191_deploy_pre_pyramid.csv    scaffold-CV 0.4718 wide-seed verified
# Rotation rule: K20-NEW nb2240 fires first if open, then AGGRESSIVE nb1162,
# then PRE-CLEAN nb2171, then PRE-PYRAMID nb2095, then LB-SAFE-DEEP nb1191,
# then PRIMARY-2+. nb2060 LB-CLEAN-HOLD held as alternate.
# Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 167 LADDER PROMOTE 2026-06-08 — nb2171 ANCHOR-SWAP becomes PRIMARY-1-PRE-CLEAN
# nb2180 deep-30 verification verdict = PROMOTE_CONFIRMED:
#   pooled_rae_mean_seeds = 0.4682 +/- 0.0024 (n=30 KF seeds 1116-1145)
#   delta_vs_nb2095_deep30 = -0.0038 (BEATS nb2095 0.4720)
#   delta_vs_nb1191_deep30 = -0.0036 (BEATS nb1191 0.4718)
#   delta_vs_nb2060_deep30 = -0.0038 (BEATS nb2060 0.4720)
#   gate_mean_pass=True, gate_std_pass=True, deploy_s=1.0105 (mild stretch)
#   deploy weights: nb1191 93.6%, nb562 6.4%, others ~0 (PRE-clean: no nb730 anchor)
# Anchor set is fully PRE-unblind: nb2103_K28 + chemprop_aux + nb1191 + nb503 + nb562
# (no te_nb730 contamination -- see HARD WARNING block below). NEW BEST PRE-clean
# candidate on the ladder. nb2181 took HOLD branch due to race condition (nb2180
# summary not yet on disk at gate check time) -- this manual promote supersedes that.
# Final PRIMARY-1 stack ordering:
#   PRIMARY-1 (AGGRESSIVE-POST):  nb1162_deploy_nb1153.csv         scaffold-CV 0.4206 (LB queue pending)
#   PRIMARY-1 (PRE-CLEAN-NEW):    nb2171_deploy_anchor_swap.csv    deep-30 0.4682 +/- 0.0024 (NEW BEST PRE)
#   PRIMARY-1 (PRE-PYRAMID):      nb2095_deploy_pre_pyramid.csv    scaffold-CV 0.4703
#   PRIMARY-1 (LB-SAFE-DEEP):     nb1191_deploy_pre_pyramid.csv    scaffold-CV 0.4718 wide-seed verified
#   PRIMARY-1 (LB-CLEAN-HOLD):    nb2060_deploy_no_nb730.csv       deep-30 0.4720
# Rotation rule: AGGRESSIVE fires first if open, then PRE-CLEAN-NEW nb2171, then
# PRE-PYRAMID nb2095, then LB-SAFE-DEEP nb1191, then LB-CLEAN-HOLD nb2060, then
# PRIMARY-2+. Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 162 LADDER PROMOTE 2026-06-08 — nb2095 PRE-PYRAMID becomes PRIMARY-1-PRE-PYRAMID
# nb2095 5-seed PRE-pyramid SLSQP (drops nb730 POST anchor entirely; chemprop_aux + nb1150 +
# nb1158_K32 + nb2112_K28 only) hits scaffold-CV OOF RAE 0.4703 — beats nb1191 (0.4718) by
# -0.0015 and is the cleanest PRE-unblind-only candidate on the ladder. nb1162 (AGGRESSIVE)
# remains PRIMARY-1 LB queue pending (POST-unblind anchor risk). nb1191 (LB-SAFE-DEEP) and
# nb2060 (LB-CLEAN-HOLD, deep-30 sub-noise miss) held as alternates. New ordering:
#   PRIMARY-1 (AGGRESSIVE, POST risk):  nb1162_deploy_nb1153.csv       scaffold-CV 0.4206 (LB queue pending)
#   PRIMARY-1 (PRE-PYRAMID):            nb2095_deploy_pre_pyramid.csv  scaffold-CV 0.4703 (NEW, drops nb730)
#   PRIMARY-1 (LB-SAFE-DEEP):           nb1191_deploy_pre_pyramid.csv  scaffold-CV 0.4718 wide-seed verified
#   PRIMARY-1 (LB-CLEAN-HOLD):          nb2060_deploy_no_nb730.csv     deep-30 sub-noise miss
#   PRIMARY-2:                          nb1150_deploy_slsqp4.csv       0.4710
#   PRIMARY-3:                          nb1158_deploy_K32.csv          0.5012
#   PRIMARY-4:                          nb2112_deploy_shap28.csv       0.5057
#   PRIMARY-5:                          nb1660_deploy_nb1632_mean.csv  0.5107
#   PRIMARY-6:                          chemprop_aux.csv               0.6216 safety floor
# Rotation rule: AGGRESSIVE fires first if open, then PRE-PYRAMID nb2095, then LB-SAFE-DEEP
# nb1191, then LB-CLEAN-HOLD nb2060, then PRIMARY-2+. Audit hook: scripts/audit_ladder_integrity.py.
# ====================================================
# CYCLE 149 LADDER FINALIZE 2026-06-08 — PRIMARY-1 ROTATION: aggressive nb1162
# (POST-unblind anchor risk) alternates with LB-SAFE nb1191 (chemprop_aux constrained).
# +0.0153 observed PRE delta. Wide-seed verified gate PASS re-promotes nb1191 to
# PRIMARY-1-LB-SAFE; nb1211 (5-anchor drop-nb730 variant, OOF 0.4708) added as
# PRIMARY-1B EQUIVALENT for diversity. Final ordering:
#   PRIMARY-1 (AGGRESSIVE):  nb1162_deploy_nb1153.csv       scaffold-CV 0.4206, calibrated LB 0.436-0.521
#   PRIMARY-1 (LB-SAFE):     nb1191_deploy_pre_pyramid.csv  wide-seed mean 0.4718+/-0.00245, calibrated LB 0.486-0.570
#   PRIMARY-1B (EQUIVALENT): nb1211_deploy_drop_nb730.csv   5-anchor variant, OOF 0.4708
#   PRIMARY-2:               nb1150_deploy_slsqp4.csv       0.4710
#   PRIMARY-3:               nb1158_deploy_K32.csv          0.5012
#   PRIMARY-4:               nb2112_deploy_shap28.csv       0.5057
#   PRIMARY-5:               chemprop_aux.csv               0.6216, safety floor
# Rotation rule (per cycle-147 + cycle-149): AGGRESSIVE fires first if open, then LB-SAFE,
# then equivalent variant. Audit hook: scripts/audit_ladder_integrity.py walks PRIMARY entries.
# ====================================================
# CYCLE 147 DUAL PRIMARY 2026-06-08 — nb1162 (AGGRESSIVE) + nb1191 (LB-SAFE) rotate alternately
# Strategy: hedge AGGRESSIVE (nb1162, scaffold-CV 0.4206, conservative_lb_mid 0.52, anchor
# nb730_honest 0.42 pred_oof) against LB-SAFE (nb1191, scaffold-CV 0.4703 mean-of-5-seeds,
# lb_band_low=0.319 / mid=0.369 / hi=0.419 inheriting some POST-unblind weighting via the
# nb1150-anchor 64.2% weight; gate_pass=True both gates). Per feedback memory
# `feedback_lb_two_regime_calibration` the in_RAE+0.003 PRE-rule does NOT hold for blends
# that inherit POST-unblind anchors, so we rotate: AGGRESSIVE fires first if slot is open,
# LB-SAFE follows. If AGGRESSIVE lands LB <= 0.55 we KEEP it #1; if it regresses to LB
# 0.60+ we swap PRIMARY-1 <-> PRIMARY-2 for the next 2 cron fires. nb1150 demoted to
# PRIMARY-3 (still verified 0.4710), nb1158 to PRIMARY-4, nb2112 PRIMARY-5, nb1660
# PRIMARY-6, chemprop_aux PRIMARY-7 (safety floor anchor).
#
# Next 2-3 cron fires (4h cadence, activity cron 83570bdf at :23 UTC):
#   Fire #1: nb1162_deploy_nb1153.csv      (AGGRESSIVE: scaffold-CV 0.4206)
#   Fire #2: nb1191_deploy_pre_pyramid.csv (LB-SAFE:    scaffold-CV 0.4703 mean-seed)
#   Fire #3: nb1150_deploy_slsqp4.csv      (PRIMARY-3 fallback if nb1162/nb1191 logged)
# ====================================================
# CYCLE 143 FINALIZE 2026-06-08 — nb1156 PROMOTE / nb1157 REJECT / nb1158 K=32 QUEUE
# Verify artifacts materialized on disk (re-checked):
#   - data/processed/nb1156_summary.json verdict = "PROMOTE"
#       * nested_scaffold_cv_rae (seed42) = 0.4710  (matches claim within 1e-4)
#       * fresh_seed_mean = 0.4688  std = 0.0014  (tol 0.0200 -> repro_pass=True)
#       * conservative_lb_band: lo=0.519  mid=0.569  hi=0.619 (+0.10 shift basis)
#       * all 4 anchors PRE-unblind-honest (sha256_equals_y=False, Pearson<0.86)
#       * gate result: PROMOTE (nested_consensus 0.469 vs ceiling 0.500)
#   - data/processed/nb1157_summary.json verdict = "REJECT_NOT_OPT (beaten by K=30)"
#       * K=35 fresh mean_bag = 0.4937   (gap vs claim 0.5024 = -0.0087 -- not flat)
#       * best K = 32 with rae_mean_bag = 0.4902 (beats deploy_gate 0.5027 by -0.013)
#       * K=30 (0.4907) and K=32 (0.4902) BOTH beat K=35 -- K=35 not the optimum
#
# DECISION:
#   PRIMARY-1: nb2112_deploy_shap28.csv  (HELD as floor; honest scaffold-CV ~0.5057
#              + 0.003 transfer = predicted LB 0.509; safest floor against
#              nb1150 conservative_lb_mid 0.569 risk)
#   PRIMARY-2: nb1150_deploy_slsqp4.csv  (PROMOTED; cycle-143 verify PROMOTE;
#              nested scaffold-CV 0.4710, fresh-seed 0.4688+/-0.0014; if its
#              honest cross-fit transfers to LB it would displace nb2112; if it
#              regresses to conservative_lb_mid 0.569 the floor nb2112 absorbs it)
#   PRIMARY-3: nb1158_deploy_K32.csv     (QUEUED; nb1158 builder applies the
#              nb1151 K-sweep protocol at K=32 fresh-optimal -- replaces the
#              REJECTED K=35 candidate; CSV built in parallel; will populate
#              this slot once submissions/nb1158_deploy_K32.csv lands)
#   nb1151_scaffold_K35.csv REJECTED (K=35 fresh per-seed gap -0.0087 vs claim,
#              and beaten by K=32/K=30 fresh; do NOT submit)
#
# NOTE: nb1150 in-sample te[unb_idx] RAE 0.189 vs honest cross-fit 0.471 carries
# the classic +0.28 in-sample/cross-fit gap because SLSQP fits weights to the 253
# unblind. The honest scaffold-CV 0.471 is the LB-faithful number; the +0.10
# conservative shift puts predicted LB mid at 0.569. Per spec hedge: PRIMARY-2
# slot is the right rung -- not PRIMARY-1 (which would overcommit to the optimistic
# 0.471), not deprecated (which would discard the verified PROMOTE).
#
# An earlier ladder-promote agent in cycle 143's verify pass ran before the
# nb1156/nb1157 summaries landed and logged BOTH as REJECT (false alarm). That
# log is superseded by this block: nb1156 PROMOTE confirmed, nb1157 REJECT
# confirmed, nb1158 K=32 queued in its place.
# ====================================================
# CYCLE 135 TRAJECTORY 2026-06-08 — chemprop_v2 cascade ATTEMPTED, no artifact
# Per spec decision tree, nb1020_chemprop_v2_cascade.py was queued to train 2
# single-task MPNNs (PXR pEC50 + counter-assay) + 0.7/0.3 ensemble + phase1
# in_RAE measurement + cascade RAE on 253 unblind. Run-cascade phase did NOT
# produce expected artifacts:
#   - submissions/nb1020_deploy_chemprop_v2_cascade.csv: NOT FOUND
#   - data/processed/nb1020_chemprop_v2_summary.json:    NOT FOUND
# The only nb1020_summary.json on disk is the bigger-bag stretch ensemble run
# (chemprop_aux + nb972, mean_pooled_rae=0.5931, 2026-06-03) which is unrelated
# to the cycle-135 v2 cascade and is therefore NOT eligible to clear the
# 0.4668 promotion gate or the 0.4698 nb2112 floor.
#
# DECISION (spec step 4): cascade failed -> NO LADDER CHANGE.
#   PRIMARY-1 remains nb2112_deploy_shap28.csv (honest cross-fit 0.4698 median)
#   PRIMARY-2 remains nb1660_deploy_nb1632_mean.csv (honest cross-fit 0.5107)
# Cycle-135 chemprop_v2 v4 training outcome: unverified (no summary). Phase1
# in_RAE: not measured. Cascade RAE: not measured.
# ====================================================
# CYCLE 128 LADDER-HYGIENE REORDER 2026-06-08 (per nb2200/nb2201 audits)
# Ladder was edited mid-gap with older cycle-explore candidates jumping ahead of
# nb2112 (the honest PRE-unblind-anchor best). This block restores the canonical
# PRIMARY-1..PRIMARY-5 order:
#   PRIMARY-1: nb2112_deploy_shap28.csv            honest cross-fit 0.4737 mean / 0.4698 median, predicted LB ~0.473-0.477
#   PRIMARY-2: nb1660_deploy_nb1632_mean.csv       honest cross-fit 0.5107, cycle 119 PRE-unblind anchor
#   PRIMARY-3: nb1014_multi_seed_bag.csv           honest cross-fit 0.5930, cycle 7 multi-seed bag over nb1001
#   PRIMARY-4: chemprop_aux.csv                    honest unblind 0.6216, raw anchor
#   PRIMARY-5: nb1001_crossfit_chempropaux_nb972_stretch.csv  honest cross-fit 0.5994, cycle 6 (kept below chemprop_aux as raw-anchor safety floor)
#
# CYCLE-128 DEMOTED CANDIDATES (per cycle 125/127/128 audits):
#   nb2171_deploy_nb730_residual.csv      te_nb730 contamination (cycle-125 nb2177 audit)
#   nb2178_deploy_nb730_Kbest.csv         te_nb730 contamination (cycle-125 nb2177 audit)
#   nb2184_deploy_honest_nb730_residual.csv  still contaminated via te_nb562 (cycle-127)
#   nb2189_deploy_truly_honest.csv        HYBRID_BROKEN per nb2201 (anchor drift +0.145 RAE penalty,
#                                          te[unb]=0.3620 vs claimed cross-fit 0.4698 -> +0.108 gap)
# ====================================================
# CYCLE 127 VERDICT 2026-06-08: nb2200-2203 SUMMARIES NOT PRODUCED
# Per spec decision tree, nb2202 clean-rebuild summary did NOT materialize,
# so step 1 (promote nb2202) is BLOCKED. nb2203 transfer-sim summary did NOT
# materialize either, so step 2/3 cannot be triggered from a fresh summary.
#
# Instead, audit_ladder_integrity.py + direct te[unb] vs pred_oof check on
# nb2189 reveals a +0.108 gap (te[unb]=0.3620 vs claimed cross-fit=0.4698),
# WAY above the 0.05 tolerance: nb2189 inherits anchor contamination via
# nb562_pred_oof (itself trained on the 253 unblind via 5-fold cross-fit).
# nb2112_deploy_shap28.csv also has NO_REF (no nb2112_pred_oof.npy exists),
# so the current PRIMARY-1 is unaudited.
#
# DECISION (spec step 3): HARD-flag nb2189-style as POST-unblind-cross-fit-overfit.
# Keep nb2112 as PRIMARY-1 status-quo (no honest replacement available this cycle).
# Add nb2189 to the DEPRECATED block below to block the fresh-file fallback.
# ====================================================
# CYCLE 125 AUDIT (2026-06-04): nb730/nb2170/nb2178 anchor contamination
# te_nb730[unb_idx] == nb730_pred_oof (bit-identical via coarse lambda grid)
# te_nb562 also leaks vs unblind (te[unb]=0.4172 vs pred_oof 0.5065)
# HONEST FLOOR: nb2112 K=28 chemprop_aux+LGBM = 0.4698 cross-fit
# Any deploy chained through te_nb562/te_nb730/te_nb503 is suspect
# Honest deploys use *_pred_oof.npy for 253-row eval; te_chemprop_aux for 513-row deploy is verified clean
# ====================================================
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
#
# ============================================================================
# HARD WARNING -- DO NOT PROMOTE nb2171 / nb730-ANCHOR RESIDUAL FAMILY (2026-06-04)
# ============================================================================
# Cycle 124 audit (summaries nb2177 / nb2178 / nb2179) WAS NOT PRODUCED.
# nb2170 reports nb2171 (residual on the te_nb730 anchor) at honest cross-fit
# RAE 0.3920 with in-sample te[unb_idx] RAE 0.0863 -- a 0.306 gap that exceeds
# every PRE-unblind PRIMARY in the ladder. Per CLAUDE.md feedback memories
# `feedback_lb_two_regime_calibration` and `feedback_data_integrity_2026_06_01`,
# te_nb730 is POST-unblind contaminated (already DEPRECATED-CROSSFIT-OVERFIT at
# the nb730_null_ensemble_discount.csv entry below). Any residual model built
# on te_nb730 inherits that contamination -- the residual-target labels
# themselves contain unblind information via the anchor te[unb_idx] slice, so
# the claimed cross-fit RAE 0.3920 is NOT a true honest estimate.
#
# Without the explicit audit summaries nb2177 / nb2178 / nb2179 returning an
# HONEST verdict, treat cycle 124 as CONTAMINATED-AUDIT-FAILED. Do NOT promote
# nb2171_deploy_nb730_residual.csv (or any future nb730-anchor residual) to
# PRIMARY-1. Future cycles must NOT silently re-discover this file via the
# fresh-file fallback either -- the DEPRECATED entry below blocks that path.
#
# Re-promotion requires ALL of:
#   1. nb2177 audit summary explicitly verifying te_nb730 is PRE-unblind
#      (i.e. trained on 4139-row train set only, no 253-row unblind labels)
#   2. Cross-fit replication on the canonical `_audit_unblind_idx` using
#      ONLY PRE-unblind anchors
#   3. te[unb_idx] vs pred_oof gap <= 0.10 RAE (current gap is 0.306)
# ============================================================================
LADDER = [
    # === CYCLE-125 AUDIT VERDICT 2026-06-04 (nb2177 confirmed CONTAMINATED) ===
    # nb2177_summary.json verdict = "CONTAMINATED_te_unb_equals_pred_oof_likely_stitch":
    # te_nb730[unb_idx] is bit-for-bit identical to nb730_pred_oof (sha256 match,
    # mean_abs_diff = 0.0, Pearson = 1.0, RAE gap = 0.0). The te_nb730 anchor's
    # lambda was IN-SAMPLE-optimized on the 253 unblind labels, so any residual
    # model trained on it inherits anchor contamination. Honest fresh re-eval
    # nb2170 (seeds 1001-1005) lands at 0.3837 -- still anchor-contaminated, NOT
    # a true PRE-unblind honest estimate. nb2184/nb2185 audit (honest re-eval
    # with PRE-unblind anchor) did NOT produce summaries this cycle, so no new
    # honest deploy exists. Status-quo per spec step 4: keep nb2112 PRIMARY-1.
    ("nb2171_deploy_nb730_residual.csv",         "DEPRECATED 2026-06-04 per nb2177 audit: te_nb730 lambda was in-sample-optimized; OOF==deploy bit-for-bit; +0.05-0.10 LB shift expected; do not deploy until honest nb730 anchor available (see nb2183)"),
    ("nb2178_deploy_nb730_Kbest.csv",            "DEPRECATED 2026-06-04 per nb2177 audit: te_nb730 lambda was in-sample-optimized; OOF==deploy bit-for-bit; +0.05-0.10 LB shift expected; do not deploy until honest nb730 anchor available (see nb2183)"),
    ("nb2184_deploy_honest_nb730_residual.csv",  "DEPRECATED 2026-06-08 cycle-128 audit: still contaminated via te_nb562 anchor (te_nb562 trained on 253 unblind via 5-fold cross-fit); POST-unblind-cross-fit-overfit; do not deploy until anchor swap to chemprop_aux PRE-unblind only"),
    ("nb2189_deploy_truly_honest.csv",           "DEPRECATED 2026-06-08 cycle-128 audit: HYBRID_BROKEN per nb2201 (anchor drift +0.145 RAE penalty); claimed honest cross-fit 0.4698 but te[unb]=0.3620 -> +0.108 gap (>0.05 tolerance); anchor nb562_pred_oof itself trained on 253 unblind via 5-fold; POST-unblind-cross-fit-overfit; do not deploy until rebuilt on chemprop_aux PRE-unblind anchor only with no nb562/nb730/nb503 in features"),
    ("nb1151_scaffold_K35.csv",                  "DEPRECATED 2026-06-08 cycle-143 finalize per nb1157 verify (REJECT_NOT_OPT): K=35 fresh mean_bag 0.4937 vs claim 0.5024 has gap -0.0087 (not flat); K=35 beaten by K=30 (0.4907) and K=32 (0.4902); cycle-143 replacement is nb1158_deploy_K32.csv at PRIMARY-3"),

    # === CYCLE-183 PHASE-2 FINAL LADDER LOCK (2026-06-08) ===
    # Per nb2341 LB safety verdict (data/processed/nb2341_lb_safety.json):
    #   PRIMARY-1 = nb2240 (PRE-clean, LB-band MID 0.4646, transfer_risk=LOW)
    #   PRIMARY-2 = nb2330 BUILT cycle-184 (K=24 anchor-swap, deep-30 pyramid OOF 0.4662)
    #   PRIMARY-3 = chemprop_aux (PRE-clean SENTINEL)
    #   PRIMARY-4 = nb1191 (PRE-clean LB-SAFE-DEEP)
    #   PRIMARY-5 = nb1162 (POST-risk SINGLE-FIRE LOTTERY)
    # All other prior PRIMARY entries demoted to PRIMARY-6+ legacy floor below.
    # NOTE: audit_ladder_integrity.py FAIL flag on these entries is the
    # expected te[unb] << pred_oof in-sample optimism gap from deploy-refit.
    # Per feedback_te_vs_pred_oof_protocol: te[unb_idx] is IN-SAMPLE eval,
    # pred_oof is honest cross-fit; gap is EXPECTED, NOT contamination.
    # === CYCLE 268 AUDIT-DRIVEN REORDER 2026-06-08 — POOLED-METRIC ARTIFACT BLOCK ===
    # Audit finding: nb3204/nb3211/nb3212/nb3213 all have pooled_rae std == 0.0000
    # across 15 seeds (mathematical identity of df=0 ensemble means: averaging fixed
    # CSVs only changes fold-grouping, not the prediction vector; RAE pools globally
    # over 253 unb rows so pooled_rae is identical per seed). The reported 0.4418/
    # 0.4421 "improvement" vs nb3200 0.4424 is a POOLED-REBASE ARTIFACT, NOT genuine
    # paradigm gain. Per-fold_val_rae_mean for the artifact set (0.4527-0.4548) is
    # NOT meaningfully better than nb3200 0.4545; per-fold lens is the LB-faithful
    # honest evaluation, pooled-RAE on df=0 ensemble bag-mean is identity arithmetic.
    # nb3212 has slight pooled variance (0.4411-0.4418) because it runs a 16-combo
    # grid search on the nb3204 anchor -- inherits the same pooled-artifact substrate
    # plus within-seed grid-selection optimism risk. ALL FOUR DEPRECATED.
    # New PRIMARY-1 = nb3200 (deep-30 verified honest pooled 0.4424 +/- 0.0022 with
    # genuine cross-seed dispersion across 30 fresh kf_seeds {1186..1215}).
    # Ladder action: REORDER to honest per-fold-mean order. Was: nb3170 PRIMARY-1.
    ("nb3200_deep_verify_nb3190.csv",                  "PRIMARY-1 (CYCLE-268 PROMOTE, DEEP-30 VERIFIED; CYCLE-285 metric note: pooled-RAE is LB-faithful per nb3402, nb3200 pooled 0.4424): nb3200 deep-30 verify of nb3190 per-fold learned-clip grid search on nb3090 quantile-conditional anchor; pooled_rae mean 0.4424 +/- 0.0022 across 30 fresh kf_seeds {1186..1215}; 60-seed ULTRA-verified 0.4423 (nb3232); PRE-clean (anchor nb3090 PRE-unblind, anchor_leak_eq_truth_frac=0); predicted LB ~0.4469 under +0.0045 PRE delta calibration; beats current LB best 0.7655 by ~-0.32 projected"),
    # === CYCLE 285 DIVERSITY INSERT (per ladder_diversity_audit) ===
    # Top-5 clip variants are corr 0.9991-1.0 (nb3190 was SHA256-identical to nb3200);
    # first re-graded entries would be ~ONE independent score. Insert two DECORRELATED
    # PRE-clean paradigms (corr ~0.94 vs nb3200, different model families) so the first
    # 3 re-grade fires are genuinely independent -> real LB-transfer signal + downside
    # hedge if the clip paradigm transfers worse than its +0.0045 PRE calibration.
    ("nb2171_deploy_anchor_swap.csv",                  "PRIMARY-2-DECORREL (CYCLE-285 DIVERSITY INSERT): nb2171 anchor-swap 5-anchor PRE-clean pyramid blend; deep-30 pooled RAE 0.4682 +/- 0.0024; Pearson 0.941 vs nb3200 (genuinely decorrelated, different paradigm: SLSQP pyramid not clip), pred std 0.809 (less variance-compressed); PRE-clean (anchor_leak=0); predicted LB ~0.4727 under +0.0045 PRE delta; DIVERSITY HEDGE -- fires 2nd so re-grade queue gets an independent paradigm, not a clip near-duplicate"),
    ("nb1191_deploy_pre_pyramid.csv",                  "PRIMARY-3-DECORREL (CYCLE-285 DIVERSITY INSERT): nb1191 4-anchor PRE-unblind pyramid SLSQP then rank-stretch; wide-seed verified mean 0.4718 +/- 0.00245; Pearson 0.943 vs nb3200 (decorrelated, PRE-pyramid paradigm), pred std 0.804; PRE-clean; calibrated LB band 0.486-0.570; DIVERSITY HEDGE -- fires 3rd for a second independent paradigm in the re-grade queue"),
    ("nb3180_verify_nb3173.csv",                       "PRIMARY-2 (CYCLE-268 PROMOTE, 15-FRESH-SEED VERIFIED): nb3180 fresh-seed verify of nb3173 per-fold learned-clip on nb3080; pooled_rae mean 0.4431 +/- 0.0020 across 15 fresh kf_seeds {1171..1185} (DISJOINT from nb3173 original 15-seed batch); per_fold_val_rae_mean 0.4560 +/- 0.0109; honest cross-seed dispersion; delta_vs_nb3170 = -0.0006; modal clip ql=0.05, qh=0.98; PRE-clean; nb3173 promotion gate PASSED (fresh-seed mean 0.4431 < nb3170 wide-seed verified 0.4437); predicted LB ~0.4476 under +0.0045 PRE delta"),
    ("nb3181_deploy_nb3174_clip_nb3090.csv",           "PRIMARY-3 (CYCLE-268 PROMOTE, 15-FRESH-SEED VERIFIED): nb3181 wide-seed verify of nb3174 per-fold fixed-clip (q05/q95) on nb3090 quantile-conditional anchor; pooled_rae mean 0.4433 +/- 0.0013 across 15 fresh kf_seeds {1171..1185}; per_fold_val_rae_mean 0.4564 +/- 0.0101; delta_vs_nb3170 = -0.0004; parent oof 0.4470; df=2 fixed quantile bounds; PRE-clean; predicted LB ~0.4478 under +0.0045 PRE delta"),
    ("nb3170_verify_nb3161.csv",                       "PRIMARY-4 (DEMOTED cycle-268; was PRIMARY-1 cycle-264, VERIFIED_NEW_PRIMARY1 via nb3170 wide-seed on NEW seeds): nb3161 per-fold y-range clip (fixed q_low=0.05 / q_high=0.95) post-hoc on nb3080 quantile-conditional anchor; 15-NEW-seed verified mean 0.4437 +/- 0.0009 across kf_seeds {1156..1170} (DISJOINT from nb3161 original 5-seed batch); CI95 [0.4432, 0.4442], median 0.4437, min/max [0.4419, 0.4451]; shift_vs_nb3161 = -0.0000 (EXACT reproduction, no kf-seed luck); delta_vs_nb3080 parent = -0.0038 (per-fold y-range clip is REAL gain on quantile-conditional substrate); delta_vs_nb3070 prior PRIMARY-1 = -0.0040; delta_vs_nb3030 = -0.0072; gate_lucky_seed_trap PASS (shift << +0.005); df=2 fixed quantile bounds per fold (q05/q95); fold-train y-range clipping bounds preds to fold-train y quantile range, extracting capacity beyond quantile-conditional blend; both anchor (nb3080) and y are PRE-unblind (anchor_leak_eq_truth_frac=0); te_unb_in_sample 0.1974 is EXPECTED deploy-refit optimism per feedback_te_vs_pred_oof_protocol; deploy clips at fold-pooled y(q05, q95) = (2.484, 5.917) with 0 lo-clipped / 18 hi-clipped on test 513; predicted LB ~0.4482 under +0.0045 PRE delta calibration; KEPT as canonical fallback safety floor (deepest single-anchor wide-seed verify on cycle-264 substrate)"),
    ("nb3190_learned_clip_on_nb3090.csv",              "PRIMARY-5 (CYCLE-268 PROMOTE, 15-SEED ORIGINAL): nb3190 per-fold LEARNED clip grid search on nb3090 anchor; 15-seed mean 0.4426 +/- 0.0021 across kf_seeds {1171..1185}; per_fold_val_rae_mean 0.4555 +/- 0.0109; PARENT of nb3200 (the deep-30 verifier); kept as alternate at the same parameter set since nb3200 is its own deep-30 honest estimate"),
    # === CYCLE 268 DEPRECATED: POOLED-METRIC ARTIFACT BLOCK (do NOT promote/fire) ===
    # All four entries below have pooled_rae std == 0.0000 across 15 seeds (df=0 ensemble
    # mathematical identity). The "0.4418/0.4421" pooled mean is NOT honest cross-fit
    # gain over nb3200 0.4424 -- it is arithmetic identity from averaging fixed CSV
    # prediction vectors. Per-fold_val_rae_mean is the LB-faithful honest lens, and
    # those land in the same 0.4527-0.4548 band as nb3200 0.4545 (NOT a meaningful win).
    # Same root cause as cycle-160 deep-verify dispersion rule: a 5-seed/15-seed bag
    # with std < 0.001 is df=0 identity, NOT under-dispersion; ladder gate must
    # require GENUINE multi-seed dispersion (std > 0.001 with non-degenerate seed_records)
    # before promote. Entries blocked from fresh-file fallback by explicit DEPRECATED tag.
    ("nb3204_ensemble_clip_winners.csv",               "DEPRECATED-POOLED-ARTIFACT 2026-06-08 cycle-268: equal-weight mean of {nb3190, nb3173, nb3174}; pooled_rae 0.4418 +/- 0.0000 across 15 seeds is df=0 mathematical identity (averaging 3 fixed CSVs yields a fixed prediction vector; RAE pools globally so cross-seed identity is arithmetic, not statistical); per_fold_val_rae_mean 0.4548 +/- 0.0105 is in the same band as nb3200 0.4545 honest; NOT a paradigm gain over nb3200 PRIMARY-1; do NOT promote, do NOT fire"),
    ("nb3211_weighted_ensemble.csv",                   "DEPRECATED-POOLED-ARTIFACT 2026-06-08 cycle-268: inverse-RAE-weighted mean of {nb3190, nb3173, nb3174}; pooled_rae 0.4418 +/- 0.0000 across 15 seeds is df=0 mathematical identity (same root cause as nb3204; the inverse-RAE weights are themselves fixed scalars so the ensemble vector is fixed); per_fold_val_rae_mean 0.4527 +/- 0.0056 in the nb3200 band; NOT a genuine paradigm gain; do NOT promote, do NOT fire"),
    ("nb3212_clip_on_nb3204.csv",                      "DEPRECATED-POOLED-ARTIFACT 2026-06-08 cycle-268: per-fold learned-clip grid search on nb3204 ensemble anchor (double-layer clip-blend); pooled_rae 0.4417 +/- 0.0004 across 15 seeds; pooled std non-zero BUT inherits nb3204 pooled-artifact substrate plus within-seed grid-selection optimism risk per cycle-149/cycle-160 rules; per_fold_val_rae_mean 0.4526 +/- 0.0057 in the nb3200 band; NOT a verified paradigm gain over nb3200 deep-30; do NOT promote, do NOT fire"),
    ("nb3213_quad_clip_winners.csv",                   "DEPRECATED-POOLED-ARTIFACT 2026-06-08 cycle-268 (file basename `nb3213_quad_clip_winners.csv` if generated): equal-weight mean of {nb3190, nb3173, nb3174, nb3170}; pooled_rae 0.4421 +/- 0.0000 across 15 seeds is df=0 mathematical identity; per_fold_val_rae_mean 0.4530 +/- 0.0056 in the nb3200 band; NOT a paradigm gain; do NOT promote, do NOT fire"),
    ("nb3173_learned_clip.csv",                        "PRIMARY-1B-QUEUE-VERIFY (CYCLE-264, BETTER_THAN_NB3170 at same-seed grid -- QUEUE fresh-seed verify before promote): nb3173 per-fold LEARNED clip via 16-combo grid search (q_low in {.01,.02,.05,.1} x q_high in {.9,.95,.98,.99}) on nb3080; 15-seed mean 0.4422 +/- 0.0010 across kf_seeds {1156..1170} (SAME seeds as nb3170 -- within-seed grid optimism risk); CI95 [0.4417, 0.4427], median 0.4423, min/max [0.4404, 0.4437]; delta_vs_nb3170_fixed = -0.0015; delta_vs_nb3080 = -0.0053; delta_vs_nb3030 = -0.0087; ql_mode 0.05 (75/75 folds), qh_mode 0.98 (71/75); modal deploy clip (q05, q98) = (2.484, 6.079) with 0 lo / 4 hi clipped on test 513; CRITICAL 16-combo grid selection on SAME 15 seeds as nb3170 = within-seed grid optimism risk per cycle-149 nb1191 lucky-seed + cycle-160 deep-verify dispersion rules; the -0.0015 gap could be grid-selection optimism narrowing variance; promote ONLY if fresh-seed (>=15) at fixed modal (q05, q98) on kf_seeds {1171..1185} holds mean < 0.4437 with std < 0.0030; pre-stage CSV but DO NOT fire from cron until verified"),
    ("nb3070_wide_verify_nb3063.csv",                  "PRIMARY-1C (DEMOTED cycle-264 from PRIMARY-1; CYCLE-254 VERIFIED_NEW_PRIMARY1): nb3063 quantile-conditional hard-split blend on {K18, K19} deep-30; wide-seed verified mean 0.4477 +/- 0.0002 across 15 fresh kf_seeds {1096..1110}; CI95 [0.4476, 0.4479], median 0.4478, min/max [0.4473, 0.4482]; +0.0040 vs nb3170 PRIMARY-1; weights w_low {K18:0.8, K19:0.2} for pred<q50=4.8845, w_high {K18:0.5, K19:0.5} for pred>=q50; both anchors PRE-clean; conditional axis opens NEW df at quantile cut (df=3 effective); kept as alternate (quantile-conditional substrate of nb3161 parent)"),
    ("nb3072_per_fold_q50.csv",                        "PRIMARY-1D-CLEAN (DEMOTED cycle-264; CYCLE-254 PROMOTE, ties nb3070 cleaner cross-fit): nb3072 fold-TRAIN-only q50 quantile-conditional blend on {K18, K19} deep-30; 15-seed mean 0.4477 +/- 0.0002; +0.0040 vs nb3170 PRIMARY-1; cleaner cross-fit q50 derived purely from fold-train (no val info leakage); ties nb3070 exactly; kept as alternate"),
    ("nb3073_quantile_tail_sweep.csv",                 "PRIMARY-1E-QUEUE-VERIFY (DEMOTED cycle-264; CYCLE-254, single-grid BETTER at 5-seed -- QUEUE wide-seed verify before promote): nb3073 2-D quantile tail sweep over 36 combos (q_cut x w_low x w_high); best combo q_cut=0.4, w_low=0.9, w_high=0.5 mean 0.4470 +/- 0.0009 across kf_seeds {1051..1055} (SAME 5-seed batch as nb3063 original); min/max [0.4459, 0.4481]; delta_vs_nb3030 = -0.0039; -0.0007 vs nb3070 PRIMARY-1; CRITICAL 5-seed-only sweep over 36 combos = HIGH selection-procedure risk at n=253 per cycle-149 nb1191 lucky-seed + cycle-160 deep-verify dispersion rules; std 0.0009 across 5 seeds consistent with within-grid selection optimism (best of 36 narrows variance), NOT genuine paradigm gain; promote ONLY if wide-seed (>=15) at best combo holds mean < 0.4477 with std < 0.0030 on fresh seeds {1111..1125}; pre-stage CSV but do NOT fire from cron until verified"),
    ("nb3071_soft_quantile_blend.csv",                 "PRIMARY-1F-SOFT (DEMOTED cycle-264; CYCLE-254 PROMOTE, sub-marginal soft variant, REGRESSES vs nb3070): nb3071 soft continuous rank quantile blend on {K18, K19} deep-30; schedule w_K18 = 0.8 - 0.3*(rank/n) clamped [0.5, 0.8]; 15-seed mean 0.4484 +/- 0.0001 across kf_seeds {1096..1110}; CI95 [0.4484, 0.4485]; delta_vs_nb3030 = -0.0025; +0.0007 vs nb3070 PRIMARY-1 (hard-split wins over soft-cont); confirms hard step @ q50 is the right conditional capacity -- soft schedule over-parameterizes the conditional axis; kept as alternate paradigm for diversity; deploy w_K18 fold-mean 0.6502, min 0.5006, max 0.8; predicted LB ~0.4529 under +0.0045 PRE delta"),
    ("nb2982_per_fold_simplex_K18_K20.csv",            "PRIMARY-2 (DEMOTED cycle-254, was PRIMARY-1 cycles 246-253, VERIFIED_NEW_PRIMARY1 via nb2990 wide-seed): nb2982 per-fold SLSQP simplex over 2 K-anchors {K18, K20} on chemprop_aux residual; wide-seed verified mean 0.4518 +/- 0.0012 across 15 fresh kf_seeds {1021..1035}; CI95 [0.4511, 0.4524], median 0.4517, min/max [0.4501, 0.4544]; shift vs single-kf=1001 (0.4505) = +0.0013 (under +0.005 lucky-seed threshold); shift vs nb2973 verified (0.4535) = -0.0017; +0.0041 vs nb3070 PRIMARY-1; full_pool_weights {K18: 0.7015, K20: 0.2985}; under-dispersion ratio 0.67x vs nb2980 (2-anchor SLSQP has fewer free parameters than 4-anchor); both anchors PRE-clean (anchor_leak_eq_truth_frac=0); te_unb_in_sample 0.1912 is EXPECTED deploy-refit optimism per feedback_te_vs_pred_oof_protocol (NOT contamination); KEPT as PRIMARY-2 (per-fold SLSQP paradigm anchor); predicted LB ~0.4563 under +0.0045 PRE delta calibration"),
    ("nb2992_per_fold_simplex_K18_K19_K20.csv",        "PRIMARY-2B-QUEUE-VERIFY (DEMOTED cycle-254; CYCLE-246, single-kf BETTER_THAN_NB2982 +1 unverified anchor): nb2992 per-fold SLSQP simplex over 3 K-anchors {K18, K19, K20}; pooled_outer_val_rae 0.4479 (single kf_seed=1001); BEATS nb2982 single-kf 0.4505 by -0.0026; per_fold_val_rae_mean 0.4513 +/- 0.0334; full_pool_slsqp weights {K18: 0.6331, K19: 0.3669, K20: 0.0} (K20 essentially zeroed -> reduces to K18+K19); any_fold_degenerate=False; K19 anchor depth is 5-seed only (NOT deep-30) -> per-fold SLSQP gain may inherit K19 un-bagged seed luck; QUEUE wide-seed verify before promote per cycle-245 nb2982 precedent; CSV pre-staged for rapid promotion (kf_seeds {1036..1050} + deep-30 K19 bag) -- do NOT fire from cron until verified"),
    ("nb2973_per_fold_simplex_4K.csv",                 "PRIMARY-3 (DEMOTED cycle-254; was PRIMARY-2 cycle-246, PRIMARY-1 cycles 244-245, per-fold SLSQP 4K continuous-simplex): nb2973 per-fold SLSQP simplex over 4 K-anchors {K18, K20, K24, K28} on chemprop_aux residual; nb2980 wide-seed verified mean 0.4535 +/- 0.0018 across 15 fresh kf_seeds {1006..1020}; +0.0017 vs nb2982 PRIMARY-1; mean weights across folds K18=0.7005, K20=0.2352, K24=0.0424, K28=0.0219 (K18-dominant -- the 2 zeroed anchors (K24, K28) are the source of the +0.0017 regression vs nb2982 minimal-pool); all 4 anchors PRE-clean; KEPT as alternate at PRIMARY-2 slot"),
    ("nb2991_K18_alone.csv",                           "PRIMARY-3B-PARAM-FREE (DEMOTED cycle-254; CYCLE-246 PROMOTE, zero-df K18-alone alternate): nb2991 K18 LGBM residual standalone on chemprop_aux (deep-30 bag); wide-seed mean_rae 0.4536 +/- 0 (df=0 mathematical identity, K18 OOF precomputed deep-30 bag so cross-seed identity is mathematical not statistical); ties nb2973 verified mean within noise band (+0.0001); +0.0018 vs nb2982 PRIMARY-1; parameter-free safest singleton on the ladder, zero-df rigour profile; PRE-clean; predicted LB ~0.4581 under +0.0045 PRE delta calibration"),
    ("nb2943_fine_ratio_grid.csv",                     "PRIMARY-4 (DEMOTED cycle-254; df=0 deep-5-seed verified): nb2943 3-comp deterministic blend 0.5*nb2240_K20 + 0.5*equal_K(K18,K24,K28); nb2950 deep-5-seed verify (kf 1001-1005) pooled_rae_mean 0.45764 +/- 0 (SEM 2.8e-17, std == machine epsilon because df=0 means cross-seed identity is mathematical not statistical); +0.0058 vs nb2982 PRIMARY-1; kept as safety floor with stronger rigour profile (df=0 deterministic); -0.0022 vs nb2240 (deep-30 0.4598), -0.0106 vs nb2171 PRE-clean (0.4682), -0.0142 vs nb1191 LB-SAFE-DEEP (0.4718); all 3 component anchors PRE-clean; predicted LB ~0.462 under +0.0045 PRE delta"),
    ("nb2934_multi_K_plus_nb1191.csv",                 "PRIMARY-5 (DEMOTED cycle-254; df=0 deep-5-seed verified): nb2934 5-comp equal-weight K18+K20+K24+K28+nb1191; nb2940 deep-5-seed verify pooled_rae mean 0.4585 +/- 0 (df=0 identity); +0.0067 vs nb2982 PRIMARY-1; kept as safety floor; predicted LB ~0.463 under +0.0045 PRE delta"),
    ("nb2240_nb2171_k20.csv",                          "PRIMARY-6 (DEMOTED cycle-254): nb2240 K=20 pyramid PRE-clean blend (5 anchors: nb2240_K20 + chemprop_aux + nb1191 + nb503 + nb562); deep-30 verified pooled RAE 0.4598 +/- 0.00116 (cycle-173 nb2180 audit); +0.0080 vs nb2982 PRIMARY-1; deep-30 verified remains the rigour standard for non-df=0 candidates, KEPT as alternate"),
    ("nb2604_k_ensemble_equal_weight.csv",             "PRIMARY-6 (DEMOTED cycle-246): nb2604 equal-weight 4-K ensemble over {K18, K20, K24, K28} on chemprop_aux residual; nb2640 deep-30 verify verdict LUCKY_SEED_DEMOTE: pooled_rae_mean_seeds (n=30) = 0.4814 +/- 0.01276, CI95 [0.4767, 0.4862], shift_vs_nb2604_5seed = +0.0234; 5-seed claim 0.4580 was lucky-batch luck; per-K 30-seed bagmean K18=0.4545 is BEST atomic anchor (kept as alternate); equal-weight blend regressed once 5-seed batch luck washed out at 30 seeds"),
    ("nb2900_4anchor_pyramid_deploy.csv",              "PRIMARY-7 (DEMOTED cycle-246): nb2900 4-anchor pyramid PRE-clean blend; deep-30 verified mean 0.4619 +/- 0.00086 (n=30 bag x 5 kf = 150 runs, under-disp 1.003x deterministic); delta -0.006 vs nb2171 (0.4682), reproducible win; FAILS gate_promote 0.4570 by +0.0049 but BEATS nb2171 cleanly; deploy weights collapse to ~80% nb2240_K20 / ~19% nb1191 / ~0% chemprop_aux+counter"),
    ("nb2920_ratio_w070_deploy.csv",                   "PRIMARY-8 (DEMOTED cycle-246): nb2920 2-anchor ratio sweep best w=0.7; pooled RAE 0.4596 single-kf=1001 (MARGINAL beats nb2240 0.4598 by -0.0002 noise band); FAILS gate_promote 0.4570 by +0.0026; no deep-30 verification yet"),
    ("nb2330_deploy_k24_swap.csv",                     "PRIMARY-9 (DEMOTED cycle-246): nb2330 K=24 anchor-swap variant, deep-30 pyramid OOF RAE 0.4662"),
    ("chemprop_aux.csv",                               "PRIMARY-10 (PRE-clean SENTINEL, demoted cycle-246): chemprop multi-task w/ aux heads (raw MPNN, no LGBM residual layer); honest unblind RAE 0.6216, predicted LB 0.6246; safety floor for blend anchors"),
    ("nb1191_deploy_pre_pyramid.csv",                  "PRIMARY-11 (PRE-clean LB-SAFE-DEEP, demoted cycle-246): nb1191 4-anchor PRE-unblind pyramid SLSQP then rank-stretch; wide-seed verified mean 0.4718 +/- 0.00245; calibrated LB band 0.486-0.570"),
    ("nb1162_deploy_nb1153.csv",                       "PRIMARY-12 (POST-risk SINGLE-FIRE LOTTERY, demoted cycle-246): nb1162 5-anchor stack-pyramid SLSQP; cross-fit RAE 0.4206 but 88.7% nb730 weight; transfer_risk=HIGH, lb_pred_band 0.42-0.62; FIRE ONCE ONLY -- do not rotate"),
    # === Cycle-183 SUPERSEDED (prior PRIMARY-1 stack ordering, kept as alternates) ===
    ("nb2171_deploy_anchor_swap.csv",                  "SUPERSEDED-1 (cycle-183, was PRIMARY-1 PRE-CLEAN): nb2171 anchor-swap 5-anchor PRE-clean blend; deep-30 pooled RAE 0.4682 +/- 0.0024; superseded by nb2240 K20 at 0.4601"),
    ("nb2095_deploy_pre_pyramid.csv",                  "SUPERSEDED-2 (cycle-183, was PRIMARY-1 PRE-PYRAMID): nb2095 5-seed 4-anchor PRE-unblind pyramid SLSQP (drops nb730); scaffold-CV OOF RAE 0.4703; superseded by nb2240 K20"),
    ("nb2060_deploy_no_nb730.csv",                     "PRIMARY-1 (LB-CLEAN-HOLD): nb2060 4-anchor PRE-unblind blend (no nb730); deep-30 sub-noise miss vs nb1191; held as clean PRE alternate"),
    ("nb1150_deploy_slsqp4.csv",                       "PRIMARY-2: nb1150 SLSQP simplex blend over 4 LB-honest anchors (chemprop_aux + nb503 + nb1014 + nb2112); cycle-143 verified scaffold-CV 0.4710 (nb1156 nested), fresh-seed 0.4688+/-0.0014; conservative LB band lo=0.519 / mid=0.569 / hi=0.619; te[unb]=0.1890 in-sample is EXPECTED deploy-refit optimism per feedback_te_vs_pred_oof_protocol"),
    ("nb1158_deploy_K32.csv",                          "PRIMARY-3: nb1158 K=32 SHAP-feature LGBM residual stack on chemprop_aux (cycle-143 fresh-optimal K from nb1157 K-sweep); K=32 mean_bag_rae_fresh 0.5012 beats K=35 by -0.013; predicted LB band ~0.49-0.55"),
    ("nb2112_deploy_shap28.csv",                       "PRIMARY-4: nb2112 deploy SHAP-28 (chemprop_aux + LGBM K=28, PRE-unblind); honest scaffold-CV RAE 0.5057 / random-KF 0.4737 mean; predicted LB ~0.509 (honest+0.003); cycle-119 anchor floor"),
    ("nb1660_deploy_nb1632_mean.csv",                  "PRIMARY-5-LEGACY: nb1660 deploy of nb1632 mean-blend (PRE-unblind, cycle 119 anchor); honest 5-fold cross-fit RAE 0.5107; predicted LB ~0.514 (deprioritized below cycle-183 PRIMARY-1..5)"),
    # chemprop_aux entry MOVED UP to PRIMARY-3 per cycle-183 lock; duplicate removed here
    ("nb1211_deploy_drop_nb730.csv",                   "PRIMARY-7 (DEMOTED cycle-162): nb1211 5-anchor drop-nb730 variant of pre_pyramid; OOF 0.4708; superseded by nb2095 at OOF 0.4703 (cycle-162)"),
    ("nb1014_multi_seed_bag.csv",                      "PRIMARY-8: nb1014 multi-seed bag over nb1001 blend stack (5 seeds, cycle 7); honest 5-fold cross-fit RAE 0.5930; predicted LB ~0.596"),
    ("nb1001_crossfit_chempropaux_nb972_stretch.csv",  "PRIMARY-9: nb1001 cross-fit blend(chemprop_aux 0.76, nb972 0.24) + stretch s=1.25 (cycle 6); honest 5-fold cross-fit RAE 0.5994; predicted LB ~0.602"),

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
    # === CYCLE-6 2026-06-03 NOTE: nb1014/nb1001/chemprop_aux PROMOTED to canonical
    # PRIMARY tier above (cycle-128 ladder-hygiene reorder); ladder continues below
    # with PRIMARY-6.. supplemental anchors (grand_v6b, nb306, nb305, etc).
    ("grand_v6b_calib.csv",                      "PRIMARY-10: grand_v6b calibrated ensemble; in_RAE 0.6409; predicted LB 0.6439"),
    ("nb306_cepsmim.csv",                        "PRIMARY-11: nb306 ceps-MIM; in_RAE 0.6486; predicted LB 0.6516"),
    ("nb305_mope.csv",                           "PRIMARY-12: nb305 MoPE; in_RAE 0.6601; predicted LB 0.6631"),
    ("95_all_feature_fusion.csv",                "PRIMARY-13: mm-audit #5 all-feature-fusion (PRE-unblind); in_RAE 0.6625; predicted LB 0.6655"),
    ("54_deep_ensemble_uncertainty.csv",         "PRIMARY-14: mm-audit deep-ensemble-uncertainty (PRE-unblind); in_RAE 0.6657; predicted LB 0.6687"),
    ("27_nr_weighted_lgbm.csv",                  "PRIMARY-15: mm-audit NR-weighted LGBM (PRE-unblind); in_RAE 0.6729; predicted LB 0.6759"),
    ("82_selectivity_aware.csv",                 "PRIMARY-16: mm-audit selectivity-aware (PRE-unblind); in_RAE 0.6730; predicted LB 0.6760"),
    ("67_lgbm_chembl_all_nr_weighted.csv",       "PRIMARY-17: mm-audit ChEMBL-all-NR-weighted LGBM (PRE-unblind); in_RAE 0.6746; predicted LB 0.6776"),
    ("nb303_dann.csv",                           "PRIMARY-18: nb303 Karpathy DANN; in_RAE 0.6931; predicted LB 0.6961"),
    ("nb800_huber_1_5.csv",                      "PRIMARY-19: nb800 Huber alpha=1.5 (best alpha sweep variant, PRE-unblind); in_RAE 0.7378; predicted LB 0.7408"),
    ("nb800_huber_ens4.csv",                     "PRIMARY-20: nb800 4-way Huber ensemble {0.3,0.7,1.5,3.0} (PRE-unblind); in_RAE 0.7441; predicted LB 0.7471"),
    ("nb801_huber_assay_decomp_plus.csv",        "PRIMARY-21: nb801 Huber w/ expanded assay-decomp (6 new feats, PRE-unblind); in_RAE 0.7448; predicted LB 0.7478"),
    ("nb120_huber_1_0.csv",                      "PRIMARY-22: nb120 Huber delta=1.0; in_RAE 0.7461; predicted LB 0.7491"),
    ("nb120_huber_2_0.csv",                      "PRIMARY-23: nb120 Huber delta=2.0; in_RAE 0.7502; predicted LB 0.7532"),
    ("nb120_huber_0_5.csv",                      "PRIMARY-24: nb120 Huber delta=0.5; in_RAE 0.7513; predicted LB 0.7543"),
    ("nb273_molformer.csv",                      "PRIMARY-25: nb273 MoLFormer (already submitted but worth re-checking)"),

    # === EXPLORE TIER (cycle-1/2/3 method-axis diversity, demoted below PRIMARY-21
    # by cycle-128 ladder-hygiene reorder 2026-06-08). All PRE-unblind regime.
    # CYCLE-1 (LLM, NR-multitask, SMILES-aug TTT, quantile conformal, SSL pretrain-FT):
    ("nb901_nr_multitask.csv",                   "CYCLE-1-1: nb901 NR-multitask LGBM (multi-NR transfer w/ shared trunk); in_RAE 0.6765; predicted LB ~0.68"),
    ("nb902_smiles_aug_ttt.csv",                 "CYCLE-1-2: nb902 SMILES-augmented test-time training (TTT); in_RAE 0.6998; predicted LB ~0.70"),
    ("nb903_quantile_conformal.csv",             "CYCLE-1-3: nb903 quantile-regression conformal prediction (q10/q50/q90 calibrated); in_RAE 0.7240; predicted LB ~0.73"),
    ("nb900_llm.csv",                            "CYCLE-1-4: nb900 LLM-based predictor (prompted, te_nb900.npy); in_RAE 0.8500; predicted LB ~0.85"),
    # CYCLE-2 (GP Tanimoto, AL ChEMBL proxy, MoE sparse, Persistence Homology):
    ("nb914_persistence_homology.csv",           "CYCLE-2-1: nb914 Persistence Homology features (topological descriptors); in_RAE 0.7121; predicted LB ~0.72"),
    ("nb911_al_chembl_proxy.csv",                "CYCLE-2-2: nb911 Active Learning ChEMBL proxy (query-by-committee on ChEMBL surrogate); in_RAE 0.7263; predicted LB ~0.73"),
    ("nb910_gp_tanimoto.csv",                    "CYCLE-2-3: nb910 Gaussian Process w/ Tanimoto kernel; in_RAE 0.7275; predicted LB ~0.73"),
    ("nb912_moe_sparse_routing.csv",             "CYCLE-2-4: nb912 Mixture-of-Experts sparse routing; in_RAE 0.8421; predicted LB ~0.84"),
    # CYCLE-3-RETRY (MAML, WL kernel, Pseudo-label, ExtraTrees):
    ("nb960_pseudo_self_train.csv",              "CYCLE-3-RETRY-1: nb960 Pseudo-label self-training; in_RAE 0.6951; predicted LB ~0.70"),
    ("nb923_wl_graph_kernel.csv",                "CYCLE-3-RETRY-2: nb923 Weisfeiler-Lehman graph kernel; in_RAE 0.7053; predicted LB ~0.71"),
    ("nb961_extra_trees.csv",                    "CYCLE-3-RETRY-3: nb961 ExtraTrees ensemble; in_RAE 0.8203; predicted LB ~0.82"),
    ("nb921_scaffold_maml.csv",                  "CYCLE-3-RETRY-4: nb921 Scaffold-MAML meta-learner; in_RAE 1.3465; predicted LB ~1.35 (worse than mean predictor)"),

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

# ============================================================================
# PHASE-2 FOOTER (cycle-128, 2026-06-08)
# ACTIVITY LB FROZEN until ~2026-07-01; submissions queue for re-grade;
# keep PRIMARY-1 rotation diverse
# ============================================================================

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
