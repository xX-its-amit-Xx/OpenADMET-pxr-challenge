# PXR Challenge - Trajectory v2 (cycles 130-140 extension)

Auto-generated 2026-06-08 cycle-140 build. Companion to `trajectory.md` (cycles 1-130). This file uses the orthogonal nb1300-nb1499 bucket scheme (cycle N = nb1(N)00 - nb1(N)09) -- a separate substrate from the main trajectory.md nb91x-nb220x indexing. Both indexings coexist; the cycle-140 floor in BOTH systems is **0.4698 RAE honest cross-fit** at the `chemprop_aux` anchor refit / nb2112 deploy.

LB context (per `feedback_lb_frozen_phase2.md`): activity-track LB frozen since 2026-05-27 at 0.7655. Honest 5-fold cross-fit RAE on the 253-row unblind subset (`*_pred_oof.npy`) is the only live signal.

---

## Cycles 130-140: nb1300-nb1409 substrate (residual + SHAP-prune family)

### Run-level table

`mean_bag` = `rae_mean_bag` field if present; `-` = composite/blend script with no single-bag metric. `gate` = PASS if verdict contains BEATS without WORSE_THAN/FLAT/HURTS; FAIL if verdict says FLAT/HURTS/WORSE_THAN; MIXED otherwise (incl. reproductions). Notes are abridged from `verdict` field.

| Cycle | NB     | Method                                  | RAE mean_bag | Gate  | Notes                                                            |
|------:|--------|-----------------------------------------|-------------:|-------|------------------------------------------------------------------|
| 130   | nb1300 | ChemBERTa-77M-MTR residual (PCA-100)     | 0.5545       | MIXED | Helps nb1070, but worse than nb1183/nb1242                       |
| 130   | nb1301 | Per-row routing (fixed-w gate)           |       -      | FAIL  | Flat vs nb1290 (baseline_fixed_w0.35 @ 0.5390)                   |
| 130   | nb1302 | Uncertainty-weighted blend               |       -      | FAIL  | Hurts vs nb1290 (C_invstd_floored @ 0.5420)                      |
| 130   | nb1303 | Alt-objective bag                        | 0.5466       | MIXED | Helps nb1070, worse than nb1242                                  |
| 130   | nb1304 | Train-all + external-union               | 0.5559       | FAIL  | Helps nb1070, hurts refs                                         |
| 131   | nb1310 | SVR-Tanimoto residual                    | 0.5636       | MIXED | Helps nb1070, worse than nb1183                                  |
| 131   | nb1311 | GP-Tanimoto residual                     | 0.5747       | FAIL  | Flat, no new signal                                              |
| 131   | nb1312 | Chemprop embed residual                  | 0.5647       | MIXED | Helps nb1070, worse than prior residuals                         |
| 131   | nb1314 | Bayesian model averaging                 |       -      | FAIL  | Flat vs nb1290 (map @ 0.5400)                                    |
| 132   | nb1321 | (reserved)                                |       -      | MIXED |                                                                  |
| 132   | nb1322 | Quantile q05 residual                    | 0.5474       | FAIL  | Hurts nb1290                                                     |
| 132   | nb1323 | Isotonic on nb1290                       | 0.5630       | FAIL  | Hurts vs nb1290                                                  |
| 132   | nb1324 | Augmented-anchor                         |       -      | FAIL  | 0.6020 vs floor 0.5390 (-0.063 vs ref)                          |
| 133   | nb1330 | Rank-stretch on nb1290                   |       -      | FAIL  | Hurts cross-fit (cf RAE 0.5448, delta +0.0058)                  |
| 133   | nb1331 | (reserved)                                |       -      | MIXED |                                                                  |
| 133   | nb1332 | Mordred + ChEMBL kNN feat                | 0.5576       | FAIL  | Helps nb1070, worse than nb1242                                  |
| 133   | nb1333 | DW-pessimistic                            |       -      | FAIL  | DW=0.5463 vs random=0.5390                                       |
| 134   | nb1340 | PLS residual                             | 0.6254       | FAIL  | Hurts nb1070                                                     |
| 134   | nb1341 | CatBoost residual                        | 0.5420       | FAIL  | Ties nb1242 flat                                                 |
| 134   | nb1342 | K-sequence residual                      | 0.5486       | FAIL  | Helps nb1070, worse than nb1242                                  |
| 134   | nb1343 | ElasticNet residual                      | 0.5833       | FAIL  | Hurts                                                            |
| 134   | nb1344 | LGBM stumps residual                     | 0.5722       | FAIL  | Helps nb1070, worse than depth=3                                 |
| 135   | nb1350 | CatBoost blend                           |       -      | PASS  | grid_best @ 0.5359, beats nb1290                                 |
| 135   | nb1351 | (reserved)                                |       -      | MIXED |                                                                  |
| 135   | nb1352 | SHAP-pruned MACCS K=20                   | **0.5323**   | PASS  | Beats nb1242 -- new bucket primary candidate                     |
| 135   | nb1353 | Combined bag                             |       -      | FAIL  | Beats nb1242 only; flat vs nb1290                                |
| 135   | nb1354 | (reserved)                                |       -      | MIXED |                                                                  |
| 136   | nb1360 | (reserved)                                |       -      | MIXED |                                                                  |
| 136   | nb1361 | nb1352 reproduces                         |       -      | MIXED | Stability check passed                                           |
| 136   | nb1362 | SHAP K-grid                              |       -      | FAIL  | Flat vs nb1352, best K=30                                        |
| 136   | nb1363 | Combine 1352 + 1290                      |       -      | FAIL  | bestw_median @ 0.5315, flat vs nb1352                            |
| 136   | nb1364 | SHAP-pruned Mordred                      | **0.5242**   | PASS  | Beats nb1352 -- new bucket primary candidate                     |
| 137   | nb1370 | (reserved)                                |       -      | MIXED |                                                                  |
| 137   | nb1371 | CatBoost pruned blend                    | 0.5329       | PASS  | Beats nb1352 -- new blend candidate                              |
| 137   | nb1372 | Dual-pruned (MACCS + Mordred)            | **0.5207**   | PASS  | Beats nb1364 -- new bucket primary candidate                     |
| 137   | nb1373 | SHAP-pruned AtomPair K=30                | **0.5095**   | PASS  | Beats nb1352, new bucket primary; pearson 0.99 to nb1372         |
| 137   | nb1374 | Per-row gated                            |       -      | FAIL  | Flat vs nb1352 (k=10.0, thr=0.60 -> 0.5328)                      |
| 138   | nb1380 | (reserved)                                |       -      | MIXED |                                                                  |
| 138   | nb1381 | nb1373 reproduces                         |       -      | MIXED |                                                                  |
| 138   | nb1382 | SHAP-pruned Morgan                       | 0.5527       | FAIL  | Helps nb1070, worse than nb1352                                  |
| 138   | nb1383 | Triple-pruned (MACCS+Mord+AP)            | 0.5189       | FAIL  | Flat vs nb1372                                                   |
| 138   | nb1384 | K-grid best K=20                          |       -      | FAIL  | Flat vs nb1373                                                   |
| 139   | nb1390 | (reserved)                                |       -      | MIXED |                                                                  |
| 139   | nb1391 | Triple-blend grid                         |       -      | FAIL  | Flat vs nb1373                                                   |
| 139   | nb1392 | SHAP-pruned Avalon                       | 0.5391       | FAIL  | Beats nb1163, worse than nb1373                                  |
| 139   | nb1393 | SHAP-pruned TopologicalTorsion           | 0.5581       | FAIL  | Beats nb1173, worse than nb1373                                  |
| 139   | nb1394 | Triple-blend cross-fit                    |       -      | FAIL  | Flat vs nb1373                                                   |
| 140   | nb1401 | Quad-blend                                |       -      | FAIL  | Flat vs nb1373                                                   |
| 140   | nb1402 | Ext K-grid (best K=25)                    |       -      | FAIL  | Flat vs nb1373                                                   |
| 140   | nb1403 | nb1391 reproduces                         |       -      | MIXED |                                                                  |
| 140   | nb1404 | Dual-pruned (AP + Avalon K=30)           | 0.5191       | FAIL  | Beats nb1392, worse than nb1373                                  |

### Floor check

Minimum `rae_mean_bag` in the nb1300-nb1409 substrate: **0.5095** (nb1373 SHAP-pruned AtomPair K=30 residual on nb1070 anchor).

**0.5095 is the bucket-local floor on the nb1070 anchor**, not a global floor. The global cycle-140 floor on the `chemprop_aux` PRE-unblind anchor remains **0.4698** (nb2112 deploy alias for nb2189 K=28 honest median-bag).

Cycle-140 cross-check (`nb2202`) reproduces 0.4698 exactly with refreshed SHAP basis. Cycles 127-130 audit chain (`nb2170/nb2178/nb2184/nb2189/nb2201`) is in `trajectory.md` Section 2.

### Candidates inspected for beating 0.4698

None in the nb1300-nb1409 substrate. All values are residuals on `nb1070` anchor (RAE 0.5771), not `chemprop_aux` PRE-unblind. Their honest LB-faithful equivalents would land in the 0.51-0.53 band -- noisily flat with the chemprop_aux baseline (0.6216 honest), not breakthroughs.

| Candidate | Reported     | Anchor     | LB-faithful? | Verdict           |
|-----------|--------------|------------|--------------|-------------------|
| nb1373    | 0.5095       | nb1070     | NO           | Bucket-local only |
| nb1372    | 0.5207       | nb1070     | NO           | Bucket-local only |
| nb1364    | 0.5242       | nb1070     | NO           | Bucket-local only |
| nb1352    | 0.5323       | nb1070     | NO           | Bucket-local only |

**Verdict (per cycle 137-139 audits): NO candidate in nb1300-nb1409 GENUINELY beats 0.4698.** The 0.51 floor on the nb1070 anchor is ~0.04 above the chemprop_aux refit-anchor 0.4698 floor.

---

## Final floor (cycle 140)

| Metric                                | Value     |
|---------------------------------------|-----------|
| Honest cross-fit RAE (253 unblind)    | **0.4698**|
| Reference candidate                    | nb2112 (deploy alias for nb2189 K=28 median-bag) |
| Anchor                                 | chemprop_aux PRE-unblind (RAE 0.6216) |
| Compounding delta from chemprop_aux    | -0.1518 RAE (-24.4% relative) |
| LB predicted band                      | ~0.47 (Phase-2 re-grade expected ~2026-07-01) |
| Floor held since                       | cycle 121 (nb2103 K=28 SHAP substrate) |
| Audit attempts since cycle 123         | 6 (all reverted to band or flagged anchor-contaminated) |

Pre-frozen LB calibration (per `feedback_lb_two_regime_calibration.md`): PRE-unblind anchor `te[unb_idx] ≈ LB + 0.003`. The nb2189/nb2112 residuals on the regenerated PRE-unblind chemprop_aux anchor satisfy this rule (in_RAE 0.42 + ~0.05 anchor-refit bias + ~0.003 LB delta = ~0.47 LB).

Open axes (per `trajectory.md` Section 4): external scaffold-diverse data (nb950 v2 pending), conformal-shrink abstention (nb2190 in flight), narrow-substrate transformer (nb2191 in flight), structure-track pose-quality features (nb1010 plan), OOD-aware router (nb1012 plan).
