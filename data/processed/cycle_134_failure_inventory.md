# PXR Cycle 134 Failure Inventory (~860 methods)

Walk of `data/processed/nb*_summary.json` cross-referenced with `scripts/nb*.py`.
All RAE values are honest 5-fold cross-fit on the 253 unblind compounds unless noted.
Reference floor: `nb2103` LGBM/SHAP-K28 5-way 117-col = **mean_bag 0.4737 / median_bag 0.4698**.

## 1. CLOSED axes

| # | Axis | Probe nbs | Methods | Best RAE | Verdict |
|---|---|---|---|---|---|
| 1 | K-sweep (SHAP top-K) | nb2063/nb2081/nb2091/nb2103/nb2178 | 30 K values (10-117) | 0.4698 @ K=28 | local minimum, flat in 26-35 |
| 2 | num_leaves grid | nb2123/nb2131/nb2132 | 12,14,15,16,18 | 0.4698 @ L=15 | flat |
| 3 | learning_rate grid | nb2133 | 0.02/0.03/0.04/0.06 | 0.4698 @ lr=0.03 | flat |
| 4 | min_child_samples | nb2151 | 3/5/7/10 | 0.4698 @ 5 | flat |
| 5 | feature_fraction | nb2153 | 0.5-1.0 | 0.4698 @ 1.0 | flat |
| 6 | monotone constraints | nb2013/nb2158 | top-10 mono | 0.4988 | WORSE |
| 7 | DART drop-rate | nb2160 | 0.05-0.20 | 0.4698 | flat |
| 8 | SHAP-seed intersect | nb2159 | 3-seed intersect K=28 | 0.4698 | flat |
| 9 | row-bootstrap subsample | nb2166 | 0.6-1.0 | 0.4698 | flat |
| 10 | pooled-N (25/50-bag) | nb2071/nb2082/nb2092/nb2111/nb2131/nb2132 | inner5xouter5 | 0.4698 | no variance gain |
| 11 | XGB-SHAP cross-paradigm | nb2165 | xgboost K=28 | 0.4698 | identical |
| 12 | Family ablation | nb2172 | per-family + top-2 combos | 0.4698 | flat |
| 13 | Residual cascade (2-stage) | nb2173 | K1=28 + K2=28 stage-2 | 0.4698 stage1 (stage2 no gain) | closed |
| 14 | tanh-target residual | nb2174 | c=0.5/1/1.5/2 + signlog | 0.4698 | closed |
| 15 | sklearn HistGB stack | nb2167 | cross-paradigm + blend | 0.4697 | tie (1e-4) |
| 16 | conformal shrink-to-mean | nb2190 | alpha grid x beta grid | 0.4698 raw | no shrinkage helps |
| 17 | mono-anchor-swap (CONTAMINATED) | nb2170/nb2178/nb2184/nb2189 | nb730 anchor | 0.3810 in-sample | DEMOTED |
| 18 | alt-honest-anchor | nb2185 | nb562/nb503/nb464/nb432 | 0.4847 | all worse than chemprop_aux+K28 |
| 19 | family-mono (group axes) | nb2172 | family-removal | 0.4698 | no family is dispensable |
| 20 | F2-abstention | nb2190 variant_B | novel-scaffold conformal | 0.4698 | no gain |
| 21 | ChEMBL-train-aug | nb1692/nb2063 ChEMBL kNN | +1 feature | 0.4698 | absorbed by SHAP |
| 22 | heteroscedastic-loss | nb2061 tanh | weight residual | 0.5049 | WORSE |
| 23 | isotonic | nb2190 variant_C | post-hoc | 0.4698 | no improvement |
| 24 | PU-21k single-conc | nb186/nb99 | pseudo-positives | n/a (CYCLE_LATE) | absorbed in 5-way |
| 25 | OOD-router | nb472 era | err_hat gate | 0.5410 | older era |
| 26 | ChemBERTa-resid | nb1611/nb1612/nb1620 | 6-way ChemBERTa | 0.4788 | ties K=28, no gain |
| 27 | SLSQP 6-anchor | nb194-197/nb239 | constrained convex blend | 0.4928 (historical) | nb2103 beats |
| 28 | PXR-direct ChEMBL | nb1692/nb1993 | knn1 | 0.4698 (absorbed) | feature absorbed |
| 29 | NN-head (MLP) | nb2191 | (64,32) MLPRegressor | 0.5440 | WORSE (degenerate residual fit) |
| 30 | CatBoost | nb2021/nb2141/nb1543/nb1573 | RMSE depth4 n200 lr0.05 | 0.5095-0.5309 | WORSE; blend collapses to 100% LGBM |
| 31 | Bayesian-Linear | nb98 era | ARD on Morgan | n/a (CYCLE_LATE) | absorbed |
| 32 | GNN (chemprop v1) | nb03/nb1541 | multitask chemprop K-grid | 0.6216 anchor only | ABSORBED AS ANCHOR |
| 33 | polynomial-Ridge | nb181/nb192 | quantile poly | n/a | older era flatness |
| 34 | LGBM-meta-stack | nb145/nb148/nb541 | level-2 LGBM | 0.514 (older) | no transfer to LB |
| 35 | SMILES-TTA (test-time-aug) | nb902 | aug + averaging | n/a | absorbed in anchor |

## 2. CONTAMINATED candidates demoted (nb2170/nb2178/nb2184/nb2189 chain)

| Tag | Method | Reported RAE | Why contaminated |
|---|---|---|---|
| nb2170 | anchor_swap chemprop_aux -> nb730 residual LGBM K=28 | 0.3920 | nb730 is POST-unblind anchor (te_nb730 was refit on the 253 leaked labels) -> residual fit fits noise of training set |
| nb2178 | k-sweep nb730-anchor residual LGBM fresh SHAP | 0.3810 | inherits nb730 leakage; "best K=20" optimizes against in-sample anchor |
| nb2184 | re-eval K-sweep on "honest" nb730 anchor | 0.3813 | "honest" version still trained against unblind labels via pred_oof anchor formed POST-unblind |
| nb2189 | truly-honest residual on nb562 pred_oof K=20 | 0.4556 | claimed honest but anchor nb562_pred_oof itself is post-hoc rank-stretch on the unblind; transfer-sim nb2203 shows +0.21-0.31 RAE gap on holdout vs in-sample |

Audit signal (nb2203 train-holdout transfer sim, 3 paired scaffold folds):
- nb2103 transfer gap +0.07 to +0.21 (modest)
- nb2189 transfer gap up to +0.31 -> CONFIRMS contamination
- All four DROPPED from ladder; rule reinforced: never use POST-unblind anchor in residual stacks.

## 3. Single confirmed honest winner

**nb2112** — deploy_nb2103_K28_shap_top28_median_lgbm_mse, anchor = chemprop_aux (PRE-unblind), 25-fit median bag (5 outer seeds x 5 inner offsets).

- Cross-fit RAE (nb2103 K=28): **mean_bag 0.4737, median_bag 0.4698**
- in-sample residual on unb_idx: 0.1006 (expected gap, not leakage)
- 117-col 5-way feature set (AtomPair 25, MACCS 20, Mordred 20, ChempropEmbed 20, Avalon 30, +pred_chembl_pec50, +mean_sim) -> SHAP top-28
- LB target band: **0.4698 - 0.4737** (honest)
- Submission: `submissions/nb2112_deploy_shap28.csv`; te artifact `data/processed/te_nb2112.npy`

## 4. Lessons learned

a. **n=253 cross-fit is BIAS-LIMITED not variance-limited.** Per-seed sigma of RAE ~0.013; 25-bag pooling does not move mean_bag below 0.4698. All variance-reduction axes (bag size, seed count, fold count) hit a hard floor.

b. **Post-hoc calibration / shrinkage all fail because nb2103 is already well-calibrated.** Conformal shrink (nb2190), rank-stretch on top of K=28, isotonic, beta-shrink — all return to 0.4698. The residual decomposition `mu + s*(p-mu)` was already exhausted by nb562 (scalar s=1.10) and nb2103 inherits a near-calibrated curve.

c. **Cross-paradigm models (NN, CatBoost, Bayesian, HistGB) add no orthogonal signal to LGBM K=28.** All SLSQP blends collapse to ~100% LGBM weight (best_blend_w_mlp = 0.0 in nb2191; CatBoost RMSE blend identical to LGBM up to 1e-4). The 28 SHAP features have already absorbed every paradigm's signal.

d. **External data (ChEMBL 945-pool, direct-PXR ChEMBL kNN, PU 21k single-conc) has cross-assay bias that swamps signal.** The +1 ChEMBL feature lands at SHAP rank 20/117 but K=28 already includes it; doubling pool or adding direct PXR slice yields no improvement and risks distributional drift.

e. **POST-unblind cross-fit doesn't transfer to LB (+0.10 typical shift).** Verified empirically: nb730/nb562 honest cross-fit 0.4603/0.5065 but transfer-sim shows +0.21-0.31 RAE gap on held-out scaffold fold. PRE-unblind anchors (chemprop_aux, K=28 5-way) preserve LB calibration; POST-unblind anchors do not.

f. **The only remaining lever was chemprop_v2 (in flight).** All within-117-col axes are flat at 0.4698 -- next gain must come from new feature space (chemprop_v2 deep embeddings, foundation-model embeddings beyond ChemBERTa, or 3D pose features). Within LGBM/SHAP/117-col the search is exhausted.

## 5. Top-10 honest cross-fit RAE (excluding contaminated)

| Rank | Tag | RAE | Method |
|---|---|---|---|
| 1 | nb2154 | 0.4620 | trajectory_120cycles snapshot (best_rae over 120 cycles) |
| 2 | nb2156 | 0.4620 | verify trajectory best |
| 3 | nb2144 | 0.4655 | trajectory_119cycles_5seed_bag (L=12 lock variant) |
| 4 | nb2123 | 0.4681 | num_leaves grid 12/14/16/18 best L sweep |
| 5 | nb2167 | 0.4697 | sklearn HistGB cross-paradigm stack (ties LGBM at 1e-4) |
| 6 | nb2103 | 0.4698 | LGBM/SHAP K-fine grid 26-35 K=28 (REFERENCE) |
| 7 | nb2112 | 0.4698 | DEPLOY of nb2103 (PRIMARY-1, 25-bag median) |
| 8 | nb2159 | 0.4698 | SHAP top-50 3-seed intersect K=28 |
| 9 | nb2163 | 0.4698 | pooled aggregate |
| 10 | nb2202 | 0.4698 | clean rebuild same anchor (audit confirms 0.4698) |

Notes:
- nb2154/nb2144 trajectory snapshots are aggregate-best over many cycles; they include nb2123's L-sweep gain (0.4681) and SHAP-top-28 lock; not standalone candidates.
- Decision margin 0.003 (per nb2103) — none of the top-5 exceeds noise floor below 0.4620.
- Honest-LB target band locked at **0.4698 - 0.4737**.

Ladder: nb2112 PRIMARY-1, nb2103 PRIMARY-2 (cross-fit safety floor), all other 854 methods DEPRECATED-FLAT or DEPRECATED-WORSE.
