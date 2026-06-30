# nb1594 - Final LB Candidates Report (cycle 60-63 refresh)

- Total candidates analyzed: **420** (+4 from cycle 60-63)
- Submission-ready (have CSV + te_513): **33** (+2: nb1570 mean/median deploys + nb1583)
- Skipped (no honest_RAE key): 13
- Calibration margin: **predicted_LB = honest_RAE + 0.003** (PRE-unblind regime)
- Promotion margin for ladder change: 0.003 RAE

## Cycle 60-63 update (2026-06-04)

| nb | Description | Metric | Value | Status |
|---|---|---|---|---|
| nb1554 | CatBoost 5-way K-tuned (AP+MACCS+Mord+ChempropEmbed+Avalon, MAE, depth=4, 5 seeds) | mean_bag cross-fit RAE | **0.5163** | PROMOTED (beats nb1543 by 0.0074) |
| nb1561 | CatBoost BoB (nested 5x5 outer/inner reproduction of nb1554) | BoB MEAN | **0.5155** | PROMOTED (reproduces nb1554 within 0.001) |
| nb1571 | SLSQP blend (nb1561 CatBoost + nb1560 LGBM) | cross-fit RAE | **0.5130** | HOLD (sub-margin: 0.0025 < 0.003 vs nb1561) |
| nb1582 | CatBoost Avalon-K20 BoB (K=20 instead of K=30) | BoB MEAN | **0.5184** | DROPPED (lucky-seed regression; worse than nb1561) |

**Ladder verdict:** nb1561 (BoB MEAN 0.5155) is the new top PRE-unblind candidate, displacing nb1450 (0.4990 honest-anchor was over-confident; nb1561 is the strongest *cross-fit* reproducible result). nb1571 fails the 0.003 margin gate so nb1561 stands. nb1582's K=20 Avalon was a "lucky-seed" mirage that did not reproduce in nested BoB.

## A. Top-15 SUBMISSION-READY candidates (have deploy CSV)

| Rank | Tag | honest_RAE | in_RAE(253) | predicted_LB | Confidence | Submission CSV |
|---:|---|---:|---:|---:|---|---|
| 1 | nb1450 | 0.4990 | 0.3250 | 0.5020 | honest-anchor | nb1450_deploy_nb1441_blend.csv |
| 2 | nb1481 | 0.4995 | 0.3701 | 0.5025 | cross-fit | nb1481_deploy_nb1471.csv |
| 3 | nb1453 | 0.4999 | 0.3889 | 0.5029 | honest-anchor | nb1453_deploy_nb1443.csv |
| 4 | nb1430 | 0.5022 | 0.3851 | 0.5052 | honest-anchor | nb1430_deploy_nb1422_mean.csv |
| 5 | nb1410 | 0.5079 | 0.4079 | 0.5109 | honest-anchor | nb1410_deploy_nb1403_mean.csv |
| 6 | **nb1571** | **0.5130** | - | **0.5160** | **cross-fit-blend (sub-margin)** | **nb1583_deploy_nb1571.csv** |
| 7 | **nb1561** | **0.5155** | - | **0.5185** | **BoB-MEAN nested 5x5 (NEW)** | **nb1570_deploy_nb1561_mean.csv** |
| 8 | **nb1554** | **0.5163** | - | **0.5193** | **CatBoost 5-way K-tuned (NEW)** | **nb1570_deploy_nb1561_mean.csv** |
| 9 | **nb1582** | **0.5184** | - | **0.5214** | **CatBoost Avalon-K20 BoB (DROPPED)** | n/a (not deployed) |
| 10 | nb1491 | 0.5231 | 0.3908 | 0.5261 | cross-fit | nb1491_deploy_nb1484.csv |
| 11 | nb1503 | 0.5231 | 0.3702 | 0.5261 | cross-fit | nb1503_deploy_nb1484_slsqp.csv |
| 12 | nb1480 | 0.5330 | 0.4088 | 0.5360 | cross-fit | nb1480_deploy_nb1472.csv |
| 13 | nb1470 | 0.5550 | 0.4376 | 0.5580 | cross-fit | nb1470_deploy_nb1460.csv |
| 14 | te_grand_v6b_calib | 0.6409 | 0.6409 | 0.6439 | legacy-PRE-unblind | grand_v6b_calib.csv |
| 15 | te_nb306_cepsmim | 0.6486 | 0.6486 | 0.6516 | legacy-PRE-unblind | nb306_cepsmim.csv |

### Absolute submission paths (top-10 deployable)

- **#1** `nb1450` -> `D:\Users\ashenoy00000\.windsurf\OpenADMET-pxr-challenge\submissions\nb1450_deploy_nb1441_blend.csv` (predicted LB **0.5020**)
- **#2** `nb1481` -> `D:\Users\ashenoy00000\.windsurf\OpenADMET-pxr-challenge\submissions\nb1481_deploy_nb1471.csv` (predicted LB **0.5025**)
- **#3** `nb1453` -> `D:\Users\ashenoy00000\.windsurf\OpenADMET-pxr-challenge\submissions\nb1453_deploy_nb1443.csv` (predicted LB **0.5029**)
- **#4** `nb1430` -> `D:\Users\ashenoy00000\.windsurf\OpenADMET-pxr-challenge\submissions\nb1430_deploy_nb1422_mean.csv` (predicted LB **0.5052**)
- **#5** `nb1410` -> `D:\Users\ashenoy00000\.windsurf\OpenADMET-pxr-challenge\submissions\nb1410_deploy_nb1403_mean.csv` (predicted LB **0.5109**)
- **#6** `nb1571` -> `D:\Users\ashenoy00000\.windsurf\OpenADMET-pxr-challenge\submissions\nb1583_deploy_nb1571.csv` (predicted LB **0.5160**) [NEW, sub-margin HOLD]
- **#7** `nb1561` -> `D:\Users\ashenoy00000\.windsurf\OpenADMET-pxr-challenge\submissions\nb1570_deploy_nb1561_mean.csv` (predicted LB **0.5185**) [NEW, BoB MEAN]
- **#8** `nb1554` -> `D:\Users\ashenoy00000\.windsurf\OpenADMET-pxr-challenge\submissions\nb1570_deploy_nb1561_mean.csv` (predicted LB **0.5193**) [NEW, 5-way K-tuned; shares deploy with nb1561]
- **#9** `nb1582` -> n/a (not promoted, predicted LB **0.5214**)
- **#10** `nb1491` -> `D:\Users\ashenoy00000\.windsurf\OpenADMET-pxr-challenge\submissions\nb1491_deploy_nb1484.csv` (predicted LB **0.5261**)

## B. Top-15 ALL candidates (including OOF-only diagnostics)

| Rank | Tag | honest_RAE | in_RAE(253) | predicted_LB | Confidence | Deployable |
|---:|---|---:|---:|---:|---|---|
| 1 | nb1450 | 0.4990 | 0.3250 | 0.5020 | honest-anchor | yes |
| 2 | nb1481 | 0.4995 | 0.3701 | 0.5025 | cross-fit | yes |
| 3 | nb1451 | 0.4996 | - | 0.5026 | cross-fit (OOF-only) | no |
| 4 | nb1453 | 0.4999 | 0.3889 | 0.5029 | honest-anchor | yes |
| 5 | nb1454 | 0.4999 | - | 0.5029 | cross-fit (OOF-only) | no |
| 6 | nb1471 | 0.5019 | - | 0.5049 | cross-fit (OOF-only) | no |
| 7 | nb1431 | 0.5020 | - | 0.5050 | cross-fit (OOF-only) | no |
| 8 | nb1422 | 0.5022 | - | 0.5052 | BoB-validated (OOF-only) | no |
| 9 | nb1430 | 0.5022 | 0.3851 | 0.5052 | honest-anchor | yes |
| 10 | nb1452 | 0.5023 | - | 0.5053 | BoB-validated (OOF-only) | no |
| 11 | nb1463 | 0.5025 | - | 0.5055 | cross-fit (OOF-only) | no |
| 12 | nb1444 | 0.5036 | - | 0.5066 | cross-fit (OOF-only) | no |
| 13 | nb1421 | 0.5037 | - | 0.5067 | cross-fit (OOF-only) | no |
| 14 | nb1413 | 0.5042 | - | 0.5072 | single-seed (OOF-only) | no |
| 15 | nb1411 | 0.5045 | - | 0.5075 | cross-fit (OOF-only) | no |

## C. Cycle 60-63 cross-fit / BoB diagnostic detail

- **nb1554** (5-way K-tuned CatBoost on chemprop_aux residual):
  - features: AtomPair K=25 + MACCS K=20 + Mordred K=20 + ChempropEmbed K=20 + Avalon K=30 (117-dim)
  - per-seed RAE: [0.5294, 0.5445, 0.5220, 0.5224, 0.5223]; mean_bag **0.5163**
  - Delta vs anchor chemprop_aux (0.6216): **-0.1053**
  - Delta vs nb1543 (0.5237): **-0.0074** -> promoted

- **nb1561** (BoB nested 5x5 reproduction of nb1554):
  - per-outer RAE: [0.5163, 0.5233, 0.5178, 0.5138, 0.5199]; mean 0.5182, std 0.0032
  - **BoB MEAN: 0.5155** (preferred deploy aggregator)
  - BoB MEDIAN: 0.5156
  - reproduces nb1554 within 0.001 -> stable

- **nb1571** (SLSQP blend of nb1561 + nb1560_LGBM):
  - in-sample best at w=0.55: 0.5120
  - **cross-fit RAE: 0.5130** (proper held-out evaluation)
  - delta vs nb1561 BoB (0.5155): -0.0025 -> **sub-margin, HOLD nb1561**

- **nb1582** (CatBoost K=20 Avalon instead of K=30):
  - per-outer RAE: [0.5262, 0.5260, 0.5203, 0.5178, 0.5147]; mean 0.5210
  - **BoB MEAN: 0.5184** -> WORSE than nb1561 (0.5155) by 0.0029
  - K=20 Avalon was a single-seed lucky-seed result (nb1581 = 0.514); nested BoB shows it does not reproduce
  - DROPPED, K=30 Avalon retained as canonical

## D. Source-key distribution (top-15 deployable)

- `legacy_pre_unblind_in_RAE`: 2 (nb14XX-era and below)
- `honest_lb_anchor*`: 4 (nb1450, nb1453, nb1430, nb1410)
- `honest_crossfit_RAE_nb14*`: 5 (nb1481, nb1491, nb1503, nb1480, nb1470)
- `catboost_5way_K_tuned`: 1 (nb1554)
- `catboost_BoB_mean`: 1 (nb1561)
- `catboost_BoB_K20_Avalon`: 1 (nb1582, dropped)
- `slsqp_crossfit_blend`: 1 (nb1571, sub-margin)
