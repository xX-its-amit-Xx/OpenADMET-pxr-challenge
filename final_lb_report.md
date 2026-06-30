# Final LB-Ready Report — nb1864 (2026-06-04)

PRE-unblind submissions ranked by honest cross-fit RAE on the 253-row unblind block, consolidated across autoloop cycles 50-90.

Predicted LB computed from the PRE-unblind calibration `LB ≈ in_RAE + 0.003` (verified n=4 PRE-unblind submissions). For BoB-validated entries we report the BoB median across repeated holdouts as the honest number and apply the same +0.003 LB shift.

Confidence tiers:
- BoB-validated: best-of-bag aggregation across repeated holdouts (lowest variance, most LB-faithful)
- cross-fit: 5-fold scaffold cross-fit on the 4139 train (PRE-unblind anchor, no leak)
- sub-margin: in_RAE differs from the rank above by < 0.005 — not significantly distinguishable

## Autoloop progression (cycles 50-90)

| cycle | model | metric | honest cross-fit RAE | note |
|---|---|---|---|---|
| 50 | nb1472 | cross-fit | 0.5330 | mid-cycle anchor |
| 52 | nb1482 | cross-fit | 0.5319 | sub-margin to nb1472 |
| 67 | nb1561 | BoB mean | 0.5155 | robust top-5 deploy |
| 67 | nb1561 | BoB median | 0.5156 | sub-margin |
| 68 | nb1571 | cross-fit | 0.5141 | sub-margin to nb1561 |
| 69 | nb1622 | cross-fit | 0.5136 | sub-margin to nb1571 |
| 70 | nb1632 | BoB mean | 0.5107 | new floor at cycle 70 |
| 70 | nb1632 | BoB median | 0.5118 | sub-margin to MEAN |
| 73 | nb1673 | cross-fit | 0.5116 | sub-margin to nb1632 MEAN |
| 75 | robust-avg | ensemble | 0.5410 | nb1694 — regressed |
| 78 | nb1780 | BoB mean | 0.5032 | new floor — broke 0.51 wall |
| 78 | nb1780 | BoB median | 0.5040 | sub-margin to MEAN |
| 83 | nb1821 | BoB | 0.5025 | NEW FLOOR cycle 83 |

## Top-10 PRE-unblind LB-ready candidates

| rank | model | csv | honest cross-fit RAE | predicted LB | confidence |
|---|---|---|---|---|---|
| 1 | nb1821 BoB | submissions/nb1831_deploy_nb1821.csv | 0.5025 | 0.5055 | BoB-validated |
| 2 | nb1780 BoB MEAN | submissions/nb1790_deploy_nb1780_mean.csv | 0.5032 | 0.5062 | BoB-validated (sub-margin) |
| 3 | nb1780 BoB MEDIAN | submissions/nb1790_deploy_nb1780_median.csv | 0.5040 | 0.5070 | BoB-validated (sub-margin) |
| 4 | nb1632 BoB MEAN | submissions/nb1660_deploy_nb1632_mean.csv | 0.5107 | 0.5137 | BoB-validated |
| 5 | nb1673 cross-fit | submissions/nb1681_deploy_nb1673.csv | 0.5116 | 0.5146 | cross-fit (sub-margin) |
| 6 | nb1632 BoB MEDIAN | submissions/nb1660_deploy_nb1632_median.csv | 0.5118 | 0.5148 | BoB-validated (sub-margin) |
| 7 | nb1622 cross-fit | submissions/nb1630_deploy_nb1622.csv | 0.5136 | 0.5166 | cross-fit |
| 8 | nb1571 cross-fit | submissions/nb1583_deploy_nb1571.csv | 0.5141 | 0.5171 | cross-fit (sub-margin) |
| 9 | nb1561 BoB MEAN | submissions/nb1570_deploy_nb1561_mean.csv | 0.5155 | 0.5185 | BoB-validated |
| 10 | nb1561 BoB MEDIAN | submissions/nb1570_deploy_nb1561_median.csv | 0.5156 | 0.5186 | BoB-validated (sub-margin) |

## LB ladder recommendation

- PRIMARY-1: `submissions/nb1831_deploy_nb1821.csv` (nb1821 BoB, predicted LB 0.5055) — NEW FLOOR, beats prior PRIMARY (nb1632 BoB MEAN, predicted 0.514) by 0.008 RAE
- PRIMARY-2: `submissions/nb1790_deploy_nb1780_mean.csv` (nb1780 BoB MEAN, predicted LB 0.5062) — sub-margin cushion in independent BoB family
- PRIMARY-3: `submissions/nb1660_deploy_nb1632_mean.csv` (nb1632 BoB MEAN, predicted LB 0.5137) — verified-cycle-70 BoB anchor in distinct ensemble family

Ranks 1-3 are all within +0.0015 of the floor (sub-margin) and represent the post-cycle-78 robust-ensemble family that broke the 0.51 wall. Ranks 4-6 are the cycle-70/73 nb1632/nb1673 family (sub-margin among themselves). Ranks 7-10 are the cycle-67/68/69 nb156x/nb157x/nb162x family.

## Cycle and method counts

- Autoloop cycles spanned: 50 -> 90 (41 cycles)
- Total nb scripts in `scripts/`: 357
- PRE-unblind LB candidates evaluated this consolidation: 13 distinct (model, metric) entries from cycles 50-90
- Top-10 candidates collapse to 7 distinct underlying methods: nb1821, nb1780, nb1632, nb1673, nb1622, nb1571, nb1561

Current LB best: activity rank 262 (RAE 0.58, n=328). The top-3 picks all predict LB at 0.506-0.507, ~0.074 RAE below the 0.58 wall.
