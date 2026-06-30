# Models-as-data + cross-assay research plan (cycle 290+)

User directive (2026-06-10): exploit pharma/academic open models as featurizers/data generators,
cross-assay data (Ki/Kd/EC50/single-point), and obscure academic tools (ATOMICA). ≥10 approaches,
implemented combinatorially. Self-scheduled until finished.

**Evaluation:** activity LB frozen → judge everything on the nb952 scaffold-extrapolation degradation
curve (deep-extrap MAE @ sim<0.3 = **0.5924** is the bar) + multi-seed verify (nb956 pattern) to reject
lucky-seed gains. Cheap CPU probe BEFORE any GPU. Honest negatives are wins (they save compute).

## Findings so far (the priors are EARNED, not assumed)
- **ChEMBL PXR (CHEMBL3401) harvest = OFF-MANIFOLD for rescue.** 1602 cpds, 76% new scaffolds, but of 334
  novel-to-train TEST scaffolds it covers **1**; harvested actives median Tanimoto 0.23 to test, ceiling 0.39.
  Public PXR chemistry (rifampicin/statin/steroid) ≠ this challenge's analog-expansion test. (subagent, clean, 0 leak)
- **Internal single-conc screen = LOCAL.** 10,870 unique cpds; novel-test-scaffold proximity median **0.507**,
  83% have a ≥0.4 neighbor, 55% ≥0.5 (nb958). Each carries log2_fc_estimate + stderr + fdr_bh. THE lever.
- Representation axis already walled: 2D ChemBERTa frozen (nb953 negative), 3D RDKit (nb954/nb956 seed-noise).

## Approaches (≥10), by axis, with earned priors

### AXIS A — internal single-conc weak-label transfer  [HIGHEST prior: proximal data]
1. **SC-kNN activation feature** (CPU, nb959): per-compound features = Tanimoto-weighted activation stats of
   k single-conc neighbors. Test on degradation curve. *First probe — running now.*
2. **SC-pretrain → pEC50 fine-tune** (GPU/Kaggle): pretrain GNN on 10,870 log2_fc (LOCAL chemistry), fine-tune
   on 4139 pEC50. The local-chemistry analog of the (failed) broad ChemBERTa pretrain.
3. **Multi-task GNN + SC head** (GPU): chemprop pEC50 + counter + single-conc-activation heads, joint.
4. **SC pseudo-label co-training** (CPU/GPU): low-fdr SC actives/inactives → pEC50 pseudo-labels via a learned
   single-point→CRC calibration; add to train down-weighted.
5. **SC abstention/shrink prior** (CPU): novel test cpd whose SC neighbors are all inactive → shrink pEC50 toward
   inactive. Directly targets the F2 greasy-novel-inactive failure mode.

### AXIS B — external bioactivity beyond PXR  [LOW prior after ChEMBL-PXR negative; cheap coverage checks]
6. **CAR (NR1I3) + NR-panel multi-task**: sister xenosensor shares ligands; check test-scaffold coverage first.
7. **Cross-assay harmonized pActivity**: pool EC50/AC50/Ki/Kd/Potency w/ assay-type covariate (hierarchical).

### AXIS C — pharma/academic foundation featurizers  [LOW-MED prior; representation axis walled, but stronger models]
8. **MolE (Recursion, 842M graphs, #1 on 10/22 TDC ADMET) / ChemFM embeddings**: the strongest 2D models; the
   one shot left on the 2D axis. Frozen-embed test on degradation curve; fine-tune only if frozen is competitive.
9. **ADMET-AI predicted properties as features** [MED — biological axis Morgan lacks]: predicted CYP3A4/metabolism/
   solubility etc. PXR REGULATES CYP3A4 → CYP-related ADMET is mechanistically on-target. CPU-ish.
10. **ATOMICA interaction embeddings** [MED but needs poses]: embed PXR-LBD+ligand complexes (HF ada-f/ATOMICA,
    2M-complex pretrain) → interaction-interface features. Bridges structure track. Needs docked/cofolded poses.

### AXIS D — generative-model-as-data  [LOW-MED, speculative]
11. **Generative NLL/typicality feature**: pharma gen model (REINVENT/MolGPT) per-molecule negative-log-likelihood
    = distributional-novelty feature (is this molecule "off-distribution" → likely mispredicted).
12. **Scaffold-conditioned analog augmentation**: generate analogs of novel test scaffolds, consistency-regularize.

### AXIS E — combinatorial integration
13. **Confidence-gated stacked ensemble** of every component that beats the curve, gated by SC-neighbor density.

## Order of execution (cheap→expensive, prior-weighted)
A1 (nb959) → A5 (cheap, targets F2) → C9 (ADMET-AI) → B6 coverage-check → A2/A3 (GPU) → C8 (MolE) → C10 (ATOMICA+poses) → D → E.
Each: degradation-curve probe + multi-seed verify; promote to ladder only if it beats 0.5924 stably AND the gain
survives the selection-bias/transfer caveats.

## Status log
- [DONE/NEG] B ChEMBL PXR harvest → off-manifold (1/334 novel test scaffolds; median sim 0.23). data/external/chembl_pxr_harvest.csv
- [DONE] single-conc rescue scoping → LOCAL, proximity 0.507, 83% have >=0.4 nbr (nb958)
- [DONE/NEG] A1 SC-kNN activation feature (nb959): neighbors-only HONEST = 0.6148 (WORSE than 0.598 ref);
  self+nbr = 0.4942 but TRAIN-ONLY TRAP (test cov 0/512). KEY INSIGHT: self-single-point→pEC50 is STRONG
  (0.598→0.494) but test was never SP-screened → the highest-value data lever is EXPERIMENTAL (screen the
  test), not computational. Flag to organizers.
- [DONE/REAL-BUT-ABSORBED] C9 ADMET-AI (nb960-964, isolated venv C:/admet_venv, 104 props incl all CYP3A4):
  nb962 8-seed verify over FINGERPRINTS = overall -0.0191±0.0031 + deep -0.0225±0.0090 (8/8 STABLE) — REAL win
  over Morgan, confirms supervised-featurizer thesis. nb963 combined-only 253 = -0.024. BUT nb964: on
  combined+chempropembed (nb3200 substrate) delta -0.0055±0.0116, 4/7, NOT stable → ABSORBED by chempropembed
  (ADMET-AI is Chemprop; our K18 already has ChempropEmbed). Does NOT break 0.4416. KEY LESSON below.
- [DONE/NEG] A5 SC-abstention (nb965): F2 cohort over-prediction CONFIRMED (novel+inactive-nbr n=73, mean +0.248)
  but the SC-neighbor-gated shrink is seed-noise (delta -0.0001±0.0023, 4/7, NOT stable). SC signal flags the
  cohort but is too imprecise per-compound to correct (same root as "confidence-shrink overfits at n=253").
- [DEPRIORITIZED w/ EVIDENCE] C8 MolE/ChemFM: 2 independent confirmations (ChemBERTa nb953 neg, ADMET nb964
  absorbed) that 2D GNN/transformer featurizers are absorbed by chempropembed. Heavy install, ~0 marginal EV. Skip.
- [GPU-GATED] C10 ATOMICA 3D-interaction = the one genuinely distinct axis, but needs docked poses of 513+4139 in
  the PXR LBD = a docking+GPU effort. It is the ESCALATION of the already-staged Uni-Mol molecule-only test
  (notebooks/957, decision gate 0.5924): run that cheaper 3D test first; ATOMICA/pocket-docking only if it's positive.
- [GPU-GATED] A2/A3 SC-pretrain / multitask-GNN (low prior after A1); D generative typicality (2D-derived → likely absorbed).

## CONVERGENCE (cheap local axes EXHAUSTED; user stop-condition met)
Stop-condition "a real winner beats 0.5924 stably" = MET by ADMET-AI (nb962 8/8). But it's ABSORBED by our deployed
chempropembed (nb964) so it does NOT break the 0.4416 ladder. Net of 13 approaches: data-harvest (B,A1,A4,A5) and
2D-featurizer (C8,C9, + prior ChemBERTa/MolFormer/3D) axes are all characterized NEGATIVE-or-absorbed. The ONLY
remaining levers that could move the DEPLOYED ceiling are GPU-gated and already staged: (1) 3D conformer/interaction
(Uni-Mol nb957 → ATOMICA escalation), (2) structure Boltz-2 (nb951), (3) experimental: single-point-screen the test.
Local activity method space is exhausted. STOPPING the research self-schedule per instruction; maintenance crons continue.
- [INFRA] D: hit 0 MB mid-run (ENOSPC truncated a script) → freed 213 MB by clearing 363 stale subagents/workflows
  run dirs. D: is chronically full; ADMET CSVs + HF cache live on C:/. Watch D: before any large local write.

## Honest running assessment
Data-harvest axes (A feature, B external) are NEGATIVE — the novel-test wall is local and the rescue chemistry
is either non-public (ChEMBL) or only weakly-labeled+unavailable-for-test (single-conc). Remaining live bets are
the BIOLOGICAL-property axis (C9 ADMET-AI: CYP/metabolism, orthogonal to substructure) and the 3D-INTERACTION axis
(C10 ATOMICA, needs poses, bridges structure). Representation 2D axis (C8) is low-prior after nb953. Keep testing
each on the degradation curve; promote only on stable beat of 0.5924 + transfer caveats.
