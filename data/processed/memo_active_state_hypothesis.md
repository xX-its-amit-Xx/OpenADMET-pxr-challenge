# Research Memo: Active-State Stabilization vs. Affinity for PXR pEC50 Prediction

**Question:** Is modeling transcriptionally-active PXR states / coactivator recruitment / PXR–RXR a plausible route to improving pEC50 prediction beyond affinity-focused / ligand-only models?

**Verdict (up front, skeptical): NO — not a plausible route to a step-change, and for PXR *specifically* it is less promising than for almost any other nuclear receptor.** The hypothesis is *mechanistically correct in framing* (the label is activation, not binding) but the implied predictive signal is either (a) already implicit in 2D ligand structure, or (b) structurally ill-defined and empirically saturated for PXR. The one orthogonal axis (efficacy) is real but governs a ~3% partial-agonist tail AND does not explain our model's error. Confidence: high — three independent lines converge (assay design, structural biology, our own data + 20 yr literature).

---

## 1. Assay reverse-engineering (primary sources)

The pEC50 labels are from a **bespoke Octant Bio in-house cell-based reporter assay** (NOT the NCATS/Tox21 hPXR-luc assay; HF card + OpenADMET blog confirm "in-house data generation at Octant"). Architecture:

- **Construct: GAL4-DBD–PXR-LBD chimera (LBD-only transactivation).** OpenADMET announcement blog, verbatim: *"a two-part chimeric design… the ligand-binding domain (LBD) of human PXR attached to a heterologous DNA-binding domain, which acts via a reporter construct containing the corresponding DNA response element upstream of a luciferase gene."*
- **Readout:** NanoLuc luciferase (Promega Nano-Glo), 1536-well, 4,500 cells/well, 18–24 h agonist incubation; **agonism / transactivation mode**.
- **Cell line:** *not disclosed*; dox-inducible Tet-On stable single-copy integration, DMEM+10% FBS, poly-D-lysine → most consistent with a HEK293-Tet-On chassis (inference, not stated). Paired WT vs PXR-KO (nonsense-mutation null) lines.
- **Label fit:** Bayesian 3-parameter Hill (EC50, Emax, Hillslope); `pEC50 = -log10(EC50_M)`. Data ships `pEC50`, `emax`, `emax_rel`, with SEs and 95% CIs.
- Sources: OpenADMET announcement blog (`announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction`); Octant HTChem blog + `PXR Activation Assay Protocol.md`; local `data/raw/README.md`.

**Why this matters (decisive):** The GAL4-LBD chimera **deliberately removes** the native cascade — there is **no RXR heterodimerization on a native CYP3A4-XREM/PXRE element, and no native DNA binding**. The signal is *isolated to*: ligand → PXR-LBD agonist conformation → coactivator recruitment → reporter. Therefore **PXR–RXR complex modeling and DNA-binding/full-cascade features are mechanistically irrelevant to this label by construction** (Q on RXR-PXR is answered: no). Permeability/metabolic-stability over 18–24 h is a secondary whole-cell confounder.

## 2. Mechanistic analysis: what pEC50 represents here

The chimeric LBD-transactivation design means pEC50 is **one step removed from pure binding**: it reports how well the ligand drives the agonist-active LBD conformation + coactivator recruitment. So the hypothesis's *framing* is right: a tight binder that fails to stabilize the active LBD will score low. Ranked contributors to pEC50 variance:

| Contributor | Likely share of pEC50 variance | Capturable from ligand 2D? |
|---|---|---|
| Ligand binding/occupancy (affinity) | **Dominant** (within agonist series EC50≈affinity, r≈0.89; Ki≈EC50) | **Yes — already** |
| LBD active-conformation stabilization (efficacy) | Small; orthogonal tail (~3% partial agonists) | Partly (2D-predictable Emax) |
| Coactivator (SRC-1) recruitment efficiency | Graded but **downstream of pocket α12-stabilization**; interface near-constant | Implicit in above |
| RXR heterodimer | **Removed by assay design** / constant partner | N/A |
| DNA binding | **Removed by assay design** | N/A |
| Permeability / metabolic stability (18–24 h) | Secondary confounder | Partly (2D ADME) |

## 3. Literature: PXR activation structural mechanisms (the PXR-specific problem)

PXR is *uniquely hostile* to active-state structural features among NRs:

- **No canonical helix-12 "mousetrap."** PXR's AF-2/H12 is **pre-stabilized in the active position even apo** (Phe420–Leu411/Ile414 packing → high constitutive activity; Shizu/Wang 2021, PMC8390552). Ligand binding causes only "very moderate conformational changes" (Ngan/Vajda 2009, PMC5079256). → **There is no discrete ligand-defined "active state" to encode.**
- **Largest, most plastic NR pocket**, expanding by *disorder-on-demand* (~1150 Å³ apo → 1544 Å³ hyperforin → >1600 Å³); SR12813 binds in **3–7 distinct orientations in one pocket** (Watkins 2001 Science, 1ILG/1ILH; Motta 2018, PMC6212460). → **Single-pose structural descriptors are under-determined.**
- **Coactivator interface is near-constant across agonists** (PXR–SRC-1: 1NRL, 2O9I, 5X0R). The variance lives in the pocket/loops, not the SRC-1 groove → a coactivator-interface descriptor is **degenerate over an analog series** (mechanistic explanation of our Cycle-302 redundancy).
- **RXRα is a constant background** (4J5W/4J5X; agonist orders but does not remodel the interface; no full-length/cryo-EM PXR–RXR–DNA exists) → cancels per-analog.
- **Affinity vs efficacy DOES diverge** (SPA70 vs SJB7 differ by one para-methoxy, similar affinity, opposite efficacy; SJPYT-331 binds 3.6 nM but is an antagonist — Lin 2017 PMC5622171; Garcia-Maldonado 2024, PDB 8SVN–8SVX, PMC11094003). The 2024 structures localize the agonist switch to **filling the M425/L428/F429 cleft**. BUT this governs **agonist/antagonist identity + Emax (a tail)**, *not* EC50 rank within an agonist series.
- **Structure has NEVER beaten 2D for PXR potency** in 20 years of head-to-heads: Ekins 2009 (PLoS Comput Biol e1000594) 2D ROC 0.84 vs GOLD docking ~chance, CoMFA/4D-QSAR overfit; Khandelwal 2008 (PMC2574557) ligand-SVM 66.9% vs FlexX 51%; modern SOTA = ligand-only tree ensembles (Gou 2023 AUC 0.86). Conformational metrics (H12 displacement, IFP) exist only as **N≈3 binary classifiers, never regressed vs continuous EC50**.

## 4. Structural-modeling opportunities — assessed (do not assume they work)

| Approach | Assessment for THIS task |
|---|---|
| Docking / ensemble docking / MM-GBSA | **Dead.** Multi-pose ambiguity + binding≠activation; our Cycle-294 Vina = signal-less (corr −0.03); 20 yr literature agrees. |
| AF3 / Boltz-2 / Chai-1 cofold **interaction (z) embedding** | The **only** approach with weak real signal: our rich-z ≈ −0.008 RAE (Cycle-295), and the structural biology agrees this (ligand→α12 contact stability, marginalized over the diffusion ensemble) is the *only* ligand-discriminating active-state quantity. **But already mined + saturated** (Cycle-298: 5 independent cuts all redundant ~−0.010). |
| Coactivator (SRC-1) ternary cofold | **Tested, redundant/negative.** Cycle-302 added −0.0003 over rich-z; Cycle-309 train-side ternary on the honest gate **HURTS +0.0082**. Structural biology explains why: coactivator interface is constant across agonists. |
| PXR–RXR heterodimer cofold | **Irrelevant** — removed by the GAL4-LBD assay design AND a constant partner. |
| MD-derived active-state probability / ensemble stats | Conceptually the right idea, but our pose-fluctuation features (Cycle-298) already capture a cheap version and saturated; full MD at n=4139 is enormous cost for a feature class already shown redundant. |
| Helix-12 displacement / AF-2 geometry | Ill-defined for PXR (pre-formed AF-2); only ever an N≈3 binary classifier in the literature. |

## 5. Specific feature ideas (and why each likely fails here)

1. **M425/L428/F429 agonist-cleft occupancy** (from 2024 antagonist structures) — the single most *specific*, mechanistically-novel idea. A pose-derived score for whether the ligand fills the cleft that flips antagonist→agonist. *Risk:* requires resolving PXR's 3–7-pose ambiguity; governs the ~3% antagonist/partial tail; per-compound pose is under-determined.
2. **Predicted-Emax as an efficacy correction** (`pEC50_corr = pEC50 + γ·log10(Emax)`) — **already built (nb100), nb1070 Emax-aux, nb107 assay-decomposition.** Killer fact below.
3. **Cofold ensemble variance / contact-stability of the ligand→helix-12 residues** — = our rich-z + geom, already the one weak win, saturated.
4. **Coactivator-interface stabilization score** — degenerate (constant interface).
5. **Active-state probability from a classifier head** — no continuous-EC50 precedent; for an agonist-heavy set it's near-constant.

## 6. The decisive empirical test (our own data)

- **Emax DOES decouple from potency** (corr(pEC50, emax) = −0.116; corr(pEC50, emax_rel) = −0.133) — the hypothesis's premise is *real*.
- **But partial agonists are only 2.9%** of compounds (emax_rel < 0.5; median = 1.0 full agonist).
- **DECISIVE:** the magnitude of our strong ligand-only model's error is **uncorrelated with efficacy**: corr(|error|, emax_rel) = **−0.018 ≈ 0** (signed corr(error, emax_rel) = −0.136, i.e. a tiny linear tilt at most). **Even with perfect active-state/efficacy knowledge, the compounds we get wrong are NOT the efficacy-discordant ones.** The active-state axis does not live where our residual error lives.
- **Emax is train-only** (the 513 test is blinded SMILES) → to use it you must *predict* Emax from the ligand, and an Emax predictor is itself a 2D problem (absorbed by the chempropembed sink). No free lunch.
- Prior work already covered this: `nb100_emax_correction`, `nb1070_emax_aux`, `nb107_assay_decomposition` — none is in the deployed model, consistent with the ≈0 residual correlation.

## 7. Experimental plan (IF pursued despite the verdict — minimal, cheap, falsifiable)

Only one probe is worth the keystrokes, and it's a *label* probe, not a structural campaign:
- **P1 (cheap, 1 worker tick):** Re-run the Emax-correction honestly: cross-fit an Emax predictor from current features, apply `pEC50 + γ·log10(Emax_pred)` (tune γ on never-tuned holdouts only), honest-gate vs 0.4268. *Falsifies in one tick.* Expected: null (corr(|error|,emax)≈0 predicts no gain).
- **P2 (only if P1 shows ANY signal):** M425/L428/F429-cleft occupancy from the existing Boltz cofold poses (we already have them) as a single scalar; honest-gate. Expected: null/tail-only.
- **Do NOT** run: new RXR/coactivator cofold, MD ensembles, docking — all tested or irrelevant.

## 8. Expected upside

- **Realistic:** ~0.000 to −0.002 RAE, on the ~3% partial-agonist tail at most; below the n=253 noise floor and the blinded-transfer threshold.
- **Optimistic ceiling (if the cleft-occupancy idea somehow generalizes):** ~−0.003, almost certainly selection-inflated. No path to a step-change.

## 9. Major risks

- **Overfitting trap (highest):** exactly the Ekins-2009 pattern — good internal CV, fails external — which maps onto our documented blinded-transfer risk (Cycle-305, adversarial AUC 0.984 train-vs-blinded). A small-n structural feature tuned on 253 will not transfer to the blinded 260.
- **Pose under-determination:** PXR's 3–7-orientation pocket makes any single-pose feature noise-dominated.
- **Opportunity cost:** GPU/effort spent here is not spent on the only proven step-change lever (new on-manifold measurements — the 2026-07-01 unblind).

---

## Recommendation

**Do not invest in active-state / coactivator / PXR–RXR structural modeling as a route to beating ligand-only models on this challenge.** The hypothesis is mechanistically sound but, for PXR specifically, the active state is not a discrete encodable object (pre-formed AF-2, promiscuous breathing pocket, constant coactivator/RXR), the orthogonal efficacy axis is a ~3% tail that does not explain our error, and the one structural feature class with weak real signal (cofold interaction-z) is already mined and saturated. This is independently corroborated by 20 years of PXR literature (2D > structure, every head-to-head) and by our own Cycles 294/295/298/302/309.

**Allowed exception:** the single cheap, falsifiable Emax-correction probe (P1) — run it in one worker tick to formally close the efficacy question on the current best (0.4268); expect null. The real step-change remains the **2026-07-01 Analog-Set-1 unblind** (new on-manifold labels), not structural features.

*Most important takeaway:* The reason structure fails here is not that activation is the wrong mechanism — it is that **PXR's activation is "always-on" and pocket-promiscuous, so the activation signal that varies across analogs is precisely the part that is already encoded in 2D ligand structure (affinity), while the part that is genuinely structural (efficacy tail) is tiny, train-only, and uncorrelated with where our models actually err.**
