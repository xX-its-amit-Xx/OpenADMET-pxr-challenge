# Deep Cofolder Search — Underground / Orthogonal Models for the PXR Pose Ensemble

Date: 2026-06-20
Goal: surface protein-ligand cofolding/docking models we are SLEEPING ON — especially lesser-known, recent (2024-2026), with a UNIQUE training set or architecture, that are both ORTHOGONAL to the AF3-diffusion family AND plausibly accurate (>~0.7 PoseBusters / AF3-class). Hard lesson encoded: orthogonality is worthless without high base accuracy (RoseTTAFold-AA dragged the pool at ~half AF3 accuracy).

Current pool (do not re-test): Boltz-1/2, OpenFold3, AlphaFold3, Chai-1, Protenix-v1/v2 (all AF3-family diffusion); RFAA + RF3 (testing); NeuralPlexer-1 (testing); HelixFold3 (clone, skip); ESMFold (protein-only). Closed: NeuralPlexer-2/3, PEARL, Chai-2.

---

## CRITICAL CONTEXT FOR OUR TEST SET

Two independent 2025 findings make the "orthogonal training data" thesis fragile for our PanDDA-fragment test set:

1. **Cofolding accuracy is predicted almost perfectly by training-set similarity** — models do NOT extrapolate to unusual unseen systems (Nat Commun 2025, s41467-025-63947-5). Our test ligands are PanDDA fragments with sim<0.3 to all crystals, i.e. exactly the extrapolation regime. ANY new model is at risk here, AF3-family or not.
2. **No publicly-trained cofolder is known to use XChem/PanDDA fragment-screening crystal data as a distinct training source.** I searched specifically for this (DFT/cryoEM/fragment/XChem/PanDDA-trained cofolders) and found NONE. The XChem/PanDDA "big data on small fragments" work (Nat Commun 2025, s41467-025-59233-z; Fearon 2025) is about *data infrastructure*, not a trained pose model. **This is a genuine gap — there is no fragment-trained cofolder to grab.** The closest orthogonal-data signals are DEL-trained scorers (Hermes) and pretrained-on-docked-decoys models (CarsiDock, Uni-Mol), not fragment-crystal-trained generators.

Implication: the highest-value adds are models that are (a) AF3-class accurate AND (b) architecturally orthogonal so they make *different errors* on flexible analogs — value comes from ensemble error-decorrelation, not from "learned fragment chemistry" (which no model has).

---

## TOP 3-5 GENUINELY-NEW CANDIDATES TO INSTALL + GT-TEST (ranked)

### 1. IntFold (IntelliFold) — HIGHEST VALUE
- **Single reason:** AF3-class accuracy (PoseBusters v2 **76.1%**, beats Protenix 72.6%; FoldBench protein-ligand 58.5%, #2 after AF3) with a genuinely useful orthogonality lever: **modular controllable adapters** for constraints/binding-affinity/allosteric states — lets us inject pocket anchors (Ser247/Gln285) as native constraints rather than hacks. Apache-2.0, open weights on HuggingFace, `pip install intellifold`.
- Open weights + license: **Apache 2.0 — commercial OK.** Weights on HF + PyPI.
- HPC install effort: **1-2** (pip; standard CUDA-12 torch).
- Type: **true cofolding** (AF3-style, sequence+SMILES). IntFold+ variant further improves the PL interface to 61.8%.
- Orthogonality: moderate — it is AF3-family-adjacent (Triangle-attention + diffusion lineage), but a *separately trained* network with controllable adapters; its errors should partly decorrelate from Boltz/Chai. The adapter control is the unique differentiator, not the backbone.
- Accuracy bar: **CLEARS IT (76% PoseBusters).**
- Date/URL: arXiv 2507.02025 (Jul 2025); https://github.com/qiaoqiaoLF/IntFold

### 2. SurfDock — STRONGEST ARCHITECTURAL ORTHOGONALITY THAT IS ALSO ACCURATE
- **Single reason:** the only high-accuracy model whose representation is fundamentally NON-sequence/NON-MSA: a **MaSIF molecular-surface + ESM-2** SE(3) diffusion over the binding-site *surface geometry*. This is the most orthogonal learned signal to the entire AF3/MSA family while still topping PoseBusters/Astex/PDBbind2020 — exactly the "orthogonal AND accurate" profile we want.
- Open weights + license: open on GitHub (CAODH/SurfDock), weights included; Nature Methods 2024.
- HPC install effort: **3** (conda py3.10, torch 2.2.2, PyG, MaSIF surface-precompute step adds friction).
- Type: **pocket docking** (needs a pocket/holo protein, not blind cofold). We have pockets from the apo soaks + our Boltz/AF3 holo predictions, so usable — feed it Boltz/Chai holo pockets like FlowDock does.
- Orthogonality: **HIGH** (surface-fingerprint representation absent from every model we run).
- Accuracy bar: **CLEARS IT** (reported top of PoseBusters/Astex/DEKOIS, ~90%+ PB-valid).
- Date/URL: bioRxiv 2023.12.13.571408; https://github.com/CAODH/SurfDock

### 3. Uni-Mol Docking V2 — ORTHOGONAL TRAINING DATA + ACCURATE + CHEMICALLY CLEAN
- **Single reason:** trained on **millions of large-scale pretrained molecular/pocket representations** (Uni-Mol 3D pretraining, an entirely different data pipeline from PDB-MSA cofolders) and explicitly engineered to KILL chirality inversions and steric clashes — the exact failure mode that hurts on flexible drug-like analogs. **77% PoseBusters <2 Å, 75% pass all quality checks.**
- Open weights + license: **MIT**, weights + data on GitHub (deepmodeling/Uni-Mol). (Note: hosted *service* is non-commercial, but the MIT code+weights are not — verify before commercial use; for our research/Kaggle use it's fine.)
- HPC install effort: **2** (mature deepmodeling stack).
- Type: **pocket docking** — requires a docking grid (center + box JSON). We have pocket centers (Ser247/Gln285).
- Orthogonality: **HIGH** (Uni-Mol SE(3) transformer + 3D molecular pretraining; no MSA, no diffusion-cofold).
- Accuracy bar: **CLEARS IT (77%).**
- Date/URL: arXiv 2405.11769 (May 2024); https://github.com/deepmodeling/Uni-Mol (unimol_docking_v2)

### 4. PocketXMol — UNIQUE GENERATIVE FOUNDATION MODEL, FRAGMENT-NATIVE
- **Single reason:** a **pocket-interacting generative foundation model** (Cell 2026) that natively handles docking AND fragment linking/growing — the ONLY candidate explicitly built around fragment-scale chemistry, which is what our test set IS. MolDiff-derived (orthogonal to AF3 entirely). MIT.
- Open weights + license: **MIT**, weights + data on Zenodo; https://github.com/pengxingang/PocketXMol
- HPC install effort: **3** (MolDiff-style env, Zenodo asset download).
- Type: **pocket docking / pocket-conditioned generation** (small-molecule docking is a supported task).
- Orthogonality: **VERY HIGH** (generative-diffusion-over-molecule, not cofold; fragment-aware training tasks).
- Accuracy bar: **UNCERTAIN** — paper reports PoseBusters/PDBbind-MOAD test sets but I could not confirm the headline RMSD<2 Å rate; benchmark numbers not surfaced. **Install but GT-test BEFORE trusting** — risk it is a design model first, docking second (RFAA lesson: verify accuracy before adding to pool).
- Date/URL: Cell 2026; https://github.com/pengxingang/PocketXMol

### 5. CarsiDock — ACCURATE, ORTHOGONAL DATA, BUT WATCH PHYSICAL VALIDITY
- **Single reason:** pretrained on **millions of *predicted* (docked-decoy) protein-ligand complexes** — a training distribution unlike any PDB-crystal cofolder — and posts **79.7% top-1 PoseBusters** (above many cofolders). Comes with CarsiInduce (pocket induced-fit refine) and CarsiDock-Cov (covalent), useful side-tools.
- Open weights + license: **Apache-2.0 code; weights free for academic** (commercial needs contact). Weights on Google Drive via repo.
- HPC install effort: **2-3**.
- Type: **pocket docking** (needs pocket).
- Orthogonality: **HIGH** (Uni-Mol-style large-scale pretraining on synthetic complexes; no MSA/cofold).
- Caveat: **PB-validity only 47.7%** (below Vina) — generates good-RMSD but sometimes physically-strained poses. Must gate through PoseBusters before submitting; risky as a raw pool member but strong as a *diversity* source + re-rank.
- Date/URL: PubMed 38274053 (2024); https://github.com/carbonsilicon-ai/CarsiDock

---

## ORTHOGONAL-DATA SCORERS / RE-RANKERS (not pose generators — use to SELECT, not generate)

- **Hermes (DEL-trained PLI model)** — **the standout unique-training-data find.** Lightweight transformer trained EXCLUSIVELY on **DNA-encoded-library (DEL) screens vs hundreds of targets** — the largest, most target-diverse DEL training set for PLI modeling, and a data source NO cofolder uses. Generalizes to held-out targets/scaffolds. It is a **binding/PLI predictor, not a pose generator** — so its role is an *orthogonal cross-pose / cross-model selection signal* (our documented wall is selection, not pool quality). High value as a re-ranker to break the IPDE/plddt ceiling. arXiv 2602.13503 (2026). https://arxiv.org/abs/2602.13503
- **Interformer** — Graph-Transformer with interaction-aware mixture-density-network energy; open (Nat Commun 2024, s41467-024-54440-6). Good as a *physics-aware pose re-scorer* for our existing poses; not a strong standalone generator. https://www.nature.com/articles/s41467-024-54440-6
- **Boltz-2 affinity head / Boltzina** — already in family; affinity signal for selection (noted, not new).

---

## CHECKED BUT SKIP (one-line reasons)

- **NeuralPlexer-3** — flow-based, physio-realistic, induced-fit; **no open weights found** (closed, consistent with our memo). Skip until released. arXiv 2412.10743.
- **FlowDock** — open MIT flow-matching, apo→holo, multi-ligand; but **only ~51% PoseBusters blind** — below our accuracy bar (RFAA lesson). Skip as pool member; could fine-tune on PLINDER later. github.com/BioinfoMachineLearning/FlowDock.
- **Umol / Umol2** — open Apache-2.0 cofolder, but documented to **underperform RFAA at high-accuracy threshold**; RFAA already too weak for us → Umol weaker. Skip. github.com/patrickbryant1/Umol.
- **ArtiDock** — 29-38% better than Vina/Glide and fast, BUT **commercial/HTVS-oriented; weights not openly released** (JCIM 2025, behind ACS; bioRxiv 2024.03.14.585019). Skip unless weights surface.
- **DiffDock-L / DiffDock-Pocket / DiffDock-PP** — older diffusion docking, **sub-50% PoseBusters, poor PB-validity**; superseded. Skip.
- **DynamicBind / Re-Dock / FlexDock** — flexible/induced-fit diffusion; interesting architecture but **mid-tier accuracy, no PoseBusters dominance**; lower priority than SurfDock/Uni-Mol. Skip for now.
- **KarmaDock / TankBind / E3Bind / EquiBind / DeltaDock / PLANTAIN / TEMPL** — older or template-baseline regression dockers, **below cofold accuracy**; skip.
- **GenMol (NVIDIA), PocketXMol-design tasks** — molecular *generation*, not pose prediction of a fixed ligand. N/A.
- **HelixFold3, OpenFold3 NIM, SiteAF3, IntFold backbone duplicates** — AF3-family clones / already-have. Skip (SiteAF3 = AF3 site-conditioning, useful idea but server-bound).
- **PEARL** — Genesis foundation model, **closed**. Skip (known).

---

## EXPLICIT FRAGMENT / PanDDA / XChem CALLOUT

- **No cofolding model trained on XChem/PanDDA fragment-screening crystals exists publicly** (searched specifically). The XChem/PanDDA 2 ecosystem (Diamond) is data-infrastructure, not a released pose model. This remains our unique unfair-advantage gap: *we hold fragment data others' models never saw, and there is no off-the-shelf model that did.*
- **PocketXMol is the only candidate with fragment-native training tasks** (fragment linking/growing/SBDD) — closest thing to "fragment chemistry," worth a GT-test for that reason alone, but verify accuracy first.
- **Hermes (DEL)** is the most genuinely-orthogonal training data found, but it scores rather than poses.

---

## RECOMMENDED ACTION ORDER

1. **IntFold** — install first (pip, Apache-2.0, 76% PoseBusters, controllable-constraint adapters for our pocket anchors). Lowest effort, highest accuracy-confidence new add.
2. **SurfDock** — install second (most orthogonal accurate representation; feed it our Boltz/Chai holo pockets). The decorrelation play.
3. **Uni-Mol Docking V2** — install third (MIT, 77%, chirality/clash-clean, orthogonal Uni-Mol pretraining; we have pocket grids).
4. **Hermes** — wire in as a cross-model selection signal (DEL-orthogonal; attacks the selection wall directly).
5. **PocketXMol** — GT-test (fragment-native) but DO NOT pool until PoseBusters-verified; **CarsiDock** likewise (gate on PB-validity).

All five clear the "orthogonal" requirement; IntFold/SurfDock/Uni-Mol clear the accuracy bar with published numbers; PocketXMol/CarsiDock need our own GT harness (scripts/pose_lib.py + validate_selection.py, 18 holos) to confirm before they touch a submission — per the RFAA lesson.

---

## SOURCES
- PoseX benchmark + leaderboard: https://github.com/CataAI/PoseX ; arXiv 2505.01700
- PoseBench benchmark: https://github.com/BioinfoMachineLearning/PoseBench
- IntFold: arXiv 2507.02025 ; https://github.com/qiaoqiaoLF/IntFold
- SurfDock: bioRxiv 2023.12.13.571408 ; https://github.com/CAODH/SurfDock
- Uni-Mol Docking V2: arXiv 2405.11769 ; https://github.com/deepmodeling/Uni-Mol
- PocketXMol: Cell 2026 ; https://github.com/pengxingang/PocketXMol
- CarsiDock: PubMed 38274053 ; https://github.com/carbonsilicon-ai/CarsiDock
- Hermes (DEL): arXiv 2602.13503 ; https://arxiv.org/html/2602.13503v1
- Interformer: Nat Commun 2024 s41467-024-54440-6
- FlowDock: arXiv 2412.10966 ; https://github.com/BioinfoMachineLearning/FlowDock
- NeuralPlexer3: arXiv 2412.10743
- Umol: https://github.com/patrickbryant1/Umol ; Nat Commun s41467-024-48837-6
- ArtiDock: JCIM 2025 10.1021/acs.jcim.5c02777 ; bioRxiv 2024.03.14.585019
- "Cofolding learns physics?" (training-similarity ceiling): Nat Commun 2025 s41467-025-63947-5
- XChem/PanDDA data infra (NOT a model): Nat Commun 2025 s41467-025-59233-z ; Fearon 2025 appl.202400192
- Pat Walters cofolding-evolution survey: https://patwalters.github.io/Cofolding-Evolution/
