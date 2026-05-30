# PXR Iteration Roadmap

## 4 Methods to Build (in order)

### Method 1: Meta-Analogy Transformer
Train transformer on (fragment FP × pocket residue type × residue position) → contact strength + binding score.
- Input: BRICS fragment Morgan FP (2048) + residue one-hot (20 AAs) + position bin (10 bins)
- Output: predicted contact distance OR per-fragment binding contribution
- Training data: 8509 atom-residue contacts from pdb64 (nb277 extraction)
- For test: decompose, predict contact patterns for each pocket residue, aggregate
- **Combinatorial layers**:
  - L1: PXR-only contacts
  - L2: + activity cliff weighting (compounds in cliff pairs get higher loss weight)
  - L3: + assay-noise correction (downweight compounds with high pec50_SE)

### Method 2: Generative De Novo + Scoring
- Use JT-VAE or REINVENT to generate PXR-like compounds
- Score each generated compound with nb239 (or even better, the meta-analogy model from Method 1)
- Take top-1000 highest-scoring as "synthetic high-affinity" augmentation
- Take bottom-1000 as "synthetic non-binder" augmentation
- Add to training as semi-supervised data with weight 0.3
- Retrain LGBM, check if augmentation helps

### Method 3: JUMP-CP Imaging VAE
- Pull JUMP-Cell-Painting morphology profiles for compounds overlapping our 4652
- Use as additional feature OR as standalone modality
- VAE: input/output = compound's image embedding
- Embedding becomes new feature

### Method 4: Custom PXR-Aware GNN
- PyTorch Geometric MPNN with PXR-pocket residue-aware node features
- Atom features: standard + "predicted distance to nearest pocket residue type"
- Edge features: bond type + distance encoding
- Output: pec50

### Method 5: Knowledge Graph + Multi-Modal GNN
- Build heterogeneous knowledge graph with:
  - PXR (NR1I2) as central node
  - Compound nodes (our train + Papyrus + PubChem + ChEMBL)
  - Other NR target nodes (FXR, PPARγ, LXRα, CAR, RXRα, VDR — share pocket biology)
  - Downstream gene nodes (CYP3A4, MDR1/ABCB1, etc. — PXR regulatory targets)
  - Disease nodes (cholestasis, drug-drug interactions, liver toxicity)
  - RNA/DNA binding element nodes (PXR response element DR3, ER6)
- Edges: binding affinity, regulatory, expression correlation, sequence homology
- Data sources to integrate:
  - PrimeKG / Hetionet (1-2GB pull, biomedical knowledge)
  - GO annotations, UniProt
  - LINCS L1000 for transcriptomics
  - GTEx for expression
- Architecture: heterogeneous GNN (R-GCN or HAN) on the KG
- For test compound: add as node, predict its PXR-binding edge weight

## Compute Budget
- 30h Kaggle GPU (used ~25h on Boltz2 attempts; ~5h remaining)
- 20h Kaggle TPU (unused)
- Local CPU: unlimited

## Ideas to Vet by User
1. **Atomistic SE(3)-equivariant networks** (e.g. e3nn, EquiBind) — proper 3D geometry
2. **Active learning loop**: submit candidate, get LB feedback, refine
3. **Tanimoto-MMP pair training**: for each cliff pair (sim≥0.7, |ΔpEC50|≥1.0), train explicit transform model
4. **Bayesian per-compound uncertainty**: GP regression on top of LGBM predictions
5. **Distillation from MolGPT-3B / ChemLLM** (if accessible via HF)
6. **Test-time fine-tune via similar Papyrus compounds for each test point** (per-compound model)

## Status (Phase 1 → Phase 2 transition)
- **Current best LB**: 239 (4-way SLSQP) = 0.7487
- **Phase 2 unblinds**: 2026-05-26 — 250 new analog labels
- **Action when Phase 2 lands**: retrain nb239 components + nb224 with new labels, re-SLSQP
