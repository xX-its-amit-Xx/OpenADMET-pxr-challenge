# External Data + Leaderboard Dossier (cycle 309, 2026-06-23)

Deep-research output for the SAIR / GatorAffinity / structurally-augmented-data push + activity-track leaderboard recon.
Crons read this file. Update it as datasets are pulled / probes resolved. Honest rules at bottom.

## ★ BIGGEST FINDING — leaderboard methods we MISSED (highest priority, NOT external data)

Public top methods (RyeCatcher / De La Vega writeups; final board not yet public) all used:
- **CheMeleon** — descriptor-based molecular FOUNDATION model, D-MPNN 2048-d, ships INSIDE chemprop. arxiv 2506.15792. 79% win Polaris. EVERY public top method used it. WE HAVE NONE.
- **TabPFN (v2/v2.5)** — tabular foundation model, in-context learning, no fine-tune, EXCELLENT for small data (513 analogs). TabPFN+CheMeleon → up to 100% win on MoleculeACE. Top-3 standalone (MAE 0.528). `pip install tabpfn`. WE HAVE NONE.
- **Macau** — Bayesian multi-relational matrix factorization w/ side info (MCMC), naturally multitask. Top-3 (MAE 0.533).
- **AutoGluon-tabular** on CheMeleon embeddings (cheap stacking diversity).

**MECHANISM WE GOT WRONG (critical):** RyeCatcher made external data (ChEMBL NR1I2 ~907 cpds, NCATS qHTS PXR PubChem AID 1346982/1346985 ~10k, Tox21 SR-ARE) LIFT by injecting it as **AUXILIARY MULTITASK HEADS inside Chemprop** (5-head model T1v5), NOT as post-hoc residual features. Our cycles 297/301 found ChEMBL/Tox21 "absorbed by the chempropembed sink" — but we tested them as POST-HOC RESIDUALS on a saturated base. Multitask-head injection is a DIFFERENT, working mechanism. → RE-TEST external data as multitask heads, not residual features.

Winning RyeCatcher submission: v43 = 0.78·v31 + 0.22·T1v5, per-fold ISOTONIC calibrated (in-sample isotonic ~0.009 RAE optimistic — matches our cycle-303). CV = parent-cluster LOCO via FCFP4 NN (reproduces 513-analogs-of-~89-parents). Lesson: diversity > scale; CV champion (CheMeleon-only) was WORST on held-out.
Writeups: huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026 ; delavega.ai/posts/2026_06_18_unblinded_analysis.html
Public scores ~MAE 0.495 best (RAE≈MAE/0.84). Our honest ~0.44 internal / 0.53-0.62 blinded is COMPETITIVE or better.

## SAIR (Structurally Augmented IC50 Repository) — SandboxAQ, 2025-06
- HF: `SandboxAQ/SAIR`. License **CC BY 4.0** (commercial OK). Gated by instant auto-approve contact form -> need `huggingface-cli login`.
- 5.24M Boltz-1x co-folded structures, 1,048,857 unique protein-ligand pairs, 5 conformers each. Labels = **IC50** curated from ChEMBL35 + BindingDB.
- `sair.parquet` = 635 MB master table (START HERE: SMILES + target + IC50 + Boltz confidence). Structures ~1.6 TB (105 shards, DON'T pull all).
- PXR coverage: INHERITED from ChEMBL (CHEMBL3401 = 557 PXR IC50) + BindingDB. **Must filter parquet by UniProt O75469 / NR1I2.** NOT confirmed by direct row inspection (gated).
- ⚠ Binding/IC50 ≠ our pEC50 ACTIVATION endpoint (same gap as docking cycle-294). Value = target-aware structural prior / pretrained rep, not direct labels.
- Download: `huggingface-cli download SandboxAQ/SAIR sair.parquet --repo-type dataset --local-dir C:/pxr_work/sair`
- **AQAffinity** (SandboxAQ, 2026-01, Apache-2.0, `SandboxAQ/AQAffinity`): OpenFold3 trunk + affinity head, **STRUCTURE-FREE** (seq+SMILES->affinity), ships WEIGHTS (`model_weights_only.safetensors`) + `src/aqaffinity` + inference examples. Directly usable as inference OR embedding extractor. Degrades OOD (relevant: our test = novel analogs).
- Siblings: SAIR-FEP (~80k absolute-FEP), SAIR-OOD (OOD splits). OpenReview aiQyNhZ3s5.

## GatorAffinity — Univ. Florida Li Lab (AIDD-LiLab), 2025-09  [NOTE: "Li Lab" not "Lee Lab"]
- GitHub: `https://github.com/AIDD-LiLab/GatorAffinity`. Code MIT; **weights CC BY-NC-SA (non-commercial)**.
- Architecture: geometric DL scorer on **ATOMICA backbone** (SE(3)-equivariant, mims-harvard/ATOMICA, weights `ada-f/ATOMICA`). **Requires 3D protein-ligand COMPLEX** (protein PDB + ligand PDB + chains + SMILES), NOT ligand-only. Output = pK (pKd/pKi/pIC50). Needs a docked/co-folded pose first.
- Checkpoints in `model_checkpoints/`: `Kd+Ki+IC50_experimental_fine_tuning.ckpt` (recommended).
- Install: `git clone ...; bash environment.sh` (e3nn/SE3 stack). Infer: `python inference.py --model_ckpt ... --test_set_path X.pkl`. Custom: `data/process_pdbs.py --data_index_file your.csv` (cols: pdb_id,protein_pdb,ligand_pdb,protein_chains,lig_code,smiles,lig_resi,label).
- **gator-affinity-db** (`AIDD-LiLab/GatorAffinity-DB`, gated, data CC-BY-4.0): 4.03 TB full, 456,526 complexes (69,201 Kd + 387,325 Ki from BindingDB) + SAIR IC50. **Index `GatorAffnity_structure_index.csv`** (typo intentional, 463,867 rows) has `UniProt Primary ID`, `Target Name`, SMILES, Kd/Ki, ChEMBL ID. PULL ONLY THE INDEX (few MB), filter UniProt O75469 for PXR.
- Kd/Ki-only filter may THIN PXR coverage (PXR data is mostly EC50/IC50). ⚠ Binding≠activation again.
- **ATOMICA embeddings = our memory's flagged high-prior "sink-escape" axis we never fully tested.** GatorAffinity = ATOMICA fine-tuned for affinity -> concrete way to probe it.
- Related (same lab): Apo2Mol (flexible-pocket gen, `AIDD-LiLab/Apo2Mol_Dataset`).

## Other structurally-augmented datasets (2024-26), ranked
1. SAIR (above) — top pick, CC BY 4.0, Boltz+IC50, subset-downloadable.
2. GatorAffinity-DB (above) — ships a pretrained affinity model.
3. PLINDER (VantAI) — 449k systems, crystal+apo+AF2, BindingDB affinity, Apache/CC-BY. PDB-bound (~80 PXR ceiling).
4. Boltz-2 weights (MIT, `jwohlwend/boltz`) — the trunk producing our rich-z. Already in lineage.
5. PDBbind CleanSplit / GEMS (ETH, Zenodo 15482796) — leak-cleaned, ships trained GEMS + ESM2/Ankh/ChemBERTa embeds.
6. BindingNet v2 (hnlab, Zenodo 11218329) — 689k template-docked + ChEMBL affinity, CC BY 4.0, 1794 targets, filter for NR.
7. MISATO (Zenodo 7711953) — MD traj + QM, 17k complexes. PDBBind-bound.
8. AI3 PL Binding Affinity (AWS Open Data) — MD + MM-PBSA energy-decomposed.
9. HiQBind (figshare 27430305, CC BY-NC) ; LP-PDBBind (THGLab, split file) ; PDBBind-Opt ; DecoyDB ; Q-BioLiP.
**Coverage reality:** NO turnkey PXR/NR structural-affinity set. PDB-derived cap at ~80 PXR structures. Only new lever = filter SAIR/Gator co-folded corpora for NR1I2 (metadata filter, not 4TB).

## HONEST RULES (apply before deploying ANY of this)
1. DEPLOY METRIC = corr-with-nb3200-ERROR and honest clean-holdout RAE (nb1127 gate), NOT alignment-with-truth, NOT the contaminated 253.
2. External data as MULTITASK HEADS (RyeCatcher mechanism), re-test — our post-hoc-residual negative may not apply.
3. Binding/IC50/Kd/Ki labels ≠ pEC50 activation. Structural REPRESENTATIONS (AQAffinity/ATOMICA/SAIR-z) may still help even if labels don't.
4. Coverage is the wall: filter for test-manifold overlap (Tanimoto to 513) before training. Acceptance test cycles 300/301.
5. Subset-download (metadata/index first), route to C:/pxr_work (D: full). Confirm before any multi-100GB pull.
6. Validate on NEVER-TUNED holdouts; per-fold isotonic only (in-sample = +0.009 optimistic).
```
