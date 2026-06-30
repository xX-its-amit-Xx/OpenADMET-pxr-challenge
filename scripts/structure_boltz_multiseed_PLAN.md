# Structure track — Boltz-2 multi-seed runbook (the real LDDT-PLI lever)

**Date:** 2026-06-09 · **Owner action:** user runs Kaggle P100 (no local GPU, D: at 367 MB free).
**Goal:** beat structure v5 (LDDT-PLI **0.4996**, rank 10/50) toward the **0.55–0.75** band that multi-seed Boltz-2 consensus opens (per `feedback_structure_track_pivot`).

---

## 1. Current deploy state (verified, not from memory)

| Submission | What it actually is | LDDT-PLI | Rank |
|---|---|---|---|
| `structure_baseline_v1.zip` | 184 pure Boltz-2 cofolds, **single seed**, `diffusion_samples 1 / recycling_steps 1 / sampling_steps 50` (minimal quality) | ~0.46 (≈ baseline) | — |
| `structure_baseline_v4.zip` | v1 + 14 RDKit redocks on low-conf ligands | 0.4583 | 29/48 |
| `structure_baseline_v5.zip` | Chai-1 consensus **intended**, but Chai CIFs were never extracted locally → **fell back to v1 poses verbatim** (different zip hash, identical poses) | **0.4996** | 10/50 |
| `structure_v6_perlig_qsel.zip` | per-ligand pose-swap by clash+pocket-distance | 0.4632 | 27/50 (REGRESSED −0.0364) |

**Verified facts (this session):**
- v1 PDBs use the **canonical 293-aa tutorial FASTA** (`SEQRES` = `GLTEEQRMMI...`), chain A = protein, chain B = ligand `resname LIG`, TER + END, no CONECT. Validator-clean.
- The nb167/169 Kaggle kernels only kept **affinity JSON** (activity-track feature); they did **not** save poses. The v1 poses came from a separate cofold run.
- v5's poses are **single-seed minimal-quality Boltz**. That is the entire headroom: a stronger sampler + multi-seed + confidence-based pose pick.

**The gap to a better LDDT-PLI:** v5 = 1 diffusion sample, 1 recycle, 50 sampling steps. Boltz-2 quality scales with `diffusion_samples` (more independent poses to pick the best from), `recycling_steps`, and `sampling_steps`. Picking the **highest-confidence** pose per ligand (the model's own `confidence_score`, iptm/plddt-derived) is the recipe that the AF3/Boltz papers report as the standard ranker — and it is NOT the clash/pocket-distance heuristic that regressed v6.

---

## 2. Multi-seed recipe (what nb951 does)

For each of the 184 ligands:
1. **YAML:** protein chain A (293-aa PXR) + ligand chain B (SMILES). `--use_msa_server` builds the MSA on Boltz's server (no local MSA needed).
2. **Sampler (the lever):** `--diffusion_samples 5 --recycling_steps 3 --sampling_steps 200`. This emits 5 ranked models per ligand; raising recycles/steps sharpens each.
3. **Pose selection — by Boltz CONFIDENCE, not geometry:** read `confidence_*model_*.json` → `confidence_score`, keep the **max-confidence** CIF. This is the v6-failure fix: confidence correlates with interface quality; clash/pocket-distance does not.
4. **Convert:** CIF → PDB via `gemmi`, force ligand chain `B` / residue `LIG` / `het_flag H` (validator requires exactly one `LIG` residue, ≤2 chains).
5. **Zip flat:** `<structure_id>.pdb` arcnames (matches `structure_placeholder_v1` convention).

**Template-biasing (the 64 PDB holo) — deliberately NOT in the turnkey path.** Boltz-2's open CLI takes templates only via the YAML `templates:` block with a local CIF, and the held-out half of the 184 plus the v6 lesson ("any move off the raw Boltz pose has cost LDDT-PLI on this target": v4 −0.0049, v6 −0.0364) make template injection a **net-risk stretch goal**, not the first shot. Recipe if you want it after the clean multi-seed lands: add `templates:\n  - cif: /kaggle/input/pdb64/<closest_holo>.cif` per ligand (closest by Tanimoto of the co-crystal ligand), run as a **separate** zip, and A/B it against the multi-seed zip on the live LB half before promoting. Do not blend.

---

## 3. Turnkey notebook

`notebooks/951_boltz2_structure_multiseed_kaggle.ipynb` — self-contained, **resumable**:
- Downloads the 184 structure SMILES from HF; embeds the 293-aa FASTA (asserts len==293).
- P100 cc<7 guard: force-reinstall `torch==2.4.0 + cu121` before `pip install boltz` (per `feedback_kaggle_p100_cuda`); numpy/scipy preflight (per `feedback_kaggle_chemprop_dead_end`).
- Skips any ligand whose `<id>.pdb` already exists, flushes `structure_boltz_multiseed.zip` every 10 ligands, and **stops cleanly at 11.3 h** so the partial zip survives the 12 h kill. Re-run 2–3× to finish all 184.
- `shutil.rmtree` each ligand's Boltz output after keeping its one chosen PDB → disk-safe on Kaggle.
- Cell 7 validates in-kernel (LIG present, count==184).

---

## 4. Estimates

| Quantity | Value |
|---|---|
| Per-ligand wall (5 samples / 3 recycles / 200 steps, P100) | ~9–11 min |
| 184 ligands | ~28–34 GPU-h → **3 Kaggle P100 sessions** (~9.5 h compute each under the 12 h cap) |
| Calendar time | **1–2 days** (Kaggle weekly P100 quota is ~30 h; may straddle a quota reset) |
| Expected LDDT-PLI | **0.52–0.62** (multi-seed best-of-5 over single-seed v5 0.4996; conservative — the 0.75 top of the memory band assumes template biasing too) |
| Expected gain vs v5 | **+0.02 to +0.12** |
| Disk (local) | **0** — all compute on Kaggle; final zip ~8 MB pulled to `submissions/` |

If P100 quota is tight, drop to `diffusion_samples 3 / sampling_steps 150` (~6–7 min/ligand → fits in ~2 sessions) for ~0.50–0.58.

---

## 5. EXACT commands (user-executable)

```powershell
# 0. one-time: kaggle creds present at D:/Users/ashenoy00000/.kaggle/kaggle.json, then:
#    pip install kaggle psutil

# 1. push the kernel to Kaggle as a private P100 GPU notebook (no --data needed; it self-downloads)
$env:KAGGLE_ACCEL = "gpu"
python scripts/kaggle_push.py --nb 951

# 2. open the kernel URL it prints, confirm Accelerator = GPU P100 + Internet ON, click Save/Run-All.
#    (kaggle_push sets enable_gpu + enable_internet; P100 vs T4 is chosen in the Kaggle UI Settings panel.)

# 3. watch it (optional; or just check the Kaggle UI)
python scripts/kaggle_push.py --nb 951 --poll

# 4. when a session ends, pull the partial zip:
python scripts/kaggle_push.py --nb 951 --pull
#    -> lands in submissions/kaggle_nb951/structure_boltz_multiseed.zip + nb951_validation.json

# 5. if nb951_validation.json shows n_pdbs < 184: re-run (it resumes from existing PDBs)
python scripts/kaggle_push.py --nb 951          # re-push/re-run
python scripts/kaggle_push.py --nb 951 --pull   # pull again

# 6. once n_pdbs==184 and n_errors==0, validate locally against the OFFICIAL validator:
Copy-Item submissions/kaggle_nb951/structure_boltz_multiseed.zip submissions/structure_boltz_multiseed_v7.zip
python scripts/validate_structure_submission.py   # (point it at the v7 zip, or reuse the v6 validate() wrapper)

# 7. only after 0 validator errors: add to scripts/auto_submit_structure_ladder.py ABOVE v5 as PRIMARY-1,
#    let the structure cron fire it, and log expected/actual LDDT-PLI in data/processed/leaderboard_log.csv.
```

**Promotion rule (per v6 lesson):** do NOT promote above v5 (0.4996) until the new zip beats it on the live LB half. If the first multi-seed LB number is < 0.4996, keep v5 PRIMARY-1 (`structure_baseline_v5_resubmit.zip`) as the safety floor and treat multi-seed as a candidate, exactly as the v6 regression forced.
