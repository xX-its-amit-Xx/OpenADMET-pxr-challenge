# Med-Chem Hand-Annotation Fleet Review — PXR Under-Determined Fragments

Date: 2026-06-24. Reviewer: fleet manager (10 agents, 1 fragment each). All 10 poses saved, all report 0 clashes, all files present at `C:/tb/annot/<lig>/annotated.pdb` with a proper `LIG` residue.

## 1. Trends — anchor group → residue recurrence

| Anchor group | Ligands | Residue(s) hit |
|---|---|---|
| Amide C=O / NH | x00113, x01401, x00714 | Ser247 + Gln285 (and Ser247 alone for x00113) |
| Sulfonamide (1°/2°) | x03463, x00543, x03273 | Ser247 + His407 (+ Gln285) |
| Aniline/2-amino (aminoazole) | x00046, x00252 | Ser247 + Gln285 (bidentate) |
| Urea (bidentate) | x01438 | His407 (bidentate) |
| Phenol | x03273 | Ser247 + His407 |
| Carboxylate (anionic) | x01234 | Arg410 salt bridge (+ Ser247/His407) |

**Dominant recurring motif:** the polar head (amide / sulfonamide / amino / phenol) is anchored to the **Ser247 + Gln285 + His407** polar cluster at the pocket mouth, with the flat aromatic body buried against the **Phe288 / Trp299 / Tyr306** aromatic subpocket. This is the textbook PXR LBD recognition pattern and is **chemically sound** — these are the canonical PXR polar anchors, and the hydrophobic burial is correct for PXR's large promiscuous pocket. The one anionic fragment (x01234) correctly uses Arg410; every neutral fragment correctly leaves Arg410 unengaged (6–9 Å). That negative-control consistency is a good sign of disciplined annotation.

## 2. Quality concern — systematically over-short H-bond distances

The single most consistent QUALITY problem across the set is that reported donor–acceptor distances are **physically too short for real H-bonds**. A genuine N/O···O H-bond sits at ~2.7–3.1 Å (heavy-atom). Many poses report contacts at **2.2–2.5 Å**, which is at or below the sum of heavy-atom van der Waals contact and implies the polar atoms were dragged into each other to "win" on contact count:

- x03273 phenol: Ser247 2.35 Å, His407 2.48 Å, Gln285 2.37 Å — all three too short, simultaneously (a tridentate at 2.3–2.5 Å is geometrically implausible).
- x01234 carboxylate: His407 **2.25 Å**, Ser247 2.77 Å — 2.25 Å is essentially a clash distance for the His contact.
- x01401: Ser247 2.23 Å; x00714: Ser247 2.24 Å; x00252: Ser247 2.23 Å; x00046: 2.33/2.38 Å.

These "0 clashes / 2.2–2.5 Å H-bond" reports are internally inconsistent — the clash checker is almost certainly heavy-atom-only and not flagging the over-compression. This is a fleet-wide artifact, not one bad agent.

## 3. Per-ligand verdicts

- **x01438-1 (diaryl urea → His407 bidentate):** TRUST. Urea as central H-bond hub to His407 (donor+acceptor) is the strongest single annotation in the set; distances 2.9–3.3 Å are realistic. Keep.
- **x00113-1 (benzothiazole amide → Ser247):** TRUST. Dual Ser247/Gln285 anchor at 3.3/3.7 Å, sensible aromatic burial, Arg410 correctly idle. Keep.
- **x01234-1 (acridone-N-acetic acid → Arg410):** TRUST THE ANCHOR, REFINE THE GEOMETRY. The carboxylate→Arg410 salt bridge is the correct call (only ionizable fragment). But His407 at 2.25 Å is a clash-distance artifact, and the reasoning admits the rigid acridone can't seat deep without clashing. Relax/minimize before trusting the exact coordinates.
- **x03273-1 (phenol sulfonamide → Ser247/His407):** REDO / FALL BACK. Claims a tridentate 2.35/2.48/2.37 Å phenol+sulfonamide network — three sub-2.5 Å H-bonds at once is geometrically over-fit. Highest risk of an artificially compressed pose. Relax-and-recheck, else fall back to model pose.
- **x03463-1 (primary sulfonamide + urea, azetidine/phenyl → Ser247/His407):** CAUTION. Largest, most flexible fragment here; mixes two anchor groups (sulfonamide + urea). Sulfonamide→His407 2.9 Å is fine, but flexible multi-group fragments are exactly where hand-placement is least reliable. Verify the phenyl/azetidine burial is real, not a single rotamer guess.
- **x00046-1 (2-amino-5-Cl-benzoxazole → Gln285/Ser247):** TRUST, minor. Small rigid fragment, clean bidentate amino anchor; only the 2.33/2.38 Å distances are slightly tight. Keep.
- **x00543-1 (methyl 4-sulfamoylbenzoate → Ser247+His407 bidentate):** TRUST. Bidentate sulfonyl-oxygen network at 2.88/2.92 Å is the most realistic distance set in the whole batch. Keep — model-quality.
- **x00252-1 (2-amino-6-Cl-benzothiazole → Gln285/Ser247):** TRUST, minor. Same clean aminothiazole bidentate as x00046; Ser247 2.23 Å slightly tight. Keep.
- **x01401-1 (glycine-ester isoxazole carboxamide → Ser247/Gln285):** TRUST THE ANCHOR. Central amide to Ser247+Gln285 is right and Arg410 correctly idle. Trp299 at **2.26 Å** is an over-close aromatic contact — verify it's not a buried-atom clash the heavy-atom checker missed.
- **x00714-1 (triazole–amide–benzofuran → Gln285):** TRUST. Amide→Gln285 2.42 Å with Ser247 backup and benzofuran pi-stack into Phe288/Tyr306/Trp299; coherent. Aromatic contacts reported at 2.6–3.1 Å (pi-stack centroid-ish) are acceptable. Keep.

## 4. Single consistent refinement that would improve the whole set

**Run a short protein-restrained complex minimization (OpenMM, protein heavy atoms restrained, ligand + H-bond network free) on all 10 poses.** This directly fixes the fleet-wide 2.2–2.5 Å over-compression: it will relax over-short H-bonds back to ~2.8–3.0 Å, expose any latent clashes the heavy-atom checker missed (x01401 Trp299 2.26 Å, x01234 His407 2.25 Å), and confirm whether the hand-placed anchors are real minima or just contact-count maxima. This is the same protein-aware mini-MD already endorsed in project memory (ligand-only MMFF was GT-refuted; protein-aware complex relax is the right tool). It is one uniform pass, anchor-preserving, and converts "geometrically suspicious but plausible" into submittable coordinates.

## 5. Overall verdict — submit, with two carve-outs

The anchor *logic* is sound across all 10 (right groups, right residues, correct Arg410 negative control), so the set is worth advancing as a candidate — but NOT at the raw coordinates. Recommended action:

1. **Apply the restrained minimization (Section 4) to all 10**, then accept.
2. Even after relaxation, treat **x03273-1** (over-fit tridentate phenol) and **x03463-1** (large flexible 2-anchor fragment) as the two highest-risk poses: validate each against its original model/cofold pose on the GT harness, and **fall those two back to the model pose if the relaxed hand-pose does not hold its anchors**.
3. **x00543-1** and **x01438-1** are the most trustworthy as-is and can serve as the confidence anchors of the candidate.

Do not submit the raw 2.2–2.5 Å poses directly — the compressed H-bonds are the kind of geometric artifact that LDDT-PLI / scoring-perception checks penalize.
