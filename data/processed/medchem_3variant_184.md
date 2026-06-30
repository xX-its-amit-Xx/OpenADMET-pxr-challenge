# Med-Chem 3-Variant Annotation Synthesis — 184-Ligand PXR Structure Pass

Manager review of the per-ligand light/medium/drastic pose-variant annotation. Each of 184 ligands
now has three honest tier attempts plus a recommendation; all-three-valid = 184/184.

## 1. Recommendation distribution

| rec | count | share |
|--------|-------|-------|
| light | 103 | 56.0% |
| medium | 31 | 16.8% |
| drastic | 24 | 13.0% |
| none | 26 | 14.1% |
| **total** | **184** | 100% |

158 ligands carry a recommendation OVER the model start pose (light+medium+drastic); 26 keep the
model pose unchanged. The mass on **light (56%)** is the headline: in the large majority of cases the
annotator concluded the model start pose already has the correct binding MODE and only the H-bond
*geometry* needs polishing (tighten a 2.4–2.6 A over-compressed anchor, or relax a 3.4–3.8 A loose one,
into the realistic 2.7–3.2 A band). This is the conservative, low-regret action and is consistent with
the project's hard-won GT lesson that hand-built / relocated poses lose to model-native geometry.

Confidence labels on the 158: overwhelmingly **medium** (~140), a handful **high** (~11), one **low**.
So even where a change is recommended, the annotator is rarely highly certain — appropriate for OOD
PanDDA fragments in PXR's huge promiscuous pocket.

## 2. Well-evidenced vs speculative recommendations

Two evidence types make a recommendation trustworthy here.

### Well-evidenced (trust these)

**(a) Cross-model consensus** — the only reliable confidence signal for OOD fragments (model
self-pLDDT is meaningless on memorized PXR receptors). The strongest cases pair a `light` rec with
4–5 model agreement on the same anchor after CA-superposition:
- **x00358-1, x01217-1** — all 5 models converge (apparent scatter was a coordinate-frame artifact);
  only defect is an over-tight shared anchor → light polish is a strict, near-zero-risk gain.
- **x00409-1, x00433-1, x00463-1, x00757-1, x01016-1, x01401-1, x02696-1, x00819-1** — `high`/`medium`-conf
  light recs backed by multi-model consensus on a single dominant pharmacophore (urea/amide/sulfonamide).
  These are the safest "recommended" set.
- **x00543-1, x01131-1 (sulfonamides)** — 4/5 models anchor the SO2 to the canonical
  Ser247/His407 clamp; AF3 outliers correctly discarded.

**(b) Clear pharmacophore + defect correction** — `medium`/`drastic` recs that fix a genuine chemical
error in the start pose are well-evidenced even without full consensus:
- **x00088-1** (amide donor/acceptor FLIP → medium adopts AF3's correct antiparallel pairing),
  **x00773-1** (acceptor–acceptor 2.68 A repulsion → medium re-pairs to NE2-H), **x01334-1**
  (drastic OF3 satisfies a buried unsatisfied pyridine N + bidentate Ser247/Gln285), **x00242-1**
  (drastic — start's charged carboxylate buried but UNsatisfied, consensus says Gln285 clamp),
  **x01382-1** (drastic — start leaves polar cap empty, Chai reaches Ser247+Gln285).
  These correct a buried-unsatisfied-polar or wrong-donor/acceptor-sense error — the highest-value
  edits in the set.

### Speculative (lower trust)
- **`drastic` recs resting on a single non-consensus model** — x00229-1, x00406-1, x00462-1,
  x00558-1, x01438-1, x01502-1, x02746-1, x02777-1. Each adopts one model's pose that
  reaches a "fuller" anchor network, but cross-model support is weak (the annotator self-labels these
  medium-conf and explicitly flags the fragment as orientation-ambiguous). Chemically rational, but a
  coin-flip against GT for intrinsically degenerate fragments.
- **`medium` re-anchors that move a confident, consensus-agreeing pose** — x00625-1 (low conf, gain
  "within pose noise"), x01306-1, x01126-1, x02698-1. Plausible second-anchor recruitment but trades a
  validated single anchor for an unproven two-anchor mode.

## 3. Systematic patterns by chemotype

- **Sulfonamides (primary aryl-SO2NH2):** strong, repeatable signal. The Ser247/His407 (and Gln285)
  clamp is the canonical mode and models converge on it → most are **light** with high/medium conf
  (x00543, x00757, x01131, x01415, x02696). The few `drastic`/`medium` sulfonamide recs (x00337,
  x00558, x02698, x01511) fire only when the start pose under-uses the clamp (single/no contact).
  Sulfonamides are the most confidently handled chemotype.
- **Secondary amides / lactams / ureas:** dominated by **light** — the C=O→Gln285 (±N-H→Ser247)
  two-point anchor is consensus-stable; recs almost always just fix anchor distance (x00358, x00463,
  x00509, x00644, x00652, x00714, x00794, x00913, x01009, x01105, x01108, x01401). When `medium`/
  `drastic` appears it is a flagged donor/acceptor-flip or acceptor–acceptor-clash fix (x00088, x00773)
  or a genuinely anchorless start (x01334, x01438 ureas).
- **2-aminoazoles / 2-aminobenzothiazoles (riluzole-core):** mixed. The bidentate NH2-donor/ring-N-
  acceptor hinge to Gln285(±Ser247) is the prior; `light` when start already makes it (x00046, x01233,
  x01249, x01260), `medium`/`drastic` when start makes only one anchor and another model bridges both
  (x00011, x01126). Genuinely orientation-degenerate → mostly medium conf.
- **Small rigid acids (carboxylate fragments):** lean **drastic/medium** — start poses frequently bury
  the charged group UNsatisfied (over-confident artifact), and consensus says route to the
  Ser247/Gln285 polar cap (x00242 drastic, x00261 medium, x01234 medium). The charge anchor is high-
  priority so correcting it is high-value.
- **Acceptor-only / HBD-0 fragments (nitriles, tertiary amides, click-triazoles):** pose pinned by one
  weak anchor + hydrophobic burial → high under-determination, recs split between `light` (keep the one
  good contact: x00527, x00652, x01325, x02704) and `drastic` (only when the start leaves the sole
  acceptor unsatisfied: x02746).
- **Melatonin / N-acetyltryptamine analogs:** consistent **light** — acetamide to Zone A + indole to
  the aromatic wall is a 3-model-consensus mode; only the strained NH→Ser247 distance needs relaxing
  (x00819 high-conf, x00979).

**Cross-cutting pattern:** `drastic` clusters on chemotypes with a charged or strongly-directional
handle that the start pose left buried-and-unsatisfied (acids, some sulfonamides, ureas, pyridine-N).
`light` clusters on neutral amide/lactam/sulfonamide fragments where the mode is consensus-correct and
only geometry is off. This is the right division: be bold only when there is a demonstrable chemical
defect, conservative otherwise.

## 4. Which tier to trust for a 'recommended' submission

**Build the recommended ensemble as: take the annotator's `rec` tier per ligand, but down-weight
`drastic` toward the start/light fallback unless the rationale cites a concrete defect.**

Three tiers of trust for assembling a submission:

1. **Adopt the rec as-is** for all **light** recs (103) and for the **medium/drastic** recs that cite a
   *named chemical defect* in the start pose (donor/acceptor flip, acceptor–acceptor clash, buried
   unsatisfied charge/polar, over-tight <2.5 A or loose >3.5 A anchor). These are strict improvements or
   genuine error corrections — the well-evidenced core (~120–130 ligands).
2. **Prefer the LIGHT fallback** over the recommended `medium`/`drastic` for the ~15–20 speculative
   re-anchors that move a confident, consensus-agreeing pose with only a "fuller network" justification
   and no cited defect (x00625, x01126, x01306, x02698, and the single-model `drastic` set). For these,
   the light variant (or start) is the lower-variance bet; GT can punish a wrong relocation harder than
   it rewards a marginal extra contact.
3. **Keep start** for the 26 `none` ligands.

**Net recommendation:** the safest submittable ensemble is a **"light-biased recommended" build** —
honor every `light`/defect-corrected `medium` rec, but substitute the light variant for any
`drastic`/`medium` that is a pure relocation on a non-consensus single model. This captures the
high-value anchor-geometry and defect fixes (where LDDT-PLI reliably rewards getting the contact atoms
right) while avoiding the documented failure mode of relocating an already-correct fragment pose.
A more aggressive "rec-as-annotated" build (honoring all 24 drastics) is a reasonable *second* ladder
slot to A/B against the light-biased build on the held-out holo GT before trusting it on the leaderboard.

---
*Generated 2026-06-28. Source: manager review of the 184-ligand 3-variant annotation pass.*
