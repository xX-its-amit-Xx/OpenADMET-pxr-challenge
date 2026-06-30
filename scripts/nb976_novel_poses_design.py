"""nb976 — PHASE 3: hypothesize 3 NOVEL PXR binding modes from the UNDER-ENGAGED pocket
regions (nb971: AF-2 edge ~0.36 and flexible insert ~0.25-0.41 are the least-used), and
design one drug-like binder for each, with mechanistic rationale + 2D + predicted mode.

These are HYPOTHESES (rationale + geometry + property fit), not docked/validated poses —
validation would need docking (ties to the structure-track tooling).
"""
import os, json
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, Descriptors3D
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

OUT = "C:/pxr_struct"

# --- 3 novel poses, each targeting an under-engaged region + a designed binder ---
DESIGNS = [
    dict(
        pose="N1 · AF-2 LEVER",
        region="AF-2 edge (res 205-211, near helix H12) — engaged by only ~36% of known ligands",
        mechanism=("A ligand that reaches and H-bonds the AF-2 edge directly props helix H12 into its "
                   "active (agonist-competent) position — the classic nuclear-receptor SUPER-AGONIST strategy. "
                   "Core anchors Ser247/His407; a rigid biaryl arm carries a terminal amide to Ser208/Leu209."),
        name="PXR-LEV1",
        # biphenyl: sulfonyl-aminopyridine anchor (one end) + primary carboxamide arm (reaches AF-2 edge)
        smiles="NC(=O)c1ccc(cc1)-c1ccc(cc1)S(=O)(=O)Nc1ccncc1",
        analogy="a lever arm reaching the H12 edge to prop it open",
    ),
    dict(
        pose="N2 · INSERT BOLT",
        region="flexible insert (res 227-251) — the LEAST-used region (~25-41%)",
        mechanism=("A polar head grips the core anchors while a lipophilic tail THREADS into the flexible "
                   "insert side-channel and locks its conformation -> potential subtype selectivity / partial "
                   "agonism. Morpholino-sulfonamide head (anchors), tert-butyl-aryl amide tail (into insert)."),
        name="PXR-BOLT1",
        smiles="CC(C)(C)c1ccc(cc1)C(=O)Nc1ccc(cc1)S(=O)(=O)N1CCOCC1",
        analogy="a bolt threaded into the side-channel and torqued home",
    ),
    dict(
        pose="N3 · TWIN-ANCHOR STAPLE",
        region="BOTH anchor clusters at once (Ser247/His407 AND His327/Arg410, ~13 A apart)",
        mechanism=("Most ligands grip ONE anchor cluster. A rigid linear bivalent presenting two H-bond "
                   "pharmacophores ~13-15 A apart can STAPLE both ends of the polar floor simultaneously -> "
                   "high affinity + a single locked pose. p-Terphenyl spacer with terminal sulfonamides."),
        name="PXR-STAPLE1",
        smiles="NS(=O)(=O)c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)S(=O)(=O)N",
        analogy="a staple gripping both ends of the polar floor",
    ),
]


def shape_class(n1, n2):
    return ["rod", "disc", "sphere"][int(np.argmin([np.hypot(n1, n2-1), np.hypot(n1-.5, n2-.5), np.hypot(n1-1, n2-1)]))]


def assign_mode(h, hba, arom, sh):
    if h <= 20: return "A_tripod"
    if h >= 40 and hba >= 4: return "D_blob"
    if sh == "sphere" and h >= 30: return "D_blob"
    if sh == "disc" and arom >= 3: return "B_blade"
    if sh == "rod" and hba <= 2: return "E_reach"
    if sh == "rod": return "C_skewer"
    return "B_blade" if arom >= 2 else "A_tripod"


def main():
    mode_med = json.load(open(f"{OUT}/nb974_phase4_summary.json"))["mode_stats"]
    mols, legends, rows = [], [], []
    for d in DESIGNS:
        m = Chem.MolFromSmiles(d["smiles"])
        assert m is not None, f"bad SMILES {d['name']}"
        h = Descriptors.HeavyAtomCount(m); hba = Descriptors.NumHAcceptors(m)
        hbd = Descriptors.NumHDonors(m); arom = Descriptors.NumAromaticRings(m)
        mw = Descriptors.MolWt(m); logp = Descriptors.MolLogP(m)
        # 3D shape + extent
        mh = Chem.AddHs(m); p = AllChem.ETKDGv3(); p.randomSeed = 42
        AllChem.EmbedMolecule(mh, p); AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
        n1, n2 = Descriptors3D.NPR1(mh), Descriptors3D.NPR2(mh)
        conf = mh.GetConformer(); xyz = conf.GetPositions()
        extent = float(np.sqrt(((xyz[:, None]-xyz[None])**2).sum(-1)).max())
        sh = shape_class(n1, n2); mode = assign_mode(h, hba, arom, sh)
        pec50 = mode_med.get(mode, {}).get("median", np.nan)
        rng = mode_med.get(mode, {})
        rows.append(dict(name=d["name"], pose=d["pose"], smiles=d["smiles"], heavy=h, mw=round(mw, 1),
                         hba=hba, hbd=hbd, arom=arom, logp=round(logp, 2), shape=sh, extent=round(extent, 1),
                         predicted_mode=mode, predicted_pec50=round(pec50, 2) if pec50 == pec50 else None,
                         pec50_range=[rng.get("q25"), rng.get("q75")],
                         region=d["region"], mechanism=d["mechanism"], analogy=d["analogy"]))
        mols.append(m)
        legends.append(f"{d['name']} | {sh} | {h} heavy | extent {extent:.1f}A | -> {mode} (pEC50~{pec50:.1f})")
        print(f"{d['name']:12s} {d['pose']:22s} heavy={h} MW={mw:.0f} HBA={hba} arom={arom} "
              f"logP={logp:.1f} shape={sh} extent={extent:.1f}A -> {mode} (pred pEC50 {pec50:.2f})")

    # 2D render grid
    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=3, subImgSize=(430, 360))
    img.save(f"{OUT}/nb976_designed_binders.png")
    json.dump(rows, open(f"{OUT}/nb976_novel_designs.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/nb976_designed_binders.png + nb976_novel_designs.json")


if __name__ == "__main__":
    main()
