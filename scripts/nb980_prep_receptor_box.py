"""nb980 — prep the PXR LBD receptor + docking box for real docking (Vina/smina in Codespace).
Windows-validatable (gemmi only). Extracts the protein from a clean holo (2O9I), strips ligand/
solvent, and computes the docking box from the pocket residues. Outputs to C: for the Codespace
docking script to consume.
"""
import os, json
import numpy as np
import gemmi

OUT = "C:/pxr_struct/dock"
os.makedirs(OUT, exist_ok=True)
REF = "2o9i"   # clean PXR + T0901317
POCKET = [209, 243, 246, 247, 281, 285, 288, 299, 306, 327, 407, 408, 410, 411, 420, 425]  # pocket-lining
EXCL = set("HOH NA CL ZN MG K CA SO4 PO4 GOL EDO PEG PG4 MPD DMS ACT FMT EOH IPA NAG".split())


def is_aa(r):
    info = gemmi.find_tabulated_residue(r.name); return info is not None and info.is_amino_acid()


def main():
    st = gemmi.read_structure(f"data/external/pdb64_structures/{REF}.cif"); st.setup_entities()
    model = st[0]
    # pick the protein chain with the most amino acids (the LBD monomer)
    bestc, bestn = None, 0
    for ch in model:
        n = sum(1 for r in ch if is_aa(r))
        if n > bestn: bestn, bestc = n, ch
    print(f"receptor chain {bestc.name} with {bestn} residues")

    # build a receptor-only structure (protein chain, no HET/solvent)
    st2 = gemmi.Structure(); st2.cell = st.cell; st2.spacegroup_hm = st.spacegroup_hm
    m2 = gemmi.Model("1"); ch2 = gemmi.Chain(bestc.name)
    pocket_coords = []
    for r in bestc:
        if is_aa(r):
            ch2.add_residue(r)
            if int(r.seqid.num) in POCKET:
                for a in r:
                    if a.element.name != "H": pocket_coords.append([a.pos.x, a.pos.y, a.pos.z])
    m2.add_chain(ch2); st2.add_model(m2)
    rec_pdb = f"{OUT}/pxr_receptor_{REF}.pdb"
    st2.write_pdb(rec_pdb)
    print(f"receptor -> {rec_pdb}")

    # docking box from pocket residues
    P = np.array(pocket_coords)
    center = P.mean(0)
    size = (P.max(0) - P.min(0)) + 8.0  # +8A margin; PXR pocket is large (~1300 A^3)
    box = {"receptor_pdb": os.path.basename(rec_pdb), "ref": REF,
           "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2]),
           "size_x": float(min(size[0], 30)), "size_y": float(min(size[1], 30)), "size_z": float(min(size[2], 30)),
           "pocket_residues": POCKET, "n_pocket_atoms": len(P)}
    json.dump(box, open(f"{OUT}/dock_box.json", "w"), indent=2)
    print(f"box center=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f}) "
          f"size=({size[0]:.1f},{size[1]:.1f},{size[2]:.1f}) -> {OUT}/dock_box.json")
    print("receptor + box ready for Codespace docking (see DOCKING_RUNBOOK.md)")


if __name__ == "__main__":
    main()
