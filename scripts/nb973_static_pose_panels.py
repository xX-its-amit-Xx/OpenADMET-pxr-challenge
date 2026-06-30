"""nb973 — static multi-panel 3D render of the 5 canonical PXR poses (inline-viewable).
Ligand sticks (cyan) + engaged polar-anchor residues (red sticks, labeled) from real coords.
Complements the interactive HTML (nb972). Crude matplotlib 3D but shows pose + anchor engagement.
"""
import os
import numpy as np
import gemmi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

STRUCT_DIR = "data/external/pdb64_structures"
OUT = "C:/pxr_struct"
ANCHORS = {247: "S247", 285: "Q285", 327: "H327", 407: "H407", 410: "R410"}
PANELS = [
    ("1m13", "HYF", "A · TRIPOD", "SR12813/hyperforin-class"),
    ("8f5y", "JQ1", "B · BLADE", "(+)-JQ1 (flat)"),
    ("8svp", "WSX", "C · SKEWER", "extended agonist (rod)"),
    ("1skx", "RFP", "D · BLOB/CLAW", "rifampicin (space-filling)"),
    ("6bns", "XGH", "E · REACH-THROUGH", "XGH (side-channel)"),
]


def is_aa(res):
    info = gemmi.find_tabulated_residue(res.name); return info is not None and info.is_amino_acid()


def atoms_bonds(coords, names=None, cut=1.85):
    """Return list of bonds (i,j) for atoms within cut (covalent)."""
    d = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    bonds = [(i, j) for i in range(len(coords)) for j in range(i + 1, len(coords))
             if 0.4 < d[i, j] < cut]
    return bonds


def get_pose(pdb, ligcode):
    st = gemmi.read_structure(f"{STRUCT_DIR}/{pdb}.cif"); st.setup_entities(); model = st[0]
    lig = None; lig_chain = None
    for ch in model:
        for r in ch:
            if not is_aa(r) and r.name == ligcode:
                lig = r; lig_chain = ch.name; break
        if lig: break
    lpos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in lig if a.element.name != "H"])
    bind, bd = None, 1e9
    for ch in model:
        ca = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in ch if is_aa(r) for a in r if a.name == "CA"])
        if len(ca) == 0: continue
        dd = np.linalg.norm(ca[:, None] - lpos[None], axis=2).min()
        if dd < bd: bd, bind = dd, ch
    anchors = {}
    for r in bind:
        if is_aa(r) and int(r.seqid.num) in ANCHORS:
            rp = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r if a.element.name != "H"])
            if len(rp) and np.linalg.norm(rp[:, None] - lpos[None], axis=2).min() <= 4.5:
                anchors[int(r.seqid.num)] = rp
    return lpos, anchors


def main():
    fig = plt.figure(figsize=(22, 5))
    for k, (pdb, lig, title, sub) in enumerate(PANELS):
        ax = fig.add_subplot(1, 5, k + 1, projection="3d")
        lpos, anchors = get_pose(pdb, lig)
        # ligand
        for i, j in atoms_bonds(lpos):
            ax.plot(*zip(lpos[i], lpos[j]), c="#00b3b3", lw=2.5)
        ax.scatter(*lpos.T, c="#007a7a", s=14)
        # anchors
        for num, rp in anchors.items():
            for i, j in atoms_bonds(rp):
                ax.plot(*zip(rp[i], rp[j]), c="#d62728", lw=1.6)
            cen = rp.mean(0)
            ax.text(*cen, ANCHORS[num], color="#d62728", fontsize=8, weight="bold")
        ax.set_title(f"{title}\n{pdb.upper()}:{lig} — {sub}\n{len(anchors)} anchors engaged", fontsize=10)
        ax.set_axis_off()
        # equal aspect around ligand
        c = lpos.mean(0); r = np.abs(lpos - c).max() + 3
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.view_init(elev=18, azim=k * 40)
    plt.suptitle("PXR canonical binding poses — ligand (cyan) + engaged polar anchors (red). Real crystal coords.",
                 fontsize=13, y=1.02)
    plt.tight_layout(); plt.savefig(f"{OUT}/nb973_pose_panels.png", dpi=135, bbox_inches="tight"); plt.close()
    print(f"saved -> {OUT}/nb973_pose_panels.png")


if __name__ == "__main__":
    main()
