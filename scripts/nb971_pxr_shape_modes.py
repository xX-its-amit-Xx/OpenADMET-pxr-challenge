"""nb971 — PHASE 1b: shape + anchor + sub-region refinement of the PXR binding taxonomy.

nb970's contact clustering is size-dominated. Here each bound ligand is characterized by:
  - SHAPE of the crystallographic pose: NPR1/NPR2 (rod-disc-sphere triangle), Rg, max-extent
  - ANCHOR pattern: which of the 5 polar anchors (Ser247/Gln285/His327/His407/Arg410)
  - SUB-REGION engagement: fraction of each pocket region contacted
This yields mechanistically-meaningful binding MODES (the physical-analogy shapes) + the
NPR-triangle figure for the report. Real bound coordinates, not generated conformers.
"""
import os, json, glob
import numpy as np
import gemmi

STRUCT_DIR = "data/external/pdb64_structures"
OUT = "C:/pxr_struct"
CONTACT_A = 4.5
ANCHORS = [247, 285, 327, 407, 410]
ANCHOR_NAMES = {247: "Ser247", 285: "Gln285", 327: "His327", 407: "His407", 410: "Arg410"}
# pocket sub-regions (human PXR LBD numbering, from topology)
REGIONS = {
    "polar_floor":   [247, 285, 327, 407, 410],            # H-bond anchors
    "aromatic_wall": [281, 288, 299, 306, 408],            # Phe281/Phe288/Trp299/Tyr306/Phe408
    "hphobic_roof":  [209, 243, 246, 411, 420, 425, 429],  # Leu/Met/Phe hydrophobic
    "af2_edge":      [205, 206, 207, 208, 210, 211, 240],  # helix region toward AF-2
    "flex_insert":   [227, 236, 239, 244, 251, 323, 414],  # flexible insert / beta region
}
EXCLUDE = set("""HOH DOD NA CL ZN MG K CA MN FE CU NI CD HG CO SO4 PO4 NO3 BR IOD GOL EDO PEG PG4 PG0
1PE 2PE P6G PGE PG5 7PE 12P MPD DMS DMF ACT FMT EOH IPA TRS BME MES EPE BTB CIT TAR MLA OGA ACY NH4
IMD BMA NAG FUC MAN BGC GAL XYL PLM MYR OLA OLC LDA SO3 ACE FLC MRD GLC SUC TLA NO2 PCA UNX UNL UNK SCN AZI""".split())


def is_aa(res):
    info = gemmi.find_tabulated_residue(res.name); return info is not None and info.is_amino_acid()


def shape_descriptors(coords):
    """NPR1, NPR2, radius of gyration, max pairwise extent from heavy-atom coords."""
    X = coords - coords.mean(0)
    # inertia tensor (unit mass)
    I = np.zeros((3, 3))
    for x in X:
        I += np.dot(x, x) * np.eye(3) - np.outer(x, x)
    pmi = np.sort(np.linalg.eigvalsh(I))  # ascending I1<=I2<=I3
    pmi = np.maximum(pmi, 1e-6)
    npr1, npr2 = pmi[0] / pmi[2], pmi[1] / pmi[2]
    rg = float(np.sqrt(np.mean(np.sum(X ** 2, axis=1))))
    # max extent (longest atom-atom distance)
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return float(npr1), float(npr2), rg, float(d.max())


def classify_shape(npr1, npr2):
    """Physical-analogy bin on the rod-disc-sphere triangle."""
    # vertices: rod (0,1), disc (0.5,0.5), sphere (1,1)
    drod = np.hypot(npr1 - 0.0, npr2 - 1.0)
    ddisc = np.hypot(npr1 - 0.5, npr2 - 0.5)
    dsph = np.hypot(npr1 - 1.0, npr2 - 1.0)
    return ["rod", "disc", "sphere"][int(np.argmin([drod, ddisc, dsph]))]


def main():
    recs = []
    for f in sorted(glob.glob(f"{STRUCT_DIR}/*.cif")):
        pdb = os.path.basename(f).replace(".cif", "").upper()
        try:
            st = gemmi.read_structure(f); st.setup_entities(); model = st[0]
            # largest drug-like HET
            best = (None, -1)
            for ch in model:
                for r in ch:
                    if is_aa(r) or r.name in EXCLUDE: continue
                    info = gemmi.find_tabulated_residue(r.name)
                    if info and (info.is_nucleic_acid()): continue
                    h = sum(1 for a in r if a.element.name != "H")
                    if h >= 12 and h > best[1]: best = (r, h)
            lig, heavy = best
            if lig is None: continue
            lpos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in lig if a.element.name != "H"])
            npr1, npr2, rg, ext = shape_descriptors(lpos)
            shape = classify_shape(npr1, npr2)
            # binding chain = closest by CA
            bind, bd = None, 1e9
            for ch in model:
                ca = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in ch if is_aa(r) for a in r if a.name == "CA"])
                if len(ca) == 0: continue
                dd = np.linalg.norm(ca[:, None] - lpos[None], axis=2).min()
                if dd < bd: bd, bind = dd, ch
            contacts = set()
            for r in bind:
                if not is_aa(r): continue
                rp = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r if a.element.name != "H"])
                if len(rp) and np.linalg.norm(rp[:, None] - lpos[None], axis=2).min() <= CONTACT_A:
                    contacts.add(int(r.seqid.num))
            anchors = [a for a in ANCHORS if a in contacts]
            region_frac = {k: round(len(set(v) & contacts) / len(v), 2) for k, v in REGIONS.items()}
            recs.append({"pdb": pdb, "ligand": lig.name, "heavy": int(heavy),
                         "npr1": round(npr1, 3), "npr2": round(npr2, 3), "rg": round(rg, 2),
                         "extent": round(ext, 2), "shape": shape,
                         "anchors": [ANCHOR_NAMES[a] for a in anchors], "n_anchors": len(anchors),
                         "regions": region_frac})
        except Exception as e:
            print(f"  {pdb}: ERR {str(e)[:50]}")
    print(f"characterized {len(recs)} holo ligands\n")

    # shape distribution
    from collections import Counter
    print("SHAPE distribution:", dict(Counter(r["shape"] for r in recs)))
    print("\nshape x size (median heavy atoms):")
    for s in ("rod", "disc", "sphere"):
        sr = [r for r in recs if r["shape"] == s]
        if sr:
            print(f"  {s:6s}: n={len(sr):2d} med_heavy={np.median([r['heavy'] for r in sr]):.0f} "
                  f"med_extent={np.median([r['extent'] for r in sr]):.1f}A "
                  f"med_anchors={np.median([r['n_anchors'] for r in sr]):.0f}")
    print("\nmean region engagement by shape:")
    for s in ("rod", "disc", "sphere"):
        sr = [r for r in recs if r["shape"] == s]
        if sr:
            mr = {k: round(np.mean([r["regions"][k] for r in sr]), 2) for k in REGIONS}
            print(f"  {s:6s}: {mr}")

    json.dump(recs, open(f"{OUT}/nb971_shape_modes.json", "w"), indent=1)

    # NPR rod-disc-sphere triangle figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 10))
    cmap = {"rod": "#d62728", "disc": "#1f77b4", "sphere": "#2ca02c"}
    for r in recs:
        ax.scatter(r["npr1"], r["npr2"], s=12 + r["heavy"] * 3, c=cmap[r["shape"]],
                   alpha=0.6, edgecolors="k", linewidths=0.4)
        ax.annotate(f"{r['pdb']}:{r['ligand']}", (r["npr1"], r["npr2"]), fontsize=5, alpha=0.7)
    # triangle guide
    ax.plot([0, 0.5, 1, 0], [1, 0.5, 1, 1], "k--", lw=1, alpha=0.5)
    ax.annotate("ROD\n(skewer/zipper)", (0.0, 1.01), fontsize=11, color=cmap["rod"], ha="left", weight="bold")
    ax.annotate("DISC\n(blade/coin)", (0.5, 0.46), fontsize=11, color=cmap["disc"], ha="center", va="top", weight="bold")
    ax.annotate("SPHERE\n(blob/claw)", (1.0, 1.01), fontsize=11, color=cmap["sphere"], ha="right", weight="bold")
    ax.set_xlabel("NPR1 (I1/I3)"); ax.set_ylabel("NPR2 (I2/I3)")
    ax.set_title("PXR bound-ligand SHAPE space (64 holo poses) — marker size ∝ heavy atoms")
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(0.4, 1.1)
    plt.tight_layout(); plt.savefig(f"{OUT}/nb971_shape_triangle.png", dpi=140); plt.close()
    print(f"\nsaved -> {OUT}/nb971_shape_modes.json + nb971_shape_triangle.png")


if __name__ == "__main__":
    main()
