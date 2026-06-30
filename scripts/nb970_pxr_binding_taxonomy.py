"""nb970 — PHASE 1: data-driven PXR binding-mode taxonomy from 64 holo structures.

For each PXR LBD holo structure: identify the bound ligand (largest drug-like HET),
compute its residue-contact fingerprint (<=4.5 A), flag the canonical polar-anchor
contacts, then cluster structures by contact fingerprint -> empirical binding-shape
classes. Grounded in real coordinates, not memory.

Outputs (to C:, D: is full):
  C:/pxr_struct/nb970_contacts.json        per-structure ligand + contacts + anchors
  C:/pxr_struct/nb970_contact_matrix.npz   matrix (struct x residue) + labels
  C:/pxr_struct/nb970_clusters.json        cluster assignment + per-cluster consensus
  C:/pxr_struct/nb970_taxonomy.png         dendrogram + contact heatmap
"""
import os, sys, json, glob
import numpy as np
import gemmi

STRUCT_DIR = "data/external/pdb64_structures"
OUT = "C:/pxr_struct"
os.makedirs(OUT, exist_ok=True)
CONTACT_A = 4.5

# canonical PXR LBD polar-anchor residues (human PXR numbering) — from PXR literature
ANCHORS = {247: "Ser247", 285: "Gln285", 327: "His327", 407: "His407", 410: "Arg410"}
# aromatic/hydrophobic key residues to also track
KEY_HYDROPHOBIC = {288: "Phe288", 299: "Trp299", 306: "Tyr306", 281: "Phe281",
                   209: "Leu209", 243: "Met243", 246: "Met246", 411: "Leu411",
                   420: "Phe420", 425: "Met425", 284: "Cys284", 408: "Phe408"}

# HET codes that are NOT the ligand (solvent/ion/cryo/buffer/sugar)
EXCLUDE = set("""HOH DOD NA CL ZN MG K CA MN FE CU NI CD HG CO SO4 PO4 NO3 BR IOD
GOL EDO PEG PG4 PG0 1PE 2PE P6G PGE PG5 7PE 12P MPD DMS DMF DMSO ACT FMT EOH IPA
TRS BME MES EPE BTB CIT TAR MLA OGA ACY NH4 IMD BMA NAG FUC MAN BGC GAL XYL
PLM MYR OLA OLC LDA D10 D12 DD9 HTG SO3 PER OXY ACE FLC MRD GLC SUC TLA NO2
PCA CSO CME OCS SEP TPO PTR FME MLY KCX UNX UNL UNK ID DMSO BU3 BU1 ETX SCN AZI""".split())


def is_aa(res):
    info = gemmi.find_tabulated_residue(res.name)
    return info is not None and info.is_amino_acid()


def is_na(res):
    info = gemmi.find_tabulated_residue(res.name)
    return info is not None and info.is_nucleic_acid()


def pick_ligand(model):
    """Return (chain_name, residue, heavy_atom_count) for the largest drug-like HET."""
    best = (None, None, -1)
    for chain in model:
        for res in chain:
            if is_aa(res) or is_na(res):
                continue
            if res.name in EXCLUDE:
                continue
            heavy = sum(1 for a in res if a.element.name != "H")
            if heavy >= 12 and heavy > best[2]:
                best = (chain.name, res, heavy)
    return best


def main():
    files = sorted(glob.glob(f"{STRUCT_DIR}/*.cif"))
    print(f"parsing {len(files)} PXR structures ...", flush=True)
    records = []
    for f in files:
        pdb = os.path.basename(f).replace(".cif", "").upper()
        try:
            st = gemmi.read_structure(f); st.setup_entities()
            model = st[0]
            cname, lig, heavy = pick_ligand(model)
            if lig is None:
                records.append({"pdb": pdb, "ligand": None, "note": "apo/no-druglike-HET"}); continue
            lig_pos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in lig if a.element.name != "H"])
            # protein residues in the SAME chain as the ligand (the binding monomer)
            bind_chain = None; best_d = 1e9
            for chain in model:
                aa = [r for r in chain if is_aa(r)]
                if not aa: continue
                ca = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in aa for a in r if a.name == "CA"])
                if len(ca) == 0: continue
                d = np.min(np.linalg.norm(ca[:, None, :] - lig_pos[None, :, :], axis=2))
                if d < best_d: best_d, bind_chain = d, chain
            contacts = {}
            for res in bind_chain:
                if not is_aa(res): continue
                rp = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res if a.element.name != "H"])
                if len(rp) == 0: continue
                mind = np.min(np.linalg.norm(rp[:, None, :] - lig_pos[None, :, :], axis=2))
                if mind <= CONTACT_A:
                    contacts[int(res.seqid.num)] = res.name
            anchor_hits = {ANCHORS[n]: (n in contacts) for n in ANCHORS}
            records.append({
                "pdb": pdb, "ligand": lig.name, "ligand_heavy": int(heavy),
                "n_contacts": len(contacts), "contacts": contacts,
                "anchor_contacts": anchor_hits,
                "n_anchors": int(sum(anchor_hits.values())),
            })
            print(f"  {pdb}: lig={lig.name:4s} heavy={heavy:3d} contacts={len(contacts):2d} "
                  f"anchors={sum(anchor_hits.values())}", flush=True)
        except Exception as e:
            records.append({"pdb": pdb, "ligand": None, "note": f"ERR {e}"})
            print(f"  {pdb}: ERR {str(e)[:60]}", flush=True)

    holo = [r for r in records if r.get("contacts")]
    print(f"\n{len(holo)}/{len(files)} holo with a defined ligand pocket")

    # contact matrix over union of residues contacted in >=2 structures
    from collections import Counter
    rc = Counter()
    for r in holo:
        rc.update(r["contacts"].keys())
    res_union = sorted([res for res, c in rc.items() if c >= 2])
    M = np.zeros((len(holo), len(res_union)), int)
    for i, r in enumerate(holo):
        for j, res in enumerate(res_union):
            M[i, j] = 1 if res in r["contacts"] else 0
    labels = [r["pdb"] for r in holo]
    np.savez(f"{OUT}/nb970_contact_matrix.npz", M=M, residues=np.array(res_union), labels=np.array(labels))
    json.dump(records, open(f"{OUT}/nb970_contacts.json", "w"), indent=1)

    # cluster by Jaccard on contact fingerprints
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    D = pdist(M, metric="jaccard")
    Z = linkage(D, method="average")
    for k in (3, 4, 5, 6):
        cl = fcluster(Z, k, criterion="maxclust")
        sizes = Counter(cl)
        print(f"k={k}: cluster sizes {dict(sizes)}")
    K = 5
    cl = fcluster(Z, K, criterion="maxclust")
    clusters = {}
    for c in sorted(set(cl)):
        idx = [i for i in range(len(labels)) if cl[i] == c]
        consensus = [res_union[j] for j in range(len(res_union))
                     if M[idx][:, j].mean() >= 0.6]  # residues hit by >=60% of cluster
        heavy = [holo[i]["ligand_heavy"] for i in idx]
        nanc = [holo[i]["n_anchors"] for i in idx]
        clusters[f"C{c}"] = {
            "members": [labels[i] for i in idx],
            "ligands": [holo[i]["ligand"] for i in idx],
            "n": len(idx),
            "consensus_residues": consensus,
            "median_ligand_heavy": float(np.median(heavy)),
            "median_anchors": float(np.median(nanc)),
            "mean_ncontacts": float(np.mean([holo[i]["n_contacts"] for i in idx])),
        }
    json.dump({"k": K, "clusters": clusters, "linkage_method": "average_jaccard"},
              open(f"{OUT}/nb970_clusters.json", "w"), indent=1)
    print(f"\n=== {K} empirical binding-mode clusters ===")
    for c, d in clusters.items():
        print(f"{c}: n={d['n']:2d} med_heavy={d['median_ligand_heavy']:.0f} "
              f"med_anchors={d['median_anchors']:.0f} ncontacts={d['mean_ncontacts']:.0f}")
        print(f"    ligands: {d['ligands']}")
        print(f"    consensus pocket: {d['consensus_residues']}")

    # figure: dendrogram + heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [1, 3]})
    dn = dendrogram(Z, labels=labels, orientation="left", ax=ax1, leaf_font_size=7)
    ax1.set_title("PXR holo structures — contact-fingerprint clustering (Jaccard)")
    order = dn["leaves"]
    im = ax2.imshow(M[order], aspect="auto", cmap="Greys", interpolation="nearest")
    ax2.set_xticks(range(len(res_union))); ax2.set_xticklabels(res_union, rotation=90, fontsize=6)
    ax2.set_yticks(range(len(order))); ax2.set_yticklabels([labels[i] for i in order], fontsize=6)
    for j, res in enumerate(res_union):
        if res in ANCHORS: ax2.get_xticklabels()[j].set_color("red"); ax2.get_xticklabels()[j].set_fontweight("bold")
    ax2.set_title("Ligand–residue contact matrix (red x-labels = polar anchors)")
    plt.tight_layout(); plt.savefig(f"{OUT}/nb970_taxonomy.png", dpi=130); plt.close()
    print(f"\nsaved -> {OUT}/nb970_taxonomy.png + contacts.json + clusters.json + matrix.npz")


if __name__ == "__main__":
    main()
