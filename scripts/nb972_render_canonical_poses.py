"""nb972 — PHASE 2: render canonical PXR binder poses as openable interactive 3D HTML.

For each of the 5 binding modes, render the canonical structure(s): protein cartoon +
ligand sticks + the 5 polar-anchor residues as labeled sticks, colored by zone. Output =
standalone py3Dmol HTML (rotate in any browser) + a cropped pocket PDB (open in PyMOL/ChimeraX/Mol*).
"""
import os, glob
import gemmi
import py3Dmol

STRUCT_DIR = "data/external/pdb64_structures"
OUT = "C:/pxr_struct/poses"
os.makedirs(OUT, exist_ok=True)

ANCHORS = {247: "Ser247", 285: "Gln285", 327: "His327", 407: "His407", 410: "Arg410"}
AROMATIC = {281, 288, 299, 306, 408}
HPHOBIC = {209, 243, 246, 411, 420, 425, 429}

# canonical binder per mode: (pdb, ligand_code, mode, analogy, compound_name)
CANON = [
    ("1m13", "HYF", "A_tripod", "tripod", "SR12813 / hyperforin-class"),
    ("7axi", "EST", "A_tripod", "tripod", "17b-estradiol"),
    ("8f5y", "JQ1", "B_blade", "blade/coin", "(+)-JQ1"),
    ("2o9i", "444", "B_blade", "blade/coin", "T0901317"),
    ("8svp", "WSX", "C_skewer", "skewer/zipper", "extended agonist WSX"),
    ("5x0r", "4WH", "C_skewer", "skewer/zipper", "4WH agonist"),
    ("1skx", "RFP", "D_blob", "blob/fist(claw)", "rifampicin"),
    ("6nx1", "L7D", "D_blob", "blob/fist(claw)", "L7D (largest, 62 heavy)"),
    ("6bns", "XGH", "E_reachthrough", "key-in-side-channel", "XGH extended fragment"),
]
EXCLUDE = set("HOH NA CL ZN MG K CA SO4 PO4 GOL EDO PEG PG4 MPD DMS ACT FMT EOH IPA NAG".split())


def is_aa(res):
    info = gemmi.find_tabulated_residue(res.name); return info is not None and info.is_amino_acid()


def render(pdb, ligcode, tag, analogy, name):
    f = f"{STRUCT_DIR}/{pdb}.cif"
    st = gemmi.read_structure(f); st.setup_entities(); model = st[0]
    # find ligand + binding chain
    lig = None
    for ch in model:
        for r in ch:
            if not is_aa(r) and r.name == ligcode:
                lig = r; lig_chain = ch.name; break
        if lig: break
    if lig is None:  # fallback: largest HET
        best = (None, -1)
        for ch in model:
            for r in ch:
                if is_aa(r) or r.name in EXCLUDE: continue
                h = sum(1 for a in r if a.element.name != "H")
                if h > best[1]: best = (r, h); lig_chain = ch.name
        lig = best[0]
    import numpy as np
    lpos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in lig])
    # binding-monomer chain (closest CA)
    bind, bd = None, 1e9
    for ch in model:
        ca = np.array([[a.pos.x, a.pos.y, a.pos.z] for r in ch if is_aa(r) for a in r if a.name == "CA"])
        if len(ca) == 0: continue
        dd = np.linalg.norm(ca[:, None] - lpos[None], axis=2).min()
        if dd < bd: bd, bind = dd, ch.name

    # write a cropped pocket PDB (binding chain + ligand) for PyMOL/ChimeraX
    st2 = gemmi.Structure(); st2.cell = st.cell; st2.spacegroup_hm = st.spacegroup_hm
    m2 = gemmi.Model("1");
    for ch in model:
        if ch.name == bind:
            m2.add_chain(ch.clone())
    # ensure ligand chain present
    if not any(c.name == lig_chain for c in m2):
        for ch in model:
            if ch.name == lig_chain:
                m2.add_chain(ch.clone())
    st2.add_model(m2)
    pdb_out = f"{OUT}/{tag}__{pdb}_{ligcode}.pdb"
    st2.write_pdb(pdb_out)

    # py3Dmol interactive HTML from the cropped PDB text
    pdb_text = open(pdb_out).read()
    view = py3Dmol.view(width=900, height=700)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "lightgrey", "opacity": 0.55}})
    # ligand sticks (by resname)
    view.setStyle({"resn": ligcode}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}})
    # anchor residues = red sticks + labels
    for num, nm in ANCHORS.items():
        sel = {"resi": str(num), "chain": bind}
        view.addStyle(sel, {"stick": {"colorscheme": "redCarbon", "radius": 0.18}})
        view.addResLabels(sel, {"fontSize": 11, "backgroundColor": "white", "fontColor": "red"})
    # aromatic wall = orange, hydrophobic roof = green (thin)
    for num in AROMATIC:
        view.addStyle({"resi": str(num), "chain": bind}, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.12}})
    for num in HPHOBIC:
        view.addStyle({"resi": str(num), "chain": bind}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.10}})
    view.zoomTo({"resn": ligcode})
    view.zoom(0.85)
    html = f"""<html><head><meta charset='utf-8'><title>PXR {tag} {pdb}:{ligcode}</title></head>
<body style='font-family:sans-serif'>
<h3>PXR {analogy.upper()} mode — {name}</h3>
<p><b>{pdb.upper()}</b> · ligand <b>{ligcode}</b> (cyan) · <span style='color:red'>red = polar anchors
Ser247/Gln285/His327/His407/Arg410</span> · <span style='color:orange'>orange = aromatic wall (Phe288/Trp299/Tyr306)</span>
· <span style='color:green'>green = hydrophobic roof</span>. Drag to rotate, scroll to zoom.</p>
{view._make_html()}
</body></html>"""
    html_out = f"{OUT}/{tag}__{pdb}_{ligcode}.html"
    open(html_out, "w", encoding="utf-8").write(html)
    return os.path.basename(html_out), os.path.basename(pdb_out)


def main():
    print("rendering canonical poses ...")
    made = []
    for pdb, lig, tag, analogy, name in CANON:
        try:
            h, p = render(pdb, lig, tag, analogy, name)
            print(f"  {tag:16s} {pdb}:{lig:5s} -> {h}")
            made.append((tag, h, p))
        except Exception as e:
            print(f"  {tag:16s} {pdb}:{lig:5s} ERR {str(e)[:70]}")
    # index page linking all poses
    idx = "<html><body style='font-family:sans-serif'><h2>PXR canonical binding poses — index</h2><ul>"
    for tag, h, p in made:
        idx += f"<li><b>{tag}</b>: <a href='{h}'>interactive 3D ({h})</a> · PDB crop: {p}</li>"
    idx += "</ul></body></html>"
    open(f"{OUT}/index.html", "w", encoding="utf-8").write(idx)
    print(f"\n{len(made)} poses -> {OUT}/  (open index.html)")


if __name__ == "__main__":
    main()
