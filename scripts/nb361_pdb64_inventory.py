"""Inventory 64 PXR holo templates + Tanimoto match vs 184 structure-track test ligands.

Parses CIF files header-only (no coords), builds ligand SMILES from the embedded
_chem_comp_atom / _chem_comp_bond loops via RDKit RWMol, then computes Morgan/Tanimoto
top-1 similarity for each test ligand against all template ligands.
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
CIF_DIR = ROOT / "data/external/pdb64_structures"
TEST_CSV = ROOT / "data/raw/pxr-challenge_structure_TEST_BLINDED.csv"
OUT_DIR = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Residue codes to ignore as ligands
SKIP = {
    "HOH", "WAT", "DOD", "NA", "K", "MG", "CA", "ZN", "CL", "BR", "I", "SO4",
    "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "DMS", "ACT", "ACY", "BME",
    "MES", "TRS", "HEPES", "IPA", "MPD", "FMT", "EPE", "BOG", "LDA",
    "OLA", "OLC", "PLM", "MYR", "STE", "NAG", "MAN", "BMA", "FUC", "GAL",
    "GLC", "BGC", "XYS", "BCT", "CO3",
}

# ---- CIF parsing helpers ----------------------------------------------------

def _read_loop(text: str, prefix: str) -> list[dict]:
    """Return list of row-dicts for a CIF loop_ whose tags start with prefix."""
    out = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            tags = []
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith(prefix):
                tags.append(lines[j].strip())
                j += 1
            if tags:
                # consume rows until blank/#/loop_/non-data
                k = j
                rows = []
                while k < len(lines):
                    ln = lines[k].rstrip()
                    if not ln or ln.startswith("#") or ln.startswith("loop_") or ln.startswith("_") or ln.startswith("data_"):
                        break
                    rows.append(ln)
                    k += 1
                # tokenize (respect single quotes)
                joined = " ".join(rows)
                toks = re.findall(r"'[^']*'|\"[^\"]*\"|\S+", joined)
                ncol = len(tags)
                for idx in range(0, (len(toks) // ncol) * ncol, ncol):
                    row = {tags[c]: toks[idx + c].strip("'\"") for c in range(ncol)}
                    out.append(row)
                i = k
                continue
        i += 1
    return out


_BOND_ORDER = {
    "sing": Chem.BondType.SINGLE,
    "doub": Chem.BondType.DOUBLE,
    "trip": Chem.BondType.TRIPLE,
    "arom": Chem.BondType.AROMATIC,
}


def smiles_from_cif_chem_comp(text: str, comp_id: str) -> str | None:
    """Reconstruct SMILES for a HET comp_id from _chem_comp_atom + _chem_comp_bond."""
    atoms = _read_loop(text, "_chem_comp_atom.")
    bonds = _read_loop(text, "_chem_comp_bond.")
    a_rows = [a for a in atoms if a.get("_chem_comp_atom.comp_id") == comp_id]
    b_rows = [b for b in bonds if b.get("_chem_comp_bond.comp_id") == comp_id]
    if not a_rows:
        return None
    rw = Chem.RWMol()
    idx_of: dict[str, int] = {}
    for a in a_rows:
        sym = a.get("_chem_comp_atom.type_symbol", "C")
        if sym in ("H", "D", "T"):
            continue
        sym = sym.capitalize()
        atom = Chem.Atom(sym)
        # leaving-atom flag means it's typically a placeholder for linkage; skip
        leaving = a.get("_chem_comp_atom.pdbx_leaving_atom_flag", "N")
        if leaving == "Y":
            continue
        ai = rw.AddAtom(atom)
        idx_of[a["_chem_comp_atom.atom_id"]] = ai
    for b in b_rows:
        a1 = b.get("_chem_comp_bond.atom_id_1")
        a2 = b.get("_chem_comp_bond.atom_id_2")
        order = b.get("_chem_comp_bond.value_order", "sing").lower()
        arom = b.get("_chem_comp_bond.pdbx_aromatic_flag", "N").upper() == "Y"
        if a1 in idx_of and a2 in idx_of:
            bt = Chem.BondType.AROMATIC if arom else _BOND_ORDER.get(order, Chem.BondType.SINGLE)
            try:
                bi = rw.AddBond(idx_of[a1], idx_of[a2], bt) - 1
                if arom:
                    rw.GetBondWithIdx(bi).SetIsAromatic(True)
                    rw.GetAtomWithIdx(idx_of[a1]).SetIsAromatic(True)
                    rw.GetAtomWithIdx(idx_of[a2]).SetIsAromatic(True)
            except Exception:
                pass
    try:
        mol = rw.GetMol()
        # Use partial sanitization + aromaticity perception so rings get
        # perceived as aromatic where appropriate (CIF stores Kekule SING/DOUB).
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_RDKIT)
        return Chem.MolToSmiles(mol)
    except Exception:
        try:
            mol = rw.GetMol()
            Chem.SanitizeMol(mol, sanitizeOps=(
                Chem.SANITIZE_FINDRADICALS | Chem.SANITIZE_SETHYBRIDIZATION |
                Chem.SANITIZE_SETCONJUGATION | Chem.SANITIZE_SYMMRINGS))
            Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_RDKIT)
            return Chem.MolToSmiles(mol)
        except Exception:
            try:
                return Chem.MolToSmiles(rw.GetMol(), canonical=True)
            except Exception:
                return None


def parse_template(cif_path: Path) -> list[dict]:
    txt = cif_path.read_text(encoding="utf-8", errors="ignore")
    pdb_id = cif_path.stem.upper()

    # Resolution
    res = None
    m = re.search(r"_refine\.ls_d_res_high\s+([\d.?]+)", txt)
    if m and m.group(1) not in ("?", "."):
        try:
            res = float(m.group(1))
        except Exception:
            pass
    if res is None:
        m = re.search(r"_reflns\.d_resolution_high\s+([\d.?]+)", txt)
        if m and m.group(1) not in ("?", "."):
            try:
                res = float(m.group(1))
            except Exception:
                pass

    # HET codes via _pdbx_nonpoly_scheme (one HET-code per row, robust)
    nps = _read_loop(txt, "_pdbx_nonpoly_scheme.")
    het_codes = []
    seen = set()
    for r in nps:
        cid = r.get("_pdbx_nonpoly_scheme.mon_id")
        if cid and cid not in SKIP and cid not in seen:
            het_codes.append(cid)
            seen.add(cid)

    # Build formula-weight + name from the messy _chem_comp loop only for lookup
    cc = _read_loop(txt, "_chem_comp.")
    cc_lookup = {}
    for row in cc:
        cid = row.get("_chem_comp.id")
        if cid:
            try:
                fw = float(row.get("_chem_comp.formula_weight", "0") or 0)
            except Exception:
                fw = 0.0
            cc_lookup[cid] = {
                "name": row.get("_chem_comp.name", "?").strip("'\""),
                "formula": row.get("_chem_comp.formula", "?").strip("'\""),
                "mw": fw,
            }

    out = []
    for cid in het_codes:
        info = cc_lookup.get(cid, {"name": "?", "formula": "?", "mw": 0.0})
        fw = info["mw"]
        # If chem_comp metadata missing, MW will be computed from RDKit
        if fw and fw < 150:
            continue
        formula = info["formula"]
        smi = smiles_from_cif_chem_comp(txt, cid)
        # Heavy atom count + MW backfill from RDKit if missing
        heavy = None
        if smi:
            m_ = Chem.MolFromSmiles(smi)
            if m_:
                heavy = m_.GetNumHeavyAtoms()
                if not fw:
                    from rdkit.Chem import Descriptors
                    fw = float(Descriptors.MolWt(m_))
        if heavy is None:
            heavy = sum(int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if el and el != "H")
        if fw and fw < 150:
            continue
        out.append({
            "pdb_id": pdb_id,
            "het_code": cid,
            "name": info["name"],
            "formula": formula,
            "mw": fw,
            "heavy_atoms": heavy,
            "smiles": smi,
            "resolution": res,
        })
    return out


# ---- run --------------------------------------------------------------------

def main():
    cif_files = sorted(CIF_DIR.glob("*.cif"))
    print(f"Parsing {len(cif_files)} CIFs ...")
    rows = []
    for p in cif_files:
        try:
            rows.extend(parse_template(p))
        except Exception as e:
            print(f"  ! {p.name}: {e}")
    inv = pd.DataFrame(rows)
    print(f"Found {len(inv)} candidate ligand entries across {inv['pdb_id'].nunique()} PDBs")

    # Filter to plausible drug-like ligands: MW 150-900, has SMILES
    inv = inv[inv["smiles"].notna() & (inv["mw"] >= 150) & (inv["mw"] <= 900)].copy()
    # Per PDB, keep the largest ligand (proxy for the actual bound ligand)
    inv = inv.sort_values(["pdb_id", "mw"], ascending=[True, False])
    inv_primary = inv.drop_duplicates("pdb_id", keep="first").reset_index(drop=True)
    print(f"Primary ligands kept: {len(inv_primary)} / 64 PDBs")
    inv_primary.to_csv(OUT_DIR / "pdb64_template_inventory.csv", index=False)

    # ---- Tanimoto: test ligands vs template ligands ------------------------
    test = pd.read_csv(TEST_CSV)
    print(f"Test ligands: {len(test)}")
    test["mol"] = test["smiles"].apply(Chem.MolFromSmiles)
    test = test[test["mol"].notna()].reset_index(drop=True)
    inv_primary["mol"] = inv_primary["smiles"].apply(Chem.MolFromSmiles)
    inv_primary = inv_primary[inv_primary["mol"].notna()].reset_index(drop=True)

    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    test_fps = [gen.GetFingerprint(m) for m in test["mol"]]
    tmpl_fps = [gen.GetFingerprint(m) for m in inv_primary["mol"]]

    # nearest template per test ligand
    matches = []
    template_hit_counts = np.zeros(len(tmpl_fps), dtype=int)
    for i, fp in enumerate(test_fps):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, tmpl_fps))
        j = int(sims.argmax())
        matches.append({
            "test_structure": test["structure"].iat[i],
            "test_smiles": test["smiles"].iat[i],
            "top1_pdb": inv_primary["pdb_id"].iat[j],
            "top1_het": inv_primary["het_code"].iat[j],
            "top1_sim": float(sims[j]),
            "top1_template_smiles": inv_primary["smiles"].iat[j],
        })
        template_hit_counts[j] += 1
    mdf = pd.DataFrame(matches)
    mdf.to_csv(OUT_DIR / "pdb64_test_top1_match.csv", index=False)

    inv_primary["test_top1_hits"] = template_hit_counts
    inv_primary.drop(columns=["mol"]).to_csv(OUT_DIR / "pdb64_template_inventory.csv", index=False)

    # ---- Summary ----------------------------------------------------------
    sims = mdf["top1_sim"].values
    print("\n=== Top-1 Tanimoto distribution (test -> nearest template) ===")
    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        print(f"  q{int(q*100):02d}: {np.quantile(sims, q):.3f}")
    print(f"  mean: {sims.mean():.3f}  | frac >=0.4: {(sims>=0.4).mean():.2%}  | frac >=0.5: {(sims>=0.5).mean():.2%}  | frac >=0.7: {(sims>=0.7).mean():.2%}")

    print("\n=== Top 10 'general purpose' templates (most test ligands matched) ===")
    top = inv_primary.sort_values("test_top1_hits", ascending=False).head(10)
    for _, r in top.iterrows():
        print(f"  {r['pdb_id']} {r['het_code']:>4s}  hits={int(r['test_top1_hits']):3d}  mw={r['mw']:6.1f}  res={r['resolution']}  {r['name'][:50]}")

    print(f"\nWrote {OUT_DIR/'pdb64_template_inventory.csv'} and {OUT_DIR/'pdb64_test_top1_match.csv'}")


if __name__ == "__main__":
    main()
