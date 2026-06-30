"""Generate TERNARY Boltz-2 cofold YAMLs: PXR-LBD + SRC-1 coactivator peptide + each test ligand.

Activation hypothesis (Q3b): pEC50 = transactivation = coactivator recruitment driven by ligand-induced AF-2/helix-12
stabilization, NOT mere binding. Ligand-only cofold sees binding; adding the SRC-1 LXXLL peptide makes the system
represent the activation-competent state explicitly. SRC-1 NR-box-2 peptide 'SLTERHKILHRLLQE' (LXXLL core ILHRLL),
from PDB 2O9I (PXR-SRC1-T0901317, the structure our docking box uses).

Token order = PXR(434) + peptide(15) + ligand. PXR msa = msa/pxr.a3m (precomputed); peptide msa = empty (short).
Args: [msa_path=msa/pxr.a3m] [csv=inputs/test513.csv] [outdir=yamls_coact]
"""
import csv, os, sys

seq = "".join(l.strip() for l in open("inputs/pxr.fasta") if not l.startswith(">"))
PEP = "SLTERHKILHRLLQE"   # SRC-1 NR-box-2 coactivator peptide (2O9I chains C/D)
msa = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("-", "none") else "msa/pxr.a3m"
csvpath = sys.argv[2] if len(sys.argv) > 2 else "inputs/test513.csv"
outdir = sys.argv[3] if len(sys.argv) > 3 else "yamls_coact"
os.makedirs(outdir, exist_ok=True)
rows = list(csv.DictReader(open(csvpath)))
prot = f"  - protein:\n      id: A\n      sequence: {seq}\n      msa: {msa}\n"
pep = f"  - protein:\n      id: C\n      sequence: {PEP}\n      msa: empty\n"
for i, r in enumerate(rows):
    smi = r["smiles"].strip()
    with open(f"{outdir}/{i:05d}.yaml", "w") as f:
        f.write("version: 1\nsequences:\n" + prot + pep +
                f"  - ligand:\n      id: B\n      smiles: '{smi}'\n")
print(f"wrote {len(rows)} ternary yamls to {outdir}; pxr={len(seq)} pep={len(PEP)} total_prot={len(seq)+len(PEP)}")
