"""Boltz-2 affinity cofold YAMLs: PXR + ligand + properties.affinity (binder=ligand)."""
import csv, os, sys
seq="".join(l.strip() for l in open("inputs/pxr.fasta") if not l.startswith(">"))
msa=sys.argv[1] if len(sys.argv)>1 else "msa/pxr.a3m"
csvpath=sys.argv[2]; outdir=sys.argv[3]; os.makedirs(outdir,exist_ok=True)
rows=list(csv.DictReader(open(csvpath)))
for i,r in enumerate(rows):
    smi=r["smiles"].strip()
    with open(f"{outdir}/{i:05d}.yaml","w") as f:
        f.write("version: 1\nsequences:\n")
        f.write(f"  - protein:\n      id: A\n      sequence: {seq}\n      msa: {msa}\n")
        f.write(f"  - ligand:\n      id: B\n      smiles: '{smi}'\n")
        f.write("properties:\n  - affinity:\n      binder: B\n")
print(f"wrote {len(rows)} affinity yamls to {outdir}")
