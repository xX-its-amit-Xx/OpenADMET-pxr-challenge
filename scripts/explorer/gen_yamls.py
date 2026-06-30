"""Generate Boltz-2 cofold YAMLs: PXR protein + each activity-test ligand. Plain csv (no pandas dep).
Run on the cluster from /scratch/$USER/boltz_pxr/. Optional arg = path to a precomputed PXR MSA (.a3m);
if given, all YAMLs reference it (avoids 513 redundant MSA-server queries since the protein is constant).
"""
import csv, os, sys

seq = "".join(l.strip() for l in open("inputs/pxr.fasta") if not l.startswith(">"))
msa = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("-", "none") else None
csvpath = sys.argv[2] if len(sys.argv) > 2 else "inputs/test513.csv"
outdir = sys.argv[3] if len(sys.argv) > 3 else "yamls"
os.makedirs(outdir, exist_ok=True)
rows = list(csv.DictReader(open(csvpath)))
prot = f"  - protein:\n      id: A\n      sequence: {seq}\n"
if msa:
    prot += f"      msa: {msa}\n"
for i, r in enumerate(rows):
    smi = r["smiles"].strip()
    with open(f"{outdir}/{i:05d}.yaml", "w") as f:
        f.write("version: 1\nsequences:\n" + prot +
                f"  - ligand:\n      id: B\n      smiles: '{smi}'\n")
print(f"wrote {len(rows)} yamls to {outdir} (msa={'precomputed' if msa else 'server'}); seq_len={len(seq)}")
