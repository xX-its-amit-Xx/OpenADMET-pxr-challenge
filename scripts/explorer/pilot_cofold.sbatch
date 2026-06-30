#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -o /scratch/%u/boltz_pxr/pilot.log
# Pilot: cofold ONE PXR+ligand complex with embeddings -> validate boltz + inspect the embedding format,
# and produce the PXR MSA (reused for the full 513 run).
module load miniconda3/25.9.1
cd /scratch/$USER/boltz_pxr
./env/bin/python gen_yamls.py
echo "=== running boltz on yamls/0000.yaml ==="
./env/bin/boltz predict yamls/0000.yaml --use_msa_server --write_embeddings --no_kernels \
    --out_dir pilot_out --cache ./boltz_cache --output_format pdb 2>&1 | tail -25
echo PILOT_DONE
echo "=== embedding / msa outputs ==="
find pilot_out -name '*.npz' 2>/dev/null | head
find . -name '*.a3m' 2>/dev/null | head
