#!/bin/bash
#SBATCH -p sharing
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o /scratch/%u/boltz_pxr/extract.log
# Pool the 513 Boltz embeddings on a compute node (login node kills it; z arrays are 107MB each).
module load miniconda3/25.9.1
cd /scratch/$USER/boltz_pxr
env/bin/python extract_emb.py
echo EXTRACT_SBATCH_DONE
