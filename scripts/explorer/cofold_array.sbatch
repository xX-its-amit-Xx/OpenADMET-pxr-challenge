#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 05:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -a 0-3
#SBATCH -o /scratch/%u/boltz_pxr/cofold_%a.log
# Cofold a 1/16 slice of the 513 PXR+ligand YAMLs with --write_embeddings.
# Assumes yamls/ already reference a precomputed PXR MSA (run gen_yamls.py <a3m> first) -> no msa server.
module load miniconda3/25.9.1
cd /scratch/$USER/boltz_pxr
N=4; T=$SLURM_ARRAY_TASK_ID
mkdir -p chunks/$T out
i=0
for y in yamls/*.yaml; do
  if [ $((i % N)) -eq $T ]; then ln -sf "$PWD/$y" "chunks/$T/$(basename $y)"; fi
  i=$((i+1))
done
echo "task $T: $(ls chunks/$T | wc -l) ligands"
./env/bin/boltz predict "chunks/$T" --write_embeddings --no_kernels --out_dir "out/$T" \
    --cache ./boltz_cache --output_format pdb --num_workers 4
echo CHUNK_${T}_DONE
