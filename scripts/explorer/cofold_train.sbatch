#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -a 0-5
#SBATCH -o /scratch/%u/boltz_pxr/cofold_train_%a.log
# Cofold the 4139 PXR training ligands (enables the interaction-graph head + pretrain/finetune).
# 8 tasks x ~517 ligands. Resumable: skips ligands whose embedding npz already exists.
module load miniconda3/25.9.1
cd /scratch/$USER/boltz_pxr
N=6; T=$SLURM_ARRAY_TASK_ID
mkdir -p "chunks_train/$T" out_train
rm -f chunks_train/$T/*.yaml
i=0
for y in yamls_train/*.yaml; do
  if [ $((i % N)) -eq $T ]; then
    stem=$(basename "$y" .yaml)
    if ! ls "out_train/$T/boltz_results_$stem/predictions/$stem/embeddings_$stem.npz" >/dev/null 2>&1; then
      ln -sf "$PWD/$y" "chunks_train/$T/$(basename $y)"
    fi
  fi
  i=$((i+1))
done
echo "task $T: $(ls chunks_train/$T 2>/dev/null | wc -l) ligands to do"
./env/bin/boltz predict "chunks_train/$T" --write_embeddings --no_kernels --out_dir "out_train/$T" \
    --cache ./boltz_cache --output_format pdb --num_workers 4
echo CHUNK_TRAIN_${T}_DONE
