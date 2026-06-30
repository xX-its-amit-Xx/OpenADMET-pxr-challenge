#!/bin/bash
#SBATCH -p short
#SBATCH -t 08:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -a 0-15
#SBATCH -o /scratch/%u/xtb_pxr/solv_%a.log
# GFN2-xTB ALPB-water solvation descriptors for a 1/16 slice of the 4652-mol PXR corpus.
# Reuses the xtb_pxr env + corpus from the xTB QM run. CPU only, resumable, hang-guarded.
module load miniconda3/25.9.1
WORK=/scratch/$USER/xtb_pxr
cd "$WORK"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
N=16; T=$SLURM_ARRAY_TASK_ID
./env/bin/python xtb_solv.py corpus.csv solv_out_${T}.csv $N $T
echo SOLV_ARRAY_${T}_DONE
