#!/bin/bash
#SBATCH -p short
#SBATCH -t 08:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -a 0-15
#SBATCH -o /scratch/%u/xtb_pxr/xtb_%a.log
# GFN2-xTB QM descriptors for a 1/16 slice of the 4652-compound PXR corpus. CPU only, resumable.
module load miniconda3/25.9.1
WORK=/scratch/$USER/xtb_pxr
cd "$WORK"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
N=16; T=$SLURM_ARRAY_TASK_ID
./env/bin/python xtb_qm.py corpus.csv out_${T}.csv $N $T
echo XTB_ARRAY_${T}_DONE
