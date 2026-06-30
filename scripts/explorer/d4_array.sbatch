#!/bin/bash
#SBATCH -p short
#SBATCH -t 06:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -a 0-15
#SBATCH -o /scratch/%u/xtb_pxr/d4_%a.log
# DFT-D4 dispersion + polarizability descriptors for a 1/16 slice of the 4652-compound PXR corpus.
# CPU only, resumable. Reuses the xtb_pxr conda env (rdkit + dftd4-python).
module load miniconda3/25.9.1
WORK=/scratch/$USER/xtb_pxr
cd "$WORK"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
N=16; T=$SLURM_ARRAY_TASK_ID
./env/bin/python d4_qm.py corpus.csv d4_out_${T}.csv $N $T
echo D4_ARRAY_${T}_DONE
