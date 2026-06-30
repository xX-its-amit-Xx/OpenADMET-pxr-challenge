#!/bin/bash
#SBATCH --job-name=crest_pxr
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=/scratch/shenoy.am/crest_pxr/logs/crest_%a.out
#SBATCH --error=/scratch/shenoy.am/crest_pxr/logs/crest_%a.err
# CREST conformational-entropy descriptors, 1/16 slice of the 4652-mol PXR corpus.
# Uses a PREFIX conda env (./env) so no conda-init/activate is needed (the bug that
# silently failed v2: `conda: command not found` in batch -> system python, no pandas).
module load miniconda3/25.9.1
WORK=/scratch/shenoy.am/crest_pxr
cd "$WORK"
export PATH="$WORK/env/bin:$PATH"        # crest binary + python from the prefix env
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$WORK/logs" "$WORK/results"

# Wait for the prefix env to finish building (login-node conda create) before running.
# Prevents the v2 failure mode of falling back to system python (no pandas).
for w in $(seq 1 60); do
  if [ -x "$WORK/env/bin/crest" ] && [ -x "$WORK/env/bin/python" ]; then break; fi
  echo "waiting for env ($((w*30))s)..."; sleep 30
done
if [ ! -x "$WORK/env/bin/crest" ] || [ ! -x "$WORK/env/bin/python" ]; then
  echo "ERROR: env not ready after 30min; aborting task $SLURM_ARRAY_TASK_ID"; exit 1
fi

N_TOTAL=4652
TASK_SIZE=$(( (N_TOTAL + 15) / 16 ))
START=$(( SLURM_ARRAY_TASK_ID * TASK_SIZE ))
END=$(( START + TASK_SIZE ))
if [ $END -gt $N_TOTAL ]; then END=$N_TOTAL; fi

echo "Task $SLURM_ARRAY_TASK_ID: mols $START-$END  (crest=$(which crest))"
"$WORK/env/bin/python" "$WORK/run_crest_slice.py" $START $END

echo "CREST_ARRAY_${SLURM_ARRAY_TASK_ID}_DONE" > "$WORK/logs/CREST_ARRAY_${SLURM_ARRAY_TASK_ID}_DONE"
