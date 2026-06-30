#!/bin/bash
# Submit MACE-OFF23 strain featurization job array on Explorer
# Usage: bash submit_mace_strain.sh
# Each array task processes 100 SMILES -> ~47 tasks for 4652 mols

set -e

SCRATCH=/scratch/shenoy.am
DATA_DIR=$SCRATCH/pxr_data
WORK_DIR=$SCRATCH/pxr_work/mace_strain
ENV=$SCRATCH/xtb_pxr/env
SCRIPT=$SCRATCH/pxr_scripts/mace_strain_features.py
LOG_DIR=$WORK_DIR/logs

mkdir -p $WORK_DIR $LOG_DIR

# Copy input data if not present
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p $DATA_DIR
    echo "WARNING: $DATA_DIR is empty — copy data CSVs there first."
    echo "Expected: pxr-challenge_TRAIN.csv and pxr-challenge_TEST_BLINDED.csv"
    exit 1
fi

# Count total SMILES (approximate) to determine n_tasks
N_SMILES=$(python3 -c "
import glob, pandas as pd
files = glob.glob('$DATA_DIR/*.csv')
total = 0
for f in files:
    try:
        df = pd.read_csv(f)
        smi_col = next((c for c in df.columns if 'smiles' in c.lower()), None)
        if smi_col: total += len(df[smi_col].dropna())
    except: pass
print(total)
" 2>/dev/null || echo "5000")

CHUNK_SIZE=100
N_TASKS=$(( ($N_SMILES + $CHUNK_SIZE - 1) / $CHUNK_SIZE ))
echo "Submitting array of $N_TASKS tasks for ~$N_SMILES SMILES (chunk_size=$CHUNK_SIZE)"

sbatch --array=0-$((N_TASKS-1))%4 <<EOF
#!/bin/bash
#SBATCH --job-name=mace_strain
#SBATCH --output=$LOG_DIR/mace_strain_%A_%a.out
#SBATCH --error=$LOG_DIR/mace_strain_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=gpu_mpi_qos

module load miniconda3/24.11.1
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV

python $SCRIPT \\
    --data_dir $DATA_DIR \\
    --device cuda \\
    --model "MACE-OFF23(S)" \\
    --chunk_size $CHUNK_SIZE \\
    --chunk_id \$SLURM_ARRAY_TASK_ID

EOF

echo "Array job submitted. Monitor with: squeue -u shenoy.am"
echo "Merge when done: python $SCRATCH/pxr_scripts/merge_mace_strain.py"
