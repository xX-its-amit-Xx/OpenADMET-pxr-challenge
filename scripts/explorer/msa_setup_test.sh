#!/bin/bash
# Stage the pilot's PXR MSA for reuse across all 513 (protein is constant -> 1 MSA, no 513 server queries),
# regenerate YAMLs to reference it, and submit a 1-ligand validation (a DIFFERENT ligand, no --use_msa_server).
cd /scratch/$USER/boltz_pxr || exit 1
mkdir -p msa
cp pilot_out/boltz_results_0000/msa/0000_unpaired_tmp_env/uniref.a3m msa/pxr.a3m
module load miniconda3/25.9.1
env/bin/python gen_yamls.py "$PWD/msa/pxr.a3m"
echo "yaml 0001 msa line: $(grep -m1 msa yamls/0001.yaml)"
cat > msa_test.sbatch <<'SB'
#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -o /scratch/%u/boltz_pxr/msa_test.log
module load miniconda3/25.9.1
cd /scratch/$USER/boltz_pxr
./env/bin/boltz predict yamls/0001.yaml --write_embeddings --no_kernels \
    --out_dir msa_test_out --cache ./boltz_cache --output_format pdb 2>&1 | tail -15
echo MSA_TEST_DONE
find msa_test_out -name "embeddings_*.npz"
SB
sbatch msa_test.sbatch
