#!/bin/bash
#SBATCH -p sharing
#SBATCH -t 00:40:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o /scratch/%u/boltz_pxr/torch_fix.log
# pip pulled torch built for CUDA 13; cluster driver is CUDA 12.8 -> reinstall torch with cu128 wheels.
module load miniconda3/25.9.1
cd /scratch/$USER/boltz_pxr
echo "OLD: $(./env/bin/python -c 'import torch; print(torch.__version__)')"
TVER=$(./env/bin/python -c "import torch; print(torch.__version__.split('+')[0])")
echo "pinning torch==$TVER on cu128"
./env/bin/pip install "torch==$TVER" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall \
  || ./env/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
./env/bin/python -c "import torch; print('NEW:', torch.__version__, 'cuda_build', torch.version.cuda)"
echo TORCH_FIX_DONE
