#!/bin/bash
# Set up a Boltz conda env on Explorer (/scratch) for PXR cofold + embeddings.
set -e
module load miniconda3/25.9.1
WORK=/scratch/$USER/boltz_pxr
mkdir -p "$WORK/inputs" "$WORK/out" "$WORK/msa"
cd "$WORK"
if [ ! -x ./env/bin/python ]; then
  conda create -y -p ./env python=3.11
fi
./env/bin/pip install --quiet --upgrade pip
./env/bin/pip install --quiet boltz
./env/bin/python - <<'PY'
import importlib.metadata as m
try:
    print("boltz version:", m.version("boltz"))
except Exception as e:
    print("boltz import check failed:", e)
import torch
print("torch:", torch.__version__, "cuda_build:", torch.version.cuda)
PY
echo SETUP_DONE
