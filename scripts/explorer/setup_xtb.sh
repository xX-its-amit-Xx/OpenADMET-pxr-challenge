#!/bin/bash
# Set up a CPU conda env on Explorer (/scratch) for GFN2-xTB QM descriptors (tblite + rdkit + morfeus).
# No GPU. Run on login node (solve/download only, not heavy compute).
set -e
module load miniconda3/25.9.1
WORK=/scratch/$USER/xtb_pxr
mkdir -p "$WORK"
cd "$WORK"
if [ ! -x ./env/bin/python ]; then
  conda create -y -p ./env -c conda-forge python=3.11 rdkit tblite tblite-python numpy pandas pyarrow
fi
./env/bin/pip install --quiet morfeus-ml || echo "morfeus pip failed (optional, continue)"
./env/bin/python - <<'PY'
import numpy as np
from tblite.interface import Calculator
# tiny H2 smoke test (coords in BOHR)
nums=np.array([1,1]); pos=np.array([[0,0,0],[0,0,1.4]])
c=Calculator("GFN2-xTB",nums,pos); c.set("verbosity",0); r=c.singlepoint()
print("tblite OK energy=",float(r.get("energy")))
try:
    import morfeus; print("morfeus OK")
except Exception as e:
    print("morfeus missing:",e)
import rdkit; print("rdkit",rdkit.__version__)
PY
echo XTB_SETUP_DONE
