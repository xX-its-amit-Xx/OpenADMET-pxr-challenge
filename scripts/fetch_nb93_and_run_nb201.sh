#!/bin/bash
# Download nb93 Kaggle outputs and run nb201 immediately.
# Run after knowledgegraphlover/pxr-challenge-nb93 shows COMPLETE.

set -e
KERNEL="knowledgegraphlover/pxr-challenge-nb93"
DL_DIR="submissions/kaggle_nb93v27"

echo "=== Downloading nb93 Kaggle outputs ==="
mkdir -p "$DL_DIR"
kaggle kernels output "$KERNEL" -p "$DL_DIR"
ls -la "$DL_DIR/"

echo ""
echo "=== Verifying OOF predictions ==="
.venv/Scripts/python -c "
import numpy as np, sys
from pathlib import Path

dl = Path('$DL_DIR')
oof_p = dl / 'oof_nb93_chemprop_large_gpu.npy'
te_p  = dl / 'te_nb93_chemprop_large_gpu.npy'

if not oof_p.exists():
    print('ERROR: OOF file not found')
    sys.exit(1)

oof = np.load(str(oof_p)).flatten()
te  = np.load(str(te_p)).flatten()

print(f'OOF: shape={oof.shape}  std={oof.std():.4f}  min={oof.min():.3f}  max={oof.max():.3f}')
print(f'Test: shape={te.shape}  std={te.std():.4f}  min={te.min():.3f}  max={te.max():.3f}')

if oof.std() < 0.01:
    print('WARNING: OOF std is near zero -- placeholder predictions only')
    sys.exit(1)
else:
    print('Real predictions confirmed (std > 0.01)')
"

echo ""
echo "=== Copying to data/processed/ ==="
cp "$DL_DIR/oof_nb93_chemprop_large_gpu.npy" data/processed/
cp "$DL_DIR/te_nb93_chemprop_large_gpu.npy"  data/processed/
echo "Copied."

echo ""
echo "=== Computing quick OOF RAE ==="
.venv/Scripts/python -c "
import numpy as np
from src.pxr.data import load_train
from src.pxr.eval import rae

tr = load_train()
y_tr = tr['pec50'].values
oof = np.load('data/processed/oof_nb93_chemprop_large_gpu.npy').flatten()
te  = np.load('data/processed/te_nb93_chemprop_large_gpu.npy').flatten()
ratio = te.std() / oof.std()
print(f'nb93 Chemprop OOF RAE: {rae(y_tr, oof):.4f}  ratio={ratio:.4f}')
"

echo ""
echo "=== Running nb201 (Chemprop + external in full SLSQP) ==="
.venv/Scripts/python scripts/nb201_chemprop_slsqp.py 2>&1 | tee scripts/nb_logs/nb201.log
