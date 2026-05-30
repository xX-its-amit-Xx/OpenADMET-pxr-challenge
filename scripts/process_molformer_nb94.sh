#!/bin/bash
# Fetch nb94 MolFormer results and check if they improve the nb211 blend.
# Run after knowledgegraphlover/pxr-challenge-nb94 shows COMPLETE.

set -e
KERNEL="knowledgegraphlover/pxr-challenge-nb94"
DL_DIR="submissions/kaggle_nb94"

echo "=== Downloading nb94 MolFormer outputs ==="
mkdir -p "$DL_DIR"
kaggle kernels output "$KERNEL" -p "$DL_DIR" 2>&1
ls -la "$DL_DIR/"*.npy 2>/dev/null || echo "No .npy files found — MolFormer might not have run"

echo ""
echo "=== Checking OOF predictions ==="
.venv/Scripts/python -c "
import numpy as np, sys
from pathlib import Path

dl = Path('$DL_DIR')
oof_candidates = list(dl.glob('oof_*.npy')) + list(dl.glob('*molformer*oof*.npy'))
te_candidates  = list(dl.glob('te_*.npy'))  + list(dl.glob('*molformer*te*.npy'))

print('OOF candidates:', [f.name for f in oof_candidates])
print('Test candidates:', [f.name for f in te_candidates])

if not oof_candidates:
    print('ERROR: No OOF file found')
    sys.exit(1)

oof = np.load(str(oof_candidates[0])).flatten()
print(f'OOF shape: {oof.shape}  std: {oof.std():.4f}  range: {oof.min():.2f} - {oof.max():.2f}')
if oof.std() < 0.01:
    print('WARNING: OOF std near zero -- only placeholder predictions')
    sys.exit(1)
print('Real predictions confirmed')
"

echo ""
echo "=== Copying to data/processed/ ==="
.venv/Scripts/python -c "
import shutil
from pathlib import Path

dl = Path('$DL_DIR')
dp = Path('data/processed')

for f in dl.glob('oof_nb94*.npy'):
    shutil.copy(f, dp / f.name)
    print(f'Copied {f.name}')
for f in dl.glob('te_nb94*.npy'):
    shutil.copy(f, dp / f.name)
    print(f'Copied {f.name}')
"

echo ""
echo "=== Computing OOF RAE and blend quality ==="
.venv/Scripts/python -c "
import numpy as np, pandas as pd
from pathlib import Path
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae
from src.pxr.paths import DATA_PROCESSED, SUBMISSIONS

tr = load_train(); te_df = load_test()
y = tr['pec50'].values

# Find nb94 OOF/test predictions
oof_nb94_files = list(DATA_PROCESSED.glob('oof_nb94*.npy'))
te_nb94_files  = list(DATA_PROCESSED.glob('te_nb94*.npy'))

if not oof_nb94_files:
    print('No nb94 OOF files found in data/processed/')
    exit(1)

oof_nb94 = np.load(str(oof_nb94_files[0])).flatten()
te_nb94  = np.load(str(te_nb94_files[0])).flatten()
oof_nb94 = np.where(np.isfinite(oof_nb94), oof_nb94, np.nanmean(oof_nb94))
te_nb94  = np.where(np.isfinite(te_nb94),  te_nb94,  np.nanmean(te_nb94))

r_nb94 = rae(y, oof_nb94)
ratio_nb94 = te_nb94.std() / oof_nb94.std()
print(f'nb94 MolFormer: OOF RAE={r_nb94:.4f}  ratio={ratio_nb94:.4f}')

# Load nb211 blend
oof211 = np.load(str(DATA_PROCESSED / 'oof_nb211_div15_chemprop_blend.npy'))
te211  = np.load(str(DATA_PROCESSED / 'te_nb211_div15_chemprop_blend.npy'))
corr = np.corrcoef(y - oof211, y - oof_nb94)[0,1]
print(f'Error correlation with nb211 blend: {corr:.4f}')
print(f'(nb93 Chemprop had corr=0.5563 with blend)')

if corr > 0.6:
    print('Correlation too high to improve blend significantly')
    exit(0)

# Try blend
THRESH = 0.58
PREV_BEST = 0.296886
best_r, best_w = 1e9, 1.0

for w in np.arange(0.90, 1.001, 0.001):
    oof_b = w * oof211 + (1-w) * oof_nb94
    te_b  = w * te211  + (1-w) * te_nb94
    oof_s = oof_b.std()
    if te_b.std() < 1e-9: continue
    te_b_inf = te_b.mean() + (te_b - te_b.mean()) * (THRESH * oof_s / te_b.std())
    ratio = te_b_inf.std() / oof_s
    r = rae(y, oof_b)
    if ratio >= THRESH - 1e-9 and r < best_r:
        best_r, best_w = r, w

print(f'Best blend nb211+MolFormer: w={best_w:.3f}  OOF RAE={best_r:.6f}  (improvement={0.296886-best_r:.6f})')

if best_r < PREV_BEST:
    w = best_w
    oof_b = w * oof211 + (1-w) * oof_nb94
    te_b  = w * te211  + (1-w) * te_nb94
    oof_s = oof_b.std()
    te_b_inf = te_b.mean() + (te_b - te_b.mean()) * (THRESH * oof_s / te_b.std())
    sub = pd.DataFrame({'Molecule Name': te_df['name'].values, 'pEC50': te_b_inf})
    np.save(str(DATA_PROCESSED / 'oof_nb212_molformer_blend.npy'), oof_b)
    np.save(str(DATA_PROCESSED / 'te_nb212_molformer_blend.npy'), te_b_inf)
    sub.to_csv(str(SUBMISSIONS / 'nb212_molformer_blend.csv'), index=False)
    print(f'*** NEW BEST! Saved nb212_molformer_blend.csv ***')
else:
    print('No improvement over nb211 (0.296886)')
"
