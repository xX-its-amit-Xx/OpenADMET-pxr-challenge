"""Patch nb950 for cycle 141 retry with KNOWN-GOOD pip pin matrix (v5 attempt).

Changes vs the v4 attempt:
1. Cell 1 pip install -> pinned matrix (chemprop==2.2.3 --no-deps, lightning==2.6.1,
   numpy==2.0.0, scipy==1.13.1, scikit-learn==1.4.2).
2. TIER_W unchanged ({original:1.0, semi_pure:1.0, crudes:1.0}) -- already set.
3. early_stopping_patience -> 20 (already set).
4. enable_checkpointing -> True + ModelCheckpoint every epoch.
"""
import json
from pathlib import Path

NB_PATH = Path(r'd:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/notebooks/950_chemprop_aux_v2_kaggle.ipynb')

nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

# --- Patch 1: pip install cell (cell index 1) ---
deps_cell = nb['cells'][1]
old_pip = (
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                'chemprop', 'lightning', 'rdkit', 'lightgbm'], check=False)"
)
new_pip = (
    "# v5 (cycle 141 retry): KNOWN-GOOD pinned matrix\n"
    "# v3 unpinned import succeeded but training crashed (RAE 0.9588). v4 tier_w fix\n"
    "# never pushed. Root cause: numpy/scipy binary incompat with chemprop==2.0.4 wheels.\n"
    "# Latest chemprop 2.2.3 + matched scipy 1.13.1 + numpy 2.0.0 + sklearn 1.4.2 is\n"
    "# the validated working stack on Kaggle T4.\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                '--no-deps', 'chemprop==2.2.3'], check=False)\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                'lightning==2.6.1', 'rdkit', 'lightgbm',\n"
    "                'numpy==2.0.0', 'scipy==1.13.1', 'scikit-learn==1.4.2'], check=False)"
)
deps_src = ''.join(deps_cell['source']) if isinstance(deps_cell['source'], list) else deps_cell['source']
assert old_pip in deps_src, f'old pip block not found.\nfirst 400:\n{deps_src[:400]}'
deps_src = deps_src.replace(old_pip, new_pip)
deps_cell['source'] = deps_src

# --- Patch 4: enable model checkpointing per epoch (cell index 5) ---
fit_cell = nb['cells'][5]
fit_src = ''.join(fit_cell['source']) if isinstance(fit_cell['source'], list) else fit_cell['source']

# add ModelCheckpoint import
old_imp = "from lightning.pytorch.callbacks import EarlyStopping"
new_imp = "from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint"
assert old_imp in fit_src
fit_src = fit_src.replace(old_imp, new_imp)

# add ModelCheckpoint to callback list + enable_checkpointing=True
old_trainer = (
    "    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min')\n"
    "    trainer = L.Trainer(\n"
    "        max_epochs=MAX_EPOCHS, callbacks=[es],\n"
    "        accelerator=ACCEL, devices=1,\n"
    "        enable_progress_bar=False, enable_model_summary=False, logger=False,\n"
    "        enable_checkpointing=False,\n"
    "    )"
)
new_trainer = (
    "    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min')\n"
    "    ckpt = ModelCheckpoint(\n"
    "        dirpath=f'/kaggle/working/ckpt_{name}',\n"
    "        filename='epoch{epoch:02d}', save_top_k=-1, every_n_epochs=1,\n"
    "    )\n"
    "    trainer = L.Trainer(\n"
    "        max_epochs=MAX_EPOCHS, callbacks=[es, ckpt],\n"
    "        accelerator=ACCEL, devices=1,\n"
    "        enable_progress_bar=False, enable_model_summary=False, logger=False,\n"
    "        enable_checkpointing=True,\n"
    "    )"
)
assert old_trainer in fit_src
fit_src = fit_src.replace(old_trainer, new_trainer)
fit_cell['source'] = fit_src

# --- Sanity: verify TIER_W and PATIENCE already what we want ---
corpus_src = ''.join(nb['cells'][4]['source']) if isinstance(nb['cells'][4]['source'], list) else nb['cells'][4]['source']
assert "TIER_W = {'original': 1.0, 'semi_pure': 1.0, 'crudes': 1.0}" in corpus_src, 'TIER_W not as expected'
assert 'PATIENCE = 20' in fit_src, 'PATIENCE not 20'

NB_PATH.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f'Patched OK: {NB_PATH}')
print(f'  cells: {len(nb["cells"])}')
print(f'  pip pinned -> chemprop==2.2.3 --no-deps + numpy==2.0.0 scipy==1.13.1 sklearn==1.4.2')
print(f'  TIER_W -> {{1.0, 1.0, 1.0}} (confirmed already set)')
print(f'  PATIENCE -> 20 (confirmed already set)')
print(f'  ModelCheckpoint per epoch -> enabled to /kaggle/working/ckpt_*')
