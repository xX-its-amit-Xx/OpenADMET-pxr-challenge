"""Diagnostic: full training notebook with all cells wrapped in try/except."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "94_molformer_finetune.ipynb"


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


# Cell that tries to import transformers and load ChemBERTa step by step
CELL_TEST = """\
import os, sys, traceback

log = []

def chk(step, fn):
    try:
        result = fn()
        log.append(f'OK: {step}')
        return result
    except Exception as e:
        log.append(f'FAIL: {step} -> {type(e).__name__}: {e}')
        log.append(traceback.format_exc())
        return None

# Step 1: basic imports
chk("import numpy", lambda: __import__("numpy"))
chk("import pandas", lambda: __import__("pandas"))
chk("import torch", lambda: __import__("torch"))

# Step 2: check torch.cuda
def check_cuda():
    import torch
    log.append(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        log.append(f'GPU: {torch.cuda.get_device_name(0)}')
    return True
chk("torch.cuda", check_cuda)

# Step 3: import transformers top-level
chk("import transformers", lambda: __import__("transformers"))

# Step 4: import AutoTokenizer
def import_autotok():
    from transformers import AutoTokenizer
    log.append(f'AutoTokenizer: {AutoTokenizer}')
    return AutoTokenizer
AutoTokenizer = chk("AutoTokenizer", import_autotok)

# Step 5: import AutoModel
def import_automodel():
    from transformers import AutoModel
    log.append(f'AutoModel: {AutoModel}')
    return AutoModel
AutoModel = chk("AutoModel", import_automodel)

# Step 6: check dataset path
ds_base = "/kaggle/input/datasets/knowledgegraphlover/pxr-challenge-data"
chk("dataset exists", lambda: log.append(f'dataset exists: {os.path.exists(ds_base)}') or True)

# Step 7: check chemberta paths
for p in [ds_base + "/chemberta_mtr", ds_base + "/chemberta_model"]:
    is_dir = os.path.isdir(p)
    log.append(f'  {p}: is_dir={is_dir}')
    if is_dir:
        log.append(f'    contents: {sorted(os.listdir(p))[:10]}')

# Step 8: try loading model
def load_model():
    from transformers import AutoTokenizer, AutoModel
    model_dir = ds_base + "/chemberta_mtr"
    if os.path.isdir(model_dir) and os.path.exists(model_dir + "/config.json"):
        log.append(f'Loading from {model_dir}')
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        n = sum(p.numel() for p in model.parameters())
        log.append(f'Model: {n:,} params d={model.config.hidden_size}')
        return model
    else:
        log.append(f'Trying HuggingFace: DeepChem/ChemBERTa-77M-MTR')
        tok = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR", local_files_only=False)
        model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR", local_files_only=False)
        n = sum(p.numel() for p in model.parameters())
        log.append(f'HF Model: {n:,} params d={model.config.hidden_size}')
        return model
model = chk("load_model", load_model)

# Save
report = "\\n".join(log)
print(report, flush=True)
with open("/kaggle/working/diag.txt", "w") as f:
    f.write(report)
print("DONE", flush=True)
"""

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": [code_cell(CELL_TEST)],
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written {NB_PATH} — step-by-step import/load diagnostic")

code = nb["cells"][0]["source"]
try:
    compile(code, "diag_cell", "exec")
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
