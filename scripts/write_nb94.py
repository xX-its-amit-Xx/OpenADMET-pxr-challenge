"""Rewrite nb94 — self-contained ChemBERTa fine-tuning, no pxr import needed."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "94_molformer_finetune.ipynb"


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


CELL_MD = """# 94 — Kaggle GPU: ChemBERTa-77M-MTR Fine-tuning

ChemBERTa-77M-MTR: RoBERTa pre-trained on 77M SMILES from ZINC.
3.4M params, 384-dim, 6-layer. Loaded from local dataset (no internet needed).

Self-contained notebook — no pxr import.

Run on Kaggle: `python scripts/kaggle_push.py --nb 94`"""


CELL_1 = """\
import os, sys, warnings, copy
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

# Dataset base path (Kaggle mounts datasets at this location)
DS_BASE = "/kaggle/input/datasets/knowledgegraphlover/pxr-challenge-data"
WORK    = Path("/kaggle/working")
WORK.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

import transformers
print(f"transformers: {transformers.__version__}", flush=True)
from transformers import AutoTokenizer, AutoModel
print("imports OK", flush=True)
"""


CELL_2 = """\
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# --- Inline helpers (no pxr needed) ---
def rae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum(np.abs(y_true - y_true.mean()))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else 0.0

def bemis_murcko(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return str(smi)
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except: return str(smi)

def scaffold_kfold_indices(scaffolds, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    scaffold_to_idx = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        key = sc if (sc and sc == sc) else f"__singleton_{i}__"
        scaffold_to_idx[key].append(i)
    groups = list(scaffold_to_idx.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_assignments = [[] for _ in range(n_splits)]
    for group in groups:
        k = int(np.argmin(fold_sizes))
        fold_assignments[k].append(group)
        fold_sizes[k] += len(group)
    all_idx = np.arange(sum(len(g) for grp in fold_assignments for g in grp))
    splits = []
    for fold in range(n_splits):
        val_idx = np.array([i for grp in fold_assignments[fold] for i in grp])
        train_idx = np.setdiff1d(all_idx, val_idx)
        splits.append((train_idx, val_idx))
    return splits

# --- Load data directly from dataset CSV ---
raw_dir = DS_BASE + "/rawdata"
tr = pd.read_csv(raw_dir + "/pxr-challenge_TRAIN.csv")
te = pd.read_csv(raw_dir + "/pxr-challenge_TEST_BLINDED.csv")
tr = tr.rename(columns={"Molecule Name": "name", "SMILES": "smiles", "pEC50": "pec50"})
te = te.rename(columns={"Molecule Name": "name", "SMILES": "smiles"})
print(f"Train: {len(tr)}  Test: {len(te)}", flush=True)

y_tr = tr["pec50"].values.astype(np.float64)
print("Computing scaffolds...", flush=True)
scaffolds = [bemis_murcko(s) for s in tr["smiles"]]
splits = scaffold_kfold_indices(scaffolds, 5, SEED)
print(f"Scaffold 5-fold splits ready  fold sizes: {[len(v) for _,v in splits]}", flush=True)
"""


CELL_3 = """\
BERT_AVAIL = False
MODEL_NAME = "chemberta_mtr"
D_MODEL = 384

# Try local dataset path first, then HuggingFace fallback
_LOCAL = DS_BASE + "/chemberta_mtr"
_HF    = "DeepChem/ChemBERTa-77M-MTR"

if os.path.isdir(_LOCAL) and os.path.exists(_LOCAL + "/config.json"):
    MODEL_PATH = _LOCAL
    _local = True
    print(f"Model: {MODEL_PATH}", flush=True)
else:
    MODEL_PATH = _HF
    _local = False
    print(f"Fallback to HuggingFace: {MODEL_PATH}", flush=True)

try:
    tok  = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=_local)
    base = AutoModel.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True,
                                     local_files_only=_local)
    n_params = sum(p.numel() for p in base.parameters())
    D_MODEL  = base.config.hidden_size
    print(f"Loaded: {n_params:,} params  D_MODEL={D_MODEL}  layers={base.config.num_hidden_layers}", flush=True)
    BERT_AVAIL = True
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}", flush=True)
"""


CELL_4 = """\
if BERT_AVAIL:
    # Freeze all, unfreeze last 2 encoder layers
    for param in base.parameters():
        param.requires_grad = False
    encoder_layers = None
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        encoder_layers = base.encoder.layer
    elif hasattr(base, "transformer") and hasattr(base.transformer, "layer"):
        encoder_layers = base.transformer.layer
    if encoder_layers is not None:
        n_layers = len(encoder_layers)
        for layer in encoder_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"Unfroze last 2 of {n_layers} encoder layers", flush=True)

    n_trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print(f"Trainable: {n_trainable:,} / {sum(p.numel() for p in base.parameters()):,}", flush=True)

    class RoBERTaRegressor(nn.Module):
        def __init__(self, encoder, d_model=D_MODEL, dropout=0.3):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 64),  nn.GELU(), nn.Dropout(dropout / 2),
                nn.Linear(64, 1),
            )
        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.head(cls).squeeze(-1)

    def tokenize_batch(smiles_list, max_len=128):
        enc = tok(smiles_list, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        # Keep only input_ids and attention_mask (RoBERTa doesn't need token_type_ids)
        return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    EPOCHS   = 25
    BATCH    = 64
    LR       = 3e-5
    PATIENCE = 6

    oof_bert = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\\n=== Fold {fold+1}/5 ===", flush=True)
        # Deep-copy encoder so each fold starts from original pre-trained weights
        base_f   = copy.deepcopy(base)
        model_f  = RoBERTaRegressor(base_f, D_MODEL).to(device)
        trainable = [p for p in model_f.parameters() if p.requires_grad]
        opt      = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
        sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn  = nn.L1Loss()

        tr_smi  = tr.iloc[tr_idx]["smiles"].tolist()
        va_smi  = tr.iloc[va_idx]["smiles"].tolist()
        tr_y_f  = torch.tensor(y_tr[tr_idx], dtype=torch.float32)

        best_val  = float("inf")
        patience_cnt = 0
        best_preds = np.full(len(va_idx), np.nan)

        for epoch in range(EPOCHS):
            model_f.train()
            perm  = torch.randperm(len(tr_smi))
            ep_loss = 0; nb_ = 0
            for b in range(0, len(tr_smi), BATCH):
                idx_b = perm[b:b+BATCH].tolist()
                enc   = tokenize_batch([tr_smi[i] for i in idx_b])
                enc   = {k: v.to(device) for k, v in enc.items()}
                pred  = model_f(**enc)
                loss  = loss_fn(pred, tr_y_f[idx_b].to(device))
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); nb_ += 1
            sched.step()

            model_f.eval()
            with torch.no_grad():
                va_preds = []
                for b in range(0, len(va_smi), BATCH*2):
                    enc = tokenize_batch(va_smi[b:b+BATCH*2])
                    enc = {k: v.to(device) for k, v in enc.items()}
                    va_preds.append(model_f(**enc).cpu().numpy())
                va_pred = np.concatenate(va_preds)
                val_rae = rae(y_tr[va_idx], va_pred)
            print(f"  Ep {epoch+1:2d}  loss={ep_loss/nb_:.4f}  val_RAE={val_rae:.4f}", flush=True)
            if val_rae < best_val:
                best_val = val_rae; patience_cnt = 0; best_preds = va_pred.copy()
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  Early stop ep {epoch+1}", flush=True); break

        oof_bert[va_idx] = best_preds
        print(f"Fold {fold+1}: best val RAE={best_val:.4f}", flush=True)
        del model_f, base_f; torch.cuda.empty_cache()

    oof_rae = rae(y_tr, oof_bert)
    print(f"\\nChemBERTa OOF RAE: {oof_rae:.4f}", flush=True)
    print(f"OOF std={oof_bert.std():.4f}  mean={oof_bert.mean():.4f}", flush=True)
else:
    oof_bert = np.full(len(y_tr), y_tr.mean())
    print("Using mean predictor", flush=True)
"""


CELL_5 = """\
if BERT_AVAIL and np.isfinite(oof_bert).all() and oof_bert.std() > 0.05:
    print("Training final model on all data...", flush=True)
    base_final  = copy.deepcopy(base)
    model_final = RoBERTaRegressor(base_final, D_MODEL).to(device)
    trainable_f = [p for p in model_final.parameters() if p.requires_grad]
    opt_f   = torch.optim.AdamW(trainable_f, lr=LR, weight_decay=1e-4)
    sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=EPOCHS)
    all_smi = tr["smiles"].tolist()
    all_y_t = torch.tensor(y_tr, dtype=torch.float32)
    loss_fn2 = nn.L1Loss()

    for epoch in range(EPOCHS):
        model_final.train()
        perm = torch.randperm(len(all_smi))
        ep_loss = 0; nb_ = 0
        for b in range(0, len(all_smi), BATCH):
            idx_b = perm[b:b+BATCH].tolist()
            enc   = tokenize_batch([all_smi[i] for i in idx_b])
            enc   = {k: v.to(device) for k, v in enc.items()}
            pred  = model_final(**enc)
            loss  = loss_fn2(pred, all_y_t[idx_b].to(device))
            opt_f.zero_grad(); loss.backward(); opt_f.step()
            ep_loss += loss.item(); nb_ += 1
        sched_f.step()
        print(f"  Final ep {epoch+1:2d}  loss={ep_loss/nb_:.4f}", flush=True)

    model_final.eval()
    te_smi = te["smiles"].tolist()
    te_preds_raw = []
    with torch.no_grad():
        for b in range(0, len(te_smi), BATCH*2):
            enc = tokenize_batch(te_smi[b:b+BATCH*2])
            enc = {k: v.to(device) for k, v in enc.items()}
            te_preds_raw.append(model_final(**enc).cpu().numpy())
    te_preds = np.clip(np.concatenate(te_preds_raw)[:513],
                       y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"Test: min={te_preds.min():.2f}  med={np.median(te_preds):.2f}  max={te_preds.max():.2f}", flush=True)
    del model_final, base_final; torch.cuda.empty_cache()
else:
    te_preds = np.full(513, y_tr.mean())
    print("Mean predictor for test", flush=True)
"""


CELL_6 = """\
oof_fname = f"oof_nb94_{MODEL_NAME}.npy"
te_fname  = f"te_nb94_{MODEL_NAME}.npy"

np.save(str(WORK / oof_fname), oof_bert)
np.save(str(WORK / te_fname),  te_preds)

sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub) == 513 and sub["pEC50"].notna().all(), f"Bad submission: {len(sub)} rows"
sub_path = WORK / f"94_{MODEL_NAME}.csv"
sub.to_csv(str(sub_path), index=False)

print(f"Saved {oof_fname}  OOF std={oof_bert.std():.4f}", flush=True)
print(f"Saved {te_fname}   te  std={te_preds.std():.4f}", flush=True)
print(f"OOF RAE: {rae(y_tr, oof_bert):.4f}", flush=True)
print("DONE", flush=True)
"""


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [
        md_cell(CELL_MD),
        code_cell(CELL_1),
        code_cell(CELL_2),
        code_cell(CELL_3),
        code_cell(CELL_4),
        code_cell(CELL_5),
        code_cell(CELL_6),
    ],
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written {NB_PATH} — {len(nb['cells'])} cells")

# Syntax check all cells
import sys
ok = True
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        try:
            compile(cell["source"], f"cell_{i}", "exec")
            print(f"  Cell {i}: syntax OK")
        except SyntaxError as e:
            print(f"  Cell {i}: SYNTAX ERROR: {e}")
            ok = False
if ok:
    print("All cells: syntax OK")
