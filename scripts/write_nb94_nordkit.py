"""nb94 — ChemBERTa fine-tuning, NO rdkit dependency (stratified k-fold)."""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "94_molformer_finetune.ipynb"


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


CELL_1 = """\
import os, sys, warnings, copy
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

DS_BASE = "/kaggle/input/datasets/knowledgegraphlover/pxr-challenge-data"
WORK    = Path("/kaggle/working")
WORK.mkdir(exist_ok=True)
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# P100 (cc=6.0) is incompatible with modern PyTorch CUDA kernels (sm_60 dropped).
# Detect compute capability; fall back to CPU if Pascal or older.
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    cc_str = f"{major}.{minor}"
    if major >= 7:
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} (cc={cc_str}) → using CUDA", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
    else:
        device = torch.device("cpu")
        print(f"GPU cc={cc_str} < 7.0 (Pascal) — falling back to CPU for kernel compatibility", flush=True)
else:
    device = torch.device("cpu")
    print("No GPU detected — using CPU", flush=True)
print(f"device={device}", flush=True)

import transformers
print(f"transformers={transformers.__version__}", flush=True)
from transformers import AutoTokenizer, AutoModel
print("CELL_1 OK", flush=True)
Path("/kaggle/working/chk_1.txt").write_text(f"device={device} transformers={transformers.__version__}\\nCELL_1 OK")
"""

CELL_2 = """\
# No rdkit needed — stratified k-fold by pEC50 quintile
def rae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum(np.abs(y_true - y_true.mean()))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else 0.0

def stratified_kfold(y, n_splits=5, seed=42):
    \"\"\"Stratified k-fold by sorted value (ensures similar distribution per fold).\"\"\"
    rng = np.random.default_rng(seed)
    n = len(y)
    sorted_idx = np.argsort(y)
    folds = [[] for _ in range(n_splits)]
    for i, idx in enumerate(sorted_idx):
        folds[i % n_splits].append(idx)
    for fold in folds:
        rng.shuffle(fold)
    all_idx = np.arange(n)
    return [(np.setdiff1d(all_idx, fold), np.array(fold)) for fold in folds]

import sys as _sys
_pxr_d = _sys.modules.get('pxr.data')
if _pxr_d is not None:
    print("Loading via pxr.data shim (HuggingFace download if needed)...", flush=True)
    tr = _pxr_d.load_train()
    te = _pxr_d.load_test()
else:
    raw_dir = DS_BASE + "/rawdata"
    print(f"Loading from {raw_dir}...", flush=True)
    tr = pd.read_csv(raw_dir + "/pxr-challenge_TRAIN.csv")
    te = pd.read_csv(raw_dir + "/pxr-challenge_TEST_BLINDED.csv")
    tr = tr.rename(columns={"Molecule Name": "name", "SMILES": "smiles", "pEC50": "pec50"})
    te = te.rename(columns={"Molecule Name": "name", "SMILES": "smiles"})
y_tr = tr["pec50"].values.astype(np.float64)
splits = stratified_kfold(y_tr, 5, SEED)
print(f"Train={len(tr)} Test={len(te)}", flush=True)
print(f"Fold sizes: {[len(v) for _,v in splits]}", flush=True)
print("CELL_2 OK", flush=True)
Path("/kaggle/working/chk_2.txt").write_text(f"Train={len(tr)} Test={len(te)}\\nCELL_2 OK")
"""

CELL_3 = """\
BERT_AVAIL = False
MODEL_NAME = "chemberta_mtr"
D_MODEL = 384

_LOCAL = DS_BASE + "/chemberta_mtr"
_HF    = "DeepChem/ChemBERTa-77M-MTR"
MODEL_PATH = _LOCAL if (os.path.isdir(_LOCAL) and os.path.exists(_LOCAL + "/config.json")) else _HF
_local = MODEL_PATH.startswith("/")
print(f"Loading from: {MODEL_PATH}", flush=True)

try:
    tok  = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=_local)
    base = AutoModel.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True,
                                     local_files_only=_local)
    D_MODEL  = base.config.hidden_size
    n_params = sum(p.numel() for p in base.parameters())
    BERT_AVAIL = True
    print(f"Loaded: {n_params:,} params  D_MODEL={D_MODEL}", flush=True)
    Path("/kaggle/working/chk_3.txt").write_text(f"params={n_params} D_MODEL={D_MODEL}\\nCELL_3 OK")
except Exception as e:
    import traceback
    err = traceback.format_exc()
    print(f"FAIL: {type(e).__name__}: {e}", flush=True)
    Path("/kaggle/working/chk_3.txt").write_text(f"FAIL: {type(e).__name__}: {e}\\n{err}")
print(f"BERT_AVAIL={BERT_AVAIL}", flush=True)
"""

CELL_4 = """\
import traceback as _tb
try:
    if not BERT_AVAIL:
        raise RuntimeError("BERT_AVAIL=False — skipping")

    # Freeze all, unfreeze last 2 encoder layers
    for param in base.parameters(): param.requires_grad = False
    encoder_layers = None
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        encoder_layers = base.encoder.layer
    elif hasattr(base, "transformer") and hasattr(base.transformer, "layer"):
        encoder_layers = base.transformer.layer
    if encoder_layers is not None:
        for layer in list(encoder_layers)[-2:]:
            for param in layer.parameters(): param.requires_grad = True
    n_tr_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print(f"Trainable params: {n_tr_params:,}", flush=True)
    Path("/kaggle/working/chk_4a.txt").write_text(f"trainable={n_tr_params} encoder_layers={encoder_layers is not None}")

    # Smoke test: check encoder output attributes
    _tok_test = tok(["CCO"], return_tensors="pt", padding=True, truncation=True, max_length=32)
    _enc_test = {k: v for k, v in _tok_test.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        _out_test = base(**_enc_test)
    _attrs = [a for a in dir(_out_test) if not a.startswith("_")]
    print(f"Encoder output attrs: {_attrs}", flush=True)
    _has_lhs = hasattr(_out_test, "last_hidden_state") and _out_test.last_hidden_state is not None
    print(f"last_hidden_state available: {_has_lhs}  shape: {_out_test.last_hidden_state.shape if _has_lhs else 'N/A'}", flush=True)
    Path("/kaggle/working/chk_4b.txt").write_text(f"has_lhs={_has_lhs} attrs={_attrs[:5]}")

    class RoBERTaRegressor(nn.Module):
        def __init__(self, encoder, d_model=D_MODEL, dropout=0.3):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout/2),
                nn.Linear(64, 1),
            )
        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.head(cls).squeeze(-1)

    def tokenize_batch(smiles_list, max_len=64):
        enc = tok(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    EPOCHS = 20; BATCH = 64; LR = 3e-5; PATIENCE = 5
    oof_bert = np.full(len(y_tr), np.nan)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\\n=== Fold {fold+1}/5 ===", flush=True)
        base_f  = copy.deepcopy(base)
        model_f = RoBERTaRegressor(base_f, D_MODEL).to(device)
        opt     = torch.optim.AdamW([p for p in model_f.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.L1Loss()
        tr_smi  = tr.iloc[tr_idx]["smiles"].tolist()
        va_smi  = tr.iloc[va_idx]["smiles"].tolist()
        tr_y_f  = torch.tensor(y_tr[tr_idx], dtype=torch.float32)
        best_val = float("inf"); patience_cnt = 0
        best_preds = np.full(len(va_idx), np.nan)

        for epoch in range(EPOCHS):
            model_f.train()
            perm = torch.randperm(len(tr_smi))
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
        Path(f"/kaggle/working/chk_4_fold{fold+1}.txt").write_text(f"fold={fold+1} rae={best_val:.4f} OK")
        del model_f, base_f; torch.cuda.empty_cache()

    oof_rae = rae(y_tr, oof_bert)
    print(f"OOF RAE: {oof_rae:.4f}", flush=True)
    Path("/kaggle/working/chk_4.txt").write_text(f"oof_rae={oof_rae:.4f}\\nCELL_4 OK")

except Exception as _e:
    _err = _tb.format_exc()
    print(f"CELL_4 FAILED: {type(_e).__name__}: {_e}", flush=True)
    print(_err, flush=True)
    Path("/kaggle/working/chk_4.txt").write_text(f"FAILED: {type(_e).__name__}: {_e}\\n{_err[:2000]}")
    oof_bert = np.full(len(y_tr), y_tr.mean())
    print("Fell back to mean predictor", flush=True)
"""

CELL_5 = """\
if BERT_AVAIL and np.isfinite(oof_bert).all() and oof_bert.std() > 0.05:
    base_final  = copy.deepcopy(base)
    model_final = RoBERTaRegressor(base_final, D_MODEL).to(device)
    opt_f   = torch.optim.AdamW([p for p in model_final.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
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
    te_preds = np.clip(np.concatenate(te_preds_raw)[:513], y_tr.min()-0.5, y_tr.max()+0.5)
    del model_final, base_final; torch.cuda.empty_cache()
    print(f"Test: min={te_preds.min():.2f} max={te_preds.max():.2f}", flush=True)
    Path("/kaggle/working/chk_5.txt").write_text(f"te_min={te_preds.min():.2f} te_max={te_preds.max():.2f}\\nCELL_5 OK")
else:
    te_preds = np.full(513, y_tr.mean())
    Path("/kaggle/working/chk_5.txt").write_text("mean predictor\\nCELL_5 OK")
print(f"te_preds std={te_preds.std():.4f}", flush=True)
"""

CELL_6 = """\
oof_fname = f"oof_nb94_{MODEL_NAME}.npy"
te_fname  = f"te_nb94_{MODEL_NAME}.npy"
np.save(str(WORK / oof_fname), oof_bert)
np.save(str(WORK / te_fname),  te_preds)
sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
assert len(sub) == 513 and sub["pEC50"].notna().all()
sub.to_csv(str(WORK / f"94_{MODEL_NAME}.csv"), index=False)
final_rae = rae(y_tr, oof_bert)
print(f"OOF RAE: {final_rae:.4f}", flush=True)
print("DONE", flush=True)
Path("/kaggle/working/chk_6.txt").write_text(f"oof_rae={final_rae:.4f}\\nCELL_6 OK")
"""

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [
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

import sys
ok = True
for i, cell in enumerate(nb["cells"]):
    try:
        compile(cell["source"], f"cell_{i}", "exec")
    except SyntaxError as e:
        print(f"Cell {i}: SYNTAX ERROR: {e}"); ok = False
print("Syntax OK" if ok else "SYNTAX ERRORS")
