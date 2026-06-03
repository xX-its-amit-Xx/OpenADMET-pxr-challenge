"""nb904 -- Self-supervised char-level transformer pretrain + fine-tune.

Hypothesis: pretraining on the 21k single-concentration SMILES (which DOES
include test-distribution chemistry) gives a representation distribution-
matched to the test set, beating 4139-only training.

Pipeline:
  1) Load 21k single-conc SMILES (labels unused).
  2) Char-level tokenize, max_len=128.
  3) Pretrain a 3-layer / d=128 / 4-head transformer encoder with 15% MLM
     for 5 epochs (CPU OK; ~200K params, batch 32).
  4) Pool (mean over tokens) -> Linear(128, 1) regression head.
  5) Fine-tune on 4139 CRC pEC50 with Huber loss + AdamW lr=1e-4 for 30 epochs.
  6) Predict on 513 test, score on 253 unblind via Molecule Name join.
  7) Save checkpoint C:/pxr_artifacts/nb904_ssl.pt + submission CSV.
"""
from __future__ import annotations

import os
import sys
import math
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger

from pxr.data import load_train
from pxr.eval import rae
from pxr.paths import DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 0
MAX_LEN = 128
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
BATCH = 32
PRE_EPOCHS = 5
FT_EPOCHS = 30
LR_PRE = 3e-4
LR_FT = 1e-4
MASK_PROB = 0.15
TAG = "nb904"
ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cpu")  # transformer is tiny


# ------------------------------------------------------------------ Tokenizer

PAD, CLS, MASK, UNK = "<pad>", "<cls>", "<mask>", "<unk>"
SPECIAL = [PAD, CLS, MASK, UNK]


def build_vocab(smiles_list):
    chars = set()
    for s in smiles_list:
        if isinstance(s, str):
            chars.update(list(s))
    vocab = SPECIAL + sorted(chars)
    stoi = {c: i for i, c in enumerate(vocab)}
    return vocab, stoi


def encode(s: str, stoi: dict, max_len: int = MAX_LEN) -> np.ndarray:
    ids = [stoi[CLS]]
    for ch in str(s)[: max_len - 1]:
        ids.append(stoi.get(ch, stoi[UNK]))
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [stoi[PAD]] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)


def std_smi(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


# ----------------------------------------------------------------- Model

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, max_len=MAX_LEN, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.reg_head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok(x) + self.pos(pos_ids)
        key_padding = (x == self.pad_id)
        h = self.encoder(h, src_key_padding_mask=key_padding)
        h = self.norm(h)
        return h, key_padding

    def pool(self, h, key_padding):
        # mean over non-pad tokens
        mask = (~key_padding).float().unsqueeze(-1)
        s = (h * mask).sum(dim=1)
        n = mask.sum(dim=1).clamp(min=1.0)
        return s / n


# ----------------------------------------------------------------- Datasets

class MLMDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids  # (N, L) int64

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return torch.from_numpy(self.ids[i])


class RegDataset(Dataset):
    def __init__(self, ids, y):
        self.ids = ids
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return torch.from_numpy(self.ids[i]), torch.tensor(self.y[i])


def mask_batch(x, mask_id, pad_id, vocab_size, p=MASK_PROB):
    """15% MLM: 80% mask, 10% random, 10% keep."""
    x = x.clone()
    labels = torch.full_like(x, -100)
    rand = torch.rand_like(x, dtype=torch.float32)
    valid = (x != pad_id) & (x > len(SPECIAL) - 1)  # don't mask specials
    mask_pos = (rand < p) & valid
    labels[mask_pos] = x[mask_pos]
    r = torch.rand_like(x, dtype=torch.float32)
    do_mask = mask_pos & (r < 0.8)
    do_rand = mask_pos & (r >= 0.8) & (r < 0.9)
    x[do_mask] = mask_id
    if do_rand.any():
        x[do_rand] = torch.randint(len(SPECIAL), vocab_size, size=(int(do_rand.sum().item()),))
    return x, labels


# ----------------------------------------------------------------- Main

def main() -> dict:
    print("=" * 78)
    print("nb904 -- SSL char-transformer pretrain + fine-tune")
    print("=" * 78)

    needed = {
        "SP":         DATA_RAW / "pxr-challenge_single_concentration_TRAIN.csv",
        "TEST":       DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":  DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    miss = [k for k, p in needed.items() if not Path(p).exists()]
    if miss:
        print("MISSING:", miss)
        return {"success": False, "missing": miss}

    # ---- Load 21k single-conc SMILES (no labels needed) ----
    sp = pd.read_csv(needed["SP"], usecols=["SMILES"])
    sp_smi = [std_smi(s) for s in sp["SMILES"].tolist()]
    sp_smi = [s for s in sp_smi if isinstance(s, str) and len(s) > 0]
    print(f"Single-conc SMILES (pretrain pool): {len(sp_smi)}")

    # ---- Load 4139 CRC ----
    tr = load_train()
    tr_smi_raw = tr["smiles"].tolist() if "smiles" in tr.columns else tr["SMILES"].tolist()
    tr_smi = [std_smi(s) for s in tr_smi_raw]
    y_tr = tr["pec50"].values.astype(np.float32)
    keep = [i for i, s in enumerate(tr_smi) if isinstance(s, str)]
    tr_smi = [tr_smi[i] for i in keep]
    y_tr = y_tr[keep]
    print(f"Fine-tune CRC: n={len(tr_smi)}")

    # ---- Load 513 test ----
    te_df = pd.read_csv(needed["TEST"])
    te_smi = [std_smi(s) for s in te_df["SMILES"].tolist()]
    te_smi_filled = [s if isinstance(s, str) else "" for s in te_smi]
    print(f"Test: n={len(te_smi_filled)}")

    # ---- Build vocab from union ----
    vocab, stoi = build_vocab(sp_smi + tr_smi + [s for s in te_smi_filled if s])
    pad_id = stoi[PAD]
    mask_id = stoi[MASK]
    V = len(vocab)
    print(f"Vocab size: {V}")

    sp_ids = np.stack([encode(s, stoi) for s in sp_smi])
    tr_ids = np.stack([encode(s, stoi) for s in tr_smi])
    te_ids = np.stack([encode(s, stoi) for s in te_smi_filled])
    print(f"sp_ids={sp_ids.shape}  tr_ids={tr_ids.shape}  te_ids={te_ids.shape}")

    # ---- Build model ----
    model = CharTransformer(V, pad_id=pad_id).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ---- Pretrain (MLM) ----
    print("\n[pretrain] 5 epochs MLM ...")
    pre_loader = DataLoader(MLMDataset(sp_ids), batch_size=BATCH, shuffle=True,
                            num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PRE, weight_decay=0.01)
    model.train()
    for ep in range(PRE_EPOCHS):
        tot, nbat = 0.0, 0
        for batch in pre_loader:
            batch = batch.to(DEVICE)
            x, labels = mask_batch(batch, mask_id, pad_id, V)
            h, _ = model(x)
            logits = model.mlm_head(h)
            loss = F.cross_entropy(logits.view(-1, V), labels.view(-1),
                                   ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
            nbat += 1
        print(f"  epoch {ep+1}/{PRE_EPOCHS}  mlm_loss={tot/max(nbat,1):.4f}")

    # ---- Fine-tune on CRC pEC50 ----
    print("\n[fine-tune] 30 epochs Huber ...")
    ft_loader = DataLoader(RegDataset(tr_ids, y_tr), batch_size=BATCH,
                           shuffle=True, num_workers=0, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=0.01)
    huber = nn.HuberLoss(delta=1.0)
    model.train()
    for ep in range(FT_EPOCHS):
        tot, nbat = 0.0, 0
        for x, y in ft_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            h, kp = model(x)
            pooled = model.pool(h, kp)
            yhat = model.reg_head(pooled).squeeze(-1)
            loss = huber(yhat, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
            nbat += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{FT_EPOCHS}  huber={tot/max(nbat,1):.4f}")

    # ---- Predict on train + test ----
    model.eval()
    def _predict(ids):
        out = np.zeros(len(ids), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(ids), 64):
                xb = torch.from_numpy(ids[i:i+64]).to(DEVICE)
                h, kp = model(xb)
                p = model.reg_head(model.pool(h, kp)).squeeze(-1).cpu().numpy()
                out[i:i+64] = p
        return out

    tr_pred = _predict(tr_ids)
    te_pred = _predict(te_ids)
    train_fit_rae = float(rae(y_tr, tr_pred))
    print(f"\n[train-fit RAE] {train_fit_rae:.4f}  (sanity; not OOF)")

    # ---- Score on 253 unblind ----
    te_names = te_df["Molecule Name"].tolist()
    n2i = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(n2i)].reset_index(drop=True)
    unb_idx = np.array([n2i[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    in_rae = float(rae(unb_y, te_pred[unb_idx]))

    print("\n" + "=" * 78)
    print(f"UNBLIND RAE (n={len(unb_idx)})  = {in_rae:.4f}")
    print(f"  truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"  pred  mean/std = {te_pred.mean():.3f} / {te_pred.std():.3f}")
    print("=" * 78)

    # ---- Save checkpoint + submission ----
    ckpt = ART / f"{TAG}_ssl.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "stoi": stoi,
        "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                       max_len=MAX_LEN, vocab_size=V, pad_id=pad_id),
        "in_rae": in_rae,
    }, ckpt)
    print(f"Saved checkpoint -> {ckpt}")

    np.save(ART / f"{TAG}_te_pred.npy", te_pred.astype(np.float32))
    np.save(ART / f"{TAG}_tr_pred.npy", tr_pred.astype(np.float32))

    sub = SUBMISSIONS / f"{TAG}_ssl_pretrain_ft.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": te_pred.astype(np.float32),
    }).to_csv(sub, index=False)
    print(f"Wrote {sub}")

    return {
        "success": True,
        "n_params": int(n_params),
        "vocab_size": int(V),
        "n_pretrain": int(len(sp_ids)),
        "n_finetune": int(len(tr_ids)),
        "train_fit_rae": train_fit_rae,
        "in_rae": in_rae,
        "checkpoint": str(ckpt),
        "submission": str(sub),
    }


if __name__ == "__main__":
    r = main()
    print("\n==== SUMMARY ====")
    for k, v in r.items():
        print(f"  {k}: {v}")
