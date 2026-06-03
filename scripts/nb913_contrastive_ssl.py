"""nb913 -- SimCSE-style contrastive SSL on SMILES + linear-head fine-tune.

Distinct from nb904 (MLM/generative): instance-discrimination contrastive
objective. For each SMILES we sample one random non-canonical variant as the
positive; all other in-batch entries (including their positives) are negatives.
InfoNCE with temperature=0.1.

Pipeline:
  1) Load 21k SP-only SMILES + 4139 CRC -> ~25k pretrain pool.
  2) Char-level transformer, same arch as nb904 (3 layers / d=128 / 4h / L=128).
  3) Contrastive pretrain 15 epochs (CPU), batch 128, temperature 0.1.
  4) Freeze encoder; linear regression head; fine-tune 50 epochs on 4139 CRC.
  5) Score on 253 unblind by Molecule Name join; save submission.
"""
from __future__ import annotations

import os
import sys
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
PRE_BATCH = 128
FT_BATCH = 64
PRE_EPOCHS = 15
FT_EPOCHS = 50
LR_PRE = 3e-4
LR_FT = 1e-3   # higher: only the head trains
TEMP = 0.1
TAG = "nb913"
ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

PAD, CLS, UNK = "<pad>", "<cls>", "<unk>"
SPECIAL = [PAD, CLS, UNK]


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


def rand_smi(smi):
    """Generate one random non-canonical SMILES variant (positive sample)."""
    try:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return smi
        return Chem.MolToSmiles(m, doRandom=True, canonical=False)
    except Exception:
        return smi


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
        # projection head for contrastive (discarded after pretrain)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        B, L = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok(x) + self.pos(pos_ids)
        kp = (x == self.pad_id)
        h = self.encoder(h, src_key_padding_mask=kp)
        h = self.norm(h)
        return h, kp

    def embed(self, x):
        h, kp = self.forward(x)
        mask = (~kp).float().unsqueeze(-1)
        s = (h * mask).sum(dim=1)
        n = mask.sum(dim=1).clamp(min=1.0)
        return s / n


class ContrastiveDataset(Dataset):
    """Re-randomize positive each epoch; encode lazily."""
    def __init__(self, smiles, stoi):
        self.smiles = smiles
        self.stoi = stoi

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        s = self.smiles[i]
        a = encode(s, self.stoi)
        b = encode(rand_smi(s), self.stoi)
        return torch.from_numpy(a), torch.from_numpy(b)


def info_nce(z1, z2, temp=TEMP):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def main() -> dict:
    print("=" * 78)
    print("nb913 -- SimCSE contrastive SSL + frozen-encoder linear fine-tune")
    print("=" * 78)

    needed = {
        "SP": DATA_RAW / "pxr-challenge_single_concentration_TRAIN.csv",
        "TEST": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    miss = [k for k, p in needed.items() if not Path(p).exists()]
    if miss:
        return {"success": False, "missing": miss}

    sp = pd.read_csv(needed["SP"], usecols=["SMILES"])
    sp_smi = [std_smi(s) for s in sp["SMILES"].tolist()]
    sp_smi = sorted({s for s in sp_smi if isinstance(s, str) and s})

    tr = load_train()
    tr_smi_raw = tr["smiles"].tolist() if "smiles" in tr.columns else tr["SMILES"].tolist()
    tr_smi = [std_smi(s) for s in tr_smi_raw]
    y_tr = tr["pec50"].values.astype(np.float32)
    keep = [i for i, s in enumerate(tr_smi) if isinstance(s, str)]
    tr_smi = [tr_smi[i] for i in keep]
    y_tr = y_tr[keep]

    te_df = pd.read_csv(needed["TEST"])
    te_smi = [std_smi(s) for s in te_df["SMILES"].tolist()]
    te_smi_filled = [s if isinstance(s, str) else "" for s in te_smi]

    pretrain_pool = sorted(set(sp_smi) | set(tr_smi))
    print(f"Pretrain pool (SP+CRC, dedup): {len(pretrain_pool)}")
    print(f"Fine-tune CRC: n={len(tr_smi)}   Test: n={len(te_smi_filled)}")

    vocab, stoi = build_vocab(pretrain_pool + [s for s in te_smi_filled if s])
    pad_id = stoi[PAD]
    V = len(vocab)
    print(f"Vocab size: {V}")

    model = CharTransformer(V, pad_id=pad_id).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ---- Contrastive pretrain ----
    print(f"\n[pretrain] {PRE_EPOCHS} epochs SimCSE (T={TEMP}, batch={PRE_BATCH})")
    pre_ds = ContrastiveDataset(pretrain_pool, stoi)
    pre_loader = DataLoader(pre_ds, batch_size=PRE_BATCH, shuffle=True,
                            num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PRE, weight_decay=0.01)
    model.train()
    for ep in range(PRE_EPOCHS):
        tot, nbat = 0.0, 0
        for a, b in pre_loader:
            a = a.to(DEVICE); b = b.to(DEVICE)
            z1 = model.proj(model.embed(a))
            z2 = model.proj(model.embed(b))
            loss = info_nce(z1, z2, TEMP)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); nbat += 1
        print(f"  epoch {ep+1:2d}/{PRE_EPOCHS}  info_nce={tot/max(nbat,1):.4f}")

    enc_ckpt = ART / f"{TAG}_encoder.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab, "stoi": stoi,
        "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                       max_len=MAX_LEN, vocab_size=V, pad_id=pad_id),
    }, enc_ckpt)
    print(f"Saved encoder -> {enc_ckpt}")

    # ---- Freeze encoder, train linear head ----
    print(f"\n[fine-tune] freeze encoder, linear head, {FT_EPOCHS} epochs")
    for p in model.parameters():
        p.requires_grad_(False)
    head = nn.Linear(D_MODEL, 1).to(DEVICE)
    opt_h = torch.optim.AdamW(head.parameters(), lr=LR_FT, weight_decay=0.01)
    huber = nn.HuberLoss(delta=1.0)

    tr_ids = np.stack([encode(s, stoi) for s in tr_smi])
    te_ids = np.stack([encode(s, stoi) for s in te_smi_filled])

    # Precompute embeddings once (encoder is frozen)
    model.eval()
    def _embed_batch(ids):
        out = np.zeros((len(ids), D_MODEL), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(ids), 64):
                xb = torch.from_numpy(ids[i:i+64]).to(DEVICE)
                out[i:i+64] = model.embed(xb).cpu().numpy()
        return out

    tr_emb = _embed_batch(tr_ids)
    te_emb = _embed_batch(te_ids)
    print(f"emb shapes: tr={tr_emb.shape}  te={te_emb.shape}")

    tr_emb_t = torch.from_numpy(tr_emb)
    y_t = torch.from_numpy(y_tr)
    n = len(tr_emb_t)
    head.train()
    for ep in range(FT_EPOCHS):
        perm = torch.randperm(n)
        tot, nbat = 0.0, 0
        for i in range(0, n, FT_BATCH):
            idx = perm[i:i+FT_BATCH]
            xb = tr_emb_t[idx]; yb = y_t[idx]
            yhat = head(xb).squeeze(-1)
            loss = huber(yhat, yb)
            opt_h.zero_grad()
            loss.backward()
            opt_h.step()
            tot += float(loss.item()); nbat += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:2d}/{FT_EPOCHS}  huber={tot/max(nbat,1):.4f}")

    # ---- Predict ----
    head.eval()
    with torch.no_grad():
        tr_pred = head(torch.from_numpy(tr_emb)).squeeze(-1).numpy()
        te_pred = head(torch.from_numpy(te_emb)).squeeze(-1).numpy()
    train_fit_rae = float(rae(y_tr, tr_pred))
    print(f"\n[train-fit RAE] {train_fit_rae:.4f}  (sanity; not OOF)")

    # ---- Unblind score ----
    te_names = te_df["Molecule Name"].tolist()
    n2i = {n_: i for i, n_ in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(n2i)].reset_index(drop=True)
    unb_idx = np.array([n2i[n_] for n_ in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    in_rae = float(rae(unb_y, te_pred[unb_idx]))

    print("\n" + "=" * 78)
    print(f"UNBLIND RAE (n={len(unb_idx)})  = {in_rae:.4f}")
    print(f"  truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"  pred  mean/std = {te_pred.mean():.3f} / {te_pred.std():.3f}")
    print("=" * 78)

    np.save(ART / f"{TAG}_te_pred.npy", te_pred.astype(np.float32))
    np.save(ART / f"{TAG}_tr_pred.npy", tr_pred.astype(np.float32))
    np.save(ART / f"{TAG}_te_emb.npy", te_emb)
    np.save(ART / f"{TAG}_tr_emb.npy", tr_emb)

    sub = SUBMISSIONS / f"{TAG}_contrastive_ssl.csv"
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
        "n_pretrain": int(len(pretrain_pool)),
        "n_finetune": int(len(tr_smi)),
        "train_fit_rae": train_fit_rae,
        "in_rae": in_rae,
        "encoder": str(enc_ckpt),
        "submission": str(sub),
    }


if __name__ == "__main__":
    r = main()
    print("\n==== SUMMARY ====")
    for k, v in r.items():
        print(f"  {k}: {v}")
