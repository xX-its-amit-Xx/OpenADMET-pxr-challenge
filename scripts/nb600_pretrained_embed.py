"""nb600 — Pretrained ChemBERTa embeddings for PXR train + test.

Loads deepchem/ChemBERTa-77M-MTR (already cached locally), tokenizes each
SMILES, and extracts attention-mask-pooled last_hidden_state embeddings.

Outputs:
    data/processed/tr_chemberta.npy   (4139, d)
    data/processed/te_chemberta.npy   (513, d)

CPU-only, batch=32, fp32, no_grad.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxr.data import load_train, load_test  # noqa: E402
from pxr.paths import DATA_PROCESSED  # noqa: E402

MODEL_NAME = "deepchem/ChemBERTa-77M-MTR"
BATCH = 32
MAX_LEN = 256
WALL_BUDGET_S = 600  # 10 min abort budget


def main() -> int:
    t0 = time.time()
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:  # noqa: BLE001
        print(f"[nb600] import failure: {e}")
        return 2

    print(f"[nb600] loading {MODEL_NAME} (cached)...")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
    except Exception as e:  # noqa: BLE001
        print(f"[nb600] model load failed: {e}")
        return 2

    hidden = model.config.hidden_size
    print(f"[nb600] hidden_size={hidden}")

    tr = load_train()[["name", "smiles"]].drop_duplicates(subset=["name"]).reset_index(drop=True)
    te = load_test()[["name", "smiles"]].reset_index(drop=True)
    # Aggregate to per-unique-train compound by name first occurrence
    tr_full = load_train()[["name", "smiles"]].reset_index(drop=True)
    # We want a (4139, d) array matching the *training rows* (not unique). Loaders
    # already return 4139 rows on load_train; keep row order.
    print(f"[nb600] n_train_rows={len(tr_full)}  n_test_rows={len(te)}")

    def embed(smiles_list: list[str]) -> np.ndarray:
        out = np.zeros((len(smiles_list), hidden), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(smiles_list), BATCH):
                if time.time() - t0 > WALL_BUDGET_S:
                    raise TimeoutError("nb600 wall-clock budget exceeded")
                chunk = smiles_list[i : i + BATCH]
                enc = tok(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                )
                hs = model(**enc).last_hidden_state  # (B, L, H)
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                out[i : i + len(chunk)] = pooled.cpu().numpy().astype(np.float32)
                if (i // BATCH) % 20 == 0:
                    print(f"  [{i + len(chunk)}/{len(smiles_list)}]  t={time.time() - t0:.0f}s")
        return out

    try:
        tr_emb = embed(tr_full["smiles"].tolist())
        te_emb = embed(te["smiles"].tolist())
    except TimeoutError as e:
        print(f"[nb600] {e}")
        return 3

    tr_path = DATA_PROCESSED / "tr_chemberta.npy"
    te_path = DATA_PROCESSED / "te_chemberta.npy"
    np.save(tr_path, tr_emb)
    np.save(te_path, te_emb)
    print(f"[nb600] saved {tr_path} {tr_emb.shape}")
    print(f"[nb600] saved {te_path} {te_emb.shape}")
    print(f"[nb600] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
