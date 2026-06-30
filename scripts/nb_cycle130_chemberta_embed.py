"""cycle-130 M2 — ChemBERTa-77M-MTR embeddings for all PXR compounds.

Loads all train + unblind + test SMILES, standardizes, dedupes by InChIKey,
and extracts CLS-pooled embeddings from DeepChem/ChemBERTa-77M-MTR.

Outputs:
    data/processed/chemberta_77m_mtr_embeddings.npy   (N, 384)
    data/processed/chemberta_77m_mtr_index.csv        InChIKey -> row index

Note: ChemBERTa-77M-MTR hidden_size is 384 (not 768 as one might assume from
"77M" parameter count). This is the actual model dimension.
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
from pxr.chem import standardize, standardize_smiles  # noqa: E402
from pxr.paths import DATA_PROCESSED  # noqa: E402

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
BATCH = 16
MAX_LEN = 256
WALL_BUDGET_S = 1800  # 30 min


def main() -> int:
    t0 = time.time()
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:  # noqa: BLE001
        print(f"[cycle130] import failure: {e}")
        return 2

    # 1. Load all SMILES
    tr = load_train()[["name", "smiles"]]
    te = load_test()[["name", "smiles"]]
    unb = pd.read_csv(ROOT / "data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")[["Molecule Name", "SMILES"]]
    unb = unb.rename(columns={"Molecule Name": "name", "SMILES": "smiles"})
    print(f"[cycle130] loaded train={len(tr)} test={len(te)} unblind={len(unb)}")

    all_df = pd.concat([tr, te, unb], ignore_index=True)
    print(f"[cycle130] total raw rows: {len(all_df)}")

    # 2. Standardize + InChIKey dedup
    print("[cycle130] standardizing...")

    def std_to_smiles(smi: str) -> str | None:
        mol = standardize(smi)
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    all_df["std_smiles"] = all_df["smiles"].astype(str).apply(std_to_smiles)
    all_df = all_df[all_df["std_smiles"].notna()].reset_index(drop=True)
    print(f"[cycle130] after standardize: {len(all_df)}")

    def to_inchikey(smi: str) -> str | None:
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return None
            return MolToInchiKey(m)
        except Exception:
            return None

    all_df["inchikey"] = all_df["std_smiles"].apply(to_inchikey)
    all_df = all_df.dropna(subset=["inchikey"]).reset_index(drop=True)
    unique = all_df.drop_duplicates(subset=["inchikey"]).reset_index(drop=True)
    print(f"[cycle130] unique by InChIKey: {len(unique)}")

    # 3. Load ChemBERTa
    print(f"[cycle130] loading {MODEL_NAME}...")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
    except Exception as e:  # noqa: BLE001
        print(f"[cycle130] model load failed: {e}")
        return 2

    hidden = model.config.hidden_size
    print(f"[cycle130] hidden_size={hidden}")

    # 4. Compute CLS-pooled embeddings, batch=16
    smiles_list = unique["std_smiles"].tolist()
    N = len(smiles_list)
    out = np.zeros((N, hidden), dtype=np.float32)
    print(f"[cycle130] embedding {N} unique compounds (batch={BATCH}, CLS-pooled)...")
    with torch.no_grad():
        for i in range(0, N, BATCH):
            if time.time() - t0 > WALL_BUDGET_S:
                print(f"[cycle130] wall budget exceeded at {i}/{N}")
                break
            chunk = smiles_list[i : i + BATCH]
            enc = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            hs = model(**enc).last_hidden_state  # (B, L, H)
            # CLS pooling: [CLS] token is at position 0
            cls = hs[:, 0, :]  # (B, H)
            out[i : i + len(chunk)] = cls.cpu().numpy().astype(np.float32)
            if (i // BATCH) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  [{i + len(chunk)}/{N}]  t={elapsed:.0f}s", flush=True)

    # 5. Save
    emb_path = DATA_PROCESSED / "chemberta_77m_mtr_embeddings.npy"
    idx_path = DATA_PROCESSED / "chemberta_77m_mtr_index.csv"
    np.save(emb_path, out)
    index_df = pd.DataFrame({
        "row_idx": np.arange(N),
        "inchikey": unique["inchikey"].values,
        "name": unique["name"].values,
        "std_smiles": unique["std_smiles"].values,
    })
    index_df.to_csv(idx_path, index=False)
    size_mb = emb_path.stat().st_size / 1e6
    print(f"[cycle130] saved {emb_path}  shape={out.shape}  size={size_mb:.2f} MB")
    print(f"[cycle130] saved {idx_path}  rows={len(index_df)}")

    # 6. Sanity check: nearest neighbors of a known PXR active
    # Rifampicin SMILES: a well-known PXR agonist; check by InChIKey if present
    known_actives = {
        "rifampicin": "CC1C=CC=C(C)C(=O)NC2=C(O)C3=C(O)C(C)=C4OC(C)(O)C(OC(C)=O)C(C)C(O)C(C)C(O)C(C)/C=C/C=C(C)/C(=O)NC1=C2C3=O4",
        "SR12813": "CCC(C)C1=NC2=C(C1)C(=O)c1ccccc1C2=O",
    }
    # Sanity: find a high-activity compound from training
    tr_full = load_train()
    top_active = tr_full.nlargest(1, "pec50")
    top_smi = top_active["smiles"].iloc[0]
    top_name = top_active["name"].iloc[0]
    top_std = standardize(top_smi)
    top_ik = to_inchikey(top_std) if top_std else None
    if top_ik in index_df["inchikey"].values:
        anchor_idx = index_df.index[index_df["inchikey"] == top_ik][0]
        anchor_emb = out[anchor_idx]
        # cosine sim
        norms = np.linalg.norm(out, axis=1) + 1e-9
        sims = (out @ anchor_emb) / (norms * np.linalg.norm(anchor_emb) + 1e-9)
        topk = np.argsort(-sims)[:6]
        print(f"\n[cycle130] SANITY CHECK: nearest neighbors of top active '{top_name}' (pEC50={top_active['pec50'].iloc[0]:.2f})")
        for k in topk:
            ik = index_df["inchikey"].iloc[k]
            nm = index_df["name"].iloc[k]
            sm = index_df["std_smiles"].iloc[k]
            print(f"  sim={sims[k]:.4f}  name={nm}  smi={sm[:60]}")

    print(f"[cycle130] DONE  wall_time={time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
