"""nb252 -- MolFormer-XL embeddings for PXR challenge.

Pull ibm/MoLFormer-XL-both-10pct from HuggingFace. Extract pooled embeddings
(~768-dim) for all 4139 train + 513 test SMILES. Train LGBM on those.

If embeddings provide genuinely orthogonal signal, blending should help.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModel

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def extract_embeddings(smiles_list, batch_size=32):
    print(f"Loading MolFormer-XL from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True, deterministic_eval=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")

    embeddings = []
    t0 = time.time()
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean-pooled embedding over tokens (mask-aware)
        last_hidden = outputs.last_hidden_state  # (B, L, D)
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        embeddings.append(pooled.cpu().numpy())
        if (i + batch_size) % 256 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + batch_size) * (len(smiles_list) - i - batch_size)
            print(f"  {i+batch_size}/{len(smiles_list)} elapsed={elapsed:.0f}s ETA={eta:.0f}s")
    return np.vstack(embeddings)


def main():
    print("=== nb252: MolFormer-XL embeddings ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].tolist()

    emb_tr_path = DATA_PROCESSED / "X_molformer_tr.npy"
    emb_te_path = DATA_PROCESSED / "X_molformer_te.npy"

    if emb_tr_path.exists() and emb_te_path.exists():
        print("Loading cached embeddings...")
        X_emb_tr = np.load(emb_tr_path)
        X_emb_te = np.load(emb_te_path)
    else:
        print("Extracting train embeddings...")
        X_emb_tr = extract_embeddings(smiles_tr)
        print(f"Train embeddings shape: {X_emb_tr.shape}")
        np.save(emb_tr_path, X_emb_tr)

        print("\nExtracting test embeddings...")
        X_emb_te = extract_embeddings(smiles_te)
        print(f"Test embeddings shape: {X_emb_te.shape}")
        np.save(emb_te_path, X_emb_te)

    print(f"\nEmbeddings ready: tr={X_emb_tr.shape}, te={X_emb_te.shape}")

    # Train LGBM on MolFormer embeddings alone
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    print("\n=== OOF on MolFormer embeddings alone ===")
    oof_mf = np.zeros(len(y_tr))
    te_mf_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_emb_tr[ti], y_tr[ti], eval_set=[(X_emb_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_mf[vi] = md.predict(X_emb_tr[vi])
        te_mf_preds.append(md.predict(X_emb_te))
    te_mf = np.mean(te_mf_preds, axis=0)
    r_mf = rae(y_tr, oof_mf)
    print(f"MolFormer-only OOF RAE: {r_mf:.4f}")
    np.save(DATA_PROCESSED / "oof_nb252_molformer.npy", oof_mf)
    np.save(DATA_PROCESSED / "te_nb252_molformer.npy", te_mf)

    # Combined: base + MolFormer
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    X_tr_aug = np.column_stack([X_tr, X_emb_tr])
    X_te_aug = np.column_stack([X_te, X_emb_te])
    print(f"\n=== OOF on base + MolFormer ===")
    oof_aug = np.zeros(len(y_tr))
    te_aug_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr_aug[ti], y_tr[ti], eval_set=[(X_tr_aug[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof_aug[vi] = md.predict(X_tr_aug[vi])
        te_aug_preds.append(md.predict(X_te_aug))
    te_aug = np.mean(te_aug_preds, axis=0)
    print(f"Base+MolFormer OOF RAE: {rae(y_tr, oof_aug):.4f}")
    np.save(DATA_PROCESSED / "oof_nb252_combined.npy", oof_aug)
    np.save(DATA_PROCESSED / "te_nb252_combined.npy", te_aug)

    # Stack with 239
    print(f"\n=== Stack nb252_molformer with 239 components ===")
    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    print(f"nb239 baseline OOF: {rae(y_tr, nb239_oof):.4f}")
    # Correlation
    print(f"corr(nb239, mf): {np.corrcoef(nb239_oof, oof_mf)[0,1]:.4f}")
    # Best blend
    for w in [0.05, 0.10, 0.15, 0.20, 0.30]:
        b = (1-w)*nb239_oof + w*oof_mf
        r = rae(y_tr, b)
        sign = " ***" if r < rae(y_tr, nb239_oof) else ""
        print(f"  +mf w={w}: OOF={r:.4f}{sign}")


if __name__ == "__main__":
    main()
