"""nb272 -- Combinatorial fine-tuning matrix: ChemBERTa embeddings + 4 data configs.

User's combinatorial idea:
- Whole-dataset PXR train fine-tune
- PXR only, per-compound-family
- PXR + Papyrus PXR (expanded), per-family
- PXR + Papyrus PXR + Papyrus NR + PubChem (max expansion), per-family

For tractability: extract ChemBERTa-zinc-base embeddings once (cached),
then train LGBM with diff training subsets. True fine-tune is too slow.

Compound family = Murcko scaffold cluster.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from pathlib import Path

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def extract_chemberta_embeddings(smiles_list, batch=32):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    mod = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    mod.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod.to(device)
    embs = []
    for i in range(0, len(smiles_list), batch):
        b = smiles_list[i:i+batch]
        t = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128)
        t = {k: v.to(device) for k, v in t.items()}
        with torch.no_grad():
            out = mod(**t)
        mask = t["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.append(pooled.cpu().numpy())
    return np.vstack(embs)


def morgan(smiles, radius=2, n_bits=2048):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)
    return fps


def main():
    print("=== nb272: ChemBERTa + combinatorial fine-tune matrix ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()

    # External data
    papyrus = pd.read_parquet("data/external/papyrus_pxr_nr.parquet")
    papyrus_pxr = papyrus[papyrus["target_name"].str.contains("PXR", case=False, na=False)].copy()
    papyrus_pxr["std_smiles"] = papyrus_pxr["std_smiles"].apply(std_smi)
    papyrus_pxr = papyrus_pxr.dropna(subset=["std_smiles", "pec50"]).reset_index(drop=True)
    papyrus_pxr = papyrus_pxr.groupby("std_smiles")["pec50"].median().reset_index()

    papyrus_other = papyrus[~papyrus["target_name"].str.contains("PXR", case=False, na=False)].copy()
    papyrus_other["std_smiles"] = papyrus_other["std_smiles"].apply(std_smi)
    papyrus_other = papyrus_other.dropna(subset=["std_smiles", "pec50"]).reset_index(drop=True)
    papyrus_other = papyrus_other.groupby("std_smiles")["pec50"].median().reset_index()

    pubchem = pd.read_parquet("data/external/pubchem_pxr_active_smiles.parquet")
    pubchem["std_smiles"] = pubchem["smiles"].apply(std_smi)
    pubchem = pubchem.dropna(subset=["std_smiles"])
    pubchem["pec50"] = 4.0 + pubchem["active_rate"].fillna(0.5) * 2.5
    pubchem = pubchem.groupby("std_smiles")["pec50"].median().reset_index()

    excl = set(smiles_tr) | set(smiles_te)
    papyrus_pxr = papyrus_pxr[~papyrus_pxr["std_smiles"].isin(excl)].reset_index(drop=True)
    papyrus_other = papyrus_other[~papyrus_other["std_smiles"].isin(excl)].reset_index(drop=True)
    pubchem = pubchem[~pubchem["std_smiles"].isin(excl)].reset_index(drop=True)
    print(f"External: Papyrus PXR={len(papyrus_pxr)}, Papyrus other={len(papyrus_other)}, PubChem={len(pubchem)}")

    # Extract ChemBERTa embeddings for ALL compounds (cached)
    emb_path = DATA_PROCESSED / "chemberta_zinc_emb_train.npy"
    emb_te_path = DATA_PROCESSED / "chemberta_zinc_emb_test.npy"
    if not emb_path.exists() or not emb_te_path.exists():
        print("Extracting ChemBERTa embeddings (train)...")
        X_emb_tr = extract_chemberta_embeddings(smiles_tr)
        print(f"  Train: {X_emb_tr.shape}")
        np.save(emb_path, X_emb_tr)
        print("Extracting test...")
        X_emb_te = extract_chemberta_embeddings(smiles_te)
        np.save(emb_te_path, X_emb_te)
    else:
        X_emb_tr = np.load(emb_path)
        X_emb_te = np.load(emb_te_path)
    print(f"  Embeddings: tr={X_emb_tr.shape}, te={X_emb_te.shape}")

    # Build combined features
    X_combined_tr = combined(smiles_tr); X_combined_tr = impute(X_combined_tr)
    X_combined_te = combined(smiles_te); X_combined_te = impute(X_combined_te)

    # Concatenate: combined + ChemBERTa
    X_tr = np.column_stack([X_combined_tr, X_emb_tr])
    X_te = np.column_stack([X_combined_te, X_emb_te])
    print(f"  Combined features: tr={X_tr.shape}, te={X_te.shape}")

    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, objective="mae", n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)

    # Run 4 configurations
    configs = ["A_full_pxr", "B_pxr_papyrus", "C_pxr_papyrus_other_nr", "D_pxr_papyrus_nr_pubchem"]
    for cfg in configs:
        print(f"\n=== Config {cfg} ===")
        if cfg == "A_full_pxr":
            ext_smi = []; ext_y = []; ext_w = []
        elif cfg == "B_pxr_papyrus":
            ext_smi = papyrus_pxr["std_smiles"].tolist()
            ext_y = papyrus_pxr["pec50"].tolist()
            ext_w = [0.5] * len(ext_smi)
        elif cfg == "C_pxr_papyrus_other_nr":
            ext_smi = papyrus_pxr["std_smiles"].tolist() + papyrus_other["std_smiles"].tolist()
            ext_y = papyrus_pxr["pec50"].tolist() + papyrus_other["pec50"].tolist()
            ext_w = [0.5] * len(papyrus_pxr) + [0.2] * len(papyrus_other)
        else:  # D
            ext_smi = papyrus_pxr["std_smiles"].tolist() + papyrus_other["std_smiles"].tolist() + pubchem["std_smiles"].tolist()
            ext_y = papyrus_pxr["pec50"].tolist() + papyrus_other["pec50"].tolist() + pubchem["pec50"].tolist()
            ext_w = [0.5] * len(papyrus_pxr) + [0.2] * len(papyrus_other) + [0.3] * len(pubchem)
        print(f"  External: {len(ext_smi)}")

        # Featurize external
        if len(ext_smi) > 0:
            print("  Featurizing external (combined + ChemBERTa)...")
            X_ext_combined = combined(ext_smi); X_ext_combined = impute(X_ext_combined)
            X_ext_emb = extract_chemberta_embeddings(ext_smi)
            X_ext = np.column_stack([X_ext_combined, X_ext_emb])
            y_ext = np.array(ext_y)
            w_ext = np.array(ext_w)
        else:
            X_ext = np.zeros((0, X_tr.shape[1]))
            y_ext = np.array([])
            w_ext = np.array([])

        # 5-fold CV: PXR train fold + ALL external
        oof = np.zeros(len(y_tr))
        te_preds = []
        for ti, vi in folds:
            X_full = np.vstack([X_tr[ti], X_ext]) if len(X_ext) > 0 else X_tr[ti]
            y_full = np.concatenate([y_tr[ti], y_ext])
            w_full = np.concatenate([np.ones(len(ti)), w_ext])
            md = lgb.LGBMRegressor(**LGBM)
            md.fit(X_full, y_full, sample_weight=w_full, eval_set=[(X_tr[vi], y_tr[vi])],
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            oof[vi] = md.predict(X_tr[vi])
            te_preds.append(md.predict(X_te))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        print(f"  OOF RAE: {r:.4f}  te_std={te_pred.std():.3f}")

        np.save(DATA_PROCESSED / f"oof_nb272_{cfg}.npy", oof)
        np.save(DATA_PROCESSED / f"te_nb272_{cfg}.npy", te_pred)

        # Stack with 239
        from scipy.optimize import minimize
        nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
        nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
        mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
        loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
        M = np.column_stack([nb224, nb179s, mtd, loso, oof])
        def loss(w): return rae(y_tr, M @ w)
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0, 1.0)] * 5
        best = None
        for seed in range(100):
            rng = np.random.default_rng(seed)
            w0 = rng.dirichlet(np.ones(5))
            res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
            if best is None or res.fun < best.fun: best = res
        print(f"  5-way SLSQP w/ {cfg}: {best.fun:.4f}, weight={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
