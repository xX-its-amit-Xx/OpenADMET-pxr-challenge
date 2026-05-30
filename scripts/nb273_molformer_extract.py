"""nb273 -- MolFormer-XL embedding extraction (1.6B-pretrained chem foundation model).

Uses our patched cached files (3 monkey-patches applied to make transformers 5.x compatible).
Extracts 768-dim embeddings for all 4139 train + 513 test compounds.
Trains LGBM, checks OOF, blends with nb239.
"""
import os, sys, types, importlib.util, time, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from rdkit import Chem

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED


SRC = "D:/Users/ashenoy00000/.cache/huggingface/modules/transformers_modules/ibm/MolFormer_hyphen_XL_hyphen_both_hyphen_10pct/7b12d946c181a37f6012b9dc3b002275de070314"


def load_molformer():
    for n in ['transformers_modules', 'transformers_modules.ibm', 'transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct']:
        if n not in sys.modules:
            sys.modules[n] = types.ModuleType(n)
    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, file)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod
    cfg = load('transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.configuration_molformer', f'{SRC}/configuration_molformer.py')
    sys.modules['transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.configuration_molformer'] = cfg
    mdl = load('transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.modeling_molformer', f'{SRC}/modeling_molformer.py')
    sys.modules['transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.modeling_molformer'] = mdl
    tok = load('transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.tokenization_molformer', f'{SRC}/tokenization_molformer.py')
    sys.modules['transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.tokenization_molformer'] = tok
    return cfg, mdl, tok


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except: return None


def extract(smiles_list, model, tok, batch=24):
    embs = []
    t0 = time.time()
    device = next(model.parameters()).device
    for i in range(0, len(smiles_list), batch):
        b = smiles_list[i:i+batch]
        inputs = tok(b, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.append(pooled.cpu().numpy())
        if (i+batch) % 256 == 0:
            elapsed = time.time() - t0
            eta = elapsed/(i+batch)*(len(smiles_list)-i-batch)
            print(f"  {i+batch}/{len(smiles_list)} elapsed={elapsed:.0f}s ETA={eta:.0f}s")
    return np.vstack(embs)


def main():
    print("=== nb273: MolFormer-XL embeddings (WORKING) ===\n")
    cfg, mdl, tok = load_molformer()
    config = cfg.MolformerConfig.from_pretrained('ibm/MoLFormer-XL-both-10pct', deterministic_eval=True)
    model = mdl.MolformerModel.from_pretrained('ibm/MoLFormer-XL-both-10pct', config=config)
    model.eval()
    tokenizer = tok.MolformerTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct')
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} params")

    tr = load_train(); tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()

    emb_tr_path = DATA_PROCESSED / "X_molformer_xl_tr.npy"
    emb_te_path = DATA_PROCESSED / "X_molformer_xl_te.npy"
    if not emb_tr_path.exists():
        print("Extracting train embeddings...")
        X_emb_tr = extract(smiles_tr, model, tokenizer)
        np.save(emb_tr_path, X_emb_tr)
    else:
        X_emb_tr = np.load(emb_tr_path)
    print(f"Train embs: {X_emb_tr.shape}")
    if not emb_te_path.exists():
        print("Extracting test embeddings...")
        X_emb_te = extract(smiles_te, model, tokenizer)
        np.save(emb_te_path, X_emb_te)
    else:
        X_emb_te = np.load(emb_te_path)
    print(f"Test embs: {X_emb_te.shape}")

    # LGBM on MolFormer-XL embeddings only
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, objective="mae", n_jobs=4, random_state=42, verbose=-1)
    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)
    print("\n=== OOF on MolFormer-XL only ===")
    oof = np.zeros(len(y_tr))
    te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_emb_tr[ti], y_tr[ti], eval_set=[(X_emb_tr[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_emb_tr[vi])
        te_preds.append(md.predict(X_emb_te))
    te_pred = np.mean(te_preds, axis=0)
    print(f"MolFormer-only OOF RAE: {rae(y_tr, oof):.4f}")
    np.save(DATA_PROCESSED / "oof_nb273_molformer_only.npy", oof)
    np.save(DATA_PROCESSED / "te_nb273_molformer_only.npy", te_pred)

    # Combined + MolFormer
    X_combined_tr = combined(smiles_tr); X_combined_tr = impute(X_combined_tr)
    X_combined_te = combined(smiles_te); X_combined_te = impute(X_combined_te)
    X_tr_aug = np.column_stack([X_combined_tr, X_emb_tr])
    X_te_aug = np.column_stack([X_combined_te, X_emb_te])
    print("\n=== OOF on combined + MolFormer ===")
    oof = np.zeros(len(y_tr))
    te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr_aug[ti], y_tr[ti], eval_set=[(X_tr_aug[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr_aug[vi])
        te_preds.append(md.predict(X_te_aug))
    te_pred = np.mean(te_preds, axis=0)
    print(f"Combined+MolFormer OOF RAE: {rae(y_tr, oof):.4f}")
    np.save(DATA_PROCESSED / "oof_nb273_combined.npy", oof)
    np.save(DATA_PROCESSED / "te_nb273_combined.npy", te_pred)

    # Stack with 239
    print("\n=== 5-way SLSQP w/ MolFormer ===")
    from scipy.optimize import minimize
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")

    for nm, oof_x in [('molformer_only', np.load(DATA_PROCESSED / "oof_nb273_molformer_only.npy")),
                      ('combined+mol', oof)]:
        M = np.column_stack([nb224, nb179s, mtd, loso, oof_x])
        def loss(w): return rae(y_tr, M @ w)
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0, 1.0)] * 5
        best = None
        for seed in range(100):
            rng = np.random.default_rng(seed)
            w0 = rng.dirichlet(np.ones(5))
            res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9})
            if best is None or res.fun < best.fun: best = res
        print(f"  +{nm}: {best.fun:.4f}, weight={best.x[4]:.4f}")


if __name__ == "__main__":
    main()
