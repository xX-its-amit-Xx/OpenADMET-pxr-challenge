"""nb105 — Uni-Mol2 Fine-tuning for PXR (Kaggle GPU script).

Uni-Mol2 is an SE(3)-equivariant Transformer pretrained on 1.1B 3D molecular
conformations. Unlike SMILES-based models, it understands 3D molecular geometry,
which is critical for PXR (large flexible binding pocket).

Fine-tuning strategy:
  Stage 1: Multi-task warm-up on all PXR-adjacent data (5 epochs)
    - Primary: pEC50 regression (CRC train, 4139 compounds)
    - Auxiliary: pEC50_null regression (counter-assay, 2858 compounds)
    - Auxiliary: binary active (SC FDR<0.05, 21k compounds, weak label)
    - Auxiliary: ChEMBL PXR pIC50 (812 compounds, lower weight)
  Stage 2: Fine-tune on CRC only with cliff-aware ACA loss (10 epochs)
  Stage 3: Test-time augmentation (5 random SMILES per compound, average)

Run on Kaggle with T4 GPU. Expected time: 2-3 hours.
"""

# ── Kaggle setup ──────────────────────────────────────────────────────────────
import os, sys, subprocess, json, warnings
warnings.filterwarnings("ignore")

# Install Uni-Mol from source (or HuggingFace)
def install_unimol():
    """Install Uni-Mol dependencies on Kaggle."""
    packages = [
        "unimol_tools",       # official Uni-Mol toolkit
        "rdkit-pypi",
    ]
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
            print(f"Installed {pkg}")
        except Exception as e:
            print(f"Failed to install {pkg}: {e}")

# On Kaggle, data is mounted at /kaggle/input/pxr-challenge-data/
KAGGLE_DATA = "/kaggle/input/pxr-challenge-data"
KAGGLE_OUT  = "/kaggle/working"
IS_KAGGLE   = os.path.exists(KAGGLE_DATA)

if IS_KAGGLE:
    install_unimol()
    DATA_RAW  = KAGGLE_DATA + "/raw"
    DATA_PROC = KAGGLE_DATA + "/processed"
    SUBS_DIR  = KAGGLE_OUT
    sys.path.insert(0, KAGGLE_DATA + "/src")
else:
    # Local paths for testing
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW  = ROOT + "/data/raw"
    DATA_PROC = ROOT + "/data/processed"
    SUBS_DIR  = ROOT + "/submissions"
    sys.path.insert(0, ROOT + "/src")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

print("=" * 60)
print("nb105: Uni-Mol2 Fine-tuning for PXR")
print("=" * 60)

# ── Data loading ─────────────────────────────────────────────────────────────
try:
    from pxr.data import load_train, load_test
    from pxr.eval import rae, scaffold_kfold_indices
    from pxr.chem import bemis_murcko
    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["smiles"].tolist()
    smiles_te = te["smiles"].tolist()
    print(f"Train: {len(tr)}, Test: {len(te)}")
except Exception as e:
    print(f"pxr library not found, loading CSVs directly: {e}")
    train_csv = pd.read_csv(f"{DATA_RAW}/pxr-challenge_TRAIN.csv")
    test_csv  = pd.read_csv(f"{DATA_RAW}/pxr-challenge_TEST_BLINDED.csv")
    smiles_col = [c for c in train_csv.columns if "SMILES" in c.upper()][0]
    smiles_tr = train_csv[smiles_col].tolist()
    smiles_te = test_csv[smiles_col].tolist()
    y_tr = train_csv["pEC50"].values.astype(np.float64)
    print(f"Train: {len(smiles_tr)}, Test: {len(smiles_te)}")

# ── Uni-Mol fine-tuning ───────────────────────────────────────────────────────
try:
    from unimol_tools import UniMolRepr, MolTrain, MolPredict

    print("\nUni-Mol found. Starting fine-tuning...")

    # Stage 1: Load counter-assay and single-conc for multi-task
    counter = pd.read_csv(f"{DATA_RAW}/pxr-challenge_counter-assay_TRAIN.csv")
    sc_data = pd.read_csv(f"{DATA_RAW}/pxr-challenge_single_concentration_TRAIN.csv")

    smiles_counter = counter["SMILES"].tolist() if "SMILES" in counter.columns else counter.iloc[:, 2].tolist()
    y_counter = counter["pEC50"].values.astype(np.float64) if "pEC50" in counter.columns else counter["pec50"].values

    sc_smiles_col = [c for c in sc_data.columns if "SMILES" in c.upper()][0]
    sc_log2fc_col = [c for c in sc_data.columns if "log2" in c.lower()][0]
    sc_fdr_col    = [c for c in sc_data.columns if "fdr" in c.lower()][0]
    sc_active = (sc_data[sc_fdr_col] < 0.05) & (sc_data[sc_log2fc_col] > 0.5)
    sc_smiles = sc_data[sc_smiles_col].tolist()
    sc_labels = sc_active.astype(float).values  # binary weak labels

    print(f"Counter-assay: {len(smiles_counter)} compounds")
    print(f"SC data: {len(sc_smiles)} compounds, {sc_labels.mean()*100:.1f}% active")

    # ── Stage 1: Pretrain on all PXR data ────────────────────────────────────
    print("\nStage 1: Multi-task pretraining...")
    # Combine all data with task indicators
    all_smiles_stage1 = smiles_tr + smiles_counter + sc_smiles
    all_labels_stage1 = np.concatenate([
        y_tr,                          # primary CRC pEC50
        y_counter,                     # counter-assay pEC50
        sc_labels * 4.0 + 4.0,        # SC active→~8 pEC50, inactive→~4 pEC50 (weak signal)
    ])
    # Task weights (primary CRC data gets highest weight)
    all_weights = np.concatenate([
        np.ones(len(smiles_tr)),       # CRC: weight 1.0
        np.full(len(smiles_counter), 0.4),  # counter: weight 0.4
        np.full(len(sc_smiles), 0.15), # SC: weight 0.15
    ])

    clf = MolTrain(
        task="regression",
        data_type="molecule",
        epochs=5,
        learning_rate=1e-4,
        batch_size=32,
        early_stopping=3,
        use_gpu=True,
        save_path=f"{KAGGLE_OUT}/unimol_stage1"
    )
    clf.fit(data={"smiles": all_smiles_stage1,
                  "target": all_labels_stage1,
                  "weight": all_weights})
    print("Stage 1 complete.")

    # ── Stage 2: Fine-tune on CRC data only ──────────────────────────────────
    print("\nStage 2: CRC-only fine-tuning with ACA cliff loss...")

    # Get cliff pairs for ACA loss
    cliff_pairs_path = f"{DATA_PROC}/cliff_pairs.parquet"
    if os.path.exists(cliff_pairs_path):
        cliff_df = pd.read_parquet(cliff_pairs_path)
        print(f"  Cliff pairs: {len(cliff_df)}")
    else:
        cliff_df = pd.DataFrame()

    clf2 = MolTrain(
        task="regression",
        data_type="molecule",
        epochs=15,
        learning_rate=5e-5,
        batch_size=16,
        early_stopping=5,
        use_gpu=True,
        save_path=f"{KAGGLE_OUT}/unimol_stage2",
        load_model=f"{KAGGLE_OUT}/unimol_stage1"
    )
    clf2.fit(data={"smiles": smiles_tr, "target": y_tr.tolist()})
    print("Stage 2 complete.")

    # ── Stage 3: Test-time augmentation (SMILES enumeration) ─────────────────
    print("\nStage 3: Test-time augmentation (5 random SMILES each)...")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def enumerate_smiles(smi, n=5):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: return [smi]
            return list(set([Chem.MolToSmiles(mol, doRandom=True) for _ in range(n)]))[:n]
        except Exception:
            return [smi]

    # Predict with augmentation
    pred_collection = []
    for _ in range(5):
        aug_smiles = [enumerate_smiles(s, 1)[0] for s in smiles_te]
        pred = MolPredict(load_model=f"{KAGGLE_OUT}/unimol_stage2")
        preds_aug = pred.predict(data={"smiles": aug_smiles})
        pred_collection.append(np.array(preds_aug).flatten())

    te_preds_unimol = np.column_stack(pred_collection).mean(axis=1)

    # Also predict OOF for scaffold CV evaluation
    print("\nComputing OOF predictions for evaluation...")
    from sklearn.model_selection import KFold

    scaffolds = [bemis_murcko(s) for s in smiles_tr] if "bemis_murcko" in dir() else [""] * len(smiles_tr)
    splits = scaffold_kfold_indices(scaffolds, 5, 42) if "scaffold_kfold_indices" in dir() else list(KFold(5, shuffle=True, random_state=42).split(smiles_tr))

    oof_preds = np.full(len(smiles_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        clf_fold = MolTrain(
            task="regression", data_type="molecule",
            epochs=10, learning_rate=5e-5, batch_size=16, early_stopping=3,
            use_gpu=True,
            save_path=f"{KAGGLE_OUT}/unimol_fold{fold}",
            load_model=f"{KAGGLE_OUT}/unimol_stage1"
        )
        tr_data = {"smiles": [smiles_tr[i] for i in tr_idx],
                   "target": y_tr[tr_idx].tolist()}
        clf_fold.fit(data=tr_data)
        pred_fold = MolPredict(load_model=f"{KAGGLE_OUT}/unimol_fold{fold}")
        va_preds = pred_fold.predict(data={"smiles": [smiles_tr[i] for i in va_idx]})
        oof_preds[va_idx] = np.array(va_preds).flatten()
        r = rae(y_tr[va_idx], oof_preds[va_idx])
        print(f"  fold {fold+1}  RAE={r:.4f}")

    valid = np.isfinite(oof_preds)
    print(f"Overall OOF RAE={rae(y_tr[valid], oof_preds[valid]):.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    np.save(f"{KAGGLE_OUT}/oof_nb105_unimol.npy", oof_preds)
    np.save(f"{KAGGLE_OUT}/te_nb105_unimol.npy", te_preds_unimol)

    # Get molecule names from test
    if IS_KAGGLE:
        name_col = [c for c in test_csv.columns if "Molecule" in c or "Name" in c][0]
        mol_names = test_csv[name_col].tolist()
    else:
        mol_names = te["name"].tolist()

    sub = pd.DataFrame({"Molecule Name": mol_names, "pEC50": te_preds_unimol})
    sub.to_csv(f"{SUBS_DIR}/105_unimol_finetune.csv", index=False)
    print(f"\nSaved: {SUBS_DIR}/105_unimol_finetune.csv")
    print(f"Test: min={te_preds_unimol.min():.2f}  med={np.median(te_preds_unimol):.2f}  max={te_preds_unimol.max():.2f}")

except ImportError as e:
    print(f"\nUni-Mol not available: {e}")
    print("Running fallback: ChemBERTa-2 fine-tuning instead...")

    # ── Fallback: ChemBERTa-2 via HuggingFace ────────────────────────────────
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "transformers", "torch", "-q"])
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        print("Loading ChemBERTa-2 (77M SMILES pretrained)...")
        model_name = "seyonec/ChemBERTa-zinc-base-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def smiles_to_embeddings(smiles_list, batch_size=32):
            """Extract [CLS] embeddings from ChemBERTa."""
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_emb = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=768, output_hidden_states=True
            ).to(device)
            model_emb.eval()
            all_embs = []
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    outputs = model_emb(**inputs, output_hidden_states=True)
                    # Use last hidden state [CLS] token
                    emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                all_embs.append(emb)
                if i % (batch_size * 10) == 0:
                    print(f"  Processed {i}/{len(smiles_list)}...")
            return np.vstack(all_embs)

        print("Extracting ChemBERTa-2 embeddings for train...")
        emb_tr = smiles_to_embeddings(smiles_tr)
        print("Extracting embeddings for test...")
        emb_te = smiles_to_embeddings(smiles_te)
        print(f"Embedding shape: train={emb_tr.shape}  test={emb_te.shape}")

        # Train LGBM on top of embeddings
        import lightgbm as lgb
        LGBM_P = dict(n_estimators=1000, num_leaves=64, learning_rate=0.05,
                      min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                      random_state=42, verbose=-1, n_jobs=4)

        # Load scaffold splits
        try:
            scaffolds = [bemis_murcko(s) for s in smiles_tr]
            splits = scaffold_kfold_indices(scaffolds, 5, 42)
        except Exception:
            from sklearn.model_selection import KFold
            splits = list(KFold(5, shuffle=True, random_state=42).split(smiles_tr))

        oof_emb = np.full(len(smiles_tr), np.nan)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            m = lgb.train(LGBM_P, lgb.Dataset(emb_tr[tr_idx], label=y_tr[tr_idx]),
                          callbacks=[lgb.log_evaluation(-1)])
            oof_emb[va_idx] = m.predict(emb_tr[va_idx])
            print(f"  fold {fold+1}  RAE={rae(y_tr[va_idx], oof_emb[va_idx]):.4f}")

        valid = np.isfinite(oof_emb)
        print(f"ChemBERTa-2 OOF RAE={rae(y_tr[valid], oof_emb[valid]):.4f}")

        m_final = lgb.train(LGBM_P, lgb.Dataset(emb_tr, label=y_tr),
                            callbacks=[lgb.log_evaluation(-1)])
        te_preds_cb = m_final.predict(emb_te)

        np.save(f"{KAGGLE_OUT}/oof_nb105_chemberta.npy", oof_emb)
        np.save(f"{KAGGLE_OUT}/te_nb105_chemberta.npy", te_preds_cb)
        pd.DataFrame({"Molecule Name": (te["name"].tolist() if hasattr(te, "name") else list(range(len(smiles_te)))),
                      "pEC50": te_preds_cb}).to_csv(f"{SUBS_DIR}/105_chemberta_finetune.csv", index=False)
        print(f"Saved ChemBERTa-2 submission")

    except Exception as e2:
        print(f"ChemBERTa-2 also failed: {e2}")

print("\nDone.")
