"""nb309 -- Novartis ADMET cross-feature augmentation.

Data source: https://github.com/molecularinformatics/Computational-ADME
File:        ADME_public_set_3521.csv  (3,521 diverse compounds, 6 endpoints)
Endpoints:   HLM, RLM, Solubility, MDR1-MDCK ER (LOG), hPPB, rPPB

Pipeline:
  1. Download CSV into data/external/novartis_admet/ via urllib.
  2. Standardize SMILES, compute InChIKey + first-block (FB) for lookup.
  3. Build per-FB endpoint vector (mean per endpoint where multiple matches).
  4. For PXR train+test compounds, look up matching FB and add 6 cols
     (NaN if missing). Add binary "has_novartis" indicator (7 cols total).
  5. Augment combined feature matrix; train LGBM scaffold-5fold CV.
  6. Save oof_nb309_novartis.npy + te_nb309_novartis.npy
  7. SLSQP 5-way blend with nb239 base components.
"""
import os, sys, warnings, urllib.request
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from scipy.optimize import minimize
from rdkit import Chem

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns, standardize_smiles
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL


NOVARTIS_URL = "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv"
NOVARTIS_DIR = DATA_EXTERNAL / "novartis_admet"
NOVARTIS_CSV = NOVARTIS_DIR / "ADME_public_set_3521.csv"


def std_smi(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def inchikey_fb(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        ik = Chem.MolToInchiKey(mol)
        return ik.split("-")[0] if ik else None
    except Exception:
        return None


def download_novartis():
    NOVARTIS_DIR.mkdir(parents=True, exist_ok=True)
    if NOVARTIS_CSV.exists() and NOVARTIS_CSV.stat().st_size > 1000:
        print(f"  Already cached: {NOVARTIS_CSV} ({NOVARTIS_CSV.stat().st_size} bytes)")
        return
    print(f"  Downloading {NOVARTIS_URL}")
    urllib.request.urlretrieve(NOVARTIS_URL, NOVARTIS_CSV)
    print(f"  Saved -> {NOVARTIS_CSV} ({NOVARTIS_CSV.stat().st_size} bytes)")


def main():
    print("=== nb309: Novartis ADMET cross-augmentation ===\n")

    # --- 1. Download
    print("Step 1: Download Novartis ADME public set")
    download_novartis()
    nov = pd.read_csv(NOVARTIS_CSV)
    print(f"  Loaded Novartis rows: {len(nov)}, columns: {list(nov.columns)}")

    # --- 2. Standardize + InChIKey FB
    print("\nStep 2: Standardize Novartis SMILES")
    smi_col = "SMILES" if "SMILES" in nov.columns else nov.columns[0]
    nov["std_smiles"] = nov[smi_col].apply(std_smi)
    nov["fb"] = nov[smi_col].apply(inchikey_fb)
    nov = nov.dropna(subset=["fb"]).reset_index(drop=True)
    print(f"  After std: {len(nov)} rows")

    # Endpoint columns: everything numeric except identifiers
    id_cols = {smi_col, "Vendor", "Vendor_ID", "std_smiles", "fb",
               "Internal_ID", "Internal_id", "Vendor ID"}
    endpoint_cols = [c for c in nov.columns
                     if c not in id_cols and pd.api.types.is_numeric_dtype(nov[c])]
    print(f"  Endpoints: {endpoint_cols}")

    # Mean per FB across endpoints (multiple measurements collapse)
    fb_endpoints = nov.groupby("fb")[endpoint_cols].mean()
    print(f"  Unique FBs in Novartis: {len(fb_endpoints)}")

    # --- 3. Load PXR train + test, look up
    print("\nStep 3: PXR train + test lookup")
    tr = load_train()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    tr_fbs = [inchikey_fb(s) for s in smiles_tr]

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()
    te_fbs = [inchikey_fb(s) for s in smiles_te]

    n_match_tr = sum(1 for fb in tr_fbs if fb in fb_endpoints.index)
    n_match_te = sum(1 for fb in te_fbs if fb in fb_endpoints.index)
    print(f"  Train matches: {n_match_tr}/{len(tr_fbs)}")
    print(f"  Test  matches: {n_match_te}/{len(te_fbs)}")

    def make_feats(fbs):
        feats = []
        for fb in fbs:
            if fb in fb_endpoints.index:
                row = fb_endpoints.loc[fb].values
                feats.append(np.concatenate([row, [1.0]]))  # +has_novartis
            else:
                feats.append(np.concatenate([np.full(len(endpoint_cols), np.nan),
                                             [0.0]]))
        return np.array(feats, dtype=np.float64)

    Nov_tr = make_feats(tr_fbs)
    Nov_te = make_feats(te_fbs)

    # --- 4. Combine with standard features
    print("\nStep 4: Featurize + train LGBM")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    X_tr_aug = np.hstack([X_tr, Nov_tr])
    X_te_aug = np.hstack([X_te, Nov_te])
    print(f"  X_tr shape: {X_tr_aug.shape}, X_te shape: {X_te_aug.shape}")

    folds = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)
    LGBM = dict(n_estimators=1500, num_leaves=63, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                objective="mae", n_jobs=4, random_state=42, verbose=-1)

    oof = np.zeros(len(y_tr))
    te_preds = []
    for ti, vi in folds:
        md = lgb.LGBMRegressor(**LGBM)
        md.fit(X_tr_aug[ti], y_tr[ti],
               eval_set=[(X_tr_aug[vi], y_tr[vi])],
               callbacks=[lgb.early_stopping(80, verbose=False),
                          lgb.log_evaluation(-1)])
        oof[vi] = md.predict(X_tr_aug[vi])
        te_preds.append(md.predict(X_te_aug))
    te_pred = np.mean(te_preds, axis=0)
    r = rae(y_tr, oof)
    sp, _ = spearmanr(y_tr, oof)
    print(f"\n  OOF RAE: {r:.4f}  Spearman: {sp:.4f}  te_std: {te_pred.std():.3f}")

    np.save(DATA_PROCESSED / "oof_nb309_novartis.npy", oof)
    np.save(DATA_PROCESSED / "te_nb309_novartis.npy", te_pred)
    print(f"  Saved oof_nb309_novartis.npy + te_nb309_novartis.npy")

    # --- 5. SLSQP 5-way blend
    print("\nStep 5: SLSQP 5-way blend w/ nb239 base")
    try:
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
            res = minimize(loss, w0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"ftol": 1e-9})
            if best is None or res.fun < best.fun:
                best = res
        print(f"  5-way SLSQP OOF: {best.fun:.4f}")
        print(f"  Weights: nb224={best.x[0]:.3f} nb179s={best.x[1]:.3f} "
              f"mtd={best.x[2]:.3f} loso={best.x[3]:.3f} nb309={best.x[4]:.3f}")
    except FileNotFoundError as e:
        print(f"  Blend skipped: {e}")


if __name__ == "__main__":
    main()
