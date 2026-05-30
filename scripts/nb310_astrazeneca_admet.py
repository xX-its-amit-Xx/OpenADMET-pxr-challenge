"""nb310 -- AstraZeneca ADMET cross-feature augmentation.

Data sources (Harvard Dataverse via TDC, DOI 10.7910/DVN/21LKWG):
  - Lipophilicity_AstraZeneca (4,200 rows)  -> file id 4259595
  - PPBR_AZ                  (1,614+ rows) -> file id 6413140
  - Clearance_Hepatocyte_AZ  (1,102 rows)  -> file id 4266187
  - Clearance_Microsome_AZ   (1,020 rows)  -> file id 4266186

Files are served as .tab (TSV) from https://dataverse.harvard.edu/api/access/datafile/<id>
which 303-redirects to a signed S3 link. urllib follows the redirect.

Pipeline mirrors nb309:
  1. Download 4 AZ datasets into data/external/astrazeneca_admet/
  2. Standardize SMILES, compute InChIKey first-block.
  3. Build per-FB endpoint vector: 4 cols (lipo, ppbr, cl_hep, cl_mic) + has_az.
  4. Augment PXR features; LGBM scaffold-5fold CV.
  5. Save oof_nb310_az.npy + te_nb310_az.npy
  6. SLSQP 5-way blend with nb239 base components.
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
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL


AZ_DATAVERSE = {
    "lipophilicity_az":      4259595,
    "ppbr_az":               6413140,
    "clearance_hepatocyte":  4266187,
    "clearance_microsome":   4266186,
}
AZ_DIR = DATA_EXTERNAL / "astrazeneca_admet"


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


def download_az(name, fid):
    AZ_DIR.mkdir(parents=True, exist_ok=True)
    out = AZ_DIR / f"{name}.tab"
    if out.exists() and out.stat().st_size > 1000:
        print(f"  Cached {name} -> {out} ({out.stat().st_size} bytes)")
        return out
    url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}"
    print(f"  Downloading {name} <- {url}")
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, out)
        print(f"    -> {out} ({out.stat().st_size} bytes)")
    except Exception as e:
        print(f"    FAILED ({e}); skipping {name}")
        return None
    return out


def load_az_table(path):
    """TDC .tab files are TSV with header row including 'Drug', 'Y' (sometimes 'Drug_ID')."""
    if path is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path)
    # Find SMILES column ("Drug") and label column ("Y")
    smi_col = "Drug" if "Drug" in df.columns else (
        "SMILES" if "SMILES" in df.columns else None)
    y_col = "Y" if "Y" in df.columns else None
    if smi_col is None or y_col is None:
        print(f"    Schema unknown: cols={list(df.columns)[:8]}")
        return None
    df = df[[smi_col, y_col]].rename(columns={smi_col: "smiles", y_col: "y"})
    df["smiles"] = df["smiles"].astype(str)
    return df


def fb_vector(df, label):
    """Return dict FB -> mean(y) for this AZ endpoint."""
    if df is None:
        return {}
    df = df.copy()
    df["fb"] = df["smiles"].apply(inchikey_fb)
    df = df.dropna(subset=["fb", "y"])
    out = df.groupby("fb")["y"].mean().to_dict()
    print(f"    {label}: {len(out)} unique FBs")
    return out


def main():
    print("=== nb310: AstraZeneca ADMET cross-augmentation ===\n")

    print("Step 1: Download AZ datasets from Harvard Dataverse")
    paths = {name: download_az(name, fid) for name, fid in AZ_DATAVERSE.items()}

    print("\nStep 2: Parse + per-FB lookup tables")
    fb_maps = {}
    for name, p in paths.items():
        df = load_az_table(p)
        if df is None or len(df) == 0:
            print(f"    {name}: empty / unavailable; skipped")
            fb_maps[name] = {}
            continue
        print(f"    {name}: {len(df)} rows raw")
        fb_maps[name] = fb_vector(df, name)

    if all(len(m) == 0 for m in fb_maps.values()):
        print("\n  ALL AZ DOWNLOADS FAILED -- producing fallback zero-impact predictions")
        tr = load_train(); tr = add_standard_columns(tr)
        y_tr = tr["pec50"].values.astype(np.float64)
        te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
        oof = np.full(len(y_tr), y_tr.mean())
        te_pred = np.full(len(te_df), y_tr.mean())
        np.save(DATA_PROCESSED / "oof_nb310_az.npy", oof)
        np.save(DATA_PROCESSED / "te_nb310_az.npy", te_pred)
        print(f"  Fallback OOF RAE: {rae(y_tr, oof):.4f}")
        return

    # --- 3. Load PXR train + test
    print("\nStep 3: PXR train + test lookup")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    tr_fbs = [inchikey_fb(s) for s in smiles_tr]
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df["SMILES"].apply(std_smi).tolist()
    te_fbs = [inchikey_fb(s) for s in smiles_te]

    endpoint_order = list(AZ_DATAVERSE.keys())
    for ep in endpoint_order:
        nm_tr = sum(1 for fb in tr_fbs if fb in fb_maps[ep])
        nm_te = sum(1 for fb in te_fbs if fb in fb_maps[ep])
        print(f"  {ep}: train matches={nm_tr}, test matches={nm_te}")

    def make_feats(fbs):
        feats = []
        for fb in fbs:
            vec = [fb_maps[ep].get(fb, np.nan) for ep in endpoint_order]
            has = float(any(not np.isnan(v) for v in vec))
            feats.append(vec + [has])
        return np.array(feats, dtype=np.float64)

    AZ_tr = make_feats(tr_fbs)
    AZ_te = make_feats(te_fbs)

    # --- 4. Featurize + train
    print("\nStep 4: Featurize + train LGBM")
    X_tr = combined(smiles_tr); X_tr = impute(X_tr)
    X_te = combined(smiles_te); X_te = impute(X_te)
    X_tr_aug = np.hstack([X_tr, AZ_tr])
    X_te_aug = np.hstack([X_te, AZ_te])
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

    np.save(DATA_PROCESSED / "oof_nb310_az.npy", oof)
    np.save(DATA_PROCESSED / "te_nb310_az.npy", te_pred)
    print(f"  Saved oof_nb310_az.npy + te_nb310_az.npy")

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
              f"mtd={best.x[2]:.3f} loso={best.x[3]:.3f} nb310={best.x[4]:.3f}")
    except FileNotFoundError as e:
        print(f"  Blend skipped: {e}")


if __name__ == "__main__":
    main()
