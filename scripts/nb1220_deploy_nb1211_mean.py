"""nb1220 — Deploy nb1211 (naive 0.5/0.5 of nb1190 BoB + nb1200 BoB MEAN) to 513-row CSV."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"
SUB = ROOT / "submissions"
RAW = ROOT / "data" / "raw"

# 1. Load components
te_nb1190 = np.load(PROC / "te_nb1190.npy")
te_nb1210 = np.load(PROC / "te_nb1210.npy")
print(f"te_nb1190 shape={te_nb1190.shape} mean={te_nb1190.mean():.4f} std={te_nb1190.std():.4f}")
print(f"te_nb1210 shape={te_nb1210.shape} mean={te_nb1210.mean():.4f} std={te_nb1210.std():.4f}")

assert te_nb1190.shape == (513,), f"nb1190 not 513: {te_nb1190.shape}"
assert te_nb1210.shape == (513,), f"nb1210 not 513: {te_nb1210.shape}"

# 2. Naive 0.5/0.5 blend
te_nb1220 = 0.5 * te_nb1190 + 0.5 * te_nb1210
print(f"te_nb1220 shape={te_nb1220.shape}")
print(f"  mean={te_nb1220.mean():.4f}")
print(f"  std ={te_nb1220.std():.4f}")
print(f"  min ={te_nb1220.min():.4f}")
print(f"  max ={te_nb1220.max():.4f}")
print(f"  nan ={np.isnan(te_nb1220).sum()}")

# 3. Save .npy
out_npy = PROC / "te_nb1220.npy"
np.save(out_npy, te_nb1220)
print(f"Saved: {out_npy}")

# 4. In-sample evaluation on unblind (if available)
unb_idx_path = PROC / "unblind_idx_in_test.npy"
y_unb_path = PROC / "y_unblind_253.npy"
if unb_idx_path.exists() and y_unb_path.exists():
    unb_idx = np.load(unb_idx_path)
    y_unb = np.load(y_unb_path)
    pred_unb = te_nb1220[unb_idx]
    rae = np.mean(np.abs(pred_unb - y_unb)) / np.mean(np.abs(y_unb - np.mean(y_unb)))
    print(f"in_RAE (unblind 253) = {rae:.4f}")
else:
    # Try alternative paths
    alt_unb = list(PROC.glob("*unblind*"))
    print(f"unblind files: {alt_unb[:5]}")

# 5. Build submission CSV — SMILES + Molecule Name + pEC50
test_csv = RAW / "pxr-challenge_TEST_BLINDED.csv"
test_df = pd.read_csv(test_csv)
print(f"test_df cols: {list(test_df.columns)}")
print(f"test_df rows: {len(test_df)}")

# Identify SMILES and name columns
smi_col = None
name_col = None
for c in test_df.columns:
    if c.lower() in ("smiles", "smi"):
        smi_col = c
    if "name" in c.lower() or "molecule" in c.lower() or "id" in c.lower():
        if name_col is None:
            name_col = c

print(f"smi_col={smi_col}, name_col={name_col}")

out_df = pd.DataFrame({
    "SMILES": test_df[smi_col].values,
    "Molecule Name": test_df[name_col].values,
    "pEC50": te_nb1220,
})
assert len(out_df) == 513
assert out_df["pEC50"].isna().sum() == 0

out_csv = SUB / "nb1220_deploy_nb1211_mean.csv"
out_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
print(f"CSV rows: {len(out_df)}, NaN: {out_df['pEC50'].isna().sum()}")
print(out_df.head(3))
