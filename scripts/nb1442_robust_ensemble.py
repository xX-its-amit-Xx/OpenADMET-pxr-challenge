"""nb1442 — Deploy 5-way deployable-CSV ensemble (variance reduction at deploy level).

Stacks 5 honest BoB anchors and computes row-level median (primary) + mean (diagnostic).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"
SUBS = ROOT / "submissions"

# Load the 5 deployable anchor te arrays
anchors = {
    "nb1430": np.load(PROC / "te_nb1430_median.npy"),
    "nb1410": np.load(PROC / "te_nb1410_median.npy"),
    "nb1390": np.load(PROC / "te_nb1390_median.npy"),
    "nb1360": np.load(PROC / "te_nb1360_median.npy"),
    "nb1380": np.load(PROC / "te_nb1380_median.npy"),
}

for name, arr in anchors.items():
    print(f"{name}: shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f} "
          f"min={arr.min():.4f} max={arr.max():.4f}")

# Stack to (513, 5)
M = np.stack([anchors["nb1430"], anchors["nb1410"], anchors["nb1390"],
              anchors["nb1360"], anchors["nb1380"]], axis=1)
print(f"\nStacked matrix shape: {M.shape}")

te_median = np.median(M, axis=1)
te_mean = M.mean(axis=1)

print(f"\nte_nb1442 MEDIAN: mean={te_median.mean():.4f} std={te_median.std():.4f} "
      f"min={te_median.min():.4f} max={te_median.max():.4f}")
print(f"te_nb1442 MEAN:   mean={te_mean.mean():.4f} std={te_mean.std():.4f} "
      f"min={te_mean.min():.4f} max={te_mean.max():.4f}")

# Save te arrays
np.save(PROC / "te_nb1442.npy", te_median)
np.save(PROC / "te_nb1442_median.npy", te_median)
np.save(PROC / "te_nb1442_mean.npy", te_mean)

# In-sample evaluation on unblind 253
# Locate unblind indices + truth
unb_truth_path = PROC / "_audit_unblind_y.npy"
unb_idx_path = PROC / "_audit_unblind_idx.npy"

in_rae_median = None
in_rae_mean = None
if unb_truth_path.exists() and unb_idx_path.exists():
    y_true = np.load(unb_truth_path)
    unb_idx = np.load(unb_idx_path)
    y_hat_med = te_median[unb_idx]
    y_hat_mean = te_mean[unb_idx]

    def rae(y, p):
        return np.abs(y - p).sum() / np.abs(y - y.mean()).sum()

    in_rae_median = rae(y_true, y_hat_med)
    in_rae_mean = rae(y_true, y_hat_mean)
    print(f"\nIn-sample RAE on 253 unblind (MEDIAN): {in_rae_median:.4f}")
    print(f"In-sample RAE on 253 unblind (MEAN):   {in_rae_mean:.4f}")
else:
    print(f"\nWarning: missing unblind truth/idx ({unb_truth_path.exists()},{unb_idx_path.exists()})")

# Load test SMILES + Molecule Name for CSV
test_blind = pd.read_csv(ROOT / "data" / "raw" / "pxr-challenge_TEST_BLINDED.csv")
print(f"\nTest blind columns: {list(test_blind.columns)}")
print(f"Test blind shape: {test_blind.shape}")

# Map columns to required submission format
# Required: SMILES, Molecule Name, pEC50
name_col = None
smi_col = None
for c in test_blind.columns:
    if c.lower() in ("molecule name", "molecule_name", "name", "id"):
        name_col = c
    if c.lower() in ("smiles", "canonical_smiles", "std_smiles"):
        smi_col = c

print(f"Detected: name_col={name_col} smi_col={smi_col}")

# Median CSV (primary)
sub_med = pd.DataFrame({
    "SMILES": test_blind[smi_col],
    "Molecule Name": test_blind[name_col],
    "pEC50": te_median,
})
out_med = SUBS / "nb1442_robust_ensemble.csv"
sub_med.to_csv(out_med, index=False)
print(f"\nWrote MEDIAN: {out_med} rows={len(sub_med)}")

# Mean CSV (diagnostic)
sub_mean = pd.DataFrame({
    "SMILES": test_blind[smi_col],
    "Molecule Name": test_blind[name_col],
    "pEC50": te_mean,
})
out_mean = SUBS / "nb1442_robust_ensemble_mean.csv"
sub_mean.to_csv(out_mean, index=False)
print(f"Wrote MEAN:   {out_mean} rows={len(sub_mean)}")

# Summary JSON
import json
summary = {
    "nb": "nb1442",
    "anchors": list(anchors.keys()),
    "n_anchors": 5,
    "te_median": {
        "mean": float(te_median.mean()),
        "std": float(te_median.std()),
        "min": float(te_median.min()),
        "max": float(te_median.max()),
    },
    "te_mean": {
        "mean": float(te_mean.mean()),
        "std": float(te_mean.std()),
        "min": float(te_mean.min()),
        "max": float(te_mean.max()),
    },
    "in_rae_median": in_rae_median,
    "in_rae_mean": in_rae_mean,
    "csv_median": str(out_med),
    "csv_mean": str(out_mean),
    "te_median_npy": str(PROC / "te_nb1442_median.npy"),
    "te_mean_npy": str(PROC / "te_nb1442_mean.npy"),
    "te_npy": str(PROC / "te_nb1442.npy"),
}
out_json = PROC / "nb1442_summary.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nWrote summary: {out_json}")
print(json.dumps(summary, indent=2))
