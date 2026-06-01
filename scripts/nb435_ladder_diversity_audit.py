"""nb435 ladder diversity audit -- compute pairwise correlation across LADDER entries."""
import os, sys, re
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# Load LADDER from auto_submit_ladder.py via regex (avoid importing -- may pull heavy deps)
LADDER_SRC = Path(__file__).parent / "auto_submit_ladder.py"
src = LADDER_SRC.read_text(encoding="utf-8")
# pull lines like ("filename.csv", "note"),
ladder_entries = re.findall(r'\("([^"]+\.csv)",\s*"([^"]+)"\)', src)
print(f"Parsed {len(ladder_entries)} LADDER entries")

# Load unblind truth (Phase 1 unblinded -- 253 compounds)
truth_path = Path("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
truth_df = pd.read_csv(truth_path)
truth_map = dict(zip(truth_df["Molecule Name"], truth_df["pEC50"]))
print(f"Loaded {len(truth_map)} unblind truth labels")

# Load each submission CSV; align to a canonical Molecule Name index
preds = {}  # filename -> pd.Series indexed by Molecule Name
notes = {}
missing = []
for fn, note in ladder_entries:
    path = SUBMISSIONS / fn
    if not path.exists():
        missing.append(fn)
        continue
    try:
        df = pd.read_csv(path)
        if "Molecule Name" not in df.columns or "pEC50" not in df.columns:
            missing.append(fn + " (bad cols)")
            continue
        s = pd.Series(df["pEC50"].values, index=df["Molecule Name"].values)
        preds[fn] = s
        notes[fn] = note
    except Exception as e:
        missing.append(f"{fn} ({e})")

print(f"Loaded {len(preds)} present, {len(missing)} missing")
for m in missing:
    print(f"  MISSING: {m}")

# Build a 513xN matrix on common index (use union, fill NaN)
common_idx = sorted(set().union(*[s.index for s in preds.values()]))
print(f"Common index size: {len(common_idx)}")
M = pd.DataFrame({fn: s.reindex(common_idx) for fn, s in preds.items()})
print(f"Matrix shape: {M.shape}")

# Pairwise Pearson correlation (column-wise)
corr = M.corr(method="pearson")

# Honest unblind RAE for each entry (if it has unblind labels)
def honest_rae(s):
    """Compute RAE on the unblind subset where labels exist."""
    common = [k for k in s.index if k in truth_map and pd.notna(s[k])]
    if len(common) < 10:
        return np.nan
    y_pred = np.array([s[k] for k in common], dtype=float)
    y_true = np.array([truth_map[k] for k in common], dtype=float)
    mae = np.mean(np.abs(y_pred - y_true))
    mean_pred = np.mean(y_true)  # mean predictor on the same subset
    denom = np.mean(np.abs(y_true - mean_pred))
    return mae / denom if denom > 0 else np.nan

rae_map = {fn: honest_rae(s) for fn, s in preds.items()}

# Flag redundant pairs (corr > 0.99)
files = list(preds.keys())
redundant_pairs = []
for i in range(len(files)):
    for j in range(i + 1, len(files)):
        c = corr.iloc[i, j]
        if pd.notna(c) and c > 0.99:
            a, b = files[i], files[j]
            ra = rae_map.get(a, np.nan)
            rb = rae_map.get(b, np.nan)
            # keep the lower-RAE one; recommend removing the higher
            if pd.notna(ra) and pd.notna(rb):
                keep = a if ra <= rb else b
                drop = b if keep == a else a
            elif pd.notna(ra):
                keep, drop = a, b
            elif pd.notna(rb):
                keep, drop = b, a
            else:
                keep, drop = a, b  # tie: keep first
            redundant_pairs.append({
                "file_a": a, "file_b": b, "pearson_r": round(float(c), 5),
                "rae_a": ra, "rae_b": rb, "recommend_keep": keep,
                "recommend_drop": drop,
            })

print(f"\nRedundant pairs (corr > 0.99): {len(redundant_pairs)}")
for rp in redundant_pairs[:15]:
    print(f"  r={rp['pearson_r']}  KEEP {rp['recommend_keep']}  DROP {rp['recommend_drop']}")

# Avg pairwise correlation per entry (lower = more diverse)
avg_corr = {}
for fn in files:
    others = [c for c in corr[fn].drop(fn).values if pd.notna(c)]
    avg_corr[fn] = float(np.mean(others)) if others else np.nan
top5_diverse = sorted(avg_corr.items(), key=lambda x: x[1])[:5]
print("\nTop-5 most diverse (lowest avg pairwise corr):")
for fn, ac in top5_diverse:
    print(f"  {ac:.4f}  {fn}")

# Overall avg pairwise correlation
all_offdiag = []
for i in range(len(files)):
    for j in range(i + 1, len(files)):
        v = corr.iloc[i, j]
        if pd.notna(v):
            all_offdiag.append(v)
overall_avg = float(np.mean(all_offdiag)) if all_offdiag else np.nan
print(f"\nOverall avg pairwise correlation: {overall_avg:.4f}")
print(f"N pairs scored: {len(all_offdiag)}")

# Write audit CSV
rows = []
for fn in files:
    rows.append({
        "file": fn,
        "note": notes[fn],
        "honest_unblind_rae": rae_map.get(fn, np.nan),
        "avg_pairwise_corr": avg_corr.get(fn, np.nan),
        "n_redundant_with": sum(1 for rp in redundant_pairs
                                if rp["file_a"] == fn or rp["file_b"] == fn),
        "recommend_drop": fn in {rp["recommend_drop"] for rp in redundant_pairs},
    })
audit_df = pd.DataFrame(rows).sort_values("avg_pairwise_corr")
out_path = DATA_PROCESSED / "nb435_ladder_diversity_audit.csv"
audit_df.to_csv(out_path, index=False)
print(f"\nWrote {out_path}")

# Save redundant pairs detail too
if redundant_pairs:
    rp_df = pd.DataFrame(redundant_pairs).sort_values("pearson_r", ascending=False)
    rp_out = DATA_PROCESSED / "nb435_ladder_redundant_pairs.csv"
    rp_df.to_csv(rp_out, index=False)
    print(f"Wrote {rp_out}")

print("\n=== SUMMARY ===")
print(f"ladder_entries_audited: {len(preds)}")
print(f"redundant_pairs (r>0.99): {len(redundant_pairs)}")
print(f"avg_pairwise_corr: {overall_avg:.4f}")
print(f"top5_diverse: {[fn for fn, _ in top5_diverse]}")
