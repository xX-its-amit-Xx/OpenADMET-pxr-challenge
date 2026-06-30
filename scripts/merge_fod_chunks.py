"""Merge FOD chunk CSVs from Explorer into a single local file."""
import glob, pandas as pd, pathlib

REMOTE_PATTERN = "/scratch/shenoy.am/pxr_work/fod/fod_chunk_*.csv"
LOCAL_CHUNKS   = "C:/pxr_work/fod/"
OUT = "C:/pxr_work/fod/fod_features_merged.csv"

chunks = sorted(glob.glob(LOCAL_CHUNKS + "fod_chunk_*.csv"))
if not chunks:
    print(f"No chunks found in {LOCAL_CHUNKS}. Run: scp explorer:{REMOTE_PATTERN} {LOCAL_CHUNKS}")
    raise SystemExit(1)

dfs = [pd.read_csv(f) for f in chunks]
df  = pd.concat(dfs, ignore_index=True)
print(f"Merged {len(chunks)} chunks → {len(df)} rows")
print(f"NaN rates:\n{df.isna().mean()}")
df.to_csv(OUT, index=False)
print(f"Wrote: {OUT}")
