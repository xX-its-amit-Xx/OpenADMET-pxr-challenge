"""Project paths, resolved from the package location."""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve()

# On Kaggle the package lives inside a read-only dataset mount:
#   /kaggle/input/datasets/<user>/<slug>/src/pxr/paths.py
# parents[2] == <slug>/  which is read-only.
# Detect this by checking for /kaggle AND that data/ subdir is absent.
_KAGGLE = os.path.exists("/kaggle/working")

if _KAGGLE:
    # Dataset root (read-only), writable scratch dir
    PROJECT_ROOT = _HERE.parents[2]          # slug dir
    DATA_RAW = PROJECT_ROOT / "rawdata"      # CSVs uploaded to rawdata/
    DATA_PROCESSED = Path("/kaggle/working")
    DATA_EXTERNAL = Path("/kaggle/working")
    SUBMISSIONS = Path("/kaggle/working")
    FIGURES = Path("/kaggle/working/figures")
else:
    PROJECT_ROOT = _HERE.parents[2]
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
    SUBMISSIONS = PROJECT_ROOT / "submissions"
    FIGURES = DATA_PROCESSED / "figures"

for _p in (DATA_PROCESSED, DATA_EXTERNAL, FIGURES, SUBMISSIONS):
    _p.mkdir(parents=True, exist_ok=True)
