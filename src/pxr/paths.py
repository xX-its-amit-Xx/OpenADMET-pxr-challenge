"""Project paths, resolved from the package location."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
TUTORIAL = PROJECT_ROOT / "tutorial"
SUBMISSIONS = PROJECT_ROOT / "submissions"
FIGURES = DATA_PROCESSED / "figures"

for _p in (DATA_PROCESSED, DATA_EXTERNAL, FIGURES, SUBMISSIONS):
    _p.mkdir(parents=True, exist_ok=True)
