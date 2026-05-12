#!/usr/bin/env python3
"""
Auto-update README.md model tables when new OOF arrays appear.
Designed to run as a Claude Code Stop hook after every response.

Fast path: compares a fingerprint of data/processed/oof_*.npy files
against a cached value. Exits in ~5ms if nothing changed.
"""
import re
import json
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
NOTEBOOKS = ROOT / "notebooks"
README = ROOT / "README.md"
FINGERPRINT_FILE = ROOT / ".readme_fingerprint"

# nb prefix -> (display name, feature description)
# Add new rows here when adding new base model notebooks.
BASE_MODEL_META = {
    "02":  ("LGBM baseline",      "Morgan (2048) + RDKit (217), combined 2265-dim"),
    "03":  ("Chemprop multitask", "GNN, dual head: PXR + counter-assay"),
    "05":  ("Tanimoto k-NN",      "ECFP4, k=5, similarity-weighted"),
    "13":  ("ChemBERTa-MLM",      "`ChemBERTa-zinc-base-v1`, 768-dim CLS token"),
    "14":  ("ChemBERTa-MTR",      "`ChemBERTa-PubChem-base-v1`, 768-dim CLS token"),
    "16":  ("LGBM tuned",         "Morgan + RDKit, Optuna TPE 60-trial search"),
    "19":  ("Uni-Mol",            "3D conformer-aware transformer, 512-dim"),
    "20":  ("BERT-SMILES",        "`unikei/bert-base-smiles`, 768-dim CLS token"),
    "21":  ("SELFormer",          "`HUBioDataLab/SELFormer`, SELFIES RoBERTa, 768-dim CLS"),
    "22":  ("GROVER-base",        "Graph transformer atom FP 1600-dim; pretrained on 10M molecules"),
    "22b": ("GROVER-large",       "Graph transformer atom FP 2400-dim"),
}

# nb prefix -> (version label, description of what's new)
# Add new rows here when adding new ensemble notebooks.
ENSEMBLE_META = {
    "15": ("Grand v1", "lgbm_aug + knn + ChemBERTa × 2 + chemprop"),
    "18": ("Grand v2", "lgbm_tuned (Optuna) replaces lgbm_aug"),
    "23": ("Grand v3", "v2 + Uni-Mol"),
    "24": ("Grand v4", "v3 + GROVER-base"),
    "25": ("Grand v5", "v4 + GROVER-large"),
}


# ── Fingerprinting ─────────────────────────────────────────────────────────────

def compute_fingerprint() -> str:
    files = sorted(PROCESSED.glob("oof_*.npy"))
    content = "|".join(f"{p.name}:{p.stat().st_size}" for p in files if p.exists())
    return hashlib.md5(content.encode()).hexdigest()


def fingerprint_unchanged() -> bool:
    current = compute_fingerprint()
    if FINGERPRINT_FILE.exists() and FINGERPRINT_FILE.read_text().strip() == current:
        return True
    return False


def save_fingerprint():
    FINGERPRINT_FILE.write_text(compute_fingerprint())


# ── Notebook parsing ───────────────────────────────────────────────────────────

def _cell_text(cell) -> str:
    out = []
    for o in cell.get("outputs", []):
        out.extend(o.get("text", o.get("data", {}).get("text/plain", [])))
    return "".join(out)


def _read_notebook(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_oof_rae(nb_path: Path) -> float | None:
    nb = _read_notebook(nb_path)
    if not nb:
        return None
    for cell in nb.get("cells", []):
        text = _cell_text(cell)
        m = re.search(r"OOF RAE[^:]*:\s*(0\.\d+)", text)
        if m:
            return float(m.group(1))
    return None


def extract_nested_rae(nb_path: Path) -> float | None:
    nb = _read_notebook(nb_path)
    if not nb:
        return None
    for cell in nb.get("cells", []):
        text = _cell_text(cell)
        m = re.search(r"Nested CV meta-learner RAE:\s*(0\.\d+)", text)
        if m:
            return float(m.group(1))
    return None


def find_notebook(prefix: str) -> Path | None:
    """Find the executed notebook for a given prefix (e.g. '22b')."""
    for nb in sorted(NOTEBOOKS.glob(f"{prefix}_*.ipynb")):
        return nb
    return None


# ── Table builders ─────────────────────────────────────────────────────────────

def build_base_table(rows: list[tuple]) -> str:
    lines = [
        "| # | Model | Features / Architecture | OOF RAE |",
        "|---|---|---|---|",
    ]
    for nb, name, feats, rae in rows:
        lines.append(f"| {nb} | {name} | {feats} | {rae:.4f} |")
    return "\n".join(lines)


def build_ensemble_table(rows: list[tuple]) -> str:
    lines = [
        "| Version | Notebook | New model added | Nested CV RAE |",
        "|---|---|---|---|",
    ]
    for nb, version, desc, rae in rows:
        marker = "**" if rows[-1][0] == nb else ""
        lines.append(f"| {marker}{version}{marker} | {marker}nb{nb}{marker} | {marker}{desc}{marker} | {marker}{rae:.4f}{marker} |")
    return "\n".join(lines)


# ── README section replacement ─────────────────────────────────────────────────

def replace_section(text: str, header: str, new_table: str) -> str:
    """Replace the markdown table that follows `header` in `text`."""
    # Match: header line + blank line + table (lines starting with |)
    pattern = rf"({re.escape(header)}\n\n)(\|[^\n]*\n(?:\|[^\n]*\n)*)"
    replacement = rf"\g<1>{new_table}\n"
    result, n = re.subn(pattern, replacement, text)
    if n == 0:
        # Header not found — skip silently
        return text
    return result


# ── Git commit ─────────────────────────────────────────────────────────────────

def git_commit(message: str):
    subprocess.run(["git", "add", "README.md"], cwd=ROOT, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, capture_output=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if fingerprint_unchanged():
        return  # ~5ms exit; nothing new

    # Gather base model results
    base_rows = []
    for prefix, (name, feats) in BASE_MODEL_META.items():
        nb = find_notebook(prefix)
        if nb is None:
            continue
        rae = extract_oof_rae(nb)
        if rae is not None:
            base_rows.append((prefix, name, feats, rae))

    # Gather ensemble results
    ens_rows = []
    for prefix, (version, desc) in ENSEMBLE_META.items():
        nb = find_notebook(prefix)
        if nb is None:
            continue
        rae = extract_nested_rae(nb)
        if rae is not None:
            ens_rows.append((prefix, version, desc, rae))

    if not base_rows and not ens_rows:
        save_fingerprint()
        return

    readme = README.read_text(encoding="utf-8")
    updated = readme

    if base_rows:
        updated = replace_section(updated, "### Base Models", build_base_table(base_rows))

    if ens_rows:
        updated = replace_section(updated, "### Ensemble Progression", build_ensemble_table(ens_rows))

    if updated == readme:
        save_fingerprint()
        return  # Tables already current

    README.write_text(updated, encoding="utf-8")
    save_fingerprint()

    # Find the latest model names for the commit message
    new_base = base_rows[-1][1] if base_rows else ""
    new_ens = f"grand_v{len(ens_rows)}" if ens_rows else ""
    label = ", ".join(filter(None, [new_base, new_ens]))
    git_commit(f"Auto-update README: sync tables [{label}]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # Never surface hook errors to user
