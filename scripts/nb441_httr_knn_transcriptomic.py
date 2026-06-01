"""
nb441_httr_knn_transcriptomic.py

Goal: build a sim-weighted PXR-pathway score for the 513 blinded test compounds
from HTTr TempO-Seq / L1000 transcriptomic data — k-NN by Morgan/Tanimoto on
SMILES, then aggregate log2FC of PXR-induced genes (CYP3A4, ABCB1/MDR1, UGT1A1)
weighted by sim^2. Calibrate to pEC50 via isotonic on the 253-compound unblind
release. Save te_nb441.npy, submissions/nb441_httr_pathway.csv, soft07 variant.

Memory-safe gate:
  - If HTTr / L1000 SMILES->expression matrix is not on disk in a usable form,
    print a structured JSON summary with status='httr_unavailable_on_disk' and
    exit 0 without crashing. The parent agent reads stdout for the structured
    result.

Audit performed (2026-05-31):
  HTTr (GSE272548 / EPA ToxCast HTTr in MCF-7):
    - data/external/httr_tempo_seq/GSE272548_mcf7_pg1_count_data.csv.gz
        only 1 of 41 plate groups present; file is truncated (EOF mid-stream
        when read past the first ~5 rows). Holds raw counts, not log2FC, and
        no chemical->well->SMILES mapping is on disk.
    - data/external/httr_tempo_seq/GSE272548_metadata.xlsx
        GEO submission template; sample table has 'chemical name' +
        'chemical sample ID' (TP00... codes) but NO canonical SMILES column,
        and the well IDs reference TC00284655_* whereas the count file
        contains TC00284658_* — different plate run. Cell line is MCF-7
        (breast), which is not a hepatic PXR-responsive model; CYP3A4 /
        UGT1A1 are not robustly inducible in MCF-7.
    - No standardized signature matrix (per-chemical log2FC) on disk.
  L1000 / CMap (data/external/multimodal_transcriptomic/):
    - pert_info_70138.txt.gz: per-perturbation table with canonical_smiles
        and inchi_key — useful, but ONLY metadata.
    - gene_info_92742.txt.gz: landmark/best-inferred gene list.
    - cellinfo_2020.txt: cell line metadata.
    - No signatures / level5 GCTX / per-pert log2FC matrix on disk.
  => For both data sources we are missing the SMILES->expression matrix.
     The task spec explicitly says: emit 'httr_unavailable_on_disk' and stop.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
HTTR_DIR = ROOT / "data" / "external" / "httr_tempo_seq"
L1000_DIR = ROOT / "data" / "external" / "multimodal_transcriptomic"
OUT_NPY = ROOT / "data" / "processed" / "te_nb441.npy"
OUT_SUB = ROOT / "submissions" / "nb441_httr_pathway.csv"
OUT_SOFT = ROOT / "submissions" / "nb441_httr_pathway_soft07_truth.csv"

PXR_TARGET_GENES = [
    # Classic PXR-induced
    "CYP3A4", "CYP3A5", "CYP2B6", "CYP2C9", "CYP2C19",
    "ABCB1", "MDR1",            # P-gp
    "UGT1A1", "UGT1A3",
    "SULT2A1",
    "NR1I2",                    # PXR itself (feedback)
    "GSTA1", "GSTA2",
    "CYP1A1", "CYP1A2",         # AhR overlap, often co-reported
]


def audit_httr_files():
    """Probe the HTTr / L1000 dirs for a usable SMILES->expression matrix.

    Returns: (ok: bool, info: dict)
    """
    info: dict = {
        "httr_dir_exists": HTTR_DIR.exists(),
        "l1000_dir_exists": L1000_DIR.exists(),
        "httr_files": [],
        "l1000_files": [],
        "pxr_genes_in_httr_probes": [],
        "smiles_mapping_found": False,
        "expression_matrix_readable": False,
        "reason": "",
    }

    if HTTR_DIR.exists():
        info["httr_files"] = sorted(p.name for p in HTTR_DIR.iterdir())
    if L1000_DIR.exists():
        info["l1000_files"] = sorted(p.name for p in L1000_DIR.iterdir())

    # --- Check L1000 first: it has SMILES in pert_info, but no signatures here.
    pert_info_path = L1000_DIR / "pert_info_70138.txt.gz"
    has_pert_info = pert_info_path.exists()
    sig_glob = (
        list(L1000_DIR.glob("*.gctx"))
        + list(L1000_DIR.glob("*level5*"))
        + list(L1000_DIR.glob("*signatures*"))
    )
    if has_pert_info and not sig_glob:
        info["smiles_mapping_found"] = True  # via pert_info
        info["reason"] = (
            "L1000 pert_info present (canonical_smiles) but no per-perturbation "
            "expression / signature matrix on disk."
        )

    # --- Check HTTr count file + metadata
    count_files = list(HTTR_DIR.glob("*count_data.csv.gz"))
    md_files = list(HTTR_DIR.glob("*metadata*"))
    found_pxr_genes: list[str] = []
    truncated = False
    if count_files:
        cf = count_files[0]
        # Probe first column without reading full file (safe against truncation)
        try:
            with gzip.open(cf, "rt", encoding="utf-8", errors="replace") as f:
                header = f.readline().strip().split(",")
                probes = []
                for _ in range(20000):  # cap rows scanned
                    line = f.readline()
                    if not line:
                        break
                    probes.append(line.split(",", 1)[0])
            gene_syms = {p.split("_")[0] for p in probes}
            found_pxr_genes = [g for g in PXR_TARGET_GENES if g in gene_syms]
            info["pxr_genes_in_httr_probes"] = found_pxr_genes
            info["httr_n_sample_columns"] = max(0, len(header) - 1)
            info["httr_n_probes_scanned"] = len(probes)
            info["expression_matrix_readable"] = True
        except (EOFError, OSError) as e:
            truncated = True
            info["expression_matrix_readable"] = False
            info["reason"] = f"HTTr count file truncated/corrupt: {e!s}"

    # --- Check metadata for SMILES column
    smiles_in_metadata = False
    if md_files:
        try:
            md = pd.read_excel(md_files[0], sheet_name="Metadata", header=None,
                               nrows=120)
            md_text = " ".join(
                str(v) for v in md.values.ravel() if pd.notna(v)
            ).lower()
            smiles_in_metadata = "canonical_smiles" in md_text or "smiles" in md_text
        except Exception as e:
            info["metadata_read_error"] = str(e)

    info["smiles_in_httr_metadata"] = smiles_in_metadata

    # --- Decision
    ok = False
    # We'd need: (a) per-compound SMILES, (b) per-compound log2FC for >=1 PXR gene
    have_httr_full = (
        bool(count_files)
        and not truncated
        and bool(found_pxr_genes)
        and smiles_in_metadata
    )
    have_l1000_full = bool(sig_glob) and has_pert_info
    if have_httr_full or have_l1000_full:
        ok = True
        info["reason"] = "usable matrix found"
    else:
        # Compose a precise reason
        bits = []
        if not count_files:
            bits.append("no HTTr count file")
        elif truncated:
            bits.append("HTTr count file truncated")
        if not found_pxr_genes and count_files and not truncated:
            bits.append("no PXR pathway genes in HTTr probes")
        if not smiles_in_metadata:
            bits.append("no SMILES column in HTTr metadata")
        if not sig_glob:
            bits.append("no L1000 signature/GCTX matrix on disk")
        info["reason"] = "; ".join(bits) if bits else info["reason"]

    return ok, info


def emit_summary(ok: bool, info: dict, extra: dict | None = None):
    summary = {
        "ok": ok,
        "note": "httr_unavailable_on_disk" if not ok else "ran",
        **info,
    }
    if extra:
        summary.update(extra)
    print("NB441_SUMMARY_JSON_BEGIN")
    print(json.dumps(summary, indent=2, default=str))
    print("NB441_SUMMARY_JSON_END")


def main():
    ok, info = audit_httr_files()
    if not ok:
        emit_summary(ok, info)
        # Touch placeholder npy so downstream stack scripts can skip cleanly
        OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
        np.save(OUT_NPY, np.full(513, np.nan, dtype=np.float32))
        return 0

    # --- If we ever reach here in the future (data drops in), the pipeline below
    # is the intended one. It is wired but never triggered with the current disk
    # state. Kept short to avoid silently running on incomplete data.

    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from sklearn.isotonic import IsotonicRegression
    from pxr.data import load_test
    from pxr.chem import standardize

    raise NotImplementedError(
        "Reached HTTr-available code path, but no curated SMILES->log2FC "
        "matrix was wired in this audit. Re-run nb441 after adding either "
        "(a) a HTTr per-compound log2FC matrix + SMILES side-table, or "
        "(b) an L1000 level5 signature file + pert_info join, in "
        f"{HTTR_DIR} or {L1000_DIR}."
    )


if __name__ == "__main__":
    sys.exit(main())
