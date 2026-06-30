"""
Cycle-130 M3 prep: Build external ChEMBL PXR + NR knowledge base for F2 Tanimoto-kNN.

Inputs (already on disk from prior fetches):
  data/external/chembl_pxr_CHEMBL3401.parquet   (5000 raw activities, PXR)
  data/external/chembl_nr_extended.parquet      (11496 NR-family rows: PPARg, FXR, RXRa, LXRa, PXR, VDR, PPARa)

Outputs:
  data/external/chembl_pxr_nr_kb.parquet        [smiles, inchikey, scaffold, pec50_chembl, source_target]
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, DataStructs, inchi as rd_inchi
import warnings
warnings.filterwarnings("ignore")

from src.pxr.data import load_train, load_test
from src.pxr.chem import standardize as _std_mol


def standardize(smi):
    """Return canonical std SMILES string (or None)."""
    m = _std_mol(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None

ROOT = Path("data/external")
OUT = ROOT / "chembl_pxr_nr_kb.parquet"


def murcko(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    try:
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s) if s else ""
    except Exception:
        return ""


def inchikey(smi: str) -> str:
    if not isinstance(smi, str) or not smi:
        return ""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        return rd_inchi.MolToInchiKey(m) or ""
    except Exception:
        return ""


def main() -> dict:
    # ---- Source 1: ChEMBL PXR (CHEMBL3401) ----
    pxr = pd.read_parquet(ROOT / "chembl_pxr_CHEMBL3401.parquet")
    pxr = pxr[pxr["pchembl_value"].notna() & pxr["canonical_smiles"].notna()].copy()
    pxr = pxr[pxr["standard_type"].isin(["EC50", "IC50", "AC50", "Ki", "Kd"])]
    pxr_rows = pxr.rename(
        columns={"canonical_smiles": "smiles_raw", "pchembl_value": "pec50_chembl"}
    )[["smiles_raw", "pec50_chembl", "standard_type"]].copy()
    pxr_rows["source_target"] = "PXR_CHEMBL3401"
    n_pxr_raw = len(pxr_rows)

    # ---- Source 2: NR-extended (already filtered/cleaned) ----
    nr = pd.read_parquet(ROOT / "chembl_nr_extended.parquet")
    nr["source_target"] = "NR_" + nr["target_name"]
    nr_rows = nr.rename(columns={"std_smiles": "smiles_raw", "pec50": "pec50_chembl"})[
        ["smiles_raw", "pec50_chembl", "standard_type", "source_target"]
    ].copy()
    n_nr_raw_per_target = nr["target_name"].value_counts().to_dict()

    # Combine
    raw = pd.concat([pxr_rows, nr_rows], ignore_index=True)
    raw["pec50_chembl"] = pd.to_numeric(raw["pec50_chembl"], errors="coerce")
    raw = raw.dropna(subset=["pec50_chembl"])
    print(f"Combined raw: {len(raw)}  (PXR={n_pxr_raw}, NR={len(nr_rows)})")

    # ---- Standardize ----
    print("Standardizing SMILES ...")
    raw["smiles"] = raw["smiles_raw"].apply(lambda s: standardize(s) if isinstance(s, str) else None)
    raw = raw.dropna(subset=["smiles"])
    raw["inchikey"] = raw["smiles"].apply(inchikey)
    raw = raw[raw["inchikey"] != ""]
    print(f"After std/inchikey: {len(raw)}")

    # ---- Median-aggregate per inchikey x target (multiple assays per compound) ----
    agg = (
        raw.groupby(["inchikey", "source_target"], as_index=False)
        .agg(
            smiles=("smiles", "first"),
            pec50_chembl=("pec50_chembl", "median"),
            n_assays=("pec50_chembl", "size"),
        )
    )
    # Keep one row per inchikey: prefer PXR_CHEMBL3401, else max-n source
    agg["is_pxr"] = (agg["source_target"] == "PXR_CHEMBL3401").astype(int)
    agg = agg.sort_values(["inchikey", "is_pxr", "n_assays"], ascending=[True, False, False])
    dedup = agg.drop_duplicates(subset=["inchikey"], keep="first").copy()
    print(f"Unique inchikeys after dedupe: {len(dedup)}")

    # ---- Scaffold ----
    print("Computing Murcko scaffolds ...")
    dedup["scaffold"] = dedup["smiles"].apply(murcko)

    # ---- Set difference vs OUR train+test ----
    tr = load_train()
    te = load_test()
    tr["std_smi"] = tr["smiles"].apply(standardize)
    te["std_smi"] = te["smiles"].apply(standardize)
    tr["ik"] = tr["std_smi"].apply(inchikey)
    te["ik"] = te["std_smi"].apply(inchikey)
    ours_iks = set(tr["ik"]) | set(te["ik"])
    print(f"Our train+test unique inchikeys: {len(ours_iks)}")

    pre = len(dedup)
    dedup = dedup[~dedup["inchikey"].isin(ours_iks)].copy()
    print(f"After set diff (no leakage): {len(dedup)}  (removed {pre - len(dedup)} overlaps)")

    # ---- Novel-scaffold count vs test 513 ----
    te_scaf = set(te["std_smi"].apply(murcko))
    te_scaf.discard("")
    dedup["scaffold_in_test"] = dedup["scaffold"].isin(te_scaf)
    n_novel_scaf = (~dedup["scaffold_in_test"]).sum()
    n_shared_scaf = dedup["scaffold_in_test"].sum()
    n_unique_external_scaffolds = dedup["scaffold"].nunique()

    # ---- F2 overlap projection: max Tanimoto from each test compound to external KB ----
    print("Computing test->external Tanimoto stats (ECFP4 2048-bit) ...")

    def fp(smi):
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None

    te_fps = [fp(s) for s in te["std_smi"]]
    kb_fps = [fp(s) for s in dedup["smiles"]]
    kb_fps_valid = [f for f in kb_fps if f is not None]

    sims_top1 = []
    for fpi in te_fps:
        if fpi is None:
            sims_top1.append(0.0)
            continue
        s = DataStructs.BulkTanimotoSimilarity(fpi, kb_fps_valid)
        sims_top1.append(max(s) if s else 0.0)
    sims_top1 = np.array(sims_top1)

    # F2 = greasy-novel-inactive tail; proxy: test compounds with low max-similarity to our TRAIN
    # We measure the *gain* from adding external KB: compounds whose max sim to KB > max sim to train
    print("Computing test->train Tanimoto (for delta) ...")
    tr_fps = [fp(s) for s in tr["std_smi"]]
    tr_fps_valid = [f for f in tr_fps if f is not None]
    tr_top1 = []
    for fpi in te_fps:
        if fpi is None:
            tr_top1.append(0.0)
            continue
        s = DataStructs.BulkTanimotoSimilarity(fpi, tr_fps_valid)
        tr_top1.append(max(s) if s else 0.0)
    tr_top1 = np.array(tr_top1)

    # F2 cohort proxy: test compounds with low train similarity (< 0.40)
    f2_mask = tr_top1 < 0.40
    n_f2 = int(f2_mask.sum())
    kb_helps_f2 = ((sims_top1 > tr_top1) & f2_mask).sum()
    median_gain_f2 = float(np.median((sims_top1 - tr_top1)[f2_mask])) if n_f2 else 0.0
    median_kb_sim_f2 = float(np.median(sims_top1[f2_mask])) if n_f2 else 0.0

    # ---- Save final KB ----
    out = dedup[["smiles", "inchikey", "scaffold", "pec50_chembl", "source_target"]].copy()
    out.to_parquet(OUT, index=False)
    print(f"\nSaved KB: {OUT}  rows={len(out)}")

    # ---- Summary ----
    per_target_kept = out["source_target"].value_counts().to_dict()
    summary = {
        "n_pxr_chembl3401_raw_high_quality": int(n_pxr_raw),
        "n_nr_extended_raw": int(len(nr_rows)),
        "nr_per_target_raw": {k: int(v) for k, v in n_nr_raw_per_target.items()},
        "n_kb_rows_final": int(len(out)),
        "kb_per_target_final": {k: int(v) for k, v in per_target_kept.items()},
        "n_unique_external_scaffolds": int(n_unique_external_scaffolds),
        "n_novel_scaffolds_vs_test513": int(n_novel_scaf),
        "n_shared_scaffolds_with_test513": int(n_shared_scaf),
        "test_top1_sim_to_kb_median": float(np.median(sims_top1)),
        "test_top1_sim_to_train_median": float(np.median(tr_top1)),
        "f2_cohort_size_train_sim_lt_040": n_f2,
        "f2_kb_helps_count": int(kb_helps_f2),
        "f2_median_kb_top1_sim": median_kb_sim_f2,
        "f2_median_sim_gain_from_kb": median_gain_f2,
        "output_path": str(OUT),
    }
    import json
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/cycle130_chembl_kb_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
