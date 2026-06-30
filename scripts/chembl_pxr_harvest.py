"""ChEMBL PXR harvest scoping: can external ChEMBL bioactivity bring NEW scaffolds
that rescue our failing TEST compounds?

Processes two large MCP-result JSON files on disk (EC50 + AC50 for NR1I2 / human PXR),
builds a unique-compound harvest, computes scaffold-rescue + proximity + leakage metrics,
and writes data/external/chembl_pxr_harvest.csv + data/processed/chembl_pxr_harvest_summary.json.
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

ROOT = r"D:\Users\ashenoy00000\.windsurf\OpenADMET-pxr-challenge"
EC50_PATH = r"D:\Users\ashenoy00000\.claude\projects\d--Users-ashenoy00000--windsurf-OpenADMET-pxr-challenge\c394e07c-82cc-4430-93ed-20bdeff94a09\tool-results\mcp-claude_ai_ChEMBL-get_bioactivity-1781030681749.txt"
AC50_PATH = r"D:\Users\ashenoy00000\.claude\projects\d--Users-ashenoy00000--windsurf-OpenADMET-pxr-challenge\c394e07c-82cc-4430-93ed-20bdeff94a09\tool-results\mcp-claude_ai_ChEMBL-get_bioactivity-1781030683420.txt"
TRAIN_PARQUET = os.path.join(ROOT, "data", "processed", "unimol_train.parquet")
TEST_PARQUET = os.path.join(ROOT, "data", "processed", "unimol_test513.parquet")
OUT_CSV = os.path.join(ROOT, "data", "external", "chembl_pxr_harvest.csv")
OUT_JSON = os.path.join(ROOT, "data", "processed", "chembl_pxr_harvest_summary.json")

BAD_VALIDITY = {
    "Potential author error",
    "Outside typical range",
    "Potential transcription error",
    "Potential missing data",
}
REPORTER_KEYWORDS = ("reporter", "luciferase", "transactivation", "cyp3a4")

_normalizer = rdMolStandardize.Normalizer()
_lfc = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize_smiles(smi):
    """Largest fragment, normalize, neutralize, canonical SMILES. Returns None on failure."""
    if not smi or not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = _lfc.choose(mol)
        mol = _normalizer.normalize(mol)
        mol = _uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return None


def murcko_scaffold(std_smi):
    """Bemis-Murcko scaffold SMILES from a (standardized) SMILES. Returns None on failure."""
    if not std_smi:
        return None
    try:
        scaf = MurckoScaffoldSmiles_safe(std_smi)
        return scaf
    except Exception:
        return None


def MurckoScaffoldSmiles_safe(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        core = MurckoScaffold.GetScaffoldForMol(m)
        if core is None:
            return None
        return Chem.MolToSmiles(core)


def inchikey14(std_smi):
    if not std_smi:
        return None
    m = Chem.MolFromSmiles(std_smi)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


def morgan_fp(std_smi, radius=2, nbits=2048):
    if not std_smi:
        return None
    m = Chem.MolFromSmiles(std_smi)
    if m is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
    except Exception:
        return None


def load_activities(path, tag):
    d = json.load(open(path, encoding="utf-8"))
    total = d.get("total")
    count = d.get("count")
    acts = d.get("activities", [])
    rows = []
    for a in acts:
        rows.append(
            {
                "source": tag,
                "molecule_chembl_id": a.get("molecule_chembl_id"),
                "canonical_smiles": a.get("canonical_smiles"),
                "standard_type": a.get("standard_type"),
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
                "standard_relation": a.get("standard_relation"),
                "pchembl_value": a.get("pchembl_value"),
                "assay_type": a.get("assay_type"),
                "assay_chembl_id": a.get("assay_chembl_id"),
                "assay_description": a.get("assay_description"),
                "data_validity_comment": a.get("data_validity_comment"),
            }
        )
    return pd.DataFrame(rows), total, count


def pct(n, d):
    return round(100.0 * n / d, 2) if d else 0.0


def main():
    ec50_df, ec50_total, ec50_count = load_activities(EC50_PATH, "EC50")
    ac50_df, ac50_total, ac50_count = load_activities(AC50_PATH, "AC50")
    raw = pd.concat([ec50_df, ac50_df], ignore_index=True)
    n_raw_records = len(raw)

    # --- filtering ---
    raw["pchembl_value"] = pd.to_numeric(raw["pchembl_value"], errors="coerce")
    # keep records with non-null canonical SMILES
    raw = raw[raw["canonical_smiles"].notna() & (raw["canonical_smiles"].astype(str).str.len() > 0)].copy()
    n_after_smiles = len(raw)
    # drop bad validity
    raw = raw[~raw["data_validity_comment"].isin(BAD_VALIDITY)].copy()
    n_after_validity = len(raw)
    # for pChEMBL, prefer relation '=' : null out pchembl where relation is not '=' so it
    # doesn't pollute the median (censored '>'/'<' values are not real potencies).
    rel = raw["standard_relation"].astype(str)
    raw["pchembl_clean"] = raw["pchembl_value"].where(rel.eq("="))

    # --- standardize SMILES per record (cache by canonical_smiles to save work) ---
    uniq_can = raw["canonical_smiles"].dropna().unique()
    std_cache = {s: standardize_smiles(s) for s in uniq_can}
    raw["std_smiles"] = raw["canonical_smiles"].map(std_cache)
    raw = raw[raw["std_smiles"].notna()].copy()
    n_after_std = len(raw)

    # --- dedup to UNIQUE compounds by molecule_chembl_id ---
    def has_reporter(descs):
        for de in descs:
            if not de:
                continue
            dl = de.lower()
            if any(k in dl for k in REPORTER_KEYWORDS):
                return True
        return False

    grp = raw.groupby("molecule_chembl_id", sort=False)
    recs = []
    for cid, g in grp:
        # pick a representative std_smiles: the most common one for this compound
        std_smi = g["std_smiles"].mode().iloc[0]
        pch = g["pchembl_clean"].dropna()
        pch_median = float(np.median(pch)) if len(pch) else np.nan
        recs.append(
            {
                "molecule_chembl_id": cid,
                "smiles": g["canonical_smiles"].iloc[0],
                "std_smiles": std_smi,
                "pchembl_median": pch_median,
                "n_records": int(len(g)),
                "assay_types": ",".join(sorted(set(str(x) for x in g["standard_type"].dropna()))),
                "has_reporter": bool(has_reporter(g["assay_description"].tolist())),
            }
        )
    comp = pd.DataFrame(recs)
    n_unique = len(comp)

    # --- scaffolds + inchikey14 for harvested compounds ---
    comp["scaffold"] = comp["std_smiles"].map(murcko_scaffold)
    comp["inchikey14"] = comp["std_smiles"].map(inchikey14)

    # --- our train/test ---
    tr = pd.read_parquet(TRAIN_PARQUET)
    te = pd.read_parquet(TEST_PARQUET)
    tr_std = tr["smiles"].map(standardize_smiles)
    te_std = te["smiles"].map(standardize_smiles)
    tr_scaf = set(s for s in tr_std.map(murcko_scaffold).tolist() if s)
    te_scaf_list = [s for s in te_std.map(murcko_scaffold).tolist() if s]
    te_scaf = set(te_scaf_list)
    te_scaf_novel = te_scaf - tr_scaf  # test scaffolds NOT covered by train

    tr_ik = set(s for s in tr_std.map(inchikey14).tolist() if s)
    te_ik = set(s for s in te_std.map(inchikey14).tolist() if s)

    # --- membership flags ---
    comp["in_train_scaffold"] = comp["scaffold"].map(lambda s: bool(s) and s in tr_scaf)
    comp["in_test_scaffold"] = comp["scaffold"].map(lambda s: bool(s) and s in te_scaf)

    # ===== metric 1 =====
    n_with_pchembl = int(comp["pchembl_median"].notna().sum())

    # ===== metric 2: pchembl distribution =====
    pser = comp["pchembl_median"].dropna()
    if len(pser):
        pdist = {
            "min": float(pser.min()),
            "p25": float(pser.quantile(0.25)),
            "median": float(pser.median()),
            "p75": float(pser.quantile(0.75)),
            "max": float(pser.max()),
            "n_ge_5": int((pser >= 5).sum()),
        }
    else:
        pdist = {"min": None, "p25": None, "median": None, "p75": None, "max": None, "n_ge_5": 0}

    # ===== metric 3: NEW-SCAFFOLD yield =====
    has_scaf = comp["scaffold"].notna() & (comp["scaffold"].astype(str).str.len() > 0)
    n_with_scaf = int(has_scaf.sum())
    new_scaf_mask = has_scaf & (~comp["in_train_scaffold"])
    n_new_scaffold = int(new_scaf_mask.sum())

    # ===== metric 4: TEST-SCAFFOLD RESCUE =====
    test_hit_mask = has_scaf & comp["in_test_scaffold"]
    n_compounds_match_test_scaf = int(test_hit_mask.sum())
    # of those, how many are scaffolds NOT in train (external covering novel-to-train test scaffolds)
    test_hit_novel_mask = test_hit_mask & (~comp["in_train_scaffold"])
    n_compounds_match_test_scaf_not_train = int(test_hit_novel_mask.sum())
    # distinct test scaffolds that get >=1 chembl compound
    chembl_scaf_set = set(comp.loc[has_scaf, "scaffold"].tolist())
    test_scaf_covered = te_scaf & chembl_scaf_set
    test_scaf_novel_covered = te_scaf_novel & chembl_scaf_set
    n_test_scaf_total = len(te_scaf)
    n_test_scaf_novel = len(te_scaf_novel)
    n_test_scaf_covered = len(test_scaf_covered)
    n_test_scaf_novel_covered = len(test_scaf_novel_covered)

    # ===== metric 5: NN proximity (sample up to 300) =====
    rng = np.random.RandomState(42)
    cand_idx = comp.index[comp["std_smiles"].notna()].tolist()
    sample_idx = list(rng.choice(cand_idx, size=min(300, len(cand_idx)), replace=False)) if cand_idx else []
    te_fps = [fp for fp in (morgan_fp(s) for s in te_std.tolist()) if fp is not None]
    comp["max_sim_to_test"] = np.nan
    sims = []
    for i in sample_idx:
        fp = morgan_fp(comp.at[i, "std_smiles"])
        if fp is None or not te_fps:
            continue
        ms = max(BulkTanimotoSimilarity(fp, te_fps))
        comp.at[i, "max_sim_to_test"] = ms
        sims.append(ms)
    sims = np.array(sims, dtype=float)
    if len(sims):
        prox = {
            "n_sampled": int(len(sims)),
            "median_max_sim": float(np.median(sims)),
            "frac_max_sim_ge_0.4": float(np.mean(sims >= 0.4)),
            "frac_max_sim_ge_0.6": float(np.mean(sims >= 0.6)),
            "n_max_sim_eq_1.0": int(np.sum(sims >= 0.999)),
        }
    else:
        prox = {"n_sampled": 0, "median_max_sim": None, "frac_max_sim_ge_0.4": None,
                "frac_max_sim_ge_0.6": None, "n_max_sim_eq_1.0": 0}

    # ===== metric 6: LEAKAGE (inchikey14 overlap) =====
    harvest_ik = set(s for s in comp["inchikey14"].dropna().tolist() if s)
    ik_overlap_test = harvest_ik & te_ik
    ik_overlap_train = harvest_ik & tr_ik
    # per-compound counts (a compound is flagged if its ik14 hits)
    comp_in_test_ik = comp["inchikey14"].map(lambda k: bool(k) and k in te_ik)
    comp_in_train_ik = comp["inchikey14"].map(lambda k: bool(k) and k in tr_ik)
    n_harvest_compounds_in_test = int(comp_in_test_ik.sum())
    n_harvest_compounds_in_train = int(comp_in_train_ik.sum())

    # ===== write CSV =====
    out_cols = [
        "molecule_chembl_id", "smiles", "std_smiles", "inchikey14", "scaffold",
        "pchembl_median", "n_records", "assay_types", "has_reporter",
        "in_train_scaffold", "in_test_scaffold", "max_sim_to_test",
    ]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    comp[out_cols].to_csv(OUT_CSV, index=False)

    # ===== summary JSON =====
    summary = {
        "inputs": {
            "ec50_total": ec50_total, "ec50_count": ec50_count,
            "ac50_total": ac50_total, "ac50_count": ac50_count,
            "raw_records_combined": int(n_raw_records),
        },
        "filtering": {
            "after_nonnull_smiles": int(n_after_smiles),
            "after_drop_bad_validity": int(n_after_validity),
            "after_standardize_ok": int(n_after_std),
            "bad_validity_set": sorted(BAD_VALIDITY),
        },
        "metric1_harvest": {
            "n_unique_compounds": int(n_unique),
            "n_with_scaffold": n_with_scaf,
            "n_with_pchembl_value": n_with_pchembl,
        },
        "metric2_pchembl_distribution": pdist,
        "metric3_new_scaffold_yield": {
            "n_new_scaffold": n_new_scaffold,
            "pct_new_scaffold_of_with_scaffold": pct(n_new_scaffold, n_with_scaf),
            "pct_new_scaffold_of_all": pct(n_new_scaffold, n_unique),
            "n_train_scaffolds": len(tr_scaf),
        },
        "metric4_test_scaffold_rescue": {
            "n_compounds_match_test_scaffold": n_compounds_match_test_scaf,
            "n_compounds_match_test_scaffold_NOT_in_train": n_compounds_match_test_scaf_not_train,
            "n_test_scaffolds_total": n_test_scaf_total,
            "n_test_scaffolds_novel_to_train": n_test_scaf_novel,
            "n_test_scaffolds_covered_by_chembl": n_test_scaf_covered,
            "n_test_scaffolds_novel_AND_covered_by_chembl": n_test_scaf_novel_covered,
            "test_scaf_novel_covered_smiles": sorted(test_scaf_novel_covered)[:50],
        },
        "metric5_nn_proximity": prox,
        "metric6_leakage": {
            "n_harvest_compounds_inchikey14_in_TEST": n_harvest_compounds_in_test,
            "n_harvest_compounds_inchikey14_in_TRAIN": n_harvest_compounds_in_train,
            "n_distinct_inchikey14_overlap_test": len(ik_overlap_test),
            "n_distinct_inchikey14_overlap_train": len(ik_overlap_train),
            "test_overlap_inchikey14_sample": sorted(ik_overlap_test)[:25],
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nWrote:", OUT_CSV, "rows=", len(comp))
    print("Wrote:", OUT_JSON)


if __name__ == "__main__":
    main()
