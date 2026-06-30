"""nb1031 — consolidate ALL external PXR-activity data and build a per-compound NEAR-NEIGHBOR prior for the test.

Prior cycles judged external data by whole-test MEDIAN similarity (for training scaffold-rescue) and declared it
'doesn't cover'. Different question here (user): for EACH test compound, is there ANY external neighbor at Tanimoto
0.9/0.8/0.7 carrying PXR activity we can bake into a soft prior? Even partial coverage of the novel tail helps.

Sources (all have SMILES + pEC50 unless noted): aid_1346985 (Tox21 PXR), aid_720659 (NCATS PXR), papyrus_pxr_nr,
chembl_pxr_all_types, chembl_pxr_new_external, bindingdb_pxr_direct, pubchem_pxr_pool, + pubchem_pxr_active_smiles
(active_rate only). Output test_external_prior.csv: per test compound, best external sim + neighbor-weighted pEC50.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "data/processed"; E = "data/external"


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


def load_pec50_sources():
    out = []
    specs = [
        (f"{E}/pubchem_aid_1346985_tox21_pxr/aid_1346985.parquet", "std_smiles", "pec50", "tox21_pxr"),
        (f"{E}/pubchem_aid_720659_ncats_pxr/aid_720659.parquet", "std_smiles", "pec50", "ncats_pxr"),
        (f"{E}/papyrus_pxr_nr.parquet", "std_smiles", "pec50", "papyrus"),
        (f"{E}/chembl_pxr_all_types.parquet", "smiles", "pec50", "chembl_all"),
        (f"{E}/chembl_pxr_new_external.parquet", "canonical_smiles", "pec50", "chembl_new"),
        (f"{E}/bindingdb_pxr_direct.parquet", "std_smiles", "pec50", "bindingdb"),
        (f"{D}/pubchem_pxr_pool.parquet", "std_smiles", "pec50", "pubchem_pool"),
    ]
    for path, sc, yc, src in specs:
        if not os.path.exists(path):
            print(f"  MISSING {path}"); continue
        df = pd.read_parquet(path)
        if sc not in df.columns or yc not in df.columns:
            print(f"  skip {src}: cols {list(df.columns)[:6]}"); continue
        sub = df[[sc, yc]].rename(columns={sc: "smiles", yc: "pec50"}).dropna()
        sub["pec50"] = pd.to_numeric(sub["pec50"], errors="coerce")
        sub = sub.dropna(); sub["src"] = src
        out.append(sub)
        print(f"  {src}: {len(sub)} rows  pec50 [{sub.pec50.min():.1f},{sub.pec50.max():.1f}] med {sub.pec50.median():.2f}")
    return pd.concat(out, ignore_index=True)


def main():
    ref = load_pec50_sources()
    # keep only sane pEC50 (-log10 M, ~3..10)
    ref = ref[(ref.pec50 > 2) & (ref.pec50 < 11)].copy()
    ref["ik"] = [ik(s) for s in ref["smiles"]]
    ref = ref.dropna(subset=["ik"])
    # dedup by inchikey: mean pec50, count sources
    agg = ref.groupby("ik").agg(pec50=("pec50", "mean"), n_meas=("pec50", "size"),
                                 smiles=("smiles", "first")).reset_index()
    print(f"\nconsolidated external PXR reference: {len(agg)} unique compounds (from {len(ref)} measurements)")

    # exclude any external compound that IS a test compound (would be exact leakage if matched to itself)
    te = load_test().reset_index(drop=True)
    te_ik = set(ik(s) for s in te["smiles"])
    exact = agg["ik"].isin(te_ik).sum()
    print(f"external compounds that are EXACT test matches: {exact} (kept separately; flagged not used as 'neighbor')")

    # fingerprints
    reffp = morgan_fp_batch(agg["smiles"].tolist()).astype(np.uint8)
    tefp = morgan_fp_batch(te["smiles"].tolist()).astype(np.uint8)
    ref_pec = agg["pec50"].to_numpy(); ref_ik = agg["ik"].to_numpy()
    te_ik_arr = np.array([ik(s) for s in te["smiles"]])

    refsum = reffp.sum(1)  # popcounts
    rows = []
    for i in range(len(te)):
        inter = reffp @ tefp[i]                       # (Nref,) intersection counts
        uni = refsum + tefp[i].sum() - inter
        sims = inter / np.clip(uni, 1, None)
        # exclude exact self-match (same inchikey)
        same = (ref_ik == te_ik_arr[i])
        sims_neigh = sims.copy(); sims_neigh[same] = -1
        order = np.argsort(-sims_neigh)[:5]
        top5 = sims_neigh[order]
        w = np.clip(top5, 0, None) ** 2
        prior = float(np.sum(w * ref_pec[order]) / np.sum(w)) if w.sum() > 0 else np.nan
        rows.append({"test_pos": i, "ext_top1_sim": float(top5[0]), "ext_top5_mean_sim": float(top5.mean()),
                     "ext_pec50_prior": prior, "ext_exact_match": int(same.any()),
                     "ext_exact_pec50": float(ref_pec[same].mean()) if same.any() else np.nan})
    out = pd.DataFrame(rows)
    nov = pd.read_csv(f"{D}/novel_targets.csv")[["test_pos", "top1_sim", "in_unb"]]
    out = out.merge(nov, on="test_pos")
    out.to_csv(f"{D}/test_external_prior.csv", index=False)

    print(f"\n=== external NEIGHBOR coverage of the test ===")
    for thr in [0.9, 0.8, 0.7, 0.6]:
        n = (out.ext_top1_sim >= thr).sum(); nu = ((out.ext_top1_sim >= thr) & (out.in_unb == 1)).sum()
        print(f"  external neighbor sim >= {thr}: {n}/513  ({nu} in 253 eval)")
    print(f"  EXACT external matches: {out.ext_exact_match.sum()}/513 ({(out[(out.ext_exact_match==1)].in_unb==1).sum()} in eval)")
    # focus on the NOVEL tail (low train sim) — do THEY have external neighbors?
    novel = out[out.top1_sim < 0.5]
    print(f"\n=== for the NOVEL tail (train top1_sim < 0.5, n={len(novel)}) ===")
    for thr in [0.8, 0.7, 0.6, 0.5]:
        print(f"  external neighbor sim >= {thr}: {(novel.ext_top1_sim>=thr).sum()}/{len(novel)}")
    print(f"  median external sim for novel tail: {novel.ext_top1_sim.median():.3f}")
    json.dump({"n_ref": len(agg), "exact_matches": int(out.ext_exact_match.sum()),
               "neighbor_ge_0.7": int((out.ext_top1_sim >= 0.7).sum()),
               "novel_neighbor_ge_0.7": int((novel.ext_top1_sim >= 0.7).sum())},
              open(f"{D}/nb1031_external_summary.json", "w"), indent=2)
    print(f"\nsaved {D}/test_external_prior.csv")


if __name__ == "__main__":
    main()
