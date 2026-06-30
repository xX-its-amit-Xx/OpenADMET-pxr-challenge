"""nb1023 — export multi-assay data for the PXR FOUNDATION (neural Hill-equation) model.

Builds compact parquets (uploaded to Kaggle via kaggle_push --data) so the GPU notebook can train a
dose-response foundation model:
  - one molecular encoder -> per-compound (pEC50, Emax, Hill-n)
  - CRC pEC50 (4139) supervises pEC50 directly
  - single-conc log2FC (21k obs over 4 concentrations, 8131 scaffold-EXCLUSIVE compounds) supervises via the
    Hill forward model R(c)=Emax/(1+10^(n*(pConc-pEC50))) -> injects off-manifold scaffold coverage into pEC50
  - counter-assay null pEC50 (2647) -> PXR-specificity auxiliary head

LEAKAGE GUARD: any train-assay row whose standardized InChIKey matches a TEST-513 compound is dropped.
The 253 unblinded subset is NEVER used for training -> the 513 Kaggle predictions are an honest held-out test.

Outputs (data/processed/):
  found_compounds.parquet  comp_id, smiles, inchikey, scaffold, in_test, test_pos
  found_features.parquet   comp_id + f0..f2264 (combined Morgan+RDKit, imputed)
  found_crc.parquet        comp_id, pec50, emax
  found_counter.parquet    comp_id, pec50_null
  found_sc.parquet         comp_id, log10_conc, log2fc, w     (long format)
  found_meta.json          counts + the unb (253) positions for local validation
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test, load_counter, load_single_conc, load_crudes
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "data/processed"


def std_ik(smi):
    m = Chem.MolFromSmiles(str(smi))
    if not m:
        return None, None
    try:
        ik = Chem.MolToInchiKey(m)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None, None
    return ik, scaf


def main():
    # ---------- TEST 513 (prediction targets + leakage reference) ----------
    te = load_test().reset_index(drop=True)
    te_ik = []
    for s in te["smiles"]:
        ik, _ = std_ik(s); te_ik.append(ik)
    te["ik"] = te_ik
    test_ik_set = set(i for i in te_ik if i)
    unb = np.load(f"{D}/_audit_unblind_idx.npy")
    print(f"test: {len(te)} compounds, {len(test_ik_set)} valid IKs | unblind 253: {len(unb)}")

    # ---------- gather every compound across assays ----------
    crc = load_train().reset_index(drop=True)
    crc["pec50"] = pd.to_numeric(crc.get("pec50"), errors="coerce")
    if "emax" in crc.columns:
        crc["emax"] = pd.to_numeric(crc["emax"], errors="coerce")
    crc = crc.dropna(subset=["pec50"]).reset_index(drop=True)
    cnt = load_counter().reset_index(drop=True)
    cnt["pec50"] = pd.to_numeric(cnt.get("pec50"), errors="coerce")
    cnt = cnt.dropna(subset=["pec50"]).reset_index(drop=True)
    sc = load_single_conc()
    cr = load_crudes()

    # registry: inchikey -> comp_id; keep first smiles/scaffold seen
    reg = {}            # ik -> dict(comp_id, smiles, scaffold)
    order = []          # ik in insertion order

    def register(smi):
        ik, scaf = std_ik(smi)
        if ik is None:
            return None
        if ik not in reg:
            reg[ik] = {"comp_id": len(order), "smiles": str(smi), "scaffold": scaf or "", "ik": ik}
            order.append(ik)
        return reg[ik]["comp_id"]

    # register test FIRST so test compounds always have features
    te_pos = {}         # comp_id -> test position
    for pos, s in enumerate(te["smiles"]):
        cid = register(s)
        if cid is not None and cid not in te_pos:
            te_pos[cid] = pos

    # ---------- CRC labels ----------
    crc_rows = []
    for _, r in crc.iterrows():
        ik, _ = std_ik(r["smiles"])
        if ik is None or ik in test_ik_set:
            continue
        cid = register(r["smiles"])
        emax = float(r["emax"]) if "emax" in crc.columns and np.isfinite(r.get("emax", np.nan)) else np.nan
        crc_rows.append((cid, float(r["pec50"]), emax))
    crc_df = pd.DataFrame(crc_rows, columns=["comp_id", "pec50", "emax"]).groupby("comp_id", as_index=False).mean()

    # ---------- counter (null) labels ----------
    cnt_rows = []
    for _, r in cnt.iterrows():
        ik, _ = std_ik(r["smiles"])
        if ik is None or ik in test_ik_set:
            continue
        cid = register(r["smiles"])
        cnt_rows.append((cid, float(r["pec50"])))
    cnt_df = pd.DataFrame(cnt_rows, columns=["comp_id", "pec50_null"]).groupby("comp_id", as_index=False).mean()

    # ---------- single-conc long-format (dose) ----------
    sc_rows = []
    for _, r in sc.iterrows():
        c = r.get("concentration_M", np.nan); fc = r.get("log2_fc_estimate", np.nan)
        if not (np.isfinite(c) and np.isfinite(fc)) or c <= 0:
            continue
        ik, _ = std_ik(r["smiles"])
        if ik is None or ik in test_ik_set:
            continue
        cid = register(r["smiles"])
        se = r.get("log2_fc_stderr", np.nan); nrep = r.get("n_replicates", np.nan)
        w = 1.0
        if np.isfinite(se) and se > 0:
            w = float(min(4.0, 1.0 / (se ** 2 + 0.25)))   # precision weight, clipped
        sc_rows.append((cid, float(-np.log10(c)), float(fc), w))   # store pConc = -log10(c)
    sc_df = pd.DataFrame(sc_rows, columns=["comp_id", "pconc", "log2fc", "w"])

    # ---------- crudes (extra pEC50 -> fold into CRC) ----------
    extra = []
    cr["pec50"] = pd.to_numeric(cr.get("pec50"), errors="coerce")
    for _, r in cr.iterrows():
        v = r.get("pec50", np.nan)
        if not (isinstance(v, (int, float, np.floating)) and np.isfinite(v)):
            continue
        ik, _ = std_ik(r["smiles"])
        if ik is None or ik in test_ik_set:
            continue
        cid = register(r["smiles"])
        extra.append((cid, float(v), np.nan))
    if extra:
        ex_df = pd.DataFrame(extra, columns=["comp_id", "pec50", "emax"]).groupby("comp_id", as_index=False).mean()
        # only add crude compounds NOT already in CRC
        ex_df = ex_df[~ex_df["comp_id"].isin(set(crc_df["comp_id"]))]
        crc_df = pd.concat([crc_df, ex_df], ignore_index=True)

    # ---------- compound table + features ----------
    n = len(order)
    comp = pd.DataFrame({
        "comp_id": range(n),
        "smiles": [reg[ik]["smiles"] for ik in order],
        "inchikey": order,
        "scaffold": [reg[ik]["scaffold"] for ik in order],
    })
    comp["in_test"] = comp["comp_id"].map(lambda c: c in te_pos).astype(int)
    comp["test_pos"] = comp["comp_id"].map(lambda c: te_pos.get(c, -1)).astype(int)

    print(f"compounds: {n} unique | CRC: {len(crc_df)} | counter: {len(cnt_df)} | SC obs: {len(sc_df)} "
          f"(SC compounds {sc_df['comp_id'].nunique()})")

    print("featurizing (combined Morgan+RDKit)...")
    X = impute(combined(comp["smiles"].tolist())).astype(np.float32)
    feat = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    feat.insert(0, "comp_id", range(n))

    # ---------- save ----------
    comp.to_parquet(f"{D}/found_compounds.parquet", index=False)
    feat.to_parquet(f"{D}/found_features.parquet", index=False)
    crc_df.to_parquet(f"{D}/found_crc.parquet", index=False)
    cnt_df.to_parquet(f"{D}/found_counter.parquet", index=False)
    sc_df.to_parquet(f"{D}/found_sc.parquet", index=False)

    # test_pos -> comp_id map for the 513 (so notebook can build the 513-ordered prediction)
    pos2cid = {te_pos[c]: c for c in te_pos}
    test_comp = [pos2cid.get(p, -1) for p in range(len(te))]
    meta = {"n_compounds": n, "n_features": int(X.shape[1]), "n_crc": len(crc_df), "n_counter": len(cnt_df),
            "n_sc_obs": len(sc_df), "n_sc_compounds": int(sc_df["comp_id"].nunique()),
            "test_comp_id_by_pos": test_comp, "unb_pos": unb.tolist()}
    json.dump(meta, open(f"{D}/found_meta.json", "w"))
    json.dump({k: v for k, v in meta.items() if k not in ("test_comp_id_by_pos", "unb_pos")},
              open(f"{D}/found_meta_summary.json", "w"), indent=2)
    print(f"saved found_* parquets. features {X.shape}. test-covered {sum(1 for c in test_comp if c>=0)}/{len(te)}")


if __name__ == "__main__":
    main()
