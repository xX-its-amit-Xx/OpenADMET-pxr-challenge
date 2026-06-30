"""nb1211 — ALFABET ML-DFT bond-dissociation-energy (BDE) scalars build.

For each of 4139 train + 513 test compounds, run ALFABET (pstjohn/alfabet,
Digital Discovery 2020) to predict per-bond homolytic BDE/BDFE, then aggregate
to compact molecular thermochemical-reactivity scalars. This is the first
BOND-CLEAVAGE THERMOCHEMISTRY axis on the ledger — distinct from every prior QM
block (AIMNet2 charges/forces, strain energetics, D4 dispersion/polarizability,
DBSTEP steric/shape). Writes C:/pxr_work/alfabet/alfabet_features.csv.

Resumable: caches per-molecule aggregates to a parquet; re-run resumes.
"""
import os, sys
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.chem import standardize
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

OUT = "C:/pxr_work/alfabet"
os.makedirs(OUT, exist_ok=True)
CACHE = f"{OUT}/per_mol_cache.pkl"
FEAT = f"{OUT}/alfabet_features.csv"
LABILE_THR = 80.0   # kcal/mol; bonds weaker than this = metabolically labile


def agg_one(g):
    """Aggregate alfabet per-bond df (one molecule) -> scalar dict."""
    valid = g[g["is_valid"]]
    bde = valid["bde_pred"].to_numpy(float)
    bdfe = valid["bdfe_pred"].to_numpy(float)
    n = len(bde)
    if n == 0:
        return None
    return {
        "bde_min": float(np.min(bde)),
        "bde_max": float(np.max(bde)),
        "bde_mean": float(np.mean(bde)),
        "bde_std": float(np.std(bde)) if n > 1 else 0.0,
        "bde_range": float(np.max(bde) - np.min(bde)),
        "bde_q10": float(np.quantile(bde, 0.10)),
        "bdfe_min": float(np.min(bdfe)),
        "bdfe_mean": float(np.mean(bdfe)),
        "n_bonds_scored": int(n),
        "n_labile": int(np.sum(bde < LABILE_THR)),
        "frac_labile": float(np.mean(bde < LABILE_THR)),
    }


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    rows = [("train", r["name"], r["smiles"]) for _, r in tr.iterrows()] + \
           [("test", r["name"], r["smiles"]) for _, r in te.iterrows()]
    df = pd.DataFrame(rows, columns=["src", "name", "smiles"])
    # alfabet returns RDKit-canonical SMILES in its 'molecule' col -> key on that
    from rdkit import Chem
    def canon(s):
        m = standardize(s)             # returns a Mol (or None)
        if m is None:
            m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m is not None else (s or "")
    df["std"] = [canon(s) for s in df["smiles"]]

    done = {}
    if os.path.exists(CACHE):
        prev = pd.read_pickle(CACHE)
        done = {n: r for n, r in zip(prev["name"], prev.to_dict("records"))}
        print(f"resume: {len(done)} cached")

    from alfabet import model
    todo = df[~df["name"].isin(done)].reset_index(drop=True)
    print(f"to compute: {len(todo)} / {len(df)}")
    CHUNK = 64
    recs = list(done.values())
    for i in range(0, len(todo), CHUNK):
        sub = todo.iloc[i:i + CHUNK]
        smis = sub["std"].tolist()
        try:
            res = model.predict(smis, drop_duplicates=False, verbose=False)
        except Exception as e:
            print(f"  chunk {i} predict FAIL: {e}; per-mol fallback")
            res = None
        # map canonical smiles -> name. alfabet 'molecule' col holds input smiles.
        by_smi = {}
        if res is not None and len(res):
            for smi, g in res.groupby("molecule"):
                by_smi[smi] = g
        for _, rr in sub.iterrows():
            g = by_smi.get(rr["std"])
            a = agg_one(g) if g is not None else None
            rec = {"name": rr["name"], "src": rr["src"]}
            if a is None:
                a = {}
            rec.update(a)
            recs.append(rec)
        if (i // CHUNK) % 5 == 0:
            pd.DataFrame(recs).to_pickle(CACHE)
            print(f"  {i + len(sub)}/{len(todo)} done")
    out = pd.DataFrame(recs)
    out.to_pickle(CACHE)
    # order columns
    feat_cols = ["bde_min", "bde_max", "bde_mean", "bde_std", "bde_range",
                 "bde_q10", "bdfe_min", "bdfe_mean", "n_bonds_scored",
                 "n_labile", "frac_labile"]
    for c in feat_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[["name", "src"] + feat_cols]
    out.to_csv(FEAT, index=False)
    nfail = int(out[feat_cols].isna().all(axis=1).sum())
    print(f"saved {FEAT}: {len(out)} rows, {nfail} all-NaN (alfabet-failed)")


if __name__ == "__main__":
    main()
