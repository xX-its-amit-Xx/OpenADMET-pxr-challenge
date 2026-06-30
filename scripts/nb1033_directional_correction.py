"""nb1033 — TARGETED directional correction from the external PXR neighbor signal.

nb1032 found corr(ext_active, nb3200 err)=+0.222 on close-neighbor compounds, in the F2 direction (inactive
neighbors -> nb3200 over-predicts). As a global feature it washed out (signal concentrated in ~45 of 253). Here:
fit a per-fold linear correction  err_hat = b*(ext_active - c)  on COVERED+CONFIDENT train compounds only, apply
to covered+confident val compounds, leave the rest = nb3200. Honest scaffold 5-fold x 30 seeds. Tests whether the
real-but-sparse external signal yields a stable RAE gain when applied only where it exists.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "data/processed"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def build_ext():
    m = pd.read_csv(f"{D}/test_pxr_neighbor_matches.csv")
    nsmi = pd.read_csv(f"{D}/test_pxr_neighbor_smiles.csv").dropna(subset=["smiles"])
    cid2smi = dict(zip(nsmi["cid"], nsmi["smiles"]))
    te = load_test().reset_index(drop=True)
    ucids = [c for c in m["cid"].unique() if c in cid2smi]
    nfp = morgan_fp_batch([cid2smi[c] for c in ucids]).astype(np.uint8)
    cidx = {c: i for i, c in enumerate(ucids)}
    tefp = morgan_fp_batch(te["smiles"].tolist()).astype(np.uint8); nsum = nfp.sum(1)
    ea = np.full(len(te), np.nan); eb = np.zeros(len(te)); en = np.zeros(len(te))
    for pos, grp in m.groupby("test_pos"):
        cids = [c for c in grp["cid"].tolist() if c in cidx]
        if not cids:
            continue
        idx = [cidx[c] for c in cids]
        inter = nfp[idx] @ tefp[pos]; uni = nsum[idx] + tefp[pos].sum() - inter
        sims = inter / np.clip(uni, 1, None)
        ar = grp.set_index("cid").loc[cids, "active_rate"].to_numpy().astype(float)
        w = sims ** 2
        ea[pos] = float(np.sum(w * ar) / np.sum(w)) if w.sum() > 0 else np.nan
        eb[pos] = float(sims.max()); en[pos] = len(cids)
    return ea, eb, en


def main():
    ea, eb, en = build_ext()
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    anchor = np.load(f"{D}/nb3200_pred_oof.npy")
    te = load_test(); smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    ea, eb, en = ea[unb], eb[unb], en[unb]
    nov = pd.read_csv(f"{D}/novel_targets.csv").set_index("test_pos").loc[unb, "top1_sim"].to_numpy()

    r_anchor = rae(y, anchor)
    for SIM_THR in [0.35, 0.40, 0.50]:
        conf = (~np.isnan(ea)) & (eb >= SIM_THR)
        deltas = []
        for s in range(1400, 1430):
            folds = scaffold_kfold_indices(scaf, 5, seed=s)
            pred = anchor.copy()
            for tri, vai in folds:
                ct = conf & np.isin(np.arange(len(y)), tri)
                cv = conf & np.isin(np.arange(len(y)), vai)
                if ct.sum() < 8 or cv.sum() == 0:
                    continue
                # fit err = b*(ext_active - c) by least squares (c = train mean ext_active)
                c0 = ea[ct].mean(); x = (ea[ct] - c0); e = y[ct] - anchor[ct]
                b = float(np.dot(x, e) / (np.dot(x, x) + 1e-9))
                b = np.clip(b, -3, 3)
                pred[cv] = anchor[cv] + b * (ea[cv] - c0)
            deltas.append(rae(y, pred) - r_anchor)
        deltas = np.array(deltas); st = deltas.mean() < 0 and abs(deltas.mean()) > deltas.std()
        print(f"sim>={SIM_THR}: covered {conf.sum():3d}/253  delta {deltas.mean():+.5f} +/- {deltas.std():.5f} "
              f"wins {int((deltas<0).sum())}/30 stable={st}")

    # covered-subset only (does the correction help WHERE applied?) at sim>=0.4
    conf = (~np.isnan(ea)) & (eb >= 0.40)
    print(f"\ncovered-subset (sim>=0.4, n={conf.sum()}): anchor RAE on subset = {rae(y[conf], anchor[conf]):.4f}")
    # leave-one-out style fit on subset to gauge ceiling
    c0 = ea[conf].mean(); x = ea[conf] - c0; e = y[conf] - anchor[conf]
    b = float(np.dot(x, e) / (np.dot(x, x) + 1e-9))
    pred_sub = anchor[conf] + np.clip(b, -3, 3) * (ea[conf] - c0)
    print(f"  in-sample corrected subset RAE = {rae(y[conf], pred_sub):.4f}  (b={b:+.2f}; optimistic ceiling)")
    json.dump({"anchor": r_anchor, "covered_sim040": int(conf.sum())},
              open(f"{D}/nb1033_directional.json", "w"), indent=2)


if __name__ == "__main__":
    main()
