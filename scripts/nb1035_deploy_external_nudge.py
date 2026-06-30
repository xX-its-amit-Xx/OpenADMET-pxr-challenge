"""nb1035 — DEPLOY the conservative external-neighbor prior as a downward-only nudge on nb3200.

Rule (a-priori, directionally validated in nb1032/1033; NOT fitted on eval labels): for test compounds with a
CONFIDENT PXR-tested near-neighbor (best Morgan sim >= SIM_MIN) whose neighbors are CLEARLY INACTIVE
(sim-weighted active_rate < THR), nudge the prediction DOWN by BETA*(THR - active_rate), capped at CAP. This
targets the documented F2 failure (novel inactives over-predicted) and can only LOWER predictions -> bounded risk.

Validates on the 253 (must not hurt), then applies to the full 513 nb3200 predictions and writes a submission.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "data/processed"
SIM_MIN, THR, BETA, CAP = 0.40, 0.30, 0.50, 0.40


def build_ext_active():
    """sim-weighted neighbor active_rate + best sim + n, for all 513 test compounds."""
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


def nudge(pred, ea, eb):
    qualify = (~np.isnan(ea)) & (eb >= SIM_MIN) & (ea < THR)
    dn = np.zeros_like(pred)
    dn[qualify] = np.clip(BETA * (THR - ea[qualify]), 0, CAP)
    return pred - dn, qualify


def main():
    ea, eb, en = build_ext_active()
    te = load_test().reset_index(drop=True)

    # ---- validate on the 253 (must not hurt) ----
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    anchor = np.load(f"{D}/nb3200_pred_oof.npy")
    p253, q253 = nudge(anchor.copy(), ea[unb], eb[unb])
    r0, r1 = rae(y, anchor), rae(y, p253)
    nov = pd.read_csv(f"{D}/novel_targets.csv").set_index("test_pos").loc[unb, "top1_sim"].to_numpy()
    print(f"253 validation: anchor {r0:.4f} -> nudged {r1:.4f}  (delta {r1-r0:+.5f}); {q253.sum()} compounds nudged")
    for nm, mk in [("novel<0.5", nov < 0.5), ("near>=0.5", nov >= 0.5)]:
        print(f"  {nm:10s} n={mk.sum():3d}  anchor {rae(y[mk],anchor[mk]):.4f} -> nudged {rae(y[mk],p253[mk]):.4f}  "
              f"({q253[mk].sum()} nudged)")
    if r1 > r0 + 0.0005:
        print("WARNING: nudge HURTS the 253 beyond tolerance — NOT writing submission.")
        return

    # ---- apply to full 513 + write submission ----
    te_pred = np.load(f"{D}/te_nb3200.npy").astype(float)
    p513, q513 = nudge(te_pred.copy(), ea, eb)
    print(f"\nfull 513: {q513.sum()} compounds nudged down (mean shrink {np.mean((te_pred-p513)[q513]):.3f}, "
          f"max {np.max(te_pred-p513):.3f})")
    out = pd.DataFrame({"SMILES": te["smiles"], "Molecule Name": te["name"], "pEC50": p513})
    path = "submissions/nb1035_external_neighbor_nudge.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")
    json.dump({"r253_anchor": r0, "r253_nudged": r1, "n_nudged_253": int(q253.sum()),
               "n_nudged_513": int(q513.sum()), "params": {"SIM_MIN": SIM_MIN, "THR": THR, "BETA": BETA, "CAP": CAP}},
              open(f"{D}/nb1035_deploy.json", "w"), indent=2)


if __name__ == "__main__":
    main()
