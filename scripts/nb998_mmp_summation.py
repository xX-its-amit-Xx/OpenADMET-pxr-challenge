"""nb998 — MMP / summation analysis: what structural change adds how much pEC50? (the user's
'delta-sum of interactions' idea, grounded in real data). Single-cut matched molecular pairs from
the 4139 train -> per-transformation median delta-pEC50 + counts. Plus a fragment-contribution view.
Answers 'what takes pEC50 from 2->4->6->8'. Outputs -> C:/pxr_struct/dash/.
"""
import os, sys, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from rdkit import Chem
from rdkit.Chem import rdMMPA, Draw, Descriptors
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "C:/pxr_struct/dash"; os.makedirs(OUT, exist_ok=True)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    smiles = tr["smiles"].tolist(); y = list(tr["pec50"].to_numpy(float))
    # + the htchem analog series (organizers flagged its SAR) for richer matched pairs
    import pandas as pd
    for f, col in [("pxr-challenge_htchem-libraries_TRAIN.csv", "Corrected Crude pEC50 (log)"),
                   ("pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv", "Corrected Semi-Pure pEC50 (log)")]:
        p = f"data/external/htchem/{f}"
        if os.path.exists(p):
            d = pd.read_csv(p); pe = pd.to_numeric(d[col], errors="coerce")
            for s, v in zip(d["SMILES"], pe):
                if pd.notna(v): smiles.append(s); y.append(float(v))
    y = np.array(y); mols = [Chem.MolFromSmiles(s) for s in smiles]
    print(f"MMP over {len(mols)} compounds (train + htchem analogs)")

    # single-cut fragmentation: FragmentMol returns ('', 'fragA.fragB'); larger frag = context
    print("fragmenting (rdMMPA single-cut) ...", flush=True)
    core_to = defaultdict(list)
    for i, m in enumerate(mols):
        if m is None: continue
        try:
            frags = rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False)
        except Exception:
            continue
        for _core, frag_str in frags:
            parts = frag_str.split(".")
            if len(parts) != 2: continue
            a, b = parts
            ctx, rg = (a, b) if len(a) >= len(b) else (b, a)   # larger = scaffold context
            core_to[ctx].append((rg, i))

    # transformations within each shared core
    print("collecting matched pairs ...", flush=True)
    trans = defaultdict(list)             # (rA, rB) -> [delta pEC50]
    npairs = 0
    for core, members in core_to.items():
        if len(members) < 2: continue
        for a in range(len(members)):
            for b in range(len(members)):
                if a == b: continue
                rA, iA = members[a]; rB, iB = members[b]
                if rA == rB: continue
                trans[(rA, rB)].append(y[iB] - y[iA]); npairs += 1
    print(f"{len(core_to)} cores, {npairs} ordered matched pairs, {len(trans)} transformations")

    # aggregate (require >=6 observations for a stable median)
    rows = []
    for (rA, rB), deltas in trans.items():
        if len(deltas) >= 6:
            rows.append({"from": rA, "to": rB, "n": len(deltas),
                         "median_dpEC50": float(np.median(deltas)), "iqr": float(np.percentile(deltas, 75) - np.percentile(deltas, 25))})
    rows.sort(key=lambda r: -r["median_dpEC50"])

    def clean(r):  # strip the dummy-atom attachment for display
        return r.replace("[*:1]", "*").replace("[*]", "*")

    top_up = rows[:12]; top_dn = rows[-12:][::-1]
    print("\n=== top ACTIVITY-INCREASING transformations (median dpEC50) ===")
    for r in top_up[:8]: print(f"  {clean(r['from']):28s} -> {clean(r['to']):28s} dpEC50 {r['median_dpEC50']:+.2f} (n={r['n']})")
    print("=== top ACTIVITY-DECREASING ===")
    for r in top_dn[:8]: print(f"  {clean(r['from']):28s} -> {clean(r['to']):28s} dpEC50 {r['median_dpEC50']:+.2f} (n={r['n']})")

    # figure: top up/down transformations
    fig, ax = plt.subplots(figsize=(13, 9))
    sel = top_up[::-1] + top_dn[::-1]
    labels = [f"{clean(r['from'])[:18]} -> {clean(r['to'])[:18]}" for r in sel]
    vals = [r["median_dpEC50"] for r in sel]; ns = [r["n"] for r in sel]
    cols = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
    ax.barh(range(len(sel)), vals, color=cols, alpha=0.8)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(labels, fontsize=8, family="monospace")
    for i, (v, n) in enumerate(zip(vals, ns)): ax.text(v + (0.03 if v > 0 else -0.03), i, f"{v:+.2f} (n={n})", va="center", ha="left" if v > 0 else "right", fontsize=7)
    ax.axvline(0, c="k"); ax.set_xlabel("median delta-pEC50 of the transformation")
    ax.set_title("What structural change moves pEC50? (matched molecular pairs, n>=6)\n"
                 "green = activity-increasing, red = decreasing — the data-grounded 'delta-sum' table")
    plt.tight_layout(); plt.savefig(f"{OUT}/nb998_mmp_transformations.png", dpi=140); plt.close()

    # fragment-contribution: each R-group's mean pEC50 vs the global mean (Free-Wilson-ish)
    rg_y = defaultdict(list)
    for core, members in core_to.items():
        for rg, i in members: rg_y[rg].append(y[i])
    gm = y.mean()
    contrib = [{"rgroup": clean(rg), "n": len(v), "contrib": float(np.mean(v) - gm)} for rg, v in rg_y.items() if len(v) >= 15]
    contrib.sort(key=lambda r: -r["contrib"])
    json.dump({"n_pairs": npairs, "n_transformations": len(trans), "top_increasing": top_up,
               "top_decreasing": top_dn, "rgroup_contrib_top": contrib[:15], "rgroup_contrib_bottom": contrib[-15:]},
              open(f"{OUT}/nb998_mmp_summation.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/nb998_mmp_transformations.png + nb998_mmp_summation.json")


if __name__ == "__main__":
    main()
